import os
import secrets

import torch
import torch.utils.data as torch_data
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
from transformers import set_seed as tf_set_seed

from model import WebContentClassifier
from dataset import HDF5Dataset
from constants import IS_MASTER_NODE


def allow_cuda_tf32():
    # issue with ray tune so use eval instead of running the code directly
    eval('setattr(torch.backends.cuda.matmul, "allow_tf32", True)')
    eval('setattr(torch.backends.cudnn, "allow_tf32", True)')


def set_seed(seed: int):
    """Set all seeds to make results reproducible"""
    tf_set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True  # issue with ray tune
    # torch.backends.cudnn.benchmark = False  # issue with ray tune
    eval('setattr(torch.backends.cudnn, "deterministic", True)')
    eval('setattr(torch.backends.cudnn, "benchmark", False)')
    np.random.seed(seed)
    secrets.SystemRandom().seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_data_parallel_model():
    model = WebContentClassifier()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    return model


def get_data_loaders(batch_size, TRAIN_H5, model=None):
    num_workers = 8 if IS_MASTER_NODE else 12
    train_set = HDF5Dataset(TRAIN_H5, model)
    train_ratio = 0.90
    val_ratio = 0.1
    # Calculate the lengths of each subset
    train_length = int(train_ratio * len(train_set))
    val_length = int(val_ratio * len(train_set))
    remainder = len(train_set) - train_length - val_length
    val_length += remainder

    train_subset, val_subset = torch_data.random_split(
        train_set,
        [train_length, val_length],
    )
    train_loader = torch_data.DataLoader(
        train_subset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = torch_data.DataLoader(
        val_subset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        shuffle=False,
    )
    return train_loader, val_loader, train_set.class_weights


def ddp_setup(rank, world_size):
    master_addr = "192.168.0.199"
    master_port = "12355"
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )


def get_ddp_model(rank, model: nn.Module = WebContentClassifier()):
    IS_MASTER_TRAINER = os.environ.get("MASTER_TRAINER", 0)
    device = (
        torch.device(f"cuda:{rank}") if IS_MASTER_TRAINER else torch.device(f"cuda:{0}")
    )
    model.to(device)
    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False,
    )
    return model


def get_ddp_data_loaders(
    batch_size: int, TRAIN_H5: str, TEST_H5: str, rank: int, world_size: int
):
    num_workers = 8 if IS_MASTER_NODE else 12
    train_set = HDF5Dataset(TRAIN_H5)
    test_set = HDF5Dataset(TEST_H5)
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True
    )
    test_sampler = DistributedSampler(
        test_set, num_replicas=world_size, rank=rank, shuffle=False
    )
    train_loader = torch_data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    test_loader = torch_data.DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=test_sampler,
    )
    return train_loader, test_loader, train_set.class_weights
