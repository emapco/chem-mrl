from dataclasses import asdict, dataclass
from typing import Any, TypeVar

from .Enums import MultiProcessContextOption, SchedulerOption, WatchLogOption

BoundConfigType = TypeVar("BoundConfigType", bound="BaseConfig")


@dataclass
class WandbConfig:
    api_key: str | None = None
    project_name: str | None = None
    run_name: str | None = None
    use_watch: bool = False
    watch_log: WatchLogOption = WatchLogOption.all
    watch_log_freq: int = 1000
    watch_log_graph: bool = True
    enabled: bool = True
    asdict = asdict

    def __post_init__(self):
        # check types
        if self.api_key is not None and not isinstance(self.api_key, str):
            raise TypeError("api_key must be a string or None")
        if self.project_name is not None and not isinstance(self.project_name, str):
            raise TypeError("project_name must be a string or None")
        if self.run_name is not None and not isinstance(self.run_name, str):
            raise TypeError("run_name must be a string or None")
        if not isinstance(self.use_watch, bool):
            raise TypeError("use_watch must be a boolean")
        if not isinstance(self.watch_log, str):
            raise TypeError("watch_log must be a string")
        if not isinstance(self.watch_log_freq, int):
            raise TypeError("watch_log_freq must be an integer")
        if not isinstance(self.watch_log_graph, bool):
            raise TypeError("watch_log_graph must be a boolean")
        # check values
        if self.watch_log is not None and not isinstance(self.watch_log, WatchLogOption):
            raise ValueError(f"watch_log must be one of {WatchLogOption.to_list()}")
        if self.watch_log_freq < 1:
            raise ValueError("watch_log_freq must be positive")


@dataclass
class BaseConfig:
    # Hydra's structured config schema doesn't support
    # generics nor unions of containers (e.g. ChemMRLConfig)
    model: Any
    train_dataset_path: str
    val_dataset_path: str
    test_dataset_path: str | None = None
    smiles_a_column_name: str = "smiles_a"
    smiles_b_column_name: str | None = "smiles_b"
    label_column_name: str = "similarity"
    n_train_samples: int | None = None
    n_val_samples: int | None = None
    n_test_samples: int | None = None
    n_dataloader_workers: int = 0
    pin_memory: bool = False
    multiprocess_context: MultiProcessContextOption | None = None
    generate_dataset_examples_at_init: bool = True
    train_batch_size: int = 32
    eval_batch_size: int = 32
    num_epochs: int = 2
    lr_base: float | int = 2e-05
    weight_decay: float | None = 0.0
    use_normalized_weight_decay: bool = False
    scheduler: SchedulerOption = SchedulerOption.warmuplinear
    warmup_steps_percent: float = 0.0
    use_fused_adamw: bool = False
    use_tf32: bool = False
    use_amp: bool = False
    seed: int | None = 42
    model_output_path: str = "output"
    evaluation_steps: int = 0
    checkpoint_save_steps: int = 0
    checkpoint_save_total_limit: int = 20
    show_progress_bar: bool = True
    wandb: WandbConfig | None = None
    asdict = asdict

    def __post_init__(self):
        # check types
        if not isinstance(self.train_dataset_path, str):
            raise TypeError("train_dataset_path must be a string")
        if not isinstance(self.val_dataset_path, str):
            raise TypeError("val_dataset_path must be a string")
        if not isinstance(self.test_dataset_path, str | None):
            raise TypeError("test_dataset_path must be a string or None")
        if not isinstance(self.smiles_a_column_name, str):
            raise TypeError("smiles_a_column_name must be a string")
        if not isinstance(self.smiles_b_column_name, str | None):
            raise TypeError("smiles_b_column_name must be a string or None")
        if not isinstance(self.label_column_name, str):
            raise TypeError("label_column_name must be a string")
        if self.n_train_samples is not None and not isinstance(self.n_train_samples, int):
            raise TypeError("n_train_samples must be an integer or None")
        if not isinstance(self.n_val_samples, int) and self.n_val_samples is not None:
            raise TypeError("n_val_samples must be an integer or None")
        if self.n_test_samples is not None and not isinstance(self.n_test_samples, int):
            raise TypeError("n_test_samples must be an integer or None")
        if not isinstance(self.n_dataloader_workers, int):
            raise TypeError("n_dataloader_workers must be an integer")
        if not isinstance(self.pin_memory, bool):
            raise TypeError("pin_memory must be a boolean")
        if not isinstance(self.multiprocess_context, str | None):
            raise TypeError("multiprocess_context must be a string pr None")
        if not isinstance(self.generate_dataset_examples_at_init, bool):
            raise TypeError("generate_dataset_examples_at_init must be a boolean")
        if not isinstance(self.train_batch_size, int):
            raise TypeError("train_batch_size must be an integer")
        if not isinstance(self.eval_batch_size, int):
            raise TypeError("eval_batch_size must be an integer")
        if not isinstance(self.num_epochs, int):
            raise TypeError("num_epochs must be an integer")
        if not isinstance(self.lr_base, float | int):
            raise TypeError("lr_base must be a float or int")
        if not isinstance(self.weight_decay, float | None):
            raise TypeError("weight_decay must be a float or None")
        if not isinstance(self.use_normalized_weight_decay, bool):
            raise TypeError("use_normalized_weight_decay must be a boolean")
        if not isinstance(self.scheduler, str):
            raise TypeError("scheduler must be a string")
        if not isinstance(self.warmup_steps_percent, float):
            raise TypeError("warmup_steps_percent must be a float")
        if not isinstance(self.use_fused_adamw, bool):
            raise TypeError("use_fused_adamw must be a boolean")
        if not isinstance(self.use_tf32, bool):
            raise TypeError("use_tf32 must be a boolean")
        if not isinstance(self.use_amp, bool):
            raise TypeError("use_amp must be a boolean")
        if not isinstance(self.seed, int | None):
            raise TypeError("seed must be an integer or None")
        if not isinstance(self.model_output_path, str):
            raise TypeError("model_output_path must be a string")
        if not isinstance(self.evaluation_steps, int):
            raise TypeError("evaluation_steps must be an integer")
        if not isinstance(self.checkpoint_save_steps, int):
            raise TypeError("checkpoint_save_steps must be an integer")
        if not isinstance(self.checkpoint_save_total_limit, int):
            raise TypeError("checkpoint_save_total_limit must be an integer")
        if not isinstance(self.show_progress_bar, bool):
            raise TypeError("show_progress_bar must be a boolean")
        if not isinstance(self.wandb, WandbConfig | None):
            raise TypeError("wandb must be a WandbConfig or None")
        # check values
        if self.train_dataset_path == "":
            raise ValueError("train_dataset_path must be set")
        if self.val_dataset_path == "":
            raise ValueError("val_dataset_path must be set")
        if self.test_dataset_path is not None and self.test_dataset_path == "":
            raise ValueError("test_dataset_path must be set")
        if self.smiles_a_column_name == "":
            raise ValueError("smiles_a_column_name must be set")
        if self.smiles_b_column_name is not None and self.smiles_b_column_name == "":
            raise ValueError("smiles_b_column_name must be set")
        if self.label_column_name == "":
            raise ValueError("label_column_name must be set")
        if self.n_train_samples is not None and self.n_train_samples < 1:
            raise ValueError("n_train_samples must be greater than 0")
        if self.n_val_samples is not None and self.n_val_samples < 1:
            raise ValueError("n_val_samples must be greater than 0")
        if self.n_test_samples is not None and self.n_test_samples < 1:
            raise ValueError("n_test_samples must be greater than 0")
        if self.n_dataloader_workers < 0:
            raise ValueError("n_dataloader_workers must be positive")
        if self.multiprocess_context is not None and not isinstance(
            self.multiprocess_context, MultiProcessContextOption
        ):
            raise ValueError(
                f"multiprocess_context must be one of {MultiProcessContextOption.to_list()}"
            )
        if self.train_batch_size < 1:
            raise ValueError("train_batch_size must be greater than 0")
        if self.eval_batch_size < 1:
            raise ValueError("eval_batch_size must be greater than 0")
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be greater than 0")
        if self.lr_base <= 0:
            raise ValueError("lr_base must be positive")
        if self.weight_decay is not None and self.weight_decay < 0:
            raise ValueError("weight_decay must be positive")
        if self.weight_decay is not None and self.use_normalized_weight_decay:
            raise ValueError("weight_decay and use_normalized_weight_decay cannot be used together")
        if self.weight_decay is None and not self.use_normalized_weight_decay:
            raise ValueError("either weight_decay or use_normalized_weight_decay must be set")
        if not isinstance(self.scheduler, SchedulerOption):
            raise ValueError(f"scheduler must be one of {SchedulerOption.to_list()}")
        if not (0 <= self.warmup_steps_percent <= 1.0):
            raise ValueError("warmup_steps_percent must be between 0 and 1")
        if self.model_output_path == "":
            raise ValueError("model_output_path cannot be empty")
        if self.evaluation_steps < 0:
            raise ValueError("evaluation_steps must be positive")
        if self.checkpoint_save_steps < 0:
            raise ValueError("checkpoint_save_steps must be positive")
        if self.checkpoint_save_total_limit < 0:
            raise ValueError("checkpoint_save_total_limit must be positive")
