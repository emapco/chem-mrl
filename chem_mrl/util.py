import logging
import os
import queue

import torch
from sentence_transformers import LoggingHandler


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )


def get_file_extension(filename: str) -> str:
    """Extract file extension and map to dataset format."""
    ext_mapping = {".parquet": "parquet", ".csv": "csv", ".json": "json", ".jsonl": "json"}
    ext = os.path.splitext(filename)[1].lower()
    return ext_mapping.get(ext, "parquet")  # default fallback


class CudaDeviceManager:
    def __init__(self, max_cpu_processes_fallback: int = 1, use_logging: bool = True):
        if max_cpu_processes_fallback < 1:
            raise ValueError("max_cpu_processes must be at least 1")
        self.use_logging = use_logging

        if torch.cuda.is_available():
            self._num_cuda_devices = torch.cuda.device_count()
            self._num_processes = self._num_cuda_devices
            if self.use_logging:
                logging.info(f"Using {self._num_processes} CUDA device(s)")
        else:
            self._num_cuda_devices = 0
            self._num_processes = max_cpu_processes_fallback
            if self.use_logging:
                logging.info(
                    "No CUDA devices detected. "
                    f"Falling back to using {self._num_processes} CPU process(es)"
                )

        self._device_id_queue = queue.Queue(maxsize=self._num_processes)
        for i in range(max(self._num_processes, 1)):
            self._device_id_queue.put(i)

    @property
    def num_processes(self) -> int:
        return self._num_processes

    def get_device(self) -> str:
        device_id = self._device_id_queue.get()

        if self._num_cuda_devices == 0:
            if self.use_logging:
                logging.info("No CUDA devices available. Using CPU.")
            return "cpu"

        if self.use_logging:
            logging.info(f"Acquired device: cuda:{device_id}")
        return f"cuda:{device_id}"

    def release_device(self, device: str) -> None:
        if self._num_cuda_devices == 0:
            logging.info("Releasing CPU device.")
            self._device_id_queue.put(device)
            return

        # when using cuda the device is formatted as cuda:id as required by PyTorch
        device_id = int(device.split(":")[-1])
        self._device_id_queue.put(device_id)
        if self.use_logging:
            logging.info(f"Released device: {device}")
