from __future__ import annotations

import logging

from sentence_transformers import LoggingHandler

__version__ = "0.4.0"

from . import (
    benchmark,
    constants,
    datasets,
    device_manager,
    evaluation,
    losses,
    molecular_embedder,
    molecular_fingerprinter,
    schemas,
    trainers,
)

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


__all__ = [
    "benchmark",
    "constants",
    "datasets",
    "device_manager",
    "evaluation",
    "losses",
    "molecular_embedder",
    "molecular_fingerprinter",
    "schemas",
    "trainers",
]
