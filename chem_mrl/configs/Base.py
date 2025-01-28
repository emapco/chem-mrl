from dataclasses import asdict, dataclass
from typing import Literal

_watch_log_option_type = Literal["gradients", "parameters", "all"]
_watch_log_options: tuple[_watch_log_option_type, ...] = (
    "gradients",
    "parameters",
    "all",
)
_scheduler_option_type = Literal[
    "warmupconstant",
    "warmuplinear",
    "warmupcosine",
    "warmupcosinewithhardrestarts",
]
_scheduler_options: tuple[_scheduler_option_type, ...] = (
    "warmupconstant",
    "warmuplinear",
    "warmupcosine",
    "warmupcosinewithhardrestarts",
)


@dataclass
class WandbConfig:
    api_key: str | None = None
    project_name: str | None = None
    run_name: str | None = None
    use_watch: bool = False
    watch_log: _watch_log_option_type | None = "all"
    watch_log_freq: int = 1000
    watch_log_graph: bool = True
    asdict = asdict

    def __post_init__(self):
        if self.watch_log is not None and self.watch_log not in _watch_log_options:
            raise ValueError(f"watch_log must be one of {_watch_log_options}")
        if self.watch_log_freq < 1:
            raise ValueError("watch_log_freq must be positive")


@dataclass
class BaseConfig:
    train_batch_size: int = 32
    num_epochs: int = 3
    use_wandb: bool = False
    wandb_config: WandbConfig | None = None
    lr_base: float = 1.1190785944700813e-05
    scheduler: _scheduler_option_type = "warmuplinear"
    warmup_steps_percent: float = 0.0
    use_amp: bool = False
    seed: int | None = 42
    model_output_path: str = "output"
    evaluation_steps: int = 0
    checkpoint_save_steps: int = 1000000
    checkpoint_save_total_limit: int = 20
    asdict = asdict

    def __post_init__(self):
        if self.train_batch_size < 1:
            raise ValueError("train_batch_size must be positive")
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be positive")
        if self.lr_base <= 0:
            raise ValueError("lr_base must be positive")
        if self.scheduler not in _scheduler_options:
            raise ValueError(f"scheduler must be one of {_scheduler_options}")
        if not (0 <= self.warmup_steps_percent <= 1.0):
            raise ValueError("warmup_steps_percent must be between 0 and 1")
        try:
            if self.seed is not None:
                self.seed = int(self.seed)
        except ValueError:
            raise ValueError("seed must be an integer")
        if self.evaluation_steps < 0:
            raise ValueError("evaluation_steps must be positive")
        if self.checkpoint_save_steps < 0:
            raise ValueError("checkpoint_save_steps must be positive")
        if self.checkpoint_save_total_limit < 0:
            raise ValueError("checkpoint_save_total_limit must be positive")
