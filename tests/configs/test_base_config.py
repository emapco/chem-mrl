import pytest

from chem_mrl.configs import BaseConfig, WandbConfig


def test_wandb_config_default_values():
    config = WandbConfig()
    assert config.api_key is None
    assert config.project_name is None
    assert config.run_name is None
    assert config.use_watch is False
    assert config.watch_log == "all"
    assert config.watch_log_freq == 1000
    assert config.watch_log_graph is True


def test_wandb_config_custom_values():
    config = WandbConfig(
        api_key="test_key",
        project_name="test_project",
        run_name="test_run",
        use_watch=True,
        watch_log="gradients",
        watch_log_freq=500,
        watch_log_graph=False,
    )
    assert config.api_key == "test_key"
    assert config.project_name == "test_project"
    assert config.run_name == "test_run"
    assert config.use_watch is True
    assert config.watch_log == "gradients"
    assert config.watch_log_freq == 500
    assert config.watch_log_graph is False


def test_wandb_config_validation():
    with pytest.raises(ValueError, match="watch_log must be one of"):
        WandbConfig(watch_log="invalid")

    with pytest.raises(ValueError, match="watch_log_freq must be positive"):
        WandbConfig(watch_log_freq=0)


def test_base_config_default_values():
    config = BaseConfig()
    assert config.train_batch_size == 32
    assert config.num_epochs == 3
    assert config.use_wandb is False
    assert config.wandb_config is None
    assert config.lr_base == 1.1190785944700813e-05
    assert config.scheduler == "warmuplinear"
    assert config.warmup_steps_percent == 0.0
    assert config.use_amp is False
    assert config.seed == 42
    assert config.model_output_path == "output"
    assert config.evaluation_steps == 0
    assert config.checkpoint_save_steps == 1000000
    assert config.checkpoint_save_total_limit == 20


def test_base_config_custom_values():
    wandb_config = WandbConfig(api_key="test_key")
    config = BaseConfig(
        train_batch_size=64,
        num_epochs=5,
        use_wandb=True,
        wandb_config=wandb_config,
        lr_base=0.001,
        scheduler="warmupcosine",
        warmup_steps_percent=0.1,
        use_amp=True,
        seed=123,
        model_output_path="custom_output",
        evaluation_steps=100,
        checkpoint_save_steps=500,
        checkpoint_save_total_limit=10,
    )

    assert config.train_batch_size == 64
    assert config.num_epochs == 5
    assert config.use_wandb is True
    assert config.wandb_config == wandb_config
    assert config.lr_base == 0.001
    assert config.scheduler == "warmupcosine"
    assert config.warmup_steps_percent == 0.1
    assert config.use_amp is True
    assert config.seed == 123
    assert config.model_output_path == "custom_output"
    assert config.evaluation_steps == 100
    assert config.checkpoint_save_steps == 500
    assert config.checkpoint_save_total_limit == 10


def test_base_config_validation():
    with pytest.raises(ValueError, match="train_batch_size must be positive"):
        BaseConfig(train_batch_size=0)

    with pytest.raises(ValueError, match="num_epochs must be positive"):
        BaseConfig(num_epochs=0)

    with pytest.raises(ValueError, match="lr_base must be positive"):
        BaseConfig(lr_base=0)

    with pytest.raises(ValueError, match="scheduler must be one of"):
        BaseConfig(scheduler="invalid")

    with pytest.raises(
        ValueError, match="warmup_steps_percent must be between 0 and 1"
    ):
        BaseConfig(warmup_steps_percent=1.5)

    with pytest.raises(
        ValueError, match="warmup_steps_percent must be between 0 and 1"
    ):
        BaseConfig(warmup_steps_percent=-0.1)

    with pytest.raises(ValueError, match="seed must be an integer"):
        BaseConfig(seed="invalid")

    with pytest.raises(ValueError, match="evaluation_steps must be positive"):
        BaseConfig(evaluation_steps=-1)

    with pytest.raises(ValueError, match="checkpoint_save_steps must be positive"):
        BaseConfig(checkpoint_save_steps=-1)

    with pytest.raises(
        ValueError, match="checkpoint_save_total_limit must be positive"
    ):
        BaseConfig(checkpoint_save_total_limit=-1)


def test_config_asdict():
    wandb_config = WandbConfig()
    base_config = BaseConfig()

    wandb_dict = wandb_config.asdict()
    base_dict = base_config.asdict()

    assert isinstance(wandb_dict, dict)
    assert isinstance(base_dict, dict)
    assert "api_key" in wandb_dict
    assert "train_batch_size" in base_dict
