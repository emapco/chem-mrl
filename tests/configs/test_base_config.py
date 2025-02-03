# type: ignore
import pytest

from chem_mrl.configs import WandbConfig
from chem_mrl.configs.BaseConfig import _BaseConfig
from chem_mrl.constants import BASE_MODEL_NAME


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


def test_wandb_config_none_values():
    """Test WandbConfig handles None values appropriately"""
    config = WandbConfig(project_name=None, run_name=None)
    assert config.api_key is None
    assert config.project_name is None
    assert config.run_name is None


def test_wandb_config_in_base_config():
    """Test WandbConfig integration in BaseConfig"""
    wandb_config = WandbConfig(api_key="test_key", project_name="test_project")
    base_config = _BaseConfig(use_wandb=True, wandb_config=wandb_config)
    config_dict = base_config.asdict()
    assert config_dict["wandb_config"]["api_key"] == "test_key"
    assert config_dict["wandb_config"]["project_name"] == "test_project"


def test_wandb_config_validation():
    with pytest.raises(ValueError, match="watch_log must be one of"):
        WandbConfig(watch_log="invalid")
    with pytest.raises(ValueError, match="watch_log_freq must be positive"):
        WandbConfig(watch_log_freq=0)


def test_wandb_config_type_validation():
    """Test type validation for WandbConfig parameters"""
    with pytest.raises(TypeError):
        WandbConfig(api_key=123)
    with pytest.raises(TypeError):
        WandbConfig(project_name=123)
    with pytest.raises(TypeError):
        WandbConfig(run_name=123)
    with pytest.raises(TypeError):
        WandbConfig(use_watch=123)
    with pytest.raises(TypeError):
        WandbConfig(watch_log=123)
    with pytest.raises(TypeError):
        WandbConfig(watch_log_freq="123")
    with pytest.raises(TypeError):
        WandbConfig(watch_log_graph=123)


def test_base_config_default_values():
    config = _BaseConfig()
    assert config.train_dataset_path is None
    assert config.val_dataset_path is None
    assert config.test_dataset_path is None
    assert config.n_train_samples is None
    assert config.n_val_samples is None
    assert config.n_test_samples is None
    assert config.n_dataloader_workers == 0
    assert config.generate_dataset_examples_at_init is True
    assert config.model_name == BASE_MODEL_NAME
    assert config.train_batch_size == 32
    assert config.num_epochs == 3
    assert config.lr_base == 1.1190785944700813e-05
    assert config.scheduler == "warmuplinear"
    assert config.warmup_steps_percent == 0.0
    assert config.use_fused_adamw is False
    assert config.use_tf32 is False
    assert config.use_amp is False
    assert config.seed == 42
    assert config.model_output_path == "output"
    assert config.evaluation_steps == 0
    assert config.checkpoint_save_steps == 0
    assert config.checkpoint_save_total_limit == 20
    assert config.use_wandb is False
    assert config.wandb_config is None
    assert config.return_eval_metric is False


def test_base_config_custom_values():
    wandb_config = WandbConfig(api_key="test_key")
    config = _BaseConfig(
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
        test_dataset_path="test.parquet",
        n_train_samples=1000,
        n_val_samples=500,
        n_test_samples=200,
        n_dataloader_workers=4,
        generate_dataset_examples_at_init=False,
        model_name="test",
        train_batch_size=64,
        num_epochs=5,
        lr_base=0.001,
        scheduler="warmupcosine",
        warmup_steps_percent=0.1,
        use_amp=True,
        seed=123,
        model_output_path="custom_output",
        evaluation_steps=100,
        checkpoint_save_steps=500,
        checkpoint_save_total_limit=10,
        return_eval_metric=True,
        use_wandb=True,
        wandb_config=wandb_config,
    )
    assert config.train_dataset_path == "train.parquet"
    assert config.val_dataset_path == "val.parquet"
    assert config.test_dataset_path == "test.parquet"
    assert config.n_train_samples == 1000
    assert config.n_val_samples == 500
    assert config.n_test_samples == 200
    assert config.n_dataloader_workers == 4
    assert config.generate_dataset_examples_at_init is False
    assert config.train_batch_size == 64
    assert config.num_epochs == 5
    assert config.lr_base == 0.001
    assert config.scheduler == "warmupcosine"
    assert config.warmup_steps_percent == 0.1
    assert config.use_amp is True
    assert config.seed == 123
    assert config.model_output_path == "custom_output"
    assert config.evaluation_steps == 100
    assert config.checkpoint_save_steps == 500
    assert config.checkpoint_save_total_limit == 10
    assert config.use_wandb is True
    assert config.wandb_config == wandb_config
    assert config.return_eval_metric is True


def test_base_config_validation():
    with pytest.raises(ValueError, match="model_name must be set"):
        _BaseConfig(
            model_name="",
        )
    with pytest.raises(ValueError, match="train_dataset_path must be set"):
        _BaseConfig(
            train_dataset_path="",
        )
    with pytest.raises(ValueError, match="val_dataset_path must be set"):
        _BaseConfig(
            val_dataset_path="",
        )
    with pytest.raises(ValueError, match="n_train_samples must be positive"):
        _BaseConfig(
            n_train_samples=0,
        )
    with pytest.raises(ValueError, match="n_val_samples must be positive"):
        _BaseConfig(
            n_val_samples=0,
        )
    with pytest.raises(ValueError, match="n_test_samples must be positive"):
        _BaseConfig(
            n_test_samples=0,
        )
    with pytest.raises(ValueError, match="n_dataloader_workers must be positive"):
        _BaseConfig(
            n_dataloader_workers=-1,
        )
    with pytest.raises(ValueError, match="train_batch_size must be positive"):
        _BaseConfig(
            train_batch_size=0,
        )
    with pytest.raises(ValueError, match="num_epochs must be positive"):
        _BaseConfig(
            num_epochs=0,
        )
    with pytest.raises(ValueError, match="lr_base must be positive"):
        _BaseConfig(
            lr_base=0,
        )
    with pytest.raises(ValueError, match="scheduler must be one of"):
        _BaseConfig(
            scheduler="invalid",
        )
    with pytest.raises(
        ValueError, match="warmup_steps_percent must be between 0 and 1"
    ):
        _BaseConfig(
            warmup_steps_percent=1.5,
        )
    with pytest.raises(
        ValueError, match="warmup_steps_percent must be between 0 and 1"
    ):
        _BaseConfig(
            warmup_steps_percent=-0.1,
        )
    with pytest.raises(ValueError, match="model_output_path cannot be empty"):
        _BaseConfig(
            model_output_path="",
        )
    with pytest.raises(ValueError, match="evaluation_steps must be positive"):
        _BaseConfig(
            evaluation_steps=-1,
        )
    with pytest.raises(ValueError, match="checkpoint_save_steps must be positive"):
        _BaseConfig(
            checkpoint_save_steps=-1,
        )
    with pytest.raises(
        ValueError, match="checkpoint_save_total_limit must be positive"
    ):
        _BaseConfig(
            checkpoint_save_total_limit=-1,
        )


def test_config_asdict():
    wandb_config = WandbConfig()
    base_config = _BaseConfig()
    wandb_dict = wandb_config.asdict()
    base_dict = base_config.asdict()
    assert isinstance(wandb_dict, dict)
    assert isinstance(base_dict, dict)
    assert "api_key" in wandb_dict
    assert "train_batch_size" in base_dict


def test_base_config_with_none_wandb():
    """Test BaseConfig with None WandbConfig when use_wandb is True"""
    with pytest.raises(
        ValueError, match="wandb_config must be provided when use_wandb is True"
    ):
        _BaseConfig(use_wandb=True, wandb_config=None)


def test_configs_immutable_defaults():
    """Test that default values remain unchanged between instances"""
    config1 = _BaseConfig()
    config2 = _BaseConfig()
    assert config1.train_batch_size == config2.train_batch_size
    assert config1.num_epochs == config2.num_epochs
    assert id(config1) != id(config2)


def test_base_config_type_validation():
    """Test type validation for base config parameters"""
    with pytest.raises(TypeError):
        _BaseConfig(model_name=1)
    with pytest.raises(TypeError):
        _BaseConfig(train_dataset_path=123)
    with pytest.raises(TypeError):
        _BaseConfig(val_dataset_path=123)
    with pytest.raises(TypeError):
        _BaseConfig(test_dataset_path=123)
    with pytest.raises(TypeError):
        _BaseConfig(n_train_samples=1.5)
    with pytest.raises(TypeError):
        _BaseConfig(n_val_samples=1.5)
    with pytest.raises(TypeError):
        _BaseConfig(n_test_samples=1.5)
    with pytest.raises(TypeError):
        _BaseConfig(n_dataloader_workers=1.5)
    with pytest.raises(TypeError):
        _BaseConfig(generate_dataset_examples_at_init=1.5)
    with pytest.raises(TypeError):
        _BaseConfig(n_train_samples="123")
    with pytest.raises(TypeError):
        _BaseConfig(n_val_samples="123")
    with pytest.raises(TypeError):
        _BaseConfig(n_test_samples="123")
    with pytest.raises(TypeError):
        _BaseConfig(train_batch_size="123")
    with pytest.raises(TypeError):
        _BaseConfig(num_epochs="123")
    with pytest.raises(TypeError):
        _BaseConfig(wandb_config="123")
    with pytest.raises(TypeError):
        _BaseConfig(lr_base="123")
    with pytest.raises(TypeError):
        _BaseConfig(scheduler=123)
    with pytest.raises(TypeError):
        _BaseConfig(warmup_steps_percent="123")
    with pytest.raises(TypeError):
        _BaseConfig(use_fused_adamw=123)
    with pytest.raises(TypeError):
        _BaseConfig(use_tf32=123)
    with pytest.raises(TypeError):
        _BaseConfig(use_amp=123)
    with pytest.raises(TypeError):
        _BaseConfig(seed="123")
    with pytest.raises(TypeError):
        _BaseConfig(model_output_path=123)
    with pytest.raises(TypeError):
        _BaseConfig(evaluation_steps="123")
    with pytest.raises(TypeError):
        _BaseConfig(checkpoint_save_steps="123")
    with pytest.raises(TypeError):
        _BaseConfig(checkpoint_save_total_limit="123")
    with pytest.raises(TypeError):
        _BaseConfig(return_eval_metric=123)
