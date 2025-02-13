# type: ignore
import pytest

from chem_mrl.schemas import ChemMRLConfig
from chem_mrl.schemas.BaseConfig import (
    BaseConfig,
    MultiProcessContextOption,
    SchedulerOption,
    WandbConfig,
    WatchLogOption,
)


def test_wandb_config_custom_values():
    config = WandbConfig(
        api_key="test_key",
        project_name="test_project",
        run_name="test_run",
        use_watch=True,
        watch_log=WatchLogOption.gradients,
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
    base_config = BaseConfig(
        model=ChemMRLConfig(),
        train_dataset_path="test",
        val_dataset_path="test",
        wandb=wandb_config,
    )
    config_dict = base_config.asdict()
    assert config_dict["wandb"]["api_key"] == "test_key"
    assert config_dict["wandb"]["project_name"] == "test_project"


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


@pytest.mark.parametrize("scheduler", SchedulerOption)
def test_scheduler_option(scheduler):
    """Test SchedulerOption enum"""
    assert isinstance(scheduler, SchedulerOption)
    assert scheduler in SchedulerOption


def test_base_config_custom_values():
    wandb_config = WandbConfig(api_key="test_key")
    config = BaseConfig(
        model=ChemMRLConfig(),
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
        test_dataset_path="test.parquet",
        smiles_a_column_name="asdf",
        smiles_b_column_name=None,
        label_column_name="adsf",
        n_train_samples=1000,
        n_val_samples=500,
        n_test_samples=200,
        n_dataloader_workers=4,
        pin_memory=True,
        multiprocess_context=MultiProcessContextOption.fork,
        generate_dataset_examples_at_init=False,
        train_batch_size=64,
        eval_batch_size=128,
        num_epochs=5,
        lr_base=0.001,
        weight_decay=0.1,
        use_normalized_weight_decay=False,
        scheduler=SchedulerOption.warmupcosine,
        warmup_steps_percent=0.1,
        use_amp=True,
        seed=123,
        model_output_path="custom_output",
        evaluation_steps=100,
        checkpoint_save_steps=500,
        checkpoint_save_total_limit=10,
        show_progress_bar=False,
        wandb=wandb_config,
    )
    assert config.train_dataset_path == "train.parquet"
    assert config.val_dataset_path == "val.parquet"
    assert config.test_dataset_path == "test.parquet"
    assert config.smiles_a_column_name == "asdf"
    assert config.smiles_b_column_name is None
    assert config.label_column_name == "adsf"
    assert config.n_train_samples == 1000
    assert config.n_val_samples == 500
    assert config.n_test_samples == 200
    assert config.n_dataloader_workers == 4
    assert config.pin_memory is True
    assert config.multiprocess_context == MultiProcessContextOption.fork
    assert config.generate_dataset_examples_at_init is False
    assert config.train_batch_size == 64
    assert config.eval_batch_size == 128
    assert config.num_epochs == 5
    assert config.lr_base == 0.001
    assert config.weight_decay == 0.1
    assert config.use_normalized_weight_decay is False
    assert config.scheduler == "warmupcosine"
    assert config.warmup_steps_percent == 0.1
    assert config.use_amp is True
    assert config.seed == 123
    assert config.model_output_path == "custom_output"
    assert config.evaluation_steps == 100
    assert config.checkpoint_save_steps == 500
    assert config.checkpoint_save_total_limit == 10
    assert config.show_progress_bar is False
    assert config.wandb == wandb_config


def test_base_config_validation():
    with pytest.raises(ValueError, match="train_dataset_path must be set"):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="",
            val_dataset_path="test",
        )
    with pytest.raises(ValueError, match="val_dataset_path must be set"):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="test",
            val_dataset_path="",
        )
    with pytest.raises(ValueError, match="test_dataset_path must be set"):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="test",
            val_dataset_path="test",
            test_dataset_path="",
        )
    with pytest.raises(ValueError, match="smiles_a_column_name must be set"):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="test",
            val_dataset_path="test",
            smiles_a_column_name="",
        )
    with pytest.raises(ValueError, match="smiles_b_column_name must be set"):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="test",
            val_dataset_path="test",
            smiles_b_column_name="",
        )
    with pytest.raises(ValueError, match="label_column_name must be set"):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="test",
            val_dataset_path="test",
            label_column_name="",
        )
    with pytest.raises(ValueError, match="n_train_samples must be greater than 0"):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="test",
            val_dataset_path="test",
            n_train_samples=0,
        )
    with pytest.raises(ValueError, match="n_val_samples must be greater than 0"):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="test",
            val_dataset_path="test",
            n_val_samples=0,
        )
    with pytest.raises(ValueError, match="n_test_samples must be greater than 0"):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="test",
            val_dataset_path="test",
            n_test_samples=0,
        )
    with pytest.raises(ValueError, match="n_dataloader_workers must be positive"):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="test",
            val_dataset_path="test",
            n_dataloader_workers=-1,
        )
    with pytest.raises(ValueError, match="multiprocess_context must be one of"):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="test",
            val_dataset_path="test",
            multiprocess_context="invalid",
        )
    with pytest.raises(ValueError, match="train_batch_size must be greater than 0"):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="test",
            val_dataset_path="test",
            train_batch_size=0,
        )
    with pytest.raises(ValueError, match="eval_batch_size must be greater than 0"):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="test",
            val_dataset_path="test",
            eval_batch_size=0,
        )
    with pytest.raises(ValueError, match="num_epochs must be greater than 0"):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="test",
            val_dataset_path="test",
            num_epochs=0,
        )
    with pytest.raises(ValueError, match="lr_base must be positive"):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="test",
            val_dataset_path="test",
            lr_base=0,
        )
    with pytest.raises(ValueError, match="weight_decay must be positive"):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="test",
            val_dataset_path="test",
            weight_decay=-0.1,
        )
    with pytest.raises(ValueError, match="scheduler must be one of"):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="test",
            val_dataset_path="test",
            scheduler="invalid",
        )
    with pytest.raises(ValueError, match="warmup_steps_percent must be between 0 and 1"):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="test",
            val_dataset_path="test",
            warmup_steps_percent=1.5,
        )
    with pytest.raises(ValueError, match="warmup_steps_percent must be between 0 and 1"):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="test",
            val_dataset_path="test",
            warmup_steps_percent=-0.1,
        )
    with pytest.raises(ValueError, match="model_output_path cannot be empty"):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="test",
            val_dataset_path="test",
            model_output_path="",
        )
    with pytest.raises(ValueError, match="evaluation_steps must be positive"):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="test",
            val_dataset_path="test",
            evaluation_steps=-1,
        )
    with pytest.raises(ValueError, match="checkpoint_save_steps must be positive"):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="test",
            val_dataset_path="test",
            checkpoint_save_steps=-1,
        )
    with pytest.raises(ValueError, match="checkpoint_save_total_limit must be positive"):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="test",
            val_dataset_path="test",
            checkpoint_save_total_limit=-1,
        )


def test_base_config_weight_decay():
    with pytest.raises(
        ValueError, match="weight_decay and use_normalized_weight_decay cannot be used together"
    ):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="test",
            val_dataset_path="test",
            weight_decay=0.0,
            use_normalized_weight_decay=True,
        )
    with pytest.raises(
        ValueError, match="either weight_decay or use_normalized_weight_decay must be set"
    ):
        BaseConfig(
            model=ChemMRLConfig(),
            train_dataset_path="test",
            val_dataset_path="test",
            weight_decay=None,
            use_normalized_weight_decay=False,
        )


def test_config_asdict():
    wandb_config = WandbConfig()
    base_config = BaseConfig(
        model=ChemMRLConfig(), train_dataset_path="test", val_dataset_path="test"
    )
    wandb_dict = wandb_config.asdict()
    base_dict = base_config.asdict()
    assert isinstance(wandb_dict, dict)
    assert isinstance(base_dict, dict)
    assert "api_key" in wandb_dict
    assert "train_batch_size" in base_dict


def test_base_config_type_validation():
    """Test type validation for base config parameters"""
    with pytest.raises(TypeError):
        BaseConfig(train_dataset_path=123)
    with pytest.raises(TypeError):
        BaseConfig(val_dataset_path=123)
    with pytest.raises(TypeError):
        BaseConfig(test_dataset_path=123)
    with pytest.raises(TypeError):
        ChemMRLConfig(smiles_a_column_name=1)
    with pytest.raises(TypeError):
        ChemMRLConfig(smiles_b_column_name=1)
    with pytest.raises(TypeError):
        ChemMRLConfig(label_column_name=1)
    with pytest.raises(TypeError):
        BaseConfig(n_train_samples=1.5)
    with pytest.raises(TypeError):
        BaseConfig(n_val_samples=1.5)
    with pytest.raises(TypeError):
        BaseConfig(n_test_samples=1.5)
    with pytest.raises(TypeError):
        BaseConfig(n_dataloader_workers=1.5)
    with pytest.raises(TypeError):
        BaseConfig(pin_memory=1.5)
    with pytest.raises(TypeError):
        BaseConfig(multiprocess_context=1.5)
    with pytest.raises(TypeError):
        BaseConfig(generate_dataset_examples_at_init=1.5)
    with pytest.raises(TypeError):
        BaseConfig(n_train_samples="123")
    with pytest.raises(TypeError):
        BaseConfig(n_val_samples="123")
    with pytest.raises(TypeError):
        BaseConfig(n_test_samples="123")
    with pytest.raises(TypeError):
        BaseConfig(train_batch_size="123")
    with pytest.raises(TypeError):
        BaseConfig(eval_batch_size="123")
    with pytest.raises(TypeError):
        BaseConfig(num_epochs="123")
    with pytest.raises(TypeError):
        BaseConfig(wandb_config="123")
    with pytest.raises(TypeError):
        BaseConfig(lr_base="123")
    with pytest.raises(TypeError):
        BaseConfig(weight_decay="123")
    with pytest.raises(TypeError):
        BaseConfig(use_normalized_weight_decay=123)
    with pytest.raises(TypeError):
        BaseConfig(scheduler=123)
    with pytest.raises(TypeError):
        BaseConfig(warmup_steps_percent="123")
    with pytest.raises(TypeError):
        BaseConfig(use_fused_adamw=123)
    with pytest.raises(TypeError):
        BaseConfig(use_tf32=123)
    with pytest.raises(TypeError):
        BaseConfig(use_amp=123)
    with pytest.raises(TypeError):
        BaseConfig(seed="123")
    with pytest.raises(TypeError):
        BaseConfig(model_output_path=123)
    with pytest.raises(TypeError):
        BaseConfig(evaluation_steps="123")
    with pytest.raises(TypeError):
        BaseConfig(checkpoint_save_steps="123")
    with pytest.raises(TypeError):
        BaseConfig(checkpoint_save_total_limit="123")
    with pytest.raises(TypeError):
        BaseConfig(wandb="invalid")
    with pytest.raises(TypeError):
        BaseConfig(show_progress_bar="invalid")
