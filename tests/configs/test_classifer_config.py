# type: ignore
import pytest

from chem_mrl.configs import ClassifierConfig, DiceLossClassifierConfig
from chem_mrl.constants import CHEM_MRL_DIMENSIONS, MODEL_NAME_KEYS, MODEL_NAMES


def test_classifier_config_default_values():
    config = ClassifierConfig(
        model_name=MODEL_NAMES[MODEL_NAME_KEYS[1]],
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
    )
    assert config.smiles_column_name == "smiles"
    assert config.label_column_name == "label"
    assert config.eval_metric == "accuracy"
    assert config.loss_func == "softmax"
    assert config.classifier_hidden_dimension == CHEM_MRL_DIMENSIONS[0]
    assert config.dropout_p == 0.1
    assert config.freeze_model is False


def test_classifier_config_custom_values():
    config = ClassifierConfig(
        model_name=MODEL_NAMES[MODEL_NAME_KEYS[0]],
        train_dataset_path="custom_train.parquet",
        val_dataset_path="custom_val.parquet",
        smiles_column_name="smiles_col",
        label_column_name="labels_col",
        loss_func="selfadjdice",
        classifier_hidden_dimension=CHEM_MRL_DIMENSIONS[1],
        dropout_p=0.3,
        freeze_model=True,
    )

    assert config.model_name == MODEL_NAMES[MODEL_NAME_KEYS[0]]
    assert config.train_dataset_path == "custom_train.parquet"
    assert config.val_dataset_path == "custom_val.parquet"
    assert config.smiles_column_name == "smiles_col"
    assert config.label_column_name == "labels_col"
    assert config.loss_func == "selfadjdice"
    assert config.classifier_hidden_dimension == CHEM_MRL_DIMENSIONS[1]
    assert config.dropout_p == 0.3
    assert config.freeze_model is True


def test_classifier_config_validation():
    with pytest.raises(ValueError, match="smiles_column_name must be set"):
        ClassifierConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            smiles_column_name="",
        )
    with pytest.raises(ValueError, match="label_column_name must be set"):
        ClassifierConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            label_column_name="",
        )
    with pytest.raises(ValueError, match="eval_metric must be one of"):
        ClassifierConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            eval_metric="invalid_metric",
        )
    with pytest.raises(ValueError, match="loss_func must be one of"):
        ClassifierConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            loss_func="invalid",
        )
    with pytest.raises(
        ValueError, match="classifier_hidden_dimension must be greater than 0"
    ):
        ClassifierConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            classifier_hidden_dimension=0,
        )
    with pytest.raises(ValueError, match="dropout_p must be between 0 and 1"):
        ClassifierConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            dropout_p=1.5,
        )


def test_dice_loss_classifier_config_default_values():
    config = DiceLossClassifierConfig(
        model_name=MODEL_NAMES[MODEL_NAME_KEYS[1]],
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
    )
    assert config.dice_reduction == "mean"
    assert config.dice_gamma == 1.0


def test_dice_loss_classifier_config_custom_values():
    config = DiceLossClassifierConfig(
        model_name=MODEL_NAMES[MODEL_NAME_KEYS[1]],
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
        dice_reduction="sum",
        dice_gamma=2.0,
    )
    assert config.dice_reduction == "sum"
    assert config.dice_gamma == 2.0


def test_dice_loss_classifier_config_validation():
    with pytest.raises(ValueError, match="dice_gamma must be positive"):
        DiceLossClassifierConfig(
            model_name=MODEL_NAMES[MODEL_NAME_KEYS[1]],
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            dice_gamma=-1.0,
        )
    with pytest.raises(
        ValueError, match="dice_reduction must be either 'mean' or 'sum'"
    ):
        DiceLossClassifierConfig(
            model_name=MODEL_NAMES[MODEL_NAME_KEYS[1]],
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            dice_reduction="invalid",
        )


def test_classifier_configs_asdict():
    classifier_config = ClassifierConfig(
        model_name=MODEL_NAMES[MODEL_NAME_KEYS[1]],
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
    )
    dice_config = DiceLossClassifierConfig(
        model_name=MODEL_NAMES[MODEL_NAME_KEYS[1]],
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
    )

    classifier_dict = classifier_config.asdict()
    dice_dict = dice_config.asdict()

    assert isinstance(classifier_dict, dict)
    assert isinstance(dice_dict, dict)
    assert "model_name" in classifier_dict
    assert "dice_gamma" in dice_dict


def test_classifier_config_dimension_boundaries():
    """Test classifier dimension boundary values"""
    config = ClassifierConfig(
        model_name=MODEL_NAMES[MODEL_NAME_KEYS[1]],
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
        classifier_hidden_dimension=CHEM_MRL_DIMENSIONS[-1],
    )
    assert config.classifier_hidden_dimension == CHEM_MRL_DIMENSIONS[-1]


def test_classifier_config_dropout_boundaries():
    """Test dropout probability boundary values"""
    config = ClassifierConfig(
        model_name=MODEL_NAMES[MODEL_NAME_KEYS[1]],
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
        dropout_p=0.0,
    )
    assert config.dropout_p == 0.0

    config = ClassifierConfig(
        model_name=MODEL_NAMES[MODEL_NAME_KEYS[1]],
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
        dropout_p=1.0,
    )
    assert config.dropout_p == 1.0


def test_classifier_config_all_model_names():
    """Test configuration with all valid model names"""
    for key in MODEL_NAME_KEYS:
        config = ClassifierConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            model_name=MODEL_NAMES[key],
        )
        assert config.model_name == MODEL_NAMES[key]


def test_dice_loss_classifier_config_combined():
    """Test combined configuration settings"""
    config = DiceLossClassifierConfig(
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
        model_name=MODEL_NAMES[MODEL_NAME_KEYS[1]],
        dice_reduction="mean",
        dice_gamma=1.5,
        classifier_hidden_dimension=CHEM_MRL_DIMENSIONS[0],
        dropout_p=0.5,
        freeze_model=True,
    )
    assert config.dice_reduction == "mean"
    assert config.dice_gamma == 1.5
    assert config.classifier_hidden_dimension == CHEM_MRL_DIMENSIONS[0]
    assert config.dropout_p == 0.5
    assert config.freeze_model is True


def test_classifier_config_type_validation():
    """Test type validation for classifier config parameters"""
    with pytest.raises(TypeError):
        ClassifierConfig(
            model_name="test",
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            smiles_column_name=1,
        )
    with pytest.raises(TypeError):
        ClassifierConfig(
            model_name="test",
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            label_column_name=1,
        )
    with pytest.raises(TypeError):
        ClassifierConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            eval_similarity_fct=1,
        )
    with pytest.raises(TypeError):
        ClassifierConfig(
            model_name="test",
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            loss_func=1,
        )
    with pytest.raises(TypeError):
        ClassifierConfig(
            model_name="test",
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            classifier_hidden_dimension="1",
        )
    with pytest.raises(TypeError):
        ClassifierConfig(
            model_name="test",
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            dropout_p="1",
        )


def test_dice_loss_classifier_config_type_validation():
    """Test type validation for dice loss classifier config parameters"""
    with pytest.raises(TypeError):
        DiceLossClassifierConfig(
            model_name="test",
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            dice_reduction=1,
        )
    with pytest.raises(TypeError):
        DiceLossClassifierConfig(
            model_name="test",
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            dice_gamma="1",
        )
