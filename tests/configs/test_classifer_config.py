import pytest

from chem_mrl.configs import ClassifierConfig, DiceLossClassifierConfig
from chem_mrl.constants import CHEM_MRL_DIMENSIONS, MODEL_NAME_KEYS, MODEL_NAMES


def test_classifier_config_default_values():
    config = ClassifierConfig(
        train_dataset_path="train.csv", val_dataset_path="val.csv"
    )
    assert config.model_name == MODEL_NAMES[MODEL_NAME_KEYS[1]]
    assert config.smiles_column_name == "smiles"
    assert config.label_column_name == "label"
    assert config.loss_func == "softmax"
    assert config.classifier_hidden_dimension == CHEM_MRL_DIMENSIONS[0]
    assert config.dropout_p == 0.15
    assert config.freeze_model is False


def test_classifier_config_custom_values():
    config = ClassifierConfig(
        model_name=MODEL_NAMES[MODEL_NAME_KEYS[0]],
        train_dataset_path="custom_train.csv",
        val_dataset_path="custom_val.csv",
        smiles_column_name="smiles_col",
        label_column_name="labels_col",
        loss_func="selfadjdice",
        classifier_hidden_dimension=CHEM_MRL_DIMENSIONS[1],
        dropout_p=0.3,
        freeze_model=True,
    )

    assert config.model_name == MODEL_NAMES[MODEL_NAME_KEYS[0]]
    assert config.train_dataset_path == "custom_train.csv"
    assert config.val_dataset_path == "custom_val.csv"
    assert config.smiles_column_name == "smiles_col"
    assert config.label_column_name == "labels_col"
    assert config.loss_func == "selfadjdice"
    assert config.classifier_hidden_dimension == CHEM_MRL_DIMENSIONS[1]
    assert config.dropout_p == 0.3
    assert config.freeze_model is True


def test_classifier_config_validation():
    with pytest.raises(ValueError, match="model_name must be set"):
        ClassifierConfig(
            model_name="", train_dataset_path="train.csv", val_dataset_path="val.csv"
        )

    with pytest.raises(ValueError, match="train_dataset_path must be set"):
        ClassifierConfig(train_dataset_path="", val_dataset_path="val.csv")

    with pytest.raises(ValueError, match="val_dataset_path must be set"):
        ClassifierConfig(train_dataset_path="train.csv", val_dataset_path="")

    with pytest.raises(ValueError, match="smiles_column_name must be set"):
        ClassifierConfig(
            train_dataset_path="train.csv",
            val_dataset_path="val.csv",
            smiles_column_name="",
        )

    with pytest.raises(ValueError, match="label_column_name must be set"):
        ClassifierConfig(
            train_dataset_path="train.csv",
            val_dataset_path="val.csv",
            label_column_name="",
        )

    with pytest.raises(ValueError, match="loss_func must be one of"):
        ClassifierConfig(
            train_dataset_path="train.csv",
            val_dataset_path="val.csv",
            loss_func="invalid",
        )

    with pytest.raises(ValueError, match="classifier_hidden_dimension must be one of"):
        ClassifierConfig(
            train_dataset_path="train.csv",
            val_dataset_path="val.csv",
            classifier_hidden_dimension=999,
        )

    with pytest.raises(ValueError, match="dropout_p must be between 0 and 1"):
        ClassifierConfig(
            train_dataset_path="train.csv", val_dataset_path="val.csv", dropout_p=1.5
        )


def test_dice_loss_classifier_config_default_values():
    config = DiceLossClassifierConfig(
        train_dataset_path="train.csv", val_dataset_path="val.csv"
    )
    assert config.dice_reduction == "mean"
    assert config.dice_gamma == 1.0


def test_dice_loss_classifier_config_custom_values():
    config = DiceLossClassifierConfig(
        train_dataset_path="train.csv",
        val_dataset_path="val.csv",
        dice_reduction="sum",
        dice_gamma=2.0,
    )
    assert config.dice_reduction == "sum"
    assert config.dice_gamma == 2.0


def test_dice_loss_classifier_config_validation():
    with pytest.raises(ValueError, match="dice_gamma must be positive"):
        DiceLossClassifierConfig(
            train_dataset_path="train.csv", val_dataset_path="val.csv", dice_gamma=-1.0
        )

    with pytest.raises(
        ValueError, match="dice_reduction must be either 'mean' or 'sum'"
    ):
        DiceLossClassifierConfig(
            train_dataset_path="train.csv",
            val_dataset_path="val.csv",
            dice_reduction="invalid",
        )


def test_classifier_configs_asdict():
    classifier_config = ClassifierConfig(
        train_dataset_path="train.csv", val_dataset_path="val.csv"
    )
    dice_config = DiceLossClassifierConfig(
        train_dataset_path="train.csv", val_dataset_path="val.csv"
    )

    classifier_dict = classifier_config.asdict()
    dice_dict = dice_config.asdict()

    assert isinstance(classifier_dict, dict)
    assert isinstance(dice_dict, dict)
    assert "model_name" in classifier_dict
    assert "dice_gamma" in dice_dict
