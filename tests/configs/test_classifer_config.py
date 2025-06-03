# type: ignore
import pytest

from chem_mrl.constants import CHEM_MRL_DIMENSIONS
from chem_mrl.schemas.ClassifierConfig import (
    ClassifierConfig,
    ClassifierLossFctOption,
    DiceReductionOption,
)


def test_classifier_config_custom_values():
    config = ClassifierConfig(
        loss_func=ClassifierLossFctOption.selfadjdice,
        classifier_hidden_dimension=CHEM_MRL_DIMENSIONS[1],
        dropout_p=0.3,
        freeze_model=True,
        num_labels=3,
        dice_reduction=DiceReductionOption.sum,
        dice_gamma=2.0,
    )
    assert config.loss_func == "selfadjdice"
    assert config.classifier_hidden_dimension == CHEM_MRL_DIMENSIONS[1]
    assert config.dropout_p == 0.3
    assert config.freeze_model is True
    assert config.num_labels == 3
    assert config.dice_reduction == "sum"
    assert config.dice_gamma == 2.0


@pytest.mark.parametrize("loss", ClassifierLossFctOption)
def test_classifier_config_loss_func(loss: str):
    config = ClassifierConfig(loss_func=loss)
    assert config.loss_func == loss


@pytest.mark.parametrize("reduction", DiceReductionOption)
def test_classifier_config_dice_reduction(reduction: str):
    config = ClassifierConfig(dice_reduction=reduction)
    assert config.dice_reduction == reduction


def test_classifier_config_validation():
    with pytest.raises(ValueError, match="model_name must be set"):
        ClassifierConfig(model_name="")
    with pytest.raises(ValueError, match="eval_metric must be one of"):
        ClassifierConfig(eval_metric="invalid_metric")
    with pytest.raises(ValueError, match="loss_func must be one of"):
        ClassifierConfig(loss_func="invalid")
    with pytest.raises(ValueError, match="classifier_hidden_dimension must be greater than 0"):
        ClassifierConfig(classifier_hidden_dimension=0)
    with pytest.raises(ValueError, match="dropout_p must be between 0 and 1"):
        ClassifierConfig(dropout_p=1.5)
    with pytest.raises(ValueError, match="num_labels must be greater than 0"):
        ClassifierConfig(num_labels=0)
    with pytest.raises(ValueError, match="dice_gamma must be positive"):
        ClassifierConfig(dice_gamma=-1.0)
    with pytest.raises(ValueError, match="dice_reduction must be either 'mean' or 'sum'"):
        ClassifierConfig(dice_reduction="invalid")


def test_classifier_configs_asdict():
    classifier_config = ClassifierConfig()
    classifier_dict = classifier_config.asdict()
    assert isinstance(classifier_dict, dict)


def test_classifier_config_dropout_boundaries():
    """Test dropout probability boundary values"""
    config = ClassifierConfig(dropout_p=0.0)
    assert config.dropout_p == 0.0
    config = ClassifierConfig(dropout_p=1.0)
    assert config.dropout_p == 1.0


def test_classifier_config_type_validation():
    """Test type validation for classifier config parameters"""
    with pytest.raises(TypeError):
        ClassifierConfig(model_name=1)
    with pytest.raises(TypeError):
        ClassifierConfig(eval_similarity_fct=1)
    with pytest.raises(TypeError):
        ClassifierConfig(loss_func=1)
    with pytest.raises(TypeError):
        ClassifierConfig(classifier_hidden_dimension="1")
    with pytest.raises(TypeError):
        ClassifierConfig(dropout_p="1")
    with pytest.raises(TypeError):
        ClassifierConfig(num_labels="1")
    with pytest.raises(TypeError):
        ClassifierConfig(dice_reduction=1)
    with pytest.raises(TypeError):
        ClassifierConfig(dice_gamma="1")
