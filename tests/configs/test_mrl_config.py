import pytest

from chem_mrl.configs import Chem2dMRLConfig, ChemMRLConfig
from chem_mrl.configs.MRL import (
    _tanimoto_loss_func_options,
    _tanimoto_similarity_base_loss_func_options,
)
from chem_mrl.constants import (
    BASE_MODEL_NAME,
    CHEM_MRL_DATASET_KEYS,
    CHEM_MRL_DIMENSIONS,
)


def test_chem_mrl_config_default_values():
    config = ChemMRLConfig()
    assert config.dataset_key == CHEM_MRL_DATASET_KEYS[0]
    assert config.model_name == BASE_MODEL_NAME
    assert config.loss_func == "tanimotosentloss"
    assert config.tanimoto_similarity_loss_func is None
    assert config.mrl_dimensions == CHEM_MRL_DIMENSIONS
    assert len(config.mrl_dimension_weights) == len(CHEM_MRL_DIMENSIONS)
    assert config.use_2d_matryoshka is False


def test_chem_mrl_config_custom_values():
    custom_weights = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
    config = ChemMRLConfig(
        dataset_key=CHEM_MRL_DATASET_KEYS[1],
        model_name="custom_model",
        loss_func="cosentloss",
        tanimoto_similarity_loss_func="mse",
        mrl_dimension_weights=custom_weights,
        use_2d_matryoshka=True,
    )

    assert config.dataset_key == CHEM_MRL_DATASET_KEYS[1]
    assert config.model_name == "custom_model"
    assert config.loss_func == "cosentloss"
    assert config.tanimoto_similarity_loss_func == "mse"
    assert config.mrl_dimension_weights == custom_weights
    assert config.use_2d_matryoshka is True


def test_chem_mrl_config_validation():
    with pytest.raises(ValueError, match="dataset_key must be one of"):
        ChemMRLConfig(dataset_key="invalid_key")

    with pytest.raises(ValueError, match="model_name must be set"):
        ChemMRLConfig(model_name="")

    with pytest.raises(ValueError, match="loss_func must be one of"):
        ChemMRLConfig(loss_func="invalid_loss")

    with pytest.raises(
        ValueError, match="tanimoto_similarity_loss_func must be one of"
    ):
        ChemMRLConfig(tanimoto_similarity_loss_func="invalid_loss")

    invalid_weights = (1.0, 1.2, 1.4)  # Wrong length
    with pytest.raises(ValueError, match="Number of dimension weights must match"):
        ChemMRLConfig(mrl_dimension_weights=invalid_weights)

    negative_weights = (1.0, -1.2, 1.4, 1.6, 1.8, 2.0)
    with pytest.raises(ValueError, match="All dimension weights must be positive"):
        ChemMRLConfig(mrl_dimension_weights=negative_weights)

    non_increasing_weights = (2.0, 1.0, 1.5, 1.6, 1.8, 2.0)
    with pytest.raises(
        ValueError, match="Dimension weights must be in increasing order"
    ):
        ChemMRLConfig(mrl_dimension_weights=non_increasing_weights)


def test_chem_2d_mrl_config_default_values():
    config = Chem2dMRLConfig()
    assert config.use_2d_matryoshka is True
    assert config.last_layer_weight == 1.8708220063487997
    assert config.prior_layers_weight == 1.4598249321447245


def test_chem_2d_mrl_config_custom_values():
    config = Chem2dMRLConfig(last_layer_weight=2.0, prior_layers_weight=1.5)
    assert config.last_layer_weight == 2.0
    assert config.prior_layers_weight == 1.5


def test_chem_2d_mrl_config_validation():
    with pytest.raises(ValueError, match="use_2d_matryoshka must be True"):
        Chem2dMRLConfig(use_2d_matryoshka=False)

    with pytest.raises(ValueError, match="last_layer_weight must be positive"):
        Chem2dMRLConfig(last_layer_weight=0)

    with pytest.raises(ValueError, match="prior_layers_weight must be positive"):
        Chem2dMRLConfig(prior_layers_weight=-1.0)


def test_mrl_configs_asdict():
    chem_config = ChemMRLConfig()
    chem_2d_config = Chem2dMRLConfig()

    chem_dict = chem_config.asdict()
    chem_2d_dict = chem_2d_config.asdict()

    assert isinstance(chem_dict, dict)
    assert isinstance(chem_2d_dict, dict)
    assert "dataset_key" in chem_dict
    assert "last_layer_weight" in chem_2d_dict


def test_tanimoto_loss_options():
    for loss_func in _tanimoto_loss_func_options:
        config = ChemMRLConfig(loss_func=loss_func)
        assert config.loss_func == loss_func


def test_tanimoto_similarity_base_loss_options():
    for base_loss in _tanimoto_similarity_base_loss_func_options:
        config = ChemMRLConfig(tanimoto_similarity_loss_func=base_loss)
        assert config.tanimoto_similarity_loss_func == base_loss
