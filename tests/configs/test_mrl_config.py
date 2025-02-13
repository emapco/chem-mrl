# type: ignore
import pytest

from chem_mrl.schemas.ChemMRLConfig import (
    ChemMRLConfig,
    ChemMrlLossFctOption,
    EmbeddingPoolingOption,
    TanimotoSimilarityBaseLossFctOption,
)


def test_chem_mrl_config_custom_values():
    custom_weights = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4)
    config = ChemMRLConfig(
        use_query_tokenizer=True,
        embedding_pooling=EmbeddingPoolingOption.weightedmean,
        loss_func=ChemMrlLossFctOption.angleloss,
        tanimoto_similarity_loss_func=TanimotoSimilarityBaseLossFctOption.mse,
        mrl_dimension_weights=custom_weights,
        n_dims_per_step=2,
        use_2d_matryoshka=True,
        n_layers_per_step=2,
        last_layer_weight=2.0,
        prior_layers_weight=1.5,
        kl_div_weight=0.5,
        kl_temperature=0.7,
    )

    assert config.use_query_tokenizer is True
    assert config.embedding_pooling == "weightedmean"
    assert config.loss_func == "angleloss"
    assert config.tanimoto_similarity_loss_func == "mse"
    assert config.mrl_dimension_weights == custom_weights
    assert config.n_dims_per_step == 2
    assert config.use_2d_matryoshka is True
    assert config.n_layers_per_step == 2
    assert config.last_layer_weight == 2.0
    assert config.prior_layers_weight == 1.5
    assert config.kl_div_weight == 0.5
    assert config.kl_temperature == 0.7


def test_chem_mrl_config_validation():
    with pytest.raises(ValueError, match="model_name must be set"):
        ChemMRLConfig(model_name="")
    with pytest.raises(ValueError, match="embedding_pooling must be one of"):
        ChemMRLConfig(embedding_pooling="invalid_pooling")
    with pytest.raises(ValueError, match="loss_func must be one of"):
        ChemMRLConfig(loss_func="invalid_loss")
    with pytest.raises(ValueError, match="tanimoto_similarity_loss_func must be one of"):
        ChemMRLConfig(tanimoto_similarity_loss_func="invalid_loss")
    with pytest.raises(ValueError, match="eval_similarity_fct must be one of"):
        ChemMRLConfig(eval_similarity_fct="invalid_fct")
    with pytest.raises(ValueError, match="eval_metric must be one of"):
        ChemMRLConfig(eval_metric="invalid_metric")
    invalid_weights = (1.0, 1.2, 1.4)  # Wrong length
    with pytest.raises(ValueError, match="Number of dimension weights must match"):
        ChemMRLConfig(mrl_dimension_weights=invalid_weights)
    negative_weights = (1.0, -1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4)
    with pytest.raises(ValueError, match="All dimension weights must be positive"):
        ChemMRLConfig(mrl_dimension_weights=negative_weights)
    non_increasing_weights = (2.0, 1.0, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4)
    with pytest.raises(ValueError, match="Dimension weights must be in increasing order"):
        ChemMRLConfig(mrl_dimension_weights=non_increasing_weights)
    with pytest.raises(ValueError, match="n_dims_per_step must be positive or -1"):
        ChemMRLConfig(n_dims_per_step=0)
    with pytest.raises(ValueError, match="n_layers_per_step must be positive or -1"):
        ChemMRLConfig(n_layers_per_step=0)
    with pytest.raises(ValueError, match="last_layer_weight must be positive"):
        ChemMRLConfig(last_layer_weight=0)
    with pytest.raises(ValueError, match="prior_layers_weight must be positive"):
        ChemMRLConfig(prior_layers_weight=-1.0)
    with pytest.raises(ValueError, match="kl_div_weight must be positive"):
        ChemMRLConfig(kl_div_weight=0)
    with pytest.raises(ValueError, match="kl_temperature must be positive"):
        ChemMRLConfig(kl_temperature=0.0)


def test_mrl_configs_asdict():
    chem_config = ChemMRLConfig()
    chem_dict = chem_config.asdict()
    assert isinstance(chem_dict, dict)
    assert "loss_func" in chem_dict
    assert "last_layer_weight" in chem_dict


@pytest.mark.parametrize("pooling", EmbeddingPoolingOption)
def test_embedding_pooling_options(pooling):
    config = ChemMRLConfig(embedding_pooling=pooling)
    assert config.embedding_pooling == pooling


@pytest.mark.parametrize("loss_func", ChemMrlLossFctOption)
def test_tanimoto_loss_options(loss_func):
    config = ChemMRLConfig(loss_func=loss_func)
    assert config.loss_func == loss_func


@pytest.mark.parametrize("base_loss", TanimotoSimilarityBaseLossFctOption)
def test_tanimoto_similarity_base_loss_options(base_loss):
    config = ChemMRLConfig(tanimoto_similarity_loss_func=base_loss)
    assert config.tanimoto_similarity_loss_func == base_loss


def test_chem_mrl_config_equality():
    config1 = ChemMRLConfig()
    config2 = ChemMRLConfig()
    config3 = ChemMRLConfig(use_2d_matryoshka=True)

    assert config1 == config2
    assert config1 != config3
    assert config1 != "not_a_config"


def test_dimension_weights_edge_cases():
    # Test minimum valid weights
    min_weights = (
        1.0,
        1.000001,
        1.000002,
        1.000003,
        1.000004,
        1.000005,
        1.000006,
        1.000007,
    )
    config = ChemMRLConfig(mrl_dimension_weights=min_weights)
    assert config.mrl_dimension_weights == min_weights

    # Test large weight differences
    max_weights = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0)
    config = ChemMRLConfig(mrl_dimension_weights=max_weights)
    assert config.mrl_dimension_weights == max_weights


def test_multiple_invalid_parameters():
    with pytest.raises(ValueError):
        ChemMRLConfig(
            loss_func="invalid_loss",
            mrl_dimension_weights=(1.0, 0.5, 0.2),
        )


def test_chem_2d_mrl_weight_precision():
    config = ChemMRLConfig(
        use_2d_matryoshka=True,
        last_layer_weight=1.87082200634879971234,  # Extra precision
        prior_layers_weight=1.45982493214472451234,  # Extra precision
    )
    assert abs(config.last_layer_weight - 1.8708220063487997) < 1e-15
    assert abs(config.prior_layers_weight - 1.4598249321447245) < 1e-15


def test_chem_mrl_config_type_validation():
    """Test type validation for chem mrl config parameters"""
    with pytest.raises(TypeError):
        ChemMRLConfig(model_name=1)
    with pytest.raises(TypeError):
        ChemMRLConfig(use_query_tokenizer=1)
    with pytest.raises(TypeError):
        ChemMRLConfig(embedding_pooling=1)
    with pytest.raises(TypeError):
        ChemMRLConfig(loss_func=1)
    with pytest.raises(TypeError):
        ChemMRLConfig(tanimoto_similarity_loss_func=1)
    with pytest.raises(TypeError):
        ChemMRLConfig(eval_similarity_fct=1)
    with pytest.raises(TypeError):
        ChemMRLConfig(eval_metric=1)
    with pytest.raises(TypeError):
        ChemMRLConfig(mrl_dimensions=1)
    with pytest.raises(TypeError):
        ChemMRLConfig(mrl_dimension_weights=1)
    with pytest.raises(TypeError):
        ChemMRLConfig(n_dims_per_step="1")
    with pytest.raises(TypeError):
        ChemMRLConfig(use_2d_matryoshka="1")
    with pytest.raises(TypeError):
        ChemMRLConfig(n_layers_per_step="1")
    with pytest.raises(TypeError):
        ChemMRLConfig(last_layer_weight="1")
    with pytest.raises(TypeError):
        ChemMRLConfig(prior_layers_weight="1")
    with pytest.raises(TypeError):
        ChemMRLConfig(kl_div_weight="1")
    with pytest.raises(TypeError):
        ChemMRLConfig(kl_temperature="1")
