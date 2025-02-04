# type: ignore
import pytest

from chem_mrl.configs import Chem2dMRLConfig, ChemMRLConfig
from chem_mrl.configs.MrlConfig import (
    CHEM_MRL_LOSS_FCT_OPTIONS,
    TANIMOTO_SIMILARITY_BASE_LOSS_FCT_OPTIONS,
)
from chem_mrl.constants import BASE_MODEL_NAME, CHEM_MRL_DIMENSIONS


def test_chem_mrl_config_default_values():
    config = ChemMRLConfig(
        model_name=BASE_MODEL_NAME,
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
    )

    assert config.smiles_a_column_name == "smiles_a"
    assert config.smiles_b_column_name == "smiles_b"
    assert config.label_column_name == "similarity"
    assert config.model_name == BASE_MODEL_NAME
    assert config.embedding_pooling == "mean"
    assert config.loss_func == "tanimotosentloss"
    assert config.tanimoto_similarity_loss_func is None
    assert config.eval_similarity_fct == "tanimoto"
    assert config.eval_metric == "spearman"
    assert config.mrl_dimensions == tuple(CHEM_MRL_DIMENSIONS)
    assert len(config.mrl_dimension_weights) == len(CHEM_MRL_DIMENSIONS)
    assert config.n_dims_per_step == -1
    assert config.use_2d_matryoshka is False


def test_chem_mrl_config_custom_values():
    custom_weights = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4)
    config = ChemMRLConfig(
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
        model_name="custom_model",
        embedding_pooling="weightedmean",
        loss_func="angleloss",
        tanimoto_similarity_loss_func="mse",
        mrl_dimension_weights=custom_weights,
        n_dims_per_step=2,
        use_2d_matryoshka=True,
    )

    assert config.train_dataset_path == "train.parquet"
    assert config.val_dataset_path == "val.parquet"
    assert config.model_name == "custom_model"
    assert config.embedding_pooling == "weightedmean"
    assert config.loss_func == "angleloss"
    assert config.tanimoto_similarity_loss_func == "mse"
    assert config.mrl_dimension_weights == custom_weights
    assert config.n_dims_per_step == 2
    assert config.use_2d_matryoshka is True


def test_chem_mrl_config_validation():
    with pytest.raises(ValueError, match="smiles_a_column_name must be set"):
        ChemMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            smiles_a_column_name="",
        )
    with pytest.raises(ValueError, match="smiles_b_column_name must be set"):
        ChemMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            smiles_b_column_name="",
        )
    with pytest.raises(ValueError, match="label_column_name must be set"):
        ChemMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            label_column_name="",
        )
    with pytest.raises(ValueError, match="embedding_pooling must be one of"):
        ChemMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            embedding_pooling="invalid_pooling",
        )
    with pytest.raises(ValueError, match="loss_func must be one of"):
        ChemMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            loss_func="invalid_loss",
        )
    with pytest.raises(
        ValueError, match="tanimoto_similarity_loss_func must be one of"
    ):
        ChemMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            tanimoto_similarity_loss_func="invalid_loss",
        )
    with pytest.raises(ValueError, match="eval_similarity_fct must be one of"):
        ChemMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            eval_similarity_fct="invalid_fct",
        )
    with pytest.raises(ValueError, match="eval_metric must be one of"):
        ChemMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            eval_metric="invalid_metric",
        )
    invalid_weights = (1.0, 1.2, 1.4)  # Wrong length
    with pytest.raises(ValueError, match="Number of dimension weights must match"):
        ChemMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            mrl_dimension_weights=invalid_weights,
        )
    negative_weights = (1.0, -1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4)
    with pytest.raises(ValueError, match="All dimension weights must be positive"):
        ChemMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            mrl_dimension_weights=negative_weights,
        )
    non_increasing_weights = (2.0, 1.0, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4)
    with pytest.raises(
        ValueError, match="Dimension weights must be in increasing order"
    ):
        ChemMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            mrl_dimension_weights=non_increasing_weights,
        )
    with pytest.raises(ValueError, match="n_dims_per_step must be positive or -1"):
        ChemMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            n_dims_per_step=0,
        )


def test_chem_2d_mrl_config_default_values():
    config = Chem2dMRLConfig(
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
    )
    assert config.use_2d_matryoshka is True
    assert config.n_layers_per_step == -1
    assert config.last_layer_weight == 1.0
    assert config.prior_layers_weight == 1.0
    assert config.kl_div_weight == 1.0
    assert config.kl_temperature == 0.3


def test_chem_2d_mrl_config_custom_values():
    config = Chem2dMRLConfig(
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
        n_layers_per_step=2,
        last_layer_weight=2.0,
        prior_layers_weight=1.5,
        kl_div_weight=0.5,
        kl_temperature=0.7,
    )
    assert config.n_layers_per_step == 2
    assert config.last_layer_weight == 2.0
    assert config.prior_layers_weight == 1.5
    assert config.kl_div_weight == 0.5
    assert config.kl_temperature == 0.7


def test_chem_2d_mrl_config_validation():
    with pytest.raises(ValueError, match="use_2d_matryoshka must be True"):
        Chem2dMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            use_2d_matryoshka=False,
        )
    with pytest.raises(ValueError, match="n_layers_per_step must be positive"):
        Chem2dMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            n_layers_per_step=0,
        )
    with pytest.raises(ValueError, match="last_layer_weight must be positive"):
        Chem2dMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            last_layer_weight=0,
        )
    with pytest.raises(ValueError, match="prior_layers_weight must be positive"):
        Chem2dMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            prior_layers_weight=-1.0,
        )
    with pytest.raises(ValueError, match="kl_div_weight must be positive"):
        Chem2dMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            kl_div_weight=0,
        )
    with pytest.raises(ValueError, match="kl_temperature must be positive"):
        Chem2dMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            kl_temperature=0.0,
        )


def test_mrl_configs_asdict():
    chem_config = ChemMRLConfig(
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
    )
    chem_2d_config = Chem2dMRLConfig(
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
    )

    chem_dict = chem_config.asdict()
    chem_2d_dict = chem_2d_config.asdict()

    assert isinstance(chem_dict, dict)
    assert isinstance(chem_2d_dict, dict)
    assert "loss_func" in chem_dict
    assert "last_layer_weight" in chem_2d_dict


@pytest.mark.parametrize("loss_func", CHEM_MRL_LOSS_FCT_OPTIONS)
def test_tanimoto_loss_options(loss_func):
    config = ChemMRLConfig(
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
        loss_func=loss_func,
    )
    assert config.loss_func == loss_func


@pytest.mark.parametrize("base_loss", TANIMOTO_SIMILARITY_BASE_LOSS_FCT_OPTIONS)
def test_tanimoto_similarity_base_loss_options(base_loss):
    config = ChemMRLConfig(
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
        tanimoto_similarity_loss_func=base_loss,
    )
    assert config.tanimoto_similarity_loss_func == base_loss


def test_chem_mrl_config_equality():
    config1 = ChemMRLConfig(
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
    )
    config2 = ChemMRLConfig(
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
    )
    config3 = ChemMRLConfig(
        train_dataset_path="train2.parquet",
        val_dataset_path="val.parquet",
    )

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
    config = ChemMRLConfig(
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
        mrl_dimension_weights=min_weights,
    )
    assert config.mrl_dimension_weights == min_weights

    # Test large weight differences
    max_weights = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0)
    config = ChemMRLConfig(
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
        mrl_dimension_weights=max_weights,
    )
    assert config.mrl_dimension_weights == max_weights


def test_chem_2d_mrl_inheritance():
    base_config = ChemMRLConfig(
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
        model_name="inherited_model",
    )
    derived_config = Chem2dMRLConfig(
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
        model_name="inherited_model",
    )

    assert derived_config.model_name == base_config.model_name
    assert derived_config.use_2d_matryoshka is True
    assert base_config.use_2d_matryoshka is False


def test_multiple_invalid_parameters():
    with pytest.raises(ValueError):
        ChemMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            model_name="",
            loss_func="invalid_loss",
            mrl_dimension_weights=(1.0, 0.5, 0.2),
        )


def test_chem_2d_mrl_weight_precision():
    config = Chem2dMRLConfig(
        train_dataset_path="train.parquet",
        val_dataset_path="val.parquet",
        last_layer_weight=1.87082200634879971234,  # Extra precision
        prior_layers_weight=1.45982493214472451234,  # Extra precision
    )
    assert abs(config.last_layer_weight - 1.8708220063487997) < 1e-15
    assert abs(config.prior_layers_weight - 1.4598249321447245) < 1e-15


def test_chem_mrl_config_type_validation():
    """Test type validation for chem mrl config parameters"""
    with pytest.raises(TypeError):
        ChemMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            smiles_a_column_name=1,
        )
    with pytest.raises(TypeError):
        ChemMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            smiles_b_column_name=1,
        )
    with pytest.raises(TypeError):
        ChemMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            label_column_name=1,
        )
    with pytest.raises(TypeError):
        ChemMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            embedding_pooling=1,
        )
    with pytest.raises(TypeError):
        ChemMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            loss_func=1,
        )
    with pytest.raises(TypeError):
        ChemMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            tanimoto_similarity_loss_func=1,
        )
    with pytest.raises(TypeError):
        ChemMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            eval_similarity_fct=1,
        )
    with pytest.raises(TypeError):
        ChemMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            eval_metric=1,
        )
    with pytest.raises(TypeError):
        ChemMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            mrl_dimensions=1,
        )
    with pytest.raises(TypeError):
        ChemMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            mrl_dimension_weights=1,
        )
    with pytest.raises(TypeError):
        ChemMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            n_dims_per_step="1",
        )


def test_chem_2d_mrl_config_type_validation():
    """Test type validation for chem 2d mrl config parameters"""
    with pytest.raises(TypeError):
        Chem2dMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            n_layers_per_step="1",
        )
    with pytest.raises(TypeError):
        Chem2dMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            last_layer_weight="1",
        )
    with pytest.raises(TypeError):
        Chem2dMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            prior_layers_weight="1",
        )
    with pytest.raises(TypeError):
        Chem2dMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            kl_div_weight="1",
        )
    with pytest.raises(TypeError):
        Chem2dMRLConfig(
            train_dataset_path="train.parquet",
            val_dataset_path="val.parquet",
            kl_temperature="1",
        )
