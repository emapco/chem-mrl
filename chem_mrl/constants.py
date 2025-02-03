import os

_curr_file_dir = os.path.dirname(os.path.abspath(__file__))
_project_root_dir = os.path.dirname(_curr_file_dir)
_data_dir = os.path.join(_project_root_dir, "data", "chem")
OUTPUT_MODEL_DIR = os.path.join(_project_root_dir, "output", "old")
OUTPUT_DATA_DIR = os.path.join(_project_root_dir, "data", "chem")
EMBEDDING_MODEL_HIDDEN_DIM = 768
TEST_FP_SIZES = [8, 16, 32, 64, 128, 256, 512, 768, 2048]
CHEM_MRL_DIMENSIONS = [768, 512, 256, 128, 64, 32, 16, 8]
BASE_MODEL_DIMENSIONS = [EMBEDDING_MODEL_HIDDEN_DIM]
BASE_MODEL_NAME = "seyonec/SMILES_tokenized_PubChem_shard00_160k"
OPTUNA_DB_URI = "postgresql://postgres:password@127.0.0.1:5432/postgres"


##############################
# CHEM-MRL TRAINED MODEL PATHS
##############################
MODEL_NAMES = {
    # full dataset 2d-mrl-embed preferred in init. hyperparam. search
    # followed by QED_morgan dataset with NON-functional morgan fingerprints
    "base": BASE_MODEL_NAME,  # for comparison
    "full_dataset": os.path.join(
        OUTPUT_MODEL_DIR,
        "ChemBERTa-zinc-base-v1-2d-matryoshka-embeddings"
        "-n_layers_per_step_2-TaniLoss-lr_1.1190785944700813e-05-batch_size_8"
        "-num_epochs_2-epoch_2-best-model-1900000_steps",
    ),
    "qed_functional_fingerprints": os.path.join(
        OUTPUT_MODEL_DIR,
        "ChemBERTa-zinc-base-v1-QED_functional_morgan_fingerprint-2d-matryoshka-embeddings"
        "-num_epochs_2-epoch_2-best-model-1900000_steps",
    ),
    "qed_fingerprints": os.path.join(
        OUTPUT_MODEL_DIR,
        "ChemBERTa-zinc-base-v1-QED_morgan_fingerprint-2d-matryoshka-embeddings"
        "-num_epochs_2-epoch_4-best-model-1900000_steps",
    ),
}
MODEL_NAME_KEYS = sorted(list(MODEL_NAMES.keys()))

##############################
# CHEM-MRL DATASET MAPS
##############################
TRAIN_DS_DICT = {
    "functional-qed-pfizer-fp-sim": os.path.join(
        _data_dir,
        "train_QED-pfizer_func_fp_sim_8192.parquet",
    ),
    "functional-qed-fp-sim": os.path.join(
        _data_dir, "train_QED_func_fp_sim_8192.parquet"
    ),
    "functional-fp-sim": os.path.join(_data_dir, "train_func_fp_sim_8192.parquet"),
    "functional-pubchem-10m-fp-sim": os.path.join(
        _data_dir, "train_pubchem_10m_fp_sim_8192.parquet"
    ),
    "qed-pfizer-fp-sim": os.path.join(
        _data_dir, "train_QED-pfizer_fp_sim_8192.parquet"
    ),
    "qed-fp-sim": os.path.join(_data_dir, "train_QED_fp_sim_8192.parquet"),
    "fp-sim": os.path.join(_data_dir, "train_fp_sim_8192.parquet"),
    "pubchem-10m-fp-sim": os.path.join(
        _data_dir, "train_pubchem_10m_fp_sim_8192.parquet"
    ),
}
CHEM_MRL_DATASET_KEYS = sorted(list(TRAIN_DS_DICT.keys()))

VAL_DS_DICT = {
    "functional-qed-pfizer-fp-sim": os.path.join(
        _data_dir,
        "val_QED-pfizer_func_fp_sim_8192.parquet",
    ),
    "functional-qed-fp-sim": os.path.join(
        _data_dir, "val_QED_func_fp_sim_8192.parquet"
    ),
    "functional-fp-sim": os.path.join(_data_dir, "val_func_fp_sim_8192.parquet"),
    "functional-pubchem-10m-fp-sim": os.path.join(
        _data_dir, "val_pubchem_10m_fp_sim_8192.parquet"
    ),
    "qed-pfizer-fp-sim": os.path.join(_data_dir, "val_QED-pfizer_fp_sim_8192.parquet"),
    "qed-fp-sim": os.path.join(_data_dir, "val_QED_fp_sim_8192.parquet"),
    "fp-sim": os.path.join(_data_dir, "val_fp_sim_8192.parquet"),
    "pubchem-10m-fp-sim": os.path.join(
        _data_dir, "val_pubchem_10m_fp_sim_8192.parquet"
    ),
}
TEST_DS_DICT = {
    "functional-qed-pfizer-fp-sim": os.path.join(
        _data_dir,
        "test_QED-pfizer_func_fp_sim_8192.parquet",
    ),
    "functional-qed-fp-sim": os.path.join(
        _data_dir, "test_QED_func_fp_sim_8192.parquet"
    ),
    "functional-fp-sim": os.path.join(_data_dir, "test_func_fp_sim_8192.parquet"),
    "functional-pubchem-10m-fp-sim": os.path.join(
        _data_dir, "test_pubchem_10m_fp_sim_8192.parquet"
    ),
    "qed-pfizer-fp-sim": os.path.join(_data_dir, "test_QED-pfizer_fp_sim_8192.parquet"),
    "qed-fp-sim": os.path.join(_data_dir, "test_QED_fp_sim_8192.parquet"),
    "fp-sim": os.path.join(_data_dir, "test_fp_sim_8192.parquet"),
    "pubchem-10m-fp-sim": os.path.join(
        _data_dir, "test_pubchem_10m_fp_sim_8192.parquet"
    ),
}


def _check_dataset_files():
    all_dicts = {
        "Training": TRAIN_DS_DICT,
        "Validation": VAL_DS_DICT,
        "Testing": TEST_DS_DICT,
    }

    for dataset_type, dataset_dict in all_dicts.items():
        print(f"\nChecking {dataset_type} datasets:")
        for model_type, file_path in dataset_dict.items():
            exists = os.path.exists(file_path)
            status = "✓" if exists else "✗"
            print(f"{status} {model_type}: {file_path}")


if __name__ == "__main__":
    _check_dataset_files()
