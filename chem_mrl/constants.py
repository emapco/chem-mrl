import os

_curr_file_dir = os.path.dirname(os.path.abspath(__file__))
_project_root_dir = os.path.dirname(_curr_file_dir)
_root_data_dir = os.path.join(_project_root_dir, "data")
_data_dir = os.path.join(_root_data_dir, "processed")
OUTPUT_MODEL_DIR = os.path.join(_project_root_dir, "training_output", "old")
OUTPUT_DATA_DIR = _root_data_dir
BASE_MODEL_HIDDEN_DIM = 768
TEST_FP_SIZES = [8, 16, 32, 64, 128, 256, 512, 768, 2048]
CHEM_MRL_DIMENSIONS = [768, 512, 256, 128, 64, 32, 16, 8]
BASE_MODEL_DIMENSIONS = [BASE_MODEL_HIDDEN_DIM]
BASE_MODEL_NAME = "Derify/ChemBERTa_augmented_pubchem_13m"
CHEM_MRL_MODEL_NAME = "Derify/ChemMRL-alpha"
TRAINED_CHEM_MRL_DIMENSIONS = [1024, 768, 512, 256, 128, 64, 32, 16, 8, 4]
OPTUNA_DB_URI = "postgresql://postgres:password@127.0.0.1:5432/postgres"


##############################
# CHEM-MRL TRAINED MODEL PATHS
##############################
MODEL_NAMES = {
    # full dataset 2d-mrl-embed preferred in init. hyperparam. search
    # followed by QED_morgan dataset with NON-functional morgan fingerprints
    "base": BASE_MODEL_NAME,  # for comparison
    "alpha": CHEM_MRL_MODEL_NAME,
}
MODEL_NAME_KEYS = sorted(list(MODEL_NAMES.keys()))

##############################
# CHEM-MRL DATASET MAPS
##############################
TRAIN_DS_DICT = {
    "pubchem-10m-fp-sim": os.path.join(_data_dir, "train_pubchem_10m_fp_sim_8192.parquet"),
}
CHEM_MRL_DATASET_KEYS = sorted(list(TRAIN_DS_DICT.keys()))

VAL_DS_DICT = {
    "pubchem-10m-fp-sim": os.path.join(_data_dir, "val_pubchem_10m_fp_sim_8192.parquet"),
}
TEST_DS_DICT = {
    "pubchem-10m-fp-sim": os.path.join(_data_dir, "test_pubchem_10m_fp_sim_8192.parquet"),
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
