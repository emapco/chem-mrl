import os

_const_file_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_const_file_dir)
_data_dir = os.path.join(_parent_dir, "data", "chem")
OUTPUT_MODEL_DIR = os.path.join(_parent_dir, "output")
CHEM_MRL_DIMENSIONS = [768, 512, 256, 128, 64, 32]

TRAIN_ISOMER_DESIGN_DS_PATH = os.path.join(
    _data_dir, "isomer_design", "train_isomer_design.parquet"
)
VAL_ISOMER_DESIGN_DS_PATH = os.path.join(
    _data_dir, "isomer_design", "val_isomer_design.parquet"
)
TEST_ISOMER_DESIGN_DS_PATH = os.path.join(
    _data_dir, "isomer_design", "test_isomer_design.parquet"
)

MODEL_NAMES = {
    # full dataset 2d-mrl-embed preferred in init. hyperparam. search
    # followed by QED_morgan dataset with NON-functional morgan fingerprints
    "base": "seyonec/ChemBERTa-zinc-base-v1",  # for comparison
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

CAT_TO_LABEL = {
    "unknown": 0,
    "cannabinoid": 1,
    "tryptamine": 2,
    "aryldiazepine": 3,
    "fentanyl": 4,
    "arylcycloalkylamine": 5,
    "peyote alkaloid": 6,
    "essential oil": 7,
    "neurotoxin": 8,
}
LABEL_TO_CAT = {v: k for k, v in CAT_TO_LABEL.items()}
