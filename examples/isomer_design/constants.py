import os

_const_file_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_const_file_dir)
_project_root_dir = os.path.dirname(_parent_dir)
_data_dir = os.path.join(_project_root_dir, "data", "chem")


##############################
# ISOMER DESIGN CLASSIFICATION
##############################

TRAIN_ISOMER_DESIGN_DS_PATH = os.path.join(
    _data_dir, "isomer_design", "train_isomer_design.parquet"
)
VAL_ISOMER_DESIGN_DS_PATH = os.path.join(
    _data_dir, "isomer_design", "val_isomer_design.parquet"
)
TEST_ISOMER_DESIGN_DS_PATH = os.path.join(
    _data_dir, "isomer_design", "test_isomer_design.parquet"
)

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
