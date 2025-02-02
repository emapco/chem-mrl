import os
from pathlib import Path

curr_file_path = Path(__file__).parent
_parent_dir = Path(curr_file_path).parent
_test_data_dir = Path(_parent_dir, "data")


TEST_CHEM_MRL_PATH = os.path.join(_test_data_dir, "test_chem_mrl.parquet")
TEST_CLASSIFICATION_PATH = os.path.join(_test_data_dir, "test_classification.parquet")
