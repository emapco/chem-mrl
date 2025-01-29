import os

_curr_file_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_curr_file_dir)
_test_data_dir = os.path.join(_parent_dir, "data")


TEST_CHEM_MRL_PATH = os.path.join(_test_data_dir, "test_chem_mrl.parquet")
TEST_CLASSIFICATION_PATH = os.path.join(_test_data_dir, "test_classification.parquet")
