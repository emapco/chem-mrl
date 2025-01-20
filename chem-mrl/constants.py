import os

_const_file_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_const_file_dir)
_data_dir = os.path.join(_parent_dir, "data", "chem")
OUTPUT_MODEL_DIR = os.path.join(_parent_dir, "output")

# test datasets
FUNCTIONAL_SIMILARITY_TEST_DS_PATH = os.path.join(
    _data_dir, "test_functional_morgan_fingerprint_similarity_8192.parquet"
)
SIMILARITY_TEST_DS_PATH = os.path.join(
    _data_dir, "test_morgan_fingerprint_similarity_8192.parquet"
)
# validation datasets
FUNCTIONAL_SIMILARITY_VALIDATE_DS_PATH = os.path.join(
    _data_dir, "validate_functional_morgan_fingerprint_similarity_8192.parquet"
)
SIMILARITY_VALIDATE_DS_PATH = os.path.join(
    _data_dir, "validate_morgan_fingerprint_similarity_8192.parquet"
)

TRAIN_DS_DICT = {
    "functional-qed-pfizer-morgan-similarity": os.path.join(
        _data_dir,
        "train_QED-Pfizer_functional_morgan_fingerprint_similarity_8192.parquet",
    ),
    "functional-qed-morgan-similarity": os.path.join(
        _data_dir, "train_QED_functional_morgan_fingerprint_similarity_8192.parquet"
    ),
    "functional-morgan-similarity": os.path.join(
        _data_dir, "train_functional_morgan_fingerprint_similarity_8192.parquet"
    ),
    "qed-pfizer-morgan-similarity": os.path.join(
        _data_dir, "train_QED-Pfizer_morgan_fingerprint_similarity_8192.parquet"
    ),
    "qed-morgan-similarity": os.path.join(
        _data_dir, "train_QED_morgan_fingerprint_similarity_8192.parquet"
    ),
    "morgan-similarity": os.path.join(
        _data_dir, "train_morgan_fingerprint_similarity_8192.parquet"
    ),
}

VAL_DS_DICT = {
    "functional-qed-pfizer-morgan-similarity": FUNCTIONAL_SIMILARITY_VALIDATE_DS_PATH,
    "functional-qed-morgan-similarity": FUNCTIONAL_SIMILARITY_VALIDATE_DS_PATH,
    "functional-morgan-similarity": FUNCTIONAL_SIMILARITY_VALIDATE_DS_PATH,
    "qed-pfizer-morgan-similarity": SIMILARITY_VALIDATE_DS_PATH,
    "qed-morgan-similarity": SIMILARITY_VALIDATE_DS_PATH,
    "morgan-similarity": SIMILARITY_VALIDATE_DS_PATH,
}

TEST_DS_DICT = {
    "functional-qed-pfizer-morgan-similarity": FUNCTIONAL_SIMILARITY_TEST_DS_PATH,
    "functional-qed-morgan-similarity": FUNCTIONAL_SIMILARITY_TEST_DS_PATH,
    "functional-morgan-similarity": FUNCTIONAL_SIMILARITY_TEST_DS_PATH,
    "qed-pfizer-morgan-similarity": SIMILARITY_TEST_DS_PATH,
    "qed-morgan-similarity": SIMILARITY_TEST_DS_PATH,
    "morgan-similarity": SIMILARITY_TEST_DS_PATH,
}
