import os

_const_file_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_const_file_dir)
_data_dir = os.path.join(_parent_dir, "data", "chem")
OUTPUT_MODEL_DIR = os.path.join(_parent_dir, "output")

TRAIN_DS_DICT = {
    "functional-qed-pfizer-fp-similarity": os.path.join(
        _data_dir,
        "train_QED-pfizer_functional_fp_similarity_8192.parquet",
    ),
    "functional-qed-fp-similarity": os.path.join(
        _data_dir, "train_QED_functional_fp_similarity_8192.parquet"
    ),
    "functional-fp-similarity": os.path.join(
        _data_dir, "train_functional_fp_similarity_8192.parquet"
    ),
    "qed-pfizer-fp-similarity": os.path.join(
        _data_dir, "train_QED-pfizer_fp_similarity_8192.parquet"
    ),
    "qed-fp-similarity": os.path.join(
        _data_dir, "train_QED_fp_similarity_8192.parquet"
    ),
    "fp-similarity": os.path.join(_data_dir, "train_fp_similarity_8192.parquet"),
}

VAL_DS_DICT = {
    "functional-qed-pfizer-fp-similarity": os.path.join(
        _data_dir,
        "val_QED-pfizer_functional_fp_similarity_8192.parquet",
    ),
    "functional-qed-fp-similarity": os.path.join(
        _data_dir, "val_QED_functional_fp_similarity_8192.parquet"
    ),
    "functional-fp-similarity": os.path.join(
        _data_dir, "val_functional_fp_similarity_8192.parquet"
    ),
    "qed-pfizer-fp-similarity": os.path.join(
        _data_dir, "val_QED-pfizer_fp_similarity_8192.parquet"
    ),
    "qed-fp-similarity": os.path.join(_data_dir, "val_QED_fp_similarity_8192.parquet"),
    "fp-similarity": os.path.join(_data_dir, "val_fp_similarity_8192.parquet"),
}

TEST_DS_DICT = {
    "functional-qed-pfizer-fp-similarity": os.path.join(
        _data_dir,
        "test_QED-pfizer_functional_fp_similarity_8192.parquet",
    ),
    "functional-qed-fp-similarity": os.path.join(
        _data_dir, "test_QED_functional_fp_similarity_8192.parquet"
    ),
    "functional-fp-similarity": os.path.join(
        _data_dir, "test_functional_fp_similarity_8192.parquet"
    ),
    "qed-pfizer-fp-similarity": os.path.join(
        _data_dir, "test_QED-pfizer_fp_similarity_8192.parquet"
    ),
    "qed-fp-similarity": os.path.join(_data_dir, "test_QED_fp_similarity_8192.parquet"),
    "fp-similarity": os.path.join(_data_dir, "test_fp_similarity_8192.parquet"),
}


def check_dataset_files():
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
    check_dataset_files()
