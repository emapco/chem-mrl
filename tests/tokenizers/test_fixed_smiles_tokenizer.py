# type: ignore
from pathlib import Path

import pandas as pd
import pytest
from transformers import RobertaTokenizerFast

from chem_mrl.tokenizers import SmilesTokenizerFast

curr_file_path = Path(__file__).parent
_parent_dir = Path(curr_file_path).parent
_test_data_dir = Path(_parent_dir, "data")


TEST_FIXED_SMILES_TOKENIZER_DATA_PATH = Path(
    _test_data_dir, "test_fixed_smiles_tokenizer_data.parquet"
)


@pytest.mark.parametrize("max_len", range(128, 1024, 128))
def test_fixed_smiles_tokenizer_arbitrary_length(max_len):
    fixed_tokenizer = SmilesTokenizerFast(max_len=max_len)
    assert fixed_tokenizer.model_max_length == max_len


def test_fixed_smiles_tokenizer():
    """Test that custom tokenizer tokenizes smiles string (RDKit canonical or not)
    identically to `DeepChem/SmilesTokenizer_PubChem_1M` tokenizer.
    Note these tokenizer's have different vocab arrangements so this checks the decoded smiles.
    """
    fixed_tokenizer = SmilesTokenizerFast(max_len=128)
    assert fixed_tokenizer.unk_token == "<unk>"
    assert fixed_tokenizer.cls_token == "<s>"
    assert fixed_tokenizer.sep_token == "</s>"
    assert fixed_tokenizer.pad_token == "<pad>"
    assert fixed_tokenizer.mask_token == "<mask>"
    assert fixed_tokenizer.unk_token_id == 0
    assert fixed_tokenizer.cls_token_id == 1
    assert fixed_tokenizer.sep_token_id == 2
    assert fixed_tokenizer.pad_token_id == 3
    assert fixed_tokenizer.mask_token_id == 4
    assert fixed_tokenizer.vocab_size == 581

    df = pd.read_parquet(TEST_FIXED_SMILES_TOKENIZER_DATA_PATH)
    ground_truth_tokenizer: RobertaTokenizerFast = RobertaTokenizerFast.from_pretrained(
        "DeepChem/SmilesTokenizer_PubChem_1M", max_len=128
    )

    map_fixed_to_truth_tokens = {
        fixed_tokenizer.unk_token: "[UNK]",
        fixed_tokenizer.cls_token: "[CLS]",
        fixed_tokenizer.sep_token: "[SEP]",
        fixed_tokenizer.pad_token: "[PAD]",
        fixed_tokenizer.mask_token: "[MASK]",
    }
    map_truth_to_fixed_tokens = {v: k for k, v in map_fixed_to_truth_tokens.items()}

    for _, row in df.iterrows():
        for smiles_column in ["smiles", "canonical_smiles"]:
            curr_smiles = row[smiles_column]
            encoded = ground_truth_tokenizer(curr_smiles)

            # Different vocab arrangement
            # So we need to convert the encoded inputs back to their tokens
            decoded_truth_tokens = [
                ground_truth_tokenizer._convert_id_to_token(token_id)
                for token_id in encoded["input_ids"]
            ]
            # SmilesTokenizer_PubChem_1M uses BERT special token syntax
            # while FixedSmileTokenizer uses roBERTa special tokens
            mapped_decoded_truth_tokens = [
                map_truth_to_fixed_tokens.get(token, token)
                for token in decoded_truth_tokens
            ]

            # SmilesTokenizerFast encodings
            fixed_encoded = fixed_tokenizer(curr_smiles)
            decoded_fixed_tokens = [
                fixed_tokenizer._convert_id_to_token(token_id)
                for token_id in fixed_encoded["input_ids"]
            ]
            assert decoded_fixed_tokens == mapped_decoded_truth_tokens
            assert encoded["attention_mask"] == fixed_encoded["attention_mask"]
