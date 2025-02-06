from pathlib import Path

from transformers import RobertaTokenizerFast

curr_file_path = Path(__file__).parent


class SmilesTokenizerFast(RobertaTokenizerFast):
    def __init__(self, max_len: int = 512, **kwargs):
        vocab_path = Path(curr_file_path, "vocab.json")
        merges_path = Path(curr_file_path, "merges.txt")
        super().__init__(
            vocab_file=vocab_path,
            merges_file=merges_path,
            max_len=max_len,
            **kwargs,
        )
        self.unk_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 3
