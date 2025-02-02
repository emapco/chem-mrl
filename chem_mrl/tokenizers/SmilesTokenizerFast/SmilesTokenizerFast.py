import json
from pathlib import Path

from transformers import RobertaTokenizerFast

curr_file_path = Path(__file__).parent


class SmilesTokenizerFast(RobertaTokenizerFast):
    def __init__(self, max_len: int = 512, **kwargs):
        vocab_path = Path(curr_file_path, "vocab.json")
        merges_path = Path(curr_file_path, "merges.txt")
        with open(vocab_path, "r") as f:
            self._token_ids: dict[str, int] = json.load(f)

        self._id_tokens: dict[int, str] = {
            value: key for key, value in self._token_ids.items()
        }
        if max_len is None:
            max_len = len(self._token_ids)
        super().__init__(
            vocab_file=vocab_path, merges_file=merges_path, max_len=max_len, **kwargs
        )

        # Initialize special tokens for RoBERTa
        self.unk_token = "<unk>"
        self.cls_token = "<s>"
        self.sep_token = "</s>"
        self.pad_token = "<pad>"
        self.mask_token = "<mask>"
        self.unk_token_id = self._token_ids.get(self.unk_token, 0)
        self.cls_token_id = self._token_ids.get(self.cls_token, 1)
        self.sep_token_id = self._token_ids.get(self.sep_token, 2)
        self.pad_token_id = self._token_ids.get(self.pad_token, 3)
        self.mask_token_id = self._token_ids.get(self.mask_token, 4)

    def _convert_token_to_id(self, token: str) -> int:
        return self._token_ids[token] if token in self._token_ids else self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_tokens[index] if index in self._id_tokens else self.unk_token

    def get_vocab(self) -> dict[str, int]:
        return self._token_ids.copy()

    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = None
    ) -> tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)

    @property
    def vocab_size(self) -> int:
        return len(self._token_ids)
