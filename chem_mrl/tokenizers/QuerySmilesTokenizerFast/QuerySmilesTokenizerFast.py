from pathlib import Path

from transformers.models.roberta import RobertaTokenizerFast


class QuerySmilesTokenizerFast(RobertaTokenizerFast):
    """This tokenizer include additional tokens for enhanced smiles embedding querying.
    similar: Computes SMILES similarity between two molecular structures.
    substructure: Determines the presence of a substructure within the second SMILES string.
    families: Identifies the presence of chemical feature families in the second SMILES string.
    """

    def __init__(self, max_len: int = 512, **kwargs):
        curr_file_path = Path(__file__).parent
        vocab_path = Path(curr_file_path, "vocab.json")
        merges_path = Path(curr_file_path, "merges.txt")
        super().__init__(
            vocab_file=vocab_path,
            merges_file=merges_path,
            max_len=max_len,
            **kwargs,
        )
        self.bos_token_id = 0
        self.pad_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.pad_token = "<pad>"
        self.__class__.__name__ = "RobertaTokenizerFast"
