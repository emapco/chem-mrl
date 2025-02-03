from typing import Sequence

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, models

from chem_mrl.constants import BASE_MODEL_NAME, EMBEDDING_MODEL_HIDDEN_DIM


class ChemMRL:
    """A class to generate molecular (SMILES) embeddings using transformer models."""

    def __init__(
        self,
        model_name: str,
        fp_size: int | None = None,
        use_half_precision: bool = True,
        device: str = "cuda",
        batch_size: int = 4096,
        normalize_embeddings: bool | None = True,
    ) -> None:
        """
        Initialize the SMILES embedder with specified parameters.

        Args:
            model_name: Name or file path of the transformer model to use.
                Can either `seyonec/SMILES_tokenized_PubChem_shard00_160k` or a path to a trained chem-mrl model.
            fp_size: Size of the embedding vector (for truncation)
            use_half_precision: Whether to use FP16 precision
            device: Device to run the model on ('cuda' or 'cpu')
            batch_size: Batch size for inference
            normalize_embeddings: Whether to normalize the embeddings
        """
        if fp_size is not None and fp_size < 32:
            raise ValueError("fp_size must be greater than 32")
        if fp_size is not None and fp_size > EMBEDDING_MODEL_HIDDEN_DIM:
            raise ValueError(f"fp_size must be less than {EMBEDDING_MODEL_HIDDEN_DIM}")
        self._model_name = model_name
        self._fp_size = fp_size
        self._use_half_precision = use_half_precision
        self._device = device
        self._batch_size = batch_size
        if normalize_embeddings is None:
            normalize_embeddings = (
                fp_size is not None and fp_size < EMBEDDING_MODEL_HIDDEN_DIM
            )
        self._normalize_embeddings = normalize_embeddings

        if model_name == BASE_MODEL_NAME:
            if fp_size is not None and fp_size != EMBEDDING_MODEL_HIDDEN_DIM:
                raise ValueError(
                    f"{BASE_MODEL_NAME} only supports embeddings of size 768"
                )
            word_embedding_model = models.Transformer(model_name)
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(), pooling_mode="mean"
            )
            self._model = SentenceTransformer(
                modules=[word_embedding_model, pooling_model],
                device=device,
            )
        else:
            enable_truncate_dim = (
                fp_size is not None and fp_size < EMBEDDING_MODEL_HIDDEN_DIM
            )
            self._model = SentenceTransformer(
                model_name,
                device=device,
                truncate_dim=fp_size if enable_truncate_dim else None,
            )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def fp_size(self) -> int | None:
        return self._fp_size

    @property
    def use_half_precision(self) -> bool:
        return self._use_half_precision

    @property
    def device(self) -> str:
        return self._device

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def normalize_embeddings(self) -> bool:
        return self._normalize_embeddings

    def _set_model_half_precision(self) -> None:
        if self._use_half_precision:
            self._model = self._model.half()

    def get_embeddings(
        self,
        smiles_list: Sequence[str] | pd.Series,
        show_progress_bar=False,
        convert_to_numpy=True,
    ) -> np.ndarray:
        """
        Generate embeddings for a list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings to embed

        Returns:
            numpy array of embeddings
        """
        embeddings: np.ndarray = self._model.encode(
            smiles_list,  # type: ignore
            batch_size=self._batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            device=self._device,
            normalize_embeddings=self._normalize_embeddings,
        )

        if self._use_half_precision:
            embeddings = embeddings.astype(np.float16)

        return embeddings

    def get_embedding(self, smiles: str, convert_to_numpy=True) -> np.ndarray:
        """
        Generate embedding for a single SMILES string.

        Args:
            smiles: SMILES string to embed

        Returns:
            numpy array of embedding
        """
        return self.get_embeddings([smiles], convert_to_numpy=convert_to_numpy)[0]
