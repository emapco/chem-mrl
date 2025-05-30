from collections.abc import Sequence

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.util import get_device_name

from chem_mrl.constants import BASE_MODEL_HIDDEN_DIM, BASE_MODEL_NAME, CHEM_MRL_MODEL_NAME


class ChemMRL:
    """A class to generate molecular (SMILES) embeddings using transformer models."""

    def __init__(
        self,
        model_name: str = CHEM_MRL_MODEL_NAME,
        embedding_size: int | None = None,
        use_half_precision: bool = False,
        batch_size: int = 64,
        normalize_embeddings: bool | None = True,
        device: str | None = None,
    ) -> None:
        """
        Initialize the SMILES embedder with specified parameters.

        Args:
            model_name: Name or file path of the transformer model to use.
                Can either a path to a trained chem-mrl model or a pretrained model on HuggingFace.
            embedding_size: Size of the embedding vector
            use_half_precision: Whether to use FP16 precision
            device: Device to run the model on
            batch_size: Batch size for inference
            normalize_embeddings: Whether to normalize the embeddings
        """
        self._model_name = model_name
        self._embedding_size = embedding_size
        self._use_half_precision = use_half_precision
        self._batch_size = batch_size
        self._device = device

        self._model = self._init_model()
        model_embedding_dimension = (
            self._model.get_sentence_embedding_dimension() or BASE_MODEL_HIDDEN_DIM
        )
        if self._use_half_precision:
            self._model = self._model.half()
        # Check if the requested embedding size is greater than model hidden dimension
        if self._embedding_size is not None and self._embedding_size > model_embedding_dimension:
            raise ValueError(f"embedding_size must be less than equal to {BASE_MODEL_HIDDEN_DIM}")

        # normalize if embeddings are truncated and not specified otherwise
        if normalize_embeddings is None:
            normalize_embeddings = (
                embedding_size is not None and embedding_size < model_embedding_dimension
            )
        self._normalize_embeddings = normalize_embeddings

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def fp_size(self) -> int | None:
        return self._embedding_size

    @property
    def use_half_precision(self) -> bool:
        return self._use_half_precision

    @property
    def device(self) -> str | None:
        return self._device

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def normalize_embeddings(self) -> bool:
        return self._normalize_embeddings

    def _init_model(self) -> SentenceTransformer:
        if self._model_name == BASE_MODEL_NAME:
            embedding_model = models.Transformer(self._model_name)
            embedding_dimension = embedding_model.get_word_embedding_dimension()
            if self._embedding_size is not None and self._embedding_size != embedding_dimension:
                raise ValueError(
                    f"{BASE_MODEL_NAME} only supports embeddings of size {embedding_dimension}"
                )

            return SentenceTransformer(
                modules=[
                    embedding_model,
                    models.Pooling(
                        embedding_model.get_word_embedding_dimension(),
                        pooling_mode="mean",
                    ),
                    models.Normalize(),
                ],
                device=self._device,
            )

        enable_truncate_dim = (
            self._embedding_size is not None and self._embedding_size < BASE_MODEL_HIDDEN_DIM
        )
        return SentenceTransformer(
            self._model_name,
            device=self._device,
            truncate_dim=self._embedding_size if enable_truncate_dim else None,
        )

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
            device=self._device or get_device_name(),
            normalize_embeddings=self._normalize_embeddings or False,
        )

        if self._use_half_precision:
            embeddings = embeddings.astype(np.float16)

        return embeddings

    def get_embedding(
        self,
        smiles: str,
        convert_to_numpy=True,
        show_progress_bar=False,
    ) -> np.ndarray:
        """
        Generate embedding for a single SMILES string.

        Args:
            smiles: SMILES string to embed

        Returns:
            numpy array of embedding
        """
        return self.get_embeddings(
            [smiles], convert_to_numpy=convert_to_numpy, show_progress_bar=show_progress_bar
        )[0]
