from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData
from transformers.modeling_utils import PreTrainedModel

from chem_mrl.constants import CHEM_MRL_MODEL_NAME
from chem_mrl.similarity_functions import SimilarityFunction, patch_sentence_transformer

# Patch to support SentenceTransformer with custom similarity functions
patch_sentence_transformer()


class ChemMRL:
    """A class to generate molecular (SMILES) embeddings using a pretrained transformer model."""

    def __init__(
        self,
        model_name_or_path: str | None = CHEM_MRL_MODEL_NAME,
        modules: Iterable[torch.nn.Module] | None = None,
        device: str | None = None,
        prompts: dict[str, str] | None = None,
        default_prompt_name: str | None = None,
        similarity_fn_name: str | SimilarityFunction | None = None,
        cache_folder: str | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
        token: bool | str | None = None,
        use_auth_token: bool | str | None = None,
        truncate_dim: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        model_card_data: SentenceTransformerModelCardData | None = None,
        use_half_precision: bool = False,
    ) -> None:
        """
        Loads or creates a ChemMRL model that can be used to map sentences / text to embeddings.

        Args:
            model_name_or_path (str, optional): If it is a filepath on disc, it loads the model from that path. If it is not a path,
                it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model
                from the Hugging Face Hub with that name.
            modules (Iterable[nn.Module], optional): A list of torch Modules that should be called sequentially, can be used to create custom
                SentenceTransformer models from scratch.
            device (str, optional): Device (like "cuda", "cpu", "mps", "npu") that should be used for computation. If None, checks if a GPU
                can be used.
            prompts (Dict[str, str], optional): A dictionary with prompts for the model. The key is the prompt name, the value is the prompt text.
                The prompt text will be prepended before any text to encode. For example:
                `{"query": "query: ", "passage": "passage: "}` or `{"clustering": "Identify the main category based on the
                titles in "}`.
            default_prompt_name (str, optional): The name of the prompt that should be used by default. If not set,
                no prompt will be applied.
            similarity_fn_name (str or SimilarityFunction, optional): The name of the similarity function to use. Valid options are "cosine", "dot",
                "euclidean", "manhattan", and "tanimoto. If not set, it is automatically set to "cosine" if `similarity` or
                `similarity_pairwise` are called while `model.similarity_fn_name` is still `None`.
            cache_folder (str, optional): Path to store models. Can also be set by the SENTENCE_TRANSFORMERS_HOME environment variable.
            trust_remote_code (bool, optional): Whether or not to allow for custom models defined on the Hub in their own modeling files.
                This option should only be set to True for repositories you trust and in which you have read the code, as it
                will execute code present on the Hub on your local machine.
            revision (str, optional): The specific model version to use. It can be a branch name, a tag name, or a commit id,
                for a stored model on Hugging Face.
            local_files_only (bool, optional): Whether or not to only look at local files (i.e., do not try to download the model).
            token (bool or str, optional): Hugging Face authentication token to download private models.
            use_auth_token (bool or str, optional): Deprecated argument. Please use `token` instead.
            truncate_dim (int, optional): The dimension to truncate sentence embeddings to. Defaults to None.
            model_kwargs (Dict[str, Any], optional): Additional model configuration parameters to be passed to the Hugging Face Transformers model.
                Particularly useful options are:

                - ``torch_dtype``: Override the default `torch.dtype` and load the model under a specific `dtype`.
                The different options are:

                        1. ``torch.float16``, ``torch.bfloat16`` or ``torch.float``: load in a specified
                        ``dtype``, ignoring the model's ``config.torch_dtype`` if one exists. If not specified - the model will
                        get loaded in ``torch.float`` (fp32).

                        2. ``"auto"`` - A ``torch_dtype`` entry in the ``config.json`` file of the model will be
                        attempted to be used. If this entry isn't found then next check the ``dtype`` of the first weight in
                        the checkpoint that's of a floating point type and use that as ``dtype``. This will load the model
                        using the ``dtype`` it was saved in at the end of the training. It can't be used as an indicator of how
                        the model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32.
                - ``attn_implementation``: The attention implementation to use in the model (if relevant). Can be any of
                `"eager"` (manual implementation of the attention), `"sdpa"` (using `F.scaled_dot_product_attention
                <https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html>`_),
                or `"flash_attention_2"` (using `Dao-AILab/flash-attention <https://github.com/Dao-AILab/flash-attention>`_).
                By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"`
                implementation.
                - ``provider``: If backend is "onnx", this is the provider to use for inference, for example "CPUExecutionProvider",
                "CUDAExecutionProvider", etc. See https://onnxruntime.ai/docs/execution-providers/ for all ONNX execution providers.
                - ``file_name``: If backend is "onnx" or "openvino", this is the file name to load, useful for loading optimized
                or quantized ONNX or OpenVINO models.
                - ``export``: If backend is "onnx" or "openvino", then this is a boolean flag specifying whether this model should
                be exported to the backend. If not specified, the model will be exported only if the model repository or directory
                does not already contain an exported model.

                See the `PreTrainedModel.from_pretrained
                <https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained>`_
                documentation for more details.
            tokenizer_kwargs (Dict[str, Any], optional): Additional tokenizer configuration parameters to be passed to the Hugging Face Transformers tokenizer.
                See the `AutoTokenizer.from_pretrained
                <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained>`_
                documentation for more details.
            config_kwargs (Dict[str, Any], optional): Additional model configuration parameters to be passed to the Hugging Face Transformers config.
                See the `AutoConfig.from_pretrained
                <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoConfig.from_pretrained>`_
                documentation for more details.
            model_card_data (:class:`~sentence_transformers.model_card.SentenceTransformerModelCardData`, optional): A model
                card data object that contains information about the model. This is used to generate a model card when saving
                the model. If not set, a default model card data object is created.
            use_half_precision (bool, optional): Whether to use half precision for the model. This can reduce memory usage
                and speed up inference, but may reduce accuracy. Defaults to False.
        """  # noqa: E501
        self.backbone = SentenceTransformer(
            model_name_or_path=model_name_or_path,
            modules=modules,
            device=device,
            prompts=prompts,
            default_prompt_name=default_prompt_name,
            similarity_fn_name=similarity_fn_name,  # type: ignore[arg-type]
            cache_folder=cache_folder,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
            token=token,
            use_auth_token=use_auth_token,
            truncate_dim=truncate_dim,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
            model_card_data=model_card_data,
            backend="torch",
        )
        self._use_half_precision = use_half_precision
        if self._use_half_precision:
            for module in self.backbone.modules():
                if isinstance(module, PreTrainedModel):
                    module.half()

    @property
    def use_half_precision(self) -> bool:
        return self._use_half_precision

    def embed(
        self,
        smiles: str | list[str],
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str | list[str | torch.device] | None = None,
        normalize_embeddings: bool = False,
        truncate_dim: int | None = None,
        pool: dict[Literal["input", "output", "processes"], Any] | None = None,
        chunk_size: int | None = None,
        **kwargs,
    ):
        """
        Computes sentence embeddings.

        .. tip::

            If you are unsure whether you should use :meth:`encode`, :meth:`encode_query`, or :meth:`encode_document`,
            your best bet is to use :meth:`encode_query` and :meth:`encode_document` for Information Retrieval tasks
            with clear query and document/passage distinction, and use :meth:`encode` for all other tasks.

            Note that :meth:`encode` is the most general method and can be used for any task, including Information
            Retrieval, and that if the model was not trained with predefined prompts and/or task types, then all three
            methods will return identical embeddings.

        Args:
            smiles (Union[str, List[str]]): The SMILES to embed.
            batch_size (int, optional): The batch size used for the computation. Defaults to 32.
            show_progress_bar (bool, optional): Whether to output a progress bar when encode sentences. Defaults to None.
            precision (Literal["float32", "int8", "uint8", "binary", "ubinary"], optional): The precision to use for the embeddings.
                Can be "float32", "int8", "uint8", "binary", or "ubinary". All non-float32 precisions are quantized embeddings.
                Quantized embeddings are smaller in size and faster to compute, but may have a lower accuracy. They are useful for
                reducing the size of the embeddings of a corpus for semantic search, among other tasks. Defaults to "float32".
            convert_to_numpy (bool, optional): Whether the output should be a list of numpy vectors. If False, it is a list of PyTorch tensors.
                Defaults to True.
            convert_to_tensor (bool, optional): Whether the output should be one large tensor. Overwrites `convert_to_numpy`.
                Defaults to False.
            device (Union[str, List[str], None], optional): Device(s) to use for computation. Can be:

                - A single device string (e.g., "cuda:0", "cpu") for single-process encoding
                - A list of device strings (e.g., ["cuda:0", "cuda:1"], ["cpu", "cpu", "cpu", "cpu"]) to distribute
                  encoding across multiple processes
                - None to auto-detect available device for single-process encoding
                If a list is provided, multi-process encoding will be used. Defaults to None.
            normalize_embeddings (bool, optional): Whether to normalize returned vectors to have length 1. In that case,
                the faster dot-product (util.dot_score) instead of cosine similarity can be used. Defaults to False.
            truncate_dim (int, optional): The dimension to truncate sentence embeddings to.
                Truncation is especially interesting for `Matryoshka models <https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html>`_,
                i.e. models that are trained to still produce useful embeddings even if the embedding dimension is reduced.
                Truncated embeddings require less memory and are faster to perform retrieval with, but note that inference
                is just as fast, and the embedding performance is worse than the full embeddings. If None, the ``truncate_dim``
                from the model initialization is used. Defaults to None.
            pool (Dict[Literal["input", "output", "processes"], Any], optional): A pool created by `start_multi_process_pool()`
                for multi-process encoding. If provided, the encoding will be distributed across multiple processes.
                This is recommended for large datasets and when multiple GPUs are available. Defaults to None.
            chunk_size (int, optional): Size of chunks for multi-process encoding. Only used with multiprocessing, i.e. when
                ``pool`` is not None or ``device`` is a list. If None, a sensible default is calculated. Defaults to None.

        Returns:
            Union[List[Tensor], ndarray, Tensor]: By default, a 2d numpy array with shape [num_inputs, output_dimension] is returned.
            If only one string input is provided, then the output is a 1d array with shape [output_dimension]. If ``convert_to_tensor``,
            a torch Tensor is returned instead. If ``self.truncate_dim <= output_dimension`` then output_dimension is ``self.truncate_dim``.
        """  # noqa: E501

        embeddings = self.backbone.encode(
            sentences=smiles,  # type: ignore[arg-type]
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            output_value="sentence_embedding",
            precision=precision,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
            device=device,
            normalize_embeddings=normalize_embeddings,
            truncate_dim=truncate_dim,
            pool=pool,
            chunk_size=chunk_size,
            **kwargs,
        )

        if self._use_half_precision and precision == "float32":
            if isinstance(embeddings, np.ndarray | pd.DataFrame | pd.Series):
                embeddings = embeddings.astype(np.float16)
            elif isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.half()

        return embeddings
