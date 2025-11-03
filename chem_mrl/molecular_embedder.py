# Copyright 2025 Emmanuel Cortes. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
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


logger = logging.getLogger(__name__)


class ChemMRL(SentenceTransformer):
    def __init__(
        self,
        model_name_or_path: str | None = CHEM_MRL_MODEL_NAME,
        modules: Iterable[torch.nn.Module] | None = None,
        device: str | None = None,
        prompts: dict[str, str] | None = None,
        default_prompt_name: str | None = None,
        similarity_fn_name: str | SimilarityFunction | None = None,
        cache_folder: str | None = None,
        trust_remote_code: bool | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        token: bool | str | None = None,
        use_auth_token: bool | str | None = None,
        truncate_dim: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        compile_kwargs: dict[str, Any] | None = None,
        model_card_data: SentenceTransformerModelCardData | None = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
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
                See the `PreTrainedModel.from_pretrained
                <https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained>`_
                documentation for more details.

                Particularly useful options are:
                - **dtype**: Override the default `torch.dtype` and load the model under a specific `dtype`.
                    The different options are:
                    1. `torch.float16`, `torch.bfloat16` or `torch.float`: load in a specified
                        `dtype`, ignoring the model's `config.dtype` if one exists. If not specified - the model will
                        get loaded in `torch.float` (fp32).
                    2. `"auto"` - A `dtype` entry in the `config.json` file of the model will be
                        attempted to be used. If this entry isn't found then next check the `dtype` of the first weight in
                        the checkpoint that's of a floating point type and use that as `dtype`. This will load the model
                        using the `dtype` it was saved in at the end of the training. It can't be used as an indicator of how
                        the model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32.
                - **attn_implementation**: The attention implementation to use in the model (if relevant). Can be any of
                    `"eager"` (manual implementation of the attention), `"sdpa"` (using `F.scaled_dot_product_attention`),
                    or `"flash_attention_2"` (using `Dao-AILab/flash-attention`).
                    By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"`
                    implementation.
                - **provider**: If backend is "onnx", this is the provider to use for inference, for example "CPUExecutionProvider",
                    "CUDAExecutionProvider", etc. See https://onnxruntime.ai/docs/execution-providers/ for all ONNX execution providers.
                - **file_name**: If backend is "onnx" or "openvino", this is the file name to load, useful for loading optimized
                    or quantized ONNX or OpenVINO models.
                - **export**: If backend is "onnx" or "openvino", then this is a boolean flag specifying whether this model should
                    be exported to the backend. If not specified, the model will be exported only if the model repository or directory
                    does not already contain an exported model.
            tokenizer_kwargs (Dict[str, Any], optional): Additional tokenizer configuration parameters to be passed to the Hugging Face Transformers tokenizer.
                See the `AutoTokenizer.from_pretrained
                <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained>`_
                documentation for more details.
            config_kwargs (Dict[str, Any], optional): Additional model configuration parameters to be passed to the Hugging Face Transformers config.
                See the `AutoConfig.from_pretrained
                <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoConfig.from_pretrained>`_
                documentation for more details.
            compile_kwargs (dict, optional): Configuration for PyTorch 2.0+ model compilation via torch.compile.
                If None (default), torch.compile is not applied. If provided, applies torch.compile to the transformer model with the specified options.
                See the `torch.compile
                <https://docs.pytorch.org/docs/2.8/generated/torch.compile.html>`_
                documentation for more details.

                Particularly useful options are:
                - **fullgraph** (bool): Whether it is OK to break model into several subgraphs. If False (default),
                    torch.compile attempts to discover compilable regions in the function that it will optimize.
                    If True, requires the entire function be capturable into a single graph. Defaults to False.
                - **dynamic** (bool or None): Use dynamic shape tracing. When True, generates kernels that are as dynamic
                    as possible to avoid recompilations when sizes change. When False, never generates dynamic kernels
                    and always specializes.
                    By default (None), automatically detects if dynamism has occurred and compiles a more dynamic kernel upon recompile.
                - **backend** (str or Callable): Backend to use for compilation. Defaults to "inductor".
                - **mode** (str or None): Optimization mode. Options include:
                    - `"default"`: Good balance between performance and overhead
                    - `"reduce-overhead"`: Reduces Python overhead with CUDA graphs, useful for small batches
                    - `"max-autotune"`: Leverages Triton/template-based operations, enables CUDA graphs on GPU
                    - `"max-autotune-no-cudagraphs"`: Similar to max-autotune but without CUDA graphs
                - **options** (dict or None): Dictionary of backend-specific options. Notable options include:
                    - `epilogue_fusion`: Fuses pointwise ops into templates (requires max_autotune)
                    - `max_autotune`: Profiles to pick the best matmul configuration
                    - `fallback_random`: Useful for debugging accuracy issues
                    - `shape_padding`: Pads matrix shapes for better alignment on GPUs
                    - `triton.cudagraphs`: Reduces Python overhead with CUDA graphs
                    - `trace.enabled`: Useful debugging flag
                    - `trace.graph_diagram`: Shows graph visualization after fusion
                - **disable** (bool): Turn torch.compile() into a no-op for testing. Defaults to False.
                Example: `{"backend": "inductor", "mode": "max-autotune", "fullgraph": False}`
            model_card_data (:class:`~sentence_transformers.model_card.SentenceTransformerModelCardData`, optional): A model
                card data object that contains information about the model. This is used to generate a model card when saving
                the model. If not set, a default model card data object is created.
            backend (str): The backend to use for inference. Can be one of "torch" (default), "onnx", or "openvino". Only torch is tested extensively.
                See https://sbert.net/docs/sentence_transformer/usage/efficiency.html for benchmarking information
                on the different backends.
            use_half_precision (bool, optional): Whether to use half precision for the model. This can reduce memory usage
                and speed up inference, but may reduce accuracy. Defaults to False.
        """  # noqa: E501

        # Automatically trust remote code for Derify models unless explicitly specified otherwise
        if model_name_or_path and model_name_or_path.startswith("Derify/") and trust_remote_code is None:
            trust_remote_code = True

        # Temporary change of class name to ensure proper loading of pretrained models
        original_name = self.__class__.__name__
        self.__class__.__name__ = "SentenceTransformer"

        super().__init__(
            model_name_or_path=model_name_or_path,
            modules=modules,
            device=device,
            prompts=prompts,
            default_prompt_name=default_prompt_name,
            similarity_fn_name=similarity_fn_name,  # type: ignore[arg-type]
            cache_folder=cache_folder,
            trust_remote_code=bool(trust_remote_code),
            revision=revision,
            local_files_only=local_files_only,
            token=token,
            use_auth_token=use_auth_token,
            truncate_dim=truncate_dim,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
            model_card_data=model_card_data,
            backend=backend,
        )
        self.__class__.__name__ = original_name  # Restore original class name

        self.__use_half_precision = use_half_precision
        if self.__use_half_precision:
            for module in self.modules():
                if isinstance(module, PreTrainedModel):
                    module.half()

        self.__compile_kwargs = compile_kwargs
        if self.__compile_kwargs is not None and not self.__compile_kwargs.get("disable", False):
            self._apply_torch_compile()

    def _apply_torch_compile(self) -> None:
        """Apply torch.compile to the transformer model with the provided compile_kwargs."""
        if self.__compile_kwargs is None:
            return

        # Enable capturing scalar outputs to avoid compilation issues - ModChemBERT specific
        if hasattr(torch, "_dynamo"):
            torch._dynamo.config.capture_scalar_outputs = True

        first_module = self._first_module()
        if hasattr(first_module, "auto_model"):
            try:
                compile_kwargs = {
                    k: v
                    for k, v in self.__compile_kwargs.items()
                    if k in ["fullgraph", "dynamic", "backend", "mode", "options", "disable"]
                }

                first_module.auto_model = torch.compile(first_module.auto_model, **compile_kwargs)  # type: ignore[arg-type,call-overload]
                backend_name = compile_kwargs.get("backend", "inductor")
                logger.info(f"Applied torch.compile with backend={backend_name} to transformer model")
            except Exception as e:
                logger.warning(f"Failed to apply torch.compile: {e}")
                if hasattr(torch, "_dynamo"):
                    logger.info(f"Available backends: {torch._dynamo.list_backends()}")
                raise

    @property
    def backbone(self) -> "ChemMRL":
        """Returns self for backward compatibility.

        Note:
            In earlier versions (< 0.8.0), ChemMRL used composition with a `backbone`
            attribute that was a SentenceTransformer instance. Now ChemMRL directly
            inherits from SentenceTransformer, so `backbone` returns `self`.
        """
        return self

    @property
    def use_half_precision(self) -> bool:
        """Whether half precision is used for embeddings."""
        return self.__use_half_precision

    def embed(
        self,
        smiles: str | list[str] | np.ndarray,
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
        Computes SMILES embeddings.

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

        embeddings = self.encode(
            sentences=smiles,
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

        if self.__use_half_precision and precision == "float32":
            if isinstance(embeddings, np.ndarray | pd.DataFrame | pd.Series):
                embeddings = embeddings.astype(np.float16)
            elif isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.half()

        return embeddings
