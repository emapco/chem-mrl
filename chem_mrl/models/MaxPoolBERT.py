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

from __future__ import annotations

import torch
from torch import Tensor, nn

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from sentence_transformers.models.Module import Module


class MaxPoolBERT(Module):
    """
    Performs max pooling with multi-head attention over BERT representations.

    This module implements various pooling strategies from the MaxPoolBERT paper, combining depth-wise
    and token-wise information aggregation for improved sentence embeddings.

    Args:
        word_embedding_dimension: Size of the token embeddings
        num_attention_heads: Number of attention heads for multi-head attention. Default: 4
        last_k_layers: Number of last hidden layers to use for pooling. Default: 3
        pooling_strategy: Strategy for pooling. Default: "max_seq_mha"

            - "cls": Use the CLS token from the last layer (baseline)
            - "max_cls": Max pool CLS tokens across last k layers
            - "mha": Additional MHA layer where CLS token attends to all tokens of the last layer
            - "max_seq_mha": Max pool entire sequence across last k layers,
                then MHA with CLS attending to all tokens
            - "mean_seq_mha": Mean pool entire sequence across last k layers,
                then MHA with CLS attending to all tokens
            - "sum_seq_mha": Sum pool entire sequence across last k layers,
                then MHA with CLS attending to all tokens

    References:
        - MaxPoolBERT paper: https://arxiv.org/abs/2505.15696
    """

    config_keys: list[str] = [
        "word_embedding_dimension",
        "num_attention_heads",
        "last_k_layers",
        "pooling_strategy",
    ]

    def __init__(
        self,
        word_embedding_dimension: int,
        num_attention_heads: int = 4,
        last_k_layers: int = 3,
        pooling_strategy: str = "mha",
    ) -> None:
        super().__init__()

        valid_strategies = ["cls", "max_cls", "mha", "max_seq_mha", "mean_seq_mha", "sum_seq_mha"]
        if pooling_strategy not in valid_strategies:
            raise ValueError(f"Invalid pooling_strategy '{pooling_strategy}'. Must be one of: {valid_strategies}")

        self.word_embedding_dimension = word_embedding_dimension
        self.num_attention_heads = num_attention_heads
        self.last_k_layers = last_k_layers
        self.pooling_strategy = pooling_strategy

        if pooling_strategy in ["mha", "max_seq_mha", "mean_seq_mha", "sum_seq_mha"]:
            if word_embedding_dimension % num_attention_heads != 0:
                raise ValueError(
                    f"word_embedding_dimension ({word_embedding_dimension}) "
                    f"must be divisible by num_attention_heads ({num_attention_heads})"
                )
            self.multi_head_attention = nn.MultiheadAttention(
                embed_dim=word_embedding_dimension,
                num_heads=num_attention_heads,
                batch_first=True,
            )
        else:
            self.multi_head_attention = None

    def forward(self, features: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        """
        Compute sentence embeddings using the specified pooling strategy.

        Args:
            features: Dictionary with keys:

                - "all_layer_embeddings": List of hidden states from all transformer layers
                - "attention_mask": Attention mask for the input sequence (optional)

        Returns:
            Updated features dictionary with "sentence_embedding" key
        """
        if "all_layer_embeddings" not in features:
            raise ValueError("MaxPoolBERT requires 'all_layer_embeddings' in features")

        all_layer_embeddings = features["all_layer_embeddings"]
        attention_mask = features.get("attention_mask")

        last_layer = all_layer_embeddings[-1]

        if self.pooling_strategy == "cls":
            # Baseline: Use CLS token from the last layer
            sentence_embedding = last_layer[:, 0, :]

        elif self.pooling_strategy == "max_cls":
            # Max pool CLS tokens across last k layers
            if len(all_layer_embeddings) < self.last_k_layers:
                raise ValueError(
                    f"Not enough layers: got {len(all_layer_embeddings)}, need at least {self.last_k_layers}"
                )

            k_cls_tokens = [layer[:, 0:1, :] for layer in all_layer_embeddings[-self.last_k_layers :]]
            stacked_cls = torch.stack(k_cls_tokens, dim=1)
            stacked_cls = stacked_cls.squeeze(2)

            sentence_embedding = torch.max(stacked_cls, dim=1).values

        elif self.pooling_strategy == "mha":
            # CLS token attends to all tokens of the last layer
            cls_token = last_layer[:, 0:1, :]

            assert self.multi_head_attention is not None
            # Prepare key_padding_mask: ensure it matches the sequence length of last_layer
            key_padding_mask = None
            if attention_mask is not None:
                if attention_mask.dim() > 2:
                    attention_mask = attention_mask.squeeze()
                # Truncate or pad attention_mask to match last_layer sequence length
                seq_len = last_layer.size(1)
                if attention_mask.size(1) > seq_len:
                    attention_mask = attention_mask[:, :seq_len]
                elif attention_mask.size(1) < seq_len:
                    # Pad with ones (valid tokens)
                    batch_size = attention_mask.size(0)
                    padding = torch.ones(
                        batch_size,
                        seq_len - attention_mask.size(1),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    )
                    attention_mask = torch.cat([attention_mask, padding], dim=1)
                key_padding_mask = ~attention_mask.bool()

            attn_output, _ = self.multi_head_attention(
                query=cls_token,
                key=last_layer,
                value=last_layer,
                key_padding_mask=key_padding_mask,
            )

            sentence_embedding = attn_output.squeeze(1)

        elif self.pooling_strategy in {"max_seq_mha", "mean_seq_mha", "sum_seq_mha"}:
            # Max/mean pool entire sequence across last k layers, then MHA
            if len(all_layer_embeddings) < self.last_k_layers:
                raise ValueError(
                    f"Not enough layers: got {len(all_layer_embeddings)}, need at least {self.last_k_layers}"
                )

            k_hidden_states = list(all_layer_embeddings[-self.last_k_layers :])
            stacked_layers = torch.stack(k_hidden_states, dim=1)
            if self.pooling_strategy == "max_seq_mha":
                pooled_seq = torch.max(stacked_layers, dim=1).values
            elif self.pooling_strategy == "mean_seq_mha":
                pooled_seq = torch.mean(stacked_layers, dim=1)
            else:  # sum_seq_mha
                pooled_seq = torch.sum(stacked_layers, dim=1)

            cls_token = pooled_seq[:, 0:1, :]

            assert self.multi_head_attention is not None
            # Prepare key_padding_mask: ensure it matches the sequence length of pooled_seq
            key_padding_mask = None
            if attention_mask is not None:
                if attention_mask.dim() > 2:
                    attention_mask = attention_mask.squeeze()
                # Truncate or pad attention_mask to match pooled_seq sequence length
                seq_len = pooled_seq.size(1)
                if attention_mask.size(1) > seq_len:
                    attention_mask = attention_mask[:, :seq_len]
                elif attention_mask.size(1) < seq_len:
                    # Pad with ones (valid tokens)
                    batch_size = attention_mask.size(0)
                    padding = torch.ones(
                        batch_size,
                        seq_len - attention_mask.size(1),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    )
                    attention_mask = torch.cat([attention_mask, padding], dim=1)
                key_padding_mask = ~attention_mask.bool()

            attn_output, _ = self.multi_head_attention(
                query=cls_token,
                key=pooled_seq,
                value=pooled_seq,
                key_padding_mask=key_padding_mask,
            )

            sentence_embedding = attn_output.squeeze(1)
        else:
            raise ValueError(f"Unsupported pooling_strategy: {self.pooling_strategy}")

        features.update({"sentence_embedding": sentence_embedding})
        return features

    def get_sentence_embedding_dimension(self) -> int:
        return self.word_embedding_dimension

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)

    def __repr__(self) -> str:
        return f"MaxPoolBERT({self.get_config_dict()})"

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> Self:
        hub_kwargs = {
            "subfolder": subfolder,
            "token": token,
            "cache_folder": cache_folder,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        config = cls.load_config(model_name_or_path=model_name_or_path, **hub_kwargs)
        model = cls(**config)
        cls.load_torch_weights(model_name_or_path=model_name_or_path, model=model, **hub_kwargs)
        return model

    @property
    def citation(self) -> str:
        return """
@misc{behrendt2025maxpoolbertenhancingbertclassification,
      title={MaxPoolBERT: Enhancing BERT Classification via Layer- and Token-Wise Aggregation},
      author={Maike Behrendt and Stefan Sylvius Wagner and Stefan Harmeling},
      year={2025},
      eprint={2505.15696},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.15696},
}
"""
