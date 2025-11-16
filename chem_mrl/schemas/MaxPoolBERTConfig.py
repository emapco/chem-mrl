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

from dataclasses import asdict, dataclass

from .Enums import MaxPoolBERTStrategyOption


@dataclass
class MaxPoolBERTConfig:
    enable: bool = False
    num_attention_heads: int = 4
    last_k_layers: int = 3
    pooling_strategy: MaxPoolBERTStrategyOption = MaxPoolBERTStrategyOption.mha
    asdict = asdict

    def __post_init__(self):
        # check types
        if not isinstance(self.enable, bool):
            raise TypeError("enable must be a bool")
        if not isinstance(self.num_attention_heads, int):
            raise TypeError("num_attention_heads must be an int")
        if not isinstance(self.last_k_layers, int):
            raise TypeError("last_k_layers must be an int")

        # check values
        if not isinstance(self.pooling_strategy, MaxPoolBERTStrategyOption):
            raise ValueError(f"pooling_strategy must be one of {MaxPoolBERTStrategyOption.to_list()}")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.last_k_layers <= 0:
            raise ValueError("last_k_layers must be positive")
