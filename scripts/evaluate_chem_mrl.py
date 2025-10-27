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

from datasets import load_dataset

from chem_mrl import ChemMRL
from chem_mrl.evaluation import EmbeddingSimilarityEvaluator
from chem_mrl.schemas.Enums import ChemMrlEvalMetricOption, EvalSimilarityFctOption

chem_mrl = ChemMRL(
    "Derify/ChemMRL",
    trust_remote_code=True,
    model_kwargs={"dtype": "bfloat16"},
)
dataset = load_dataset("Derify/pubchem_10m_genmol_similarity", split="test")

evaluator = EmbeddingSimilarityEvaluator(
    dataset["smiles_a"],
    dataset["smiles_b"],
    dataset["similarity"],
    batch_size=2048,
    main_similarity=EvalSimilarityFctOption.tanimoto,
    metric=ChemMrlEvalMetricOption.spearman,
    name="pubchem_10m_genmol_similarity",
    show_progress_bar=True,
    write_csv=True,
    precision="float32",
)

chem_mrl.backbone.evaluate(evaluator, output_path="evaluation_results")
