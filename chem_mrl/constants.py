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

import os

_curr_file_dir = os.path.dirname(os.path.abspath(__file__))
_project_root_dir = os.path.dirname(_curr_file_dir)
_root_data_dir = os.path.join(_project_root_dir, "data")
OUTPUT_DATA_DIR = _root_data_dir
BASE_MODEL_HIDDEN_DIM = 1024
TEST_FP_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
CHEM_MRL_DIMENSIONS = [1024, 512, 256, 128, 64, 32, 16, 8]
BASE_MODEL_DIMENSIONS = [BASE_MODEL_HIDDEN_DIM]
BASE_MODEL_NAME = "Derify/ModChemBERT-IR-BASE"
CHEM_MRL_MODEL_NAME = "Derify/ChemMRL"
TRAINED_CHEM_MRL_DIMENSIONS = [1024, 512, 256, 128, 64, 32, 16, 8]
OPTUNA_DB_URI = "postgresql://postgres:password@127.0.0.1:5432/postgres"
