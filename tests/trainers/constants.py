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
from pathlib import Path

curr_file_path = Path(__file__).parent
_parent_dir = Path(curr_file_path).parent
_test_data_dir = Path(_parent_dir, "data")


TEST_CHEM_MRL_PATH = os.path.join(_test_data_dir, "test_chem_mrl.parquet")
TEST_CLASSIFICATION_PATH = os.path.join(_test_data_dir, "test_classification.parquet")
