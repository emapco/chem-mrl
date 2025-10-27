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

import csv
import os


def _write_results_to_csv(
    write_csv: bool,
    csv_file: str,
    csv_headers: list[str],
    output_path: str,
    results: list,
):
    if output_path == "" or not write_csv:
        return
    csv_path = os.path.join(output_path, csv_file)
    output_file_exists = os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(
        csv_path,
        newline="",
        mode="a" if output_file_exists else "w",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        if not output_file_exists:
            writer.writerow(csv_headers)

        writer.writerow(results)
