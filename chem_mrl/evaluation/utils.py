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
