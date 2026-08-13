"""
Read/write helpers for the vector CSV format: one row per image with
columns file_path, file_name, vector. The vector is stored as a JSON array
string so a 256-value fp16 embedding round-trips cleanly through a single
CSV cell.
"""

import csv
import json
from pathlib import Path

import numpy as np

CSV_COLUMNS = ["file_path", "file_name", "vector"]


def write_vectors_csv(rows: list[tuple[Path, np.ndarray]], out_path: Path) -> None:
    """rows: list of (image_path, vector) pairs. Writes out_path with
    columns file_path, file_name, vector."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for path, vector in rows:
            vector_str = json.dumps([float(v) for v in vector])
            writer.writerow([str(path), path.name, vector_str])


def read_vectors_csv(csv_path: Path) -> list[dict]:
    """Load a vector CSV back into a list of dicts, with 'vector' restored
    as a numpy fp16 array."""
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        records = []
        for row in reader:
            row["vector"] = np.array(json.loads(row["vector"]), dtype=np.float16)
            records.append(row)
    return records
