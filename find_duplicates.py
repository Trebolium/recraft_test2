"""
Scans a vector CSV (produced by vectorize.py) for suspected duplicate
images: any two images whose embedding cosine similarity exceeds
DUPLICATE_THRESHOLD are written to a JSONL file, one suspected-duplicate
pair per line, at outputs/duplicate_pairs/<dataset-name>.jsonl.

Run for real (expects outputs/vector_csvs/<dataset-name>.csv to exist,
i.e. vectorize.py has already been run on that dataset):
  python find_duplicates.py --dataset-name my_dataset

Run the built-in test (scans the 24-image test set, generating its vector
CSV first via vectorize.py --test if it doesn't exist yet):
  python find_duplicates.py --test
"""

import argparse
import json
from pathlib import Path

from similarity import find_duplicate_pairs
from vector_io import read_vectors_csv

DUPLICATE_THRESHOLD = 0.71
VECTOR_CSV_ROOT = Path("outputs/vector_csvs")
OUTPUT_ROOT = Path("outputs/duplicate_pairs")


def scan_for_duplicates(csv_path: Path, dataset_name: str, threshold: float) -> Path:
    """Read a vector CSV, find every pair of images above `threshold`
    similarity, and write them to outputs/duplicate_pairs/<dataset_name>.jsonl.
    Returns the output path."""
    records = read_vectors_csv(csv_path)
    pairs = find_duplicate_pairs(records, threshold)

    out_path = OUTPUT_ROOT / f"{dataset_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    return out_path


def _ensure_test_vectors(dataset_name: str) -> Path:
    """Make sure outputs/vector_csvs/<dataset_name>.csv exists, generating
    it (and the 24-image test dataset behind it) via vectorize.py's --test
    path if it doesn't."""
    csv_path = VECTOR_CSV_ROOT / f"{dataset_name}.csv"
    if csv_path.exists():
        return csv_path

    print(f"[--test] {csv_path} not found -- generating it via vectorize.py --test first...")
    import vectorize  # imported lazily to avoid a hard dependency (and its torch import) for normal runs

    test_output = Path("test_data/output")
    vectorize._ensure_test_dataset(Path("test_data/input"), test_output)
    vectorize.vectorize_directory(test_output, dataset_name)
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Find suspected duplicate images via embedding similarity.")
    parser.add_argument("--dataset-name", type=str, help="Name matching an existing outputs/vector_csvs/<name>.csv.")
    parser.add_argument("--threshold", type=float, default=DUPLICATE_THRESHOLD, help=f"Cosine similarity threshold above which a pair counts as a duplicate (default {DUPLICATE_THRESHOLD}).")
    parser.add_argument("--test", action="store_true", help="Scan the 24-image test set (test_dataset_24).")
    args = parser.parse_args()

    if args.test:
        dataset_name = "test_dataset_24"
        csv_path = _ensure_test_vectors(dataset_name)
    else:
        if args.dataset_name is None:
            parser.error("--dataset-name is required unless --test is set.")
        dataset_name = args.dataset_name
        csv_path = VECTOR_CSV_ROOT / f"{dataset_name}.csv"
        if not csv_path.exists():
            parser.error(f"{csv_path} does not exist -- run vectorize.py first.")

    out_path = scan_for_duplicates(csv_path, dataset_name, args.threshold)

    pair_count = sum(1 for _ in out_path.open())
    print(f"Done. Found {pair_count} suspected duplicate pair(s) (threshold={args.threshold}). Wrote {out_path}")


if __name__ == "__main__":
    main()
