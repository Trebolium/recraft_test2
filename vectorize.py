"""
Loads images from a directory and stores a DINOv2 embedding vector for each
one (see embeddings.py) as a CSV at outputs/vector_csvs/<dataset-name>.csv,
with columns file_path, file_name, vector.

Run for real:
  python vectorize.py --input-dir path/to/images --dataset-name my_dataset

Run the built-in test (vectorizes the 24-image augmented set -- 20
originals + 4 "_dup" -- from test_data/output, generating that set first
via the augmentation pipeline if it doesn't exist yet, then sanity-checks
that each "_dup" image's embedding is in fact closest to its original):
  python vectorize.py --test
"""

import argparse
from pathlib import Path

import numpy as np

import dataset_utils
import download_samples
from embeddings import embed_image
from similarity import cosine_similarity
from vector_io import write_vectors_csv

OUTPUT_ROOT = Path("outputs/vector_csvs")


def vectorize_directory(input_dir: Path, dataset_name: str) -> tuple[Path, list[tuple[Path, np.ndarray]]]:
    """Embed every image in input_dir and write them to
    outputs/vector_csvs/<dataset_name>.csv. Returns (csv_path, rows)."""
    images = dataset_utils.list_images(input_dir)
    rows = [(path, embed_image(path)) for path in images]

    out_path = OUTPUT_ROOT / f"{dataset_name}.csv"
    write_vectors_csv(rows, out_path)
    return out_path, rows


def _print_duplicate_sanity_check(rows: list[tuple[Path, np.ndarray]]) -> None:
    """For each "_dup" image, check that its embedding is more similar to
    its own original than to any other image in the set -- i.e. that
    cosine similarity is actually a reliable duplicate signal here."""
    by_name = {path.name: vector for path, vector in rows}

    print("\nDuplicate-detection sanity check (cosine similarity):")
    for name, vector in by_name.items():
        stem, suffix = Path(name).stem, Path(name).suffix
        if not stem.endswith("_dup"):
            continue

        original_name = stem[: -len("_dup")] + suffix
        if original_name not in by_name:
            continue

        sim_to_original = cosine_similarity(vector, by_name[original_name])
        other_sims = [cosine_similarity(vector, v) for n, v in by_name.items() if n not in (name, original_name)]
        max_other_sim = max(other_sims)

        verdict = "OK" if sim_to_original > max_other_sim else "WARN: not the closest match!"
        print(f"  {name}: sim to {original_name} = {sim_to_original:.3f}  |  "
              f"max sim to any other image = {max_other_sim:.3f}  [{verdict}]")


def _ensure_test_dataset(test_input: Path, test_output: Path) -> None:
    """Make sure test_data/output (20 originals + 4 "_dup") exists, building
    it via the augmentation pipeline (main.py's --test path) if it doesn't."""
    if test_output.exists() and len(dataset_utils.list_images(test_output)) == 24:
        return

    print("[--test] test_data/output not found (or incomplete) -- generating "
          "it via the augmentation pipeline first...")
    import main as augmentation_main  # imported lazily to avoid a hard dependency for normal runs

    download_samples.download(n=20, dest=test_input)
    augmentation_main.run_pipeline(test_input, test_output, fraction=0.2)


def main():
    parser = argparse.ArgumentParser(description="Vectorize an image dataset into a CSV of embeddings.")
    parser.add_argument("--input-dir", type=Path, help="Directory of images to vectorize.")
    parser.add_argument("--dataset-name", type=str, help="Name for the output CSV (outputs/vector_csvs/<name>.csv). Defaults to the input directory's folder name.")
    parser.add_argument("--test", action="store_true", help="Vectorize the 24-image augmented test set (test_data/output).")
    args = parser.parse_args()

    if args.test:
        test_input, test_output = Path("test_data/input"), Path("test_data/output")
        _ensure_test_dataset(test_input, test_output)

        out_path, rows = vectorize_directory(test_output, "test_dataset_24")
        print(f"[--test] equivalent manual command:\n"
              f"  python vectorize.py --input-dir {test_output} --dataset-name test_dataset_24\n")
        print(f"Done. Wrote {out_path}")

        _print_duplicate_sanity_check(rows)
        return

    if args.input_dir is None:
        parser.error("--input-dir is required unless --test is set.")

    dataset_name = args.dataset_name or args.input_dir.name
    out_path, _ = vectorize_directory(args.input_dir, dataset_name)
    print(f"Done. Wrote {out_path}")


if __name__ == "__main__":
    main()
