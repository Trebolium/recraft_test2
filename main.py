"""
Entry point for the dataset augmentation pipeline.

Pipeline:
  1. Duplicate the whole input dataset into output-dir.
  2. Within output-dir, duplicate a fraction (default 20%) of images,
     appending "_dup" to their filenames.
  3. Apply modest, random augmentations to just the "_dup" files, in place.

Run for real:
  python main.py --input-dir path/to/images --output-dir path/to/out

Run the built-in test (downloads 20 sample images and runs the pipeline
on them):
  python main.py --test
"""

import argparse
import random
from pathlib import Path

import dataset_utils
import download_samples
from augment import augment_image


def run_pipeline(input_dir: Path, output_dir: Path, fraction: float) -> None:
    """Copy the dataset, duplicate `fraction` of it with "_dup" names, and
    augment those duplicates in place. Prints a short summary at the end."""
    copied = dataset_utils.copy_dataset(input_dir, output_dir)
    dup_paths = dataset_utils.select_and_duplicate(output_dir, fraction)

    for dup_path in dup_paths:
        applied = augment_image(dup_path)
        print(f"  augmented {dup_path.name}: {', '.join(applied)}")

    print(
        f"\nDone. {len(copied)} images copied to {output_dir}, "
        f"{len(dup_paths)} duplicated (~{fraction:.0%}) and augmented."
    )


def main():
    parser = argparse.ArgumentParser(description="Duplicate and augment an image dataset.")
    parser.add_argument("--input-dir", type=Path, help="Directory of source images.")
    parser.add_argument("--output-dir", type=Path, help="Where to write the duplicated+augmented dataset.")
    parser.add_argument("--dup-fraction", type=float, default=0.2, help="Fraction of images to duplicate+augment (default 0.2).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed, for reproducible selection/augmentation.")
    parser.add_argument("--test", action="store_true", help="Download 20 sample images and run the full pipeline on them.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.test:
        test_input = Path("test_data/input")
        test_output = Path("test_data/output")

        print(f"[--test] downloading 20 sample images into {test_input} ...")
        download_samples.download(n=20, dest=test_input)

        print(f"[--test] equivalent manual command:\n"
              f"  python main.py --input-dir {test_input} --output-dir {test_output} --dup-fraction {args.dup_fraction}\n")

        run_pipeline(test_input, test_output, args.dup_fraction)
        return

    if args.input_dir is None:
        parser.error("--input-dir is required unless --test is set.")

    output_dir = args.output_dir or args.input_dir.parent / f"{args.input_dir.name}_augmented"
    run_pipeline(args.input_dir, output_dir, args.dup_fraction)


if __name__ == "__main__":
    main()
