"""
Filesystem-level helpers for the augmentation pipeline: copying a dataset
and picking a random subset of it to duplicate as "_dup" files.
No image-transform logic lives here -- see augment.py for that.
"""

import random
import shutil
from pathlib import Path

# Extensions we treat as "images". Anything else in the dir is ignored.
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(directory: Path) -> list[Path]:
    """Return all supported image files directly inside `directory` (non-recursive)."""
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def copy_dataset(input_dir: Path, output_dir: Path) -> list[Path]:
    """
    Duplicate the entire dataset from input_dir into output_dir.
    Returns the list of copied image paths (in output_dir).

    output_dir is wiped first if it already exists, so re-running the
    pipeline never leaves stale "_dup" files (from a previous run, possibly
    against different source images) mixed in with the current dataset.
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    copied = []
    for src in list_images(input_dir):
        dst = output_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def select_and_duplicate(output_dir: Path, fraction: float = 0.2) -> list[Path]:
    """
    Randomly select `fraction` of the images currently in output_dir, and for
    each one create a copy named "<stem>_dup<ext>". These _dup files are the
    only ones later augmented -- originals are left untouched.

    Returns the list of newly created "_dup" paths.
    """
    images = list_images(output_dir)
    n_to_duplicate = round(len(images) * fraction)
    chosen = random.sample(images, n_to_duplicate)

    dup_paths = []
    for src in chosen:
        dup_path = src.with_name(f"{src.stem}_dup{src.suffix}")
        shutil.copy2(src, dup_path)
        dup_paths.append(dup_path)
    return dup_paths
