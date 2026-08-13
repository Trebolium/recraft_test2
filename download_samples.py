"""
Downloads a small throwaway test dataset from the internet, used only by
main.py's --test flag. Uses picsum.photos, which redirects each request to
a different random stock photo -- no API key required.
"""

import shutil
from pathlib import Path

import requests

PICSUM_URL = "https://picsum.photos/640/480"


def download(n: int, dest: Path) -> list[Path]:
    """Download `n` random images into `dest`, named sample_01.jpg etc.
    Returns the list of downloaded file paths.

    `dest` is wiped first: each --test run fetches a fresh random batch, so a
    stale image left over from a previous run would sit under the same
    filename as new (different) content."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    downloaded = []
    for i in range(1, n + 1):
        response = requests.get(PICSUM_URL, timeout=15)
        response.raise_for_status()

        out_path = dest / f"sample_{i:02d}.jpg"
        out_path.write_bytes(response.content)
        downloaded.append(out_path)

    return downloaded
