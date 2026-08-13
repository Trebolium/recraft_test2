"""
Pairwise cosine similarity helpers, shared by vectorize.py's duplicate
sanity check and find_duplicates.py's full duplicate scan.
"""

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float32), b.astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def find_duplicate_pairs(records: list[dict], threshold: float) -> list[dict]:
    """
    records: list of dicts with at least 'file_path', 'file_name', 'vector'
    (a numpy array), as produced by vector_io.read_vectors_csv.

    Returns one entry per unique pair whose cosine similarity exceeds
    `threshold`, sorted most-similar first, ready to be JSONL-serialized.
    """
    pairs = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            sim = cosine_similarity(records[i]["vector"], records[j]["vector"])
            if sim > threshold:
                pairs.append({
                    "file_path_a": records[i]["file_path"],
                    "file_name_a": records[i]["file_name"],
                    "file_path_b": records[j]["file_path"],
                    "file_name_b": records[j]["file_name"],
                    "similarity": round(sim, 4),
                })

    pairs.sort(key=lambda p: p["similarity"], reverse=True)
    return pairs
