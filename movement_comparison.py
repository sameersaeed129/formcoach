"""
movement_comparison.py
--------------------------
Compares a performed rep against a reference "ideal" rep using Dynamic
Time Warping (DTW), which -- unlike simple point-by-point distance --
correctly handles the fact that two people (or two reps) rarely move at
exactly the same speed. DTW finds the best alignment between the two
sequences before computing a distance, so a rep performed slightly faster
or slower than the reference isn't penalized just for tempo.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class DTWResult:
    distance: float              # raw DTW distance (lower = more similar)
    normalized_distance: float    # distance / path length, comparable across rep lengths
    alignment_path: List[Tuple[int, int]]


def dtw(seq_a: np.ndarray, seq_b: np.ndarray) -> DTWResult:
    """Classic dynamic time warping with Euclidean local cost, implemented
    directly (no external DTW dependency) so the distance metric and
    alignment path are fully transparent and easy to verify.

    seq_a, seq_b: 1D arrays (e.g. knee angle over time for two reps).
    """
    n, m = len(seq_a), len(seq_b)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            local_cost = abs(seq_a[i - 1] - seq_b[j - 1])
            cost[i, j] = local_cost + min(
                cost[i - 1, j],      # insertion
                cost[i, j - 1],      # deletion
                cost[i - 1, j - 1],  # match
            )

    # Backtrack to recover the alignment path
    path = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step = min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
            if step == cost[i - 1, j - 1]:
                i, j = i - 1, j - 1
            elif step == cost[i - 1, j]:
                i -= 1
            else:
                j -= 1
    path.reverse()

    total_distance = float(cost[n, m])
    return DTWResult(
        distance=total_distance,
        normalized_distance=total_distance / len(path),
        alignment_path=path,
    )


def compare_to_reference(rep_sequence: np.ndarray, reference_sequence: np.ndarray) -> DTWResult:
    """Convenience wrapper: compares one rep's knee-angle curve to a
    reference 'ideal' curve. Both are 1D arrays; lengths may differ."""
    return dtw(np.asarray(rep_sequence, dtype=float), np.asarray(reference_sequence, dtype=float))


def similarity_score(dtw_result: DTWResult, scale_deg: float = 15.0) -> float:
    """Converts a normalized DTW distance (in degrees) into an intuitive
    0-100 similarity score using an exponential decay -- a normalized
    distance of 0 gives 100, and larger deviations decay smoothly toward 0.
    `scale_deg` controls how many degrees of average deviation correspond
    to roughly a 63% score."""
    return float(100 * np.exp(-dtw_result.normalized_distance / scale_deg))
