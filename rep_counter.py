"""
rep_counter.py
------------------
Segments a continuous knee-angle time series into individual repetitions
using peak/valley detection on the signal, and smooths noisy per-frame
pose estimates first so detection isn't thrown off by jitter.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy.signal import find_peaks, savgol_filter


@dataclass
class Rep:
    rep_number: int
    start_frame: int
    bottom_frame: int   # frame of maximum knee bend (minimum angle) within this rep
    end_frame: int
    min_knee_angle: float
    max_knee_angle: float


def smooth_signal(values: np.ndarray, window: int = 7, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing to reduce per-frame pose-estimation jitter
    before doing peak detection. Falls back to the raw signal if there
    isn't enough data for the requested window."""
    n = len(values)
    if n < window:
        return values
    w = window if window % 2 == 1 else window + 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w < 3:
        return values
    return savgol_filter(values, window_length=w, polyorder=min(polyorder, w - 1))


def segment_reps(knee_angles: List[float], frame_indices: List[int],
                  min_rep_depth_deg: float = 15.0, min_frames_between_reps: int = 10) -> List[Rep]:
    """
    Finds each rep as a full down-up cycle in the knee angle signal.
    A "standing" moment is a local maximum (near-straight leg, large angle);
    a "bottom of squat" moment is a local minimum (bent leg, small angle).
    A rep = standing -> bottom -> standing.
    """
    angles = np.array(knee_angles, dtype=float)
    valid_mask = ~np.isnan(angles)
    if valid_mask.sum() < min_frames_between_reps:
        return []

    # Interpolate small gaps (frames where pose wasn't detected) so the
    # signal is continuous for peak detection.
    if not valid_mask.all():
        angles = np.interp(
            np.arange(len(angles)),
            np.where(valid_mask)[0],
            angles[valid_mask],
        )

    smoothed = smooth_signal(angles)

    # Peaks = standing positions (local maxima in knee angle)
    peaks, _ = find_peaks(smoothed, distance=min_frames_between_reps, prominence=min_rep_depth_deg / 2)
    # Valleys = bottom-of-squat positions (local minima)
    valleys, _ = find_peaks(-smoothed, distance=min_frames_between_reps, prominence=min_rep_depth_deg / 2)

    reps = []
    rep_num = 1
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]
        # Find the deepest valley strictly between this pair of standing peaks
        between = [v for v in valleys if start < v < end]
        if not between:
            continue
        bottom = min(between, key=lambda v: smoothed[v])

        depth = smoothed[start] - smoothed[bottom]
        if depth < min_rep_depth_deg:
            continue  # too shallow to count as a real rep

        reps.append(Rep(
            rep_number=rep_num,
            start_frame=frame_indices[start],
            bottom_frame=frame_indices[bottom],
            end_frame=frame_indices[end],
            min_knee_angle=float(smoothed[bottom]),
            max_knee_angle=float(smoothed[start]),
        ))
        rep_num += 1

    return reps


def extract_rep_segment(knee_angles: List[float], rep: Rep, frame_indices: List[int]) -> np.ndarray:
    """Return the knee-angle sub-sequence for a single rep (start_frame..end_frame)."""
    start_i = frame_indices.index(rep.start_frame)
    end_i = frame_indices.index(rep.end_frame)
    return np.array(knee_angles[start_i:end_i + 1], dtype=float)
