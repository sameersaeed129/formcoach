"""
reference_rep.py
--------------------
Provides a canonical "ideal" squat rep knee-angle curve to compare user
reps against via DTW. Built from the same biomechanical parameters used
to label synthetic training data as good_form (standing ~172deg, depth
~88deg, smooth bell-curve descent/ascent), not from an arbitrary shape.
"""

import numpy as np


def ideal_squat_curve(length: int = 50, standing_angle: float = 172.0, depth_angle: float = 88.0) -> np.ndarray:
    t = np.linspace(0, 1, length)
    bump = np.exp(-((t - 0.5) ** 2) / (2 * 0.18 ** 2))
    return standing_angle - bump * (standing_angle - depth_angle)
