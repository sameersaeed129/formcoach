"""
synthetic_data.py
---------------------
Generates biomechanically-plausible synthetic training data for the form
classifier. Real labeled "bad squat form" video datasets are not publicly
available at scale, so this generates parametric joint-angle curves for
each form class based on known biomechanics of each error pattern, with
randomized noise/timing/amplitude so the classifier doesn't just memorize
one fixed curve shape. This is a standard technique (synthetic/simulated
training data) used when real labeled data is scarce or expensive to
collect -- see README for a full discussion of this design choice.

Each sample is a (T, 4) time series: [knee_angle, hip_angle, torso_lean, knee_valgus]
resampled to a fixed length T, representing one repetition.
"""

import numpy as np

CLASSES = ["good_form", "knee_valgus", "insufficient_depth", "back_rounding"]
SEQ_LEN = 50  # resampled length per rep


def _bell_curve(t, center, width):
    """A smooth 0->1->0 bump used to shape a rep's descent/ascent."""
    return np.exp(-((t - center) ** 2) / (2 * width ** 2))


def _base_rep(t, depth_angle=90.0, standing_angle=172.0, noise=2.0, rng=None):
    """Baseline knee-angle curve for a normal rep: standing -> bottom -> standing."""
    rng = rng or np.random.default_rng()
    bump = _bell_curve(t, center=0.5, width=0.18)
    knee_angle = standing_angle - bump * (standing_angle - depth_angle)
    knee_angle += rng.normal(0, noise, size=t.shape)
    return knee_angle


def generate_sample(label: str, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0, 1, SEQ_LEN)

    # Randomize normal rep parameters slightly for variety
    standing_angle = rng.uniform(165, 178)
    tempo_jitter = rng.uniform(0.85, 1.15)
    t_shifted = np.clip(t * tempo_jitter, 0, 1)

    if label == "good_form":
        depth = rng.uniform(80, 95)
        knee_angle = _base_rep(t_shifted, depth_angle=depth, standing_angle=standing_angle,
                                noise=1.5, rng=rng)
        hip_angle = standing_angle - 0.9 * (standing_angle - knee_angle) + rng.normal(0, 2, SEQ_LEN)
        torso_lean = 15 * _bell_curve(t_shifted, 0.5, 0.22) + rng.normal(0, 1.5, SEQ_LEN)
        knee_valgus = rng.normal(0, 0.02, SEQ_LEN)

    elif label == "knee_valgus":
        depth = rng.uniform(80, 95)
        knee_angle = _base_rep(t_shifted, depth_angle=depth, standing_angle=standing_angle,
                                noise=1.5, rng=rng)
        hip_angle = standing_angle - 0.9 * (standing_angle - knee_angle) + rng.normal(0, 2, SEQ_LEN)
        torso_lean = 15 * _bell_curve(t_shifted, 0.5, 0.22) + rng.normal(0, 1.5, SEQ_LEN)
        # Pronounced inward knee deviation concentrated near the bottom of the rep
        valgus_magnitude = rng.uniform(0.12, 0.25)
        knee_valgus = valgus_magnitude * _bell_curve(t_shifted, 0.5, 0.15) + rng.normal(0, 0.02, SEQ_LEN)

    elif label == "insufficient_depth":
        # Never reaches proper depth -- knee angle bottoms out much higher
        depth = rng.uniform(125, 150)
        knee_angle = _base_rep(t_shifted, depth_angle=depth, standing_angle=standing_angle,
                                noise=1.5, rng=rng)
        hip_angle = standing_angle - 0.9 * (standing_angle - knee_angle) + rng.normal(0, 2, SEQ_LEN)
        torso_lean = 10 * _bell_curve(t_shifted, 0.5, 0.22) + rng.normal(0, 1.5, SEQ_LEN)
        knee_valgus = rng.normal(0, 0.02, SEQ_LEN)

    elif label == "back_rounding":
        depth = rng.uniform(80, 95)
        knee_angle = _base_rep(t_shifted, depth_angle=depth, standing_angle=standing_angle,
                                noise=1.5, rng=rng)
        hip_angle = standing_angle - 0.9 * (standing_angle - knee_angle) + rng.normal(0, 2, SEQ_LEN)
        # Excessive, sustained forward torso lean well beyond normal squat lean
        lean_magnitude = rng.uniform(35, 55)
        torso_lean = lean_magnitude * _bell_curve(t_shifted, 0.55, 0.28) + rng.normal(0, 2, SEQ_LEN)
        knee_valgus = rng.normal(0, 0.02, SEQ_LEN)

    else:
        raise ValueError(f"Unknown label: {label}")

    return np.stack([knee_angle, hip_angle, torso_lean, knee_valgus], axis=1)  # (SEQ_LEN, 4)


def generate_dataset(n_per_class: int = 300, seed: int = 42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for class_idx, label in enumerate(CLASSES):
        for _ in range(n_per_class):
            X.append(generate_sample(label, rng))
            y.append(class_idx)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y
