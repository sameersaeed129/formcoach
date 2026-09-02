"""
biomechanics.py
------------------
Converts raw pose landmarks into real biomechanical measurements: joint
angles (knee flexion, hip hinge, torso lean) and knee-valgus (knee
cave-in) deviation, using standard vector-angle geometry.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from pose_extractor import FramePose


def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b formed by points a-b-c, in degrees.
    Uses only the x,y coordinates (2D projection is standard for
    single-camera video-based biomechanics)."""
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return float("nan")
    cosine = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


@dataclass
class FrameMetrics:
    frame_index: int
    knee_angle: float          # degrees; ~180 = straight leg, smaller = deeper bend
    hip_angle: float            # degrees; torso-hip-knee angle, tracks hip hinge
    torso_lean: float            # degrees from vertical; 0 = upright
    knee_valgus: float            # normalized horizontal knee-vs-ankle-vs-hip deviation
    valid: bool


def compute_frame_metrics(frame_pose: FramePose, visibility_threshold: float = 0.4) -> FrameMetrics:
    """Compute joint angles for a single frame. Averages left/right side
    when both are visible; falls back to whichever side is visible."""
    if not frame_pose.detected:
        return FrameMetrics(frame_pose.frame_index, float("nan"), float("nan"),
                             float("nan"), float("nan"), valid=False)

    lm = frame_pose.landmarks

    def visible(name):
        return lm[name][2] >= visibility_threshold

    knee_angles = []
    hip_angles = []
    valgus_scores = []

    for side in ["LEFT", "RIGHT"]:
        hip = f"{side}_HIP"
        knee = f"{side}_KNEE"
        ankle = f"{side}_ANKLE"
        shoulder = f"{side}_SHOULDER"

        if visible(hip) and visible(knee) and visible(ankle):
            knee_angles.append(_angle_between(lm[hip], lm[knee], lm[ankle]))

        if visible(shoulder) and visible(hip) and visible(knee):
            hip_angles.append(_angle_between(lm[shoulder], lm[hip], lm[knee]))

        # Knee valgus proxy: horizontal distance of knee from the hip-ankle
        # line, normalized by leg length. A healthy squat keeps the knee
        # roughly in line with the foot; a positive score means the knee is
        # caving inward relative to that line.
        if visible(hip) and visible(knee) and visible(ankle):
            hip_xy, knee_xy, ankle_xy = lm[hip][:2], lm[knee][:2], lm[ankle][:2]
            leg_vec = ankle_xy - hip_xy
            leg_len = np.linalg.norm(leg_vec)
            if leg_len > 1e-6:
                # Perpendicular distance of the knee point from the hip-ankle line
                t = np.dot(knee_xy - hip_xy, leg_vec) / (leg_len ** 2)
                projected = hip_xy + t * leg_vec
                deviation = (knee_xy - projected)[0]  # signed horizontal offset
                valgus_scores.append(float(deviation / leg_len))

    torso_lean = float("nan")
    if visible("LEFT_SHOULDER") and visible("LEFT_HIP"):
        shoulder_xy = lm["LEFT_SHOULDER"][:2]
        hip_xy = lm["LEFT_HIP"][:2]
        torso_vec = shoulder_xy - hip_xy
        vertical = np.array([0.0, -1.0])  # up, in image coords y grows downward
        norm_t = np.linalg.norm(torso_vec)
        if norm_t > 1e-6:
            cosine = np.clip(np.dot(torso_vec, vertical) / norm_t, -1.0, 1.0)
            torso_lean = float(np.degrees(np.arccos(cosine)))

    if not knee_angles:
        return FrameMetrics(frame_pose.frame_index, float("nan"), float("nan"),
                             torso_lean, float("nan"), valid=False)

    return FrameMetrics(
        frame_index=frame_pose.frame_index,
        knee_angle=float(np.mean(knee_angles)),
        hip_angle=float(np.mean(hip_angles)) if hip_angles else float("nan"),
        torso_lean=torso_lean,
        knee_valgus=float(np.mean(valgus_scores)) if valgus_scores else float("nan"),
        valid=True,
    )
