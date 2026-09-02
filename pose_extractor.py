"""
pose_extractor.py
--------------------
Wraps MediaPipe Pose to extract 33 body landmarks per video frame.
Landmarks are normalized image coordinates (x, y in [0, 1]) plus a
visibility score MediaPipe assigns to each point.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# The subset of MediaPipe's 33 landmarks relevant to lower-body lifting
# exercises (squat, deadlift, lunge). Keys map to mp_pose.PoseLandmark names.
KEY_LANDMARKS = [
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
    "NOSE",
]


@dataclass
class FramePose:
    frame_index: int
    landmarks: Dict[str, np.ndarray]  # name -> [x, y, visibility]
    detected: bool
    raw_landmarks: Optional[object] = None  # MediaPipe's native landmark list, for drawing


class PoseExtractor:
    """Runs MediaPipe Pose on a video and yields per-frame landmark data."""

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame_bgr: np.ndarray, frame_index: int) -> FramePose:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if not results.pose_landmarks:
            return FramePose(frame_index=frame_index, landmarks={}, detected=False, raw_landmarks=None)

        landmarks = {}
        for name in KEY_LANDMARKS:
            idx = getattr(mp_pose.PoseLandmark, name).value
            lm = results.pose_landmarks.landmark[idx]
            landmarks[name] = np.array([lm.x, lm.y, lm.visibility])

        return FramePose(frame_index=frame_index, landmarks=landmarks, detected=True,
                          raw_landmarks=results.pose_landmarks)

    def process_video(self, video_path: str, max_frames: Optional[int] = None) -> List[FramePose]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        poses = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break
            poses.append(self.process_frame(frame, frame_idx))
            frame_idx += 1

        cap.release()
        return poses

    def draw_landmarks(self, frame_bgr: np.ndarray, frame_pose: FramePose) -> np.ndarray:
        """Draw the full MediaPipe skeleton overlay on a frame (for display)."""
        annotated = frame_bgr.copy()
        if frame_pose is not None and frame_pose.raw_landmarks is not None:
            mp_drawing.draw_landmarks(
                annotated, frame_pose.raw_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        return annotated

    def close(self):
        self.pose.close()
