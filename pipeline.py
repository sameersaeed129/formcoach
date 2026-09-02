"""
pipeline.py
--------------
Orchestrates the full FormCoach pipeline: video -> pose -> joint angles ->
rep segmentation -> per-rep form classification -> DTW similarity scoring
-> coaching feedback. Used by both app.py (Streamlit) and any CLI/test
scripts, so the logic lives in exactly one place.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np

from pose_extractor import PoseExtractor, FramePose
from biomechanics import compute_frame_metrics, FrameMetrics
from rep_counter import segment_reps, Rep
from form_classifier import FormClassifier
from movement_comparison import compare_to_reference, similarity_score
from coach_feedback import generate_feedback, RepFeedback
from reference_rep import ideal_squat_curve


@dataclass
class RepAnalysis:
    rep: Rep
    predicted_label: str
    confidence: float
    similarity: float
    feedback: RepFeedback


@dataclass
class PipelineResult:
    frame_poses: List[FramePose]
    metrics: List[FrameMetrics]
    reps: List[Rep]
    analyses: List[RepAnalysis] = field(default_factory=list)


def run_pipeline(video_path: str, classifier_checkpoint: str, use_llm_feedback: bool = False,
                  max_frames: int = None, min_rep_depth_deg: float = 15.0) -> PipelineResult:
    extractor = PoseExtractor()
    try:
        frame_poses = extractor.process_video(video_path, max_frames=max_frames)
    finally:
        extractor.close()

    metrics = [compute_frame_metrics(fp) for fp in frame_poses]

    knee_angles = [m.knee_angle for m in metrics]
    hip_angles = [m.hip_angle for m in metrics]
    torso_leans = [m.torso_lean for m in metrics]
    knee_valgus = [m.knee_valgus for m in metrics]
    frame_indices = [m.frame_index for m in metrics]

    reps = segment_reps(knee_angles, frame_indices, min_rep_depth_deg=min_rep_depth_deg)

    classifier = FormClassifier(classifier_checkpoint)
    reference_curve = ideal_squat_curve()

    analyses = []
    for rep in reps:
        start_i = frame_indices.index(rep.start_frame)
        end_i = frame_indices.index(rep.end_frame)

        seq = np.stack([
            knee_angles[start_i:end_i + 1],
            hip_angles[start_i:end_i + 1],
            torso_leans[start_i:end_i + 1],
            knee_valgus[start_i:end_i + 1],
        ], axis=1)
        seq = np.nan_to_num(seq, nan=0.0)

        label, probs = classifier.predict(seq)
        confidence = float(np.max(probs))

        knee_seq = seq[:, 0]
        dtw_result = compare_to_reference(knee_seq, reference_curve)
        sim = similarity_score(dtw_result)

        peak_torso_lean = float(np.nanmax(torso_leans[start_i:end_i + 1]))

        feedback = generate_feedback(
            rep_number=rep.rep_number, label=label, confidence=confidence,
            similarity=sim, min_knee=rep.min_knee_angle, torso_lean=peak_torso_lean,
            use_llm=use_llm_feedback,
        )

        analyses.append(RepAnalysis(
            rep=rep, predicted_label=label, confidence=confidence,
            similarity=sim, feedback=feedback,
        ))

    return PipelineResult(frame_poses=frame_poses, metrics=metrics, reps=reps, analyses=analyses)
