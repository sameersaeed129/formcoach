"""
app.py
--------
FormCoach — Streamlit interface. Upload a workout video, watch the
skeleton overlay, and get rep-by-rep form analysis with coaching feedback.

Run with:
    streamlit run app.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
import numpy as np
import cv2
import pandas as pd

from pose_extractor import PoseExtractor
from biomechanics import compute_frame_metrics
from rep_counter import segment_reps
from form_classifier import FormClassifier
from movement_comparison import compare_to_reference, similarity_score
from coach_feedback import generate_feedback
from reference_rep import ideal_squat_curve

st.set_page_config(page_title="FormCoach", page_icon="🏋️", layout="wide")

DEFAULT_CHECKPOINT = str(Path(__file__).parent / "models" / "form_classifier.pt")


@st.cache_resource
def load_classifier(checkpoint_path: str):
    return FormClassifier(checkpoint_path)


def process_video(video_path: str, min_rep_depth: float, sample_every: int):
    extractor = PoseExtractor()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_poses = []
    annotated_frames = []
    progress = st.progress(0, text="Analyzing video frame by frame...")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fp = extractor.process_frame(frame, idx)
        frame_poses.append(fp)

        if idx % sample_every == 0:
            annotated = extractor.draw_landmarks(frame, fp)
            annotated_frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

        idx += 1
        if total_frames > 0:
            progress.progress(min(idx / total_frames, 1.0), text=f"Processing frame {idx}/{total_frames}")

    cap.release()
    extractor.close()
    progress.empty()

    metrics = [compute_frame_metrics(fp) for fp in frame_poses]
    return frame_poses, metrics, annotated_frames, fps


def main():
    st.title("🏋️ FormCoach")
    st.caption("AI movement-quality coach — pose estimation, biomechanics, and rep-by-rep feedback")

    with st.sidebar:
        st.header("Settings")
        min_rep_depth = st.slider("Minimum rep depth to count (degrees)", 5, 40, 15,
                                   help="Lower this if your test clip has a shallow range of motion.")
        sample_every = st.slider("Preview frame sampling", 1, 10, 3,
                                  help="Show every Nth frame in the skeleton preview (higher = faster).")
        use_llm = st.checkbox("Use local LLM for feedback (requires Ollama running)", value=False,
                               help="Falls back to built-in coaching templates automatically if Ollama isn't available.")
        st.divider()
        st.markdown(
            "**Tip:** works best with a side-on view of a squat, lunge, or similar "
            "lower-body exercise, with your full body in frame."
        )

    uploaded_file = st.file_uploader("Upload a workout video", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is None:
        st.info("Upload a video of a squat (or similar lower-body exercise) to get started.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    classifier = load_classifier(DEFAULT_CHECKPOINT)

    with st.spinner("Running pose estimation..."):
        frame_poses, metrics, annotated_frames, fps = process_video(video_path, min_rep_depth, sample_every)

    detected_frac = sum(1 for m in metrics if m.valid) / max(len(metrics), 1)
    if detected_frac < 0.5:
        st.warning(
            f"Only {detected_frac:.0%} of frames had a clear pose detected. "
            "Make sure your full body is visible and well-lit for best results."
        )

    knee_angles = [m.knee_angle for m in metrics]
    hip_angles = [m.hip_angle for m in metrics]
    torso_leans = [m.torso_lean for m in metrics]
    knee_valgus = [m.knee_valgus for m in metrics]
    frame_indices = [m.frame_index for m in metrics]

    reps = segment_reps(knee_angles, frame_indices, min_rep_depth_deg=min_rep_depth)

    st.subheader(f"Detected {len(reps)} rep(s)")

    if annotated_frames:
        st.subheader("Skeleton tracking preview")
        preview_idx = st.slider("Preview frame", 0, len(annotated_frames) - 1, 0)
        st.image(annotated_frames[preview_idx], use_container_width=True)

    st.subheader("Knee angle over time")
    angle_df = pd.DataFrame({"frame": frame_indices, "knee_angle": knee_angles})
    st.line_chart(angle_df.set_index("frame"))

    if not reps:
        st.info("No full reps detected. Try lowering the 'minimum rep depth' setting in the sidebar, "
                 "or make sure the exercise's full range of motion is visible in the video.")
        return

    reference_curve = ideal_squat_curve()

    st.subheader("Rep-by-rep analysis")
    summary_rows = []
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

        dtw_result = compare_to_reference(seq[:, 0], reference_curve)
        sim = similarity_score(dtw_result)

        peak_torso_lean = float(np.nanmax(torso_leans[start_i:end_i + 1]))

        feedback = generate_feedback(
            rep_number=rep.rep_number, label=label, confidence=confidence,
            similarity=sim, min_knee=rep.min_knee_angle, torso_lean=peak_torso_lean,
            use_llm=use_llm,
        )

        icon = "✅" if label == "good_form" else "⚠️"
        with st.expander(f"{icon} Rep {rep.rep_number} — {label.replace('_', ' ').title()} "
                          f"(similarity {sim:.0f}/100)"):
            st.write(feedback.text)
            col1, col2, col3 = st.columns(3)
            col1.metric("Min knee angle", f"{rep.min_knee_angle:.0f}\u00b0")
            col2.metric("Model confidence", f"{confidence:.0%}")
            col3.metric("Similarity score", f"{sim:.0f}/100")

        summary_rows.append({
            "Rep": rep.rep_number,
            "Classification": label.replace("_", " ").title(),
            "Confidence": f"{confidence:.0%}",
            "Similarity": f"{sim:.0f}/100",
            "Min knee angle": f"{rep.min_knee_angle:.0f}\u00b0",
        })

    st.subheader("Summary table")
    st.table(pd.DataFrame(summary_rows))


if __name__ == "__main__":
    main()
