"""
demo_cli.py
--------------
Command-line entry point that runs the full FormCoach pipeline on a video
file and prints the results -- useful for quickly verifying your install
works, or for batch-processing without the Streamlit UI.

Usage:
    python demo_cli.py path/to/your_squat_video.mp4
    python demo_cli.py path/to/video.mp4 --min-rep-depth 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run FormCoach on a video file from the command line.")
    parser.add_argument("video_path", help="Path to a squat (or similar lower-body exercise) video")
    parser.add_argument("--checkpoint", default="models/form_classifier.pt",
                         help="Path to the trained classifier checkpoint")
    parser.add_argument("--min-rep-depth", type=float, default=15.0,
                         help="Minimum knee-angle change (degrees) to count as a rep")
    parser.add_argument("--use-llm", action="store_true",
                         help="Use a local Ollama model for feedback text (falls back to templates if unavailable)")
    args = parser.parse_args()

    print(f"Processing {args.video_path} ...")
    result = run_pipeline(
        args.video_path, args.checkpoint,
        use_llm_feedback=args.use_llm,
    )

    detected_frac = sum(1 for m in result.metrics if m.valid) / max(len(result.metrics), 1)
    print(f"\nProcessed {len(result.frame_poses)} frames ({detected_frac:.0%} had a pose detected)")
    print(f"Detected {len(result.reps)} rep(s)\n")

    if not result.reps:
        print("No full reps detected. Try --min-rep-depth with a lower value, "
              "or make sure the exercise's full range of motion is visible in the video.")
        return

    for analysis in result.analyses:
        print(f"--- Rep {analysis.rep.rep_number} ---")
        print(f"  Classification : {analysis.predicted_label} ({analysis.confidence:.0%} confidence)")
        print(f"  Similarity     : {analysis.similarity:.0f}/100")
        print(f"  Min knee angle : {analysis.rep.min_knee_angle:.0f}\u00b0")
        print(f"  Feedback       : {analysis.feedback.text}")
        print()


if __name__ == "__main__":
    main()
