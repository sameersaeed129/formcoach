"""
make_test_video.py — NOT part of the shipped app.
Generates a synthetic test video by periodically warping a real photo of a
standing person to simulate squat-like vertical motion, purely so the
video-processing pipeline (pose -> angles -> rep counting -> classifier)
can be integration-tested without needing a real recorded workout video.
"""
import cv2
import numpy as np

SRC_IMAGE = "/tmp/person2.jpg"
OUT_VIDEO = "/tmp/synthetic_squat.mp4"
N_FRAMES = 150  # ~5 seconds at 30fps, roughly 3 squat cycles
FPS = 30


def warp_squat(img, phase):
    """Simulate a squat by vertically compressing the lower 60% of the frame
    (legs/hips) proportional to sin(phase), and shifting it down slightly,
    which changes the apparent knee bend when MediaPipe re-detects pose."""
    h, w = img.shape[:2]
    compression = 0.55 * max(0, np.sin(phase))  # deeper compression for a more realistic angle swing

    split_y = int(h * 0.40)  # roughly hip height
    upper = img[:split_y]
    lower = img[split_y:]

    new_lower_h = int(lower.shape[0] * (1 - compression))
    lower_resized = cv2.resize(lower, (w, new_lower_h))

    canvas = np.zeros_like(img)
    canvas[:split_y] = upper
    y_offset = split_y + int((lower.shape[0] - new_lower_h))
    end_y = min(h, y_offset + new_lower_h)
    canvas[y_offset:end_y] = lower_resized[: end_y - y_offset]

    return canvas


def main():
    img = cv2.imread(SRC_IMAGE)
    h, w = img.shape[:2]
    writer = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))

    for i in range(N_FRAMES):
        phase = 2 * np.pi * (i / N_FRAMES) * 3  # 3 full cycles across the clip
        frame = warp_squat(img, phase)
        writer.write(frame)

    writer.release()
    print(f"Wrote {N_FRAMES} frames to {OUT_VIDEO}")


if __name__ == "__main__":
    main()
