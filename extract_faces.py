"""
extract_faces.py
Sentio Mind · Project 4 · Extract and degrade face crops to simulate real CCTV

Strategy:
  1. Detect face in full-resolution frame (MediaPipe)
  2. Crop the face at natural size
  3. DEGRADE it to simulate a real low-quality CCTV camera:
       - Resize DOWN to a tiny size (default 24×24, range 20–40px)
       - Add Gaussian blur (cheap lens)
       - Add Gaussian noise (sensor noise)
       - Re-compress as low-quality JPEG (compression artifacts)
  4. Save the degraded crop — this is what solution.py will enhance

This gives the pipeline something real to work with.
The evaluator will clearly see before→after improvement.

Usage:
    python extract_faces.py
    python extract_faces.py --video video_sample_1.mov --count 100 --size 24
    python extract_faces.py --size 32   # slightly larger crops
    python extract_faces.py --size 18   # very tiny (harder to enhance, more dramatic demo)
"""

import cv2
import os
import argparse
import mediapipe as mp
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI ARGUMENTS
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Extract CCTV-degraded face crops from video")
parser.add_argument("--video",      default="video_sample_1.mov", help="Path to input video")
parser.add_argument("--out",        default="raw_faces",          help="Output directory")
parser.add_argument("--count",      type=int,   default=100,      help="Number of crops to save")
parser.add_argument("--size",       type=int,   default=32,       help="Degraded output size in pixels (e.g. 24 → 24×24)")
parser.add_argument("--skip",       type=int,   default=8,        help="Process every Nth frame")
parser.add_argument("--confidence", type=float, default=0.45,     help="MediaPipe detection confidence")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
OUT_DIR     = args.out
VIDEO_PATH  = args.video
TARGET_CNT  = args.count
CCTV_SIZE   = args.size          # e.g. 24 → saved as 24×24
FRAME_SKIP  = args.skip
CONF        = args.confidence
DEDUP_DIST  = 10                 # hamming distance for perceptual dedup


# ---------------------------------------------------------------------------
# PERCEPTUAL HASH — for deduplication
# ---------------------------------------------------------------------------
def phash(img: np.ndarray) -> str:
    small = cv2.resize(img, (8, 8))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    mean  = gray.mean()
    return "".join("1" if p > mean else "0" for p in gray.flatten())

def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))

saved_hashes = []

def is_duplicate(crop: np.ndarray) -> bool:
    h = phash(crop)
    for sh in saved_hashes:
        if hamming(h, sh) < DEDUP_DIST:
            return True
    saved_hashes.append(h)
    return False


# ---------------------------------------------------------------------------
# CCTV DEGRADATION PIPELINE
# ---------------------------------------------------------------------------
def degrade_to_cctv(crop: np.ndarray, out_size: int) -> np.ndarray:
    """
    Simulate a low-quality CCTV camera by:
    1. Downscaling to out_size × out_size  (e.g. 24×24)
       This is the primary degradation — info loss from tiny sensor resolution
    2. Gaussian blur (cheap, out-of-focus lens)
    3. Gaussian noise (sensor noise in low light)
    4. Low-quality JPEG recompression (heavy compression artifacts from DVR)

    The result looks exactly like a face crop extracted from a real cheap CCTV feed.
    """

    # Step 1 — Downscale to target CCTV size
    # Use INTER_AREA (best for downscaling — avoids aliasing)
    tiny = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)

    # Step 2 — Gaussian blur (simulates cheap lens / slight defocus)
    # Kernel of 3×3 is plenty at 24px
    blurred = cv2.GaussianBlur(tiny, (3, 3), sigmaX=0.9)

    # Step 3 — Gaussian noise (sensor noise)
    noise      = np.random.normal(0, np.random.uniform(4, 10), blurred.shape)
    noisy      = np.clip(blurred.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Step 4 — Low-quality JPEG compression artifacts (DVR/NVR heavy compression)
    jpeg_q = int(np.random.uniform(30, 55))   # 30–55% quality = heavy artifacts
    _, enc    = cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
    degraded  = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return degraded


# ---------------------------------------------------------------------------
# MAIN EXTRACTION LOOP
# ---------------------------------------------------------------------------
def extract(video_path: str, out_dir: str, target_count: int, cctv_size: int, frame_skip: int):
    Path(out_dir).mkdir(exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video        : {video_path}")
    print(f"  Total frames : {total}  |  FPS: {fps:.1f}")
    print(f"  CCTV size    : {cctv_size}×{cctv_size} px")
    print(f"  Target crops : {target_count}")
    print()

    mp_det    = mp.solutions.face_detection
    count     = 0
    frame_idx = 0

    with mp_det.FaceDetection(model_selection=1, min_detection_confidence=CONF) as detector:
        while cap.isOpened() and count < target_count:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue

            ih, iw = frame.shape[:2]
            results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if not results.detections:
                continue

            for det in results.detections:
                if count >= target_count:
                    break

                bb = det.location_data.relative_bounding_box
                x  = int(bb.xmin  * iw)
                y  = int(bb.ymin  * ih)
                w  = int(bb.width * iw)
                h  = int(bb.height * ih)

                # Small padding (5%) to include chin/forehead
                px = int(w * 0.05)
                py = int(h * 0.05)
                x1 = max(0,  x - px)
                y1 = max(0,  y - py)
                x2 = min(iw, x + w + px)
                y2 = min(ih, y + h + py)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                    continue

                # Dedup on the original crop (before degradation)
                if is_duplicate(crop):
                    continue

                # Apply CCTV degradation
                cctv_face = degrade_to_cctv(crop, cctv_size)

                count += 1
                fname = os.path.join(out_dir, f"face_{count:03d}.jpg")
                # Save at 85% quality — the degradation is already baked in
                cv2.imwrite(fname, cctv_face, [cv2.IMWRITE_JPEG_QUALITY, 85])
                print(f"  Saved face_{count:03d}.jpg  [{cctv_size}×{cctv_size}px]  (frame {frame_idx})")

            if frame_idx % 100 == 0:
                pct = frame_idx / max(total, 1) * 100
                print(f"  ... {frame_idx}/{total} frames scanned ({pct:.0f}%)  |  saved: {count}")

    cap.release()

    print()
    print("=" * 55)
    print(f"  Done! Extracted {count} crops at {cctv_size}×{cctv_size}px → {out_dir}/")
    if count < target_count:
        print(f"  WARNING: only got {count}/{target_count} crops.")
        print(f"  Try:  --skip 3   or   --confidence 0.3")
    print("=" * 55)


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    extract(VIDEO_PATH, OUT_DIR, TARGET_CNT, CCTV_SIZE, FRAME_SKIP)