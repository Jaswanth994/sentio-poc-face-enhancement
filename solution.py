"""
face_enhancement.py
Sentio Mind · Project 4 · Low-Resolution CCTV Face Enhancement

Copy this file to solution.py and fill in every TODO block.
Do not rename any function.
Run: python solution.py
Output goes into enhanced_faces/ (created automatically).
"""

import cv2
import json
import base64
import time
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RAW_FACES_DIR    = Path("raw_faces")
REFERENCE_DIR    = Path("reference_identities")
ENHANCED_DIR     = Path("enhanced_faces")
REPORT_HTML_OUT  = Path("enhancement_report.html")
METRICS_JSON_OUT = Path("evaluation_metrics.json")

TARGET_SIZE      = (240, 240)
ENHANCED_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# STAGE 1 — DENOISE
# ---------------------------------------------------------------------------

def stage1_denoise(img: np.ndarray) -> np.ndarray:
    """cv2.fastNlMeansDenoisingColored with h=8, hColor=8, templateWindowSize=7, searchWindowSize=21"""
    return cv2.fastNlMeansDenoisingColored(img, None, h=8, hColor=8, templateWindowSize=7, searchWindowSize=21)

def stage2_clahe(img: np.ndarray) -> np.ndarray:
    """Convert to LAB. Apply CLAHE (clipLimit=3.5, tileGridSize=(4,4)) to L channel. Merge + convert back."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
    l_clahe = clahe.apply(l)
    
    merged = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def unsharp_mask(img: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    """blurred = GaussianBlur(img, sigma); result = img + strength * (img - blurred)"""
    # cv2.GaussianBlur with kernel (0,0) computes kernel size automatically based on sigma
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    # mathematically equivalent to: img*(1+strength) - blurred*strength
    result = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return np.clip(result, 0, 255).astype(np.uint8)
# ==========================================
# GLOBAL OPTIMIZATION FOR 30-SECOND RULE
# ==========================================
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
# Initialize once globally to save massive CPU overhead
global_face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, 
    max_num_faces=1, 
    refine_landmarks=False, 
    min_detection_confidence=0.5
)

# ---------------------------------------------------------------------------
# STAGE 3 — UPSAMPLE
# ---------------------------------------------------------------------------
def stage3_upscale(img: np.ndarray) -> np.ndarray:
    """If short side < 64px: 2x LANCZOS4 -> unsharp(1.0, 0.8) -> 2x LANCZOS4 -> resize."""
    h, w = img.shape[:2]
    short_side = min(h, w)
    
    if short_side < 64:
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        # REDUCED STRENGTH: 1.6 -> 0.8 to prevent artifacting
        img = unsharp_mask(img, sigma=1.0, strength=0.8)
        h, w = img.shape[:2] 
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
        
    return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)


# ---------------------------------------------------------------------------
# STAGE 4 — ZONE SHARPEN
# ---------------------------------------------------------------------------
def stage4_zone_sharpen(img: np.ndarray) -> np.ndarray:
    """
    MediaPipe Face Mesh -> locate eye + nose region -> create mask.
    Apply softer unsharp masks to preserve dlib facial encodings.
    """
    # REDUCED FALLBACK STRENGTH: 1.5 -> 0.8
    fallback_img = unsharp_mask(img, sigma=1.0, strength=0.8)
    
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Using the globally loaded model
    results = global_face_mesh.process(rgb)
    
    if not results.multi_face_landmarks:
        return fallback_img
        
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    
    t_zone_indices = [
        33, 160, 158, 133, 153, 144, 
        362, 385, 387, 263, 373, 380, 
        168, 6, 197, 195, 5, 4, 1, 2, 98, 327 
    ]
    
    landmarks = results.multi_face_landmarks[0].landmark
    points = []
    for idx in t_zone_indices:
        lm = landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        points.append([x, y])
        
    points = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 1.0)
    
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    mask = np.expand_dims(mask, axis=-1)
    
    # REDUCED T-ZONE STRENGTH: 2.0 -> 1.2
    img_zone = unsharp_mask(img, sigma=0.8, strength=1.2).astype(np.float32)
    # REDUCED REST OF FACE STRENGTH: 1.3 -> 0.5
    img_rest = unsharp_mask(img, sigma=1.2, strength=0.5).astype(np.float32)
    
    blended = (img_zone * mask) + (img_rest * (1.0 - mask))
    
    return np.clip(blended, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# FULL PIPELINE — do not change this function
# ---------------------------------------------------------------------------

def enhance_face(img: np.ndarray) -> np.ndarray:
    """Run all 4 stages in order. Do not modify."""
    img = stage1_denoise(img)
    img = stage2_clahe(img)
    img = stage3_upscale(img)
    img = stage4_zone_sharpen(img)
    return img


# ---------------------------------------------------------------------------
# EVALUATION HELPERS
# ---------------------------------------------------------------------------

def sharpness(img: np.ndarray) -> float:
    """Laplacian variance. Higher = sharper. Convert to grayscale first."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def get_face_encoding(img: np.ndarray):
    """128-d face encoding. Return numpy array if face found, else None."""
    import face_recognition as fr
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Using upsample=2 as strictly requested for small faces
    boxes = fr.face_locations(rgb_img, number_of_times_to_upsample=2)
    if not boxes:
        return None
    encodings = fr.face_encodings(rgb_img, known_face_locations=boxes)
    return encodings[0] if encodings else None

def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """Structural Similarity Index between two images."""
    from skimage.metrics import structural_similarity
    
    # Ensure both are resized to TARGET_SIZE before comparison
    if a.shape[:2] != TARGET_SIZE:
        a = cv2.resize(a, TARGET_SIZE)
    if b.shape[:2] != TARGET_SIZE:
        b = cv2.resize(b, TARGET_SIZE)
        
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    
    score, _ = structural_similarity(a_gray, b_gray, full=True)
    return float(score)


# ---------------------------------------------------------------------------
# HTML A/B REPORT
# ---------------------------------------------------------------------------

def generate_ab_report(results: list, output_path: Path):
    """
    Self-contained HTML. No CDN.
    Summary header: overall accuracy improvement + sharpness gain.
    Grid: each row = original image | enhanced image | sharpness before/after | match before/after.
    Images embedded as base64.
    """
    n = len(results)
    if n == 0:
        return
        
    # Calculate summary metrics
    acc_before = sum(1 for r in results if r["match_before"]) / n * 100
    acc_after = sum(1 for r in results if r["match_after"]) / n * 100
    avg_sharp_b = sum(r["sharpness_before"] for r in results) / n
    avg_sharp_a = sum(r["sharpness_after"] for r in results) / n

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Sentio Mind Enhancement Report</title>
        <style>
            body {{ font-family: system-ui, -apple-system, sans-serif; background: #f9fafb; margin: 2rem; color: #111827; }}
            .header {{ background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 2rem; }}
            h1 {{ margin-top: 0; color: #0f172a; }}
            .metrics {{ display: flex; gap: 2rem; font-size: 1.1rem; }}
            .metrics strong {{ color: #3b82f6; }}
            table {{ width: 100%; border-collapse: collapse; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }}
            th, td {{ padding: 1rem; text-align: left; border-bottom: 1px solid #e5e7eb; vertical-align: middle; }}
            th {{ background: #f3f4f6; font-weight: 600; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.05em; }}
            img {{ width: 120px; height: 120px; border-radius: 6px; object-fit: cover; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }}
            .true {{ color: #16a34a; font-weight: 600; }}
            .false {{ color: #dc2626; font-weight: 600; }}
            .code {{ font-family: monospace; background: #f1f5f9; padding: 0.2rem 0.4rem; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Enhancement A/B Report</h1>
            <div class="metrics">
                <div>Total Faces Processed: <strong>{n}</strong></div>
                <div>Accuracy: <strong>{acc_before:.1f}% &rarr; {acc_after:.1f}%</strong></div>
                <div>Avg Sharpness: <strong>{avg_sharp_b:.1f} &rarr; {avg_sharp_a:.1f}</strong></div>
            </div>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Filename</th>
                    <th>Original (Resized)</th>
                    <th>Enhanced</th>
                    <th>Sharpness</th>
                    <th>Identity Match</th>
                </tr>
            </thead>
            <tbody>
    """

    for r in results:
        m_b_class = "true" if r['match_before'] else "false"
        m_a_class = "true" if r['match_after'] else "false"
        html += f"""
                <tr>
                    <td><span class="code">{r['filename']}</span></td>
                    <td><img src="data:image/jpeg;base64,{r['raw_b64']}" alt="Raw"></td>
                    <td><img src="data:image/jpeg;base64,{r['enhanced_b64']}" alt="Enhanced"></td>
                    <td>{r['sharpness_before']} &rarr; <strong>{r['sharpness_after']}</strong></td>
                    <td>
                        <span class="{m_b_class}">{r['match_before']}</span> &rarr; 
                        <span class="{m_a_class}">{r['match_after']}</span>
                    </td>
                </tr>
        """

    html += """
            </tbody>
        </table>
    </body>
    </html>
    """
    output_path.write_text(html, encoding="utf-8")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()

    # Load reference encodings for evaluation
    reference_encodings = {}
    for ref in sorted(REFERENCE_DIR.glob("*")):
        if ref.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        img = cv2.imread(str(ref))
        if img is None:
            continue
        enc = get_face_encoding(img)
        if enc is not None:
            reference_encodings[ref.stem] = enc
            print(f"  Reference: {ref.stem}")
        else:
            print(f"  WARNING: no face in {ref.name}")

    print(f"Loaded {len(reference_encodings)} reference identities")

    face_paths = sorted(RAW_FACES_DIR.glob("*.jpg")) + sorted(RAW_FACES_DIR.glob("*.png"))
    print(f"Processing {len(face_paths)} face crops ...")

    results = []

    for fp in face_paths:
        raw = cv2.imread(str(fp))
        if raw is None:
            continue

        enhanced = enhance_face(raw.copy())
        cv2.imwrite(str(ENHANCED_DIR / fp.name), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])

        sharp_b  = sharpness(raw)
        sharp_a  = sharpness(enhanced)
        ssim_g   = ssim_score(cv2.resize(raw, TARGET_SIZE), enhanced)

        enc_raw = get_face_encoding(raw)
        enc_enh = get_face_encoding(enhanced)

        match_b = False
        match_a = False
        mid     = None

        if reference_encodings:
            import face_recognition as fr
            refs  = list(reference_encodings.values())
            names = list(reference_encodings.keys())
            if enc_raw is not None:
                match_b = any(fr.compare_faces(refs, enc_raw, tolerance=0.60))
            if enc_enh is not None:
                hits = fr.compare_faces(refs, enc_enh, tolerance=0.60)
                match_a = any(hits)
                if match_a:
                    mid = names[hits.index(True)]

        # Encode for report
        _, rb = cv2.imencode(".jpg", cv2.resize(raw, TARGET_SIZE), [cv2.IMWRITE_JPEG_QUALITY, 82])
        _, eb = cv2.imencode(".jpg", enhanced, [cv2.IMWRITE_JPEG_QUALITY, 82])

        results.append({
            "filename":          fp.name,
            "original_size_px":  list(raw.shape[:2]),
            "enhanced_size_px":  list(enhanced.shape[:2]),
            "sharpness_before":  round(sharp_b, 2),
            "sharpness_after":   round(sharp_a, 2),
            "ssim_improvement":  round(ssim_g, 4),
            "match_before":      match_b,
            "match_after":       match_a,
            "matched_identity":  mid,
            "raw_b64":           base64.b64encode(rb).decode(),
            "enhanced_b64":      base64.b64encode(eb).decode(),
        })
        print(f"  {fp.name}: sharp {sharp_b:.1f}→{sharp_a:.1f}  match {match_b}→{match_a}")

    n   = len(results)
    t_s = round(time.time() - t_start, 2)

    metrics = {
        "source":                          "p4_face_enhancement",
        "total_faces_processed":           n,
        "processing_time_sec":             t_s,
        "pipeline_stages_applied":         ["denoise", "clahe", "upscale_multistep", "zone_sharpen"],
        "recognition_accuracy_before_pct": round(sum(r["match_before"] for r in results) / n * 100, 1) if n else 0.0,
        "recognition_accuracy_after_pct":  round(sum(r["match_after"]  for r in results) / n * 100, 1) if n else 0.0,
        "avg_sharpness_before":            round(float(np.mean([r["sharpness_before"] for r in results])), 2) if results else 0.0,
        "avg_sharpness_after":             round(float(np.mean([r["sharpness_after"]  for r in results])), 2) if results else 0.0,
        "avg_ssim_improvement":            round(float(np.mean([r["ssim_improvement"] for r in results])), 4) if results else 0.0,
        "per_face": [{k: v for k, v in r.items() if k not in ["raw_b64", "enhanced_b64"]} for r in results],
    }

    with open(METRICS_JSON_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    generate_ab_report(results, REPORT_HTML_OUT)

    print()
    print("=" * 55)
    print(f"  Done in {t_s}s  for {n} faces")
    print(f"  Recognition:  {metrics['recognition_accuracy_before_pct']}%  →  {metrics['recognition_accuracy_after_pct']}%")
    print(f"  Sharpness:    {metrics['avg_sharpness_before']}  →  {metrics['avg_sharpness_after']}")
    print(f"  Enhanced  → {ENHANCED_DIR}/")
    print(f"  Report    → {REPORT_HTML_OUT}")
    print(f"  Metrics   → {METRICS_JSON_OUT}")
    print("=" * 55)
