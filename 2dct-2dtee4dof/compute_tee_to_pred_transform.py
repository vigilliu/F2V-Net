#!/usr/bin/env python3
"""
Compute tee_random_mid_256.png -> pred_slice.png transform from point JSON labels,
then save warped image and overlay visualization in each case folder.

Note:
- With 2 points, a full unconstrained affine transform is underdetermined.
- This script estimates a similarity transform (scale + rotation + translation),
  which is an affine transform with additional constraints.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image


Point = Tuple[float, float]

AFFINE_MODE = Image.Transform.AFFINE if hasattr(Image, "Transform") else Image.AFFINE
RESAMPLE_BILINEAR = (
    Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
)


def load_two_points(points_json: Path) -> List[Point]:
    with open(points_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    pts = data.get("points", [])
    if len(pts) != 2:
        raise ValueError(f"{points_json} must contain exactly 2 points.")
    return [(float(pts[0]["x"]), float(pts[0]["y"])), (float(pts[1]["x"]), float(pts[1]["y"]))]


def estimate_similarity(src_pts: Sequence[Point], dst_pts: Sequence[Point]) -> np.ndarray:
    """
    Estimate 2D similarity transform (forward) mapping src -> dst:
      [x_d, y_d, 1]^T = M @ [x_s, y_s, 1]^T
    where M is 3x3 homogeneous.
    """
    p1 = np.array(src_pts[0], dtype=np.float64)
    p2 = np.array(src_pts[1], dtype=np.float64)
    q1 = np.array(dst_pts[0], dtype=np.float64)
    q2 = np.array(dst_pts[1], dtype=np.float64)

    v_src = p2 - p1
    v_dst = q2 - q1
    n_src = float(np.linalg.norm(v_src))
    n_dst = float(np.linalg.norm(v_dst))
    if n_src < 1e-8 or n_dst < 1e-8:
        raise ValueError("Point pairs are degenerate (distance too small).")

    scale = n_dst / n_src
    ang_src = math.atan2(v_src[1], v_src[0])
    ang_dst = math.atan2(v_dst[1], v_dst[0])
    theta = ang_dst - ang_src

    c = math.cos(theta)
    s = math.sin(theta)
    r = scale * np.array([[c, -s], [s, c]], dtype=np.float64)
    t = q1 - r @ p1

    m = np.eye(3, dtype=np.float64)
    m[:2, :2] = r
    m[:2, 2] = t
    return m


def to_pil_inverse_coeffs(forward_m: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """
    PIL Image.transform(..., Image.AFFINE, data=coeffs) expects inverse mapping:
      x_in = a*x_out + b*y_out + c
      y_in = d*x_out + e*y_out + f
    """
    inv = np.linalg.inv(forward_m)
    a, b, c = inv[0, 0], inv[0, 1], inv[0, 2]
    d, e, f = inv[1, 0], inv[1, 1], inv[1, 2]
    return float(a), float(b), float(c), float(d), float(e), float(f)


def apply_transform_to_point(m: np.ndarray, p: Point) -> Point:
    x, y = p
    out = m @ np.array([x, y, 1.0], dtype=np.float64)
    return float(out[0]), float(out[1])


def process_case(
    case_dir: Path,
    pred_img_name: str,
    tee_img_name: str,
    pred_json_name: str,
    tee_json_name: str,
    alpha: float,
    overwrite: bool,
) -> None:
    pred_img = case_dir / pred_img_name
    tee_img = case_dir / tee_img_name
    pred_json = case_dir / pred_json_name
    tee_json = case_dir / tee_json_name

    required = [pred_img, tee_img, pred_json, tee_json]
    if not all(p.exists() for p in required):
        return

    transform_json = case_dir / "tee_to_pred_transform.json"
    warped_tee_png = case_dir / "tee_warp_to_pred.png"
    overlay_png = case_dir / "tee_pred_overlay.png"

    if (
        not overwrite
        and transform_json.exists()
        and warped_tee_png.exists()
        and overlay_png.exists()
    ):
        print(f"Skip (exists): {case_dir}")
        return

    tee_pts = load_two_points(tee_json)
    pred_pts = load_two_points(pred_json)

    # Forward transform: tee -> pred
    m = estimate_similarity(tee_pts, pred_pts)
    coeffs = to_pil_inverse_coeffs(m)

    pred_img_pil = Image.open(pred_img).convert("RGB")
    tee_img_pil = Image.open(tee_img).convert("RGB")

    warped_tee = tee_img_pil.transform(
        pred_img_pil.size,
        AFFINE_MODE,
        coeffs,
        resample=RESAMPLE_BILINEAR,
    )
    overlay = Image.blend(pred_img_pil, warped_tee, alpha=alpha)

    warped_tee.save(warped_tee_png)
    overlay.save(overlay_png)

    warped_pts = [apply_transform_to_point(m, p) for p in tee_pts]
    errors = [
        float(
            math.hypot(
                warped_pts[i][0] - pred_pts[i][0],
                warped_pts[i][1] - pred_pts[i][1],
            )
        )
        for i in range(2)
    ]

    payload: Dict[str, object] = {
        "transform_type": "similarity (affine-constrained)",
        "from_image": tee_img_name,
        "to_image": pred_img_name,
        "matrix_3x3_forward_tee_to_pred": m.tolist(),
        "source_points_tee": [{"x": tee_pts[0][0], "y": tee_pts[0][1]}, {"x": tee_pts[1][0], "y": tee_pts[1][1]}],
        "target_points_pred": [{"x": pred_pts[0][0], "y": pred_pts[0][1]}, {"x": pred_pts[1][0], "y": pred_pts[1][1]}],
        "warped_source_points": [{"x": warped_pts[0][0], "y": warped_pts[0][1]}, {"x": warped_pts[1][0], "y": warped_pts[1][1]}],
        "point_errors_pixels": errors,
        "mean_error_pixels": float(sum(errors) / len(errors)),
        "outputs": {
            "warped_tee_image": warped_tee_png.name,
            "overlay_image": overlay_png.name,
        },
    }
    with open(transform_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Processed: {case_dir}")


def collect_case_dirs(root: Path) -> List[Path]:
    return [p for p in sorted(root.iterdir(), key=lambda x: x.name) if p.is_dir()]


def parse_case_ids(case_ids_text: str) -> set[str] | None:
    text = case_ids_text.strip()
    if not text:
        return None
    return {x.strip() for x in text.split(",") if x.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute tee->pred transform from 2-point labels and generate overlay images."
    )
    parser.add_argument("--root", type=str, default=".", help="Root directory containing case subfolders.")
    parser.add_argument(
        "--case-ids",
        type=str,
        default="",
        help="Optional comma-separated case folder names, e.g. 49,50,51",
    )
    parser.add_argument("--pred-image", type=str, default="pred_slice.png")
    parser.add_argument("--tee-image", type=str, default="tee_random_mid_256.png")
    parser.add_argument("--pred-json", type=str, default="pred_slice_points.json")
    parser.add_argument("--tee-json", type=str, default="tee_random_mid_256_points.json")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Blend ratio for overlay: overlay = (1-alpha)*pred + alpha*warped_tee",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0, 1]")

    case_id_filter = parse_case_ids(args.case_ids)
    case_dirs = collect_case_dirs(root)
    if case_id_filter is not None:
        case_dirs = [p for p in case_dirs if p.name in case_id_filter]

    if not case_dirs:
        raise RuntimeError("No case directories found.")

    for case_dir in case_dirs:
        process_case(
            case_dir=case_dir,
            pred_img_name=args.pred_image,
            tee_img_name=args.tee_image,
            pred_json_name=args.pred_json,
            tee_json_name=args.tee_json,
            alpha=args.alpha,
            overwrite=args.overwrite,
        )

    print("Finished.")


if __name__ == "__main__":
    main()
