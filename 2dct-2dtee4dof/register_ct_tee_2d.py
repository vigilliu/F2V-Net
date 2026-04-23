#!/usr/bin/env python3
"""
2D CT-TEE registration utility (single pair and batch).

Examples
--------
Single pair:
python register_ct_tee_2d.py \
  --fixed /root/dof_project/single_infer_case1_crop96/pred_slice.png \
  --moving /root/dof_project/3dct_point_dataset_DS4/3dtee_3dct_DS4/49/tee_random_mid_256.png \
  --out-dir /root/dof_project/single_infer_case1_crop96/registration

Batch:
python register_ct_tee_2d.py \
  --cases-root /root/dof_project/3dct_point_dataset_DS4/3dtee_3dct_DS4 \
  --fixed-name pred_slice.png \
  --moving-name tee_random_mid_256.png
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import SimpleITK as sitk


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register 2D CT slice and 2D TEE image.")
    parser.add_argument("--fixed", type=Path, default=None, help="Fixed image path (e.g. CT slice PNG).")
    parser.add_argument("--moving", type=Path, default=None, help="Moving image path (e.g. TEE PNG).")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for single-pair mode.")
    parser.add_argument("--prefix", type=str, default="ct_tee", help="Output filename prefix in single mode.")

    parser.add_argument(
        "--cases-root",
        type=Path,
        default=None,
        help="Batch mode root. Each case is a numeric subfolder under this directory.",
    )
    parser.add_argument("--fixed-name", type=str, default="pred_slice.png", help="Fixed image filename in each case.")
    parser.add_argument(
        "--moving-name",
        type=str,
        default="tee_random_mid_256.png",
        help="Moving image filename in each case.",
    )
    parser.add_argument(
        "--batch-out-subdir",
        type=str,
        default="registration_2d",
        help="Output subdirectory name inside each case folder in batch mode.",
    )

    parser.add_argument("--mi-bins", type=int, default=50, help="Mattes MI histogram bins.")
    parser.add_argument("--sample-ratio", type=float, default=0.2, help="Metric sampling ratio in (0,1].")
    parser.add_argument("--max-iter", type=int, default=300, help="Max optimization iterations.")
    return parser.parse_args()


def _to_float01(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def _save_visuals(fixed_arr: np.ndarray, warped_arr: np.ndarray, out_dir: Path, prefix: str) -> None:
    fixed_u8 = (np.clip(fixed_arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    warped_u8 = (np.clip(warped_arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    overlay = np.stack([fixed_u8, warped_u8, fixed_u8], axis=2)

    h, w = fixed_u8.shape
    tile = max(8, min(h, w) // 16)
    checker = np.empty_like(fixed_u8)
    for y in range(h):
        for x in range(w):
            if ((y // tile) + (x // tile)) % 2 == 0:
                checker[y, x] = fixed_u8[y, x]
            else:
                checker[y, x] = warped_u8[y, x]

    imageio.imwrite(out_dir / f"{prefix}_fixed.png", fixed_u8)
    imageio.imwrite(out_dir / f"{prefix}_warped.png", warped_u8)
    imageio.imwrite(out_dir / f"{prefix}_overlay.png", overlay)
    imageio.imwrite(out_dir / f"{prefix}_checkerboard.png", checker)


def register_pair(
    fixed_path: Path,
    moving_path: Path,
    out_dir: Path,
    prefix: str,
    mi_bins: int,
    sample_ratio: float,
    max_iter: int,
) -> dict:
    if not fixed_path.exists():
        raise FileNotFoundError(f"Fixed image not found: {fixed_path}")
    if not moving_path.exists():
        raise FileNotFoundError(f"Moving image not found: {moving_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_np = _to_float01(imageio.imread(fixed_path))
    moving_np = _to_float01(imageio.imread(moving_path))
    fixed = sitk.GetImageFromArray(fixed_np)
    moving = sitk.GetImageFromArray(moving_np)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=mi_bins)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sample_ratio)
    reg.SetInterpolator(sitk.sitkLinear)

    initial_tx = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg.SetInitialTransform(initial_tx, inPlace=False)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=max_iter,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-8,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_tx = reg.Execute(fixed, moving)

    warped = sitk.Resample(moving, fixed, final_tx, sitk.sitkLinear, 0.0, moving.GetPixelID())
    warped_np = sitk.GetArrayFromImage(warped).astype(np.float32)

    _save_visuals(fixed_np, warped_np, out_dir, prefix)
    sitk.WriteTransform(final_tx, str(out_dir / f"{prefix}_transform.tfm"))

    summary = {
        "fixed_path": str(fixed_path),
        "moving_path": str(moving_path),
        "metric_final": float(reg.GetMetricValue()),
        "optimizer_stop_condition": reg.GetOptimizerStopConditionDescription(),
        "iterations": int(reg.GetOptimizerIteration()),
        "transform_type": final_tx.GetName(),
        "transform_parameters": [float(v) for v in final_tx.GetParameters()],
        "fixed_parameters": [float(v) for v in final_tx.GetFixedParameters()],
    }
    with (out_dir / f"{prefix}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_batch(args: argparse.Namespace) -> None:
    if args.cases_root is None:
        raise ValueError("--cases-root is required in batch mode.")
    if not args.cases_root.exists():
        raise FileNotFoundError(f"Cases root not found: {args.cases_root}")

    cases = sorted(
        [p for p in args.cases_root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    ok = 0
    fail = 0
    report = []

    for case_dir in cases:
        fixed_path = case_dir / args.fixed_name
        moving_path = case_dir / args.moving_name
        out_dir = case_dir / args.batch_out_subdir
        prefix = f"{case_dir.name}_ct_tee"
        try:
            summary = register_pair(
                fixed_path=fixed_path,
                moving_path=moving_path,
                out_dir=out_dir,
                prefix=prefix,
                mi_bins=args.mi_bins,
                sample_ratio=args.sample_ratio,
                max_iter=args.max_iter,
            )
            ok += 1
            report.append(
                {
                    "case": case_dir.name,
                    "status": "OK",
                    "metric_final": summary["metric_final"],
                    "out_dir": str(out_dir),
                }
            )
            logging.info("Case %s OK (metric=%.6f)", case_dir.name, summary["metric_final"])
        except Exception as ex:
            fail += 1
            report.append({"case": case_dir.name, "status": "FAIL", "error": f"{type(ex).__name__}: {ex}"})
            logging.warning("Case %s FAIL: %s", case_dir.name, ex)

    report_path = args.cases_root / "registration_2d_batch_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump({"ok": ok, "fail": fail, "total": len(cases), "cases": report}, f, indent=2)
    logging.info("Batch done. ok=%d fail=%d total=%d report=%s", ok, fail, len(cases), report_path)


def main() -> None:
    setup_logger()
    args = parse_args()

    single_mode = args.fixed is not None or args.moving is not None or args.out_dir is not None
    batch_mode = args.cases_root is not None

    if single_mode and batch_mode:
        raise ValueError("Choose either single mode (--fixed/--moving/--out-dir) or batch mode (--cases-root), not both.")
    if not single_mode and not batch_mode:
        raise ValueError("Please provide single mode args or --cases-root for batch mode.")

    if not (0.0 < args.sample_ratio <= 1.0):
        raise ValueError("--sample-ratio must be in (0, 1].")
    if args.mi_bins <= 0:
        raise ValueError("--mi-bins must be > 0.")
    if args.max_iter <= 0:
        raise ValueError("--max-iter must be > 0.")

    if batch_mode:
        run_batch(args)
        return

    if args.fixed is None or args.moving is None or args.out_dir is None:
        raise ValueError("Single mode requires --fixed, --moving, and --out-dir.")

    summary = register_pair(
        fixed_path=args.fixed,
        moving_path=args.moving,
        out_dir=args.out_dir,
        prefix=args.prefix,
        mi_bins=args.mi_bins,
        sample_ratio=args.sample_ratio,
        max_iter=args.max_iter,
    )
    logging.info("Done. metric=%.6f iterations=%d", summary["metric_final"], summary["iterations"])


if __name__ == "__main__":
    main()

