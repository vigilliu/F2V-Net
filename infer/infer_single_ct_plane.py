#!/usr/bin/env python3
"""
Single-case inference for CT plane prediction.

Pipeline:
1) Load one CT NIfTI (.nii/.nii.gz)
2) Apply the same preprocessing as training/testing
3) Run VNet6DOF with pretrained weights
4) Decode predicted 6DoF
5) Extract predicted plane image with configurable output size
"""

import argparse
import logging
from pathlib import Path

import imageio.v2 as imageio
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import map_coordinates, zoom

from networks.Vnet_6DoF import VNet6DOF


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer one CT case and generate predicted slice only (no metrics)."
    )
    parser.add_argument(
        "--ct-path",
        type=Path,
        required=True,
        help="Input CT path (.nii or .nii.gz).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Pretrained model .pth path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output folder for inference artifacts.",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=128,
        help="Predicted slice size (output_size x output_size).",
    )
    parser.add_argument(
        "--center-crop-size",
        type=int,
        default=96,
        help="Optional center crop size on predicted slice. 0 means no crop.",
    )
    parser.add_argument(
        "--final-resize-size",
        type=int,
        default=0,
        help="Optional final resize size after crop. 0 means no resize.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Inference device.",
    )
    parser.add_argument(
        "--window-width",
        type=float,
        default=600.0,
        help="CT window width for output slice visualization.",
    )
    parser.add_argument(
        "--window-level",
        type=float,
        default=300.0,
        help="CT window level for output slice visualization.",
    )
    return parser.parse_args()


def load_and_preprocess_ct(ct_path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(ct_path))
    ct = img.get_fdata().astype(np.float32)  # raw volume
    if ct.ndim != 3:
        raise ValueError(f"Expected 3D CT, got shape={ct.shape}")

    ct_proc = np.clip(ct, 0.0, 2000.0)
    ct_proc = (ct_proc - ct_proc.min()) / (ct_proc.max() - ct_proc.min() + 1e-8)
    ct_proc = ct_proc * 2.0 - 1.0
    return ct, ct_proc


def pose_to_plane(pose_xyz_rxyz_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    point = pose_xyz_rxyz_deg[:3]
    angles = pose_xyz_rxyz_deg[3:]
    rad = np.deg2rad(angles)

    normal = np.array([np.cos(rad[0]), np.cos(rad[1]), np.cos(rad[2])], dtype=np.float64)
    normal /= np.linalg.norm(normal) + 1e-8

    tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(tmp, normal)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    u = np.cross(normal, tmp)
    u /= np.linalg.norm(u) + 1e-8
    v = np.cross(normal, u)
    v /= np.linalg.norm(v) + 1e-8
    return point, normal, u, v


def extract_slice(
    ct: np.ndarray,
    point: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    size: int,
    window_width: float,
    window_level: float,
) -> np.ndarray:
    coords = np.linspace(-size // 2, size // 2, size)
    uu, vv = np.meshgrid(coords, coords)

    sample_points = point.reshape(3, 1, 1) + u.reshape(3, 1, 1) * uu + v.reshape(3, 1, 1) * vv
    sampled = map_coordinates(
        ct,
        [sample_points[0], sample_points[1], sample_points[2]],
        order=1,
        mode="nearest",
    )

    lower = window_level - window_width / 2.0
    upper = window_level + window_width / 2.0
    sampled = np.clip(sampled, lower, upper)
    sampled = (sampled - lower) / (upper - lower + 1e-8)
    return (sampled * 255).astype(np.uint8)


def center_crop_square(img: np.ndarray, crop_size: int) -> np.ndarray:
    if crop_size <= 0:
        return img
    h, w = img.shape
    if crop_size > min(h, w):
        raise ValueError(f"center crop size {crop_size} exceeds slice shape ({h}, {w})")
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[start_h : start_h + crop_size, start_w : start_w + crop_size]


def resize_square(img: np.ndarray, target_size: int) -> np.ndarray:
    if target_size <= 0:
        return img
    h, w = img.shape
    if h == target_size and w == target_size:
        return img
    resized = zoom(img.astype(np.float32), (target_size / h, target_size / w), order=1)
    return np.clip(resized, 0, 255).astype(np.uint8)


def build_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    net = VNet6DOF(n_channels=1, n_filters=16, normalization="batchnorm")
    state_dict = torch.load(str(model_path), map_location=device)
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    return net


@torch.no_grad()
def run_inference(net: torch.nn.Module, ct_proc: np.ndarray, device: torch.device) -> np.ndarray:
    vol = torch.from_numpy(ct_proc[None, None]).float().to(device)  # [1,1,D,H,W]
    pred_pose_norm = net(vol)[0].detach().cpu().numpy().astype(np.float32)  # [6]
    return pred_pose_norm


def denormalize_pose(pred_pose_norm: np.ndarray) -> np.ndarray:
    pred_pose_real = pred_pose_norm.copy()
    pred_pose_real[:3] *= 128.0
    pred_pose_real[3:] *= 180.0
    return pred_pose_real


def save_outputs(
    out_dir: Path,
    pred_slice_full: np.ndarray,
    pred_slice: np.ndarray,
    pred_pose_norm: np.ndarray,
    pred_pose_real: np.ndarray,
    input_shape: tuple[int, int, int],
    output_size: int,
    crop_size: int,
    final_resize_size: int,
    window_width: float,
    window_level: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    imageio.imwrite(out_dir / "pred_slice_full.png", pred_slice_full)
    np.save(out_dir / "pred_slice_full.npy", pred_slice_full)
    imageio.imwrite(out_dir / "pred_slice.png", pred_slice)
    np.save(out_dir / "pred_slice.npy", pred_slice)

    report_path = out_dir / "inference_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("===== Inference Report =====\n")
        f.write(f"Input CT size (D,H,W): {input_shape}\n")
        f.write(f"Model input tensor size: (1, 1, {input_shape[0]}, {input_shape[1]}, {input_shape[2]})\n")
        f.write(f"Extracted slice size (H,W): ({output_size}, {output_size})\n")
        f.write(f"Display window (WW/WL): ({window_width}, {window_level})\n")
        if crop_size > 0:
            f.write(f"Center crop size (H,W): ({crop_size}, {crop_size})\n")
        else:
            f.write("Center crop: disabled\n")
        if final_resize_size > 0:
            f.write(f"Final resize size (H,W): ({final_resize_size}, {final_resize_size})\n")
        else:
            f.write("Final resize: disabled\n")
        f.write(f"Final output slice size (H,W): {pred_slice.shape}\n\n")

        f.write("Pred pose (normalized):\n")
        f.write(" ".join([f"{v:.6f}" for v in pred_pose_norm]) + "\n\n")

        f.write("Pred pose (real scale):\n")
        f.write("x y z rx ry rz\n")
        f.write(" ".join([f"{v:.6f}" for v in pred_pose_real]) + "\n")


def main() -> None:
    setup_logger()
    args = parse_args()

    if not args.ct_path.exists():
        raise FileNotFoundError(f"CT file not found: {args.ct_path}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if args.output_size <= 0:
        raise ValueError("--output-size must be > 0")
    if args.center_crop_size < 0:
        raise ValueError("--center-crop-size must be >= 0")
    if args.center_crop_size > 0 and args.center_crop_size > args.output_size:
        raise ValueError("--center-crop-size must be <= --output-size")
    if args.final_resize_size < 0:
        raise ValueError("--final-resize-size must be >= 0")
    if args.window_width <= 0:
        raise ValueError("--window-width must be > 0")
    if args.device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA unavailable, fallback to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logging.info("Loading CT: %s", args.ct_path)
    ct_raw, ct_proc = load_and_preprocess_ct(args.ct_path)
    input_shape = tuple(ct_raw.shape)  # (D,H,W)
    logging.info("Input CT size (D,H,W): %s", input_shape)

    logging.info("Loading model: %s", args.model_path)
    net = build_model(args.model_path, device)

    logging.info("Running inference on %s", device)
    pred_pose_norm = run_inference(net, ct_proc, device)
    pred_pose_real = denormalize_pose(pred_pose_norm)

    point, _, u, v = pose_to_plane(pred_pose_real)
    pred_slice_full = extract_slice(
        ct_raw,
        point,
        u,
        v,
        size=args.output_size,
        window_width=args.window_width,
        window_level=args.window_level,
    )
    pred_slice = center_crop_square(pred_slice_full, args.center_crop_size)
    pred_slice = resize_square(pred_slice, args.final_resize_size)

    save_outputs(
        out_dir=args.output_dir,
        pred_slice_full=pred_slice_full,
        pred_slice=pred_slice,
        pred_pose_norm=pred_pose_norm,
        pred_pose_real=pred_pose_real,
        input_shape=input_shape,
        output_size=args.output_size,
        crop_size=args.center_crop_size,
        final_resize_size=args.final_resize_size,
        window_width=args.window_width,
        window_level=args.window_level,
    )

    if args.center_crop_size > 0:
        logging.info("Center crop enabled: (%d, %d)", args.center_crop_size, args.center_crop_size)
    if args.final_resize_size > 0:
        logging.info("Final resize enabled: (%d, %d)", args.final_resize_size, args.final_resize_size)
    logging.info("Display window (WW/WL): (%.1f, %.1f)", args.window_width, args.window_level)
    logging.info("Final output slice size (H,W): (%d, %d)", pred_slice.shape[0], pred_slice.shape[1])
    logging.info("Saved results to: %s", args.output_dir)


if __name__ == "__main__":
    main()
