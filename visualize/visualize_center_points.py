#!/usr/bin/env python3
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_center_from_6dof(file_path: Path) -> np.ndarray:
    text = file_path.read_text(encoding="utf-8").strip()
    parts = text.split()
    if len(parts) < 3:
        raise ValueError(f"{file_path} does not contain at least 3 values.")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)


def build_sphere_mask(shape_xyz: tuple[int, int, int], center_xyz: np.ndarray, radius: int) -> np.ndarray:
    center_idx = np.round(center_xyz).astype(np.int32)
    center_idx = np.clip(center_idx, [0, 0, 0], np.array(shape_xyz) - 1)
    x = np.arange(shape_xyz[0])[:, None, None]
    y = np.arange(shape_xyz[1])[None, :, None]
    z = np.arange(shape_xyz[2])[None, None, :]
    dist2 = (x - center_idx[0]) ** 2 + (y - center_idx[1]) ** 2 + (z - center_idx[2]) ** 2
    return dist2 <= (radius ** 2)


def make_rgb_volume(gray_volume: np.ndarray, sphere_mask: np.ndarray) -> np.ndarray:
    vol = gray_volume.astype(np.float32)
    vmin = float(np.min(vol))
    vmax = float(np.max(vol))
    if vmax <= vmin:
        norm = np.zeros_like(vol, dtype=np.uint8)
    else:
        norm = ((vol - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)

    rgb = np.stack([norm, norm, norm], axis=-1)
    rgb[sphere_mask, 0] = 255
    rgb[sphere_mask, 1] = 0
    rgb[sphere_mask, 2] = 0
    return rgb


def process_one_case(label_file: Path, data_root: Path, output_root: Path, radius: int) -> tuple[bool, str]:
    case_id = label_file.parent.name
    ct_path = data_root / case_id / "cta" / f"{case_id}ct.nii.gz"
    if not ct_path.exists():
        return False, f"[missing_ct] {case_id}: {ct_path}"

    try:
        center = parse_center_from_6dof(label_file)
    except Exception as exc:  # noqa: BLE001
        return False, f"[bad_6dof] {case_id}: {exc}"

    img = nib.load(str(ct_path))
    arr = img.get_fdata()
    if arr.ndim != 3:
        return False, f"[bad_dim] {case_id}: expected 3D, got {arr.ndim}D"

    sphere = build_sphere_mask(arr.shape, center, radius=radius)
    rgb = make_rgb_volume(arr, sphere)

    out_case_dir = output_root / case_id
    out_case_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = out_case_dir / f"{case_id}ct_center_red.nii.gz"
    out_img = nib.Nifti1Image(rgb, affine=img.affine, header=img.header)
    nib.save(out_img, str(out_img_path))
    return True, f"[ok] {case_id}: {out_img_path}"


def process_label_set(label_root: Path, data_root: Path, output_root: Path, radius: int) -> list[str]:
    logs: list[str] = []
    if not label_root.exists():
        logs.append(f"[missing_label_root] {label_root}")
        return logs

    label_files = sorted(label_root.glob("*/6DoF.txt"), key=lambda p: int(p.parent.name))
    if not label_files:
        logs.append(f"[empty] no 6DoF.txt found under {label_root}")
        return logs

    for lf in label_files:
        ok, msg = process_one_case(lf, data_root, output_root, radius=radius)
        logs.append(msg)
    return logs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize first 3 values in 6DoF.txt as red sphere on CT .nii.gz."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/root/dof_project/3dct_point_dataset_DS4"),
        help="Root path containing data/, ctpoint_label_voxel_revise_DS4/, ctpoint_label_voxel_right_DS4/.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=3,
        help="Sphere radius in voxels.",
    )
    args = parser.parse_args()

    dataset_root: Path = args.dataset_root
    data_root = dataset_root / "data"
    revise_root = dataset_root / "ctpoint_label_voxel_revise_DS4"
    right_root = dataset_root / "ctpoint_label_voxel_right_DS4"
    out_root = dataset_root / "check_data_centre_point"
    out_root.mkdir(parents=True, exist_ok=True)

    revise_out = out_root / "ctpoint_label_voxel_revise_DS4"
    right_out = out_root / "ctpoint_label_voxel_right_DS4"
    revise_out.mkdir(parents=True, exist_ok=True)
    right_out.mkdir(parents=True, exist_ok=True)

    logs: list[str] = []
    logs.append("=== revise set ===")
    logs.extend(process_label_set(revise_root, data_root, revise_out, radius=args.radius))
    logs.append("=== right set ===")
    logs.extend(process_label_set(right_root, data_root, right_out, radius=args.radius))

    log_file = out_root / "visualization_log.txt"
    log_file.write_text("\n".join(logs) + "\n", encoding="utf-8")
    print(f"Done. Outputs in: {out_root}")
    print(f"Log: {log_file}")


if __name__ == "__main__":
    main()
