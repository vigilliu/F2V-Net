import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


def _normalize_to_minus1_1(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x * 2.0 - 1.0


def _load_tee_frame_as_2d(tee_path: Path, frame_index: int = -1, as_gray: bool = True) -> np.ndarray:
    """
    Load one frame from TEE NIfTI and convert to 2D tensor-friendly array.

    Supported common layouts:
    - [H, W]
    - [H, W, T]
    - [H, W, T, C]
    - [H, W, T, 1, C]
    """
    img_obj = nib.load(str(tee_path))
    tee_obj = img_obj.dataobj
    tee_shape = img_obj.shape

    if len(tee_shape) == 2:
        img = np.asanyarray(tee_obj[:, :], dtype=np.float32)
    elif len(tee_shape) == 3:
        t = tee_shape[2] - 1 if frame_index < 0 else min(frame_index, tee_shape[2] - 1)
        img = np.asanyarray(tee_obj[:, :, t], dtype=np.float32)
    elif len(tee_shape) == 4:
        t = tee_shape[2] - 1 if frame_index < 0 else min(frame_index, tee_shape[2] - 1)
        img = np.asanyarray(tee_obj[:, :, t, :], dtype=np.float32)
    elif len(tee_shape) == 5:
        t = tee_shape[2] - 1 if frame_index < 0 else min(frame_index, tee_shape[2] - 1)
        img = np.asanyarray(tee_obj[:, :, t, 0, :], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported TEE shape: {tee_shape}")

    if img.ndim == 2:
        img = _normalize_to_minus1_1(img)
        return img[None]  # [1, H, W]

    if img.ndim == 3:
        # Assume [H,W,C]
        if as_gray:
            if img.shape[-1] >= 3:
                img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
            else:
                img = img[..., 0]
            img = _normalize_to_minus1_1(img)
            return img[None]  # [1, H, W]

        img = _normalize_to_minus1_1(img)
        img = np.transpose(img, (2, 0, 1))  # [C,H,W]
        return img

    raise ValueError(f"Unexpected parsed TEE frame shape: {img.shape}")


class CTPoseTEEDataset(Dataset):
    """
    Late-fusion dataset:
      - CT branch input:  [1, D, H, W]
      - TEE branch input: [1, H, W] or [3, H, W]
      - target pose:      [6] (same normalization as existing pipeline)
    """

    def __init__(
        self,
        data_root: str,
        label_root: str,
        split: str = "train",
        tee_frame_index: int = -1,
        tee_as_gray: bool = True,
    ):
        self.data_root = Path(data_root)
        self.label_root = Path(label_root)
        self.tee_frame_index = tee_frame_index
        self.tee_as_gray = tee_as_gray

        self.ids = sorted(os.listdir(self.data_root))
        split_idx = int(0.8 * len(self.ids))
        if split == "train":
            self.ids = self.ids[:split_idx]
        else:
            self.ids = self.ids[split_idx:]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        case_id = self.ids[idx]

        ct_path = self.data_root / case_id / "cta" / f"{case_id}ct.nii.gz"
        tee_path = self.data_root / case_id / "tee" / f"{case_id}tee.nii.gz"
        pose_path = self.label_root / case_id / "6DoF.txt"

        if not ct_path.exists():
            raise FileNotFoundError(f"Missing CT: {ct_path}")
        if not tee_path.exists():
            raise FileNotFoundError(f"Missing TEE: {tee_path}")
        if not pose_path.exists():
            raise FileNotFoundError(f"Missing label: {pose_path}")

        # CT preprocess (same as current CTPoseDataset)
        ct = nib.load(str(ct_path)).get_fdata().astype(np.float32)
        ct = np.clip(ct, 0, 2000)
        ct = _normalize_to_minus1_1(ct)
        ct = ct[None]  # [1, D, H, W]

        # TEE preprocess
        tee = _load_tee_frame_as_2d(
            tee_path=tee_path,
            frame_index=self.tee_frame_index,
            as_gray=self.tee_as_gray,
        )  # [1,H,W] or [3,H,W]

        # Pose preprocess (same scaling as current CTPoseDataset)
        pose = np.genfromtxt(pose_path).astype(np.float32).reshape(6)
        pose[0:3] = pose[0:3] / 128.0
        pose[3:6] = pose[3:6] / 180.0

        return {
            "image_ct": torch.from_numpy(ct),
            "image_tee": torch.from_numpy(tee),
            "pose": torch.from_numpy(pose),
            "case_id": case_id,
        }


if __name__ == "__main__":
    dataset = CTPoseTEEDataset(
        data_root="/root/dof_project/3dct_point_dataset/data",
        label_root="/root/dof_project/3dct_point_dataset/ctpoint_label_voxel_revise",
        split="train",
        tee_frame_index=-1,
        tee_as_gray=True,
    )
    sample = dataset[0]
    print("case:", sample["case_id"])
    print("ct shape:", tuple(sample["image_ct"].shape))
    print("tee shape:", tuple(sample["image_tee"].shape))
    print("pose shape:", tuple(sample["pose"].shape))
