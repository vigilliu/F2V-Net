import os
import argparse
import logging
import random
import time

import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import zoom

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter

from networks.Vnet_6DoF import VNet6DOF


# -------------------------
# Dataset: CT + segmentation
# -------------------------
class CTPoseSegDataset(Dataset):
    """
    返回:
      image: [2, D, H, W]  (CT, SEG)
      pose : [6]           (归一化: xyz/128, angles/180)
    """
    def __init__(
        self,
        data_root: str,
        label_root: str,
        seg_root: str,
        split: str = "train",
        train_ratio: float = 0.8,
        seg_mode: str = "binary",
    ):
        self.data_root = data_root
        self.label_root = label_root
        self.seg_root = seg_root
        self.seg_mode = seg_mode

        ids = sorted(os.listdir(data_root))
        cut = int(train_ratio * len(ids))
        if split == "train":
            self.ids = ids[:cut]
        else:
            self.ids = ids[cut:]

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _load_nii(path: str) -> np.ndarray:
        return nib.load(path).get_fdata().astype(np.float32)

    @staticmethod
    def _normalize_ct(ct: np.ndarray) -> np.ndarray:
        ct = np.clip(ct, 0, 2000)
        ct = (ct - ct.min()) / (ct.max() - ct.min() + 1e-8)
        return ct * 2 - 1

    @staticmethod
    def _downsample_seg(seg: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
        if seg.shape == target_shape:
            return seg

        ratios = [seg.shape[i] / target_shape[i] for i in range(3)]
        is_int_ratio = all(abs(r - round(r)) < 1e-6 for r in ratios)
        if is_int_ratio:
            step = [int(round(r)) for r in ratios]
            return seg[:: step[0], :: step[1], :: step[2]]

        # fallback: nearest-neighbor resample
        zf = [target_shape[i] / seg.shape[i] for i in range(3)]
        return zoom(seg, zoom=zf, order=0)

    def __getitem__(self, idx):
        case_id = self.ids[idx]

        ct_path = os.path.join(self.data_root, case_id, "cta", f"{case_id}ct.nii.gz")
        seg_path = os.path.join(self.seg_root, case_id, "cta", f"{case_id}ct_seg.nii.gz")
        pose_path = os.path.join(self.label_root, case_id, "6DoF.txt")

        ct = self._load_nii(ct_path)
        ct = self._normalize_ct(ct)

        seg = self._load_nii(seg_path)
        seg = self._downsample_seg(seg, ct.shape)

        if self.seg_mode == "binary":
            seg = (seg > 0).astype(np.float32)
        elif self.seg_mode == "scaled":
            vmax = float(seg.max()) if seg.size else 1.0
            seg = (seg / (vmax + 1e-8)).astype(np.float32)
        else:
            raise ValueError(f"Unsupported seg_mode: {self.seg_mode}")

        image = np.stack([ct, seg], axis=0)  # [2, D, H, W]

        pose = np.genfromtxt(pose_path).astype(np.float32).reshape(6)
        pose[0:3] = pose[0:3] / 128.0
        pose[3:6] = pose[3:6] / 180.0

        return {
            "image": torch.from_numpy(image),
            "pose": torch.from_numpy(pose),
        }


# -------------------------
# MI loss
# -------------------------
class MutualInformation(torch.nn.Module):
    """
    Differentiable Mutual Information Loss (same style as train.py).
    """
    def __init__(self, num_bins=64):
        super().__init__()
        self.num_bins = num_bins

    def forward(self, x, y):
        """
        x,y: [B,1,H,W] in [0,1]
        """
        B = x.shape[0]
        x = x.view(B, -1)
        y = y.view(B, -1)

        mi_total = 0.0
        for b in range(B):
            xi = x[b]
            yi = y[b]

            joint_hist = torch.histc(
                xi * self.num_bins + yi,
                bins=self.num_bins,
                min=0,
                max=self.num_bins,
            )
            joint_prob = joint_hist / torch.sum(joint_hist) + 1e-8
            px = torch.sum(joint_prob)
            py = torch.sum(joint_prob)
            mi = torch.sum(joint_prob * torch.log(joint_prob / (px * py)))
            mi_total += mi

        return -mi_total / B


# -------------------------
# Plane helpers (same as updated train.py)
# -------------------------
def pose_to_normal(pose: torch.Tensor) -> torch.Tensor:
    """
    pose: [B,6]  (angles already normalized by /180 in dataset)
    return: normal [B,3]
    """
    angles = pose[:, 3:6] * torch.pi
    normal = torch.stack(
        [torch.cos(angles[:, 0]), torch.cos(angles[:, 1]), torch.cos(angles[:, 2])],
        dim=1,
    )
    return F.normalize(normal, dim=1)


def default_tangent_frame(normal: torch.Tensor):
    B, _ = normal.shape
    device = normal.device
    tmp = torch.tensor([1.0, 0.0, 0.0], device=device).expand(B, 3).clone()
    dot = torch.abs((tmp * normal).sum(dim=1, keepdim=True))
    mask = (dot > 0.9).squeeze(1)
    tmp[mask] = torch.tensor([0.0, 1.0, 0.0], device=device)
    u = torch.cross(normal, tmp, dim=1)
    u = F.normalize(u, dim=1, eps=1e-8)
    v = torch.cross(normal, u, dim=1)
    v = F.normalize(v, dim=1, eps=1e-8)
    return u, v


def tangent_frame_align_to_ref(normal: torch.Tensor, u_ref: torch.Tensor):
    proj = (u_ref * normal).sum(dim=1, keepdim=True)
    u = u_ref - proj * normal
    unorm = u.norm(dim=1, keepdim=True)
    tiny = (unorm.squeeze(1) < 1e-4)
    u_def, _ = default_tangent_frame(normal)
    u = torch.where(tiny.unsqueeze(1).expand_as(u), u_def, u / unorm.clamp(1e-8))
    v = torch.cross(normal, u, dim=1)
    v = F.normalize(v, dim=1, eps=1e-8)
    return u, v


def extract_slice_torch(volume_ct: torch.Tensor, pose: torch.Tensor, size=128, align_u_ref=None):
    """
    volume_ct: [B,1,D,H,W]  (只用 CT 做 MI)
    pose: [B,6]
    """
    B, _, _, _, _ = volume_ct.shape
    device = volume_ct.device

    center = pose[:, 0:3]
    normal = pose_to_normal(pose)

    if align_u_ref is not None:
        u, v = tangent_frame_align_to_ref(normal, align_u_ref)
    else:
        u, v = default_tangent_frame(normal)

    coords = torch.linspace(-1, 1, size, device=device)
    uu, vv = torch.meshgrid(coords, coords, indexing="ij")
    uu = uu[None, None]
    vv = vv[None, None]

    center = center.view(B, 3, 1, 1)
    u = u.view(B, 3, 1, 1)
    v = v.view(B, 3, 1, 1)

    sample_points = center + u * uu + v * vv
    grid = sample_points.permute(0, 2, 3, 1).unsqueeze(1)

    slice_img = F.grid_sample(volume_ct, grid, mode="bilinear", align_corners=True)
    return slice_img.squeeze(2)


@torch.no_grad()
def evaluate(net, valloader, mi_loss_fn):
    net.eval()
    total_mi = 0.0
    count = 0
    for batch in valloader:
        volume = batch["image"].cuda()  # [B,2,D,H,W]
        volume_ct = volume[:, 0:1]
        gt_pose = batch["pose"].cuda()

        pred_pose = net(volume)

        n_gt = pose_to_normal(gt_pose)
        u_gt, _ = default_tangent_frame(n_gt)
        pred_slice = extract_slice_torch(volume_ct, pred_pose, align_u_ref=u_gt)
        gt_slice = extract_slice_torch(volume_ct, gt_pose)

        pred_slice = (pred_slice - pred_slice.min()) / (pred_slice.max() - pred_slice.min() + 1e-6)
        gt_slice = (gt_slice - gt_slice.min()) / (gt_slice.max() - gt_slice.min() + 1e-6)

        mi = -mi_loss_fn(pred_slice, gt_slice)
        total_mi += mi.item()
        count += 1

    net.train()
    return total_mi / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="/root/dof_project/3dct_point_dataset_DS4", help="dataset root")
    parser.add_argument("--seg_root", type=str, default="/root/dof_project/results", help="segmentation results root")
    parser.add_argument("--label_root", type=str, default="/root/dof_project/3dct_point_dataset_DS4/ctpoint_label_voxel_revise_DS4", help="6DoF label root")
    parser.add_argument("--exp", type=str, default="vnet_6dof_ct+seg", help="experiment name")
    parser.add_argument("--max_iterations", type=int, default=15000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seg_mode", type=str, default="binary", choices=["binary", "scaled"])
    parser.add_argument("--lambda_t", type=float, default=1.0)
    parser.add_argument("--lambda_r", type=float, default=0.1)
    parser.add_argument("--lambda_mi", type=float, default=0.5)
    args = parser.parse_args()

    snapshot_path = os.path.join("./model", args.exp)
    os.makedirs(snapshot_path, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    batch_size = args.batch_size * len(args.gpu.split(","))

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # model: 2-channel input (CT + Seg)
    net = VNet6DOF(n_channels=2, n_filters=16, normalization="batchnorm").cuda()
    net.train()

    db_train = CTPoseSegDataset(
        data_root=os.path.join(args.root_path, "data"),
        label_root=args.label_root,
        seg_root=args.seg_root,
        split="train",
        seg_mode=args.seg_mode,
    )
    db_val = CTPoseSegDataset(
        data_root=os.path.join(args.root_path, "data"),
        label_root=args.label_root,
        seg_root=args.seg_root,
        split="val",
        seg_mode=args.seg_mode,
    )

    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    valloader = DataLoader(
        db_val,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
    )

    optimizer = optim.SGD(net.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    mi_loss_fn = MutualInformation().cuda()
    writer = SummaryWriter(os.path.join(snapshot_path, "log"))

    best_mi = -1e9
    iter_num = 0
    max_epoch = args.max_iterations // max(len(trainloader), 1) + 1

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    logging.info("Train cases: %d | Val cases: %d", len(db_train), len(db_val))
    logging.info("Snapshot: %s", snapshot_path)

    for epoch_num in tqdm(range(max_epoch), ncols=80):
        time1 = time.time()
        for sampled_batch in trainloader:
            volume = sampled_batch["image"].cuda()      # [B,2,D,H,W]
            volume_ct = volume[:, 0:1]                 # [B,1,D,H,W]
            gt_pose = sampled_batch["pose"].cuda()     # [B,6]

            pred_pose = net(volume)

            # pose supervision (translation + normal-vector)
            pred_t = pred_pose[:, 0:3]
            gt_t = gt_pose[:, 0:3]
            trans_loss = F.mse_loss(pred_t, gt_t)

            n_pred = pose_to_normal(pred_pose)
            n_gt = pose_to_normal(gt_pose)
            rot_loss = F.mse_loss(n_pred, n_gt)

            pose_loss = args.lambda_t * trans_loss + args.lambda_r * rot_loss

            # MI: align Pred in-plane u to GT u
            u_gt, _ = default_tangent_frame(n_gt)
            pred_slice = extract_slice_torch(volume_ct, pred_pose, align_u_ref=u_gt)
            gt_slice = extract_slice_torch(volume_ct, gt_pose)

            pred_slice_n = (pred_slice - pred_slice.min()) / (pred_slice.max() - pred_slice.min() + 1e-6)
            gt_slice_n = (gt_slice - gt_slice.min()) / (gt_slice.max() - gt_slice.min() + 1e-6)
            mi_loss = mi_loss_fn(pred_slice_n, gt_slice_n)

            loss = pose_loss + args.lambda_mi * mi_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            if iter_num % 50 == 0:
                writer.add_scalar("loss/total", float(loss.item()), iter_num)
                writer.add_scalar("loss/trans", float(trans_loss.item()), iter_num)
                writer.add_scalar("loss/normal", float(rot_loss.item()), iter_num)
                writer.add_scalar("loss/mi", float(mi_loss.item()), iter_num)

            if iter_num % 200 == 0:
                val_mi = evaluate(net, valloader, mi_loss_fn)
                writer.add_scalar("val/mi", float(val_mi), iter_num)
                logging.info("iter=%d | loss=%.4f | val_mi=%.4f", iter_num, float(loss.item()), float(val_mi))

                if val_mi > best_mi:
                    best_mi = val_mi
                    save_path = os.path.join(snapshot_path, "best_model.pth")
                    torch.save(net.state_dict(), save_path)
                    logging.info("Saved BEST model: %s", save_path)

            if iter_num % 2500 == 0:
                lr_ = args.base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_

            if iter_num % 1000 == 0:
                save_path = os.path.join(snapshot_path, f"iter_{iter_num}.pth")
                torch.save(net.state_dict(), save_path)

            if iter_num > args.max_iterations:
                break

        if iter_num > args.max_iterations:
            break

        time2 = time.time()
        logging.info("epoch=%d done | %.2fs", epoch_num, time2 - time1)

    final_path = os.path.join(snapshot_path, f"iter_{args.max_iterations+1}.pth")
    torch.save(net.state_dict(), final_path)
    writer.close()
    logging.info("Training done. Final saved: %s", final_path)


if __name__ == "__main__":
    main()

