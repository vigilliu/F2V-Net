#!/usr/bin/env python3
import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_ct_pose_tee_late_fusion import CTPoseTEEDataset
from networks.Vnet_6DoF_late_fusion import VNet6DOFLateFusion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Late fusion training: 3D CT + 2D TEE -> 6DoF")
    parser.add_argument("--root-path", type=str, default="/root/dof_project/3dct_point_dataset")
    parser.add_argument("--label-root-name", type=str, default="ctpoint_label_voxel_revise")
    parser.add_argument("--exp", type=str, default="vnet_late_fusion_ct_tee")
    parser.add_argument("--max-iterations", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--tee-frame-index", type=int, default=-1)
    parser.add_argument("--tee-rgb", action="store_true", help="Use RGB TEE input (3 channels).")
    return parser


def set_seed(seed: int, deterministic: bool) -> None:
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def pose_to_normal(pose: torch.Tensor) -> torch.Tensor:
    # pose expected in normalized range, same convention as existing codebase
    angles = pose[:, 3:6] * torch.pi
    normal = torch.stack(
        [torch.cos(angles[:, 0]), torch.cos(angles[:, 1]), torch.cos(angles[:, 2])],
        dim=1,
    )
    return F.normalize(normal, dim=1)


def main():
    args = build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed, bool(args.deterministic))

    snapshot_path = f"./model/{args.exp}"
    os.makedirs(snapshot_path, exist_ok=True)

    data_root = os.path.join(args.root_path, "data")
    label_root = os.path.join(args.root_path, args.label_root_name)

    train_set = CTPoseTEEDataset(
        data_root=data_root,
        label_root=label_root,
        split="train",
        tee_frame_index=args.tee_frame_index,
        tee_as_gray=(not args.tee_rgb),
    )
    val_set = CTPoseTEEDataset(
        data_root=data_root,
        label_root=label_root,
        split="val",
        tee_frame_index=args.tee_frame_index,
        tee_as_gray=(not args.tee_rgb),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

    model = VNet6DOFLateFusion(
        ct_in_channels=1,
        tee_in_channels=3 if args.tee_rgb else 1,
        n_filters=16,
        normalization="batchnorm",
    ).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)

    best_val = 1e9
    iter_num = 0
    max_epoch = args.max_iterations // max(1, len(train_loader)) + 1

    for _ in tqdm(range(max_epoch), ncols=70):
        model.train()
        for batch in train_loader:
            ct = batch["image_ct"].to(device)
            tee = batch["image_tee"].to(device)
            gt = batch["pose"].to(device)

            pred = model(ct, tee)
            trans_loss = F.mse_loss(pred[:, 0:3], gt[:, 0:3])
            rot_loss = F.mse_loss(pose_to_normal(pred), pose_to_normal(gt))
            loss = trans_loss + rot_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            if iter_num % 200 == 0:
                model.eval()
                val_loss_sum = 0.0
                with torch.no_grad():
                    for vb in val_loader:
                        ct_v = vb["image_ct"].to(device)
                        tee_v = vb["image_tee"].to(device)
                        gt_v = vb["pose"].to(device)
                        pr_v = model(ct_v, tee_v)
                        tl = F.mse_loss(pr_v[:, 0:3], gt_v[:, 0:3])
                        rl = F.mse_loss(pose_to_normal(pr_v), pose_to_normal(gt_v))
                        val_loss_sum += (tl + rl).item()
                val_loss = val_loss_sum / max(1, len(val_loader))
                print(f"iter={iter_num} train_loss={loss.item():.4f} val_loss={val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(model.state_dict(), os.path.join(snapshot_path, "best_model.pth"))
                    print("Saved best_model.pth")

            if iter_num % 1000 == 0:
                torch.save(model.state_dict(), os.path.join(snapshot_path, f"iter_{iter_num}.pth"))
            if iter_num >= args.max_iterations:
                break
        if iter_num >= args.max_iterations:
            break

    torch.save(model.state_dict(), os.path.join(snapshot_path, "last_model.pth"))
    print("Training done.")


if __name__ == "__main__":
    main()
