"""
与 train_with_seg.py 一致：CT + 分割（2 通道）输入的 6DoF 测试脚本。
切片可视化与 MI 仍基于 CT 通道。
"""
import argparse
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import map_coordinates
from torch.utils.data import DataLoader

from networks.Vnet_6DoF import VNet6DOF
from train_with_seg import CTPoseSegDataset


def mutual_information(img1, img2, bins=64):
    img1 = img1.ravel()
    img2 = img2.ravel()
    hist_2d, _, _ = np.histogram2d(img1, img2, bins=bins)
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nz = pxy > 0
    mi = np.sum(pxy[nz] * np.log(pxy[nz] / px_py[nz]))
    return mi


def pose_to_plane(pose):
    point = pose[:3]
    angles = pose[3:]
    rad = np.deg2rad(angles)
    normal = np.array(
        [np.cos(rad[0]), np.cos(rad[1]), np.cos(rad[2])], dtype=np.float64
    )
    normal /= np.linalg.norm(normal) + 1e-8
    tmp = np.array([1, 0, 0], dtype=np.float64)
    if abs(np.dot(tmp, normal)) > 0.9:
        tmp = np.array([0, 1, 0], dtype=np.float64)
    u = np.cross(normal, tmp)
    u /= np.linalg.norm(u) + 1e-8
    v = np.cross(normal, u)
    v /= np.linalg.norm(v) + 1e-8
    return point, normal, u, v


def align_inplane_to_reference(normal, u_ref):
    normal = np.asarray(normal, dtype=np.float64)
    u_ref = np.asarray(u_ref, dtype=np.float64)
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    u = u_ref - np.dot(u_ref, normal) * normal
    nu = np.linalg.norm(u)
    if nu < 1e-6:
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, normal)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        u = np.cross(normal, tmp)
        u /= np.linalg.norm(u) + 1e-8
    else:
        u = u / nu
    v = np.cross(normal, u)
    v /= np.linalg.norm(v) + 1e-8
    return u, v


def extract_slice(ct, point, u, v, size=128):
    coords = np.linspace(-size // 2, size // 2, size)
    uu, vv = np.meshgrid(coords, coords)
    sample_points = (
        point.reshape(3, 1, 1) + u.reshape(3, 1, 1) * uu + v.reshape(3, 1, 1) * vv
    )
    sampled = map_coordinates(
        ct,
        [sample_points[0], sample_points[1], sample_points[2]],
        order=1,
        mode="nearest",
    )
    img = sampled - sampled.min()
    img = img / (img.max() + 1e-8)
    img = (img * 255).astype(np.uint8)
    return img


def visualize_planes(ct, gt_pose, pred_pose, save_path, plane_size=60):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    _, _, u_gt_ref, _ = pose_to_plane(gt_pose)

    def draw_plane(pose, color, label, align_u_ref=None):
        point, normal, u, v = pose_to_plane(pose)
        if align_u_ref is not None:
            u, v = align_inplane_to_reference(normal, align_u_ref)
        coords = np.linspace(-plane_size, plane_size, 20)
        uu, vv = np.meshgrid(coords, coords)
        plane = (
            point.reshape(3, 1, 1)
            + u.reshape(3, 1, 1) * uu
            + v.reshape(3, 1, 1) * vv
        )
        ax.plot_surface(plane[0], plane[1], plane[2], alpha=0.4, color=color)
        ax.scatter(point[0], point[1], point[2], color=color, s=40, label=label)

    draw_plane(gt_pose, "green", "GT")
    draw_plane(pred_pose, "red", "Pred", align_u_ref=u_gt_ref)

    z, y, x = ct.shape
    ax.set_xlim(0, z)
    ax.set_ylim(0, y)
    ax.set_zlim(0, x)
    ax.set_xlabel("Z")
    ax.set_ylabel("Y")
    ax.set_zlabel("X")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        type=str,
        default="/root/dof_project/model/vnet_6dof_ct+seg/best_model.pth",
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="/root/dof_project/3dct_point_dataset_DS4/data",
    )
    p.add_argument(
        "--label_root",
        type=str,
        default="/root/dof_project/3dct_point_dataset_DS4/ctpoint_label_voxel_revise_DS4",
    )
    p.add_argument(
        "--seg_root",
        type=str,
        default="/root/dof_project/results",
    )
    p.add_argument(
        "--save_dir",
        type=str,
        default="./test_resultsDS4_ct+seg",
    )
    p.add_argument("--seg_mode", type=str, default="binary", choices=["binary", "scaled"])
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    net = VNet6DOF(
        n_channels=2,
        n_filters=16,
        normalization="batchnorm",
    ).to(device)
    state = torch.load(args.model, map_location=device)
    net.load_state_dict(state)
    net.eval()
    print("✅ Model loaded (n_channels=2):", args.model)

    db_test = CTPoseSegDataset(
        data_root=args.data_root,
        label_root=args.label_root,
        seg_root=args.seg_root,
        split="test",
        seg_mode=args.seg_mode,
    )
    testloader = DataLoader(
        db_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    mi_list = []
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            volume = batch["image"].to(device)
            gt_pose = batch["pose"][0].cpu().numpy()

            pred_pose = net(volume)[0].cpu().numpy()

            pred_pose_real = pred_pose.copy()
            pred_pose_real[:3] *= 128.0
            pred_pose_real[3:] *= 180.0

            gt_pose_real = gt_pose.copy()
            gt_pose_real[:3] *= 128.0
            gt_pose_real[3:] *= 180.0

            ct = volume[0, 0].cpu().numpy()

            p, n, u, v = pose_to_plane(gt_pose_real)
            gt_img = extract_slice(ct, p, u, v)

            _, _, u_gt, _ = pose_to_plane(gt_pose_real)
            p_pred, n_pred, _, _ = pose_to_plane(pred_pose_real)
            u_p, v_p = align_inplane_to_reference(n_pred, u_gt)
            pred_img = extract_slice(ct, p_pred, u_p, v_p)

            case_id = db_test.ids[idx]
            case_dir = os.path.join(args.save_dir, str(case_id))
            os.makedirs(case_dir, exist_ok=True)

            imageio.imwrite(os.path.join(case_dir, "gt_pose.png"), gt_img)
            imageio.imwrite(os.path.join(case_dir, "pred_pose.png"), pred_img)

            pose_txt = os.path.join(case_dir, "pose_info.txt")
            trans_error = np.linalg.norm(gt_pose_real[:3] - pred_pose_real[:3])
            rot_error = np.linalg.norm(gt_pose_real[3:] - pred_pose_real[3:])

            with open(pose_txt, "w") as f:
                f.write("===== GT Pose =====\n")
                f.write(
                    "x y z rx ry rz:\n"
                    + " ".join([f"{v:.4f}" for v in gt_pose_real])
                    + "\n\n"
                )
                f.write("===== Pred Pose =====\n")
                f.write(
                    "x y z rx ry rz:\n"
                    + " ".join([f"{v:.4f}" for v in pred_pose_real])
                    + "\n\n"
                )
                f.write("===== Error =====\n")
                f.write(f"Translation L2 Error: {trans_error:.4f}\n")
                f.write(f"Rotation L2 Error: {rot_error:.4f}\n")
                f.write("\nPer-dimension difference:\n")
                diff = pred_pose_real - gt_pose_real
                f.write(" ".join([f"{v:.4f}" for v in diff]))

            visualize_planes(
                ct,
                gt_pose_real,
                pred_pose_real,
                os.path.join(case_dir, "plane_3d.png"),
            )

            mi = mutual_information(gt_img, pred_img)
            mi_list.append(mi)
            print(f"[{idx}] case={case_id} MI = {mi:.4f}")

    print("=================================")
    print("Mean MI:", float(np.mean(mi_list)))
    print("=================================")


if __name__ == "__main__":
    main()
