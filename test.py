import os
import numpy as np
import nibabel as nib
import torch
import imageio

from scipy.ndimage import map_coordinates
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader

from networks.Vnet_6DoF import VNet6DOF
from dataset_ct_point import CTPoseDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# =========================
# 参数
# =========================
MODEL_PATH = "/root/dof_project/model/vnet_3points_BS8_0.001lr_1rotloss_data_right/best_model.pth"
DATA_ROOT = "/root/dof_project/3dct_point_dataset_DS4/"
SAVE_DIR = "./test_resultsDS4_cursor_vnet_6dof_right"

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# Mutual Information
# =========================
def mutual_information(img1, img2, bins=64):

    img1 = img1.ravel()
    img2 = img2.ravel()

    # joint histogram
    hist_2d, _, _ = np.histogram2d(
        img1,
        img2,
        bins=bins
    )

    # probability
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    px_py = px[:, None] * py[None, :]

    nz = pxy > 0

    mi = np.sum(
        pxy[nz] * np.log(pxy[nz] / px_py[nz])
    )

    return mi


# =========================
# pose -> normal
# =========================
def pose_to_plane(pose):
    point = pose[:3]
    angles = pose[3:]

    rad = np.deg2rad(angles)

    normal = np.array([
        np.cos(rad[0]),
        np.cos(rad[1]),
        np.cos(rad[2])
    ])

    normal /= np.linalg.norm(normal)

    # 构造平面基
    tmp = np.array([1, 0, 0])
    if abs(np.dot(tmp, normal)) > 0.9:
        tmp = np.array([0, 1, 0])

    u = np.cross(normal, tmp)
    u /= np.linalg.norm(u)

    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    return point, normal, u, v


def align_inplane_to_reference(normal, u_ref):
    """
    将参考切向量 u_ref 投影到法向为 normal 的平面上并单位化，使 Pred 切片面内方向与 GT 一致，
    避免同一平面因「选不同正交基」在 2D 上看起来旋转/倾斜。
    """
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


# =========================
# 提取切面
# =========================
def extract_slice(ct, point, u, v, size=128):

    coords = np.linspace(-size//2, size//2, size)
    uu, vv = np.meshgrid(coords, coords)

    sample_points = (
        point.reshape(3,1,1)
        + u.reshape(3,1,1)*uu
        + v.reshape(3,1,1)*vv
    )

    sampled = map_coordinates(
        ct,
        [
            sample_points[0],
            sample_points[1],
            sample_points[2]
        ],
        order=1,
        mode='nearest'
    )

    img = sampled - sampled.min()
    img = img / (img.max() + 1e-8)
    img = (img * 255).astype(np.uint8)

    return img

def visualize_planes(ct, gt_pose, pred_pose, save_path, plane_size=60):

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    _, _, u_gt_ref, _ = pose_to_plane(gt_pose)

    def draw_plane(pose, color, label, align_u_ref=None):

        point, normal, u, v = pose_to_plane(pose)
        if align_u_ref is not None:
            u, v = align_inplane_to_reference(normal, align_u_ref)

        coords = np.linspace(-plane_size, plane_size, 20)
        uu, vv = np.meshgrid(coords, coords)

        plane = (
            point.reshape(3,1,1)
            + u.reshape(3,1,1)*uu
            + v.reshape(3,1,1)*vv
        )

        ax.plot_surface(
            plane[0],
            plane[1],
            plane[2],
            alpha=0.4,
            color=color
        )

        ax.scatter(
            point[0], point[1], point[2],
            color=color,
            s=40,
            label=label
        )

    # GT = green
    draw_plane(gt_pose, 'green', 'GT')

    # Pred = red（面内基与 GT 对齐，便于与绿面对比）
    draw_plane(pred_pose, 'red', 'Pred', align_u_ref=u_gt_ref)

    # volume bounding box
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


# =========================
# 加载模型
# =========================
net = VNet6DOF(
    n_channels=1,
    n_filters=16,
    normalization='batchnorm'
).cpu()

net.load_state_dict(torch.load(MODEL_PATH))
net.eval()

print("✅ Model loaded")


# =========================
# Dataset
# =========================
db_test = CTPoseDataset(
    data_root=DATA_ROOT + "/data",
    label_root=DATA_ROOT + "/ctpoint_label_voxel_right_DS4",
    split='test'
)

testloader = DataLoader(
    db_test,
    batch_size=1,
    shuffle=False,
    num_workers=2
)

db_train = CTPoseDataset(
    data_root=DATA_ROOT + "/data",
    label_root=DATA_ROOT + "/ctpoint_label_voxel_right",
    split='train'
)

trainloader = DataLoader(
    db_train,
    batch_size=1,
    shuffle=False,
    num_workers=2
)

# =========================
# 测试循环
# =========================
mi_list = []

with torch.no_grad():

    for idx, batch in enumerate(testloader):

        volume = batch['image'].cpu()
        gt_pose = batch['pose'][0].numpy()

        # -------- predict --------
        pred_pose = net(volume)[0].cpu().numpy()

        # ===== 反归一化（与你训练一致）=====
        pred_pose_real = pred_pose.copy()
        pred_pose_real[:3] *= 128.0
        pred_pose_real[3:] *= 180.0

        gt_pose_real = gt_pose.copy()
        gt_pose_real[:3] *= 128.0
        gt_pose_real[3:] *= 180.0

        # -------- CT volume --------
        ct = volume[0,0].cpu().numpy()

        # =========================
        # GT slice
        # =========================
        p, n, u, v = pose_to_plane(gt_pose_real)
        gt_img = extract_slice(ct, p, u, v)

        # =========================
        # Pred slice（面内 u,v 对齐到 GT，避免法向略有差异时 2D 切片面内旋转）
        # =========================
        _, _, u_gt, _ = pose_to_plane(gt_pose_real)
        p_pred, n_pred, _, _ = pose_to_plane(pred_pose_real)
        u_p, v_p = align_inplane_to_reference(n_pred, u_gt)
        pred_img = extract_slice(ct, p_pred, u_p, v_p)

        # =========================
        # 保存
        # =========================
        case_dir = os.path.join(SAVE_DIR, f"case_{idx:03d}")
        os.makedirs(case_dir, exist_ok=True)

        imageio.imwrite(os.path.join(case_dir, "gt_pose.png"), gt_img)
        imageio.imwrite(os.path.join(case_dir, "pred_pose.png"), pred_img)
        # =========================
        # 保存 pose 信息
        # =========================
        pose_txt = os.path.join(case_dir, "pose_info.txt")

        trans_error = np.linalg.norm(
            gt_pose_real[:3] - pred_pose_real[:3]
        )

        rot_error = np.linalg.norm(
            gt_pose_real[3:] - pred_pose_real[3:]
        )

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
            f.write(
                " ".join([f"{v:.4f}" for v in diff])
            )
            
        visualize_planes(
            ct,
            gt_pose_real,
            pred_pose_real,
            os.path.join(case_dir, "plane_3d.png")
        )

        # =========================
        # SSIM
        # =========================
        mi = mutual_information(gt_img, pred_img)
        mi_list.append(mi)

        print(f"[{idx}] MI  = {mi:.4f}")

# =========================
# 平均SSIM
# =========================
print("=================================")
print("Mean MI:", np.mean(mi_list))
print("=================================")