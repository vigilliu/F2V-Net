import os
import numpy as np

def compute_plane_properties(A, B, C):
    """
    输入:
        A, B, C: 三个点坐标 (list 或 np.array, 形状为 [3])
    输出:
        center: 三点中心
        normal: 法向量
        unit_normal: 单位法向量
        angles: 法向量与 x,y,z 轴夹角 (度)
    """
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)

    # 1. 中心点
    center = (A + B + C) / 3

    # 2. 法向量
    AB = B - A
    AC = C - A
    normal = np.cross(AB, AC)

    # 3. 单位法向量
    norm_length = np.linalg.norm(normal)
    unit_normal = normal / norm_length

    # 4. 夹角
    angles = np.arccos(np.clip(unit_normal, -1.0, 1.0)) * 180 / np.pi

    return center, normal, unit_normal, angles

# 数据路径
src_root = "/public/home/jiacheng.liu/f2v_reg/3dct_point_dataset/ctpoint_label_voxel_revise"

for case in sorted(os.listdir(src_root)):
    txt_path = os.path.join(src_root, case, "Points_voxel.txt")
    if not os.path.exists(txt_path):
        print(f"Warning: {txt_path} 不存在")
        continue

    points = np.loadtxt(txt_path)  # shape: (3,3)
    if points.shape != (3,3):
        print(f"Warning: {txt_path} 不是三点数据，跳过")
        continue

    A, B, C = points
    center, normal, unit_normal, angles = compute_plane_properties(A, B, C)

    # 合并结果，6DoF: cx, cy, cz, ax, ay, az
    dof = np.concatenate([center, angles])

    # 保存到 6DoF.txt
    dst_path = os.path.join(src_root, case, "6DoF.txt")
    np.savetxt(dst_path, dof[None,:], fmt="%.6f")  # 保存为一行
    print(f"Saved 6DoF for case {case}")
