import os
import numpy as np

POINT_ROOT = "/public/home/jiacheng.liu/f2v_reg/3dct_point_dataset/ctpoint_label"
OUT_ROOT   = "/public/home/jiacheng.liu/f2v_reg/3dct_point_dataset/ctpoint_label_voxel"

# CTA 尺寸
CTA_VOXEL_SIZE = np.array([512, 512, 320])
CTA_WORLD_SIZE = np.array([234.541, 234.541, 159.5])

SPACING = CTA_WORLD_SIZE / CTA_VOXEL_SIZE
print("CTA spacing (mm):", SPACING)

os.makedirs(OUT_ROOT, exist_ok=True)

for case_id in sorted(os.listdir(POINT_ROOT)):
    case_dir = os.path.join(POINT_ROOT, case_id)
    if not os.path.isdir(case_dir):
        continue

    in_file = os.path.join(case_dir, "Points.txt")
    if not os.path.exists(in_file):
        in_file = os.path.join(case_dir, "point.txt")
        if not os.path.exists(in_file):
            print(f"[WARN] {case_id} 没有 Points/point.txt")
            continue

    # 读取真实世界坐标 (mm)
    points_world = np.loadtxt(in_file, delimiter=",")
    points_world = np.atleast_2d(points_world)

    # world -> voxel
    points_voxel = points_world / SPACING

    # （可选）限制在体素范围内
    points_voxel[:, 0] = np.clip(points_voxel[:, 0], 0, 511)
    points_voxel[:, 1] = np.clip(points_voxel[:, 1], 0, 511)
    points_voxel[:, 2] = np.clip(points_voxel[:, 2], 0, 319)

    # 保存
    out_case_dir = os.path.join(OUT_ROOT, case_id)
    os.makedirs(out_case_dir, exist_ok=True)

    out_file = os.path.join(out_case_dir, "Points_voxel.txt")
    np.savetxt(out_file, points_voxel, fmt="%.6f")

    print(f"[OK] {case_id}: {points_voxel.shape[0]} points converted")

print("\n🎯 所有 Points.txt 已从 world 坐标转换为 voxel 坐标")
