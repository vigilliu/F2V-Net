import os
import numpy as np

# 原始数据路径
src_root = "/public/home/jiacheng.liu/f2v_reg/3dct_point_dataset/ctpoint_label_voxel"
# 新文件夹路径
dst_root = "/public/home/jiacheng.liu/f2v_reg/3dct_point_dataset/ctpoint_label_voxel_revise"

os.makedirs(dst_root, exist_ok=True)

case_folders = sorted(os.listdir(src_root))

for case in case_folders:
    src_case_path = os.path.join(src_root, case, "Points_voxel.txt")
    dst_case_folder = os.path.join(dst_root, case)
    dst_case_path = os.path.join(dst_case_folder, "Points_voxel.txt")
    
    if not os.path.exists(src_case_path):
        print(f"Warning: {src_case_path} 不存在！")
        continue

    os.makedirs(dst_case_folder, exist_ok=True)

    points = np.loadtxt(src_case_path)  # shape: (N,3)
    # 变换坐标
    points[:, 1] = 512 - points[:, 1]
    points[:, 2] = 320 - points[:, 2]

    # 保存到新文件
    np.savetxt(dst_case_path, points, fmt="%.6f")
    print(f"Saved revised points for case {case}")
