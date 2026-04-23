import nibabel as nib
import numpy as np


def load_points_txt(txt_path):
    points = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            x, y, z = map(float, line.split(","))
            points.append([x, y, z])
    return np.array(points)


def world_to_voxel(points_world, affine):
    affine_inv = np.linalg.inv(affine)
    ones = np.ones((points_world.shape[0], 1))
    points_h = np.concatenate([points_world, ones], axis=1)
    points_voxel = (affine_inv @ points_h.T).T[:, :3]
    return np.round(points_voxel).astype(int)


def highlight_points(
    nii_path,
    points_txt,
    save_path,
    radius=2,
    value=3000
):
    nii = nib.load(nii_path)
    print(nii.affine)
    print(nii.header)
    data = nii.get_fdata()
    affine = nii.affine

    points_world = load_points_txt(points_txt)
    points_voxel = world_to_voxel(points_world, affine)

    print("World coordinates (mm):")
    print(points_world)
    print("Voxel coordinates:")
    print(points_voxel)

    data_highlight = data.copy()
    D, H, W = data.shape

    for (x, y, z) in points_voxel:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    xx, yy, zz = x + dx, y + dy, z + dz
                    if 0 <= xx < D and 0 <= yy < H and 0 <= zz < W:
                        data_highlight[xx, yy, zz] = value

    new_nii = nib.Nifti1Image(data_highlight, affine, nii.header)
    nib.save(new_nii, save_path)

    print(f"Saved to: {save_path}")

highlight_points(
    nii_path="/public/home/jiacheng.liu/f2v_reg/3dct_point_dataset/data/1/cta/1ct.nii.gz",
    points_txt="/public/home/jiacheng.liu/f2v_reg/3dct_point_dataset/ctpoint_label/1/Points.txt",
    save_path="cta_with_points.nii.gz",
    radius=2,
    value=3000
)

