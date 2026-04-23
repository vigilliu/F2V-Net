from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nib
import numpy as np

nii = nib.load("/root/dof_project/3dct_point_dataset_DS4/data/1/cta/1ct.nii.gz")
data = nii.get_fdata()

data = np.clip(data, -200, 300)

# marching cubes
verts, faces, _, _ = measure.marching_cubes(data, level=100)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

mesh = Poly3DCollection(verts[faces])

# ⭐ 核心：灰色不透明
mesh.set_facecolor([0.3, 0.3, 0.3])
mesh.set_alpha(1.0)
mesh.set_edgecolor('none')

ax.add_collection3d(mesh)

# 统一比例（否则会拉伸）
ax.set_box_aspect(data.shape)

# 去坐标轴（论文必须）
ax.axis('off')

# 设置视角（更好看）
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig("ct_mesh.png", dpi=300)