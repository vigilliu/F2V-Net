import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import random

import nibabel as nib

def generate_gaussian_heatmap(center, shape, sigma=3):
    """
    center: (z, y, x)
    shape: (D, H, W)
    """
    x, y, z = center
    H, W, D = shape

    xx, yy, zz = np.meshgrid(
        np.arange(H), np.arange(W), np.arange(D), indexing='ij'
    )

    heatmap = np.exp(
        -((xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2) / (2 * sigma ** 2)
    )
    return heatmap.astype(np.float32)

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib


class CTPoseDataset(Dataset):

    def __init__(self, data_root, label_root, split='train'):

        self.data_root = data_root
        self.label_root = label_root

        self.ids = sorted(os.listdir(data_root))

        if split == 'train':
            self.ids = self.ids[:int(0.8 * len(self.ids))]
        else:
            self.ids = self.ids[int(0.8 * len(self.ids)):]

    def __len__(self):
        return len(self.ids)

    
    def __getitem__(self, idx):

        case_id = self.ids[idx]

        # ========= load CT =========
        ct_path = os.path.join(
            self.data_root,
            case_id,
            "cta",
            f"{case_id}ct.nii.gz"
        )

        ct = nib.load(ct_path).get_fdata().astype(np.float32)

        # window
        ct = np.clip(ct, 0, 2000)

        # normalize to [-1,1]
        ct = (ct - ct.min()) / (ct.max() - ct.min() + 1e-8)
        ct = ct * 2 - 1

        # add channel
        ct = ct[None]  # [1, D, H, W]

        # ========= load pose =========
        pose_path = os.path.join(
            self.label_root,
            case_id,
            "6DoF.txt"
        )

        pose = np.genfromtxt(pose_path).astype(np.float32)
        pose = pose.reshape(6)

        # ========= pose normalization =========
        # translation (假设CT尺寸≈512)
        pose[0:3] = pose[0:3] / 128.0

        # rotation (degree → [-1,1])
        pose[3:6] = pose[3:6] / 180.0

        return {
            "image": torch.from_numpy(ct),
            "pose": torch.from_numpy(pose)
        }

class CTPointDataset(Dataset):
    def __init__(self, data_root, label_root, split='train'):
        self.data_root = data_root
        self.label_root = label_root

        self.ids = sorted(os.listdir(data_root))
        if split == 'train':
            self.ids = self.ids[:int(0.8 * len(self.ids))]
        else:
            self.ids = self.ids[int(0.8 * len(self.ids)):]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        case_id = self.ids[idx]
        # === load CT ===
        ct_path = os.path.join(self.data_root, case_id, 'cta', f'{case_id}ct.nii.gz')
        ct = nib.load(ct_path).get_fdata().astype(np.float32)
        print('ct.shape:',ct.shape)

        ct = np.clip(ct, 0, 2000)  # CTA 常用窗宽
        # ct = (ct - ct.mean()) / (ct.std() + 1e-5)
        ct = (ct - ct.min()) / (ct.max() - ct.min())
        ct = ct * 2 - 1
        ct = ct[None]  # [1, D, H, W]
        print('ct.shape:',ct.shape)

        # === load points ===
        point_path = os.path.join(self.label_root, case_id, 'Points_voxel.txt')
        print('point_path:',point_path)

        points = np.genfromtxt(point_path, dtype=np.float32)#split by space, not delimiter=','

        print('point:',points)
        # if points.shape != (3, 3):
        #     raise ValueError(f"Invalid point shape {points.shape} in {point_path}")  # [3, 3]

        H, W, D = ct.shape[1:]
        print('H, W, D', H, W, D)
        heatmaps = np.zeros((3, H, W, D), dtype=np.float32)

        for i in range(3):
            x, y, z = points[i]
            print(x, y, z)
            
            x_idx = x      # 如果y是H方向坐标
            y_idx = W - int(y)      # x是W方向坐标
            z_idx = D - int(z)      # z是D方向坐标
            print('x_idx, y_idx, z_idx:', x_idx, y_idx, z_idx)

            heatmaps[i] = generate_gaussian_heatmap(
                center=(int(x_idx), int(y_idx), int(z_idx)),
                shape=(H, W, D),
                sigma=8
            )   
            #512-W,320-D
            heatmaps[i] = (heatmaps[i] > 0.5).astype(np.uint8)  # 0 或 1



        return {
            'image': torch.from_numpy(ct),
            'heatmap': torch.from_numpy(heatmaps)
        }

if __name__ == '__main__':
    data_root = '/public/home/jiacheng.liu/f2v_reg/3dct_point_dataset/data'    # 这里改成你的 CTA 数据路径
    label_root = '/public/home/jiacheng.liu/f2v_reg/3dct_point_dataset/ctpoint_label_voxel'  # 这里改成你的 Points_voxel.txt 所在路径

    dataset = CTPointDataset(data_root, label_root, split='train')


    print("dataset length:", len(dataset))

    from torch.utils.data import DataLoader

    # 用 DataLoader 测试 batch
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in loader:
        image = batch['image']       # [B, 1, D, H, W]
        heatmap = batch['heatmap']   # [B, 3, D, H, W]
        print('image shape:', image.shape)
        print('heatmap shape:', heatmap.shape)
        print('image min/max:', image.min().item(), image.max().item())
        print('heatmap min/max:', heatmap.min().item(), heatmap.max().item())
        break  # 只看第一个 case
