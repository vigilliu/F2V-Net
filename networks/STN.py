import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]# size fixed img[D, W, H]
        #vectors = [[D],[W],[H]]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode)

    
class RegistrationHead(nn.Sequential):
    #形变场
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)

class RegistrationHeadX3(nn.Module):
    def __init__(self, in_channels, out_channels=3, kernel_size=3):
        super(RegistrationHeadX3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.InstanceNorm3d(in_channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv3d(in_channels // 2, in_channels // 4, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.InstanceNorm3d(in_channels // 4),
            nn.ReLU(inplace=True),

            nn.Conv3d(in_channels // 4, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        )

        # 初始化最后一层权重为较小值
        nn.init.normal_(self.conv[-1].weight, mean=0.0, std=1e-5)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, x):
        return self.conv(x)

class SimpleRigidShift(nn.Module):
    """ 只输出3个数：dx, dy, dz """
    def __init__(self, in_channels):
        super(SimpleRigidShift, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3)  # dx, dy, dz
        )

    def forward(self, x):
        x = self.global_pool(x).view(x.size(0), -1)
        shift = self.fc(x)  # [B, 3]
        return shift
    
class SimpleRigidShift(nn.Module):
    def __init__(self, in_channels):
        super(SimpleRigidShift, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3)  # dx, dy, dz
        )

    def forward(self, x, shape):
        x = self.pool(x).view(x.size(0), -1)  # [B, C]
        shift = self.fc(x)                    # [B, 3]

        # 构造 dense flow（广播到每个 voxel）
        B, _, D, H, W = shape
        flow = shift.view(B, 3, 1, 1, 1).expand(B, 3, D, H, W)
        return flow

class SimpleRigidShiftWithConv(nn.Module):
    def __init__(self, in_channels, max_shift=128.0):
        super(SimpleRigidShiftWithConv, self).__init__()
        self.max_shift = max_shift
        
        # 卷积层
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),  # 输出3个值，表示仿射变换的偏移量
            nn.Tanh()  # 输出限制在 [-1, 1]
        )

    def forward(self, x, shape):
        # 卷积层提取特征
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # 池化层得到全局特征
        x = self.pool(x).view(x.size(0), -1)  # [B, C]
        
        # 通过全连接层计算仿射变换
        shift = self.fc(x) * self.max_shift   # 缩放偏移值
        
        # 获取输入的形状信息
        B, _, D, H, W = shape
        
        # 扩展到与输入图像相同的大小
        flow = shift.view(B, 3, 1, 1, 1).expand(B, 3, D, H, W)
        
        return flow


class SimpleRigidAffineField(nn.Module):
    def __init__(self, in_channels, max_shift=128.0, grid_size=(128, 128, 128)):
        super(SimpleRigidAffineField, self).__init__()
        self.max_shift = max_shift
        self.grid_size = grid_size  # (D, H, W)

        # 卷积层
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d(1)

        # 输出 6 个仿射参数（3 平移 + 3 旋转）
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6),   # 3 平移 + 3 旋转
            nn.Tanh()
        )

    def forward(self, x):
        B = x.size(0)
        D, H, W = self.grid_size

        # 提取全局特征
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).view(B, -1)  # [B, 128]

        # 输出平移和旋转
        params = self.fc(x)  # [B, 6]
        shift = params[:, :3] * self.max_shift        # 平移，单位为体素
        angles = params[:, 3:] * 3.1416               # 旋转，单位为弧度（限制在 ±π）

        # 生成标准网格（单位坐标，中心在 0）
        grid = self.create_grid(B, D, H, W, x.device)  # [B, D, H, W, 3]
        grid_flat = grid.view(B, -1, 3)                # [B, N, 3]

        # 构建旋转矩阵并应用
        rot_mat = self.compute_rotation_matrix(angles)    # [B, 3, 3]
        rotated = torch.bmm(grid_flat, rot_mat.transpose(1, 2))  # [B, N, 3]
        rotated = rotated.view(B, D, H, W, 3)

        # 平移偏移加入
        shift = shift.view(B, 1, 1, 1, 3)
        transformed = rotated + shift  # [B, D, H, W, 3]

        # 位移 = 变换后坐标 - 原始坐标
        flow = (transformed - grid).permute(0, 4, 1, 2, 3).contiguous()  # [B, 3, D, H, W]
        return flow  # 形变场（仿射 flow）

    def create_grid(self, B, D, H, W, device):
        """创建标准体素网格"""
        z = torch.linspace(-1, 1, D, device=device)
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        zz, yy, xx = torch.meshgrid(z, y, x)
        grid = torch.stack((xx, yy, zz), dim=-1)  # [D, H, W, 3]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1)  # [B, D, H, W, 3]
        return grid

    def compute_rotation_matrix(self, angles):
        """
        将欧拉角转换为旋转矩阵
        angles: [B, 3] -> (rx, ry, rz)
        return: [B, 3, 3]
        """
        rx, ry, rz = angles[:, 0], angles[:, 1], angles[:, 2]

        cosx, sinx = torch.cos(rx), torch.sin(rx)
        cosy, siny = torch.cos(ry), torch.sin(ry)
        cosz, sinz = torch.cos(rz), torch.sin(rz)

        # 每个旋转矩阵独立构建后 batch 合成
        B = angles.shape[0]
        rot_x = torch.zeros(B, 3, 3, device=angles.device)
        rot_y = torch.zeros(B, 3, 3, device=angles.device)
        rot_z = torch.zeros(B, 3, 3, device=angles.device)

        rot_x[:, 0, 0] = 1
        rot_x[:, 1, 1] = cosx
        rot_x[:, 1, 2] = -sinx
        rot_x[:, 2, 1] = sinx
        rot_x[:, 2, 2] = cosx

        rot_y[:, 0, 0] = cosy
        rot_y[:, 0, 2] = siny
        rot_y[:, 1, 1] = 1
        rot_y[:, 2, 0] = -siny
        rot_y[:, 2, 2] = cosy

        rot_z[:, 0, 0] = cosz
        rot_z[:, 0, 1] = -sinz
        rot_z[:, 1, 0] = sinz
        rot_z[:, 1, 1] = cosz
        rot_z[:, 2, 2] = 1

        return torch.bmm(rot_z, torch.bmm(rot_y, rot_x))  # Z * Y * X




