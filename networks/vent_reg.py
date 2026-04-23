import torch
from torch import nn
import torch.nn.functional as F
from .STN import RegistrationHead as RegHead,RegistrationHeadX3,SimpleRigidShiftWithConv,SimpleRigidAffineField
from .vnet import VNetFeatureExtractor



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
    

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

# class VNetRegistration(nn.Module):
#     def __init__(self, vnet1, vnet2, reg_in_level='x5'):
#         super(VNetRegistration, self).__init__()
#         self.vnet1 = vnet1
#         self.vnet2 = vnet2

#         # 你可以选择在哪个层级进行拼接，例如：x1~x5
#         self.reg_in_level = reg_in_level

#         level_channels = {
#             'x1': 16,
#             'x2': 32,
#             'x3': 64,
#             'x4': 128,
#             'x5': 256
#         }
#         in_ch = level_channels[self.reg_in_level] * 2  # 拼接后的通道数
#         self.registration_head = RegistrationHead(in_channels=in_ch)

#     def forward(self, input1, input2):
#         feat1 = self.vnet1.encoder(input1)  # [x1, x2, x3, x4, x5]
#         feat2 = self.vnet2.encoder(input2)

#         # 选择要拼接的层
#         if self.reg_in_level == 'x1':
#             x = torch.cat([feat1[0], feat2[0]], dim=1)
#         elif self.reg_in_level == 'x2':
#             x = torch.cat([feat1[1], feat2[1]], dim=1)
#         elif self.reg_in_level == 'x3':
#             x = torch.cat([feat1[2], feat2[2]], dim=1)
#         elif self.reg_in_level == 'x4':
#             x = torch.cat([feat1[3], feat2[3]], dim=1)
#         elif self.reg_in_level == 'x5':
#             x = torch.cat([feat1[4], feat2[4]], dim=1)
#         else:
#             raise ValueError("Unsupported reg_in_level")

#         dvf = self.registration_head(x)
#         return dvf

# class RigidTransform3D(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 每个 block 都有自己的一组旋转+平移参数
#         self.rotation = nn.Parameter(torch.zeros(3))  # θx, θy, θz
#         self.translation = nn.Parameter(torch.zeros(3))  # tx, ty, tz

#     def forward(self, x):
#         B, C, D, H, W = x.shape

#         # 旋转矩阵
#         rx, ry, rz = self.rotation
#         cos = torch.cos; sin = torch.sin
#         Rx = torch.tensor([[1, 0, 0], [0, cos(rx), -sin(rx)], [0, sin(rx), cos(rx)]], device=x.device)
#         Ry = torch.tensor([[cos(ry), 0, sin(ry)], [0, 1, 0], [-sin(ry), 0, cos(ry)]], device=x.device)
#         Rz = torch.tensor([[cos(rz), -sin(rz), 0], [sin(rz), cos(rz), 0], [0, 0, 1]], device=x.device)
#         R = Rz @ Ry @ Rx  # (3, 3)

#         # 仿射矩阵
#         affine = torch.eye(4, device=x.device)
#         affine[:3, :3] = R
#         affine[:3, 3] = self.translation
#         affine = affine[:3, :].unsqueeze(0).repeat(B, 1, 1)  # (B, 3, 4)

#         grid = F.affine_grid(affine, x.size(), align_corners=True)
#         warped = F.grid_sample(x, grid, align_corners=True, mode='bilinear', padding_mode='border')
#         return warped
    

class DualVNetDecoder(nn.Module):
    def __init__(self, n_filters=16, n_classes=2, normalization='none', has_dropout=False):
        super(DualVNetDecoder, self).__init__()
        self.has_dropout = has_dropout

        self.dropout = nn.Dropout3d(p=0.1, inplace=False)

        # 创建用于图像配准的ResNet模型
        self.regvnet1 = VNetFeatureExtractor()  # 用于第一个模态
        self.regvnet2 = VNetFeatureExtractor()  # 用于第二个模态

        # 加入多个刚性模块，分别对 feat2 的浅层做对齐
        # self.rigid0 = RigidTransform3D()
        # self.rigid1 = RigidTransform3D()
        # self.rigid2 = RigidTransform3D()
        # self.rigid3 = RigidTransform3D()



        self.block_five_up = UpsamplingDeconvBlock(2 * n_filters * 16, n_filters * 8, normalization=normalization)
        self.block_six = ConvBlock(3, n_filters * 8 + 2 * n_filters * 8, n_filters * 8, normalization=normalization)# 128 + 256 = 384
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4 + 2 * n_filters * 4, n_filters * 4, normalization=normalization)# 64 + 128 = 192
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2 + 2 * n_filters * 2, n_filters * 2, normalization=normalization)#32+64=96
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters + 2 * n_filters, n_filters, normalization=normalization)#16+32=48
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        #self.out_reg = RegHead(n_filters, 3)
        # self.out_reg = RegistrationHeadX3(n_filters, 3)             # 非刚性 flow
        self.shift_predictor = SimpleRigidAffineField(n_filters)          # 刚性 shift flow
        #这里换了
        # self.stn = SpatialTransformer(img_size)

    def forward(self, feat1, feat2):
        # 每层特征进行channel维拼接

        # feat2 = [
        #     self.rigid0(feat2[0]),  # torch.Size([B, 16, 128, 128, 128])
        #     self.rigid1(feat2[1]),  # torch.Size([B, 32, 64, 64, 64])
        #     self.rigid2(feat2[2]),  # torch.Size([B, 64, 32, 32, 32])
        #     self.rigid3(feat2[3]),  # torch.Size([B, 128, 16, 16, 16])
        #     feat2[4]                # 深层特征不刚性处理
        # ]

        # 将原始输入图像送入ResNet以提取特征
        feat1_regvnet1 = self.regvnet1.encoder(feat1[0])  # 使用ResNet提取第一个模态的特征
        feat2_regvnet2 = self.regvnet2.encoder(feat2[0])  # 使用ResNet提取第二个模态的特征



        x1 = torch.cat([feat1[0], feat2[0]], dim=1)

        x2 = torch.cat([feat1[1], feat2[1]], dim=1)

        x3 = torch.cat([feat1[2], feat2[2]], dim=1)

        x4 = torch.cat([feat1[3], feat2[3]], dim=1)

        x5 = torch.cat([feat1_regvnet1, feat2_regvnet2], dim=1)  # 使用配准后的特征替代原x5

        #x5 = torch.cat([feat1[4], feat2[4]], dim=1)       (--batchsize=4)

        # torch.Size([4, 32, 128, 128, 128])
        # torch.Size([4, 64, 64, 64, 64])
        # torch.Size([4, 128, 32, 32, 32])
        # torch.Size([4, 256, 16, 16, 16])
        # torch.Size([4, 512, 8, 8, 8])

        x5_up = self.block_five_up(x5)
        print(x5_up.size(),'x5_up')
        #.Size([4, 128, 16, 16, 16])
        x5_up = torch.cat([x5_up, x4], dim=1)#.Size([4, 384, 16, 16, 16])


        x6 = self.block_six(x5_up)

        x6_up = self.block_six_up(x6)
        x6_up = torch.cat([x6_up, x3], dim=1)

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = torch.cat([x7_up, x2], dim=1)

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = torch.cat([x8_up, x1], dim=1)

        x9 = self.block_nine(x8_up)

        if self.has_dropout:
            x9 = self.dropout(x9)

        # nonrigid_flow = self.out_reg(x9)  # 非刚性 flow
        flow = self.shift_predictor(x9) #shape=x9.shape)  # 刚性平移 flow


        flow = flow #+ nonrigid_flow  # 总flow

        
        #torch.Size([4, 3, 128, 128, 128])
        # warped = self.stn(src, flow)  # 🔵 利用 flow 变换原图

        return flow , feat1 , feat2

class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec
    
if __name__ == "__main__":
    net = ResNetRegNet()
    print('download')