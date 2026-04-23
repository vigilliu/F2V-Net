import torch
from torch import nn
import torch.nn.functional as F


# ===============================
# Basic Conv Block
# ===============================
class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super().__init__()

        ops = []
        for i in range(n_stages):

            input_channel = n_filters_in if i == 0 else n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))

            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(16, n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                raise ValueError("Unsupported normalization")

            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


# ===============================
# Downsampling Block
# ===============================
class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super().__init__()

        ops = [
            nn.Conv3d(
                n_filters_in,
                n_filters_out,
                kernel_size=stride,
                stride=stride,
                padding=0
            )
        ]

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(16, n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            raise ValueError("Unsupported normalization")

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


# ===============================
# VNet Encoder + 6DOF Head
# ===============================
class VNet6DOF(nn.Module):
    """
    Output:
        [B,6]  -> (Tx,Ty,Tz,Rx,Ry,Rz)
    """

    def __init__(
        self,
        n_channels=1,
        n_filters=16,
        normalization='none',
        has_dropout=False
    ):
        super().__init__()

        self.has_dropout = has_dropout

        # -------- Encoder --------
        self.block_one = ConvBlock(1, n_channels, n_filters, normalization)
        self.block_one_dw = DownsamplingConvBlock(
            n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(
            2, 2 * n_filters, 2 * n_filters, normalization)
        self.block_two_dw = DownsamplingConvBlock(
            2 * n_filters, 4 * n_filters, normalization=normalization)

        self.block_three = ConvBlock(
            3, 4 * n_filters, 4 * n_filters, normalization)
        self.block_three_dw = DownsamplingConvBlock(
            4 * n_filters, 8 * n_filters, normalization=normalization)

        self.block_four = ConvBlock(
            3, 8 * n_filters, 8 * n_filters, normalization)
        self.block_four_dw = DownsamplingConvBlock(
            8 * n_filters, 16 * n_filters, normalization=normalization)

        self.block_five = ConvBlock(
            3, 16 * n_filters, 16 * n_filters, normalization)

        self.dropout = nn.Dropout3d(p=0.5)

        # -------- 6DOF Regression Head --------
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.reg_head = nn.Sequential(
            nn.Linear(16 * n_filters, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6)
        )
        # translation head
        self.trans_head = nn.Sequential(
            nn.Linear(16 * n_filters, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )

        # rotation head
        self.rot_head = nn.Sequential(
            nn.Linear(16 * n_filters, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )

        self._init_weight()

    # ===============================
    # Encoder
    # ===============================
    def encoder(self, x):

        x1 = self.block_one(x)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        return x5

    # ===============================
    # Regression Head
    # ===============================
    # def regression_head(self, x):

    #     # Global pooling
    #     x = self.global_pool(x)        # [B,C,1,1,1]
    #     x = x.view(x.size(0), -1)      # [B,C]

    #     dof6 = self.reg_head(x)        # [B,6]

    #     return dof6
    def regression_head(self, x):

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        t = self.trans_head(x)
        r = self.rot_head(x)

        dof6 = torch.cat([t, r], dim=1)

        return dof6

    # ===============================
    # Forward
    # ===============================
    def forward(self, x, turnoff_drop=False):

        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False

        feat = self.encoder(x)
        out = self.regression_head(feat)

        if turnoff_drop:
            self.has_dropout = has_dropout

        return out

    # ===============================
    # Weight Init
    # ===============================
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# ===============================
# Test
# ===============================
if __name__ == "__main__":

    model = VNet6DOF(
        n_channels=1,
        n_filters=16,
        normalization='batchnorm'
    )

    x = torch.randn(2, 1, 112, 112, 80)

    y = model(x)

    print("Output shape:", y.shape)