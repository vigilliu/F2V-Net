import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization="none"):
        super().__init__()
        ops = []
        for i in range(n_stages):
            in_ch = n_filters_in if i == 0 else n_filters_out
            ops.append(nn.Conv3d(in_ch, n_filters_out, kernel_size=3, padding=1))
            if normalization == "batchnorm":
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == "groupnorm":
                ops.append(nn.GroupNorm(16, n_filters_out))
            elif normalization == "instancenorm":
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != "none":
                raise ValueError("Unsupported normalization")
            ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class DownsamplingConvBlock3D(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization="none"):
        super().__init__()
        ops = [
            nn.Conv3d(
                n_filters_in,
                n_filters_out,
                kernel_size=stride,
                stride=stride,
                padding=0,
            )
        ]
        if normalization == "batchnorm":
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == "groupnorm":
            ops.append(nn.GroupNorm(16, n_filters_out))
        elif normalization == "instancenorm":
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != "none":
            raise ValueError("Unsupported normalization")
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class TeeEncoder2D(nn.Module):
    """
    Lightweight 2D encoder for TEE.
    Input: [B, 1 or 3, H, W]
    Output: [B, feat_dim]
    """

    def __init__(self, in_channels=1, base_channels=16, feat_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        return self.fc(x)


class VNet6DOFLateFusion(nn.Module):
    """
    Late fusion model:
      - 3D CT branch: VNet-style encoder
      - 2D TEE branch: lightweight CNN encoder
      - Fusion head: concatenate [ct_feat, tee_feat] for 6DoF regression

    Forward input:
      ct  : [B, 1, D, H, W]
      tee : [B, C, H, W], C in {1,3}
    """

    def __init__(
        self,
        ct_in_channels=1,
        tee_in_channels=1,
        n_filters=16,
        normalization="batchnorm",
        tee_feat_dim=128,
        fusion_hidden=128,
        has_dropout=False,
    ):
        super().__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock3D(1, ct_in_channels, n_filters, normalization)
        self.block_one_dw = DownsamplingConvBlock3D(n_filters, 2 * n_filters, normalization=normalization)
        self.block_two = ConvBlock3D(2, 2 * n_filters, 2 * n_filters, normalization)
        self.block_two_dw = DownsamplingConvBlock3D(2 * n_filters, 4 * n_filters, normalization=normalization)
        self.block_three = ConvBlock3D(3, 4 * n_filters, 4 * n_filters, normalization)
        self.block_three_dw = DownsamplingConvBlock3D(4 * n_filters, 8 * n_filters, normalization=normalization)
        self.block_four = ConvBlock3D(3, 8 * n_filters, 8 * n_filters, normalization)
        self.block_four_dw = DownsamplingConvBlock3D(8 * n_filters, 16 * n_filters, normalization=normalization)
        self.block_five = ConvBlock3D(3, 16 * n_filters, 16 * n_filters, normalization)
        self.dropout3d = nn.Dropout3d(p=0.5)
        self.global_pool3d = nn.AdaptiveAvgPool3d(1)

        self.tee_encoder = TeeEncoder2D(
            in_channels=tee_in_channels,
            base_channels=16,
            feat_dim=tee_feat_dim,
        )
        self.dropout_fuse = nn.Dropout(p=0.3)

        ct_feat_dim = 16 * n_filters
        fusion_dim = ct_feat_dim + tee_feat_dim
        self.trans_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden, 3),
        )
        self.rot_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden, 3),
        )

        self._init_weight()

    def encode_ct(self, ct):
        x1 = self.block_one(ct)
        x1_dw = self.block_one_dw(x1)
        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)
        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)
        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)
        x5 = self.block_five(x4_dw)
        if self.has_dropout:
            x5 = self.dropout3d(x5)
        x5 = self.global_pool3d(x5).flatten(1)
        return x5

    def forward(self, ct, tee, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False

        ct_feat = self.encode_ct(ct)
        tee_feat = self.tee_encoder(tee)
        fused = torch.cat([ct_feat, tee_feat], dim=1)
        fused = self.dropout_fuse(fused)

        t = self.trans_head(fused)
        r = self.rot_head(fused)
        out = torch.cat([t, r], dim=1)

        if turnoff_drop:
            self.has_dropout = has_dropout
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


if __name__ == "__main__":
    model = VNet6DOFLateFusion(
        ct_in_channels=1,
        tee_in_channels=1,
        n_filters=16,
        normalization="batchnorm",
    )
    ct = torch.randn(2, 1, 128, 128, 80)
    tee = torch.randn(2, 1, 256, 256)
    y = model(ct, tee)
    print("Output shape:", y.shape)
