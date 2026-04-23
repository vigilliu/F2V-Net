#!/usr/bin/env python3

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch.nn.functional as F
import torch
import torch.nn as nn
from timm.models.registry import register_model
import math
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models._builder import resolve_pretrained_cfg
try:
    from timm.models._builder import _update_default_kwargs as update_args
except:
    from timm.models._builder import _update_default_model_kwargs as update_args
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from .registry import register_pip_model
from pathlib import Path
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numbers
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
#==============
import os
import torch
import torchvision.utils as vutils
from torchvision import transforms

from torchvision.utils import save_image

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def save_colormap(tensors, save_dir, prefix_list=["p4", "p3", "p2", "fuse"], cmap='bwr'):
    """
    tensors: list of torch.Tensor, shape [B, C, H, W], assume single-channel
    save_dir: 保存路径
    prefix_list: 每个 tensor 的前缀名
    cmap: matplotlib colormap, 蓝黄可以用 'bwr' 或 'coolwarm'
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for t, prefix in zip(tensors, prefix_list):
        t = t.detach().cpu()
        B, C, H, W = t.shape
        for i in range(B):
            # 只取第一个通道
            img = t[i, 0].numpy()
            # 归一化到 [0,1]
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
            # 使用 colormap
            cmap_img = plt.get_cmap(cmap)(img_norm)[:, :, :3]  # 去掉 alpha
            # 转成 0-255
            cmap_img = (cmap_img * 255).astype(np.uint8)
            plt.imsave(os.path.join(save_dir, f"{prefix}_b{i}.png"), cmap_img)


def save_tensor_as_jpg(tensor, save_path, prefix):
    """
    将 [B, C, H, W] 张量保存为 JPG
    如果是单通道，会复制成 3 通道以便保存为彩色 JPG
    """
    os.makedirs(save_path, exist_ok=True)

    B, C, H, W = tensor.shape

    # 如果是单通道，复制为 3 通道
    if C == 1:
        tensor = tensor.repeat(1, 3, 1, 1)

    for i in range(B):
        vutils.save_image(
            tensor[i],
            os.path.join(save_path, f"{prefix}_{i}.jpg")
        )





def get_edge(mask, ksize=3):
    """
    输入:
        mask: [N,1,H,W] 概率图 (0~1)
        ksize: 卷积核大小，控制边缘厚度
    输出:
        edge: [N,1,H,W] 二值边缘图 (0/1)
    """
    # 二值化 (>=0.5 为前景)
    mask_bin = (mask > 0.5).float()

    # 膨胀 (dilate)
    dilated = F.max_pool2d(mask_bin, kernel_size=ksize, stride=1, padding=ksize//2)

    # 腐蚀 (erode) = 1 - dilate(1-mask)
    eroded = 1 - F.max_pool2d(1 - mask_bin, kernel_size=ksize, stride=1, padding=ksize//2)

    # 边缘 = 膨胀 - 腐蚀
    edge = (dilated - eroded).clamp(0, 1)

    return edge

#edge-Attention
def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear): 
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

    def initialize(self):
        weight_init(self)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
    def initialize(self):
        weight_init(self)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

    def initialize(self):
        weight_init(self)

class MSAAttention(nn.Module):
    def __init__(self, dim, num_heads, bias,mode):
        super(MSAAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
        self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
        self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
    
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
    def forward(self, x,mask=None):
        b,c,h,w = x.shape
        q=self.qkv1conv(self.qkv_0(x))
        k=self.qkv2conv(self.qkv_1(x))
        v=self.qkv3conv(self.qkv_2(x))
        if mask is not None:
            q=q*mask
            k=k*mask

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

    def initialize(self):
        weight_init(self)

class MSA_head(nn.Module):
    def __init__(self, mode='dilation',dim=768, num_heads=8, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'):
        super(MSA_head, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = MSAAttention(dim, num_heads, bias,mode)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x,mask=None):
        x = x + self.attn(self.norm1(x),mask)
        x = x + self.ffn(self.norm2(x))
        return x

    def initialize(self):
        weight_init(self)

class MSA_module(nn.Module):
    def __init__(self, dim=768):
        super(MSA_module, self).__init__()
        self.B_TA = MSA_head()
        self.F_TA = MSA_head()
        self.TA = MSA_head()
        self.Fuse = nn.Conv2d(3*dim,dim,kernel_size=3,padding=1)
        self.Fuse2 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.Conv2d(dim, dim, kernel_size=3, padding=1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
    
    def forward(self,x,side_x,mask):
        N,C,H,W = x.shape
        mask = F.interpolate(mask,size=x.size()[2:],mode='bilinear')
        mask_d = mask.detach()

        # side_x 通道和空间对齐
        side_x = F.interpolate(side_x, size=(H,W), mode='bilinear', align_corners=False)
        if side_x.shape[1] != C:
            conv_proj = nn.Conv2d(side_x.shape[1], C, kernel_size=1).to(x.device)
            side_x = conv_proj(side_x)

        xf = self.F_TA(x,mask_d)
        xb = self.B_TA(x,1-mask_d)
        x = self.TA(x)
        x = torch.cat((xb,xf,x),1)
        x = x.view(N,3*C,H,W)
        x = self.Fuse(x)
        D = self.Fuse2(side_x+side_x*x)
        return D
    
    def initialize(self):
        weight_init(self)

#================


class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class ChannelDecoupledEdgeAttention(nn.Module):
    def __init__(self, channels, factor=8, vis=False, vis_dir="./paper_vis"):
        super(ChannelDecoupledEdgeAttention, self).__init__()
        self.vis = vis
        self.vis_dir = vis_dir
        self.vis_step = 0
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        
    def _visualize_groups_heatmap(self, weights, max_groups=8):
        os.makedirs(self.vis_dir, exist_ok=True)

        # weights: [B*G, 1, H, W]
        BxG, _, H, W = weights.shape
        G = min(self.groups, max_groups)

        for g in range(G):
            heat = weights[g, 0].detach().cpu().numpy()

            # normalize to [0,1]
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)

            plt.figure(figsize=(3, 3))
            plt.imshow(heat, cmap='jet')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')

            plt.savefig(
                os.path.join(
                    self.vis_dir,
                    f"step{self.vis_step:05d}_group{g}_heatmap.png"
                ),
                dpi=300,
                bbox_inches='tight',
                pad_inches=0
            )
            plt.close()

        self.vis_step += 1


    def forward(self, x, mask):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        #================
        mask = F.interpolate(mask,size=x.size()[2:],mode='bilinear')
        #print(mask.shape)
        mask_d = mask.detach()
        mask_d = mask_d.repeat_interleave(self.groups, dim=0)  # (16*groups, 1, H, W)
        #print(mask_d.shape)
        group_x = mask_d * group_x
        #print(group_x.shape)
        #=================
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)

        self.debug_group_x = group_x.detach()      # [B*G, Cg, H, W]
        self.debug_weights = weights.detach()      # [B*G, 1, H, W]
        self.debug_out = (group_x * weights.sigmoid()).detach()

        if self.vis:
            self._visualize_groups_heatmap(weights)
            

        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class MYMLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class MYMLP2(nn.Module):
    """
    ViT-style MLP projection
    Linear -> GELU -> Dropout -> Linear -> Dropout
    """
    def __init__(self, input_dim=2048, embed_dim=768, mlp_ratio=4, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)

        self.proj_in = nn.Linear(input_dim, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 输入 x: [B, C, H, W]
        x = x.flatten(2).transpose(1, 2)   # [B, N, C]
        x = self.proj_in(x)                # [B, N, embed_dim]
        x = self.mlp(x)                    # [B, N, embed_dim]
        return x
    
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class SegFormerHeadWithCDEA(nn.Module):
    def __init__(self, num_classes=2, in_channels=[96, 192, 384, 768], embedding_dim=768, dropout_ratio=0.1):
        super().__init__()
        c1_in, c2_in, c3_in, c4_in = in_channels

        # MLP projection
        self.linear_c4 = MYMLP2(input_dim=c4_in, embed_dim=embedding_dim)
        self.linear_c3 = MYMLP2(input_dim=c3_in, embed_dim=embedding_dim)
        self.linear_c2 = MYMLP2(input_dim=c2_in, embed_dim=embedding_dim)
        self.linear_c1 = MYMLP2(input_dim=c1_in, embed_dim=embedding_dim)

        # 每层预测器
        self.pred4 = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim // 2, num_classes, 1)
        )
        self.pred3 = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim // 2, num_classes, 1)
        )
        self.pred2 = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim // 2, num_classes, 1)
        )
        self.pred1 = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embedding_dim // 2, num_classes, 1)
        )


        self.cdea3 = ChannelDecoupledEdgeAttention(768)
        self.cdea2 = ChannelDecoupledEdgeAttention(768)
        self.cdea1 = ChannelDecoupledEdgeAttention(768)

        self.linear_fuse = ConvModule(
            c1=embedding_dim*4,
            c2=embedding_dim,
            k=1,
        )
        self.linear_pred    = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        _, c1, c2, c3, c4 = inputs
        n, _, H, W = c1.shape   # 以最浅层 c1 的分辨率作为 target size

        # -------- Stage 4 --------
        #print(c4.shape,'c4.shape')#([16, 1536, 4, 4])
        f4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(f4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        #print(f4.shape,'f4.shape')#([16, 768, 4, 4])
        p4 = self.pred4(f4)
        p4_up = F.interpolate(p4, size=(H, W), mode="bilinear", align_corners=False)
        mask4 = torch.sigmoid(p4_up)
        edge4 = get_edge(mask4[:, 0:1, :, :], ksize=2)

        # -------- Stage 3 --------
        f3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(f3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        #print(f3.shape,'f3.shape')#([16, 768, 8, 8])
        f3 = self.cdea3(f3, mask=edge4)   # side_x 用原始 c3
        p3 = self.pred3(f3)
        p3_up = F.interpolate(p3, size=(H, W), mode="bilinear", align_corners=False)
        mask3 = torch.sigmoid(p3_up)
        edge3 = get_edge(mask3[:, 0:1, :, :], ksize=2)

        # -------- Stage 2 --------
        f2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(f2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        f2 = self.cdea2(f2, mask=edge3)   # side_x 用原始 c2
        p2 = self.pred2(f2)
        p2_up = F.interpolate(p2, size=(H, W), mode="bilinear", align_corners=False)
        mask2 = torch.sigmoid(p2_up)
        edge2 = get_edge(mask2[:, 0:1, :, :], ksize=2)

        # #保存 edge2 / edge3 / edge4
        # save_tensor_as_jpg(edge2, save_dir, prefix="edge2")
        # save_tensor_as_jpg(edge3, save_dir, prefix="edge3")
        # save_tensor_as_jpg(edge4, save_dir, prefix="edge4")
        # exit()

        # -------- Stage 1 --------
        f1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        f1 = self.cdea1(f1, mask=edge2)   # side_x 用原始 c1
        p1 = self.pred1(f1)
        p1_up = F.interpolate(p1, size=(H, W), mode="bilinear", align_corners=False)

        

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, f1 ], dim=1))

        _c = self.dropout(_c)
        _c = self.linear_pred(_c)

        # save_colormap([p4_up, p3_up, p2_up, _c], save_dir="./outputs", cmap='coolwarm')
        # exit()

        return [p4_up, p3_up, p2_up, _c]

# class SegFormerHeadWithMSA(nn.Module):
#     def __init__(self, num_classes=2, in_channels=[96, 192, 384, 768], embedding_dim=768, dropout_ratio=0.1):
#         super().__init__()
#         c1_in, c2_in, c3_in, c4_in = in_channels

#         # MLP projection
#         self.linear_c4 = MYMLP2(input_dim=c4_in, embed_dim=embedding_dim)
#         self.linear_c3 = MYMLP2(input_dim=c3_in, embed_dim=embedding_dim)
#         self.linear_c2 = MYMLP2(input_dim=c2_in, embed_dim=embedding_dim)
#         self.linear_c1 = MYMLP2(input_dim=c1_in, embed_dim=embedding_dim)

#         # 每层预测器
#         self.pred4 = nn.Conv2d(embedding_dim, num_classes, 1)
#         self.pred3 = nn.Conv2d(embedding_dim, num_classes, 1)
#         self.pred2 = nn.Conv2d(embedding_dim, num_classes, 1)
#         self.pred1 = nn.Conv2d(embedding_dim, num_classes, 1)

#         # 每层的 MSA
#         self.msa3 = MSA_module(dim=embedding_dim)
#         self.msa2 = MSA_module(dim=embedding_dim)
#         self.msa1 = MSA_module(dim=embedding_dim)

#         self.dropout = nn.Dropout2d(dropout_ratio)

#     def forward(self, inputs):
#         _, c1, c2, c3, c4 = inputs
#         n, _, H, W = c1.shape   # 以最浅层 c1 的分辨率作为 target size

#         # -------- Stage 4 --------
#         #print(c4.shape,'c4.shape')([16, 1536, 4, 4])
#         f4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
#         #print(f4.shape,'f4.shape')([16, 768, 4, 4])
#         p4 = self.pred4(f4)
#         p4_up = F.interpolate(p4, size=(H, W), mode="bilinear", align_corners=False)
#         mask4 = torch.sigmoid(p4_up)

#         # -------- Stage 3 --------
#         f3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
#         #print(f3.shape,'f3.shape')([16, 768, 8, 8])
#         f3 = self.msa3(f3, side_x=f4, mask=mask4[:, 0:1, :, :])   # side_x 用原始 c3
#         p3 = self.pred3(f3)
#         p3_up = F.interpolate(p3, size=(H, W), mode="bilinear", align_corners=False)
#         mask3 = torch.sigmoid(p3_up)

#         # -------- Stage 2 --------
#         f2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
#         f2 = self.msa2(f2, side_x=f3, mask=mask3[:, 0:1, :, :])   # side_x 用原始 c2
#         p2 = self.pred2(f2)
#         p2_up = F.interpolate(p2, size=(H, W), mode="bilinear", align_corners=False)
#         mask2 = torch.sigmoid(p2_up)
#         edge2 = get_edge(mask2[:, 0:1, :, :], ksize=5)

#         # -------- Stage 1 --------
#         f1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
#         f1 = self.msa1(f1, side_x=f2, mask=edge2)   # side_x 用原始 c1
#         p1 = self.pred1(f1)
#         p1_up = F.interpolate(p1, size=(H, W), mode="bilinear", align_corners=False)

#         return [p4_up, p3_up, p2_up, p1_up]

# class SegFormerHead(nn.Module):
#     """
#     SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
#     """
#     def __init__(self, num_classes=2, in_channels=[96, 192, 384, 768], embedding_dim=768, dropout_ratio=0.1):
#         super(SegFormerHead, self).__init__()
#         c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels
        
#         self.linear_c4 = MYMLP2(input_dim=c4_in_channels, embed_dim=embedding_dim)
#         self.linear_c3 = MYMLP2(input_dim=c3_in_channels, embed_dim=embedding_dim)
#         self.linear_c2 = MYMLP2(input_dim=c2_in_channels, embed_dim=embedding_dim)
#         self.linear_c1 = MYMLP2(input_dim=c1_in_channels, embed_dim=embedding_dim)

#         # self.ema4 = EMA(c4_in_channels)
#         # self.ema3 = EMA(c3_in_channels)
#         # self.ema2 = EMA(c2_in_channels)
#         # self.ema1 = EMA(c1_in_channels)


#         self.linear_fuse = ConvModule(
#             c1=embedding_dim*4,
#             c2=embedding_dim,
#             k=1,
#         )

#         self.linear_pred    = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
#         self.dropout        = nn.Dropout2d(dropout_ratio)
    
#     def forward(self, inputs):
#         #c1, c2, c3,_ , c4 = inputs
#         _, c1, c2, c3, c4 = inputs
#         print('c1:',c1.shape)
#         print('c2:',c2.shape)
#         print('c3:',c3.shape)
#         print('c4:',c4.shape)

#         ############## MLP decoder on C1-C4 ###########
#         n, _, h, w = c4.shape
#         # c4 = self.ema4(c4)
#         # c3 = self.ema3(c3)
#         # c2 = self.ema2(c2)
#         # c1 = self.ema1(c1)

#         _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
#         _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

#         _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
#         _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

#         _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
#         _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

#         _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

#         _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        
#         x = self.dropout(_c)
#         x = self.linear_pred(x)


#         return x

def _cfg(url='', **kwargs):
    return {'url': url,
            'num_classes': 1000,
            'input_size': (3, 224, 224),
            'pool_size': None,
            'crop_pct': 0.875,
            'interpolation': 'bicubic',
            'fixed_input_size': True,
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            **kwargs
            }


default_cfgs = {
    'mamba_vision_T': _cfg(url='https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar',
                           crop_pct=1.0,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_T2': _cfg(url='https://huggingface.co/nvidia/MambaVision-T2-1K/resolve/main/mambavision_tiny2_1k.pth.tar',
                            crop_pct=0.98,
                            input_size=(3, 224, 224),
                            crop_mode='center'),
    'mamba_vision_S': _cfg(url='https://huggingface.co/nvidia/MambaVision-S-1K/resolve/main/mambavision_small_1k.pth.tar',
                           crop_pct=0.93,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_B': _cfg(url='https://huggingface.co/nvidia/MambaVision-B-1K/resolve/main/mambavision_base_1k.pth.tar',
                           crop_pct=1.0,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_B_21k': _cfg(url='https://huggingface.co/nvidia/MambaVision-B-21K/resolve/main/mambavision_base_21k.pth.tar',
                           crop_pct=1.0,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_L': _cfg(url='https://huggingface.co/nvidia/MambaVision-L-1K/resolve/main/mambavision_large_1k.pth.tar',
                           crop_pct=1.0,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_L_21k': _cfg(url='https://huggingface.co/nvidia/MambaVision-L-21K/resolve/main/mambavision_large_21k.pth.tar',
                           crop_pct=1.0,
                           input_size=(3, 224, 224),
                           crop_mode='center'),
    'mamba_vision_L2': _cfg(url='https://huggingface.co/nvidia/MambaVision-L2-1K/resolve/main/mambavision_large2_1k.pth.tar',
                            crop_pct=1.0,
                            input_size=(3, 224, 224),
                            crop_mode='center'),
    'mamba_vision_L2_512_21k': _cfg(url='https://huggingface.co/nvidia/MambaVision-L2-512-21K/resolve/main/mambavision_L2_21k_240m_512.pth.tar',
                            crop_pct=0.93,
                            input_size=(3, 512, 512),
                            crop_mode='squash'),
    'mamba_vision_L3_256_21k': _cfg(url='https://huggingface.co/nvidia/MambaVision-L3-256-21K/resolve/main/mambavision_L3_21k_740m_256.pth.tar',
                            crop_pct=1.0,
                            input_size=(3, 256, 256),
                            crop_mode='center'),
    'mamba_vision_L3_512_21k': _cfg(url='https://huggingface.co/nvidia/MambaVision-L3-512-21K/resolve/main/mambavision_L3_21k_740m_512.pth.tar',
                            crop_pct=0.93,
                            input_size=(3, 512, 512),
                            crop_mode='squash'),                               
}


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B,windows.shape[2], H, W)
    return x


def _load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
    
    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    
    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def _load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = torch.load(filename, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    _load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class Downsample(nn.Module):
    """
    Down-sampling block"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch embedding block"
    """

    def __init__(self, in_chans=1, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.medical=1
            dim: feature size dimension.
        """
        # in_dim = 1
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
            )
        #self.pos_embed = nn.Parameter(torch.zeros(1, dim, 64, 64))#
    


    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        # x = x + self.pos_embed

        return x


class ConvBlock(nn.Module):

    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate= 'tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x


class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out
    

class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 counter, 
                 transformer_blocks, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=False, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, 
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if counter in transformer_blocks:
            self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        else:
            self.mixer = MambaVisionMixer(d_model=dim, 
                                          d_state=8,  
                                          d_conv=3,    
                                          expand=1
                                          )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class MambaVisionLayer(nn.Module):
    """
    MambaVision layer"
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 conv=False,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks = [],
    ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            window_size: window size in each stage.
            conv: bool argument for conv stage flag.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
            transformer_blocks: list of transformer blocks.
        """

        super().__init__()
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.blocks = nn.ModuleList([ConvBlock(dim=dim,
                                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                   layer_scale=layer_scale_conv)
                                                   for i in range(depth)])
            self.transformer_block = False
        else:
            self.blocks = nn.ModuleList([Block(dim=dim,
                                               counter=i, 
                                               transformer_blocks=transformer_blocks,
                                               num_heads=num_heads,
                                               mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias,
                                               qk_scale=qk_scale,
                                               drop=drop,
                                               attn_drop=attn_drop,
                                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                               layer_scale=layer_scale)
                                               for i in range(depth)])
            self.transformer_block = True

        self.downsample = None if not downsample else Downsample(dim=dim)
        self.do_gt = False
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape

        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = torch.nn.functional.pad(x, (0,pad_r,0,pad_b))
                _, _, Hp, Wp = x.shape
            else:
                Hp, Wp = H, W
            x = window_partition(x, self.window_size)

        for _, blk in enumerate(self.blocks):
            x = blk(x)
        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()
        if self.downsample is None:
            return x
        return self.downsample(x)


class MambaVision(nn.Module):
    """
    MambaVision,
    """

    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 drop_path_rate=0.2,
                 in_chans=1,
                 num_classes=2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 **kwargs):
        """
        Args:
            dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            drop_path_rate: drop path rate.
            in_chans: number of input channels.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
        """
        super().__init__()
        print('len(depths):',len(depths))
        
        num_features = int(dim * 2 ** (len(depths) - 1))
        print(num_features)
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            conv = True if (i == 0 or i == 1) else False
            level = MambaVisionLayer(dim=int(dim * 2 ** i),
                                     depth=depths[i],
                                     num_heads=num_heads[i],
                                     window_size=window_size[i],
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     conv=conv,
                                     drop=drop_rate,
                                     attn_drop=attn_drop_rate,
                                     drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                     downsample=(i < 4),#org 3
                                     layer_scale=layer_scale,
                                     layer_scale_conv=layer_scale_conv,
                                     transformer_blocks=list(range(depths[i]//2+1, depths[i])) if depths[i]%2!=0 else list(range(depths[i]//2, depths[i])),
                                     )
            self.levels.append(level)
        self.norm = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #classifacation self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
        #seghead：
        #self.head = SegMambarHead(in_channels=num_features, num_classes=num_classes)
        #self.conv_seg = nn.Conv2d(channels = , self.out_channels = , kernel_size=1)
        #self.MLP = MLP_SegmentationHead(in_channels, out_channels, img_size=128)
        #self.headin_channels = [96, 192, 384, 768]
        self.headin_channels = [192, 384, 768, 1536]
        self.embedding_dim = 768
        self.decode_head = SegFormerHeadWithCDEA(num_classes, self.headin_channels, self.embedding_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x):
        #print("Shape befotr patch embedding:", x.shape)  # 打印中间的形状torch.Size([16, 1, 512, 512])
        x = self.patch_embed(x)
        #print("Shape after patch embedding:", x.shape)  # 打印中间的形状([16, 96, 128, 128])
        # 保存不同尺度的输出
        outs = []
        outs.append(x)

        for level in self.levels:
            x = level(x)
            #print('levels_shape:', x.shape)
            outs.append(x)  # 保存每个尺度的输出
        #print("Shape after x = level(x):", x.shape)  # 打印中间的形状

        # upsampled_outs = []
        # for i, out in enumerate(outs):
        #     upsampled = F.interpolate(out, size=(32,32), mode='bilinear', align_corners=False)
        #     upsampled_outs.append(upsampled)
        #     print(f"Upsampled shape for level {i}:", upsampled.shape)

        # # 融合多个尺度的输出
        # fused = torch.cat(upsampled_outs, dim=1)  # 沿通道维度拼接
        # print("Shape after concatenation:", fused.shape)



        # x = self.norm(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        return x,outs

    def forward(self, x):
        H, W = x.size(2), x.size(3)#HW
        x,outs = self.forward_features(x)
        
        x = self.decode_head.forward(outs)
        
        #==========version1============
        # print('decode_head.forward:',x.shape)
        
        # x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        #==========version1============

        #==========version2============
        upsampled_x = []
        for i, p in enumerate(x):
            # print(f'  p{i} shape before upsample: {p.shape}')
            # 上采样到指定大小
            p_up = F.interpolate(p, size=(H, W), mode='bilinear', align_corners=True)
            #print(f'  p{i} shape after upsample: {p_up.shape}')
            upsampled_x.append(p_up)

        # 替换原始输出
        x = upsampled_x
        #==========version2============
        return x

    def _load_state_dict(self, 
                         pretrained, 
                         strict: bool = False):
        _load_checkpoint(self, 
                         pretrained, 
                         strict=strict)