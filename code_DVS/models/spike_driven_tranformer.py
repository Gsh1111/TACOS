import torch
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_
from spikingjelly.activation_based.neuron import surrogate
from tacos_neuron import PaperLIFNode as LIFNode
from .drop_feature import compress_embedding
from .utils import (
    print_accumulated_drop_stats,
    reset_drop_stats,
)


class Erode(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

    def forward(self, x):
        return self.pool(x)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        self.fc1_lif = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )

        self.fc2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm2d(out_features)
        self.fc2_lif = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()

        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()
        return x


class SDTAttention(nn.Module):
    def __init__(self, args, dim, num_heads=16, dvs=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.args = args
        self.num_heads = num_heads
        self.dvs = dvs
        if dvs:
            self.pool = Erode()
        self.scale = 0.125

        self.shortcut_lif = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )

        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim)
        self.q_lif = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )

        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm2d(dim)
        self.k_lif = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )

        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm2d(dim)
        self.v_lif = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )

        self.talking_heads_lif = LIFNode(
            tau=2.0,
            v_threshold=0.5,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )
        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm2d(dim)

    def forward(self, x, x_com):
        T, B, C, H, W = x.shape
        N = H * W
        x = self.shortcut_lif(x)

        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_com.flatten(0, 1))
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, H, W).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = (
            q_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, H, W).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        if self.dvs:
            k_conv_out = self.pool(k_conv_out)
        k = (
            k_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, H, W).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        if self.dvs:
            v_conv_out = self.pool(v_conv_out)
        v = (
            v_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )  # T B head N C//h

        kv = k.mul(v)
        if self.dvs:
            kv = self.pool(kv)
        kv = kv.sum(dim=-2, keepdim=True)
        kv = self.talking_heads_lif(kv)
        x = q.mul(kv)
        if self.dvs:
            x = self.pool(x)

        x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous()
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W).contiguous()

        return x


class Block(nn.Module):
    def __init__(self, args, dim, num_heads, mlp_ratio=4.0, drop=0.0, dvs=False):
        super().__init__()
        self.attn = SDTAttention(args, dim, num_heads=num_heads, dvs=dvs)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, x_com):
        x = x + self.attn(x, x_com)
        x = x + self.mlp(x)
        return x


class SPS(nn.Module):
    def __init__(
        self,
        img_size_h=128,
        img_size_w=128,
        patch_size=4,
        in_channels=2,
        embed_dims=256,
        pooling_stat="1111",
    ):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        self.pooling_stat = pooling_stat
        print(f"pooling_stat: {pooling_stat}")
        print(f"patch_size: {patch_size}")
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj_conv = nn.Conv2d(
            in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv1 = nn.Conv2d(
            embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv2 = nn.Conv2d(
            embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.proj_conv3 = nn.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
            store_v_seq=True,
        )
        self.maxpool3 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.rpe_conv = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.rpe_bn = nn.BatchNorm2d(embed_dims)

    def forward(self, x):
        T, B, C, H, W = x.shape
        ratio = 1
        x = self.proj_conv(x.flatten(0, 1))  # have some fire value
        x = self.proj_bn(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()
        x = self.proj_lif(x).flatten(0, 1)
        if self.pooling_stat[0] == "1":
            x = self.maxpool(x)
            ratio *= 2

        x = self.proj_conv1(x)
        H1, W1 = x.shape[-2], x.shape[-1]
        x = self.proj_bn1(x).reshape(T, B, -1, H1, W1).contiguous()
        x = self.proj_lif1(x).flatten(0, 1)
        if self.pooling_stat[1] == "1":
            x = self.maxpool1(x)
            ratio *= 2

        x = self.proj_conv2(x)
        H2, W2 = x.shape[-2], x.shape[-1]
        x = self.proj_bn2(x).reshape(T, B, -1, H2, W2).contiguous()
        x = self.proj_lif2(x).flatten(0, 1)
        if self.pooling_stat[2] == "1":
            x = self.maxpool2(x)
            ratio *= 2

        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        if self.pooling_stat[3] == "1":
            x = self.maxpool3(x)
            ratio *= 2

        x_feat = x
        H3, W3 = x.shape[-2], x.shape[-1]
        x = self.proj_lif3(x.reshape(T, B, -1, H3, W3).contiguous())
        x = x.flatten(0, 1).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x)
        x = (x + x_feat).reshape(T, B, -1, H3, W3).contiguous()

        v_seq = self.proj_lif3.paper_u_seq
        return x, v_seq


class SpikeDrivenTransformer(nn.Module):
    def __init__(
        self,
        args,
        img_size_h=128,
        img_size_w=128,
        patch_size=16,
        in_channels=2,
        num_classes=11,
        embed_dims=256,
        num_heads=16,
        mlp_ratios=4,
        depths=2,
        pooling_stat="1111",
    ):
        super().__init__()
        self.config = args
        self.num_classes = num_classes
        self.depths = depths

        self.patch_embed = SPS(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            pooling_stat=self.config.pooling_stat,
        )
        dvs = self.config.dataset.lower() == "dvs128"
        print(f"dvs: {dvs}")
        self.block = nn.ModuleList(
            [
                Block(args, dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, dvs=dvs)
                for j in range(depths)
            ]
        )

        # classification head
        if num_classes > 0:
            self.head = nn.Linear(embed_dims, num_classes)
        else:
            self.head = nn.Identity()
        self.head_lif = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )

        self.apply(self._init_weights)
        self.has_printed_drop_info = False

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.permute(1, 0, 2, 3, 4)  # [T, B, C, H, W]

        compress = self.config.drop_sequence

        x_len = torch.full((B,), T, device=x.device, dtype=torch.long)

        # embedding
        x, v_seq = self.patch_embed(x)  # [T, B, C, H, W]

        if compress:
            x, query_input, x_len = compress_embedding(
                self, x, x_len, v_seq, self.patch_embed.proj_lif3.paper_s_seq
            )
        else:
            query_input = x

        for blk in self.block:
            x = blk(x, query_input)
            query_input = x

        x = x.flatten(3).mean(3)
        x = self.head_lif(x)
        x = self.head(x.mean(0))

        return x

    def print_drop_stats(self):
        print_accumulated_drop_stats(self)

    def reset_drop_stats(self):
        reset_drop_stats(self)
