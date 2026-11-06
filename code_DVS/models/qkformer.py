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


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)
        self.mlp1_lif = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )

        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)
        self.mlp2_lif = LIFNode(
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

        x = self.mlp1_conv(x.flatten(0, 1))
        x = self.mlp1_bn(x).reshape(T, B, self.c_hidden, H, W)
        x = self.mlp1_lif(x)

        x = self.mlp2_conv(x.flatten(0, 1))
        x = self.mlp2_bn(x).reshape(T, B, C, H, W)
        x = self.mlp2_lif(x)
        return x


class Token_QK_Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )

        self.attn_lif = LIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy")

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )

    def forward(self, x, x_com):
        T, B, C, H, W = x.shape

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_com.flatten(3).flatten(0, 1))
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        q = torch.sum(q, dim=3, keepdim=True)
        attn = self.attn_lif(q)
        x = torch.mul(attn, k)

        x = x.flatten(2, 3)
        x = self.proj_bn(self.proj_conv(x.flatten(0, 1))).reshape(T, B, C, H, W)
        x = self.proj_lif(x)
        return x


class SSA(nn.Module):
    def __init__(self, dim, num_heads=16):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )
        self.attn_lif = LIFNode(
            tau=2.0,
            v_threshold=0.5,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )

        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = (
            q_conv_out.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = (
            k_conv_out.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = (
            v_conv_out.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T, B, C, H, W)
        return x


class TokenSpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.tssa = Token_QK_Attention(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x, x_com):

        x = x + self.tssa(x, x_com)
        x = x + self.mlp(x)

        return x


class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.ssa = SSA(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.ssa(x)
        x = x + self.mlp(x)

        return x


class PatchEmbedInit(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
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

        self.proj1_conv = nn.Conv2d(
            embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj1_bn = nn.BatchNorm2d(embed_dims // 4)
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        self.proj1_lif = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )

        self.proj2_conv = nn.Conv2d(
            embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj2_bn = nn.BatchNorm2d(embed_dims // 2)
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        self.proj2_lif = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )

        self.proj3_conv = nn.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.maxpool3 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        self.proj3_lif = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )

        self.proj_res_conv = nn.Conv2d(
            embed_dims // 4, embed_dims, kernel_size=1, stride=4, padding=0, bias=False
        )
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
            store_v_seq=True,
        )

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W)
        x = self.proj_lif(x).flatten(0, 1).contiguous()

        x = self.proj1_conv(x)
        x = self.proj1_bn(x)
        x = self.maxpool1(x)
        H1, W1 = x.shape[-2], x.shape[-1]
        x = x.reshape(T, B, -1, H1, W1).contiguous()
        x = self.proj1_lif(x).flatten(0, 1).contiguous()

        x_feat = x
        x = self.proj2_conv(x)
        x = self.proj2_bn(x)
        x = self.maxpool2(x)
        H2, W2 = x.shape[-2], x.shape[-1]
        x = x.reshape(T, B, -1, H2, W2).contiguous()
        x = self.proj2_lif(x).flatten(0, 1).contiguous()

        x = self.proj3_conv(x)
        x = self.proj3_bn(x)
        x = self.maxpool3(x)
        H3, W3 = x.shape[-2], x.shape[-1]
        x = x.reshape(T, B, -1, H3, W3).contiguous()
        x = self.proj3_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = x_feat.reshape(T, B, -1, H3, W3).contiguous()
        x_feat = self.proj_res_lif(x_feat)
        x = x + x_feat  # shortcut

        v_seq = self.proj_res_lif.paper_u_seq
        return x, v_seq


class PatchEmbeddingStage(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj_conv = nn.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj_bn = nn.BatchNorm2d(embed_dims)
        self.proj_lif = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )

        self.proj4_conv = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.proj4_bn = nn.BatchNorm2d(embed_dims)
        self.proj4_maxpool = torch.nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        self.proj4_lif = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )

        self.proj_res_conv = nn.Conv2d(
            embed_dims // 2, embed_dims, kernel_size=1, stride=2, padding=0, bias=False
        )
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = LIFNode(
            tau=2.0,
            detach_reset=True,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            backend="cupy",
            step_mode="m",
        )

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = x.flatten(0, 1).contiguous()
        x_feat = x

        x = self.proj_conv(x)
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x).flatten(0, 1).contiguous()

        x = self.proj4_conv(x)
        x = self.proj4_bn(x)
        x = self.proj4_maxpool(x)
        H1, W1 = x.shape[-2], x.shape[-1]
        x = x.reshape(T, B, -1, H1, W1).contiguous()
        x = self.proj4_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat)
        x_feat = x_feat.reshape(T, B, -1, H1, W1).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat  # shortcut

        return x


class QKFormer(nn.Module):
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
        mlp_ratios=1,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=2,
        sr_ratios=1,
    ):
        super().__init__()
        self.config = args
        self.num_classes = num_classes
        self.depths = depths

        self.patch_embed1 = PatchEmbedInit(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims // 2,
        )

        self.stage1 = TokenSpikingTransformer(
            dim=embed_dims // 2, num_heads=num_heads, mlp_ratio=mlp_ratios
        )

        self.patch_embed2 = PatchEmbeddingStage(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
        )

        self.stage2 = SpikingTransformer(dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios)

        # classification head
        if num_classes > 0:
            self.head = nn.Linear(embed_dims, num_classes)
        else:
            self.head = nn.Identity()
        self.apply(self._init_weights)
        self.has_printed_drop_info = False

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        return None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.permute(1, 0, 2, 3, 4)  # [T, B, C, H, W]

        compress = self.config.drop_sequence

        x_len = torch.full((B,), T, device=x.device, dtype=torch.long)

        # embedding
        x, v_seq = self.patch_embed1(x)  # [T, B, C, H, W]

        if compress:
            x, query_input, x_len = compress_embedding(
                self, x, x_len, v_seq, self.patch_embed1.proj_res_lif.paper_s_seq
            )
        else:
            query_input = x

        x = self.stage1(x, query_input)

        x = self.patch_embed2(x)
        x = self.stage2(x)
        x = x.flatten(3).mean(3)
        x = self.head(x.mean(0))

        return x

    def print_drop_stats(self):
        print_accumulated_drop_stats(self)

    def reset_drop_stats(self):
        reset_drop_stats(self)
