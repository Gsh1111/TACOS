from tacos_neuron import PaperLIFNode
import torch
import torch.nn as nn

from .modules import Transpose, PointwiseConv1d, DepthwiseConv1d
from .modules import MS_Block_Conv_spikcommander
from torch.nn.init import xavier_uniform_
from spikingjelly.activation_based import neuron, layer
from .drop_feature import compress_embedding
from .utils import (
    get_x_len_from_mask,
    rebuild_attention_mask,
    print_accumulated_drop_stats,
    reset_drop_stats,
)


class SpikingEmbedv2p(nn.Module):
    def __init__(self, config, kernel_size=7):
        super(SpikingEmbedv2p, self).__init__()
        self.config = config

        self.pwconv = PointwiseConv1d(
            config.n_inputs, config.n_hidden_neurons, stride=1, padding=0, bias=True
        )  # True
        self.dwconv = DepthwiseConv1d(
            config.n_hidden_neurons,
            config.n_hidden_neurons,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=config.use_dw_bias,
        )  # config.use_dw_bias

        self.linear = layer.Linear(
            self.config.n_hidden_neurons,
            self.config.n_hidden_neurons,
            bias=False,  # False，True
            step_mode="m",
        )  # self.config.bias

        self.bn1 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode="m")
        self.bn2 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode="m")

        self.dropout1 = layer.Dropout(config.dropout_l, step_mode="m")
        self.dropout2 = layer.Dropout(config.dropout_l, step_mode="m")

        self.trans1 = Transpose(0, 2, 1)
        self.trans2 = Transpose(2, 0, 1)
        self.trans3 = Transpose(1, 2, 0)

        self.lif1 = PaperLIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold,
            v_reset=config.v_reset,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode="m",
            decay_input=False,
            store_v_seq=False,
            backend=config.backend,
        )

        self.lif2 = PaperLIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold,
            v_reset=config.v_reset,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode="m",
            decay_input=False,
            store_v_seq=True,
            backend=config.backend,
        )

    def forward(self, x):
        # batch, time, dim =>  batch, dim, time
        x = self.trans1(x)
        x = self.pwconv(x)
        x = self.dwconv(x)

        # batch, dim, time =>  time, batch, dim
        x = self.trans2(x)
        x = self.bn1(x)
        x = self.lif1(x)
        x = self.dropout1(x)

        x_res = x
        x = self.linear(x)
        x = self.bn2(x)
        x = self.lif2(x)
        v_seq = self.lif2.paper_u_seq
        x = self.dropout2(x)
        x = x + x_res

        return x, v_seq


class SpikeDrivenTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.spike_embed = SpikingEmbedv2p(config, kernel_size=7)
        self.blocks = nn.ModuleList(
            [
                MS_Block_Conv_spikcommander(
                    config=self.config,
                    dim=self.config.n_hidden_neurons,
                    num_heads=self.config.num_heads,
                    init_tau=self.config.init_tau,
                    layers=j,
                )
                for j in range(self.config.depths)
            ]
        )

        self.final_dp = layer.Dropout(self.config.dropout_l, step_mode="m")
        self.head = layer.Linear(
            self.config.n_hidden_neurons, self.config.n_outputs, bias=False, step_mode="m"
        )
        self._reset_parameters()
        self.has_printed_drop_info = False

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def _init_x_len(self, x, attention_mask):
        """
        x before embedding: [B, T, D]
        """
        B, T, _ = x.shape
        if attention_mask is not None:
            x_len, original_mask_shape = get_x_len_from_mask(attention_mask, x.shape)
        else:
            x_len = torch.full((B,), T, dtype=torch.long, device=x.device)
            original_mask_shape = None
        return x_len, original_mask_shape

    def forward(self, x, attention_mask):
        B, T, _ = x.shape

        compress = self.config.drop_sequence

        x_len, original_mask_shape = self._init_x_len(x, attention_mask)

        # embedding
        x, v_seq = self.spike_embed(x)  # [T, B, D]

        if compress:
            x, query_input, x_len = compress_embedding(
                self, x, x_len, v_seq, self.spike_embed.lif2.paper_s_seq
            )
            if attention_mask is not None:
                attention_mask = rebuild_attention_mask(
                    x_len, x.shape[0], original_mask_shape, x.device
                )
        else:
            query_input = x

        # blocks
        for block in self.blocks:
            x = block(x, query_input, attention_mask)

            query_input = x

        x = self.head(self.final_dp(x))
        return x

    def print_drop_stats(self):
        print_accumulated_drop_stats(self)

    def reset_drop_stats(self):
        reset_drop_stats(self)
