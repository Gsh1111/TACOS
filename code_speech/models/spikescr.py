from tacos_neuron import PaperLIFNode
import torch
import torch.nn as nn

from .modules import Transpose
from .modules import MS_Block_Conv_spikescr
from torch.nn.init import xavier_uniform_
from spikingjelly.activation_based import neuron, layer
from .drop_feature import compress_embedding
from .utils import (
    get_x_len_from_mask,
    rebuild_attention_mask,
    print_accumulated_drop_stats,
    reset_drop_stats,
)


class SpikingEmbed(nn.Module):
    def __init__(self, config):
        super(SpikingEmbed, self).__init__()
        self.config = config

        self.conv = nn.Conv1d(
            config.n_inputs,
            config.n_hidden_neurons,
            kernel_size=3,
            stride=1,
            bias=config.bias,
            padding=1,
        )
        self.bn = layer.BatchNorm1d(config.n_hidden_neurons, step_mode="m")
        self.dropout = layer.Dropout(config.dropout_l, step_mode="m")

        self.trans1 = Transpose(0, 2, 1)
        self.trans2 = Transpose(2, 0, 1)

        self.lif = PaperLIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode="m",
            decay_input=False,
            store_v_seq=True,
            backend=config.backend,
        )

    def forward(self, x):
        x = self.trans1(x)
        x = self.conv(x)
        x = self.trans2(x)
        x = self.bn(x)

        x = self.lif(x)
        v_seq = self.lif.paper_u_seq
        # print(f"v_seq: {v_seq}")
        x = self.dropout(x)

        return x, v_seq


class SpikeDrivenTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.spike_embed = SpikingEmbed(config)
        self.blocks = nn.ModuleList(
            [
                MS_Block_Conv_spikescr(
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
                self, x, x_len, v_seq, self.spike_embed.lif.paper_s_seq
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
