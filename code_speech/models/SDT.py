from tacos_neuron import PaperLIFNode
import torch
import torch.nn as nn

from .modules import Transpose
from .modules import MS_Block_SDT
from torch.nn.init import xavier_uniform_
from spikingjelly.activation_based import neuron, layer
from .drop_feature import apply_drop_feature
from .modules import Transpose, PointwiseConv1d, DepthwiseConv1d
from .utils import (
    get_x_len_from_mask,
    rebuild_attention_mask,
    log_drop_event,
    print_tensor_stats_before_drop,
    print_tensor_stats_after_drop,
    ensure_drop_stats_initialized,
    update_drop_stats,
    print_accumulated_drop_stats,
    reset_drop_stats,
)

class SpikingEmbed(nn.Module):
    def __init__(self, config):
        super(SpikingEmbed, self).__init__()
        self.config = config

        self.conv = nn.Conv1d(config.n_inputs, config.n_hidden_neurons, kernel_size=3, stride=1, bias=config.bias, padding=1)
        self.bn = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
        self.dropout = layer.Dropout(config.dropout_l, step_mode='m')

        self.trans1 = Transpose(0,2,1)
        self.trans2 = Transpose(2,0,1)

        self.lif = PaperLIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode='m',
            decay_input=False,
            store_v_seq=True,
            backend=config.backend
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

class SpikingEmbedv2p(nn.Module):
    def __init__(self, config, kernel_size=7):
        super(SpikingEmbedv2p, self).__init__()
        self.config = config

        self.pwconv = PointwiseConv1d(config.n_inputs, config.n_hidden_neurons, stride=1, padding=0, bias=True) # True
        self.dwconv = DepthwiseConv1d(config.n_hidden_neurons, config.n_hidden_neurons, kernel_size, stride=1,
                                      padding=(kernel_size - 1) // 2,
                                      bias=config.use_dw_bias) # config.use_dw_bias

        self.bn1 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')
        if config.embed_projection:
            self.linear = layer.Linear(
                config.n_hidden_neurons, config.n_hidden_neurons, bias=False, step_mode='m'
            )
            self.bn2 = layer.BatchNorm1d(config.n_hidden_neurons, step_mode='m')

        self.dropout1 = layer.Dropout(config.dropout_l, step_mode='m')
        self.dropout2 = layer.Dropout(config.dropout_l, step_mode='m')

        self.trans1 = Transpose(0, 2, 1)
        self.trans2 = Transpose(2, 0, 1)
        self.trans3 = Transpose(1, 2, 0)

        self.lif1 = PaperLIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode='m',
            decay_input=False,
            store_v_seq=False,
            backend=config.backend
        )

        self.lif2 = PaperLIFNode(
            tau=config.init_tau,
            v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function,
            detach_reset=config.detach_reset,
            step_mode='m',
            decay_input=False,
            store_v_seq=True,
            backend=config.backend
        )

    def forward(self, x):
        x = self.trans1(x)
        x = self.pwconv(x)
        x = self.dwconv(x)
        
        x = self.trans2(x)
        x = self.bn1(x)
        x = self.lif1(x)
        x = self.dropout1(x)
        
        x_res = x
        if self.config.embed_projection:
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
        # self.spike_embed = SpikingEmbed(config)
        self.spike_embed = SpikingEmbedv2p(config)
        self.blocks = nn.ModuleList(
            [
                MS_Block_SDT(
                    config=self.config,
                    dim=self.config.n_hidden_neurons,
                    num_heads=self.config.num_heads,
                    init_tau=self.config.init_tau,
                    layers=j,
                )
                for j in range(self.config.depths)
            ]
        )

        self.final_dp = layer.Dropout(self.config.dropout_l, step_mode='m')
        self.head = layer.Linear(self.config.n_hidden_neurons, self.config.n_outputs, bias=False, step_mode='m')
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

    def _maybe_rebuild_attention_mask(self, attention_mask, x_len, x, original_mask_shape):
        """
        x after embedding / block: [T, B, D]
        """
        if attention_mask is None:
            return None
        return rebuild_attention_mask(x_len, x.shape[0], original_mask_shape, x.device)

    def _apply_feature_drop(
        self,
        x,
        x_len,
        attention_mask,
        original_mask_shape,
        log_state,
        tag,
        batch_size,
        v_seq=None,
    ):
        """
        Unified drop + print + stats + mask rebuild logic.

        Args:
            x: [T, B, D]
            x_len: [B]
            attention_mask: original / rebuilt mask
            original_mask_shape: original mask shape
            log_state: for log_drop_event
            tag: e.g. 'after_embedding'
            batch_size: B
            v_seq: optional, used when apply_drop_feature needs it

        Returns:
            x, x_com, x_len, attention_mask, dropped_this_call
        """
        x_len_before = x_len.clone()
        T_before = x.shape[0]

        print_tensor_stats_before_drop(x, self.has_printed_drop_info)
        ensure_drop_stats_initialized(self, T_before)

        if getattr(self.config, "drop_compensation", False):
            if v_seq is None:
                x, x_com, x_len, kept_indices = apply_drop_feature(self.config, x, x_len, return_indices=True)
            else:
                x, x_com, x_len, kept_indices = apply_drop_feature(self.config, x, x_len, v_seq, return_indices=True, spike_seq=self.spike_embed.lif2.paper_s_seq.reshape_as(v_seq))
        else:
            if v_seq is None:
                x, x_len, kept_indices = apply_drop_feature(self.config, x, x_len, return_indices=True)
            else:
                x, x_len, kept_indices = apply_drop_feature(self.config, x, x_len, v_seq, return_indices=True, spike_seq=self.spike_embed.lif2.paper_s_seq.reshape_as(v_seq))
            x_com = x

        log_drop_event(log_state, tag, x_len_before, x_len, T_before, x.shape[0])

        attention_mask = self._maybe_rebuild_attention_mask(
            attention_mask, x_len, x, original_mask_shape
        )

        update_drop_stats(self, kept_indices, batch_size)
        print_tensor_stats_after_drop(x, T_before, self.has_printed_drop_info)

        if not self.has_printed_drop_info:
            self.has_printed_drop_info = True

        return x, x_com, x_len, attention_mask, True

    def forward(self, x, attention_mask):
        B, T, _ = x.shape

        compress = self.config.drop_sequence

        log_state = {
            "enabled": (not self.has_printed_drop_info),
            "started": False,
        }
        drop_logged_this_forward = False

        x_len, original_mask_shape = self._init_x_len(x, attention_mask)


        # embedding
        x, v_seq = self.spike_embed(x)  # [T, B, D]

        if compress:
            x, x_com, x_len, attention_mask, dropped = self._apply_feature_drop(
                x=x,
                x_len=x_len,
                attention_mask=attention_mask,
                original_mask_shape=original_mask_shape,
                log_state=log_state,
                tag="after_embedding",
                batch_size=B,
                v_seq=v_seq,
            )
            drop_logged_this_forward |= dropped
        else:
            x_com = x

        # blocks
        for idx, block in enumerate(self.blocks):
            x = block(x, x_com, attention_mask)

            x_com = x

        if log_state["enabled"] and drop_logged_this_forward:
            self.has_printed_drop_info = True

        x = self.head(self.final_dp(x))
        return x

    def print_drop_stats(self):
        print_accumulated_drop_stats(self)

    def reset_drop_stats(self):
        reset_drop_stats(self)