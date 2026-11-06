from tacos_neuron import PaperLIFNode
import math
import torch.nn as nn

from rotary_embedding_torch import RotaryEmbedding
from spikingjelly.activation_based import neuron, layer
from tacos_neuron import PaperLIFNode as LIFNode
from .conv import Transpose, PointwiseConv1d, DepthwiseConv1d


class MS_MLP(nn.Module):
    def __init__(
        self,
        config,
        in_features,
        hidden_features=None,
        out_features=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.config = config

        # batch, time, dim
        self.trans1 = Transpose(1,0,2)
        self.fc1_conv = layer.Linear(in_features, hidden_features, bias=False, step_mode='m')
        self.fc1_bn = layer.BatchNorm1d(hidden_features, step_mode='m')
        self.fc1_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True ,backend=config.backend)
        self.dropout1 = layer.Dropout(config.dropout_p, step_mode='m')

        self.fc2_conv = layer.Linear(hidden_features, out_features, bias=False, step_mode='m')
        self.fc2_bn = layer.BatchNorm1d(out_features, step_mode='m')
        self.fc2_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True,backend=config.backend)
        self.dropout2 = layer.Dropout(config.dropout_p, step_mode='m')



    def forward(self, x):

        x = self.fc1_conv(x)
        x = self.fc1_bn(x).contiguous()
        x = self.fc1_lif(x)
        x = self.dropout1(x)

        x = self.fc2_conv(x)
        x = self.fc2_bn(x).contiguous()
        x = self.fc2_lif(x)
        x = self.dropout2(x)
        return x


class MS_SSA(nn.Module):
    def __init__(
        self,
        dim,
        config,
        num_heads=8,
        init_tau = 2.0,
        layers=0,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.config = config

        # batch, time, dim
        self.trans1 = Transpose(1,0,2)
        self.scale = 1/math.sqrt(self.dim//self.num_heads)
        
        self.q_conv = layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=self.config.bias, step_mode='m')
        self.q_bn = layer.BatchNorm1d(self.config.n_hidden_neurons, step_mode='m')
        self.q_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq=False, backend=config.backend)
        self.q_lif2 = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq=False, backend=config.backend)

        self.k_conv = layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=self.config.bias,step_mode='m')
        self.k_bn = layer.BatchNorm1d(self.config.n_hidden_neurons, step_mode='m')
        self.k_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True,backend=config.backend)
        self.k_lif2 = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True,backend=config.backend)

        self.v_conv = layer.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=self.config.bias,step_mode='m')
        self.v_bn = layer.BatchNorm1d(self.config.n_hidden_neurons, step_mode='m')
        self.v_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True,backend=config.backend)

        self.attn_dropout = layer.Dropout(config.dropout_p, step_mode='m')
        self.attn_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True,backend=config.backend)

        self.proj_conv = layer.Linear(dim, dim, bias=self.config.bias,step_mode='m')
        self.proj_bn = layer.BatchNorm1d(dim, step_mode='m')
        self.dropout = layer.Dropout(config.dropout_p, step_mode='m')
        self.proj_lif = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
            surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
            step_mode='m', decay_input=False, store_v_seq = True,backend=config.backend)

        self.layers = layers
        self.rotary_emb = RotaryEmbedding(dim=self.dim//self.num_heads)

    def forward(self, x, x_com, attention_mask=None):

        T, B, N = x.shape
        x_qkv = x # B*N*T

        # q_conv_out = self.q_conv(x_qkv)
        q_conv_out = self.q_conv(x_com)
        q_conv_out = self.q_bn(q_conv_out)
        q_conv_out = self.q_lif(q_conv_out)

        q = (
            q_conv_out
                .reshape(T, B, self.num_heads, N // self.num_heads)
                .permute(1, 2, 0, 3) # => (B, H, T, D)
                .contiguous()
        )
        q = self.rotary_emb.rotate_queries_or_keys(q).reshape(T,B,-1)  # .permute(2,0,1,3)
        q = self.q_lif2(q).reshape(B,self.num_heads, T, N // self.num_heads) # .permute(1,2,0,3) # (B,H,T,D)

        k_conv_out = self.k_conv(x_qkv)
        k_conv_out = self.k_bn(k_conv_out)
        k_conv_out = self.k_lif(k_conv_out)

        k = (
            k_conv_out
                .reshape(T, B, self.num_heads, N // self.num_heads)
                .permute(1, 2, 0, 3) # => (B,H,T,D)
                .contiguous()
        )
        k = self.rotary_emb.rotate_queries_or_keys(k).reshape(T,B,-1)  # .permute(2,0,1,3) # (T,B,H,D)
        k = self.k_lif2(k).reshape(B, self.num_heads, T, N // self.num_heads) # .permute(1,2,0,3) # (B,H,T,D)

        v_conv_out = self.v_conv(x_qkv)
        v_conv_out = self.v_bn(v_conv_out)
        v_conv_out = self.v_lif(v_conv_out)

        v = (
            v_conv_out
                .reshape(T, B, self.num_heads, N // self.num_heads)
                .permute(1, 2, 0, 3) # => (B,H,T,D)
                .contiguous()
        )

        if attention_mask is not None: # attention_mask:(B,T)

            attn = (q @ k.transpose(-2, -1))

            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1) # (Batch, 1,1,Time)
            
            attention_mask = attention_mask.expand(attn.size())

            
            attn = attn.masked_fill(attention_mask == False, float(0.))

            x = attn@v # (B,H,T,T) * (B,H,T,D)
            x = x*self.scale  # (B,H,T,T)
            x = x.permute(2, 0, 1, 3)  # (T,B,H,D)

            
            x = self.attn_lif(x)
            x = self.attn_dropout(x)
            x = x.reshape(T, B, -1).contiguous()  # Ensure the tensor is stored in a contiguous chunk of memory

            x = self.proj_bn(self.proj_conv(x)).contiguous()
            x = self.proj_lif(x)
            x = self.dropout(x)
            return x




class GSU(nn.Module):
    """
    The gating mechanism is called Gated Spiking Units (GSU)
    """
    def __init__(self, config, split_dim: int, dim: int) -> None:
        super(GSU, self).__init__()
        self.split_dim = split_dim
        self.config = config
        self.lif1 = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
                        surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                        step_mode='m', decay_input=False, store_v_seq = True, backend=config.backend)
        self.lif2 = LIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
                        surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                        step_mode='m', decay_input=False, store_v_seq = True, backend=config.backend)

        self.proj_conv = layer.Linear(dim, dim, bias=config.bias, step_mode='m')
        self.proj_bn = layer.BatchNorm1d(dim, step_mode='m')


    def forward(self, inputs):
        outputs, gate = inputs.chunk(2, dim=self.split_dim)
        gate = self.lif1(gate)
        gate = self.proj_conv(gate)
        gate = self.lif2(self.proj_bn(gate))
        outputs = outputs * gate
        return outputs


class ConformerConvModule(nn.Module):
    def __init__(
            self,
            config,
            in_channels: int,
            kernel_size: int = 31,
            expansion_factor: int = 2,
            dropout_p: float = 0.1
    ) -> None:
        super(ConformerConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        # assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"
        self.config = config
        # batch, time, dim
        self.trans1 = Transpose(1,0,2)
        # batch, dim, time
        self.trans2 = Transpose(1, 2, 0)
        # time, batch ,dim
        self.trans3 = Transpose(2, 0, 1)
        
        self.bn3 = layer.BatchNorm1d(in_channels*expansion_factor, step_mode='m') #  self.bn1 = layer.BatchNorm1d(in_channels*expansion_factor, step_mode='m')
        self.bn2 = layer.BatchNorm1d(in_channels, step_mode='m')
        self.bn1 = layer.BatchNorm1d(in_channels, step_mode='m')
        
        self.lif1 = PaperLIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
                        surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                        step_mode='m', decay_input=False, store_v_seq = True,backend=config.backend)
        self.lif2 = PaperLIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
                        surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                        step_mode='m', decay_input=False, store_v_seq=False,backend=config.backend)
        self.lif3 = PaperLIFNode(tau=config.init_tau, v_threshold=config.v_threshold,
                        surrogate_function=config.surrogate_function, detach_reset=config.detach_reset,
                        step_mode='m', decay_input=False, store_v_seq=False,backend=config.backend)
        
        self.dw = DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=config.use_dw_bias) 
        self.pw1 = PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True)  # bias= config.bias
        self.pw2 = PointwiseConv1d(in_channels, in_channels* expansion_factor, stride=1, padding=0, bias=True) # # bias= config.bias

        self.dropout1 = layer.Dropout(dropout_p, step_mode='m')
        self.dropout2 = layer.Dropout(dropout_p, step_mode='m')
        self.dropout3 = layer.Dropout(dropout_p, step_mode='m')
            
        self.gsu = GSU(self.config, split_dim=-1, dim=in_channels)

    def forward(self, inputs):
        x = inputs
        # time, batch, dim
        x = self.trans2(x)
        # batch, dim, time
        x = self.pw1(x)
        # time, batch, dim*2
        x = self.trans3(x)
        x = self.bn1(x)

        x = self.lif1(x)
        x = self.dropout1(x)

        # batch, dim, time
        x = self.trans2(x)
        x = self.dw(x)
        # time, batch, dim
        x = self.trans3(x)
        x = self.bn2(x)
        x = self.lif2(x)
        x = self.dropout2(x)

        # batch, dim, time
        x = self.trans2(x)
        x = self.pw2(x)
        # time, batch, dim
        x = self.trans3(x)
        x = self.bn3(x)
        x = self.gsu(x) 
        x = self.lif3(x)
        x = self.dropout3(x)

        return x


class MS_Block_Conv(nn.Module):
    def __init__(
        self,
        config,
        dim,
        num_heads,
        init_tau=2.0,
        layers=0,
    ):
        super().__init__()
        self.config = config

        # SDT
        self.attn = MS_SSA(
            dim,
            config,
            init_tau = init_tau,
            num_heads=num_heads,
            layers=layers,
        )

        mlp_hidden_dim = config.hidden_dims
        # MLP
        self.mlp1 = MS_MLP(
            config,
            in_features=dim,
            hidden_features=mlp_hidden_dim,
        )
        self.mlp2 = MS_MLP(
            config,
            in_features=dim,
            hidden_features=mlp_hidden_dim
        )

        self.conformer_module = ConformerConvModule(
            config,
            in_channels=dim,
            kernel_size=config.kernel_size,
            expansion_factor=2,
            dropout_p=config.dropout_l
        )


    def forward(self, x, x_com, attention_mask=None):

        # Attention with residual
        attn_output = self.attn(x, x_com, attention_mask=attention_mask)
        x = x + attn_output

        # First MLP with residual
        mlp1_output = self.mlp1(x)
        x = x + mlp1_output
        
        # Convolution module with residual
        conv_output = self.conformer_module(x)
        x = x + conv_output

        # Second MLP with residual
        mlp2_output = self.mlp2(x)
        x = x + mlp2_output

        return x