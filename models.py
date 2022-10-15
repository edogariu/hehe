import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import numpy as np

import math
from abc import abstractmethod
from typing import List

from datasets import TOP_DIR_NAME


# Abstract wrapper class to be used when forward propagation needs step embeddings
class UsesDays(nn.Module):
    @abstractmethod
    def forward(self, x, day):
        """
        To be used when forward propagation needs day embeddings
        """

def override(a):  # silly function i had to write to write @override, sometimes python can be annoying lol
    return a

# Wrapper sequential class that knows when to pass time step embedding to its children or not
class UsesDaysSequential(nn.Sequential, UsesDays):
    @override
    def forward(self, x, day):
        for layer in self:
            if isinstance(layer, UsesDays):
                x = layer(x, day)
            else:
                x = layer(x)
        return x

# ------------------------------------------------------------------------------
# -------------------------------------  MODELS  -------------------------------
# ------------------------------------------------------------------------------

# TODO ADD DILATED TO ENCODR/decodr evan ur a goon for not doing this yet if u forget i will never forgive u, use FPN over dilated layerr
# TODO EVANN U NEED TO ADD DAY EMBEDDINGS AS INPUT EVANN
# TODO positional embeaingia

class LinearCoder(nn.Module):
    def __init__(self, 
                 layer_dims: List[int], 
                 dropout: float,
                 input_2d: bool,):
        super(LinearCoder, self).__init__()
        
        self.in_dim = layer_dims[0]
        self.out_dim = layer_dims[-1]
        
        self.blocks = []
        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            self.blocks.append(LinearBlock(in_dim, out_dim, dropout, input_2d))
        self.blocks = nn.Sequential(*self.blocks)
        
    def forward(self, x):
        return self.blocks(x)

      
class LinearCoderGroup(UsesDays):
    def __init__(self, 
                 layer_dims: List[int], 
                 dropout: float,
                 input_2d: bool,
                 days: List[int]):
        super(LinearCoderGroup, self).__init__()
        
        self.in_dim = layer_dims[0]
        self.out_dim = layer_dims[-1]
        
        self.days = days
        
        self.models = {}
        for d in self.days:
            self.models[str(d)] = LinearCoder(layer_dims, dropout, input_2d).blocks
        self.models = nn.ModuleDict(self.models)
        
    def forward(self, x, day):
        # TODO fix this so that we can use batching, but inference over different days in same batch
        outs = []
        for h, d in zip(x, day):
            outs.append(self.models[str(d.item())](h.unsqueeze(0)))
        return torch.cat(outs, dim=0)
    
    
class Encoder(UsesDays):
    """
    dilated model inspired by basenji and Sei <3
    enformer model inspired by enformer :)
    """
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 num_channels: int,
                 tower_length: int,
                 body_length: int,
                 body_type: str,  # one of ['enformer', 'dilated']
                 pooling_type: str,  # one of ['max', 'average', 'attention'] 
                 output_2d: bool,  # whether to have latent tensors be 2d `(B, out_dim, log_2(in_dim))` or 1d `(B, out_dim)`
                 ):
        super(Encoder, self).__init__()
        
        assert body_type in ['enformer', 'dilated']
        assert pooling_type in ['max', 'average', 'attention'] 
        
        pools = {'max': MaxPool,
                 'average': AvgPool,
                 'attention': AttentionPool}
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.tower_length = tower_length
        self.body_length = body_length
        self.body_type = body_type
        self.pooling_type = pooling_type
        self.num_channels = num_channels
            
        # stem is (conv, RConvBlock, Pool)
        self.stem = nn.Sequential(
            nn.Conv1d(1, self.num_channels // 2, kernel_size=15, padding=15 // 2),
            Residual(ConvBlock(self.num_channels // 2, self.num_channels // 2, 1, 1)),
            pools[self.pooling_type](self.num_channels // 2, 2),
        )
        
        self.tower_dims = [self.in_dim]
        for _ in range(self.tower_length + 1):
            if self.pooling_type == 'attention':
                self.tower_dims.append(math.floor((self.tower_dims[-1] + 1) / 2))
            else:
                self.tower_dims.append(math.ceil((self.tower_dims[-1] + 1) / 2))
        self.tower_dims = self.tower_dims[1:]
        self.tower_out_length = self.tower_dims[-1]
        
        # tower is tower_length x (ConvBlock, RConvBlock, Pool)
        tower_channels = exponential_linspace_int(start=self.num_channels // 2, end=self.num_channels, num=self.tower_length, divisible_by=1)
        self.tower = []
        for i in range(len(tower_channels)):
            c_in = tower_channels[i - 1] if i > 1 else self.num_channels // 2
            c_out = tower_channels[i]
            level = nn.Sequential(
                ConvBlock(c_in, c_out, 5, 1),
                Residual(ConvBlock(c_out, c_out, 1, 1)),
                pools[self.pooling_type](c_out, 2),
            )
            self.tower.append(level)
        self.tower = nn.Sequential(*self.tower)
        
        # body is `length` x body_type
        if self.body_type == 'enformer':
            self.body = nn.Sequential(*[TransformerBlock(c_out, num_heads=8, dropout=0.2) for _ in range(self.body_length)])
        elif self.body_type == 'dilated':
            self.body = nn.Identity() # TODO ADD DILATED MODEL!!!!
        else:
            raise NotImplementedError('huh????')
        
        self.pointwise = nn.Sequential(
            # TargetLengthCrop(320), 
            ConvBlock(c_out, 2 * self.num_channels, 1, 1),
            nn.Dropout(0.05),
            GeLU(),
        )
        
        self.output_2d = output_2d
        if self.output_2d:
            self.out = nn.Sequential(
            nn.Conv1d(2 * self.num_channels, self.out_dim, 1, 1),
            GeLU(),
        )
        else:
            self.out = nn.Sequential(
                nn.Conv1d(2 * self.num_channels, self.out_dim, 1, 1),
                GeLU(),
                nn.Linear(self.tower_out_length, self.tower_out_length // 2),
                GeLU(),
                nn.Linear(self.tower_out_length // 2, 1),
            )
        
    def forward(self, x, day):
        h = x
        h = h.unsqueeze(1)
        h = self.stem(h)
        h = self.tower(h)
        # maybe add here a LinearCoder to take days as input and do something? something like      h = self.lc(h.reshape(N, -1), day).reshape(N, C, L)
        h = self.body(h)
        h = self.pointwise(h)
        h = self.out(h)
        if not self.output_2d: h = h.squeeze(-1)
        return h
    
class Decoder(UsesDays):
    """
    dilated model inspired by basenji and Sei <3
    enformer model inspired by enformer :)
    """
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 num_channels: int,
                 tower_length: int,
                 body_length: int,
                 body_type: str,  # one of ['enformer', 'dilated']
                 unpooling_type: str,  # one of ['interpolation', 'conv'] 
                 input_2d: bool,  # whether to have latent tensors be 2d `(B, in_dim, log_2(out_dim))` or 1d `(B, in_dim)`
                 ):
        super(Decoder, self).__init__()
        
        assert body_type in ['enformer', 'dilated']
        assert unpooling_type in ['interpolation', 'conv'] 
        
        unpools = {'interpolation': UnpoolInterpolate,
                   'conv': UnpoolConv,}
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.tower_length = tower_length
        self.body_length = body_length
        self.body_type = body_type
        self.unpooling_type = unpooling_type
        self.num_channels = num_channels
        
        self.tower_dims = [self.out_dim]
        for _ in range(self.tower_length + 1):
            self.tower_dims.append(math.ceil((self.tower_dims[-1] + 1) / 2))
        self.tower_dims = self.tower_dims[::-1]
        self.tower_in_length = self.tower_dims[0]
        self.tower_dims = self.tower_dims[1:]
        
        self.input_2d = input_2d
        if self.input_2d:
            self.rev_out = nn.Sequential(
                nn.Conv1d(self.in_dim, 2 * self.num_channels, 1, 1),
            )
        else: 
            self.rev_out = nn.Sequential(
                nn.Linear(1, self.tower_in_length // 2),
                GeLU(),
                nn.Linear(self.tower_in_length // 2, self.tower_in_length),
                GeLU(),
                nn.Conv1d(self.in_dim, 2 * self.num_channels, 1, 1),
            )
        
        self.rev_pointwise = ConvBlock(2 * self.num_channels, self.num_channels, 1, 1)
        
        # body is `length` x body_type
        if self.body_type == 'enformer':
            self.rev_body = nn.Sequential(*[TransformerBlock(self.num_channels, num_heads=8, dropout=0.2) for _ in range(self.body_length)])
        elif self.body_type == 'dilated':
            self.rev_body = nn.Identity()  # TODO ADD DILATED MODELL!!!!
        else:
            raise NotImplementedError('huh????')
        
        tower_channels = exponential_linspace_int(start=self.num_channels, end=self.num_channels // 2, num=self.tower_length, divisible_by=1)
        self.rev_tower = []
        for i, d in zip(range(len(tower_channels)), self.tower_dims):
            c_in = tower_channels[i]
            c_out = tower_channels[i + 1] if i < self.tower_length - 1 else self.num_channels // 2
            level = nn.Sequential(
                unpools[self.unpooling_type](c_in, d),
                Residual(TransConvBlock(c_in, c_in, 1, 1)),
                TransConvBlock(c_in, c_out, 5, 1),
            )
            self.rev_tower.append(level)
        self.rev_tower = nn.Sequential(*self.rev_tower)
        
        self.rev_stem = nn.Sequential(
            unpools[self.unpooling_type](self.num_channels // 2, self.out_dim),
            Residual(TransConvBlock(self.num_channels // 2, self.num_channels // 2, 1, 1)),
            nn.ConvTranspose1d(self.num_channels // 2, 1, kernel_size=15, padding=15 // 2)
        )
        
    def forward(self, x, day):
        h = x
        if not self.input_2d: h = h.unsqueeze(-1)
        h = self.rev_out(h)
        h = self.rev_pointwise(h)
        h = self.rev_body(h)
        h = self.rev_tower(h)
        h = self.rev_stem(h)
        h = h.squeeze(1)
        return h
    
# ------------------------------------------------------------------------------
# -------------------------------------  BLOCKS  -------------------------------
# ------------------------------------------------------------------------------

class LinearBlock(nn.Module):
    """
    Uses no embedding
    """
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 dropout: float,
                 input_2d: bool):
        super(LinearBlock, self).__init__()
                
        self.in_linear = nn.Conv1d(in_dim, out_dim, 1) if input_2d else nn.Linear(in_dim, out_dim)
        self.out_linear = nn.Conv1d(out_dim, out_dim, 1) if input_2d else nn.Linear(out_dim, out_dim)

        self.activation = nn.ReLU()
        
        self.out_norm = ChannelNorm(out_dim) if input_2d else nn.LayerNorm(out_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = x
        h = self.in_linear(h)
        h = self.activation(h)
        h = self.out_norm(h)
        h = self.dropout(h)
        t = h
        h = self.out_linear(h) + t
        return h
   
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super(ConvBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        self.layers = nn.Sequential(nn.BatchNorm1d(in_channels),  # TODO figure out good init for batchnorm
                                    GeLU(),
                                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, dilation=dilation))
    
    def forward(self, x):
        return self.layers(x)
    
class TransConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super(TransConvBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        self.layers = nn.Sequential(nn.BatchNorm1d(in_channels),  # TODO figure out good init for batchnorm
                                    GeLU(),
                                    nn.ConvTranspose1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, dilation=dilation))
    
    def forward(self, x):
        return self.layers(x)
    
class AttentionBlock(nn.Module):
    """
    Creates an attention-style Transformer block containing QKV-attention, a residual connection,
    and a linear projection out.
    Uses 1 layer of LayerNorm.
    
    Parameters
    -------------
        - channels (int): number of channels to pass in and return out
        - num_heads (int): number of heads to use in Multi-Headed-Attention
        - num_head_channels (int): number of channels to use for each head. if not None, then this block uses
        (channels // num_head_channels) heads and ignores num_heads
        - split_qkv_first (bool): whether to split qkv first or split heads first during attention
        - dropout (float): dropout probability
        
    Returns
    -------------
        - An nn.Module to be used to compose a network.
    """

    def __init__(self, channels, num_heads=1, num_head_channels=None, split_qkv_first=True, dropout=0.3):
        super(AttentionBlock, self).__init__()

        if num_head_channels is None:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), "channels {} is not divisible by num_head_channels {}".format(channels, num_head_channels)
            self.num_heads = channels // num_head_channels

        self.split_qkv_first = split_qkv_first
        self.scale = (channels // self.num_heads) ** -0.5

        self.qkv_nin = nn.Conv1d(in_channels=channels, out_channels=3 * channels,
                                 kernel_size=(1,), stride=(1,))

        self.norm = ChannelNorm(channels)
        self.proj_out = zero_module(nn.Conv1d(in_channels=channels, out_channels=channels,
                                              kernel_size=(1,), stride=(1,)))
        self.dropout = nn.Dropout(dropout)

    # My implementation of MHA
    def forward(self, x):
        # compute attention
        B, C, L = x.shape
        qkv = self.norm(x)

        if self.split_qkv_first:
            # Get q, k, v
            qkv = self.qkv_nin(qkv).permute(0, 2, 1)  # b,c,hw -> b,hw,c
            qkv = qkv.reshape(B, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # q/k/v.shape = b,num_heads,hw,c//num_heads

            # w = softmax(q @ k / sqrt(d_k))
            w = (q @ k.transpose(2, 3)) * self.scale
            w = torch.softmax(w, dim=-1)

            # h = w @ v
            h = (w @ v).transpose(1, 2).reshape(B, L, C).permute(0, 2, 1)
        else:
            qkv = self.qkv_nin(qkv)  # b, 3 * c, hw
            q, k, v = qkv.reshape(B * self.num_heads, 3 * C // self.num_heads, L).split(C // self.num_heads, dim=1)

            # w = softmax(q @ k / sqrt(d_k))
            w = q.transpose(1, 2) @ k * self.scale
            w = torch.softmax(w, dim=2)

            # h = w @ v
            h = torch.einsum("bts,bcs->bct", w, v).reshape(B, -1, L)

        h = self.proj_out(h)
        h = self.dropout(h)

        return h + x

class TransformerBlock(nn.Module):
    """
    based on Enformer blocks
    """
    def __init__(self, channels: int, num_heads: int, dropout: float):
        super(TransformerBlock, self).__init__()
        
        self.mha = AttentionBlock(channels=channels, num_heads=num_heads, split_qkv_first=True, dropout=dropout)
        self.feed_forward = Residual(nn.Sequential(
            ChannelNorm(channels),
            nn.Conv1d(channels, 2 * channels, 1, 1),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Conv1d(2 * channels, channels, 1, 1),
            nn.Dropout(dropout),
        ))
    
    def forward(self, x):
        h = x
        h = self.mha(h)
        h = self.feed_forward(h)
        return h
    
class ChannelNorm(nn.Module):
    """
    LayerNorm across channel dim (NCL -> NLC -> LayerNorm -> NCL)
    """
    def __init__(self, num_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
    
    def forward(self, x):
        h = x.swapaxes(1, 2)
        h = self.norm(h)
        h = h.swapaxes(1, 2)
        return h
        
class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super(Residual, self).__init__()
        
        self.module = module
        
    def forward(self, x):
        h = self.module(x)
        return x + h
    
class GeLU(nn.Module):
    """
    GeLU approximation from section 2 of https://arxiv.org/abs/1606.08415 
    """
    def __init__(self):
        super(GeLU, self).__init__()
                
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x
        

class TargetLengthCrop(nn.Module):
    def __init__(self, target_length: int):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-1], self.target_length

        if target_len == -1:
            return x

        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')

        trim = (seq_len - target_len) // 2

        if trim == 0:
            return x

        return x[:, trim:-trim]
    
class MaxPool(nn.Module):
    def __init__(self, in_channels: int, pool_size: int):
        super().__init__()
        self.fn = nn.MaxPool1d(pool_size, padding=pool_size // 2)
    
    def forward(self, x):
        return self.fn(x)
    
class AvgPool(nn.Module):
    def __init__(self, in_channels: int, pool_size: int):
        super().__init__()
        self.fn = nn.AvgPool1d(pool_size, padding=pool_size // 2)
    
    def forward(self, x):
        return self.fn(x)
    
class AttentionPool(nn.Module):
    def __init__(self, in_channels: int, pool_size: int):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange('b d (n p) -> b d n p', p = pool_size)
        self.to_attn_logits = nn.Conv2d(in_channels, in_channels, 1, bias = False)  # softmax is agnostic to shifts

    def forward(self, x):
        """
        number of features is in channel dim here
        """
        b, _, length = x.shape
        remainder = length % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value = 0)
            mask = torch.zeros((b, 1, length), dtype = torch.bool, device = x.device)
            mask = F.pad(mask, (0, remainder), value = True)

        x = self.pool_fn(x)
        logits = self.to_attn_logits(x)

        if needs_padding:
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(self.pool_fn(mask), mask_value)

        attn = logits.softmax(dim = -1)

        return (x * attn).sum(dim = -1) 
    
class UnpoolInterpolate(nn.Module):
    """
    1D upsampling module, basically a drunk man's inverse pooling function
    """
    def __init__(self, in_channels: int, out_length: int):
        super().__init__()
        self.out_length = out_length
    
    def forward(self, x):
        x = F.interpolate(x, size=self.out_length, mode='nearest')
        return x

class UnpoolConv(nn.Module):
    """
    1D upsampling module, basically a drunk man's inverse pooling function, but this time with conv
    """
    def __init__(self, in_channels: int, out_length: int):
        super().__init__()
        self.out_length = out_length
        self.conv = nn.Conv1d(in_channels, in_channels, 3, 1, 1)
    
    def forward(self, x):
        x = F.interpolate(x, size=self.out_length, mode='nearest')
        x = self.conv(x)
        return x

# -----------------------------------------------------------------------------
# -------------------------------------  UTILS  -------------------------------
# -----------------------------------------------------------------------------

# Method to create sinusoidal timestep embeddings, much like positional encodings found in many Transformers
def timestep_embedding(timesteps, embedding_dim, method='identity', max_period=10000):
    """
    Embeds input timesteps

    Parameters
    ----------
    timesteps : torch.tensor
        input timesteps
    embedding_dim : int
        dimension for each scalar to embed to
    method : str, optional
        how to perform embedding, by default 'sin', must be one of `['sin', 'identity']`
    max_period : int, optional
        maximum period for sinusoidal embeddings, by default 10000

    Returns
    -------
    torch.tensor
        embedded timesteps
    """
    if method == 'sin':
        half = embedding_dim // 2
        emb = math.log(max_period) / half
        emb = torch.exp(torch.arange(half, dtype=torch.float32) * -emb).to(timesteps.device)
        emb = timesteps[:, None].float() * emb[None]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=1)
        # Zero pad for odd dimensions
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
    elif method == 'identity':
        emb = timesteps.unsqueeze(-1).expand([*timesteps.shape, embedding_dim]) / 5 - 1  # rescale to be in [-1, 1], where -1 is day 0 and 1 is day 10
    return emb

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())

def exponential_linspace_int(start, end, num, divisible_by=1):
    """Exponentially increasing values of integers."""
    base = np.exp(np.log(end / start) / (num - 1))
    return [int(np.round(start * base**i / divisible_by) * divisible_by) for i in range(num)]

def zero_module(module):
    """
    Helpful method that zeros all the parameters in a nn.Module, used for initialization
    """
    for param in module.parameters():
        param.detach().zero_()
    return module
