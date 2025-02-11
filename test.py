import torch
import torch.nn as nn
import torch.nn.functional as F
import intel_extension_for_pytorch as ipex

from einops import rearrange
from einops.layers.torch import Rearrange

import time

torch.set_default_device("xpu")
torch.set_default_dtype(torch.bfloat16)
torch.backends.cuda.enable_flash_sdp(True)
######### helpers #########


######### classes #########



class ParallelMLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.RMSNorm(dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.norm = nn.RMSNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b s (h d) -> b h s d', h=self.heads), qkv) #B, hc, s, hs
        

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        out = rearrange(attn_out, 'b h s d -> b s (h d)')
        out = self.norm(out)
        return self.to_out(out)



if __name__ == '__main__':
    seq = 2048
    dim = 1024
    mlp_dim = dim*2
    heads = 8
    dim_head = dim//heads

    x = torch.randn(1, seq, dim) #B, hc, s, hs
    layer = nn.Sequential(Attention(dim, heads=heads, dim_head=dim_head),ParallelMLP(dim, mlp_dim))

    print(layer)
    print(f'=> Trainable Params: {sum(p.numel() for p in layer.parameters() if p.requires_grad)}')
    y = layer(x)
    print(y.shape)
