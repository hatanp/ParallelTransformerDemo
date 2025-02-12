from mpi4py import MPI

import torch
import torch.nn as nn
import torch.nn.functional as F
import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch
from einops import rearrange
from einops.layers.torch import Rearrange

import time
import socket
import os

torch.set_default_device("xpu")
torch.set_default_dtype(torch.bfloat16)
######### helpers #########


######### classes #########



rank = int(MPI.COMM_WORLD.Get_rank())
world_size = int(MPI.COMM_WORLD.Get_size())

#print(f"rank {rank}/{world_size}")
#logging.info(f"rank {rank}/{world_size}")

local_rank = rank % 12
torch.xpu.set_device(local_rank)

if rank == 0:
   master_addr              = socket.gethostname()
   sock                     = socket.socket()
   sock.bind(('',0))
   # master_port  = sock.getsockname()[1]
   master_port              = 2345
else:
   master_addr              = None
   master_port              = None

master_addr                 = MPI.COMM_WORLD.bcast(master_addr, root=0)
master_port                 = MPI.COMM_WORLD.bcast(master_port, root=0)
os.environ["MASTER_ADDR"]   = master_addr
os.environ["MASTER_PORT"]   = str(master_port)

MPI.COMM_WORLD.Barrier()
start = time.time()
torch.distributed.init_process_group(
    backend="ccl",
    init_method="env://",
    world_size=world_size,
    rank=rank,
)
TP = 12
tp_group=None
for i in range(world_size//TP):
    ranks = [j for j in range(i*TP,(i+1)*TP)]
    group = torch.distributed.new_group(ranks)
    if rank in ranks:
        tp_group=group
        #print(f"TP group = {ranks} on {rank}")

dp_group = None
for i in range(TP):
    ranks = [i for i in range(i,world_size,TP)]
    group = torch.distributed.new_group(ranks)
    if rank in ranks:
        dp_group=group
        #print(f"DP group = {ranks} on {rank}")

class ColumnParallelLinear(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.net = nn.Linear(dim1, dim2//TP)

    def forward(self, x):
        return self.net(x)

class RowParallelLinear(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.net = nn.Linear(dim1//TP, dim2)

    def forward(self, x):
        intermediate = self.net(x)
        torch.distributed.all_reduce(
            intermediate, group=tp_group
        )
        return intermediate

class ParallelMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            ColumnParallelLinear(dim, hidden_dim),
            nn.GELU(),
            RowParallelLinear(hidden_dim, dim),
            nn.RMSNorm(dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.heads = heads//TP
        self.norm = nn.RMSNorm(dim)

        self.to_qkv = ColumnParallelLinear(dim, dim * 3)
        self.to_out = RowParallelLinear(dim, dim)

    def forward(self, x):

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b s (h d) -> b h s d', h=self.heads), qkv) #B, hc, s, hs
        

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        out = rearrange(attn_out, 'b h s d -> b s (h d)')
        out = self.to_out(out)
        return self.norm(out)


if __name__ == '__main__':
    seq = 2048
    dim = 128*12
    mlp_dim = dim*2
    heads = 12
    dim_head = dim//heads

    assert dim % TP == 0

    x = torch.randn(1, seq, dim) #B, hc, s, hs
    layer = nn.Sequential(Attention(dim, heads=heads, dim_head=dim_head),ParallelMLP(dim, mlp_dim))

    #print(layer)
    if rank == 0:
        print(f'=> Trainable Params: {sum(p.numel() for p in layer.parameters() if p.requires_grad)}')
    y = layer(x)
    if rank == 0:
        print(y.shape)
