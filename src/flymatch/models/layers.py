"""Directed GNN layer using sparse-dense matmul (plan.md 7.3.2)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DirLayer(nn.Module):
    """h_i <- h_i + GELU(LN(W_s h_i + W_in · mean_inputs(h) + W_out · mean_outputs(h)))"""

    def __init__(self, d: int, p: float = 0.1):
        super().__init__()
        self.lin_self = nn.Linear(d, d)
        self.lin_in = nn.Linear(d, d, bias=False)
        self.lin_out = nn.Linear(d, d, bias=False)
        self.norm, self.drop = nn.LayerNorm(d), nn.Dropout(p)

    def forward(self, h, A_in, A_out):
        m = (self.lin_self(h)
             + self.lin_in(torch.sparse.mm(A_in, h))
             + self.lin_out(torch.sparse.mm(A_out, h)))
        return h + self.drop(F.gelu(self.norm(m)))
