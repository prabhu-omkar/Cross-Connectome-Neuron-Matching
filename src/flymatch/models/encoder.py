"""Shared-weight encoder (plan.md 7.3.2). The SAME instance embeds both datasets.
layers = 0 is the MLP baseline B4 with everything else identical."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DirLayer


class Encoder(nn.Module):
    def __init__(self, f_in: int, d: int = 256, out: int = 128, layers: int = 2, p: float = 0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(f_in, d), nn.GELU(), nn.LayerNorm(d))
        self.layers = nn.ModuleList([DirLayer(d, p) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(d * (layers + 1), d), nn.GELU(), nn.Linear(d, out))

    def forward(self, x, A_in=None, A_out=None):
        h = self.inp(x)
        hs = [h]
        for layer in self.layers:
            h = layer(h, A_in, A_out)
            hs.append(h)
        return F.normalize(self.head(torch.cat(hs, dim=1)), dim=1)   # jumping-knowledge concat

    @torch.no_grad()
    def embed(self, x, A_in=None, A_out=None, chunk: int = 65536):
        """Inference; for layers = 0 runs in chunks (no graph needed)."""
        self.eval()
        if len(self.layers) == 0:
            return torch.cat([self(x[i:i + chunk]) for i in range(0, len(x), chunk)])
        return self(x, A_in, A_out)
