"""Losses (plan.md 7.3.3-7.3.4)."""
import torch


def multi_pos_infonce(za, zb, pos, log_tau, valid=None):
    """za: [na, d], zb: [nb, d] unit vectors; pos: bool [na, nb]; rows/cols with no positive are
    ignored as anchors. valid: optional bool [na, nb]; False entries are removed from both
    softmax denominators (e.g. same type, other side — neither positive nor negative)."""
    tau = log_tau.exp().clamp(0.01, 1.0)
    logits = za @ zb.T / tau
    if valid is not None:
        logits = logits.masked_fill(~(valid | pos), float("-inf"))
    lp_ab = logits - logits.logsumexp(dim=1, keepdim=True)
    lp_ba = logits.T - logits.T.logsumexp(dim=1, keepdim=True)

    def side(lp, P):
        n = P.sum(1)
        keep = n > 0
        if not keep.any():
            return lp.sum() * 0
        return (-(lp.masked_fill(~P.bool(), 0) * P).sum(1)[keep] / n[keep]).mean()

    return 0.5 * (side(lp_ab, pos.float()) + side(lp_ba, pos.T.float()))


def coral(za, zb):
    """CORAL: ||Cov(Za) - Cov(Zb)||_F^2 / (4 d^2)."""
    d = za.shape[1]

    def cov(z):
        z = z - z.mean(0, keepdim=True)
        return z.T @ z / max(len(z) - 1, 1)
    return ((cov(za) - cov(zb)) ** 2).sum() / (4 * d * d)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None
