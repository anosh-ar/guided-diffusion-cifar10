"""Small building blocks shared by the U-Net."""
from inspect import isfunction

import torch
from einops.layers.torch import Rearrange
from torch import nn


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num: int, divisor: int) -> list:
    """Split ``num`` into chunks of size ``divisor`` (plus a remainder chunk)."""
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def extract(a: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
    """
    Gather the per-timestep constants ``a[t]`` and reshape them to broadcast over a batch of images.

    Args:
        a: 1-D tensor of length ``timesteps``.
        t: Long tensor of shape ``(B,)`` with timestep indices in ``[0, timesteps - 1]``.
        x_shape: Shape of the batch the result is broadcast against, e.g. ``(B, C, H, W)``.

    Returns:
        Tensor of shape ``(B, 1, 1, 1)`` (for 4-D ``x_shape``) on the device of ``t``.
    """
    batch_size = t.shape[0]
    out = a.to(t.device).gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No more strided convolutions or pooling: fold 2x2 patches into channels
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )
