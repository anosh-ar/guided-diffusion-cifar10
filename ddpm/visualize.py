"""Figures: labelled image grids for samples, denoising trajectories and guidance sweeps."""
from pathlib import Path
from typing import Optional, Sequence

import torch
from torchvision.utils import make_grid

from .data import to_unit_range


def save_labeled_grid(images: torch.Tensor, ncol: int, path: Path, row_labels: Optional[Sequence[str]] = None,
                      col_labels: Optional[Sequence[str]] = None, title: Optional[str] = None,
                      cell_inches: float = 0.6) -> None:
    """
    Save a grid of images (row-major, ``ncol`` images per row) with optional row / column labels.

    Args:
        images: Tensor of shape ``(N, C, H, W)`` in ``[-1, 1]``.
        ncol (int): Images per row.
        path: Output file (PNG / PDF / ...).
        row_labels, col_labels: Optional text for each row / column.
        title: Optional figure title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    padding = 2
    images = to_unit_range(images.detach().float().cpu())
    grid = make_grid(images, nrow=ncol, padding=padding, pad_value=1.0).permute(1, 2, 0).numpy()
    nrow = (images.shape[0] + ncol - 1) // ncol
    h, w = images.shape[-2:]

    fig, ax = plt.subplots(figsize=(ncol * cell_inches + 1.5, nrow * cell_inches + 0.8))
    ax.imshow(grid, interpolation="nearest")
    ax.set_xticks([padding + (w + padding) * i + w / 2 for i in range(ncol)] if col_labels else [])
    ax.set_yticks([padding + (h + padding) * i + h / 2 for i in range(nrow)] if row_labels else [])
    if col_labels:
        ax.set_xticklabels(col_labels, rotation=45, ha="left", fontsize=8)
        ax.xaxis.tick_top()
    if row_labels:
        ax.set_yticklabels(row_labels, fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, pad=40 if col_labels else 6)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
