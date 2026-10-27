"""Training utilities: seeding, EMA, checkpoints, model construction and plotting."""
import copy
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .data import CHANNELS, IMAGE_SIZE, NUM_CLASSES
from .diffusion import Diffusion, get_beta_schedule
from .model import Unet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(no_cuda: bool = False) -> str:
    return "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"


class EMA:
    """
    Exponential moving average of model weights, which gives noticeably better samples for diffusion models.

    The effective decay is warmed up as ``min(decay, (1 + step) / (10 + step))`` so that the average is not
    dominated by the random initialisation early in training.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = decay
        self.step = 0
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        decay = min(self.decay, (1 + self.step) / (10 + self.step))
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.lerp_(p.detach(), 1 - decay)
        for ema_b, b in zip(self.model.buffers(), model.buffers()):
            ema_b.copy_(b)

    def state_dict(self) -> Dict:
        return {"model": self.model.state_dict(), "step": self.step, "decay": self.decay}

    def load_state_dict(self, state: Dict) -> None:
        self.model.load_state_dict(state["model"])
        self.step = state["step"]
        self.decay = state["decay"]


# ---------------------------------------------------------------------------
# Model / diffusion construction from a config dict (the training args)
# ---------------------------------------------------------------------------

def build_model(config: Dict) -> Unet:
    return Unet(
        dim=config["dim"],
        dim_mults=tuple(config["dim_mults"]),
        channels=CHANNELS,
        num_classes=None if config["unconditional"] else NUM_CLASSES,
    )


def build_diffusion(config: Dict, device: str) -> Diffusion:
    return Diffusion(
        timesteps=config["timesteps"],
        get_noise_schedule=get_beta_schedule(config["schedule"]),
        img_size=IMAGE_SIZE,
        device=device,
        guidance_scale=config.get("guidance_scale", 0.0),
    )


def save_checkpoint(path: Path, model: nn.Module, ema: EMA, optimizer, scheduler, epoch: int, step: int,
                    best_val_loss: float, config: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val_loss,
        "config": config,
    }, path)


def load_checkpoint(path: str, device: str, use_ema: bool = True) -> Tuple[Unet, Diffusion, Dict]:
    """Load a trained model (EMA weights by default) and its diffusion process for inference."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["ema"]["model"] if use_ema else ckpt["model"])
    model.eval()
    return model, build_diffusion(config, device), config


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Logging and plotting
# ---------------------------------------------------------------------------

class CSVLogger:
    """Appends one row of metrics per call to a CSV file."""

    def __init__(self, path: Path, fieldnames: Sequence[str], append: bool = False):
        self.path = path
        self.fieldnames = list(fieldnames)
        if not (append and path.exists()):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def log(self, row: Dict) -> None:
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)


def read_csv(path: Path) -> List[Dict[str, float]]:
    with open(path, newline="") as f:
        return [{k: float(v) for k, v in row.items() if v != ""} for row in csv.DictReader(f)]


def save_json(obj, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def plot_loss_curves(runs: Dict[str, Path], out_path: Path, title: str = "Training curves") -> None:
    """Plot train / validation loss per epoch for one or more runs (``label -> metrics.csv``)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (label, csv_path) in enumerate(runs.items()):
        rows = read_csv(csv_path)
        epochs = [r["epoch"] for r in rows]
        color = f"C{i}"
        prefix = f"{label} " if len(runs) > 1 else ""
        ax.plot(epochs, [r["train_loss"] for r in rows], color=color, linestyle="--", alpha=0.7,
                label=f"{prefix}train")
        ax.plot(epochs, [r["val_loss"] for r in rows], color=color, label=f"{prefix}val (EMA)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("noise-prediction MSE")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def latest_run(out_dir: str = "runs") -> Optional[Path]:
    runs = [p for p in Path(out_dir).glob("*") if (p / "checkpoints" / "best.pt").exists()]
    return max(runs, key=os.path.getmtime) if runs else None
