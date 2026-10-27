"""Plot the beta schedules and the resulting signal level alpha_bar_t used for training."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ddpm.diffusion import SCHEDULES, get_beta_schedule


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--out", type=str, default="assets/beta_schedules.png")
    args = p.parse_args()

    fig, (ax_beta, ax_abar) = plt.subplots(1, 2, figsize=(11, 4))
    for name in SCHEDULES:
        betas = get_beta_schedule(name)(args.timesteps)
        alphas_cumprod = torch.cumprod(1 - betas, dim=0)
        ax_beta.plot(betas.numpy(), label=name)
        ax_abar.plot(alphas_cumprod.numpy(), label=name)

    ax_beta.set_title(r"Noise schedule $\beta_t$")
    ax_beta.set_yscale("log")
    ax_abar.set_title(r"Signal level $\bar\alpha_t = \prod_{s \leq t} (1 - \beta_s)$")
    for ax in (ax_beta, ax_abar):
        ax.set_xlabel("timestep t")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
