"""
Generate qualitative figures from a trained checkpoint.

Figures (written to ``--out_dir``, default ``<run>/figures``):

    class_grid.png       one row per CIFAR-10 class
    trajectory.png       reverse diffusion from pure noise (left) to the final sample (right)
    guidance_sweep.png   the same initial noise sampled with increasing guidance weight w (rows)

Example:
    python sample.py --checkpoint runs/<run>/checkpoints/best.pt --guidance_scale 2
"""
import argparse
from pathlib import Path

import torch

from ddpm.data import CIFAR10_CLASSES, NUM_CLASSES
from ddpm.utils import get_device, latest_run, load_checkpoint, seed_everything
from ddpm.visualize import save_labeled_grid

FIGURES = ("grid", "trajectory", "guidance")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Sample figures from a trained DDPM",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", type=str, default=None, help="defaults to best.pt of the latest run in ./runs")
    p.add_argument("--out_dir", type=str, default=None, help="defaults to <run>/figures")
    p.add_argument("--figures", nargs="+", default=list(FIGURES), choices=FIGURES)
    p.add_argument("--guidance_scale", type=float, default=None, help="w for grid / trajectory (default: from config)")
    p.add_argument("--guidance_scales", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0],
                   help="w values for the guidance sweep")
    p.add_argument("--n_per_class", type=int, default=10)
    p.add_argument("--n_trajectories", type=int, default=6)
    p.add_argument("--n_frames", type=int, default=10)
    p.add_argument("--no_ema", action="store_true", help="use raw instead of EMA weights")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args(argv)


def main(args):
    device = get_device(args.no_cuda)
    if args.checkpoint is None:
        run = latest_run()
        if run is None:
            raise SystemExit("No trained run found in ./runs; pass --checkpoint.")
        args.checkpoint = str(run / "checkpoints" / "best.pt")
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, diffusion, config = load_checkpoint(args.checkpoint, device, use_ema=not args.no_ema)
    conditional = not config["unconditional"]
    w = config.get("guidance_scale", 0.0) if args.guidance_scale is None else args.guidance_scale
    print(f"Loaded {ckpt_path} ({config['schedule']} schedule, T={config['timesteps']}, "
          f"{'conditional, w=' + str(w) if conditional else 'unconditional'})")

    if "grid" in args.figures:
        seed_everything(args.seed)
        if conditional:
            classes = torch.arange(NUM_CLASSES, device=device).repeat_interleave(args.n_per_class)
            images = diffusion.sample(model, classes=classes, guidance_scale=w)
            save_labeled_grid(images, args.n_per_class, out_dir / "class_grid.png", row_labels=CIFAR10_CLASSES,
                              title=f"Class-conditional samples (w = {w:g})")
        else:
            images = diffusion.sample(model, batch_size=NUM_CLASSES * args.n_per_class)
            save_labeled_grid(images, args.n_per_class, out_dir / "class_grid.png", title="Unconditional samples")
        print(f"saved {out_dir / 'class_grid.png'}")

    if "trajectory" in args.figures:
        seed_everything(args.seed + 1)
        classes = torch.arange(args.n_trajectories, device=device) % NUM_CLASSES if conditional else None
        _, frames = diffusion.sample(model, batch_size=args.n_trajectories, classes=classes, guidance_scale=w,
                                     return_trajectory=True, num_frames=args.n_frames)
        steps = torch.linspace(diffusion.timesteps - 1, 1, args.n_frames - 1).round().long().tolist()
        col_labels = [f"t={s}" for s in steps] + ["t=0 (out)"]
        # frames: list over time of (B, C, H, W) -> rows = samples, columns = time
        images = torch.stack(frames, dim=1).flatten(0, 1)
        row_labels = [CIFAR10_CLASSES[c] for c in classes.tolist()] if conditional else None
        save_labeled_grid(images, len(frames), out_dir / "trajectory.png", row_labels=row_labels,
                          col_labels=col_labels, title="Reverse diffusion process")
        print(f"saved {out_dir / 'trajectory.png'}")

    if "guidance" in args.figures and conditional:
        classes = torch.arange(NUM_CLASSES, device=device)
        rows = []
        for gw in args.guidance_scales:
            seed_everything(args.seed + 2)  # identical noise for every w, so only guidance changes
            rows.append(diffusion.sample(model, classes=classes, guidance_scale=gw).cpu())
        save_labeled_grid(torch.cat(rows), NUM_CLASSES, out_dir / "guidance_sweep.png",
                          row_labels=[f"w = {gw:g}" for gw in args.guidance_scales], col_labels=CIFAR10_CLASSES,
                          title="Effect of the classifier-free guidance weight")
        print(f"saved {out_dir / 'guidance_sweep.png'}")


if __name__ == "__main__":
    main(parse_args())
