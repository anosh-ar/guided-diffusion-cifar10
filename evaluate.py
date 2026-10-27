"""
Quantitative evaluation: FID and Inception Score of generated samples, swept over the guidance weight w.

Generated images are class-balanced (``num_samples / 10`` per class). FID is computed against the CIFAR-10
test set (10k images) by default, using the 2048-d pool features of InceptionV3 (torchmetrics / torch-fidelity).

Outputs (in ``--out_dir``, default ``<run>/eval``):

    results.json           one entry per w: FID, IS mean/std, sampling time
    results.md             the same as a Markdown table
    fid_vs_guidance.png    FID and IS as a function of w

Example:
    python evaluate.py --checkpoint runs/<run>/checkpoints/best.pt --num_samples 10000 --amp
"""
import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from ddpm.data import NUM_CLASSES, to_unit_range
from ddpm.utils import get_device, latest_run, load_checkpoint, save_json, seed_everything


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="FID / IS evaluation of a trained DDPM",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", type=str, default=None, help="defaults to best.pt of the latest run in ./runs")
    p.add_argument("--out_dir", type=str, default=None, help="defaults to <run>/eval")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--real_split", type=str, default="test", choices=("train", "test"),
                   help="CIFAR-10 split used as reference distribution for FID")
    p.add_argument("--num_samples", type=int, default=10000, help="generated images per guidance weight")
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--guidance_scales", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--amp", action="store_true", help="bfloat16 autocast while sampling")
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args(argv)


def write_markdown(results, path: Path, config) -> None:
    lines = [f"Schedule: `{config['schedule']}`, T = {config['timesteps']}, "
             f"{results[0]['num_samples']} samples per row, FID reference: CIFAR-10 {results[0]['real_split']} set.",
             "",
             "| guidance w | FID ↓ | Inception Score ↑ | sampling time |",
             "|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['guidance_scale']:g} | {r['fid']:.2f} | {r['is_mean']:.2f} ± {r['is_std']:.2f} "
                     f"| {r['sampling_time_min']:.1f} min |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(results, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ws = [r["guidance_scale"] for r in results]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(ws, [r["fid"] for r in results], "o-", color="C0", label="FID")
    ax1.set_xlabel("guidance weight w")
    ax1.set_ylabel("FID (lower is better)", color="C0")
    ax2 = ax1.twinx()
    ax2.errorbar(ws, [r["is_mean"] for r in results], yerr=[r["is_std"] for r in results], fmt="s--",
                 color="C1", capsize=3, label="IS")
    ax2.set_ylabel("Inception Score (higher is better)", color="C1")
    ax1.grid(alpha=0.3)
    ax1.set_title("Sample quality vs. classifier-free guidance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(args):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    device = get_device(args.no_cuda)
    if args.checkpoint is None:
        run = latest_run()
        if run is None:
            raise SystemExit("No trained run found in ./runs; pass --checkpoint.")
        args.checkpoint = str(run / "checkpoints" / "best.pt")
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, diffusion, config = load_checkpoint(args.checkpoint, device, use_ema=not args.no_ema)
    conditional = not config["unconditional"]
    guidance_scales = args.guidance_scales if conditional else [0.0]
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if args.amp and device == "cuda" else nullcontext()

    # ---- reference statistics (computed once, kept across guidance weights) ----
    fid = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False).to(device)
    real_set = datasets.CIFAR10(args.data_dir, train=args.real_split == "train", download=True,
                                transform=transforms.ToTensor())
    for images, _ in tqdm(DataLoader(real_set, batch_size=args.batch_size, num_workers=4), desc="real features"):
        fid.update(images.to(device), real=True)

    results = []
    for w in guidance_scales:
        fid.reset()
        inception = InceptionScore(normalize=True).to(device)
        seed_everything(args.seed)
        t0 = time.time()
        for start in tqdm(range(0, args.num_samples, args.batch_size), desc=f"w = {w:g}"):
            n = min(args.batch_size, args.num_samples - start)
            classes = (torch.arange(start, start + n, device=device) % NUM_CLASSES) if conditional else None
            with amp:
                images = diffusion.sample(model, batch_size=n, classes=classes, guidance_scale=w, progress=False)
            images = to_unit_range(images.float())
            fid.update(images, real=False)
            inception.update(images)
        is_mean, is_std = inception.compute()
        results.append({
            "guidance_scale": w,
            "fid": fid.compute().item(),
            "is_mean": is_mean.item(),
            "is_std": is_std.item(),
            "num_samples": args.num_samples,
            "real_split": args.real_split,
            "sampling_time_min": (time.time() - t0) / 60,
        })
        print(f"w = {w:g}: FID {results[-1]['fid']:.2f} | IS {is_mean:.2f} ± {is_std:.2f}")
        save_json({"checkpoint": str(ckpt_path), "config": config, "results": results}, out_dir / "results.json")

    write_markdown(results, out_dir / "results.md", config)
    if len(results) > 1:
        plot_results(results, out_dir / "fid_vs_guidance.png")
    print(f"Results written to {out_dir}")


if __name__ == "__main__":
    main(parse_args())
