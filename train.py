"""
Train a (class-conditional) DDPM on CIFAR-10.

Every run writes to ``<out_dir>/<run_name>/``:

    config.json            training arguments
    metrics.csv            per-epoch train / validation loss, learning rate and wall time
    loss_curve.png         plot of metrics.csv
    samples/epoch_XXX.png  EMA samples (one row per class) during training
    checkpoints/last.pt    latest checkpoint (for --resume)
    checkpoints/best.pt    checkpoint with the lowest validation loss
    summary.json           final numbers, including the test-set loss

Example:
    python train.py --schedule cosine --epochs 100 --amp
"""
import argparse
import datetime
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from ddpm.data import CIFAR10_CLASSES, NUM_CLASSES, get_dataloaders
from ddpm.diffusion import SCHEDULES
from ddpm.utils import (EMA, CSVLogger, build_diffusion, build_model, count_parameters, get_device,
                        plot_loss_curves, save_checkpoint, save_json, seed_everything)
from ddpm.visualize import save_labeled_grid


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train a class-conditional DDPM with classifier-free guidance on CIFAR-10",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data / io
    p.add_argument("--data_dir", type=str, default="./data", help="where CIFAR-10 is (or will be downloaded)")
    p.add_argument("--out_dir", type=str, default="./runs", help="parent directory for run folders")
    p.add_argument("--run_name", type=str, default=None, help="defaults to <timestamp>_<schedule>")
    p.add_argument("--num_workers", type=int, default=4)
    # diffusion
    p.add_argument("--timesteps", type=int, default=1000, help="number of diffusion steps T")
    p.add_argument("--schedule", type=str, default="linear", choices=SCHEDULES, help="beta schedule")
    p.add_argument("--loss", type=str, default="l2", choices=("l1", "l2", "huber"))
    # model
    p.add_argument("--dim", type=int, default=64, help="base channel width of the U-Net")
    p.add_argument("--dim_mults", type=int, nargs="+", default=[1, 2, 4], help="channel multipliers per level")
    p.add_argument("--unconditional", action="store_true", help="train without class conditioning")
    p.add_argument("--p_uncond", type=float, default=0.1,
                   help="probability of dropping the label during training (classifier-free guidance)")
    p.add_argument("--guidance_scale", type=float, default=1.0, help="guidance weight w for preview samples")
    # optimisation
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=1000, help="linear learning-rate warm-up")
    p.add_argument("--grad_clip", type=float, default=1.0, help="max gradient norm (0 disables)")
    p.add_argument("--ema_decay", type=float, default=0.9995)
    p.add_argument("--amp", action="store_true", help="bfloat16 mixed precision (recommended on Ampere GPUs)")
    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_every", type=int, default=10, help="save preview samples every N epochs")
    p.add_argument("--n_per_class", type=int, default=8, help="preview samples per class")
    p.add_argument("--log_interval", type=int, default=100, help="print the running loss every N steps")
    p.add_argument("--resume", type=str, default=None,
                   help="path to a last.pt checkpoint to continue from (writes into the same run directory)")
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="run a couple of batches end-to-end to test the setup")
    return p.parse_args(argv)


def autocast(device: str, enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if enabled and device == "cuda" else nullcontext()


def train_one_epoch(model, ema, loader, optimizer, scheduler, diffusion, device, args, epoch, max_batches=None):
    model.train()
    running, total, n = 0.0, 0.0, 0
    pbar = tqdm(loader, desc=f"epoch {epoch}", leave=False)
    for step, (images, labels) in enumerate(pbar):
        if max_batches is not None and step >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = None if args.unconditional else labels.to(device, non_blocking=True)

        # Algorithm 1, line 3: sample a timestep uniformly for every image
        t = torch.randint(0, diffusion.timesteps, (images.shape[0],), device=device)
        with autocast(device, args.amp):
            loss = diffusion.p_losses(model, images, t, loss_type=args.loss, labels=labels, p_uncond=args.p_uncond)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        ema.update(model)

        loss_val = loss.item()
        total += loss_val * images.shape[0]
        n += images.shape[0]
        running = loss_val if step == 0 else 0.98 * running + 0.02 * loss_val
        pbar.set_postfix(loss=f"{running:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        if step % args.log_interval == 0:
            tqdm.write(f"epoch {epoch} step {step}/{len(loader)}  loss {loss_val:.5f}")
    return total / max(n, 1)


@torch.no_grad()
def evaluate_loss(model, loader, diffusion, device, args, seed=0, max_batches=None):
    """
    Average noise-prediction loss over a dataset.

    Timesteps and noise come from a fixed-seed generator, so the number is comparable across epochs and runs.
    """
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    total, n = 0.0, 0
    for step, (images, labels) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = None if args.unconditional else labels.to(device, non_blocking=True)
        t = torch.randint(0, diffusion.timesteps, (images.shape[0],), device=device, generator=gen)
        noise = torch.randn(images.shape, device=device, generator=gen)
        with autocast(device, args.amp):
            loss = diffusion.p_losses(model, images, t, noise=noise, loss_type=args.loss, labels=labels)
        total += loss.item() * images.shape[0]
        n += images.shape[0]
    return total / max(n, 1)


@torch.no_grad()
def save_preview(model, diffusion, device, args, path: Path, n_per_class: int):
    """Sample a grid with one row per class (or a plain grid for unconditional models)."""
    model.eval()
    gen_state = torch.random.get_rng_state(), (torch.cuda.get_rng_state_all() if device == "cuda" else None)
    torch.manual_seed(args.seed)  # same noise every time, so previews are comparable across epochs
    with autocast(device, args.amp):
        if args.unconditional:
            images = diffusion.sample(model, batch_size=NUM_CLASSES * n_per_class)
            row_labels = None
        else:
            classes = torch.arange(NUM_CLASSES, device=device).repeat_interleave(n_per_class)
            images = diffusion.sample(model, classes=classes, guidance_scale=args.guidance_scale)
            row_labels = CIFAR10_CLASSES
    torch.random.set_rng_state(gen_state[0])
    if gen_state[1] is not None:
        torch.cuda.set_rng_state_all(gen_state[1])
    save_labeled_grid(images, ncol=n_per_class, path=path, row_labels=row_labels)


def main(args):
    seed_everything(args.seed)
    device = get_device(args.no_cuda)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    max_batches = 2 if args.dry_run else None
    if args.dry_run:
        args.epochs, args.n_per_class, args.num_workers = 1, 2, 0

    # ---- run directory -----------------------------------------------------
    if args.resume and args.run_name is None:
        # continue writing into the run the checkpoint belongs to (<out_dir>/<run_name>/checkpoints/last.pt)
        resume_run = Path(args.resume).resolve().parent.parent
        args.out_dir, args.run_name = str(resume_run.parent), resume_run.name
    if args.run_name is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.run_name = f"{stamp}_{args.schedule}" + ("_uncond" if args.unconditional else "")
    run_dir = Path(args.out_dir) / args.run_name
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    config = vars(args).copy()
    save_json(config, run_dir / "config.json")

    # ---- data, model, optimisation -----------------------------------------
    train_loader, val_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size, args.num_workers,
                                                            seed=args.seed)
    model = build_model(config).to(device)
    diffusion = build_diffusion(config, device)
    ema = EMA(model, decay=args.ema_decay)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(args.warmup_steps, 1)))

    start_epoch, step, best_val = 0, 0, math.inf
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, step, best_val = ckpt["epoch"] + 1, ckpt["step"], ckpt["best_val_loss"]
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    print(f"Run: {run_dir}  |  device: {device}  |  parameters: {count_parameters(model) / 1e6:.2f}M  |  "
          f"train/val/test: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")

    logger = CSVLogger(run_dir / "metrics.csv", ["epoch", "step", "train_loss", "val_loss", "lr", "epoch_time_s"],
                       append=bool(args.resume))

    # ---- training loop -----------------------------------------------------
    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, ema, train_loader, optimizer, scheduler, diffusion, device, args, epoch,
                                     max_batches)
        step += min(len(train_loader), max_batches or len(train_loader))
        val_loss = evaluate_loss(ema.model, val_loader, diffusion, device, args, max_batches=max_batches)
        epoch_time = time.time() - t0

        logger.log({"epoch": epoch, "step": step, "train_loss": f"{train_loss:.6f}", "val_loss": f"{val_loss:.6f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.3e}", "epoch_time_s": f"{epoch_time:.1f}"})
        print(f"epoch {epoch:3d} | train {train_loss:.5f} | val (EMA) {val_loss:.5f} | {epoch_time:.0f}s")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(ckpt_dir / "best.pt", model, ema, optimizer, scheduler, epoch, step, best_val, config)
        save_checkpoint(ckpt_dir / "last.pt", model, ema, optimizer, scheduler, epoch, step, best_val, config)

        if epoch % args.sample_every == 0 or epoch == args.epochs - 1:
            save_preview(ema.model, diffusion, device, args, run_dir / "samples" / f"epoch_{epoch:03d}.png",
                         args.n_per_class)

    # ---- final evaluation --------------------------------------------------
    test_loss = evaluate_loss(ema.model, test_loader, diffusion, device, args, max_batches=max_batches)
    plot_loss_curves({args.run_name: run_dir / "metrics.csv"}, run_dir / "loss_curve.png",
                     title=f"DDPM on CIFAR-10 ({args.schedule} schedule)")
    summary = {
        "run_name": args.run_name,
        "schedule": args.schedule,
        "conditional": not args.unconditional,
        "parameters": count_parameters(model),
        "epochs": args.epochs,
        "steps": step,
        "best_val_loss": best_val,
        "test_loss_ema": test_loss,
        "train_time_h": (time.time() - t_start) / 3600,
    }
    save_json(summary, run_dir / "summary.json")
    print(f"Done. Test loss (EMA): {test_loss:.5f}. Results in {run_dir}")
    return run_dir


if __name__ == "__main__":
    main(parse_args())
