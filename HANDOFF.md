# Handoff notes for the next assistant

> Internal working notes, pushed so they reach the workstation. **Delete this file once the README has real results.**

## Goal

Make this repo a complete, professional GitHub project: a class-conditional DDPM with classifier-free guidance on
CIFAR-10, with real results in the README. The code is finished and tested on CPU. What's left is to **run it on
the workstation (RTX 3090)**, put the results into the README, then commit and push.

## Background

- This started as Exercise 2 of the *Advanced Deep Learning* course at FAU (WS 2023/24). The git history includes
  commits by course staff ("Leonard Fischer", "--replace-all"). Those old commits only exist on the user's laptop
  and the course GitLab (`gitlab.cs.fau.de`).
- **Public repo: https://github.com/anosh-ar/guided-diffusion-cifar10** (branch `main`, MIT license). It has a
  **fresh history**: GitHub's initial commit, then one commit with this whole project. On the workstation just
  `git clone` it and work and push on `main` normally.
- **What may be public:** saying the project started in the *Advanced Deep Learning* course at FAU (winter semester
  2023/24) is fine; the README's Acknowledgements section already says so. What must **not** be published is the
  course's exercise sheet (`exercise-02.pdf`) and the other course-provided files (`ex02_test_values.pt`).
- **Never push the old laptop history** (branches `dev`/`main` of the original repo). Old commits contain
  `exercise-02.pdf` and the course test values. This doesn't matter on the workstation, where you work from the
  GitHub clone.
- Course material stays local and is gitignored: `exercise-02.pdf` (all `*.pdf` except `docs/pipeline.pdf`) and
  `tests/data/`.

## What was changed (vs. the original exercise code)

The code was restructured into the `ddpm/` package (`git mv`, so history is kept). Bugs fixed:

1. **Cosine schedule** returned cos² (that's ᾱ, not β) in float64, which crashed training. It now uses the
   Nichol & Dhariwal formula and returns float32.
2. **Classifier-free guidance null class**: training zeroed the embedding, but sampling used class index 0
   ("airplane") as "unconditional". There is now a learned null token at index `num_classes`, used via
   `drop_mask` in both training and sampling. Both CFG branches run in one batched forward pass.
3. **Timestep off-by-one** between training (t ∈ [0, T−1]) and sampling (the model was given t = T..1). Now
   0-indexed everywhere.
4. `classes=None` (unconditional) crashed. It works now, and `--unconditional` trains without labels.
5. `ex02_main.py` problems: output directories were never created, which crashed sampling and saving; `args.epochs`
   was overridden; the run name contained `:`; the dataset path was cluster-only; no `no_grad` in the test loop;
   lr 3e-3. All fixed.

Added:
- **Training:** EMA, warm-up, gradient clipping, bf16 `--amp`, `--resume`, and a fixed-seed validation and test
  loss.
- **Outputs:** CSV metrics, loss curve, preview samples, `summary.json`.
- **Scripts:** `sample.py` (class grid, trajectory, guidance sweep), `evaluate.py` (FID + IS via torchmetrics,
  swept over w), `compare_runs.py`, `plot_schedules.py`.
- **Tooling:** `scripts/reproduce.sh`, `scripts/run_in_tmux.sh`, the pytest suite (18 tests), a GitHub Actions
  CI, `pyproject.toml`, `requirements.txt`, MIT `LICENSE` (with the lucidrains attribution), and the README.
- **Diagram:** `docs/pipeline.tex` (TikZ standalone). It is compiled to `docs/pipeline.pdf` and `docs/pipeline.png`,
  and the PNG is shown in the README. Rebuild with `cd docs && pdflatex pipeline.tex`.

## What was verified (Windows, CPU)

- `pytest -q`: all 18 tests pass.
- `python train.py --dry_run --timesteps 20 --batch_size 16`: runs end to end and writes every expected file.
- `sample.py` on the dry-run checkpoint produced all three figures.
- **Not verified yet:** `evaluate.py` (the run was stopped while it downloaded the InceptionV3 weights),
  `compare_runs.py`, `reproduce.sh` on Linux, and any real GPU training. Watch the first runs of these.

## Next steps on the workstation

1. Set up the environment:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match the CUDA driver
   pip install -r requirements.txt
   pytest -q
   python train.py --dry_run --timesteps 50                                             # ~1 min sanity check
   ```
2. Smoke-test the evaluation path before the long run (fixes any FID/IS issue cheaply):
   ```bash
   python evaluate.py --checkpoint runs/<dry-run>/checkpoints/best.pt --num_samples 500 --guidance_scales 0 1 --amp
   python compare_runs.py runs/<dry-run>
   ```
   Delete the dry-run folders from `runs/` afterwards. `reproduce.sh` uses fixed names `runs/cifar10_<schedule>`.
3. Run the full pipeline detached:
   ```bash
   scripts/run_in_tmux.sh ddpm bash scripts/reproduce.sh
   ```
   - Defaults: 100 epochs × 3 schedules (linear, cosine, sigmoid), then FID/IS on 10k samples for
     w ∈ {0, 0.5, 1, 2, 4} per model, then figures for the best run. Everything is collected in `assets/`.
   - Wall time on a 3090 hasn't been measured; expect many hours. A quicker variant:
     `EPOCHS=50 SCHEDULES="cosine" bash scripts/reproduce.sh`.
   - The script is idempotent: it skips finished steps and resumes from `last.pt`.
4. Look at `runs/*/samples/epoch_*.png`. Objects should be recognisable by roughly epoch 20–30. If samples stay
   noisy or blurry, check that the loss is decreasing in `metrics.csv` before changing anything.
5. Update the README with the real numbers:
   - Replace the two placeholder tables (marked `<!-- paste ... -->`) with `assets/results.md` and
     `assets/schedule_comparison.md`.
   - Check that every image the README references exists in `assets/`: `class_grid.png`, `guidance_sweep.png`,
     `trajectory.png`, `fid_vs_guidance.png`, `loss_curve.png`, `beta_schedules.png`.
   - Adjust the qualitative statements ("small w gives the best FID", "cosine usually works better") so they match
     what was actually measured.
   - Optionally add the hardware and training time from `runs/*/summary.json`.
6. Publish:
   - Commit `assets/` and the README update, and delete `HANDOFF.md` in the same commit (commits in this repo end
     with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`).
   - Push to `origin main` (the GitHub clone). Ask the user before pushing.
   - Optionally attach `runs/cifar10_<best>/checkpoints/best.pt` (≈45 MB with EMA and optimizer state) as a GitHub
     Release asset. `*.pt` files are gitignored.

## Design notes / gotchas

- Timesteps are 0-indexed (`t ∈ [0, T−1]`) everywhere. `p_sample(model, x, t_index, ...)` takes an int.
- Guidance: `eps = (1+w)·eps_cond − w·eps_uncond`. `w=0` means plain conditional sampling. The default for previews
  is `--guidance_scale 1.0`, which is stored in the checkpoint config.
- `Unet.forward(x, t, c=None, drop_mask=None)`: `c=None` means null class. Passing labels to a model built with
  `num_classes=None` raises a `ValueError`.
- Checkpoints store `config` (the training args). `ddpm.utils.load_checkpoint` rebuilds the model and diffusion
  from it and loads the EMA weights by default (`--no_ema` for raw weights).
- Model: `dim=64, dim_mults=(1,2,4)` gives ≈10.9M parameters. The reverse process uses the posterior variance β̃ₜ.
- Validation and test loss use a fixed-seed generator for t and noise, so the numbers are comparable across epochs
  and runs. Validation is computed with the EMA model.
- On Windows, MiKTeX may pop up a package-install dialog the first time `pdflatex` runs.
