# Class-Conditional DDPM with Classifier-Free Guidance on CIFAR-10

[![tests](https://github.com/anosh-ar/guided-diffusion-cifar10/actions/workflows/tests.yml/badge.svg)](https://github.com/anosh-ar/guided-diffusion-cifar10/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A from-scratch PyTorch diffusion model that **generates new 32×32 images of a chosen class** (airplane, car, bird,
cat, …) from pure noise. It implements DDPM (Ho et al., 2020) with classifier-free guidance (Ho & Salimans, 2022)
and is trained on CIFAR-10.

## Results

**FID 13.97** and **Inception Score 9.11** on CIFAR-10 (10k samples). The model has 10.9M parameters and was trained
for 300 epochs, which took 4.2 h on one RTX 3090.

<p align="center">
  <img src="assets/class_grid.png" width="62%" alt="Generated samples, one row per class">
</p>
<p align="center"><sub>Generated images, one row per class. None of them are in the dataset.</sub></p>

<p align="center">
  <img src="assets/trajectory.png" width="62%" alt="Reverse diffusion from noise to image">
</p>
<p align="center"><sub>From noise (left) to image (right): the model removes a little noise at each of 1000 steps.</sub></p>

## How it works

![pipeline](docs/pipeline.png)

1. **Training:** add a random amount of noise to an image and train a U-Net to predict that noise. The class label
   is hidden 10% of the time, so the network also learns to work without it.
2. **Generation:** start from random noise and let the U-Net remove a little noise, 1000 times. The guidance weight
   `w` sets how strongly the image should match the chosen class.

### Guidance weight

| guidance w | FID ↓ | Inception Score ↑ |
|---:|---:|---:|
| 0 | 15.26 | 8.38 |
| **0.5** | **13.97** | 8.99 |
| 1 | 20.34 | **9.11** |
| 2 | 41.85 | 8.75 |
| 4 | 91.93 | 7.21 |

A little guidance helps (best FID at `w = 0.5`). Too much makes the images over-saturated and less diverse.

<p align="center">
  <img src="assets/guidance_sweep.png" width="60%" alt="Effect of the guidance weight">
  <img src="assets/fid_vs_guidance.png" width="38%" alt="FID and IS vs guidance weight">
</p>

## Quick start

```bash
git clone https://github.com/anosh-ar/guided-diffusion-cifar10.git
cd guided-diffusion-cifar10
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your CUDA version
pip install -r requirements.txt

python train.py --schedule cosine --epochs 300 --amp                     # train (CIFAR-10 downloads automatically)
python sample.py --checkpoint runs/<run>/checkpoints/best.pt             # make figures
python evaluate.py --checkpoint runs/<run>/checkpoints/best.pt --amp     # FID / IS
```

To reproduce all results above with one command (about 9 h on an RTX 3090):

```bash
EPOCHS=300 SCHEDULES="cosine" bash scripts/reproduce.sh
```

Run `pytest -q` for the unit tests. Every script has `-h` for its options (linear / cosine / sigmoid noise
schedules, model size, guidance weight, …).

## Project structure

```
ddpm/            diffusion process, U-Net, data loading, utilities
train.py         training (EMA, mixed precision, checkpoints, metrics)
sample.py        sample grids, denoising trajectory, guidance sweep
evaluate.py      FID / Inception Score over guidance weights
compare_runs.py  compare several runs (loss curves, summary table)
scripts/         reproduce.sh: train, evaluate and collect figures in assets/
docs/            LaTeX (TikZ) source of the pipeline diagram
tests/           unit tests
```

## References

- J. Ho, A. Jain, P. Abbeel. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). NeurIPS 2020.
- J. Ho, T. Salimans. [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). NeurIPS Workshop 2021.
- A. Nichol, P. Dhariwal. [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672). ICML 2021.

## Acknowledgements

The U-Net building blocks are adapted from Phil Wang's
[denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) (MIT), through
[The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion). The project started in the
*Advanced Deep Learning* course at FAU Erlangen-Nürnberg (winter semester 2023/24).

## License

[MIT](LICENSE)
