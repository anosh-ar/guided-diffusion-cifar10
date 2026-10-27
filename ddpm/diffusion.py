"""Noise schedules and the DDPM forward / reverse process with classifier-free guidance."""
import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .layers import extract


# ---------------------------------------------------------------------------
# Beta schedules
# ---------------------------------------------------------------------------

def linear_beta_schedule(beta_start: float, beta_end: float, timesteps: int) -> torch.Tensor:
    """
    Linear beta schedule between ``beta_start`` and ``beta_end`` (Ho et al., 2020).

    Returns:
        torch.Tensor: Float32 tensor of shape ``(timesteps,)``.
    """
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule of Nichol & Dhariwal (2021), "Improved Denoising Diffusion Probabilistic Models".

    The cumulative product ``alpha_bar(t) = f(t) / f(0)`` with ``f(t) = cos^2(((t / T) + s) / (1 + s) * pi / 2)``
    is defined first; the betas are recovered via ``beta_t = 1 - alpha_bar(t) / alpha_bar(t - 1)`` and clipped
    to 0.999 to avoid singularities close to ``t = T``.

    Args:
        timesteps (int): Number of diffusion steps ``T``.
        s (float): Small offset that keeps ``beta_t`` from being too small near ``t = 0``.

    Returns:
        torch.Tensor: Float32 tensor of shape ``(timesteps,)``.
    """
    steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999).float()


def sigmoid_beta_schedule(beta_start: float, beta_end: float, timesteps: int, s_limit: float = 6) -> torch.Tensor:
    """
    Sigmoid schedule: starts at ``beta_start`` and smoothly transitions to ``beta_end``.

    Args:
        beta_start (float): The starting value of beta.
        beta_end (float): The ending value of beta.
        timesteps (int): Number of diffusion steps.
        s_limit (float): The sigmoid is evaluated on ``[-s_limit, s_limit]``; larger values give a sharper transition.

    Returns:
        torch.Tensor: Float32 tensor of shape ``(timesteps,)``.
    """
    steps = torch.linspace(-s_limit, s_limit, timesteps, dtype=torch.float32)
    return beta_start + (beta_end - beta_start) * torch.sigmoid(steps)


SCHEDULES = ("linear", "cosine", "sigmoid")


def get_beta_schedule(name: str, beta_start: float = 1e-4, beta_end: float = 0.02) -> Callable[[int], torch.Tensor]:
    """Return a ``timesteps -> betas`` callable for the schedule called ``name``."""
    if name == "linear":
        return lambda T: linear_beta_schedule(beta_start, beta_end, T)
    if name == "cosine":
        return lambda T: cosine_beta_schedule(T)
    if name == "sigmoid":
        return lambda T: sigmoid_beta_schedule(beta_start, beta_end, T)
    raise ValueError(f"Unknown schedule '{name}', expected one of {SCHEDULES}")


# ---------------------------------------------------------------------------
# Diffusion process
# ---------------------------------------------------------------------------

class Diffusion:
    """
    Forward (noising) and reverse (denoising) process of a DDPM.

    Timesteps are 0-indexed everywhere: ``t = 0`` is the first noising step and ``t = timesteps - 1`` the last,
    both during training and sampling.

    Classifier-free guidance (Ho & Salimans, 2022) combines conditional and unconditional noise predictions as
    ``eps = (1 + w) * eps(x_t, c) - w * eps(x_t, null)``. ``w = 0`` is plain conditional sampling.

    Attributes:
        timesteps (int): Number of diffusion steps ``T``.
        img_size (int): Spatial size of the (square) images.
        device (str): Device the precomputed constants live on.
        guidance_scale (float): Default guidance weight ``w`` used by :meth:`sample`.
        betas (torch.Tensor): Beta schedule of shape ``(T,)``.
    """

    def __init__(self, timesteps: int, get_noise_schedule: Callable[[int], torch.Tensor], img_size: int,
                 device: Optional[str] = None, guidance_scale: float = 0.0):
        """
        Args:
            timesteps (int): Number of diffusion steps.
            get_noise_schedule (callable): Maps ``timesteps`` to a beta tensor of shape ``(timesteps,)``.
            img_size (int): Spatial size of the (square) images.
            device (str, optional): Defaults to CUDA when available.
            guidance_scale (float): Default classifier-free guidance weight ``w``.
        """
        self.timesteps = timesteps
        self.img_size = img_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.guidance_scale = guidance_scale

        betas = get_noise_schedule(timesteps).float()
        if betas.shape != (timesteps,):
            raise ValueError(f"Noise schedule must have shape ({timesteps},), got {tuple(betas.shape)}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        to_dev = lambda x: x.to(self.device)
        self.betas = to_dev(betas)
        self.alphas_cumprod = to_dev(alphas_cumprod)

        # forward process q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)
        self.sqrt_alphas_cumprod = to_dev(alphas_cumprod.sqrt())
        self.sqrt_one_minus_alphas_cumprod = to_dev((1.0 - alphas_cumprod).sqrt())

        # reverse process mean: 1/sqrt(alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps)
        self.sqrt_recip_alphas = to_dev((1.0 / alphas).sqrt())
        self.eps_coef = to_dev(betas / (1.0 - alphas_cumprod).sqrt())

        # variance of the posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = to_dev(betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    # -- forward process -----------------------------------------------------

    def q_sample(self, x_zero: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample ``x_t ~ q(x_t | x_0)`` in closed form.

        Args:
            x_zero: Clean images of shape ``(B, C, H, W)`` scaled to ``[-1, 1]``.
            t: Long tensor of shape ``(B,)`` with timesteps in ``[0, T - 1]``.
            noise: Optional Gaussian noise of the same shape as ``x_zero``.

        Returns:
            torch.Tensor: The noised images ``x_t``.
        """
        if noise is None:
            noise = torch.randn_like(x_zero)
        return (extract(self.sqrt_alphas_cumprod, t, x_zero.shape) * x_zero
                + extract(self.sqrt_one_minus_alphas_cumprod, t, x_zero.shape) * noise)

    def p_losses(self, denoise_model, x_zero: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None,
                 loss_type: str = "l2", labels: Optional[torch.Tensor] = None, p_uncond: float = 0.0) -> torch.Tensor:
        """
        Noise-prediction loss of Algorithm 1 in Ho et al. (2020).

        Args:
            denoise_model: Network predicting the noise ``eps`` from ``(x_t, t, c)``.
            x_zero: Clean images of shape ``(B, C, H, W)``.
            t: Timesteps of shape ``(B,)``.
            noise: Optional noise; sampled if ``None``.
            loss_type (str): ``'l1'``, ``'l2'`` or ``'huber'``.
            labels: Optional class labels of shape ``(B,)``. ``None`` trains unconditionally.
            p_uncond (float): Probability of replacing a label with the null class (classifier-free guidance).

        Returns:
            torch.Tensor: Scalar loss.
        """
        if noise is None:
            noise = torch.randn_like(x_zero)

        x_noisy = self.q_sample(x_zero, t, noise)

        drop_mask = None
        if labels is not None and p_uncond > 0:
            drop_mask = torch.rand(labels.shape[0], device=labels.device) < p_uncond

        predicted_noise = denoise_model(x_noisy, t, labels, drop_mask=drop_mask)

        if loss_type == "l1":
            return F.l1_loss(predicted_noise, noise)
        if loss_type == "l2":
            return F.mse_loss(predicted_noise, noise)
        if loss_type == "huber":
            return F.smooth_l1_loss(predicted_noise, noise)
        raise ValueError(f"Unknown loss type '{loss_type}'")

    # -- reverse process -----------------------------------------------------

    def predict_noise(self, model, x: torch.Tensor, t: torch.Tensor, classes: Optional[torch.Tensor],
                      guidance_scale: float) -> torch.Tensor:
        """Noise prediction, using classifier-free guidance when ``classes`` is given and ``guidance_scale != 0``."""
        if classes is None or guidance_scale == 0:
            return model(x, t, classes)

        # evaluate conditional and unconditional branch in a single batched forward pass
        batch_size = x.shape[0]
        drop_mask = torch.cat([torch.zeros(batch_size, dtype=torch.bool, device=x.device),
                               torch.ones(batch_size, dtype=torch.bool, device=x.device)])
        eps = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([classes, classes]), drop_mask=drop_mask)
        eps_cond, eps_uncond = eps.chunk(2)
        return (1 + guidance_scale) * eps_cond - guidance_scale * eps_uncond

    @torch.no_grad()
    def p_sample(self, model, x: torch.Tensor, t_index: int, classes: Optional[torch.Tensor] = None,
                 guidance_scale: Optional[float] = None) -> torch.Tensor:
        """
        One reverse step ``x_t -> x_{t-1}`` (Algorithm 2, lines 3-4, of Ho et al., 2020).

        Args:
            model: The noise-prediction network.
            x: Noisy images ``x_t`` of shape ``(B, C, H, W)``.
            t_index (int): Current timestep in ``[0, T - 1]``.
            classes: Optional class labels of shape ``(B,)``; ``None`` samples unconditionally.
            guidance_scale (float, optional): Guidance weight ``w``; defaults to ``self.guidance_scale``.

        Returns:
            torch.Tensor: The images ``x_{t-1}``.
        """
        w = self.guidance_scale if guidance_scale is None else guidance_scale
        t = torch.full((x.shape[0],), t_index, device=x.device, dtype=torch.long)

        eps = self.predict_noise(model, x, t, classes, w).float()
        model_mean = extract(self.sqrt_recip_alphas, t, x.shape) * (x - extract(self.eps_coef, t, x.shape) * eps)

        if t_index == 0:
            # no noise is added in the final step
            return model_mean
        noise = torch.randn_like(x)
        return model_mean + extract(self.posterior_variance, t, x.shape).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, image_size: Optional[int] = None, batch_size: int = 16, channels: int = 3,
               classes: Optional[torch.Tensor] = None, guidance_scale: Optional[float] = None,
               return_trajectory: bool = False, num_frames: int = 10, progress: bool = True,
               ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Full reverse diffusion from Gaussian noise to images (Algorithm 2 of Ho et al., 2020).

        Args:
            model: The noise-prediction network.
            image_size (int, optional): Spatial size; defaults to ``self.img_size``.
            batch_size (int): Number of images; ignored if ``classes`` is given.
            channels (int): Number of image channels.
            classes: Optional class labels of shape ``(B,)``.
            guidance_scale (float, optional): Guidance weight ``w``; defaults to ``self.guidance_scale``.
            return_trajectory (bool): Additionally return intermediate images.
            num_frames (int): Number of intermediate images (evenly spaced in ``t``) when ``return_trajectory``.
            progress (bool): Show a progress bar.

        Returns:
            Images of shape ``(B, C, H, W)`` in ``[-1, 1]``, and optionally a list of intermediate images
            (from pure noise to the final sample).
        """
        image_size = image_size or self.img_size
        if classes is not None:
            batch_size = classes.shape[0]
            classes = classes.to(self.device)

        x = torch.randn(batch_size, channels, image_size, image_size, device=self.device)

        # snapshot x_t before these steps; together with the final sample this gives `num_frames` frames
        frame_steps = set(torch.linspace(self.timesteps - 1, 1, max(num_frames - 1, 1)).round().long().tolist())
        frames = []
        for t_index in tqdm(reversed(range(self.timesteps)), total=self.timesteps, desc="sampling",
                            leave=False, disable=not progress):
            if return_trajectory and t_index in frame_steps:
                frames.append(x.clamp(-1, 1).cpu())
            x = self.p_sample(model, x, t_index, classes, guidance_scale)

        x = x.clamp(-1, 1)
        if return_trajectory:
            frames.append(x.cpu())
            return x, frames
        return x
