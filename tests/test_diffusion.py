import pytest
import torch

from ddpm.diffusion import (Diffusion, SCHEDULES, cosine_beta_schedule, get_beta_schedule, linear_beta_schedule,
                            sigmoid_beta_schedule)
from ddpm.layers import extract

T = 50


@pytest.mark.parametrize("name", SCHEDULES)
def test_schedules_are_valid(name):
    betas = get_beta_schedule(name)(T)
    assert betas.shape == (T,)
    assert betas.dtype == torch.float32
    assert torch.all(betas > 0) and torch.all(betas < 1)


def test_linear_schedule_endpoints():
    betas = linear_beta_schedule(1e-4, 0.02, T)
    assert betas[0].item() == pytest.approx(1e-4)
    assert betas[-1].item() == pytest.approx(0.02)


def test_sigmoid_schedule_is_monotone_within_bounds():
    betas = sigmoid_beta_schedule(1e-4, 0.02, T)
    assert torch.all(betas[1:] >= betas[:-1])
    assert betas.min() >= 1e-4 and betas.max() <= 0.02


def test_cosine_schedule_matches_closed_form_alpha_bar():
    s = 0.008
    betas = cosine_beta_schedule(T, s=s).double()
    alphas_cumprod = torch.cumprod(1 - betas, dim=0)
    f = lambda t: torch.cos(((t / T) + s) / (1 + s) * torch.pi / 2) ** 2
    t = torch.arange(1, T + 1, dtype=torch.float64)
    expected = f(t) / f(torch.tensor(0.0, dtype=torch.float64))
    # the last beta is clipped to 0.999, so compare all but the final step
    torch.testing.assert_close(alphas_cumprod[:-1], expected[:-1], rtol=1e-4, atol=1e-6)


def test_extract_broadcast_shape():
    a = torch.arange(T, dtype=torch.float32)
    t = torch.tensor([0, 3, T - 1])
    out = extract(a, t, (3, 3, 8, 8))
    assert out.shape == (3, 1, 1, 1)
    assert out.flatten().tolist() == [0.0, 3.0, T - 1.0]


def test_q_sample_closed_form():
    diffusion = Diffusion(T, get_beta_schedule("linear"), img_size=8, device="cpu")
    x0 = torch.randn(4, 3, 8, 8)
    noise = torch.randn_like(x0)
    t = torch.tensor([0, 10, 25, T - 1])
    alpha_bar = torch.cumprod(1 - diffusion.betas, 0)[t].view(-1, 1, 1, 1)
    expected = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * noise
    torch.testing.assert_close(diffusion.q_sample(x0, t, noise), expected)


def test_posterior_variance_is_zero_at_t0_and_bounded_by_beta():
    diffusion = Diffusion(T, get_beta_schedule("cosine"), img_size=8, device="cpu")
    assert diffusion.posterior_variance[0].item() == pytest.approx(0.0)
    assert torch.all(diffusion.posterior_variance <= diffusion.betas + 1e-7)


class _ConstantModel(torch.nn.Module):
    """Predicts noise 1 for the conditional branch and 0 for the null class."""

    def forward(self, x, t, c=None, drop_mask=None):
        out = torch.ones_like(x)
        if c is None:
            return torch.zeros_like(x)
        if drop_mask is not None:
            out[drop_mask] = 0
        return out


def test_classifier_free_guidance_combination():
    diffusion = Diffusion(T, get_beta_schedule("linear"), img_size=4, device="cpu")
    x = torch.randn(2, 3, 4, 4)
    t = torch.zeros(2, dtype=torch.long)
    classes = torch.tensor([0, 1])
    w = 2.5
    eps = diffusion.predict_noise(_ConstantModel(), x, t, classes, guidance_scale=w)
    torch.testing.assert_close(eps, torch.full_like(x, 1 + w))  # (1 + w) * 1 - w * 0
    eps_uncond = diffusion.predict_noise(_ConstantModel(), x, t, None, guidance_scale=w)
    torch.testing.assert_close(eps_uncond, torch.zeros_like(x))
