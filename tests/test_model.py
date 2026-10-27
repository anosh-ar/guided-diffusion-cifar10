import pytest
import torch

from ddpm.diffusion import Diffusion, get_beta_schedule
from ddpm.model import Unet
from ddpm.utils import EMA

T = 10
IMG = 16


@pytest.fixture
def model():
    torch.manual_seed(0)
    return Unet(dim=16, dim_mults=(1, 2), num_classes=10).eval()


@pytest.fixture
def diffusion():
    return Diffusion(T, get_beta_schedule("linear"), img_size=IMG, device="cpu")


def test_output_shape(model):
    x = torch.randn(3, 3, IMG, IMG)
    t = torch.randint(0, T, (3,))
    assert model(x, t, torch.tensor([0, 5, 9])).shape == x.shape
    assert model(x, t).shape == x.shape


def test_null_class_equals_dropped_condition(model):
    x = torch.randn(2, 3, IMG, IMG)
    t = torch.tensor([1, 7])
    c = torch.tensor([2, 3])
    with torch.no_grad():
        uncond = model(x, t, None)
        dropped = model(x, t, c, drop_mask=torch.tensor([True, True]))
        cond = model(x, t, c)
    torch.testing.assert_close(uncond, dropped)
    # the null class must not coincide with any real class (in particular class 0)
    with torch.no_grad():
        class0 = model(x, t, torch.zeros_like(c))
    assert not torch.allclose(uncond, class0)
    assert not torch.allclose(uncond, cond)


def test_unconditional_model_rejects_labels():
    model = Unet(dim=16, dim_mults=(1, 2), num_classes=None)
    x = torch.randn(1, 3, IMG, IMG)
    t = torch.zeros(1, dtype=torch.long)
    assert model(x, t).shape == x.shape
    with pytest.raises(ValueError):
        model(x, t, torch.tensor([1]))


def test_p_losses_backward(model, diffusion):
    model.train()
    x0 = torch.rand(4, 3, IMG, IMG) * 2 - 1
    t = torch.randint(0, T, (4,))
    loss = diffusion.p_losses(model, x0, t, labels=torch.arange(4), p_uncond=0.5)
    assert loss.ndim == 0
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("classes", [None, torch.tensor([0, 1, 2])])
def test_sample_shape_and_range(model, diffusion, classes):
    out, frames = diffusion.sample(model, batch_size=3, classes=classes, guidance_scale=1.0,
                                   return_trajectory=True, num_frames=4, progress=False)
    assert out.shape == (3, 3, IMG, IMG)
    assert out.min() >= -1 and out.max() <= 1
    assert len(frames) == 4


def test_guidance_zero_is_plain_conditional(model, diffusion):
    classes = torch.tensor([3, 4])
    torch.manual_seed(1)
    a = diffusion.sample(model, classes=classes, guidance_scale=0.0, progress=False)
    torch.manual_seed(1)
    x = torch.randn(2, 3, IMG, IMG)
    for i in reversed(range(T)):
        x = diffusion.p_sample(lambda x_, t_, c_, drop_mask=None: model(x_, t_, c_), x, i, classes, 0.0)
    torch.testing.assert_close(a, x.clamp(-1, 1))


def test_ema_tracks_weights(model):
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    p_model = next(model.parameters())
    p_ema = next(ema.model.parameters())
    assert not torch.equal(p_model, p_ema)
    assert not any(p.requires_grad for p in ema.model.parameters())
