"""Conditional U-Net noise predictor for DDPM."""
import math
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import nn, einsum

from .layers import Downsample, Residual, Upsample, default, exists


# Note: This code employs large parts of the following sources:
# Niels Rogge (nielsr) & Kashif Rasul (kashif): https://huggingface.co/blog/annotated-diffusion (last access: 23.05.2023),
# which is based on
# Phil Wang (lucidrains): https://github.com/lucidrains/denoising-diffusion-pytorch (last access: 23.05.2023)

class SinusoidalPositionEmbeddings(nn.Module):
    """
    A module for generating sinusoidal position embeddings.

    Attributes:
        dim (int): Dimension of the embeddings.
    """
    def __init__(self, dim: int):
        """
        Initializes the SinusoidalPositionEmbeddings module.

        Args:
            dim (int): Dimension of the embeddings.
        """
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Generates sinusoidal embeddings based on input time.

        Args:
            time (torch.Tensor): The input tensor representing time.

        Returns:
            torch.Tensor: The sinusoidal embeddings.
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class WeightStandardizedConv2d(nn.Conv2d):
    """
    A convolutional layer with weight standardization.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for WeightStandardizedConv2d.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying weight standardized convolution.
        """
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    """
    A basic block module for use in neural networks, implementing FiLM conditioning.

    Attributes:
        proj (nn.Module): Projection layer.
        norm (nn.Module): Normalization layer.
        act (nn.Module): Activation function.
    """
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        """
        Initializes the Block module.

        Args:
            dim (int): Dimension of the input.
            dim_out (int): Dimension of the output.
            groups (int): Number of groups for normalization.
        """
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
        # NOTE: This implements FiLM conditioning, see https://distill.pub/2018/feature-wise-transformations/ and
        # http://arxiv.org/pdf/1709.07871.pdf

    def forward(self, x: torch.Tensor, scale_shift: tuple = None) -> torch.Tensor:
        """
        Forward pass for Block.

        Args:
            x (torch.Tensor): Input tensor.
            scale_shift (tuple, optional): Tuple of scale and shift tensors for FiLM conditioning.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """
    A ResNet block with optional time and class embeddings.

    Attributes:
        mlp (nn.Sequential): A sequential MLP module.
        block1 (Block): The first block module.
        block2 (Block): The second block module.
        res_conv (nn.Module): Residual convolutional layer.
    """

    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: int = None, 
                 classes_emb_dim: int = None, groups: int = 8):
        """
        Initializes the ResnetBlock module.

        Args:
            dim (int): Dimension of the input.
            dim_out (int): Dimension of the output.
            time_emb_dim (int, optional): Dimension of the time embedding.
            classes_emb_dim (int, optional): Dimension of the class embedding.
            groups (int): Number of groups for normalization.
        """
        super().__init__()
        full_emb_dim = int(default(time_emb_dim, 0)) + int(default(classes_emb_dim, 0))
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(full_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor = None, 
                class_emb: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the ResnetBlock.

        Args:
            x (torch.Tensor): Input tensor.
            time_emb (torch.Tensor, optional): Time embedding.
            class_emb (torch.Tensor, optional): Class embedding.

        Returns:
            torch.Tensor: The output tensor after processing through the ResNet block.
        """
        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1 1')
            scale_shift = cond_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    """
    An attention mechanism in neural networks.

    Attributes:
        scale (float): Scaling factor for attention.
        heads (int): Number of attention heads.
        to_qkv (nn.Module): Convolutional layer to generate query, key, and value.
        to_out (nn.Module): Convolutional layer for output processing.
    """

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        """
        Initializes the Attention module.

        Args:
            dim (int): Dimension of the input.
            heads (int): Number of attention heads.
            dim_head (int): Dimension of each attention head.
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Attention module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output tensor after applying attention.
        """
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    """
    A linear attention mechanism, scaling linearly with sequence length.

    Attributes:
        scale (float): Scaling factor for attention.
        heads (int): Number of attention heads.
        to_qkv (nn.Conv2d): Convolutional layer to generate query, key, and value.
        to_out (nn.Sequential): Sequential layer for output processing.
    """

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        """
        Initializes the LinearAttention module.

        Args:
            dim (int): Dimension of the input.
            heads (int): Number of attention heads.
            dim_head (int): Dimension of each attention head.
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LinearAttention module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The output tensor after applying linear attention.
        """
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    """
    A normalization wrapper for a neural network module.

    Attributes:
        fn (nn.Module): The neural network module to normalize.
        norm (nn.GroupNorm): Group normalization layer.
    """

    def __init__(self, dim: int, fn: nn.Module):
        """
        Initializes the PreNorm module.

        Args:
            dim (int): Dimension for group normalization.
            fn (nn.Module): The neural network module to normalize.
        """
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PreNorm module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The normalized output tensor.
        """
        x = self.norm(x)
        return self.fn(x)




class Unet(nn.Module):
    """
    U-Net noise predictor ``eps_theta(x_t, t, c)`` for Denoising Diffusion Probabilistic Models.

    Time and (optional) class embeddings are concatenated and injected into every ResNet block via FiLM.
    For classifier-free guidance the class embedding table has one extra entry, the learned *null class*
    (index ``num_classes``), which stands for "no condition".

    Args:
        dim (int): Base channel width of the U-Net (also the size of the sinusoidal time embedding).
        init_dim (int, optional): Channels after the initial convolution. Defaults to ``dim``.
        out_dim (int, optional): Output channels. Defaults to ``channels``.
        dim_mults (tuple of int): Channel multipliers per resolution level.
        channels (int): Number of input image channels.
        resnet_block_groups (int): Number of groups for GroupNorm in the ResNet blocks.
        num_classes (int, optional): Number of classes for conditioning. ``None`` builds an unconditional model.
    """

    def __init__(
        self,
        dim: int,
        init_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        dim_mults: tuple = (1, 2, 4, 8),
        channels: int = 3,
        resnet_block_groups: int = 4,
        num_classes: Optional[int] = None,
    ):
        super().__init__()

        self.channels = channels
        self.num_classes = num_classes
        init_dim = default(init_dim, dim)

        # Initial convolution layer to match the input dimension
        self.init_conv = nn.Conv2d(channels, init_dim, 1, padding=0)

        # Define dimensions for each layer in the U-Net
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # Time embedding layers for temporal information encoding
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Class embedding; the extra last entry is the null class used for classifier-free guidance
        if exists(num_classes):
            classes_dim = time_dim
            self.class_emb = nn.Embedding(num_classes + 1, classes_dim)
        else:
            classes_dim = None
            self.class_emb = None

        block = partial(block_klass, time_emb_dim=time_dim, classes_emb_dim=classes_dim)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Construct downsampling layers with attention and residual connections
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block(dim_in, dim_in),
                        block(dim_in, dim_in),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        # Middle blocks of the U-Net with full attention
        mid_dim = dims[-1]
        self.mid_block1 = block(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block(mid_dim, mid_dim)

        # Each upsampling stage consists of two ResNet blocks, one attention block, and an upsampling operation
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block(dim_out + dim_in, dim_out),
                        block(dim_out + dim_in, dim_out),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        # Final residual block (on the concatenation with the initial features) and output projection
        self.final_res_block = block(init_dim * 2, dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    @property
    def null_class(self) -> Optional[int]:
        """Index of the learned null class, or ``None`` for an unconditional model."""
        return self.num_classes

    def forward(self, x: torch.Tensor, time: torch.Tensor, c: Optional[torch.Tensor] = None,
                drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict the noise contained in ``x``.

        Args:
            x (torch.Tensor): Noisy images of shape ``(B, C, H, W)``.
            time (torch.Tensor): Timesteps of shape ``(B,)``.
            c (torch.Tensor, optional): Class labels of shape ``(B,)``. ``None`` means unconditional (null class).
            drop_mask (torch.Tensor, optional): Bool tensor of shape ``(B,)``; where ``True`` the label is replaced
                by the null class. Used for condition dropout during training and for guidance at sampling time.

        Returns:
            torch.Tensor: Predicted noise of shape ``(B, out_dim, H, W)``.
        """
        x = self.init_conv(x)
        r = x.clone()  # kept for the final skip connection

        t = self.time_mlp(time)

        class_emb = None
        if exists(self.class_emb):
            if c is None:
                c = torch.full((x.shape[0],), self.null_class, device=x.device, dtype=torch.long)
            elif exists(drop_mask):
                c = torch.where(drop_mask, torch.full_like(c, self.null_class), c)
            class_emb = self.class_emb(c.long())
        elif exists(c):
            raise ValueError("Class labels were given, but the model was built without `num_classes`.")

        h = []  # intermediate outputs for skip connections

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, class_emb)
            h.append(x)

            x = block2(x, t, class_emb)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, class_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, class_emb)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, class_emb)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t, class_emb)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t, class_emb)
        return self.final_conv(x)
