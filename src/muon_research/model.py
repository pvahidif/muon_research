"""GPT architecture with optional GeoMuon covariance capture sinks."""

# pylint: disable=abstract-method
# pylint: disable=arguments-differ
# pylint: disable=arguments-renamed

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from muon_research.rules import TrainConfig


def covariance_stat(x: Tensor, cov_ndim: int) -> Tensor:
    """Compute E[x_i²] or E[x_i x_j] over all non-feature dimensions."""
    flat = x.reshape(-1, x.shape[-1]).float()
    n = flat.shape[0]

    if cov_ndim == 1:
        return flat.square().mean(dim=0)

    if cov_ndim == 2:
        return (flat.mT @ flat) / n

    raise RuntimeError(f"Invalid covariance rank: {cov_ndim}")


class CaptureActivationCov(torch.autograd.Function):
    """Identity whose backward emits an activation covariance."""

    @staticmethod
    def forward(ctx, x: Tensor, stat_sink: Tensor):
        if stat_sink.ndim not in (1, 2):
            raise ValueError(
                f"stat_sink must be a vector or matrix, got {stat_sink.ndim}D"
            )

        d = x.shape[-1]
        expected_shape = (d,) if stat_sink.ndim == 1 else (d, d)
        if stat_sink.shape != expected_shape:
            raise ValueError(
                f"Expected stat_sink shape {expected_shape}, "
                f"got {tuple(stat_sink.shape)}"
            )

        ctx.cov_ndim = stat_sink.ndim
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_x: Tensor):
        (x,) = ctx.saved_tensors
        stat = covariance_stat(x, ctx.cov_ndim)

        # Gradient for x, gradient for stat_sink.
        return grad_x, stat


class CaptureErrorCov(torch.autograd.Function):
    """Identity whose backward emits an output-gradient covariance."""

    @staticmethod
    def forward(ctx, y: Tensor, stat_sink: Tensor):
        if stat_sink.ndim not in (1, 2):
            raise ValueError(
                f"stat_sink must be a vector or matrix, got {stat_sink.ndim}D"
            )

        d = y.shape[-1]
        expected_shape = (d,) if stat_sink.ndim == 1 else (d, d)
        if stat_sink.shape != expected_shape:
            raise ValueError(
                f"Expected stat_sink shape {expected_shape}, "
                f"got {tuple(stat_sink.shape)}"
            )

        ctx.cov_ndim = stat_sink.ndim
        return y

    @staticmethod
    def backward(ctx, grad_y: Tensor):
        stat = covariance_stat(grad_y, ctx.cov_ndim)
        return grad_y, stat


def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (norm(x.float()) * self.gains).type_as(x)


class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)
        self.capture_cov = False
        self.register_buffer(
            "_act_sink",
            torch.zeros(
                in_features,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_err_sink",
            torch.zeros(
                out_features,
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def forward(self, x):
        capture = self.capture_cov and self.training and torch.is_grad_enabled()
        if capture:
            x = CaptureActivationCov.apply(x, self._act_sink)
        # pylint: disable=not-callable
        y = F.linear(x, self.weight.type_as(x), self.bias.type_as(x))
        if capture:
            y = CaptureErrorCov.apply(y, self._err_sink)
        return y


class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(
            0, 1, steps=dim // 4, dtype=torch.float32
        )
        self.register_buffer(
            "angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        )

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        # pylint: disable=not-callable
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale=0.12,
            is_causal=True,
        ).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, dim: int, expansion_ratio: float = 4.0):
        super().__init__()
        hdim = int(expansion_ratio * dim)
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        num_heads: int,
        expansion_ratio: float = 4.0,
    ):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim=head_dim, num_heads=num_heads)
        self.mlp = MLP(dim, expansion_ratio=expansion_ratio)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        head_dim: int,
        num_heads: int,
        expansion_ratio: float = 4.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.expansion_ratio = expansion_ratio

        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=model_dim,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    expansion_ratio=expansion_ratio,
                )
                for _ in range(num_layers)
            ]
        )
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def logits(self, inputs: Tensor) -> Tensor:
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        return 15 * logits * (logits.square() + 15**2).rsqrt()

    def forward(self, inputs: Tensor, targets: Tensor):
        logits = self.logits(inputs)
        return F.cross_entropy(
            logits.view(targets.numel(), -1), targets.view(-1), reduction="sum"
        )


class GPLBlock(nn.Module):
    def __init__(self, dim: int, expansion_ratio: float = 4.0):
        super().__init__()
        hdim = int(expansion_ratio * dim)
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        return x + self.proj(F.relu(self.fc(x)))


class GPL(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        embed_dim: int,
        num_tokens: int,
        expansion_ratio: float = 4.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.expansion_ratio = expansion_ratio

        self.embed = nn.Embedding(vocab_size, embed_dim).bfloat16()
        self.embed_proj = Linear(num_tokens * embed_dim, model_dim)
        self.blocks = nn.ModuleList(
            [
                GPLBlock(model_dim, expansion_ratio=expansion_ratio)
                for _ in range(num_layers)
            ]
        )
        self.proj = Linear(model_dim, vocab_size)
        self.norm2 = RMSNorm(model_dim)

    def context_window(self, inputs: Tensor) -> Tensor:
        """Concat the embeddings of the last num_tokens tokens (causal, zero-padded)."""
        B, T = inputs.size(0), inputs.size(1)
        embed = self.embed(inputs)
        padded = F.pad(embed, (0, 0, self.num_tokens - 1, 0))
        windows = padded.unfold(1, self.num_tokens, 1)  # (B, T, embed_dim, num_tokens)
        return windows.permute(0, 1, 3, 2).reshape(
            B, T, self.num_tokens * self.embed_dim
        )

    def logits(self, inputs: Tensor) -> Tensor:
        x = self.embed_proj(self.context_window(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        return 15 * logits * (logits.square() + 15**2).rsqrt()

    def forward(self, inputs: Tensor, targets: Tensor):
        logits = self.logits(inputs)
        return F.cross_entropy(
            logits.view(targets.numel(), -1), targets.view(-1), reduction="sum"
        )


def build_model(train_config: TrainConfig) -> GPT | GPL:
    """Construct the (CPU, un-``.cuda()``'d) model for ``train_config``
    -- left un-moved so a caller (e.g. ``fork.build_model_and_geon``, via
    ``RuleSet.resolve``) can validate every rule/param before spending any
    GPU memory."""
    if train_config.model_type == "gpt":
        return GPT(
            vocab_size=train_config.vocab_size,
            num_layers=train_config.num_layers,
            model_dim=train_config.model_dim,
            head_dim=train_config.head_dim,
            num_heads=train_config.num_heads,
            expansion_ratio=train_config.expansion_ratio,
        )
    if train_config.model_type == "gpl":
        return GPL(
            vocab_size=train_config.vocab_size,
            num_layers=train_config.num_layers,
            model_dim=train_config.model_dim,
            embed_dim=train_config.embed_dim,
            num_tokens=train_config.num_tokens,
            expansion_ratio=train_config.expansion_ratio,
        )
    raise ValueError(f"invalid model_type {train_config.model_type}")
