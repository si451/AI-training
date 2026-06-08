"""
CortexV7 v3 — New modules: YaRN, HyperConnections, MTP
These get imported into Model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple

try:
    from flash_attn.ops.rms_norm import rms_norm
    HAS_FLASH_RMSNORM = True
except ImportError:
    HAS_FLASH_RMSNORM = False

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if HAS_FLASH_RMSNORM and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
            return rms_norm(x, self.weight, self.eps)
            
        x_float = x.float()
        norm = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_float * norm).to(x.dtype) * self.weight

# ─────────────────────────────────────────────────────────────────────────────
# YaRN-ENABLED ROTARY EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

class YaRNRotaryEmbedding(nn.Module):
    """YaRN RoPE (Yet another RoPE extensioN).

    Trains at original_max_len (2048). Extends to original_max_len * yarn_scale
    (e.g. 8192) without quality loss via NTK-aware interpolation.

    - High-freq dims: unchanged (encode local token patterns)
    - Low-freq dims:  interpolated (encode absolute position)
    - Smooth ramp between the two controlled by beta_fast/beta_slow
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000,
                 original_max_len: int = 2048, yarn_scale: float = 4.0,
                 yarn_beta_fast: float = 32.0, yarn_beta_slow: float = 1.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.original_max_len = original_max_len
        self.yarn_scale = yarn_scale

        # NTK-aware frequency mixing
        freq_extra = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        freq_inter = 1.0 / (yarn_scale * base ** (torch.arange(0, dim, 2).float() / dim))

        low = max(int(math.floor(
            dim * math.log(original_max_len / (yarn_beta_fast * 2 * math.pi))
            / (2 * math.log(base)))), 0)
        high = min(int(math.ceil(
            dim * math.log(original_max_len / (yarn_beta_slow * 2 * math.pi))
            / (2 * math.log(base)))), dim // 2 - 1)

        mask = torch.zeros(dim // 2)
        for i in range(dim // 2):
            if i < low:
                mask[i] = 0.0
            elif i > high:
                mask[i] = 1.0
            else:
                mask[i] = (i - low) / max(high - low, 1)

        inv_freq = freq_extra * (1.0 - mask) + freq_inter * mask
        self.register_buffer("inv_freq", inv_freq)

        # Attention scaling for extended sequences
        self.attn_scale = 0.1 * math.log(yarn_scale) + 1.0

        self._cached_cos: Optional[torch.Tensor] = None
        self._cached_sin: Optional[torch.Tensor] = None
        self._cached_seq_len: int = 0

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        if seq_len is None:
            seq_len = x.shape[1]
        if hasattr(torch, "_dynamo") and torch._dynamo.is_compiling():
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            return freqs.cos(), freqs.sin()
        if seq_len > self._cached_seq_len or self._cached_cos is None:
            self._cached_seq_len = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            self._cached_cos = freqs.cos()
            self._cached_sin = freqs.sin()
        return self._cached_cos[:seq_len], self._cached_sin[:seq_len]


# ─────────────────────────────────────────────────────────────────────────────
# HYPER-CONNECTIONS (ICLR 2025)
# ─────────────────────────────────────────────────────────────────────────────

class HyperConnection(nn.Module):
    """Learnable residual connections replacing standard x + f(x).

    Maintains n_streams parallel hidden streams. The "active" stream (index 0)
    gets transformed by the sublayer, then all streams are contracted back
    to a single representation via learned weights.

    At initialization, behaves exactly like a standard residual connection.
    During training, learns optimal cross-layer information routing.
    """

    def __init__(self, dim: int, n_streams: int = 2, layer_id: int = 0,
                 total_layers: int = 1):
        super().__init__()
        self.n_streams = n_streams
        self.dim = dim

        # Expansion: scale factors for creating each stream from input
        alpha_init = torch.zeros(n_streams)
        alpha_init[0] = 1.0   # active stream = full signal
        for i in range(1, n_streams):
            alpha_init[i] = 1.0   # residual streams = full signal too
        self.alpha = nn.Parameter(alpha_init)

        beta_init = torch.zeros(n_streams)
        beta_init[0] = 1.0        # full sublayer contribution
        beta_init[1] = 1.0        # full residual (starts exactly as standard x + f(x))
        self.beta = nn.Parameter(beta_init)

    def forward(self, x: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        """Apply hyper-connection: merge sublayer output with residual streams.

        x:            original input to the sublayer
        sublayer_out: output of the sublayer (e.g. attention or FFN)
        returns:      merged output
        """
        # Stream 0 = transformed, streams 1+ = residual
        result = self.beta[0] * sublayer_out
        for i in range(1, self.n_streams):
            result = result + self.beta[i] * self.alpha[i] * x
            
        return result


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-TOKEN PREDICTION HEAD (DeepSeek-V3 style)
# ─────────────────────────────────────────────────────────────────────────────

class MTPHead(nn.Module):
    """Lightweight sequential MTP head.

    Takes hidden state from previous depth + ground-truth embedding of the
    target token, projects them together, and produces a new hidden state
    that gets projected to vocabulary via the shared output head.

    Each MTP head adds only ~3.3M params at dim=1280.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim, bias=False)
        self.norm = RMSNorm(dim)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, h_prev: torch.Tensor,
                token_embed: torch.Tensor) -> torch.Tensor:
        """
        h_prev:      [B, S, dim] hidden state from previous depth
        token_embed: [B, S, dim] embedding of ground-truth token at this offset
        returns:     [B, S, dim] new hidden state for vocab projection
        """
        combined = torch.cat([h_prev, token_embed], dim=-1)
        return self.norm(self.proj(combined))
