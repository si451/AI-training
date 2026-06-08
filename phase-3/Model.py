"""
Nexus — Simplified Transformer Architecture
=============================================
Clean 20-layer transformer with integrated neural memory.

Architecture (brain-inspired):
  Embed → Memory → [Layers 0-6]  → Memory → [Layers 7-13] → Memory → [Layers 14-19] → Output
           ↑ input     sensory        ↑ bridge    processing      ↑ bridge    output
           context                    update                      update

Components:
  - GQA Attention (heads=16, kv_heads=4) + Flash Attention 2
  - SwiGLU FFN (2.67× expansion)
  - GlobalNeuralMemory at 3 strategic bridge points
  - YaRN RoPE for context extension
  - MTP head for multi-token prediction training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple, Dict, Any
from v3_modules import YaRNRotaryEmbedding, MTPHead, HyperConnection, RMSNorm


# ─────────────────────────────────────────────────────────────────────────────
# BASIC BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────────────────────




def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings. x: [B, H, S, D]"""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class SwiGLU(nn.Module):
    """SwiGLU FFN — the standard modern FFN choice."""
    def __init__(self, dim: int, hidden_dim: Optional[int] = None,
                 num_layers: int = 1, layer_id: int = 0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2.67 * dim)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        
        self.layer_id = layer_id
        self.inner_norm = RMSNorm(hidden_dim)
        
        # Standard init for w1, w2
        nn.init.normal_(self.w1.weight, std=0.02)
        nn.init.normal_(self.w2.weight, std=0.02)
        # Scale down the output projection to stabilize deep residuals
        std_proj = 0.02 / math.sqrt(2 * num_layers)
        nn.init.normal_(self.w3.weight, std=std_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.w1(x)) * self.w2(x)
        if hidden.requires_grad:
            from torch.utils.checkpoint import checkpoint
            hidden_normed = checkpoint(self.inner_norm, hidden, use_reentrant=False)
        else:
            hidden_normed = self.inner_norm(hidden)
        return self.w3(hidden_normed)


# ─────────────────────────────────────────────────────────────────────────────
# GQA ATTENTION — Grouped Query Attention with Flash Attention 2
# ─────────────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """Grouped Query Attention with KV cache and Flash Attention 2 support."""

    def __init__(self, dim: int, num_heads: int = 16, num_kv_heads: int = 4,
                 num_layers: int = 20, layer_id: int = 0):
        super().__init__()
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = dim // num_heads
        self.n_rep        = num_heads // num_kv_heads

        self.q   = nn.Linear(dim, num_heads    * self.head_dim, bias=False)
        self.k   = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v   = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(num_heads * self.head_dim, dim,    bias=False)
        
        self.layer_id = layer_id
        self.inner_norm = RMSNorm(num_heads * self.head_dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # Standard init for q, k, v
        for layer in (self.q, self.k, self.v):
            nn.init.normal_(layer.weight, std=0.02)
        # Scale down output projection
        std_proj = 0.02 / math.sqrt(2 * num_layers)
        nn.init.normal_(self.out.weight, std=std_proj)

        self._flash_available = False
        try:
            import flash_attn
            self._flash_available = True
        except ImportError:
            pass

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
        past_kv=None, use_flash: bool = False,
    ) -> Tuple[torch.Tensor, Any]:
        B, S, D = x.shape

        q = self.q(x).view(B, S, self.num_heads,    self.head_dim)
        k = self.k(x).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # KV cache
        if past_kv is not None:
            if hasattr(past_kv, 'update'):
                k, v = past_kv.update(k, v)
            else:
                past_k, past_v = past_kv
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
        new_past_kv = past_kv if hasattr(past_kv, 'update') else (k, v)

        is_causal = (past_kv is None)

        # Flash Attention 2
        if use_flash and self._flash_available:
            from flash_attn import flash_attn_func
            q_fa = q.transpose(1, 2)
            if self.n_rep > 1:
                k_fa = k.repeat_interleave(self.n_rep, dim=1).transpose(1, 2)
                v_fa = v.repeat_interleave(self.n_rep, dim=1).transpose(1, 2)
            else:
                k_fa, v_fa = k.transpose(1, 2), v.transpose(1, 2)
            orig_dtype = q_fa.dtype
            attn = flash_attn_func(
                q_fa.to(torch.bfloat16), k_fa.to(torch.bfloat16),
                v_fa.to(torch.bfloat16), causal=is_causal,
            ).to(orig_dtype).reshape(B, S, -1)
        else:
            # PyTorch SDPA (also very fast)
            if self.n_rep > 1:
                k_s = k.repeat_interleave(self.n_rep, dim=1)
                v_s = v.repeat_interleave(self.n_rep, dim=1)
            else:
                k_s, v_s = k, v
            attn = F.scaled_dot_product_attention(q, k_s, v_s, is_causal=is_causal)
            attn = attn.transpose(1, 2).reshape(B, S, -1)

        if attn.requires_grad:
            from torch.utils.checkpoint import checkpoint
            attn_normed = checkpoint(self.inner_norm, attn, use_reentrant=False)
        else:
            attn_normed = self.inner_norm(attn)
            
        return self.out(attn_normed), new_past_kv


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMER BLOCK — Attention + FFN with pre-norm residuals & HyperConnections
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Single transformer layer: RMSNorm → Attention → RMSNorm → SwiGLU."""

    def __init__(self, dim: int, num_heads: int = 16, num_kv_heads: int = 4,
                 num_layers: int = 20, layer_id: int = 0, use_flash: bool = False):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn      = Attention(dim, num_heads, num_kv_heads, num_layers, layer_id)
        self.ffn_norm  = RMSNorm(dim)
        self.ffn       = SwiGLU(dim, num_layers=num_layers, layer_id=layer_id)
        
        # Hyper connections
        self.hyper_attn = HyperConnection(dim, n_streams=2)
        self.hyper_ffn  = HyperConnection(dim, n_streams=2)
        
        self.use_flash = use_flash

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                past_kv=None) -> Tuple[torch.Tensor, Any]:
        # Attention with residual
        attn_out, new_kv = self.attn(
            self.attn_norm(x), cos, sin, past_kv, use_flash=self.use_flash)
        h = self.hyper_attn(x, attn_out)          # learnable residual
        
        # FFN with residual
        ffn_out = self.ffn(self.ffn_norm(h))
        h = self.hyper_ffn(h, ffn_out)            # second learnable residual
        return h, new_kv


# ─────────────────────────────────────────────────────────────────────────────
# CROSS ATTENTION — For Neural Memory read/write
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttention(nn.Module):
    """Cross-attention for memory read/write operations."""

    def __init__(self, dim: int, num_heads: int = 16, num_kv_heads: int = 4):
        super().__init__()
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = dim // num_heads
        self.n_rep        = num_heads // num_kv_heads

        self.q   = nn.Linear(dim, num_heads    * self.head_dim, bias=False)
        self.k   = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v   = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(num_heads * self.head_dim, dim,    bias=False)

        # Standard init for q, k, v
        for layer in (self.q, self.k, self.v):
            nn.init.normal_(layer.weight, std=0.02)
        # Standard scale for out projection
        nn.init.normal_(self.out.weight, std=0.02)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        _, S_ctx, _ = context.shape

        q = self.q(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).view(B, S_ctx, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).view(B, S_ctx, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        attn = F.scaled_dot_product_attention(q, k, v)
        return self.out(attn.transpose(1, 2).reshape(B, S, -1))


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL NEURAL MEMORY — Persistent memory that learns across the sequence
# ─────────────────────────────────────────────────────────────────────────────

class GlobalNeuralMemory(nn.Module):
    """Cognitive Neural Memory — persistent memory that the model MUST use.

    Redesigned from auxiliary latent space into causally necessary cognitive state:
      - Gated competitive reads (not additive residual — model decides when memory matters)
      - Sparse top-k writes (slots compete for ownership → specialization)
      - Role-specific configs (lexical / semantic / reasoning)
      - Temporal persistence via forget gate with role-specific retention floors
    """

    def __init__(self, dim: int, num_slots: int = 128, heads: int = 16,
                 retain_floor: float = 0.5, write_scale: float = 0.3,
                 role: str = "generic"):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.role = role

        # Learnable initial memory state
        self.memory_init = nn.Parameter(torch.randn(num_slots, dim) * 0.02)

        # Norms
        self.norm_x   = RMSNorm(dim)
        self.norm_mem = RMSNorm(dim)
        self.mem_norm = RMSNorm(dim)

        # Read: query from x, key/value from memory
        self.read_attn  = CrossAttention(dim, heads, max(1, heads // 4))
        # Write: query from memory, key/value from x
        self.write_attn = CrossAttention(dim, heads, max(1, heads // 4))

        # ── STEP 1: Gated competitive read ──────────────────────────────────
        # Instead of x + scale * read_out, we use:
        #   gate = sigmoid(W[x ; read_out])
        #   x_new = gate * read_out + (1 - gate) * x
        # This forces the model to DECIDE when memory matters.
        self.read_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        # Initialize to 0.0 so sigmoid(0.0) = 0.5. Memory starts 50% open.
        nn.init.zeros_(self.read_gate[0].weight)
        nn.init.constant_(self.read_gate[0].bias, 0.0)

        # ── STEP 2: Sparse competitive write ────────────────────────────────
        # Dropout on write path to encourage diverse slot usage
        self.write_dropout = nn.Dropout(0.05)

        self.write_scale = write_scale
        self.retain_floor = retain_floor

        # Forget gate — controls how much old memory is retained
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        nn.init.zeros_(self.gate[0].weight)
        nn.init.constant_(self.gate[0].bias, 2.0)  # Start with high retention

        # ── Diagnostic state (not parameters, not saved) ────────────────────
        self.last_read_gate: Optional[torch.Tensor] = None
        self.last_write_mask: Optional[torch.Tensor] = None

    def init_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.memory_init.unsqueeze(0).expand(batch_size, -1, -1).clone()

    def forward(self, x: torch.Tensor, memory_state: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_norm   = self.norm_x(x)

        # DDP Unused Parameter Fix: Ensure memory_init is always in the graph.
        memory_state = memory_state + (self.memory_init * 0.0).unsqueeze(0)

        mem_norm = self.norm_mem(memory_state)

        # ── READ: Gated competitive routing ─────────────────────────────────
        read_out = self.read_attn(x_norm, mem_norm)
        gate_raw = self.read_gate(torch.cat([x, read_out], dim=-1))
        
        gate = gate_raw
            
        x_new = gate * read_out + (1 - gate) * x
        # Store for diagnostics and usefulness loss (DO NOT DETACH OR LOSS GRADIENTS WILL SEVER)
        self.last_read_gate = gate

        # ── WRITE: Sparse top-k competitive routing ─────────────────────────
        # 1. Score each slot's affinity to the current context (Scale by sqrt(D) to prevent sigmoid saturation!)
        context_summary = x_norm.mean(dim=1, keepdim=True)  # [B, 1, D]
        slot_scores = (mem_norm * context_summary).sum(dim=-1) / math.sqrt(x_norm.shape[-1])  # [B, num_slots]

        # 1.5. Prevent Dead Slot Collapse (Exploration Noise)
        if self.training:
            # Add scaled Gumbel-like noise to encourage exploring unused slots
            noise = torch.randn_like(slot_scores) * slot_scores.std(dim=-1, keepdim=True) * 0.5
            routing_scores = slot_scores + noise
        else:
            routing_scores = slot_scores

        # 2. Select slots for update (dynamic independent routing via STE)
        soft_mask = torch.sigmoid(routing_scores)
        hard_mask = (soft_mask > 0.5).float()
        
        # 2.5. Create Hard Write Mask
        # Forward pass uses hard_mask, backward pass flows gradients to soft_mask
        write_mask = hard_mask - soft_mask.detach() + soft_mask  # [B, S]
        write_mask_3d = write_mask.unsqueeze(-1)  # [B, S, 1]
        
        # Compute MoE Load Balancing Loss (only during training)
        lb_loss = torch.tensor(0.0, device=x.device)
        if self.training:
            # P: mean routing probability across the batch
            P = F.softmax(slot_scores, dim=-1).mean(dim=0)  # [S]
            # f: fraction of times each slot was actually selected
            f = hard_mask.mean(dim=0)  # [S]
            # Standard MoE load balancing loss: S * sum(f_i * P_i)
            num_slots_s = slot_scores.size(-1)
            lb_loss = num_slots_s * torch.sum(f * P)

        # Store for diagnostics (store exact hard mask for accurate metric counting)
        self.last_write_mask = hard_mask.detach()

        # 3. Compute write update with slot identity injection
        mem_query = mem_norm + self.memory_init.unsqueeze(0)
        mem_update = self.write_attn(mem_query, x_norm)
        mem_update = self.write_dropout(mem_update)

        # 4. Gated update — winning slots get full candidate, losing slots get 5% leak
        forget_gate = self.gate(memory_state).clamp(self.retain_floor, 0.95)
        candidate = forget_gate * memory_state + (1 - forget_gate) * (self.write_scale * mem_update)
        new_memory = write_mask_3d * candidate + (1 - write_mask_3d) * memory_state
        new_memory = self.mem_norm(new_memory)

        return x_new, new_memory, lb_loss


# ─────────────────────────────────────────────────────────────────────────────
# KV CACHE — Simple flat KV cache for inference
# ─────────────────────────────────────────────────────────────────────────────

class KVCache:
    """Simple KV cache for autoregressive inference."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.kvs: List[Any] = [None] * num_layers

    def get_past_len(self) -> int:
        """Get the sequence length of cached keys."""
        for kv in self.kvs:
            if kv is not None:
                if hasattr(kv, 'seq_len'):
                    return kv.seq_len
                return kv[0].shape[2]
        return 0

    def update(self, layer_idx: int, new_kv):
        self.kvs[layer_idx] = new_kv

    def reset(self):
        self.kvs = [None] * self.num_layers


# ─────────────────────────────────────────────────────────────────────────────
# NEXUS — The Main Model
# ─────────────────────────────────────────────────────────────────────────────

class Nexus(nn.Module):
    """Nexus — Clean transformer with neural memory.

    Architecture:
        Embed → Memory₁ → [Layer 0-6] → Memory₂ → [Layer 7-13] → Memory₃ → [Layer 14-19] → Head

    Brain analogy:
        Memory₁ = "What do I already know about this input?" (hippocampal retrieval)
        Layers 0-6 = Sensory cortex (perception, pattern recognition)
        Memory₂ = "Update what I've learned from perception" (hippocampal encoding)
        Layers 7-13 = Association cortex (reasoning, integration)
        Memory₃ = "Consolidate reasoning into memory" (hippocampal consolidation)
        Layers 14-19 = Motor cortex (output generation)
    """

    def __init__(
        self,
        vocab_size:    int  = 100_277,
        dim:           int  = 1280,
        heads:         int  = 16,
        kv_heads:      int  = 4,
        num_layers:    int  = 20,
        memory_slots:  int  = 128,
        use_flash:     bool = False,
        mtp_depths:    int  = 1,
    ):
        super().__init__()
        self.dim        = dim
        self.heads      = heads
        self.num_layers = num_layers

        # ── Embedding ────────────────────────────────────────────────────────
        self.embed = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embed.weight, std=0.01)

        head_dim = dim // heads
        self.rope = YaRNRotaryEmbedding(
            head_dim, max_seq_len=8192,
            original_max_len=2048, yarn_scale=8.0)

        # ── Transformer Layers ───────────────────────────────────────────────
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, kv_heads, num_layers, i, use_flash)
            for i in range(num_layers)
        ])

        # ── Neural Memory (called at 3 strategic bridge points) ──────────────
        # Step 9: Hierarchical Memory Specialization
        memory_configs = [
            {"slots": 32,  "retain_floor": 0.3, "write_scale": 0.5, "role": "lexical"},
            {"slots": 64,  "retain_floor": 0.5, "write_scale": 0.3, "role": "semantic"},
            {"slots": 128, "retain_floor": 0.7, "write_scale": 0.2, "role": "reasoning"},
        ]
        self.memory = nn.ModuleList([
            GlobalNeuralMemory(
                dim, 
                num_slots=cfg["slots"], 
                heads=heads,
                retain_floor=cfg["retain_floor"],
                write_scale=cfg["write_scale"],
                role=cfg["role"]
            )
            for cfg in memory_configs
        ])

        # Step 5: Predictive Memory Objective Head
        self.memory_predictor = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.memory_predictor.weight, std=0.01)

        # Memory bridge schedule: after which layers to call memory
        # Bridges shifted slightly deeper: [3, 10, 17]
        self.memory_bridges = [3, 10, 17]

        # ── MTP Head ────────────────────────────────────────────────────────
        self.mtp_heads = nn.ModuleList([MTPHead(dim) for _ in range(mtp_depths)])

        # ── Output ──────────────────────────────────────────────────────────
        self.norm = RMSNorm(dim)
        
        # Output Reflection: lets the final layers read the newly written memory
        # This ties the memory writing mechanism to the LM loss.
        self.final_memory_read = CrossAttention(dim, heads, max(1, heads // 4))
        self.final_memory_scale = nn.Parameter(torch.tensor(0.1))
        self.final_memory_gate = nn.Linear(dim, 1)
        nn.init.zeros_(self.final_memory_gate.weight)
        nn.init.constant_(self.final_memory_gate.bias, 0.0) # Starts at gate=0.5
        
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # Weight tying

        # ── Print summary ───────────────────────────────────────────────────
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[OK] Nexus initialized: {total_params:,} parameters ({total_params/1e9:.3f}B)")
        print(f"   Architecture: {num_layers} layers (dim={dim}, heads={heads}, kv_heads={kv_heads})")
        print(f"   Memory: {memory_slots} slots × {dim}d at bridges {self.memory_bridges}")
        print(f"   FFN: SwiGLU ({int(2.67 * dim)}d hidden)")
        print(f"   MTP Heads: {mtp_depths}")
        print(f"   Flash Attn 2: {'enabled' if use_flash else 'disabled'}")

    def project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(hidden))

    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[List[torch.Tensor]] = None,
        kv_cache: Optional[KVCache] = None,
        targets: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> Any:
        B, S = x.shape
        h = self.embed(x)

        # ── RoPE ────────────────────────────────────────────────────────────
        past_len = kv_cache.get_past_len() if kv_cache else 0
        total_len = past_len + S
        cos_full, sin_full = self.rope(h, total_len)
        cos = cos_full[past_len:total_len]
        sin = sin_full[past_len:total_len]

        # ── Initialize memory ──────────────────────────────────────────────
        if memory_state is None:
            memory_state = [mem.init_memory(B, h.device) for mem in self.memory]

        # ── Forward through layers with memory bridges ─────────────────────
        bridge_idx = 0
        new_memory_state = []
        aux_losses = []
        for i, layer in enumerate(self.layers):
            # Memory bridge BEFORE this layer group starts
            if i in self.memory_bridges:
                h, m_state, lb_loss = self.memory[bridge_idx](h, memory_state[bridge_idx])
                new_memory_state.append(m_state)
                aux_losses.append(lb_loss)
                bridge_idx += 1

            # Transformer layer
            pkv = kv_cache.kvs[i] if kv_cache else None

            # Step 6: Memory Dependency Pressure (Scheduled KV Masking)
            if self.training and getattr(self, '_kv_mask_ratio', 0.0) > 0.0:
                if pkv is not None and isinstance(pkv, tuple):
                    mask_ratio = self._kv_mask_ratio
                    seq_len = pkv[0].shape[2]
                    keep = int(seq_len * (1 - mask_ratio))
                    if keep > 0 and keep < seq_len:
                        indices = torch.randperm(seq_len, device=pkv[0].device)[:keep].sort().values
                        pkv = (pkv[0][:, :, indices], pkv[1][:, :, indices])

            h, new_kv = layer(h, cos, sin, pkv)
            if kv_cache:
                kv_cache.update(i, new_kv)

        # ── Output ──────────────────────────────────────────────────────────
        h_normed = self.norm(h)
        
        # Output Reflection: use past memory to shape output (using past state prevents future data leaks)
        if memory_state is not None:
            # Concatenate all memory states (sensory, processing, output) along seq dim
            # shape: (B, num_slots * 3, dim)
            all_mem = torch.cat(memory_state, dim=1)
            reflection = self.final_memory_read(h_normed, all_mem)
            
            # Adaptive Output Gate
            gate = torch.sigmoid(self.final_memory_gate(h_normed))
            
            # Add reflection to the un-normalized residual stream
            h = h + gate * self.final_memory_scale * reflection
            
            # Re-apply the final norm so the LM head input is strictly bounded!
            h_normed = self.norm(h)
            
        logits = self.head(h_normed)

        # ── MTP loss (multi-token prediction, training only) ────────────────
        mtp_loss = torch.tensor(0.0, device=h.device)
        if targets is not None and self.training and len(self.mtp_heads) > 0:
            mtp_losses = []
            h_prev = h_normed
            for d, mtp_head in enumerate(self.mtp_heads):
                offset = d + 2  # predict token at position +2, +3, ...
                if S <= offset:
                    break
                # tgt_ids: the embedding we inject. For d=0, we inject targets[:, 0:-1]
                tgt_ids = targets[:, d : -1]
                tgt_embed = self.embed(tgt_ids)
                # h_prev: we drop the last state since we don't predict beyond the sequence
                h_prev_trunc = h_prev[:, : -1]
                h_mtp = mtp_head(h_prev_trunc, tgt_embed)
                logits_mtp = self.head(h_mtp)
                # labels: the token we are predicting. For d=0, we predict targets[:, 1:]
                labels = targets[:, d + 1 :]
                if labels.shape[1] == 0:
                    break
                
                loss_flat = F.cross_entropy(
                    logits_mtp.reshape(-1, logits_mtp.size(-1)),
                    labels.reshape(-1), ignore_index=-1)
                mtp_losses.append(loss_flat)
                h_prev = h_mtp
            if mtp_losses:
                mtp_loss = sum(mtp_losses) / len(mtp_losses)

        # ── Return ──────────────────────────────────────────────────────────
        # Ponder is always 1.0 (no ACT), aux is MoE load balancing loss
        ponder = torch.tensor(1.0, device=h.device)
        if aux_losses:
            aux = sum(aux_losses) / len(aux_losses)
        else:
            aux = torch.tensor(0.0, device=h.device)

        if targets is not None:
            return logits, ponder, aux, new_memory_state, mtp_loss, h_normed
        elif return_intermediates:
            return logits, ponder, aux, new_memory_state, h_normed
        return logits, ponder, aux, new_memory_state

    def memory_usefulness_loss(self, read_gates: List[torch.Tensor]) -> torch.Tensor:
        """Penalize memory if the read gate never opens.
        
        read_gates: list of [B, S, D] gate activations from each bridge.
        If gate values are always ~0, memory is being bypassed.
        """
        if not read_gates or read_gates[0] is None:
            return torch.tensor(0.0, device=self.embed.weight.device)
        total = torch.tensor(0.0, device=read_gates[0].device)
        for gate in read_gates:
            mean_gate = gate.mean()
            # Penalty: encourage gate to be at least 0.3 on average
            usage_penalty = F.relu(0.3 - mean_gate)
            total += usage_penalty
        return total / len(read_gates)

    def memory_prediction_loss(self, memory_states: List[torch.Tensor], final_hidden: torch.Tensor) -> torch.Tensor:
        """Train memory to predict the final hidden state.
        
        This forces memory to store information that will be
        useful for future prediction, not just current snapshots.
        """
        total = torch.tensor(0.0, device=final_hidden.device)
        target = final_hidden.mean(dim=1).detach()  # [B, D] — detached!
        for mem in memory_states:
            # Pool memory slots → single prediction
            mem_pooled = mem.max(dim=1)[0]  # [B, D]
            predicted = self.memory_predictor(mem_pooled)  # [B, D]
            # Cosine similarity loss
            sim = F.cosine_similarity(predicted, target, dim=-1).mean()
            total += (1.0 - sim)
        return total / len(memory_states)

    def memory_diversity_loss(self, memory_states: List[torch.Tensor]) -> torch.Tensor:
        """Penalize memory slot collapse by encouraging orthogonality.
        
        Computes mean cosine similarity between all pairs of memory slots
        across all bridges. Returns a scalar loss that should be MINIMIZED
        (high similarity = high loss).
        """
        total_loss = torch.tensor(0.0, device=memory_states[0].device)
        for mem in memory_states:  # mem: [B, num_slots, dim]
            # Normalize slots to unit vectors
            mem_norm = F.normalize(mem.float(), dim=-1)  # [B, S, D]
            # Cosine similarity matrix: [B, S, S]
            sim = torch.bmm(mem_norm, mem_norm.transpose(1, 2))
            # Zero out diagonal (self-similarity = 1.0, not interesting)
            eye = torch.eye(sim.size(1), device=sim.device).unsqueeze(0)
            sim = sim * (1.0 - eye)
            # Mean of squared similarities (penalize high similarity)
            total_loss = total_loss + (sim ** 2).mean()
        return total_loss / len(memory_states)

    # ─────────────────────────────────────────────────────────────────────────
    def configure_optimizers(
        self,
        learning_rate: float,
        weight_decay:  float,
        betas: Tuple[float, float] = (0.9, 0.95),
    ) -> torch.optim.AdamW:
        decay, no_decay, hyper_decay = set(), set(), set()
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters(recurse=False):
                fpn = f"{mn}.{pn}" if mn else pn
                # HyperConnection alpha/beta get their own mild decay group
                if isinstance(m, HyperConnection) and pn in ("alpha", "beta"):
                    hyper_decay.add(fpn)
                elif pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, (nn.Linear, nn.Embedding)):
                    decay.add(fpn)
                else:
                    no_decay.add(fpn)

        param_dict = dict(self.named_parameters())
        optim_groups = [
            {"params": [param_dict[p] for p in sorted(decay)    if p in param_dict],
             "weight_decay": weight_decay},
            {"params": [param_dict[p] for p in sorted(no_decay) if p in param_dict],
             "weight_decay": 0.0},
            {"params": [param_dict[p] for p in sorted(hyper_decay) if p in param_dict],
             "weight_decay": 0.01},  # mild decay to anchor alpha/beta near init
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=torch.cuda.is_available())


# Backward-compatible alias so old checkpoint loading code still works
CortexV7 = Nexus


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running Nexus sanity check …\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Nexus(
        vocab_size   = 100_277,
        dim          = 1280,
        heads        = 16,
        kv_heads     = 4,
        num_layers   = 20,
        memory_slots = 128,
        use_flash    = True,
        mtp_depths   = 1,
    ).to(device)

    # ── Training forward pass ──────────────────────────────────────────────
    print("\n[1] Training pass (batch=2, seq=64) …")
    x = torch.randint(0, 100_277, (2, 64), device=device)
    targets = torch.randint(0, 100_277, (2, 64), device=device)
    model.train()
    with torch.no_grad():
        logits, ponder, aux, new_mem, mtp_loss, h_normed = model(x, targets=targets)
    print(f"    logits: {logits.shape}  ponder: {ponder:.3f}  mtp_loss: {mtp_loss:.4f}")
    assert logits.shape == (2, 64, 100_277), f"Shape mismatch: {logits.shape}"
    print("    [OK] Training pass OK")

    # ── Inference with KV cache ────────────────────────────────────────────
    print("\n[2] Inference with KV cache (prefill 16 + generate 8) …")
    model.eval()
    cache = KVCache(num_layers=20)
    prompt = torch.randint(0, 100_277, (1, 16), device=device)
    with torch.no_grad():
        logits_pre, _, _, mem, *_ = model(prompt, kv_cache=cache)
    print(f"    Prefill: {logits_pre.shape}, cache len: {cache.get_past_len()}")

    next_token = logits_pre[:, -1, :].argmax(-1, keepdim=True)
    for step in range(8):
        with torch.no_grad():
            logits_step, _, _, mem, *_ = model(next_token, memory_state=mem, kv_cache=cache)
        next_token = logits_step[:, -1, :].argmax(-1, keepdim=True)
    print(f"    Generated 8 tokens. Cache len: {cache.get_past_len()}")
    print("    [OK] KV Cache inference OK")

    print(f"\n[OK] Nexus sanity check passed!")
    print(f"    {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"    Memory bridges at layers: {model.memory_bridges}")

