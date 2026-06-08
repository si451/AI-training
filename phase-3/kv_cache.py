"""
DynamicKVCache for CortexV7 v4 — Simplified High-Performance KV Cache
Architecture: 20 layers (dim=1280, heads=16, kv_heads=4) + 3 memory bridges
Handles: GQA + Flash Attn 2 KV caching, memory state persistence

V4: Simplified — no more sensory/motor/reasoner split, just flat layers.
"""
import torch
import torch.nn.functional as F
import os
import re
import sys
from typing import Optional, Tuple, List, Dict, Any
import functools

# Module-level cache for blocked token IDs (computed once per encoder)
_blocked_ids_cache: Dict[int, set] = {}


def _maybe_compile_decode(fn):
    return fn


def _build_cjk_token_ids(enc) -> set:
    if enc is None or not hasattr(enc, "n_vocab"):
        return set()
    cache_key = id(enc)
    if cache_key in _blocked_ids_cache:
        return _blocked_ids_cache[cache_key]
    blocked = set()
    ranges = ((0x4E00, 0x9FFF), (0xAC00, 0xD7AF), (0x3040, 0x30FF))
    for token_id in range(enc.n_vocab):
        try:
            text = enc.decode([token_id])
        except Exception:
            continue
        if any(any(lo <= ord(ch) <= hi for ch in text) for lo, hi in ranges):
            blocked.add(token_id)
    _blocked_ids_cache[cache_key] = blocked
    return blocked


def _build_artifact_token_ids(enc) -> set:
    blocked = set()
    if enc is None or not hasattr(enc, "encode"):
        return blocked
    for text in ("Sig", " Sig", "tgtg", " tgtg", "集", " 集", "�"):
        try:
            token_ids = enc.encode(text, allowed_special=set())
        except Exception:
            continue
        if len(token_ids) == 1:
            blocked.add(token_ids[0])
    return blocked


def _apply_recent_repeat_penalty(tok_logits: torch.Tensor, generated: List[int], strength: float = 1.35) -> None:
    if not generated:
        return
    recent = generated[-32:]
    device = tok_logits.device
    recent_tensor = torch.tensor(list(set(recent)), dtype=torch.long, device=device)
    gathered = tok_logits[recent_tensor]
    pos_mask = gathered > 0
    tok_logits[recent_tensor] = torch.where(pos_mask, gathered / strength, gathered * strength)
    if len(generated) >= 2 and generated[-1] == generated[-2]:
        tok_logits[generated[-1]] = float("-inf")
    if len(generated) >= 4 and generated[-4:-2] == generated[-2:]:
        tok_logits[generated[-2]] = float("-inf")


def _has_repeated_suffix(token_ids, min_size=1, max_size=6, repeats=3):
    if len(token_ids) < min_size * repeats:
        return False
    for size in range(min_size, max_size + 1):
        chunk = token_ids[-size:]
        if all(token_ids[-size * (idx + 1): -size * idx if idx > 0 else None] == chunk
               for idx in range(repeats)):
            return True
    return False


def _looks_like_artifact_loop(decoded: str) -> bool:
    tail = decoded[-64:]
    if re.search(r'\b(Sig|tgtg)(?:\s+\1){1,}', tail, re.IGNORECASE):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# FAST KV BUFFER (Zero-Copy O(1) Pre-allocated Cache)
# ─────────────────────────────────────────────────────────────────────────────

class FastKVBuffer:
    """Pre-allocated ring buffer for KV cache. Avoids O(N²) torch.cat overhead."""

    def __init__(self, max_seq_len: int, num_kv_heads: int, head_dims: list,
                 dtype, device, batch_size: int = 1):
        self.max_seq_len = max_seq_len
        self.num_tensors = len(head_dims)
        self.seq_len = 0
        self.buffers = tuple(
            torch.zeros((batch_size, num_kv_heads, max_seq_len, dim),
                        dtype=dtype, device=device)
            for dim in head_dims
        )

    @classmethod
    def from_tuple(cls, t: tuple, max_seq_len: int):
        if t is None:
            return None
        B, num_kv_heads, S, _ = t[0].shape
        head_dims = [tensor.shape[3] for tensor in t]
        buf = cls(max_seq_len, num_kv_heads, head_dims, t[0].dtype,
                  t[0].device, batch_size=B)
        buf.seq_len = S
        for i in range(buf.num_tensors):
            buf.buffers[i][:, :, :S, :].copy_(t[i])
        return buf

    def update(self, *tensors):
        """In-place write to the buffer. Returns sliced views."""
        new_len = tensors[0].shape[2]
        end = self.seq_len + new_len
        if end > self.max_seq_len:
            end = self.max_seq_len
            start = end - new_len
        else:
            start = self.seq_len
        for i, t in enumerate(tensors):
            self.buffers[i][:, :, start:end, :].copy_(t)
        self.seq_len = end
        return tuple(b[:, :, :self.seq_len, :] for b in self.buffers)

    def get(self):
        return tuple(b[:, :, :self.seq_len, :].contiguous() for b in self.buffers)


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC KV CACHE — Simplified for CortexV7 v4
# ─────────────────────────────────────────────────────────────────────────────

class DynamicKVCache:
    """
    Simplified KV cache for CortexV7 v4 (flat 20-layer transformer + memory).

    Usage:
        cache = DynamicKVCache(model, max_seq_len=2048)
        logits, ponder, aux, mem = cache.prefill(prompt_tokens, enc=enc)
        for _ in range(max_new):
            logits, ponder, aux, mem = cache.decode_one(next_token)
    """

    def __init__(self, model, max_seq_len: int = 2048, batch_size: int = 1,
                 inference_mode: bool = False):
        self.model = model
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.inference_mode = inference_mode

        # Import KVCache from Model.py
        from Model import KVCache
        self.kv_cache = KVCache(num_layers=model.num_layers)
        self.memory_state = None
        self.seq_len = 0
        self.prompt_len = 0

    def _init_fast_buffers(self):
        """Upgrade standard tuples to FastKVBuffers after prefill."""
        for i in range(self.kv_cache.num_layers):
            kv = self.kv_cache.kvs[i]
            if kv is not None and not hasattr(kv, 'update'):
                self.kv_cache.kvs[i] = FastKVBuffer.from_tuple(kv, self.max_seq_len)

    @torch.no_grad()
    def prefill(self, tokens: torch.Tensor, enc=None,
                memory_state: Optional[List[torch.Tensor]] = None):
        """Process full prompt in one pass, populating the cache."""
        self.model.eval()
        from Model import KVCache
        self.kv_cache = KVCache(num_layers=self.model.num_layers)
        self.memory_state = memory_state

        self.prompt_len = tokens.shape[1]

        result = self.model(
            tokens, memory_state=self.memory_state, kv_cache=self.kv_cache)

        # Unpack: logits, ponder, aux, memory_state
        logits, ponder, aux, new_mem = result[:4]

        self.memory_state = new_mem
        self.seq_len = tokens.shape[1]
        self._init_fast_buffers()
        return logits, ponder, aux, self.memory_state

    @torch.no_grad()
    def decode_one(self, token: torch.Tensor):
        """Process token(s) [B, 1] using cached K/V."""
        result = self.model(
            token, memory_state=self.memory_state, kv_cache=self.kv_cache)

        logits, ponder, aux, new_mem = result[:4]
        self.memory_state = new_mem
        self.seq_len += 1
        return logits, ponder, aux, self.memory_state

    def reset(self):
        """Clear all cached state."""
        from Model import KVCache
        self.kv_cache = KVCache(num_layers=self.model.num_layers)
        self.memory_state = None
        self.seq_len = 0
        self.prompt_len = 0

    def truncate_kv_keep_memory(self, keep_last_n: int = 32):
        """Truncate KV cache but preserve memory state.
        
        This is the critical cognitive memory benchmark:
        If generation stays coherent after truncation → memory is genuinely cognitive.
        If output collapses → memory was auxiliary decoration.
        """
        for i in range(self.kv_cache.num_layers):
            kv = self.kv_cache.kvs[i]
            if kv is not None:
                if hasattr(kv, 'buffers'):
                    # FastKVBuffer: truncate to last N
                    old_len = kv.seq_len
                    if old_len > keep_last_n:
                        for j in range(kv.num_tensors):
                            kv.buffers[j][:, :, :keep_last_n] = kv.buffers[j][:, :, old_len - keep_last_n:old_len].clone()
                        kv.seq_len = keep_last_n
                elif isinstance(kv, tuple):
                    seq_len = kv[0].shape[2]
                    if seq_len > keep_last_n:
                        self.kv_cache.kvs[i] = (kv[0][:, :, -keep_last_n:].contiguous(),
                                                kv[1][:, :, -keep_last_n:].contiguous())
        self.seq_len = min(self.seq_len, keep_last_n)


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLING — Vectorized GPU sampling
# ─────────────────────────────────────────────────────────────────────────────

def _sample(logits_1d, generated, temperature=0.6, rep_penalty=1.35,
            top_k=32, top_p=0.88, blocked_ids=None):
    """Sample one token from 1-D logits with penalties."""
    tok_logits = logits_1d.clone().float()
    device = logits_1d.device

    if generated is not None:
        if not torch.is_tensor(generated):
            generated = torch.tensor(generated, dtype=torch.long, device=device)
        else:
            generated = generated.to(device)

        if generated.numel() > 0:
            unique_gen = torch.unique(generated)
            gathered = tok_logits[unique_gen]
            pos_mask = gathered > 0
            tok_logits[unique_gen] = torch.where(pos_mask,
                                                 gathered / rep_penalty,
                                                 gathered * rep_penalty)

            recent = generated[-32:] if generated.numel() > 32 else generated
            unique_recent = torch.unique(recent)
            gathered_r = tok_logits[unique_recent]
            pos_r = gathered_r > 0
            tok_logits[unique_recent] = torch.where(pos_r,
                                                    gathered_r / 1.35,
                                                    gathered_r * 1.35)

            if generated.numel() >= 2 and int(generated[-1].item()) == int(generated[-2].item()):
                tok_logits[int(generated[-1].item())] = float("-inf")
            if generated.numel() >= 4 and torch.equal(generated[-4:-2], generated[-2:]):
                tok_logits[int(generated[-2].item())] = float("-inf")

    if blocked_ids is not None:
        if not torch.is_tensor(blocked_ids):
            blocked_ids = torch.tensor(list(blocked_ids), dtype=torch.long, device=device)
        if blocked_ids.numel() > 0:
            tok_logits[blocked_ids] = float("-inf")

    tok_logits /= max(temperature, 1e-8)
    if top_k > 0:
        k = min(top_k, tok_logits.size(-1))
        kth = torch.topk(tok_logits, k).values[-1]
        tok_logits[tok_logits < kth] = float("-inf")
    probs = F.softmax(tok_logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    if top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        remove = cum - sp > top_p
        sp[remove] = 0.0
        tot = sp.sum()
        if tot <= 0.0 or not torch.isfinite(tot):
            sp = torch.ones_like(sp) / sp.numel()
        else:
            sp /= tot
        return si[torch.multinomial(sp, 1)].item()
    tot = probs.sum()
    if tot <= 0.0 or not torch.isfinite(tot):
        probs = torch.ones_like(probs) / probs.numel()
    else:
        probs /= tot
    return torch.multinomial(probs, 1).item()


# ─────────────────────────────────────────────────────────────────────────────
# FAST GENERATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_fast(model, enc, user_input: str, system_prompt: str = "",
                  max_tokens: int = 200, temperature: float = 0.6,
                  rep_penalty: float = 1.35, top_k: int = 32,
                  top_p: float = 0.88, stream: bool = True,
                  memory_state=None, max_seq_len: int = 2048,
                  cache=None):

    device = next(model.parameters()).device
    param_dtype = next(model.parameters()).dtype
    if param_dtype == torch.float16:
        dtype = torch.float16
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    prompt = (f"System: {system_prompt}\nUser: {user_input}\nAssistant:"
              if system_prompt else f"User: {user_input}\nAssistant:")
    tokens = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    blocked_token_ids = _build_cjk_token_ids(enc) | _build_artifact_token_ids(enc)

    if cache is None:
        cache = DynamicKVCache(model, max_seq_len=max_seq_len, batch_size=1, inference_mode=True)
    else:
        cache.reset()

    with torch.amp.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
        logits, ponder, _, mem = cache.prefill(x, enc=enc, memory_state=memory_state)

    generated = []
    generated_gpu = torch.empty(max_tokens, dtype=torch.long, device=device)
    last_yield_tok = 0
    ponder_sum = float(ponder)

    blocked_tensor = torch.tensor(sorted(blocked_token_ids), dtype=torch.long, device=device) if blocked_token_ids else None

    first_logits = logits[0, -1].clone()
    if blocked_tensor is not None and blocked_tensor.numel() > 0:
        first_logits[blocked_tensor] = float("-inf")
    next_token = _sample(first_logits, None, temperature,
                         rep_penalty, top_k, top_p, blocked_ids=blocked_tensor)
    if next_token == enc.eot_token:
        if not stream:
            yield "", 0.0
        return

    generated.append(next_token)
    generated_gpu[0] = next_token

    token_t = torch.zeros((1, 1), dtype=torch.long, device=device)

    # Precompute token sequences for stop markers to avoid decoding every step
    stop_texts = ("\nUser:", "\nSystem:", "\nHuman:")
    stop_seqs = []
    for s in stop_texts:
        try:
            stop_seqs.append(enc.encode(s, allowed_special={"<|endoftext|>"}))
        except Exception:
            stop_seqs.append([])

    def _endswith_seq(lst, seq):
        if not seq:
            return False
        if len(lst) < len(seq):
            return False
        return lst[-len(seq):] == seq

    for step in range(1, max_tokens):
        token_t[0, 0] = next_token

        with torch.amp.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
            logits, ponder, _, mem = cache.decode_one(token_t)

        ponder_sum += float(ponder)
        step_logits = logits[0, -1].clone()
        if blocked_tensor is not None and blocked_tensor.numel() > 0:
            step_logits[blocked_tensor] = float("-inf")
        next_token = _sample(step_logits, generated_gpu[:step], temperature,
                             rep_penalty, top_k, top_p, blocked_ids=blocked_tensor)

        if next_token == enc.eot_token:
            break

        generated.append(next_token)
        generated_gpu[step] = next_token

        # Token-based stop checks (avoid decoding tokens each step)
        stop_hit = False
        for seq in stop_seqs:
            if _endswith_seq(generated, seq):
                # trim the stop sequence from generated
                gen_trimmed = generated[:-len(seq)] if len(seq) > 0 else generated
                if stream and len(gen_trimmed) > last_yield_tok:
                    chunk = enc.decode(gen_trimmed[last_yield_tok:])
                    yield chunk, float(ponder)
                return

        # Token-level loop/artifact checks
        if _has_repeated_suffix(generated, min_size=1, max_size=6, repeats=3):
            if stream and len(generated) > last_yield_tok:
                chunk = enc.decode(generated[last_yield_tok:])
                yield chunk, float(ponder)
            break

        # Stream: decode only the newly generated token slice
        if stream:
            if len(generated) > last_yield_tok:
                new_part = enc.decode(generated[last_yield_tok:])
                last_yield_tok = len(generated)
                if new_part:
                    yield new_part, float(ponder)

    if not stream:
        full = enc.decode(generated)
        avg_ponder = ponder_sum / max(len(generated), 1)
        yield full, avg_ponder
