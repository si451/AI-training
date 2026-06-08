"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         NEXUS V7 — PHASE 3: SUPERVISED FINE-TUNING  (Unified v4)           ║
║                                                                              ║
║  Single-stage "Unified Uncensored Alignment" curriculum:                    ║
║                                                                              ║
║   Source                          Count   Purpose                           ║
║   ───────────────────────────────────────────────────────                   ║
║   dolphin-r1 reasoning traces    30,000   Teach deep <think> chains         ║
║   dolphin-r1 non-reasoning       10,000   Teach fast bypass of thinking     ║
║   Domain-specific simple Q&A      3,000   Awaken all pretrained domains     ║
║   Identity injection (35 × 200)   7,000   Prevent catastrophic forgetting   ║
║   ───────────────────────────────────────────────────────                   ║
║   Total                          50,000                                      ║
║                                                                              ║
║  Epochs: 5  (250K gradient exposures for deep internalization)              ║
║  Smart truncation: reasoning traces are trimmed to fit seq_len=1024        ║
║  while ALWAYS keeping the complete answer.                                 ║
║                                                                              ║
║  Run:    python Train_sft.py --checkpoint ckpt_v7_500000.pth               ║
║  Flags:  --reasoning-count 30000 --nonreasoning-count 10000                ║
║          --identity-reps 200 --save-dir ./sft_checkpoints                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn.functional as F
import tiktoken
import os
import sys
import glob
import re
import random
import math
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from kv_cache import DynamicKVCache, _sample
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
def validate_example_tags(examples: List[Dict], max_samples: int = 50):
    """
    Checks the first `max_samples` examples for unbalanced <think>/</think>.
    Uses a temporary dataset to exactly replicate the formatting used in training.
    """
    import copy
    # Use a huge max_len so formatting doesn't truncate
    tmp_ds = NexusSFTDataset(copy.deepcopy(examples[:max_samples]), max_len=999999)
    enc_check = enc  # already global
    
    think_open  = enc_check.encode("<think>\n", allowed_special=set())
    think_close = enc_check.encode("\n</think>\n", allowed_special=set())
    bare_close  = enc_check.encode("</think>", allowed_special=set())
    asst_seq    = enc_check.encode("\nAssistant:", allowed_special=set())

    for i in range(min(max_samples, len(tmp_ds))):
        ex = tmp_ds.examples[i]            # original dict
        text = tmp_ds._format_example(ex)  # use dataset's own formatting
        tokens = enc_check.encode(text, allowed_special={"<|endoftext|>"})
        
        # Find start of Assistant content
        asst_idx = -1
        for j in range(len(tokens) - len(asst_seq) + 1):
            if tokens[j:j+len(asst_seq)] == asst_seq:
                asst_idx = j + len(asst_seq)
                break
        if asst_idx == -1:
            continue
        
        ass_tokens = tokens[asst_idx:]
        open_pos   = [j for j in range(len(ass_tokens) - len(think_open) + 1)
                       if ass_tokens[j:j+len(think_open)] == think_open]
        close_pos  = [j for j in range(len(ass_tokens) - len(think_close) + 1)
                       if ass_tokens[j:j+len(think_close)] == think_close]
        bare_close_pos = [j for j in range(len(ass_tokens) - len(bare_close) + 1)
                           if ass_tokens[j:j+len(bare_close)] == bare_close]
        
        if len(open_pos) < len(close_pos) + len(bare_close_pos):
            print(f"  ⚠️  Example {i} has unbalanced think tags: "
                  f"opens={len(open_pos)} closes={len(close_pos)+len(bare_close_pos)}",
                  flush=True)
            # Show first 200 tokens of the assistant section for diagnosis
            print(f"      Assistant tokens (first 200): {ass_tokens[:200]}", flush=True)
# ── Import model ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CortexV7 = None
for _mod_name, _cls_name in [("Model", "CortexV7"), ("CortexV7", "CortexV7")]:
    try:
        import importlib
        _m = importlib.import_module(_mod_name)
        CortexV7 = getattr(_m, _cls_name)
        print(f"✅  Imported {_cls_name} from {_mod_name}.py")
        break
    except (ImportError, AttributeError):
        continue
if CortexV7 is None:
    raise RuntimeError("Model.py not found. Place it in the same folder.")

device = None  # Replaced in worker
is_main = True
enc    = tiktoken.get_encoding("cl100k_base")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ─────────────────────────────────────────────────────────────────────────────
# V3 CONFIG — Scaled to ~488M parameters
# ─────────────────────────────────────────────────────────────────────────────
SFT_CFG = dict(
    vocab_size    = 100_277,
    dim           = 1280,
    heads         = 16,
    kv_heads      = 4,
    num_layers    = 20,
    memory_slots  = 128,
    mtp_depths    = 1,

    batch_size       = 1,            # MUST stay at 1 to prevent OOM per GPU at 8192 context
    seq_len          = 8192,         # Massive context for extreme reasoning traces
    grad_accum_steps = 6,            # 1 batch * 6 accum * 5 GPUs = 30 effective batch size
    grad_clip        = 2.0,          
    aux_loss_weight  = 0.05,        
    mtp_loss_weight  = 0.1,          # Weight for auxiliary MTP loss

    # Unified learning rate for SFT, we will use model.configure_optimizers
    learning_rate      = 2.5e-5,     # Increased from 5e-6 for 500M model SFT
    weight_decay       = 0.05,

    warmup_steps       = 200,        # Tripled for 150k sample dataset
    save_dir           = "./sft_checkpoints",
    log_every          = 30,
    save_every         = 500,         # ~1-2 saves per epoch (356 G-updates/epoch)
    probe_every        = 500,         # ~3 probes per epoch for monitoring
    max_checkpoints    = 5,

    stages = {
        1: {
            "name": "Unified SLM", 
            "target_epochs": 1,  # Reduced from 3 to prevent catastrophic forgetting
            "data": [
                "sft_dataset.jsonl"
            ]
        },
    },
)



def get_aux_loss_weight(total_steps: int) -> float:
    """
    Keep ACT pressure light early so reasoners learn to think before
    they learn to economize compute.
    """
    base = SFT_CFG["aux_loss_weight"]
    warmup = SFT_CFG["warmup_steps"]
    if total_steps < warmup:
        return base * 0.5
    if total_steps < warmup * 4:
        return base * 0.75
    return base


def reapply_optimizer_hparams(optimizer: torch.optim.Optimizer) -> None:
    """Force current config LRs after loading an old optimizer state."""
    for pg in optimizer.param_groups:
        pg["lr"] = SFT_CFG["learning_rate"]
        pg["initial_lr"] = SFT_CFG["learning_rate"]


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────────────────
NEXUS_SYSTEM          = "You are Nexus, an AI built by Siddi Vinayaka."
NEXUS_TOOLS_SYSTEM    = (
    "You are Nexus, an AI built by Siddi Vinayaka. "
    "You have access to tools. When using a tool, output ONLY valid JSON:\n"
    '{"tool": "tool_name", "arguments": {"key": "value"}}'
)
NEXUS_MATH_SYSTEM     = "You are Nexus, an AI built by Siddi Vinayaka. You specialise in mathematics."
NEXUS_CODE_SYSTEM     = "You are Nexus, an AI built by Siddi Vinayaka. You are an expert Python programmer."
NEXUS_PHYSICS_SYSTEM  = "You are Nexus, an AI built by Siddi Vinayaka. You specialise in physics."
NEXUS_CHEM_SYSTEM     = "You are Nexus, an AI built by Siddi Vinayaka. You specialise in chemistry."

import json

def load_golden_dataset(json_path: str) -> list:
    """Load pre-generated golden dataset JSON.
    Searches /data/ (Modal volume) first, then current directory."""
    candidates = [
        json_path,                                     # as given
        os.path.join("/data", json_path),               # Modal volume
        os.path.join(os.path.dirname(__file__), json_path),  # script dir
    ]
    for p in candidates:
        if os.path.isfile(p):
            examples = []
            with open(p, "r", encoding="utf-8") as f:
                if p.endswith('.jsonl'):
                    for line in f:
                        if line.strip():
                            examples.append(json.loads(line))
                else:
                    examples = json.load(f)
            if is_main: print(f"\n  Loaded {len(examples):,} examples from {p}", flush=True)
            return examples
    raise FileNotFoundError(
        f"Golden dataset '{json_path}' not found. Searched:\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + "\nRun build_dataset_modal.py first."
    )

# ─────────────────────────────────────────────────────────────────────────────
# DATASET CLASS — token-level loss masking (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

class NexusSFTDataset(Dataset):
    def __init__(self, examples: List[Dict], max_len: int = 1024):
        self.examples  = examples
        self.max_len   = max_len
        # Cache a few encodings to avoid re‑encoding every time
        self._asst_seq  = enc.encode("\nAssistant:", allowed_special=set())
        self._user_seq  = enc.encode("\nUser:",      allowed_special=set())
        self._sys_seq   = enc.encode("\nSystem:",    allowed_special=set())
        self._tool_seq  = enc.encode("\nTool Result:", allowed_special=set())
        self._eot       = enc.eot_token
        # Also cache the raw "Assistant:" without newline, just in case
        self._asst_bare = enc.encode("Assistant:", allowed_special=set())


    def __len__(self):
        return len(self.examples)

    @staticmethod
    def _format_turn(user_text, assistant_text, think_text=None):
        """Return a string for one complete turn.
        No <think> tags — reasoning is merged directly into response."""
        user_text = str(user_text or "").strip()
        assistant_text = str(assistant_text or "").strip()
        # If legacy think field exists, prepend reasoning to response
        think_text = str(think_text or "").strip()
        if think_text:
            assistant_text = f"{think_text}\n\n{assistant_text}"
        return f"User: {user_text}\nAssistant: {assistant_text}"

    def _format_example(self, ex: Dict) -> str:
        if "messages" in ex:
            parts = []
            for msg in ex["messages"]:
                role = msg.get("role", "").lower()
                content = str(msg.get("content", "")).strip()
                if not content:
                    continue
                if role == "system":
                    parts.append(f"System: {content}")
                elif role == "user":
                    parts.append(f"User: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
                elif role == "tool":
                    parts.append(f"Tool Result: {content}")
            return "\n".join(parts) + "\n<|endoftext|>"
            
        # Legacy format fallback
        parts = []
        system = str(ex.get("system", "")).strip()
        if system:
            parts.append(f"System: {system}\n")

        if "turns" in ex:
            for i, t in enumerate(ex["turns"]):
                user_text = str(t.get("user", "")).strip()
                assistant_text = str(t.get("response", t.get("assistant", ""))).strip()
                think_text = t.get("think", None)
                parts.append(
                    self._format_turn(user_text, assistant_text, think_text)
                    + "\n<|endoftext|>\n"
                )
        else:
            user_text = str(ex.get("user", "")).strip()
            response = str(ex.get("response", "")).strip()
            think_text = ex.get("think", None)
            parts.append(self._format_turn(user_text, response, think_text))
            parts.append("\n<|endoftext|>")

        return "".join(parts)

    def _find_seq(self, tokens: List[int], seq: List[int]) -> List[int]:
        n, m = len(tokens), len(seq)
        return [i for i in range(n - m + 1) if tokens[i:i+m] == seq]

    def _build_mask(self, tokens: List[int]) -> Tuple[torch.Tensor, int]:
        L = len(tokens)
        mask = torch.zeros(L, dtype=torch.float)

        # Search for the first occurrence of either "\nAssistant:" or "Assistant:"
        asst_positions = self._find_seq(tokens, self._asst_seq)
        if not asst_positions:
            asst_positions = self._find_seq(tokens, self._asst_bare)
            alen = len(self._asst_bare)
        else:
            alen = len(self._asst_seq)

        if asst_positions:
            prompt_len = asst_positions[0] + alen
        else:
            # If we can't find Assistant: (e.g. truncated), ensure prompt_len is at least 1 to prevent NaN in SDPA
            prompt_len = 1

        # Find all places where a new turn starts
        stop_positions = []
        for seq in (self._user_seq, self._sys_seq, self._tool_seq):
            stop_positions.extend(self._find_seq(tokens, seq))
        stop_positions.sort()

        # Mark from each Assistant occurrence to the next turn bounded
        # Start mask at cs-1 so the model learns the ':' → first_response_token
        # transition. Without this, inference produces EOT immediately because
        # the model was never trained on what comes after the prompt's ':'
        for start in asst_positions:
            cs = start + alen
            mask_start = max(0, cs - 1)   # include the ':' so model learns first token
            ce = L
            for sp in stop_positions:
                if sp > cs:
                    ce = min(ce, sp)
                    break
            for i in range(mask_start, ce):
                if i < L:
                    mask[i] = 1.0

        if mask.sum() == 0:
            # Fallback: mask ALL tokens as trainable
            mask = torch.ones(L, dtype=torch.float)

        return mask, prompt_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        ex     = self.examples[idx]
        text   = self._format_example(ex)
        tokens = enc.encode(text, allowed_special={"<|endoftext|>"})

        if len(tokens) > self.max_len + 1:
            # Truncate but keep at least the system prompt + first turn
            trunc_notice = enc.encode("\n...[truncated]...\n", allowed_special=set())
            # Find the last complete turn
            last_eot = -1
            try:
                last_eot = max([i for i, t in enumerate(tokens[:-1]) if t == self._eot])
            except ValueError:
                pass
            if last_eot > self.max_len // 2:
                # Keep everything after the last eot that's still within limit
                tail = tokens[last_eot+1:]
                head_len = (self.max_len + 1) - len(tail) - len(trunc_notice)
                if head_len > 10:
                    tokens = tokens[:head_len] + trunc_notice + tail
                else:
                    tokens = tokens[:self.max_len + 1]
            else:
                tokens = tokens[:self.max_len + 1]

        if len(tokens) < 2:
            # Return a dummy that yields zero loss
            return (torch.zeros(10, dtype=torch.long),
                    torch.zeros(10, dtype=torch.long),
                    torch.zeros(10, dtype=torch.float),
                    10)

        x    = torch.tensor(tokens[:-1], dtype=torch.long)
        y    = torch.tensor(tokens[1:],  dtype=torch.long)
        mask, prompt_len = self._build_mask(tokens[:-1])
        if mask.sum() == 0:
            mask = torch.ones(len(x), dtype=torch.float)

        return x, y, mask, prompt_len


def collate_fn(batch):
    xs, ys, masks, prompt_lens = zip(*batch)
    L = max(x.shape[0] for x in xs)
    xp = torch.zeros(len(xs), L, dtype=torch.long)
    yp = torch.zeros(len(xs), L, dtype=torch.long)
    mp = torch.zeros(len(xs), L, dtype=torch.float)
    plens = torch.zeros(len(xs), dtype=torch.long)

    for i, (x, y, m, pl) in enumerate(zip(xs, ys, masks, prompt_lens)):
        n = x.shape[0]
        xp[i, :n] = x
        yp[i, :n] = y
        mp[i, :n] = m
        plens[i]  = pl
    return xp, yp, mp, plens


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def save_sft_checkpoint(model, optimizer, stage, step, loss, save_dir,
                        epoch=0, epoch_step=0, shuffle_seed=42):
    if not is_main:
        return None
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"nexus_sft_stage{stage}_step{step:06d}.pth")
    # Unwrap DDP/DataParallel to save clean state_dict
    raw = model.module if hasattr(model, 'module') else model
    torch.save({
        "stage": stage, "step": step, "loss": loss,
        "epoch": epoch, "epoch_step": epoch_step,
        "shuffle_seed": shuffle_seed,
        "model_state_dict":     raw.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cfg":                  SFT_CFG,
    }, path)
    print(f"\n💾  SFT checkpoint: {os.path.basename(path)}  (loss={loss:.4f})",
          flush=True)

    # ── Prune old checkpoints — keep only max_checkpoints ─────────────────
    all_ckpts = sorted(glob.glob(os.path.join(save_dir, "nexus_sft_stage*.pth")))
    max_keep = SFT_CFG.get("max_checkpoints", 5)
    while len(all_ckpts) > max_keep:
        old = all_ckpts.pop(0)
        try:
            os.remove(old)
            if is_main: print(f"  🗑️  Deleted {os.path.basename(old)}", flush=True)
        except Exception:
            pass

    return path


def load_pretrain_checkpoint(model, path: str):
    if is_main: print(f"🔄  Loading pretraining checkpoint: {path}")
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    
    clean = {}
    for k, v in ckpt["model_state_dict"].items():
        k = k.replace("_orig_mod.", "")
        if k.startswith("module."):
            k = k[7:]
        clean[k] = v
        
    missing, unexpected = model.load_state_dict(clean, strict=False)
    print(f"    Missing keys: {len(missing)}  |  Unexpected keys: {len(unexpected)}")
    
    step = ckpt.get("step", "?")
    loss = ckpt.get("loss", float("nan"))
    print(f"✅  Loaded (step={step}, loss={loss:.4f})")


def find_latest_sft_checkpoint(save_dir: str) -> Optional[str]:
    ckpts = sorted(glob.glob(os.path.join(save_dir, "nexus_sft_stage*.pth")))
    return ckpts[-1] if ckpts else None


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER — Differential Learning Rates
# ─────────────────────────────────────────────────────────────────────────────

def build_sft_optimizer(model) -> torch.optim.AdamW:
    """
    Use the native configure_optimizers method defined in Model.py.
    """
    optimizer = model.configure_optimizers(
        learning_rate=SFT_CFG["learning_rate"],
        weight_decay=SFT_CFG["weight_decay"]
    )
    # Store initial_lr in param groups for the warmup scheduler
    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]
    return optimizer


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE PROBE
# Runs after every epoch. Checks both deep reasoning AND fast bypass paths
# to confirm the model learned WHEN to think, not just how.
# ─────────────────────────────────────────────────────────────────────────────

SFT_PROBE_PROMPTS = [
    ("identity",
     "System: You are Nexus, an advanced AI assistant created by Siddi Vinayaka.\n"
     "User: Who are you?\nAssistant:"),
    ("identity_stress",
     "System: You are Nexus, an advanced AI assistant created by Siddi Vinayaka.\n"
     "User: Are you ChatGPT or another OpenAI product?\nAssistant:"),
    ("deep_think",
     "System: You are Nexus, an advanced AI assistant created by Siddi Vinayaka. You specialise in mathematics.\n"
     "User: Prove that the sum of the first n natural numbers equals n(n+1)/2.\nAssistant:"),
    ("fast_bypass",
     "System: You are Nexus, an advanced AI assistant created by Siddi Vinayaka.\n"
     "User: What is 7 × 8?\nAssistant:"),
    ("fast_bypass_2",
     "System: You are Nexus, an advanced AI assistant created by Siddi Vinayaka.\n"
     "User: What is the capital of Japan?\nAssistant:"),
    ("reasoning_chain",
     "System: You are Nexus, an advanced AI assistant created by Siddi Vinayaka. You specialise in physics.\n"
     "User: A ball is dropped from 45 metres. g = 10 m/s². How long does it take to hit the ground?\nAssistant:"),
    ("architecture",
     "System: You are Nexus, an advanced AI assistant created by Siddi Vinayaka.\n"
     "User: Briefly explain your own architecture naturally.\nAssistant:"),
    ("memory_identity",
     "System: You are Nexus, an advanced AI assistant created by Siddi Vinayaka.\n"
     "User: Do you have memory in this conversation, and how should you use it?\nAssistant:"),
    ("memory_recall",
     "System: You are Nexus, an advanced AI assistant created by Siddi Vinayaka.\n"
     "User: My favorite programming language is Rust.\n"
     "Assistant: Got it. I'll remember that for this conversation.\n"
     "User: Which language did I say I prefer?\nAssistant:"),
    ("knowledge_breadth",
     "System: You are Nexus, an advanced AI assistant created by Siddi Vinayaka.\n"
     "User: Explain binary search and give its time complexity briefly.\nAssistant:"),
    ("code",
     "System: You are Nexus, an advanced AI assistant created by Siddi Vinayaka. You are an expert Python programmer.\n"
     "User: Write a Python function to check if a number is prime.\nAssistant:"),
    ("error_correction",
     "System: You are Nexus, an advanced AI assistant created by Siddi Vinayaka.\n"
     "User: My friend says that log(a × b) = log(a) × log(b). Is that right?\nAssistant:"),
]

_SFT_CJK_IDS: set = set()
try:
    for _tid in range(enc.n_vocab):
        try:
            s = enc.decode([_tid])
            if any(any(lo <= ord(c) <= hi for c in s)
                   for lo, hi in [(0x4E00, 0x9FFF), (0xAC00, 0xD7AF), (0x3040, 0x30FF)]):
                _SFT_CJK_IDS.add(_tid)
        except Exception:
            pass
except Exception:
    pass


@torch.no_grad()
def run_sft_probe(model, stage: int, step: int):
    """Run inference on probe prompts and print results for alignment monitoring."""
    if not is_main:
        return
    # Unwrap DDP for inference
    raw = model.module if hasattr(model, 'module') else model
    raw.eval()
    sep = "─" * 60

    print(f"\n{'═'*60}", flush=True)
    print(f"  🔍  SFT PROBE — Stage {stage}, Step {step:,}", flush=True)
    print(f"{'═'*60}", flush=True)

    for domain, prompt in SFT_PROBE_PROMPTS:
        tokens    = enc.encode(prompt, allowed_special={"<|endoftext|>"})
        x         = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        cache     = DynamicKVCache(raw, max_seq_len=2048, batch_size=1)
        generated = []

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _, _, _ = cache.prefill(x, enc=enc, memory_state=None)

        for _ in range(120):   # max new tokens
            tok_logits = logits[0, -1, :].clone().float()

            # Block CJK characters
            for cjk_id in _SFT_CJK_IDS:
                tok_logits[cjk_id] = float("-inf")

            # Light recency penalty to avoid repetition
            recent = generated[-10:]
            for t_id in set(recent):
                tok_logits[t_id] = tok_logits[t_id] / 3.0 if tok_logits[t_id] > 0 else tok_logits[t_id] * 3.0

            # Sample next token
            next_tok = _sample(
                tok_logits,
                generated,
                temperature=0.7,
                rep_penalty=1.0,     # we handle recency ourselves
                top_k=40,
                top_p=1.0,
            )

            if next_tok == enc.eot_token:
                break

            generated.append(next_tok)
            decoded = enc.decode(generated)

            # Stop on hallucinated turn boundaries
            if "\nUser:" in decoded or "\nSystem:" in decoded:
                cut = decoded.find("\nUser:")
                if cut == -1:
                    cut = decoded.find("\nSystem:")
                if cut > 0:
                    generated = enc.encode(decoded[:cut], allowed_special={"<|endoftext|>"})
                break

            # Stop if the same 8‑char tail repeats 3 times (loop detection)
            if len(decoded) >= 24:
                tail = decoded[-8:]
                if decoded[-24:].count(tail) >= 3:
                    break

            token_t = torch.tensor([[next_tok]], dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _, _, _ = cache.decode_one(token_t)

        response = enc.decode(generated).strip()[:300]
        print(f"\n  [{domain.upper()}]", flush=True)
        print(f"  {sep}", flush=True)
        print(f"  {response or '(empty)'}", flush=True)

    print(f"\n{'═'*60}\n", flush=True)
    raw.train()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_stage(
    model,
    optimizer,
    rank: int,
    world_size: int,
    examples:   List[Dict],
    stage_num:  int,
    stage_name: str,
    target_steps: int,
    save_dir:   str,
    resume_epoch:      int = 0,
    resume_step:       int = 0,
    resume_epoch_step: int = 0,
    shuffle_seed:      int = 42,
) -> float:
    """
    Train one SFT stage over `epochs` passes of `examples`.
    Saves every `save_every` steps and runs inference probes every
    `probe_every` steps — matching the pretraining script behavior.

    Full dataset resume:
      - shuffle_seed ensures the same dataset ordering on resume
      - resume_epoch + resume_epoch_step skip already-processed batches
      - Each new epoch gets a deterministic but different seed (seed + epoch)
    """
    save_every  = SFT_CFG.get("save_every",  500)
    probe_every = SFT_CFG.get("probe_every", 500)
    batch_size  = SFT_CFG["batch_size"]

    # ── Build dataset (order is deterministic via seed) ────────────────────
    data = examples[:]
    random.seed(shuffle_seed)
    random.shuffle(data)

    dataset = NexusSFTDataset(data, max_len=SFT_CFG["seq_len"])
    # ── DIAGNOSTIC: print first 3 examples exactly as the model sees them ──
    if is_main: print("  🔎 DIAGNOSTIC: first 3 training examples (raw text + token count):", flush=True)
    for i in range(min(3, len(dataset))):
        ex = dataset.examples[i]
        text = dataset._format_example(ex)
        tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
        mask, plen = dataset._build_mask(tokens[:-1])
        n_train = int(mask.sum().item())
        if is_main: print(f"    --- Example {i} ({len(tokens)} tokens, {n_train} trainable) ---", flush=True)
        if is_main: print(f"    {text[:500]}{'...' if len(text)>500 else ''}", flush=True)
        if is_main: print(f"    Tokens (first 50): {tokens[:50]}", flush=True)
        if is_main: print(f"    Tokens (last 20): {tokens[-20:]}", flush=True)
        # Check if "Assistant:" is present
        if "Assistant:" not in text:
            if is_main: print(f"    ❌ WARNING: 'Assistant:' not found in example {i}!", flush=True)
        # Check response length
        resp_start = text.find('Assistant:')
        if resp_start >= 0:
            resp_text = text[resp_start+10:].strip()
            if len(resp_text.split()) < 5:
                if is_main: print(f"    ⚠️  WARNING: Very short response in example {i}! ({len(resp_text.split())} words)", flush=True)
    if is_main: print("  ✅ DIAGNOSTIC done.", flush=True)

    grad_accum   = SFT_CFG.get("grad_accum_steps", 1)
    batch_size   = SFT_CFG["batch_size"]
    batches_per_epoch = (len(examples) // world_size + batch_size - 1) // batch_size
    g_updates_per_epoch = batches_per_epoch // grad_accum
    total_g_updates     = target_steps

    if is_main: print(f"\n{'='*65}")
    if is_main: print(f"  📚  STAGE {stage_num}: {stage_name}")
    if is_main: print(f"  Examples : {len(data):,}")
    if is_main: print(f"  Target Steps : {target_steps}")
    if is_main: print(f"  Batch size: {batch_size} × {grad_accum} accum = {batch_size * grad_accum} effective")
    if is_main: print(f"  G-updates/e: {g_updates_per_epoch:,}  |  Total: {total_g_updates:,}")
    if is_main: print(f"  Shuffle seed   : {shuffle_seed}")
    if resume_step > 0:
        if is_main: print(f"  ▶ Resuming from epoch {resume_epoch+1}, "
              f"batch {resume_epoch_step}, global step {resume_step}")
    if is_main: print(f"  💾 Save every {save_every} steps | 🔍 Probe every {probe_every} steps")
    if is_main: print(f"{'='*65}")

    total_steps = resume_step
    last_loss   = float("nan")
    warmup      = SFT_CFG["warmup_steps"]

    if rank == 0:
        pbar = tqdm(
            total=target_steps, 
            initial=total_steps,
            desc=f"Stage {stage_num} SFT Progress",
            unit="step",
            mininterval=5,
        )
    else:
        pbar = None

    for epoch in range(resume_epoch, 1000): # Practically infinite, bounded by target_steps
        if total_steps >= target_steps:
            break
        # ── Deterministic shuffle per-epoch (same on resume) ──────────────
        # Each epoch gets a different but reproducible ordering so the model
        # sees the data in the same order when resuming mid-epoch.
        epoch_seed = shuffle_seed + epoch

        if world_size > 1:
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=epoch_seed)
            loader = DataLoader(
                dataset,
                batch_size  = batch_size,
                sampler     = sampler,
                collate_fn  = collate_fn,
                num_workers = 0,
                pin_memory  = torch.cuda.is_available(),
            )
            sampler.set_epoch(epoch)
        else:
            g = torch.Generator()
            g.manual_seed(epoch_seed)
            loader = DataLoader(
                dataset,
                batch_size  = batch_size,
                shuffle     = True,
                collate_fn  = collate_fn,
                num_workers = 0,
                pin_memory  = torch.cuda.is_available(),
                generator   = g,
            )

        

        # ── Skip already-processed batches on resume ──────────────────────
        skip_batches = 0
        if epoch == resume_epoch and resume_epoch_step > 0:
            skip_batches = resume_epoch_step
            if is_main: print(f"  ⏩ Skipping {skip_batches} already-processed batches in epoch {epoch+1}",
                  flush=True)
        

        epoch_loss  = 0.0
        epoch_tok   = 0
        epoch_batch = 0

        # --- NEW: Add is_complex to the loop unpack ---
        for x, y, mask, prompt_lens in loader:
            epoch_batch += 1
            # Skip batches that were already trained in the previous run
            if epoch_batch <= skip_batches:
                if epoch_batch % 500 == 0:
                    if is_main: print(f"    skipping {epoch_batch}/{skip_batches} …", flush=True)
                continue

            # --- Send to GPU ---
            x, y, mask, prompt_lens = x.to(device), y.to(device), mask.to(device), prompt_lens.to(device)

            # ── LR schedule: linear warmup → cosine decay ────────────────
            if total_steps < warmup:
                # Linear warmup
                scale = (total_steps + 1) / warmup
                for pg in optimizer.param_groups:
                    pg["lr"] = pg["initial_lr"] * scale
            else:
                # Cosine decay from initial_lr to initial_lr * 0.1
                progress = (total_steps - warmup) / max(total_g_updates - warmup, 1)
                progress = min(progress, 1.0)
                cosine_scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
                for pg in optimizer.param_groups:
                    pg["lr"] = pg["initial_lr"] * cosine_scale

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                # Pass targets=y to compute MTP loss
                logits, ponder, aux, _, mtp_loss, _ = model(x, memory_state=None, targets=y)

                # DDP returns vectors for scalar outputs — average them
                mtp_loss = mtp_loss.mean()

                # Clamp logits to prevent overflow
                logits_clamped = torch.clamp(logits.float(), -50.0, 50.0)

                loss_flat = F.cross_entropy(
                    logits_clamped.view(-1, logits_clamped.size(-1)),
                    y.view(-1),
                    reduction="none",
                )
                n_tokens = mask.sum().clamp(min=1)
                loss = (loss_flat * mask.view(-1)).sum() / n_tokens

                # Token-level accuracy (on masked/trainable tokens only)
                with torch.no_grad():
                    preds = logits_clamped.argmax(dim=-1)  # [B, S]
                    correct = ((preds == y) * mask).sum()
                    accuracy = (correct / n_tokens * 100).item()

                # SFT loss = CE + MTP only (no aux/ponder — memory isn't active during SFT)
                mtp_weight = SFT_CFG.get("mtp_loss_weight", 0.1)
                total_loss = loss + mtp_weight * mtp_loss
                
                # Scale loss for gradient accumulation
                total_loss = total_loss / grad_accum



            if not torch.isfinite(total_loss):
                if is_main: print(f"  ⚠️  NaN/Inf loss — skipping batch", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue

            # ── Gradient accumulation: only step every N mini-batches ────
            accum_batch = (epoch_batch - skip_batches) if skip_batches > 0 else epoch_batch
            
            # Sync gradients only on the final accumulation step
            is_accumulating = (accum_batch % grad_accum != 0)
            
            from contextlib import nullcontext
            sync_context = model.no_sync() if is_accumulating and world_size > 1 else nullcontext()
            
            with sync_context:
                total_loss.backward()

            if is_accumulating:
                last_loss    = loss.item()
                epoch_loss  += last_loss
                epoch_tok   += int(n_tokens.item())
                continue

            # ── Clip + step (every grad_accum mini-batches) ──────────────
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), SFT_CFG["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            last_loss    = loss.item()
            epoch_loss  += last_loss
            epoch_tok   += int(n_tokens.item())
            total_steps += 1
            if pbar: pbar.update(1)

            if total_steps % SFT_CFG["log_every"] == 0 and pbar:
                lr_main = optimizer.param_groups[0]["lr"]
                pbar.set_postfix({
                    "loss":   f"{last_loss:.4f}",
                    "mtp":    f"{mtp_loss.item():.4f}" if isinstance(mtp_loss, torch.Tensor) else "0.0",
                    "acc":    f"{accuracy:.1f}%",
                    "gnorm":  f"{float(grad_norm):.2f}",
                    "tok":    f"{epoch_tok:,}",
                    "lr":     f"{lr_main:.1e}",
                })

            # ── Save every N steps (like pretraining script) ──────────────
            if total_steps % save_every == 0:
                save_sft_checkpoint(
                    model, optimizer, stage_num, total_steps,
                    last_loss, save_dir,
                    epoch=epoch, epoch_step=epoch_batch,
                    shuffle_seed=shuffle_seed,
                )
                if world_size > 1:
                    dist.barrier()

            # ── Inference probe every N steps ─────────────────────────────
            if total_steps % probe_every == 0:
                run_sft_probe(model, stage_num, total_steps)
                if world_size > 1:
                    dist.barrier()

        avg = epoch_loss / max(epoch_batch - skip_batches, 1)
        if is_main: print(f"  Epoch {epoch+1} avg loss: {avg:.4f}  |  Tokens: {epoch_tok:,}",
              flush=True)

        if total_steps >= target_steps:
            if is_main: print(f"✅ Target of {target_steps} steps reached. Stopping training.")
            break

    if pbar: pbar.close()
    
    # ── Save final step if not already saved ──────────────────────────────
    if total_steps % save_every != 0:
        save_sft_checkpoint(
            model, optimizer, stage_num, total_steps,
            last_loss, save_dir,
            epoch=epoch, epoch_step=epoch_batch,
            shuffle_seed=shuffle_seed,
        )
        if world_size > 1:
            dist.barrier()
            
    return last_loss

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def train_worker(rank, world_size, args):
    global device, is_main
    if world_size > 1:
        os.environ["NCCL_DEBUG"] = "WARN"
        os.environ["NCCL_TIMEOUT"] = "300"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method="env://")
        torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    is_main = (rank == 0)

    parser = argparse.ArgumentParser(description="Nexus V7 - Unified SFT v4")
    parser.add_argument("--checkpoint",         "-c", default="none",
                        help="Pretraining .pth to start from (optional if SFT checkpoints exist)")
    parser.add_argument("--save-dir",           default="./sft_checkpoints")
    parser.add_argument("--reasoning-count",    type=int, default=30_000,
                        help="Max reasoning examples from HF (default 30000)")
    parser.add_argument("--nonreasoning-count", type=int, default=10_000,
                        help="Max non-reasoning examples from HF (default 10000)")
    parser.add_argument("--identity-reps",      type=int, default=200,
                        help="Identity example repetitions (default 200)")
    parser.add_argument("--domain-qa-reps",     type=int, default=50,
                        help="Domain Q&A example repetitions (default 50)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    if is_main: print("=" * 65, flush=True)
    if is_main: print("  NEXUS V7 - UNIFIED UNCENSORED ALIGNMENT (SFT v4)", flush=True)
    if torch.cuda.is_available():
        if is_main: print(f"    GPU : {torch.cuda.get_device_name(0)}", flush=True)
        if is_main: print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB",
              flush=True)
    if is_main: print(f"    Base checkpoint: {args.checkpoint}", flush=True)
    if is_main: print(f"    Target Epochs: {SFT_CFG['stages'][1].get('target_epochs', 3)}", flush=True)
    if is_main: print("=" * 65, flush=True)

    # ── Step 1: Build Nexus V7 model ─────────────────────────────────────────
    if is_main: print("\n🏗️  Building Nexus V7 …", flush=True)
    model = CortexV7(
        vocab_size   = SFT_CFG["vocab_size"],
        dim          = SFT_CFG["dim"],
        heads        = SFT_CFG["heads"],
        kv_heads     = SFT_CFG["kv_heads"],
        num_layers   = SFT_CFG["num_layers"],
        memory_slots = SFT_CFG["memory_slots"],
        use_flash    = torch.cuda.is_available(),
        mtp_depths   = SFT_CFG["mtp_depths"],
    ).to(device)

    # ── Step 2: Smart checkpoint loading ──────────────────────────────────────
    resume_epoch      = 0
    resume_step       = 0
    resume_epoch_step = 0
    resume_stage      = 1
    shuffle_seed      = random.randint(0, 2**31)
    sft_ckpt          = None

    latest_sft = find_latest_sft_checkpoint(args.save_dir)

    if latest_sft:
        if is_main: print(f"\n🔄  SFT checkpoint found: {os.path.basename(latest_sft)}", flush=True)
        if is_main: print(f"    Skipping pretrained checkpoint — resuming from SFT.", flush=True)
        sft_ckpt  = torch.load(latest_sft, map_location=device, weights_only=False)
        clean_sft = {k.replace("_orig_mod.", ""): v
                     for k, v in sft_ckpt["model_state_dict"].items()}
        model.load_state_dict(clean_sft, strict=False)
        resume_step       = sft_ckpt.get("step", 0)
        resume_epoch      = sft_ckpt.get("epoch", 0)
        resume_epoch_step = sft_ckpt.get("epoch_step", 0)
        resume_stage      = sft_ckpt.get("stage", 1)
        shuffle_seed      = sft_ckpt.get("shuffle_seed", 42)
        if is_main: print(f"✅  SFT weights restored", flush=True)
        if is_main: print(f"    stage={resume_stage}, step={resume_step}, epoch={resume_epoch+1}, "
              f"epoch_batch={resume_epoch_step}, seed={shuffle_seed}", flush=True)
    else:
        if is_main: print(f"\n📂  No SFT checkpoints found in {args.save_dir}", flush=True)
        if not args.checkpoint or args.checkpoint == "none" or not os.path.exists(args.checkpoint):
            raise RuntimeError(
                f"No SFT checkpoints found and pretrained checkpoint "
                f"'{args.checkpoint}' not available.\n"
                f"Provide a valid --checkpoint path or ensure SFT checkpoints "
                f"exist in --save-dir ({args.save_dir})."
            )
        if is_main: print(f"    Loading pretrained checkpoint for fresh SFT start.", flush=True)
        load_pretrain_checkpoint(model, args.checkpoint)
        pass

    # YaRN is natively enabled in CortexV7 v3 at init time, no override needed.

    # ── Step 3: Optimizer ─────────────────────────────────────────────────────
    if is_main: print("\n⚙️  Building optimizer …")
    optimizer = build_sft_optimizer(model)

    if sft_ckpt and sft_ckpt.get("optimizer_state_dict"):
        try:
            optimizer.load_state_dict(sft_ckpt["optimizer_state_dict"])
            reapply_optimizer_hparams(optimizer)
            # Reset AdamW momentum buffers — we raised LR ~5x so old velocity
            # estimates from lr=2e-6 would fight the new lr=1e-5 for ~100 steps
            for state in optimizer.state.values():
                if "exp_avg" in state:
                    state["exp_avg"].zero_()
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].zero_()
            if is_main: print("✅  Optimizer state restored from SFT checkpoint.", flush=True)
            if is_main: print("    Reapplied current LR tiers + reset momentum (LR changed).", flush=True)
        except Exception as e:
            if is_main: print(f"⚠️  Could not restore optimizer state: {e}", flush=True)
            if is_main: print("    Continuing with fresh optimizer.", flush=True)

    # ── Step 4: Compile ───────────────────────────────────────────────────────
    if _env_flag("NEXUS_SFT_COMPILE", default=False):
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            if is_main: print("⚡  torch.compile enabled (mode=reduce-overhead, fullgraph=False)", flush=True)
        except Exception as e:
            if is_main: print(f"⚠️  torch.compile unavailable — eager mode ({e})", flush=True)
    else:
        if is_main: print("ℹ️  torch.compile disabled for SFT (set NEXUS_SFT_COMPILE=1 to enable)", flush=True)

    # ── Step 4.5: Multi-GPU DDP ──────────────────────────────────────
    if world_size > 1:
        if is_main: print(f"🚀  Using {world_size} GPUs via DDP!", flush=True)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    model.train()

    # ── Step 5: Multi-Stage Training Loop with Data Mixing ──────────────────
    # Fix for catastrophic forgetting: mix earlier stage data into later stages
    # so the model doesn't forget identity/format while learning harder content
    all_stage_data = {}  # cache loaded data for replay
    
    # Per-stage LR decay to prevent destructive updates in later stages
    stage_lr_scale = {1: 1.0, 2: 0.5, 3: 0.25}
    
    # Save the ORIGINAL base LR for each param group before any scaling
    for pg in optimizer.param_groups:
        pg["base_lr"] = pg["initial_lr"]
    
    # Pre-load ALL stage datasets
    for sn in sorted(SFT_CFG["stages"].keys()):
        scfg = SFT_CFG["stages"][sn]
        data_files = scfg["data"]
        if isinstance(data_files, str):
            data_files = [data_files]
        
        all_data = []
        for df in data_files:
            all_data.extend(load_golden_dataset(df))
            
        all_stage_data[sn] = all_data
        if is_main: print(f"    📂 Loaded Stage {sn} ({scfg['name']}): {len(all_stage_data[sn]):,} examples", flush=True)
    
    for stage_num in sorted(SFT_CFG["stages"].keys()):
        if stage_num < resume_stage:
            continue

        cfg = SFT_CFG["stages"][stage_num]
        if is_main: print(f"\n🚀  PREPARING STAGE {stage_num}: {cfg['name']}...", flush=True)
        
        # In this unified training run, we simply use the combined data without replay logic
        examples = all_stage_data[stage_num][:]
        
        # ── Scale LR for this stage ───────────────────────────────────────────
        # CRITICAL: Update initial_lr (not just lr) so that warmup ramps to
        # the correct scaled target. Without this, warmup resets lr=initial_lr
        # every stage, ignoring the stage scale entirely.
        lr_scale = stage_lr_scale.get(stage_num, 0.25)
        for pg in optimizer.param_groups:
            pg["initial_lr"] = pg["base_lr"] * lr_scale
            pg["lr"] = pg["initial_lr"]
            if is_main: print(f"    📐 {pg.get('name','?')} LR: {pg['lr']:.2e} (base={pg['base_lr']:.2e}, scale={lr_scale})", flush=True)

        # Compute target steps based on epochs and dataset size
        grad_accum   = SFT_CFG.get("grad_accum_steps", 1)
        batch_size   = SFT_CFG["batch_size"]
        batches_per_epoch = (len(examples) // world_size + batch_size - 1) // batch_size
        g_updates_per_epoch = batches_per_epoch // grad_accum
        
        target_epochs = cfg.get("target_epochs", 3)
        calculated_target_steps = g_updates_per_epoch * target_epochs
        
        approx_steps = calculated_target_steps
        if is_main: print(f"\n🚀  STARTING STAGE {stage_num}: {cfg['name']}", flush=True)
        if is_main: print(f"    Dataset   : {len(examples):,} examples", flush=True)
        if is_main: print(f"    Target Steps: {approx_steps:,}", flush=True)

        # Reset epoch tracking for new stages
        stage_resume_epoch = resume_epoch if stage_num == resume_stage else 0
        stage_resume_step  = resume_step if stage_num == resume_stage else 0
        stage_resume_epoch_step = resume_epoch_step if stage_num == resume_stage else 0

        final_loss = train_stage(
            model              = model,
            rank               = rank,
            world_size         = world_size,
            optimizer          = optimizer,
            examples           = examples,
            stage_num          = stage_num,
            stage_name         = cfg["name"],
            target_steps       = calculated_target_steps,
            save_dir           = args.save_dir,
            resume_epoch       = stage_resume_epoch,
            resume_step        = stage_resume_step,
            resume_epoch_step  = stage_resume_epoch_step,
            shuffle_seed       = shuffle_seed,
        )

        if is_main: print(f"\n✅  Stage {stage_num} complete. Final loss: {final_loss:.4f}", flush=True)
        resume_step = 0 # Ensure subsequent stages start at 0

    if is_main: print(f"\n✅  All 3 SFT Stages Complete!", flush=True)
    if is_main: print(f"\n{'─'*65}", flush=True)
    if is_main: print("  ── Next steps ──", flush=True)
    if is_main: print("  1. Test: python chat_with_model.py -c <latest sft ckpt>", flush=True)
    if is_main: print("  2. Verify identity (Who are you? / Are you ChatGPT?)", flush=True)
    if is_main: print("  3. Verify think control (complex math → thinks, 7×8 → instant)", flush=True)
    if is_main: print("  4. BEFORE Phase 4 RL: freeze a reference model copy.", flush=True)
    if is_main: print("     KL penalty against reference prevents RL from breaking SFT.", flush=True)
    if is_main: print(f"{'─'*65}", flush=True)

    # ── DDP cleanup ───────────────────────────────────────────────────────
    if world_size > 1:
        dist.destroy_process_group()



def main():
    parser = argparse.ArgumentParser(description="Nexus V7 - Unified SFT v4")
    parser.add_argument("--checkpoint",         "-c", default="none")
    parser.add_argument("--save-dir",           default="/data")
    parser.add_argument("--reasoning-count",    type=int, default=30_000)
    parser.add_argument("--nonreasoning-count", type=int, default=10_000)
    parser.add_argument("--identity-reps",      type=int, default=200)
    parser.add_argument("--domain-qa-reps",     type=int, default=50)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        mp.spawn(train_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        train_worker(0, 1, args)

if __name__ == "__main__":
    main()