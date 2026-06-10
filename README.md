# Nexus — 512M Parameter Transformer with Neural Memory

A custom transformer architecture built from scratch, featuring **Global Neural Memory** — a persistent memory system that flows through all layers. Trained on 10B+ tokens from scratch using 8x H100 GPUs on Modal.

**Creator:** Siddi Vinayaka — Independent AI Researcher

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Details](#component-details)
4. [Training](#training)
5. [Inference & Chat](#inference--chat)
6. [File Structure](#file-structure)
7. [Checkpoint Format](#checkpoint-format)
8. [Hyperparameters](#hyperparameters)

---

## Architecture Overview

Nexus is a 20-layer transformer (~488M parameters) with a brain-inspired memory architecture. Unlike standard transformers that process each token independently through residual streams, Nexus has **3 GlobalNeuralMemory bridges** that act as a persistent "hippocampus" — storing, retrieving, and updating information across the entire sequence.

### Key Specifications

| Component | Value |
|-----------|-------|
| Parameters | 488M (0.488B) |
| Layers | 20 |
| Hidden dim | 1280 |
| Attention heads | 16 (GQA with 4 KV heads) |
| Head dim | 80 |
| FFN | SwiGLU, 3417d hidden (2.67x expansion) |
| Vocab size | 100,277 |
| Tokenizer | cl100k_base (GPT-4 compatible) |
| Max seq len | 2,048 (trainable), extends to 8,192 via YaRN |
| Memory bridges | 3 (at layers 3, 10, 17) |
| Memory slots | 32 + 64 + 128 = 224 total |
| Precision | BFloat16 with Flash Attention 2 |

---

## Architecture Diagram

### High-Level Flow

```
Input Tokens
     |
     v
 [Embedding]  (vocab_size=100277 -> dim=1280)
     |
     v
 +-------+     +-----------+     +-----------+     +-------+
 | Embed | --> | Memory_1  | --> | Layers    | --> | Head  |
 |       |     | (32 slots)|     | 0-6       |     |       |
 +-------+     +-----------+     +-----------+     +-------+
                                    |
                              +-----------+
                              | Memory_2  |
                              | (64 slots)|
                              +-----------+
                                    |
                              +-----------+
                              | Layers    |
                              | 7-13      |
                              +-----------+
                                    |
                              +-----------+
                              | Memory_3  |
                              | (128 slots)|
                              +-----------+
                                    |
                              +-----------+
                              | Layers    |
                              | 14-19     |
                              +-----------+
                                    |
                              +-----------+
                              | Output    |
                              | Reflection|
                              +-----------+
                                    |
                              [LM Head]  (weight-tied with embedding)
                                    |
                              Output Logits
```

### Transformer Block (Each Layer)

```
     x (input)
      |
      +------ HyperConnection (learnable residual) ------+
      |                                                  |
  [RMSNorm]                                        (residual)
      |                                                  |
  [GQA Attention]  <-- RoPE (YaRN)                     |
      |               + Flash Attention 2                |
      +------ HyperConnection (learnable residual) ------+
      |                                                  |
  [RMSNorm]                                        (residual)
      |                                                  |
  [SwiGLU FFN]  (1280 -> 3417 -> 1280)               |
      |               + Inner RMSNorm                    |
      +------ HyperConnection (learnable residual) ------+
      |
     h (output)
```

### GlobalNeuralMemory Bridge

Each memory bridge operates in two phases:

```
                    READ PHASE
                    ==========
    x (hidden)  ──────┐
                      v
              +--------------+
              | Cross-Attn   | <── Memory Slots
              | (Q from x,   |     [B, N_slots, 1280]
              |  K,V from mem)|
              +--------------+
                      |
              +--------------+
              | Gated Merge  | gate = sigmoid(W[x; read_out])
              | x_new = gate * read_out + (1-gate) * x
              +--------------+
                      |
                 x_new (output)


                    WRITE PHASE
                    ===========
    x (hidden)  ──────┐
                      v
              +--------------+
              | Slot Scoring | score = mem_slot * context_summary / sqrt(D)
              | + Gumbel Noise| (exploration during training)
              +--------------+
                      |
              +--------------+
              | Hard Mask    | STE (Straight-Through Estimator)
              | top-k slots  | binary mask for slot selection
              +--------------+
                      |
              +--------------+
              | Cross-Attn   | <── x (hidden)
              | (Q from mem, |
              |  K,V from x) |
              +--------------+
                      |
              +--------------+
              | Gated Update | forget_gate * old_mem + (1-forget_gate) * write_scale * update
              | + MemNorm    |
              +--------------+
                      |
                 new_memory
```

### Memory Specialization

The 3 bridges have different roles and retention characteristics:

```
Bridge 0 (Lexical)     Bridge 1 (Semantic)    Bridge 2 (Reasoning)
After Layer 3          After Layer 10         After Layer 17
32 slots               64 slots               128 slots
retain_floor=0.3       retain_floor=0.5       retain_floor=0.7
write_scale=0.5        write_scale=0.3        write_scale=0.2
Role: "What do I        Role: "Update what      Role: "Consolidate
already know"           I've learned"           reasoning"
                       
Short retention,        Medium retention,      Long retention,
aggressive writes      balanced updates       conservative updates
```

### Output Reflection

After the final layer, the output head reads from all memory bridges:

```
                  h (final hidden)
                       |
                  [RMSNorm]
                       |
              +------------------+
              | Cross-Attention  | <── cat(all 3 memory bridges, dim=seq)
              | (final_memory    |     [B, 224, 1280]
              |  _read)          |
              +------------------+
                       |
              +------------------+
              | Adaptive Gate    | gate = sigmoid(W * h_normed)
              | h = h + gate * 0.1 * reflection
              +------------------+
                       |
                  [RMSNorm]
                       |
                  [LM Head] --> logits
```

---

## Component Details

### 1. SwiGLU FFN

```python
hidden = silu(w1(x)) * w2(x)    # SwiGLU activation
hidden = RMSNorm(hidden)         # Inner normalization (stabilizes deep networks)
output = w3(hidden)              # Project back to dim
```

The inner RMSNorm is a key stability feature — it prevents the hidden state from growing unbounded through the 20 layers.

### 2. GQA Attention (Grouped Query Attention)

- **16 query heads**, **4 KV heads** (shared across 4 query heads each)
- **RoPE**: YaRN extension — trains at 2048, extends to 8192 without quality loss
- **Flash Attention 2**: ~30% faster matmul on H100
- **QK Norm**: RMSNorm on Q and K before RoPE (prevents attention score explosion)

### 3. HyperConnections (ICLR 2025)

Replaces standard `x + f(x)` residuals with learnable multi-stream routing:

```python
# Standard:  result = x + f(x)
# Hyper:     result = beta[0] * f(x) + beta[1] * alpha[1] * x
#            alpha, beta are learned parameters
#            At init: alpha=[1,1], beta=[1,1] → identical to standard residual
#            During training: learns optimal cross-layer information routing
```

Applied to both attention and FFN outputs in every layer.

### 4. Multi-Token Prediction (MTP)

Inspired by DeepSeek-V3, the MTP head predicts the next-next token:

```python
# Input: hidden_state + embedding of next token
combined = concat(h_prev, embed(next_token))
h_mtp = MTPHead(combined)  # Linear(2*dim -> dim) + RMSNorm
logits_mtp = head(h_mtp)
loss_mtp = cross_entropy(logits_mtp, token_at_position+2)
```

This auxiliary loss (weight=0.1) improves sample efficiency during pretraining.

### 5. YaRN RoPE

Trains at original_max_len=2048, extends to 8192 via NTK-aware interpolation:
- High-frequency dimensions: unchanged (local patterns)
- Low-frequency dimensions: interpolated (absolute position)
- Smooth ramp controlled by beta_fast=32, beta_slow=1
- Attention scaling: `0.1 * log(yarn_scale) + 1.0`

---

## Training

### Prerequisites

- Python 3.11+
- PyTorch 2.3.1+ with CUDA 12.1+
- Modal account with H100 access
- HuggingFace token (for dataset access)

### Setup

```bash
pip install modal torch transformers datasets tiktoken tqdm flash-attn
```

### Training on Modal (8x H100)

```bash
# Start training (attached — see logs in real-time)
modal run Train.py

# Start training (detached — runs in background)
modal run --detach Train.py

# View logs
modal logs nexus-v1-training

# Download a checkpoint
modal volume get nexus-v1-ckpts ckpt_nexus_076500.pth .
```

### Training Configuration (Phase 3)

```python
# Model
dim = 1280
heads = 16 (GQA with 4 KV heads)
num_layers = 20
seq_len = 2048

# Training schedule
max_steps = 153,000        # Phase 3: 10B more tokens (total ~20B)
batch_size = 32 per GPU     # 4 sequences per GPU
grad_accum_steps = 4        # Effective global batch = 128 sequences
learning_rate = 1.5e-4      # Peak, cosine decay to 1.5e-5
warmup_steps = 500
weight_decay = 0.1
grad_clip = 2.0

# Regularization
hyper_beta_weight = 0.001   # L2 on HyperConnection beta
mem_diversity_weight = 0.05 # Prevent memory slot collapse
mem_temporal_weight = 0.3   # Encourage memory persistence
mtp_loss_weight = 0.1       # Multi-token prediction auxiliary loss

# Hardware
gpus = 8x H100 (80GB VRAM)
precision = bfloat16
timeout = 3 days
```

### Dataset Mix

```
20%  FineWeb-Edu 100BT     — General English, science, history
18%  FineMath               — Math reasoning
12%  Stack-v2 Dedup         — High-quality code
15%  Science                — Academic papers (OpenWebMath, textbooks)
18%  STEM-Reasoning CoT    — Step-by-step reasoning chains
8%   Code-Feedback          — Code with explanations
5%   Tool calling           — Function calling examples
2%   Identity               — Nexus + creator identity QA
```

### Checkpoints

- Saved every 500 steps to Modal Volume `nexus-v1-ckpts`
- Keeps last 5 checkpoints (older ones auto-deleted)
- Self-contained: architecture config stored inside `.pth` file
- Resume: automatic — detects latest checkpoint and continues

### Monitoring

Every 500 steps:
- Inference probe across 9 domains (identity, creator, math, physics, chemistry, code, English, tool_call, agentic)
- Memory inspection (live state from forward pass)
- HyperConnection mixing rates
- Write router statistics
- Read gate activation rates

Every 5,000 steps:
- Activation health check (logits std monitoring)

---

## Inference & Chat

### Interactive Chat

```bash
python chat_with_model.py -c ckpt_nexus_076500.pth
```

### Validation Suite

```bash
python chat_with_model.py -c ckpt_nexus_076500.pth --validate
```

### Benchmark

```bash
python chat_with_model.py -c ckpt_nexus_076500.pth --benchmark
```

### Single Domain Test

```bash
python chat_with_model.py -c ckpt_nexus_076500.pth --test math
python chat_with_model.py -c ckpt_nexus_076500.pth --test code
python chat_with_model.py -c ckpt_nexus_076500.pth --test tool_call
```

### Memory Probe

```bash
python chat_with_model.py -c ckpt_nexus_076500.pth --memory
```

Tests cross-chunk context retention — verifies the memory system actually stores and retrieves information across long sequences.

### Programmatic Usage

```python
import torch
from Model import Nexus
from kv_cache import DynamicKVCache, _sample
import tiktoken

# Load checkpoint
ckpt = torch.load("ckpt_nexus_076500.pth", map_location="cuda")
model = Nexus(vocab_size=100277, dim=1280, heads=16, kv_heads=4,
              num_layers=20, memory_slots=128, use_flash=True).cuda()
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

enc = tiktoken.get_encoding("cl100k_base")

# Generate
prompt = "User: What is your architecture?\nAssistant:"
tokens = enc.encode(prompt, allowed_special={""})
x = torch.tensor(tokens).unsqueeze(0).cuda()

cache = DynamicKVCache(model, max_seq_len=2048, batch_size=1)
with torch.no_grad():
    logits, _, _, mem = cache.prefill(x, enc=enc)

# Generate tokens
generated = []
for _ in range(80):
    next_token = _sample(logits[0, -1], generated, temperature=0.8,
                        rep_penalty=1.5, top_k=50, top_p=1.0)
    if next_token == enc.eot_token:
        break
    generated.append(next_token)
    token_t = torch.tensor([[next_token]], device="cuda")
    with torch.no_grad():
        logits, _, _, mem = cache.decode_one(token_t)

print(enc.decode(generated))
```

---

## File Structure

```
phase-3/
├── README.md                  # This file
├── Model.py                   # Nexus model definition
├── v3_modules.py              # YaRN RoPE, HyperConnections, MTP Head, RMSNorm
├── kv_cache.py                # DynamicKVCache for fast autoregressive inference
├── Train.py                   # Pretraining script (Modal H100)
├── Train_sft.py               # Supervised fine-tuning script
├── chat_with_model.py         # Interactive chat + validation + benchmarks
├── analyze_model.py           # Activation analysis & visualization
├── build_sft_dataset.py       # SFT dataset builder
├── prepare_sft_dataset.py     # SFT dataset preparation
├── run_modal_sft.py           # SFT runner (Modal)
├── check_routing.py           # Memory routing diagnostics
├── ckpt_nexus_076500.pth      # Latest checkpoint (76.5K steps)
├── nexus_sft_stage1_step004743.pth  # SFT checkpoint
├── sft_dataset.jsonl          # SFT training data
└── analysis/                  # Activation analysis outputs
    ├── 76.5k/
    │   ├── summary.json       # Activation statistics
    │   └── *.png              # Visualization plots
    └── ...
```

---

## Checkpoint Format

Checkpoints are self-contained `.pth` files with:

```python
{
    "step": 76500,                        # Training step
    "loss": 3.0167,                       # Current loss
    "ema_loss": 3.0167,                   # EMA smoothed loss
    "model_state_dict": {...},            # Model weights
    "optimizer_state_dict": {...},        # AdamW optimizer state (momentum + variance)
    "cfg": {...},                         # Full training config (for validation)
    "torch_version": "2.3.1",           # PyTorch version
    "stream_skip_counts": {...},           # Dataset position counters
    "scaler_state_dict": {...},           # Grad scaler (if used)
}
```

To load:

```python
ckpt = torch.load("ckpt_nexus_076500.pth", map_location="cuda")
model.load_state_dict(ckpt["model_state_dict"])
optimizer.load_state_dict(ckpt["optimizer_state_dict"])
resume_step = ckpt["step"] + 1
```

---

## Hyperparameters

### Model Architecture

| Parameter | Value | Notes |
|-----------|-------|-------|
| vocab_size | 100,277 | cl100k_base tokenizer |
| dim | 1280 | Hidden dimension |
| heads | 16 | Query attention heads |
| kv_heads | 4 | KV heads (GQA ratio 4:1) |
| num_layers | 20 | Transformer blocks |
| head_dim | 80 | dim / heads |
| ffn_hidden | 3,417 | 2.67 x dim |
| memory_bridges | [3, 10, 17] | After which layers |
| memory_slots | [32, 64, 128] | Per bridge |
| mtp_depths | 1 | Multi-token prediction depth |
| rope_base | 10,000 | RoPE frequency base |
| yarn_scale | 8.0 | Context extension factor |

### Memory Configuration

| Bridge | Slots | Retain Floor | Write Scale | Role |
|--------|-------|--------------|-------------|------|
| 0 (lexical) | 32 | 0.3 | 0.5 | Fast retention, aggressive writes |
| 1 (semantic) | 64 | 0.5 | 0.3 | Balanced retention and writes |
| 2 (reasoning) | 128 | 0.7 | 0.2 | Long retention, conservative writes |

### Training

| Parameter | Phase 1 | Phase 3 (Current) |
|-----------|---------|-------------------|
| max_steps | 76,500 | 153,000 |
| batch_size/GPU | 32 (4 seqs) | 32 (4 seqs) |
| grad_accum | 2 | 2 |
| global_batch | 64 seqs | 64 seqs |
| tokens/step | 131,072 | 131,072 |
| total_tokens | ~10B | ~20B |
| learning_rate | 3e-4 | 1.5e-4 |
| min_lr | 3e-5 | 1.5e-5 |
| warmup_steps | 0 (250 resume) | 500 |
| weight_decay | 0.1 | 0.1 |
| grad_clip | 2.0 | 2.0 |
| hyper_beta_weight | — | 0.001 |
| mem_diversity_weight | 0.01 | 0.05 |
| mem_temporal_weight | 0.1 | 0.3 |

### Loss Components

```
total = loss_lm
      + mtp_loss_weight * mtp_loss
      + mem_diversity_weight * mem_div_loss
      + mem_usage_weight * mem_use_loss
      + mem_prediction_weight * mem_pred_loss
      + mem_temporal_weight * temporal_loss
      + moe_lb_weight * aux_loss
      + hyper_beta_weight * beta_l2_reg
```

---

## License

Copyright (c) 2026 Siddi Vinayaka. All rights reserved.
