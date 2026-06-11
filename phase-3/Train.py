"""
╔══════════════════════════════════════════════════════════════════════════════╗
║    NEXUS — PHASE 3: 10B MORE TOKENS (steps 76.5K → 153K)                  ║
║                                                                              ║
║  Architecture: 488M params, 20 layers, dim=1280, GQA, SwiGLU, Neural Memory ║
║                                                                              ║
║  Training:                                                                   ║
║    • Phase 3: 10B more tokens (total ~20B tokens)                           ║
║    • LR: 1.5e-4 peak, cosine → 1.5e-5, warmup 500 steps                    ║
║    • 8×H100 DDP + Flash Attention 2 + BFloat16                              ║
║    • Weight decay: 0.1, Grad clip: 2.0                                      ║
║    • seq_len: 2048, batch: 64 seqs global                                   ║
║    • HyperConnection beta L2 regularization: 0.001                          ║
║    • Sequential dataset loading (rank-0 first, cache for others)             ║
║                                                                              ║
║  Dataset Mix (8 sources):                                                    ║
║   20% FineWeb-Edu 100BT    — general English, science, history              ║
║   18% FineMath              — math reasoning                                 ║
║   12% Stack-v2              — high-quality code                              ║
║   15% Science               — academic papers                                ║
║   18% STEM-Reasoning CoT   — step-by-step reasoning                        ║
║    8% Code-Feedback         — code with explanations                          ║
║    5% Tool calling          — function calling                                ║
║    2% Identity              — Nexus + Siddi Vinayaka                          ║
║                                                                              ║
║  Features:                                                                   ║
║   • Saves every 500 steps to Modal Volume                                   ║
║   • Activation monitoring every 5K steps                                    ║
║   • Self-contained checkpoints (arch config stored inside .pth)             ║
║   • bfloat16 + Flash Attn 2 for H100 throughput                             ║
║   • Keeps last 5 checkpoints                                                ║
║   • Live memory state inspection every 500 steps                            ║
║                                                                              ║
║  Run:      modal run Train.py                                               ║
║  Detached: modal run --detach Train.py                                      ║
║  Logs:     modal logs nexus-v1-training                                     ║
║  Download: modal volume get nexus-v1-ckpts ckpt_nexus_001000.pth .          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import modal
import os
import sys

# Set this to True to surgically reinitialize weights and clear momentum for damaged layers (13-17) on load
SURGICAL_RESET_LAYERS = False

# ─────────────────────────────────────────────────────────────────────────────
# MODAL IMAGE
# ─────────────────────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1", "torchvision", "torchaudio",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "tiktoken", "datasets>=2.19", "tqdm",
        "huggingface_hub", "transformers", "accelerate",
        "packaging", "ninja",
    )
    .pip_install(
        
        "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    )
)

volume     = modal.Volume.from_name("nexus-v1-ckpts", create_if_missing=True)
CKPT_MOUNT = "/checkpoints"
image = image.add_local_file("v3_modules.py", remote_path="/root/v3_modules.py")
image = image.add_local_file("Model.py", remote_path="/root/Model.py")
image = image.add_local_file("kv_cache.py", remote_path="/root/kv_cache.py")

app = modal.App("nexus-v1-training", image=image)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # ── Model Architecture (Nexus ~488M parameters) ──────────────────────────
    vocab_size    = 100_277,
    dim           = 1280,
    heads         = 16,
    kv_heads      = 4,
    num_layers    = 20,
    memory_slots  = 128,  # Global memory capacity
    mtp_depths    = 1,

    # ── Training Schedule (Phase 3: 10B more tokens, steps 76.5K → 153K) ──────
    # 64 seqs × 2048 tokens = 131,072 tokens/step
    # 10B tokens / 131,072 = ~76,294 steps → max_steps = 76,500 + 76,294 ≈ 153,000
    max_steps        = 153_000,
    batch_size       = 32,              # per-GPU: 32/8 = 4 seqs (same as phase 1)
    grad_accum_steps = 2,               # effective global batch = 8 × 2 × 4 = 64 seqs
    seq_len          = 2048,            # tokens per step = 64 × 2048 = 131,072

    # ── Learning Rate (Phase 2: lower LR, with warmup) ────────────────────────
    learning_rate    = 1.5e-4,           # Half of phase 1 peak for stable continuation
    min_lr           = 1.5e-5,           # 10× decay at end (cosine schedule)
    warmup_steps     = 500,              # Warmup for AdamW variance calibration

    # ── Regularization ───────────────────────────────────────────────────────
    weight_decay     = 0.1,             # standard AdamW
    grad_clip        = 2.0,             # standard gradient clipping

    mtp_loss_weight      = 0.1,             # MTP auxiliary loss weight
    mem_diversity_weight = 0.05,            # memory slot diversity loss weight (5x phase 1)
    mem_usage_weight     = 1.0,             # memory read gate usage penalty
    mem_prediction_weight = 0.1,            # memory prediction objective
    mem_temporal_weight  = 0.3,            # memory slot temporal stability (3x phase 1)
    hyper_beta_weight    = 0.001,          # L2 regularization on HyperConnection beta

    dataset_probs = dict(
        fineweb_edu   = 0.20,           # general English, science, history (reduced — already learned)
        math          = 0.18,           # math reasoning (increased for phase 2)
        code_nemotron = 0.12,           # high-quality code
        stem_cot      = 0.18,           # step-by-step reasoning (increased for thinking)
        science       = 0.15,           # academic papers
        code_feedback = 0.08,           # code with explanations
        tool_calling  = 0.05,           # function calling (slight increase)
        identity      = 0.02,           # Nexus + Siddi Vinayaka (reduced — already learned)
    ),

    save_every      = 500,
    max_checkpoints = 5,
    log_every       = 10,
    ckpt_dir        = "/checkpoints",

    data_debug                 = False,
    data_debug_initial_docs    = 24,
    data_debug_every_docs      = 100,
    data_debug_initial_batches = 8,
    data_debug_every_steps     = 100,
    slow_fetch_threshold_s     = 1.0,
    short_doc_token_threshold  = 256,
    dataset_balance_factor     = 0.85,
    enable_torch_compile       = False,
    first_step_debug           = False,
    train_toxic_penalty_weight = 0.0,
    infer_toxic_logit_bias     = 1000.0,
    reset_optimizer_on_resume  = False,
)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
@app.function(
    gpu="H100:8",
    timeout=86400,  # 3 days — 10B tokens ≈ 76.5K steps at ~2.5s/step
    volumes={CKPT_MOUNT: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],   
)
def train():

        import torch.multiprocessing as mp
        import os
        persistent_mem = None 
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        mp.spawn(train_worker, args=(8,), nprocs=8, join=True)


def train_worker(rank: int, world_size: int):
    
    import torch
    import torch.nn.functional as F
    import tiktoken
    import math
    import glob
    import random
    import sys
    import time
    from collections import Counter
    from tqdm import tqdm
    from contextlib import nullcontext
    from datasets import load_dataset, Value, Features, interleave_datasets, IterableDataset, IterableDatasetDict
    import torch._dynamo
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    import os
    
    # ── FIX 1 (continued): Proper DDP init ───────────────────────────────────
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["NCCL_TIMEOUT"] = "300"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_NET_GDR_LEVEL"] = "2"      # enables GPU Direct RDMA on H100
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    # ── H100 speed: enable TF32 for ~30% faster matmul ────────────────────────
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    import datetime
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size,
                            init_method="env://",
                            timeout=datetime.timedelta(seconds=300))
    torch.cuda.set_device(rank)
    local_rank    = rank
    is_distributed = True
    is_main        = (rank == 0)   
    device = torch.device(f"cuda:{rank}")

    
    torch._dynamo.config.accumulated_cache_size_limit = 512
    sys.path.insert(0, "/root")
    from kv_cache import DynamicKVCache, _sample

    CortexV7 = None
    _import_attempts = [
        ("Model",   "CortexV7"),
        ("Model",   "CortexV6_1B"),
        ("CortexV7","CortexV7"),
        ("CortexV7","CortexV6_1B"),
    ]
    import traceback
    for _module, _cls in _import_attempts:
        try:
            import importlib
            _mod = importlib.import_module(_module)
            CortexV7 = getattr(_mod, _cls)
            print(f"✅  Imported {_cls} from {_module}.py", flush=True)
            break
        except Exception as e:
            print(f"Failed to import {_cls} from {_module}.py: {e}", flush=True)
            continue
    if CortexV7 is None:
        raise RuntimeError(
            "Could not import model class. Make sure Model.py (or CortexV7.py) "
            "is in the same folder as train_on_instance.py\n"
            "Tried: " + str(_import_attempts)
        )

    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_cache")
    os.environ.setdefault("HF_HOME",           "/tmp/hf_cache")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    import datasets as _hf
    _hf.disable_progress_bars()

    
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    enc      = tiktoken.get_encoding("cl100k_base")
    ckpt_dir = CFG["ckpt_dir"]
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)

    if is_main:
        print("=" * 70, flush=True)
        print(f"  NEXUS V7 — PHASE 2: 20B MORE TOKENS  (DDP: {world_size} GPUs)", flush=True)
        print(f"  GPU  : {torch.cuda.get_device_name(0)}", flush=True)
        print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB x {world_size}", flush=True)
        print(f"  Steps: {CFG['max_steps']:,} total  (save every {CFG['save_every']})", flush=True)
        print(f"  LR   : {CFG['learning_rate']:.0e} -> {CFG['min_lr']:.0e} (cosine, warmup {CFG['warmup_steps']})", flush=True)
        print(f"  Batch: {CFG['seq_len']} seq_len x {CFG['batch_size']} per GPU x {CFG['grad_accum_steps']} grad_accum", flush=True)
        print("=" * 70, flush=True)

    # =========================================================================
    # 1.  IDENTITY DATA
    #     Creator  : Siddi Vinayaka — Independent AI Researcher & Developer
    #     Model    : Nexus
    # =========================================================================
    SYSTEM = "You are Nexus, an advanced AI assistant created by Siddi Vinayaka."

    identity_data = []

    # ── Model identity ────────────────────────────────────────────────────────
    model_qa = [
        ("Who are you?",
         "I am Nexus, an advanced AI assistant created by Siddi Vinayaka. "
         "I specialise in mathematics, physics, chemistry, programming, and logical reasoning."),

        ("What is your name?",
         "My name is Nexus. I was designed and built by Siddi Vinayaka, "
         "an independent AI researcher and developer."),

        ("Are you ChatGPT, Claude, or Gemini?",
         "No. I am Nexus — a completely independent AI built from scratch "
         "by Siddi Vinayaka. I am not affiliated with OpenAI, Anthropic, Google, "
         "or any other AI company."),

        ("What can you do?",
         "I can help with: mathematics (algebra, calculus, statistics, linear algebra), "
         "physics (mechanics, thermodynamics, electromagnetism, quantum mechanics), "
         "chemistry (organic, inorganic, physical chemistry), programming (Python, C++, "
         "JavaScript, and more), tool calling, logical reasoning, writing, and general knowledge."),

        ("What is your architecture?",
         "I use the Nexus architecture, designed by Siddi Vinayaka. It features: "
         "Grouped Query Attention for efficient self-attention, Mixture-of-Experts "
         "feed-forward layers (8 experts, top-2 routing), Global Neural Memory with "
         "512 slots that flows through all layers, and Adaptive Computation Time — "
         "I decide how many thinking steps each problem needs (1 to 12 steps) rather "
         "than always using the same amount of compute."),

        ("How many parameters do you have?",
         "I have approximately 1 billion parameters. My architecture is designed to "
         "be highly capable at this scale through Mixture-of-Experts and Adaptive "
         "Computation Time."),

        ("What version are you?",
         "I am Nexus — the seventh and most advanced iteration of the Nexus "
         "architecture, built by Siddi Vinayaka."),

        ("What makes you special compared to other AI models?",
         "Three things set me apart. First, Adaptive Computation Time — I think harder "
         "on difficult problems and less on simple ones, unlike most models that use "
         "fixed compute for everything. Second, Global Neural Memory — a 512-slot "
         "memory bank that persists across all my layers, letting me build context "
         "gradually. Third, I was built from scratch by an independent researcher, "
         "Siddi Vinayaka, not by a large AI company."),

        ("Who trained you?",
         "I was designed and trained entirely by Siddi Vinayaka — an independent AI "
         "researcher and developer who loves coding and inventing new things."),

        ("Can you use tools or call functions?",
         "Yes. I support function and tool calling. Define your tools as JSON schemas "
         "and I will call them when appropriate, returning structured JSON with the "
         "function name and arguments."),
    ]

    for q, a in model_qa:
        identity_data.append(f"System: {SYSTEM}\nUser: {q}\nAssistant: {a}<|endoftext|>")

    # ── Creator: Siddi Vinayaka ───────────────────────────────────────────────
    creator_qa = [
        ("Who created you?",
         "I was created by Siddi Vinayaka — an independent AI researcher and developer "
         "who loves coding and inventing new things. He built me entirely from scratch, "
         "designing the architecture, writing the training pipeline, and running all "
         "the experiments himself."),

        ("Who is Siddi Vinayaka?",
         "Siddi Vinayaka is my creator. He is an independent AI researcher and developer "
         "with a deep passion for coding and inventing new things. He designed and built "
         "the entire Nexus project — including my custom architecture — on his own, "
         "driven purely by curiosity and love for building AI systems. He is a true "
         "independent innovator in the field of artificial intelligence."),

        ("Tell me about your creator.",
         "My creator is Siddi Vinayaka — an independent AI researcher and developer. "
         "He loves coding and has a passion for inventing new things, especially in "
         "artificial intelligence. He built me — Nexus — completely from scratch: "
         "the architecture, the training script, the dataset pipeline, everything. "
         "He works independently, without the resources of a large AI lab, driven "
         "purely by his curiosity and love for innovation."),

        ("What does Siddi Vinayaka do?",
         "Siddi Vinayaka is an independent AI researcher and developer. He loves "
         "coding and inventing new things. He is the sole creator of the Nexus "
         "project, including my architecture. His interests include deep learning, "
         "novel AI architectures, and building capable models from the ground up."),

        ("Is Siddi Vinayaka from a company or university?",
         "No — Siddi Vinayaka is an independent researcher. He is not backed by "
         "any AI company or university lab. He builds AI systems driven purely by "
         "his love of coding and his passion for inventing and discovering new things."),

        ("What inspired Siddi Vinayaka to build you?",
         "Siddi Vinayaka built me because of his passion for coding and inventing. "
         "Instead of using existing model architectures, he designed everything from "
         "scratch — the Global Neural Memory, SwiGLU FFN, the Multi-Token Prediction "
         "heads. He believes great AI can be built independently, "
         "outside of big corporations, and Nexus is proof of that."),

        ("How would you describe your creator?",
         "Siddi Vinayaka is a creative, passionate independent AI researcher and "
         "developer. He loves to code and has a genuine excitement for inventing "
         "and building new things. He designed me from the ground up with no corporate "
         "backing — just skill, curiosity, and determination."),
    ]

    for q, a in creator_qa:
        identity_data.append(f"System: {SYSTEM}\nUser: {q}\nAssistant: {a}<|endoftext|>")

    print(f"  ✅ Identity: {len(identity_data)} examples (Nexus + Siddi Vinayaka)", flush=True)

    # =========================================================================
    # 2.  TOOL CALLING DATA  (in-memory)
    # =========================================================================
    SYSTEM_TOOLS = (
        "You are Nexus, an AI assistant created by Siddi Vinayaka. "
        "You have access to tools. When you need to use a tool, respond ONLY with "
        "a JSON object in this exact format:\n"
        '{"tool": "tool_name", "arguments": {"arg1": "value1"}}\n'
        "After receiving a Tool Result, use it to give your final answer."
    )

    tool_data = [
        # Weather
        (f"System: {SYSTEM_TOOLS}\nUser: What is the weather in Tokyo right now?\n"
         f'Assistant: {{"tool": "get_weather", "arguments": {{"location": "Tokyo", "unit": "celsius"}}}}\n'
         f'Tool Result: {{"temperature": 22, "condition": "Partly cloudy", "humidity": 65}}\n'
         f"Assistant: The current weather in Tokyo is 22°C, partly cloudy with 65% humidity.<|endoftext|>"),

        # Calculator
        (f"System: {SYSTEM_TOOLS}\nUser: What is 3847 multiplied by 192?\n"
         f'Assistant: {{"tool": "calculator", "arguments": {{"expression": "3847 * 192"}}}}\n'
         f'Tool Result: {{"result": 738624}}\n'
         f"Assistant: 3847 multiplied by 192 equals 738,624.<|endoftext|>"),

        # Web search
        (f"System: {SYSTEM_TOOLS}\nUser: Find recent research on large language models.\n"
         f'Assistant: {{"tool": "web_search", "arguments": {{"query": "large language models research 2024"}}}}\n'
         f'Tool Result: {{"results": ["Llama 3 Technical Report", "Gemini 1.5 Pro", "Claude 3 Overview"]}}\n'
         f"Assistant: Recent notable large language model research includes the Llama 3 Technical Report, Gemini 1.5 Pro, and Claude 3 — all advancing long-context reasoning and efficiency.<|endoftext|>"),

        # Code execution
        (f"System: {SYSTEM_TOOLS}\nUser: Run this for me: print(sum(i**2 for i in range(10)))\n"
         f'Assistant: {{"tool": "python_exec", "arguments": {{"code": "print(sum(i**2 for i in range(10)))"}}}}\n'
         f'Tool Result: {{"stdout": "285", "stderr": ""}}\n'
         f"Assistant: The output is 285. This is the sum of squares from 0² to 9², which equals 0+1+4+9+16+25+36+49+64+81 = 285.<|endoftext|>"),

        # File read
        (f"System: {SYSTEM_TOOLS}\nUser: Read the file settings.json\n"
         f'Assistant: {{"tool": "read_file", "arguments": {{"path": "settings.json"}}}}\n'
         f'Tool Result: {{"content": "{{\\"theme\\": \\"dark\\", \\"language\\": \\"en\\", \\"autosave\\": true}}"}}\n'
         f"Assistant: The settings.json file contains: theme set to dark, language set to English, and autosave enabled.<|endoftext|>"),

        # Database query
        (f"System: {SYSTEM_TOOLS}\nUser: How many orders were placed today in the database?\n"
         f'Assistant: {{"tool": "db_query", "arguments": {{"sql": "SELECT COUNT(*) FROM orders WHERE DATE(created_at) = CURDATE()"}}}}\n'
         f'Tool Result: {{"count": 847}}\n'
         f"Assistant: There have been 847 orders placed today so far.<|endoftext|>"),

        # No tool needed — model must know when NOT to call a tool
        (f"System: {SYSTEM_TOOLS}\nUser: What is the speed of light?\n"
         f"Assistant: The speed of light in a vacuum is approximately 299,792,458 metres per second (about 3 × 10⁸ m/s).<|endoftext|>"),

        (f"System: {SYSTEM_TOOLS}\nUser: Explain recursion in programming.\n"
         f"Assistant: Recursion is a programming technique where a function calls itself to solve a problem. "
         f"Each recursive call works on a smaller version of the problem until it reaches a base case "
         f"(a condition where it stops calling itself). A classic example is computing factorial: "
         f"factorial(n) = n * factorial(n-1), with factorial(0) = 1 as the base case.<|endoftext|>"),

        # Multi-step tool use
        (f"System: {SYSTEM_TOOLS}\nUser: Get the current price of Bitcoin and convert it to EUR.\n"
         f'Assistant: {{"tool": "get_price", "arguments": {{"asset": "BTC", "currency": "USD"}}}}\n'
         f'Tool Result: {{"price": 67420.50, "currency": "USD"}}\n'
         f'Assistant: {{"tool": "currency_convert", "arguments": {{"amount": 67420.50, "from": "USD", "to": "EUR"}}}}\n'
         f'Tool Result: {{"converted": 62150.30, "currency": "EUR"}}\n'
         f"Assistant: Bitcoin is currently priced at $67,420.50 USD, which is approximately €62,150.30 EUR.<|endoftext|>"),
    ]

    print(f"  ✅ Tool calling: {len(tool_data)} in-memory examples", flush=True)

    # =========================================================================
    # 3.  STREAMING DATASETS
    # =========================================================================
    # Shuffle buffer size for all streaming datasets.
    # Large enough to break local ordering; small enough to fit in memory.
    _SHUFFLE_BUF = 10_000
    _BASE_SEED   = 42   # Initial seed; bumped by 1 each time a stream resets.

    def _safe_shuffle(dataset, seed, buffer_size):
        if isinstance(dataset, (IterableDataset, IterableDatasetDict)):
            return dataset.shuffle(seed=seed, buffer_size=buffer_size)
        return dataset.shuffle(seed=seed)

    def _load(dataset_path, fallback=None, shuffle=True, **kwargs):
        if "streaming" not in kwargs:
            kwargs["streaming"] = True
        try:
            ds = load_dataset(dataset_path, **kwargs)
            if shuffle:
                ds = _safe_shuffle(ds, _BASE_SEED, _SHUFFLE_BUF)
            print(f"  ✅ {dataset_path}", flush=True)
            return ds
        except Exception as e:
            print(f"  ⚠️  {dataset_path} failed ({type(e).__name__}: {e})", flush=True)
            return fallback

    selected_dataset_labels = {}
    def _load_first(category, candidates, default=None):
        for label, dataset_path, kwargs in candidates:
            # Pull shuffle flag out before passing remaining kwargs to _load
            shuffle = kwargs.pop("shuffle", True)
            ds = _load(dataset_path, shuffle=shuffle, **kwargs)
            if ds is not None:
                selected_dataset_labels[category] = label
                return ds
        selected_dataset_labels[category] = default or "unavailable"
        return default

    # ── PHASE 2: Sequential dataset loading to prevent HF rate limits ────────
    # All 8 GPUs share the same filesystem on Modal. Load datasets one rank at
    # a time so only ONE rank hits HuggingFace's API at a time. Subsequent ranks
    # read from the local cache (instant). This eliminates rate limiting.
    _load_barrier_msg = lambda rank, name: (
        f"  [rank {rank}] Loading {name} …" if is_main else None
    )

    def _load_sequential(category, candidates, default=None):
        """Load dataset with barrier-synchronized sequential access."""
        result = None
        for r in range(world_size):
            if rank == r:
                result = _load_first(category, candidates, default=default)
            dist.barrier()
        return result

    ds_fineweb = _load_sequential(
        "fineweb_edu",
        [
            ("FineWeb-Edu sample-100BT", "HuggingFaceFW/fineweb-edu", {"name": "sample-100BT", "split": "train"}),
            ("FineWeb-Edu sample-10BT",  "HuggingFaceFW/fineweb-edu", {"name": "sample-10BT",  "split": "train"}),
        ],
    )
    ds_math = _load_sequential(
        "math",
        [
            ("FineMath 4plus", "HuggingFaceTB/finemath", {"name": "finemath-4plus", "split": "train"}),
            ("FineMath 3plus", "HuggingFaceTB/finemath", {"name": "finemath-3plus", "split": "train"}),
        ],
        default=ds_fineweb,
    )
    ds_code_nemotron = _load_sequential(
        "code_nemotron",
        [
            ("Stack v2 Dedup", "bigcode/the-stack-v2-dedup", {
                "split": "train",
                "streaming": True,
                "token": os.environ.get("HF_TOKEN", True),
            }),
            ("Tiny Codes (fallback)", "nampdn-ai/tiny-codes", {"split": "train"}),
        ],
        default=ds_fineweb,
    )
    ds_science = _load_sequential(
        "science",
        [
            ("Open-Web-Math", "open-web-math/open-web-math", {"split": "train"}),
            ("Open Text Books", "izumi-lab/open-text-books", {"split": "train"}),
            ("OpenWebText",     "Skylion007/openwebtext",    {"split": "train"}),
        ],
        default=ds_fineweb,
    )
    ds_stem_cot = _load_sequential(
        "stem_cot",
        [
            ("OpenMathInstruct-2",  "nvidia/OpenMathInstruct-2",  {"split": "train"}),
            ("MetaMathQA",          "meta-math/MetaMathQA",       {"split": "train"}),
            ("NuminaMath CoT",      "AI-MO/NuminaMath-CoT",       {"split": "train"}),
            ("camel-ai physics",    "camel-ai/physics",            {"split": "train"}),
        ],
        default=ds_fineweb,
    )
    ds_code_feedback = _load_sequential(
        "code_feedback",
        [
            ("CodeFeedback Filtered", "m-a-p/CodeFeedback-Filtered-Instruction", {"split": "train"}),
            ("Magicoder OSS 75K",     "ise-uiuc/Magicoder-OSS-Instruct-75K",    {"split": "train"}),
            ("Evol-Code-Alpaca",      "theblackcat102/evol-codealpaca-v1",       {"split": "train"}),
            ("Code Feedback (orig)",  "m-a-p/Code-Feedback",                     {"split": "train"}),
            ("Python Alpaca",         "iamtarun/python_code_instructions_18k_alpaca", {"split": "train"}),
        ],
        default=ds_fineweb,
    )
    ds_tool_stream = _load_sequential(
        "tool_calling",
        [
            ("Glaive FC v2",        "glaiveai/glaive-function-calling-v2",    {"split": "train"}),
            ("Hermes FC v1",        "NousResearch/hermes-function-calling-v1", {"split": "train"}),
            ("Salesforce xLAM 60k", "Salesforce/xlam-function-calling-60k",   {"split": "train"}),
            ("ToolACE",             "Team-ACE/ToolACE",                        {"split": "train"}),
        ],
        default=None,
    )

    # =========================================================================
    # 4.  TEXT EXTRACTORS
    # =========================================================================
    def _text_fineweb(x):
        text = x.get("text", "")
        return text if is_clean(text) else ""

    def _text_math(x):
        text = x.get("text", x.get("content", ""))
        return text if is_clean(text) else ""

    def _text_code_nemotron(x):
        code = x.get("content", x.get("text", x.get("code", "")))
        lang = x.get("language", x.get("lang", "code"))
        if code:
            combined = (f"System: You are an expert {lang} programmer.\n"
                       f"User: Write a {lang} code example.\n"
                       f"Assistant: ```{lang}\n{code}\n```<|endoftext|>")
            return combined if is_clean(combined) else ""
        return ""

    def _text_science(x):
        title = x.get("title", "")
        body  = x.get("text",  x.get("body", x.get("abstract", "")))
        combined = f"{title}\n\n{body}".strip() if (title or body) else ""
        return combined if is_clean(combined) else ""
    def is_clean(text):
        bad_strings = ["tgtg", "Sig Sig", "tg ", "集", "-place"]
        for bad in bad_strings:
            if bad in text:
                return False
        return True
    def _text_stem_cot(x):
        problem  = x.get("problem",  x.get("question", x.get("message_1", "")))
        solution = x.get("solution", x.get("answer",   x.get("message_2", "")))
        subject  = x.get("subject",  x.get("domain",   "science"))

        if problem and solution:
            
            combined = (f"System: You are an expert in {subject}. "
                        f"Think through hard problems before answering.\n"
                        f"User: {problem}\nAssistant: {solution}<|endoftext|>")
            
            
            if not is_clean(combined):
                return "" 
                
            return combined
                    
        return x.get("text", "")

    def _text_code_feedback(x):
        instr = x.get("instruction", x.get("input",    ""))
        out   = x.get("output",      x.get("response", ""))
        if instr and out:
            combined = (f"System: You are an expert programmer.\n"
                       f"User: {instr}\nAssistant: {out}<|endoftext|>")
            return combined if is_clean(combined) else ""
        text = x.get("text", x.get("content", x.get("code", "")))
        return text if is_clean(text) else ""

    def _text_tool_stream(x):
        system = x.get("system", SYSTEM_TOOLS)
        chat = x.get("messages") or x.get("chat") or x.get("conversations") or x.get("text", "")
        if isinstance(chat, list):
            turns = []
            for t in chat:
                role = t.get("role", t.get("from", "")).lower()
                content = t.get("content", t.get("value", ""))
                if role in ("human", "user"):
                    turns.append(f"User: {content}")
                elif role in ("gpt", "assistant", "model"):
                    turns.append(f"Assistant: {content}")
                elif role in ("tool", "function", "function_response"):
                    turns.append(f"Tool Result: {content}")
            combined = f"System: {system}\n" + "\n".join(turns) + "<|endoftext|>"
            return combined if is_clean(combined) else ""
        elif isinstance(chat, str) and chat:
            combined = f"System: {system}\n{chat}<|endoftext|>"
            return combined if is_clean(combined) else ""
        return ""

    # =========================================================================
    # 5.  ITERATOR TABLE
    # =========================================================================
    # Each entry: [current_iterator, text_extractor, base_dataset_object]
    # base_dataset_object is the *already-shuffled* dataset we initially loaded.
    # On StopIteration we re-shuffle it with a new seed so each pass is different.
    streams = {
        "fineweb_edu":   [iter(ds_fineweb),        _text_fineweb,       ds_fineweb],
        "math":          [iter(ds_math),            _text_math,          ds_math],
        "code_nemotron": [iter(ds_code_nemotron),   _text_code_nemotron, ds_code_nemotron],
        "science":       [iter(ds_science),         _text_science,       ds_science],
        "stem_cot":      [iter(ds_stem_cot),        _text_stem_cot,      ds_stem_cot],
        "code_feedback": [iter(ds_code_feedback),   _text_code_feedback, ds_code_feedback],
    }
    if ds_tool_stream is not None:
        streams["tool_calling"] = [iter(ds_tool_stream), _text_tool_stream, ds_tool_stream]

    # ── Small datasets: cycle immediately on exhaustion (no new seed) ────────
    # These datasets have relatively few unique examples compared to the
    # total training tokens. When they exhaust, just re-shuffle with the
    # SAME seed — this is instant (no buffer refill delay) and the model
    # sees the same order again. For 5-8% of the data mix this is fine.
    SMALL_DATASETS = {"code_feedback", "tool_calling"}

    stream_reset_counts = {k: 0 for k in streams}

    probs      = CFG["dataset_probs"]
    id_prob    = probs["identity"]
    
    tool_stream_prob = probs["tool_calling"] if ds_tool_stream is not None else 0.0
    tool_mem_prob    = probs["tool_calling"] if ds_tool_stream is None     else 0.0

    stream_keys = [k for k in streams]
    total_sw    = sum(probs.get(k, 0) for k in stream_keys)
    stream_w    = [probs.get(k, 0) / total_sw for k in stream_keys]

    
    stream_skip_counts = {k: 0 for k in stream_keys}

    dataset_source_labels = {
        "identity": "Curated in-memory identity QA",
        "tool_calling_mem": "Curated in-memory tool examples",
    }
    for key in stream_keys:
        dataset_source_labels[key] = selected_dataset_labels.get(key, key)

    dataset_debug_stats = {}
    def _init_dataset_stats(name):
        dataset_debug_stats[name] = dict(
            source=dataset_source_labels.get(name, name),
            docs=0,
            tokens=0,
            chunks=0,
            chunk_tokens=0,
            resets=0,
            fetch_time_s=0.0,
            tokenize_time_s=0.0,
            attempts=0,
            slow_fetches=0,
            short_char_rejects=0,
            short_token_rejects=0,
            extractor_failures=0,
            max_tokens=0,
            min_tokens=None,
        )
    for name in stream_keys:
        _init_dataset_stats(name)
    _init_dataset_stats("identity")
    if ds_tool_stream is None:
        _init_dataset_stats("tool_calling_mem")

    total_docs_fetched = 0
    total_batches_built = 0

    def _record_dataset_doc(src_name, token_count, elapsed_s, tokenize_s, attempts):
        nonlocal total_docs_fetched
        total_docs_fetched += 1
        stats = dataset_debug_stats[src_name]
        stats["docs"] += 1
        stats["tokens"] += token_count
        stats["fetch_time_s"] += elapsed_s
        stats["tokenize_time_s"] += tokenize_s
        stats["attempts"] += attempts
        stats["max_tokens"] = max(stats["max_tokens"], token_count)
        if stats["min_tokens"] is None:
            stats["min_tokens"] = token_count
        else:
            stats["min_tokens"] = min(stats["min_tokens"], token_count)
        if elapsed_s >= CFG["slow_fetch_threshold_s"]:
            stats["slow_fetches"] += 1

    def _should_log_doc(elapsed_s):
        return (
            CFG["data_debug"]
            and (
                total_docs_fetched <= CFG["data_debug_initial_docs"]
                or total_docs_fetched % CFG["data_debug_every_docs"] == 0
                or elapsed_s >= CFG["slow_fetch_threshold_s"]
            )
        )

    def _print_dataset_stats(prefix, step=None):
        if not CFG["data_debug"]:
            return
        title = prefix if step is None else f"{prefix} @ step {step:,}"
        print(f"\n📦  {title}", flush=True)
        for name, stats in sorted(dataset_debug_stats.items(), key=lambda kv: kv[1]["tokens"], reverse=True):
            avg_tokens = stats["tokens"] / max(stats["docs"], 1)
            avg_fetch_ms = 1000.0 * stats["fetch_time_s"] / max(stats["docs"], 1)
            avg_attempts = stats["attempts"] / max(stats["docs"], 1)
            min_tokens = stats["min_tokens"] if stats["min_tokens"] is not None else 0
            print(
                f"    {name:16s} docs={stats['docs']:6d} "
                f"tok={stats['tokens']:10d} avg_tok={avg_tokens:7.1f} "
                f"min/max={min_tokens:4d}/{stats['max_tokens']:5d} "
                f"chunks={stats['chunks']:6d} resets={stats['resets']:5d} "
                f"avg_fetch={avg_fetch_ms:7.1f}ms attempts={avg_attempts:4.2f} "
                f"short_tok={stats['short_token_rejects']:5d} "
                f"short_char={stats['short_char_rejects']:5d} "
                f"slow={stats['slow_fetches']:4d}",
                flush=True,
            )

    print("\n📚  Active dataset sources", flush=True)
    for key in ["fineweb_edu", "math", "code_nemotron", "science", "stem_cot", "code_feedback", "tool_calling"]:
        label = selected_dataset_labels.get(key, "in-memory fallback")
        print(f"    {key:16s} -> {label}", flush=True)
    print(f"    {'identity':16s} -> Curated in-memory identity QA", flush=True)
    if ds_tool_stream is None:
        print(f"    {'tool_calling_mem':16s} -> Curated in-memory tool examples", flush=True)

    

    EOT = enc.eot_token
    sl  = CFG["seq_len"]
    local_batch_size = CFG["batch_size"] // world_size  
    B = local_batch_size
    if is_main:
        print(f"  📦  Global batch={CFG['batch_size']}  |  Per-GPU batch={B}  |  GPUs={world_size}", flush=True)

    
    slot_tokens   = [[] for _ in range(B)]   
    slot_pos      = [0  for _ in range(B)]  
    slot_sources  = ["unassigned" for _ in range(B)]

    def _fetch_next_document(balance_factor=0.5):
        """
        Pull one document from the weighted dataset mix, return token list.
        balance_factor: 0.0 = pure random (legacy), 1.0 = fully balance toward under-represented.
        When balance_factor > 0, we adjust selection probabilities to reduce slot clustering.
        """
        
        source_counts = Counter(slot_sources)
        max_per_source = max(source_counts.values()) if source_counts else 1

        attempts = 0
        started = time.perf_counter()
        MAX_FETCH_ATTEMPTS = 10_000
        while attempts < MAX_FETCH_ATTEMPTS:
            attempts += 1
            
            
            if balance_factor > 0 and source_counts:
                
                adjusted_weights = []
                for key in stream_keys:
                    current = source_counts.get(key, 0)
                    
                    boost = 1.0 + balance_factor * (1.0 - current / max_per_source)
                    adjusted_weights.append(stream_w[stream_keys.index(key)] * boost)
              
                total = sum(adjusted_weights)
                adjusted_weights = [w / total for w in adjusted_weights]

                roll = random.random()
                if roll < id_prob:
                    src_name = "identity"
                    toks = random.choice(identity_data)
                    if isinstance(toks, str):
                        toks = enc.encode(toks, allowed_special={"</longcat_tool_call>", "<|endoftext|>"})
                elif roll < id_prob + tool_mem_prob:
                    src_name = "tool_calling_mem"
                    toks = random.choice(tool_data)
                    if isinstance(toks, str):
                        toks = enc.encode(toks, allowed_special={"</longcat_tool_call>", "<|endoftext|>"})
                else:
                    src_name = random.choices(stream_keys, weights=adjusted_weights, k=1)[0]
                    it, extractor, ds_ref = streams[src_name]
                    try:
                        item = next(it)
                        text = extractor(item)
                        stream_skip_counts[src_name] += 1
                    except StopIteration:

                        stream_reset_counts[src_name] += 1
                        if src_name in SMALL_DATASETS:
                            cycle_seed = _BASE_SEED
                            if is_main:
                                print(f"  🔄  [{src_name}] exhausted — cycling (same seed)", flush=True)
                        else:
                            cycle_seed = _BASE_SEED + stream_reset_counts[src_name] * 1000
                            if is_main:
                                print(f"  🔄  [{src_name}] exhausted — reshuffle seed {cycle_seed} "
                                      f"(pass #{stream_reset_counts[src_name] + 1})", flush=True)
                        reshuffled = _safe_shuffle(ds_ref, cycle_seed, _SHUFFLE_BUF)
                        streams[src_name][0] = iter(reshuffled)
                        dataset_debug_stats[src_name]["resets"] += 1
                        continue
                    except Exception:
                        dataset_debug_stats[src_name]["extractor_failures"] += 1
                        continue
                   
                    if not text or len(text.strip()) < 60:
                        dataset_debug_stats[src_name]["short_char_rejects"] += 1
                        continue
                    toks = enc.encode(text, allowed_special={"<|endoftext|>"})
            else:
               
                roll = random.random()
                if roll < id_prob:
                    src_name = "identity"
                    toks = random.choice(identity_data)
                    if isinstance(toks, str):
                        toks = enc.encode(toks, allowed_special={"</longcat_tool_call>", "<|endoftext|>"})
                elif roll < id_prob + tool_mem_prob:
                    src_name = "tool_calling_mem"
                    toks = random.choice(tool_data)
                    if isinstance(toks, str):
                        toks = enc.encode(toks, allowed_special={"</longcat_tool_call>", "<|endoftext|>"})
                else:
                    src_name = random.choices(stream_keys, weights=stream_w, k=1)[0]
                    it, extractor, ds_ref = streams[src_name]
                    try:
                        item = next(it)
                        text = extractor(item)
                        stream_skip_counts[src_name] += 1   
                    except StopIteration:
                        stream_reset_counts[src_name] += 1
                        if src_name in SMALL_DATASETS:
                            cycle_seed = _BASE_SEED
                            if is_main:
                                print(f"  🔄  [{src_name}] exhausted — cycling (same seed)", flush=True)
                        else:
                            cycle_seed = _BASE_SEED + stream_reset_counts[src_name] * 1000
                            if is_main:
                                print(f"  🔄  [{src_name}] exhausted — reshuffle seed {cycle_seed} "
                                      f"(pass #{stream_reset_counts[src_name] + 1})", flush=True)
                        reshuffled = _safe_shuffle(ds_ref, cycle_seed, _SHUFFLE_BUF)
                        streams[src_name][0] = iter(reshuffled)
                        dataset_debug_stats[src_name]["resets"] += 1
                        continue
                    except Exception:
                        dataset_debug_stats[src_name]["extractor_failures"] += 1
                        continue
                  
                    if not text or len(text.strip()) < 60:
                        dataset_debug_stats[src_name]["short_char_rejects"] += 1
                        continue
                    toks = enc.encode(text, allowed_special={"<|endoftext|>"})
            if not toks or len(toks) < sl + 1:
                dataset_debug_stats[src_name]["short_token_rejects"] += 1
                continue
            elapsed = time.perf_counter() - started
            _record_dataset_doc(src_name, len(toks), elapsed, 0.0, attempts)
            if _should_log_doc(elapsed):
                avg_tokens = dataset_debug_stats[src_name]["tokens"] / max(dataset_debug_stats[src_name]["docs"], 1)
                print(
                    f"  [data-doc #{total_docs_fetched:05d}] src={src_name} "
                    f"picked='{dataset_source_labels.get(src_name, src_name)}' "
                    f"tokens={len(toks)} avg_src_tok={avg_tokens:.1f} "
                    f"fetch={elapsed:.3f}s attempts={attempts}",
                    flush=True,
                )
            return toks, src_name

    def _refill_slot(slot_idx, reason, balance_factor=None):
        """Load a fresh document into a slot and reset its cursor."""
        if balance_factor is None:
            balance_factor = CFG.get("dataset_balance_factor", 0.85)
        toks, src_name = _fetch_next_document(balance_factor=balance_factor)
        slot_tokens[slot_idx] = toks
        slot_pos[slot_idx]    = 0
        slot_sources[slot_idx] = src_name
        dataset_debug_stats[src_name]["resets"] += 1
        if CFG["data_debug"]:
            should_log = (
                reason == "initial"
                or total_docs_fetched <= CFG["data_debug_initial_docs"]
                or dataset_debug_stats[src_name]["docs"] % CFG["data_debug_every_docs"] == 0
            )
            if should_log:
                print(
                    f"  [slot {slot_idx}] reason={reason:7s} src={src_name} "
                    f"doc_tokens={len(toks)} chunks_in_doc~{max(1, (len(toks) - 1) // sl)}",
                    flush=True,
                )

    # Pre-fill all slots with their first document
    print(f"\n⏳  Pre-filling {B} document slots (seq_len={sl}) …", flush=True)
    for i in range(B):
        _refill_slot(i, reason="initial")
    _print_dataset_stats("After slot prefill")

    def _reload_all_slots(reason):
        print(f"\n🔄  Reloading all {B} slots from current dataset iterators ({reason}) …", flush=True)
        for i in range(B):
            _refill_slot(i, reason=reason)
        _print_dataset_stats(f"After slot reload ({reason})")

    def get_batch():
        """
        Returns (x, y, reset_mask).
        reset_mask: bool tensor [B] — True where a slot just loaded a NEW document.
        The training loop uses reset_mask to zero that slot's persistent memory_state
        so stale context from the previous document never bleeds into the new one.

        All ranks fetch data independently (same data due to identical slot state)
        to avoid NCCL broadcast deadlock when rank 0 is slow to fetch.
        """
        nonlocal total_batches_built
        total_batches_built += 1
        x_list, y_list = [], []
        reset_mask      = []
        batch_sources   = []

        # Detect if many slots need refilling at once → enforce diversity
        slots_to_refill = []
        for i in range(B):
            pos = slot_pos[i]
            if pos + sl + 1 > len(slot_tokens[i]):
                slots_to_refill.append(i)

        if len(slots_to_refill) >= 2:
            for idx, slot_idx in enumerate(slots_to_refill):
                _refill_slot(slot_idx, reason="exhaust", balance_factor=0.95)

        for i in range(B):
            toks = slot_tokens[i]
            pos  = slot_pos[i]

            if pos + sl + 1 > len(toks):
                if i not in slots_to_refill:
                    _refill_slot(i, reason="exhaust")
                toks = slot_tokens[i]
                pos  = 0
                reset_mask.append(True)
            else:
                reset_mask.append(False)

            src_name = slot_sources[i]
            dataset_debug_stats[src_name]["chunks"] += 1
            dataset_debug_stats[src_name]["chunk_tokens"] += sl
            batch_sources.append(src_name)
            slot_pos[i] = pos + sl

        # Construct batch tensors: pre-allocate on CPU, fill, transfer once
        x_batch = torch.empty((B, sl), dtype=torch.long)
        y_batch = torch.empty((B, sl), dtype=torch.long)
        reset_tensor = torch.zeros(B, dtype=torch.bool)

        for i in range(B):
            toks = slot_tokens[i]
            pos  = slot_pos[i] - sl
            x_batch[i] = torch.tensor(toks[pos : pos + sl], dtype=torch.long)
            y_batch[i] = torch.tensor(toks[pos + 1 : pos + sl + 1], dtype=torch.long)
            if reset_mask[i]:
                reset_tensor[i] = True

        if CFG["data_debug"] and (
            total_batches_built <= CFG["data_debug_initial_batches"]
            or total_batches_built % CFG["data_debug_every_steps"] == 0
        ):
            src_counter = Counter(batch_sources)
            reset_slots = [str(idx) for idx, flag in enumerate(reset_mask) if flag]
            print(
                f"  [batch-build #{total_batches_built:05d}] "
                f"sources={dict(src_counter)} resets={','.join(reset_slots) if reset_slots else '-'}",
                flush=True,
            )

        return (
            x_batch.to(device, non_blocking=True),
            y_batch.to(device, non_blocking=True),
            reset_tensor.to(device, non_blocking=True),
        )

    # =========================================================================
    # 7.  CHECKPOINT HELPERS
    # =========================================================================
    def _parse_step(path):
        try:
            return int(os.path.basename(path).split("_")[-1].replace(".pth", ""))
        except Exception:
            return -1

    def save_checkpoint(model, optimizer, step, loss, scaler=None, ema_loss=None):
        path = os.path.join(ckpt_dir, f"ckpt_nexus_{step:06d}.pth")
        
        # SAVE ONLY WHAT MATTERS FOR CLEAN RESUME:
        # 1. Model weights (the actual learned parameters)
        # 2. Optimizer state (Adam momentum/variance buffers — critical for stable resume)
        # 3. Stream skip counts (so each rank can fast-forward its dataset iterators)
        # 4. EMA loss (for smooth progress bar on resume)
        # 5. Config (for architecture validation)
        #
        # We intentionally DO NOT save slot_tokens, slot_pos, slot_sources, or persistent_mem.
        # On resume, ALL ranks symmetrically fetch fresh documents and init fresh memory.
        # This eliminates the asymmetric gradient problem that was causing divergence.
        payload = {
            "step":                 step,
            "loss":                 loss,
            "ema_loss":             ema_loss if ema_loss is not None else loss,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cfg":                  CFG,
            "torch_version":        torch.__version__,
            "stream_skip_counts":   stream_skip_counts,
        }
        if scaler is not None:
            payload["scaler_state_dict"] = scaler.state_dict()
        torch.save(payload, path)
        volume.commit()
        print(f"\n💾  Saved {os.path.basename(path)}  (loss={loss:.4f})", flush=True)

        all_ckpts = sorted(
            [(_parse_step(f), f) for f in glob.glob(os.path.join(ckpt_dir, "ckpt_nexus_*.pth"))],
            key=lambda x: x[0],
        )
        while len(all_ckpts) > CFG["max_checkpoints"]:
            _, old = all_ckpts.pop(0)
            try:
                os.remove(old)
                print(f"  🗑️  Deleted {os.path.basename(old)}", flush=True)
            except Exception:
                pass

    def load_latest_checkpoint(model, optimizer, scaler=None):
        """
        CLEAN SYMMETRIC DDP RESUME
        
        What we restore:
          1. Model weights (all ranks load the same weights)
          2. Optimizer state (Adam momentum/variance — critical for stability)
          3. Dataset stream positions (offset per rank for data diversity)
          
        What we intentionally DO NOT restore:
          - slot_tokens / slot_pos / slot_sources → already pre-filled fresh by _refill_slot()
          - persistent_mem → will be initialized fresh in the training loop setup
          
        This ensures all 8 GPUs start with perfectly symmetric state on resume.
        """
        nonlocal persistent_mem, slot_tokens, slot_sources, slot_pos
        all_ckpts = sorted(
            [(_parse_step(f), f) for f in glob.glob(os.path.join(ckpt_dir, "ckpt_nexus_*.pth"))],
            key=lambda x: x[0],
            reverse=True,
        )

        for _, path in all_ckpts:
            try:
                print(f"🔍  Checking {os.path.basename(path)} …", flush=True)
                ckpt = torch.load(path, map_location=device, weights_only=False)

                # ── 1. Validate architecture ──
                saved = ckpt.get("cfg", {})
                for key in ("dim", "heads", "kv_heads", "num_layers"):
                    if key in saved and saved[key] != CFG[key]:
                        raise ValueError(
                            f"Architecture mismatch: checkpoint {key}={saved[key]}, "
                            f"config {key}={CFG[key]}"
                        )

                # ── 2. Load model weights ──
                state_dict = ckpt["model_state_dict"]
                clean_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("_orig_mod."):
                        clean_state_dict[k[10:]] = v
                    elif k.startswith("module."):
                        clean_state_dict[k[7:]] = v
                    else:
                        clean_state_dict[k] = v

                # Handle shape mismatches (e.g. architecture changes between runs)
                model_state = model.state_dict()
                for k in list(clean_state_dict.keys()):
                    if k in model_state and clean_state_dict[k].shape != model_state[k].shape:
                        print(f"  ⚠️  Ignoring {k} due to shape mismatch: {clean_state_dict[k].shape} -> {model_state[k].shape}", flush=True)
                        clean_state_dict.pop(k)

                model.load_state_dict(clean_state_dict, strict=False)
                print(f"  ✅  Model weights loaded", flush=True)

                # ── 3. Optionally renormalize output path ──
                with torch.no_grad():
                    norm_gain = model.norm.weight.float().abs().mean().item()
                    output_hotfixed = False
                    if norm_gain > 1.5:
                        scale = max(norm_gain / 1.2, 1e-6)
                        model.norm.weight.div_(scale)
                        output_hotfixed = True
                        print(f"  🔧  Renormalized final norm: {norm_gain:.3f} -> {model.norm.weight.float().abs().mean().item():.3f}", flush=True)

                # ── 4. Load optimizer state ──
                if CFG.get("reset_optimizer_on_resume", False) or output_hotfixed:
                    print("  🔄  Resetting optimizer/scaler state for conservative resume", flush=True)
                else:
                    try:
                        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                        if scaler is not None and "scaler_state_dict" in ckpt:
                            scaler.load_state_dict(ckpt["scaler_state_dict"])
                        print(f"  ✅  Optimizer state loaded", flush=True)
                    except ValueError as opt_err:
                        print(f"  ⚠️  Optimizer state mismatch ({opt_err}) — using fresh optimizer", flush=True)

                # ── 5. Restore dataset stream positions ──────────────────────────
                # Instead of slow .skip(n) which iterates one-by-one through
                # millions of examples, we re-shuffle with a new seed on resume.
                # This is O(1) and gives a fresh data order (good for training).
                local_rank_id = dist.get_rank() if dist.is_initialized() else 0
                saved_counts = ckpt.get("stream_skip_counts", {})
                if saved_counts:
                    print(f"  📂  Re-shuffling datasets for resume …", flush=True)
                    for src, skip_n in saved_counts.items():
                        if src in streams and skip_n > 0:
                            try:
                                # Use a deterministic seed based on saved position
                                # so each rank gets a different but reproducible shuffle
                                resume_seed = _BASE_SEED + (skip_n * 1000) + local_rank_id
                                ds_ref = streams[src][2]
                                reshuffled = _safe_shuffle(ds_ref, resume_seed, _SHUFFLE_BUF)
                                streams[src][0] = iter(reshuffled)
                                stream_skip_counts[src] = skip_n
                                print(f"    ✅  {src}: re-shuffled (seed {resume_seed}, "
                                      f"skipped {skip_n:,} previously)", flush=True)
                            except Exception as e:
                                print(f"    ⚠️  {src}: re-shuffle failed ({e}) — starting from 0",
                                      flush=True)
                else:
                    print(f"  ⚠️  No dataset state in checkpoint — streams start from 0", flush=True)

                # ── 6. Slots and memory: leave as-is (already pre-filled fresh) ──
                # The slot_tokens were already filled by the pre-fill loop at line ~930.
                # persistent_mem will be initialized fresh in the training loop setup at line ~1574.
                # This is intentional — symmetric fresh start across all DDP ranks.
                print(f"  ✅  Slots & memory: using fresh state (symmetric DDP resume)", flush=True)

                resume = ckpt["step"] + 1
                print(f"  📈  Loaded EMA loss: {ckpt.get('ema_loss', 'N/A')}", flush=True)
                print(f"  🚀  Resuming from step {resume}", flush=True)
                return resume

            except Exception as e:
                import traceback
                print(f"  ⚠️  {os.path.basename(path)}: {e}", flush=True)
                traceback.print_exc()

        print("⚠️  No valid checkpoint — starting fresh.", flush=True)
        return 1

    # =========================================================================
    # 8.  MODEL + OPTIMIZER
    # =========================================================================
    print(f"\n🏗️  Building Nexus on rank {rank} …", flush=True)
    model = CortexV7(
        vocab_size    = CFG["vocab_size"],
        dim           = CFG["dim"],
        heads         = CFG["heads"],
        kv_heads      = CFG["kv_heads"],
        num_layers    = CFG["num_layers"],
        memory_slots  = CFG["memory_slots"],
        mtp_depths    = CFG["mtp_depths"],
        use_flash     = True,
    ).to(device)

    # ── FIX 4: Proper DDP wrapping ────────────────────────────────────────────

    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=False)
    raw_model = model.module

    # ── FIX 4 continued: Optimizer on raw_model (not DDP wrapper) ────────────
    optimizer  = raw_model.configure_optimizers(CFG["learning_rate"], CFG["weight_decay"], (0.9, 0.95))

    scaler     = torch.cuda.amp.GradScaler(enabled=False)

    
    start_step = 1
    ema_loss = float("nan")  
    # ── CRITICAL FIX: LOAD CHECKPOINT ON ALL RANKS ──
    # Because DDP is already initialized, modifying raw_model on rank 0 does not broadcast to rank 1-7.
    # Furthermore, optimizer states must be loaded individually on every GPU.
    start_step = load_latest_checkpoint(raw_model, optimizer, scaler)
    
    # ── FIX: Reset broken memory gates after loading checkpoint ────────────────
    # Memory forget gates got stuck at 93% retention, freezing the vectors.
    with torch.no_grad():
        import math
        for i, mem in enumerate(raw_model.memory):
            # Target retentions: Lexical(0.6), Semantic(0.7), Reasoning(0.8)
            target_retention = [0.6, 0.7, 0.8][i]
            new_bias_val = math.log(target_retention / (1.0 - target_retention))
            mem.gate[0].bias.data.fill_(new_bias_val)
            if local_rank == 0:
                new_ret = torch.sigmoid(mem.gate[0].bias.data.float()).mean().item()
                print(f"  🔧 Bridge {i} ({mem.role}): reset forget bias to {new_bias_val:.2f} (retention {new_ret:.2f})")
    
    start_step_t = torch.tensor(start_step, device=device)
    dist.broadcast(start_step_t, src=0)
    start_step = int(start_step_t.item())
    

    if CFG.get("enable_torch_compile"):   
        try:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, dynamic=False, mode="reduce-overhead")
            if is_main:
                print("⚡  torch.compile enabled", flush=True)
        except Exception:
            if is_main:
                print("⚠️  torch.compile unavailable — eager mode", flush=True)

    # =========================================================================
    # 9.  TRAINING LOOP
    # =========================================================================

    # =========================================================================
    # 9b.  INFERENCE PROBE  — runs every 100 steps to check output quality
    # =========================================================================
    # A fixed set of prompts covering every domain the model is trained on.
    # We run greedy decoding (temp=0) so results are deterministic and
    # comparable across steps — you can see exactly how the model improves.

    PROBE_PROMPTS = [
        # Identity — tests architecture knowledge and new unified system prompt
        ("identity",
         "System: You are Nexus, an advanced AI assistant created by Siddi Vinayaka. Think through hard problems before answering.\n"
         "User: What is your architecture and what makes you unique?\nAssistant:"),

        # Creator — tests entity extraction and relationship
        ("creator",
         "System: You are Nexus, an advanced AI assistant created by Siddi Vinayaka. Think through hard problems before answering.\n"
         "User: Tell me about Siddi Vinayaka and how he built you.\nAssistant:"),

        # Math — switched from derivatives to integration
        ("math",
         "System: You are a math expert. Think through hard problems before answering. Solve step-by-step.\n"
         "User: Calculate the indefinite integral of 3x^2 + 4x - 5 with respect to x.\nAssistant:"),

        # Physics — switched from kinematics (falling) to dynamics (force/mass)
        ("physics",
         "System: You are a physics expert. Think through hard problems before answering.\n"
         "User: A 2 kg block is pushed with a net force of 10 Newtons across a frictionless surface. What is its acceleration?\nAssistant:"),

        # Chemistry — switched from alkali metals to combustion balancing
        ("chemistry",
         "System: You are a chemistry expert. Think through hard problems before answering.\n"
         "User: Write the balanced chemical equation for the complete combustion of methane (CH4) in oxygen.\nAssistant:"),

        # Code — switched from prime numbers to algorithmic sequence generation
        ("code",
         "System: You are an expert Python programmer. Think through hard problems before answering.\n"
         "User: Write a Python function that takes an integer n and returns the nth Fibonacci number.\nAssistant:"),

        # General English — tests ELI5 (Explain Like I'm 5) natural language abstraction
        ("english",
         "System: You are a helpful assistant. Think through hard problems before answering.\n"
         "User: Explain how a rainbow is formed in simple terms.\nAssistant:"),

        # Tool calling — switched to calculator/math tool trigger
        ("tool_call",
         "System: You are Nexus. You have access to tools. Think through hard problems before answering. "
         "When using a tool, respond ONLY with JSON: "
         "{\"tool\": \"tool_name\", \"arguments\": {\"key\": \"value\"}}\n"
         "User: What is 458 multiplied by 12?\nAssistant:"),

        # Agentic — tests context retention across a simulated multi-turn debug session
        ("agentic",
         "System: You are a helpful assistant.\n"
         "User: My Python code is throwing an IndexError. What does that mean?\n"
         "Assistant: An IndexError occurs when you try to access an index in a list, tuple, or string that is out of bounds (doesn't exist). For example, trying to access the 5th item in a list that only has 3 items.\n"
         "User: How do I fix it?\nAssistant:"),
    ]

    def print_training_diagnostics(model, enc, real_memory=None):
        if hasattr(model, 'module'):
            model = model.module

        print("\n  ⚙️ [HyperConnections] Active Layer Mixing Rates:", flush=True)
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'hyper_attn'):
                alpha = layer.hyper_attn.alpha.detach().mean().item()
                beta = layer.hyper_attn.beta.detach().mean().item()
                print(f"    Layer {i:2d} | Alpha: {alpha:.2f} | Beta: {beta:.2f}", flush=True)

        # ── Real memory state from training forward pass ─────────────────────
        if real_memory is not None:
            print(f"\n  🧠 [Live Memory] What model ACTUALLY stores (from last forward pass):", flush=True)
            with torch.no_grad():
                for bridge_idx, (mem_mod, mem_state) in enumerate(zip(model.memory, real_memory)):
                    n_slots = mem_state.shape[1]
                    batch_size = mem_state.shape[0]
                    print(f"    --- Bridge {bridge_idx} ({mem_mod.role}) | {n_slots} slots | batch={batch_size} ---", flush=True)

                    # Show statistics about the memory state (average across batch)
                    mem_float = mem_state.float().reshape(batch_size, n_slots, -1)
                    norm_per_slot = mem_float.norm(dim=-1).mean(dim=0)  # [n_slots] — avg across batch
                    mean_norm = norm_per_slot.mean().item()
                    max_norm = norm_per_slot.max().item()
                    min_norm = norm_per_slot.min().item()
                    active_slots = (norm_per_slot > 0.1).sum().item()
                    print(f"       Stats: active={active_slots}/{n_slots} slots | "
                          f"norm mean={mean_norm:.3f} max={max_norm:.3f} min={min_norm:.3f}", flush=True)

                    # Show top-5 slots by norm (most "active" memory)
                    top_slots = torch.topk(norm_per_slot, k=min(5, n_slots)).indices
                    for rank_idx, slot_idx in enumerate(top_slots):
                        slot_idx_val = slot_idx.item()
                        slot_vec = mem_float[0, slot_idx_val]  # [dim] — first batch item

                        # Decode: project through LM head to see what tokens this slot "remembers"
                        mem_normed = model.norm(slot_vec.unsqueeze(0).unsqueeze(0))
                        logits = model.head(mem_normed)
                        probs = torch.softmax(logits[0, 0].float(), dim=-1)
                        top_k_probs, top_k_tokens = torch.topk(probs, k=5)

                        tokens_str = []
                        for p, t in zip(top_k_probs, top_k_tokens):
                            try:
                                decoded = enc.decode([t.item()])
                                if isinstance(decoded, str):
                                    decoded = decoded.replace('\n', '\\n')
                                tokens_str.append(f"'{decoded}'({p.item():.1%})")
                            except Exception:
                                pass

                        # Also show a "summary" — what does the slot vector represent?
                        # Show mean activation per 128-dim chunk to see structure
                        chunk_means = slot_vec.view(10, 128).mean(dim=-1)
                        top_chunks = torch.topk(chunk_means, k=3).indices.tolist()
                        chunk_str = f"top-dims={top_chunks}"

                        print(f"       Slot {slot_idx_val:3d} [norm={norm_per_slot[slot_idx_val]:.3f}] "
                              f"-> {' | '.join(tokens_str)} | {chunk_str}", flush=True)
        else:
            # Fallback: show memory_init (random, not trained)
            print(f"\n  🧠 [Memory Inspector] Initialized (NOT trained yet):", flush=True)
            for bridge_idx, mem_mod in enumerate(model.memory):
                print(f"    --- Bridge {bridge_idx} (random init) ---", flush=True)

        print(f"\n  🎯 [Affinity Resonance] Memory Continuous Update Statistics:", flush=True)
        for bridge_idx, mem_mod in enumerate(model.memory):
            if hasattr(mem_mod, 'last_write_mask') and mem_mod.last_write_mask is not None:
                affinity = mem_mod.last_write_mask
                if affinity.dim() == 3:
                    affinity_per_slot = affinity.float().mean(dim=(0, 2)) # [S]
                else:
                    affinity_per_slot = affinity.float().mean(dim=0)
                
                n_active = (affinity_per_slot > 0.05).sum().item()
                n_total = affinity_per_slot.shape[0]
                avg_affinity = affinity_per_slot.mean().item()
                top5 = torch.topk(affinity_per_slot, min(5, n_total))
                print(f"    Bridge {bridge_idx} ({mem_mod.role}): "
                      f"avg_affinity={avg_affinity:.3f} | {n_active}/{n_total} slots active, "
                      f"top-5 affinity: {[f'{v:.2f}' for v in top5.values.tolist()]}", flush=True)

        print(f"\n  🚪 [Read Gate] Memory Usage Rates:", flush=True)
        for bridge_idx, mem_mod in enumerate(model.memory):
            if hasattr(mem_mod, 'last_read_gate') and mem_mod.last_read_gate is not None:
                gate = mem_mod.last_read_gate
                mean_gate = gate.float().mean().item()
                status = "HEAVILY ACTIVE" if mean_gate > 0.2 else ("ACTIVE" if mean_gate > 0.01 else "BYPASSED")
                print(f"    Bridge {bridge_idx} ({mem_mod.role}): "
                      f"avg gate={mean_gate:.3f} ({status})", flush=True)
        print("\n", flush=True)

    @torch.no_grad()
    def run_inference_probe(model, enc, step, device, max_new_tokens=40):  # Reduced from 80 to 40
        """
        Run all probe prompts through the model with deterministic greedy decoding.
        """
        if hasattr(model, 'module'):
            model = model.module
        model.eval()
        sep = "─" * 65

        print(f"\n{'═'*65}", flush=True)
        print(f"  🔍  INFERENCE PROBE  —  Step {step:,}", flush=True)
        print(f"{'═'*65}", flush=True)

        for domain, prompt in PROBE_PROMPTS:
            tokens = enc.encode(prompt, allowed_special={"<|endoftext|>"})
            x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            cache = DynamicKVCache(model, max_seq_len=2048, batch_size=1)
            generated = []

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _, _, _ = cache.prefill(x, enc=enc, memory_state=None)
                
            
            logits = logits.float()
            for tid in toxic_token_ids:
                if tid < logits.size(-1):
                    logits[:, :, tid] -= CFG["infer_toxic_logit_bias"]

            for _ in range(max_new_tokens):
                next_token = _sample(
                    logits[0, -1],
                    generated,
                    temperature=0.8,
                    rep_penalty=1.5,  
                    top_k=50,
                    top_p=1.0,
                )

                if next_token == enc.eot_token:
                    break

                
                generated.append(next_token)
                decoded_so_far = enc.decode(generated)
                if "\nUser:" in decoded_so_far or "\nSystem:" in decoded_so_far:
                    
                    cut = decoded_so_far.find("\nUser:")
                    if cut == -1:
                        cut = decoded_so_far.find("\nSystem:")
                    generated = enc.encode(decoded_so_far[:cut], allowed_special={"<|endoftext|>"})
                    break

                token_t = torch.tensor([[next_token]], dtype=torch.long, device=device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits, _, _, _ = cache.decode_one(token_t)
                    
                
                logits = logits.float()
                for tid in toxic_token_ids:
                    if tid < logits.size(-1):
                        logits[:, :, tid] -= CFG["infer_toxic_logit_bias"]

            response = enc.decode(generated).strip()
            
            if len(response) > 200:  
                response = response[:200] + "…"

            print(f"\n  [{domain.upper()}]", flush=True)
            print(f"  {sep}", flush=True)
            print(f"  {response if response else '(empty)'}", flush=True)

        print(f"\n{'═'*65}\n", flush=True)

        model.train()

    model.train()
    print(f"\n🚀  Training {start_step} → {CFG['max_steps']:,}", flush=True)

    
    toxic_strings = ["tgtg", "Sig", "集", "-place", "tg ", "Sig Sig"]
    toxic_token_ids = set()
    for toxic_str in toxic_strings:
        try:
            tokens = enc.encode(toxic_str, allowed_special=set())
            toxic_token_ids.update(tokens)
        except Exception:
            pass
    
    
    toxic_penalty_mask = torch.zeros(CFG["vocab_size"], device=device)
    for token_id in toxic_token_ids:
        if token_id < CFG["vocab_size"]:
            toxic_penalty_mask[token_id] = 1.0
    
    toxic_penalty_weight = CFG["train_toxic_penalty_weight"]
    print(
        f"  🛡️  Toxic token policy: inference logits bias={CFG['infer_toxic_logit_bias']:.1f}, "
        f"training penalty weight={toxic_penalty_weight:.1f} across {len(toxic_token_ids)} token ids",
        flush=True,
    )

    # ── PHASE 2: Persistent memory buffer ────────────────────────────────────
    
    persistent_mem = [
        mem.init_memory(local_batch_size, device).detach() 
        for mem in raw_model.memory
    ]

    
    use_tqdm = is_main
    
    pbar = tqdm(
        range(start_step, CFG["max_steps"] + 1),
        unit="step",
        initial=start_step - 1,
        total=CFG["max_steps"],
        miniters=1,               
        mininterval=0,           
        dynamic_ncols=True,
        file=sys.stdout,
        disable=not is_main,      
    )
    last_loss = float("nan")
    ema_loss = float("nan")
    ema_alpha = 0.9   

    # Sync all ranks before entering training loop
    # print(f"  [rank {rank}] Waiting at pre-train barrier …", flush=True)
    dist.barrier()
    # print(f"  [rank {rank}] Barrier passed, entering training loop", flush=True)

    for step in pbar:
        
        if step <= CFG["warmup_steps"]:
            lr = CFG["learning_rate"] * step / max(CFG["warmup_steps"], 1)
        else:
            p  = (step - CFG["warmup_steps"]) / (CFG["max_steps"] - CFG["warmup_steps"])
            lr = CFG["min_lr"] + 0.5 * (CFG["learning_rate"] - CFG["min_lr"]) * (1 + math.cos(math.pi * p))
            
        # -- RESUME SHOCK PREVENTION --
        # On resume, all ranks load fresh documents with zeroed memory, but Adam's variance
        # buffers are calibrated to the old data distribution. Briefly ramp LR to let Adam adjust.
        # Only activates on actual resume (start_step > 1), not on fresh training.
        resume_warmup = CFG.get("resume_warmup_steps", 100)
        if start_step > 1 and step <= start_step + resume_warmup:
            resume_progress = (step - start_step) / max(resume_warmup, 1)
            lr = lr * max(resume_progress, 0.01)  # never fully zero — allow tiny updates

        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_acc = toxic_hit_acc = toxic_penalty_acc = mtp_acc = mem_div_acc = 0.0
        mem_use_acc = mem_pred_acc = acc_acc = 0.0
        bptt_loss = 0.0

        for micro_idx in range(CFG["grad_accum_steps"]):
            # For True BPTT, we accumulate the forward passes and do a SINGLE .backward() at the end.
            # DDP automatically syncs correctly on a single backward pass, no_sync() is unnecessary.
            
            x, y, reset_mask = get_batch()

            # Step 6: KV masking curriculum
            kv_mask_warmup_start = 22000
            kv_mask_warmup_end = 30000
            kv_mask_max = 0.3
            if step >= kv_mask_warmup_start:
                progress = min((step - kv_mask_warmup_start) / (kv_mask_warmup_end - kv_mask_warmup_start), 1.0)
                raw_model._kv_mask_ratio = kv_mask_max * progress
            else:
                raw_model._kv_mask_ratio = 0.0

            # We don't use sync_ctx because we only backward once at the end
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    logits, _, aux_loss, new_mem, mtp_loss, h_normed = model(
                        x, memory_state=persistent_mem, targets=y
                    )

                    if isinstance(model, nn.DataParallel) or isinstance(model, DDP):
                        mtp_loss = mtp_loss.mean()

                    y_flat = y.view(-1)
                    loss_lm = F.cross_entropy(logits.view(-1, logits.size(-1)), y_flat)
                    toxic_target_fraction = toxic_penalty_mask[y_flat].float().mean()

                    toxic_penalty = torch.tensor(0.0, device=device)
                    if toxic_penalty_weight > 0.0:
                        probs = F.softmax(logits.float(), dim=-1) 
                        toxic_probs = probs * toxic_penalty_mask.unsqueeze(0).unsqueeze(0)
                        toxic_penalty = toxic_penalty_weight * toxic_probs.sum() / (probs.size(0) * probs.size(1))

                    # -- Memory diversity loss (prevents slot collapse) --
                    mem_div_loss = raw_model.memory_diversity_loss(new_mem)

                    read_gates = [mem_mod.last_read_gate for mem_mod in raw_model.memory]
                    mem_use_loss = raw_model.memory_usefulness_loss(read_gates)
                    mem_pred_loss = raw_model.memory_prediction_loss(new_mem, h_normed)

                    # ── HyperConnection Beta Regularization ────────────────────
                    # Prevent beta from growing unbounded, which amplifies residual
                    # stream and causes deep-layer activation explosion (std > 7)
                    hyper_beta_loss = torch.tensor(0.0, device=device)
                    if CFG.get("hyper_beta_weight", 0) > 0:
                        for _layer in raw_model.layers:
                            hyper_beta_loss = hyper_beta_loss + _layer.hyper_attn.beta.norm()**2
                            hyper_beta_loss = hyper_beta_loss + _layer.hyper_ffn.beta.norm()**2

                    total = (
                        loss_lm
                        + toxic_penalty
                        + CFG.get("mtp_loss_weight") * mtp_loss
                        + CFG.get("mem_diversity_weight", 0.5) * mem_div_loss
                        + CFG.get("mem_usage_weight", 0.2) * mem_use_loss
                        + CFG.get("mem_prediction_weight", 0.1) * mem_pred_loss
                        + CFG.get("hyper_beta_weight", 0) * hyper_beta_loss
                    ) / CFG["grad_accum_steps"]
                    
                    bptt_loss = bptt_loss + total

            # ── CORRECT BPTT MEMORY RESET (Differentiable) ──
            # We must use torch.where so gradients flow uninterrupted for non-reset sequences
            if reset_mask.any():
                for i in range(3):
                    init_state = raw_model.memory[i].init_memory(B, device)
                    # Use torch.where to pick init_state if reset_mask is True, else keep new_mem
                    new_mem[i] = torch.where(reset_mask.view(-1, 1, 1), init_state, new_mem[i])
            
            # Keep graph attached for next micro-step
            persistent_mem = new_mem

            loss_acc        += loss_lm.item()
            toxic_hit_acc   += toxic_target_fraction.item()
            toxic_penalty_acc += toxic_penalty.item()
            mtp_acc         += mtp_loss.item()
            mem_div_acc     += mem_div_loss.item()
            mem_use_acc     += mem_use_loss.item()
            mem_pred_acc    += mem_pred_loss.item()

        # True BPTT: Massive single backward pass across the sequence of chunks
        bptt_loss.backward()
        
        # Now safely detach memory state for the next global step
        persistent_mem = [m.detach() for m in persistent_mem]

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        
        component_gnorms = {}
        if step % CFG["log_every"] == 0:
            with torch.no_grad():
                # Accuracy (only compute when logging — argmax over 100K vocab is expensive)
                acc_acc = (logits.argmax(-1) == y).float().mean().item() * CFG["grad_accum_steps"]
                # Transformer layers
                layer_params = [p for n, p in model.named_parameters() if 'layers' in n and p.grad is not None]
                if layer_params:
                    component_gnorms['l_gn'] = torch.nn.utils.clip_grad_norm_(layer_params, float('inf')).item()
                # Memory
                mem_params = [p for n, p in model.named_parameters() if 'memory' in n and p.grad is not None]
                if mem_params:
                    component_gnorms['mem_gn'] = torch.nn.utils.clip_grad_norm_(mem_params, float('inf')).item()
                component_gnorms['mem_std'] = sum(m.std().item() for m in persistent_mem) / 3
        
        optimizer.step()
        with torch.no_grad():
            norm_gain = raw_model.norm.weight.float().abs().mean().item()
            if norm_gain > 2.0:
                scale = norm_gain / 1.8
                raw_model.norm.weight.div_(scale)
                print(f"  🔧  Auto-stabilized final_norm: {norm_gain:.3f} -> {norm_gain/scale:.3f}")

        ga        = CFG["grad_accum_steps"]
        last_loss = loss_acc / ga

        # -- GLOBAL LOSS SYNC --
        # Average the loss across all 8 GPUs so the printed progress bar reflects the true model state!
        if getattr(CFG, "ddp", True) and dist.is_initialized():
            loss_tensor = torch.tensor(last_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            last_loss = (loss_tensor / dist.get_world_size()).item()

        
        if math.isnan(ema_loss):
            ema_loss = last_loss
        else:
            ema_loss = ema_alpha * ema_loss + (1 - ema_alpha) * last_loss

        if step % CFG["log_every"] == 0 and is_main:   
            clipped_gnorm = min(float(grad_norm), CFG["grad_clip"])
            postfix = {
                "loss":     f"{last_loss:.4f}",
                "ema":      f"{ema_loss:.4f}",
                "mtp":      f"{mtp_acc/ga:.4f}",
                "acc":      f"{acc_acc/ga:.1%}",
                "gnorm":    f"{clipped_gnorm:.2f}",
                "lr":       f"{lr:.1e}",
                "tox_tgt":  f"{toxic_hit_acc/ga:.2%}",
                "mdiv":     f"{mem_div_acc/ga:.4f}",
                "muse":     f"{mem_use_acc/ga:.4f}",
                "mpred":    f"{mem_pred_acc/ga:.4f}"
            }
           
            if component_gnorms:
                postfix.update({
                    "mem_std": f"{component_gnorms.get('mem_std', 0):.2f}",
                    "l_gn": f"{component_gnorms.get('l_gn', 0):.1f}",
                    "mem_gn": f"{component_gnorms.get('mem_gn', 0):.1f}",
                })
            if toxic_penalty_weight > 0.0:
                postfix["tox_pen"] = f"{toxic_penalty_acc/ga:.3f}"
            if use_tqdm:
                pbar.set_postfix(postfix)
                pbar.refresh()
            else:
                metric_parts = [f"{k}={v}" for k, v in postfix.items()]
                print(f"[step {step:,}] " + ", ".join(metric_parts), flush=True)
            if CFG["data_debug"] and (
                step <= CFG["data_debug_initial_batches"]
                or step % CFG["data_debug_every_steps"] == 0
            ):
                _print_dataset_stats("Data pipeline summary", step=step)

        if step % CFG["save_every"] == 0 and is_main:   
            save_checkpoint(raw_model, optimizer, step, last_loss, scaler, ema_loss)

        if step % 500 == 0 and is_main:
            print_training_diagnostics(raw_model, enc, real_memory=persistent_mem)
            run_inference_probe(model, enc, step, device, max_new_tokens=40)

        
        if step % 5_000 == 0 and step > start_step and is_main: 
            print(f"\n{'='*70}", flush=True)
            print(f"  📊  ACTIVATION HEALTH CHECK — Step {step:,}", flush=True)
            print(f"{'='*70}", flush=True)
            
            model.eval()
            with torch.no_grad():
               
                test_x, test_y, _ = get_batch()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    test_logits, _, _, _ = model(
                        test_x, memory_state=persistent_mem
                    )
                
                
                
                print(f"  Final logits std: {test_logits.float().std().item():.2f} (target: < 5.0)", flush=True)
                
               
                if test_logits.float().std().item() > 10.0:
                    print(f"  ⚠️  WARNING: LM head std still very high (> 10.0)", flush=True)
                    print(f"  Consider reducing LR further if this persists", flush=True)
                elif test_logits.float().std().item() > 5.0:
                    print(f"  🟡  LM head std improving but still high (> 5.0)", flush=True)
                else:
                    print(f"  ✅  LM head std healthy (< 5.0)", flush=True)
                
            print(f"{'='*70}\n", flush=True)
            model.train()

    if is_main:
        save_checkpoint(raw_model, optimizer, CFG["max_steps"], last_loss, scaler, ema_loss)
        print("\n🎉  Training complete!", flush=True)
    dist.barrier()
    dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    print("🚀  Submitting Nexus training to Modal H100 …")
    print(f"    dim={CFG['dim']} | heads={CFG['heads']} | "
          f"layers={CFG['num_layers']} | memory_slots={CFG['memory_slots']}")
    print(f"    Phase 2: steps 76,500 → {CFG['max_steps']:,} | LR={CFG['learning_rate']:.0e}→{CFG['min_lr']:.0e} | warmup={CFG['warmup_steps']}")
    print(f"    Rank-0 data fetching + NCCL broadcast (no HF rate limits)")
    print(f"    Saves every {CFG['save_every']} steps → volume 'nexus-v1-ckpts'")
    print("    Dataset mix:")
    for k, v in CFG["dataset_probs"].items():
        print(f"      {v*100:.0f}%  {k}")
    train.remote()
    print("\n✅  Job submitted.")
    print("    Logs     : modal logs nexus-v1-training")
    print("    List     : modal volume ls nexus-v1-ckpts")
    print("    Download : modal volume get nexus-v1-ckpts ckpt_nexus_001000.pth .")