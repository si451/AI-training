# import modal
# import os
# import subprocess
# import glob

# # 1. Define the App
# app = modal.App("nexus-sft-phase3")

# # 2. Define the Image & Add Local Files
# image = (
#     modal.Image.debian_slim()
#     .pip_install(
#         "torch",
#         "tiktoken",
#         "tqdm",
#         "numpy",
#         "datasets"
#     )
#     .add_local_file("Train_sft.py", remote_path="/root/Train_sft.py")
#     .add_local_file("Model.py", remote_path="/root/Model.py")
# )

# # 3. Connect the Persistent Volume
# volume = modal.Volume.from_name("nexus-v1-ckpts", create_if_missing=True)

# # Save dir — SFT checkpoints go directly into /data (same volume root)
# SFT_SAVE_DIR = "/data"

# # 4. Define the GPU Function
# @app.function(
#     image=image,
#     gpu="H100",
#     timeout=86400,
#     volumes={"/data": volume},
#     secrets=[modal.Secret.from_name("huggingface-secret")]
# )
# def execute_sft(checkpoint_filename: str):
#     """
#     Smart SFT launcher:
#       1. Check for existing SFT checkpoints → resume from latest
#       2. If no SFT checkpoints → require pretrained checkpoint
#     """
#     print(f"🚀 Starting Modal Container for Nexus SFT")

#     # List all files in volume
#     volume_files = os.listdir("/data")
#     print(f"📂 Volume contents: {volume_files}")

#     # Check for existing SFT checkpoints
#     sft_ckpts = sorted(glob.glob(os.path.join(SFT_SAVE_DIR, "nexus_sft_stage*.pth")))

#     if sft_ckpts:
#         # ── SFT checkpoint found → resume (pretrained not needed) ─────────
#         print(f"✅ Found {len(sft_ckpts)} SFT checkpoint(s):")
#         for c in sft_ckpts:
#             print(f"   • {os.path.basename(c)}")
#         print(f"⏩ Will resume from latest SFT checkpoint (pretrained skipped)")

#         cmd = [
#             "python",
#             "/root/Train_sft.py",
#             "--save-dir", SFT_SAVE_DIR,
#         ]

#         # Pass pretrained checkpoint only if it exists (as fallback info)
#         ckpt_path = f"/data/{checkpoint_filename}"
#         if os.path.exists(ckpt_path):
#             cmd.extend(["--checkpoint", ckpt_path])
#         else:
#             # Use a dummy — Train_sft.py won't load it when SFT ckpts exist
#             cmd.extend(["--checkpoint", "none"])

#     else:
#         # ── No SFT checkpoint → pretrained is required ────────────────────
#         ckpt_path = f"/data/{checkpoint_filename}"
#         if not os.path.exists(ckpt_path):
#             print(f"❌ ERROR: No SFT checkpoints AND pretrained not found at {ckpt_path}")
#             print("Available files in volume:")
#             print(volume_files)
#             return

#         print(f"📂 No SFT checkpoints found — starting fresh from {checkpoint_filename}")
#         cmd = [
#             "python",
#             "/root/Train_sft.py",
#             "--checkpoint", ckpt_path,
#             "--save-dir", SFT_SAVE_DIR,
#         ]

#     print(f"Executing: {' '.join(cmd)}")
#     print("=" * 60)

#     process = subprocess.Popen(
#         cmd,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.STDOUT,
#         universal_newlines=True
#     )

#     for line in process.stdout:
#         print(line, end="")

#     process.wait()

#     if process.returncode != 0:
#         print(f"❌ SFT script failed with return code {process.returncode}")
#     else:
#         print(f"✅ SFT successfully completed on Modal.")

# # 5. Local Entrypoint
# @app.local_entrypoint()
# def main(checkpoint: str = "ckpt_v7_500000.pth"):
#     """
#     To run: modal run run_modal_sft.py
#     Detached: modal run --detach run_modal_sft.py
#     """
#     print(f"Submitting SFT job to Modal for checkpoint: {checkpoint}...")
#     execute_sft.remote(checkpoint)

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         NEXUS V7 — MODAL SFT RUNNER  (run_modal_sft.py)                    ║
║                                                                              ║
║  Fixes vs original:                                                         ║
║   • torch==2.3.1 pinned (matches pretraining)                              ║
║   • flash-attn 2.6.3 wheel installed (H100 required)                       ║
║   • huggingface_hub + transformers + accelerate added (HF streaming)        ║
║   • python_version="3.11" locked                                            ║
║   • volume.commit() called after every checkpoint save                      ║
║                                                                             ║
║  Run:     modal run run_modal_sft.py                                        ║
║  Detach:  modal run --detach run_modal_sft.py                               ║
║  Logs:    modal logs nexus-sft-phase3                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import modal
import os
import subprocess
import glob

app = modal.App("nexus-sft-phase3")

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE — identical stack to pretraining (torch 2.3.1, flash-attn 2.6.3)
# ─────────────────────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")

    # ── Core ML stack — PINNED to match pretraining ──────────────────────────
    .pip_install(
        "torch==2.3.1", "torchvision", "torchaudio",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )

    # ── HuggingFace ecosystem — ALL required for dataset streaming ────────────
    .pip_install(
        "tiktoken",
        "datasets>=2.19",
        "tqdm",
        "huggingface_hub",
        "transformers",
        "accelerate",
        "packaging",
        "ninja",
        "numpy",
    )

    # ── Flash Attention — REQUIRED: CortexV7 enables use_flash on CUDA ───────
    # Same prebuilt wheel as pretraining. No nvcc needed, installs in seconds.
    .pip_install(
        "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/"
        "download/v2.6.3/flash_attn-2.6.3+cu123torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    )

    # ── Bake in model + SFT training script dependencies at image build time ─
    .add_local_file("v3_modules.py", remote_path="/root/v3_modules.py")
    .add_local_file("Model.py",      remote_path="/root/Model.py")
    .add_local_file("kv_cache.py",   remote_path="/root/kv_cache.py")
    .add_local_file("Train_sft.py",  remote_path="/root/Train_sft.py")
)

# Persistent volume — same one used for pretraining checkpoints
volume       = modal.Volume.from_name("nexus-v1-ckpts", create_if_missing=True)
SFT_SAVE_DIR = "/data"


# ─────────────────────────────────────────────────────────────────────────────
# SFT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
@app.function(
    image   = image,
    gpu     = "H100:5",   # 🚀 Request 5x H100 GPUs as requested by user
    timeout = 86_400,                    # 24h — 5 epochs over 50k examples
    volumes = {"/data": volume},
    secrets = [modal.Secret.from_name("huggingface-secret")],
)
def execute_sft(
    checkpoint_filename: str = "ckpt_nexus_076500.pth",
    resume_existing_sft: bool = True,
):
    """
    Smart SFT launcher:
      1. Check for existing SFT checkpoints → resume from latest
      2. If no SFT checkpoints → load pretrained checkpoint, start fresh SFT
    """
    import sys
    sys.path.insert(0, "/root")

    print(f"🚀  Nexus SFT Phase 3 — Modal H100", flush=True)
    print(f"    GPU  : {__import__('torch').cuda.get_device_name(0)}", flush=True)
    print(f"    VRAM : {__import__('torch').cuda.get_device_properties(0).total_memory/1e9:.1f} GB",
          flush=True)

    # List volume contents
    try:
        volume_files = sorted(os.listdir("/data"))
        print(f"\n📂  Volume contents ({len(volume_files)} files):", flush=True)
        for f in volume_files[-20:]:          # last 20 to avoid wall of text
            size = os.path.getsize(f"/data/{f}") / 1e6
            print(f"    {f}  ({size:.0f} MB)", flush=True)
    except Exception as e:
        print(f"  ⚠️  Could not list volume: {e}", flush=True)

    # ── Decide: resume SFT vs start fresh ────────────────────────────────────
    sft_ckpts = sorted(glob.glob(os.path.join(SFT_SAVE_DIR, "nexus_sft_stage*.pth")))

    if sft_ckpts and resume_existing_sft:
        # SFT checkpoint exists → resume (pretrained checkpoint not needed)
        print(f"\n✅  Found {len(sft_ckpts)} SFT checkpoint(s) — resuming:", flush=True)
        for c in sft_ckpts:
            print(f"    • {os.path.basename(c)}", flush=True)

        cmd = [
            "python", "/root/Train_sft.py",
            "--save-dir", SFT_SAVE_DIR,
            "--checkpoint", "none",
        ]

    else:
        # No SFT checkpoint → pretrained checkpoint required
        ckpt_path = os.path.join(SFT_SAVE_DIR, checkpoint_filename)
        if not os.path.exists(ckpt_path):
            print(f"\n❌  ERROR: Pretrained checkpoint not found: {ckpt_path}", flush=True)
            print("     Available files:", flush=True)
            for f in os.listdir(SFT_SAVE_DIR):
                print(f"       {f}", flush=True)
            return

        if sft_ckpts and not resume_existing_sft:
            print(f"\n♻️  Ignoring {len(sft_ckpts)} existing SFT checkpoint(s) by request.", flush=True)
            for c in sft_ckpts:
                print(f"    • {os.path.basename(c)}", flush=True)
        print(f"\n📂  Starting fresh from base checkpoint {checkpoint_filename}",
              flush=True)
        cmd = [
            "python", "/root/Train_sft.py",
            "--checkpoint", ckpt_path,
            "--save-dir",   SFT_SAVE_DIR,
        ]

    print(f"\n▶  Command: {' '.join(cmd)}", flush=True)
    print("=" * 65, flush=True)

    child_env = os.environ.copy()
    child_env.setdefault("NEXUS_SFT_COMPILE", "0")

    process = subprocess.Popen(
        cmd,
        env            = child_env,
        stdout         = subprocess.PIPE,
        stderr         = subprocess.STDOUT,
        universal_newlines = True,
        bufsize        = 1,
    )

    # Stream output line-by-line (so logs show in modal logs --follow)
    for line in process.stdout:
        print(line, end="", flush=True)

        # Commit volume every time a checkpoint is saved
        # The save line always contains this signature:
        if "SFT checkpoint:" in line and ".pth" in line:
            try:
                volume.commit()
                print("  ✅  Volume committed.", flush=True)
            except Exception as e:
                print(f"  ⚠️  Volume commit failed: {e}", flush=True)

    process.wait()

    # Final commit — ensure the last save is flushed
    try:
        volume.commit()
        print("\n✅  Final volume commit done.", flush=True)
    except Exception as e:
        print(f"\n⚠️  Final volume commit failed: {e}", flush=True)

    if process.returncode != 0:
        print(f"\n❌  SFT script exited with code {process.returncode}", flush=True)
    else:
        print(f"\n🎉  SFT Phase 3 complete!", flush=True)
        print(f"    List: modal volume ls nexus-v1-ckpts | grep sft", flush=True)
        print(f"    Get:  modal volume get nexus-v1-ckpts nexus_sft_stage1_step*.pth .", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(
    checkpoint: str = "ckpt_nexus_076500.pth",
    fresh_start: bool = False,
):
    """
    Usage:
      modal run run_modal_sft.py                           # default checkpoint
      modal run --detach run_modal_sft.py                  # run in background
      modal run run_modal_sft.py --checkpoint my_ckpt.pth  # custom checkpoint
      modal logs nexus-sft-phase3                          # follow logs
    """
    print(f"🚀  Submitting Nexus SFT to Modal H100 …")
    print(f"    Checkpoint : {checkpoint}")
    print(f"    Mode       : {'fresh base checkpoint' if fresh_start else 'resume latest SFT if present'}")
    print(f"    Save dir   : {SFT_SAVE_DIR} (volume: nexus-v1-ckpts)")
    print(f"    Timeout    : 24h")
    print(f"    Torch      : 2.3.1 (pinned to match pretraining)")
    print(f"    Flash-attn : 2.6.3 (H100 required)")
    execute_sft.remote(
        checkpoint_filename=checkpoint,
        resume_existing_sft=not fresh_start,
    )
    print("\n✅  Job submitted.")
    print("    Logs : modal logs nexus-sft-phase3 --follow")
    print("    List : modal volume ls nexus-v1-ckpts")
