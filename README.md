# Nexus AI - Hierarchical Memory Model

This repository contains the training codebase for **Nexus**, a custom AI model built by Siddi Vinayaka. The core innovation of this architecture is its **Hierarchical Memory** mechanism, which utilizes sparse, layered read/write memory gates that selectively access persistent memory slots.

## Architecture Highlights
- **Base Architecture**: Transformer-based decoder with custom modifications.
- **Hierarchical Memory**: Uses state, read, and write gates across specific "bridge" layers (e.g., Layer 3, Layer 10, Layer 17). This allows the network to bypass deep memory reads for simple syntax parsing but open wide for factual or identity retrieval at later layers.
- **Optimized for 8192 Context Length**: Designed to handle extremely long `<thought>` reasoning traces.

## Training Pipeline

### Phase 1 & 2: Pre-training
The model undergoes pre-training on a dense mix of tokens to build foundational algorithmic logic and language comprehension. The dataset focuses heavily on coding, mathematics, and high-quality general text.

### Phase 3: Supervised Fine-Tuning (SFT)
The SFT phase teaches the model specific behavioral formatting without overfitting to factual data:
- **Reasoning Traces**: Wraps logic in `<thought>...</thought>` blocks.
- **Mathematical Accuracy**: Uses strict LaTeX formatting and `<<x*y=z>>` calculator tokens based on GSM8K styles.
- **Identity Maintenance**: A robust identity injection prevents the model from hallucinating that it was created by OpenAI or Anthropic, solidly reinforcing its identity as **Nexus**, created by **Siddi Vinayaka**.

### Hardware & Infrastructure
The training scripts are designed to run on **Modal** utilizing multiple **NVIDIA H100** GPUs. 
- Using `batch_size=1` with large `grad_accum_steps` safely avoids CUDA Out Of Memory (OOM) errors even on 8192 sequence lengths.
- Achieves high step throughput leveraging PyTorch DDP across multiple GPUs.

## Repository Structure
- `phase-2/`: Scripts and architectures related to base pre-training.
- `phase-3/`: SFT training scripts, loss functions, and evaluation scripts.
- *(Note: Model checkpoints, `.safetensors`, `.pth` files, datasets, and massive analysis logs are excluded from version control).*
