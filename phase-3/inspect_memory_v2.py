"""
Nexus Memory Deep Inspector V2
================================
Goes far beyond single-word projection to reveal what memory ACTUALLY stores.

Analyses performed:
  1. Vector Space Analysis   — norms, distributions, dimensional structure
  2. Semantic Decoding       — multi-token sequence reconstruction via beam search
  3. Slot Specialization     — clustering to find what each slot "specializes" in
  4. Cross-Slot Relationships — similarity matrix, redundancy detection
  5. Read Gate Dynamics      — how much each bridge is actually being used
  6. Write Routing Analysis  — which slots are alive, dead, or monopolizing
  7. Dimensional Analysis    — which dimensions are active, PCA decomposition
  8. Memory vs Input Comparison — does memory differ from raw embeddings?

Usage:
  python inspect_memory_v2.py -c ckpt_nexus_079500.pth
  python inspect_memory_v2.py -c ckpt_nexus_079500.pth --prompt "User: What is gravity?\nAssistant:"
  python inspect_memory_v2.py -c ckpt_nexus_079500.pth --detailed
"""

import torch
import torch.nn.functional as F
import tiktoken
import argparse
import math
import sys
from collections import defaultdict


def inspect_memory_v2(model, enc, memory_states, prompt_text=None, detailed=False):
    """
    Deep inspection of memory states.
    
    Args:
        model: Nexus model (unwrapped)
        enc: tiktoken encoder
        memory_states: list of [B, num_slots, dim] tensors (one per bridge)
        prompt_text: optional prompt text that was fed to get these memory states
        detailed: if True, print extra analysis
    """
    if hasattr(model, "module"):
        model = model.module

    device = next(model.parameters()).device
    dim = model.dim

    print("\n" + "=" * 80)
    print("  NEXUS MEMORY DEEP INSPECTOR V2")
    print("=" * 80)

    if prompt_text:
        print(f"\n  Input context: \"{prompt_text[:120]}{'...' if len(prompt_text) > 120 else ''}\"")

    # =========================================================================
    # 1. VECTOR SPACE ANALYSIS — What does the memory look like geometrically?
    # =========================================================================
    print(f"\n{'─' * 80}")
    print("  1. VECTOR SPACE ANALYSIS")
    print(f"{'─' * 80}")

    for bridge_idx, (mem_mod, mem_state) in enumerate(zip(model.memory, memory_states)):
        mem_float = mem_state.float()
        B, S, D = mem_float.shape
        
        # Per-slot norms
        norms = mem_float.norm(dim=-1)  # [B, S]
        avg_norms = norms.mean(dim=0)   # [S]
        
        # Activity classification
        active_mask = avg_norms > 0.1
        n_active = active_mask.sum().item()
        n_dead = S - n_active
        
        # Norm distribution
        print(f"\n  Bridge {bridge_idx} ({mem_mod.role}) — {S} slots × {D}d")
        print(f"    Active: {n_active}/{S} | Dead: {n_dead}/{S}")
        print(f"    Norm distribution: min={avg_norms.min():.4f} | "
              f"mean={avg_norms.mean():.4f} | max={avg_norms.max():.4f} | "
              f"std={avg_norms.std():.4f}")
        
        # Norm histogram (text-based)
        if detailed:
            buckets = torch.histc(avg_norms, bins=10, min=0, max=avg_norms.max().item())
            max_count = buckets.max().item()
            print(f"    Norm histogram:")
            for i, count in enumerate(buckets):
                bar = "█" * int(count / max(max_count, 1) * 30)
                lo = avg_norms.max().item() * i / 10
                hi = avg_norms.max().item() * (i + 1) / 10
                print(f"      [{lo:.2f}-{hi:.2f}] {bar} ({int(count.item())})")

    # =========================================================================
    # 2. SEMANTIC DECODING — What concepts do memory slots encode?
    # =========================================================================
    print(f"\n{'─' * 80}")
    print("  2. SEMANTIC DECODING — What each slot encodes")
    print(f"{'─' * 80}")
    print("  (Beyond single-word: showing multi-token context reconstruction)")

    for bridge_idx, (mem_mod, mem_state) in enumerate(zip(model.memory, memory_states)):
        mem_float = mem_state.float()
        B, S, D = mem_float.shape
        norms = mem_float[0].norm(dim=-1)  # [S] — batch 0
        
        # Show top-8 most active slots
        top_k = min(8, S)
        top_slots = torch.topk(norms, k=top_k).indices
        
        print(f"\n  Bridge {bridge_idx} ({mem_mod.role}) — Top {top_k} active slots:")
        
        for rank, slot_idx in enumerate(top_slots):
            slot_vec = mem_float[0, slot_idx.item()]  # [D]
            slot_norm = norms[slot_idx.item()].item()
            
            # --- Method 1: Standard LM head projection (single-token) ---
            with torch.no_grad():
                mem_normed = model.norm(slot_vec.unsqueeze(0).unsqueeze(0))
                logits = model.head(mem_normed)[0, 0]
                probs = torch.softmax(logits.float(), dim=-1)
            
            top_probs, top_tokens = torch.topk(probs, k=10)
            
            # Decode tokens
            single_tokens = []
            for p, t in zip(top_probs, top_tokens):
                try:
                    decoded = enc.decode([t.item()]).replace('\n', '\\n').replace('\r', '\\r')
                    single_tokens.append(f"'{decoded}'({p.item():.1%})")
                except Exception:
                    pass
            
            # --- Method 2: Entropy analysis (how "focused" is this slot?) ---
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            max_entropy = math.log(probs.shape[0])
            focus_pct = 1.0 - (entropy / max_entropy)  # 1.0 = perfectly focused on one token
            
            # --- Method 3: Top-90% mass coverage ---
            sorted_probs, _ = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            effective_tokens = (cumsum < 0.9).sum().item() + 1
            
            # --- Method 4: Semantic similarity to known token embeddings ---
            # Find which GROUPS of tokens this vector is similar to
            slot_normed = F.normalize(slot_vec.unsqueeze(0), dim=-1)  # [1, D]
            embed_weight = model.embed.weight.float()  # [V, D]
            embed_normed = F.normalize(embed_weight, dim=-1)
            cosine_sims = torch.mm(slot_normed, embed_normed.t())[0]  # [V]
            
            # Top similar embeddings (different from logit projection!)
            top_sim_vals, top_sim_ids = torch.topk(cosine_sims, k=5)
            embed_neighbors = []
            for s, t in zip(top_sim_vals, top_sim_ids):
                try:
                    decoded = enc.decode([t.item()]).replace('\n', '\\n')
                    embed_neighbors.append(f"'{decoded}'(cos={s.item():.3f})")
                except Exception:
                    pass
            
            # Print comprehensive slot info
            print(f"\n    Slot {slot_idx.item():3d} [norm={slot_norm:.3f}] "
                  f"focus={focus_pct:.1%} eff_tokens={effective_tokens}")
            print(f"      LM Head decode:  {' | '.join(single_tokens[:6])}")
            print(f"      Embed neighbors: {' | '.join(embed_neighbors)}")
            
            # --- Method 5: Greedy multi-token sequence reconstruction ---
            if detailed:
                # Use the memory vector as initial hidden state and greedily decode
                # This shows what "sentence" the memory slot is encoding
                with torch.no_grad():
                    h = slot_vec.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
                    generated_tokens = []
                    for _ in range(12):  # generate up to 12 tokens
                        h_normed = model.norm(h)
                        logits = model.head(h_normed)
                        next_token = logits[0, -1].argmax().item()
                        generated_tokens.append(next_token)
                        # Feed the token embedding back as next input
                        h = model.embed(torch.tensor([[next_token]], device=device)).float()
                    
                    reconstructed = enc.decode(generated_tokens)
                    reconstructed = reconstructed.replace('\n', '\\n')[:80]
                    print(f"      Sequence reconstruction: \"{reconstructed}\"")

    # =========================================================================
    # 3. SLOT SPECIALIZATION — Do slots learn different things?
    # =========================================================================
    print(f"\n{'─' * 80}")
    print("  3. SLOT SPECIALIZATION ANALYSIS")
    print(f"{'─' * 80}")

    for bridge_idx, (mem_mod, mem_state) in enumerate(zip(model.memory, memory_states)):
        mem_float = mem_state.float()
        B, S, D = mem_float.shape
        norms = mem_float[0].norm(dim=-1)
        
        # Cross-slot cosine similarity
        slot_normed = F.normalize(mem_float[0], dim=-1)  # [S, D]
        sim_matrix = torch.mm(slot_normed, slot_normed.t())  # [S, S]
        
        # Mask diagonal
        eye = torch.eye(S, device=sim_matrix.device)
        off_diag = sim_matrix * (1 - eye)
        
        avg_sim = off_diag.sum() / (S * (S - 1))
        max_sim = off_diag.max()
        
        # Find the most similar pair
        max_idx = off_diag.argmax()
        max_i, max_j = max_idx // S, max_idx % S
        
        # Find the most dissimilar pair (negative = opposing)
        min_sim = off_diag.min()
        min_idx = off_diag.argmin()
        min_i, min_j = min_idx // S, min_idx % S
        
        diversity_score = 1.0 - avg_sim.item()  # Higher = more diverse
        
        print(f"\n  Bridge {bridge_idx} ({mem_mod.role}):")
        print(f"    Diversity score: {diversity_score:.4f} (1.0 = fully diverse, 0.0 = collapsed)")
        print(f"    Avg cross-slot similarity: {avg_sim.item():.4f}")
        print(f"    Most similar pair: slots {max_i.item()},{max_j.item()} (cos={max_sim.item():.4f})")
        print(f"    Most opposing pair: slots {min_i.item()},{min_j.item()} (cos={min_sim.item():.4f})")
        
        # Simple clustering: group slots by similarity
        if detailed and S <= 128:
            # Quick k-means-like grouping using cosine similarity
            active_mask = norms > 0.1
            active_indices = torch.where(active_mask)[0]
            
            if len(active_indices) >= 4:
                active_vecs = F.normalize(mem_float[0, active_indices], dim=-1)
                active_sim = torch.mm(active_vecs, active_vecs.t())
                
                # Find clusters by thresholding similarity
                clusters = []
                assigned = set()
                for i in range(len(active_indices)):
                    if i in assigned:
                        continue
                    cluster = [i]
                    assigned.add(i)
                    for j in range(i + 1, len(active_indices)):
                        if j not in assigned and active_sim[i, j] > 0.7:
                            cluster.append(j)
                            assigned.add(j)
                    clusters.append(cluster)
                
                print(f"    Slot clusters (cos > 0.7): {len(clusters)} groups")
                for ci, cluster in enumerate(clusters[:5]):  # Show top 5 clusters
                    slot_ids = [active_indices[c].item() for c in cluster]
                    # What does this cluster encode?
                    cluster_vec = mem_float[0, slot_ids].mean(dim=0)  # average vector
                    with torch.no_grad():
                        cl_normed = model.norm(cluster_vec.unsqueeze(0).unsqueeze(0))
                        cl_logits = model.head(cl_normed)
                        cl_probs = torch.softmax(cl_logits[0, 0].float(), dim=-1)
                    top_p, top_t = torch.topk(cl_probs, k=3)
                    tokens_str = []
                    for p, t in zip(top_p, top_t):
                        try:
                            tokens_str.append(f"'{enc.decode([t.item()])}'")
                        except Exception:
                            pass
                    print(f"      Cluster {ci}: slots={slot_ids} → {', '.join(tokens_str)}")

    # =========================================================================
    # 4. DIMENSIONAL ANALYSIS — Which dimensions are doing the work?
    # =========================================================================
    print(f"\n{'─' * 80}")
    print("  4. DIMENSIONAL ANALYSIS")
    print(f"{'─' * 80}")

    for bridge_idx, (mem_mod, mem_state) in enumerate(zip(model.memory, memory_states)):
        mem_float = mem_state.float()
        B, S, D = mem_float.shape
        
        # Per-dimension statistics across all slots
        dim_mean = mem_float[0].mean(dim=0)     # [D]
        dim_std = mem_float[0].std(dim=0)       # [D]
        dim_range = mem_float[0].max(dim=0)[0] - mem_float[0].min(dim=0)[0]  # [D]
        
        # Which dimensions are most variable (information-rich)?
        top_var_dims = torch.topk(dim_std, k=10).indices
        # Which dimensions are "dead" (low variance)?
        dead_dims = (dim_std < 0.01).sum().item()
        
        print(f"\n  Bridge {bridge_idx} ({mem_mod.role}):")
        print(f"    Dimension stats: mean_std={dim_std.mean():.4f} | "
              f"dead_dims={dead_dims}/{D} | max_range={dim_range.max():.4f}")
        print(f"    Most active dimensions: {top_var_dims.tolist()}")
        
        # PCA-like analysis: how many dimensions capture 90% of variance?
        # Use SVD on the slot matrix
        if S >= 4:
            centered = mem_float[0] - mem_float[0].mean(dim=0, keepdim=True)
            try:
                U, singular_vals, V = torch.svd(centered)
                total_var = (singular_vals ** 2).sum()
                cumvar = torch.cumsum(singular_vals ** 2, dim=0) / total_var
                dims_for_90 = (cumvar < 0.9).sum().item() + 1
                dims_for_95 = (cumvar < 0.95).sum().item() + 1
                dims_for_99 = (cumvar < 0.99).sum().item() + 1
                
                print(f"    Effective dimensionality: "
                      f"90%→{dims_for_90}d | 95%→{dims_for_95}d | 99%→{dims_for_99}d "
                      f"(out of {D}d)")
                print(f"    Top singular values: {[f'{v:.2f}' for v in singular_vals[:5].tolist()]}")
            except Exception:
                print(f"    (SVD failed — skipping PCA analysis)")

    # =========================================================================
    # 5. MEMORY INIT vs TRAINED — How far has memory evolved from initialization?
    # =========================================================================
    print(f"\n{'─' * 80}")
    print("  5. MEMORY EVOLUTION (trained state vs initialization)")
    print(f"{'─' * 80}")

    for bridge_idx, (mem_mod, mem_state) in enumerate(zip(model.memory, memory_states)):
        mem_float = mem_state.float()
        init_float = mem_mod.memory_init.float().unsqueeze(0)  # [1, S, D]
        
        # Compare current state to initial state
        init_expanded = init_float.expand_as(mem_float)
        
        # Cosine similarity between current and init
        current_normed = F.normalize(mem_float[0], dim=-1)
        init_normed = F.normalize(init_float[0], dim=-1)
        init_sim = (current_normed * init_normed).sum(dim=-1)  # [S]
        
        # L2 distance
        l2_dist = (mem_float[0] - init_float[0]).norm(dim=-1)  # [S]
        
        print(f"\n  Bridge {bridge_idx} ({mem_mod.role}):")
        print(f"    Avg cosine similarity to init: {init_sim.mean():.4f} "
              f"(0.0=completely different, 1.0=unchanged)")
        print(f"    Avg L2 distance from init: {l2_dist.mean():.4f}")
        
        # Which slots changed the most?
        most_changed = torch.topk(l2_dist, k=min(5, len(l2_dist))).indices
        least_changed = torch.topk(l2_dist, k=min(5, len(l2_dist)), largest=False).indices
        print(f"    Most evolved slots: {most_changed.tolist()} "
              f"(L2={[f'{l2_dist[i]:.3f}' for i in most_changed]})")
        print(f"    Least evolved slots: {least_changed.tolist()} "
              f"(L2={[f'{l2_dist[i]:.3f}' for i in least_changed]})")

    # =========================================================================
    # 6. READ GATE & WRITE MASK DIAGNOSTICS
    # =========================================================================
    print(f"\n{'─' * 80}")
    print("  6. READ/WRITE DYNAMICS")
    print(f"{'─' * 80}")

    for bridge_idx, mem_mod in enumerate(model.memory):
        print(f"\n  Bridge {bridge_idx} ({mem_mod.role}):")
        
        # Read gate
        if hasattr(mem_mod, 'last_read_gate') and mem_mod.last_read_gate is not None:
            gate = mem_mod.last_read_gate.float()
            mean_gate = gate.mean().item()
            min_gate = gate.min().item()
            max_gate = gate.max().item()
            
            # Per-dimension gate analysis
            dim_gate = gate.mean(dim=(0, 1))  # [D] — avg gate per dimension
            active_dims = (dim_gate > 0.3).sum().item()
            
            status = ("HEAVILY ACTIVE" if mean_gate > 0.2 
                      else "ACTIVE" if mean_gate > 0.05 
                      else "MOSTLY BYPASSED" if mean_gate > 0.01 
                      else "DEAD")
            
            print(f"    Read gate: avg={mean_gate:.4f} | min={min_gate:.4f} | "
                  f"max={max_gate:.4f} → {status}")
            print(f"    Active dimensions (gate > 0.3): {active_dims}/{dim}")
        else:
            print(f"    Read gate: (no data — run a forward pass first)")
        
        # Write mask
        if hasattr(mem_mod, 'last_write_mask') and mem_mod.last_write_mask is not None:
            mask = mem_mod.last_write_mask.float()
            if mask.dim() == 3:
                usage = mask.mean(dim=(0, 1))
            else:
                usage = mask.mean(dim=0)
            
            n_active = (usage > 0.01).sum().item()
            n_total = usage.shape[-1]
            
            # Usage distribution
            top5 = torch.topk(usage, min(5, n_total))
            bottom5 = torch.topk(usage, min(5, n_total), largest=False)
            
            print(f"    Write routing: {n_active}/{n_total} slots receiving writes")
            print(f"      Hottest slots: {list(zip(top5.indices.tolist(), [f'{v:.3f}' for v in top5.values.tolist()]))}")
            print(f"      Coldest slots: {list(zip(bottom5.indices.tolist(), [f'{v:.3f}' for v in bottom5.values.tolist()]))}")
        else:
            print(f"    Write mask: (no data — run a forward pass first)")

    # =========================================================================
    # 7. FORGET GATE ANALYSIS — How quickly does memory decay?
    # =========================================================================
    print(f"\n{'─' * 80}")
    print("  7. FORGET GATE ANALYSIS (Learned Retention)")
    print(f"{'─' * 80}")

    for bridge_idx, mem_mod in enumerate(model.memory):
        # The forget gate is: sigmoid(Linear(memory_state))
        # But we can inspect its bias to see the "default" retention
        gate_weight = mem_mod.gate[0].weight.float()
        gate_bias = mem_mod.gate[0].bias.float()
        
        # Default retention when memory is near-zero
        default_retention = torch.sigmoid(gate_bias)
        
        print(f"\n  Bridge {bridge_idx} ({mem_mod.role}) [retain_floor={mem_mod.retain_floor}]:")
        print(f"    Default retention (bias): mean={default_retention.mean():.4f} | "
              f"min={default_retention.min():.4f} | max={default_retention.max():.4f}")
        print(f"    Forget gate weight norm: {gate_weight.norm():.4f}")
        
        # How many dimensions have high default retention?
        high_retain = (default_retention > 0.8).sum().item()
        low_retain = (default_retention < 0.3).sum().item()
        print(f"    High retention dims (>0.8): {high_retain}/{dim}")
        print(f"    Low retention dims (<0.3): {low_retain}/{dim}")

    # =========================================================================
    # 8. COMPARISON: MEMORY vs RAW EMBEDDINGS
    # =========================================================================
    if prompt_text:
        print(f"\n{'─' * 80}")
        print("  8. MEMORY vs INPUT EMBEDDING COMPARISON")
        print(f"{'─' * 80}")
        
        tokens = enc.encode(prompt_text, allowed_special={"<|endoftext|>"})
        if len(tokens) > 0:
            token_ids = torch.tensor([tokens], device=device)
            with torch.no_grad():
                token_embeds = model.embed(token_ids).float()  # [1, SeqLen, D]
            
            # Average input embedding
            avg_embed = token_embeds.mean(dim=1)  # [1, D]
            avg_embed_normed = F.normalize(avg_embed, dim=-1)
            
            for bridge_idx, (mem_mod, mem_state) in enumerate(zip(model.memory, memory_states)):
                mem_float = mem_state.float()
                mem_normed = F.normalize(mem_float[0], dim=-1)  # [S, D]
                
                # Similarity between each slot and avg input
                sim_to_input = torch.mm(mem_normed, avg_embed_normed.t()).squeeze(-1)  # [S]
                
                top_similar = torch.topk(sim_to_input, k=min(3, len(sim_to_input)))
                
                print(f"\n  Bridge {bridge_idx} ({mem_mod.role}):")
                print(f"    Avg slot-to-input similarity: {sim_to_input.mean():.4f}")
                print(f"    Most input-aligned slots: "
                      f"{list(zip(top_similar.indices.tolist(), [f'{v:.4f}' for v in top_similar.values.tolist()]))}")
                
                # Are slots encoding something DIFFERENT from the input?
                orthogonal_slots = (sim_to_input.abs() < 0.1).sum().item()
                print(f"    Orthogonal to input (|cos| < 0.1): {orthogonal_slots}/{mem_float.shape[1]} "
                      f"→ {'slots encode NOVEL information beyond input' if orthogonal_slots > mem_float.shape[1]//2 else 'slots partially mirror input'}")

    print(f"\n{'=' * 80}")
    print("  INSPECTION COMPLETE")
    print(f"{'=' * 80}\n")


def run_with_checkpoint(checkpoint_path, prompt_text, detailed=False):
    """Load a checkpoint and run full inspection."""
    print(f"\n📦  Loading checkpoint: {checkpoint_path}")
    
    # Import model
    sys.path.insert(0, '.')
    from Model import Nexus
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint
    cfg = ckpt.get("cfg", {})
    
    model = Nexus(
        vocab_size=cfg.get("vocab_size", 100277),
        dim=cfg.get("dim", 1280),
        heads=cfg.get("heads", 16),
        kv_heads=cfg.get("kv_heads", 4),
        num_layers=cfg.get("num_layers", 20),
        memory_slots=cfg.get("memory_slots", 128),
        use_flash=False,  # Use SDPA for inspection
    ).to(device)
    
    # Load weights
    state_dict = ckpt["model_state_dict"]
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            clean_state_dict[k[10:]] = v
        elif k.startswith("module."):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v
    
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()
    
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Tokenize prompt and run forward pass
    tokens = enc.encode(prompt_text, allowed_special={"<|endoftext|>"})
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    
    print(f"📝  Running forward pass with {len(tokens)} tokens...")
    
    with torch.no_grad(), torch.amp.autocast(device, dtype=torch.bfloat16, enabled=device=="cuda"):
        logits, _, _, memory_states = model(x)
    
    # Now inspect
    inspect_memory_v2(model, enc, memory_states, prompt_text=prompt_text, detailed=detailed)
    
    # Bonus: show what the model would generate
    print(f"\n{'─' * 80}")
    print("  BONUS: Model generation from this prompt")
    print(f"{'─' * 80}")
    
    generated = []
    next_logits = logits
    for _ in range(30):
        next_token = next_logits[0, -1].float().argmax().item()
        if next_token == enc.eot_token:
            break
        generated.append(next_token)
        token_t = torch.tensor([[next_token]], device=device)
        with torch.no_grad(), torch.amp.autocast(device, dtype=torch.bfloat16, enabled=device=="cuda"):
            next_logits, _, _, memory_states = model(token_t, memory_state=memory_states)
    
    response = enc.decode(generated)
    print(f"  Generated: \"{response[:200]}\"")
    
    return model, enc, memory_states


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nexus Memory Deep Inspector V2")
    parser.add_argument("-c", "--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--prompt", default=(
        "System: You are Nexus, an advanced AI assistant created by Siddi Vinayaka.\n"
        "User: What is your architecture and what makes you unique?\nAssistant:"
    ), help="Prompt to feed through the model")
    parser.add_argument("--detailed", action="store_true", help="Show extra analysis")
    args = parser.parse_args()
    
    run_with_checkpoint(args.checkpoint, args.prompt, detailed=args.detailed)
