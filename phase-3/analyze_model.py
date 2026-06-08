"""
NEXUS — COMPREHENSIVE MODEL ANALYZER (CPU-friendly)
Fixes OOM by running on CPU and downsampling large tensors.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import json
import os
import sys
import time
import argparse
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import tiktoken
from collections import OrderedDict

# Import model
sys.path.insert(0, str(Path(__file__).parent))
from Model import Nexus, KVCache

class ActivationCapture:
    """Captures activations from model forward pass (CPU-safe)"""
    
    def __init__(self):
        self.activations = {}
        self.hooks = []
    
    def register_hooks(self, model: torch.nn.Module):
        """Register forward hooks to capture activations"""
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self.activations[name] = output.detach().cpu()
            return hook
        
        self.hooks.append(model.embed.register_forward_hook(make_hook("embed")))
        
        for i, layer in enumerate(model.layers):
            self.hooks.append(layer.attn.out.register_forward_hook(make_hook(f"layer_{i}_attn")))
            self.hooks.append(layer.ffn.register_forward_hook(make_hook(f"layer_{i}_ffn")))
            self.hooks.append(layer.hyper_attn.register_forward_hook(make_hook(f"layer_{i}_hyper_attn")))
            self.hooks.append(layer.hyper_ffn.register_forward_hook(make_hook(f"layer_{i}_hyper_ffn")))
            
        for i, mem in enumerate(model.memory):
            self.hooks.append(mem.read_attn.register_forward_hook(make_hook(f"memory_{i}_read")))
            self.hooks.append(mem.write_attn.register_forward_hook(make_hook(f"memory_{i}_write")))
            self.hooks.append(mem.read_gate.register_forward_hook(make_hook(f"memory_{i}_read_gate")))
            
        self.hooks.append(model.norm.register_forward_hook(make_hook("norm")))
        self.hooks.append(model.final_memory_read.register_forward_hook(make_hook("final_memory_read")))
        self.hooks.append(model.head.register_forward_hook(make_hook("head")))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        return self.activations

class WeightExtractor:
    @staticmethod
    def extract_weights(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        weights = {}
        weights["embed.weight"] = model.embed.weight.detach().cpu()
        for i, mem in enumerate(model.memory):
            weights[f"memory_{i}.read_attn.q.weight"] = mem.read_attn.q.weight.detach().cpu()
            weights[f"memory_{i}.write_attn.q.weight"] = mem.write_attn.q.weight.detach().cpu()
            weights[f"memory_{i}.read_gate.weight"] = mem.read_gate[0].weight.detach().cpu()
        for i, layer in enumerate(model.layers):
            weights[f"layer_{i}.attn.q.weight"] = layer.attn.q.weight.detach().cpu()
            weights[f"layer_{i}.ffn.w1.weight"] = layer.ffn.w1.weight.detach().cpu()
            weights[f"layer_{i}.hyper_attn.alpha"] = layer.hyper_attn.alpha.detach().cpu()
            weights[f"layer_{i}.hyper_attn.beta"] = layer.hyper_attn.beta.detach().cpu()
        weights["norm.weight"] = model.norm.weight.detach().cpu()
        return weights

class StatisticsCalculator:
    @staticmethod
    def compute_stats(tensor: torch.Tensor) -> Dict[str, Any]:
        tensor_np = tensor.detach().cpu().float().numpy()
        return {
            "shape": list(tensor.shape),
            "mean": float(np.mean(tensor_np)),
            "std": float(np.std(tensor_np)),
            "min": float(np.min(tensor_np)),
            "max": float(np.max(tensor_np)),
            "median": float(np.median(tensor_np)),
            "q25": float(np.percentile(tensor_np, 25)),
            "q75": float(np.percentile(tensor_np, 75)),
            "abs_mean": float(np.mean(np.abs(tensor_np))),
            "abs_max": float(np.max(np.abs(tensor_np))),
            "sparsity": float(np.mean(np.abs(tensor_np) < 1e-6)),
        }

class Visualizer:
    def __init__(self, output_dir: str, max_points: int = 5000, dpi: int = 90, skip_3d: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_points = max_points
        self.dpi = dpi
        self.skip_3d = skip_3d
        sns.set_style("darkgrid")
        plt.rcParams['figure.facecolor'] = 'white'
    
    def visualize_tensor(self, tensor: torch.Tensor, name: str, prefix: str = ""):
        tensor_np = tensor.detach().cpu().float().numpy()
        flat = tensor_np.flatten()
        if len(flat) > self.max_points:
            indices = np.random.choice(len(flat), self.max_points, replace=False)
            flat = flat[indices]
        self._plot_histogram(flat, name, prefix)
        self._plot_heatmap(tensor_np, name, prefix)
        if not self.skip_3d:
            self._plot_3d_embedding(tensor_np, name, prefix)
            self._plot_3d_surface(tensor_np, name, prefix)
        
        plt.close('all')
        gc.collect()
    
    def _plot_histogram(self, data: np.ndarray, name: str, prefix: str):
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=100, alpha=0.7, edgecolor='black')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'{name} - Distribution')
        stats_text = f'Mean: {np.mean(data):.4f}\nStd: {np.std(data):.4f}\nMin: {np.min(data):.4f}\nMax: {np.max(data):.4f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{prefix}_{name}_hist.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_heatmap(self, data: np.ndarray, name: str, prefix: str):
        if len(data.shape) == 1:
            data_2d = data.reshape(-1, 1)
        elif len(data.shape) == 2:
            data_2d = data
        else:
            idx = data.shape[0] // 2
            data_2d = data[idx] if len(data.shape) == 3 else data.reshape(data.shape[0], -1)
        if data_2d.shape[0] > 200:
            data_2d = data_2d[:200]
        if data_2d.shape[1] > 200:
            data_2d = data_2d[:, :200]
        plt.figure(figsize=(12, 10))
        sns.heatmap(data_2d, cmap='RdBu_r', center=0, cbar=True, xticklabels=False, yticklabels=False)
        plt.title(f'{name} - Heatmap')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{prefix}_{name}_heatmap.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_3d_embedding(self, data: np.ndarray, name: str, prefix: str):
        if len(data.shape) > 2:
            data_2d = data.reshape(-1, data.shape[-1])
        else:
            data_2d = data
        
        if len(data_2d.shape) < 2:
            return
        if data_2d.shape[0] < 3 or data_2d.shape[1] < 3:
            return
        
        if data_2d.shape[0] > self.max_points:
            indices = np.random.choice(data_2d.shape[0], self.max_points, replace=False)
            data_2d = data_2d[indices]
        
        from sklearn.decomposition import PCA
        n_comp = min(3, data_2d.shape[1])
        pca = PCA(n_components=n_comp)
        data_3d = pca.fit_transform(data_2d)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        if n_comp >= 3:
            ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2],
                    c=np.arange(data_3d.shape[0]), cmap='viridis', alpha=0.6, s=10)
            ax.set_zlabel('PC3')
        else:
            ax.scatter(data_3d[:, 0], data_3d[:, 1], np.zeros(data_3d.shape[0]),
                    c=np.arange(data_3d.shape[0]), cmap='viridis', alpha=0.6, s=10)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'{name} - 3D PCA (var={pca.explained_variance_ratio_.sum():.2%})')
        plt.colorbar(ax.collections[0], ax=ax, label='Sample Index')
        plt.savefig(self.output_dir / f'{prefix}_{name}_embed3d.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def _plot_3d_surface(self, data: np.ndarray, name: str, prefix: str):
        if len(data.shape) < 2:
            return
        if len(data.shape) == 2:
            data_2d = data
        else:
            idx = data.shape[0] // 2
            data_2d = data[idx] if len(data.shape) == 3 else data.reshape(data.shape[0], -1)
        max_dim = 50
        if data_2d.shape[0] > max_dim:
            data_2d = data_2d[:max_dim]
        if data_2d.shape[1] > max_dim:
            data_2d = data_2d[:, :max_dim]
        x = np.arange(data_2d.shape[1])
        y = np.arange(data_2d.shape[0])
        X, Y = np.meshgrid(x, y)
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, data_2d, cmap='viridis', alpha=0.8)
        ax.set_xlabel('Dim1')
        ax.set_ylabel('Dim2')
        ax.set_zlabel('Value')
        ax.set_title(f'{name} - 3D Surface')
        fig.colorbar(surf, ax=ax, shrink=0.5)
        plt.savefig(self.output_dir / f'{prefix}_{name}_surface3d.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
    
    def plot_alignment_structure(self, tensors: Dict[str, torch.Tensor], name: str, max_samples_per_layer: int = 500):
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        colors = plt.cm.tab20(np.linspace(0, 1, len(tensors)))
        for idx, (tensor_name, tensor) in enumerate(tensors.items()):
            tensor_np = tensor.detach().cpu().float().numpy().flatten()
            if len(tensor_np) > max_samples_per_layer:
                indices = np.random.choice(len(tensor_np), max_samples_per_layer, replace=False)
                samples = tensor_np[indices]
            else:
                samples = tensor_np
            n = len(samples)
            x = np.full(n, idx)
            y = np.arange(n)
            z = samples
            ax.scatter(x, y, z, color=colors[idx], label=tensor_name, alpha=0.6, s=5)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Position (downsampled)')
        ax.set_zlabel('Activation Value')
        ax.set_title(f'{name} - Alignment Structure')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{name}_alignment_structure.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()

class ModelAnalyzer:
    def __init__(self, checkpoint_path: str, output_dir: Optional[str] = None, device: str = "cpu", dpi: int = 90, skip_3d: bool = False, skip_weights: bool = False):
        self.checkpoint_path = checkpoint_path
        if output_dir is None:
            timestamp = int(time.time())
            output_dir = f"analysis/viz_{timestamp}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
        self.device = torch.device(device if device in ('cuda', 'cpu') else 'cpu')
        print(f"Using device: {self.device}")
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.visualizer = Visualizer(str(self.output_dir), dpi=dpi, skip_3d=skip_3d)
        self.skip_weights = skip_weights
        self.model = self._load_model()
    
    def _load_model(self) -> Nexus:
        print(f"Loading checkpoint: {self.checkpoint_path} (on {self.device})")
        ckpt = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        if "cfg" in ckpt:
            cfg = ckpt["cfg"]
            config = {
                "vocab_size": cfg.get("vocab_size", 100_277),
                "dim": cfg.get("dim", 1280),
                "heads": cfg.get("heads", 16),
                "kv_heads": cfg.get("kv_heads", 4),
                "num_layers": cfg.get("num_layers", 20),
                "memory_slots": cfg.get("memory_slots", 128),
                "mtp_depths": cfg.get("mtp_depths", 1),
                "use_flash": False,
            }
        else:
            config = {
                "vocab_size": 100_277,
                "dim": 1280,
                "heads": 16,
                "kv_heads": 4,
                "num_layers": 20,
                "memory_slots": 128,
                "mtp_depths": 1,
                "use_flash": False,
            }
        print(f"Model config: {config}")
        model = Nexus(**config).to(self.device)
        state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print(f"Model loaded successfully on {self.device}")
        return model
    
    def analyze(self, prompt: str = "User: Who developed you?\nAssistant:"):
        print("\n" + "="*70)
        print("STARTING COMPREHENSIVE MODEL ANALYSIS")
        print("="*70)
        print("[1/4] Capturing activations...")
        activations = self._capture_activations(prompt)
        print("[2/4] Extracting weights...")
        weights = {} if self.skip_weights else WeightExtractor.extract_weights(self.model)
        print("[3/4] Computing statistics...")
        summary = self._compute_summary(activations, weights, prompt)
        print("[4/4] Generating visualizations...")
        self._generate_visualizations(activations, weights)
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n[SUCCESS] Summary saved to: {summary_path}")
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("="*70)
    
    def _capture_activations(self, prompt: str) -> Dict[str, torch.Tensor]:
        tokens = self.enc.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        capture = ActivationCapture()
        capture.register_hooks(self.model)
        with torch.no_grad():
            outputs = self.model(input_ids)
            if isinstance(outputs, tuple) and len(outputs) >= 4 and isinstance(outputs[3], list):
                for i, ms in enumerate(outputs[3]):
                    capture.activations[f"memory_{i}_state"] = ms.detach().cpu()
        activations = capture.get_activations()
        capture.remove_hooks()
        return activations
    
    def _compute_summary(self, activations: Dict, weights: Dict, prompt: str) -> Dict:
        summary = {"prompt": prompt, "activation_stats": {}, "weight_stats": {}}
        for name, tensor in activations.items():
            summary["activation_stats"][f"prefill.{name}"] = StatisticsCalculator.compute_stats(tensor)
        for name, tensor in weights.items():
            summary["weight_stats"][name] = StatisticsCalculator.compute_stats(tensor)
        return summary
    
    def _generate_visualizations(self, activations: Dict, weights: Dict):
        print("  Visualizing activations...")
        for name, tensor in activations.items():
            print(f"    - {name}")
            self.visualizer.visualize_tensor(tensor, name, prefix="prefill")
        if weights:
            print("  Visualizing weights...")
            for name, tensor in weights.items():
                print(f"    - {name}")
                self.visualizer.visualize_tensor(tensor, name.replace(".", "_"), prefix="weight")
        print("  Creating alignment structure plots...")
        layer_acts = {k: v for k, v in activations.items() if "layer" in k and "hyper" not in k}
        if layer_acts:
            self.visualizer.plot_alignment_structure(layer_acts, "layer_alignment")
        hyper_acts = {k: v for k, v in activations.items() if "hyper" in k}
        if hyper_acts:
            self.visualizer.plot_alignment_structure(hyper_acts, "hyper_alignment")

def main():
    parser = argparse.ArgumentParser(description="Analyze Nexus model activations and weights")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="User: Who developed you?\nAssistant:", help="Prompt")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device (cpu recommended to avoid OOM)")
    parser.add_argument("--dpi", type=int, default=90, help="DPI for saved visualization plots (lower = smaller file sizes)")
    parser.add_argument("--skip-3d", action="store_true", help="Skip memory-heavy 3D surface/PCA plots")
    parser.add_argument("--skip-weights", action="store_true", help="Skip weights visualization (reduces output size & time)")
    args = parser.parse_args()
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    analyzer = ModelAnalyzer(
        args.checkpoint, 
        args.output_dir, 
        device=args.device,
        dpi=args.dpi,
        skip_3d=args.skip_3d,
        skip_weights=args.skip_weights
    )
    analyzer.analyze(args.prompt)

if __name__ == "__main__":
    main()