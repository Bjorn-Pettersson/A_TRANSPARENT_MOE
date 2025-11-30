"""
Evaluate MoE model on MMLU subjects and analyze expert routing behavior.

This script:
1. Loads a pretrained MoE checkpoint (frozen weights)
2. Runs inference on MMLU test/validation sets per subject
3. Captures router decisions (which experts are selected per sample)
4. Generates Figure 2-style heatmaps showing expert activation by domain
5. Logs performance metrics (accuracy) per subject

Usage:
    python eval_mmlu.py --checkpoint out-pretrain-4gb-mmlu/ckpt.pt --out_dir eval_results
"""

import os
import sys
import argparse
import pickle
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from tqdm import tqdm

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("Warning: matplotlib/seaborn not available. Skipping plot generation.")

# Allow running from the eval/ subfolder without modifying PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import GPT, GPTConfig

# MMLU subjects matching Figure 2 (can be extended)
DEFAULT_SUBJECTS = [
    'college_chemistry',
    'global_facts', 
    'management',
    'medical_genetics',
    'philosophy',
    'prehistory',
    'professional_accounting',
    'professional_law',
    'sociology',
    'us_foreign_policy',
]


class RouterHook:
    """Captures expert routing decisions during forward pass."""
    
    def __init__(self, model, n_layers, n_experts):
        self.model = model
        self.n_layers = n_layers
        self.n_experts = n_experts
        # Storage: {layer_idx: {expert_idx: {domain: count}}}
        # Use standard dict nesting to ease serialization
        self.expert_counts = {}
        self.current_domain = None
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on all MoE router layers."""
        for layer_idx, block in enumerate(self.model.transformer.h):
            # Check if this block has MoE
            if hasattr(block, 'mlp') and hasattr(block.mlp, 'router'):
                hook = block.mlp.register_forward_hook(
                    self._make_hook(layer_idx)
                )
                self.hooks.append(hook)
    
    def _make_hook(self, layer_idx):
        """Create hook function for a specific layer."""
        def hook_fn(module, input, output):
            # For SequenceMoE, we stored routing info in module attributes
            if hasattr(module, 'last_selected_experts'):
                selected = module.last_selected_experts  # shape: (batch, top_k)
                if selected is not None and self.current_domain is not None:
                    # Flatten and count
                    experts_flat = selected.cpu().numpy().flatten()
                    # Initialize layer dict if missing
                    layer_dict = self.expert_counts.setdefault(layer_idx, {})
                    for expert_id in experts_flat:
                        exp_dict = layer_dict.setdefault(int(expert_id), {})
                        exp_dict[self.current_domain] = exp_dict.get(self.current_domain, 0) + 1
        return hook_fn
    
    def set_domain(self, domain: str):
        """Set current MMLU subject/domain for tracking."""
        self.current_domain = domain
    
    def remove_hooks(self):
        """Clean up hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_activation_matrix(self, layer_idx: int, domains: List[str]) -> np.ndarray:
        """
        Build activation matrix for visualization.
        Rows = experts, Columns = domains.
        """
        matrix = np.zeros((self.n_experts, len(domains)))
        layer_counts = self.expert_counts.get(layer_idx, {})
        for expert_idx in range(self.n_experts):
            expert_counts = layer_counts.get(expert_idx, {})
            for domain_idx, domain in enumerate(domains):
                count = expert_counts.get(domain, 0)
                matrix[expert_idx, domain_idx] = count
        return matrix
    
    def get_all_domains(self) -> List[str]:
        """Get sorted list of all domains encountered."""
        domains = set()
        for layer_counts in self.expert_counts.values():
            for expert_counts in layer_counts.values():
                domains.update(expert_counts.keys())
        return sorted(domains)


def load_mmlu_subject_from_bin(data_dir: str, subject: str, split: str = 'val'):
    """
    Load MMLU subject data from .bin files (pickled strings).
    
    Returns list of text samples.
    """
    bin_path = os.path.join(data_dir, f"{subject}_{split}.bin")
    if not os.path.exists(bin_path):
        print(f"Warning: {bin_path} not found, skipping.")
        return []
    
    try:
        with open(bin_path, 'rb') as f:
            data = pickle.load(f)
        
        # If it's a string with separator, split it
        if isinstance(data, str):
            separator = "---"
            samples = [s.strip() for s in data.split(separator) if s.strip()]
            return samples
        elif isinstance(data, list):
            return data
        else:
            print(f"Warning: unexpected data format in {bin_path}")
            return []
    except Exception as e:
        print(f"Error loading {bin_path}: {e}")
        return []


def tokenize_samples(samples: List[str], enc, block_size: int):
    """
    Tokenize text samples and truncate/pad to block_size.
    
    Returns: List of torch tensors (seq_len,)
    """
    tokenized = []
    for sample in samples:
        ids = enc.encode_ordinary(sample)
        # Truncate or pad to block_size
        if len(ids) > block_size:
            ids = ids[:block_size]
        elif len(ids) < block_size:
            # Pad with EOT token
            ids = ids + [enc.eot_token] * (block_size - len(ids))
        tokenized.append(torch.tensor(ids, dtype=torch.long))
    return tokenized


def evaluate_subject(
    model: nn.Module,
    subject: str,
    samples: List[torch.Tensor],
    device: str,
    batch_size: int = 4,
) -> Dict:
    """
    Run inference on a subject and compute basic metrics.
    
    Returns dict with accuracy and loss (if applicable).
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            if len(batch) == 0:
                continue
            
            # Stack into batch
            x = torch.stack(batch).to(device)  # (B, T)
            
            # Targets are shifted by 1
            y = torch.cat([x[:, 1:], torch.full((x.size(0), 1), 0, device=device)], dim=1)
            
            try:
                logits, main_loss, aux_loss = model(x, y)
                total_loss += main_loss.item()
                num_batches += 1
            except Exception as e:
                print(f"Error during inference on {subject}: {e}")
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    # Perplexity
    ppl = np.exp(avg_loss) if avg_loss < 100 else float('inf')
    
    return {
        'subject': subject,
        'num_samples': len(samples),
        'avg_loss': avg_loss,
        'perplexity': ppl,
    }


def plot_activation_heatmap(
    matrix: np.ndarray,
    domains: List[str],
    layer_idx: int,
    out_path: str,
):
    """Generate Figure 2-style heatmap."""
    if not HAS_PLOT:
        print("Skipping plot: matplotlib not available.")
        return
    
    fig, ax = plt.subplots(figsize=(max(10, len(domains) * 0.6), max(6, matrix.shape[0] * 0.4)))
    
    # Normalize by column (domain) to show relative specialization
    matrix_norm = matrix / (matrix.sum(axis=0, keepdims=True) + 1e-8)
    
    sns.heatmap(
        matrix_norm,
        xticklabels=domains,
        yticklabels=[f"Expert {i}" for i in range(matrix.shape[0])],
        cmap='YlOrRd',
        cbar_kws={'label': 'Normalized Activation Frequency'},
        ax=ax,
        annot=False,
        fmt='.2f',
    )
    
    ax.set_title(f"Expert Activation by MMLU Subject (Layer {layer_idx})")
    ax.set_xlabel("MMLU Subject")
    ax.set_ylabel("Expert ID")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {out_path}")


def plot_activation_bars(
    matrix: np.ndarray,
    domains: List[str],
    layer_idx: int,
    out_path: str,
    colors: List[str] = None,
):
    """Generate grouped bar plot of expert activation percentages per subject.

    - X-axis: experts
    - Bars: subjects (domains)
    - Y-axis: percentage of activations per subject allocated to each expert
    """
    if not HAS_PLOT:
        print("Skipping plot: matplotlib not available.")
        return

    # Normalize counts per subject (column-wise) to get percentages
    col_sums = matrix.sum(axis=0, keepdims=True)
    pct = np.divide(matrix, np.where(col_sums == 0, 1.0, col_sums), where=True)

    n_experts = pct.shape[0]
    n_subjects = pct.shape[1]

    # Default color palette if not provided
    if colors is None:
        import seaborn as sns  # already guarded above
        palette = sns.color_palette("tab10", n_subjects)
        colors = [tuple(c) for c in palette]

    fig, ax = plt.subplots(figsize=(max(8, n_subjects * 1.2), 6))
    bar_width = max(0.08, 0.6 / max(n_subjects, 1))
    experts_idx = np.arange(n_experts)

    for s_idx, subject in enumerate(domains):
        offset = (s_idx - n_subjects / 2) * bar_width + bar_width / 2
        ax.bar(
            experts_idx + offset,
            pct[:, s_idx],
            width=bar_width,
            color=colors[s_idx % len(colors)],
            label=subject,
        )

    ax.set_title(f"Expert Activation Percentages per Subject (Layer {layer_idx})")
    ax.set_ylabel("Activation share (0-1)")
    ax.set_xlabel("Expert ID")
    ax.set_xticks(experts_idx)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=min(4, n_subjects), frameon=False)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved grouped bar plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MoE on MMLU and analyze routing")
    # Flexible input: either provide a checkpoint directly, or an experiment directory
    parser.add_argument('--checkpoint', type=str, required=False, help='Path to MoE checkpoint (e.g., out-<exp>/ckpt.pt)')
    parser.add_argument('--experiment_dir', type=str, required=False, help='Path to experiment directory containing ckpt.pt')
    parser.add_argument('--experiment_name', type=str, required=False, help='Optional experiment name to use under out/<name>/eval')
    parser.add_argument('--data_dir', type=str, default='data/mmlu', help='MMLU data directory')
    parser.add_argument('--out_dir', type=str, default='', help='Output directory; default auto-creates out/<experiment>/eval')
    parser.add_argument('--subjects', type=str, default=','.join(DEFAULT_SUBJECTS), help='Comma-separated MMLU subjects')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'], help='Data split to evaluate')
    parser.add_argument('--layer_to_plot', type=int, default=None, help='Which layer to plot (default: middle layer)')
    parser.add_argument('--layers_to_plot', type=str, default='', help='Comma-separated list of layers for grouped bar plots (e.g., "0,5,11")')
    
    args = parser.parse_args()
    
    # Resolve checkpoint path
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        if args.experiment_dir:
            # Try common checkpoint names inside the experiment directory
            candidates = [
                os.path.join(args.experiment_dir, 'ckpt.pt'),
                os.path.join(args.experiment_dir, 'ckpt_best.pt'),
            ]
            ckpt_path = next((p for p in candidates if os.path.exists(p)), None)
            if ckpt_path is None:
                # Search for any .pt file
                for root, _, files in os.walk(args.experiment_dir):
                    for f in files:
                        if f.endswith('.pt') and 'ckpt' in f:
                            ckpt_path = os.path.join(root, f)
                            break
                    if ckpt_path:
                        break
        if ckpt_path is None:
            raise SystemExit("Error: provide --checkpoint or --experiment_dir containing a checkpoint.")

    # Derive experiment name
    exp_name = args.experiment_name
    if not exp_name:
        # If user gave experiment_dir, use its basename; else use parent of checkpoint
        base_dir = args.experiment_dir or os.path.dirname(ckpt_path)
        exp_name = os.path.basename(os.path.normpath(base_dir)) or 'experiment'

    # Resolve output directory: default to out/<experiment>/eval
    out_dir = args.out_dir.strip()
    if not out_dir:
        out_dir = os.path.join('out', exp_name, 'eval')
    os.makedirs(out_dir, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=args.device)
    
    # Build model config (auto-adapt if MoE mismatch)
    model_args = checkpoint['model_args']
    original_n_expert = model_args.get('n_expert', 0)
    # Peek at state_dict keys to see if MoE experts are present
    sd_keys = checkpoint['model'].keys()
    has_moe_weights = any(k.startswith('transformer.h.0.mlp.experts.') for k in sd_keys)
    has_router_weight = any(k.endswith('.mlp.router.weight') for k in sd_keys)

    if original_n_expert > 0 and not has_moe_weights:
        print("[WARN] Checkpoint config says n_expert > 0 but expert weights not found. Falling back to n_expert=0 for evaluation.")
        model_args['n_expert'] = 0
        model_args['n_routed_expert'] = 1
    elif original_n_expert == 0 and has_moe_weights:
        print("[WARN] Checkpoint has MoE expert weights but config has n_expert=0. Adjusting config to match weights.")
        # Infer number of experts from first layer's expert weight keys
        inferred_experts = sorted({k.split('.')[4] for k in sd_keys if k.startswith('transformer.h.0.mlp.experts.')})
        model_args['n_expert'] = len(inferred_experts)
        # Attempt to infer routed expert top-k (cannot be exactly inferred; keep previous or default 2)
        if 'n_routed_expert' not in model_args or model_args['n_routed_expert'] <= 0:
            model_args['n_routed_expert'] = 2

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # Load state dict
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()
    
    print(f"Model loaded: {model_args['n_layer']} layers, {model_args['n_expert']} experts")
    
    # Setup tokenizer
    enc = tiktoken.get_encoding("gpt2")
    block_size = model_args['block_size']
    
    # Setup router hook only if MoE active
    n_layers = model_args['n_layer']
    n_experts = model_args['n_expert']
    router_hook = None
    if n_experts > 0:
        router_hook = RouterHook(model, n_layers, n_experts)
    else:
        print("[INFO] n_expert=0 -> running evaluation without routing heatmaps.")
    
    # Parse subjects
    subjects = [s.strip() for s in args.subjects.split(',') if s.strip()]
    
    # Evaluate each subject
    results = []
    print(f"\nEvaluating {len(subjects)} subjects on {args.split} split...")
    
    for subject in tqdm(subjects, desc="Subjects"):
        # Load data
        samples_text = load_mmlu_subject_from_bin(args.data_dir, subject, args.split)
        if not samples_text:
            print(f"No data for {subject}, skipping.")
            continue
        
        # Tokenize
        samples_tokens = tokenize_samples(samples_text, enc, block_size)
        
        # Set domain for router tracking (if MoE enabled)
        if router_hook is not None:
            router_hook.set_domain(subject)
        
        # Evaluate
        result = evaluate_subject(model, subject, samples_tokens, args.device, args.batch_size)
        results.append(result)
        
        print(f"  {subject}: loss={result['avg_loss']:.4f}, ppl={result['perplexity']:.2f}, samples={result['num_samples']}")
    
    # Save results
    results_path = os.path.join(out_dir, 'mmlu_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved evaluation results to {results_path}")
    
    # Generate activation heatmaps for each layer
    if router_hook is not None:
        domains = router_hook.get_all_domains()
        print(f"\nGenerating activation heatmaps for {n_layers} layers across {len(domains)} domains...")
        # Plot specified layer or middle+last (heatmaps)
        if args.layer_to_plot is not None:
            layers_for_heatmap = [args.layer_to_plot]
        else:
            layers_for_heatmap = [n_layers // 2, n_layers - 1]
        for layer_idx in layers_for_heatmap:
            if layer_idx >= n_layers:
                continue
            matrix = router_hook.get_activation_matrix(layer_idx, domains)
            plot_path = os.path.join(out_dir, f'expert_activation_layer{layer_idx}.png')
            plot_activation_heatmap(matrix, domains, layer_idx, plot_path)
            matrix_path = os.path.join(out_dir, f'activation_matrix_layer{layer_idx}.npy')
            np.save(matrix_path, matrix)
            print(f"Saved activation matrix to {matrix_path}")

        # Grouped bar plots for specific layers
        if args.layers_to_plot:
            # Support special keyword 'all' and filter invalid indices
            raw = args.layers_to_plot.strip().lower()
            if raw == 'all':
                bar_layers = list(range(n_layers))
            else:
                try:
                    bar_layers = [int(x.strip()) for x in raw.split(',') if x.strip()]
                except ValueError:
                    bar_layers = []
                    print("[WARN] Invalid --layers_to_plot format; expected comma-separated integers or 'all'.")
            for layer_idx in bar_layers:
                if layer_idx >= n_layers:
                    print(f"[WARN] Requested bar plot for layer {layer_idx} which exceeds n_layers={n_layers}.")
                    continue
                matrix = router_hook.get_activation_matrix(layer_idx, domains)
                bar_path = os.path.join(out_dir, f'expert_activation_bars_layer{layer_idx}.png')
                plot_activation_bars(matrix, domains, layer_idx, bar_path)
        # Serialize routing counts safely (no lambdas)
        def serialize_counts(counts):
            out = {}
            for layer_idx, experts in counts.items():
                layer_out = {}
                for exp_idx, domains_map in experts.items():
                    layer_out[exp_idx] = dict(domains_map)
                out[layer_idx] = layer_out
            return out
        routing_path = os.path.join(out_dir, 'routing_counts.pkl')
        with open(routing_path, 'wb') as f:
            pickle.dump(serialize_counts(router_hook.expert_counts), f)
        print(f"Saved routing counts to {routing_path}")
    
    # Cleanup
    if router_hook is not None:
        router_hook.remove_hooks()
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for res in results:
        print(f"{res['subject']:30s} | Loss: {res['avg_loss']:6.4f} | PPL: {res['perplexity']:8.2f} | Samples: {res['num_samples']:4d}")
    
    avg_loss = np.mean([r['avg_loss'] for r in results])
    avg_ppl = np.mean([r['perplexity'] for r in results if r['perplexity'] < 1e6])
    print("="*60)
    print(f"{'Average':30s} | Loss: {avg_loss:6.4f} | PPL: {avg_ppl:8.2f}")
    print("="*60)
    
    print(f"\nAll results saved to {out_dir}/")
    print("Done!")


if __name__ == '__main__':
    main()
