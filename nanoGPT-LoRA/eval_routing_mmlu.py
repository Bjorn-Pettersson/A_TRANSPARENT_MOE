"""Evaluate routing specialization with question files (one question per line).

This version assumes the MMLU question files already exist (no auto-download).
It produces:
1. Per-subject heatmaps (layer x expert) identical to original behavior.
2. Staple-style grouped bar plots (expert activation share per subject) like eval_mmlu.py.
3. Per-layer CSV matrices with subject rows and expert activation shares.

Example:
    python eval_routing_mmlu.py --ckpt out/out-step1-benchmark-moe/ckpt.pt \
        --questions_dir data/mmlu_questions \
        --categories college_chemistry global_facts management medical_genetics \
        --samples_per_category 100 --device cuda --out_dir out/out-step1-benchmark-moe/routing_analysis \
        --layers_to_plot all
"""
import os
import argparse
import json
from collections import defaultdict

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import GPTConfig, GPT

try:
    from transformers import GPT2TokenizerFast
except Exception:
    GPT2TokenizerFast = None


def load_questions_for_category(questions_dir, category):
    path = os.path.join(questions_dir, f"{category}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Questions file not found for category '{category}': {path}")
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


def plot_heatmap(freqs, outpath, title):
    # freqs: (n_layer, n_expert)
    plt.figure(figsize=(max(6, freqs.shape[1]), max(6, freqs.shape[0]/2)))
    plt.imshow(freqs, aspect='auto', cmap='viridis')
    plt.colorbar(label='frequency')
    plt.xlabel('expert')
    plt.ylabel('layer')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_activation_bars(matrix, subjects, layer_idx, out_path):
    """Grouped bar plot: experts on x-axis, bars per subject showing activation share."""
    n_experts = matrix.shape[0]
    n_subjects = matrix.shape[1]
    # normalize per subject (column-wise)
    col_sums = matrix.sum(axis=0, keepdims=True)
    pct = matrix / (col_sums + 1e-8)
    plt.figure(figsize=(max(8, n_subjects * 1.2), 6))
    bar_width = max(0.08, 0.6 / max(n_subjects, 1))
    experts_idx = np.arange(n_experts)
    # colors
    try:
        import seaborn as sns
        palette = sns.color_palette("tab10", n_subjects)
    except Exception:
        palette = [plt.cm.tab10(i % 10) for i in range(n_subjects)]
    for s_idx, subject in enumerate(subjects):
        offset = (s_idx - n_subjects / 2) * bar_width + bar_width / 2
        plt.bar(experts_idx + offset, pct[:, s_idx], width=bar_width, color=palette[s_idx], label=subject)
    plt.title(f"Expert Activation Percentages per Subject (Layer {layer_idx})")
    plt.ylabel("Activation share (0-1)")
    plt.xlabel("Expert ID")
    plt.xticks(experts_idx)
    plt.ylim(0, 1.0)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=min(4, n_subjects), frameon=False)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='out-moe-test/ckpt.pt')
    parser.add_argument('--questions_dir', type=str, required=True,
                        help='Directory containing one file per category named <category>.txt')
    parser.add_argument('--categories', nargs='+', required=True,
                        help='List of subjects/categories (filenames without extension) to evaluate')
    parser.add_argument('--samples_per_category', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out_dir', type=str, default='out-moe-test/routing_analysis')
    parser.add_argument('--max_tokens', type=int, default=None,
                        help='If set, crop/pad tokenized inputs to this length (default = model block_size)')
    parser.add_argument('--layers_to_plot', type=str, default='all',
                        help='Comma-separated layer indices for grouped bar plots or "all"')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load checkpoint
    print('loading checkpoint', args.ckpt)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model_args = ckpt.get('model_args', None)
    if model_args is None:
        raise KeyError('checkpoint does not contain model_args; cannot reconstruct model')

    cfg = GPTConfig(**model_args)
    model = GPT(cfg)
    state_dict = ckpt['model']
    # load
    model.load_state_dict(state_dict)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # tokenizer
    if GPT2TokenizerFast is None:
        raise RuntimeError('transformers not installed; please pip install transformers')
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    block_size = args.max_tokens if args.max_tokens is not None else cfg.block_size
    n_layer = cfg.n_layer
    n_expert = cfg.n_expert
    top_k = cfg.n_routed_expert
    print(f'model: layers={n_layer}, experts={n_expert}, top_k={top_k}, block_size={block_size}')

    summary = {}
    # ensure questions_dir exists (no auto-download)
    if not os.path.isdir(args.questions_dir):
        raise SystemExit(f"questions_dir does not exist: {args.questions_dir}")
    missing = [cat for cat in args.categories if not os.path.exists(os.path.join(args.questions_dir, f"{cat}.txt"))]
    if missing:
        raise SystemExit(f"Missing required subject files: {missing}")
    # Storage for staple-style aggregation: per layer per expert per subject
    # subject -> layer -> expert counts
    aggregate_counts = {cat: [np.zeros((n_expert,), dtype=np.int64) for _ in range(n_layer)] for cat in args.categories}

    for cat in args.categories:
        print('processing category:', cat)
        questions = load_questions_for_category(args.questions_dir, cat)
        # limit
        questions = questions[:args.samples_per_category]

        # counts per layer x expert
        counts = np.zeros((n_layer, n_expert), dtype=np.int64)
        per_sample_sel = []

        for q in questions:
            toks = tokenizer.encode(q, add_special_tokens=False)
            if len(toks) == 0:
                toks = [tokenizer.eos_token_id]
            t = min(len(toks), block_size)
            toks = toks[:t]
            idx = torch.tensor([toks], dtype=torch.long, device=device)
            with torch.no_grad():
                # call model; returns logits, None/ loss, aux
                _logits, _loss, _aux = model(idx)
            # after forward, the model blocks will have `last_selected_experts` when SequenceMoE present
            sample_sel = []
            for li, block in enumerate(model.transformer.h):
                mlp = getattr(block, 'mlp', None)
                if mlp is not None and hasattr(mlp, 'last_selected_experts') and mlp.last_selected_experts is not None:
                    # last_selected_experts shape: (B, top_k)
                    sel = mlp.last_selected_experts
                    # ensure on cpu
                    if isinstance(sel, torch.Tensor):
                        sel = sel.cpu().numpy()
                    else:
                        sel = np.array(sel)
                    # take first batch element
                    chosen = sel[0].tolist() if sel.ndim == 2 else [int(sel[0])]
                    sample_sel.append(chosen)
                    for e in chosen:
                        e_int = int(e)
                        counts[li, e_int] += 1
                        aggregate_counts[cat][li][e_int] += 1
                else:
                    sample_sel.append(None)
            per_sample_sel.append(sample_sel)

        # normalize frequencies per layer (sum over assigned counts -> divided by number of samples * top_k)
        total_assignments = max(1, len(questions) * top_k)
        freqs = counts.astype(np.float32) / float(len(questions) * top_k)

        # save csv
        csv_path = os.path.join(args.out_dir, f'routing_freq_{cat}.csv')
        with open(csv_path, 'w') as f:
            # header
            f.write('layer,' + ','.join([f'expert_{i}' for i in range(n_expert)]) + '\n')
            for li in range(n_layer):
                row = ','.join([f'{freqs[li, ei]:.6f}' for ei in range(n_expert)])
                f.write(f'{li},{row}\n')

        # plot heatmap
        png_path = os.path.join(args.out_dir, f'routing_heatmap_{cat}.png')
        plot_heatmap(freqs, png_path, title=f'Routing frequencies: {cat}')

        # save per-sample selections
        json_path = os.path.join(args.out_dir, f'per_sample_selected_{cat}.json')
        with open(json_path, 'w') as f:
            json.dump({'categories': cat, 'samples': len(questions), 'per_sample_selected': per_sample_sel}, f)

        summary[cat] = {'csv': csv_path, 'png': png_path, 'json': json_path, 'freqs': freqs.tolist()}

    # Staple-style grouped bar plots & per-layer CSV across subjects
    layer_list_raw = args.layers_to_plot.strip().lower()
    if layer_list_raw == 'all':
        target_layers = list(range(n_layer))
    else:
        try:
            target_layers = [int(x.strip()) for x in layer_list_raw.split(',') if x.strip()]
        except ValueError:
            print('[WARN] Invalid --layers_to_plot format; skipping grouped bar plots.')
            target_layers = []

    subjects = args.categories
    if target_layers:
        print(f"Generating grouped bar plots for layers: {target_layers}")
    for li in target_layers:
        if li < 0 or li >= n_layer:
            print(f"[WARN] Layer index {li} out of range; skipping")
            continue
        # Build matrix (experts x subjects)
        mat = np.zeros((n_expert, len(subjects)), dtype=np.float64)
        for s_idx, subj in enumerate(subjects):
            counts_vec = aggregate_counts[subj][li]  # (n_expert,)
            mat[:, s_idx] = counts_vec
        # Save raw counts matrix
        raw_path = os.path.join(args.out_dir, f'aggregate_counts_layer{li}.npy')
        np.save(raw_path, mat)
        # Save CSV (subject rows, expert columns normalized per subject)
        col_sums = mat.sum(axis=0, keepdims=True) + 1e-8
        norm_mat = mat / col_sums
        csv_path = os.path.join(args.out_dir, f'grouped_activation_layer{li}.csv')
        with open(csv_path, 'w') as f:
            f.write('subject,' + ','.join([f'expert_{e}' for e in range(n_expert)]) + '\n')
            for s_idx, subj in enumerate(subjects):
                row = ','.join([f"{norm_mat[e, s_idx]:.6f}" for e in range(n_expert)])
                f.write(f"{subj},{row}\n")
        # Plot
        bar_path = os.path.join(args.out_dir, f'expert_activation_bars_layer{li}.png')
        plot_activation_bars(mat, subjects, li, bar_path)
        print(f"Saved grouped bar plot and CSV for layer {li}")

    # write summary
    with open(os.path.join(args.out_dir, 'summary_routing.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('done. results in', args.out_dir)


if __name__ == '__main__':
    main()
