"""Unified evaluation script for MoE routing with text-based MMLU questions.

Features (for both benchmark and pretrained models):
- Loads a GPT MoE checkpoint and runs inference on per-subject question files
- Produces per-subject heatmaps (layer x expert) and CSVs
- Produces staple-style grouped bar plots per layer across subjects
- Writes outputs to a chosen directory (recommend: out/<experiment>/eval)

Usage examples (PowerShell):
  # Benchmark model
  python eval.py --ckpt out/out-step1-benchmark-moe/ckpt.pt \
    --questions_dir data/mmlu_questions \
    --subjects college_chemistry,global_facts,management,medical_genetics \
    --samples_per_subject 100 --device cuda --out_dir out/out-step1-benchmark-moe/eval \
    --layers_to_plot all

  # Pretrained experts model
  python eval.py --ckpt out-pretrain/ckpt.pt \
    --questions_dir data/mmlu_questions \
    --subjects college_chemistry,global_facts,management,medical_genetics \
    --samples_per_subject 100 --device cuda --out_dir out-pretrain/eval \
    --layers_to_plot all
"""
import os
import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import GPTConfig, GPT

try:
    from transformers import GPT2TokenizerFast
except Exception:
    GPT2TokenizerFast = None


def load_questions_for_subject(questions_dir: str, subject: str):
    path = os.path.join(questions_dir, f"{subject}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing subject file: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines


def plot_heatmap(freqs: np.ndarray, outpath: str, title: str):
    plt.figure(figsize=(max(6, freqs.shape[1]), max(6, freqs.shape[0]/2)))
    plt.imshow(freqs, aspect='auto', cmap='viridis')
    plt.colorbar(label='frequency')
    plt.xlabel('expert')
    plt.ylabel('layer')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_activation_bars(matrix: np.ndarray, subjects: list[str], layer_idx: int, out_path: str):
    n_experts = matrix.shape[0]
    n_subjects = matrix.shape[1]
    col_sums = matrix.sum(axis=0, keepdims=True) + 1e-8
    pct = matrix / col_sums
    plt.figure(figsize=(max(8, n_subjects * 1.2), 6))
    bar_width = max(0.08, 0.6 / max(n_subjects, 1))
    experts_idx = np.arange(n_experts)
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
    ap = argparse.ArgumentParser(description="Unified MoE routing evaluation with text questions")
    ap.add_argument('--ckpt', required=True, help='Path to checkpoint (e.g., out/.../ckpt.pt)')
    ap.add_argument('--questions_dir', required=True, help='Directory with <subject>.txt files')
    ap.add_argument('--subjects', required=True, help='Comma-separated list of subjects to evaluate')
    ap.add_argument('--samples_per_subject', type=int, default=200)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out_dir', default='eval_results/routing')
    ap.add_argument('--max_tokens', type=int, default=None)
    ap.add_argument('--layers_to_plot', type=str, default='all', help='Comma list or "all" for grouped bar plots')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if GPT2TokenizerFast is None:
        raise RuntimeError('transformers not installed; please pip install transformers')
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    print('Loading checkpoint', args.ckpt)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    model_args = ckpt.get('model_args', None)
    if model_args is None:
        raise KeyError('checkpoint missing model_args')
    cfg = GPTConfig(**model_args)
    model = GPT(cfg)
    state_dict = ckpt['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    n_layer = cfg.n_layer
    n_expert = cfg.n_expert
    top_k = cfg.n_routed_expert
    block_size = args.max_tokens if args.max_tokens is not None else cfg.block_size
    print(f"model: layers={n_layer}, experts={n_expert}, top_k={top_k}, block_size={block_size}")

    # Validate inputs
    if not os.path.isdir(args.questions_dir):
        raise SystemExit(f"questions_dir does not exist: {args.questions_dir}")
    subjects = [s.strip() for s in args.subjects.split(',') if s.strip()]
    missing = [s for s in subjects if not os.path.exists(os.path.join(args.questions_dir, f"{s}.txt"))]
    if missing:
        raise SystemExit(f"Missing subject files: {missing}")

    # Aggregate for staple plots: subject -> [layer vectors]
    aggregate_counts = {s: [np.zeros((n_expert,), dtype=np.int64) for _ in range(n_layer)] for s in subjects}
    summary = {}

    # Per-subject evaluation
    for subj in subjects:
        print('Subject:', subj)
        questions = load_questions_for_subject(args.questions_dir, subj)
        questions = questions[:args.samples_per_subject]
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
                _logits, _loss, _aux = model(idx)
            sample_sel = []
            for li, block in enumerate(model.transformer.h):
                mlp = getattr(block, 'mlp', None)
                if mlp is not None and hasattr(mlp, 'last_selected_experts') and mlp.last_selected_experts is not None:
                    sel = mlp.last_selected_experts
                    if isinstance(sel, torch.Tensor):
                        sel = sel.cpu().numpy()
                    else:
                        sel = np.array(sel)
                    chosen = sel[0].tolist() if sel.ndim == 2 else [int(sel[0])]
                    sample_sel.append(chosen)
                    for e in chosen:
                        e_int = int(e)
                        counts[li, e_int] += 1
                        aggregate_counts[subj][li][e_int] += 1
                else:
                    sample_sel.append(None)
            per_sample_sel.append(sample_sel)

        # Normalize per layer for heatmap
        freqs = counts.astype(np.float32) / float(max(1, len(questions) * max(1, top_k)))
        # Save per-subject CSV/PNG/JSON
        csv_path = os.path.join(args.out_dir, f'routing_freq_{subj}.csv')
        with open(csv_path, 'w') as f:
            f.write('layer,' + ','.join([f'expert_{i}' for i in range(n_expert)]) + '\n')
            for li in range(n_layer):
                row = ','.join([f'{freqs[li, ei]:.6f}' for ei in range(n_expert)])
                f.write(f'{li},{row}\n')
        png_path = os.path.join(args.out_dir, f'routing_heatmap_{subj}.png')
        plot_heatmap(freqs, png_path, title=f'Routing frequencies: {subj}')
        json_path = os.path.join(args.out_dir, f'per_sample_selected_{subj}.json')
        with open(json_path, 'w') as f:
            json.dump({'subject': subj, 'samples': len(questions), 'per_sample_selected': per_sample_sel}, f)
        summary[subj] = {'csv': csv_path, 'png': png_path, 'json': json_path}

    # Staple-style grouped bar plots across subjects
    raw = args.layers_to_plot.strip().lower()
    if raw == 'all':
        target_layers = list(range(n_layer))
    else:
        try:
            target_layers = [int(x.strip()) for x in raw.split(',') if x.strip()]
        except ValueError:
            print('[WARN] Invalid --layers_to_plot; skipping grouped bar plots.')
            target_layers = []

    for li in target_layers:
        if li < 0 or li >= n_layer:
            print(f"[WARN] Layer {li} out of range; skipping")
            continue
        mat = np.zeros((n_expert, len(subjects)), dtype=np.float64)
        for s_idx, subj in enumerate(subjects):
            mat[:, s_idx] = aggregate_counts[subj][li]
        np.save(os.path.join(args.out_dir, f'aggregate_counts_layer{li}.npy'), mat)
        # normalized CSV (subjects as rows)
        col_sums = mat.sum(axis=0, keepdims=True) + 1e-8
        norm_mat = mat / col_sums
        csv_path = os.path.join(args.out_dir, f'grouped_activation_layer{li}.csv')
        with open(csv_path, 'w') as f:
            f.write('subject,' + ','.join([f'expert_{e}' for e in range(n_expert)]) + '\n')
            for s_idx, subj in enumerate(subjects):
                row = ','.join([f"{norm_mat[e, s_idx]:.6f}" for e in range(n_expert)])
                f.write(f"{subj},{row}\n")
        bar_path = os.path.join(args.out_dir, f'expert_activation_bars_layer{li}.png')
        plot_activation_bars(mat, subjects, li, bar_path)
        print(f"Saved grouped bar plot and CSV for layer {li}")

    with open(os.path.join(args.out_dir, 'summary_routing.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('Done. Results in', args.out_dir)


if __name__ == '__main__':
    main()
