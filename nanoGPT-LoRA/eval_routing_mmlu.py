"""
Evaluate a trained Sequence-MoE GPT checkpoint on MMLU subject slices
to collect per-layer expert assignment frequencies per subject.

Usage examples:
  python eval_routing_mmlu.py --ckpt out-moe-test/ckpt.pt \
    --questions_dir data/mmlu_questions --categories global_facts abstract_algebra medical_genetics management college_biology college_chemistry \
    --samples_per_category 200 --device cuda --out_dir out-moe-test/routing_analysis

The script expects one text file per category in `--questions_dir`, named
`<category>.txt` with one question per line. It will produce CSV and PNG
heatmaps per category showing layer x expert frequencies.
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
        raise FileNotFoundError(f"Questions file not found for category '{category}': {path}\nPlease prepare a text file with one question per line.")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='out-moe-test/ckpt.pt')
    parser.add_argument('--questions_dir', type=str, required=True,
                        help='Directory containing one file per category named <category>.txt')
    parser.add_argument('--categories', nargs='+', required=True,
                        help='List of categories (filenames without extension) to evaluate')
    parser.add_argument('--samples_per_category', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--out_dir', type=str, default='out-moe-test/routing_analysis')
    parser.add_argument('--max_tokens', type=int, default=None,
                        help='If set, crop/pad tokenized inputs to this length (default = model block_size)')
    parser.add_argument('--use_hf', action='store_true',
                        help='If set and category files are missing, try to download MMLU via `datasets` to create them')
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
    # ensure questions_dir exists
    os.makedirs(args.questions_dir, exist_ok=True)
    # If files are missing and user requested, try to populate from HuggingFace datasets
    missing = [cat for cat in args.categories if not os.path.exists(os.path.join(args.questions_dir, f"{cat}.txt"))]
    if missing:
        print('Missing category files for:', missing)
        if args.use_hf:
            try:
                from datasets import load_dataset
            except Exception:
                print('`datasets` not installed. Install with `pip install datasets` or provide local question files.')
                missing = missing
            else:
                # try common dataset ids for MMLU
                tried = False
                for ds_id in ['hendrycks/mmlu', 'mmlu']:
                    try:
                        print('Trying to load MMLU dataset id:', ds_id)
                        ds = load_dataset(ds_id)
                        tried = True
                        break
                    except Exception:
                        ds = None
                if not tried or ds is None:
                    print('Could not fetch MMLU from common dataset ids. Please provide local files in', args.questions_dir)
                else:
                    # ds may have multiple splits; flatten all examples
                    examples = []
                    for split in ds.keys():
                        for ex in ds[split]:
                            examples.append(ex)
                    # heuristically find a text field to use as question
                    sample_keys = set().union(*(list(e.keys()) for e in examples[:10]))
                    text_key = None
                    for k in ('question', 'input', 'query', 'prompt', 'question_text'):
                        if k in sample_keys:
                            text_key = k
                            break
                    if text_key is None:
                        # fallback to first string field
                        for k in sample_keys:
                            if isinstance(examples[0].get(k), str):
                                text_key = k
                                break
                    if text_key is None:
                        print('Could not locate a text field in the downloaded dataset; aborting auto-create.')
                    else:
                        # group examples by subject if available
                        subject_key = None
                        for k in ('subject', 'task', 'category'):
                            if k in sample_keys:
                                subject_key = k
                                break
                        grouped = {}
                        if subject_key is not None:
                            for ex in examples:
                                subj = ex.get(subject_key, 'unknown')
                                grouped.setdefault(subj, []).append(ex.get(text_key, ''))
                        else:
                            # if no subject, just dump examples into a single generic file per missing category
                            grouped = {'auto': [ex.get(text_key, '') for ex in examples]}

                        # write files for the missing categories using best-effort mapping by name
                        for cat in missing:
                            out_path = os.path.join(args.questions_dir, f"{cat}.txt")
                            written = 0
                            # try exact match in subjects
                            if cat in grouped:
                                items = grouped[cat]
                            else:
                                # pick from 'auto' or sample pool
                                items = grouped.get('auto', [])
                            with open(out_path, 'w', encoding='utf-8') as f:
                                for q in items[:args.samples_per_category]:
                                    if not q:
                                        continue
                                    f.write(q.replace('\n', ' ') + '\n')
                                    written += 1
                            print(f'Wrote {written} questions to {out_path}')
        else:
            print('Missing category files and --use_hf not set. Please create the files in', args.questions_dir)
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
                        counts[li, int(e)] += 1
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

    # write summary
    with open(os.path.join(args.out_dir, 'summary_routing.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('done. results in', args.out_dir)


if __name__ == '__main__':
    main()
