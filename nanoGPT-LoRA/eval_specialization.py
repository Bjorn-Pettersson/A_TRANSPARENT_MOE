"""
Evaluate expert specialization via leave-one-expert-out ablations on subject-specific validation bins.

Outputs:
- loss_deltas.csv: rows=subjects, cols=experts; delta loss when masking expert e
- routing_freq_layer<L>.csv: rows=subjects, cols=experts; normalized routing frequencies
- routing_purity.csv: rows=subjects, cols=layers; max expert share per layer

Usage (PowerShell):
  python eval\eval_specialization.py --ckpt out-pretrain\ckpt.pt \
    --data_dir data\mmlu --subjects college_chemistry,global_facts,management,medical_genetics \
    --steps_per_subject 100 --batch_size 4 --device cuda --out_dir eval_results\specialization
"""

import os
import sys
import argparse
from typing import List, Dict

import numpy as np
import torch
from torch.nn import functional as F

# Allow running from the eval/ subfolder without modifying PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import GPT, GPTConfig


def try_load_memmap(path):
    try:
        return np.memmap(path, dtype=np.uint16, mode='r')
    except Exception:
        return None


def load_subject_val(data_dir: str, subject: str):
    val_path = os.path.join(data_dir, f"{subject}_val.bin")
    data = try_load_memmap(val_path)
    if data is None:
        raise FileNotFoundError(f"Missing or unreadable val bin: {val_path}")
    return data


def get_batch_from_memmap(data: np.memmap, block_size: int, batch_size: int, device: str):
    assert len(data) > block_size + 1, "val.bin too small for given block_size"
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device.startswith('cuda'):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def set_global_expert_mask(model: GPT, mask: torch.Tensor | None):
    """Set or clear inference_expert_mask on all MoE layers."""
    for block in model.transformer.h:
        mlp = getattr(block, 'mlp', None)
        if mlp is None:
            continue
        if hasattr(mlp, 'experts') and hasattr(mlp, 'router'):
            if mask is None:
                if hasattr(mlp, 'inference_expert_mask'):
                    delattr(mlp, 'inference_expert_mask')
            else:
                setattr(mlp, 'inference_expert_mask', mask)


@torch.no_grad()
def eval_loss_and_routing(model: GPT, val_data: np.memmap, steps: int, block_size: int, batch_size: int, device: str, eps: float = 1e-12):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    # Routing counts per layer per expert: Dict[layer, np.ndarray(n_expert,)]
    routing_counts: Dict[int, np.ndarray] = {}
    n_expert = int(getattr(model.config, 'n_expert', 0))
    n_layer = model.config.n_layer
    for li in range(n_layer):
        routing_counts[li] = np.zeros((n_expert,), dtype=np.float64) if n_expert > 0 else None

    for _ in range(steps):
        X, Y = get_batch_from_memmap(val_data, block_size, batch_size, device)
        logits, main_loss, _aux = model(X, Y)
        total_loss += float(main_loss.item())
        n_batches += 1

        # accumulate routing
        if n_expert > 0:
            for li, block in enumerate(model.transformer.h):
                mlp = getattr(block, 'mlp', None)
                if mlp is None or not hasattr(mlp, 'last_selected_experts'):
                    continue
                sel = getattr(mlp, 'last_selected_experts', None)
                w = getattr(mlp, 'last_topk_probs', None)
                if sel is None or w is None:
                    continue
                sel = torch.as_tensor(sel)
                w = torch.as_tensor(w)
                # count only positive-weight assignments
                pos = (w > eps)
                # flatten indices and weights accordingly
                sel_flat = sel[pos].reshape(-1)
                w_flat = w[pos].reshape(-1).to(torch.float32)
                if sel_flat.numel() == 0:
                    continue
                # bincount with weights
                counts = torch.bincount(sel_flat.to(torch.int64), weights=w_flat, minlength=n_expert).cpu().numpy()
                routing_counts[li] += counts

    avg_loss = total_loss / max(n_batches, 1)
    # normalize routing counts to frequencies per layer
    routing_freqs: Dict[int, np.ndarray] = {}
    if n_expert > 0:
        for li in range(n_layer):
            counts = routing_counts[li]
            if counts is None:
                continue
            s = counts.sum()
            routing_freqs[li] = counts / s if s > 0 else counts
    return avg_loss, routing_freqs


def main():
    ap = argparse.ArgumentParser(description="Evaluate expert specialization via ablations")
    ap.add_argument('--ckpt', required=True, help='Path to checkpoint (ckpt.pt)')
    ap.add_argument('--data_dir', default='data/mmlu', help='Directory with <subject>_val.bin files')
    ap.add_argument('--subjects', required=True, help='Comma-separated list of subjects')
    ap.add_argument('--steps_per_subject', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out_dir', default='eval_results/specialization')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = torch.load(args.ckpt, map_location=args.device)
    model_args = ckpt['model_args']

    # Sanity: ensure MoE shape
    n_expert = model_args.get('n_expert', 0)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    # Load state dict (strip unwanted prefix if any)
    sd = ckpt['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(sd.items()):
        if k.startswith(unwanted_prefix):
            sd[k[len(unwanted_prefix):]] = sd.pop(k)
    model.load_state_dict(sd)
    model.to(args.device)
    model.eval()

    block_size = model_args['block_size']
    subjects: List[str] = [s.strip() for s in args.subjects.split(',') if s.strip()]

    # Baseline (no mask) and leave-one-expert-out deltas
    loss_rows = {}
    purity_rows = {}

    for subj in subjects:
        val_data = load_subject_val(args.data_dir, subj)

        # baseline (no masking)
        set_global_expert_mask(model, None)
        base_loss, routing_freqs = eval_loss_and_routing(model, val_data, args.steps_per_subject, block_size, args.batch_size, args.device)

        row = []
        # collect layer-wise purity (max share per layer)
        layer_purity = []
        if n_expert > 0 and routing_freqs:
            for li in sorted(routing_freqs.keys()):
                freqs = routing_freqs[li]
                layer_purity.append(float(freqs.max() if freqs.size > 0 else 0.0))
        purity_rows[subj] = layer_purity

        if n_expert > 0:
            for ei in range(n_expert):
                mask = torch.ones(n_expert, dtype=torch.float32, device=args.device)
                mask[ei] = 0.0
                set_global_expert_mask(model, mask)
                masked_loss, _ = eval_loss_and_routing(model, val_data, args.steps_per_subject, block_size, args.batch_size, args.device)
                row.append(masked_loss - base_loss)
        loss_rows[subj] = row

    # Write CSVs
    # loss_deltas.csv
    if n_expert > 0:
        loss_path = os.path.join(args.out_dir, 'loss_deltas.csv')
        with open(loss_path, 'w') as f:
            f.write('subject,' + ','.join([f'expert_{i}' for i in range(n_expert)]) + '\n')
            for subj in subjects:
                vals = loss_rows.get(subj, [])
                vals_str = ','.join([f"{v:.6f}" for v in vals]) if vals else ','.join(['']*n_expert)
                f.write(f"{subj},{vals_str}\n")

    # routing_purity.csv
    if purity_rows:
        # determine max layers observed
        max_layers = max((len(v) for v in purity_rows.values()), default=0)
        pur_path = os.path.join(args.out_dir, 'routing_purity.csv')
        with open(pur_path, 'w') as f:
            f.write('subject,' + ','.join([f'layer_{i}' for i in range(max_layers)]) + '\n')
            for subj in subjects:
                vals = purity_rows.get(subj, [])
                # pad to max_layers
                vals = vals + [0.0] * max(0, max_layers - len(vals))
                f.write(f"{subj}," + ','.join([f"{v:.6f}" for v in vals]) + "\n")

    # also dump per-layer frequency tables
    if n_expert > 0 and subjects:
        # recompute once per subject last routing freqs to write per-layer CSVs
        for subj in subjects:
            val_data = load_subject_val(args.data_dir, subj)
            set_global_expert_mask(model, None)
            _loss, routing_freqs = eval_loss_and_routing(model, val_data, args.steps_per_subject, block_size, args.batch_size, args.device)
            for li, freqs in routing_freqs.items():
                outp = os.path.join(args.out_dir, f"routing_freq_layer{li}.csv")
                header_written = os.path.exists(outp)
                mode = 'a' if header_written else 'w'
                with open(outp, mode) as f:
                    if not header_written:
                        f.write('subject,' + ','.join([f'expert_{i}' for i in range(n_expert)]) + '\n')
                    f.write(subj + ',' + ','.join([f"{x:.6f}" for x in freqs]) + '\n')

    print(f"Wrote specialization eval to {args.out_dir}")


if __name__ == '__main__':
    main()
