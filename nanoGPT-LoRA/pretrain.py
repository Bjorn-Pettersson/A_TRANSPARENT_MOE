"""
Pretrain MoE experts per subject before general OpenWebText training.

This script sequentially pretrains a GPT with Sequence-level MoE experts on
subject-specific corpora located under data/mmlu as token ID .bin files or
pickled strings. For each subject, it runs a focused training loop and saves
checkpoints that can later be resumed from `train.py` (init_from='resume').

Expected data layout (per subject):
  data/mmlu/
    college_chemistry_train.bin
    college_chemistry_val.bin
    global_facts_train.bin
    global_facts_val.bin
    management_train.bin
    management_val.bin
    medical_genetics_train.bin
    medical_genetics_val.bin

The .bin files are expected to be GPT-2 token ids stored as np.uint16 mmap, as
produced by the project's prepare scripts. If a file is instead a pickled
string (legacy pipeline), we will decode and tokenize it on the fly.
"""

import os
import math
import time
import pickle
import traceback
import argparse
import inspect
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn

try:
    import tiktoken
    _enc = tiktoken.get_encoding("gpt2")
except Exception:
    _enc = None

from model import GPTConfig, GPT


# ----------------------------- Utilities -----------------------------

def try_load_memmap(path):
    try:
        return np.memmap(path, dtype=np.uint16, mode='r')
    except Exception:
        return None


def try_load_pickled_string_and_tokenize(path):
    if _enc is None:
        raise RuntimeError("tiktoken not available to tokenize pickled text bin")
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if not isinstance(data, str):
            raise ValueError("Pickled content is not a string; unsupported format")
        ids = _enc.encode_ordinary(data)
        # append EOT to ensure consistent boundaries
        ids.append(_enc.eot_token)
        return np.array(ids, dtype=np.uint16)
    except Exception as e:
        raise RuntimeError(f"Failed to load/tokenize pickled string from {path}: {e}")


class SubjectDataset:
    def __init__(self, data_dir, prefix):
        self.prefix = prefix
        self.train_path = os.path.join(data_dir, f"{prefix}_train.bin")
        self.val_path = os.path.join(data_dir, f"{prefix}_val.bin")

        # Try memmap first
        self.train_data = try_load_memmap(self.train_path)
        self.val_data = try_load_memmap(self.val_path)

        # Fallback to pickle+tokenize
        if self.train_data is None:
            self.train_data = try_load_pickled_string_and_tokenize(self.train_path)
        if self.val_data is None:
            self.val_data = try_load_pickled_string_and_tokenize(self.val_path)

        # Infer total token counts
        self.n_train = len(self.train_data)
        self.n_val = len(self.val_data)

    def get_batch(self, split, block_size, batch_size, device, pin_memory=True):
        data = self.train_data if split == 'train' else self.val_data
        # Ensure enough tokens
        if len(data) <= block_size + 1:
            raise ValueError(f"Dataset '{self.prefix}' too small for block_size {block_size}")
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        if isinstance(data, np.memmap):
            x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        else:
            # numpy array in memory
            x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])

        if device.startswith('cuda'):
            if pin_memory:
                x = x.pin_memory().to(device, non_blocking=True)
                y = y.pin_memory().to(device, non_blocking=True)
            else:
                x, y = x.to(device), y.to(device)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


# -------------------------- Pretraining Logic -------------------------

def main():
    # First, create a lightweight parser to capture optional config file
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--config', type=str, default=None, help='Optional Python config file to preset arguments')
    boot_known, boot_unknown = bootstrap.parse_known_args()

    parser = argparse.ArgumentParser(description="Pretrain MoE experts per subject", parents=[bootstrap])
    # I/O and training control
    parser.add_argument('--data_dir', type=str, default=os.path.join('data', 'mmlu'))
    parser.add_argument('--out_dir', type=str, default='out')
    parser.add_argument('--subjects', type=str, default='college_chemistry,global_facts,management,medical_genetics')
    parser.add_argument('--iters_per_subject', type=int, default=2000)
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--eval_iters', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--init_from', type=str, default='scratch', choices=['scratch','resume','gpt2','gpt2-medium','gpt2-large','gpt2-xl'])

    # Data/Model hyperparams
    parser.add_argument('--block_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--ffn_mult', type=float, default=4.0)
    parser.add_argument('--n_expert', type=int, default=4)
    parser.add_argument('--n_routed_expert', type=int, default=2)
    parser.add_argument('--load_balancing_lambda', type=float, default=0.01)

    # Optimizer/lr
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_iters', type=int, default=200)
    parser.add_argument('--min_lr', type=float, default=3e-5)
    parser.add_argument('--decay_lr', action='store_true')
    parser.add_argument('--compile', action='store_true')

    # Expert specialization (optional strict control)
    parser.add_argument('--subject_expert_map', type=str, default=None,
                        help="Mapping like 'college_chemistry:0,global_facts:1,management:2,medical_genetics:3' to force a specific expert per subject.")
    parser.add_argument('--freeze_non_target_experts', action='store_true',
                        help='When forcing mapping, freeze all other experts and non-expert modules by default.')
    parser.add_argument('--train_non_expert_modules', action='store_true',
                        help='Allow training attention/embeddings/etc. (by default they are frozen when forcing).')
    parser.add_argument('--export_experts', action='store_true',
                        help='Export per-expert weights after each subject to out_dir/experts/.')

    # System
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32','float16','bfloat16'])
    parser.add_argument('--seed', type=int, default=1337)

    # If a config file is provided, execute it to collect defaults
    if boot_known.config:
        cfg_path = boot_known.config
        print(f"Loading pretrain config: {cfg_path}")
        # Execute config file in isolated dict and apply known keys as defaults
        cfg_locals: dict = {}
        with open(cfg_path, 'r', encoding='utf-8') as cf:
            code = cf.read()
        exec(compile(code, cfg_path, 'exec'), {}, cfg_locals)
        # Map known config keys into parser defaults if present
        for key, val in cfg_locals.items():
            if key in { 'data_dir','out_dir','subjects','iters_per_subject','eval_interval','eval_iters','log_interval','init_from',
                        'block_size','batch_size','n_layer','n_head','n_embd','dropout','bias','ffn_mult','n_expert','n_routed_expert','load_balancing_lambda',
                        'learning_rate','weight_decay','beta1','beta2','grad_clip','warmup_iters','min_lr','decay_lr','compile',
                        'subject_expert_map','freeze_non_target_experts','train_non_expert_modules','export_experts',
                        'device','dtype','seed' }:
                parser.set_defaults(**{key: val})

    args = parser.parse_args()

    # Resolve final output directory so runs land under the base 'out' folder.
    # Rules:
    # - Absolute paths in out_dir are respected as-is.
    # - Relative paths that already start with 'out/' are respected.
    # - Other relative paths are nested under 'out/<out_dir>'.
    # - If the result is exactly 'out' (or empty), append an auto run name 'pretrain_<timestamp>'.
    try:
        _stamp = time.strftime('%Y%m%d_%H%M%S')
        if not os.path.isabs(args.out_dir):
            norm = args.out_dir.replace('\\', '/').lstrip('./')
            if not (norm == 'out' or norm.startswith('out/')):
                args.out_dir = os.path.join('out', args.out_dir)
            if norm == '' or norm == 'out':
                run_name = f"pretrain_{_stamp}"
                args.out_dir = os.path.join('out', run_name)
    except Exception:
        pass

    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if args.device.startswith('cuda') else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Build subject datasets
    subject_names = [s.strip() for s in args.subjects.split(',') if s.strip()]
    subjects = {name: SubjectDataset(args.data_dir, name) for name in subject_names}
    for name, ds in subjects.items():
        print(f"Subject '{name}': train tokens={ds.n_train:,}, val tokens={ds.n_val:,}")

    # Attempt to discover vocab_size from any existing meta.pkl (optional)
    meta_vocab_size = None
    meta_path_candidates = [os.path.join(args.data_dir, 'meta.pkl'), os.path.join('data','openwebtext','meta.pkl')]
    for mp in meta_path_candidates:
        if os.path.exists(mp):
            try:
                with open(mp, 'rb') as f:
                    meta = pickle.load(f)
                meta_vocab_size = int(meta.get('vocab_size', None))
                if meta_vocab_size:
                    print(f"found vocab_size = {meta_vocab_size} (inside {mp})")
                    break
            except Exception:
                pass

    # Model init args
    model_args = dict(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        bias=args.bias,
        vocab_size=meta_vocab_size if meta_vocab_size is not None else 50304,
        dropout=args.dropout,
        ffn_mult=args.ffn_mult,
        n_expert=args.n_expert,
        n_routed_expert=args.n_routed_expert,
    )

    # Initialize or resume model
    if args.init_from == 'scratch':
        print("Initializing a new model from scratch for MoE pretraining")
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif args.init_from == 'resume':
        ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
        print(f"Resuming pretraining from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=args.device)
        # keep critical fields compatible
        for k in ['n_layer','n_head','n_embd','block_size','bias','vocab_size','ffn_mult','n_expert','n_routed_expert']:
            model_args[k] = checkpoint['model_args'].get(k, model_args[k])
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        model.load_state_dict(checkpoint['model'])
    else:  # gpt2*
        print(f"Initializing from OpenAI GPT-2 weights: {args.init_from}")
        model = GPT.from_pretrained(args.init_from, override_args=dict(
            dropout=args.dropout,
            ffn_mult=args.ffn_mult,
            n_expert=args.n_expert,
            n_routed_expert=args.n_routed_expert,
        ))
        # refresh model_args from actual config
        for k in ['n_layer','n_head','n_embd','block_size','bias','vocab_size','ffn_mult','n_expert','n_routed_expert']:
            model_args[k] = getattr(model.config, k)

    # Crop block size if needed
    if args.block_size < model.config.block_size:
        model.crop_block_size(args.block_size)
        model_args['block_size'] = args.block_size

    model.to(args.device)

    # Optimizer
    def build_optimizer():
        return model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)

    optimizer = build_optimizer()
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

    if args.compile:
        print("compiling the model... (takes ~1 minute on first run)")
        model = torch.compile(model)

    # lr schedule (optional cosine with warmup)
    def get_lr(it):
        if not args.decay_lr:
            return args.learning_rate
        if it < args.warmup_iters:
            return args.learning_rate * it / args.warmup_iters
        # use long horizon cosine decay over the total planned iterations
        # rough estimate: iters_per_subject * num_subjects
        total_iters = args.iters_per_subject * max(1, len(subject_names))
        if it > total_iters:
            return args.min_lr
        decay_ratio = (it - args.warmup_iters) / max(1, (total_iters - args.warmup_iters))
        decay_ratio = max(0.0, min(1.0, decay_ratio))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_lr + coeff * (args.learning_rate - args.min_lr)

    # Eval helper
    @torch.no_grad()
    def estimate_loss(ds: SubjectDataset):
        model.eval()
        losses = []
        for _ in range(args.eval_iters):
            x, y = ds.get_batch('val', args.block_size, args.batch_size, args.device)
            logits, main_loss, aux_loss = model(x, y)
            total_loss = main_loss + args.load_balancing_lambda * aux_loss
            losses.append(total_loss.item())
        model.train()
        return float(np.mean(losses)) if losses else float('inf')

    # Parse subject->expert mapping if provided
    subj2exp = None
    if args.subject_expert_map:
        subj2exp = {}
        for token in args.subject_expert_map.split(','):
            token = token.strip()
            if not token:
                continue
            if ':' not in token:
                raise ValueError(f"Invalid token in --subject_expert_map: '{token}'")
            k, v = token.split(':', 1)
            subj2exp[k.strip()] = int(v.strip())

    def set_forced_expert(idx: int | None):
        for blk in model.transformer.h:
            mlp = getattr(blk, 'mlp', None)
            if mlp is not None and hasattr(mlp, 'experts'):
                setattr(mlp, 'force_expert_idx', idx)

    def set_freeze_for_subject(exp_idx: int | None):
        # default: freeze everything
        for _, p in model.named_parameters():
            p.requires_grad = False
        # unfreeze targeted expert parameters across all layers
        if exp_idx is not None:
            for blk in model.transformer.h:
                mlp = getattr(blk, 'mlp', None)
                if mlp is None or not hasattr(mlp, 'experts'):
                    continue
                expert = mlp.experts[exp_idx]
                for _, pp in expert.named_parameters():
                    pp.requires_grad = True
        # optionally allow non-expert modules to train
        if args.train_non_expert_modules:
            for mn, module in model.named_modules():
                if any(tag in mn for tag in ['attn', 'ln_', 'wte', 'wpe', 'lm_head']):
                    for _, pp in module.named_parameters(recurse=False):
                        pp.requires_grad = True

    def export_expert_weights(exp_idx: int, subject: str):
        if not args.export_experts:
            return
        exp_dir = os.path.join(args.out_dir, 'experts')
        os.makedirs(exp_dir, exist_ok=True)
        export = {}
        for li, blk in enumerate(model.transformer.h):
            mlp = getattr(blk, 'mlp', None)
            if mlp is None or not hasattr(mlp, 'experts'):
                continue
            expert = mlp.experts[exp_idx]
            export[f'layer_{li}'] = {k: v.detach().cpu() for k, v in expert.state_dict().items()}
        path = os.path.join(exp_dir, f'expert{exp_idx}_{subject}.pt')
        torch.save(export, path)
        print(f"exported expert {exp_idx} weights for subject '{subject}' -> {path}")

    # ------------------------- Training Loop -------------------------
    iter_global = 0
    model.train()
    for subj in subject_names:
        ds = subjects[subj]
        print(f"\n=== Pretraining on subject: {subj} ===")
        # If mapping provided, configure forced routing and freezing
        target_exp = None
        if subj2exp is not None:
            if subj not in subj2exp:
                raise ValueError(f"Subject '{subj}' not found in --subject_expert_map")
            target_exp = int(subj2exp[subj])
            if not (0 <= target_exp < args.n_expert):
                raise ValueError(f"Expert index {target_exp} out of range [0,{args.n_expert-1}] for subject '{subj}'")
            set_forced_expert(target_exp)
            if args.freeze_non_target_experts or not args.train_non_expert_modules:
                set_freeze_for_subject(target_exp)
            optimizer = build_optimizer()
        best_val = float('inf')
        t0 = time.time()
        for it in range(args.iters_per_subject):
            iter_global += 1
            # set LR
            lr = get_lr(iter_global)
            for g in optimizer.param_groups:
                g['lr'] = lr

            with ctx:
                X, Y = ds.get_batch('train', args.block_size, args.batch_size, args.device)
                logits, main_loss, aux_loss = model(X, Y)
                total_loss = main_loss + args.load_balancing_lambda * aux_loss

            scaler.scale(total_loss).backward()

            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if (it % args.log_interval) == 0:
                dt = time.time() - t0
                print(f"{subj} | iter {it:5d}/{args.iters_per_subject} | total {total_loss.item():.4f} | LM {main_loss.item():.4f} | AUX {aux_loss.item():.4f} | lr {lr:.2e} | {dt*1000:.1f} ms")
                t0 = time.time()

            if (it % args.eval_interval) == 0 and it > 0:
                val_loss = estimate_loss(ds)
                print(f"{subj} | eval @iter {it}: val total loss {val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    # save subject-best checkpoint snapshot
                    ckpt = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_global,  # train.py expects this key
                        'best_val_loss': best_val,  # train.py expects this key
                        'iter_global': iter_global,
                        'subject': subj,
                    }
                    subj_dir = os.path.join(args.out_dir, subj)
                    os.makedirs(subj_dir, exist_ok=True)
                    path = os.path.join(subj_dir, 'ckpt.pt')
                    print(f"saving checkpoint to {path}")
                    torch.save(ckpt, path)

        # save end-of-subject snapshot
        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_global,  # train.py expects this key
            'best_val_loss': best_val,  # train.py expects this key
            'iter_global': iter_global,
            'subject': subj,
        }
        path = os.path.join(args.out_dir, f'ckpt_{subj}.pt')
        print(f"subject '{subj}' done. saving checkpoint to {path}")
        torch.save(ckpt, path)
        # export targeted expert after finishing subject phase
        if target_exp is not None:
            export_expert_weights(target_exp, subj)

    # save a final generic checkpoint for easy resume
    final_path = os.path.join(args.out_dir, 'ckpt.pt')
    print(f"All subjects done. Saving final checkpoint to {final_path}")
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_global,  # train.py expects this key
        'best_val_loss': best_val,  # train.py expects this key
        'iter_global': iter_global,
        'subjects': subject_names,
    }, final_path)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR in pretrain.py: {e}")
        traceback.print_exc()
        raise
