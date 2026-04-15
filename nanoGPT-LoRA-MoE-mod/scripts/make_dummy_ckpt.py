"""
Create a minimal dummy checkpoint compatible with sample/train flows.
Useful to scaffold the next step when you only need a ckpt structure.

Usage (powershell):
  python scripts/make_dummy_ckpt.py --config config/step1_benchmark_moe_4gb.py --out out-dummy/ckpt.pt

This will instantiate a GPT with the config's model_args and save a ckpt.
"""
import os
import argparse
import importlib.util
import torch

from model import GPTConfig, GPT


def load_config_module(path):
    spec = importlib.util.spec_from_file_location("cfgmod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    cfgmod = load_config_module(args.config)
    # minimal set of required model args
    model_args = dict(
        n_layer=getattr(cfgmod, 'n_layer'),
        n_head=getattr(cfgmod, 'n_head'),
        n_embd=getattr(cfgmod, 'n_embd'),
        block_size=getattr(cfgmod, 'block_size', 128),
        bias=getattr(cfgmod, 'bias', True),
        vocab_size=50304,
        dropout=getattr(cfgmod, 'dropout', 0.0),
        ffn_mult=getattr(cfgmod, 'ffn_mult', 4.0),
        n_expert=getattr(cfgmod, 'n_expert', 0),
        n_routed_expert=getattr(cfgmod, 'n_routed_expert', 1),
    )
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ckpt = {
        'model': model.state_dict(),
        'optimizer': {},
        'model_args': model_args,
        'iter_num': 0,
        'best_val_loss': 1e9,
        'config': {'dummy': True},
    }
    torch.save(ckpt, args.out)
    print('wrote dummy ckpt to', args.out)


if __name__ == '__main__':
    main()
