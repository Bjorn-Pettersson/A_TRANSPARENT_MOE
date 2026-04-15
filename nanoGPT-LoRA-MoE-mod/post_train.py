"""
Assemble a MoE checkpoint by injecting per-expert weights exported from pretrain.py
into a fresh GPT MoE model (same architecture). This lets you strictly pretrain
experts per subject and then merge them for general OpenWebText training.

Usage (powershell):
  python post_train.py --config config/step1_benchmark_moe_4gb.py \
    --experts_dir out-pretrain/experts --subject_expert_map college_chemistry:0,global_facts:1,management:2,medical_genetics:3 \
    --out_ckpt out-post-transfer/ckpt.pt

Notes:
- Expects files like experts/expert0_college_chemistry.pt containing a dict
  { 'layer_0': state_dict, 'layer_1': state_dict, ... } for the expert MLPs.
- Layers must match the target model's number of layers and expert index.
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


def parse_map(s: str):
    mapping = {}
    for token in s.split(','):
        token = token.strip()
        if not token:
            continue
        if ':' not in token:
            raise ValueError(f"Invalid token in --subject_expert_map: '{token}'")
        k, v = token.split(':', 1)
        mapping[k.strip()] = int(v.strip())
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Config file describing model shape (n_layer, n_expert, etc.)')
    ap.add_argument('--experts_dir', required=True, help='Directory produced by pretrain.py with expert*_{subject}.pt files')
    ap.add_argument('--subject_expert_map', required=True, help='Mapping like subj0:0,subj1:1,... used to locate files')
    ap.add_argument('--out_ckpt', required=True, help='Where to write the merged MoE checkpoint (ckpt.pt)')
    args = ap.parse_args()

    cfgmod = load_config_module(args.config)
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

    if model_args['n_expert'] <= 0:
        raise ValueError('Target model has no experts (n_expert <= 0). Provide a MoE config.')

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    mapping = parse_map(args.subject_expert_map)

    # For each subject->expert index, load the exported expert file and copy into the model
    for subject, exp_idx in mapping.items():
        # find file
        filename = os.path.join(args.experts_dir, f"expert{exp_idx}_{subject}.pt")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Export file not found for subject '{subject}', expert {exp_idx}: {filename}")
        # Load weights-only for safety; expert export files contain plain tensors
        blob = torch.load(filename, map_location='cpu', weights_only=True)
        # iterate layers
        for li, blk in enumerate(model.transformer.h):
            key = f'layer_{li}'
            if key not in blob:
                raise KeyError(f"Layer key '{key}' missing in {filename}")
            state = blob[key]
            mlp = getattr(blk, 'mlp', None)
            assert hasattr(mlp, 'experts'), 'Target model is not MoE at this block'
            expert = mlp.experts[exp_idx]
            # Allow missing keys like biases if the source export did not include them
            expert.load_state_dict(state, strict=False)
        print(f"Loaded subject '{subject}' into expert {exp_idx} across all layers")

    # Save merged checkpoint
    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)
    ckpt = {
        'model': model.state_dict(),
        'optimizer': {},
        'model_args': model_args,
        'iter_num': 0,
        'best_val_loss': 1e9,
        'config': {'post_train_merge': True},
    }
    torch.save(ckpt, args.out_ckpt)
    print('Wrote merged MoE checkpoint to', args.out_ckpt)


if __name__ == '__main__':
    main()
