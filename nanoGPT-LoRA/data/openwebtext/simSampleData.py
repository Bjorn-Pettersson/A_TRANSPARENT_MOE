"""
Simulate how training samples sequences from the OpenWebText `train.bin` file.

This script will:
- Find a config file (given by `--config`) or search for `config/bp_train*.py`.
- Parse `block_size` from that config (fallback to 1024 if not found).
- Open `data/openwebtext/train.bin` via `numpy.memmap` (dtype=uint16 like the training scripts).
- Sample random start indices and print decoded text for `block_size` tokens using
  the GPT-2 tokenizer from `tiktoken` when available (otherwise prints token ids).

Usage examples:
python simSampleData.py                # uses first config found or 1024 block size
python simSampleData.py --config ../config/bp_train_moe_vectorized_16gb_fan_fig2.py --samples 3 --seed 42
"""

import argparse
import os
import re
import glob
import numpy as np

try:
    import tiktoken
    HAVE_TIKTOKEN = True
except Exception:
    HAVE_TIKTOKEN = False


def find_config_path(provided_path=None):
    # If provided explicitly and exists, use it
    if provided_path:
        if os.path.exists(provided_path):
            return provided_path
        # try relative to repo root
        alt = os.path.join('..', provided_path)
        if os.path.exists(alt):
            return alt
        return None
    # search for config/bp_train*.py
    matches = sorted(glob.glob(os.path.join('..', 'config', 'bp_train*.py')) + glob.glob(os.path.join('config', 'bp_train*.py')))
    if matches:
        return matches[0]
    # no bp_train found; try any config/*.py
    matches = sorted(glob.glob(os.path.join('..', 'config', '*.py')) + glob.glob(os.path.join('config', '*.py')))
    return matches[0] if matches else None


def parse_block_size_from_config(config_path):
    if not config_path or not os.path.exists(config_path):
        return None
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            txt = f.read()
        # look for a line like: block_size = 1024
        m = re.search(r'block_size\s*=\s*(\d+)', txt)
        if m:
            return int(m.group(1))
        # fallback: look for "block_size" in assignment inside all-caps CONFIG or dataclass could be different
        m = re.search(r"block_size\s*[:=].*?(\d+)", txt)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


def load_tokens(bin_path):
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Token file not found: {bin_path}")
    return np.memmap(bin_path, dtype=np.uint16, mode='r')


def decode_tokens(token_list):
    if HAVE_TIKTOKEN:
        try:
            enc = tiktoken.get_encoding('gpt2')
            return enc.decode([int(x) for x in token_list])
        except Exception:
            return None
    return None


def main():
    p = argparse.ArgumentParser(description='Sample sequences from data/openwebtext/train.bin to mimic training samples')
    p.add_argument('--config', type=str, default=None, help='Path to config file to parse block_size from')
    p.add_argument('--data-dir', type=str, default='.', help='Path to the data/openwebtext directory (default: current directory)')
    p.add_argument('--file', type=str, default='train.bin', help='File name inside data-dir to read (default: train.bin)')
    p.add_argument('--samples', type=int, default=1, help='Number of random samples to print')
    p.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = p.parse_args()

    # locate config
    config_path = find_config_path(args.config)
    block_size = None
    if config_path:
        block_size = parse_block_size_from_config(config_path)
    if block_size is None:
        # try train.py in parent dir
        parent_train = os.path.join('..', 'train.py')
        if os.path.exists(parent_train):
            block_size = parse_block_size_from_config(parent_train)

    if block_size is None:
        block_size = 1024

    data_path = os.path.join(args.data_dir, args.file)
    try:
        data = load_tokens(data_path)
    except FileNotFoundError as e:
        print(e)
        return

    total_tokens = len(data)
    if total_tokens <= block_size:
        print(f"Data file only has {total_tokens} tokens <= block_size {block_size}; nothing to sample.")
        return

    if args.seed is not None:
        np.random.seed(args.seed)

    print(f"Using config: {config_path if config_path else 'None found; defaulting to 1024'}")
    print(f"Using block_size = {block_size}")
    print(f"Data path: {data_path} (total tokens: {total_tokens:,})")
    print('-' * 80)

    for si in range(args.samples):
        start = int(np.random.randint(0, total_tokens - block_size))
        slice_ids = data[start:start + block_size].astype(np.int64).tolist()
        decoded = decode_tokens(slice_ids)
        print(f"--- SAMPLE {si + 1} (start={start}) ---")
        if decoded is not None:
            print(decoded)
        else:
            # fall back to printing token ids in groups
            print('Decoded text unavailable (tiktoken not installed). Showing token ids:')
            # show first 200 ids for brevity, then indicate how to get full
            display_ids = slice_ids[:200]
            print(display_ids)
            if len(slice_ids) > len(display_ids):
                print(f"... (total {len(slice_ids)} token ids, only first {len(display_ids)} shown) ...")
        print('-' * 80)


if __name__ == '__main__':
    main()
