"""
Prepare MMLU auxiliary train data as proper tokenized .bin files.

This script converts the MMLU auxiliary train CSV files into tokenized binary
files using the same format as OpenWebText (np.uint16 memmap arrays).

Usage:
    python data/mmlu/prepare_mmlu_binary.py
    python data/mmlu/prepare_mmlu_binary.py --csv_dir draft_BP/MMLU_AUX_25112025
    
This will regenerate all *_train.bin and *_val.bin files with proper tokenization.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import tiktoken
from pathlib import Path

# Constants
SUBJECTS = ['college_biology', 'management', 'college_chemistry', 'global_facts', 'medical_genetics']
SEPARATOR = "---"
DATA_DIR = Path(__file__).parent

# Initialize GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")

def convert_csv_to_tokenized_bin(csv_file, bin_file):
    """Convert CSV with 'qa' column to tokenized binary file."""
    if not os.path.exists(csv_file):
        print(f"  ✗ CSV not found: {csv_file}")
        return False
    
    # Read CSV
    df = pd.read_csv(csv_file)
    if 'qa' not in df.columns:
        print(f"  ✗ No 'qa' column in {csv_file}")
        return False
    
    # Concatenate all QA pairs with separator
    data_with_separators = SEPARATOR.join(df['qa'].tolist())
    
    # Tokenize
    print(f"  Tokenizing {len(data_with_separators):,} characters...")
    ids = enc.encode_ordinary(data_with_separators)
    ids.append(enc.eot_token)  # Add end-of-text token
    
    # Validate token range
    ids_array = np.array(ids, dtype=np.uint16)
    max_token = ids_array.max()
    min_token = ids_array.min()
    
    if max_token >= 50304 or min_token < 0:
        print(f"  ✗ Invalid token range: [{min_token}, {max_token}]")
        return False
    
    # Save as memmap binary
    arr_len = len(ids_array)
    arr = np.memmap(bin_file, dtype=np.uint16, mode='w+', shape=(arr_len,))
    arr[:] = ids_array[:]
    arr.flush()
    
    file_size_mb = os.path.getsize(bin_file) / (1024 * 1024)
    print(f"  ✓ Saved {arr_len:,} tokens ({file_size_mb:.2f} MB) to {bin_file}")
    print(f"    Token range: [{min_token}, {max_token}]")
    return True

def main():
    parser = argparse.ArgumentParser(description="Prepare MMLU binary data")
    parser.add_argument('--csv_dir', type=str, default=None,
                        help='Directory containing CSV files (default: search common locations)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory for .bin files (default: same as script location)')
    args = parser.parse_args()
    
    # Determine CSV source directory
    if args.csv_dir:
        csv_dir = Path(args.csv_dir)
    else:
        # Search common locations
        candidates = [
            DATA_DIR,  # Same as script (data/mmlu/)
            DATA_DIR.parent.parent / 'draft_BP' / 'MMLU_AUX_25112025',  # Original location
            Path('draft_BP/MMLU_AUX_25112025'),  # Relative to project root
        ]
        csv_dir = None
        for candidate in candidates:
            if candidate.exists() and (candidate / f"{SUBJECTS[0]}_train.csv").exists():
                csv_dir = candidate
                break
        
        if csv_dir is None:
            print("ERROR: Could not find CSV files. Searched:")
            for c in candidates:
                print(f"  - {c}")
            print("\nSpecify location with: --csv_dir <path>")
            sys.exit(1)
    
    # Determine output directory
    out_dir = Path(args.out_dir) if args.out_dir else DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("MMLU BINARY DATA PREPARATION")
    print("="*70)
    print(f"CSV source: {csv_dir}")
    print(f"Output directory: {out_dir}")
    print(f"Subjects: {', '.join(SUBJECTS)}")
    print(f"Tokenizer: GPT-2 (vocab_size=50257, padded to 50304)")
    print()
    
    success_count = 0
    total_count = 0
    
    for subject in SUBJECTS:
        print(f"\n{subject}")
        print("-" * 70)
        
        for split in ['train', 'val']:
            total_count += 1
            csv_file = csv_dir / f"{subject}_{split}.csv"
            bin_file = out_dir / f"{subject}_{split}.bin"
            
            print(f"  {split.upper()}:")
            if convert_csv_to_tokenized_bin(csv_file, bin_file):
                success_count += 1
    
    print("\n" + "="*70)
    print(f"COMPLETE: {success_count}/{total_count} files converted successfully")
    
    if success_count == total_count:
        print("✓ All MMLU binary files ready for training")
    else:
        print("✗ Some files failed - check errors above")
    print("="*70)

if __name__ == '__main__':
    main()
