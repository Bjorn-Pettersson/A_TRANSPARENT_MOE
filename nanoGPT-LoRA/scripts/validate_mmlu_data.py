"""
Validate MMLU .bin files for out-of-bounds token indices.

This script checks all MMLU subject .bin files to ensure no token IDs exceed
the GPT-2 vocabulary size (50304 rounded, 50257 original).

Usage:
    python scripts/validate_mmlu_data.py --data_dir data/mmlu
    python scripts/validate_mmlu_data.py --data_dir data/mmlu --subjects management
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path


def validate_bin_file(path, vocab_size=50304):
    """Check if a .bin file contains valid token indices."""
    try:
        data = np.memmap(path, dtype=np.uint16, mode='r')
        min_token = int(data.min())
        max_token = int(data.max())
        n_tokens = len(data)
        
        # Check for invalid indices
        invalid_mask = data >= vocab_size
        n_invalid = int(invalid_mask.sum())
        
        status = "✓ VALID" if n_invalid == 0 else f"✗ INVALID ({n_invalid} bad tokens)"
        
        print(f"\n{path.name}:")
        print(f"  Tokens: {n_tokens:,}")
        print(f"  Range: [{min_token}, {max_token}]")
        print(f"  Status: {status}")
        
        if n_invalid > 0:
            # Find first few invalid indices
            invalid_indices = np.where(invalid_mask)[0][:10]
            print(f"  First invalid indices: {invalid_indices.tolist()}")
            print(f"  Invalid token values: {[int(data[i]) for i in invalid_indices[:10]]}")
            return False
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR reading {path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate MMLU .bin files")
    parser.add_argument('--data_dir', type=str, default='data/mmlu',
                        help='Directory containing MMLU .bin files')
    parser.add_argument('--subjects', type=str, default=None,
                        help='Comma-separated subjects to check (default: all)')
    parser.add_argument('--vocab_size', type=int, default=50304,
                        help='Maximum valid token ID (exclusive)')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Collect subjects to check
    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(',')]
    else:
        # Auto-discover subjects from .bin files
        train_bins = list(data_dir.glob('*_train.bin'))
        subjects = sorted(set(f.name.replace('_train.bin', '') for f in train_bins))
    
    if not subjects:
        print(f"No subjects found in {data_dir}")
        sys.exit(1)
    
    print("="*70)
    print(f"MMLU DATA VALIDATION (vocab_size={args.vocab_size})")
    print("="*70)
    print(f"Data directory: {data_dir}")
    print(f"Subjects: {', '.join(subjects)}")
    
    all_valid = True
    invalid_files = []
    
    for subject in subjects:
        train_path = data_dir / f"{subject}_train.bin"
        val_path = data_dir / f"{subject}_val.bin"
        
        print(f"\n{'─'*70}")
        print(f"Subject: {subject}")
        print('─'*70)
        
        if not train_path.exists():
            print(f"  ✗ Missing: {train_path.name}")
            all_valid = False
            continue
        
        if not val_path.exists():
            print(f"  ✗ Missing: {val_path.name}")
            all_valid = False
            continue
        
        train_valid = validate_bin_file(train_path, args.vocab_size)
        val_valid = validate_bin_file(val_path, args.vocab_size)
        
        if not train_valid:
            invalid_files.append(str(train_path))
            all_valid = False
        if not val_valid:
            invalid_files.append(str(val_path))
            all_valid = False
    
    print("\n" + "="*70)
    if all_valid:
        print("✓ ALL FILES VALID")
    else:
        print("✗ VALIDATION FAILED")
        print("\nInvalid files:")
        for f in invalid_files:
            print(f"  - {f}")
        print("\nThese files need to be regenerated with correct tokenization.")
    print("="*70)
    
    sys.exit(0 if all_valid else 1)


if __name__ == '__main__':
    main()
