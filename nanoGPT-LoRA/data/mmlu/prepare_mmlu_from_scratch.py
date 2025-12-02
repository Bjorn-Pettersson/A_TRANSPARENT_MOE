"""
Download and prepare MMLU auxiliary train data as tokenized binary files.

This script downloads the MMLU auxiliary dataset, creates train/val splits,
and converts them to properly tokenized .bin files compatible with nanoGPT.

Usage:
    python data/mmlu/prepare_mmlu_from_scratch.py
"""

from datasets import load_dataset
import pandas as pd
import numpy as np
import tiktoken
import os
from pathlib import Path

# Configuration
DATASET_NAME = 'kz919/mmlu-auxiliary-train-auto-labelled'
SPLIT = 'train'
SUBJECTS = ['college_chemistry', 'global_facts', 'management', 'medical_genetics']
VAL_FRAC = 0.1
RANDOM_STATE = 42
SEPARATOR = "---"

# Output directory (same as script location)
OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_prepare_dataset():
    """Load MMLU dataset and create dataframe."""
    print("="*70)
    print("LOADING MMLU AUXILIARY TRAIN DATASET")
    print("="*70)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Subjects: {', '.join(SUBJECTS)}")
    print()
    
    print("Downloading dataset from HuggingFace...")
    ds = load_dataset(DATASET_NAME, split=SPLIT)
    print(f"✓ Loaded {len(ds):,} samples")
    
    # Convert to DataFrame
    df = ds.to_pandas()
    
    # Ensure QA column exists
    if 'qa' not in df.columns:
        if 'question' in df.columns and 'choices' in df.columns:
            df['qa'] = df['question'] + "\n" + df['choices']
        else:
            raise ValueError("Dataset missing required columns")
    
    # Filter to selected subjects
    df = df[df['task'].isin(SUBJECTS)].copy()
    print(f"✓ Filtered to {len(df):,} samples for selected subjects\n")
    
    # Print subject counts
    print("Subject distribution:")
    for subject in SUBJECTS:
        count = (df['task'] == subject).sum()
        print(f"  {subject}: {count:,}")
    print()
    
    return df

def create_train_val_splits(df):
    """Create and save train/val CSV splits."""
    print("="*70)
    print("CREATING TRAIN/VAL SPLITS")
    print("="*70)
    
    for subject in SUBJECTS:
        sub_df = df[df['task'] == subject].copy()
        if sub_df.empty:
            print(f"✗ No data for '{subject}', skipping")
            continue
        
        # Shuffle and split
        sub_df = sub_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        val_n = max(1, int(len(sub_df) * VAL_FRAC))
        train_df = sub_df.iloc[:-val_n]
        val_df = sub_df.iloc[-val_n:]
        
        # Save CSVs
        train_csv = OUT_DIR / f"{subject}_train.csv"
        val_csv = OUT_DIR / f"{subject}_val.csv"
        train_df[['qa']].to_csv(train_csv, index=False)
        val_df[['qa']].to_csv(val_csv, index=False)
        
        print(f"{subject}:")
        print(f"  Train: {len(train_df):,} samples -> {train_csv.name}")
        print(f"  Val:   {len(val_df):,} samples -> {val_csv.name}")
    
    print()

def convert_to_binary():
    """Convert CSV files to tokenized binary format."""
    print("="*70)
    print("CONVERTING TO TOKENIZED BINARY FILES")
    print("="*70)
    print("Using GPT-2 tokenizer (vocab_size=50257, padded to 50304)\n")
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    success_count = 0
    total_count = len(SUBJECTS) * 2  # train + val for each subject
    
    for subject in SUBJECTS:
        print(f"{subject}:")
        
        for split in ['train', 'val']:
            csv_file = OUT_DIR / f"{subject}_{split}.csv"
            bin_file = OUT_DIR / f"{subject}_{split}.bin"
            
            if not csv_file.exists():
                print(f"  ✗ {split.upper()}: CSV not found")
                continue
            
            # Read CSV
            df = pd.read_csv(csv_file)
            data_with_separators = SEPARATOR.join(df['qa'].tolist())
            
            # Tokenize
            ids = enc.encode_ordinary(data_with_separators)
            ids.append(enc.eot_token)  # Add EOT token (50256)
            
            # Convert to uint16 array
            ids_array = np.array(ids, dtype=np.uint16)
            
            # Validate token range
            max_token = int(ids_array.max())
            min_token = int(ids_array.min())
            
            if max_token >= 50304 or min_token < 0:
                print(f"  ✗ {split.upper()}: INVALID token range [{min_token}, {max_token}]")
                continue
            
            # Save as memmap binary
            arr = np.memmap(str(bin_file), dtype=np.uint16, mode='w+', shape=(len(ids_array),))
            arr[:] = ids_array[:]
            arr.flush()
            
            file_size_mb = bin_file.stat().st_size / (1024 * 1024)
            print(f"  ✓ {split.upper()}: {len(ids_array):,} tokens, range [{min_token}, {max_token}], {file_size_mb:.2f} MB")
            success_count += 1
        
        print()
    
    return success_count, total_count

def verify_binary_files():
    """Verify all binary files are valid."""
    print("="*70)
    print("VERIFYING BINARY FILES")
    print("="*70)
    
    all_valid = True
    
    for subject in SUBJECTS:
        print(f"{subject}:")
        
        for split in ['train', 'val']:
            bin_file = OUT_DIR / f"{subject}_{split}.bin"
            
            if not bin_file.exists():
                print(f"  ✗ {split.upper()}: File not found")
                all_valid = False
                continue
            
            try:
                data = np.memmap(str(bin_file), dtype=np.uint16, mode='r')
                max_token = int(data.max())
                min_token = int(data.min())
                
                if max_token >= 50304 or min_token < 0:
                    print(f"  ✗ {split.upper()}: Invalid token range [{min_token}, {max_token}]")
                    all_valid = False
                else:
                    print(f"  ✓ {split.upper()}: {len(data):,} tokens, valid range")
            except Exception as e:
                print(f"  ✗ {split.upper()}: Error reading file - {e}")
                all_valid = False
        
        print()
    
    return all_valid

def main():
    print("\n" + "="*70)
    print("MMLU BINARY DATA PREPARATION")
    print("="*70)
    print(f"Output directory: {OUT_DIR}")
    print("="*70 + "\n")
    
    try:
        # Step 1: Load dataset
        df = load_and_prepare_dataset()
        
        # Step 2: Create CSV splits
        create_train_val_splits(df)
        
        # Step 3: Convert to binary
        success_count, total_count = convert_to_binary()
        
        # Step 4: Verify
        all_valid = verify_binary_files()
        
        # Summary
        print("="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Files created: {success_count}/{total_count}")
        
        if all_valid and success_count == total_count:
            print("✓ ALL FILES VALID AND READY FOR TRAINING")
            print("\nYou can now run:")
            print("  python scripts/validate_mmlu_data.py --data_dir data/mmlu")
            print("  python autorun_workflow_16gb.py --run_name s1_16gb_1201A --device cuda")
        else:
            print("✗ SOME FILES FAILED - Check errors above")
        
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
