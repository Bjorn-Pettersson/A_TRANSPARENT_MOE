import numpy as np
import tiktoken
import os

# --- Configuration ---
# 1. Update this path to the location of your data directory
DATA_DIR = 'data/openwebtext' 
FILE_NAME = 'train.bin' # Or 'val.bin'

# 2. Specify the number of tokens you want to load and decode
TOKENS_TO_VIEW = 1000 
# Note: For very large datasets like openWebText, avoid setting this too high 
# unless you want to print a huge block of text.

# --- Execution ---

file_path = os.path.join(DATA_DIR, FILE_NAME)

print(f"1. Loading token IDs from: {file_path}")

try:
    # Use np.memmap for efficiency, as openWebText files are very large. 
    # The file content is stored as uint16 tokens (max token ID 50256 < 65536).
    data = np.memmap(file_path, dtype=np.uint16, mode='r')

    # Get a slice of the data to view
    ids_to_decode = data[:TOKENS_TO_VIEW].tolist()

    print(f"2. Loaded {len(data):,} tokens in total.")
    print(f"3. Initializing GPT-2 tokenizer (tiktoken 'gpt2')...")
    
    # Load the GPT-2 BPE encoder
    enc = tiktoken.get_encoding("gpt2")
    
    # Decode the token IDs back into a string
    decoded_text = enc.decode(ids_to_decode)

    # --- Results ---
    
    print("-" * 50)
    print(f"** SAMPLE OF THE FIRST {TOKENS_TO_VIEW} TOKENS FROM {FILE_NAME} **")
    print("-" * 50)
    print(decoded_text)
    print("-" * 50)

except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Please check the DATA_DIR path.")
except Exception as e:
    print(f"An error occurred: {e}")