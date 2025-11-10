# --- Model Architecture (Baby GPT for 4GB VRAM Test) ---
# Overrides the default GPT2-124M architecture to ensure it fits.
n_layer = 4      # Very few layers
n_head = 4       # Very few heads
n_embd = 128     # Very small embedding size
dropout = 0.0

# --- I/O and Data ---
out_dir = 'out-4gb-test'
dataset = 'openwebtext'
block_size = 128 # Small context window (reduced VRAM consumption)

# --- Training Parameters (Trivial Test Run) ---
batch_size = 4
gradient_accumulation_steps = 1 # No accumulation needed for a quick test
max_iters = 10 # Only run for 10 steps to confirm the pipeline works
learning_rate = 3e-4
lr_decay_iters = 10
warmup_iters = 0

# --- Eval and Logging ---
eval_interval = 5
eval_iters = 1
log_interval = 1

# --- System ---
device = 'cuda'
compile = False # Disable torch.compile for simplicity and speed of a tiny test run