# --- Model Architecture (Baby GPT for 4GB VRAM Test with wandb logging) ---
# Combines lightweight settings with wandb logging for quick testing.

# --- Logging ---
wandb_log = True
wandb_project = 'moe-understanding'
wandb_run_name = '4gb-test-run'

# --- Model Parameters ---
n_layer = 4      # Very few layers
n_head = 4       # Very few heads
n_embd = 128     # Very small embedding size
dropout = 0.0

# --- I/O and Data ---
out_dir = 'out-4gb-test-wandb'
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

# --- Mixture-of-Experts Parameters ---
n_expert = 4              # Number of experts
n_routed_expert = 2       # Top-2 routing
load_balancing_lambda = 0.01  # Load balancing loss weight