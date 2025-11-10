# config/test_moe_loss.py

# A very small config for a quick test run to verify the MoE loss implementation.

wandb_log = False
wandb_project = 'test'
wandb_run_name='moe-loss-check'

# -----------------------------------------------------------------------------
# MODEL CONFIG
# -----------------------------------------------------------------------------
# Small model size for quick initialization and training
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1 # Small dropout
bias = True

# MOE CONFIG
n_expert = 4
n_routed_expert = 2
load_balancing_lambda = 0.01 

# -----------------------------------------------------------------------------
# TRAINING PARAMS (Extremely Small)
# -----------------------------------------------------------------------------
# Quick Training Schedule
max_iters = 50           # Only 50 iterations
lr_decay_iters = 50
warmup_iters = 10        # Warmup for 10 steps

# Very small batch for quick processing
batch_size = 4
block_size = 64
# Minimal gradient accumulation (4 * 4 * 64 = 1,024 tokens/iter)
gradient_accumulation_steps = 4 

# eval stuff
eval_interval = 25
eval_iters = 10
log_interval = 1 # Log every step for immediate feedback
weight_decay = 0.1
learning_rate = 3e-4
min_lr = 3e-5

# System
device = 'cuda'
compile = False # Disable compile for fastest start