# Step 1: Benchmark MoE on OpenWebText (16GB GPU)
# Based on _hl_train_moe_vectorized_16gb_fan_fig2.py with load_balancing_lambda = 0.003

# ------------------------------
# Logging
# ------------------------------
wandb_log = True
wandb_project = 'moe-understanding'
wandb_run_name = 's1_16gb_1201A_benchmark'

# ------------------------------
# Mixture-of-Experts Parameters
# ------------------------------
n_expert = 4              # number of experts (N)
n_routed_expert = 2       # Top-K routing (K=2), sequence-level, layer-wise
load_balancing_lambda = 0.003  # Reduced from 0.01 for better specialization

# ------------------------------
# Model Core (GPT2-small)
# ------------------------------
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.2
bias = True
ffn_mult = 4.0

# ------------------------------
# Data / IO
# ------------------------------
out_dir = 'out/out-s1-16gb-1201A-benchmark'
dataset = 'openwebtext'
block_size = 512

# ------------------------------
# Batch Strategy (optimized for 16GB)
# Effective tokens/iter = batch_size * block_size * gradient_accumulation_steps
# 4 * 512 * 32 = 65,536 tokens/iter
# ------------------------------
batch_size = 4
gradient_accumulation_steps = 32

# ------------------------------
# Optimization
# ------------------------------
learning_rate = 9.6e-4
min_lr = 9.6e-5
weight_decay = 0.5
max_iters = 3000
lr_decay_iters = 3000
warmup_iters = 100
decay_lr = True

# AdamW optimizer
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# ------------------------------
# Evaluation / Logging
# ------------------------------
eval_interval = 200
eval_iters = 20
log_interval = 1
plot_interval = 10
eval_only = False
always_save_checkpoint = True

# ------------------------------
# System
# ------------------------------
device = 'cuda'
dtype = 'bfloat16'
compile = False
init_from = 'scratch'
