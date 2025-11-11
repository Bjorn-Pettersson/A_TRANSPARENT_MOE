# Sequence-MoE config (top_k=1, reduced expert width) for faster experimentation on small GPUs.
# Builds on bp_train_moe_vectorized.py but lowers compute by:
#   - Using n_routed_expert = 1 (single expert per sequence)
#   - Reducing FFN expansion factor (ffn_mult) inside experts to 2.0 instead of 4.0
#   - Optionally fewer layers/heads can be uncommented to shrink further.

wandb_log = True
wandb_project = 'moe-understanding'
wandb_run_name = 'sequence-moe-k1-shrink'

# Core MoE parameters
n_expert = 4             # Keep same number of experts for diversity
n_routed_expert = 1      # Route each sequence to only one expert
load_balancing_lambda = 0.001

# Model scale (inherits other defaults from train.py unless overridden)
# Uncomment to shrink depth/width if needed further:
# n_layer = 8
# n_head = 8
# n_embd = 512

# Reduce FFN width globally (affects both dense MLP and experts)
ffn_mult = 2.0           # 2x expansion vs standard 4x; halves FFN FLOPs

# Data / sequence shape
block_size = 256         # Same context length

# Batch / token budget
batch_size = 2
gradient_accumulation_steps = 16  # Effective tokens/iter: 2 * 256 * 16 = 8192

# Optimization
learning_rate = 9.6e-4
min_lr = 9.6e-5
weight_decay = 0.5
max_iters = 6000
lr_decay_iters = 6000
warmup_iters = 0

# Regularization
bias = True
dropout = 0.2

# Evaluation cadence
eval_interval = 200
eval_iters = 20
log_interval = 1

# System
device = 'cuda'
compile = False          # Enable later once stable

# Notes:
# - With top_k=1 and ffn_mult=2.0, per-block compute ~ (1/2) of original FFN and no multi-expert blending.
# - You can increase batch_size or accumulation later for better GPU utilization.
# - If memory still tight, set n_expert=2 or n_embd=384.
