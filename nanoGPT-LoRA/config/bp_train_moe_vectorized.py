# Vectorized Sequence-MoE training config (sequence-level routing, expert batching)
# Derived from bp_train_moe.py but adapted to the new optimized SequenceMoE implementation.
# Target: Run a 6K-iter exploratory training under tight VRAM (e.g. 4 GiB) while logging aux loss.

# ------------------------------
# Logging
# ------------------------------
wandb_log = True
wandb_project = 'moe-understanding'
wandb_run_name = 'sequence-moe-6K-vectorized'

# ------------------------------
# Mixture-of-Experts Parameters
# ------------------------------
# Keep experts modest to avoid memory blow-up on small GPUs.
n_expert = 4              # >0 activates SequenceMoE blocks
n_routed_expert = 2       # Top-k routing (try 1 later for speed)
load_balancing_lambda = 0.001  # Scales auxiliary load balancing loss

# ------------------------------
# Model Core (inherits unspecified defaults from train.py)
# You can optionally shrink depth/width further if still OOM.
# n_layer = 12
# n_head = 12
# n_embd = 768
# dropout can be higher for regularization in small token regimes.
dropout = 0.2
bias = True

# ------------------------------
# Data / Sequence Shape
# ------------------------------
block_size = 256          # Reduced context to save VRAM

# ------------------------------
# Batch Strategy (token budget)
# Effective tokens/iter = batch_size * block_size * gradient_accumulation_steps
# Here: 2 * 256 * 16 = 8,192 tokens/iter. Over 6K iters → ~49M tokens (exploratory scale).
# Increase grad_accum or block_size once stable.
# ------------------------------
batch_size = 2             # micro-batch per optimizer step fragment
gradient_accumulation_steps = 16

# ------------------------------
# Optimization
# ------------------------------
learning_rate = 9.6e-4
min_lr = 9.6e-5
weight_decay = 0.5
max_iters = 6000
lr_decay_iters = 6000      # cosine decay horizon
warmup_iters = 0           # optionally set >0 if you see instability

# ------------------------------
# Evaluation / Logging cadence
# ------------------------------
eval_interval = 200
eval_iters = 21
log_interval = 1

# Diagram-only plotting cadence (iterations). This only affects logging the
# multi-layer routing figure and does NOT change evaluation/checkpointing.
# Setting to 1 to start as requested.
plot_interval = 1

# ------------------------------
# System
# ------------------------------
device = 'cuda'
compile = False            # Enable later (True) after validating stability

# ------------------------------
# Suggested future tweaks (informational comments only):
# - Set n_routed_expert = 1 to reduce extra expert passes.
# - Try lower expert expansion (modify MLP to use a factor <4) if compute-bound.
# - Enable torch.compile once correctness is confirmed.
# - Use bfloat16 or float16 via train.py dtype flag to save memory.
# - Increase tokens/iter gradually for better utilization once stable.
# ------------------------------
