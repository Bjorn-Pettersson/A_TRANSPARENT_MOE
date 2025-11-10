# config for training a Mixture of Experts (MoE) model.
# This setup targets the experimental details from Fan et al. (2024), Section A.1.

# LOGGING
wandb_log = True
wandb_project = 'moe-understanding' 
wandb_run_name='sequence-moe-6K-run' 

# -----------------------------------------------------------------------------
# MOE AND LOAD BALANCING PARAMETERS 
# -----------------------------------------------------------------------------
n_expert = 4 
n_routed_expert = 2 
load_balancing_lambda = 0.001 #0.01 in paper, here reduced because it was huge when initiating.

# -----------------------------------------------------------------------------
# TRAINING AND HYPERPARAMETERS (Optimized for 4 GiB VRAM)
# -----------------------------------------------------------------------------
# ⚠️ VRAM FIX: Drastically reducing batch_size, increasing grad_accum. 
# We target an effective batch size of 262,144 tokens/iteration as a starting point.
# This is 1/4th of the original 1,048,576, but should fit 4 GiB VRAM.
# (64 grad_accum * 4 batch_size * 1024 block_size) = 262,144 tokens

batch_size = 2  # ⬅️ REDUCED from 8 (Critical VRAM saving step)
block_size = 256 

# ⚠️ VRAM FIX: Increased to compensate for reduced batch_size.
gradient_accumulation_steps = 16 # ⬅️ REDUCED from 128 (Targeting 1/4th of original tokens)

# The paper ran for 6K iterations, seeing 6B tokens total.
# If you stick to 6K iterations, you will train on 1.5B tokens (1/4th of paper).
max_iters = 6000
lr_decay_iters = 6000

# Paper's Hyperparameters (Section A.1) - Keep these the same
dropout = 0.2
learning_rate = 9.6e-4
min_lr = 9.6e-5 
weight_decay = 0.5
bias = True 

# eval stuff
eval_interval = 200 
eval_iters = 20
log_interval = 1

# System
device = 'cuda'
compile = False # Set to True if your environment supports it (optional performance boost)