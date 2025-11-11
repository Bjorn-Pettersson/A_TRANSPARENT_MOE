# Vectorized Sequence-MoE training config tuned for a single 16GB GPU
# Goal: approximate the Figure 2 setup from Fan et al. (Sequence-level, Layer-wise
# Top-2 routing with weak expert specialization) while fitting into ~16GB VRAM
# and running for 3000 iterations.
#
# Notes / assumptions:
# - Base model: GPT2-small (n_layer=12, n_head=12, n_embd=768) as in the paper.
# - MoE: add experts into FFN in every transformer block (N=4 experts, Top-2)
# - Use sequence-level, layer-wise routing (n_routed_expert = 2) to encourage
#   weak specialization similar to Figure 2.
# - Paper used a very large token budget (1,048,576 tokens/iter). We cannot match
#   that on a 16GB card – instead we increase sequence length (1024) and set
#   gradient accumulation so tokens/iter ~131k which is a practical compromise.
# - We keep learning rate / dropout / weight decay consistent with the paper defaults
#   to stay comparable.

# ------------------------------
# Logging
# ------------------------------
wandb_log = True
wandb_project = 'moe-understanding'
wandb_run_name = 'sequence-moe-3K-16GB-fan-fig2'

# ------------------------------
# Mixture-of-Experts Parameters (match paper choices)
# ------------------------------
# Keep experts modest (N=4) as in the paper experiments for Figure 2.
n_expert = 4              # number of experts (N)
n_routed_expert = 2       # Top-K routing (K=2), sequence-level, layer-wise
load_balancing_lambda = 0.01  # paper used λ = 0.01 for many MoE experiments

# ------------------------------
# Model Core (GPT2-small defaults from paper)
# ------------------------------
# These match the paper's GPT2-small backbone (124M parameters) so the MoE
# additions are comparable to the configurations used in the study.
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.2
bias = True

# ------------------------------
# Data / Sequence Shape
# ------------------------------
# Use sequence length 1024 to resemble the paper's context size while keeping
# memory manageable on 16GB.
block_size = 1024

# ------------------------------
# Batch Strategy (token budget tuned for 16GB)
# Effective tokens/iter = batch_size * block_size * gradient_accumulation_steps
# Here: 4 * 1024 * 32 = 131,072 tokens/iter. Over 3000 iters -> ~393M tokens.
# This is far smaller than the original paper, but is a reasonable compromise
# for a single 16GB run and should still allow observation of weak specialization
# patterns in routing (especially if you analyze routing behavior like in Fig.2).
# Adjust batch_size/grad_accum if you have more memory or want a different tokens/iter.
# ------------------------------
batch_size = 4             # micro-batch per optimizer step fragment
gradient_accumulation_steps = 32

# ------------------------------
# Optimization
# ------------------------------
# Keep the same LR schedule values used in the paper's runs for comparability.
learning_rate = 9.6e-4
min_lr = 9.6e-5
weight_decay = 0.5
max_iters = 3000
lr_decay_iters = 3000      # cosine decay horizon matches total iters
warmup_iters = 0           # set >0 if you run into early instability

# ------------------------------
# Evaluation / Logging cadence
# ------------------------------
eval_interval = 200
eval_iters = 21
log_interval = 1
plot_interval = 50   # plotting the routing figure less frequently to save overhead

# ------------------------------
# System
# ------------------------------
device = 'cuda'
compile = False            # enable later once stable

# ------------------------------
# Guidance & future tweaks (informational)
# ------------------------------
# - If you observe OOM: reduce batch_size to 2 and increase grad_accum to keep tokens/iter.
# - If you'd like closer token budget to the paper, increase gradient_accumulation_steps
#   and run longer (but this will increase wall-clock time).
# - To probe weak specialization, log expert assignment statistics and evaluate
#   as in Figure 2 (MMLU/XNLI slices or custom held-out tasks).
