# Quick MoE benchmark on OpenWebText (sequence-level routing), 4GB-friendly

# Logging
wandb_log = True
wandb_project = 'moe-understanding'
wandb_run_name = 'step1_benchmark_moe_4gb'

# Model (tiny)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = True

# MoE (sequence-level routing)
n_expert = 4
n_routed_expert = 2
load_balancing_lambda = 0.01

# Data / IO
out_dir = 'out-step1-benchmark-moe'
dataset = 'openwebtext'
block_size = 128

# Train (very short sanity run)
batch_size = 4
gradient_accumulation_steps = 1
max_iters = 20
learning_rate = 3e-4
lr_decay_iters = 20
warmup_iters = 0

# Eval/Log cadence
eval_interval = 5
eval_iters = 2
log_interval = 1

# System
device = 'cuda'
compile = False
