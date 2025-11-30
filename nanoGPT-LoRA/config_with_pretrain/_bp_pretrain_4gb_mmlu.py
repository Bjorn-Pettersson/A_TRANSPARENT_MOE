# --- Pretrain Config (4GB VRAM, MMLU subjects) ---
# Keep this tiny to avoid CUDA OOM on 4GB GPUs.

# I/O
out_dir = 'out-pretrain-4gb-mmlu'
data_dir = 'data/mmlu'
subjects = 'college_chemistry,global_facts,management,medical_genetics'

# Model (very small)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False
ffn_mult = 2.0  # smaller FFN to reduce memory

# MoE
n_expert = 4
n_routed_expert = 2
load_balancing_lambda = 0.01

# Data
block_size = 128  # small context window

# Training
batch_size = 4
iters_per_subject = 400  # short pretrain per subject
eval_interval = 100
eval_iters = 20
log_interval = 10

# Optimizer & LR
learning_rate = 3e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
warmup_iters = 50
min_lr = 3e-5
decay_lr = True

# System
device = 'cuda'
dtype = 'bfloat16'  # use bfloat16 if supported; else set to 'float16' or 'float32'
compile = False
seed = 1337

# Notes:
# - If you still hit OOM, reduce: block_size -> 64, batch_size -> 2, n_embd -> 96.
# - Ensure your MMLU bins are present under data/mmlu.