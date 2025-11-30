# Step 3: Transfer learning - Resume from pretrained experts on OpenWebText (16GB GPU)

# ------------------------------
# Logging
# ------------------------------
wandb_log = True
wandb_project = 'moe-understanding'
wandb_run_name = 's3_16gb_1201A_transfer'

# ------------------------------
# Model (GPT2-small - same as benchmark and pretrain)
# ------------------------------
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.2
bias = True
ffn_mult = 4.0

# ------------------------------
# MoE (sequence-level Top-2)
# ------------------------------
n_expert = 4
n_routed_expert = 2  # Enable Top-2 routing for transfer
load_balancing_lambda = 0.003  # Reduced for better specialization

# ------------------------------
# I/O
# ------------------------------
init_from = 'resume'
out_dir = 'out/out-s3-16gb-1201A-transfer'
dataset = 'openwebtext'

# ------------------------------
# Batch Strategy (same as benchmark)
# ------------------------------
block_size = 512
batch_size = 4
gradient_accumulation_steps = 32

# ------------------------------
# Optimization
# ------------------------------
learning_rate = 9.6e-4
min_lr = 9.6e-5
weight_decay = 0.5
max_iters = 2000  # Continue training
lr_decay_iters = 2000
warmup_iters = 50
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
always_save_checkpoint = True

# ------------------------------
# System
# ------------------------------
device = 'cuda'
dtype = 'bfloat16'
compile = False
