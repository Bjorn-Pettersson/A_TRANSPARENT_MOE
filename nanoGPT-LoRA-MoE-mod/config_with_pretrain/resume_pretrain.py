# Resume training from MMLU pretrain on OpenWebText
# Match the pretrain settings to avoid assertion errors
init_from = 'resume'
out_dir = 'out-pretrain-4gb-mmlu'
dataset = 'openwebtext'

# Model architecture (must match pretrain)
n_layer = 4
n_head = 4
n_embd = 128
ffn_mult = 2.0
dropout = 0.0
bias = False

# MoE settings (must match pretrain)
n_expert = 4
n_routed_expert = 2
load_balancing_lambda = 0.01

# Data (CRITICAL: must match pretrain block_size)
block_size = 128

# Training
batch_size = 4
gradient_accumulation_steps = 1
max_iters = 5000  # adjust as needed
learning_rate = 3e-4
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 3e-5
decay_lr = True

# Eval and logging
eval_interval = 200
eval_iters = 20
log_interval = 10
always_save_checkpoint = True

# System
device = 'cuda'
dtype = 'bfloat16'
compile = False  # disabled for GTX 1050 Ti