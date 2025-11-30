# MoE continuation ("transfer") run. Simplest path: resume from Step 1 ckpt
# by pointing out_dir to Step 1's directory and setting init_from='resume'.

# Logging
wandb_log = True
wandb_project = 'moe-understanding'
wandb_run_name = 'step3_moe_transfer_4gb'

# Model (same tiny MoE shape as Step 1)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = True

# MoE (sequence-level Top-2)
n_expert = 4
n_routed_expert = 2
load_balancing_lambda = 0.01

# Resume from Step 1
init_from = 'resume'
out_dir = 'out-step1-benchmark-moe'  # Step 1's ckpt directory (contains ckpt.pt)

# Continue training briefly and log
dataset = 'openwebtext'
block_size = 128
batch_size = 4
gradient_accumulation_steps = 1
max_iters = 50
learning_rate = 3e-4
lr_decay_iters = 20
warmup_iters = 0

eval_interval = 5
eval_iters = 2
log_interval = 1

device = 'cuda'
compile = False
