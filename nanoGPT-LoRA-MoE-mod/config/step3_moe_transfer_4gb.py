# MoE continuation ("transfer") run. Resume from a selected checkpoint (merged experts
# via post_train.py or the final pretrain ckpt) into a separate transfer directory.

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

init_from = 'resume'
# Use a dedicated transfer directory (aligned with README paths)
out_dir = 'out/out-step3-transfer'

# Continue training briefly and log
dataset = 'openwebtext'
block_size = 128
batch_size = 4
gradient_accumulation_steps = 1
max_iters = 65
learning_rate = 3e-4
lr_decay_iters = 20
warmup_iters = 0

eval_interval = 5
eval_iters = 2
log_interval = 1

device = 'cuda'
compile = False
