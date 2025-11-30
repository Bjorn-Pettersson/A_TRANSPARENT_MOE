# Continue training from the final multi-subject pretrain checkpoint

# Logging
wandb_log = True
wandb_project = 'moe-understanding'
wandb_run_name = 'step3_transfer_from_pretrain_4gb'

# Model shape (match pretrain)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = True

# MoE
n_expert = 4
n_routed_expert = 2  # switch back to Top-2 for general OWT training
load_balancing_lambda = 0.01

# Resume from pretrain final
init_from = 'resume'
out_dir = 'out-pretrain'  # contains ckpt.pt

# Continue training shortly on OWT
dataset = 'openwebtext'
block_size = 128
batch_size = 4
gradient_accumulation_steps = 1
max_iters = 20
learning_rate = 3e-4
lr_decay_iters = 20
warmup_iters = 0

eval_interval = 5
eval_iters = 2
log_interval = 1

device = 'cuda'
compile = False
