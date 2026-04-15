# Strict expert pretraining: each expert trained on one subject dataset
# After completion resume with a transfer config (e.g. resume_pretrain.py) for general OpenWebText.

# I/O
out_dir = 'out-pretrain-4gb-mmlu'
data_dir = 'data/mmlu'
subjects = 'college_chemistry,global_facts,management,medical_genetics'

# Model (tiny for 4GB)
n_layer = 4
n_head = 4
n_embd = 128
ffn_mult = 2.0
bias = False
dropout = 0.0

# MoE
n_expert = 4
n_routed_expert = 1  # during strict pretrain we route to a single expert via force_expert_idx
load_balancing_lambda = 0.0  # disable aux loss when forcing single expert

# Data
block_size = 128

# Training (short demo) per subject
iters_per_subject = 400
batch_size = 4
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
dtype = 'bfloat16'
compile = False
seed = 1337

# Forced expert mapping & export
# python pretrain.py --subject_expert_map college_chemistry:0,global_facts:1,management:2,medical_genetics:3 --freeze_non_target_experts --export_experts
