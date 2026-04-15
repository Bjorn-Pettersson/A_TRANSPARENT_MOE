# Step 2: Pretrain individual experts (4GB-friendly)

# I/O
data_dir = 'data/mmlu'
out_dir = 'out/out-step2-pretrain'
subjects = 'college_chemistry,global_facts,management,medical_genetics'

# Training schedule
iters_per_subject = 20
eval_interval = 200
eval_iters = 100
log_interval = 10
init_from = 'scratch'

# Model (tiny)
block_size = 128
batch_size = 4
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False
ffn_mult = 4.0

# MoE
n_expert = 4
n_routed_expert = 1
load_balancing_lambda = 0.01

# Optimizer
learning_rate = 3e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
warmup_iters = 200
min_lr = 3e-5
decay_lr = False
compile = False

# Expert specialization controls
subject_expert_map = 'college_chemistry:0,global_facts:1,management:2,medical_genetics:3'
freeze_non_target_experts = True
train_non_expert_modules = False
export_experts = True

# System
device = 'cuda'
dtype = 'bfloat16'
seed = 1337
