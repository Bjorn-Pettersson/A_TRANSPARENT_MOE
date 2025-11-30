# Step 2: Pretrain individual experts on MMLU subjects (16GB GPU)

# ------------------------------
# Logging
# ------------------------------
wandb_log = True
wandb_project = 'moe-understanding'
wandb_run_name = 's2_16gb_1201A_pretrain'

# ------------------------------
# I/O and Data
# ------------------------------
data_dir = 'data/mmlu'
out_dir = 'out/out-s2-16gb-1201A-pretrain'
subjects = 'college_chemistry,global_facts,management,medical_genetics'

# ------------------------------
# Training schedule
# ------------------------------
iters_per_subject = 600  # More iterations for larger model
eval_interval = 200
eval_iters = 50
log_interval = 10
init_from = 'scratch'

# ------------------------------
# Model (GPT2-small - same as benchmark)
# ------------------------------
block_size = 512
batch_size = 4
gradient_accumulation_steps = 16  # Balanced for subject-specific data
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.2
bias = True
ffn_mult = 4.0

# ------------------------------
# MoE Configuration
# ------------------------------
n_expert = 4
n_routed_expert = 1  # Force single expert during pretraining
load_balancing_lambda = 0.003  # Reduced for better specialization

# ------------------------------
# Expert specialization controls
# ------------------------------
subject_expert_map = 'college_chemistry:0,global_facts:1,management:2,medical_genetics:3'
freeze_non_target_experts = True
train_non_expert_modules = False  # Only train the target expert
export_experts = True  # Export individual expert weights

# ------------------------------
# Optimizer
# ------------------------------
learning_rate = 6e-4
weight_decay = 0.3
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
warmup_iters = 100
min_lr = 6e-5
decay_lr = True

# ------------------------------
# System
# ------------------------------
device = 'cuda'
dtype = 'bfloat16'
compile = False
seed = 1337
