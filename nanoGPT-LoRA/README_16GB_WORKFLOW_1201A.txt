========================================================================
16GB GPU WORKFLOW - Run 1201A
========================================================================
Configuration: GPT2-small (12 layers, 768 dim) with 4 experts, Top-2 routing
Load Balancing Lambda: 0.003 (reduced from 0.01 for better specialization)
Effective tokens/iter: ~65K (batch_size=4, block_size=512, grad_accum=32)
Expected VRAM: ~14-16GB during training

Configs: s1_16gb_1201A.py, s2_16gb_1201A.py, s3_16gb_1201A.py

-------------- Data Requirements -------------------------

1. OpenWebText for general training and benchmark
   Location: data/openwebtext/train.bin, data/openwebtext/val.bin

2. MMLU subject-specific data for expert pretraining
   Location: data/mmlu/<subject>_train.bin, data/mmlu/<subject>_val.bin
   Subjects: college_chemistry, global_facts, management, medical_genetics

3. MMLU questions for routing evaluation
   Prepare with:
   python scripts/prepare_mmlu_questions.py --out_dir data/mmlu_questions --split test --include_choices --config all --subjects college_chemistry,global_facts,management,medical_genetics

-------------- Quick Start (Automated) -------------------

python autorun_workflow.py --run_name run_1201A_16gb --device cuda --samples_per_subject 100 --benchmark_config config/s1_16gb_1201A.py --do_transfer_eval --pretrain_iters_per_subject 600 --pretrain_block_size 512 --pretrain_batch_size 4 --pretrain_n_layer 12 --pretrain_n_head 12 --pretrain_n_embd 768

-------------- Manual Workflow Steps ---------------------

═══════════════════════════════════════════════════════════
STEP 1: BENCHMARK MoE (Baseline on OpenWebText)
═══════════════════════════════════════════════════════════

Purpose: Train baseline MoE without expert specialization
Config: config/s1_16gb_1201A.py
Model: 12-layer GPT2-small, 4 experts, Top-2 routing
Iterations: 3000 (~200M tokens)
Output: out/out-s1-16gb-1201A-benchmark/

1.1 Train Benchmark
-------------------
python train.py config/s1_16gb_1201A.py

Expected output:
- out/out-s1-16gb-1201A-benchmark/ckpt.pt
- WandB logs with routing patterns (layer0/expert0, etc.)
- Training loss curves
- Routing frequency diagrams

1.2 Evaluate Benchmark Routing
-------------------------------
python eval.py --ckpt out/out-s1-16gb-1201A-benchmark/ckpt.pt --questions_dir data/mmlu_questions --subjects college_chemistry,global_facts,management,medical_genetics --samples_per_subject 100 --device cuda --out_dir out/out-s1-16gb-1201A-benchmark/eval --layers_to_plot all

Expected output:
- Heatmaps per subject (PNG + CSV)
- Staple plots per layer showing expert activation
- Summary JSON with routing statistics

═══════════════════════════════════════════════════════════
STEP 2: PRETRAIN EXPERTS (Strict Specialization)
═══════════════════════════════════════════════════════════

Purpose: Train each expert on specific MMLU subject
Config: config/s2_16gb_1201A.py
Model: Same architecture as Step 1
Method: Force routing to single expert per subject
Iterations: 600 per subject (4 subjects = 2400 total)
Output: out/out-s2-16gb-1201A-pretrain/

2.1 Pretrain All Experts Sequentially
--------------------------------------
Option A - Using pretrain.py (recommended):

python pretrain.py --data_dir data\mmlu --out_dir out-s2-16gb-1201A-pretrain --subjects college_chemistry,global_facts,management,medical_genetics --iters_per_subject 600 --block_size 512 --batch_size 4 --gradient_accumulation_steps 16 --n_layer 12 --n_head 12 --n_embd 768 --n_expert 4 --n_routed_expert 1 --subject_expert_map college_chemistry:0,global_facts:1,management:2,medical_genetics:3 --freeze_non_target_experts --export_experts --load_balancing_lambda 0.003 --learning_rate 6e-4 --warmup_iters 100

Option B - Using config file:

python train.py config/s2_16gb_1201A.py

Note: If using train.py directly with the config, it expects the config to be
set up for pretrain.py-style execution. Use Option A for cleaner workflow.

Expected output:
- out-s2-16gb-1201A-pretrain/ckpt.pt (final checkpoint)
- out-s2-16gb-1201A-pretrain/ckpt_<subject>.pt (per-subject checkpoints)
- out-s2-16gb-1201A-pretrain/experts/ (exported expert weights)
  - expert0_college_chemistry.pt
  - expert1_global_facts.pt
  - expert2_management.pt
  - expert3_medical_genetics.pt

2.2 Evaluate Pretrained Model
------------------------------
python eval.py --ckpt out-s2-16gb-1201A-pretrain/ckpt.pt --questions_dir data/mmlu_questions --subjects college_chemistry,global_facts,management,medical_genetics --samples_per_subject 100 --device cuda --out_dir out-s2-16gb-1201A-pretrain/eval --layers_to_plot all

Expected: Strong specialization patterns in routing heatmaps

2.3 Ablation Study (Optional)
------------------------------
python eval_specialization.py --ckpt out-s2-16gb-1201A-pretrain\ckpt.pt --data_dir data\mmlu --subjects college_chemistry,global_facts,management,medical_genetics --steps_per_subject 100 --batch_size 4 --device cuda --out_dir eval_results\s2_16gb_1201A_specialization

Expected output:
- loss_deltas.csv: Performance impact of masking each expert
- routing_freq_layer<L>.csv: Routing frequencies per layer
- routing_purity.csv: Specialization strength per layer

═══════════════════════════════════════════════════════════
STEP 3: TRANSFER LEARNING (Resume on OpenWebText)
═══════════════════════════════════════════════════════════

Purpose: Continue training with pretrained experts on general text
Config: config/s3_16gb_1201A.py
Model: Resume from Step 2, enable Top-2 routing
Iterations: 2000 additional (~130M tokens)
Output: out/out-s3-16gb-1201A-transfer/

3.1 Option A: Merge Exported Experts (Preferred)
-------------------------------------------------
Step 3.1a: Merge expert weights into clean checkpoint

mkdir out\out-s3-16gb-1201A-transfer
python post_train.py --config config/s1_16gb_1201A.py --experts_dir out-s2-16gb-1201A-pretrain\experts --subject_expert_map college_chemistry:0,global_facts:1,management:2,medical_genetics:3 --out_ckpt out\out-s3-16gb-1201A-transfer\ckpt.pt

Step 3.1b: Resume training on OpenWebText

python train.py config/s3_16gb_1201A.py --out_dir=out/out-s3-16gb-1201A-transfer --resume_ckpt_path=out/out-s3-16gb-1201A-transfer/ckpt.pt --reset_iter

3.2 Option B: Resume Directly from Pretrain Checkpoint
-------------------------------------------------------
mkdir out\out-s3-16gb-1201A-transfer
copy out-s2-16gb-1201A-pretrain\ckpt.pt out\out-s3-16gb-1201A-transfer\ckpt.pt
python train.py config/s3_16gb_1201A.py --out_dir=out/out-s3-16gb-1201A-transfer --resume_ckpt_path=out/out-s3-16gb-1201A-transfer/ckpt.pt --reset_iter

Note: train.py automatically cleans force_expert_idx flags from checkpoint

3.3 Evaluate Transfer Model
----------------------------
python eval.py --ckpt out/out-s3-16gb-1201A-transfer/ckpt.pt --questions_dir data/mmlu_questions --subjects college_chemistry,global_facts,management,medical_genetics --samples_per_subject 100 --device cuda --out_dir out/out-s3-16gb-1201A-transfer/eval --layers_to_plot all

Expected: Routing patterns show retained specialization with more balanced usage

═══════════════════════════════════════════════════════════
COMPARISON: Compare All Three Checkpoints
═══════════════════════════════════════════════════════════

Generate routing visualizations for all three stages side-by-side:

# Benchmark
python eval.py --ckpt out/out-s1-16gb-1201A-benchmark/ckpt.pt --questions_dir data/mmlu_questions --subjects college_chemistry,global_facts,management,medical_genetics --samples_per_subject 200 --device cuda --out_dir comparison_1201A/s1_benchmark --layers_to_plot all

# Pretrained
python eval.py --ckpt out-s2-16gb-1201A-pretrain/ckpt.pt --questions_dir data/mmlu_questions --subjects college_chemistry,global_facts,management,medical_genetics --samples_per_subject 200 --device cuda --out_dir comparison_1201A/s2_pretrain --layers_to_plot all

# Transfer
python eval.py --ckpt out/out-s3-16gb-1201A-transfer/ckpt.pt --questions_dir data/mmlu_questions --subjects college_chemistry,global_facts,management,medical_genetics --samples_per_subject 200 --device cuda --out_dir comparison_1201A/s3_transfer --layers_to_plot all

-------------- Configuration Summary ---------------------

Model Architecture:
- Layers: 12
- Heads: 12
- Embedding dim: 768
- Experts: 4
- Routing: Top-2 (benchmark & transfer), Top-1 (pretrain)
- Block size: 512 tokens
- FFN multiplier: 4.0

Training Parameters:
- Batch size: 4 micro-batches
- Gradient accumulation: 32 steps (benchmark), 16 steps (pretrain)
- Effective tokens/iter: ~65K (benchmark), ~32K (pretrain)
- Learning rate: 9.6e-4 (benchmark/transfer), 6e-4 (pretrain)
- Weight decay: 0.5 (benchmark/transfer), 0.3 (pretrain)
- Load balancing lambda: 0.003 (all stages)

Memory Optimization:
- dtype: bfloat16
- Compile: False (enable after stability check)
- Gradient checkpointing: Not enabled (sufficient VRAM with 16GB)

Expected Timeline (16GB GPU):
- Step 1 Benchmark: ~2-3 hours (3000 iters)
- Step 2 Pretrain: ~1.5-2 hours (600 iters × 4 subjects)
- Step 3 Transfer: ~1.5-2 hours (2000 iters)
Total: ~5-7 hours

-------------- Troubleshooting ---------------------------

GPU Out of Memory:
1. Reduce batch_size to 2, increase gradient_accumulation_steps to 64
2. Reduce block_size to 256 (requires config change)
3. Enable gradient checkpointing (requires model.py modification)
4. Use dtype='float16' instead of bfloat16 (if GPU lacks bfloat16 support)

No routing specialization:
1. Verify load_balancing_lambda is low (0.003 confirmed)
2. Check subject_expert_map is correct in Step 2
3. Ensure freeze_non_target_experts=True in pretrain config
4. Increase iters_per_subject (try 800-1000)

WandB not logging routing:
1. Verify wandb_log=True in config
2. Check n_expert > 0 and n_routed_expert >= 1
3. Ensure plot_interval is reasonable (10 recommended)

Data not found:
1. Verify data/openwebtext/{train,val}.bin exist
2. Check data/mmlu/<subject>_{train,val}.bin for all 4 subjects
3. Run prepare_mmlu_questions.py if evaluation files missing

-------------- Key Differences from 4GB Config -----------

16GB Config vs 4GB Config:
- Model size: 12 layers vs 4 layers (~30x more parameters)
- Block size: 512 vs 128 tokens
- Iterations: 3000 vs 20 (benchmark)
- Gradient accumulation: 32 vs 1
- More comprehensive evaluation (200 samples vs 100)
- Longer pretraining per subject (600 vs 20 iters)
- Better optimizer tuning (warmup, decay schedule)

Expected Improvements:
- Stronger specialization patterns
- Better generalization on held-out tasks
- Clearer routing distinctions in heatmaps
- More stable training dynamics

========================================================================
End of 16GB Workflow Documentation
========================================================================


AUTORUN:
Basic run with defaults:
python autorun_workflow_16gb.py --run_name s1_16gb_1201A --device cuda --do_transfer_eval

Custom lambda:
python autorun_workflow_16gb.py --run_name custom_003 --load_balancing_lambda 0.003 --device cuda

Dry run (see commands without executing):
python autorun_workflow_16gb.py --run_name test --dry_run

Skip certain steps:
python autorun_workflow_16gb.py --run_name quick_test --skip_benchmark --skip_ablation --device cuda

Adjust iterations:
python autorun_workflow_16gb.py --run_name extended --benchmark_iters 5000 --pretrain_iters_per_subject 800 --transfer_iters 3000 --device cuda

Custom model size (experimental):
python autorun_workflow_16gb.py --run_name small_test --n_layer 6 --n_embd 512 --device cuda


