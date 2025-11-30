# Transparent MoE: Sequence-Level Routing with Expert Specialization

A practical implementation of Mixture-of-Experts (MoE) GPT with **sequence-level routing**, designed for investigating expert specialization on domain-specific datasets. This project enables training specialized experts on subject-specific corpora (MMLU) and analyzing routing behavior through comprehensive evaluation tools.

## Overview

This repository implements a complete pipeline for:
- **Benchmark MoE training** on general text (OpenWebText)
- **Expert pretraining** with strict specialization on MMLU subjects
- **Transfer learning** from pretrained experts
- **Routing analysis** with heatmaps and per-layer visualizations
- **Specialization evaluation** via leave-one-expert-out ablations

### Key Features

- **Sequence-level MoE routing**: Each sequence routes to top-K experts (not token-level)
- **4GB GPU-friendly configs**: Runs on consumer hardware (GTX 1050 Ti tested)
- **Weights & Biases integration**: Automatic logging of routing patterns and training metrics
- **Flexible expert training**: Force specific experts per subject or learn routing dynamically
- **Comprehensive evaluation**: Heatmaps, staple plots, ablation studies, and routing purity metrics

## Prerequisites

### Installation
```powershell
pip install torch transformers datasets wandb tiktoken matplotlib seaborn numpy
wandb login
```

### Data Preparation

**Option 1: Automated workflow** (recommended)
```powershell
python autorun_workflow.py --run_name demo1 --device cuda --samples_per_subject 100
```

**Option 2: Manual setup**

1. **OpenWebText** for general training:
   - Place `train.bin` and `val.bin` in `data/openwebtext/`

2. **MMLU subject-specific data** for expert pretraining:
   ```powershell
   # Prepare binary files per subject (handled by pretrain.py or use existing bins)
   # Expected: data/mmlu/<subject>_{train,val}.bin
   ```

3. **MMLU questions** for routing evaluation:
   ```powershell
   python scripts/prepare_mmlu_questions.py --out_dir data/mmlu_questions --split test --include_choices --config all --subjects college_chemistry,global_facts,management,medical_genetics
   ```

## Quick Start: Automated Workflow

The `autorun_workflow.py` script orchestrates the complete pipeline:

```powershell
python autorun_workflow.py --run_name experiment1 --device cuda --subjects college_chemistry,global_facts,management,medical_genetics --samples_per_subject 100 --do_transfer_eval
```

This will:
1. Prepare MMLU question files
2. Train benchmark MoE (Step 1)
3. Evaluate benchmark routing
4. Pretrain specialized experts (Step 2)
5. Evaluate pretrained model
6. Resume training on OpenWebText (Step 3)
7. Optionally evaluate transfer checkpoint

**Output directories:**
- `out/<run_name>_benchmark/` - Benchmark model and evaluation
- `out/<run_name>_step2_pretrain/` - Pretrained experts and exports
- `out/<run_name>_transfer/` - Transfer learning checkpoint

## Manual Workflow: Step-by-Step

### Step 1: Benchmark (Baseline MoE)

Train a baseline MoE on OpenWebText without expert specialization.

**Config:** `config/step1_benchmark_moe_4gb.py`

```python
# Key settings:
n_layer = 4
n_expert = 4
n_routed_expert = 2  # Top-2 routing
load_balancing_lambda = 0.01
max_iters = 20
```

**Commands:**
```powershell
# Train
python train.py config/step1_benchmark_moe_4gb.py

# Evaluate routing
python eval.py --ckpt out/out-step1-benchmark-moe/ckpt.pt --questions_dir data/mmlu_questions --subjects college_chemistry,global_facts,management,medical_genetics --samples_per_subject 100 --device cuda --out_dir out/out-step1-benchmark-moe/eval --layers_to_plot all
```

**Outputs:**
- `out/out-step1-benchmark-moe/ckpt.pt` - Model checkpoint
- Per-layer/per-expert routing frequency heatmaps
- Staple plots showing expert activation across subjects
- WandB logs with routing scalars (`layerX/expertY`)

### Step 2: Expert Pretraining (Strict Specialization)

Train each expert on a specific MMLU subject with forced routing.

**Config:** `config/step2_pretrain_4gb.py`

```python
# Key settings:
n_expert = 4
n_routed_expert = 1  # Force single expert
subject_expert_map = 'college_chemistry:0,global_facts:1,management:2,medical_genetics:3'
freeze_non_target_experts = True
export_experts = True
```

**Method 1: Using pretrain.py** (recommended)
```powershell
python pretrain.py --data_dir data\mmlu --out_dir out-pretrain --subjects college_chemistry,global_facts,management,medical_genetics --iters_per_subject 200 --block_size 128 --batch_size 4 --n_layer 4 --n_head 4 --n_embd 128 --n_expert 4 --n_routed_expert 1 --subject_expert_map college_chemistry:0,global_facts:1,management:2,medical_genetics:3 --freeze_non_target_experts --export_experts
```

**Method 2: Using config file**
```powershell
python train.py config/step2_pretrain_4gb.py
```

**How it works:**
- Sequentially trains one expert per subject
- Forces routing to designated expert via `subject_expert_map`
- Freezes all other experts during each subject's training
- Exports expert weights to `out-pretrain/experts/expert{idx}_{subject}.pt`

**Outputs:**
- `out-pretrain/ckpt.pt` - Final checkpoint with all pretrained experts
- `out-pretrain/ckpt_<subject>.pt` - Checkpoint after each subject
- `out-pretrain/experts/` - Individual expert weight files

**Evaluate specialization:**
```powershell
python eval.py --ckpt out-pretrain/ckpt.pt --questions_dir data/mmlu_questions --subjects college_chemistry,global_facts,management,medical_genetics --samples_per_subject 100 --device cuda --out_dir out-pretrain/eval --layers_to_plot all
```

**Ablation study:**
```powershell
python eval_specialization.py --ckpt out-pretrain\ckpt.pt --data_dir data\mmlu --subjects college_chemistry,global_facts,management,medical_genetics --steps_per_subject 100 --batch_size 4 --device cuda --out_dir eval_results\specialization
```

### Step 3: Transfer Learning (Resume on OpenWebText)

Continue training pretrained experts on general text with learned routing.

**Config:** `config/step3_moe_transfer_4gb.py`

```python
# Key settings:
init_from = 'resume'
n_routed_expert = 2  # Enable Top-2 routing
out_dir = 'out/out-step3-transfer'
```

**Option A: Resume from pretrain checkpoint directly**
```powershell
# Copy pretrain checkpoint
mkdir out\out-step3-transfer
copy out\out-step2-pretrain\ckpt.pt out\out-step3-transfer\ckpt.pt

# Resume training (train.py auto-removes force_expert_idx flags)
python train.py config/step3_moe_transfer_4gb.py --out_dir=out/out-step3-transfer --resume_ckpt_path=out/out-step3-transfer/ckpt.pt --reset_iter
```

**Option B: Merge exported experts then train**
```powershell
# Merge individual expert exports into fresh model
python post_train.py --config config/step1_benchmark_moe_4gb.py --experts_dir out-pretrain\experts --subject_expert_map college_chemistry:0,global_facts:1,management:2,medical_genetics:3 --out_ckpt out/out-step3-transfer/ckpt.pt

# Train on merged checkpoint
python train.py config/step3_moe_transfer_4gb.py --out_dir=out/out-step3-transfer --resume_ckpt_path=out/out-step3-transfer/ckpt.pt --reset_iter
```

**Evaluate transfer:**
```powershell
python eval.py --ckpt out/out-step3-transfer/ckpt.pt --questions_dir data/mmlu_questions --subjects college_chemistry,global_facts,management,medical_genetics --samples_per_subject 100 --device cuda --out_dir out/out-step3-transfer/eval --layers_to_plot all
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `train.py` | Main training loop, supports MoE configs |
| `pretrain.py` | Sequential expert pretraining with forced routing |
| `eval.py` | Routing analysis with heatmaps and staple plots |
| `eval_specialization.py` | Leave-one-expert-out ablations and purity metrics |
| `post_train.py` | Merge individual expert exports into single checkpoint |
| `autorun_workflow.py` | Automated end-to-end pipeline |
| `scripts/prepare_mmlu_questions.py` | Download and format MMLU questions |

## Configuration Files

All configs are in `config/`:

- `step1_benchmark_moe_4gb.py` - Baseline MoE on OpenWebText
- `step2_pretrain_4gb.py` - Expert pretraining with strict specialization
- `step3_moe_transfer_4gb.py` - Transfer learning from pretrained experts

**MoE Parameters:**
```python
n_expert = 4              # Number of experts
n_routed_expert = 2       # Top-K routing (K=1 for strict specialization)
load_balancing_lambda = 0.01  # Auxiliary loss weight
```

## Evaluation Outputs

### Routing Heatmaps
- **Format:** PNG + CSV per subject
- **Shows:** Expert selection frequency per layer
- **Location:** `<out_dir>/eval/<subject>_heatmap.png`

### Staple Plots
- **Format:** PNG per layer
- **Shows:** Expert activation percentages across all subjects side-by-side
- **Location:** `<out_dir>/eval/layer_<L>_grouped_bars.png`

### Specialization Metrics
- **loss_deltas.csv:** Performance degradation when masking each expert
- **routing_freq_layer<L>.csv:** Normalized routing frequencies
- **routing_purity.csv:** Maximum expert share per layer (specialization measure)

## WandB Logging

Training automatically logs:
- **Loss curves:** Main loss and load balancing auxiliary loss
- **Per-layer/per-expert routing:** Scalars named `layer<L>/expert<E>`
- **Routing diagrams:** Multi-layer visualization updated every `plot_interval` iterations
- **Model checkpoints:** Linked to wandb artifacts

Access logs at: `https://wandb.ai/<your-org>/<project>/<run>`

## Advanced Usage

### Custom Subject-Expert Mapping
```powershell
python pretrain.py ... --subject_expert_map biology:0,chemistry:1,physics:2,math:3
```

### Training Non-Expert Modules
```powershell
python pretrain.py ... --train_non_expert_modules  # Train attention, embeddings, etc.
```

### Inference with Expert Masking
```python
# In eval_specialization.py or custom scripts
from model import GPT
model = GPT.from_checkpoint('ckpt.pt')

# Mask expert 0 across all layers
mask = torch.ones(model.config.n_expert)
mask[0] = 0.0
set_global_expert_mask(model, mask)
```

### Dry Run (Planning Only)
```powershell
python autorun_workflow.py --run_name test --dry_run
```

## Troubleshooting

**No routing scalars in WandB:**
- Ensure `n_expert > 0` and `wandb_log = True` in config

**Data not found:**
- Verify `data/openwebtext/{train,val}.bin` exists
- Check `data/mmlu/<subject>_{train,val}.bin` for pretrain

**GPU OOM:**
- Reduce `batch_size` or `block_size` in config
- Increase `gradient_accumulation_steps`

**Routing not specializing:**
- Use `n_routed_expert = 1` during pretrain
- Verify `freeze_non_target_experts = True`
- Check `subject_expert_map` matches your subjects

## Project Structure

```
nanoGPT-LoRA/
├── train.py                     # Main training script
├── pretrain.py                  # Expert pretraining
├── eval.py                      # Routing evaluation
├── eval_specialization.py       # Ablation studies
├── post_train.py               # Expert merging
├── autorun_workflow.py         # Automated pipeline
├── model.py                    # GPT + SequenceMoE implementation
├── config/                     # Training configurations
│   ├── step1_benchmark_moe_4gb.py
│   ├── step2_pretrain_4gb.py
│   └── step3_moe_transfer_4gb.py
├── scripts/
│   └── prepare_mmlu_questions.py
├── data/
│   ├── openwebtext/           # General training data
│   ├── mmlu/                  # Subject-specific bins
│   └── mmlu_questions/        # Evaluation questions
└── out/                       # Checkpoints and results
```

## Quick Reference Commands

Based on `README_CHECKPTS.txt` workflow:

### Data Preparation
```powershell
# Prepare MMLU questions for evaluation
python scripts/prepare_mmlu_questions.py --out_dir data/mmlu_questions --split test --include_choices --config all --subjects college_chemistry,global_facts,management,medical_genetics
```

### Step 1: Benchmark
```powershell
# 1.1 Train benchmark MoE
python train.py config/step1_benchmark_moe_4gb.py

# 1.2 Evaluate benchmark (with heatmaps and staples)
python eval.py --ckpt out/out-step1-benchmark-moe/ckpt.pt --questions_dir data/mmlu_questions --subjects college_chemistry,global_facts,management,medical_genetics --samples_per_subject 100 --device cuda --out_dir out/out-step1-benchmark-moe/eval --layers_to_plot all
```

### Step 2: Pretrain Experts
```powershell
# Pretrain individual experts on MMLU subjects
python pretrain.py --config config/step2_pretrain_4gb.py
```

### Step 3: Transfer (Resume Training)

**Option A: Merge experts first, then train**
```powershell
# Ensure experts were exported in Step 2 (directory: out/out-step2-pretrain/experts)
mkdir out\out-step3-transfer
python post_train.py --config config/step3_moe_transfer_4gb.py --experts_dir out/out-step2-pretrain/experts --subject_expert_map college_chemistry:0,global_facts:1,management:2,medical_genetics:3 --out_ckpt out/out-step3-transfer/ckpt.pt
python train.py config/step3_moe_transfer_4gb.py --out_dir=out/out-step3-transfer --resume_ckpt_path=out/out-step3-transfer/ckpt.pt --reset_iter
```

**Option B: Resume directly from pretrain checkpoint**
```powershell
mkdir out\out-step3-transfer
copy out\out-step2-pretrain\ckpt.pt out\out-step3-transfer\ckpt.pt
python train.py config/step3_moe_transfer_4gb.py --out_dir=out/out-step3-transfer --resume_ckpt_path=out/out-step3-transfer/ckpt.pt --reset_iter
```

### Evaluation
```powershell
# Evaluate pretrain checkpoint
python eval.py --ckpt out/out-step2-pretrain/ckpt.pt --questions_dir data/mmlu_questions --subjects college_chemistry,global_facts,management,medical_genetics --samples_per_subject 100 --device cuda --out_dir out/out-step2-pretrain/eval --layers_to_plot all

# Evaluate transfer checkpoint
python eval.py --ckpt out/out-step3-transfer/ckpt.pt --questions_dir data/mmlu_questions --subjects college_chemistry,global_facts,management,medical_genetics --samples_per_subject 100 --device cuda --out_dir out/out-step3-transfer/eval --layers_to_plot all
```

## Citation

If you use this code, please cite the original nanoGPT work and acknowledge the sequence-level MoE routing implementation.

## License

See LICENSE file for details.
