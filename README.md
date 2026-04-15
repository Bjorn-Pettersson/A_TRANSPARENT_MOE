# A Transparent MoE

Research and engineering workspace for building a transparent Mixture-of-Experts (MoE) language model pipeline with sequence-level routing, expert specialization, and reproducible evaluation on MMLU-style domain splits.

## Why This Project

Most MoE repos optimize for scale, but are harder to inspect and debug. This workspace focuses on:

- **Interpretability-first MoE workflows** (routing heatmaps, specialization ablations)
- **Practical training recipes** for both 4GB and 16GB GPU setups
- **Transparent experimentation** from early concept notes to reproducible scripts

## Repository At A Glance

This repo contains one main implementation track and several research/support tracks:

- `nanoGPT-LoRA/`:
  - Primary implementation for the Transparent MoE workflow
  - Training, pretraining, transfer, and evaluation scripts
  - Includes automated runners (`autorun_workflow.py`, `autorun_workflow_16gb.py`)
  - See `nanoGPT-LoRA/README.md` for detailed script-level documentation
- `nanoMoE-master/`:
  - Upstream/forked baseline for MoE-capable nanoGPT-style training
  - Useful reference point for architecture and baseline configs
- `draft_HL/`:
  - Work-in-progress training/plotting experiments
- `draft_BP/`:
  - Data prep notebooks, MMLU utilities, and report assets
- `Abstract_Initial_Idea/`:
  - Early project ideation, abstracts, and presentation materials

## Main Workflow (Transparent MoE)

The central pipeline in `nanoGPT-LoRA/` is a 3-stage workflow:

1. **Benchmark MoE training** on OpenWebText (baseline routing behavior)
2. **Subject-specific expert pretraining** with forced routing (specialization)
3. **Transfer training** back on general text (retain + blend specialization)

Evaluation includes:

- Per-subject routing heatmaps
- Layer-wise expert usage plots
- Leave-one-expert-out specialization ablations

## Quick Start

### 1) Environment

```bash
cd nanoGPT-LoRA
pip install torch transformers datasets wandb tiktoken matplotlib seaborn numpy
```

### 2) Prepare data

- OpenWebText binaries in `nanoGPT-LoRA/data/openwebtext/`
- MMLU subject binaries in `nanoGPT-LoRA/data/mmlu/`
- MMLU question files can be generated via:

```bash
python scripts/prepare_mmlu_questions.py \
  --out_dir data/mmlu_questions \
  --split test \
  --include_choices \
  --config all \
  --subjects college_chemistry,global_facts,management,medical_genetics
```

### 3) Run end-to-end workflow (16GB profile)

```bash
python autorun_workflow_16gb.py \
  --run_name run_16gb_demo \
  --device cuda \
  --do_transfer_eval
```

For detailed options, see:

- `nanoGPT-LoRA/README.md`
- `nanoGPT-LoRA/README_16GB_WORKFLOW_1201A.txt`
- `nanoGPT-LoRA/README_4GB.txt`

## What Is Production-Ready vs Experimental

- **Most stable path:** scripts and configs under `nanoGPT-LoRA/`
- **Reference baseline:** `nanoMoE-master/`
- **Exploratory / draft:** `draft_*` and parts of `Abstract_Initial_Idea/`

## Suggested Reading Order

1. `nanoGPT-LoRA/README.md`
2. `nanoGPT-LoRA/autorun_workflow_16gb.py`
3. `nanoGPT-LoRA/pretrain.py`
4. `nanoGPT-LoRA/eval.py`
5. `nanoGPT-LoRA/eval_specialization.py`

## Current Status

- Active research project with reproducible training/evaluation scripts
- Focused on understanding and visualizing expert specialization behavior
- Ongoing work on cleaner packaging, tests, and benchmark reporting

## License

This workspace contains multiple subprojects and inherited components. See per-folder license files:

- `nanoGPT-LoRA/LICENSE`
- `nanoMoE-master/LICENSE`
