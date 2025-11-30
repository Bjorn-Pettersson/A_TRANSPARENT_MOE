# Transparent MoE: End-to-end Workflow (Sequence-Level Routing)

This doc summarizes a practical 4-step workflow to explore sequence-level routing in a Mixture-of-Experts GPT, with fast configs that run on a 4GB GPU (e.g., GTX 1050 Ti) and log to Weights & Biases (wandb). Each step includes a small config, exact commands, and what to expect.

All steps use sequence-level routing. For quick demos, we use tiny models and minimal iterations so you can see logging and produce checkpoints rapidly.

## Prerequisites
- Python env with PyTorch, `transformers`, and wandb. Log in once:
  ```powershell
  pip install wandb transformers
  wandb login
  ```
- Datasets:
  - For quick runs, configs default to `openwebtext` or `shakespeare_char` bins expected at `data/<dataset>/{train,val}.bin`.
  - For MMLU subject evaluation, prepare per-subject text files (see Step 4) or let the script attempt HF download with `--use_hf`.

## Step 0 — Sanity check (optional)
Use an ultra-tiny run just to verify the pipeline and wandb logging.

- Config: `config/bp_train_4gb_test_wandb.py` (already present)
- Command:
  ```powershell
  python train.py config/bp_train_4gb_test_wandb.py
  ```
- Does: Runs ~10 iterations, logs losses and MoE routing scalars (if MoE enabled) to wandb.

## Step 1 — Benchmark (Baseline model only)
Goal: Train and evaluate a baseline MoE on OpenWebText. This model is never modified or resumed from; it is used only for comparison.

- 1.1 Train benchmark:
  - Config: `config/step1_benchmark_moe_4gb.py`
  - Command:
    ```powershell
    python train.py config/step1_benchmark_moe_4gb.py
    ```
  - Output:
    - Checkpoint at `out-step1-benchmark-moe/ckpt.pt`
    - wandb logs: training curves and per-layer/per-expert routing scalars (layerX/expertY)

- 1.last Evaluate benchmark routing:
  ```powershell
  python eval_routing_mmlu.py --ckpt out-step1-benchmark-moe/ckpt.pt --questions_dir data/mmlu_questions --categories global_facts college_biology college_chemistry medical_genetics management --samples_per_category 100 --device cuda --out_dir out-step1-benchmark-moe/routing_analysis --use_hf
  ```

## Step 2 — Pretrained Model (Strict Expert Specialization)
Goal: Train each MoE expert on its own subject-specific dataset to encourage specialization, one expert at a time.
Routing behavior in Step 2: we do NOT learn routing. Each subject run forces all MoE blocks to route to a single designated expert (via `force_expert_idx`), sets the load-balancing auxiliary loss to zero, and freezes non-target experts (and optionally other modules). The larger network does not engage beyond the target expert parameters you choose to train.

1) Quick single-subject demo (config)
  - Simulates expert specialization by forcing K=1 routing (`n_routed_expert = 1`) on a chosen MMLU subject. This config reads your existing per-subject bins at `data/mmlu/<subject>_train.bin` and `data/mmlu/<subject>_val.bin` via explicit paths.

- Config: `config/step2_pretrain_expert_bio_4gb.py`
- Set the subject in the config (e.g., `subject = 'college_biology'`).
- Command:
  ```powershell
  python train.py config/step2_pretrain_expert_bio_4gb.py
  ```
- Output:
  - Checkpoint at `out-step2-expert-bio/ckpt.pt`
  - wandb logs showing specialization (K=1) routing

2.1 Full multi-subject strict pretrain (script)
  - Sequentially trains experts: for each subject we force routing to one expert, freeze others, export weights.
  - Example:
     ```powershell
      python pretrain.py --data_dir data\mmlu --out_dir out-pretrain \
        --subjects college_chemistry,global_facts,management,medical_genetics \
        --iters_per_subject 200 --block_size 128 --batch_size 4 \
        --n_layer 4 --n_head 4 --n_embd 128 --n_expert 4 --n_routed_expert 1 \
        --subject_expert_map college_chemistry:0,global_facts:1,management:2,medical_genetics:3 \
        --freeze_non_target_experts --export_experts
     ```
   - Output:
     - Per-subject snapshots: `out-pretrain/<subject>/ckpt.pt`
     - Subject end-of-phase snapshots: `out-pretrain/ckpt_<subject>.pt`
     - Final pretrain snapshot (last subject): `out-pretrain/ckpt.pt`
     - Per-expert exports: `out-pretrain/experts/expert{idx}_{subject}.pt` containing all layers for that expert
   - Notes:
      - `--subject_expert_map` forces a single expert per subject and freezes others (unless overridden), yielding strict expert-specific training.
      - You can also allow training of non-expert modules with `--train_non_expert_modules`.

Note: True “expert pretrain-and-graft” (copying dense FFN weights into MoE experts) requires a small weight-mapping utility and consistent shapes. This demo focuses on the behavior and logging you can observe quickly.

## Step 2.2 — General Training (Resume on OpenWebText)
Use standard Top-K routing (e.g. `n_routed_expert = 2`) and allow the router to learn allocation across the pretrained experts.
Important: Step 1 (Benchmark) is separate and remains untouched. Do not resume from Step 1. Resume only from Step 2 outputs.

Option A — Continue from Step 1 baseline
- Config: `config/step3_moe_transfer_4gb.py` (points to `out-step1-benchmark-moe`)
- Command:
  ```powershell
  python train.py config/step3_moe_transfer_4gb.py
  ```
- Output: continues from `out-step1-benchmark-moe/ckpt.pt`

2.2.1 Resume from strict pretrain:
- Config: `config/step3_moe_transfer_from_pretrain_4gb.py` (ensure `out_dir = 'out-pretrain'`)
- Command:
  ```powershell
  python train.py config/step3_moe_transfer_from_pretrain_4gb.py
  ```
Auto-clean: `train.py` removes any `force_expert_idx` flags so routing works normally.

2.2.2 Optional merge (only if you trained experts separately or want a clean router re-init):
- Command (merge):
  ```powershell
  python post_train.py --config config/step1_benchmark_moe_4gb.py \
    --experts_dir out-pretrain\experts \
    --subject_expert_map college_chemistry:0,global_facts:1,management:2,medical_genetics:3 \
    --out_ckpt out-post-transfer\ckpt.pt
  ```
- Command (train on OWT):
  ```powershell
  # point `out_dir` in a config to out-post-transfer
  python train.py config/step3_moe_transfer_4gb.py
  ```

Optional (advanced): If you do implement expert weight grafting, point the resumed run to a new `out_dir` and load merged weights before continuing.

## Step 2.last — Evaluate Routing for Pretrained Model
Use `eval_routing_mmlu.py` to collect per-layer expert selection frequencies per subject and produce heatmaps.

- Prepare question files (one per subject) under a directory, e.g. `data/mmlu_questions/`, each named `<category>.txt` with one question per line. Example categories:
  `global_facts college_biology college_chemistry medical_genetics management`
  ```powershell
  python eval_routing_mmlu.py --ckpt out-pretrain/ckpt.pt \
    --questions_dir data\mmlu_questions \
    --categories global_facts college_biology college_chemistry medical_genetics management \
    --samples_per_category 100 --device cuda --out_dir out-pretrain\routing_analysis --use_hf
  ```
- Does:
  - Reconstructs the model from the checkpoint
  - Runs forward passes to capture `last_selected_experts` per layer
  - Writes CSV/PNG per category and a summary JSON under the `--out_dir`

## Notes on Routing Logging
- The training script logs per-layer/per-expert scalars (`layerL/expertE`) and maintains an internal routing history for charting in wandb.
- Sequence-level routing requires your model to be configured with `n_expert > 0` and `n_routed_expert >= 1`.
- For stronger specialization pressure, try `n_routed_expert = 1` during subject runs (Step 2).

## Quick Reference: Commands (by step)
- Strict expert pretrain (sequential):
  ```powershell
  python pretrain.py --data_dir data\mmlu --out_dir out-pretrain \
    --subjects college_chemistry,global_facts,management,medical_genetics \
    --iters_per_subject 200 --block_size 128 --batch_size 4 \
    --n_layer 4 --n_head 4 --n_embd 128 --n_expert 4 --n_routed_expert 1 \
    --subject_expert_map college_chemistry:0,global_facts:1,management:2,medical_genetics:3 \
    --freeze_non_target_experts --export_experts
  ```
- 2.2 Resume general training (from Step 2 only):
  ```powershell
  python train.py config/step3_moe_transfer_from_pretrain_4gb.py
  ```
- Optional merge then train:
  ```powershell
  python post_train.py --config config/step1_benchmark_moe_4gb.py --experts_dir out-pretrain\experts --subject_expert_map college_chemistry:0,global_facts:1,management:2,medical_genetics:3 --out_ckpt out-post-transfer\ckpt.pt
  python train.py config/step3_moe_transfer_4gb.py
  ```
- 2.last Routing evaluation (example):
  ```powershell
  python eval_routing_mmlu.py --ckpt out-pretrain/ckpt.pt --questions_dir data\mmlu_questions --categories global_facts college_chemistry management medical_genetics --samples_per_category 100 --device cuda --out_dir out-pretrain\routing_analysis --use_hf
  ```
- Specialization eval (ablation + purity for Step 2):
  ```powershell
  python eval\eval_specialization.py --ckpt out-pretrain\ckpt.pt --data_dir data\mmlu \
    --subjects college_chemistry,global_facts,management,medical_genetics \
    --steps_per_subject 100 --batch_size 4 --device cuda --out_dir eval_results\specialization
  ```

## Troubleshooting
- No routing scalars in wandb: ensure your config sets `n_expert > 0`, `n_routed_expert >= 1`, and `wandb_log = True`.
- Data not found: place `{train,val}.bin` under `data/<dataset>/`. For tiny demos, use `shakespeare_char`.
- GPU OOM: reduce `batch_size` or `block_size`, or increase `gradient_accumulation_steps`.

---
This workflow is designed for fast iteration and clear logging. When moving from “demo” to real experiments, increase model size, tokens/iter, and align datasets with your subject bins for more reliable routing effects.
