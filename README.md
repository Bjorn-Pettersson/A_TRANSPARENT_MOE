# A Transparent MoE

This repository contains our end of semester research project at the course Advanced Deep Learning at Seoul National University 2025. 

This is a workspace for creating a Mixture-of-Experts (MoE) language model pipeline with sequence-level routing, expert specialization, and reproducible evaluation on MMLU-style domain splits.

Final project paper: [A transparent MoE for sequence-level routing paper.pdf](A%20transparent%20MoE%20for%20sequence-level%20routing%20paper.pdf)

## Acknowledgments & References

This work builds on:
- **nanoGPT** (https://github.com/karpathy/nanoGPT) – A minimal, clean GPT-2 implementation that served as the foundation for our MoE adaptations.
- **Fan et al. (2024)** (https://arxiv.org/abs/2402.13089) – "Towards an Empirical Understanding of MoE Design Choices" – The seminal study on weak sequence-level domain specialization that motivated this research.

## Why This Project

Most MoE repos optimize for scale, but are harder to inspect and debug. This workspace focuses on:

- **Interpretability-first MoE workflows** (routing heatmaps, specialization ablations)
- **Practical training recipes** for both 4GB and 16GB GPU setups
- **Transparent experimentation** from early concept notes to reproducible scripts

## Highlights

- Built and analyzed a **sequence-level MoE routing pipeline** with explicit expert specialization controls.
- Implemented an **end-to-end 3-stage training workflow** (benchmark, expert pretrain, transfer) with reproducible runners.
- Added **interpretability-first evaluation tooling** (routing heatmaps, layer-wise expert usage, specialization ablations).
- Conducted **quantitative specialization analysis** using cosine distance and Jensen-Shannon metrics across training stages.
- Optimized experiments for **resource-constrained and mid-range hardware** (4GB and 16GB VRAM workflows).

Technical stack: Python, PyTorch, nanoGPT-style training loops, MoE routing, WandB logging, Matplotlib/Seaborn analysis.

## Repository Structure

- **`nanoGPT-LoRA-MoE-mod/`** – Main implementation
  - Training, pretraining, transfer, and evaluation scripts
  - Automated end-to-end workflow runners
  - Detailed READMEs: `README.md`, `README_16GB_WORKFLOW_1201A.txt`, `README_4GB.txt`
- **`A transparent MoE for sequence-level routing paper.pdf`** – Final research paper with results

## Main Workflow (Transparent MoE)

The central pipeline in `nanoGPT-LoRA-MoE-mod/` is a 3-stage workflow:

1. **Benchmark MoE training** on OpenWebText (baseline routing behavior)
2. **Subject-specific expert pretraining** with forced routing (specialization)
3. **Transfer training** back on general text (retain + blend specialization)

Evaluation includes:

- Per-subject routing heatmaps
- Layer-wise expert usage plots
- Leave-one-expert-out specialization ablations

## Final Paper Findings (Key Results)

Based on the final paper's empirical results:

- Baseline sequence-level MoE reproduces **weak domain specialization**, consistent with Fan et al. (2024).
- Pretraining experts on subject-specific data yields **stronger specialization than baseline** in early transfer learning.
- Specialization appears to **peak around 1200 iterations** then declines with continued general-data training.
- Quantitative metrics (Table 3 in paper):
  - **Benchmark baseline:** CosDist = 0.0669, JSM = 0.0333
  - **Transfer @1200 (peak):** CosDist = **0.1024**, JSM = **0.0477** (+53% improvement)
  - Transfer @3000: CosDist = 0.0876, JSM = 0.0420 (specialization decays over training)

Interpretation: Expert pre-initialization induces meaningful specialization, but continuous training on broad general-text data gradually dilutes domain expertise.

## Experiments Mapped to Paper Sections

| Paper Section | Experiment | Main Scripts | Output |
| --- | --- | --- | --- |
| 4.1 Baseline | Reproduce weak specialization | `train.py`, `eval.py` | Routing heatmaps + metrics |
| 4.2 Transfer | Expert pretraining + transfer | `pretrain.py`, `train.py` (resume) | Pretrained experts + transfer checkpoint |
| 4.5 Comparison | Quantify specialization metrics | `eval.py` | CosDist/JSM analysis |
| 4.3 Ablation | Leave-one-expert-out analysis | `eval_specialization.py` | Loss deltas + routing frequency |
| N/A Full pipeline | End-to-end reproducibility | `autorun_workflow.py`, `autorun_workflow_16gb.py` | Complete run with all outputs |

## Quick Start

### 1) Environment

```bash
cd nanoGPT-LoRA-MoE-mod
pip install torch transformers datasets wandb tiktoken matplotlib seaborn numpy
```

### 2) Prepare data

- OpenWebText binaries in `data/openwebtext/`
- MMLU subject binaries in `data/mmlu/`
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

- `nanoGPT-LoRA-MoE-mod/README.md`
- `nanoGPT-LoRA-MoE-mod/README_16GB_WORKFLOW_1201A.txt`
- `nanoGPT-LoRA-MoE-mod/README_4GB.txt`

## Implementation Status

- **Stable and reproducible:** All scripts in `nanoGPT-LoRA-MoE-mod/` have been tested on 4GB and 16GB setups
- **Paper-backed:** All experiments are documented in the final paper with quantitative results
- **Ready to use:** Automated workflows handle data prep, training, and evaluation

## Suggested Reading Order

1. [A transparent MoE for sequence-level routing paper.pdf](A%20transparent%20MoE%20for%20sequence-level%20routing%20paper.pdf) – Final project report
2. `nanoGPT-LoRA-MoE-mod/README.md` – Technical documentation
3. `nanoGPT-LoRA-MoE-mod/autorun_workflow_16gb.py` – Automated end-to-end pipeline
4. `nanoGPT-LoRA-MoE-mod/pretrain.py` – Expert pretraining logic
5. `nanoGPT-LoRA-MoE-mod/eval.py` – Routing analysis and visualization
6. `nanoGPT-LoRA-MoE-mod/eval_specialization.py` – Ablation studies and metrics

## License

See `nanoGPT-LoRA-MoE-mod/LICENSE` for implementation details.
