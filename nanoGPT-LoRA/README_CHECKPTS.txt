
-------------- Data -------------------------
OpenWebtext for general traning and benchmark

MMLU Auxillary Train Autolabelled for expert domain specific traning

MMLU Classi Questions for eval
python scripts/prepare_mmlu_questions.py --out_dir data/mmlu_questions --split test --include_choices --config all --subjects college_chemistry,global_facts,management,medical_genetics

------------- Run Steps ---------------------
if autorun:
python autorun_workflow.py --run_name demo1 --device cuda --samples_per_subject 100 --do_transfer_eval

1.1 Create MoE benchmark (no pretrained experts)
python train.py config/step1_benchmark_moe_4gb.py

1.2 Eval benchmark (heatmaps, staples (if we sepcify some layers to plot: --layers_to_plot all))
python eval.py --ckpt out/out-step1-benchmark-moe/ckpt.pt --questions_dir data/mmlu_questions --subjects college_chemistry,global_facts,management,medical_genetics --samples_per_subject 100 --device cuda --out_dir out/out-step1-benchmark-moe/eval --layers_to_plot all

2 Pretrain individual experts 
python .\pretrain.py --config config/step2_pretrain_4gb.py

3.1 Continue OpenWeb Train MoE (start from pretrained experts, not benchmark)

Option A: Merge exported experts, then train (preferred)
# Ensure experts were exported in Step 2 (directory: out/out-step2-pretrain/experts)
mkdir out\out-step3-transfer
python post_train.py --config config/step3_moe_transfer_4gb.py --experts_dir out/out-step2-pretrain/experts --subject_expert_map college_chemistry:0,global_facts:1,management:2,medical_genetics:3 --out_ckpt out/out-step3-transfer/ckpt.pt
python train.py config/step3_moe_transfer_4gb.py --out_dir=out/out-step3-transfer --resume_ckpt_path=out/out-step3-transfer/ckpt.pt --reset_iter

Option B: Skip merge and resume directly from pretrain final ckpt
mkdir out\out-step3-transfer
copy out\out-step2-pretrain\ckpt.pt out\out-step3-transfer\ckpt.pt
python train.py config/step3_moe_transfer_4gb.py --out_dir=out/out-step3-transfer --resume_ckpt_path=out/out-step3-transfer/ckpt.pt --reset_iter

3.2 Eval MoE (choose which checkpoint to evaluate)
- Evaluate pretrain:
python .\eval.py --ckpt out/out-step2-pretrain/ckpt.pt --questions_dir data/mmlu_questions --subjects college_chemistry,global_facts,management,medical_genetics --samples_per_subject 100 --device cuda --out_dir out/out-step2-pretrain/eval --layers_to_plot all
- Evaluate transfer:
python .\eval.py --ckpt out/out-step3-transfer/ckpt.pt --questions_dir data/mmlu_questions --subjects college_chemistry,global_facts,management,medical_genetics --samples_per_subject 100 --device cuda --out_dir out/out-step3-transfer/eval --layers_to_plot all