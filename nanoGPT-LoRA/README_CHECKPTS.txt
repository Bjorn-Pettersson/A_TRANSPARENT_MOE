
-------------- Data -------------------------
OpenWebtext for general traning and benchmark

MMLU Auxillary Train Autolabelled for expert domain specific traning

MMLU Classi Questions for eval
python scripts/prepare_mmlu_questions.py --out_dir data/mmlu_questions --split test --include_choices --config all --subjects college_chemistry,global_facts,management,medical_genetics

------------- Run Steps ---------------------
1.1 Create MoE benchmark (no pretrained experts)
python train.py config/step1_benchmark_moe_4gb.py

1.2 Eval benchmark (heatmaps, staples (if we sepcify some layers to plot: --layers_to_plot all))
python eval_mmlu.py --experiment_dir out/out-step1-benchmark-moe --data_dir data/mmlu --device cuda --layers_to_plot all

2 Pretrain individual experts 
python pretrain.py --data_dir data\mmlu --out_dir out-pretrain --subjects college_chemistry,global_facts,management,medical_genetics --iters_per_subject 200 --block_size 128 --batch_size 4 --n_layer 4 --n_head 4 --n_embd 128 --n_expert 4 --n_routed_expert 1 --subject_expert_map college_chemistry:0,global_facts:1,management:2,medical_genetics:3 --freeze_non_target_experts --export_experts

3.1 Post train MoE OpenWebtext (resumes from last model made in out)
python train.py config/step3_moe_transfer_4gb.py

3.2 Eval MoE
python eval_routing_mmlu.py --ckpt out-pretrain/ckpt.pt --questions_dir data/mmlu_questions --categories college_chemistry global_facts management medical_genetics --samples_per_category 100 --device cuda --out_dir out-pretrain/eval --layers_to_plot all