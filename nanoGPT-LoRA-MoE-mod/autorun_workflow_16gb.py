#!/usr/bin/env python
"""
Automated end-to-end workflow for 16GB GPU MoE experiments (1201A series).

This script orchestrates the complete pipeline for larger-scale MoE experiments:
  1. Prepare MMLU question .txt files
  2. Benchmark MoE training (Step 1) - GPT2-small, 3000 iters
  3. Benchmark evaluation (heatmaps + staple diagrams)
  4. Subject-specific expert pretraining (600 iters/subject)
  5. Pretrain checkpoint evaluation + ablation study
  6. Transfer (resume) training (Step 3) - 2000 iters on OpenWebText
  7. (Optional) Transfer checkpoint evaluation

Configuration:
  - Model: GPT2-small (12 layers, 768 dim, 4 experts)
  - Load balancing lambda: 0.003 (reduced for specialization)
  - Block size: 512 tokens
  - Gradient accumulation optimized for 16GB VRAM

Directory naming is standardized using --run_name:
  benchmark out: out/out-<run_name>-benchmark
  pretrain  out: out/out-<run_name>-pretrain
  transfer  out: out/out-<run_name>-transfer

Usage (PowerShell example):
  python autorun_workflow_16gb.py --run_name s1_16gb_1201A --device cuda --samples_per_subject 100 --do_transfer_eval

For custom load balancing:
  python autorun_workflow_16gb.py --run_name my_run --load_balancing_lambda 0.005
"""
from __future__ import annotations
import os
import sys
import time
import json
import shlex
import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()

DEFAULT_SUBJECTS = "college_chemistry,global_facts,management,medical_genetics"

def run(cmd: list[str] | str, cwd: Path, log_file: Path, env: dict | None = None):
    """Run a command, stream output, append to log file. Raises on failure."""
    if isinstance(cmd, str):
        display = cmd
        shell = True
    else:
        display = " ".join(shlex.quote(c) for c in cmd)
        shell = False
    print(f"\n[RUN] {display}")
    with log_file.open('a', encoding='utf-8') as lf:
        lf.write(f"\n===== CMD: {display} =====\n")
        lf.flush()
        proc = subprocess.Popen(cmd if not shell else display, cwd=str(cwd), shell=shell,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        for line in proc.stdout:  # type: ignore[arg-type]
            text = line.decode(errors='replace')
            print(text, end='')
            lf.write(text)
        ret = proc.wait()
        lf.write(f"\n===== EXIT CODE: {ret} =====\n")
    if ret != 0:
        raise RuntimeError(f"Command failed (exit {ret}): {display}")

def parse_args():
    ap = argparse.ArgumentParser(description="Automated 16GB MoE workflow runner")
    ap.add_argument('--run_name', required=True, help='Base name for output folders (e.g., s1_16gb_1201A)')
    ap.add_argument('--subjects', default=DEFAULT_SUBJECTS, help='Comma list of MMLU subjects')
    ap.add_argument('--samples_per_subject', type=int, default=100, help='Samples per subject during eval')
    ap.add_argument('--device', default='cuda', help='Device for training and evaluation')
    ap.add_argument('--question_split', default='test', choices=['test','val','train'], help='Split for MMLU questions')
    
    # Step control flags
    ap.add_argument('--prepare_questions', action='store_true', help='Force re-generation of MMLU questions')
    ap.add_argument('--skip_questions', action='store_true', help='Skip question preparation')
    ap.add_argument('--skip_benchmark', action='store_true', help='Skip benchmark training')
    ap.add_argument('--skip_benchmark_eval', action='store_true', help='Skip benchmark evaluation')
    ap.add_argument('--skip_pretrain', action='store_true', help='Skip expert pretraining')
    ap.add_argument('--skip_pretrain_eval', action='store_true', help='Skip pretrain evaluation')
    ap.add_argument('--skip_ablation', action='store_true', help='Skip ablation study')
    ap.add_argument('--skip_transfer', action='store_true', help='Skip transfer training')
    ap.add_argument('--do_transfer_eval', action='store_true', help='Evaluate transfer checkpoint')
    
    # Transfer seeding strategy
    ap.add_argument('--transfer_seed', choices=['pretrain_ckpt','merged_experts'], default='pretrain_ckpt',
                    help='Seed transfer from pretrain ckpt or merged experts')
    ap.add_argument('--experts_dir', default=None, help='Override experts dir for merging')
    
    # 16GB Model parameters (GPT2-small defaults)
    ap.add_argument('--n_layer', type=int, default=12, help='Number of transformer layers')
    ap.add_argument('--n_head', type=int, default=12, help='Number of attention heads')
    ap.add_argument('--n_embd', type=int, default=768, help='Embedding dimension')
    ap.add_argument('--n_expert', type=int, default=4, help='Number of experts')
    ap.add_argument('--block_size', type=int, default=512, help='Context length')
    
    # Training parameters
    ap.add_argument('--benchmark_iters', type=int, default=3000, help='Benchmark training iterations')
    ap.add_argument('--pretrain_iters_per_subject', type=int, default=600, help='Pretrain iterations per subject')
    ap.add_argument('--transfer_iters', type=int, default=2000, help='Transfer training iterations')
    ap.add_argument('--batch_size', type=int, default=4, help='Micro-batch size')
    ap.add_argument('--grad_accum_benchmark', type=int, default=32, help='Gradient accumulation for benchmark')
    ap.add_argument('--grad_accum_pretrain', type=int, default=16, help='Gradient accumulation for pretrain')
    
    # MoE parameters
    ap.add_argument('--load_balancing_lambda', type=float, default=0.003, 
                    help='Load balancing loss weight (0.003 for better specialization)')
    ap.add_argument('--n_routed_expert_pretrain', type=int, default=1, help='Top-K for pretrain (force single)')
    ap.add_argument('--n_routed_expert_transfer', type=int, default=2, help='Top-K for benchmark/transfer')
    ap.add_argument('--subject_expert_map', default='college_chemistry:0,global_facts:1,management:2,medical_genetics:3',
                    help='Subject to expert index mapping')
    
    # Evaluation parameters
    ap.add_argument('--layers_to_plot', default='all', help='Layers for staple plots ("all" or comma list)')
    ap.add_argument('--ablation_steps', type=int, default=100, help='Steps per subject for ablation study')
    
    # System
    ap.add_argument('--dtype', default='bfloat16', choices=['float32','bfloat16','float16'], help='Data type')
    ap.add_argument('--dry_run', action='store_true', help='Print commands without executing')
    
    return ap.parse_args()

def main():
    args = parse_args()

    subjects_csv = args.subjects
    benchmark_out = f"out/out-{args.run_name}-benchmark"
    pretrain_out = f"out/out-{args.run_name}-pretrain"
    transfer_out = f"out/out-{args.run_name}-transfer"
    questions_dir = REPO_ROOT / 'data' / 'mmlu_questions'
    log_dir = REPO_ROOT / 'out' / f"{args.run_name}_workflow_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'workflow.log'

    planned = []

    # ========== 1. Prepare MMLU questions ==========
    if not args.skip_questions:
        need_questions = args.prepare_questions or not questions_dir.exists() or not any(questions_dir.glob('*.txt'))
        if need_questions:
            cmd = [
                'python', 'scripts/prepare_mmlu_questions.py',
                '--out_dir', str(questions_dir),
                '--split', args.question_split,
                '--include_choices',
                '--config', 'all',
                '--subjects', subjects_csv,
            ]
            planned.append(('prepare_questions', cmd))
        else:
            print("[INFO] MMLU questions already present; skipping. Use --prepare_questions to force.")

    # ========== 2. Benchmark Training (Step 1) ==========
    if not args.skip_benchmark:
        print(f"[INFO] Step 1: Benchmark training ({args.benchmark_iters} iters)")
        cmd = [
            'python', 'train.py',
            f'--out_dir={benchmark_out}',
            '--init_from=scratch',
            f'--n_layer={args.n_layer}',
            f'--n_head={args.n_head}',
            f'--n_embd={args.n_embd}',
            f'--n_expert={args.n_expert}',
            f'--n_routed_expert={args.n_routed_expert_transfer}',
            f'--load_balancing_lambda={args.load_balancing_lambda}',
            f'--block_size={args.block_size}',
            f'--batch_size={args.batch_size}',
            f'--gradient_accumulation_steps={args.grad_accum_benchmark}',
            f'--max_iters={args.benchmark_iters}',
            '--dataset=openwebtext',
            f'--device={args.device}',
            f'--dtype={args.dtype}',
            '--wandb_log=True',
            f'--wandb_run_name={args.run_name}_benchmark',
            '--wandb_project=moe-understanding',
        ]
        planned.append(('benchmark_train', cmd))

    # ========== 3. Benchmark Evaluation ==========
    if not args.skip_benchmark_eval:
        print(f"[INFO] Step 1 Eval: Benchmark routing analysis")
        ckpt_path = f"{benchmark_out}/ckpt.pt"
        cmd = [
            'python', 'eval.py',
            '--ckpt', ckpt_path,
            '--questions_dir', str(questions_dir),
            '--subjects', subjects_csv,
            '--samples_per_subject', str(args.samples_per_subject),
            '--device', args.device,
            '--out_dir', f"{benchmark_out}/eval",
            '--layers_to_plot', args.layers_to_plot,
        ]
        planned.append(('benchmark_eval', cmd))

    # ========== 4. Expert Pretraining (Step 2) ==========
    if not args.skip_pretrain:
        print(f"[INFO] Step 2: Expert pretraining ({args.pretrain_iters_per_subject} iters/subject)")
        cmd = [
            'python', 'pretrain.py',
            '--data_dir', 'data/mmlu',
            '--out_dir', pretrain_out,
            '--subjects', subjects_csv,
            '--iters_per_subject', str(args.pretrain_iters_per_subject),
            '--block_size', str(args.block_size),
            '--batch_size', str(args.batch_size),
            '--gradient_accumulation_steps', str(args.grad_accum_pretrain),
            '--n_layer', str(args.n_layer),
            '--n_head', str(args.n_head),
            '--n_embd', str(args.n_embd),
            '--n_expert', str(args.n_expert),
            '--n_routed_expert', str(args.n_routed_expert_pretrain),
            '--subject_expert_map', args.subject_expert_map,
            '--freeze_non_target_experts',
            '--export_experts',
            '--load_balancing_lambda', str(args.load_balancing_lambda),
            '--device', args.device,
            '--dtype', args.dtype,
            '--wandb_log',
            '--wandb_run_name', f'{args.run_name}_pretrain',
            '--wandb_project', 'moe-understanding',
        ]
        planned.append(('pretrain', cmd))

    # ========== 5. Pretrain Evaluation ==========
    if not args.skip_pretrain_eval:
        print(f"[INFO] Step 2 Eval: Pretrain routing analysis")
        ckpt_path = f"{pretrain_out}/ckpt.pt"
        cmd = [
            'python', 'eval.py',
            '--ckpt', ckpt_path,
            '--questions_dir', str(questions_dir),
            '--subjects', subjects_csv,
            '--samples_per_subject', str(args.samples_per_subject),
            '--device', args.device,
            '--out_dir', f"{pretrain_out}/eval",
            '--layers_to_plot', args.layers_to_plot,
        ]
        planned.append(('pretrain_eval', cmd))

    # ========== 6. Ablation Study (Optional) ==========
    if not args.skip_ablation:
        print(f"[INFO] Step 2 Ablation: Expert specialization analysis")
        ckpt_path = f"{pretrain_out}/ckpt.pt"
        cmd = [
            'python', 'eval_specialization.py',
            '--ckpt', ckpt_path,
            '--data_dir', 'data/mmlu',
            '--subjects', subjects_csv,
            '--steps_per_subject', str(args.ablation_steps),
            '--batch_size', str(args.batch_size),
            '--device', args.device,
            '--out_dir', f"{pretrain_out}/ablation",
        ]
        planned.append(('ablation_study', cmd))

    # ========== 7. Transfer Training (Step 3) ==========
    if not args.skip_transfer:
        print(f"[INFO] Step 3: Transfer training ({args.transfer_iters} iters)")
        transfer_dir = Path(transfer_out)
        transfer_dir.mkdir(parents=True, exist_ok=True)
        seed_ckpt = transfer_dir / 'ckpt.pt'
        
        if args.transfer_seed == 'merged_experts':
            # Merge exported experts into clean checkpoint
            exp_dir = Path(args.experts_dir) if args.experts_dir else Path(pretrain_out) / 'experts'
            cmd_merge = [
                'python', 'post_train.py',
                '--config', f'config/{args.run_name}.py',  # Use step 1 config for architecture
                '--experts_dir', str(exp_dir),
                '--subject_expert_map', args.subject_expert_map,
                '--out_ckpt', str(seed_ckpt),
            ]
            planned.append(('post_train_merge', cmd_merge))
        else:
            # Copy pretrain checkpoint as seed
            pre_ckpt = Path(pretrain_out) / 'ckpt.pt'
            if pre_ckpt.exists() and not seed_ckpt.exists():
                import shutil
                print(f"[INFO] Seeding transfer with pretrain checkpoint: {pre_ckpt}")
                shutil.copy2(pre_ckpt, seed_ckpt)
        
        # Resume training on OpenWebText
        cmd = [
            'python', 'train.py',
            f'--out_dir={transfer_out}',
            f'--resume_ckpt_path={seed_ckpt}',
            '--init_from=resume',
            '--dataset=openwebtext',
            f'--n_layer={args.n_layer}',
            f'--n_head={args.n_head}',
            f'--n_embd={args.n_embd}',
            f'--n_expert={args.n_expert}',
            f'--n_routed_expert={args.n_routed_expert_transfer}',
            f'--load_balancing_lambda={args.load_balancing_lambda}',
            f'--block_size={args.block_size}',
            f'--batch_size={args.batch_size}',
            f'--gradient_accumulation_steps={args.grad_accum_benchmark}',
            f'--max_iters={args.transfer_iters}',
            f'--device={args.device}',
            f'--dtype={args.dtype}',
            '--wandb_log=True',
            f'--wandb_run_name={args.run_name}_transfer',
            '--wandb_project=moe-understanding',
        ]
        planned.append(('transfer_train', cmd))

    # ========== 8. Transfer Evaluation (Optional) ==========
    if args.do_transfer_eval:
        print(f"[INFO] Step 3 Eval: Transfer routing analysis")
        ckpt_path = f"{transfer_out}/ckpt.pt"
        cmd = [
            'python', 'eval.py',
            '--ckpt', ckpt_path,
            '--questions_dir', str(questions_dir),
            '--subjects', subjects_csv,
            '--samples_per_subject', str(args.samples_per_subject),
            '--device', args.device,
            '--out_dir', f"{transfer_out}/eval",
            '--layers_to_plot', args.layers_to_plot,
        ]
        planned.append(('transfer_eval', cmd))

    # ========== Dry Run ==========
    if args.dry_run:
        print("\n" + "="*70)
        print("PLANNED COMMANDS (DRY RUN)")
        print("="*70)
        for i, (tag, cmd) in enumerate(planned, 1):
            print(f"\n[{i}] {tag.upper()}")
            if isinstance(cmd, list):
                print("  " + " ".join(cmd))
            else:
                print("  " + cmd)
        print("\n" + "="*70)
        print(f"Total steps: {len(planned)}")
        print("="*70)
        return

    # ========== Execute Pipeline ==========
    summary = {
        'run_name': args.run_name,
        'subjects': subjects_csv.split(','),
        'benchmark_out_dir': benchmark_out,
        'pretrain_out_dir': pretrain_out,
        'transfer_out_dir': transfer_out,
        'config': {
            'n_layer': args.n_layer,
            'n_head': args.n_head,
            'n_embd': args.n_embd,
            'n_expert': args.n_expert,
            'block_size': args.block_size,
            'load_balancing_lambda': args.load_balancing_lambda,
            'benchmark_iters': args.benchmark_iters,
            'pretrain_iters_per_subject': args.pretrain_iters_per_subject,
            'transfer_iters': args.transfer_iters,
        },
        'steps': [],
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    print("\n" + "="*70)
    print(f"STARTING 16GB WORKFLOW: {args.run_name}")
    print("="*70)
    print(f"Model: {args.n_layer}L/{args.n_embd}D, {args.n_expert} experts")
    print(f"Lambda: {args.load_balancing_lambda}")
    print(f"Subjects: {subjects_csv}")
    print(f"Total steps: {len(planned)}")
    print("="*70 + "\n")

    for idx, (tag, cmd) in enumerate(planned, 1):
        print(f"\n{'='*70}")
        print(f"STEP {idx}/{len(planned)}: {tag.upper()}")
        print(f"{'='*70}")
        t0 = time.time()
        try:
            run(cmd, REPO_ROOT, log_file)
            status = 'success'
            duration = round(time.time() - t0, 2)
            print(f"✓ {tag} completed in {duration:.1f}s")
        except Exception as e:
            status = f'failed: {e}'
            duration = round(time.time() - t0, 2)
            print(f"✗ {tag} failed after {duration:.1f}s: {e}")
            summary['steps'].append({
                'step': tag,
                'cmd': cmd if isinstance(cmd, str) else ' '.join(cmd),
                'status': status,
                'duration_sec': duration
            })
            break  # Stop on first failure
        else:
            summary['steps'].append({
                'step': tag,
                'cmd': cmd if isinstance(cmd, str) else ' '.join(cmd),
                'status': status,
                'duration_sec': duration
            })

    summary['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    summary['total_duration_sec'] = sum(s['duration_sec'] for s in summary['steps'])
    
    summary_path = log_dir / 'summary.json'
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
    print(f"Total time: {summary['total_duration_sec']/3600:.2f} hours")
    print(f"Summary: {summary_path}")
    print(f"Logs: {log_file}")
    print("="*70 + "\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Workflow stopped by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
