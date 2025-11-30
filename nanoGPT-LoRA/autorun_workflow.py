#!/usr/bin/env python
"""
Automated end-to-end workflow for the MoE benchmark + expert pretraining + transfer runs.

This script encapsulates the manual steps listed in README_CHECKPTS.txt:
  1. Prepare MMLU question .txt files
  2. Benchmark MoE training (Step 1)
  3. Benchmark evaluation (heatmaps + staple diagrams)
  4. Subject-specific expert pretraining
  5. Pretrain checkpoint evaluation
  6. Transfer (resume) training (Step 3) continuing from benchmark ckpt
  7. (Optional) Transfer checkpoint evaluation

Directory naming is standardized using --run_name:
  benchmark out: out/<run_name>_benchmark
  pretrain  out: out/<run_name>_pretrain
  transfer  out: same as benchmark (resume into it)
  eval outputs land inside <out_dir>/eval for each evaluated checkpoint.

Usage (PowerShell example):
  python autorun_workflow.py --run_name demo1 --device cuda --samples_per_subject 100 --do_transfer_eval

You can skip phases with flags, or override defaults.
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
    ap = argparse.ArgumentParser(description="Automated MoE workflow runner")
    ap.add_argument('--run_name', required=True, help='Base name for output folders')
    ap.add_argument('--subjects', default=DEFAULT_SUBJECTS, help='Comma list of MMLU subjects to use')
    ap.add_argument('--samples_per_subject', type=int, default=100, help='Samples used per subject during eval')
    ap.add_argument('--benchmark_config', default='config/step1_benchmark_moe_4gb.py', help='Config file for benchmark training')
    ap.add_argument('--transfer_config', default='config/step3_moe_transfer_4gb.py', help='Config file for transfer (resume) training')
    ap.add_argument('--device', default='cuda', help='Device for evaluation and training')
    ap.add_argument('--question_split', default='test', choices=['test','val','train'], help='Split for question preparation script')
    ap.add_argument('--prepare_questions', action='store_true', help='Force re-generation of questions even if folder exists')
    ap.add_argument('--skip_questions', action='store_true', help='Skip question preparation step')
    ap.add_argument('--skip_benchmark', action='store_true', help='Skip benchmark training')
    ap.add_argument('--skip_benchmark_eval', action='store_true', help='Skip benchmark evaluation')
    ap.add_argument('--skip_pretrain', action='store_true', help='Skip expert pretraining')
    ap.add_argument('--skip_pretrain_eval', action='store_true', help='Skip evaluation of pretrain checkpoint')
    ap.add_argument('--skip_transfer', action='store_true', help='Skip transfer (resume) training')
    ap.add_argument('--do_transfer_eval', action='store_true', help='Evaluate transfer checkpoint (not in original README)')
    ap.add_argument('--transfer_seed', choices=['pretrain_ckpt','merged_experts'], default='pretrain_ckpt', help='Seed transfer from pretrain final ckpt or from merged experts via post_train.py')
    ap.add_argument('--experts_dir', default=None, help='Override experts dir (defaults to <pretrain_out>/experts) when using merged_experts')
    # Pretrain overrides
    ap.add_argument('--pretrain_iters_per_subject', type=int, default=200, help='Pretrain iterations per subject')
    ap.add_argument('--pretrain_block_size', type=int, default=128)
    ap.add_argument('--pretrain_batch_size', type=int, default=4)
    ap.add_argument('--pretrain_n_layer', type=int, default=4)
    ap.add_argument('--pretrain_n_head', type=int, default=4)
    ap.add_argument('--pretrain_n_embd', type=int, default=128)
    ap.add_argument('--pretrain_n_expert', type=int, default=4)
    ap.add_argument('--pretrain_n_routed_expert', type=int, default=1)
    ap.add_argument('--pretrain_subject_map', default='college_chemistry:0,global_facts:1,management:2,medical_genetics:3', help='Mapping for forced expert assignment in pretrain')
    ap.add_argument('--pretrain_export_experts', action='store_true', help='Export per-expert weights')
    ap.add_argument('--layers_to_plot', default='all', help='Layer selection for staple diagrams in eval ("all" or comma list)')
    ap.add_argument('--dry_run', action='store_true', help='Print planned commands and exit without running')
    return ap.parse_args()

def main():
    args = parse_args()

    subjects_csv = args.subjects
    benchmark_out = f"out/{args.run_name}_benchmark"
    pretrain_out = f"out/{args.run_name}_step2_pretrain"
    transfer_out = f"out/{args.run_name}_transfer"  # keep transfer separate from benchmark
    questions_dir = REPO_ROOT / 'data' / 'mmlu_questions'
    log_dir = REPO_ROOT / 'out' / f"{args.run_name}_workflow_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'workflow.log'

    planned = []

    # 1. Prepare questions (if needed)
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
            print("[INFO] Questions already present; skipping generation. Use --prepare_questions to force.")

    # 2. Benchmark training
    if not args.skip_benchmark:
        cmd = [
            'python', 'train.py', args.benchmark_config,
            f"--out_dir={benchmark_out}",
        ]
        planned.append(('benchmark_train', cmd))

    # 3. Benchmark eval
    if not args.skip_benchmark_eval:
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

    # 4. Pretrain experts
    if not args.skip_pretrain:
        cmd = [
            'python', 'pretrain.py',
            '--data_dir', 'data/mmlu',
            '--out_dir', pretrain_out,
            '--subjects', subjects_csv,
            '--iters_per_subject', str(args.pretrain_iters_per_subject),
            '--block_size', str(args.pretrain_block_size),
            '--batch_size', str(args.pretrain_batch_size),
            '--n_layer', str(args.pretrain_n_layer),
            '--n_head', str(args.pretrain_n_head),
            '--n_embd', str(args.pretrain_n_embd),
            '--n_expert', str(args.pretrain_n_expert),
            '--n_routed_expert', str(args.pretrain_n_routed_expert),
            '--subject_expert_map', args.pretrain_subject_map,
            '--freeze_non_target_experts',
        ]
        if args.pretrain_export_experts:
            cmd.append('--export_experts')
        planned.append(('pretrain', cmd))

    # 5. Pretrain eval
    if not args.skip_pretrain_eval:
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

    # 6. Transfer training (resume)
    if not args.skip_transfer:
        # Ensure transfer dir exists and seed it with the chosen checkpoint
        transfer_dir = Path(transfer_out)
        transfer_dir.mkdir(parents=True, exist_ok=True)
        seed_ckpt = transfer_dir / 'ckpt.pt'
        if args.transfer_seed == 'merged_experts':
            # Build ckpt by merging exported experts
            exp_dir = Path(args.experts_dir) if args.experts_dir else Path(pretrain_out) / 'experts'
            transfer_dir.mkdir(parents=True, exist_ok=True)
            cmd_merge = [
                'python', 'post_train.py',
                '--config', args.transfer_config,
                '--experts_dir', str(exp_dir),
                '--subject_expert_map', args.pretrain_subject_map,
                '--out_ckpt', str(seed_ckpt),
            ]
            planned.append(('post_train_merge', cmd_merge))
        else:
            # Default: copy pretrain final ckpt as seed
            pre_ckpt = Path(pretrain_out) / 'ckpt.pt'
            if pre_ckpt.exists() and not seed_ckpt.exists():
                import shutil
                print(f"[INFO] Seeding transfer directory with pretrain ckpt: {pre_ckpt} -> {seed_ckpt}")
                shutil.copy2(pre_ckpt, seed_ckpt)
        cmd = [
            'python', 'train.py', args.transfer_config,
            f"--out_dir={transfer_out}",  # resume from transfer directory (seeded with benchmark ckpt)
            f"--resume_ckpt_path={seed_ckpt}",
        ]
        planned.append(('transfer_train', cmd))

    # 7. Transfer eval (optional)
    if args.do_transfer_eval:
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

    # Dry run
    if args.dry_run:
        print("Planned commands (dry-run):")
        for tag, cmd in planned:
            print(f"  [{tag}] {' '.join(cmd)}")
        return

    summary = {
        'run_name': args.run_name,
        'subjects': subjects_csv.split(','),
        'benchmark_out_dir': benchmark_out,
        'pretrain_out_dir': pretrain_out,
        'transfer_out_dir': transfer_out,
        'steps': [],
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    for tag, cmd in planned:
        t0 = time.time()
        try:
            run(cmd, REPO_ROOT, log_file)
            status = 'ok'
        except Exception as e:
            status = f'failed: {e}'
            print(f"[ERROR] Step '{tag}' failed: {e}")
            summary['steps'].append({'step': tag, 'cmd': cmd, 'status': status, 'duration_sec': round(time.time()-t0,2)})
            break  # stop pipeline on first failure
        else:
            summary['steps'].append({'step': tag, 'cmd': cmd, 'status': status, 'duration_sec': round(time.time()-t0,2)})

    summary['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    summary_path = log_dir / 'summary.json'
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\nWorkflow summary written to {summary_path}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Fatal error: {e}")
        raise
