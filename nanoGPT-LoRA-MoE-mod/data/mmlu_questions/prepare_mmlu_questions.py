"""
Prepare per-category MMLU question files for eval_routing_mmlu.py

- Downloads standard MMLU from HuggingFace (cais/mmlu or hendrycks/mmlu)
- Writes one text file per category under data/mmlu_questions/<category>.txt
  with one question per line (optionally include choices)

Usage (PowerShell):
  python scripts/prepare_mmlu_questions.py --out_dir data/mmlu_questions --split test --include_choices

Then run routing eval, e.g.:
  python eval_routing_mmlu.py --ckpt out/out-step1-benchmark-moe/ckpt.pt \
    --questions_dir data/mmlu_questions \
    --categories global_facts college_biology college_chemistry medical_genetics management \
    --samples_per_category 100 --device cuda --out_dir out/out-step1-benchmark-moe/routing_analysis
"""
import os
import argparse

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


def fetch_mmlu(split: str):
    if load_dataset is None:
        raise RuntimeError("`datasets` not installed. Run: pip install datasets")
    # try common ids
    for ds_id in ["cais/mmlu", "hendrycks/mmlu", "mmlu"]:
        try:
            print(f"Trying dataset id: {ds_id}")
            # some ids require subset; try generic
            ds = load_dataset(ds_id)
            # choose split if present, else fallback to any
            if split in ds:
                return ds[split]
            # if not, pick first available split
            first_split = next(iter(ds.keys()))
            return ds[first_split]
        except Exception as e:
            print(f"Failed to load {ds_id}: {e}")
    raise RuntimeError("Could not load MMLU from known dataset ids.")


def build_question_line(row, include_choices: bool):
    q = row.get("question") or row.get("input") or row.get("prompt") or ""
    if include_choices:
        choices = row.get("choices") or row.get("options")
        if isinstance(choices, (list, tuple)):
            try:
                letters = [f"{chr(65+i)}. {c}" for i, c in enumerate(choices)]
                q = f"{q}\n" + " ".join(letters)
            except Exception:
                q = f"{q}\n" + " ".join(map(str, choices))
    return (q or "").replace("\n", " ").strip()


def main():
    ap = argparse.ArgumentParser(description="Prepare MMLU per-category question files")
    ap.add_argument("--out_dir", default="data/mmlu_questions", help="Output directory for <category>.txt files")
    ap.add_argument("--split", default="test", choices=["train", "validation", "val", "test"], help="Dataset split")
    ap.add_argument("--include_choices", action="store_true", help="Append choices to each question line")
    ap.add_argument("--max_per_category", type=int, default=None, help="Cap number of questions per category")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ds = fetch_mmlu(args.split)

    # determine field names
    sample = ds[0]
    subject_key = None
    for k in ("subject", "Category", "task", "category"):
        if k in sample:
            subject_key = k
            break
    if subject_key is None:
        raise RuntimeError("Could not find subject/category field in dataset")

    print(f"Using subject field: {subject_key}")
    # group by subject
    grouped = {}
    for row in ds:
        cat = row.get(subject_key)
        if not cat:
            cat = "unknown"
        grouped.setdefault(cat, []).append(row)

    # write files
    for cat, rows in grouped.items():
        out_path = os.path.join(args.out_dir, f"{cat}.txt")
        n = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                line = build_question_line(row, include_choices=args.include_choices)
                if not line:
                    continue
                f.write(line + "\n")
                n += 1
                if args.max_per_category and n >= args.max_per_category:
                    break
        print(f"Wrote {n} lines -> {out_path}")

    print("Done. Files ready in", args.out_dir)


if __name__ == "__main__":
    main()
