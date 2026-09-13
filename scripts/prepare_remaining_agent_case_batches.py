#!/usr/bin/env python3
"""Exclude the original holdout PMIDs and create reproducible source batches."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step1", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-pmids", type=int, default=32)
    parser.add_argument("--original-questions", type=int, default=97)
    args = parser.parse_args()

    holdout_doc = json.loads(args.holdout.read_text(encoding="utf-8"))
    holdout = {str(x) for x in holdout_doc["pmids"]}
    rows = []
    for path in args.step1.glob("*.json"):
        match = re.match(r"(\d+)_tr\.json$", path.name)
        if not match:
            continue
        pmid = match.group(1)
        rows.append({"pmid": pmid, "source_file": str(path.resolve())})
    rows.sort(key=lambda x: int(x["pmid"]))
    all_pmids = {x["pmid"] for x in rows}
    missing_holdout = sorted(holdout - all_pmids, key=int)
    remaining = [x for x in rows if x["pmid"] not in holdout]

    args.output.mkdir(parents=True, exist_ok=True)
    batches = []
    for offset in range(0, len(remaining), args.batch_pmids):
        chunk = remaining[offset : offset + args.batch_pmids]
        index = offset // args.batch_pmids + 1
        name = f"remaining_pmids_part_{index:02d}.json"
        payload = {
            "part": index,
            "n_pmids": len(chunk),
            "pmids": [x["pmid"] for x in chunk],
            "source_files": [x["source_file"] for x in chunk],
        }
        (args.output / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        batches.append({"part": index, "file": name, "n_pmids": len(chunk)})

    density = args.original_questions / len(holdout)
    summary = {
        "step1_total_pmids": len(rows),
        "original_holdout_pmids": len(holdout),
        "original_benchmark_questions": args.original_questions,
        "original_questions_per_pmid": density,
        "remaining_pmids": len(remaining),
        "batch_target_pmids": args.batch_pmids,
        "n_parts": len(batches),
        "full_parts": len(remaining) // args.batch_pmids,
        "last_part_pmids": len(remaining) % args.batch_pmids,
        "estimated_remaining_questions_at_original_density": round(len(remaining) * density),
        "estimated_total_questions_including_original": round(len(rows) * density),
        "estimate_warning": "Exact question count requires treatment-decision-node extraction; this is proportional only.",
        "missing_holdout_pmids_in_step1": missing_holdout,
        "batches": batches,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
