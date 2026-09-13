#!/usr/bin/env python3
"""Compute binary metrics for one Agent system without mixing legacy runs."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


POSITIVE = {"明显获益", "有限获益或稳定"}


def _load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def compute_metrics(cases: dict[str, dict[str, Any]], predictions: dict[str, str]) -> dict[str, Any]:
    tp = tn = fp = fn = 0
    rows = []
    for instance_id, prediction in sorted(predictions.items()):
        case = cases.get(instance_id)
        if not case:
            continue
        truth = str((case.get("ground_truth") or {}).get("overall_benefit") or "")
        true_positive = truth in POSITIVE
        pred_positive = prediction in POSITIVE
        if true_positive and pred_positive:
            tp += 1
        elif true_positive:
            fn += 1
        elif pred_positive:
            fp += 1
        else:
            tn += 1
        rows.append({
            "instance_id": instance_id,
            "ground_truth": truth,
            "prediction": prediction,
            "binary_correct": true_positive == pred_positive,
        })
    total = tp + tn + fp + fn
    return {
        "n": total,
        "confusion": {"TP": tp, "FN": fn, "TN": tn, "FP": fp},
        "positive": {"correct": tp, "count": tp + fn, "accuracy": tp / (tp + fn) if tp + fn else None},
        "negative": {"correct": tn, "count": tn + fp, "accuracy": tn / (tn + fp) if tn + fp else None},
        "overall": {"correct": tp + tn, "count": total, "accuracy": (tp + tn) / total if total else None},
        "wrong_instance_ids": [row["instance_id"] for row in rows if not row["binary_correct"]],
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-system-id", required=True)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint-root", type=Path, action="append", required=True)
    parser.add_argument("--allow-unlabeled", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cases: dict[str, dict[str, Any]] = {}
    for path in args.data:
        for row in _load(path, []):
            if isinstance(row, dict) and row.get("instance_id"):
                cases[str(row["instance_id"])] = row
    predictions: dict[str, str] = {}
    excluded_other_system = []
    for root in args.checkpoint_root:
        for path in sorted(root.rglob("case_*.json")):
            row = _load(path, {})
            system_id = str(row.get("agent_system_id") or "")
            if system_id != args.agent_system_id and not (args.allow_unlabeled and not system_id):
                excluded_other_system.append(str(path))
                continue
            result = row.get("result") or {}
            parsed = result.get("parsed_output") if isinstance(result, dict) else None
            if row.get("status") != "success" or not isinstance(parsed, dict):
                continue
            predictions[str(row.get("instance_id"))] = str(parsed.get("overall_benefit") or "")
    metrics = compute_metrics(cases, predictions)
    payload = {
        "agent_system_id": args.agent_system_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "eligibility": "checkpoint status=success; complete LLM only; no fallback/degraded",
        "excluded_other_system_records": len(excluded_other_system),
        **metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("agent_system_id", "n", "positive", "negative", "overall")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
