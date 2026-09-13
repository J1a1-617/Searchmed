from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any


INSTRUCTION = (
    "Rank oncology evidence by exact clinical applicability. Require matching cancer population, "
    "target intervention, monotherapy versus combination regimen, requested time window, reported "
    "outcome, direct result text, and outcome attribution. Demote citations, background text, wrong "
    "regimens, and outcomes outside the requested time window."
)


def prepare(root: Path, index: Path, output: Path) -> None:
    groups: list[dict[str, Any]] = []
    conn = sqlite3.connect(index)
    for path in sorted(root.glob("**/workflow_trace.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for step in data.get("steps") or []:
            if step.get("name") != "Rerank" or not step.get("atomic_rerank_goal"):
                continue
            candidates = []
            for hit in step.get("top_hits") or []:
                dimensions = hit.get("llm_rerank_dimensions") or {}
                if hit.get("llm_relevance_score") is None or not dimensions:
                    continue
                row = conn.execute(
                    "SELECT text FROM evidence_chunks WHERE chunk_id = ?", (str(hit.get("id")),)
                ).fetchone()
                text = row[0] if row else hit.get("text_preview") or ""
                candidates.append({
                    "id": hit.get("id"),
                    "text": text,
                    "historical_score": float(hit["llm_relevance_score"]),
                    "hybrid_score": float(hit.get("hybrid_score") or 0.0),
                    "evidence_type": hit.get("llm_evidence_type") or "",
                    "dimensions": dimensions,
                })
            if len(candidates) >= 2 and len({x["historical_score"] for x in candidates}) >= 2:
                groups.append({
                    "source": str(path),
                    "query": step["atomic_rerank_goal"],
                    "candidates": candidates,
                })
    conn.close()
    output.write_text(json.dumps({"groups": groups}, ensure_ascii=False, indent=2))
    print(json.dumps({
        "groups": len(groups),
        "candidates": sum(len(x["candidates"]) for x in groups),
        "output": str(output),
    }, ensure_ascii=False))


def _dcg(labels: list[float]) -> float:
    return sum((2.0 ** value - 1.0) / math.log2(i + 2) for i, value in enumerate(labels))


def score(dataset: Path, model_path: Path, output: Path, batch_size: int) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    payload = json.loads(dataset.read_text())
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True
    ).eval()
    false_id = tokenizer.convert_tokens_to_ids("no")
    true_id = tokenizer.convert_tokens_to_ids("yes")
    prefix = (
        '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query '
        'and the Instruct provided. Note that the answer can only be "yes" or "no".'
        '<|im_end|>\n<|im_start|>user\n'
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    flat: list[tuple[dict[str, Any], str]] = []
    for group in payload["groups"]:
        for candidate in group["candidates"]:
            prompt = (
                prefix + f"<Instruct>: {INSTRUCTION}\n<Query>: {group['query']}\n"
                f"<Document>: {candidate['text']}" + suffix
            )
            flat.append((candidate, prompt))

    started = time.perf_counter()
    for start in range(0, len(flat), batch_size):
        batch = flat[start:start + batch_size]
        inputs = tokenizer(
            [x[1] for x in batch], padding=True, truncation=True, max_length=2048, return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
            probabilities = torch.softmax(
                torch.stack([logits[:, false_id], logits[:, true_id]], dim=1), dim=1
            )[:, 1].float().cpu().tolist()
        for (candidate, _), probability in zip(batch, probabilities):
            candidate["qwen_score"] = probability

    ndcgs, pair_correct, pair_total, top1, useful_rr, useful_recall3 = [], 0.0, 0, 0, [], []
    mismatch_correct = mismatch_total = 0
    for group in payload["groups"]:
        candidates = group["candidates"]
        qwen_order = sorted(candidates, key=lambda x: x["qwen_score"], reverse=True)
        ideal = sorted(candidates, key=lambda x: x["historical_score"], reverse=True)
        denominator = _dcg([x["historical_score"] for x in ideal])
        ndcgs.append(_dcg([x["historical_score"] for x in qwen_order]) / denominator if denominator else 1.0)
        top1 += qwen_order[0]["id"] == ideal[0]["id"]
        useful = {x["id"] for x in candidates if x["historical_score"] >= 0.5}
        if useful:
            rank = next(i for i, x in enumerate(qwen_order, 1) if x["id"] in useful)
            useful_rr.append(1.0 / rank)
            useful_recall3.append(len(useful.intersection(x["id"] for x in qwen_order[:3])) / len(useful))
        for i, left in enumerate(candidates):
            for right in candidates[i + 1:]:
                delta = left["historical_score"] - right["historical_score"]
                if delta:
                    qdelta = left["qwen_score"] - right["qwen_score"]
                    pair_correct += 1.0 if delta * qdelta > 0 else 0.5 if qdelta == 0 else 0.0
                    pair_total += 1
                def mismatch(item: dict[str, Any]) -> bool:
                    dims = item["dimensions"]
                    return (
                        item["evidence_type"] in {"background", "citation_pointer", "irrelevant"}
                        or dims.get("population_match", 0) == 0
                        or dims.get("intervention_match", 0) == 0
                        or dims.get("outcome_match", 0) == 0
                    )
                lm, rm = mismatch(left), mismatch(right)
                if lm != rm:
                    clean, bad = (right, left) if lm else (left, right)
                    mismatch_correct += clean["qwen_score"] > bad["qwen_score"]
                    mismatch_total += 1

    metrics = {
        "groups": len(payload["groups"]),
        "candidates": len(flat),
        "ndcg": sum(ndcgs) / len(ndcgs),
        "pairwise_accuracy": pair_correct / pair_total,
        "top1_agreement": top1 / len(payload["groups"]),
        "mrr_historical_score_ge_0_5": sum(useful_rr) / len(useful_rr) if useful_rr else None,
        "recall3_historical_score_ge_0_5": sum(useful_recall3) / len(useful_recall3) if useful_recall3 else None,
        "clean_over_mismatch_pair_accuracy": mismatch_correct / mismatch_total if mismatch_total else None,
        "pair_count": pair_total,
        "clean_mismatch_pair_count": mismatch_total,
        "inference_seconds": time.perf_counter() - started,
        "max_gpu_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
    }
    output.write_text(json.dumps({"metrics": metrics, **payload}, ensure_ascii=False, indent=2))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--root", type=Path, default=Path("."))
    prepare_parser.add_argument("--index", type=Path, default=Path("indexes/structured.db"))
    prepare_parser.add_argument("--output", type=Path, required=True)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--dataset", type=Path, required=True)
    score_parser.add_argument("--model", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root, args.index, args.output)
    else:
        score(args.dataset, args.model, args.output, args.batch_size)


if __name__ == "__main__":
    main()
