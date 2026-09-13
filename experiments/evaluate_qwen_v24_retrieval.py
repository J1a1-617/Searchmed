"""Evaluate Qwen reranking on the latest v24 RAG reference pools.

This is retrieval-only: it does not run the Agent or call the API.  A reference
is considered relevant when its recorded overall_benefit matches the target
case's ground truth.  Similarity order is the baseline; Qwen reranks the same
pool.
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
from searchagent_retrieval.local_rerank import Qwen3Reranker
from searchagent_retrieval.tools import SearchHit


def text_for(ref: dict) -> str:
    inp = ref.get("input", {})
    bg = inp.get("disease_background", {})
    status = inp.get("current_status", {})
    plan = inp.get("planned_treatment", {})
    gt = ref.get("ground_truth", {})
    drugs = ", ".join(str(x.get("name", x)) for x in (plan.get("drugs") or []))
    return " | ".join(str(x) for x in (
        bg.get("diagnosis"), bg.get("molecular_profile"), status.get("imaging"),
        status.get("csf"), drugs, plan.get("combination_strategy"),
        gt.get("overall_benefit"), gt.get("body_lesion_recist"),
        gt.get("cns_lm_recist"), gt.get("symptom_trajectory"),
    ) if x is not None)


def dcg(labels):
    return sum((2 ** int(label) - 1) / math.log2(i + 2) for i, label in enumerate(labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    data = json.loads(Path(args.data).read_text())
    reranker = Qwen3Reranker(args.model, device=args.device)
    rows = []
    for case in data:
        target = case.get("ground_truth", {}).get("overall_benefit")
        refs = case.get("_rag_references") or []
        if not refs:
            continue
        hits = []
        for i, item in enumerate(refs):
            ref = item.get("reference_instance") or {}
            rid = str(ref.get("instance_id") or item.get("rank") or i)
            sim = float(item.get("similarity") or 0.0)
            relevant = int((ref.get("ground_truth") or {}).get("overall_benefit") == target)
            hits.append(SearchHit(rid, sim, "v24_reference", text_for(ref), {"relevant": relevant, "similarity_rank": i + 1}))
        base = sorted(hits, key=lambda h: h.score, reverse=True)
        qwen = reranker.rerank(str(case.get("input")), hits, top_k=len(hits))
        labels_b = [h.metadata["relevant"] for h in base]
        labels_q = [h.metadata["relevant"] for h in qwen]
        ideal = sorted(labels_b, reverse=True)
        rows.append({"instance_id": case.get("instance_id"), "baseline_labels": labels_b,
                     "qwen_labels": labels_q, "baseline_dcg": dcg(labels_b),
                     "qwen_dcg": dcg(labels_q), "ideal_dcg": dcg(ideal)})
    def mean(key): return sum(x[key] for x in rows) / max(1, len(rows))
    summary = {"n_cases": len(rows), "baseline_ndcg": mean("baseline_dcg") / max(1e-9, mean("ideal_dcg")),
               "qwen_ndcg": mean("qwen_dcg") / max(1e-9, mean("ideal_dcg")),
               "baseline_top1": sum(x["baseline_labels"][0] for x in rows) / max(1, len(rows)),
               "qwen_top1": sum(x["qwen_labels"][0] for x in rows) / max(1, len(rows)),
               "baseline_top3": sum(any(x["baseline_labels"][:3]) for x in rows) / max(1, len(rows)),
               "qwen_top3": sum(any(x["qwen_labels"][:3]) for x in rows) / max(1, len(rows)), "cases": rows}
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, ensure_ascii=False))


if __name__ == "__main__": main()
