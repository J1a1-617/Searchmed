#!/usr/bin/env python3
"""Compare the current Agent dense index with RAG_reproduce retrieval.

This is a diagnostic experiment, not a benchmark predictor. It reports the
effect of corpus choice, query formatting, BGE instruction use, normalization,
and RAG_reproduce's structured bonus independently.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from predictive_clinical_benchmark.eval.prompts import construct_agent_query
from searchagent_retrieval.query_adapter import adapt_dense_query
from searchagent_retrieval.vector_index import BGE_LARGE_ZH_QUERY_INSTRUCTION


def format_rag_case(inst: dict[str, Any], *, include_outcome: bool) -> str:
    inp = inst.get("input", inst)
    bg = inp.get("disease_background", {})
    cs = inp.get("current_status", {})
    tx = inp.get("planned_treatment", {})
    parts: list[str] = []
    mol = bg.get("molecular_profile", {})
    mol_parts = []
    if mol.get("primary_mutation"):
        mol_parts.append(str(mol["primary_mutation"]))
    for key in ("resistance_mutations", "bypass_alterations", "co_mutations"):
        if mol.get(key):
            mol_parts.append(", ".join(str(v) for v in mol[key]))
    parts.append(f"诊断: {bg.get('diagnosis', '')}. 突变: {' / '.join(mol_parts) or '无'}")
    if bg.get("metastatic_sites"):
        parts.append(f"转移: {', '.join(str(v) for v in bg['metastatic_sites'])}")
    prior = inp.get("prior_treatment_timeline", [])
    short = []
    for row in prior[:3]:
        regimen = row.get("regimen", "")
        if regimen and regimen != "NA":
            short.append(f"{regimen} → {row.get('best_response', '?')}")
    if short:
        parts.append(f"既往治疗: {'; '.join(short)}")
    if cs.get("symptoms"):
        parts.append(f"症状: {', '.join(str(v) for v in cs['symptoms'])}")
    if cs.get("performance_status") not in (None, "", "NA"):
        parts.append(f"ECOG: {cs['performance_status']}")
    names = [str(v.get("name")) for v in tx.get("drugs", []) if v.get("name") not in (None, "", "NA")]
    if not names and prior:
        last = prior[-1].get("regimen", "")
        if last and last != "NA":
            names = [str(last)]
    strategy = tx.get("combination_strategy", "")
    regimen = " + ".join(names) or (str(strategy) if strategy and strategy != "NA" else "?")
    tail = f"方案: {regimen}"
    if include_outcome:
        tail += f". 真实结局: {inst.get('ground_truth', {}).get('overall_benefit', '未知')}"
    parts.append(tail)
    return "。".join(v for v in parts if v)


def features(inst: dict[str, Any]) -> set[str]:
    inp = inst.get("input", inst)
    bg = inp.get("disease_background", {})
    mol = bg.get("molecular_profile", {})
    feats: set[str] = set()
    for key in ("primary_mutation", "resistance_mutations", "bypass_alterations"):
        value = mol.get(key)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item and item != "NA":
                feats.update(str(item).lower().replace("/", " ").replace(",", " ").split())
    if any(k in str(bg.get("metastatic_sites", [])).lower() for k in ("脑", "脑膜", "cns", "brain", "leptomeningeal")):
        feats.add("__CNS__")
    tx = inp.get("planned_treatment", {})
    drugs = [v for v in tx.get("drugs", []) if v.get("name")]
    combo = str(tx.get("combination_strategy", ""))
    if "联合" in combo or len(drugs) >= 2:
        feats.add("__COMBO__")
    elif drugs:
        feats.add("__MONO__")
    return feats


def encode(model: SentenceTransformer, texts: list[str], normalize: bool) -> np.ndarray:
    return np.asarray(model.encode(texts, batch_size=64, normalize_embeddings=normalize, show_progress_bar=True), dtype=np.float32)


def ranks(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[1])
    return np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]


def ordered_top(scores: np.ndarray, k: int) -> np.ndarray:
    rough = ranks(scores, k)
    values = np.take_along_axis(scores, rough, axis=1)
    order = np.argsort(-values, axis=1)
    return np.take_along_axis(rough, order, axis=1)


def rag_rerank(scores: np.ndarray, queries: list[dict[str, Any]], refs: list[dict[str, Any]], top_k: int = 3) -> np.ndarray:
    candidate = ordered_top(scores, top_k * 3)
    output = np.empty((len(queries), top_k), dtype=np.int64)
    ref_feats = [features(row) for row in refs]
    for i, row in enumerate(queries):
        qf = features(row)
        adjusted = [(float(scores[i, j]) + 0.02 * len(qf & ref_feats[int(j)]), int(j)) for j in candidate[i]]
        adjusted.sort(reverse=True)
        output[i] = [j for _, j in adjusted[:top_k]]
    return output


def label_metrics(top: np.ndarray, queries: list[dict[str, Any]], refs: list[dict[str, Any]]) -> dict[str, float]:
    correct1 = 0
    correct3 = 0
    for i, ids in enumerate(top):
        gold = str(queries[i].get("ground_truth", {}).get("overall_benefit", ""))
        labels = [str(refs[int(j)].get("ground_truth", {}).get("overall_benefit", "")) for j in ids]
        correct1 += bool(labels and labels[0] == gold)
        majority = Counter(labels).most_common(1)[0][0] if labels else ""
        correct3 += majority == gold
    return {"top1_outcome_agreement": round(correct1 / len(queries), 4), "top3_majority_outcome_agreement": round(correct3 / len(queries), 4)}


def clinical_terms(inst: dict[str, Any]) -> set[str]:
    inp = inst.get("input", inst)
    bg = inp.get("disease_background", {})
    mol = bg.get("molecular_profile", {})
    values: list[str] = []
    for key in ("primary_mutation", "resistance_mutations", "bypass_alterations", "co_mutations"):
        value = mol.get(key)
        values.extend(str(v) for v in (value if isinstance(value, list) else [value]) if v and v != "NA")
    values.extend(str(v) for v in bg.get("metastatic_sites", []) if v and v != "NA")
    for row in inp.get("planned_treatment", {}).get("drugs", []):
        if row.get("name") and row["name"] != "NA":
            values.append(str(row["name"]))
    terms: set[str] = set()
    for value in values:
        normalized = " ".join(value.lower().split())
        if len(normalized) >= 2:
            terms.add(normalized)
        terms.update(
            token.lower()
            for token in re.findall(r"[a-z][a-z0-9_.+-]{2,}|[\u4e00-\u9fff]{2,}", normalized)
            if token.lower() not in {"mutation", "metastasis", "metastases"}
        )
    return terms


def agent_retrieval_quality(top: np.ndarray, queries: list[dict[str, Any]], items: list[dict[str, Any]]) -> dict[str, float]:
    coverages = []
    reference_noise = 0
    outcome_signal = 0
    total = 0
    for i, ids in enumerate(top):
        texts = [" ".join(str(items[int(j)].get("text", "")).lower().split()) for j in ids]
        joined3 = " ".join(texts[:3])
        terms = clinical_terms(queries[i])
        if terms:
            coverages.append(sum(term in joined3 for term in terms) / len(terms))
        for text in texts:
            total += 1
            reference_noise += bool("google scholar" in text or text.count("[pubmed]") >= 2 or text.count("doi") >= 4)
            outcome_signal += bool(re.search(r"\b(?:cr|pr|sd|pd|orr|dcr|pfs|response|progress(?:ion|ed)?)\b|\u7f13\u89e3|\u8fdb\u5c55|\u7a33\u5b9a|\u75c7\u72b6", text))
    return {
        "mean_top3_clinical_term_coverage": round(float(np.mean(coverages)), 4),
        "top10_reference_noise_rate": round(reference_noise / max(1, total), 4),
        "top10_outcome_signal_rate": round(outcome_signal / max(1, total), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-db", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=Path("predictive_clinical_benchmark/benchmark_multinode.json"))
    parser.add_argument("--agent-vector", type=Path, default=Path("indexes_step3_v2/vector"))
    parser.add_argument("--model", type=Path, default=Path("models/bge-large-zh-v1.5"))
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/dense_rag_2x2.json"))
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text())
    rag_db = json.loads(args.rag_db.read_text())
    metadata = json.loads((args.agent_vector / "metadata.json").read_text())
    agent_items: list[dict[str, Any]] = []
    agent_vectors = []
    for space, items in metadata["space_items"].items():
        matrix = np.load(args.agent_vector / "vectors" / f"{space}.npy")
        agent_items.extend(items)
        agent_vectors.append(matrix)
    agent_matrix = np.vstack(agent_vectors).astype(np.float32)
    model = SentenceTransformer(str(args.model), local_files_only=True)

    rag_docs = [format_rag_case(row, include_outcome=True) for row in rag_db]
    rag_doc_raw = encode(model, rag_docs, normalize=False)
    rag_doc_norm = rag_doc_raw / np.maximum(np.linalg.norm(rag_doc_raw, axis=1, keepdims=True), 1e-12)
    rag_query_clean = [format_rag_case(row, include_outcome=False) for row in benchmark]
    rag_query_leaky = [format_rag_case(row, include_outcome=True) for row in benchmark]
    agent_query = [adapt_dense_query(construct_agent_query(row)) for row in benchmark]
    q_rag_clean_raw = encode(model, rag_query_clean, normalize=False)
    q_rag_leaky_raw = encode(model, rag_query_leaky, normalize=False)
    q_rag_clean_norm = q_rag_clean_raw / np.maximum(np.linalg.norm(q_rag_clean_raw, axis=1, keepdims=True), 1e-12)
    q_agent_norm = encode(model, [BGE_LARGE_ZH_QUERY_INSTRUCTION + q for q in agent_query], normalize=True)

    rag_configs = {
        "rag_original_leaky_raw_dot": q_rag_leaky_raw @ rag_doc_raw.T,
        "rag_original_clean_raw_dot": q_rag_clean_raw @ rag_doc_raw.T,
        "rag_clean_cosine_no_instruction": q_rag_clean_norm @ rag_doc_norm.T,
        "rag_db_agent_query_instruction_cosine": q_agent_norm @ rag_doc_norm.T,
    }
    rag_results = {}
    for name, score in rag_configs.items():
        top = rag_rerank(score, benchmark, rag_db, top_k=3)
        rag_results[name] = {**label_metrics(top, benchmark, rag_db), "sample_top_ids": [[rag_db[int(j)]["instance_id"] for j in row] for row in top[:10]]}

    # Agent DB is already normalized. Raw dot and cosine therefore differ only
    # by the query representation, not by score normalization.
    agent_scores_rag_query = q_rag_clean_norm @ agent_matrix.T
    agent_scores_agent_query = q_agent_norm @ agent_matrix.T
    top_rag = ordered_top(agent_scores_rag_query, 10)
    top_agent = ordered_top(agent_scores_agent_query, 10)
    jaccards = []
    for a, b in zip(top_rag, top_agent):
        sa, sb = set(map(int, a)), set(map(int, b))
        jaccards.append(len(sa & sb) / len(sa | sb))

    def samples(top: np.ndarray) -> list[list[dict[str, Any]]]:
        rows = []
        for ids in top[:10]:
            rows.append([{"id": agent_items[int(j)]["doc_id"], "space": agent_items[int(j)].get("metadata", {}).get("embedding_space"), "text": str(agent_items[int(j)].get("text", ""))[:300]} for j in ids[:3]])
        return rows

    payload = {
        "corpora": {
            "rag_case_db": {"items": len(rag_db), "mean_text_chars": round(float(np.mean([len(v) for v in rag_docs])), 1)},
            "agent_chunk_db": {"items": len(agent_items), "spaces": {k: len(v) for k, v in metadata["space_items"].items()}, "mean_text_chars": round(float(np.mean([len(str(v.get('text', ''))) for v in agent_items])), 1)},
        },
        "norms": {
            "rag_doc_raw_mean": round(float(np.linalg.norm(rag_doc_raw, axis=1).mean()), 4),
            "rag_doc_raw_std": round(float(np.linalg.norm(rag_doc_raw, axis=1).std()), 4),
            "agent_stored_mean": round(float(np.linalg.norm(agent_matrix, axis=1).mean()), 4),
            "agent_stored_std": round(float(np.linalg.norm(agent_matrix, axis=1).std()), 4),
        },
        "rag_db_benchmark_outcome_proxy": rag_results,
        "agent_db_query_comparison": {
            "mean_top10_jaccard": round(float(np.mean(jaccards)), 4),
            "median_top10_jaccard": round(float(np.median(jaccards)), 4),
            "rag_clean_query_quality": agent_retrieval_quality(top_rag, benchmark, agent_items),
            "agent_query_quality": agent_retrieval_quality(top_agent, benchmark, agent_items),
            "rag_clean_query_samples": samples(top_rag),
            "agent_query_samples": samples(top_agent),
        },
        "warnings": [
            "RAG_reproduce original query text includes benchmark ground_truth.overall_benefit.",
            "RAG_reproduce calls an unnormalized dot product cosine similarity.",
            "Outcome agreement is a retrieval proxy, not the official benchmark score.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
