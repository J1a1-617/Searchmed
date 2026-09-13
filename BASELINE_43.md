# 43-case SearchAgent baseline

This repository snapshot preserves the SearchAgent implementation used as the
reference baseline for the 43-case predictive-clinical cohort reconstructed
from the September 2026 GPU session.

## Baseline identity

- Agent system ID: `session_complete_v2_gpt54mini_20260909_090053`
- Generator/planning model: `gpt-5.4-mini`
- Dense encoder: `BAAI/bge-large-zh-v1.5`
- Reranker: `Qwen3-Reranker-4B`
- Retrieval budget: `top_k=8`, `max_steps=2`, `max_total_steps=2`
- Temporal filtering: publication cutoff enabled
- Complete cases: 43

## Binary benefit baseline

| Metric | Value |
|---|---:|
| Accuracy | 0.7442 |
| Precision (benefit) | 0.8286 |
| Recall (benefit) | 0.8529 |
| F1 (benefit) | 0.8406 |
| TP / TN / FP / FN | 29 / 3 / 6 / 5 |

## External-knowledge caveat

The historical GPU workspace did not contain the runtime `data/` directory.
Across the 43 saved trajectories, two Replanner decisions proposed expansion,
one expansion function was called, and that call failed with
`FileNotFoundError`. No external-knowledge row reached reranking, Evidence
Review, AnswerContext, or final generation. The BM25 index did contain 1,048
rule chunks, so those documents could still affect corpus statistics even
though no rule hit was propagated as evidence.

For that reason this snapshot should be described as the **case-retrieval
baseline**, not as a validated knowledge-augmented baseline.

## Reproducibility policy

Generated indexes, model weights, API credentials, sessions, full traces, and
benchmark outputs are intentionally excluded from Git. A benchmark run should
record the Git commit, configuration, data/index hashes, model identifiers,
case IDs, and artifact path in its run manifest.
