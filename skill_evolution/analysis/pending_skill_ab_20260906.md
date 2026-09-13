# Pending Skill A/B results (2026-09-06)

Model: `gpt-5.5`. A and B in each comparison received the same saved source-stage input. These are component A/B tests; they do not replace a held-out end-to-end benchmark run.

## Clinical evidence lineage

### `case_01_node_1`: weak evidence must not become target-regimen citations

| Arm | Skill loaded | Overall/body/symptom | Citations | Time |
|---|---:|---|---:|---:|
| A | no | limited/stable; SD; unchanged | 2 weak/mismatched citations | 22.49 s |
| B | yes | limited/stable; SD; unchanged | 0 | 67.88 s |

Both arms kept the same binary prediction. B retrieved the Skill at rank 1 (0.7537), called `load_skill`, and removed citations that did not directly support the target combination. Toxicity remained grade 2 in both arms although the recorded target was grade 0, so the Skill did not solve unsupported toxicity calibration.

### `case_02_node_1`: direct evidence must survive

| Arm | Skill loaded | Overall/body | Direct citations | Time |
|---|---:|---|---:|---:|
| A | no | obvious benefit; PR | 1 | 36.78 s |
| B | yes | obvious benefit; PR | 1 | 45.42 s |

B retrieved the Skill at rank 1 (0.7099), called `load_skill`, and preserved the directly matched osimertinib evidence. It did not repair the missing CNS evidence, and correctly did not turn systemic response into an intracranial response.

Decision: retain as a candidate. It improved citation lineage in one source case and did not reject direct evidence in the positive-control case. The extra function-call loop increased latency. Promotion requires non-source held-out cases.

### Paired end-to-end follow-up (`gpt-5.6-sol`)

The updated and ablated arms used identical current code, Qwen3 reranking, two retrieval steps, and the same benchmark input. The only switch was `DYNAMIC_SKILLS_ENABLED`. Two pairs completed before GPU/API tunnel failures prevented the remaining pairs from completing.

| Case | Arm | Complete | Binary prediction | Calls | Tokens | Time | Citations | Skill loaded |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `case_11_node_2` | A: disabled | yes | benefit (correct) | 9 | 48,915 | 780.25 s | 3 | no |
| `case_11_node_2` | B: enabled | yes | benefit (correct) | 10 | not retained in reduced row | 624.27 s | 2 | yes |
| `case_15_node_2` | A: disabled | yes | benefit (correct) | 9 | 34,544 | 425.13 s | 0 | no |
| `case_15_node_2` | B: enabled | yes | benefit (correct) | 10 | 49,666 | 634.42 s | 2 | yes |

Both B runs retrieved `clinical-evidence-lineage` at rank 1 with score `0.6596` from the abstract signal `cross_intervention_attribution`, then actually called `load_skill`. The binary answer did not improve because both A arms were already correct. In `case_15_node_2`, B cited two bounded analogs but explicitly retained the molecular/regimen mismatch and did not present them as direct target-regimen evidence. The Skill therefore passed routing and lineage-behavior checks, but added one LLM call and did not show task-score gain in this two-pair sample.

Infrastructure-excluded cases: `case_16_node_2` B and both `case_20_node_2` arms did not complete after three attempts because the GPU reverse tunnel disappeared; their errors were `APIConnectionError`/`APITimeoutError`, before a comparable final answer existed.

Artifacts:

- `/slow_share/yangijiayi/skills1/pending_skill_paired_ab_gpt56sol_20260906/`
- `/slow_share/yangijiayi/skills1/pending_skill_paired_ab_gpt56sol_remaining3_20260907/`

## Clinical evidence applicability

Source replay: first EvidenceReview round of `case_10_node_2`, eight identical candidate rows.

| Arm | Retained rows | Low/mismatched analogs retained | Verdict | Time |
|---|---:|---:|---|---:|
| A: current EvidenceReview prompt | 1 | 1 | insufficient evidence | 14.56 s |
| B: prompt plus Skill | 1 | 1 | partially supports | 48.05 s |

The dynamic Skill added no rejection benefit and made the task-wide verdict less conservative. Its useful rules are already present in the current EvidenceReview prompt. Decision: reject it as a separately mounted dynamic Skill and keep the prompt-level applicability constraints.

## Structured output continuation

Source replay: `case_16_node_2` AnswerContext, five eligible claims, batch size four.

| Version | Calls | Complete schema | Recovery activated | Time |
|---|---:|---:|---:|---:|
| Historical eager partition | 2 | yes | always partitioned | 39.28 s |
| Adaptive | 1 | yes | no, full call succeeded | 43.18 s |

The adaptive version removed an unnecessary call and returned all required fields. This sample's single call was slightly slower than the two historical calls combined, so no latency improvement is claimed. Existing unit tests cover the recoverable-error branch by forcing an invalid first structured result and verifying deterministic partition/merge.

Decision: keep the adaptive operational implementation. A new end-to-end source/regression run is still required before promotion.

End-to-end follow-up: all four completed A/B executions across `case_11_node_2` and `case_15_node_2` recorded complete one-part outputs for EvidenceReview, SafetyReflection, and AnswerContext with `activated=false`. Thus the adaptive implementation imposed no partition calls when the first complete structured call succeeded. `case_15_node_2` is a source case; `case_11_node_2` is a non-source regression observation. Recovery activation remains covered by forced-failure unit tests rather than a naturally truncated API response in this run.
