# Structured-output continuation source failures

These cases are the source cohort for the skill. They must not be used as the only evidence of generalization; later validation should include non-source cases and regression cases.

| Case | Attempt | Failed stage | Observed failure | Full trace |
|---|---:|---|---|---|
| `case_15_node_2` | 1 | QueryUnderstanding | First response hit `max_output_tokens`; retry returned malformed soft-schema JSON. QueryUnderstanding remains whole-task retry and is excluded from partitioning. | `/slow_share/yangjiayi/skills1/run_20260902_160626/case_15_node_2/attempt_1/llm_trace.json` |
| `case_15_node_2` | 2 | SafetyReflection | Ten claim reviews plus issues/revisions reached 8,000 output tokens and truncated in `reflection_summary`. | `/slow_share/yangjiayi/skills1/run_20260902_160626/case_15_node_2/attempt_2/llm_trace.json` |
| `case_16_node_2` | 1 | AnswerContext | Findings and seven global boundary arrays reached 5,000 output tokens and truncated in `unresolved_gaps`. | `/slow_share/yangjiayi/skills1/run_20260902_160626/case_16_node_2/attempt_1/llm_trace.json` |
| `case_16_node_2` | 3 | EvidenceReview | Eight evidence assessments with per-dimension fields reached 8,000 output tokens and truncated inside the eighth assessment. | `/slow_share/yangjiayi/skills1/run_20260902_160626/case_16_node_2/attempt_3/llm_trace.json` |
| `case_11_node_2` | 1 | QueryUnderstanding | Initial soft-schema response was incomplete; a later attempt recovered. Regression case for whole-task retry. | GPU batch trace, recovered on retry |
| `case_20_node_2` | 1 | AnswerContext | Initial context output truncated; a later attempt recovered. Regression case for partitioned context. | GPU batch trace, recovered on retry |

## A/B protocol

- A: retain the recorded pre-change attempts above. Primary completion outcome for `case_15_node_2` and `case_16_node_2` was failure after three attempts.
- B: rerun the same benchmark cases with the new code, identical model/index/retrieval limits, and full traces.
- Compare stage completion, whole-case completion, binary prediction, total/function calls, latency, output tokens, retained evidence IDs, claim IDs, and preservation of treatment-time/causal boundaries.
- Add `case_11_node_2` and `case_20_node_2` as regression checks; add at least one previously complete case to detect unnecessary call growth or answer regression.

## B-run attempts

| Run | Cases | Result | Interpretation | Artifact |
|---|---|---|---|---|
| `structured_output_ab_b_20260903` | `case_15_node_2`, `case_16_node_2` | 0/2 completed; all six case attempts stopped on the first QueryUnderstanding `chat` call | Invalid A/B observation. Each call had no token usage or model response and ended in `APITimeoutError`; the partitioning stages were never reached. Exclude this run from skill-effect estimates. | `/slow_share/yangjiayi/skills1/structured_output_ab_b_20260903/` |

The failed run exposed an infrastructure issue: an unreachable endpoint consumed about 525 seconds per logical call because the long generation timeout also governed connection establishment. The client now uses a 20-second connect timeout while preserving the long read/generation timeout, and connection failures skip internal logical-call retries so the existing case-level retry policy controls recovery.

## Replay of real A inputs

This is a deterministic input-shape check, not a substitute for the pending model A/B:

| Case/stage | Old single-call input | New LLM input | Planned complete calls |
|---|---:|---:|---:|
| `case_15_node_2` SafetyReflection | 10 claims | 3 AnswerContext-eligible, evidence-bound claims; 7 excluded | 1 |
| `case_16_node_2` SafetyReflection | 9 claims | 6 AnswerContext-eligible, evidence-bound claims; 3 excluded | 2 |
| `case_16_node_2` AnswerContext | 9 current claims | Same 9 claims; no schema or timeline fields deleted | 3 batches of at most 4 claims |
| `case_16_node_2` EvidenceReview | 8 evidence rows | Same 8 input rows; rejected/insufficient rows may be omitted from output | 2 batches of 4 evidence rows |

The old `case_15_node_2` safety request was 21,598 characters and truncated at 8,000 output tokens. The old `case_16_node_2` context request was 13,157 characters and truncated at 5,000 output tokens. Its failed EvidenceReview request had eight evidence rows and truncated at 8,000 output tokens. These source inputs therefore activate the intended prefilter or partition boundary without changing the clinical task.
