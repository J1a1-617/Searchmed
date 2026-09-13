# Skill inventory and source-error mount test (2026-09-06)

## Inventory and A/B status

| Skill | Runtime state | Source | A/B result |
|---|---|---|---|
| `clinical-evidence-applicability` | Rejected as a separately mounted dynamic Skill; its useful rules remain in the EvidenceReview prompt | Recurrent weak/mismatched analog admission in batch001_current; case 14 integration check | Clean component replay retained the same weak analog and changed the verdict from insufficient to partially supports while adding latency |
| `clinical-evidence-lineage` | Candidate `SKILL.md`; source-error retrieval tested; Generate has a `load_skill` loop | Nine source cases spanning scope collision, state/attribution drift, and binding breaks | Two component pairs plus two complete end-to-end pairs loaded 2/2. End-to-end binary predictions were unchanged and correct; citation boundaries remained explicit, at a cost of one extra LLM call |
| `structured-output-continuation` | Operational code path in EvidenceReview, SafetyReflection, and AnswerContext; not dependent on model Skill selection | `case_15_node_2`, `case_16_node_2`; regression set `case_11_node_2`, `case_20_node_2` | Adaptive component replay passed. Four complete end-to-end executions used one complete call per stage (`activated=false`); forced-error unit tests cover recovery. No natural API truncation occurred |
| `claim-scope-preservation` | Registry-only merged candidate | `case_01_node_1`, `case_02_node_1`, `case_03_node_1` | No independent A/B; merged into `clinical-evidence-lineage` |
| `evidence-context-citation` | Registry-only merged candidate | `case_01_node_1`, `case_04_node_2`, `case_10_node_2`, `case_13_node_3`, `case_19_node_2`, `case_26_node_2` | No independent A/B; merged into `clinical-evidence-lineage` |
| `evidence-absence-calibration` | Legacy `SKILL.md`, routing stage changed to `merged_candidate` | `case_12_node_1` | No independent A/B; merged into `clinical-evidence-lineage` |

## BGE source-error retrieval

Model: local `models/bge-large-zh-v1.5`, with the official Chinese retrieval instruction prepended to queries only. Queries contain stage plus abstract anomaly tags, not drug, mutation, cancer, or outcome names.

| Expected Skill / source pattern | Stage | Rank and score | In Top-K |
|---|---|---:|---:|
| `clinical-evidence-applicability` / weak mismatched analog admission | EvidenceReview | Top-1, 0.6670 | yes |
| `clinical-evidence-lineage` / scope collision | AnswerMemory | Top-1, 0.5320 | yes |
| `clinical-evidence-lineage` / evidence drop, state flip, attribution drift | AnswerContext | Top-1, 0.6574 | yes |
| `clinical-evidence-lineage` / conclusion-binding break | Generate | Top-1, 0.6180 | yes |
| `structured-output-continuation` / invalid or truncated structured output | AnswerContext | Top-1, 0.6610 | yes |

Actual saved lineage source trajectories:

| Case | Derived signals | Lineage rank / score |
|---|---|---:|
| `case_01_node_1` | scope collision, unexplained drop, attribution boundary | Top-1 / 0.7537 |
| `case_02_node_1` | scope collision, attribution boundary | Top-1 / 0.7099 |
| `case_03_node_1` | scope collision, unexplained drop, attribution boundary | Top-1 / 0.7537 |
| `case_04_node_2` | attribution boundary | Top-1 / 0.6596 |
| `case_10_node_2` | attribution boundary | Top-1 / 0.6596 |
| `case_13_node_3` | attribution boundary | Top-1 / 0.6596 |
| `case_19_node_2` | unexplained drop, attribution boundary | Top-1 / 0.7194 |
| `case_26_node_2` | attribution boundary | Top-1 / 0.6596 |

`case_12_node_1` has an analysis artifact but no local full session. Replaying its recorded abstract source signals retrieves `clinical-evidence-lineage` in Top-K; this is not counted as a raw-trajectory test.

## Interpretation

- Skill retrieval works on the tested source errors.
- Retrieval success does not prove actual mounting. Only Generate currently exposes `load_skill` to the LLM. EvidenceReview and AnswerMemory/AnswerContext do not yet have equivalent tool loops.
- `structured-output-continuation` activates directly in code after a recoverable structured-output error; it is not selected through `load_skill`.
- Skill document embeddings are persisted by embedding-model identity plus `skill_hash`. Runtime retrieval loads the index and embeds only the query. A similarity threshold is intentionally deferred while the active catalog is small.

## Embedding ablation

The same five routing queries were repeated without an embedding model, using token overlap only.

- Applicability: keyword routing ranked `structured-output-continuation` first (0.30) and the expected applicability Skill second (0.20).
- Lineage scope, context, and Generate patterns still returned the expected Skill first because their exact trigger tags appear in its metadata.
- Structured-output failure produced a 0.3333 tie between lineage and continuation.

Therefore embedding materially improves routing and should remain enabled. BGE-large-zh-v1.5's official query instruction is prepended to the short routing query only. Skill vectors are now persistent; the final index contains the two runtime Skills and reports `rebuilt=false` on later processes. After model initialization, source-query embedding and ranking took about 0.16–0.23 seconds per query in this test.

## Final active-catalog verification

- Persistent runtime Skill count: 2 (`clinical-evidence-lineage`, `structured-output-continuation`).
- Eight available raw lineage source trajectories: 8/8 expected Skill at Top-1.
- Structured-output source pattern: expected Skill at Top-1 (0.7141).
- Empty anomaly-signal set: no candidates.
- First query in a fresh process: about 4.91 seconds including BGE model initialization; subsequent queries: about 0.16–0.23 seconds.
