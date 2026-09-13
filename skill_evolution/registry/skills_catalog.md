# Skills catalog

This table is the human-readable index. `registry.json` remains the machine-readable source of truth.

| Skill ID | Purpose | Runtime stages | Source | Status | A/B status | Artifact |
|---|---|---|---|---|---|---|
| `clinical-evidence-applicability` | Gate direct, partial, analog, and counterevidence without transferring outcomes across mismatched regimens | Retired; rules remain in EvidenceReview prompt | batch001_current; case_14 | rejected as a dynamic Skill | clean component A/B found no gain | `skill_evolution/analysis/pending_skill_ab_20260906.md` |
| `claim-scope-preservation` | Prevent claims across different drugs, populations, endpoints, and timepoints from being merged | AnswerMemory, AnswerContext | batch001_current | merged into `clinical-evidence-lineage` | pending | — |
| `evidence-context-citation` | Preserve stable claim/evidence IDs through Generate | AnswerContext, Generate | batch001_current | merged into `clinical-evidence-lineage` | pending | — |
| `evidence-absence-calibration` | Keep missing evidence distinct from evidence of no benefit and prevent prior-regimen progression from becoming target-regimen failure | Evidence Review, AnswerContext, Generate | batch002; case_12_node_1 | merged into `clinical-evidence-lineage` | pending | `skill_evolution/analysis/batch_002/case_12_node_1.md` |
| `structured-output-continuation` | Retry truncated structured work as complete bounded parts and merge it without dropping timeline, causality, or schema fields | Evidence Review, SafetyReflection, AnswerContext | full_queue_batch004; case_15_node_2; case_16_node_2 | candidate | adaptive end-to-end pass on four complete runs; no natural recovery trigger | `skill_evolution/analysis/structured_output_continuation/failure_cases.md` |
| `clinical-evidence-lineage` | Preserve claim scope, evidence state, intervention attribution, and citation lineage across downstream stages | AnswerMemory, AnswerContext, Generate | batch001_current; batch002; nine source cases | candidate | loaded in 2/2 complete end-to-end pairs; no binary gain; added one LLM call | `skill_evolution/analysis/pending_skill_ab_20260906.md` |

## Maintenance rule

For every new candidate, record its source batch and cases, runtime stages, trigger, status, A/B status, and analysis artifact. Merge candidates only when their decision rule and invocation stages overlap. Do not activate a candidate until it improves a non-source evaluation set without harming previously correct cases.
