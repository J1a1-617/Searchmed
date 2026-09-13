# Skill evolution loop

`evolve_skills.py` is the controller for the iterative benchmark loop:

1. Benchmark batches remain the unit of execution (currently 10 cases).
2. Every two completed batches (20 cases) create a cohort. The cohort records
   the exact source batches and candidate skills discovered from their audits.
3. At cohort creation, the event-driven path launches a full Codex CLI agent
   (`run_codex_curator.py`, model `gpt-6` by default). Codex can inspect the
   complete failed Sessions and relevant repository code, then emits structured
   candidate Skills. `synthesize_skills.py` remains an explicit lightweight
   `--curator-backend llm` fallback.
4. A candidate skill is never activated immediately. Its registry entry starts
   as `status=candidate, ab_status=pending`; an A/B artifact must mark it
   `pass` before the runtime can consume it.
   The A/B must use the candidate's own source-error cases: arm A must reproduce
   the original binary error, arm B exposes only that one Skill, and the trace
   must pass two independent gates: **efficacy** (the reproduced source error is
   corrected) and **callability** (candidate retrieval, model selection,
   `load_skill`, and execution at every stage declared by the Skill, recorded in
   `stage_activations`). Both gates are required. A regression case or a
   binary-correct source case cannot substitute for the source-error efficacy
   gate, and an answer that happens to become correct without loading the Skill
   cannot substitute for the callability gate.
5. Repeated rules are deduplicated by a canonical rule hash. A changed rule
   becomes `candidate_update_pending_ab` and must pass A/B again.
6. After 100 completed cases, all non-source batches are recorded as a held-out
   evaluation set. This eval set is not used to author the skills.

Run a dry controller pass:

```bash
PYTHONPATH=. python skill_evolution/scripts/evolve_skills.py \
  --manifest skill_evolution/batches/full_queue/manifest.json \
  --sessions skill_evolution/runs/batch_001_current/sessions \
  --root . --cohort-size 20 --eval-every 100
```

The provenance and pending gates are written to
`skill_evolution/registry/registry.json` and `registry/cohorts/`.

The controller intentionally does not treat a failed API call or fallback
prediction as an A/B result. Once API access is available, the A/B runner must
write an artifact with `status=pass` before a skill is promoted.

Build the live wrong-case inventory and source-case A/B queue with:

```bash
PYTHONPATH=. python skill_evolution/scripts/update_skill_failure_pipeline.py \
  --data benchmark_package/data/skill_dev_batch_002.json \
  --data benchmark_package/data/full_queue/batch_003.json \
  --result results/latest_benchmark_batch002_gpu_20260902/results.json \
  --checkpoint-root results/full_benchmark_continuation_20260907 \
  --artifact-root /slow_share/yangijiayi/skills1 \
  --registry skill_evolution/registry/registry.json \
  --output skill_evolution/registry/wrong_case_pipeline.json
```

Every wrong case is explicitly `unassigned`, `candidate`, or `validated`.
Only `validated` means a source-error A/B corrected the case and the Skill was
actually retrieved and loaded at its declared call stage.

## Agent-system metric boundary

Every benchmark manifest and checkpoint carries `agent_system_id`. Aggregate
metrics must select exactly one ID; legacy predictions are never averaged with
the complete-session Agent. Legacy errors may be copied into a separate
regression batch and re-run by the new Agent, but that selected batch is
reported as a regression stratum rather than as the unbiased primary cohort.

## Automatic event-driven pipeline

The benchmark runner can emit one durable completion event per successful
case:

```bash
python predictive_clinical_benchmark/run_eval.py \
  ... \
  --skill-evolution-event-dir skill_evolution/events
```

Run the resumable consumer under systemd/supervisor, or once from cron:

```bash
PYTHONPATH=. python skill_evolution/scripts/run_evolution_worker.py \
  --events skill_evolution/events \
  --index-db indexes/structured.db \
  --model gpt-6 \
  --curate-every 2
```

The worker performs deterministic evidence-availability auditing before it
calls the curator. It automatically builds a cohort, invokes a full Codex agent
using GPT-6 (with filesystem/terminal inspection in a read-only sandbox),
materializes open-ended candidate `SKILL.md` files, and writes an
`ab_queue.json`. Candidate files are fail-closed: normal runtime discovers only
registry entries with `status=active`. Source-error A/B may expose one candidate
explicitly with `DYNAMIC_SKILL_ALLOWLIST`.

The evidence audit accepts explicit `gold_citations`, `gold_evidence`, or
`gold_evidence_groups` anchors, either at case top level or under
`ground_truth`. It classifies the first loss as `db_missing_gold`,
`candidate_retrieval_miss`, `rerank_drop`, `fetch_drop`,
`evidence_review_drop`, `answer_context_drop`, or
`generation_citation_drop`. If no explicit gold anchors exist, the result is
`unknown_no_explicit_gold` and is deliberately ineligible for retrieval-Skill
learning. A clearly wrong outcome may still enter general reasoning/structure
curation, but its audit carries an explicit prohibition against retrieval
attribution. Outcome labels are never treated as gold evidence.

GPT-6 curation is not restricted to the four legacy templates. Those templates
remain deterministic audit signals, while the curator may propose new
cross-case failure patterns and stage-specific Skills.
