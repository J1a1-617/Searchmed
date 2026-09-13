# Evidence Review soft-schema experiment

## Prompt change

The current system prompt is `_CITATION_SYSTEM_PROMPT` in
`searchagent_retrieval/agent_loop.py`. Compared with the prior prompt it now:

1. Receives and distinguishes `main_question`, `problem_representation`,
   `current_plan_step`, `executed_query`, and `rerank_goal`.
2. Treats the retrieval query as recall wording rather than confirmed patient
   facts.
3. Uses `overall_role` plus open-ended `dimension_findings` for mixed outcomes.
4. Keeps only `chunk_id` and `claim_scope` as required per-evidence fields;
   other medical fields are optional and extensible.
5. Still uses the forced `submit_evidence_review` function call, with
   `strict=false`, so the protocol remains a tool call without a rigid medical
   ontology.

## Test fixtures

- Cross-drug hepatotoxicity: capmatinib evidence for a furmonertinib question.
- Organ-specific mixed outcome: chest shrinkage with leptomeningeal worsening
  on combination treatment for an osimertinib-monotherapy question.
- Multi-outcome case: extracranial PR, leptomeningeal progression, grade-3
  thrombocytopenia, interruption and dose reduction in one chunk.

## Observed comparison

| Version | Fixtures / runs | Narrow fact preserved | Unsupported additions in claim_scope | Mixed outcomes represented by dimension | LLM calls per review |
|---|---:|---:|---:|---:|---:|
| Previous hard schema | 2 / 2 | 2/2 | 0/2 | 0/1 (one role only) | 1 |
| New soft schema, first run | 3 / 3 | 3/3 | 0/3 | 2/2 | 1 |
| New soft schema, repeated twice | 3 / 6 | 6/6 | 0/6 | 4/4 | 1 |

The six repeated calls averaged about 35.1 seconds. No function-call protocol
failure or output truncation occurred. Two naive whole-object checks found
target phrases, but inspection showed they appeared in mismatch/unreported
boundary statements, not in `claim_scope`; all six claim scopes avoided the
forbidden patient-specific inference.

One first-run inconsistency was found: a cross-drug item used
`overall_role=insufficient` while its `dimension_findings` explicitly said
`analog`. The merge layer now resolves this internal contradiction from the
dimension relations and records it as `analog_support`.

## Retrieval diagnostics change

StepMemory now expands `execute_retrieval_batch` into one ledger row per actual
search action, preserving tool-specific query, constraints, spaces, requested
top-k, hit count, cumulative candidate count, top hit IDs, and errors. It also
records the retrieved/reranked/fetched/reviewed/accepted/rejected funnel and a
compact rejection summary. Replanner receives these diagnostics for the latest
two rounds rather than only a generic query and the latest single round.
