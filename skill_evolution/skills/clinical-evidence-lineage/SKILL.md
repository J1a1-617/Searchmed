---
name: clinical-evidence-lineage
description: Preserve clinical evidence scope, state, attribution, and citation lineage from AnswerMemory through AnswerContext and Generate. Use when claims merge incompatible evidence scopes, accepted evidence changes meaning or disappears between stages, or an evidence-derived conclusion lacks stable claim/evidence bindings; do not use merely because direct evidence is absent.
metadata:
  call_stage: answer_memory|answer_context|generate
  when_to_use: claim_scope_collision, unexplained_evidence_drop, evidence_state_flip, cross_intervention_attribution, or evidence_derived_conclusion_without_binding
  when_not_to_use: ordinary retrieval mismatch, direct-evidence absence without a lineage anomaly, or a conclusion explicitly identified as an ungrounded model prior
---

# Clinical Evidence Lineage

Treat every clinical conclusion as a traceable transformation of scoped evidence. Preserve meaning across stages; do not improve apparent completeness by merging, relabeling, or inventing support.

## Shared evidence unit

Represent each evidence-derived statement with one stable lineage tuple:

`(evidence_id, claim_id, population, intervention, comparator, endpoint, timepoint, direction, evidence_state, limitations)`

An output may shorten wording but must not silently change any populated element. A treatment sequence establishes temporal order, not causal attribution.

## Trigger modes

### Scope collision

Use when one claim contains incompatible interventions, populations, endpoints, timepoints, or outcome directions. Split it into the smallest independently supportable claims. Keep efficacy, CNS control, symptoms, toxicity, and survival separate. Combination outcomes remain attributed to the combination.

### State or attribution drift

Use when accepted evidence disappears before AnswerContext, `unknown` becomes positive or negative, or an outcome is transferred from a prior intervention to the target intervention. Preserve three evidence states: `positive_evidence`, `negative_evidence`, and `unknown`. If an accepted item is excluded downstream, record its stable ID and an explicit exclusion reason.

### Binding break

Use when an AnswerContext or Generate conclusion is presented as evidence-derived but lacks valid claim/evidence IDs. Bind it to the supporting lineage tuple or relabel it as unresolved/model-prior and remove the unsupported citation. An unused bounded analog does not need to be cited merely because it is present in Context.

## Stage actions

- `answer_memory`: atomize claims and retain every source evidence ID and scope field.
- `answer_context`: compare incoming accepted IDs with forwarded, bounded, and excluded IDs; account for every evidence-bearing claim without changing its state or intervention attribution.
- `generate`: cite only evidence actually used by a conclusion; validate that cited IDs exist and support the same intervention, endpoint, timepoint, and direction. Keep unsupported dimensions unresolved or explicitly model-prior.

## Invariants

- Missing evidence is not negative evidence.
- Prior-regimen progression is not target-regimen failure without transferable resistance evidence.
- Later progression does not erase an earlier response window.
- A bounded analog never becomes a target-specific fact.
- Dropping irrelevant evidence is allowed and should be recorded compactly; silently dropping accepted support or counterevidence is not.
- Do not activate only because a drug, mutation, cancer type, or benchmark label appears. Route by lineage anomaly signals, then apply the loaded instructions to the concrete clinical entities in the stage input.
