---
name: evidence-absence-calibration
description: Preserve the distinction between missing evidence and negative evidence while building AnswerContext. Do not use it to select a final benefit label in Generate.
metadata:
  call_stage: merged_candidate
  when_to_use: accepted evidence disappears before AnswerContext, or prior-treatment progression risks being transferred to a new regimen while context is assembled
  when_not_to_use: ordinary final prediction after AnswerContext has already recorded evidence state and intervention attribution
---

# Evidence Absence Calibration

## Three-way evidence state

Keep these states separate throughout Evidence Review, AnswerContext, and Generate:

1. `positive_evidence`: an admissible finding supports benefit for the target regimen and relevant population or endpoint.
2. `negative_evidence`: an admissible finding supports failure, progression, or harm for the target regimen, or an explicit mechanism supports transferable resistance.
3. `unknown`: direct evidence was not retrieved, the endpoint was not reported, or available analogs are too weak.

`unknown` is not `negative_evidence`. Retrieval failure, database coverage gaps, and missing early-window endpoints may lower confidence but must not independently determine a no-benefit label.

## Intervention attribution

- Attach every response or progression fact to the intervention being administered at that time.
- Do not transfer progression on a prior drug to the proposed regimen solely because the drugs share a target or class.
- Transfer a negative conclusion only when a stated resistance mechanism or sufficiently matched clinical evidence supports cross-resistance.
- Mechanistic plausibility alone is not positive endpoint evidence; it may inform a low-confidence prior only when the output schema forces a classification.

## Context preservation

- If Evidence Review accepts support, counterevidence, or a bounded analog, AnswerContext must preserve it with intervention, population, endpoint, timepoint, direction, limitations, and stable evidence ID.
- If all accepted findings are removed, emit an explicit `evidence_gap` record explaining why. Never silently convert accepted findings into an empty context.
- Generate must distinguish `no admissible evidence` from `admissible evidence of no benefit` in both rationale and confidence.

## Prediction boundary

- Prefer an unresolved or `NA` endpoint when the schema permits it.
- When the benchmark schema forces an overall label, select it from admissible patient facts and bounded clinical priors; do not use “nothing was found” as the negative premise.
- Never fabricate RECIST, symptom, toxicity, or time-to-response values to fill an evidence gap.
