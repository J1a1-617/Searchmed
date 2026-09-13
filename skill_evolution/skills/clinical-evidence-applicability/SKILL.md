---
name: clinical-evidence-applicability
description: Classify retrieved clinical evidence by applicability during Evidence Review. Use for mixed direct, partial, analog, and counter evidence; do not use as a final benefit-label calibration rule in Generate.
metadata:
  call_stage: retired_candidate
  when_to_use: retrieved top-k contains competing claims with material regimen, population, endpoint, attribution, or time-window differences
  when_not_to_use: ordinary final prediction; missing direct evidence alone; or progression that occurred only under a different prior regimen
---

# Clinical Evidence Applicability

## Applicability ladder

Before accepting a candidate, compare it with the target on: intervention/regimen, prior line and rechallenge status, histology and molecular driver, anatomy or local-therapy modality, outcome, assessment window, and whether the result can be attributed to the intervention.

1. `direct_support`: the intervention and population match, the requested endpoint and time window are actually reported, and the result is attributable to the intervention.
2. `partial_support`: at least one explicitly transferable dimension is present, but record every material mismatch and never state the result as a target fact.
3. `analog_support`: use only when at least two clinically meaningful dimensions match and the candidate reports a relevant result. Keep it separate from direct evidence and retain at most the best non-redundant analog per gap.
4. `counter`: reserve for a comparable population, intervention/regimen and endpoint with an opposite result. Different drugs, populations, modalities, or missing outcomes are not counters; label them `irrelevant` or `partial_support`.

## Hard exclusions

- Citation pointers, titles, abstracts without a result, and reference lists are not outcome evidence.
- Do not use a systemic-treatment result to support a local therapy, or a prior regimen's response as the planned regimen's outcome.
- For an early-window question, a late progression result is not an early negative result.
- Different regimen plus different population is not a useful analog merely because the disease name is similar.

## Forwarding rule

Pass all admissible direct findings, then only the smallest set of partial/analog findings needed to explain a specific gap. If no direct finding survives, pass an explicit evidence-gap record and no more than two nearest analogs. Preserve each claim's population, intervention, endpoint, timepoint, direction, and limitations as separate fields. Evidence Review must not forward a low-relevance item only because it has a high retrieval score.

## Benchmark output boundary

- A partial or analog finding may affect confidence or explain uncertainty, but it must not determine target-specific RECIST, symptom direction, toxicity, or overall-benefit labels unless the same endpoint is explicitly reported for a sufficiently matching intervention and population.
- If no direct endpoint evidence exists, mark that endpoint unresolved and use the schema's conservative value; do not infer response, progression, or symptom change from a different drug, modality, or patient group.
- Keep efficacy, CNS control, symptoms, and toxicity as separate claims. Never merge them into one broad claim across different evidence items.
- Before Generate, verify that every cited item is direct or explicitly bounded analog evidence; otherwise remove it from the final context.

## Progression-risk override

- Treat a recent, explicit progression signal as high-priority negative evidence only for the intervention under which that progression occurred.
- Progression on a prior drug must not be transferred to a new target regimen merely because both drugs share a class. Transfer it only when an explicit resistance mechanism or sufficiently matched clinical evidence supports cross-resistance.
- When progression occurred on the target regimen, the target regimen has no direct response evidence, and the available positive support is only mechanism-level or an unmatched analog, do not classify the target as `获益` merely because the drug class is biologically plausible.
- In this pattern, the default binary decision is `不获益` (or an explicitly unresolved high-risk state if the output schema permits one); a positive prediction requires direct target-regimen evidence that explains why the prior progression signal should not dominate.
- Keep this override tied to the target patient's documented timeline. Do not infer progression from a late outcome in an unrelated case.

## Controlled expansion

When exact retrieval fails, relax one dimension at a time: exact regimen and setting → same intervention/class with the same population and endpoint → same disease/setting with a different regimen → mechanism or general background. Record the relaxed dimension in the query and stop when the target gap cannot be answered.
