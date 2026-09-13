# case_03_node_1 analysis

## Outcome

- Run status: complete; 28 LLM calls; 1524.5 s wall time.
- Correct: CNS `SD`, CSF `未评估`, symptoms `部分改善`.
- Incorrect: overall benefit `明显获益` vs `进展或有害`; body `PR` vs `SD`; toxicity grade 0 vs grade 2 transaminase elevation.

## Stage diagnosis

1. Query understanding correctly captured ALK fusion, TP53 co-mutation, crizotinib, brain/bone metastases. It also extracted `PR` from prior chemotherapy/current baseline imaging as a response constraint without preserving that this response preceded planned crizotinib.
2. Planner appropriately separated general crizotinib durability from TP53 as a negative effect modifier. Replanner added trial anchors and the Xalkori alias.
3. Retrieval found valid pre-cutoff TP53/crizotinib evidence (PMID 29997966) indicating worse response, mPFS about 3.3 months and frequent progression within two months. This was directly relevant to the patient's TP53 context.
4. Retrieval also admitted PMID 33889527 as a pre-cutoff 2015 item, but the article was actually published in March 2021. This is a second confirmed temporal-index leak.
5. Context preserved both general-population benefit and TP53-specific early progression risk, explicitly warning against certainty. It lacked toxicity and intracranial evidence.
6. Generate overweighted general crizotinib efficacy and the patient's pre-treatment/prior-regimen imaging state. Its rationale states that the current chest lesion is shrinking and brain lesion stable as if these were outcomes of planned crizotinib. It underweighted the more patient-specific TP53 evidence and predicted `明显获益`.
7. The final zero-toxicity prediction was unsupported; no systematic toxicity evidence reached Context, while the observed outcome was grade-2 transaminase elevation.

## Candidate reusable lessons

- Temporal attribution: never treat baseline status or response to a prior regimen as an outcome of the planned intervention.
- Modifier precedence: patient-specific negative modifiers with direct evidence should outweigh broad population averages when predicting direction and confidence.
- Dimension-specific evidence: efficacy evidence cannot justify a zero-toxicity prediction; missing toxicity evidence must remain a separate uncertainty.
- These patterns are candidates for a temporal-state/causal-attribution skill if repeated in later valid cases.

## Non-skill backlog

- Data/index bug: PMID 33889527 is stored as 2015-09-01 but PubMed reports March 2021; it must be excluded at the 2018-11-30 cutoff.
- Query-understanding schema needs treatment-relative provenance for every response/imaging fact: `prior_regimen`, `baseline_before_planned_treatment`, or `outcome_after_planned_treatment`.
- Final output again omits structured PFS/OS threshold fields.
- Performance: 28 LLM calls and about 25.4 minutes for one case.

## Promotion status

Candidate observations only. Temporal attribution is strong, but promotion requires recurrence and held-out validation.
