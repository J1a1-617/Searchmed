# case_01_node_1 analysis

## Outcome

- Run status: complete; 23 LLM calls; 1065.4 s wall time.
- Correct: overall benefit `有限获益或稳定`; body RECIST `SD`; CNS/CSF `NA/未评估`.
- Incorrect: symptom `部分改善` vs `无变化`; toxicity grade 2 vs grade 0.
- Unsupported: final prediction cites no evidence and omits the requested PFS/OS binary fields.

## Stage diagnosis

1. Query understanding correctly retained lung adenocarcinoma, EGFR 19del, ALK positivity, prior alectinib and the planned three-drug regimen.
2. Planner produced a clinically coherent hierarchy: exact post-alectinib combination evidence, then ALK-TKI-continuation plus pemetrexed analogue evidence. It used three rounds across two plan steps.
3. Retrieval failed to find direct regimen-and-time-window evidence. Reformulation changed the query but mostly returned different drugs, molecular subtypes and outcome windows.
4. Evidence Review correctly described the mismatches, yet labelled very weak/different evidence as `analog_support`: round 1 accepted 10/10, including scores 0.03-0.05 and `target_entity_match=different`; round 2 accepted 3/3; round 3 accepted 3/10.
5. AnswerMemory merged 16 heterogeneous observations into one oversized `answer_direction` claim. Its support semantics remained too broad even though the claim was marked `safety_limited`.
6. Answer Context behaved well: `key_findings=[]`, declared direct evidence absent, listed prohibited attributions, and retained only one explicitly partial/analogue finding.
7. Generate respected the low-confidence/evidence-insufficient framing but still invented directional symptom benefit and generic grade-2 chemotherapy toxicity. Those dimensions were not supported by Context.

## Candidate reusable lessons

- Evidence admission: `target_entity_match=different` plus very low relevance must remain `irrelevant/insufficient`, not `analog_support`, unless an explicit transferable dimension is named.
- Memory granularity: do not concatenate heterogeneous case observations into one answer-direction claim; store one scoped claim per intervention/population/outcome/time window.
- Generation calibration: when `key_findings` is empty, citations are empty and attribution is prohibited, categorical predictions must be labelled as model prior per dimension and must not be presented as evidence-derived.
- Retrieval hierarchy: after exact multi-drug-regimen failure, decompose by component contribution and transferable outcome dimension rather than broadening to unrelated driver populations.

## Non-skill backlog

- Performance: seven rerank calls consumed 515.6 s and about 40k output tokens; output budget is excessive for fixed-dimensional scoring.
- Performance: three Evidence Review calls consumed 210.5 s and about 16.7k output tokens.
- Logic: contradiction statistics report dozens of supporting hits despite no direct evidence; pre-review signal counts are not semantically aligned with accepted evidence.
- Schema: final benchmark output omits requested PFS/OS binary predictions.

## Promotion status

Candidate observations only. Do not create or activate a production skill until the same failure pattern recurs in later batch cases and is forward-tested on a held-out batch.
