# case_02_node_1 analysis

## Outcome

- Run status: complete; 25 LLM calls; 1342.5 s wall time.
- Correct: overall benefit `明显获益`, body RECIST `PR`, CSF `未评估`, toxicity grade 0.
- Under-called: CNS `SD` vs `PR`; symptoms `部分改善` vs `明显改善`.
- Missing schema: requested PFS/OS binary fields were discussed in rationale but not emitted as structured outputs.

## Stage diagnosis

1. Query understanding correctly captured EGFR 19del metastatic lung adenocarcinoma, baseline brain metastasis, short prior gefitinib exposure and planned first-line-like osimertinib monotherapy.
2. Planner separated systemic early response/PFS from CNS early response. This was an appropriate dimensional decomposition.
3. Retrieval did not find the requested RCT/FLAURA evidence, but did retrieve PMID 34532491 with an exact case-level match for osimertinib, EGFR 19del, 12-week PR, no significant early adverse events and 11-month disease control.
4. Replanner usefully added osimertinib aliases and switched retrieval channels. This improved recall for the systemic/PFS step. The CNS step still returned mostly irrelevant evidence.
5. Evidence Review handled the direct systemic case well, but again admitted low-scoring `target_entity_match=different` CNS candidates as analogue/partial support.
6. AnswerMemory again merged direct and irrelevant observations into one oversized claim. Answer Context repaired this by keeping one scoped key finding and explicitly prohibiting CNS extrapolation.
7. Generate used the scoped evidence appropriately for systemic PR, toxicity and likely PFS >=6 months. It conservatively predicted CNS SD because no admissible CNS evidence was available; this missed the observed CNS PR but was epistemically defensible given the retrieved context.

## Critical temporal leakage

- Benchmark cutoff: 2020-09-30.
- The decisive evidence, PMID 34532491, is indexed locally with `pub_date=2016-06-01`.
- PubMed identifies this article as Annals of Translational Medicine, August 2021 (DOI `10.21037/atm-21-3861`).
- Therefore the retrieval-time cutoff admitted post-cutoff evidence because the index publication date is wrong, likely confusing a patient timeline/event date with article publication date.
- The apparent evidence-assisted correctness for body PR, toxicity and PFS is not temporally valid for this benchmark run.

## Candidate reusable lessons

- Dimension-aware planning worked: separate systemic response/PFS from CNS response when baseline brain metastasis is present.
- Missing CNS evidence should remain an explicit gap; do not infer intracranial response from systemic response.
- Alias expansion is useful after exact trial-name queries fail in a case-report-heavy corpus.
- Evidence admission and memory granularity failures recur from case 1 and are stronger candidates for a shared skill.

## Non-skill backlog

- Data/index bug: publication-date provenance must come from bibliographic metadata, never case-event dates; rebuild temporal fields and the filtered indexes.
- Evaluation invalidation: re-run this case after the temporal metadata repair before using it as evidence that RAG improved the prediction.
- KB coverage: the corpus lacks the planned RCT/CNS subgroup evidence and is dominated by case reports.
- Performance: 10 rerank calls consumed 689.7 s and about 57k output tokens; total was 25 LLM calls.
- Schema: final output omits structured PFS/OS threshold predictions.

## Promotion status

Do not promote a skill from this case yet. The strongest apparent direct evidence is post-cutoff leakage. Retain only the planning and evidence-boundary patterns as candidates for held-out validation.
