# Clinical evidence lineage: source-case synthesis

## Common failure

The recurring defect is not a particular drug or cancer type. Evidence meaning is changed while it moves through the downstream pipeline:

`EvidenceReview -> AnswerMemory -> AnswerContext -> Generate`

The changed dimension differs by case, but each failure breaks the same lineage tuple:

`(evidence_id, claim_id, population, intervention, comparator, endpoint, timepoint, direction, evidence_state, limitations)`

## Trigger families and sources

### `claim_scope_collision`

- `case_01_node_1`: sixteen heterogeneous observations became one answer-direction claim.
- `case_02_node_1`: direct systemic evidence and irrelevant CNS analogs became one claim.
- `case_03_node_1`: population efficacy, a negative molecular modifier, baseline imaging, and toxicity uncertainty crossed scope boundaries.

### `unexplained_evidence_drop`, `evidence_state_flip`, or `cross_intervention_attribution`

- `case_12_node_1`: missing target-regimen evidence risked becoming a negative conclusion, and progression under a prior regimen risked transfer to the target regimen.
- `case_01_node_1`: no direct finding survived, while unsupported endpoint directions were still generated.
- The same attribution invariant is exercised by `case_02_node_1` and `case_03_node_1`, where systemic, CNS, baseline, and prior-treatment outcomes must remain separate.

### `evidence_derived_conclusion_without_binding`

- Detected by the batch audit in `case_01_node_1`, `case_04_node_2`, `case_10_node_2`, `case_13_node_3`, `case_19_node_2`, and `case_26_node_2` when Context contained evidence IDs but Generate emitted no citations.
- This signal is only a review trigger, not proof of an error: in several cases all retained items were weak analogs, so not citing them was correct. The Skill activates only if the final conclusion actually relies on an evidence-derived statement without a valid binding.

## Routing boundary

Skill retrieval uses only stage and abstract anomaly tags. Concrete drugs, mutations, cancer types, and outcome values remain in the stage input and are processed after the Skill is loaded. Direct-evidence absence alone must not activate the Skill.
