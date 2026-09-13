# Case Report Pipeline Step Data

This archive contains the three main data layers used by the case-report pipeline.

## Directory Layout

- `step1_extracted/`
  - Source directory: `case pip/output/pre out/`
  - Files: `*_tr.json`
  - Meaning: full-text snapshots extracted from PMC/HTML pages by Trafilatura.
  - Main fields: `pmid`, `source_file`, `extraction_method`, `title`, `author`, `date`, `text`, `text_length`, `fingerprint`.
  - Recommended use: full-text semantic search, document-level retrieval, re-processing from extracted article text.

- `step2_cleaned/`
  - Source directory: `case pip/pipeline_10_batch/step2_cleaned/`
  - Files: `*_tr_cleaned.json`
  - Meaning: LLM-cleaned case-report text. The cleaning step judges whether an article is a case report and keeps the clinically relevant `Abstract + Case Presentation` text when applicable.
  - Main fields: `text`, `is_case_report`, `provenance`, `cleaning_stats`, `metadata`, `rejection_reason`.
  - Recommended use: low-noise embedding for clinical case retrieval, especially for treatment history, mutations, response, and progression descriptions.

- `step3_structured/`
  - Source directory: `case pip/pipeline_10_batch/step3_structured/`
  - Files: `*_tr_cleaned.json.json`
  - Meaning: LLM-extracted structured case schema for case reports that passed step 2.
  - Main fields: `provenance`, `baseline_clinical_profile`, `timeline_events`, `drug_interaction`, `outcome_summary`, `knowledge_annotation`.
  - Recommended use: structured filters, event-level retrieval, timeline construction, mutation/drug matching, and evidence grounding for a medical recommendation agent.

- `pipeline_report_20260607_143832.json`
  - Original run report for the batch pipeline.
  - Records input HTML paths, step status, cache status, and output files.

## Notes

- Step 1 contains extracted article text, not raw HTML. Raw HTML pages are stored separately under `case report/literature_pipeline/output/full_texts/`.
- In this batch, the recorded pipeline run executed steps 2 and 3 using existing step 1 outputs as cache/input.
- Step 3 is more suitable for precise filtering, but may omit details present in step 2. For a doctor-assistant or drug-recommendation agent, use both:
  - Step 3 for structured matching and hard filters.
  - Step 2 or step 1 text chunks for embedding-based semantic retrieval and original evidence.
- Do not embed the whole JSON string directly. Embed curated natural-language text and keep JSON fields as metadata.

## Suggested Hybrid Retrieval Design

1. Rewrite the user query into structured constraints, such as cancer type, mutation, drug, line of therapy, response, resistance, toxicity, and DDI.
2. Use `step3_structured` to filter or rank candidate cases and timeline events.
3. Use embeddings over `step2_cleaned.text` or event-level natural-language chunks to retrieve detailed evidence.
4. Return both structured fields and original text snippets to the answer generation step.

## Counts In This Archive

Counts were generated when the zip file was built and may differ from future pipeline outputs:

- Step 1 extracted files: 780
- Step 2 cleaned files: 780
- Step 3 structured files: 417

