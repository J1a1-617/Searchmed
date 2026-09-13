# RAG 数据库 Schema、地址与抽取链路

本文档对应服务器项目：

`/home/visitor/yangijiayi/benchmark_package——`

当前 RAG 通道包括病例库（case）、临床证据（A）、知识主张（B）、队列研究（C）、人工指南规则（G）和细粒度指南摘录（Gextract）。最近的 43 题二分类 A/B 实验仅使用 case，不包含 G/A/B/C。

## 1. 数据源总览

| 通道 | 内容 | 规模 | 数据地址 |
|---|---|---:|---|
| case | PubMed 病例报告及真实结局 | 574 条 | `/home/visitor/yangijiayi/benchmark_package——/source_data/pubmed_merged_574.json` |
| A | 临床试验、队列、RWE、病例系列等临床证据 | 360 条 | `/home/visitor/yangijiayi/benchmark_package——/source_data/knowledge_rag/clinical_evidence.jsonl` |
| B | 指南、共识、系统综述和机制/选药知识主张 | 129 条 | `/home/visitor/yangijiayi/benchmark_package——/source_data/knowledge_rag/knowledge_claim.jsonl` |
| C | 结构化队列研究 | 364 个 JSON | `/home/visitor/yangijiayi/benchmark_package——/source_data/cohort/step3_cohort/` |
| G | 人工整理的指南基础规则 | 9 条 | `/home/visitor/yangijiayi/benchmark_package——/source_data/knowledge_rag/guideline_basics_v23.md` |
| Gextract | 带来源、驱动基因和日期的指南摘录 | 12 条 | `/home/visitor/yangijiayi/benchmark_package——/source_data/knowledge_rag/guideline_extracts_v23.json` |

最终挂载数据：

- v23 gated：`/home/visitor/yangijiayi/benchmark_package——/data/rag_v23_gated.json`
- case-only：`/home/visitor/yangijiayi/benchmark_package——/data/rag_v23_cases.json`
- G-only：`/home/visitor/yangijiayi/benchmark_package——/data/rag_v23_g_only.json`
- B-only：`/home/visitor/yangijiayi/benchmark_package——/data/rag_v23_b_only.json`
- Gextract-only：`/home/visitor/yangijiayi/benchmark_package——/data/rag_v23_g_extract.json`
- case+G：`/home/visitor/yangijiayi/benchmark_package——/data/rag_v24_cases_g.json`
- case+B：`/home/visitor/yangijiayi/benchmark_package——/data/rag_v24_cases_b.json`

## 2. Case 病例库

### Schema

```json
{
  "instance_id": "pubmed_40308508",
  "case_id": "pubmed_40308508",
  "time_cutoff": "2019-07-01",
  "pub_date": "YYYY-MM-DD",
  "input": {
    "disease_background": {
      "diagnosis": "...",
      "metastatic_sites": ["..."],
      "molecular_profile": {
        "primary_mutation": "...",
        "resistance_mutations": [],
        "bypass_alterations": [],
        "co_mutations": []
      }
    },
    "prior_treatment_timeline": [{
      "regimen": "...", "start_date": "...", "end_date": "...",
      "best_response": "CR/PR/SD/PD", "reason_for_discontinuation": "..."
    }],
    "current_status": {
      "symptoms": [], "imaging": "...", "csf": "...", "performance_status": "ECOG ..."
    },
    "planned_treatment": {
      "drugs": [{"name": "...", "dose": "...", "route": "..."}],
      "combination_strategy": "..."
    }
  },
  "ground_truth": {
    "overall_benefit": "明显获益/有限获益或稳定/无明显获益/进展或有害",
    "body_lesion_recist": "CR/PR/SD/PD/NA",
    "cns_lm_recist": "CR/PR/SD/PD/NA",
    "csf_trajectory": "...",
    "symptom_trajectory": "...",
    "toxicity": {"max_grade": 0, "event": "...", "has_toxicity_record": true},
    "key_evidence_items": [],
    "pfs_months": null,
    "na_flags": {}
  }
}
```

### 源文件与抽取流程

1. 从 PubMed Central HTML 读取正文并去除 HTML 标签。
2. 用 LLM 按患者拆分病例，抽取诊断、分子特征、既往治疗、当前状态、计划方案以及治疗后真实结局。
3. 合并为 `pubmed_merged_574.json`。
4. 用 NCBI/PubMed 信息补 `pub_date`，支持按题目 `time_cutoff` 防止未来信息泄漏。
5. 用 BGE embedding 建病例向量索引，检索时结合时间过滤和结构化特征 rerank。

脚本：

- PubMed 病例抽取：`/home/visitor/yangijiayi/benchmark_package——/scripts/extract_pubmed_cases.py`
- 本地脱敏病历抽取：`/home/visitor/yangijiayi/benchmark_package——/scripts/extract_cases.py`
- 补发表日期：`/home/visitor/yangijiayi/benchmark_package——/scripts/add_pub_date_to_merged.py`
- 通用病例索引/检索：`/home/visitor/yangijiayi/benchmark_package——/scripts/rag_retriever.py`
- v23 BGE 索引加载：`/home/visitor/yangijiayi/benchmark_package——/scripts/bge_store.py`
- v23 最终过滤与挂载：`/home/visitor/yangijiayi/benchmark_package——/scripts/attach_rag_v23.py`

## 3. A：Clinical evidence

### Schema

```json
{
  "doc_id": "pmid_...",
  "pmid": "...",
  "pmcid": null,
  "pub_date": "YYYY-MM-DD",
  "title": "...",
  "url": "https://pubmed.ncbi.nlm.nih.gov/...",
  "source_file": "...",
  "doc_type": "clinical_evidence",
  "evidence_level": "RCT/phase3/phase2/phase1/prospective_cohort/retrospective/case_series/case_report/NA",
  "population": {
    "disease": "NSCLC",
    "stage": "...",
    "driver_mutations": [],
    "co_mutations": [],
    "resistance_context": [],
    "cns_status": null,
    "n_patients": null
  },
  "arms": [{
    "arm_id": "...",
    "regimen": "...",
    "drugs": [],
    "line": "...",
    "n_patients": null,
    "outcomes": {
      "orr": null,
      "dcr": null,
      "mPFS_months": null,
      "mOS_months": null,
      "pfs_6mo_rate": null,
      "os_6mo_rate": null,
      "benefit_summary": "clear_benefit/limited_benefit/no_benefit/mixed/NA",
      "key_toxicities": [{"event": "...", "grade": null, "rate": null}]
    }
  }],
  "clinical_implication": "...",
  "source_excerpt": "..."
}
```

### 源文件与抽取流程

上游网页正文地址：

`/home/visitor/yangijiayi/benchmark_package——/source_data/knowledge_texts/`

流程：

1. 读取保存的 PubMed/PMC 网页文本，清除导航栏和网页噪声，按 PMID/PMCID/URL 去重。
2. 通过 NCBI ESummary/ELink 补 PMID、PMCID 和 `pub_date`。
3. LLM 将文献分成 `clinical_evidence`、`knowledge_claim` 或 `discard`。
4. 对 A 类抽取人群、分子背景、耐药背景、治疗臂、ORR/DCR/PFS/OS 和毒性。
5. 中间结果先写入 `extracted_ab.jsonl`，之后拆分为 A/B 成品文件。

主抽取脚本：

`/home/visitor/yangijiayi/benchmark_package——/scripts/extract_knowledge_rag_ab.py`

相关文件：

- 清洗中间数据：`/home/visitor/yangijiayi/benchmark_package——/source_data/knowledge_rag/cleaned.jsonl`
- A/B 合并抽取结果：`/home/visitor/yangijiayi/benchmark_package——/source_data/knowledge_rag/extracted_ab.jsonl`
- 丢弃文献：`/home/visitor/yangijiayi/benchmark_package——/source_data/knowledge_rag/discarded.jsonl`
- 抽取 checkpoint：`/home/visitor/yangijiayi/benchmark_package——/source_data/knowledge_rag/extract_ckpt.json`
- A 检索附加脚本：`/home/visitor/yangijiayi/benchmark_package——/scripts/add_evidence_rag.py`
- 真命中审计/硬过滤：`/home/visitor/yangijiayi/benchmark_package——/scripts/audit_true_hits.py`

## 4. B：Knowledge claim

### Schema

```json
{
  "doc_id": "pmid_...",
  "pmid": "...",
  "pmcid": null,
  "pub_date": "YYYY-MM-DD",
  "title": "...",
  "url": "...",
  "source_file": "...",
  "doc_type": "knowledge_claim",
  "evidence_level": "guideline/consensus/systematic_review_meta/narrative_review/mechanistic_review/expert_opinion/NA",
  "applies_to": {
    "disease": "NSCLC",
    "driver_mutations": [],
    "co_mutations": [],
    "resistance_context": [],
    "cns_status": null,
    "treatment_setting": null
  },
  "claim": "...",
  "related_regimens": [],
  "clinical_implication": "...",
  "source_excerpt": "..."
}
```

B 与 A 共用同一批原始网页文本和同一个 LLM 分类抽取脚本：

`/home/visitor/yangijiayi/benchmark_package——/scripts/extract_knowledge_rag_ab.py`

后续检索/挂载脚本：

- `/home/visitor/yangijiayi/benchmark_package——/scripts/add_evidence_rag.py`
- `/home/visitor/yangijiayi/benchmark_package——/scripts/attach_rag_v23.py`

## 5. C：Cohort

### Schema

```json
{
  "provenance": {"source_filename": "..."},
  "study_design": {
    "study_type": "retrospective_cohort/prospective_cohort/...",
    "primary_endpoint": "PFS/OS/...",
    "statistical_methods": [],
    "study_period": {},
    "arms": 1
  },
  "population": {
    "n_screened": null,
    "n_enrolled": null,
    "n_analyzed": null,
    "inclusion_criteria": [],
    "exclusion_criteria": [],
    "baseline_summary": "...",
    "biomarker_profile": []
  },
  "treatment_arms": [{
    "arm_name": "...",
    "arm_type": "...",
    "drugs": [],
    "drug_category": "...",
    "line_of_therapy": null,
    "n": null,
    "dosage": {},
    "efficacy": {
      "orr_percent": null,
      "dcr_percent": null,
      "median_pfs_months": null,
      "pfs_rate_12mo_percent": null,
      "median_os_months": null,
      "os_rate_12mo_percent": null,
      "median_dor_months": null
    },
    "safety": {
      "grade_3_plus_ae_rate_percent": null,
      "discontinuation_due_to_ae_percent": null,
      "treatment_related_death_n": null,
      "grade_3_plus_ae": [],
      "any_grade_ae_list": []
    }
  }],
  "comparative_analyses": [],
  "subgroup_analyses": [],
  "treatment_summary": {},
  "conclusions": {}
}
```

现有成品和派生表：

- 清洗队列：`/home/visitor/yangijiayi/benchmark_package——/source_data/cohort/step3_cohort/`
- 药物/突变/DDI 派生表：`/home/visitor/yangijiayi/benchmark_package——/source_data/cohort/step4_tables/`
- 最终检索与挂载：`/home/visitor/yangijiayi/benchmark_package——/scripts/attach_rag_v23.py`

注意：当前 benchmark 包中没有找到从原始论文生成 `step3_cohort` 的上游抽取脚本。每条数据仅通过 `provenance.source_filename` 保留了上游文件名。因此 C 的成品 schema 和检索代码存在，但完整抽取 pipeline 在本包中不完整，不能从该目录单独复现。

## 6. G 与 Gextract：指南

### G schema

G 是 Markdown 中编号的人工短规则，每条隐含以下字段：

```json
{
  "text": "指南规则文本",
  "pub_date": "知识时点",
  "source": "NCCN/ESMO/ASCO/..."
}
```

原始指南 PDF：

`/home/visitor/yangijiayi/benchmark_package——/source_data/guidelines_lung_bm/`

该目录包含 NCCN NSCLC、NCCN CNS、EANO-ESMO LM/脑转移、ASCO-SNO-ASTRO、ASTRO、CSCO 和中国罕见靶点指南等 PDF。

### Gextract schema

```json
{
  "id": "ge_combo_pfs",
  "pub_date": "2023-04-01",
  "source": "ESMO oncogene-addicted metastatic NSCLC 2023",
  "drivers": ["EGFR"],
  "text": "指南摘录/归纳文本"
}
```

索引和挂载脚本：

- `/home/visitor/yangijiayi/benchmark_package——/scripts/bge_store.py`
- `/home/visitor/yangijiayi/benchmark_package——/scripts/attach_rag_v23.py`

注意：仓库中未找到从 PDF 自动生成 `guideline_basics_v23.md` 或 `guideline_extracts_v23.json` 的脚本；这两份文件应视为人工整理/审核后的派生数据，不是当前可一键复现的自动抽取结果。

## 7. 最终注入 prompt 前的统一 schema

case 作为 few-shot 消息注入；G/A/B/C 先压缩成 `_kb_context`：

```json
{
  "_inject_mode": "skip/parametric/parametric_g/idiosyncratic/retrieval",
  "_channels": "cases/G/B/Gextract/casesG/casesB/gated",
  "_rag_references": [{
    "rank": 1,
    "similarity": 0.83,
    "reference_gt": "明显获益",
    "reference_instance": {"input": {}, "ground_truth": {}}
  }],
  "_kb_context": {
    "guidelines": [{"rank": 1, "text": "...", "pub_date": "...", "similarity": 0.61, "source": "..."}],
    "reviews": [{"rank": 1, "text": "...", "pub_date": "...", "similarity": 0.60}],
    "evidence": [{"rank": 1, "text": "...", "pub_date": "..."}],
    "cohorts": [{"rank": 1, "text": "...", "pub_date": "..."}]
  }
}
```

统一挂载实现：

`/home/visitor/yangijiayi/benchmark_package——/scripts/attach_rag_v23.py`

prompt 渲染实现：

`/home/visitor/yangijiayi/benchmark_package——/code/eval/prompts.py`

## 8. 当前链路的可复现性结论

- **case：基本可复现。** 有 HTML 抽取、日期补全、索引、检索和挂载脚本；但仍需确认最初 HTML 文件的外部抓取位置。
- **A/B：链路最完整。** 原始保存网页、清洗、NCBI 补元数据、LLM 分类抽取、中间结果和挂载脚本都在。
- **C：部分可复现。** 结构化成品、provenance 和挂载存在，但原始论文到 `step3_cohort` 的抽取脚本不在当前 benchmark 包。
- **G/Gextract：不可自动复现。** 原 PDF 与人工整理成品存在，但没有发现 PDF 到规则/摘录的生成脚本。

