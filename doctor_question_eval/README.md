# 医生问题引用评测集（独立于 predictive benchmark）

这套评测用于覆盖 `predictive_clinical_benchmark` 测不到的能力，重点是：

- 医生问题形态下的检索表现（而不是结构化预测输入）
- 引用锚点命中（`chunk_id/doc_id/pmid`）
- 引用可溯源完整性（`source_file + field_path`）

> 该目录不会修改 `predictive_clinical_benchmark` 的任何文件或指标定义。

## 文件说明

- `doctor_questions_citation_eval.json`：15 条医生问题及 gold 引用锚点
- `run_citation_eval.py`：评测脚本，复用当前 `searchagent_retrieval` 主链路

## 评测指标

- `chunk_hit_rate`：每题是否命中至少一个 gold `chunk_id`
- `doc_hit_rate`：每题是否命中至少一个 gold `doc_id`
- `pmid_hit_rate`：每题是否命中至少一个 gold `pmid`
- `mean_chunk_recall`：每题 gold `chunk_id` 召回率均值
- `mean_precision_at_fetch`：`fetch_evidence` 中命中 gold `chunk_id` 的比例
- `mrr_chunk`：按 `fetch_evidence` 返回顺序计算的 MRR
- `mean_anchor_integrity`：返回引用是否具有完整锚点（`chunk_id/doc_id/source_file/field_path`）

## 使用方法

在仓库根目录运行：

```bash
python doctor_question_eval/run_citation_eval.py \
  --dataset doctor_question_eval/doctor_questions_citation_eval.json \
  --index-root indexes \
  --top-k 10 \
  --output doctor_question_eval/citation_eval_results.json
```

## 输出

输出文件包含：

- `summary`：整体指标
- `by_capability`：按能力类型分组指标
- `per_question`：逐题命中、rank、召回率、引用锚点完整性

## 注意

- 该评测默认依赖本地已构建索引 `indexes/`。
- `pmid_hit_rate` 依赖 `fetch_evidence` 是否返回 `pmid`；若为 0，不代表 `doc/chunk` 级命中一定失败，可结合 `doc_hit_rate/chunk_hit_rate` 一起看。
