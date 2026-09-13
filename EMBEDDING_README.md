# SearchAgent Embedding 与向量索引说明

## 当前结果

GPU 全量构建已完成，输出位于：

```text
indexes_gpu_full/
```

构建结果：

| 向量空间 | 数量 | 维度 | 用途 |
|---|---:|---:|---|
| `case_semantic` | 27,459 | 768 | 病例正文语义检索 |
| `event_semantic` | 1,615 | 768 | 治疗、疗效、进展、毒性等事件检索 |

总计 29,074 个向量。构建 backend 为 `sentence_transformers`，不是 hashing。病例记录数为 780；BM25 文档数为 30,122。

GPU 构建耗时：`case_semantic` 424.4 秒，`event_semantic` 5.8 秒。硬件为 NVIDIA A800-SXM4-80GB，构建时使用 Torch `2.10.0+cu128`。

## 使用的模型

模型：

```text
damo/nlp_corom_sentence-embedding_chinese-base-medical
```

GPU 项目内路径：

```text
models/nlp_corom_sentence-embedding_chinese-base-medical
```

这是中文医疗领域 BERT embedding 模型。当前目录是 ModelScope/Transformers 格式，不包含 SentenceTransformer 的 `modules.json`，因此 SentenceTransformer 会自动创建 mean-pooling 包装；输出为 768 维向量，并执行 L2 normalization。检索时使用归一化查询向量与归一化文档向量的点积，等价于 cosine similarity。

日志中的 `pooler.dense.* newly initialized` 不影响当前 mean-pooling 句向量；检索不使用 BERT pooler 输出。

## 文本和 metadata 如何进入索引

`searchagent_retrieval.schema_parser` 从 `case_pipeline_steps_data` 读取统一病例记录和 evidence chunks。`index_builder` 只把具有 `text_for_embedding` 的以下 chunk 写入向量索引：

- `case_text_chunk` → `case_semantic`
- `event_chunk` → `event_semantic`

每个向量 item 同时保存原文和追溯 metadata，包括：

```text
doc_id, chunk_id, case_id, event_id, chunk_type, evidence_level,
source_file, field_path, start_char, end_char, chunk_order,
pmid, title, embedding_space
```

因此 dense/BM25 命中后可经 `fetch_evidence` 回到结构化数据库，再由 CitationAgent 生成 `claim → chunk_id → source_file` 映射。

## 全量构建命令

在 GPU 项目根目录运行：

```bash
mkdir -p indexes_gpu_full
cp indexes/structured.db indexes_gpu_full/structured.db
cp indexes/bm25.json indexes_gpu_full/bm25.json

.venv-reembed-cu128/bin/python build_indexes.py \
  --case-data-root case_pipeline_steps_data \
  --external-data-root data \
  --output-root indexes_gpu_full \
  --skip-structured \
  --skip-bm25 \
  --embed-model-path models/nlp_corom_sentence-embedding_chinese-base-medical \
  --require-sentence-transformers
```

后台构建脚本：

```bash
nohup ./run_gpu_full_reembed.sh \
  > gpu_reembed_logs/full_reembed.log 2>&1 < /dev/null &
```

状态与验证文件：

```text
gpu_reembed_logs/status.txt
gpu_reembed_logs/validation.json
indexes_gpu_full/manifest.json
```

## RetrievalTools Python 接口

初始化时必须同时指定新索引和同一个 embedding 模型：

```python
from pathlib import Path
from searchagent_retrieval.tools import RetrievalTools

tools = RetrievalTools(
    index_root=Path("indexes_gpu_full"),
    embed_model_path=Path("models/nlp_corom_sentence-embedding_chinese-base-medical"),
)

dense_hits = tools.dense_search(
    "EGFR exon 19 deletion osimertinib resistance",
    top_k=5,
    spaces=["case_semantic", "event_semantic"],
)

hybrid_hits = tools.hybrid_search(
    query="EGFR突变患者使用奥希替尼后进展",
    constraints={"gene_alterations": ["EGFR"]},
    top_k=5,
)

for hit in dense_hits:
    print(hit.id, hit.score, hit.metadata.get("source_file"))

tools.close()
```

加载时会进行两项强校验：

1. 存储 backend 是 `sentence_transformers` 时，查询 encoder 不允许回退 hashing。
2. 查询向量维度必须与索引矩阵维度一致；当前必须为 768。

## 命令行接口

只运行检索 Router：

```bash
.venv-reembed-cu128/bin/python run_retrieval.py \
  --query "EGFR突变患者使用奥希替尼后进展" \
  --index-root indexes_gpu_full \
  --embed-model-path models/nlp_corom_sentence-embedding_chinese-base-medical \
  --top-k 5
```

运行完整 Agent 工作流：

```bash
.venv-reembed-cu128/bin/python run_agent_loop.py \
  --query "EGFR突变患者使用奥希替尼后进展" \
  --index-root indexes_gpu_full \
  --embed-model-path models/nlp_corom_sentence-embedding_chinese-base-medical \
  --top-k 5 \
  --max-steps 3 \
  --print-workflow
```

完整 Agent 还需要 `.env` 中的 LLM 配置。纯 `RetrievalTools`、dense、BM25 和 hybrid 检索不需要 LLM。

## 切换为生产索引

当前 Agent 默认仍读取 `indexes/`。在没有额外复制的情况下，推荐显式传入：

```text
--index-root indexes_gpu_full
--embed-model-path models/nlp_corom_sentence-embedding_chinese-base-medical
```

不要只替换 `.npy` 文件。一个可用索引根目录必须保持以下文件来自同一构建版本：

```text
structured.db
bm25.json
manifest.json
vector/metadata.json
vector/vectors/case_semantic.npy
vector/vectors/event_semantic.npy
```

## 验证

```bash
cat gpu_reembed_logs/status.txt
cat gpu_reembed_logs/validation.json
cat indexes_gpu_full/manifest.json
```

验收条件：

- `status.txt` 为 `completed exit_code=0`。
- backend 为 `sentence_transformers`。
- shape 分别为 `[27459, 768]` 和 `[1615, 768]`。
- metadata 数量和矩阵行数一致。
- `chunk_id` 无重复。
- 每个 item 都有 `source_file` 和 `chunk_id`。
- dense/hybrid 查询可返回 768 维索引中的命中。
- 命中 metadata 可传给 CitationAgent 做源文件追溯。

## Agent 对齐验证结果

已在 GPU 上使用 `indexes_gpu_full` 和同一模型执行真实冒烟测试：

```text
query encoder backend: sentence_transformers
dense hits: 3
hybrid hits: 3
query/stored dimension: 768/768
```

dense 命中已返回 `embedding_space`、`chunk_id` 和 `source_file`。Agent 的标准路径为：

```text
dense/BM25/structured → hybrid/rerank → fetch_evidence
→ EvidenceReviewAgent → AnswerMemoryAgent → CitationAgent.trace_claims
```

因此新向量索引已与 Agent 的检索接口和 CitationAgent 的 `chunk_id` 引用接口对齐。

### 已知的源文件可移植性边界

构建验证保证每个向量 item 的 `source_file` 字段非空，但不保证该路径在 GPU 上一定存在。历史数据中有两种来源：

- `40989771_tr.json` 这类相对文件名，可映射到项目病例数据目录。
- `/Users/jiayi/.../full_texts/41772468.html` 这类开发机绝对路径；它保留了原始来源字符串，但对应 HTML 没有随项目上传到 GPU 时，不能在 GPU 直接打开。

此外，hybrid 命中若由 structured 分量提供顶层 metadata，`source_file` 可能暂时为空。Agent 不应直接把 hybrid 顶层 metadata 当最终引用，而应调用 `fetch_evidence`，使用结构化数据库中的 `citation_json` 取得最终 citation anchor。

如果要求“GPU 上每个引用都能直接打开源文件”，还需要单独迁移缺失的 HTML/PDF 原文，并在重建前把绝对 `source_file` 规范化为相对项目路径；这不要求重新训练模型，但修改 metadata 后需要重建或迁移向量 metadata 与结构化 citation 数据。
