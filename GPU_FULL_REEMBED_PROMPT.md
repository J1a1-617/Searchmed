# GPU 全量重新 Embedding 与向量索引构建 Prompt

你正在 GPU 服务器上的 `searchagentv2` 项目中工作。请为全部病例数据重新计算 embedding 并重建向量索引，同时保留现有结构化数据库和 BM25 索引。

## 目标

全量重建以下向量空间：

- `case_semantic`
- `event_semantic`

新的向量索引必须为每个 chunk 保留完整的源文件追溯 metadata：

- `chunk_id`
- `doc_id`
- `case_id`
- `event_id`
- `chunk_type`
- `evidence_level`
- `source_file`
- `field_path`
- `start_char`
- `end_char`
- `chunk_order`
- `pmid`
- `title`
- `embedding_space`

## 执行边界

- 不重建或覆盖 `indexes/structured.db`。
- 不重建或覆盖 `indexes/bm25.json`。
- 不使用 `--limit-records` 或 `--record-offset`，必须覆盖全部记录。
- 不使用 `--append-embedding`，本次需要生成干净、完整的新向量索引。
- 不改变 chunk 切分、chunk ID 或原始文本。
- 不静默退回 hashing encoder。必须确认使用医疗领域 SentenceTransformer 模型；如果模型不可用，停止并报告，不要继续构建伪向量索引。
- 不删除当前可用索引。优先输出到新的临时目录，验证成功后再由用户决定是否替换生产索引。

## 第一步：检查环境

在执行前检查：

```bash
pwd
python --version
nvidia-smi
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"
python -c "import sentence_transformers; print(sentence_transformers.__version__)"
```

确认以下数据存在：

```bash
test -d case_pipeline_steps_data
test -d data
test -f indexes/structured.db
test -f indexes/bm25.json
```

## 第二步：配置 embedding 模型

当前代码中的默认模型路径可能是开发机上的本地路径：

```text
/Users/jiayi/.cache/modelscope/hub/models/damo/nlp_corom_sentence-embedding_chinese-base-medical
```

该路径在 GPU 服务器上通常无效。先检查 `searchagent_retrieval/vector_index.py` 中的 `DEFAULT_EMBED_MODEL_PATH`，再定位 GPU 服务器上实际的医疗 embedding 模型目录。

要求：

1. 使用与项目预期一致的 `damo/nlp_corom_sentence-embedding_chinese-base-medical`，或者由用户明确指定的替代模型。
2. 模型必须能被 `SentenceTransformer` 正常加载。
3. 构建日志最终必须显示 backend 为 `sentence_transformers`，不能是 `hashing`。
4. 如果需要修改模型路径，优先为构建脚本增加可配置参数或环境变量，不要把 GPU 服务器私有路径永久硬编码进通用代码。

加载检查示例：

```bash
python -c "from pathlib import Path; from sentence_transformers import SentenceTransformer; p=Path('/实际模型目录'); assert p.exists(), p; m=SentenceTransformer(str(p)); print('model_loaded')"
```

## 第三步：先构建到隔离目录

建议先创建一个隔离输出目录，例如：

```bash
python build_indexes.py \
  --case-data-root case_pipeline_steps_data \
  --external-data-root data \
  --output-root indexes_gpu_full \
  --skip-structured \
  --skip-bm25 \
  --embed-model-path models/nlp_corom_sentence-embedding_chinese-base-medical \
  --require-sentence-transformers
```

但是 `--skip-structured` 和 `--skip-bm25` 会要求相应文件已存在于目标输出目录。执行前，应采用非破坏方式让隔离目录复用现有文件，例如复制：

```bash
mkdir -p indexes_gpu_full
cp indexes/structured.db indexes_gpu_full/structured.db
cp indexes/bm25.json indexes_gpu_full/bm25.json
```

然后执行全量向量构建命令：

```bash
python build_indexes.py \
  --case-data-root case_pipeline_steps_data \
  --external-data-root data \
  --output-root indexes_gpu_full \
  --skip-structured \
  --skip-bm25 \
  --embed-model-path models/nlp_corom_sentence-embedding_chinese-base-medical \
  --require-sentence-transformers
```

不要加入以下参数：

```text
--skip-embedding
--append-embedding
--limit-records
--record-offset
```

## 第四步：验证构建结果

确认文件存在：

```bash
test -f indexes_gpu_full/vector/metadata.json
test -f indexes_gpu_full/vector/vectors/case_semantic.npy
test -f indexes_gpu_full/vector/vectors/event_semantic.npy
test -f indexes_gpu_full/manifest.json
```

运行以下 Python 校验。可以保存为临时脚本执行，但不要修改向量数据：

```python
import json
from pathlib import Path

import numpy as np

root = Path("indexes_gpu_full")
metadata = json.loads((root / "vector" / "metadata.json").read_text(encoding="utf-8"))

assert metadata.get("backend") == "sentence_transformers", metadata.get("backend")

required = {
    "doc_id",
    "chunk_id",
    "case_id",
    "chunk_type",
    "evidence_level",
    "source_file",
    "field_path",
    "chunk_order",
    "embedding_space",
}

total = 0
missing_source = []
duplicate_ids = []
seen = set()

for space in ("case_semantic", "event_semantic"):
    items = metadata["space_items"].get(space, [])
    matrix = np.load(root / "vector" / "vectors" / f"{space}.npy", mmap_mode="r")
    assert matrix.shape[0] == len(items), (space, matrix.shape, len(items))
    assert matrix.ndim == 2 and matrix.shape[1] > 0, (space, matrix.shape)

    for item in items:
        chunk_id = item.get("doc_id")
        meta = item.get("metadata") or {}
        if chunk_id in seen:
            duplicate_ids.append(chunk_id)
        seen.add(chunk_id)
        missing = sorted(key for key in required if key not in meta or meta.get(key) is None)
        # event_id/start_char/end_char 允许因 chunk 类型不同而为空；核心来源字段不能缺失。
        if not meta.get("source_file") or not meta.get("chunk_id"):
            missing_source.append({"chunk_id": chunk_id, "missing": missing})
    total += len(items)

assert total > 0
assert not duplicate_ids, duplicate_ids[:20]
assert not missing_source, missing_source[:20]
print({"total_vectors": total, "spaces": metadata.get("spaces"), "backend": metadata.get("backend")})
```

额外验证：

1. 随机抽查至少 20 个向量 item。
2. 确认 `source_file` 指向的数据文件存在。
3. 对具有 `field_path/start_char/end_char` 的记录，从源 JSON 读取对应字段并验证片段与索引文本一致。
4. 使用 `RetrievalTools` 分别执行至少一次 `dense_search` 和 `hybrid_search`。
5. 确认命中结果的 metadata 包含 `source_file`。
6. 使用 `CitationAgent.trace_claims` 验证一个命中 chunk 能生成源文件引用映射。

检索冒烟测试示例：

```python
from pathlib import Path

from searchagent_retrieval.tools import RetrievalTools

tools = RetrievalTools(index_root=Path("indexes_gpu_full"))
hits = tools.dense_search("EGFR mutation osimertinib response", top_k=5)
assert hits
for hit in hits:
    assert hit.metadata.get("source_file"), hit.id
    print(hit.id, hit.score, hit.metadata.get("source_file"))
```

## 第五步：报告结果，不自动替换生产索引

完成后向用户报告：

- 实际使用的 embedding 模型和路径
- encoder backend
- GPU 型号
- `case_semantic` 向量数量和维度
- `event_semantic` 向量数量和维度
- 总耗时
- metadata 缺失数量
- 重复 chunk ID 数量
- 随机源文件追溯抽查结果
- dense/hybrid 检索冒烟测试结果
- 新索引目录大小
- 新索引位置

除非用户明确授权，不要删除或覆盖 `indexes/vector`，也不要自动把 `indexes_gpu_full/vector` 替换为生产向量索引。

## 验收标准

只有同时满足以下条件才算完成：

- 全量记录已参与 embedding。
- 两个向量空间均成功生成且非空。
- 使用 `sentence_transformers`，没有退回 hashing。
- metadata item 数量与向量矩阵行数完全一致。
- chunk ID 无重复。
- 每个病例向量都有可用的 `source_file` 和 `chunk_id`。
- dense search 能返回结果及源文件 metadata。
- CitationAgent 能从命中 chunk 生成源文件追溯信息。
- 原有生产索引未被破坏。
