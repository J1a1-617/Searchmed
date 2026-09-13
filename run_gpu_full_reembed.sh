#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LOG_DIR="$ROOT/gpu_reembed_logs"
STATUS_FILE="$LOG_DIR/status.txt"
MODEL="$ROOT/models/nlp_corom_sentence-embedding_chinese-base-medical"
OUTPUT="$ROOT/indexes_gpu_full"
PYTHON="$ROOT/.venv-reembed-cu128/bin/python"

mkdir -p "$LOG_DIR"
echo "running pid=$$ started_at=$(date -Is)" > "$STATUS_FILE"
trap 'code=$?; if [ "$code" -eq 0 ]; then state=completed; else state=failed; fi; echo "$state exit_code=$code finished_at=$(date -Is)" > "$STATUS_FILE"' EXIT

test -f "$MODEL/pytorch_model.bin"
test -f "$ROOT/indexes/structured.db"
test -f "$ROOT/indexes/bm25.json"

test -x "$PYTHON"
"$PYTHON" -m pip install --timeout 180 --retries 20 \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  "sentence-transformers==3.4.1" scipy scikit-learn

"$PYTHON" - <<'PY'
import torch
from sentence_transformers import SentenceTransformer

model_path = "models/nlp_corom_sentence-embedding_chinese-base-medical"
assert torch.cuda.is_available(), "CUDA is not available; refusing to run full embedding"
model = SentenceTransformer(model_path)
print({
    "torch": torch.__version__,
    "cuda": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0),
    "device": str(model.device),
    "dimension": model.get_sentence_embedding_dimension(),
}, flush=True)
assert str(model.device).startswith("cuda"), model.device
PY

mkdir -p "$OUTPUT"
cp "$ROOT/indexes/structured.db" "$OUTPUT/structured.db"
cp "$ROOT/indexes/bm25.json" "$OUTPUT/bm25.json"

"$PYTHON" build_indexes.py \
  --case-data-root case_pipeline_steps_data \
  --external-data-root data \
  --output-root indexes_gpu_full \
  --skip-structured \
  --skip-bm25 \
  --embed-model-path models/nlp_corom_sentence-embedding_chinese-base-medical \
  --require-sentence-transformers

"$PYTHON" - <<'PY'
import json
from pathlib import Path
import numpy as np

root = Path("indexes_gpu_full")
metadata = json.loads((root / "vector" / "metadata.json").read_text(encoding="utf-8"))
assert metadata["backend"] == "sentence_transformers", metadata["backend"]
seen = set()
missing_source = []
result = {"backend": metadata["backend"], "spaces": {}}
for space in ("case_semantic", "event_semantic"):
    items = metadata["space_items"][space]
    matrix = np.load(root / "vector" / "vectors" / f"{space}.npy", mmap_mode="r")
    assert matrix.shape[0] == len(items), (space, matrix.shape, len(items))
    result["spaces"][space] = {"items": len(items), "shape": list(matrix.shape)}
    for item in items:
        doc_id = item["doc_id"]
        assert doc_id not in seen, doc_id
        seen.add(doc_id)
        meta = item.get("metadata") or {}
        if not meta.get("source_file") or not meta.get("chunk_id"):
            missing_source.append(doc_id)
assert seen
assert not missing_source, missing_source[:20]
(Path("gpu_reembed_logs") / "validation.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(result, flush=True)
PY
