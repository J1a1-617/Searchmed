#!/usr/bin/env bash
set -u

ROOT="${AGENT_ROOT:-/home/visitor/yangijiayi/skill_runs/local_agent_batch_001}"
PYTHON="${AGENT_PYTHON:-/home/visitor/miniconda3/envs/vllm-interns1/bin/python}"
DATA_ROOT="${BENCHMARK_DATA_ROOT:-${ROOT}/benchmark_package/data/full_queue}"
OUTPUT_ROOT="${BENCHMARK_OUTPUT_ROOT:-${ROOT}/results/full_benchmark_continuation_20260907}"
ARTIFACT_ROOT="${BENCHMARK_ARTIFACT_ROOT:-/slow_share/yangijiayi/skills1}"
EMBED_MODEL_PATH="${BENCHMARK_EMBED_MODEL_PATH:-${ROOT}/models/bge-large-zh-v1.5}"
RERANKER_MODEL_PATH="${BENCHMARK_RERANKER_MODEL_PATH:-/slow_share/yangijiayi/reranker_runtime/models/Qwen3-Reranker-4B}"
RUNTIME_INIT_TIMEOUT="${BENCHMARK_RUNTIME_INIT_TIMEOUT:-300}"
START_BATCH="${START_BATCH:-5}"
END_BATCH="${END_BATCH:-11}"

mkdir -p "${OUTPUT_ROOT}"

batch="${START_BATCH}"
while [ "${batch}" -le "${END_BATCH}" ]; do
  batch_id="$(printf '%03d' "${batch}")"
  data="${DATA_ROOT}/batch_${batch_id}.json"
  out_dir="${OUTPUT_ROOT}/batch_${batch_id}"
  mkdir -p "${out_dir}/checkpoints"
  configured_keys="${OPENAI_API_KEYS:-${OPENAI_API_KEY:-}}"
  IFS=',' read -r -a api_keys <<< "${configured_keys}"
  echo "[$(date -Is)] batch_${batch_id} start" 
  status=1
  key_offset=0
  while [ "${key_offset}" -lt "${#api_keys[@]}" ]; do
    key_index=$(( (batch - START_BATCH + key_offset) % ${#api_keys[@]} ))
    batch_api_key="${api_keys[${key_index}]}"
    echo "[$(date -Is)] batch_${batch_id} api_slot=${key_index} attempt"
    OPENAI_API_KEYS="${batch_api_key}" CUDA_VISIBLE_DEVICES="${BENCHMARK_GPU_ID:-6}" "${PYTHON}" -u \
      "${ROOT}/predictive_clinical_benchmark/run_eval.py" \
      --data "${data}" --model searchagent-gpt5 \
      --agent-system-id "${AGENT_SYSTEM_ID:-session_complete_v2_20260907}" \
      --output "${out_dir}/results.json" --checkpoint-dir "${out_dir}/checkpoints" \
      --resume --workers 1 --agent-index-root "${ROOT}/indexes" \
      --agent-embed-model-path "${EMBED_MODEL_PATH}" \
      --agent-reranker-backend qwen \
      --agent-reranker-model "${RERANKER_MODEL_PATH}" \
      --agent-reranker-device cuda --agent-top-k 8 --agent-max-steps 2 \
      --agent-max-total-steps 2 --agent-llm-max-calls 40 --agent-llm-timeout 900 \
      --agent-llm-max-retries 2 --agent-structured-attempts 1 --agent-case-retries 3 \
      --agent-runtime-init-timeout "${RUNTIME_INIT_TIMEOUT}" --agent-case-timeout 1800 \
      --agent-temporal-filter-mode cutoff --agent-artifact-root "${ARTIFACT_ROOT}" \
      --agent-full-llm-trace
    status=$?
    [ "${status}" -eq 0 ] && break
    key_offset=$((key_offset + 1))
  done
  echo "[$(date -Is)] batch_${batch_id} exit=${status}"
  if [ "${status}" -ne 0 ]; then
    echo "[$(date -Is)] queue blocked: every API slot failed for batch_${batch_id}"
    exit "${status}"
  fi
  batch=$((batch + 1))
done

echo "[$(date -Is)] queue finished"
