#!/usr/bin/env bash
set -u

ROOT="${AGENT_ROOT:-/home/visitor/yangijiayi/skill_runs/local_agent_batch_001}"
PYTHON="${AGENT_PYTHON:-/home/visitor/miniconda3/envs/vllm-interns1/bin/python}"
SECRETS_FILE="${BENCHMARK_SECRETS_FILE:-/home/visitor/.config/searchagent/gpt54mini.env}"
DATA_ROOT="${BENCHMARK_DATA_ROOT:?BENCHMARK_DATA_ROOT is required}"
OUTPUT_ROOT="${BENCHMARK_OUTPUT_ROOT:?BENCHMARK_OUTPUT_ROOT is required}"
ARTIFACT_ROOT="${BENCHMARK_ARTIFACT_ROOT:?BENCHMARK_ARTIFACT_ROOT is required}"
EVENT_ROOT="${SKILL_EVOLUTION_EVENT_DIR:-${OUTPUT_ROOT}/skill_events}"
EMBED_MODEL_PATH="${BENCHMARK_EMBED_MODEL_PATH:-${ROOT}/models/bge-large-zh-v1.5}"
RERANKER_MODEL_PATH="${BENCHMARK_RERANKER_MODEL_PATH:-/slow_share/yangijiayi/reranker_runtime/models/Qwen3-Reranker-4B}"
START_BATCH="${START_BATCH:-5}"
END_BATCH="${END_BATCH:-11}"
SYSTEM_ID="${AGENT_SYSTEM_ID:?AGENT_SYSTEM_ID is required}"

# Credentials live outside the project/results tree and are never echoed.
set -a
. "${SECRETS_FILE}"
set +a
export MODEL_NAME="gpt-5.4-mini"
export OPENAI_BASE_URL="https://yeysai.com/v1"
export DEFAULT_BASE_URL="https://yeysai.com/v1"
export LLM_TRACE_FULL=1
export PYTHONPATH="${ROOT}"

mkdir -p "${OUTPUT_ROOT}" "${ARTIFACT_ROOT}" "${EVENT_ROOT}"

batch="${START_BATCH}"
while [ "${batch}" -le "${END_BATCH}" ]; do
  batch_id="$(printf '%03d' "${batch}")"
  data="${DATA_ROOT}/batch_${batch_id}.json"
  out_dir="${OUTPUT_ROOT}/batch_${batch_id}"
  mkdir -p "${out_dir}/checkpoints"
  case_count="$(${PYTHON} -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "${data}")"
  if [ "${case_count}" -eq 0 ]; then
    echo "[$(date -Is)] batch_${batch_id} skipped: no pending cases"
    batch=$((batch + 1))
    continue
  fi
  IFS=',' read -r -a api_keys <<< "${OPENAI_API_KEYS}"
  status=1
  key_offset=0
  echo "[$(date -Is)] batch_${batch_id} start cases=${case_count}"
  while [ "${key_offset}" -lt "${#api_keys[@]}" ]; do
    key_index=$(( (batch - START_BATCH + key_offset) % ${#api_keys[@]} ))
    export OPENAI_API_KEY="${api_keys[${key_index}]}"
    export OPENAI_API_KEYS="${api_keys[${key_index}]}"
    echo "[$(date -Is)] batch_${batch_id} api_slot=${key_index} attempt"
    CUDA_VISIBLE_DEVICES="${BENCHMARK_GPU_ID:-6}" "${PYTHON}" -u \
      "${ROOT}/predictive_clinical_benchmark/run_eval.py" \
      --data "${data}" --model searchagent-gpt54mini \
      --agent-system-id "${SYSTEM_ID}" \
      --output "${out_dir}/results.json" --checkpoint-dir "${out_dir}/checkpoints" \
      --resume --workers 1 --agent-index-root "${ROOT}/indexes" \
      --agent-embed-model-path "${EMBED_MODEL_PATH}" \
      --agent-reranker-backend qwen --agent-reranker-model "${RERANKER_MODEL_PATH}" \
      --agent-reranker-device cuda --agent-top-k 8 --agent-max-steps 2 \
      --agent-max-total-steps 2 --agent-llm-max-calls 40 --agent-llm-timeout 900 \
      --agent-llm-max-retries 2 --agent-structured-attempts 1 --agent-case-retries 3 \
      --agent-runtime-init-timeout "${BENCHMARK_RUNTIME_INIT_TIMEOUT:-1200}" --agent-case-timeout 1800 \
      --agent-temporal-filter-mode cutoff --agent-artifact-root "${ARTIFACT_ROOT}" \
      --agent-full-llm-trace --skill-evolution-event-dir "${EVENT_ROOT}"
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
