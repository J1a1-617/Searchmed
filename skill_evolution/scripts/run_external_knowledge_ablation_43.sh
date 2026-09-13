#!/usr/bin/env bash
set -u

ROOT="${AGENT_ROOT:-/home/visitor/yangijiayi/skill_runs/local_agent_batch_001}"
PYTHON="${AGENT_PYTHON:-/home/visitor/miniconda3/envs/vllm-interns1/bin/python}"
SECRETS_FILE="${BENCHMARK_SECRETS_FILE:-/home/visitor/.config/searchagent/gpt54mini.env}"
BASELINE_EVENT_DIR="${BASELINE_EVENT_DIR:-${ROOT}/results/session_complete_v2_gpt54mini_20260909_090053/skill_events}"
RUN_NAME="${ABLATION_RUN_NAME:-external_knowledge_disabled_43_gpt54mini_20260912}"
OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT:-${ROOT}/results/${RUN_NAME}}"
ARTIFACT_ROOT="${ABLATION_ARTIFACT_ROOT:-/slow_share/yangijiayi/skills1/${RUN_NAME}}"
DATA_PATH="${OUTPUT_ROOT}/cohort_43_from_baseline_events.json"
RERANKER_MODEL_PATH="${BENCHMARK_RERANKER_MODEL_PATH:-/slow_share/yangijiayi/reranker_runtime/models/Qwen3-Reranker-4B}"

set -a
. "${SECRETS_FILE}"
set +a
export MODEL_NAME="gpt-5.4-mini"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://yeysai.com/v1}"
export DEFAULT_BASE_URL="${DEFAULT_BASE_URL:-${OPENAI_BASE_URL}}"
export LLM_TRACE_FULL=1
export PYTHONPATH="${ROOT}"

mkdir -p "${OUTPUT_ROOT}" "${ARTIFACT_ROOT}" "${OUTPUT_ROOT}/skill_events"
"${PYTHON}" "${ROOT}/scripts/build_dataset_from_skill_events.py" \
  --event-dir "${BASELINE_EVENT_DIR}" --output "${DATA_PATH}"

CUDA_VISIBLE_DEVICES="${BENCHMARK_GPU_ID:-6}" "${PYTHON}" -u \
  "${ROOT}/predictive_clinical_benchmark/run_eval.py" \
  --data "${DATA_PATH}" --model searchagent-gpt54mini-no-external-kb \
  --agent-system-id "${RUN_NAME}" \
  --output "${OUTPUT_ROOT}/results.json" \
  --checkpoint-dir "${OUTPUT_ROOT}/checkpoints" --resume --workers 1 \
  --agent-index-root "${ROOT}/indexes" \
  --agent-embed-model-path "${ROOT}/models/bge-large-zh-v1.5" \
  --agent-reranker-backend qwen --agent-reranker-model "${RERANKER_MODEL_PATH}" \
  --agent-reranker-device cuda --agent-top-k 8 --agent-max-steps 2 \
  --agent-max-total-steps 2 --agent-llm-max-calls 40 --agent-llm-timeout 900 \
  --agent-llm-max-retries 2 --agent-structured-attempts 1 --agent-case-retries 3 \
  --agent-runtime-init-timeout 1200 --agent-case-timeout 1800 \
  --agent-temporal-filter-mode cutoff --agent-external-knowledge-mode disabled \
  --agent-artifact-root "${ARTIFACT_ROOT}" --agent-full-llm-trace \
  --skill-evolution-event-dir "${OUTPUT_ROOT}/skill_events"
