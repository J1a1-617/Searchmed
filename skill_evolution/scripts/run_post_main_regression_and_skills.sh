#!/usr/bin/env bash
set -eu

ROOT="${AGENT_ROOT:-/home/visitor/yangijiayi/skill_runs/local_agent_batch_001}"
PYTHON="${AGENT_PYTHON:-/home/visitor/miniconda3/envs/vllm-interns1/bin/python}"
MAIN_PID="${1:?usage: run_post_main_regression_and_skills.sh MAIN_PID}"
SYSTEM_ID="${AGENT_SYSTEM_ID:-session_complete_v2_20260907}"
SYSTEM_START_EPOCH="${AGENT_SYSTEM_START_EPOCH:-1788793238}"
MAIN_RESULTS="${MAIN_RESULTS:-${ROOT}/results/full_benchmark_continuation_20260907}"
REGRESSION_RESULTS="${REGRESSION_RESULTS:-${ROOT}/results/${SYSTEM_ID}_legacy_wrong_regression}"
ARTIFACT_ROOT="${BENCHMARK_ARTIFACT_ROOT:-/slow_share/yangijiayi/skills1}"

while kill -0 "${MAIN_PID}" 2>/dev/null; do sleep 60; done
cd "${ROOT}"

"${PYTHON}" skill_evolution/scripts/label_agent_system_checkpoints.py \
  --root "${MAIN_RESULTS}" --agent-system-id "${SYSTEM_ID}" \
  --modified-after-epoch "${SYSTEM_START_EPOCH}"

"${PYTHON}" skill_evolution/scripts/verify_benchmark_completion.py \
  --data-root benchmark_package/data/full_queue \
  --checkpoint-root "${MAIN_RESULTS}" --agent-system-id "${SYSTEM_ID}" || {
    echo "main benchmark incomplete; regression and Skill evolution are blocked"
    exit 2
  }

mkdir -p "${REGRESSION_RESULTS}/checkpoints"
CUDA_VISIBLE_DEVICES="${BENCHMARK_GPU_ID:-6}" "${PYTHON}" -u predictive_clinical_benchmark/run_eval.py \
  --data skill_evolution/batches/legacy_wrong_regression.json \
  --model searchagent-gpt5 --agent-system-id "${SYSTEM_ID}" \
  --output "${REGRESSION_RESULTS}/results.json" \
  --checkpoint-dir "${REGRESSION_RESULTS}/checkpoints" --resume --workers 1 \
  --agent-index-root indexes --agent-embed-model-path models/bge-large-zh-v1.5 \
  --agent-reranker-backend qwen \
  --agent-reranker-model /slow_share/yangijiayi/reranker_runtime/models/Qwen3-Reranker-4B \
  --agent-reranker-device cuda --agent-top-k 8 --agent-max-steps 2 --agent-max-total-steps 2 \
  --agent-llm-max-calls 40 --agent-llm-timeout 900 --agent-llm-max-retries 2 \
  --agent-structured-attempts 1 --agent-case-retries 3 --agent-runtime-init-timeout 300 \
  --agent-case-timeout 1800 \
  --agent-temporal-filter-mode cutoff --agent-artifact-root "${ARTIFACT_ROOT}" --agent-full-llm-trace

CUDA_VISIBLE_DEVICES="${SKILL_AB_GPU_ID:-7}" "${PYTHON}" -u skill_evolution/scripts/run_pending_skill_ab.py \
  --root . --data skill_evolution/batches/lineage_source_ab_case12.json \
  --output "${ARTIFACT_ROOT}/source_error_ab_lineage_case12_20260907" \
  --max-attempts 3 --retry-delay 30 --readiness-attempts 20 --top-k 8 --max-steps 2 \
  --skill-id clinical-evidence-lineage \
  --required-stage answer_memory --required-stage answer_context --required-stage generate

if [ -f "${ARTIFACT_ROOT}/source_error_ab_lineage_case12_20260907/summary.json" ]; then
  "${PYTHON}" skill_evolution/scripts/apply_skill_ab_gate.py \
    --registry skill_evolution/registry/registry.json \
    --summary "${ARTIFACT_ROOT}/source_error_ab_lineage_case12_20260907/summary.json" || true
fi

data_args=()
for data in benchmark_package/data/skill_dev_batch_001.json benchmark_package/data/skill_dev_batch_002.json benchmark_package/data/full_queue/batch_*.json; do
  data_args+=(--data "${data}")
done

"${PYTHON}" skill_evolution/scripts/summarize_agent_system_metrics.py \
  --agent-system-id "${SYSTEM_ID}" "${data_args[@]}" \
  --checkpoint-root "${MAIN_RESULTS}" \
  --output "${MAIN_RESULTS}/${SYSTEM_ID}_metrics.json"

"${PYTHON}" skill_evolution/scripts/summarize_agent_system_metrics.py \
  --agent-system-id "${SYSTEM_ID}" --data skill_evolution/batches/legacy_wrong_regression.json \
  --checkpoint-root "${REGRESSION_RESULTS}" \
  --output "${REGRESSION_RESULTS}/metrics.json"

"${PYTHON}" skill_evolution/scripts/update_skill_failure_pipeline.py \
  "${data_args[@]}" \
  --result results/latest_benchmark_batch002_gpu_20260902/results.json \
  --result results/latest_benchmark_batch003_gpu_20260902/results.json \
  --result results/latest_benchmark_batch004_gpu_20260903/results.json \
  --checkpoint-root "${MAIN_RESULTS}" --checkpoint-root "${REGRESSION_RESULTS}" \
  --artifact-root "${ARTIFACT_ROOT}" --registry skill_evolution/registry/registry.json \
  --output skill_evolution/registry/wrong_case_pipeline.json

evolve_data_args=()
for data in "${data_args[@]}"; do
  if [ "${data}" != "--data" ]; then evolve_data_args+=(--benchmark-data "${data}"); fi
done
PYTHON="${PYTHON}" PYTHONPATH=. "${PYTHON}" skill_evolution/scripts/evolve_skills.py \
  --manifest skill_evolution/batches/full_queue/manifest.json \
  --sessions "${ARTIFACT_ROOT}" --root . --cohort-size 20 --eval-every 100 \
  "${evolve_data_args[@]}" \
  --benchmark-result results/latest_benchmark_batch002_gpu_20260902/results.json \
  --benchmark-result results/latest_benchmark_batch003_gpu_20260902/results.json \
  --benchmark-result results/latest_benchmark_batch004_gpu_20260903/results.json \
  --checkpoint-root "${MAIN_RESULTS}" --checkpoint-root "${REGRESSION_RESULTS}"
