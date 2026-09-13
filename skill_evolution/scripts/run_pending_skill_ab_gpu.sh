#!/usr/bin/env bash
set -u

ROOT="${AGENT_ROOT:-/home/visitor/yangijiayi/skill_runs/local_agent_batch_001}"
PYTHON="${AGENT_PYTHON:-/home/visitor/miniconda3/envs/vllm-interns1/bin/python}"
DATA="${SKILL_AB_DATA:-${ROOT}/skill_evolution/batches/structured_lineage_ab_4.json}"
OUTPUT="${SKILL_AB_OUTPUT:-/slow_share/yangijiayi/skills1/pending_skill_paired_ab_20260906}"

exec "${PYTHON}" "${ROOT}/skill_evolution/scripts/run_pending_skill_ab.py" \
  --root "${ROOT}" \
  --data "${DATA}" \
  --output "${OUTPUT}" \
  --max-attempts 3 \
  --readiness-attempts 20 \
  --retry-delay 30 \
  --top-k 8 \
  --max-steps 2
