#!/usr/bin/env python3
"""Launch a full Codex CLI agent to diagnose sessions and propose Skills."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from skill_evolution.scripts.synthesize_skills import SKILL_SCHEMA


PROMPT = """你是 SearchAgent Skill Curator，不是一次性分类器。请使用文件与终端读取能力完成任务：
1. 完整阅读 cohort、audit、failure inventory 中每个来源错题指向的 Session artifacts；必要时检查当前仓库实现，定位错误真正发生的第一个 Agent 阶段。
2. 问题空间开放，不得只从已有四个 Skill 模板中选择。
3. 将代码 bug、API/超时、数据库覆盖缺失、一次性医学事实与可复用 Agent Skill 严格分开；前三者放入 discarded_patterns。
4. 对 explicit gold 直接检查其阶段轨迹；对 text_candidates_pending_codex 必须人工式逐条核验反查候选，只有同一患者、治疗阶段、时间点和结局事实均一致才接受为黄金证据。接受后用其 ID 对照各阶段并定位首次丢失点；候选只是主题相似时判 db_missing_gold，不能归因给检索。
5. 每个 Skill 必须对应至少一个真实来源失败题，明确作用阶段、触发条件、禁止行为和来源题 A/B 成功标准。不要把 ground-truth 结局直接写进 Skill。
6. 每个来源错题都必须写入 gold_evidence_resolutions，明确 decision、accepted_candidate_ids、first_loss_stage 和理由；只有 resolution 支持该归因时才能生成对应 Skill。
7. 只输出符合给定 JSON Schema 的最终对象；不要修改仓库或 Session。

输入文件：
- cohort: {cohort}
- deterministic audit: {audit}
- failure inventory: {failure_inventory}
- evidence availability audits: {evidence_audits}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--failure-inventory", type=Path, required=True)
    ap.add_argument("--evidence-audits", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--schema", type=Path, required=True)
    ap.add_argument("--model", default=os.environ.get("CODEX_CURATOR_MODEL") or "gpt-6-astra")
    ap.add_argument("--codex-binary", default=os.environ.get("CODEX_CLI_BINARY") or "codex")
    ap.add_argument("--workspace", type=Path, default=Path.cwd())
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.schema.write_text(json.dumps(SKILL_SCHEMA, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prompt = PROMPT.format(
        cohort=args.cohort.resolve(), audit=args.audit.resolve(),
        failure_inventory=args.failure_inventory.resolve(), evidence_audits=args.evidence_audits.resolve(),
    )
    command = [
        args.codex_binary, "exec", "--ephemeral", "--sandbox", "read-only",
        "--skip-git-repo-check", "--model", args.model,
        "--output-schema", str(args.schema.resolve()),
        "--output-last-message", str(args.output.resolve()),
        "--json", "--cd", str(args.workspace.resolve()), prompt,
    ]
    # Curator authentication is deliberately isolated from the benchmark LLM.
    # In particular, never let yeysai/OpenAI-compatible business credentials
    # override the Codex CLI's own ChatGPT/Codex login.
    codex_env = os.environ.copy()
    for name in (
        "OPENAI_API_KEY", "OPENAI_API_KEYS", "OPENAI_BASE_URL",
        "DEFAULT_OPENAI_API_KEY", "DEFAULT_BASE_URL", "MODEL_NAME",
    ):
        codex_env.pop(name, None)
    with args.trace.open("w", encoding="utf-8") as trace:
        completed = subprocess.run(
            command,
            stdout=trace,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=codex_env,
        )
    if completed.returncode:
        return completed.returncode
    try:
        value = json.loads(args.output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 3
    return 0 if isinstance(value, dict) and isinstance(value.get("skills"), list) else 4


if __name__ == "__main__":
    raise SystemExit(main())
