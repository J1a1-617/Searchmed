#!/usr/bin/env python3
"""
Predictive Clinical Benchmark — 评测入口脚本

用法:
    python run_eval.py --data benchmark_multinode.json --model gpt-4o
    python run_eval.py --data benchmark_multinode.json --model gpt-4o --judge-model gpt-4o --output results.json

环境变量:
    OPENAI_API_KEY     - API 密钥
    OPENAI_BASE_URL    - API 地址 (可选, 默认 https://api.openai.com/v1)

自定义模型接入: 修改 call_model() 函数即可
"""

import json
import os
import sys
import argparse
import re
import threading
import signal
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.runner import run_benchmark
from eval.parser import parse_model_output
from eval.bootstrap import bootstrap_ci, metric_m1_f1_from_results
from eval.prompts import construct_agent_query
from searchagent_retrieval.benchmark_runner import (
    PredictiveBenchmarkRuntime,
    run_predictive_benchmark_case,
)
from searchagent_retrieval.workflow_trace import WorkflowTrace


API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

def _agent_api_keys() -> list[str]:
    configured = os.environ.get("OPENAI_API_KEYS", "")
    keys = [value.strip() for value in configured.split(",") if value.strip()]
    if not keys and API_KEY.strip():
        keys = [API_KEY.strip()]
    return keys


def _safe_id(value: object) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return safe or "case"


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_checkpoints(checkpoint_dir: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not checkpoint_dir.exists():
        return records
    for path in checkpoint_dir.glob("*.json"):
        if path.name == "manifest.json":
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            if isinstance(record, dict) and record.get("instance_id") is not None:
                records[str(record["instance_id"])] = record
        except (OSError, json.JSONDecodeError):
            print(f"[WARN] 忽略损坏的 checkpoint: {path}")
    return records


class CaseTimeoutError(TimeoutError):
    pass


class RuntimeInitTimeoutError(TimeoutError):
    pass


class FatalBenchmarkRunError(BaseException):
    """Abort the batch so the outer supervisor can switch API accounts."""


def _case_timeout_handler(signum, frame):
    raise CaseTimeoutError("case exceeded agent-case-timeout")


def _runtime_init_timeout_handler(signum, frame):
    raise RuntimeInitTimeoutError("runtime initialization exceeded agent-runtime-init-timeout")


def _fatal_run_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in (
        "token quota is not enough",
        "pre_consume_token_quota_failed",
        "authenticationerror",
        "invalid token",
        "无效的令牌",
        "model_not_found",
        "无可用渠道",
        "runtime initialization exceeded",
    ))


def call_model(prompt: str, model_name: str = "gpt-4o") -> str:
    """调用模型 API。

    如果你用其他模型（DeepSeek、本地 vLLM、Ollama 等），在这里添加分支即可:
        elif model_name == "my-model":
            return your_model.generate(prompt)
    """
    from openai import OpenAI

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, max_retries=2)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=4096,
    )
    return response.choices[0].message.content


def call_llm_judge(prompt: str, judge_model: str = "gpt-4o") -> dict:
    """调用 LLM-Judge 进行辅助指标评分。"""
    raw = call_model(prompt, model_name=judge_model)
    parsed = parse_model_output(raw)
    if parsed is None:
        print(f"  [WARN] Judge 输出解析失败，使用默认 score=0")
        return {"score": 0}
    return parsed


def print_summary(results: dict):
    """打印评测结果摘要。"""
    meta = results["meta"]
    global_ = results["global"]
    comp = results["composite"]

    print()
    print("=" * 60)
    print("[BENCHMARK] Predictive Clinical Benchmark 评测结果")
    print("=" * 60)
    print(f"  总题数:       {meta['n_total']}")
    print(f"  成功解析:     {meta['n_instances']}")
    print(f"  解析失败率:   {meta['parse_failure_rate']:.1%}")

    print()
    print("[PRIMARY] 主指标 (Primary Metrics):")
    m1 = global_.get("M1", {}) or {}
    print(f"  M1 获益/不获益 F1:       {m1.get('f1', 'N/A')}  (Acc: {m1.get('accuracy', 'N/A')})")
    print(f"  Precision:               {m1.get('precision', 'N/A')}")
    print(f"  Recall:                  {m1.get('recall', 'N/A')}")

    print()
    print("[M5] 毒性:")
    m5 = global_.get("M5", {}) or {}
    print(f"  严重毒性 Recall:         {m5.get('severe_toxicity_recall', 'N/A')}")
    print(f"  严重毒性 Precision:      {m5.get('severe_toxicity_precision', 'N/A')}")
    print(f"  等级 ±1 准确率:          {m5.get('grade_tolerance_accuracy', 'N/A')}")

    print()
    print("[LLM-AS-JUDGE] A1-A4 平均分:")
    auxiliary = global_.get("auxiliary_avg", {}) or {}
    for dim in ["A1", "A2", "A3", "A4"]:
        print(f"  {dim}:                     {auxiliary.get(dim, 'N/A')}")

    print()
    print("[COMPOSITE] 综合得分:")
    print(f"  M1 F1 得分:    {comp['primary_score']:.4f}")
    print(f"  Judge 加权分:  {comp.get('auxiliary_score', 'N/A')}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Predictive Clinical Benchmark")
    parser.add_argument("--data", required=True, help="评测数据 JSON 文件")
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Only evaluate this instance ID; repeat to select multiple cases.",
    )
    parser.add_argument("--model", default="gpt-4o", help="待评测模型名")
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("BENCHMARK_JUDGE_MODEL") or None,
        help="A1-A4 LLM-Judge 模型；默认关闭，显式传模型名才启用",
    )
    parser.add_argument("--output", default="results.json", help="输出文件路径")
    parser.add_argument("--bootstrap", type=int, default=0, help="Bootstrap 重采样次数")
    parser.add_argument("--quiet", action="store_true", help="安静模式")
    parser.add_argument("--agent-index-root", type=Path, default=Path("indexes"), help="SearchAgent 索引根目录")
    parser.add_argument("--agent-embed-model-path", type=Path, default=None, help="SearchAgent dense query embedding 模型目录")
    parser.add_argument(
        "--agent-reranker-backend",
        default="qwen_with_llm_fallback",
        choices=["none", "llm", "qwen", "qwen_then_llm", "qwen_with_llm_fallback", "cross_encoder"],
        help="SearchAgent reranker 后端；默认使用 Qwen 本地 reranker，避免每条证据额外调用 LLM",
    )
    parser.add_argument("--agent-reranker-model", default=None, help="本地 reranker 模型路径或 HuggingFace 名称")
    parser.add_argument("--agent-reranker-device", default=None, help="reranker device，例如 cuda、cpu、mps")
    parser.add_argument("--agent-top-k", type=int, default=10, help="SearchAgent 单轮 top-k")
    parser.add_argument("--agent-max-steps", type=int, default=4, help="SearchAgent 初始检索轮数上限")
    parser.add_argument("--agent-max-total-steps", type=int, default=4, help="SearchAgent 总检索轮数硬上限")
    parser.add_argument("--agent-max-budget-extension", type=int, default=0, help="SearchAgent 单次扩容上限")
    parser.add_argument("--agent-llm-timeout", type=float, default=900.0, help="SearchAgent 单次 API 超时秒数")
    parser.add_argument("--agent-llm-max-retries", type=int, default=10, help="SearchAgent API 连接错误自动重试次数")
    parser.add_argument("--agent-structured-attempts", type=int, default=1, help="普通严格结构化调用最多尝试次数；Execution终止报告另有一次极简修复")
    parser.add_argument("--agent-llm-max-calls", type=int, default=40, help="每病例LLM调用硬上限")
    parser.add_argument("--agent-case-retries", type=int, default=3, help="SearchAgent fallback/error 单题重试次数")
    parser.add_argument("--agent-case-timeout", type=int, default=1800, help="单题无响应超时秒数；超时病例跳过且不计分")
    parser.add_argument("--agent-runtime-init-timeout", type=int, default=300, help="本地 embedding/reranker 冷启动超时秒数")
    parser.add_argument(
        "--agent-temporal-filter-mode",
        choices=["cutoff", "off"],
        default="cutoff",
        help="检索工具时间过滤；benchmark默认cutoff，off用于对照实验",
    )
    parser.add_argument(
        "--agent-external-knowledge-mode",
        choices=["enabled", "disabled"],
        default="enabled",
        help="扩展知识库消融；disabled 同时禁用显式扩展工具并从 BM25 语料中移除规则 chunk",
    )
    parser.add_argument("--workers", type=int, default=2, help="并行病例数；每个 worker 独占检索 runtime")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="逐病例 checkpoint 目录")
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 恢复，跳过已完成/已解析失败病例")
    parser.add_argument("--run-id", default=None, help="本次运行 ID；恢复时默认读取 checkpoint manifest")
    parser.add_argument(
        "--agent-system-id",
        default=os.environ.get("AGENT_SYSTEM_ID") or "unspecified",
        help="Agent implementation/cohort identifier used to prevent cross-version metric mixing",
    )
    default_agent_artifacts = Path("/slow_share/yangjiayi/skills1") if Path("/slow_share/yangjiayi/skills1").is_dir() else None
    parser.add_argument("--agent-artifact-root", type=Path, default=default_agent_artifacts, help="逐病例 SearchAgent session/trace 根目录")
    parser.add_argument(
        "--agent-full-llm-trace",
        action="store_true",
        help="保存每次 LLM 的完整 prompt、tool schema、原始返回和 function arguments",
    )
    parser.add_argument(
        "--skill-evolution-event-dir",
        type=Path,
        default=Path(os.environ["SKILL_EVOLUTION_EVENT_DIR"]) if os.environ.get("SKILL_EVOLUTION_EVENT_DIR") else None,
        help="完成每道 SearchAgent benchmark 后写入可持久化的 Skill evolution 事件；后台 worker 可自动消费",
    )
    args = parser.parse_args()

    if args.agent_full_llm_trace:
        os.environ["LLM_TRACE_FULL"] = "1"

    if not os.path.exists(args.data):
        print(f"[ERROR] 数据文件不存在: {args.data}")
        sys.exit(1)

    with open(args.data, "r", encoding="utf-8") as f:
        instances = json.load(f)

    if not isinstance(instances, list):
        print("[ERROR] 数据格式错误：顶层必须是 JSON 数组")
        sys.exit(1)

    if args.instance_id:
        requested_ids = set(args.instance_id)
        instances = [item for item in instances if str(item.get("instance_id")) in requested_ids]
        found_ids = {str(item.get("instance_id")) for item in instances}
        missing_ids = requested_ids - found_ids
        if missing_ids:
            print(f"[ERROR] 未找到 instance_id: {sorted(missing_ids)}")
            sys.exit(1)

    print(f"[OK] 加载 {len(instances)} 道预测题目")
    print(f"   待评测模型: {args.model}")
    print(f"   LLM-Judge:  {args.judge_model or '(跳过)'}")

    workers = max(1, args.workers)
    agent_api_keys = _agent_api_keys()
    if args.model.startswith("searchagent") and not agent_api_keys:
        print("[ERROR] SearchAgent requires OPENAI_API_KEYS or OPENAI_API_KEY")
        sys.exit(1)
    output_path = Path(args.output)
    checkpoint_dir = args.checkpoint_dir or output_path.with_suffix("").with_name(output_path.stem + "_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_dir / "manifest.json"
    manifest = {}
    if args.resume and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    run_id = args.run_id or manifest.get("run_id") or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    artifact_root = args.agent_artifact_root or output_path.with_suffix("").with_name(output_path.stem + "_agent_artifacts")
    _atomic_json_write(manifest_path, {
        "run_id": run_id,
        "data": str(Path(args.data).resolve()),
        "model": args.model,
        "workers": workers,
        "api_account_count": len(agent_api_keys) if args.model.startswith("searchagent") else 0,
        "artifact_root": str(artifact_root),
        "agent_system_id": args.agent_system_id,
        "external_knowledge_mode": args.agent_external_knowledge_mode,
    })
    resume_records = _load_checkpoints(checkpoint_dir) if args.resume else {}
    print(f"   Workers:    {workers}")
    if args.model.startswith("searchagent"):
        print(f"   API accounts:{len(agent_api_keys)} (worker-affine round robin)")
    print(f"   Run ID:     {run_id}")
    print(f"   Checkpoint: {checkpoint_dir}")

    runtime_local = threading.local()
    runtime_init_lock = threading.Lock()
    runtime_sequence = 0

    def get_runtime() -> PredictiveBenchmarkRuntime:
        nonlocal runtime_sequence
        runtime = getattr(runtime_local, "runtime", None)
        if runtime is None:
            # SentenceTransformer/GPU initialization is not reliable when two
            # worker threads enter it concurrently. Serialize only startup;
            # initialized worker-local runtimes still execute in parallel.
            with runtime_init_lock:
                runtime = getattr(runtime_local, "runtime", None)
                if runtime is None:
                    account_slot = runtime_sequence % len(agent_api_keys)
                    runtime_sequence += 1
                    runtime = PredictiveBenchmarkRuntime(
                        index_root=args.agent_index_root,
                        embed_model_path=args.agent_embed_model_path,
                        llm_timeout=max(1.0, args.agent_llm_timeout),
                        llm_max_retries=max(0, args.agent_llm_max_retries),
                        llm_structured_attempts=max(1, args.agent_structured_attempts),
                        llm_max_calls_per_case=max(1, args.agent_llm_max_calls),
                        api_key=agent_api_keys[account_slot],
                        base_url=BASE_URL,
                        account_slot=account_slot,
                        reranker_backend=args.agent_reranker_backend,
                        reranker_model=args.agent_reranker_model,
                        reranker_device=args.agent_reranker_device,
                        external_knowledge_enabled=args.agent_external_knowledge_mode == "enabled",
                    )
                    runtime_local.runtime = runtime
        return runtime

    def model_fn(prompt: str) -> str:
        # Non-agent models do not need per-instance artifacts.
        return call_model(prompt, model_name=args.model)

    def persist_attempt_artifacts(attempt_root: Path, result: dict, prompt: str) -> None:
        """Persist replayable LLM and agent-state artifacts for Skill mining."""
        telemetry = result.get("telemetry") or {}
        attempt_root.mkdir(parents=True, exist_ok=True)
        _atomic_json_write(attempt_root / "workflow_trace.json", result.get("workflow_trace", {}))
        _atomic_json_write(attempt_root / "telemetry.json", telemetry)
        _atomic_json_write(attempt_root / "llm_trace.json", {
            "case_id": result.get("session_id"),
            "benchmark_prompt": prompt,
            "calls": telemetry.get("llm_calls") or [],
        })
        _atomic_json_write(attempt_root / "retrieval_results.json", {
            "retrieval_plan": result.get("retrieval_plan") or {},
            "replan_decisions": result.get("replan_decisions") or [],
            "last_route_result": result.get("last_route_result") or {},
            "loop_steps": result.get("loop_steps") or [],
        })
        _atomic_json_write(attempt_root / "evidence_review.json", {
            "answer_memory": result.get("answer_memory") or {},
            "final_safety_review": result.get("final_safety_review") or result.get("safety_gate") or {},
            "loop_steps": result.get("loop_steps") or [],
        })
        _atomic_json_write(attempt_root / "answer_context.json", result.get("answer_context_summary") or {})
        _atomic_json_write(attempt_root / "session_memory.json", {
            "round_memories": result.get("round_memories") or [],
            "step_memories": result.get("step_memories") or [],
            "replanner_short_memory": result.get("replanner_short_memory") or {},
            "replanner_long_memory": result.get("replanner_long_memory") or {},
            "replanner_memory_events": result.get("replanner_memory_events") or [],
            "answer_memory": result.get("answer_memory") or {},
            "skill_runtime": result.get("skill_runtime") or {},
        })
        _atomic_json_write(attempt_root / "final_prediction.json", {
            "run_status": result.get("run_status"),
            "final_prediction_source": result.get("final_prediction_source"),
            "failed_stages": result.get("failed_stages") or [],
            "fallback_stages": result.get("fallback_stages") or [],
            "prediction": result.get("benchmark_output") or {},
        })
        trace = WorkflowTrace()
        trace.query = result.get("workflow_trace", {}).get("query", prompt)
        trace.steps = result.get("workflow_trace", {}).get("steps", [])
        _atomic_text_write(attempt_root / "workflow.txt", trace.to_text())

    def instance_model_fn(prompt: str, instance: dict) -> str:
        if args.model.startswith("searchagent"):
            case_id = _safe_id(instance["instance_id"])
            case_root = artifact_root / run_id / case_id
            last_error = None
            result = None
            for attempt in range(max(1, args.agent_case_retries)):
                attempt_case_root = case_root / f"attempt_{attempt + 1}"
                active_runtime = None
                alarm_enabled = args.workers == 1 and threading.current_thread() is threading.main_thread()
                try:
                    if alarm_enabled:
                        signal.signal(signal.SIGALRM, _runtime_init_timeout_handler)
                        signal.alarm(max(0, int(args.agent_runtime_init_timeout)))
                    active_runtime = get_runtime()
                    if alarm_enabled:
                        signal.alarm(0)
                    # Keep every attempt's workflow/telemetry for later skill
                    # mining; retries must never overwrite the previous trace.
                    session_root = attempt_case_root / "session"
                    if alarm_enabled:
                        signal.signal(signal.SIGALRM, _case_timeout_handler)
                        signal.alarm(max(0, int(args.agent_case_timeout)))
                    result = run_predictive_benchmark_case(
                        benchmark_prompt=prompt,
                        agent_query=construct_agent_query(instance),
                        index_root=args.agent_index_root,
                        embed_model_path=args.agent_embed_model_path,
                        top_k=args.agent_top_k,
                        max_steps=args.agent_max_steps,
                        max_total_steps=args.agent_max_total_steps,
                        max_budget_extension=args.agent_max_budget_extension,
                        session_id=f"{_safe_id(run_id)}__{case_id}__attempt{attempt+1}",
                        session_root=session_root,
                        runtime=active_runtime,
                        temporal_filter_mode=args.agent_temporal_filter_mode,
                        time_cutoff=instance.get("time_cutoff"),
                        external_knowledge_enabled=args.agent_external_knowledge_mode == "enabled",
                    )
                    if alarm_enabled:
                        signal.alarm(0)
                    # Persist even degraded attempts; they are excluded from
                    # metrics but are valuable training material for skills.
                    persist_attempt_artifacts(attempt_case_root, result, prompt)
                    if result.get("final_prediction_source") == "llm" and result.get("run_status") == "complete":
                        case_root = attempt_case_root
                        break
                    last_error = RuntimeError(f"fallback/degraded run: {result.get('failed_stages', [])}")
                except Exception as exc:
                    if args.workers == 1 and threading.current_thread() is threading.main_thread():
                        signal.alarm(0)
                    last_error = exc
                    # A failed structured call is valuable training data. The
                    # runtime still owns the complete per-call trace even when
                    # the agent never returned a normal result.
                    failed_calls = active_runtime.llm.trace_events() if active_runtime is not None else []
                    attempt_case_root.mkdir(parents=True, exist_ok=True)
                    _atomic_json_write(attempt_case_root / "llm_trace.json", {
                        "case_id": f"{_safe_id(run_id)}__{case_id}__attempt{attempt+1}",
                        "benchmark_prompt": prompt,
                        "calls": failed_calls,
                        "terminal_error": {"type": type(exc).__name__, "message": str(exc)},
                    })
                    _atomic_json_write(attempt_case_root / "telemetry.json", {
                        "run_status": "failed",
                        "final_prediction_source": "none",
                        "failed_stages": [str(failed_calls[-1].get("operation") or "unknown")] if failed_calls else ["unknown"],
                        "fallback_stages": [],
                        "llm_calls": failed_calls,
                    })
                    _atomic_json_write(attempt_case_root / "error.json", {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    })
                    print(f"[{instance['instance_id']}] attempt {attempt + 1}/{args.agent_case_retries} failed: {exc}", flush=True)
                    if _fatal_run_error(exc):
                        raise FatalBenchmarkRunError(f"fatal benchmark dependency error: {exc}") from exc
            if result is None or result.get("final_prediction_source") != "llm" or result.get("run_status") != "complete":
                raise RuntimeError(f"case skipped after {args.agent_case_retries} attempts: {last_error}")
            case_root.mkdir(parents=True, exist_ok=True)
            telemetry = result.get("telemetry") or {}
            persist_attempt_artifacts(case_root, result, prompt)
            if args.skill_evolution_event_dir is not None:
                event_id = f"{_safe_id(run_id)}__{case_id}"
                _atomic_json_write(args.skill_evolution_event_dir / f"{event_id}.json", {
                    "type": "benchmark.case.completed",
                    "event_id": event_id,
                    "created_at": datetime.now().isoformat(),
                    "run_id": run_id,
                    "agent_system_id": args.agent_system_id,
                    "artifact_path": str(case_root.resolve()),
                    "dataset_path": str(Path(args.data).resolve()),
                    "index_root": str(args.agent_index_root.resolve()),
                    "case": instance,
                    "prediction": result.get("benchmark_output") or {},
                    "run_status": result.get("run_status"),
                })
            llm_calls = telemetry.get("llm_calls") or []
            total_tokens = sum(int(call.get("total_tokens") or 0) for call in llm_calls)
            failed_calls = sum(1 for call in llm_calls if call.get("status") == "error")
            total_seconds = (telemetry.get("stage_timings") or {}).get("total")
            print(
                f"[{instance['instance_id']}] telemetry: status={result['run_status']} "
                f"prediction={result['final_prediction_source']} total={total_seconds}s "
                f"llm_calls={len(llm_calls)} llm_errors={failed_calls} tokens={total_tokens}",
                flush=True,
            )
            if result["final_prediction_source"] != "llm":
                raise RuntimeError(
                    "SearchAgent final prediction did not complete through the LLM; "
                    f"failed_stages={result['failed_stages']}"
                )
            return json.dumps(result["benchmark_output"], ensure_ascii=False)
        return call_model(prompt, model_name=args.model)

    llm_judge_fn = None
    if args.judge_model:
        def llm_judge_fn(prompt: str) -> dict:
            return call_llm_judge(prompt, judge_model=args.judge_model)

    print("\n>>> 评测进行中...\n")
    def checkpoint_fn(record: dict) -> None:
        record["agent_system_id"] = args.agent_system_id
        _atomic_json_write(checkpoint_dir / f"{_safe_id(record['instance_id'])}.json", record)

    try:
        results = run_benchmark(
            instances=instances,
            model_fn=model_fn,
            instance_model_fn=instance_model_fn,
            llm_judge_fn=llm_judge_fn,
            verbose=not args.quiet,
            workers=workers,
            resume_records=resume_records,
            checkpoint_fn=checkpoint_fn,
        )
    finally:
        # With one worker the runtime belongs to the main thread and can be
        # closed explicitly. Parallel runtimes are thread-local: their SQLite
        # handles must not be closed from the main thread and are released when
        # the executor worker exits (and, in all cases, at process exit).
        main_runtime = getattr(runtime_local, "runtime", None)
        if main_runtime is not None:
            main_runtime.close()

    print_summary(results)
    results.setdefault("meta", {})["agent_system_id"] = args.agent_system_id

    if args.bootstrap > 0 and results["per_instance"]:
        print(f"\n>>> Bootstrap {args.bootstrap} 次...")
        ci_m1 = bootstrap_ci(results["per_instance"], metric_m1_f1_from_results, n_bootstrap=args.bootstrap)
        print(f"  M1 F1 95% CI: {ci_m1['mean']:.4f} [{ci_m1['ci_lower']:.4f}, {ci_m1['ci_upper']:.4f}]")
        results["bootstrap"] = {"M1_f1": ci_m1}

    for r in results["per_instance"]:
        if "parsed_output" in r and "_ground_truth" in r["parsed_output"]:
            del r["parsed_output"]["_ground_truth"]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] 结果已保存至: {args.output}")
    if int((results.get("meta") or {}).get("n_instances") or 0) != int((results.get("meta") or {}).get("n_total") or 0):
        print("[INCOMPLETE] batch has missing complete+LLM results; supervisor must retry", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
