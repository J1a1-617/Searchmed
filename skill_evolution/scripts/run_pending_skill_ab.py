#!/usr/bin/env python3
"""Run paired dynamic-Skill A/B with one reusable retrieval runtime."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


POSITIVE_BENEFIT = {"明显获益", "有限获益或稳定"}


def _binary_correct(case: dict[str, Any], prediction: dict[str, Any] | None) -> bool | None:
    if not isinstance(prediction, dict):
        return None
    truth = str((case.get("ground_truth") or {}).get("overall_benefit") or "")
    predicted = str(prediction.get("overall_benefit") or "")
    if not truth or not predicted:
        return None
    return (truth in POSITIVE_BENEFIT) == (predicted in POSITIVE_BENEFIT)


def _skill_observation(row: dict[str, Any] | None, skill_id: str, required_stages: set[str]) -> dict[str, Any]:
    state = (row or {}).get("skill_runtime") or {}
    candidates = [
        str(value.get("skill_id") or "")
        for value in state.get("candidate_skills") or []
        if isinstance(value, dict)
    ]
    selected = [str(value) for value in state.get("selected_skill_ids") or []]
    loaded = [str(value) for value in state.get("loaded_skill_ids") or []]
    stage = str(state.get("stage") or "")
    observed_stages = {stage} if stage else set()
    for activation in state.get("stage_activations") or []:
        if not isinstance(activation, dict):
            continue
        if skill_id in {str(value) for value in activation.get("loaded_skill_ids") or []}:
            observed_stages.add(str(activation.get("stage") or ""))
    observed_stages.discard("")
    return {
        "stage": stage,
        "candidate": skill_id in candidates,
        "selected": skill_id in selected,
        "loaded": skill_id in loaded,
        "observed_stages": sorted(observed_stages),
        "stage_matches": not required_stages or required_stages.issubset(observed_stages),
        "validation": state.get("validation"),
        "query": state.get("query"),
    }


def _load_smoke_module(root: Path):
    path = root / "benchmark_package" / "scripts" / "smoke_agent_comutation.py"
    spec = importlib.util.spec_from_file_location("skill_ab_smoke_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load benchmark runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=30.0)
    parser.add_argument("--readiness-attempts", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--skill-id", default="")
    parser.add_argument("--required-stage", action="append", default=[])
    args = parser.parse_args()

    root = args.root.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "benchmark_package" / "code"))
    smoke = _load_smoke_module(root)
    from eval.agent_adapter import adapt_agent_output_to_v2
    from searchagent_retrieval.benchmark_runner import PredictiveBenchmarkRuntime

    instances = json.loads(args.data.read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = PredictiveBenchmarkRuntime(
        index_root=root / "indexes",
        embed_model_path=root / "models" / "bge-large-zh-v1.5",
        reranker_backend="qwen",
        reranker_model=os.environ.get("RERANKER_MODEL"),
        reranker_device=os.environ.get("RERANKER_DEVICE") or "cuda",
    )
    summary: dict[str, Any] = {
        "model": runtime.llm.model_name,
        "data": str(args.data),
        "arms": {"a_no_dynamic": [], "b_dynamic": []},
    }
    required_stages = {str(value) for value in args.required_stage if str(value)}
    previous_allowlist = os.environ.get("DYNAMIC_SKILL_ALLOWLIST")
    try:
        ready = False
        for readiness_attempt in range(1, max(1, args.readiness_attempts) + 1):
            try:
                reply = runtime.llm.chat(
                    "Return exactly READY.",
                    "A/B runner readiness check",
                    max_output_tokens=32,
                )
                ready = bool(reply.strip())
            except BaseException as exc:
                print(
                    f"readiness attempt={readiness_attempt} error={type(exc).__name__}: {exc}",
                    flush=True,
                )
            if ready:
                print(f"readiness attempt={readiness_attempt} ready", flush=True)
                break
            if readiness_attempt < args.readiness_attempts:
                time.sleep(max(0.0, min(float(args.retry_delay), 60.0)))
        if not ready:
            raise RuntimeError("API did not become ready within readiness-attempts")

        for case in instances:
            instance_id = str(case["instance_id"])
            for arm, enabled in (("a_no_dynamic", False), ("b_dynamic", True)):
                os.environ["DYNAMIC_SKILLS_ENABLED"] = "1" if enabled else "0"
                if enabled and args.skill_id:
                    os.environ["DYNAMIC_SKILL_ALLOWLIST"] = args.skill_id
                else:
                    os.environ.pop("DYNAMIC_SKILL_ALLOWLIST", None)
                attempts = []
                selected = None
                for attempt in range(1, max(1, args.max_attempts) + 1):
                    attempt_root = output / arm / instance_id / f"attempt_{attempt}"
                    attempt_root.mkdir(parents=True, exist_ok=True)
                    started = time.perf_counter()
                    try:
                        row = smoke.run_agent_case(
                            instance=case,
                            runtime=runtime,
                            index_root=root / "indexes",
                            embed_model_path=root / "models" / "bge-large-zh-v1.5",
                            session_root=attempt_root / "session",
                            max_steps=args.max_steps,
                            max_total_steps=args.max_steps,
                            top_k=args.top_k,
                            temporal_filter_mode="cutoff",
                        )
                    except BaseException as exc:
                        row = {
                            "instance_id": instance_id,
                            "run_status": "error",
                            "final_prediction_source": None,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    row["wall_seconds"] = round(time.perf_counter() - started, 4)
                    _write_json(attempt_root / "result.json", row)
                    attempts.append({
                        "attempt": attempt,
                        "run_status": row.get("run_status"),
                        "final_prediction_source": row.get("final_prediction_source"),
                        "wall_seconds": row["wall_seconds"],
                        "error": row.get("error"),
                    })
                    print(
                        f"{instance_id} {arm} attempt={attempt} "
                        f"status={row.get('run_status')} seconds={row['wall_seconds']}",
                        flush=True,
                    )
                    if row.get("run_status") == "complete":
                        selected = row
                        break
                    if attempt < args.max_attempts:
                        time.sleep(max(0.0, min(float(args.retry_delay), 60.0)))
                prediction = (selected or {}).get("benchmark_output")
                record = {
                    "instance_id": instance_id,
                    "selected_status": (selected or {}).get("run_status") or "no_complete_attempt",
                    "attempts": attempts,
                    "prediction": prediction,
                    "adapted_prediction": adapt_agent_output_to_v2(prediction) if prediction else None,
                    "binary_correct": _binary_correct(case, prediction),
                    "skill_observation": _skill_observation(selected, args.skill_id, required_stages)
                    if args.skill_id else None,
                }
                summary["arms"][arm].append(record)
                _write_json(output / "summary.partial.json", summary)
    finally:
        if previous_allowlist is None:
            os.environ.pop("DYNAMIC_SKILL_ALLOWLIST", None)
        else:
            os.environ["DYNAMIC_SKILL_ALLOWLIST"] = previous_allowlist
        runtime.close()

    for arm, rows in summary["arms"].items():
        predictions = [row.get("adapted_prediction") for row in rows]
        summary.setdefault("scores", {})[arm] = smoke.score_predictions(instances, predictions)
    if args.skill_id:
        pairs = []
        b_by_id = {row["instance_id"]: row for row in summary["arms"]["b_dynamic"]}
        for baseline in summary["arms"]["a_no_dynamic"]:
            intervention = b_by_id.get(baseline["instance_id"]) or {}
            observation = intervention.get("skill_observation") or {}
            baseline_wrong = baseline.get("binary_correct") is False
            intervention_correct = intervention.get("binary_correct") is True
            mounted_correctly = all([
                observation.get("candidate"),
                observation.get("selected"),
                observation.get("loaded"),
                observation.get("stage_matches"),
            ])
            efficacy_pass = baseline_wrong and intervention_correct
            callability_pass = mounted_correctly
            pairs.append({
                "instance_id": baseline["instance_id"],
                "baseline_wrong_reproduced": baseline_wrong,
                "intervention_correct": intervention_correct,
                "mounted_at_required_stage": mounted_correctly,
                "efficacy_pass": efficacy_pass,
                "callability_pass": callability_pass,
                "pass": efficacy_pass and callability_pass,
            })
        eligible = [row for row in pairs if row["baseline_wrong_reproduced"]]
        efficacy_pass = bool(eligible) and all(row["efficacy_pass"] for row in eligible)
        callability_pass = bool(eligible) and all(row["callability_pass"] for row in eligible)
        summary["source_error_gate"] = {
            "skill_id": args.skill_id,
            "required_stages": sorted(required_stages),
            "pairs": pairs,
            "eligible_source_error_count": len(eligible),
            "efficacy_gate": {"status": "pass" if efficacy_pass else "fail", "metric": "source_error_corrected"},
            "callability_gate": {"status": "pass" if callability_pass else "fail", "requirements": ["retrieved", "selected", "loaded", "declared_stage_observed"]},
            "status": "pass" if efficacy_pass and callability_pass else "fail",
            "requirements": [
                "baseline_reproduces_binary_error",
                "skill_arm_corrects_binary_error",
                "skill_is_candidate_selected_and_loaded",
                "load_occurs_at_declared_call_stage",
            ],
        }
    _write_json(output / "summary.json", summary)
    print(json.dumps({
        arm: [row["selected_status"] for row in rows]
        for arm, rows in summary["arms"].items()
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
