"""
评测主循环模块
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import numpy as np

from .metrics import (
    compute_binary_benefit,
    compute_weighted_kappa,
    compute_recist_metrics,
    compute_direction_accuracy,
    compute_toxicity_metrics,
    compute_calibration,
    compute_composite_score,
    CSF_MAPPING,
    SYMPTOM_MAPPING,
)
from .parser import parse_model_output, validate_output
from .prompts import (
    construct_inference_prompt,
    format_judge_prompt_A1,
    format_judge_prompt_A2,
    format_judge_prompt_A3,
    format_judge_prompt_A4,
)
from .na_filter import get_na_skip_dimensions
from .compliance import compute_temporal_compliance, compute_global_compliance


def run_benchmark(
    instances: list[dict],
    model_fn: Callable[[str], str],
    llm_judge_fn: Callable[[str], dict] = None,
    verbose: bool = True,
    workers: int = 1,
    instance_model_fn: Optional[Callable[[str, dict], str]] = None,
    resume_records: Optional[dict[str, dict]] = None,
    checkpoint_fn: Optional[Callable[[dict], None]] = None,
    primary_only: bool = True,
) -> dict:
    """完整评测主循环。

    Args:
        instances: 预测题目列表，每条包含 input 和 ground_truth
        model_fn: 模型推理函数，输入 prompt 字符串，输出 raw 响应字符串
        llm_judge_fn: LLM-Judge 函数（可选），输入 judge prompt，输出评分 JSON dict
                     如果为 None，则跳过辅助指标，只计算主指标
        verbose: 是否打印逐题进度

    Returns:
        {
            'per_instance': [{instance_id, parsed_output, validation, compliance, auxiliary}],
            'global': {M1, M2_kappa, M3_body, M3_cns, M4_csf, M4_symptom, M5, M6, M7_compliance, auxiliary_avg},
            'composite': {primary_score, auxiliary_score, composite_score},
            'meta': {n_instances, n_total, parse_failure_rate}
        }
    """
    workers = max(1, int(workers))
    resume_records = resume_records or {}
    result_by_id: dict[str, dict] = {}
    completed_failures: set[str] = set()

    for instance_id, record in resume_records.items():
        status = record.get("status")
        if status == "success" and isinstance(record.get("result"), dict):
            result_by_id[str(instance_id)] = record["result"]
        # Parse failures are deliberately not restored as completed. A later
        # supervisor attempt (possibly with another API key) must retry them.

    def evaluate_one(inst: dict) -> dict:
        instance_id = str(inst["instance_id"])
        gt = inst.get("ground_truth", {})
        prompt = construct_inference_prompt(inst)
        try:
            raw_output = (
                instance_model_fn(prompt, inst)
                if instance_model_fn is not None
                else model_fn(prompt)
            )
            parsed = parse_model_output(raw_output)
            if parsed is None:
                return {"instance_id": instance_id, "status": "parse_failure"}

            valid, missing = validate_output(parsed)
            parsed["_ground_truth"] = gt
            skip_dims = get_na_skip_dimensions(inst)
            cited = parsed.get("cited_evidence", [])
            compliance = compute_temporal_compliance(cited, inst["time_cutoff"])

            aux = {"A1": None, "A2": None, "A3": None, "A4": None}
            if llm_judge_fn is not None:
                aux_formatters = {
                    "A1": format_judge_prompt_A1,
                    "A2": format_judge_prompt_A2,
                    "A3": format_judge_prompt_A3,
                    "A4": format_judge_prompt_A4,
                }
                for dim, formatter in aux_formatters.items():
                    try:
                        judge_prompt = formatter(inst, raw_output)
                        judged = llm_judge_fn(judge_prompt)
                        score = judged.get("score", 0) if isinstance(judged, dict) else 0
                        aux[dim] = int(score)
                    except Exception:
                        aux[dim] = None

            return {
                "instance_id": instance_id,
                "status": "success",
                "result": {
                    "instance_id": inst["instance_id"],
                    "parsed_output": parsed,
                    "validation": {"valid": valid, "missing_fields": missing},
                    "compliance": compliance,
                    "auxiliary": aux,
                    "_skip_dimensions": sorted(skip_dims),
                },
            }
        except Exception as exc:
            return {
                "instance_id": instance_id,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

    pending = [
        inst for inst in instances
        if str(inst["instance_id"]) not in result_by_id
    ]
    if verbose and resume_records:
        print(f"[RESUME] 已恢复 {len(result_by_id)} 个成功病例；失败/解析失败病例将重试")

    def accept(record: dict) -> None:
        instance_id = str(record["instance_id"])
        if checkpoint_fn is not None:
            checkpoint_fn(record)
        if record["status"] == "success":
            result_by_id[instance_id] = record["result"]
            if verbose:
                print(f"[{instance_id}] 完成")
        elif record["status"] == "parse_failure":
            completed_failures.add(instance_id)
            if verbose:
                print(f"[{instance_id}] [WARN] 输出解析失败")
        elif verbose:
            print(f"[{instance_id}] [ERROR] {record.get('error', 'unknown error')}")

    if workers == 1:
        for inst in pending:
            if verbose:
                print(f"[{inst['instance_id']}] 评测中...")
            accept(evaluate_one(inst))
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="benchmark") as pool:
            futures = {pool.submit(evaluate_one, inst): inst for inst in pending}
            for future in as_completed(futures):
                accept(future.result())

    # Restore input ordering regardless of parallel completion order.
    per_instance_results = [
        result_by_id[str(inst["instance_id"])]
        for inst in instances
        if str(inst["instance_id"]) in result_by_id
    ]

    # 聚合容器
    agg = {
        "y_true_benefit": [],
        "y_pred_benefit": [],
        "y_true_overall": [],
        "y_pred_overall": [],
        "y_true_body": [],
        "y_pred_body": [],
        "y_true_cns": [],
        "y_pred_cns": [],
        "y_true_csf": [],
        "y_pred_csf": [],
        "y_true_symptom": [],
        "y_pred_symptom": [],
        "y_true_tox": [],
        "y_pred_tox": [],
        "y_true_bin": [],
        "y_pred_bin": [],
        "conf": [],
        "compliance_results": [],
        "aux_scores": {"A1": [], "A2": [], "A3": [], "A4": []},
    }

    instance_by_id = {str(inst["instance_id"]): inst for inst in instances}
    for case_result in per_instance_results:
        inst = instance_by_id[str(case_result["instance_id"])]
        gt = inst.get("ground_truth", {})
        parsed = case_result["parsed_output"]
        skip_dims = set(case_result.pop("_skip_dimensions", get_na_skip_dimensions(inst)))

        # Collect data for global aggregation.
        agg["y_true_benefit"].append(gt.get("overall_benefit", ""))
        agg["y_pred_benefit"].append(parsed.get("overall_benefit", ""))
        agg["y_true_overall"].append(gt.get("overall_benefit", ""))
        agg["y_pred_overall"].append(parsed.get("overall_benefit", ""))

        if "M3_body" not in skip_dims:
            agg["y_true_body"].append(gt.get("body_lesion_recist", "NA"))
            agg["y_pred_body"].append(parsed.get("body_lesion_recist", "NA"))

        if "M3_cns" not in skip_dims:
            agg["y_true_cns"].append(gt.get("cns_lm_recist", "NA"))
            agg["y_pred_cns"].append(parsed.get("cns_lm_recist", "NA"))

        if "M4_csf" not in skip_dims:
            agg["y_true_csf"].append(gt.get("csf_trajectory", "未评估"))
            agg["y_pred_csf"].append(parsed.get("csf_trajectory", "未评估"))

        if "M4_symptom" not in skip_dims:
            agg["y_true_symptom"].append(gt.get("symptom_trajectory", ""))
            agg["y_pred_symptom"].append(parsed.get("symptom_trajectory", ""))

        # Toxicity (always collect, M5 has its own double-track logic)
        agg["y_true_tox"].append(gt.get("toxicity", {"grade": 0}))
        agg["y_pred_tox"].append(parsed.get("toxicity", {"grade": 0}))

        # Binary benefit + confidence for M1 & M6
        yt_bin = (
            1
            if gt.get("overall_benefit", "") in ("明显获益", "有限获益或稳定")
            else 0
        )
        pred_benefit = parsed.get("overall_benefit", "")
        yp_bin = (
            1
            if pred_benefit in ("明显获益", "有限获益或稳定")
            else 0
        )
        agg["y_true_bin"].append(yt_bin)
        agg["y_pred_bin"].append(yp_bin)
        agg["conf"].append(parsed.get("confidence", "中"))

        compliance = case_result["compliance"]
        agg["compliance_results"].append(compliance)
        for dim in ["A1", "A2", "A3", "A4"]:
            agg["aux_scores"][dim].append(case_result["auxiliary"].get(dim))

    # ---- Step 5: 计算全局指标 ----
    global_metrics = {}

    # M1: 临床获益二分类
    if agg["y_true_benefit"]:
        global_metrics["M1"] = compute_binary_benefit(
            agg["y_true_benefit"], agg["y_pred_benefit"]
        )
    else:
        global_metrics["M1"] = {"accuracy": None, "precision": None, "recall": None, "f1": None}

    if primary_only:
        # Selected evaluation profile: binary benefit, toxicity, and optional
        # LLM-as-Judge only.  Do not expose M2/M3/M4/M6/M7.
        global_metrics["M5"] = compute_toxicity_metrics(
            agg["y_true_tox"], agg["y_pred_tox"]
        )
        aux_avgs = {}
        for dim in ["A1", "A2", "A3", "A4"]:
            valid_scores = [s for s in agg["aux_scores"][dim] if s is not None]
            aux_avgs[dim] = round(np.mean(valid_scores), 2) if valid_scores else None
        primary_score = float(global_metrics["M1"].get("f1") or 0.0)
        weighted_judge_parts = [
            (0.30, aux_avgs.get("A1")),
            (0.30, aux_avgs.get("A2")),
            (0.25, aux_avgs.get("A3")),
            (0.15, aux_avgs.get("A4")),
        ]
        judge_score = (
            round(sum(weight * float(value) / 5.0 for weight, value in weighted_judge_parts), 4)
            if all(value is not None for _, value in weighted_judge_parts)
            else None
        )
        return {
            "per_instance": per_instance_results,
            "global": {
                "M1": global_metrics["M1"],
                "M5": global_metrics["M5"],
                "auxiliary_avg": aux_avgs,
            },
            "composite": {
                "primary_score": round(primary_score, 4),
                "auxiliary_score": judge_score,
                "composite_score": None,
                "scoring_metric": "M1.f1 + M5 report + A1-A4 report",
            },
            "meta": {
                "n_instances": len(per_instance_results),
                "n_total": len(instances),
                "n_parse_failures": len(completed_failures),
                "n_errors": len(instances) - len(per_instance_results) - len(completed_failures),
                "workers": workers,
                "parse_failure_rate": round(1 - len(per_instance_results) / len(instances), 4) if instances else 0,
                "metric_profile": "m1_m5_llm_judge",
            },
        }

    # M2: Weighted κ
    if agg["y_true_overall"]:
        global_metrics["M2_kappa"] = compute_weighted_kappa(
            agg["y_true_overall"], agg["y_pred_overall"]
        )
    else:
        global_metrics["M2_kappa"] = None

    # M3: RECIST
    global_metrics["M3_body"] = compute_recist_metrics(
        agg["y_true_body"], agg["y_pred_body"]
    )
    global_metrics["M3_cns"] = compute_recist_metrics(
        agg["y_true_cns"], agg["y_pred_cns"]
    )

    # M4: 方向准确率
    global_metrics["M4_csf"] = compute_direction_accuracy(
        agg["y_true_csf"], agg["y_pred_csf"], CSF_MAPPING
    )
    global_metrics["M4_symptom"] = compute_direction_accuracy(
        agg["y_true_symptom"], agg["y_pred_symptom"], SYMPTOM_MAPPING
    )

    # M5: 毒性
    global_metrics["M5"] = compute_toxicity_metrics(
        agg["y_true_tox"], agg["y_pred_tox"]
    )

    # M6: 校准
    if agg["y_true_bin"]:
        global_metrics["M6"] = compute_calibration(
            agg["y_true_bin"], agg["y_pred_bin"], agg["conf"]
        )
    else:
        global_metrics["M6"] = {"ece": None, "brier_score": None, "bin_details": {}}

    # M7: 时间合规
    global_metrics["M7_compliance"] = compute_global_compliance(
        agg["compliance_results"]
    )

    # 辅助指标平均分
    aux_avgs = {}
    for dim in ["A1", "A2", "A3", "A4"]:
        valid_scores = [s for s in agg["aux_scores"][dim] if s is not None]
        aux_avgs[dim] = round(np.mean(valid_scores), 2) if valid_scores else None

    # 综合评分
    composite = compute_composite_score(global_metrics, aux_avgs)

    n_success = len(per_instance_results)

    return {
        "per_instance": per_instance_results,
        "global": {**global_metrics, "auxiliary_avg": aux_avgs},
        "composite": composite,
        "meta": {
            "n_instances": n_success,
            "n_total": len(instances),
            "n_parse_failures": len(completed_failures),
            "n_errors": len(instances) - n_success - len(completed_failures),
            "workers": workers,
            "parse_failure_rate": round(
                1 - n_success / len(instances), 4
            )
            if instances
            else 0,
        },
    }
