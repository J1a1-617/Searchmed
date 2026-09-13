"""
Bootstrap 置信区间模块
"""

from typing import Callable

import numpy as np


def bootstrap_ci(
    per_instance_results: list[dict],
    metric_fn: Callable[[list[dict]], float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap 法计算指标的 95% 置信区间。

    Args:
        per_instance_results: 每题评测结果列表 (来自 run_benchmark 的 per_instance)
        metric_fn: 从结果列表计算指标值的函数
        n_bootstrap: 重采样次数
        alpha: 显著性水平
        seed: 随机种子

    Returns:
        {mean, ci_lower, ci_upper, std, n_bootstrap}
    """
    np.random.seed(seed)
    n = len(per_instance_results)
    if n == 0:
        return {
            "mean": None,
            "ci_lower": None,
            "ci_upper": None,
            "std": None,
            "n_bootstrap": 0,
        }

    estimates = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        sample = [per_instance_results[i] for i in indices]
        try:
            estimates.append(metric_fn(sample))
        except Exception:
            continue

    if not estimates:
        return {
            "mean": None,
            "ci_lower": None,
            "ci_upper": None,
            "std": None,
            "n_bootstrap": 0,
        }

    estimates = np.array(estimates)
    lower = np.percentile(estimates, 100 * alpha / 2)
    upper = np.percentile(estimates, 100 * (1 - alpha / 2))

    return {
        "mean": round(float(np.mean(estimates)), 4),
        "ci_lower": round(float(lower), 4),
        "ci_upper": round(float(upper), 4),
        "std": round(float(np.std(estimates)), 4),
        "n_bootstrap": len(estimates),
    }


def metric_m1_f1_from_results(per_instance_results: list[dict]) -> float:
    """从 per_instance_results 计算 M1 F1（用于 bootstrap）。

    注意：需要 per_instance_results 中每个元素的 parsed_output 包含 _ground_truth 字段。
    """
    from .metrics import compute_binary_benefit

    y_true = []
    y_pred = []
    for r in per_instance_results:
        gt = r.get("parsed_output", {}).get("_ground_truth", {})
        parsed = r.get("parsed_output", {})
        y_true.append(gt.get("overall_benefit", ""))
        y_pred.append(parsed.get("overall_benefit", ""))
    metrics = compute_binary_benefit(y_true, y_pred)
    return metrics["f1"]


def metric_m2_kappa_from_results(per_instance_results: list[dict]) -> float:
    """从 per_instance_results 计算 M2 κ（用于 bootstrap）。"""
    from .metrics import compute_weighted_kappa

    y_true = []
    y_pred = []
    for r in per_instance_results:
        gt = r.get("parsed_output", {}).get("_ground_truth", {})
        parsed = r.get("parsed_output", {})
        y_true.append(gt.get("overall_benefit", ""))
        y_pred.append(parsed.get("overall_benefit", ""))
    return compute_weighted_kappa(y_true, y_pred)
