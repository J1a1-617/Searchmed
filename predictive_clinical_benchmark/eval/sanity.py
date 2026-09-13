"""
健全性检验模块
"""

import random

from .metrics import compute_binary_benefit, compute_weighted_kappa


def check_random_baseline(instances: list[dict], seed: int = 42) -> dict:
    """随机基线检验：将预测标签随机打乱，验证指标接近随机水平。

    Args:
        instances: 预测题目列表
        seed: 随机种子

    Returns:
        {random_M1_f1, random_M2_kappa, M1_expected_chance, M2_expected_chance}
    """
    random.seed(seed)

    y_true = [inst["ground_truth"]["overall_benefit"] for inst in instances]
    levels = ["明显获益", "有限获益或稳定", "无明显获益", "进展或有害"]

    # 随机生成预测
    y_pred = [random.choice(levels) for _ in y_true]

    # 计算 M1 和 M2
    m1 = compute_binary_benefit(y_true, y_pred)
    m2 = compute_weighted_kappa(y_true, y_pred)

    return {
        "random_M1_f1": m1["f1"],
        "random_M2_kappa": m2,
        "M1_expected_chance": 0.5,
        "M2_expected_chance": 0.0,
        "M1_above_chance": m1["f1"] > 0.55,
        "M2_above_chance": m2 > 0.1,
    }


def run_sanity_checks(benchmark_results: dict, instances: list[dict]) -> dict:
    """运行全部健全性检验。

    Args:
        benchmark_results: run_benchmark 返回的完整结果
        instances: 原始预测题目列表

    Returns:
        {random_baseline, parse_rate_ok, ...}
    """
    checks = {}

    # 1. Random baseline
    checks["random_baseline"] = check_random_baseline(instances)

    # 2. Parse rate
    meta = benchmark_results.get("meta", {})
    parse_rate = 1 - meta.get("parse_failure_rate", 0)
    checks["parse_rate_ok"] = parse_rate >= 0.8  # 至少 80% 可解析
    checks["parse_failure_rate"] = meta.get("parse_failure_rate", 0)

    # 3. Confidence sanity: if model always says '高' but gets many wrong → ECE high
    m6 = benchmark_results.get("global", {}).get("M6", {})
    ece = m6.get("ece")
    checks["calibration_ok"] = ece is not None and ece < 0.3  # ECE < 0.3 is reasonable
    checks["ece"] = ece

    # 4. M7 compliance
    m7 = benchmark_results.get("global", {}).get("M7_compliance", 1.0)
    checks["temporal_compliance"] = m7

    return checks
