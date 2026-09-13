#!/usr/bin/env python3
"""
Benchmark 自身量化验证脚本

用法:
    # 单模型验证（含随机基线模拟）
    python validate_benchmark.py --results results_gpt4.json

    # 多模型对比 + 稳定性（需要同一模型 3 次独立运行的结果）
    python validate_benchmark.py --results results_run1.json results_run2.json results_run3.json

    # 跨模型对比
    python validate_benchmark.py --results results_gpt4.json results_gpt35.json

依赖:
    pip install numpy
"""

import json
import sys
import argparse
import random
import numpy as np

# 将当前目录加入 path
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))

from eval.parser import parse_model_output
from eval.metrics import compute_binary_benefit, compute_weighted_kappa


# ============================================================
# 1. 区分度
# ============================================================

def compute_discrimination(model_scores: dict[str, float]) -> dict:
    """计算 benchmark 区分度。"""
    best_name = max(model_scores, key=model_scores.get)
    best = model_scores[best_name]
    random_score = model_scores.get("random", 0.35)
    discrimination = best - random_score

    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    gaps = []
    for i in range(len(sorted_models) - 1):
        gaps.append({
            "gap": round(sorted_models[i][1] - sorted_models[i + 1][1], 4),
            "pair": f"{sorted_models[i][0]} → {sorted_models[i + 1][0]}",
        })

    if discrimination > 0.3:
        verdict = "✅ 区分力强"
    elif discrimination > 0.1:
        verdict = "⚠️ 凑合能用"
    else:
        verdict = "❌ 区分力不足，建议调整题目难度"

    return {
        "scores": model_scores,
        "discrimination": round(discrimination, 4),
        "tier_gaps": gaps,
        "verdict": verdict,
    }


# ============================================================
# 2. 稳定性
# ============================================================

def compute_stability(runs: list[float]) -> dict:
    """计算 benchmark 稳定性（同一模型 3 次运行的变异系数）。"""
    mean = float(np.mean(runs))
    std = float(np.std(runs, ddof=1)) if len(runs) > 1 else 0.0
    cv = std / mean if mean > 0 else float("inf")

    if cv < 0.05:
        verdict = "✅ 很稳定"
    elif cv < 0.15:
        verdict = "⚠️ 可接受"
    else:
        verdict = "❌ 波动过大"

    return {
        "runs": runs,
        "cv": round(cv, 4),
        "mean": round(mean, 4),
        "std": round(std, 4),
        "verdict": verdict,
    }


# ============================================================
# 3. 解析鲁棒性
# ============================================================

PARSER_TEST_CASES = {
    "标准JSON": (
        '{"overall_benefit": "明显获益", "body_lesion_recist": "SD", '
        '"cns_lm_recist": "PR", "csf_trajectory": "下降", '
        '"symptom_trajectory": "部分改善", "confidence": "高"}'
    ),
    "Markdown包裹": (
        '```json\n{"overall_benefit": "明显获益", '
        '"body_lesion_recist": "SD", "cns_lm_recist": "PR", '
        '"csf_trajectory": "下降", "symptom_trajectory": "部分改善", '
        '"confidence": "高"}\n```'
    ),
    "纯文本格式": (
        "总体净获益：明显获益\n体部病灶：SD\n颅内/脑膜病灶：PR\n"
        "脑脊液指标：下降\n症状变化：部分改善\n置信度：高"
    ),
    "缺字段": '{"overall_benefit": "明显获益"}',
    "脏前缀": (
        '根据分析，我的预测如下：\n'
        '{"overall_benefit": "明显获益", "body_lesion_recist": "SD"}'
    ),
}


def _get_parse_level(output: str) -> int:
    """判断解析回退到哪一级 (1=直接JSON, 2=code block, 3=正则)。"""
    import re

    try:
        json.loads(output)
        return 1
    except Exception:
        if re.search(r"```json", output):
            return 2
        return 3


def compute_parser_robustness() -> dict:
    """计算解析器鲁棒性。"""
    results = {}
    success_count = 0

    for name, output in PARSER_TEST_CASES.items():
        parsed = parse_model_output(output)
        ok = parsed is not None and "overall_benefit" in parsed
        results[name] = {"passed": ok, "level": _get_parse_level(output)}
        if ok:
            success_count += 1

    robustness = success_count / len(PARSER_TEST_CASES)
    if robustness >= 0.9:
        verdict = "✅ 鲁棒"
    elif robustness >= 0.7:
        verdict = "⚠️ 需增强解析器"
    else:
        verdict = "❌ 解析器不可靠"

    return {
        "robustness": round(robustness, 4),
        "details": results,
        "verdict": verdict,
    }


# ============================================================
# 4. 天花板 / 地板效应
# ============================================================

def compute_ceiling_floor(
    model_scores: dict[str, float],
    random_m1_f1: float,
    random_m2_kappa: float,
) -> dict:
    """计算天花板/地板效应。"""
    best = max(model_scores.values())
    ceiling_gap = 1.0 - best
    floor_bias_m1 = abs(random_m1_f1 - 0.50)
    floor_bias_m2 = abs(random_m2_kappa - 0.0)

    checks = {
        "ceiling_ok": ceiling_gap > 0.10,
        "ceiling_gap": round(ceiling_gap, 4),
        "best_score": round(best, 4),
        "floor_m1_near_chance": floor_bias_m1 < 0.10,
        "floor_m2_near_chance": floor_bias_m2 < 0.10,
        "random_m1_f1": round(random_m1_f1, 4),
        "random_m2_kappa": round(random_m2_kappa, 4),
    }

    all_ok = all(
        [
            checks["ceiling_ok"],
            checks["floor_m1_near_chance"],
            checks["floor_m2_near_chance"],
        ]
    )
    checks["verdict"] = "✅ 通过" if all_ok else "⚠️ 需要检查"

    return checks


# ============================================================
# 5. 模拟随机基线（无真实模型时使用）
# ============================================================

def simulate_random_baseline(data_file: str, n_trials: int = 100) -> dict:
    """不调用模型，直接模拟随机预测来计算随机基线。"""
    with open(data_file, "r", encoding="utf-8") as f:
        instances = json.load(f)

    levels = ["明显获益", "有限获益或稳定", "无明显获益", "进展或有害"]
    m1_f1s = []
    m2_kappas = []

    for _ in range(n_trials):
        y_true = []
        y_pred = []
        for inst in instances:
            gt = inst.get("ground_truth", {})
            y_true.append(gt.get("overall_benefit", ""))
            y_pred.append(random.choice(levels))

        m1 = compute_binary_benefit(y_true, y_pred)
        m2 = compute_weighted_kappa(y_true, y_pred)
        m1_f1s.append(m1["f1"])
        m2_kappas.append(m2)

    return {
        "n_trials": n_trials,
        "m1_f1_mean": round(float(np.mean(m1_f1s)), 4),
        "m1_f1_std": round(float(np.std(m1_f1s)), 4),
        "m2_kappa_mean": round(float(np.mean(m2_kappas)), 4),
        "m2_kappa_std": round(float(np.std(m2_kappas)), 4),
    }


# ============================================================
# 主入口
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark 自身量化验证工具"
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="一个或多个 run_eval.py 产出的 results JSON 文件",
    )
    parser.add_argument(
        "--random-m1",
        type=float,
        default=None,
        help="随机基线 M1 F1（不指定则自动模拟）",
    )
    parser.add_argument(
        "--random-m2",
        type=float,
        default=None,
        help="随机基线 M2 κ（不指定则自动模拟）",
    )
    parser.add_argument(
        "--data",
        default="benchmark_instances.json",
        help="预测题目数据文件（用于模拟随机基线）",
    )
    args = parser.parse_args()

    # ---- 加载各模型的综合得分 ----
    model_scores = {}
    composite_runs = []

    for path in args.results:
        with open(path, "r", encoding="utf-8") as f:
            r = json.load(f)

        composite = r.get("composite", {}).get("composite_score", 0)
        model_name = path.replace("results_", "").replace(".json", "")
        model_scores[model_name] = composite
        composite_runs.append(composite)

    # ---- 模拟或使用用户指定的随机基线 ----
    if args.random_m1 is not None and args.random_m2 is not None:
        random_m1 = args.random_m1
        random_m2 = args.random_m2
    else:
        print("⏳ 模拟随机基线 (100 次)...")
        rand = simulate_random_baseline(args.data)
        random_m1 = rand["m1_f1_mean"]
        random_m2 = rand["m2_kappa_mean"]
        print(f"   随机 M1 F1: {random_m1} ± {rand['m1_f1_std']}")
        print(f"   随机 M2 κ:  {random_m2} ± {rand['m2_kappa_std']}")
        model_scores["random"] = 0.35  # 估算的综合得分

    # ---- 运行 4 项检查 ----
    print()
    print("=" * 60)
    print("🔬 Benchmark 自身量化验证报告")
    print("=" * 60)

    # 1. 区分度
    print()
    print("1️⃣  区分度")
    disc = compute_discrimination(model_scores)
    for name, score in disc["scores"].items():
        print(f"  {name:15s} {score:.4f}")
    print(f"  区分度 Δ:  {disc['discrimination']:.4f}  {disc['verdict']}")
    for g in disc["tier_gaps"]:
        print(f"  {g['pair']}: {g['gap']:+.4f}")

    # 2. 稳定性
    print()
    print("2️⃣  稳定性")
    if len(composite_runs) >= 3:
        stab = compute_stability(composite_runs)
        print(f"  CV:        {stab['cv']:.4f}  {stab['verdict']}")
        print(f"  Mean ± SD: {stab['mean']:.4f} ± {stab['std']:.4f}")
    elif len(composite_runs) == 2:
        diff = abs(composite_runs[0] - composite_runs[1])
        print(f"  2 runs diff: {diff:.4f}  {'✅ 稳定' if diff < 0.05 else '⚠️ 有波动'}")
    else:
        print("  ⚠️ 需要 ≥2 个 results 文件才能计算稳定性")

    # 3. 解析鲁棒性
    print()
    print("3️⃣  解析鲁棒性")
    rob = compute_parser_robustness()
    passed = sum(1 for v in rob["details"].values() if v["passed"])
    total = len(rob["details"])
    print(f"  成功率:    {passed}/{total} ({rob['robustness']:.0%})  {rob['verdict']}")
    for name, detail in rob["details"].items():
        status = "✅" if detail["passed"] else "❌"
        print(f"  {status} {name} (Level {detail['level']})")

    # 4. 天花板/地板
    print()
    print("4️⃣  天花板/地板")
    cf = compute_ceiling_floor(model_scores, random_m1, random_m2)
    print(f"  天花板:    {cf['best_score']:.4f} (gap: {cf['ceiling_gap']:.4f})  "
          f"{'✅' if cf['ceiling_ok'] else '⚠️ 空间不足'}")
    print(f"  随机 M1 F1: {cf['random_m1_f1']:.4f} (bias: {abs(cf['random_m1_f1']-0.50):.4f})  "
          f"{'✅' if cf['floor_m1_near_chance'] else '❌ 偏离理论值'}")
    print(f"  随机 M2 κ:  {cf['random_m2_kappa']:.4f} (bias: {abs(cf['random_m2_kappa']):.4f})  "
          f"{'✅' if cf['floor_m2_near_chance'] else '❌ 偏离理论值'}")

    # ---- 总结 ----
    verdicts = [
        disc["verdict"].startswith("✅"),
        stab.get("verdict", "").startswith("✅") if len(composite_runs) >= 3 else True,
        rob["verdict"].startswith("✅"),
        cf["verdict"].startswith("✅"),
    ]
    passed_count = sum(verdicts)
    total_count = len(verdicts)

    print()
    print("=" * 60)
    print(f"📋 总结: {passed_count}/{total_count} 通过"
          + ("" if passed_count == total_count else ", 需关注未通过项"))
    print("=" * 60)


if __name__ == "__main__":
    main()
