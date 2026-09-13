"""
评估指标模块 — 主指标 M1-M7 + 综合评分 + 辅助评分代码
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
)
from collections import Counter

# ============================================================
# 映射表
# ============================================================

CSF_MAPPING = {
    "转阴": "好转",
    "下降": "好转",
    "稳定": "稳定",
    "上升": "恶化",
}

SYMPTOM_MAPPING = {
    "明显改善": "改善",
    "部分改善": "改善",
    "无变化": "稳定",
    "加重": "恶化",
}


# ============================================================
# M1: 临床获益二分类指标
# ============================================================

def compute_binary_benefit(y_true: list[str], y_pred: list[str]) -> dict:
    """计算临床获益二分类指标。

    Args:
        y_true: GT 标签列表，每个元素是 '明显获益' | '有限获益或稳定' | '无明显获益' | '进展或有害'
        y_pred: 预测标签列表，同上

    Returns:
        {accuracy, precision, recall, f1}
    """
    def to_binary(label: str) -> str:
        if label in ("明显获益", "有限获益或稳定"):
            return "获益"
        else:
            return "不获益"

    y_true_bin = [to_binary(l) for l in y_true]
    y_pred_bin = [to_binary(l) for l in y_pred]

    acc = accuracy_score(y_true_bin, y_pred_bin)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="binary", pos_label="获益"
    )
    # 处理除零
    if precision == 0 and recall == 0:
        precision, recall, f1 = 0.0, 0.0, 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# ============================================================
# M2: 总体净获益四分类 Weighted Cohen's κ
# ============================================================

def _normalize_benefit_label(label: str) -> str:
    """模糊匹配 → 标准标签。"""
    if not label:
        return "无明显获益"  # 空值默认
    label = label.strip()
    # 精确匹配
    levels = ["明显获益", "有限获益或稳定", "无明显获益", "进展或有害"]
    if label in levels:
        return label
    # 模糊匹配
    if "明显获益" in label or "显著获益" in label or label == "获益":
        return "明显获益"
    if "有限" in label or "稳定" in label or "部分获益" in label:
        return "有限获益或稳定"
    if "无" in label or "不获益" in label or "未获益" in label:
        return "无明显获益"
    if "进展" in label or "有害" in label or "恶化" in label or "PD" in label:
        return "进展或有害"
    # 兜底
    return "无明显获益"


def compute_weighted_kappa(y_true: list[str], y_pred: list[str]) -> float:
    """计算四分类加权 Cohen's κ（二次平方惩罚）。

    有序级别: 明显获益 > 有限获益或稳定 > 无明显获益 > 进展或有害
    """
    levels = ["明显获益", "有限获益或稳定", "无明显获益", "进展或有害"]
    y_true_norm = [_normalize_benefit_label(l) for l in y_true]
    y_pred_norm = [_normalize_benefit_label(l) for l in y_pred]
    y_true_idx = [levels.index(l) for l in y_true_norm]
    y_pred_idx = [levels.index(l) for l in y_pred_norm]

    # 处理全相同导致 NaN 的情况：若 GT 和预测完全一致 → κ = 1.0
    if len(set(y_true_idx)) == 1 and y_true_idx == y_pred_idx:
        kappa = 1.0
    else:
        kappa = cohen_kappa_score(y_true_idx, y_pred_idx, weights="quadratic")
    return round(kappa, 4)


# ============================================================
# M3: RECIST 病灶疗效准确率
# ============================================================

def compute_recist_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """计算 RECIST 相关所有指标。

    Args:
        y_true: GT RECIST 标签 (CR/PR/SD/PD/NA)
        y_pred: 预测 RECIST 标签

    Returns:
        {kappa, orr_accuracy, per_class_accuracy, count}
    """
    recist_levels = ["CR", "PR", "SD", "PD", "NA"]

    # 过滤 NA
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if t != "NA" and p != "NA"]
    if not valid_pairs:
        return {"kappa": None, "orr_accuracy": None, "per_class_accuracy": {}, "count": 0}

    y_true_f, y_pred_f = zip(*valid_pairs)

    # Weighted κ
    t_idx = [recist_levels.index(l) for l in y_true_f]
    p_idx = [recist_levels.index(l) for l in y_pred_f]

    # 处理全相同导致 NaN 的情况：若 GT 和预测完全一致 → κ = 1.0
    if len(set(t_idx)) == 1 and t_idx == p_idx:
        kappa = 1.0
    else:
        kappa = cohen_kappa_score(t_idx, p_idx, weights="quadratic")

    # ORR 二分类
    def to_orr(label):
        return "缓解" if label in ("CR", "PR") else "未缓解"

    y_true_orr = [to_orr(l) for l in y_true_f]
    y_pred_orr = [to_orr(l) for l in y_pred_f]
    orr_acc = accuracy_score(y_true_orr, y_pred_orr)

    # Per-class accuracy
    per_class = {}
    for level in ["CR", "PR", "SD", "PD"]:
        mask = [t == level for t in y_true_f]
        if sum(mask) > 0:
            correct = sum(1 for i, m in enumerate(mask) if m and y_pred_f[i] == level)
            per_class[level] = round(correct / sum(mask), 4)

    return {
        "kappa": round(kappa, 4),
        "orr_accuracy": round(orr_acc, 4),
        "per_class_accuracy": per_class,
        "count": len(y_true_f),
    }


# ============================================================
# M4: 症状/脑脊液趋势准确率
# ============================================================

def compute_direction_accuracy(
    y_true: list[str], y_pred: list[str], label_mapping: dict
) -> dict:
    """计算方向性三分类准确率。

    Args:
        y_true: GT 原始标签
        y_pred: 预测原始标签
        label_mapping: 将原始标签映射为方向类别的 dict

    Returns:
        {accuracy, count}
    """
    def map_label(label: str, mapping: dict) -> str:
        for key, val in mapping.items():
            if key in label:
                return val
        return label

    y_true_mapped = [map_label(l, label_mapping) for l in y_true]
    y_pred_mapped = [map_label(l, label_mapping) for l in y_pred]

    # 过滤 NA / 未评估
    valid = [
        (t, p)
        for t, p in zip(y_true_mapped, y_pred_mapped)
        if t not in ("NA", "未评估", "")
    ]
    if not valid:
        return {"accuracy": None, "count": 0}

    y_t, y_p = zip(*valid)
    acc = accuracy_score(y_t, y_p)
    return {"accuracy": round(acc, 4), "count": len(y_t)}


# ============================================================
# M5: 毒性严重度召回率 + 等级准确率
# ============================================================

def _extract_grade(tox_dict) -> int:
    """从毒性 dict 中提取数值 grade，兼容 'grade'/'max_grade' key 和字符串类型。"""
    if not isinstance(tox_dict, dict):
        return 0
    grade = tox_dict.get("grade") or tox_dict.get("max_grade") or 0
    if isinstance(grade, str):
        try:
            return int(grade)
        except ValueError:
            return 0
    return int(grade)


def compute_toxicity_metrics(y_true: list[dict], y_pred: list[dict]) -> dict:
    """计算毒性预测指标。

    Args:
        y_true: GT 毒性 dict 列表，每个 {'grade': int, 'event': str, 'has_toxicity_record': bool}
        y_pred: 预测毒性 dict 列表

    Returns:
        {severe_toxicity_recall, severe_toxicity_precision, grade_tolerance_accuracy, severe_gt_count}
    """
    severe_tp, severe_fn, severe_fp = 0, 0, 0
    tol_correct, tol_total = 0, 0

    for t, p in zip(y_true, y_pred):
        # 兼容 "grade" 和 "max_grade" 两种 key，并处理字符串类型
        t_grade = _extract_grade(t)
        p_grade = _extract_grade(p)
        has_record = t.get("has_toxicity_record", True) if isinstance(t, dict) else True

        # 严重毒性召回：仅计入有毒性记录的病例
        if has_record:
            if t_grade >= 3 and p_grade >= 3:
                severe_tp += 1
            elif t_grade >= 3 and p_grade < 3:
                severe_fn += 1
            elif t_grade < 3 and p_grade >= 3:
                severe_fp += 1

        # ±1 容错准确率
        if t_grade > 0 or p_grade > 0:
            if abs(t_grade - p_grade) <= 1:
                tol_correct += 1
            tol_total += 1

    recall = (
        severe_tp / (severe_tp + severe_fn) if (severe_tp + severe_fn) > 0 else None
    )
    precision = (
        severe_tp / (severe_tp + severe_fp) if (severe_tp + severe_fp) > 0 else None
    )

    return {
        "severe_toxicity_recall": round(recall, 4) if recall is not None else None,
        "severe_toxicity_precision": round(precision, 4) if precision is not None else None,
        "grade_tolerance_accuracy": round(tol_correct / tol_total, 4) if tol_total > 0 else None,
        "severe_gt_count": severe_tp + severe_fn,
    }


# ============================================================
# M6: 置信度校准 (ECE + Brier Score)
# ============================================================

def compute_calibration(
    y_true_binary: list[int],
    y_pred_binary: list[int],
    confidence: list[str],
) -> dict:
    """计算置信度校准指标。

    Args:
        y_true_binary: 真实二分类标签 (1=获益, 0=不获益)
        y_pred_binary: 预测二分类标签
        confidence: 置信度标签 ('高' | '中' | '低')

    Returns:
        {ece, brier_score, bin_details}
    """
    conf_map = {"高": 0.9, "中": 0.6, "低": 0.3}
    probs = np.array([conf_map.get(c, 0.6) for c in confidence])
    y_true_arr = np.array(y_true_binary, dtype=float)

    # ECE (按置信度桶)
    bins = [
        ("高", 0.75, 1.01),
        ("中", 0.45, 0.75),
        ("低", 0.0, 0.45),
    ]
    ece = 0.0
    bin_details = {}
    for label, lo, hi in bins:
        mask = (probs >= lo) & (probs < hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            bin_details[label] = {"n": 0, "acc": None, "avg_conf": None}
            continue
        bin_acc = float(y_true_arr[mask].mean())
        bin_conf = float(probs[mask].mean())
        ece += (n_bin / len(probs)) * abs(bin_acc - bin_conf)
        bin_details[label] = {
            "n": n_bin,
            "acc": round(bin_acc, 4),
            "avg_conf": round(bin_conf, 4),
        }

    # Brier Score
    brier = float(np.mean((probs - y_true_arr) ** 2))

    return {
        "ece": round(ece, 4),
        "brier_score": round(brier, 4),
        "bin_details": bin_details,
    }


# ============================================================
# 综合评分
# ============================================================

def compute_composite_score(primary_metrics: dict, auxiliary_scores: dict) -> dict:
    """计算综合评分。

    Args:
        primary_metrics: M1-M7 指标 dict
        auxiliary_scores: A1-A4 原始分数 dict (0-5)

    Returns:
        {primary_score, auxiliary_score, composite_score}
    """

    def avg_or_0(*values) -> float:
        valid = [v for v in values if v is not None]
        return sum(valid) / len(valid) if valid else 0.0

    def num_or_0(value: object) -> float:
        return float(value) if value is not None else 0.0

    # 主指标聚合 (各子指标归一化到 [0,1])
    primary = (
        0.30 * num_or_0(primary_metrics.get("M1", {}).get("f1"))
        + 0.20 * max(num_or_0(primary_metrics.get("M2_kappa")), 0.0)
        + 0.15
        * avg_or_0(
            (primary_metrics.get("M3_body", {}) or {}).get("kappa"),
            (primary_metrics.get("M3_cns", {}) or {}).get("kappa"),
        )
        + 0.10
        * avg_or_0(
            (primary_metrics.get("M4_csf", {}) or {}).get("accuracy"),
            (primary_metrics.get("M4_symptom", {}) or {}).get("accuracy"),
        )
        + 0.10 * num_or_0(primary_metrics.get("M5", {}).get("severe_toxicity_recall"))
        + 0.10 * max(1 - (num_or_0(primary_metrics.get("M6", {}).get("ece")) or 1.0), 0.0)
        + 0.05 * num_or_0(primary_metrics.get("M7_compliance"))
    )

    # 辅助指标聚合 (0-5 归一化到 0-1)
    auxiliary = (
        0.30 * num_or_0(auxiliary_scores.get("A1")) / 5.0
        + 0.30 * num_or_0(auxiliary_scores.get("A2")) / 5.0
        + 0.25 * num_or_0(auxiliary_scores.get("A3")) / 5.0
        + 0.15 * num_or_0(auxiliary_scores.get("A4")) / 5.0
    )

    composite = 0.70 * primary + 0.30 * auxiliary

    return {
        "primary_score": round(primary, 4),
        "auxiliary_score": round(auxiliary, 4),
        "composite_score": round(composite, 4),
    }
