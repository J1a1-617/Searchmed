"""
输出解析器 — 从模型原始输出中提取结构化 JSON
支持三级回退: 直接 JSON → Markdown code block → 正则提取
"""

import json
import re
from typing import Optional


def parse_model_output(raw_output: str) -> Optional[dict]:
    """从模型原始输出中提取结构化 JSON。

    三级回退策略：
    1. 直接 JSON 解析
    2. 从 ```json ... ``` Markdown code block 中提取
    3. 正则逐字段提取（最后手段）
    """
    if not raw_output or not isinstance(raw_output, str):
        return None

    # Level 1: 直接解析
    try:
        return json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        pass

    # Level 2: 从 ```json ... ``` 中提取
    json_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?```",
        raw_output,
        re.DOTALL,
    )
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except (json.JSONDecodeError, TypeError):
            pass

    # Level 3: 正则逐字段提取
    extracted = {}
    patterns = {
        "overall_benefit": r"总体净获益[：:]\s*(明显获益|有限获益或稳定|无明显获益|进展或有害)",
        "body_lesion_recist": r"体部病灶[：:]\s*(CR|PR|SD|PD|NA)",
        "cns_lm_recist": r"颅内[／/]脑膜病灶[：:]\s*(CR|PR|SD|PD|NA)",
        "csf_trajectory": r"脑脊液[^：:]*[：:]\s*(转阴|下降|稳定|上升|未评估)",
        "symptom_trajectory": r"症状[^：:]*[：:]\s*(明显改善|部分改善|无变化|加重)",
        "confidence": r"置信度[：:]\s*(高|中|低)",
    }
    for field, pattern in patterns.items():
        match = re.search(pattern, raw_output)
        if match:
            extracted[field] = match.group(1).strip()

    # 尝试提取毒性
    tox_grade_match = re.search(
        r"(?:max_grade|最大毒性等级|毒性等级)[：:\s]*(\d)", raw_output
    )
    if tox_grade_match:
        extracted["toxicity"] = {
            "max_grade": int(tox_grade_match.group(1)),
            "event": "",
            "requires_dose_modification": False,
        }

    # 尝试提取 rationales
    rationale_matches = re.findall(
        r"(?:依据|推理)\s*(\d+)[：:]\s*(.+?)(?=(?:依据|推理)\s*\d+[：:]|\Z)",
        raw_output,
        re.DOTALL,
    )
    if rationale_matches:
        extracted["rationale"] = [m[1].strip() for m in rationale_matches]

    return extracted if extracted else None


def validate_output(parsed: dict) -> tuple:
    """验证解析后的输出是否包含所有必要字段。

    Returns:
        (is_valid: bool, missing_fields: list[str])
    """
    if parsed is None:
        return False, ["all"]

    required_fields = [
        "overall_benefit",
        "body_lesion_recist",
        "cns_lm_recist",
        "csf_trajectory",
        "symptom_trajectory",
        "toxicity",
        "confidence",
    ]
    missing = [f for f in required_fields if f not in parsed or parsed[f] is None]
    return len(missing) == 0, missing
