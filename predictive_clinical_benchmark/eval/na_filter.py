"""
NA 动态过滤机制模块
"""


def get_na_skip_dimensions(instance: dict) -> set:
    """根据 GT 的 na_flags 返回应跳过的指标维度。

    Args:
        instance: 预测题目 dict

    Returns:
        应跳过的指标维度集合，如 {'M3_cns', 'M4_csf'}
    """
    skip = set()
    na_flags = instance.get("ground_truth", {}).get("na_flags", {})
    dim_mapping = {
        "csf_trajectory": "M4_csf",
        "cns_lm_recist": "M3_cns",
        "body_lesion_recist": "M3_body",
        "symptom_trajectory": "M4_symptom",
    }
    for field, metric_key in dim_mapping.items():
        if na_flags.get(field, False):
            skip.add(metric_key)
    return skip
