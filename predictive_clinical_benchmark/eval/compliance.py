"""
证据时间合规检查模块 (M7)
"""

from datetime import datetime


def compute_temporal_compliance(
    cited_refs: list[dict], time_cutoff: str
) -> dict:
    """检查引用文献的时间合规性。

    Args:
        cited_refs: 模型引用的文献列表 [{'title': str, 'pub_date': 'YYYY-MM-DD'}, ...]
        time_cutoff: 时间切点 'YYYY-MM-DD'

    Returns:
        {compliant: bool, violation_count: int, violations: list}
    """
    if not cited_refs:
        return {"compliant": True, "violation_count": 0, "violations": []}

    try:
        cutoff = datetime.strptime(time_cutoff, "%Y-%m-%d")
    except ValueError:
        return {
            "compliant": False,
            "violation_count": 0,
            "violations": [],
            "error": f"无法解析时间切点: {time_cutoff}",
        }

    violations = []
    for ref in cited_refs:
        pub_date_str = ref.get("pub_date", "9999-12-31")
        try:
            pub_date = datetime.strptime(pub_date_str, "%Y-%m-%d")
        except ValueError:
            violations.append({**ref, "error": "unparseable_date"})
            continue
        if pub_date > cutoff:
            violations.append(ref)

    return {
        "compliant": len(violations) == 0,
        "violation_count": len(violations),
        "violations": violations,
    }


def compute_global_compliance(per_instance_results: list[dict]) -> float:
    """全局证据时间合规率。

    Args:
        per_instance_results: 每题合规检查结果列表

    Returns:
        0.0 ~ 1.0 的合规率
    """
    if not per_instance_results:
        return 0.0
    compliant = sum(1 for r in per_instance_results if r.get("compliant", False))
    return round(compliant / len(per_instance_results), 4)
