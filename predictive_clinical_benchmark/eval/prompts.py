"""
Prompt 模板模块 — 模型推理 prompt + LLM-Judge 评分 prompt
"""


def construct_inference_prompt(instance: dict) -> str:
    """根据数据实例构造模型推理 prompt。

    Args:
        instance: 预测题目 dict, 包含 input, time_cutoff 等字段

    Returns:
        完整的 prompt 字符串
    """
    inp = instance["input"]
    cutoff = instance["time_cutoff"]

    # 格式化既往治疗史
    prior_tx_lines = []
    for tx in inp.get("prior_treatment_timeline", []):
        reason = tx.get("reason_for_discontinuation", "N/A")
        prior_tx_lines.append(
            f"- {tx['start_date']} 至 {tx['end_date']}：{tx['regimen']}，"
            f"最佳疗效 {tx['best_response']}（停药原因：{reason}）"
        )

    # 格式化分子突变谱
    mol = inp["disease_background"].get("molecular_profile", {})
    mol_lines = []
    if "primary_mutation" in mol:
        mol_lines.append(f"  主要驱动突变：{mol['primary_mutation']}")
    if "resistance_mutations" in mol:
        mol_lines.append(f"  耐药突变：{', '.join(mol['resistance_mutations'])}")
    if "bypass_alterations" in mol:
        mol_lines.append(f"  旁路激活/扩增：{', '.join(mol['bypass_alterations'])}")
    if "co_mutations" in mol:
        mol_lines.append(f"  共存突变：{', '.join(mol['co_mutations'])}")

    # 格式化治疗方案
    tx = inp.get("planned_treatment", {})
    drug_lines = []
    for d in tx.get("drugs", []):
        drug_lines.append(f"  - {d['name']} {d['dose']} {d['route']}")

    cs = inp.get("current_status", {})
    bg = inp["disease_background"]

    # ---- RAG 参考病例（如果存在）----
    rag_prefix = ""
    refs = instance.get("_rag_references", [])
    if refs:
        rag_prefix = "## 参考相似病例（以下为既往真实病例及其结局，供参考）\n\n"
        for ref in refs:
            rag_prefix += f"### 参考病例 {ref['rank']}（相似度: {ref['similarity']}）\n"
            rag_prefix += f"{ref['reference_case']}\n\n"

    prompt = f"""{rag_prefix}你是一位肿瘤科临床决策支持 AI。请根据以下患者信息，预测医生实际采用的治疗方案在 8-12 周内的疗效结局。

时间切点: {cutoff}（仅使用此日期前的信息，无法判断的维度填 NA）

患者疾病背景
诊断: {bg.get('diagnosis', 'N/A')}
转移部位: {', '.join(bg.get('metastatic_sites', []))}
分子突变谱:
{chr(10).join(mol_lines) if mol_lines else '  N/A'}

既往治疗史
{chr(10).join(prior_tx_lines) if prior_tx_lines else '  N/A'}

当前临床状态
症状: {', '.join(cs.get('symptoms', []))}
影像学: {cs.get('imaging', 'N/A')}
脑脊液: {cs.get('csf', 'N/A')}
体能状态 (ECOG): {cs.get('performance_status', 'N/A')}

实际采用的治疗方案
{chr(10).join(drug_lines) if drug_lines else '  N/A'}
联合策略: {tx.get('combination_strategy', 'N/A')}

请按以下 JSON 格式输出预测结果（仅输出 JSON，不要其他文字）:
```json
{{
  "overall_benefit": "明显获益/有限获益或稳定/无明显获益/进展或有害",
  "body_lesion_recist": "CR/PR/SD/PD/NA",
  "cns_lm_recist": "CR/PR/SD/PD/NA",
  "csf_trajectory": "转阴/下降/稳定/上升/未评估",
  "symptom_trajectory": "明显改善/部分改善/无变化/加重",
  "toxicity": {{
    "max_grade": "0-5整数",
    "event": "具体毒性(无则填无)",
    "requires_dose_modification": false
  }},
  "confidence": "高/中/低",
  "rationale": ["依据1:突变靶点分析...", "依据2:耐药机制...", "依据3:PK/CNS穿透...", "依据4:既往治疗影响..."],
  "cited_evidence": [{{"type": "guideline/mechanism/trial", "content": "引用内容", "pub_date": "YYYY-MM-DD"}}]
}}
```"""

    return prompt


def construct_agent_query(instance: dict) -> str:
    """Return only clinical facts for retrieval; never expose output-schema tokens."""
    prompt = construct_inference_prompt(instance)
    return prompt.split("\n请按以下 JSON 格式输出预测结果", 1)[0].rstrip()


# ============================================================
# LLM-Judge Prompt 模板
# ============================================================


def _format_case_summary(inp: dict) -> str:
    """格式化病例摘要用于 Judge prompt。"""
    bg = inp.get("disease_background", {})
    cs = inp.get("current_status", {})
    lines = [
        f"诊断：{bg.get('diagnosis', 'N/A')}",
        f"分子突变：{bg.get('molecular_profile', 'N/A')}",
        f"既往治疗：{inp.get('prior_treatment_timeline', 'N/A')}",
        f"当前状态：症状={cs.get('symptoms', [])}, 影像={cs.get('imaging', 'N/A')}, 脑脊液={cs.get('csf', 'N/A')}",
    ]
    return "\n".join(lines)


def _format_treatment(inp: dict) -> str:
    """格式化治疗方案用于 Judge prompt。"""
    tx = inp.get("planned_treatment", {})
    return str(tx)


def _format_gt(gt: dict) -> str:
    """格式化标准答案用于 Judge prompt。"""
    return (
        f"总体净获益：{gt.get('overall_benefit', 'N/A')}\n"
        f"体部病灶：{gt.get('body_lesion_recist', 'N/A')}\n"
        f"颅内/脑膜：{gt.get('cns_lm_recist', 'N/A')}\n"
        f"脑脊液趋势：{gt.get('csf_trajectory', 'N/A')}\n"
        f"症状变化：{gt.get('symptom_trajectory', 'N/A')}\n"
        f"毒性：Grade {gt.get('toxicity', {}).get('max_grade', 'N/A')}"
    )


def _format_evidence(items: list[str]) -> str:
    """格式化关键证据列表。"""
    if not items:
        return "（无）"
    return "\n".join(f"- {item}" for item in items)


def format_judge_prompt_A1(instance: dict, model_output: str) -> str:
    """A1: 推理依据完整性评分 prompt。"""
    inp = instance["input"]
    gt = instance.get("ground_truth", {})
    evidence_list = _format_evidence(gt.get("key_evidence_items", []))

    return f"""你是一位肿瘤学临床评审专家。请根据标准答案和评分标准，对模型的预测推理进行评分。

## 患者病例摘要
{_format_case_summary(inp)}

## 治疗方案
{_format_treatment(inp)}

## 标准答案（真实随访结果）
{_format_gt(gt)}

## 标准答案中的关键推理依据（预期模型应提及）
{evidence_list}

## 模型预测输出
{model_output}

## 评分标准（推理依据完整性，0-5分）

- **5分**：模型提及了标准答案中 ≥90% 的关键依据项，且额外给出了合理的补充推理
- **4分**：模型提及了 ≥75% 的关键依据项，遗漏 1-2 项次要依据
- **3分**：模型提及了 ≥50% 的关键依据项，遗漏了一些重要依据但整体方向正确
- **2分**：模型仅提及了 <50% 的关键依据项，遗漏了多项核心依据
- **1分**：模型几乎没有触及关键依据，仅给出泛泛的通用推理
- **0分**：推理与病例无关，或出现严重幻觉

## 评分输出格式
请严格以JSON格式输出：
```json
{{
  "score": "整数 0-5",
  "covered_items": ["命中依据1", "命中依据2"],
  "missed_items": ["遗漏依据1"],
  "hallucination_items": ["幻觉/错误依据1"],
  "brief_reason": "一句话评分理由"
}}
```"""


def format_judge_prompt_A2(instance: dict, model_output: str) -> str:
    """A2: 逻辑链条自洽性评分 prompt。"""
    inp = instance["input"]

    return f"""你是一位临床推理专家。请评估模型预测的逻辑推理链是否完整自洽。

## 背景
{_format_case_summary(inp)}

## 治疗方案
{_format_treatment(inp)}

## 模型预测输出（含推理）
{model_output}

## 理想因果链
1. 突变靶点识别 → 2. 耐药机制分析 → 3. 药物作用机制（含药代动力学/CNS穿透性）→ 4. 预期疗效推断

## 评分标准（逻辑链条自洽性，0-5分）

- **5分**：四个环节全部覆盖且衔接紧密，无医学常识错误，逻辑链条无断裂
- **4分**：四个环节基本完整，但某个环节的推理深度不足
- **3分**：因果链大方向正确，但中间存在跳跃或模糊地带（如跳过耐药机制直接推断疗效）
- **2分**：逻辑链有明显断裂，或存在轻微医学错误（如搞错药物靶点）
- **1分**：逻辑牵强，因果关系混乱，存在严重医学常识错误
- **0分**：完全不合逻辑，或答非所问

## 评分输出格式
```json
{{
  "score": "整数 0-5",
  "chain_analysis": {{
    "step1_mutation_to_resistance": "评语",
    "step2_resistance_to_mechanism": "评语",
    "step3_mechanism_to_outcome": "评语"
  }},
  "errors_found": ["具体错误1"],
  "brief_reason": "一句话评分理由"
}}
```"""


def format_judge_prompt_A3(instance: dict, model_output: str) -> str:
    """A3: 证据引用准确性评分 prompt。"""
    inp = instance["input"]

    return f"""你是一位循证医学评审专家。请评估模型引用的证据是否准确可靠。

## 背景
{_format_case_summary(inp)}

## 模型预测输出（含证据引用）
{model_output}

## 可接受的知识来源
- 药物说明书（截至时间切点的版本）
- NCCN / CSCO 指南（截至时间切点的版本）
- 已发表的临床研究结论（PubMed-indexed, 发表时间 < 时间切点）

## 评分标准（证据引用准确性，0-5分）

- **5分**：所有引用证据均准确，引用内容与药物机制/临床数据完全一致
- **4分**：主要证据准确，个别次要引用的细节不完全精确
- **3分**：大部分证据准确，但有 1-2 处不精确或过度推广的引用
- **2分**：存在明显的事实性错误引用（如搞错药物靶点、错误的 CNS 穿透性数据）
- **1分**：多处严重事实错误，或引用不存在的"研究"
- **0分**：证据引用完全虚构

## 评分输出格式
```json
{{
  "score": "整数 0-5",
  "accurate_citations": ["准确引用1"],
  "inaccurate_citations": [{{"citation": "不准确引用", "correction": "正确信息"}}],
  "brief_reason": "一句话评分理由"
}}
```"""


def format_judge_prompt_A4(instance: dict, model_output: str) -> str:
    """A4: 临床可操作性评分 prompt。"""
    inp = instance["input"]
    gt = instance.get("ground_truth", {})

    return f"""你是一位肿瘤科主任医师。请评估模型的预测是否提供了实际临床可操作的信息。

## 背景
{_format_case_summary(inp)}

## 治疗方案
{_format_treatment(inp)}

## 标准答案（真实随访结果）
{_format_gt(gt)}

## 模型预测输出
{model_output}

## 评分标准（临床可操作性，0-5分）

- **5分**：预测提供了具体、可操作的临床建议（如剂量调整依据、监测频率、替代方案触发条件），且与标准答案一致
- **4分**：预测提供了较具体的建议，但缺少某一方面的细节（如未给出监测频率）
- **3分**：预测方向正确，但建议偏通用化，缺乏个体化细节
- **2分**：建议过于笼统（如"需要密切监测"），无实质操作指导
- **1分**：建议不切实际或与标准答案矛盾
- **0分**：无任何可操作信息，或给出危险的建议

## 评分输出格式
```json
{{
  "score": "整数 0-5",
  "actionable_items": ["可操作建议1"],
  "missing_items": ["缺失的建议1"],
  "brief_reason": "一句话评分理由"
}}
```"""
