#!/usr/bin/env python3
"""Binary benefit A/B smoke test on ground-truth-negative benchmark cases.

A: patient-only prompt (no retrieved examples)
B: same prompt plus existing retrieved-case few-shot examples
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

from openai import OpenAI


POSITIVE = {"明显获益", "有限获益或稳定", "获益"}


def binary_label(value: object) -> str:
    return "获益" if value in POSITIVE else "不获益"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def patient_prompt(instance: dict) -> str:
    inp = instance["input"]
    bg = inp.get("disease_background") or {}
    molecular = bg.get("molecular_profile") or {}
    status = inp.get("current_status") or {}
    treatment = inp.get("planned_treatment") or {}
    prior = inp.get("prior_treatment_timeline") or []
    drugs = [d.get("name") for d in treatment.get("drugs") or [] if isinstance(d, dict)]
    return f"""你是一位肿瘤科临床预测专家。请预测这名患者在实际采用下列方案后8–12周是否获得总体净获益。

这是严格二分类任务，只能选择“获益”或“不获益”：
- 获益：8–12周内肿瘤缩小或疾病稳定，且毒性没有抵消疗效。
- 不获益：8–12周内疾病进展、治疗失败或毒性使总体结果有害。

重要：方案符合指南、药物通常有效或存在敏感突变，都不能直接推出这名患者获益。必须同时检查既往同靶点失败、明确耐药机制、快速进展、高肿瘤负荷、差体能状态、CNS/脑膜控制不足、方案与分子机制不匹配等负向证据。不要依据数据集类别比例猜测。

时间切点：{instance.get('time_cutoff')}
诊断：{bg.get('diagnosis')}
转移部位：{bg.get('metastatic_sites')}
分子谱：{json.dumps(molecular, ensure_ascii=False)}
既往治疗：{json.dumps(prior, ensure_ascii=False)}
当前状态：{json.dumps(status, ensure_ascii=False)}
实际方案：{drugs}
联合策略：{treatment.get('combination_strategy')}

先简短列出支持获益和支持不获益的证据，再输出严格JSON：
```json
{{
  "benefit_binary": "获益/不获益",
  "confidence": "高/中/低",
  "positive_factors": ["..."],
  "negative_factors": ["..."],
  "rationale": "..."
}}
```"""


def reference_messages(instance: dict) -> list[dict]:
    messages: list[dict] = []
    for ref in (instance.get("_rag_references") or [])[:2]:
        ref_inst = ref.get("reference_instance")
        if isinstance(ref_inst, dict):
            question = patient_prompt(ref_inst)
        else:
            question = "参考患者：\n" + str(ref.get("reference_case") or "")
        label = binary_label(ref.get("reference_gt"))
        answer = json.dumps(
            {
                "benefit_binary": label,
                "confidence": "中",
                "positive_factors": [],
                "negative_factors": [],
                "rationale": "这是该参考患者的真实随访二分类标签，仅用于示范。",
            },
            ensure_ascii=False,
        )
        messages.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )
    return messages


def parse_json(text: str) -> dict:
    decoder = json.JSONDecoder()
    for pos, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[pos:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("benefit_binary") in {"获益", "不获益"}:
            return value
    return {"benefit_binary": "解析失败", "raw": text}


def call(client: OpenAI, model: str, messages: list[dict]) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1500,
    )
    return parse_json(response.choices[0].message.content or "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--rag-data", type=Path, required=True)
    parser.add_argument("--dotenv", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "gpt-5.4-mini"))
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    load_dotenv(args.dotenv)
    api_key = os.environ.get("OPENAI_API_KEY") or (os.environ.get("OPENAI_API_KEYS", "").split(",")[0])
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("DEFAULT_BASE_URL")
    if not api_key:
        raise SystemExit("missing OPENAI_API_KEY/OPENAI_API_KEYS")
    client = OpenAI(api_key=api_key, base_url=base_url)

    base = {x["instance_id"]: x for x in json.loads(args.data.read_text(encoding="utf-8"))}
    rag = {x["instance_id"]: x for x in json.loads(args.rag_data.read_text(encoding="utf-8"))}
    candidates = [
        rag[iid]
        for iid, item in base.items()
        if binary_label((item.get("ground_truth") or {}).get("overall_benefit")) == "不获益"
        and iid in rag
        and rag[iid].get("_rag_references")
    ][: args.n]

    rows = []
    system = {
        "role": "system",
        "content": "独立判断当前患者。参考病例的标签只属于参考患者，不得复制；严格输出要求的JSON。",
    }
    for index, item in enumerate(candidates, 1):
        current = {"role": "user", "content": patient_prompt(item)}
        a = call(client, args.model, [system, current])
        b = call(client, args.model, [system, *reference_messages(item), current])
        refs = [binary_label(r.get("reference_gt")) for r in (item.get("_rag_references") or [])[:2]]
        row = {
            "instance_id": item["instance_id"],
            "ground_truth_original": (item.get("ground_truth") or {}).get("overall_benefit"),
            "ground_truth_binary": "不获益",
            "fewshot_reference_labels": refs,
            "A_no_fewshot": a,
            "B_with_fewshot": b,
        }
        rows.append(row)
        print(index, item["instance_id"], "A=", a.get("benefit_binary"), "B=", b.get("benefit_binary"), "refs=", refs, flush=True)

    def summary(arm: str) -> dict:
        labels = [row[arm].get("benefit_binary") for row in rows]
        return {
            "distribution": dict(Counter(labels)),
            "negative_accuracy": sum(x == "不获益" for x in labels) / len(labels) if labels else None,
        }

    output = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "selection": "first GT-negative cases with existing few-shot references",
        "n": len(rows),
        "A_no_fewshot": summary("A_no_fewshot"),
        "B_with_fewshot": summary("B_with_fewshot"),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("model", "n", "A_no_fewshot", "B_with_fewshot")}, ensure_ascii=False, indent=2))
    print("saved", args.output)


if __name__ == "__main__":
    main()
