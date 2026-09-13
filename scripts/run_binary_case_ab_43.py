#!/usr/bin/env python3
"""Paired binary pure-LLM vs case-only RAG evaluation on batches 007--011."""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


POSITIVE = {"明显获益", "有限获益或稳定", "获益"}
SYSTEM = "独立判断当前患者。参考病例的标签只属于参考患者，不得复制；严格输出要求的JSON。"


def label(value):
    return "获益" if value in POSITIVE else "不获益"


def prompt(inst):
    inp = inst["input"]
    bg = inp.get("disease_background") or {}
    mol = bg.get("molecular_profile") or {}
    status = inp.get("current_status") or {}
    tx = inp.get("planned_treatment") or {}
    drugs = [d.get("name") for d in tx.get("drugs") or [] if isinstance(d, dict)]
    return f"""你是一位肿瘤科临床预测专家。请预测这名患者在实际采用下列方案后8–12周是否获得总体净获益。

这是严格二分类任务，只能选择“获益”或“不获益”：
- 获益：8–12周内肿瘤缩小或疾病稳定，且毒性没有抵消疗效。
- 不获益：8–12周内疾病进展、治疗失败或毒性使总体结果有害。

重要：方案符合指南、药物通常有效或存在敏感突变，都不能直接推出这名患者获益。必须同时检查既往同靶点失败、明确耐药机制、快速进展、高肿瘤负荷、差体能状态、CNS/脑膜控制不足、方案与分子机制不匹配等负向证据。不要依据数据集类别比例猜测。

时间切点：{inst.get('time_cutoff')}
诊断：{bg.get('diagnosis')}
转移部位：{bg.get('metastatic_sites')}
分子谱：{json.dumps(mol, ensure_ascii=False)}
既往治疗：{json.dumps(inp.get('prior_treatment_timeline') or [], ensure_ascii=False)}
当前状态：{json.dumps(status, ensure_ascii=False)}
实际方案：{drugs}
联合策略：{tx.get('combination_strategy')}

先简短列出支持获益和支持不获益的证据，再输出严格JSON：
{{"benefit_binary":"获益/不获益","confidence":"高/中/低","positive_factors":["..."],"negative_factors":["..."],"rationale":"..."}}"""


def parse(text):
    decoder = json.JSONDecoder()
    for pos, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[pos:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("benefit_binary") in {"获益", "不获益"}:
            return obj
    return {"benefit_binary": "解析失败", "raw": text}


def messages(inst, rag_inst, with_cases):
    out = [{"role": "system", "content": SYSTEM}]
    if with_cases:
        for ref in (rag_inst.get("_rag_references") or [])[:2]:
            ref_inst = ref.get("reference_instance")
            q = prompt(ref_inst) if isinstance(ref_inst, dict) else "参考患者：\n" + str(ref.get("reference_case") or "")
            a = json.dumps({
                "benefit_binary": label(ref.get("reference_gt")), "confidence": "中",
                "positive_factors": [], "negative_factors": [],
                "rationale": "这是该参考患者的真实随访二分类标签，仅用于示范。",
            }, ensure_ascii=False)
            out.extend([{"role": "user", "content": q}, {"role": "assistant", "content": a}])
    out.append({"role": "user", "content": prompt(inst)})
    return out


def metrics(rows):
    valid = [r for r in rows if r["prediction"] in {"获益", "不获益"}]
    confusion = {g: {p: sum(r["ground_truth"] == g and r["prediction"] == p for r in valid)
                     for p in ("获益", "不获益")} for g in ("获益", "不获益")}
    f1s = []
    for cls in ("获益", "不获益"):
        tp = sum(r["ground_truth"] == cls and r["prediction"] == cls for r in valid)
        fp = sum(r["ground_truth"] != cls and r["prediction"] == cls for r in valid)
        fn = sum(r["ground_truth"] == cls and r["prediction"] != cls for r in valid)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    neg = [r for r in valid if r["ground_truth"] == "不获益"]
    return {
        "n": len(rows), "valid": len(valid),
        "accuracy": sum(r["ground_truth"] == r["prediction"] for r in valid) / len(valid) if valid else None,
        "macro_f1": sum(f1s) / 2 if valid else None,
        "negative_caa": sum(r["prediction"] == "不获益" for r in neg) / len(neg) if neg else None,
        "prediction_distribution": dict(Counter(r["prediction"] for r in rows)), "confusion": confusion,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", type=Path, required=True)
    ap.add_argument("--rag-data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    data = []
    for i in range(7, 12):
        data.extend(json.loads((args.batch_dir / f"batch_{i:03d}.json").read_text()))
    if len(data) != 43 or len({x["instance_id"] for x in data}) != 43:
        raise SystemExit(f"expected exactly 43 unique instances, got {len(data)}")
    rag = {x["instance_id"]: x for x in json.loads(args.rag_data.read_text())}
    keys = [x.strip() for x in os.environ.get("OPENAI_API_KEYS", "").split(",") if x.strip()]
    if not keys and os.environ.get("OPENAI_API_KEY"):
        keys = [os.environ["OPENAI_API_KEY"]]
    if not keys:
        raise SystemExit("missing API key")
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("DEFAULT_BASE_URL")
    tasks = [(arm, x) for x in data for arm in ("pure_llm", "case_rag")]
    lock = threading.Lock()
    state = {"model": args.model, "prompt": "binary", "cases": 43,
             "case_injection_coverage": sum(bool(rag.get(x["instance_id"], {}).get("_rag_references")) for x in data),
             "rows": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def run(task):
        arm, inst = task
        raw = ""
        error = None
        for attempt in range(4):
            try:
                client = OpenAI(api_key=keys[(threading.get_ident() + attempt) % len(keys)], base_url=base_url, max_retries=1)
                res = client.chat.completions.create(model=args.model,
                    messages=messages(inst, rag.get(inst["instance_id"], inst), arm == "case_rag"),
                    temperature=0, max_tokens=1500)
                raw = res.choices[0].message.content or ""
                obj = parse(raw)
                break
            except Exception as exc:
                error = str(exc)
                time.sleep(2 ** attempt)
        else:
            obj = {"benefit_binary": "调用失败"}
        return {"arm": arm, "instance_id": inst["instance_id"],
                "ground_truth_original": inst["ground_truth"]["overall_benefit"],
                "ground_truth": label(inst["ground_truth"]["overall_benefit"]),
                "prediction": obj.get("benefit_binary", "解析失败"), "parsed": obj,
                "raw": raw, "error": error,
                "n_case_references": len((rag.get(inst["instance_id"], {}).get("_rag_references") or [])[:2]) if arm == "case_rag" else 0}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, t) for t in tasks]
        for n, future in enumerate(as_completed(futures), 1):
            row = future.result()
            with lock:
                state["rows"].append(row)
                state["rows"].sort(key=lambda r: (r["instance_id"], r["arm"]))
                args.output.write_text(json.dumps(state, ensure_ascii=False, indent=2))
                print(f"[{n}/86] {row['arm']} {row['instance_id']} GT={row['ground_truth']} P={row['prediction']}", flush=True)
    grouped = {arm: [r for r in state["rows"] if r["arm"] == arm] for arm in ("pure_llm", "case_rag")}
    state["metrics"] = {arm: metrics(rows) for arm, rows in grouped.items()}
    a = {r["instance_id"]: r for r in grouped["pure_llm"]}; b = {r["instance_id"]: r for r in grouped["case_rag"]}
    state["paired"] = {
        "fixed": sum(a[k]["prediction"] != a[k]["ground_truth"] and b[k]["prediction"] == b[k]["ground_truth"] for k in a),
        "harmed": sum(a[k]["prediction"] == a[k]["ground_truth"] and b[k]["prediction"] != b[k]["ground_truth"] for k in a),
        "flipped": sum(a[k]["prediction"] != b[k]["prediction"] for k in a),
    }
    state["paired"]["net"] = state["paired"]["fixed"] - state["paired"]["harmed"]
    args.output.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    print("FINAL", json.dumps({"metrics": state["metrics"], "paired": state["paired"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
