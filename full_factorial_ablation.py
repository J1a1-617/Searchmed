#!/usr/bin/env python3
"""2x2x2 prompt/model/few-shot ablation on a fixed balanced 12-case panel."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))

from binary_negative_ab import binary_label, load_dotenv, parse_json, patient_prompt
from eval.parser import parse_model_output
from eval.prompts import CASES_USAGE, construct_inference_prompt


def old_reference_answer(ref: dict) -> str:
    label = ref.get("reference_gt") or "有限获益或稳定"
    return json.dumps({"overall_benefit": label, "confidence": "中"}, ensure_ascii=False)


def old_reference_question(ref: dict) -> str:
    inst = ref.get("reference_instance")
    if isinstance(inst, dict):
        return construct_inference_prompt(inst)
    return "参考患者：\n" + str(ref.get("reference_case") or "")


def new_reference_answer(ref: dict) -> str:
    return json.dumps(
        {
            "benefit_binary": binary_label(ref.get("reference_gt")),
            "confidence": "中",
            "positive_factors": [],
            "negative_factors": [],
            "rationale": "该标签只属于参考患者。",
        },
        ensure_ascii=False,
    )


def messages_for(base_inst: dict, rag_inst: dict, prompt_kind: str, fewshot: bool) -> list[dict]:
    current = patient_prompt(base_inst) if prompt_kind == "new" else construct_inference_prompt(base_inst)
    if not fewshot:
        return [{"role": "user", "content": current}]
    system = (
        "相似病例只用于对照，参考患者的标签不能直接复制到当前患者。独立权衡当前患者的正负证据。"
        if prompt_kind == "new"
        else CASES_USAGE
    )
    messages: list[dict] = [{"role": "system", "content": system}]
    for ref in (rag_inst.get("_rag_references") or [])[:2]:
        if prompt_kind == "new":
            q, a = patient_prompt(ref["reference_instance"]) if isinstance(ref.get("reference_instance"), dict) else old_reference_question(ref), new_reference_answer(ref)
        else:
            q, a = old_reference_question(ref), old_reference_answer(ref)
        messages.extend([{"role": "user", "content": q}, {"role": "assistant", "content": a}])
    messages.append({"role": "user", "content": current})
    return messages


def parse_prediction(raw: str, prompt_kind: str) -> dict:
    parsed = parse_json(raw) if prompt_kind == "new" else (parse_model_output(raw) or {})
    original = parsed.get("benefit_binary") if prompt_kind == "new" else parsed.get("overall_benefit")
    return {
        "original": original or "解析失败",
        "binary": original if prompt_kind == "new" else (binary_label(original) if original else "解析失败"),
        "confidence": parsed.get("confidence"),
        "rationale": parsed.get("rationale"),
    }


def metrics(rows: list[dict]) -> dict:
    valid = [r for r in rows if r["prediction"]["binary"] in {"获益", "不获益"}]
    pos = [r for r in valid if r["ground_truth"] == "获益"]
    neg = [r for r in valid if r["ground_truth"] == "不获益"]
    acc = sum(r["prediction"]["binary"] == r["ground_truth"] for r in valid) / len(valid) if valid else None
    sensitivity = sum(r["prediction"]["binary"] == "获益" for r in pos) / len(pos) if pos else None
    specificity = sum(r["prediction"]["binary"] == "不获益" for r in neg) / len(neg) if neg else None
    balanced = (sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None
    return {
        "n": len(rows), "valid": len(valid), "accuracy": acc,
        "benefit_recall": sensitivity, "nonbenefit_recall": specificity,
        "balanced_accuracy": balanced,
        "prediction_distribution": dict(Counter(r["prediction"]["binary"] for r in rows)),
    }


def paired_delta(a_rows: list[dict], b_rows: list[dict]) -> dict:
    aa = {r["instance_id"]: r for r in a_rows}
    bb = {r["instance_id"]: r for r in b_rows}
    fixed = harmed = both_right = both_wrong = 0
    flips = Counter()
    for iid in sorted(aa.keys() & bb.keys()):
        ar, br = aa[iid], bb[iid]
        ac = ar["prediction"]["binary"] == ar["ground_truth"]
        bc = br["prediction"]["binary"] == br["ground_truth"]
        if not ac and bc: fixed += 1
        elif ac and not bc: harmed += 1
        elif ac and bc: both_right += 1
        else: both_wrong += 1
        if ar["prediction"]["binary"] != br["prediction"]["binary"]:
            flips[f'{ar["prediction"]["binary"]}->{br["prediction"]["binary"]}'] += 1
    return {"fixed": fixed, "harmed": harmed, "net": fixed - harmed,
            "both_right": both_right, "both_wrong": both_wrong, "prediction_flips": dict(flips)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--rag-data", type=Path, required=True)
    p.add_argument("--dotenv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    load_dotenv(args.dotenv)
    keys = [x.strip() for x in os.environ.get("OPENAI_API_KEYS", "").split(",") if x.strip()]
    if not keys and os.environ.get("OPENAI_API_KEY"): keys = [os.environ["OPENAI_API_KEY"]]
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("DEFAULT_BASE_URL")
    if not keys: raise SystemExit("missing API key")

    data = json.loads(args.data.read_text(encoding="utf-8"))
    rag_data = json.loads(args.rag_data.read_text(encoding="utf-8"))
    rag_by_id = {x["instance_id"]: x for x in rag_data}
    eligible = [x for x in data if (rag_by_id.get(x["instance_id"], {}).get("_rag_references"))]
    negatives = [x for x in eligible if binary_label(x["ground_truth"]["overall_benefit"]) == "不获益"][:6]
    positives = [x for x in eligible if binary_label(x["ground_truth"]["overall_benefit"]) == "获益"][:6]
    panel = positives + negatives
    if len(panel) != 12: raise SystemExit(f"need 6+6 eligible cases, got {len(positives)}+{len(negatives)}")

    arms = [(pmt, model, fs) for pmt in ("old", "new") for model in ("gpt-4o", "gpt-5.4-mini") for fs in (False, True)]
    tasks = [(pmt, model, fs, inst) for pmt, model, fs in arms for inst in panel]
    lock = threading.Lock()
    counter = {"n": 0}

    def run(task):
        pmt, model, fs, inst = task
        msgs = messages_for(inst, rag_by_id[inst["instance_id"]], pmt, fs)
        last = None
        for attempt in range(3):
            try:
                key = keys[(threading.get_ident() + attempt) % len(keys)]
                client = OpenAI(api_key=key, base_url=base_url, max_retries=1)
                response = client.chat.completions.create(model=model, messages=msgs, temperature=0, max_tokens=1500)
                prediction = parse_prediction(response.choices[0].message.content or "", pmt)
                break
            except Exception as exc:
                last = exc
                time.sleep(1 + attempt)
        else:
            prediction = {"original": "调用失败", "binary": "解析失败", "error": str(last)}
        row = {"instance_id": inst["instance_id"], "ground_truth": binary_label(inst["ground_truth"]["overall_benefit"]), "prediction": prediction}
        with lock:
            counter["n"] += 1
            print(f'[{counter["n"]}/96] {pmt}|{model}|fewshot={fs} {inst["instance_id"]} GT={row["ground_truth"]} P={prediction["binary"]}', flush=True)
        return (pmt, model, fs), row

    grouped = {arm: [] for arm in arms}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run, task) for task in tasks]
        for future in as_completed(futures):
            arm, row = future.result()
            grouped[arm].append(row)
    for rows in grouped.values(): rows.sort(key=lambda x: x["instance_id"])

    out = {"panel": [{"instance_id": x["instance_id"], "ground_truth": binary_label(x["ground_truth"]["overall_benefit"])} for x in panel], "arms": {}, "paired_ablations": {}}
    def name(arm): return f"prompt={arm[0]}|model={arm[1]}|fewshot={str(arm[2]).lower()}"
    for arm, rows in grouped.items(): out["arms"][name(arm)] = {"metrics": metrics(rows), "rows": rows}
    factors = {"prompt": 0, "model": 1, "fewshot": 2}
    for factor, idx in factors.items():
        pairs = []
        for arm in arms:
            if (idx == 0 and arm[idx] != "old") or (idx == 1 and arm[idx] != "gpt-4o") or (idx == 2 and arm[idx] is not False): continue
            other = list(arm)
            other[idx] = "new" if idx == 0 else ("gpt-5.4-mini" if idx == 1 else True)
            other = tuple(other)
            pairs.append({"from": name(arm), "to": name(other), **paired_delta(grouped[arm], grouped[other])})
        out["paired_ablations"][factor] = pairs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY")
    for arm in arms: print(name(arm), json.dumps(metrics(grouped[arm]), ensure_ascii=False))
    print("PAIRED", json.dumps(out["paired_ablations"], ensure_ascii=False))
    print("saved", args.output)


if __name__ == "__main__": main()
