#!/usr/bin/env python3
"""Controlled ablation of sampling, prompt, and model on benefit prediction."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))

from binary_negative_ab import binary_label, load_dotenv, parse_json, patient_prompt
from eval.parser import parse_model_output
from eval.prompts import construct_inference_prompt


def call(client: OpenAI, model: str, prompt: str, binary: bool) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1500,
    )
    raw = response.choices[0].message.content or ""
    parsed = parse_json(raw) if binary else (parse_model_output(raw) or {})
    original = parsed.get("benefit_binary") if binary else parsed.get("overall_benefit")
    predicted_binary = original if binary else binary_label(original)
    return {
        "original_prediction": original or "解析失败",
        "predicted_binary": predicted_binary if original else "解析失败",
        "confidence": parsed.get("confidence"),
        "rationale": parsed.get("rationale"),
    }


def summarize(rows: list[dict]) -> dict:
    valid = [r for r in rows if r["prediction"]["predicted_binary"] != "解析失败"]
    correct = [r for r in valid if r["prediction"]["predicted_binary"] == r["ground_truth_binary"]]
    negatives = [r for r in valid if r["ground_truth_binary"] == "不获益"]
    true_negatives = [r for r in negatives if r["prediction"]["predicted_binary"] == "不获益"]
    return {
        "n": len(rows),
        "gt_distribution": dict(Counter(r["ground_truth_binary"] for r in rows)),
        "prediction_distribution": dict(Counter(r["prediction"]["predicted_binary"] for r in rows)),
        "accuracy": len(correct) / len(valid) if valid else None,
        "negative_recall": len(true_negatives) / len(negatives) if negatives else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--dotenv", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    load_dotenv(args.dotenv)
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEYS", "").split(",")[0]
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("DEFAULT_BASE_URL")
    if not key:
        raise SystemExit("missing API key")
    client = OpenAI(api_key=key, base_url=base_url)

    data = json.loads(args.data.read_text(encoding="utf-8"))
    by_id = {x["instance_id"]: x for x in data}
    previous = json.loads(args.previous.read_text(encoding="utf-8"))
    neg_ids = [x["instance_id"] for x in previous["cases"]]
    neg6 = [by_id[i] for i in neg_ids]
    mixed6 = data[:6]

    conditions = {
        # The common control for the prompt and model comparisons.
        "neg_oldprompt_gpt4o": (neg6, "old", "gpt-4o"),
        "neg_newprompt_gpt4o": (neg6, "new", "gpt-4o"),
        "neg_oldprompt_gpt54mini": (neg6, "old", "gpt-5.4-mini"),
        # Sampling comparison against the already-computed neg/new/5.4-mini arm.
        "mixed_newprompt_gpt54mini": (mixed6, "new", "gpt-5.4-mini"),
    }

    result: dict[str, object] = {"conditions": {}}
    for name, (items, prompt_kind, model) in conditions.items():
        rows = []
        for inst in items:
            gt = binary_label((inst.get("ground_truth") or {}).get("overall_benefit"))
            prompt = patient_prompt(inst) if prompt_kind == "new" else construct_inference_prompt(inst)
            prediction = call(client, model, prompt, binary=prompt_kind == "new")
            row = {"instance_id": inst["instance_id"], "ground_truth_binary": gt, "prediction": prediction}
            rows.append(row)
            print(name, inst["instance_id"], "GT=" + gt, "P=" + prediction["predicted_binary"], flush=True)
        result["conditions"][name] = {"summary": summarize(rows), "rows": rows}

    reused_rows = []
    for old in previous["cases"]:
        prediction = old["A_no_fewshot"]
        reused_rows.append(
            {
                "instance_id": old["instance_id"],
                "ground_truth_binary": "不获益",
                "prediction": {
                    "original_prediction": prediction.get("benefit_binary"),
                    "predicted_binary": prediction.get("benefit_binary"),
                    "confidence": prediction.get("confidence"),
                    "rationale": prediction.get("rationale"),
                },
            }
        )
    result["conditions"]["neg_newprompt_gpt54mini_reused"] = {
        "summary": summarize(reused_rows),
        "rows": reused_rows,
    }
    result["comparisons"] = {
        "sampling": ["neg_newprompt_gpt54mini_reused", "mixed_newprompt_gpt54mini"],
        "prompt": ["neg_oldprompt_gpt4o", "neg_newprompt_gpt4o"],
        "model": ["neg_oldprompt_gpt4o", "neg_oldprompt_gpt54mini"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSUMMARIES")
    for name, value in result["conditions"].items():
        print(name, json.dumps(value["summary"], ensure_ascii=False))
    print("saved", args.output)


if __name__ == "__main__":
    main()
