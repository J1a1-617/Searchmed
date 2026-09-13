#!/usr/bin/env python3
"""Binary patient-only vs case-few-shot A/B using GPT-5.5."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from binary_negative_ab import binary_label, load_dotenv
from full_factorial_ablation import messages_for, metrics, paired_delta, parse_prediction


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--rag-data", type=Path, required=True)
    p.add_argument("--dotenv", type=Path, required=True)
    p.add_argument("--model", default="gpt-5.5")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    load_dotenv(args.dotenv)
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEYS", "").split(",")[0]
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("DEFAULT_BASE_URL")
    if not key:
        raise SystemExit("missing API key")
    client = OpenAI(api_key=key, base_url=base_url, max_retries=2)

    data = json.loads(args.data.read_text(encoding="utf-8"))
    rag = {x["instance_id"]: x for x in json.loads(args.rag_data.read_text(encoding="utf-8"))}
    eligible = [x for x in data if rag.get(x["instance_id"], {}).get("_rag_references")]
    pos = [x for x in eligible if binary_label(x["ground_truth"]["overall_benefit"]) == "获益"][:6]
    neg = [x for x in eligible if binary_label(x["ground_truth"]["overall_benefit"]) == "不获益"][:6]
    panel = pos + neg
    arms = {"pure_llm": [], "case_rag": []}
    for inst in panel:
        gt = binary_label(inst["ground_truth"]["overall_benefit"])
        for arm, fewshot in (("pure_llm", False), ("case_rag", True)):
            response = client.chat.completions.create(
                model=args.model,
                messages=messages_for(inst, rag[inst["instance_id"]], "new", fewshot),
                temperature=0,
                max_tokens=1500,
            )
            pred = parse_prediction(response.choices[0].message.content or "", "new")
            arms[arm].append({"instance_id": inst["instance_id"], "ground_truth": gt, "prediction": pred})
            print(arm, inst["instance_id"], "GT=" + gt, "P=" + pred["binary"], flush=True)
    out = {
        "model": args.model,
        "prompt": "binary",
        "knowledge_channels": "case_only; no G/A/B/C",
        "panel": {"n": 12, "benefit": 6, "nonbenefit": 6},
        "pure_llm": {"metrics": metrics(arms["pure_llm"]), "rows": arms["pure_llm"]},
        "case_rag": {"metrics": metrics(arms["case_rag"]), "rows": arms["case_rag"]},
        "paired_case_rag_effect": paired_delta(arms["pure_llm"], arms["case_rag"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pure_llm": out["pure_llm"]["metrics"], "case_rag": out["case_rag"]["metrics"], "paired": out["paired_case_rag_effect"]}, ensure_ascii=False, indent=2))
    print("saved", args.output)


if __name__ == "__main__":
    main()
