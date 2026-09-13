#!/usr/bin/env python3
"""Counterfactual A/B on saved trajectories: original vs applicability-gated context."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from searchagent_retrieval.llm_client import LLMClient
from searchagent_retrieval.benchmark_prediction import BenchmarkPredictionGenerator

def canon(s: str) -> str:
    return re.sub(r"case_0+(\d+)", r"case_\1", s.replace("smoke_", "").replace("_recovered", ""))

def gated_context(ctx: dict) -> dict:
    out = dict(ctx or {})
    direct = list(out.get("key_findings") or [])
    analog = list(out.get("partial_or_analog_findings") or [])
    out["key_findings"] = direct
    # Preserve direct findings. If none exist, forward only two bounded analogs;
    # all remain explicitly analog and cannot become direct evidence.
    out["partial_or_analog_findings"] = analog[:2] if not direct else analog[:4]
    limits = list(out.get("conflicts_and_limitations") or [])
    limits.append("Applicability Skill: 类比证据仅用于有边界的不确定性，不得升级为目标方案直接事实。")
    out["conflicts_and_limitations"] = list(dict.fromkeys(limits))
    return out

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--sessions", type=Path, required=True)
    ap.add_argument("--cases", required=True); ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); wanted = set(args.cases.split(",")); results=[]
    llm = LLMClient(timeout=180, max_retries=1, structured_attempts=2)
    for path in sorted(args.sessions.glob("*.json")):
        data=json.loads(path.read_text()); iid=canon(str(data.get("session_id") or path.stem))
        if iid not in wanted: continue
        base=dict(data); safety=data.get("final_safety_review") or {}
        for variant,ctx in (("original",data.get("answer_context_summary") or {}),("skill",gated_context(data.get("answer_context_summary") or {}))):
            llm.begin_trace(f"skill-ab-{iid}-{variant}")
            gen=BenchmarkPredictionGenerator(llm_client=llm,use_llm=True)
            pred=gen.generate(benchmark_prompt=str(data.get("original_query") or ""), query=str(data.get("original_query") or ""), loop_result=base, safety_result=safety, answer_context_summary=ctx)
            results.append({"instance_id":iid,"variant":variant,"prediction":pred,"source":gen.last_generation_source,"error":gen.last_error,"calls":len(llm.trace_events())})
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(results,ensure_ascii=False,indent=2))
    print(json.dumps({"cases":sorted({r["instance_id"] for r in results}),"rows":len(results)},ensure_ascii=False)); return 0
if __name__ == "__main__": raise SystemExit(main())
