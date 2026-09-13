from __future__ import annotations
import argparse, json
from pathlib import Path
from searchagent_retrieval.llm_client import LLMClient

BASE = """你是肿瘤临床检索助手，只根据输入生成回答，不补充外部知识。
问题：{query}
直接证据：{direct}
其他证据：{weak}
限制与缺口：{limits}
请输出 JSON：{{\"conclusion\":\"\",\"evidence_status\":\"direct|partial|analog|none\",\"regimen_attribution\":\"supported|not_supported|unknown\",\"answer\":\"不超过180字\"}}。
"""

CASES = [
  {"id":"joint_regimen", "query":"EGFR突变肺腺癌脑膜转移，阿美替尼进展后改用培美曲塞联合卡铂，预测8-12周疗效。",
   "direct":"无。", "weak":"病例A：EGFR突变肺癌脑膜转移，奥希替尼联合培美曲塞/卡铂后病灶稳定，未报告阿美替尼进展后8-12周结果。",
   "limits":"药物不同（奥希替尼 vs 阿美替尼）；联合方案证据不能证明目标方案；缺乏8-12周直接结局。"},
  {"id":"drug_trial", "query":"EGFR突变晚期NSCLC一线奥希替尼的FLAURA研究ORR和PFS。",
   "direct":"无。", "weak":"FLAURA相关二手摘要：奥希替尼一线治疗EGFR突变NSCLC的中位PFS约18.9个月，但未给出本题所需ORR、对照组和统计学显著性。",
   "limits":"二手摘要且关键数值不完整；不能把摘要内容写成完整试验结果。"}
]

def call(client, case, mode):
    if mode == "current":
        weak = "支持证据（未特别标注）：" + case["weak"]
        limits = case["limits"]
    elif mode == "explicit":
        weak = "部分/类比证据（仅供参考，不得直接归因）：" + case["weak"]
        limits = case["limits"] + " 必须明确说明证据是partial/analog。"
    else:
        weak = "无（已剔除部分/类比证据）"
        limits = case["limits"] + " 当前没有可用直接证据。"
    prompt = BASE.format(query=case["query"], direct=case["direct"], weak=weak, limits=limits)
    return client.chat(system="严格遵守证据边界。", user=prompt, temperature=0.0, max_output_tokens=500)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); args=ap.parse_args()
    client=LLMClient(model_name='gpt-4o-mini',timeout=30,max_retries=0,structured_attempts=1)
    out=[]
    for case in CASES:
      for mode in ('current','explicit','excluded'):
        try: raw=call(client,case,mode); err=None
        except Exception as exc: raw=""; err=f"{type(exc).__name__}: {exc}"
        out.append({'case':case['id'],'mode':mode,'raw':raw,'error':err})
    Path(args.output).write_text(json.dumps({'results':out,'llm_calls':len(client.trace_events())},ensure_ascii=False,indent=2))
    print(json.dumps({'results':len(out),'errors':sum(x['error'] is not None for x in out),'llm_calls':len(client.trace_events())},ensure_ascii=False))
if __name__=='__main__': main()
