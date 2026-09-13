# Predictive Clinical Benchmark

前瞻性临床预测基准 — 用于评估智能用药 Agent 系统的疗效预测能力。

## 任务定义

给定患者在某个时间切点之前可获取的全部临床信息 + 医生实际采用的治疗方案，预测该方案在 8-12 周内的疗效结局。Ground Truth 来自真实随访记录，全自动化评分。

## 数据集

- `benchmark_multinode.json`：135 道预测题目
- 来源：44 例真实肿瘤患者（NSCLC/SCLC/结直肠癌/卵巢癌，均含脑转移/脑膜转移），每例取多个治疗决策节点
- Ground Truth 四分类分布：明显获益 43% / 有限获益或稳定 34% / 无明显获益 10% / 进展或有害 13%

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置 API
export OPENAI_API_KEY="你的key"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # OpenAI 官方；中转站则改为对应地址

# 3. 跑评测（仅主指标）
python run_eval.py --data benchmark_multinode.json --model gpt-4o --output results.json

# 4. 完整评测（含 LLM-Judge 辅助指标）
python run_eval.py --data benchmark_multinode.json --model gpt-4o --judge-model gpt-4o --output results.json

# 5. 验证 benchmark 自身
python validate_benchmark.py --results results.json --data benchmark_multinode.json
```

## 评测指标

### 主指标 M1 — 官方当前评分

| 指标 | 说明 |
|------|------|
| M1 | 临床获益二分类 F1（获益 vs 不获益） |

当前正式 benchmark 只报告 M1（同时保留 accuracy、precision、recall 作为诊断信息）。四分类、RECIST、症状、毒性和时间合规不再进入最终得分。

### 综合得分

当前最终得分为 `M1.f1`，不再计算 composite 或 LLM-as-Judge 辅助分数。

## 自定义模型接入

编辑 `run_eval.py` 中的 `call_model()` 函数：

```python
def call_model(prompt: str, model_name: str = "gpt-4o") -> str:
    if model_name == "my-agent":
        # 你的 Agent 逻辑
        return your_agent.predict(prompt)

    # 默认走 OpenAI API
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    ...
```

然后 `--model my-agent` 即可。

## 直接评测 SearchAgent 工作流

如果你想评测当前仓库里的 `searchagent_retrieval` 整套工作流，可以直接用内置适配模式：

```bash
python run_eval.py \
  --data benchmark_multinode.json \
  --model searchagent-v2 \
  --workers 2 \
  --agent-index-root indexes \
  --agent-top-k 10 \
  --agent-max-steps 4 \
  --agent-max-total-steps 128 \
  --agent-max-budget-extension 32 \
  --output searchagent_results.json
```

全量评测默认使用 2 个 worker。每个 worker 独占一个长生命周期 SearchAgent runtime，
因此 embedding 模型、索引和 SQLite 连接只在该 worker 内复用，不会为135个病例重复加载，
也不会跨线程共享不可线程安全的对象。每个病例仍会新建 Agent loop，并使用独立 session 和 trace。

逐病例结果会原子写入 checkpoint。中断后使用相同命令补充 `--resume`：

```bash
python run_eval.py \
  --data benchmark_multinode.json \
  --model searchagent-v2 \
  --workers 2 \
  --checkpoint-dir searchagent_checkpoints \
  --agent-artifact-root searchagent_case_artifacts \
  --output searchagent_results.json \
  --resume
```

- `checkpoint-dir/<instance_id>.json`：单病例评分输入与状态，临时文件完成 `fsync` 后原子替换。
- `checkpoint-dir/manifest.json`：保存 `run_id`；恢复时沿用该 ID。
- `agent-artifact-root/<run_id>/<instance_id>/session/`：独立 SearchAgent session。
- `agent-artifact-root/<run_id>/<instance_id>/workflow.txt`：人类可读工作流。
- `agent-artifact-root/<run_id>/<instance_id>/workflow_trace.json`：完整结构化 trace。

恢复会跳过状态为 `success` 或 `parse_failure` 的病例；`error` 病例会重试。并行完成顺序
不会影响最终结果，输出会恢复为数据集原始顺序。GPU 显存不足时使用 `--workers 1`。

这个模式会先让 SearchAgent 完成检索、记忆和安全门，再用一层外部 benchmark prompt 把结果压成 benchmark 所需的 JSON schema。也就是说：

- benchmark 负责题目和评分
- SearchAgent 负责检索/推理/安全
- 外层适配器负责把 agent 输出整理成 benchmark 格式

如果你的 dense query embedding 模型不在 `indexes/` 对应的环境里，再补 `--agent-embed-model-path`。

## 数据格式

每条题目：`instance_id`, `time_cutoff`, `input`（疾病背景/既往治疗/当前状态/方案）, `ground_truth`（各维度随访结局）。完整 schema 见 `benchmark_multinode.json`。

## License

待定
