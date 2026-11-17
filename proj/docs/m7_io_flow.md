# M7 Agent 完整输入/输出流程与"严格 JSON"保证

> 对应代码：`proj/src/b01/agent/{m7_prompts,m7_protocol,m7_runtime,m7_tools}.py`、`proj/scripts/run_m7_agent.py`
> 本文说明：**模型每一步到底看到什么、输出什么、框架做了什么**，以及如何保证
> "模型自己吐出来的是结构化 JSON"，而不是"框架把模型的自然语言解析成结构化结果"。

## 1. 总体数据流

```
样本 JSON (M1 缓存)
  │  _sample_payload(): 只保留模型需要的最小字段
  ▼
system message = PROTOCOL_TEXT + 工具清单 + 参数格式规则 + 任务规则 + few-shot
user message 0 = "# 样本输入\n" + json.dumps(sample_payload, indent=2)
  │
  ▼ 循环（最多 max_steps=6 步）
messages = [{"role":"system","content":system},
            {"role":"user","content":transcript}]     # 只有两条消息！
  │  tokenizer.apply_chat_template(..., enable_thinking=False, add_generation_prompt=True)
  ▼
Qwen3-4B 贪心生成 max_new_tokens=384 → decode(skip_special_tokens=True).strip()
  ▼
raw_model_output（原始文本，不做任何修改）
  │  strict_json_load(raw): json.loads(raw) —— strict 模式下唯一入口
  ▼
动作校验（恰好一个：{"tool","args","why"} 或 {"final","evidence"}）
  ├─ call  → 工具白名单 → 参数修复(仅语义层) → 执行工具 → {
  │            "ok": true/false, "data"/"error", "_hint": 策略提示
  │          }  → 序列化 → 追加到 transcript：
  │            transcript += "\n\n=== step k ===\nASSISTANT: <raw>\nOBSERVATION: <json>"
  └─ final → 按任务校验 → 通过则结束，转成官方提交记录；不通过则把错误作为 observation 追加，继续循环
  ▼
traj_result.jsonl（官方格式）+ trace/<sample>.json + tool_calls.jsonl + report.json
```

要点：

1. **模型只看到两条 chat message**：system + user。工具结果不是单独的 `tool` role，
   而是作为文本 `OBSERVATION: {json}` 追加进同一条 user message（transcript）。
2. **模型的每个动作都是 raw 文本 JSON**，框架不做"自然语言→JSON"抽取（strict 模式）。
3. 历史对话也会被模型看到：第 k 步的 user message 里包含前 k-1 步的 `ASSISTANT: ... OBSERVATION: ...`。

## 2. 每一步模型看到的全量内容

### 2.1 system message（task2 为例，每步相同）

由 `build_system_prompt(task)` 拼接，顺序固定：

1. `PROTOCOL_TEXT`：只允许输出两种 JSON 动作；禁止解释文字；**输出语法硬性要求**（一个完整 JSON 对象、不能有 markdown/前后缀/多个对象、标准 JSON、必须能被 `json.loads` 直接解析）；
2. `可用工具：` + 每个工具的名称、参数示例、返回说明（task2：`get_sample/search_tracks/track_digest/region_hit/behavior_check`；`validate_submission` 仍保留在代码白名单中，但**不再暴露给模型**——final 由 runtime 自动校验，避免模型"先 validate 再 final"拼两个对象）；
3. `FORMAT_RULES`：范围必须是数字数组、`why`/`limit` 位置、区域名要求、事实必须来自工具；
4. `TASK2_RULES`：任务口径、final 格式、2–3 步策略、候选为 0 时怎么处理；
5. `TASK2_FEWSHOT`：3 个 JSON 输出示例。

> task1_segment / task1_region 只有第 2/4/5 段不同（工具清单、任务规则、few-shot 不同）。

### 2.2 初始 user message

```text
# 样本输入
{<sample_payload, indent=2>}
```

`_sample_payload` 字段（只给必要信息）：

- task2：`task / sample_id / text_query / top_k / regions[{region_id, boundary}]`
- task1_region：`task / sample_id / instruction / regions[{region_id, coordinates}]`
- task1_segment：`task / sample_id / instruction`

### 2.3 第 k 步（k ≥ 1）的 user message

```text
<上一步的完整 transcript>
（首次即初始 user message；之后每步追加：）

=== step k ===
ASSISTANT: <第 k 步模型原始输出>
OBSERVATION: <第 k 步工具结果 JSON，含 _hint>
```

## 3. 模型输出与框架校验（重点）

### 3.1 strict 模式（当前默认，`--json-mode strict`）

- 解析入口只有 `strict_json_load(raw)`：
  ```python
  payload = json.loads(raw.strip())      # 没有 find("{")、没有代码块剥离、没有补括号、没有多对象取第一个
  ```
- 失败即 `ProtocolError`，本步记为 `strict_json_valid=false` / `strict_json_fail`，
  写一条 `{"ok":false,"error":{"code":"PROTOCOL",...}}` 的 observation 让模型重试；
- 语义校验（`parse_action`）在解析成功之后，只允许：
  - `{"tool": str, "args": object, "why"?: str<=200}`（键集合严格，禁止多余键）；
  - `{"final": object, "evidence"?: [int]}`；
  - `{"tool"}` 与 `{"final"}` 恰好出现一个。
- 语义层的 **参数修复**（`_repair_tool_args`）仍然存在（把模型放错层级的 `why`/`limit` 移回、字符串 IR 解析等），
  但它只影响工具参数，不改变"模型原始输出是 JSON"这一事实；修复次数单独记录 `arg_repairs`。

### 3.2 lenient 模式（仅用于对照/历史兼容，`--json-mode lenient`）

- 会做：`<think>` 剥离、markdown fence 提取、`find("{")` 截取、缺括号/尾逗号修复、多对象取第一个；
- 每步仍会记录 `strict_json_valid`，以及 `lenient_recovered`（strict 不通过但 lenient 救回来的次数）。

> 因此报告里的 `steps.strict_json_ok / strict_json_fail / lenient_recovered` 可以直接回答：
> **模型自己的结构化输出率是多少、有多少是框架"救"回来的**。

### 3.3 报告字段

```json
{
  "json_mode": "strict",
  "steps": {
    "strict_json_ok": 0,          // 模型原始输出直接 json.loads 成功的步数
    "strict_json_fail": 0,        // 失败步数（strict 模式下=ProtocolError 次数）
    "lenient_recovered": 0,       // 仅 lenient 模式有意义
    "protocol_errors": 0,
    "arg_repairs": 0
  }
}
```

trace 每步字段：`raw_model_output`、`strict_json_valid`、`json_mode`、`parse_ok`、`parsed`、
`observation`、`tool`、`arg_repairs`；加 `--dump-prompts` 时还有 `prompt.system` / `prompt.user`（全量输入）。

## 4. 工具结果的格式（模型看到的 observation）

工具统一返回信封：

```json
{"ok": true, "data": {"...": "..."}}
{"ok": false, "error": {"code": "BAD_ARGS|NOT_FOUND|UNKNOWN_TOOL|PROTOCOL", "message": "...", "hint": "..."}}
```

runtime 会给 observation 加一个 `_hint` 字段（策略提示，如"候选为 0，请放宽条件"），
模型据此在下一步调整动作。工具结果超过 8KB 时会被裁剪并带 `_truncated: true`。

## 5. 最终答案校验与提交转换

- task2：`final.answer` 必须是 `{"results": [ids]}`；校验"至少调用过一次 search_tracks、id 在候选集合内、不重复、长度 1..top_k"（候选为 0 时允许空列表）；超过 top_k 自动截断并记录 `auto_truncated_finals`；
- task1：`final.answer` 必须是 `{"choice": "A/B/C/D"}`，校验格式与是否在选项内；
- 校验通过 → `_to_record()` 转成官方字段：
  - `{"样本编号": sample_id, "结果": [{"序号": n, "航迹编号": id}, ...]}`
  - `{"样本编号": sample_id, "选项": "A"}`
- 校验失败 → 错误 observation 回给模型继续循环；步数用尽 → 回退 deterministic 结果（保证一定有合法提交）。

## 6. 如何检查"模型真的自己输出了 JSON"

```bash
# 1) strict 模式跑一批样本（默认就是 strict）
docker exec -w /workspace/repo/proj b01-dev python scripts/run_m7_agent.py \
  --task task2 --limit 20 --json-mode strict --output artifacts/m7_strict

# 2) 跑单个样本并 dump 每一步的完整 system/user message
docker exec -w /workspace/repo/proj b01-dev python scripts/run_m7_agent.py \
  --task task2 --sample-id retrieval_000001 --json-mode strict --dump-prompts \
  --output artifacts/m7_strict_one

# 3) 把 trace 渲染成完整 I/O 文档
docker exec -w /workspace/repo/proj b01-dev python scripts/dump_m7_prompts.py \
  --trace artifacts/m7_strict_one/task2/trace --sample-id retrieval_000001 \
  --out artifacts/m7_strict_one/task2/flow_retrieval_000001.md

# 4) 看 strict 通过率
docker exec b01-dev python -c "import json; d=json.load(open('/workspace/repo/proj/artifacts/m7_strict/task2/report.json')); print(d['json_mode'], d['steps'])"
```

产物（均在 `proj/artifacts/` 下，gitignore）：

- `m7_strict/task2/trace/<sample>.json`：每步 raw 输出 + strict 校验 + 全量 prompt（`--dump-prompts`）
- `m7_strict/task2/flow_<sample>.md`：单样本从 system、初始输入、每步输入/输出/observation 到 final 的完整文档
- `m7_strict/task2/report.json`：strict 通过率等统计

## 7. 已知的模型违规模式（strict 模式下会被拒绝并重试）

- 一个回复里连续输出两个 JSON 对象（例如 `{"tool":...}\n{"final":...}`）；
- 输出前后带解释文字或 markdown 代码块；
- 尾随逗号、单引号、Python 字面量 `True/False/None`；
- 输出被截断（`max_new_tokens` 用尽）导致 JSON 不完整。

这些在 strict 模式下都会触发 `PROTOCOL` 错误 observation，模型带着错误提示重试；
最终仍失败则计入 `fallback`。报告里的 `strict_json_fail` 就是"模型没做到一次输出合法 JSON"的次数，
这正是需要继续优化 prompt / 推理策略的地方。

## 8. 实测：strict 模式下模型自己输出的 JSON 合规率

| 运行 | 任务 | 样本 | 模型输出次数 | strict_json_ok | strict_json_fail | lenient_recovered | fallback | strict 合规率 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 加固前（prompt 建议 validate_submission→final） | task2 | 50 | 198 | 164 | 34 | 0 | 4 | 82.8% |
| 加固前（提示"禁止同回复多对象"） | task2 | 10 | 36 | 34 | 2 | 0 | 0 | 94.4% |
| **加固后（移除 validate_submission 预检建议）** | **task2** | **50** | **179** | **179** | **0** | **0** | **0** | **100%** |
| 加固后 | task1_segment | 10 | 40 | 40 | 0 | 0 | 0 | 100% |
| 加固后 | task1_region | 10 | 30 | 30 | 0 | 0 | 0 | 100% |

失败归因（加固前 34 次）：**34/34 都是 `validate_submission` 的 tool 对象后面紧跟一个 `final` 对象**（其中 27 次 final 被截断）。
也就是说，模型本身并不是"不会写 JSON"，而是在"先自检再给答案"的引导下把两个动作写进了同一个回复。
把 `validate_submission` 从 system prompt 的工具清单中移除（final 由 runtime 自动校验）后，
task2 的 strict 合规率达到 100%（179/179），且模型反而更多使用了 `behavior_check`（44 次）等取证工具。

复现命令与产物：

- 最终 strict 运行：`artifacts/m7_strict_final/task2/`（含 `report.json`、`trace/`、`traj_result.jsonl`）
- 单样本完整 I/O：`artifacts/m7_strict_final/task2/flow_retrieval_000001.md`
- task1 strict 运行：`artifacts/m7_strict2/task1_segment/`、`artifacts/m7_strict2/task1_region/`
