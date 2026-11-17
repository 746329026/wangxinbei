# M7 工具调用 Agent（协议 / 工具 / 日志）

> 状态：**P1 完成**。task1-region / task1-segment / task2 三个任务的全量 50 条均已跑通，
> 全部由 agent 自主调用工具完成，0 回退、全部生成合法提交。
> 原则：提交格式不变（task1 单选、task2 top-k 列表）；**不开 thinking 模式**；
> 模型只输出 JSON 动作（调工具 / 给最终答案），事实必须来自工具；全量日志可审计。

## 1. 与 M6 的区别

| | M6（固定流水线） | M7（Agent） |
|---|---|---|
| 模型职责 | 解析 QueryIR / 判 A-D | 自主决定调哪个工具、传什么参数、是否再来一轮 |
| 工具调用 | controller 写死顺序 | 模型按需选择，白名单内 |
| 多轮检索 | 无（解析一次） | 支持探测→放宽→逐条核对 |
| 产物 | traj_result + report | 同上 + 每步 trace + 工具调用日志 |

## 2. 输出协议（模型每步只能二选一）

```json
{"tool": "search_tracks", "args": {"ir": {...}, "limit": 20}, "why": "先探测候选"}
{"final": {"results": ["traj_000032", "traj_000111"]}, "evidence": [1, 2]}
```

- `why` / `evidence` 仅用于审计，不参与判定；
- 最终答案由 runtime 转成官方格式（task1 `{"样本编号","选项"}`；task2 `{"样本编号","结果":[{"序号","航迹编号"}]}`）；
- 非法动作（非 JSON / 未知工具 / 白名单外工具 / final 校验失败）→ 带错误提示重试，预算耗尽则回退 M5/M6 确定性结果。

## 3. 工具清单

### task2
| 工具 | 参数 | 返回 |
|---|---|---|
| `get_sample` | `sample_id` | 查询文本、top_k、区域边界 |
| `search_tracks` | `ir`（时间/高度/速度/正负区域/行为/flags/thresholds）、`limit`、`count_only` | 候选数 + 每条候选的 matches（各约束是否满足）与 digest |
| `track_digest` | `trajectory_id` | 单条航迹摘要（时间/高度/速度/行为 flags/stages） |
| `region_hit` | `trajectory_id, region_id` | 是否到访 + 进出事件/停留 |
| `behavior_check` | `trajectory_id, behavior` | 单个行为是否成立 + 数值证据 |
| `validate_submission` | `sample_id, answer` | 格式/数量/唯一性/编号来源校验 |

### task1-segment
`get_sample`、`trajectory_digest`、`candidate_boundaries`（变化点候选）、`stage_facts`（按 `cuts` 返回每段高度/速度/空间事实与标签）。

### task1-region
`get_sample`、`trajectory_digest`、`region_events`（每区域进出时刻/停留/次数/进入顺序，UTC 精确到毫秒）。

## 4. 预算与回退

- 每样本 `max_steps=6`（task1 可调小），`max_new_tokens=384`；
- 同 `(tool,args)` 重复调用直接返回缓存并标记 `repeated`；
- 贪心输出与上一步完全相同时，自动采样一次以打破循环；
- 两种常见小模型错误自动修复：
  1. **JSON 结构**：缺少闭合括号/截断字符串 → 自动补齐后解析；
  2. **参数错位**：`why`/`limit` 放错层级、范围写成字符串 `"[4000, 7000]"`、区域名 `A` → 运行时/工具归一化；
- 失败兜底：`artifacts/m3/task2_baseline_results.jsonl`（task2）、`artifacts/m4/task1_baseline_results.jsonl`（task1）。

## 5. 日志（用于后续工具完善度/冗余度分析）

输出目录：`proj/artifacts/m7/<task>/`

```
traj_result.jsonl     # 最终提交（官方格式）
report.json           # 汇总：样本/回退/步数/工具调用分布/错误/时延/截断/重复/未使用工具
tool_calls.jsonl      # 每次工具调用一行（含 args_hash / elapsed_ms / truncated / cached / cited_in_final）
trace/<sample_id>.json# 每样本逐步：模型原始输出、解析动作、工具observation、校验、最终答案
```

`report.json` 关键字段：
`steps.protocol_errors / arg_repairs / repeated_outputs`、`tools.calls / errors / cached_repeated / truncated / cited_calls / unused_whitelisted`、`samples.fallback`。

**这些字段就是为了回答**：哪个工具从没被用（可删）、哪个工具被调用但结论从未引用（低价值）、哪个工具经常返回 BAD_ARGS/被截断（不完善）、是否出现"探测→调整→取证"的多轮序列。

## 6. 运行方式

```bash
cd /workspace/repo/proj

# task2 全量（50 条）；默认 --json-mode strict：模型原始输出必须自己就是合法 JSON
python scripts/run_m7_agent.py --task task2

# 小样本调试 / 指定样本
python scripts/run_m7_agent.py --task task2 --limit 5
python scripts/run_m7_agent.py --task task2 --sample-id retrieval_000001

# dump 每一步模型看到的完整 system/user message（用于审计 I/O）
python scripts/run_m7_agent.py --task task2 --sample-id retrieval_000001 --dump-prompts

# 生成单样本完整 I/O 文档
python scripts/dump_m7_prompts.py --trace artifacts/m7/task2/trace \
  --sample-id retrieval_000001 --out artifacts/m7/task2/flow_retrieval_000001.md

# 历史兼容：允许框架做 JSON 提取/修复（会统计 lenient_recovered），仅用于对照
python scripts/run_m7_agent.py --task task2 --json-mode lenient

# task1
python scripts/run_m7_agent.py --task task1_region --limit 5
python scripts/run_m7_agent.py --task task1_segment --limit 5
```

> 完整 I/O 流程（模型每步看到什么/输出什么/框架如何校验）见
> [`proj/docs/m7_io_flow.md`](proj/docs/m7_io_flow.md)。

## 7. 本次全量实测结果

| 任务 | 运行时长 | 样本 | 合法提交 | agent 完成 | 回退 | 平均步数 | 工具调用 |
|---|---|---|---|---|---|---|---|
| task2 | 609 s | 50 | 50 | 50 | 0 | 3.28 | search_tracks 70、validate_submission 37、behavior_check 7 |
| task1-region | 251 s | 50 | 50 | 50 | 0 | 2.84 | region_events 92 |
| task1-segment | 292 s | 50 | 50 | 50 | 0 | 4.00 | get_sample 50、candidate_boundaries 50、stage_facts 50 |

补充指标（report.json）：

- task2：`truncated=0`、`cached_repeated=5`、`BAD_ARGS=4`、`empty_results=4`、`auto_truncated_finals=4`；
  与 deterministic 基线（M3）top-k 平均 Jaccard = **0.720**，返回条数 0–15（平均 6.2）。
- task1-region：`cached_repeated=33`、`repeated_outputs=30`、`BAD_ARGS=5`、`NOT_FOUND=7`；
  与 deterministic 基线（M4）选项一致率 **17/50**。
- task1-segment：工具序列 50/50 都是 `get_sample → candidate_boundaries → stage_facts → final`；
  与 M4 一致率 **5/50**（两套方法差异很大，需要评测集或人工抽检才能判断谁更准）。

提交产物：

- `proj/artifacts/m7/task2/traj_result.jsonl`
- `proj/artifacts/m7/task1_region/traj_result.jsonl`
- `proj/artifacts/m7/task1_segment/traj_result.jsonl`

## 8. 工具问题与冗余分析（基于全量日志）

| 工具 | 现象 | 结论 |
|---|---|---|
| `get_sample` | 3 个任务中基本 0 调用（题面已由 runtime 作为初始 observation 注入） | **冗余候选**：可删除或仅保留 debug 用途 |
| `trajectory_digest` | 0 调用（task1 两个任务都没用） | **冗余候选**：与 `stage_facts`/题面信息重叠 |
| `track_digest` / `region_hit` | task2 中 0 调用（`behavior_check` 被用 7 次） | 保留观察：模型更依赖 `search_tracks` 返回的 `ok/failed`，需要在 prompt 里强化"逐条核对"场景 |
| `region_events` | 92 次调用 / 33 次 cached 重复 / 30 次 repeated output / NOT_FOUND 7 | **待完善**：模型反复请求同一区域数据；建议返回结果里直接附"区域 id 列表与选项对应关系"，或支持一次传多区域并在 prompt 给出可选 id |
| `search_tracks` | 70 次调用、BAD_ARGS 4、cached 5 | 参数归一化（字符串范围/区域短名/行为别名）已生效；BAD_ARGS 集中在时间格式 |
| `validate_submission` | 37 次调用，有效拦截了"结果条数 > top_k"问题 | 保留；这是最终提交正确率的关键工具 |

运行层面的关键修复（已固化在 runtime）：

1. JSON 处理已改为**默认 strict**：模型原始输出必须能被 `json.loads(raw)` 直接解析，否则记为 `strict_json_fail` 并重试；不再默认做提取/补括号/多对象取第一个（lenient 模式仅保留给历史对照，并统计 `lenient_recovered`）；
2. 参数错位修复：`why`/`limit` 放错层级、范围写成字符串、区域短名、行为英文别名（acceleration→step_accel 等）；
3. final 结果超过 `top_k` 自动截断并记录 `auto_truncated_finals`；
4. 重复输出检测 + 一次采样重试打破贪心死循环；同参数重复调用返回缓存并提示换策略；
5. 候选为 0 时允许空结果 final（记录 `empty_results`，避免死循环）。

## 9. 已知待改进

- task2 的检索轮次仍偏少（多数是 search → validate → final），`track_digest/region_hit` 未被主动使用，多轮取证能力需要 prompt/示例加强；
- task1-region 存在重复调用同一工具（30 次重复输出），可考虑在 observation 里注入"你已拿到该区域全部事实，请逐项比对选项"的策略提示；
- task1 的准确率无法评估（stage1 无官方答案），当前与 M4 基线差异大，需要评测集或人工抽检；
- 可考虑把 `get_sample`/`trajectory_digest` 从白名单移除，减少模型选择负担（冗余工具分析结论）。

## 10. 严格 JSON 模式（默认）与完整 I/O 流程

从 2026-11 起，`run_m7_agent.py` 默认 `--json-mode strict`：**模型原始输出必须自己就是合法 JSON**，
框架不再做"从文字里提取/补括号/多对象取第一个"这类修复。

- 解析入口只有 `json.loads(raw.strip())`（`m7_protocol.strict_json_load`）；失败即 `PROTOCOL` 错误，
  写回错误 observation 让模型重试，并累计 `steps.strict_json_fail`；
- `report.json` 新增：`json_mode`、`steps.strict_json_ok / strict_json_fail / lenient_recovered`；
- trace 每步新增：`strict_json_valid`、`json_mode`；`--dump-prompts` 时还记录 `prompt.system` 与 `prompt.user`（全量输入）；
- `--json-mode lenient` 仅保留历史对照（会统计 `lenient_recovered`，即被框架"救回来"的次数）。

**实测 strict 合规率（模型原始输出直接可解析）**：

| 任务 | 样本 | 输出次数 | strict 合规率 | fallback |
|---|---:|---:|---:|---:|
| task2（加固后） | 50 | 179 | **100%** | 0 |
| task1_segment | 10 | 40 | **100%** | 0 |
| task1_region | 10 | 30 | **100%** | 0 |

> 加固前 task2 只有 82.8%（34 次失败全部是"tool 对象 + final 对象"写进同一回复）；
> 根因是 prompt 里建议"validate_submission → final"，模型把两个动作合并输出。
> 移除 `validate_submission` 预检建议（final 由 runtime 自动校验）后达到 100%。

完整 I/O 文档见 [`proj/docs/m7_io_flow.md`](proj/docs/m7_io_flow.md)；
单样本从 system prompt、初始输入、每步 user/raw/observation 到 final 的全文见
`proj/artifacts/m7_strict_final/task2/flow_retrieval_000001.md`（用 `scripts/dump_m7_prompts.py` 生成）。
