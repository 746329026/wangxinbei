"""Prompts for the M7 tool-calling agent (no thinking mode)."""

from __future__ import annotations

import json
from typing import Any

TOOL_CATALOG = {
    "task2": [
        {
            "name": "get_sample",
            "args": {"sample_id": "retrieval_000001"},
            "returns": "query text, top_k, regions",
        },
        {
            "name": "search_tracks",
            "args": {
                "ir": {
                    "time_start_utc": "2025-01-01T00:00:00Z or null",
                    "time_end_utc": "2025-01-01T06:00:00Z or null",
                    "altitude_range_m": "[min, max] or null",
                    "speed_range_mps": "[min, max] or null",
                    "positive_regions": ["区域 A"],
                    "negative_regions": ["区域 E"],
                    "behavior_sequence": [{"name": "orbit"}],
                    "flags": {"low_speed": True, "very_high_altitude": False},
                },
            },
            "returns": "candidate_count + 每个候选 {id, score, ok, failed, start, alt_m, spd_mps}; count_only=true 时只返回候选数",
        },
        {
            "name": "track_digest",
            "args": {"trajectory_id": "traj_000032"},
            "returns": "summary of one candidate trajectory",
        },
        {
            "name": "region_hit",
            "args": {"trajectory_id": "traj_000032", "region_id": "区域 E"},
            "returns": "whether the trajectory visits that region, with entry/exit events",
        },
        {
            "name": "behavior_check",
            "args": {"trajectory_id": "traj_000032", "behavior": "orbit"},
            "returns": "whether one behavior holds, with numeric evidence",
        },
    ],
    "task1_segment": [
        {"name": "get_sample", "args": {"sample_id": "seg_000001"}, "returns": "instruction, options, trajectory meta"},
        {"name": "trajectory_digest", "args": {"sample_id": "seg_000001"}, "returns": "overall trajectory statistics"},
        {"name": "candidate_boundaries", "args": {"sample_id": "seg_000001", "bins": 20}, "returns": "candidate stage split times"},
        {
            "name": "stage_facts",
            "args": {"sample_id": "seg_000001", "cuts": [0, 30, 60, 90, 121]},
            "returns": "per-stage altitude/speed/spatial facts and labels; cuts are relative seconds",
        },
    ],
    "task1_region": [
        {"name": "get_sample", "args": {"sample_id": "region_000001"}, "returns": "instruction, options, regions"},
        {"name": "trajectory_digest", "args": {"sample_id": "region_000001"}, "returns": "overall trajectory statistics"},
        {
            "name": "region_events",
            "args": {"sample_id": "region_000001", "region_ids": ["a", "b"]},
            "returns": "每个区域的进入/离开 UTC 时刻、停留区间、区外进入次数、进入顺序；region_ids 可省略表示全部区域",
        },
    ],
}

PROTOCOL_TEXT = """你是一个只输出 JSON 的工具调用 Agent。每一步只能输出下面两种 JSON 之一，禁止输出任何解释文字：

1) 调用工具：
{"tool": "<工具名>", "args": {...}, "why": "<可选，<=40字>"}
2) 最终答案：
{"final": {...}, "evidence": [<可选，引用前面支撑结论的步骤号>]}

规则：
- 只允许使用给定工具；禁止编造工具名、禁止修改工具返回的事实。
- 事实必须来自工具返回，禁止自己计算或猜测数字/编号。
- 最终答案要尽量用工具核对过；失败时按错误提示修正 JSON 再试。
- 每个样本有步数上限，请优先调用信息量最大的工具。

输出语法硬性要求（会被严格校验，不通过即视为失败）：
- 每次回复必须且只能是"一个"完整 JSON 对象：第一个非空白字符是 {，最后一个非空白字符是 }；
- 每次回复只能做"一个动作"：要么只输出 tool 对象（等待工具结果），要么只输出 final 对象；**禁止同一个回复里同时出现 tool 和 final**，禁止一次输出多个 JSON 对象；
- 禁止 markdown 代码块（```）、禁止 JSON 前后任何解释文字/注释；
- 必须使用标准 JSON：双引号；不允许尾随逗号；不允许单引号；不允许 Python 字面量 True/False/None（用 true/false/null）；
- 你的原始输出必须能被 json.loads(原始文本) 直接解析通过，不能依赖后处理修复。"""


FORMAT_RULES = """参数格式硬性要求：
- altitude_range_m / speed_range_mps 必须是数字数组，例如 [4000, 7000]；不能写成字符串 "[4000, 7000]"。
- why 可省略；如果写，必须是 {"tool":...,"args":...,"why":...} 的顶层字段，绝不能放进 args 或 ir。
- limit / count_only 可省略（默认 limit=20，已 >= top_k）；不要放进 ir。
- 区域编号写完整形式，如 "区域 A"；"不经过/不满足"的放 negative_regions。
- 数值、时间、编号只能来自工具返回，不要自己编造。"""


def _tool_lines(task: str) -> str:
    lines = []
    for item in TOOL_CATALOG[task]:
        lines.append(
            f"- {item['name']}  参数: {json.dumps(item['args'], ensure_ascii=False)}  返回: {item['returns']}"
        )
    return "\n".join(lines)


TASK2_RULES = """任务：task2 跨模态检索。
- 输入是一个自然语言查询 + 可选区域列表 + top_k（10-15）。
- 你需要把查询解析成约束，调用 search_tracks 检索候选，必要时调整约束/核对候选，最后给出按相关度排序的航迹编号列表。
- 最终答案格式必须是：{"final": {"results": ["traj_000032", "traj_000111", ...]}}
  列表长度 1..top_k，编号不能重复，必须来自 search_tracks 返回的结果。
- 通常 2-3 步即可：search_tracks（可用 count_only 先探测）→ 需要时核对 top 候选 → final。
- final 由系统自动校验（长度/编号/重复等），**不需要先调用 validate_submission 预检**；直接给 final 即可，若被拒绝再按错误提示修正。
- candidate_count>0 时不要重复相同参数的检索；换参数最多再试 1 次，然后必须组织 final。
- 候选数少于 top_k 时按实际候选数给列表即可，禁止编造编号；两次检索都为 0 时可以给 {"results": []}。
- 查询中的"不满足/不经过"要放进 negative_regions；"低速/超高空"用 flags；模糊时间（深夜/凌晨/上午等）自行换算成 time_start_utc/time_end_utc。
"""

TASK2_FEWSHOT = """示例1（一次检索拿到候选）：
用户输入：{"sample_id":"retrieval_000001","text_query":"筛选低速移动且起始时间位于2025年1月1日深夜、经过区域 A，但不满足经过区域 E 的航迹。","top_k":11,"regions":["区域 A","区域 E"]}
输出：{"tool":"search_tracks","args":{"ir":{"time_start_utc":"2025-01-01T00:00:00Z","time_end_utc":"2025-01-01T06:00:00Z","flags":{"low_speed":true},"positive_regions":["区域 A"],"negative_regions":["区域 E"]}}}

示例2（核对单条候选）：
输出：{"tool":"region_hit","args":{"trajectory_id":"traj_000032","region_id":"区域 E"}}

示例3（给最终答案）：
输出：{"final":{"results":["traj_000032","traj_000111","traj_000124"]},"evidence":[1,2]}
"""

TASK1_SEGMENT_RULES = """任务：task1 分段理解（四选一）。
- 输入包含航迹数据与四个选项，选项描述各阶段的飞行形态及顺序。
- 你需要先确认选项的阶段数量，再用 candidate_boundaries 找候选切分点，用 stage_facts 获取每段的高度/速度/空间形态，最后选出与事实一致的选项。
- 最终答案格式必须是：{"final": {"choice": "A"}}
- cuts 是相对起始时间的秒数数组，必须严格递增且覆盖整个航迹（首尾用 0 和 duration_seconds）。
"""

TASK1_SEGMENT_FEWSHOT = """示例1（看题）：
输出：{"tool":"get_sample","args":{"sample_id":"seg_000001"},"why":"确认阶段数与选项"}

示例2（找切分点）：
输出：{"tool":"candidate_boundaries","args":{"sample_id":"seg_000001"}}

示例3（取分段事实）：
输出：{"tool":"stage_facts","args":{"sample_id":"seg_000001","cuts":[0,30,60,90,121]}}

示例4（最终答案）：
输出：{"final":{"choice":"C"},"evidence":[2,3]}
"""

TASK1_REGION_RULES = """任务：task1 区域理解（四选一）。
- 输入包含航迹数据、2/4/6 个多边形区域和四个选项；选项里给出每个区域的到访情况、进入/离开时刻、累计停留、区外进入次数和区域进入顺序。
- 你必须先调用 region_events 拿到每个区域的几何事件事实，再逐项比对选项，选出最符合的描述。
- 注意口径：起点在区内不算一次新的进入；进入顺序按时间重复列出；题干用相对时刻、选项用绝对时刻（工具返回 UTC 时刻）。
- 最终答案格式必须是：{"final": {"choice": "B"}}
"""

TASK1_REGION_FEWSHOT = """示例1（取区域事实）：
输出：{"tool":"region_events","args":{"sample_id":"region_000001"},"why":"拿全部区域进出/停留/顺序"}

示例2（最终答案）：
输出：{"final":{"choice":"A"},"evidence":[1]}
"""


def build_system_prompt(task: str) -> str:
    if task == "task2":
        rules, fewshot = TASK2_RULES, TASK2_FEWSHOT
    elif task == "task1_segment":
        rules, fewshot = TASK1_SEGMENT_RULES, TASK1_SEGMENT_FEWSHOT
    elif task == "task1_region":
        rules, fewshot = TASK1_REGION_RULES, TASK1_REGION_FEWSHOT
    else:
        raise ValueError(f"unsupported task: {task}")
    return "\n\n".join(
        [
            PROTOCOL_TEXT,
            "可用工具：\n" + _tool_lines(task),
            FORMAT_RULES,
            rules,
            fewshot,
        ]
    )


def initial_user_message(sample: dict[str, Any]) -> str:
    return "# 样本输入\n" + json.dumps(sample, ensure_ascii=False, indent=2)


def append_step(transcript: str, step: int, model_raw: str, observation: str) -> str:
    return (
        transcript
        + f"\n\n=== step {step} ===\nASSISTANT: {model_raw.strip()}\nOBSERVATION: {observation.strip()}"
    )
