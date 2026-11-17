#!/usr/bin/env python3
"""Render one M7 sample trace (with --dump-prompts) into a full input/output flow.

Usage (inside the container)::

    python scripts/dump_m7_prompts.py \
        --trace artifacts/m7_strict/task2/trace \
        --sample-id retrieval_000001 \
        --out artifacts/m7_strict/task2/flow_retrieval_000001.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fence(text: str, lang: str = "json") -> str:
    return f"```{lang}\n{text}\n```"


def render(trace: dict) -> str:
    lines: list[str] = []
    lines.append(f"# M7 单样本完整 I/O 流程：{trace['sample_id']}（{trace['task']}）")
    lines.append("")
    lines.append(
        f"- json_mode: `{trace.get('json_mode')}`；采样步数: {len(trace.get('steps', []))}；"
        f"耗时: {trace.get('elapsed_seconds')}s；最终来源: `{trace['final']['source']}`"
    )
    steps = trace.get("steps", [])
    if steps and steps[0].get("prompt"):
        lines.append("")
        lines.append("## 0. system message（每个样本固定，模型每条消息都会看到）")
        lines.append("")
        lines.append(fence(steps[0]["prompt"]["system"], "text"))
        lines.append("")
        lines.append("## 1. 初始 user message（样本输入）")
        lines.append("")
        lines.append(fence(steps[0]["prompt"]["user"], "text"))
    for step in steps:
        number = step["step"]
        lines.append("")
        lines.append(f"## step {number}")
        if step.get("prompt"):
            lines.append("")
            lines.append(f"### step {number} 模型看到的 user message（全量）")
            lines.append("")
            lines.append(fence(step["prompt"]["user"], "text"))
        lines.append("")
        lines.append(f"### step {number} 模型原始输出（未做任何修改）")
        lines.append("")
        lines.append(fence(step.get("raw_model_output", ""), "text"))
        lines.append("")
        lines.append(
            f"- strict JSON 校验: `{'PASS' if step.get('strict_json_valid') else 'FAIL'}`"
            + (f"；错误: `{step.get('protocol_error')}`" if step.get("protocol_error") else "")
        )
        parsed = step.get("parsed")
        if parsed:
            lines.append(f"- 解析动作: `{json.dumps(parsed, ensure_ascii=False)}`")
        observation = step.get("observation")
        if observation is not None:
            lines.append("")
            lines.append(f"### step {number} 工具结果 observation（序列化后原样喂回模型）")
            lines.append("")
            lines.append(fence(json.dumps(observation, ensure_ascii=False, indent=2)))
    final = trace.get("final", {})
    lines.append("")
    lines.append("## final")
    lines.append("")
    lines.append(f"- answer: `{json.dumps(final.get('answer'), ensure_ascii=False)}`")
    lines.append(f"- validation: `{json.dumps(final.get('validation'), ensure_ascii=False)}`")
    lines.append(f"- 提交记录: `{json.dumps(final.get('record'), ensure_ascii=False)}`")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True, help="trace directory")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    path = args.trace / f"{args.sample_id}.json"
    trace = json.loads(path.read_text(encoding="utf-8"))
    if not trace.get("steps") or not trace["steps"][0].get("prompt"):
        raise SystemExit("trace has no prompt dump; rerun with --dump-prompts")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(trace), encoding="utf-8")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
