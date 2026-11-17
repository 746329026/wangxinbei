#!/usr/bin/env python3
"""Run the M7 tool-calling agent on stage1 samples.

Examples (inside the container)::

    # task2 only, first 5 samples (debug)
    python scripts/run_m7_agent.py --task task2 --limit 5

    # full task2 (50 samples) with Qwen3-4B
    python scripts/run_m7_agent.py --task task2

    # task1 region / segment
    python scripts/run_m7_agent.py --task task1_region --limit 5
    python scripts/run_m7_agent.py --task task1_segment --limit 5

Outputs go to ``artifacts/m7/<task>/``: ``traj_result.jsonl``, ``report.json``,
``tool_calls.jsonl`` and per-sample traces under ``trace/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from b01.agent.m7_runtime import AgentModel, M7Runtime  # noqa: E402
from b01.agent.m7_tools import M7Tools  # noqa: E402
from b01.io.jsonl import read_jsonl  # noqa: E402
from b01.submission.schema import validate_submission_records  # noqa: E402

TASK_ALIASES = {
    "task2": "task2",
    "task1_region": "task1_region",
    "task1_segment": "task1_segment",
    "region": "task1_region",
    "segment": "task1_segment",
}


def load_fallbacks(artifacts_dir: Path, task: str) -> dict[str, dict]:
    if task == "task2":
        path = artifacts_dir / "m3" / "task2_baseline_results.jsonl"
    else:
        path = artifacts_dir / "m4" / "task1_baseline_results.jsonl"
    if not path.exists():
        print(f"[m7] warning: fallback file missing: {path}", file=sys.stderr)
        return {}
    rows = list(read_jsonl(path))
    if task == "task2":
        return {row["样本编号"]: {"样本编号": row["样本编号"], "结果": row["结果"]} for row in rows}
    prefix = "region_" if task == "task1_region" else "seg_"
    return {
        row["样本编号"]: {"样本编号": row["样本编号"], "选项": row["选项"]}
        for row in rows
        if row["样本编号"].startswith(prefix)
    }


def expected_ids(tools: M7Tools, task: str) -> list[str]:
    if task == "task2":
        return [row["sample_id"] for row in tools.queries]
    if task == "task1_region":
        return [row["sample_id"] for row in tools.region_rows]
    return [row["sample_id"] for row in tools.segment_rows]


def run_one(args: argparse.Namespace, task: str, model: AgentModel | None) -> dict:
    artifacts = args.artifacts_dir
    tools = M7Tools(str(artifacts / "m1"), str(artifacts / "m3"))
    fallbacks = load_fallbacks(artifacts, task)

    ids = expected_ids(tools, task)
    if args.sample_id:
        wanted = set(args.sample_id)
        ids = [sample_id for sample_id in ids if sample_id in wanted]
    if args.limit and args.limit > 0:
        ids = ids[: args.limit]

    output_dir = args.output / task
    runtime = M7Runtime(
        tools=tools,
        task=task,
        fallbacks=fallbacks,
        output_dir=output_dir,
        model=model,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        sample_ids=ids,
        json_mode=args.json_mode,
        dump_prompts=args.dump_prompts,
    )
    result = runtime.run()
    records = result["records"]
    validation = validate_submission_records(records, ids, task1_field_name="选项")
    report = dict(result["report"])
    report["validation"] = validation
    report["output_dir"] = str(output_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default="task2",
        choices=sorted(TASK_ALIASES),
        help="task to run (region/segment are aliases for task1_region/task1_segment)",
    )
    parser.add_argument("--model-name", default="/models/Qwen3-4B")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--limit", type=int, default=0, help="only run the first N samples (0 = all)")
    parser.add_argument("--sample-id", action="append", default=[], help="debug a specific sample id")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "m1")
    parser.add_argument("--m3-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "m3")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "m7")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument(
        "--json-mode",
        choices=("strict", "lenient"),
        default="strict",
        help="strict: model output itself must be valid JSON (no extraction/repair); lenient: legacy extraction/repair",
    )
    parser.add_argument("--dump-prompts", action="store_true", help="store the exact system/user messages the model saw on every step")
    parser.add_argument("--no-model", action="store_true", help="tools only: run without loading the model (fallback path)")
    args = parser.parse_args()

    task = TASK_ALIASES[args.task]
    model = None
    if not args.no_model:
        model = AgentModel(args.model_name, dtype=args.dtype)
        print(f"[m7] backend=qwen3 model={args.model_name} task={task}", flush=True)
    else:
        print(f"[m7] backend=no-model task={task}", flush=True)

    report = run_one(args, task, model)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["validation"]["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
