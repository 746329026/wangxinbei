"""M7 ReAct-style runtime: strict tool-call loop with full trace logging.

Design constraints (agreed with the team):
* the model only emits ``{"tool", "args"}`` or ``{"final"}`` JSON - no prose,
  no thinking mode, no free-form reasoning;
* every fact must come from a tool observation;
* a per-sample budget bounds steps/tool calls and falls back to the validated
  deterministic baseline on failure;
* each step, tool call, raw model output and validation result is logged so the
  tool set can be audited for gaps and redundancy later.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from b01.agent.m7_protocol import ProtocolError, parse_action, strict_json_load
from b01.agent.m7_prompts import append_step, build_system_prompt, initial_user_message
from b01.agent.m7_tools import TASK_TOOL_WHITELIST, M7Tools

SCHEMA_VERSION = "m7.0"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha1(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


class AgentModel:
    """Lazy local Qwen3 wrapper that never enables thinking mode."""

    def __init__(self, model_name: str, dtype: str = "float16") -> None:
        self.model_name = model_name
        self.dtype = dtype
        self._loaded = False

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
        kwargs["torch_dtype"] = torch.float16 if self.dtype == "float16" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self.model.eval()
        for field in ("temperature", "top_p", "top_k"):
            if hasattr(self.model.generation_config, field):
                setattr(self.model.generation_config, field, None)
        self._loaded = True

    def generate(self, system: str, user: str, max_new_tokens: int = 384, *, sample: bool = False) -> str:
        if not self._loaded:
            self._load()
        import torch

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            )
        except TypeError:
            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
        inputs = {"input_ids": inputs} if not hasattr(inputs, "items") else inputs
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": sample,
            "return_dict_in_generate": False,
        }
        if sample:
            kwargs.update({"temperature": 0.8, "top_p": 0.9})
        with torch.no_grad():
            output = self.model.generate(**inputs, **kwargs)
        prompt_length = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True).strip()


class M7Runtime:
    def __init__(
        self,
        *,
        tools: M7Tools,
        task: str,
        fallbacks: dict[str, dict[str, Any]],
        output_dir: Path,
        model: AgentModel | None = None,
        max_steps: int = 6,
        max_new_tokens: int = 384,
        sample_ids: list[str] | None = None,
        json_mode: str = "strict",
        dump_prompts: bool = False,
    ) -> None:
        if task not in TASK_TOOL_WHITELIST:
            raise ValueError(f"unsupported task: {task}")
        if json_mode not in {"strict", "lenient"}:
            raise ValueError(f"unsupported json_mode: {json_mode}")
        self.tools = tools
        self.task = task
        self.fallbacks = fallbacks
        self.output_dir = Path(output_dir)
        self.model = model
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.selected_sample_ids = list(sample_ids) if sample_ids else None
        self.json_mode = json_mode
        self.strict_json = json_mode == "strict"
        self.dump_prompts = dump_prompts
        self.whitelist = set(TASK_TOOL_WHITELIST[task])
        self.system_prompt = build_system_prompt(task)
        self.trace_dir = self.output_dir / "trace"
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ helpers
    def _sample_ids(self) -> list[str]:
        if self.selected_sample_ids is not None:
            return list(self.selected_sample_ids)
        if self.task == "task2":
            return [row["sample_id"] for row in self.tools.queries]
        if self.task == "task1_region":
            return [row["sample_id"] for row in self.tools.region_rows]
        return [row["sample_id"] for row in self.tools.segment_rows]

    def _sample_payload(self, sample_id: str) -> dict[str, Any] | None:
        if self.task == "task2":
            row = self.tools.query_by_id.get(sample_id)
            if row is None:
                return None
            return {
                "task": "task2",
                "sample_id": sample_id,
                "text_query": row.get("text_query", ""),
                "top_k": int(row.get("top_k", 10)),
                "regions": [
                    {"region_id": region.get("区域编号"), "boundary": region.get("边界")}
                    for region in row.get("regions", [])
                ],
            }
        row = (
            self.tools.region_by_id.get(sample_id)
            if self.task == "task1_region"
            else self.tools.segment_by_id.get(sample_id)
        )
        if row is None:
            return None
        payload = {
            "task": self.task,
            "sample_id": sample_id,
            "instruction": row.get("text_instruction", ""),
        }
        if self.task == "task1_region":
            payload["regions"] = [
                {"region_id": region.get("区域编号"), "coordinates": region.get("坐标")}
                for region in row.get("regions", [])
            ]
        return payload

    @staticmethod
    def _observation(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False)

    # ------------------------------------------------------------------ fallback
    def _fallback_record(self, sample_id: str) -> dict[str, Any] | None:
        return self.fallbacks.get(sample_id)

    @staticmethod
    def _augment_observation(
        tool_name: str,
        observation: dict[str, Any],
        step: int,
        search_calls: int,
        cached: bool,
    ) -> dict[str, Any]:
        """Add a short strategy hint to the observation shown to the model."""

        if not isinstance(observation, dict):
            return observation
        hint: str | None = None
        if not observation.get("ok"):
            error = observation.get("error") or {}
            hint = error.get("hint") or error.get("message")
        elif tool_name == "search_tracks":
            data = observation.get("data") or {}
            count = int(data.get("candidate_count", 0) or 0)
            if count == 0:
                hint = (
                    "候选为 0：必须修改参数（去掉 speed_range_mps/altitude_range_m/behavior 中至少一项）"
                    "再检索一次；若两次检索都为 0，final 可以给空列表。"
                )
            elif search_calls >= 2:
                hint = (
                    f"这是第 {search_calls} 轮检索，已有 {count} 条候选；请优先 validate_submission 后给 final，"
                    "不要再用相同参数重复检索。"
                )
            elif count > 0 and step >= 2:
                hint = "候选已就绪：可以组织 final，或先 validate_submission 自检。"
        elif tool_name == "validate_submission":
            data = observation.get("data") or {}
            if data.get("ok"):
                hint = "校验通过：下一步直接给 final（结果按你判断的相关度排序）。"
            else:
                hint = "校验失败：" + "; ".join(data.get("errors") or []) + "。请修正后重试。"
        if cached:
            hint = "该参数组合之前已调用过，结果相同；不要重复，改参数或给 final。"
        if hint:
            augmented = dict(observation)
            augmented["_hint"] = hint
            return augmented
        return observation

    @staticmethod
    def _repair_tool_args(tool_name: str, raw_args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Repair the recurring small-model argument mistakes before dispatch."""

        notes: list[str] = []
        args = dict(raw_args or {})
        if "why" in args:
            args.pop("why")
            notes.append("moved:why")
        if tool_name == "search_tracks":
            ir = args.get("ir")
            if isinstance(ir, str):
                try:
                    ir = json.loads(ir)
                    args["ir"] = ir
                    notes.append("parsed:string_ir")
                except json.JSONDecodeError:
                    pass
            if isinstance(ir, dict):
                ir = dict(ir)
                for key in ("limit", "count_only"):
                    if key in ir:
                        args.setdefault(key, ir.pop(key))
                        notes.append(f"moved:ir.{key}")
                args["ir"] = ir
        return args, notes

    # ------------------------------------------------------------------- sample
    def run_sample(self, sample_id: str) -> dict[str, Any]:
        started = time.perf_counter()
        sample = self._sample_payload(sample_id)
        if sample is None:
            record = self._fallback_record(sample_id)
            return {
                "record": record,
                "trace": {"sample_id": sample_id, "source": "missing_sample", "steps": []},
                "tool_calls": [],
            }
        self.tools.set_context(sample, self.task)
        transcript = initial_user_message(sample)
        steps: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        cache: dict[str, dict[str, Any]] = {}
        invalid_actions = 0
        fallback_reason: str | None = None
        final_answer: dict[str, Any] | None = None
        final_validation: dict[str, Any] = {}
        evidence: list[int] = []
        search_calls = 0
        repeated_outputs = 0
        arg_repairs = 0
        strict_ok_count = 0
        strict_fail_count = 0
        lenient_recovered = 0
        last_raw: str | None = None
        last_candidate_count: int | None = None
        auto_truncated_count = 0

        for step in range(1, self.max_steps + 1):
            if self.model is None:
                fallback_reason = "no_model"
                break
            try:
                raw = self.model.generate(self.system_prompt, transcript, max_new_tokens=self.max_new_tokens)
                repeated_output = last_raw is not None and raw.strip() == last_raw.strip()
                if repeated_output:
                    repeated_outputs += 1
                    # Greedy decoding can loop on the same invalid output; break the
                    # tie with one sampled attempt before giving up on this step.
                    raw = self.model.generate(
                        self.system_prompt, transcript, max_new_tokens=self.max_new_tokens, sample=True
                    )
                last_raw = raw
            except Exception as exc:  # noqa: BLE001 - model failure must fall back, not crash the run
                fallback_reason = f"model_error: {type(exc).__name__}: {exc}"
                break

            try:
                strict_json_load(raw)
                strict_ok = True
            except ProtocolError:
                strict_ok = False
            strict_ok_count += int(strict_ok)
            strict_fail_count += int(not strict_ok)
            entry: dict[str, Any] = {
                "step": step,
                "ts": _now_iso(),
                "raw_model_output": raw,
                "strict_json_valid": strict_ok,
                "json_mode": self.json_mode,
                "repeated_output": repeated_output,
                "parse_ok": False,
                "parsed": None,
                "observation": None,
                "tool": None,
            }
            if self.dump_prompts:
                entry["prompt"] = {"system": self.system_prompt, "user": transcript}
            try:
                kind, payload = parse_action(raw, strict=self.strict_json)
                if (not self.strict_json) and (not strict_ok):
                    lenient_recovered += 1
                    entry["lenient_recovered"] = True
            except ProtocolError as exc:
                invalid_actions += 1
                entry["protocol_error"] = str(exc)
                message = str(exc)
                if "Extra data" in message:
                    hint = (
                        "检测到多个 JSON 对象：每次回复只能做一件事——要么只输出 tool 对象（然后等待工具结果），"
                        "要么只输出 final 对象；禁止同一个回复里同时输出 tool 和 final。"
                    )
                else:
                    hint = (
                        '每次回复只能输出一个完整 JSON 对象（{"tool":...} 或 {"final":...}），'
                        "不要输出多个 JSON 对象、不要解释文字、不要 markdown 代码块；"
                        "必须以 { 开头、以 } 结尾，且能被 json.loads(原始文本) 直接解析。"
                    )
                error_obs = {
                    "ok": False,
                    "error": {
                        "code": "PROTOCOL",
                        "message": message,
                        "hint": hint,
                    },
                }
                entry["observation"] = error_obs
                steps.append(entry)
                transcript = append_step(transcript, step, raw, self._observation(error_obs))
                continue

            entry["parse_ok"] = True
            entry["parsed"] = (
                {"type": "call", **payload}
                if kind == "call"
                else {"type": "final", "answer": payload.get("answer"), "evidence": payload.get("evidence", [])}
            )

            if kind == "call":
                tool_name = payload["tool"]
                raw_args, arg_notes = self._repair_tool_args(tool_name, payload.get("args") or {})
                if arg_notes:
                    arg_repairs += len(arg_notes)
                args = raw_args
                entry["why"] = payload.get("why")
                entry["arg_repairs"] = arg_notes
                if tool_name not in self.whitelist:
                    invalid_actions += 1
                    error_obs = {
                        "ok": False,
                        "error": {
                            "code": "UNKNOWN_TOOL",
                            "message": f"tool {tool_name!r} is not allowed for {self.task}",
                            "hint": f"allowed tools: {sorted(self.whitelist)}",
                        },
                    }
                    entry["observation"] = error_obs
                    steps.append(entry)
                    transcript = append_step(transcript, step, raw, self._observation(error_obs))
                    continue

                args_hash = _sha1({"tool": tool_name, "args": args})
                cached = args_hash in cache
                elapsed = 0.0
                if cached:
                    observation = cache[args_hash]
                else:
                    tool_started = time.perf_counter()
                    observation = self.tools.call(tool_name, args)
                    elapsed = round((time.perf_counter() - tool_started) * 1000.0, 2)
                    cache[args_hash] = observation
                if tool_name == "search_tracks":
                    search_calls += 1
                    if observation.get("ok"):
                        data = observation.get("data") or {}
                        last_candidate_count = int(data.get("candidate_count", 0) or 0)
                ok = bool(observation.get("ok", False))
                error_code = None if ok else (observation.get("error") or {}).get("code")
                shown = self._augment_observation(tool_name, observation, step, search_calls, cached)
                entry["tool"] = {
                    "name": tool_name,
                    "args": args,
                    "args_hash": args_hash,
                    "ok": ok,
                    "error_code": error_code,
                    "elapsed_ms": elapsed,
                    "result_bytes": len(self._observation(observation).encode("utf-8")),
                    "truncated": bool(isinstance(observation.get("data"), dict) and observation["data"].get("_truncated")),
                    "cached": cached,
                }
                entry["observation"] = shown
                steps.append(entry)
                tool_calls.append(
                    {
                        "sample_id": sample_id,
                        "task": self.task,
                        "step": step,
                        "tool": tool_name,
                        "args_hash": args_hash,
                        "args": args,
                        "ok": ok,
                        "error_code": error_code,
                        "elapsed_ms": elapsed,
                        "result_bytes": entry["tool"]["result_bytes"],
                        "truncated": entry["tool"]["truncated"],
                        "cached": cached,
                        "repeated": cached,
                        "cited_in_final": False,
                    }
                )
                transcript = append_step(transcript, step, raw, self._observation(shown))
                continue

            # final answer
            answer = payload.get("answer") or {}
            evidence = payload.get("evidence", [])
            auto_truncated = False
            if self.task == "task2" and isinstance(answer.get("results"), list):
                top_k = int(sample.get("top_k", 10))
                results = answer["results"]
                if len(results) > top_k:
                    answer = {"results": results[:top_k]}
                    auto_truncated = True
                    auto_truncated_count += 1
                    entry["auto_truncated"] = True
            validation = self._validate_final(sample, answer, search_calls, last_candidate_count)
            entry["final_validation"] = validation
            if validation.get("ok"):
                final_answer = answer
                final_validation = validation
                steps.append(entry)
                break
            invalid_actions += 1
            error_obs = {
                "ok": False,
                "error": {
                    "code": "INVALID_FINAL",
                    "message": "; ".join(validation.get("errors", [])),
                    "hint": "修正后重新给 final，或先用工具核对事实",
                },
            }
            entry["observation"] = error_obs
            steps.append(entry)
            transcript = append_step(transcript, step, raw, self._observation(error_obs))
        else:
            fallback_reason = "max_steps_exhausted"

        if final_answer is None and fallback_reason is None:
            fallback_reason = "no_valid_final"

        if final_answer is not None:
            record = self._to_record(sample_id, final_answer)
            source = "agent"
        else:
            record = self._fallback_record(sample_id)
            source = "fallback"

        # mark evidence-cited tool calls
        cited_steps = set()
        for step_index in evidence:
            if isinstance(step_index, int):
                cited_steps.add(step_index)
        for call in tool_calls:
            call["cited_in_final"] = call["step"] in cited_steps

        trace = {
            "schema": f"{SCHEMA_VERSION}.trace/1",
            "sample_id": sample_id,
            "task": self.task,
            "json_mode": self.json_mode,
            "started_at": _now_iso(),
            "finished_at": _now_iso(),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "steps": steps,
            "invalid_actions": invalid_actions,
            "search_calls": search_calls,
            "final": {
                "answer": final_answer,
                "empty_result": bool(
                    final_answer is not None
                    and self.task == "task2"
                    and (final_answer.get("results") or []) == []
                ),
                "evidence": evidence,
                "source": source,
                "fallback_reason": fallback_reason,
                "validation": final_validation,
                "record": record,
            },
            "totals": {
                "steps": len(steps),
                "tool_calls": len(tool_calls),
                "invalid_actions": invalid_actions,
                "repeated_outputs": repeated_outputs,
                "arg_repairs": arg_repairs,
                "strict_json_ok": strict_ok_count,
                "strict_json_fail": strict_fail_count,
                "lenient_recovered": lenient_recovered,
                "auto_truncated_finals": auto_truncated_count,
            },
        }
        return {"record": record, "trace": trace, "tool_calls": tool_calls}

    # ---------------------------------------------------------------- validation
    def _validate_final(
        self,
        sample: dict[str, Any],
        answer: dict[str, Any],
        search_calls: int,
        last_candidate_count: int | None,
    ) -> dict[str, Any]:
        sample_id = sample["sample_id"]
        if self.task == "task2":
            if search_calls <= 0:
                return {"ok": False, "errors": ["至少先调用一次 search_tracks 再给 final"]}
            results = answer.get("results")
            if not isinstance(results, list):
                return {"ok": False, "errors": ["final 必须是 {\"results\": [...]} "]}
            if results == [] and last_candidate_count == 0:
                return {"ok": True, "errors": [], "checks": {"empty_result_allowed": True}}
            envelope = self.tools.validate_submission(sample_id, {"results": results})
            if not envelope.get("ok"):
                return {"ok": False, "errors": [str(envelope.get("error"))]}
            data = envelope["data"]
            return {"ok": bool(data.get("ok")), "errors": data.get("errors", []), "checks": data.get("checks")}
        choice = answer.get("choice")
        envelope = self.tools.validate_submission(sample_id, {"choice": choice})
        if not envelope.get("ok"):
            return {"ok": False, "errors": [str(envelope.get("error"))]}
        data = envelope["data"]
        return {"ok": bool(data.get("ok")), "errors": data.get("errors", []), "checks": data.get("checks")}

    def _to_record(self, sample_id: str, answer: dict[str, Any]) -> dict[str, Any]:
        if self.task == "task2":
            results = answer.get("results") or []
            return {
                "样本编号": sample_id,
                "结果": [{"序号": index, "航迹编号": trajectory_id} for index, trajectory_id in enumerate(results, start=1)],
            }
        return {"样本编号": sample_id, "选项": str(answer.get("choice", "")).strip().upper()}

    # --------------------------------------------------------------------- run
    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        sample_ids = self._sample_ids()
        records: list[dict[str, Any]] = []
        all_tool_calls: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []
        for index, sample_id in enumerate(sample_ids, start=1):
            result = self.run_sample(sample_id)
            traces.append(result["trace"])
            all_tool_calls.extend(result["tool_calls"])
            if result["record"] is not None:
                records.append(result["record"])
            (self.trace_dir / f"{sample_id}.json").write_text(
                json.dumps(result["trace"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            if index % 10 == 0 or index == len(sample_ids):
                print(f"[m7] {index}/{len(sample_ids)} samples done", flush=True)

        submission = self.output_dir / "traj_result.jsonl"
        with submission.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        with (self.output_dir / "tool_calls.jsonl").open("w", encoding="utf-8") as handle:
            for call in all_tool_calls:
                handle.write(json.dumps(call, ensure_ascii=False) + "\n")
        report = self._build_report(records, traces, all_tool_calls, time.perf_counter() - started)
        (self.output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return {
            "records": records,
            "traces": traces,
            "tool_calls": all_tool_calls,
            "report": report,
            "submission": str(submission),
        }

    def _build_report(
        self,
        records: list[dict[str, Any]],
        traces: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]],
        runtime_seconds: float,
    ) -> dict[str, Any]:
        from collections import Counter

        invalid_final = sum(
            1 for trace in traces if trace.get("final", {}).get("source") == "fallback"
        )
        step_counter = Counter()
        tool_counter = Counter()
        cited_counter = Counter()
        error_counter = Counter()
        expected_ids = set(self._sample_ids())
        actual_ids = {record["样本编号"] for record in records}
        for trace in traces:
            for step in trace.get("steps", []):
                parsed = step.get("parsed") or {}
                if parsed.get("type") == "call":
                    tool_counter[parsed.get("tool")] += 1
                    step_counter["tool_calls"] += 1
                elif parsed.get("type") == "final":
                    step_counter["final_attempts"] += 1
                if step.get("protocol_error"):
                    step_counter["protocol_errors"] += 1
        for call in tool_calls:
            if not call.get("ok"):
                error_counter[call.get("error_code") or "UNKNOWN"] += 1
            if call.get("cited_in_final"):
                cited_counter[call.get("tool")] += 1
        latencies = [call["elapsed_ms"] for call in tool_calls if not call.get("cached")]
        latencies.sort()
        def pct(values: list[float], q: float) -> float:
            if not values:
                return 0.0
            index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
            return round(values[index], 2)

        steps_per_sample = [trace.get("totals", {}).get("steps", 0) for trace in traces]
        return {
            "schema_version": SCHEMA_VERSION,
            "task": self.task,
            "created_at": _now_iso(),
            "runtime_seconds": round(runtime_seconds, 2),
            "samples": {
                "expected": len(expected_ids),
                "records": len(records),
                "missing": sorted(expected_ids - actual_ids),
                "fallback": invalid_final,
                "agent": len(records) - invalid_final,
                "empty_results": sum(1 for trace in traces if trace.get("final", {}).get("empty_result")),
                "auto_truncated_finals": sum(
                    trace.get("totals", {}).get("auto_truncated_finals", 0) for trace in traces
                ),
            },
            "budget": {"max_steps": self.max_steps, "max_new_tokens": self.max_new_tokens},
            "json_mode": self.json_mode,
            "steps": {
                "total": sum(steps_per_sample),
                "avg_per_sample": round(sum(steps_per_sample) / max(1, len(steps_per_sample)), 2),
                "max_per_sample": max(steps_per_sample) if steps_per_sample else 0,
                "protocol_errors": step_counter.get("protocol_errors", 0),
                "final_attempts": step_counter.get("final_attempts", 0),
                "repeated_outputs": sum(trace.get("totals", {}).get("repeated_outputs", 0) for trace in traces),
                "arg_repairs": sum(trace.get("totals", {}).get("arg_repairs", 0) for trace in traces),
                "strict_json_ok": sum(trace.get("totals", {}).get("strict_json_ok", 0) for trace in traces),
                "strict_json_fail": sum(trace.get("totals", {}).get("strict_json_fail", 0) for trace in traces),
                "lenient_recovered": sum(trace.get("totals", {}).get("lenient_recovered", 0) for trace in traces),
            },
            "tools": {
                "calls": dict(sorted(tool_counter.items())),
                "calls_per_sample": round(len(tool_calls) / max(1, len(traces)), 2),
                "errors": dict(sorted(error_counter.items())),
                "cached_repeated": sum(1 for call in tool_calls if call.get("cached")),
                "truncated": sum(1 for call in tool_calls if call.get("truncated")),
                "cited_calls": sum(1 for call in tool_calls if call.get("cited_in_final")),
                "latency_ms_p50": pct(latencies, 0.5),
                "latency_ms_p95": pct(latencies, 0.95),
                "unused_whitelisted": sorted(self.whitelist - set(tool_counter)),
            },
            "submission_sha256": self._sha256(self.output_dir / "traj_result.jsonl"),
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
