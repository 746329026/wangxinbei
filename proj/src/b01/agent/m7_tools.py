"""M7 deterministic tool registry (read-only facts for the agent).

Every tool returns a JSON envelope ``{"ok": true, "data": ...}`` or
``{"ok": false, "error": {"code", "message", "hint"}}``. Results larger than
``MAX_RESULT_BYTES`` are progressively trimmed and flagged with ``_truncated``.

The tools never call a model and never write files; they reuse the M1-M5
deterministic implementations where possible.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from datetime import datetime, timezone
from typing import Any, Iterable

from b01.geometry.regions import _polygon, region_entry_sequence, region_event_summary
from b01.io.jsonl import read_jsonl
from b01.retrieval.baseline import (
    BEHAVIOR_TERMS,
    behavior_score,
    hard_filter,
    parse_query,
    trajectory_features,
)
from b01.task1.baseline import parse_options

MAX_RESULT_BYTES = 8192

BEHAVIOR_NAMES = tuple(canonical for canonical, _ in BEHAVIOR_TERMS)

# Chinese terms from the canonical vocabulary + common English synonyms the
# small model produces (acceleration / accelerate / climbing / ...).
_BEHAVIOR_ALIASES: dict[str, str] = {}
for _canonical, _terms in BEHAVIOR_TERMS:
    _BEHAVIOR_ALIASES[_canonical] = _canonical
    for _term in _terms:
        _BEHAVIOR_ALIASES.setdefault(_term.strip().lower(), _canonical)
for _alias, _canonical in {
    "acceleration": "step_accel",
    "accelerate": "step_accel",
    "accel": "step_accel",
    "deceleration": "step_decel",
    "decelerate": "step_decel",
    "decel": "step_decel",
    "climbing": "climb",
    "descending": "descent",
    "descent": "descent",
    "circling": "orbit",
    "circling_orbit": "orbit",
    "turn": "orbit",
    "turnaround": "orbit",
    "speed_stable": "speed_hold",
    "steady": "cruise",
    "hover": "speed_hold",
    "low": "low_speed",
}.items():
    _BEHAVIOR_ALIASES.setdefault(_alias, _canonical)

TASK_TOOL_WHITELIST: dict[str, tuple[str, ...]] = {
    "task1_segment": (
        "get_sample",
        "trajectory_digest",
        "candidate_boundaries",
        "stage_facts",
    ),
    "task1_region": (
        "get_sample",
        "trajectory_digest",
        "region_events",
    ),
    "task2": (
        "get_sample",
        "search_tracks",
        "track_digest",
        "region_hit",
        "behavior_check",
        "validate_submission",
    ),
}

TASK1_SEGMENT_TOOLS = TASK_TOOL_WHITELIST["task1_segment"]
TASK1_REGION_TOOLS = TASK_TOOL_WHITELIST["task1_region"]
TASK2_TOOLS = TASK_TOOL_WHITELIST["task2"]


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def _halve_lists(value: Any) -> tuple[Any, bool]:
    """Return a copy with all long lists halved; report whether anything changed."""

    changed = False
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            new_item, did = _halve_lists(item)
            result[key] = new_item
            changed = changed or did
        return result, changed
    if isinstance(value, list):
        if len(value) > 1:
            half = max(1, len(value) // 2)
            result = []
            for item in value[:half]:
                new_item, _ = _halve_lists(item)
                result.append(new_item)
            return result, True
        return value, False
    return value, False


def _cap(payload: Any, limit: int | None = None) -> Any:
    limit = MAX_RESULT_BYTES if limit is None else limit
    if _json_size(payload) <= limit:
        return payload
    data = json.loads(json.dumps(payload, ensure_ascii=False))
    guard = 0
    while _json_size(data) > limit and guard < 20:
        guard += 1
        shrunk, changed = _halve_lists(data)
        if not changed:
            break
        data = shrunk
    if isinstance(data, dict):
        data["_truncated"] = True
    return data


def _ok(data: Any) -> dict[str, Any]:
    return {"ok": True, "data": _cap(data)}


def _err(code: str, message: str, hint: str = "") -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message, "hint": hint}}


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    middle = (len(values) - 1) / 2.0
    denominator = sum((index - middle) ** 2 for index in range(len(values))) or 1.0
    return sum((index - middle) * (value - _mean(values)) for index, value in enumerate(values)) / denominator


def _turn_delta(left: float, right: float) -> float:
    return (right - left + 180.0) % 360.0 - 180.0


def _segment_stats(points: list[dict[str, Any]]) -> dict[str, Any]:
    altitudes = [float(point["altitude_m"]) for point in points]
    speeds = [float(point["speed_mps"]) for point in points]
    bearings = [
        float(point["bearing_derived_deg"])
        for point in points
        if point.get("bearing_derived_deg") is not None
    ]
    turns = [_turn_delta(left, right) for left, right in zip(bearings, bearings[1:])]
    path_distance = sum(float(point.get("distance_m") or 0.0) for point in points)
    if len(points) >= 2:
        lon_delta = float(points[-1]["longitude"]) - float(points[0]["longitude"])
        lat_delta = float(points[-1]["latitude"]) - float(points[0]["latitude"])
        displacement = math.hypot(lon_delta, lat_delta) * 111_000.0
    else:
        displacement = 0.0
    directness = displacement / max(path_distance, 1.0)
    circularity = path_distance / max(displacement, 1.0)
    altitude_deltas = [right - left for left, right in zip(altitudes, altitudes[1:])]
    speed_deltas = [right - left for left, right in zip(speeds, speeds[1:])]

    def trend(first: float, last: float, value_range: float, sign_changes: int, base_threshold: float) -> str:
        threshold = max(base_threshold, value_range * 0.12)
        if sign_changes >= 3 and value_range > threshold:
            return "periodic"
        delta = last - first
        if delta > threshold:
            return "up"
        if delta < -threshold:
            return "down"
        return "stable"

    altitude_trend = trend(
        altitudes[0] if altitudes else 0.0,
        altitudes[-1] if altitudes else 0.0,
        max(altitudes, default=0.0) - min(altitudes, default=0.0),
        sum(
            1
            for left, right in zip(altitude_deltas, altitude_deltas[1:])
            if left * right < 0 and abs(left) > 1.0 and abs(right) > 1.0
        ),
        50.0,
    )
    speed_trend = trend(
        speeds[0] if speeds else 0.0,
        speeds[-1] if speeds else 0.0,
        max(speeds, default=0.0) - min(speeds, default=0.0),
        sum(
            1
            for left, right in zip(speed_deltas, speed_deltas[1:])
            if left * right < 0 and abs(left) > 0.2 and abs(right) > 0.2
        ),
        2.0,
    )
    turn_sign_changes = sum(
        1 for left, right in zip(turns, turns[1:]) if left * right < 0 and abs(left) > 2 and abs(right) > 2
    )
    if path_distance <= 1.0:
        pattern = "stationary"
    elif turn_sign_changes >= 3 and directness < 0.45:
        pattern = "orbit"
    elif directness >= 0.72:
        pattern = "straight"
    elif directness < 0.35:
        pattern = "runway"
    else:
        pattern = "cruise"
    return {
        "point_count": len(points),
        "altitude": {
            "first_m": round(altitudes[0], 2) if altitudes else 0.0,
            "last_m": round(altitudes[-1], 2) if altitudes else 0.0,
            "mean_m": round(_mean(altitudes), 2),
            "min_m": round(min(altitudes, default=0.0), 2),
            "max_m": round(max(altitudes, default=0.0), 2),
            "slope": round(_slope(altitudes), 4),
            "range_m": round(max(altitudes, default=0.0) - min(altitudes, default=0.0), 2),
            "sign_changes": sum(
                1
                for left, right in zip(altitude_deltas, altitude_deltas[1:])
                if left * right < 0 and abs(left) > 1.0 and abs(right) > 1.0
            ),
            "trend": altitude_trend,
        },
        "speed": {
            "first_mps": round(speeds[0], 2) if speeds else 0.0,
            "last_mps": round(speeds[-1], 2) if speeds else 0.0,
            "mean_mps": round(_mean(speeds), 2),
            "min_mps": round(min(speeds, default=0.0), 2),
            "max_mps": round(max(speeds, default=0.0), 2),
            "slope": round(_slope(speeds), 4),
            "range_mps": round(max(speeds, default=0.0) - min(speeds, default=0.0), 2),
            "sign_changes": sum(
                1
                for left, right in zip(speed_deltas, speed_deltas[1:])
                if left * right < 0 and abs(left) > 0.2 and abs(right) > 0.2
            ),
            "trend": speed_trend,
        },
        "spatial": {
            "path_distance_m": round(path_distance, 2),
            "displacement_m": round(displacement, 2),
            "directness": round(directness, 4),
            "turn_abs_deg": round(sum(abs(value) for value in turns), 2),
            "turn_sign_changes": turn_sign_changes,
            "circularity": round(circularity, 4),
            "pattern": pattern,
        },
    }


class M7Tools:
    """Deterministic tool registry backed by the M1/M3 caches."""

    def __init__(self, cache_dir: str, m3_dir: str | None = None) -> None:
        from pathlib import Path

        cache = Path(cache_dir)
        self.segment_rows = list(read_jsonl(cache / "task1_segment_normalized.jsonl"))
        self.region_rows = list(read_jsonl(cache / "task1_region_normalized.jsonl"))
        self.queries = list(read_jsonl(cache / "task2_queries_normalized.jsonl"))
        self.facts = list(read_jsonl(cache / "track_facts.jsonl"))
        self.tracks = list(read_jsonl(cache / "tracks_normalized.jsonl"))

        self.segment_by_id = {row["sample_id"]: row for row in self.segment_rows}
        self.region_by_id = {row["sample_id"]: row for row in self.region_rows}
        self.query_by_id = {row["sample_id"]: row for row in self.queries}
        self.facts_by_id = {row["trajectory_id"]: row for row in self.facts}
        self.tracks_by_id = {row["trajectory_id"]: row for row in self.tracks}
        self.library_ids = set(self.tracks_by_id)

        self.feature_by_id = {
            trajectory_id: trajectory_features(track)
            for trajectory_id, track in self.tracks_by_id.items()
        }
        self.query_region_by_id: dict[str, dict[str, Any]] = {}
        for query in self.queries:
            for region in query.get("regions", []):
                self.query_region_by_id[str(region["区域编号"])] = region
        self.region_hits_by_id = self._load_region_hits(Path(m3_dir) if m3_dir else None)

        self.context: dict[str, Any] = {}

    def _load_region_hits(self, m3_dir: Any) -> dict[str, dict[str, bool]]:
        if m3_dir is not None:
            cache_file = m3_dir / "region_hits.jsonl"
            if cache_file.exists():
                rows = list(read_jsonl(cache_file))
                if rows:
                    return {row["trajectory_id"]: row["region_hits"] for row in rows}
        return {
            trajectory_id: {
                region_id: self._region_hit(track, region)
                for region_id, region in self.query_region_by_id.items()
            }
            for trajectory_id, track in self.tracks_by_id.items()
        }

    # ------------------------------------------------------------------ context
    def set_context(self, sample: dict[str, Any], task: str) -> None:
        self.context = {
            "task": task,
            "sample": sample,
            "sample_id": sample.get("sample_id"),
            "allowed_ids": set(),
            "search_calls": 0,
        }

    # ------------------------------------------------------------------- helper
    def _task1_trajectory(self, sample_id: str) -> tuple[str, dict[str, Any]]:
        if sample_id in self.segment_by_id:
            row = self.segment_by_id[sample_id]
            return "task1_segment", row
        if sample_id in self.region_by_id:
            row = self.region_by_id[sample_id]
            return "task1_region", row
        raise KeyError(sample_id)

    def _region_hit(self, trajectory: dict[str, Any], region: dict[str, Any]) -> bool:
        geometry = _polygon(region)
        min_x, min_y, max_x, max_y = geometry.bounds
        bbox = trajectory["summary"]["bbox"]
        if bbox["max_lon"] < min_x or bbox["min_lon"] > max_x or bbox["max_lat"] < min_y or bbox["min_lat"] > max_y:
            return False
        summary = region_event_summary(trajectory, region)
        return bool(
            summary["start_inside"]
            or summary["end_inside"]
            or summary["entry_events"]
            or summary["exit_events"]
        )

    # --------------------------------------------------------------------- tools
    def get_sample(self, sample_id: str) -> dict[str, Any]:
        if sample_id in self.segment_by_id or sample_id in self.region_by_id:
            task, row = self._task1_trajectory(sample_id)
            trajectory = row["trajectory"]
            summary = trajectory["summary"]
            instruction = row.get("text_instruction", "")
            try:
                options = parse_options(instruction)
            except ValueError:
                options = {}
            data = {
                "task": task,
                "sample_id": sample_id,
                "instruction": instruction,
                "options": options,
                "trajectory": {
                    "trajectory_id": trajectory["trajectory_id"],
                    "start_time_utc": trajectory.get("start_time_utc"),
                    "point_count": summary.get("point_count", len(trajectory.get("points", []))),
                    "duration_seconds": summary.get("duration_seconds", 0.0),
                },
            }
            if task == "task1_region":
                data["regions"] = [
                    {
                        "region_id": region.get("区域编号"),
                        "type": region.get("类型"),
                        "coordinates": region.get("坐标"),
                    }
                    for region in row.get("regions", [])
                ]
            return _ok(data)

        if sample_id in self.query_by_id:
            row = self.query_by_id[sample_id]
            return _ok(
                {
                    "task": "task2",
                    "sample_id": sample_id,
                    "query": {
                        "text": row.get("text_query", ""),
                        "top_k": int(row.get("top_k", 10)),
                        "regions": [
                            {
                                "region_id": region.get("区域编号"),
                                "type": region.get("类型"),
                                "boundary": region.get("边界"),
                            }
                            for region in row.get("regions", [])
                        ],
                    },
                }
            )
        return _err("NOT_FOUND", f"unknown sample_id: {sample_id}", "use the task sample ids only")

    def trajectory_digest(self, sample_id: str) -> dict[str, Any]:
        try:
            _, row = self._task1_trajectory(sample_id)
        except KeyError:
            return _err("NOT_FOUND", f"unknown task1 sample: {sample_id}")
        trajectory = row["trajectory"]
        summary = trajectory["summary"]
        return _ok(
            {
                "sample_id": sample_id,
                "trajectory_id": trajectory["trajectory_id"],
                "start_time_utc": trajectory.get("start_time_utc"),
                "end_time_utc": summary.get("end_time_utc"),
                "point_count": summary.get("point_count", 0),
                "duration_seconds": summary.get("duration_seconds", 0.0),
                "bbox": summary.get("bbox"),
                "altitude_m": summary.get("altitude_m"),
                "speed_mps": summary.get("speed_mps"),
                "path_distance_m": summary.get("path_distance_m"),
                "directness": summary.get("directness"),
                "quality": summary.get("quality"),
            }
        )

    def candidate_boundaries(self, sample_id: str, bins: int = 20) -> dict[str, Any]:
        try:
            _, row = self._task1_trajectory(sample_id)
        except KeyError:
            return _err("NOT_FOUND", f"unknown task1 sample: {sample_id}")
        points = row["trajectory"].get("points", [])
        if len(points) < bins * 2:
            return _ok({"sample_id": sample_id, "candidates": [], "note": "too few points"})
        bins = max(6, min(int(bins), 40))
        duration = float(points[-1]["relative_seconds"]) - float(points[0]["relative_seconds"])
        if duration <= 0:
            return _ok({"sample_id": sample_id, "candidates": []})
        bin_alts: list[float] = []
        bin_speeds: list[float] = []
        edges: list[float] = []
        for index in range(bins):
            left = float(points[0]["relative_seconds"]) + duration * index / bins
            right = float(points[0]["relative_seconds"]) + duration * (index + 1) / bins
            values = [point for point in points if left <= float(point["relative_seconds"]) < right]
            edges.append(right)
            bin_alts.append(_mean(float(point["altitude_m"]) for point in values))
            bin_speeds.append(_mean(float(point["speed_mps"]) for point in values))
        alt_range = max(bin_alts) - min(bin_alts) or 1.0
        speed_range = max(bin_speeds) - min(bin_speeds) or 1.0
        candidates: list[dict[str, Any]] = []
        for index in range(1, bins - 1):
            alt_jump = abs(bin_alts[index + 1] - bin_alts[index])
            speed_jump = abs(bin_speeds[index + 1] - bin_speeds[index])
            score = alt_jump / alt_range + speed_jump / speed_range
            prev_score = abs(bin_alts[index] - bin_alts[index - 1]) / alt_range + abs(
                bin_speeds[index] - bin_speeds[index - 1]
            ) / speed_range
            next_score = (
                abs(bin_alts[min(bins - 1, index + 2)] - bin_alts[index + 1]) / alt_range
                + abs(bin_speeds[min(bins - 1, index + 2)] - bin_speeds[index + 1]) / speed_range
            )
            reasons = []
            if alt_jump / alt_range > 0.15:
                reasons.append("altitude_change")
            if speed_jump / speed_range > 0.15:
                reasons.append("speed_change")
            if not reasons:
                continue
            if score < prev_score or score < next_score:
                continue
            candidates.append(
                {
                    "t": round(edges[index], 2),
                    "score": round(score, 4),
                    "reasons": reasons,
                }
            )
        candidates.sort(key=lambda item: item["t"])
        return _ok({"sample_id": sample_id, "n_bins": bins, "candidates": candidates[:12]})

    def stage_facts(self, sample_id: str, cuts: list[float] | None = None, stage_count: int = 4) -> dict[str, Any]:
        try:
            _, row = self._task1_trajectory(sample_id)
        except KeyError:
            return _err("NOT_FOUND", f"unknown task1 sample: {sample_id}")
        points = row["trajectory"].get("points", [])
        if len(points) < 2:
            return _err("EMPTY", "sample has fewer than 2 points")
        start = float(points[0]["relative_seconds"])
        end = float(points[-1]["relative_seconds"])
        if cuts is None:
            count = max(2, min(int(stage_count), 12))
            cuts = [start + (end - start) * index / count for index in range(count + 1)]
        cuts = [float(value) for value in cuts]
        if len(cuts) < 2 or any(right <= left for left, right in zip(cuts, cuts[1:])):
            return _err("BAD_ARGS", "cuts must be strictly increasing with at least two values")
        stages: list[dict[str, Any]] = []
        for index, (left, right) in enumerate(zip(cuts, cuts[1:]), start=1):
            window = [
                point
                for point in points
                if left <= float(point["relative_seconds"]) <= right
                and (index == len(cuts) - 1 or float(point["relative_seconds"]) < right)
            ]
            if not window:
                continue
            stats = _segment_stats(window)
            stages.append(
                {
                    "index": index,
                    "t0": round(left, 2),
                    "t1": round(right, 2),
                    "duration_seconds": round(right - left, 2),
                    **stats,
                }
            )
        return _ok({"sample_id": sample_id, "cuts": [round(value, 2) for value in cuts], "stages": stages})

    def _normalize_task1_region_id(self, sample_id: str, value: Any) -> str:
        allowed = [
            str(region.get("区域编号", region.get("region_id", "")))
            for region in self.region_by_id[sample_id].get("regions", [])
        ]
        text = str(value).strip()
        short = text.replace("区域", "").strip().lower()
        for region_id in allowed:
            if text == region_id:
                return region_id
            if str(region_id).replace("区域", "").strip().lower() == short:
                return region_id
        raise ValueError(f"未知区域 {value!r}，可选: {allowed}")

    def region_events(
        self,
        sample_id: str,
        region_id: Any | None = None,
        region_ids: list[Any] | None = None,
    ) -> dict[str, Any]:
        if sample_id not in self.region_by_id:
            return _err("NOT_FOUND", f"unknown task1 region sample: {sample_id}", "only region_XXXXXX samples")
        row = self.region_by_id[sample_id]
        trajectory = row["trajectory"]
        all_regions = row.get("regions", [])
        selected: Any = region_ids if region_ids is not None else region_id
        if selected is None:
            regions = all_regions
        else:
            wanted = selected if isinstance(selected, list) else [selected]
            try:
                wanted_ids = [self._normalize_task1_region_id(sample_id, value) for value in wanted]
            except ValueError as exc:
                return _err("BAD_ARGS", str(exc))
            regions = all_regions
            wanted_set = set(wanted_ids)

            def region_name(region: dict[str, Any]) -> str:
                return str(region.get("区域编号", region.get("region_id", "")))

            regions = [region for region in regions if region_name(region) in wanted_set]
            if not regions:
                return _err("NOT_FOUND", f"unknown region_id(s): {wanted}")
        compact: list[dict[str, Any]] = []
        for region in regions:
            summary = region_event_summary(trajectory, region)
            compact.append(
                {
                    "region_id": summary["region_id"],
                    "visited": bool(
                        summary["start_inside"]
                        or summary["end_inside"]
                        or summary["entry_events"]
                        or summary["exit_events"]
                    ),
                    "start_inside": summary["start_inside"],
                    "end_inside": summary["end_inside"],
                    "entry_utc": [event.get("time_utc") for event in summary["entry_events"]],
                    "exit_utc": [event.get("time_utc") for event in summary["exit_events"]],
                    "dwell": [
                        {
                            "start_utc": interval.get("start_utc"),
                            "end_utc": interval.get("end_utc"),
                            "open_end": interval.get("open_end", False),
                        }
                        for interval in summary["dwell_intervals"]
                    ],
                    "outside_to_inside_count": summary["outside_to_inside_count"],
                    "total_dwell_seconds": round(summary["total_dwell_seconds"], 3),
                }
            )
        sequence = region_entry_sequence(trajectory, regions)
        return _ok(
            {
                "sample_id": sample_id,
                "start_time_utc": trajectory.get("start_time_utc"),
                "regions": compact,
                "entry_sequence": [
                    {
                        "region_id": item.get("region_id"),
                        "time_utc": item.get("time_utc"),
                        "sequence_index": item.get("sequence_index"),
                    }
                    for item in sequence
                ],
            }
        )

    _RANGE_RE = re.compile(r"-?\d+(?:\.\d+)?")

    @staticmethod
    def _to_number(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _parse_range(self, value: Any, name: str) -> tuple[float, float] | None:
        if value is None or value == "":
            return None
        numbers: list[float | None]
        if isinstance(value, (list, tuple)):
            numbers = [self._to_number(item) for item in value]
        elif isinstance(value, dict):
            numbers = [self._to_number(value.get("min")), self._to_number(value.get("max"))]
        elif isinstance(value, str):
            numbers = [self._to_number(item) for item in self._RANGE_RE.findall(value)]
        else:
            raise ValueError(f"{name} 必须是数组 [min, max]，例如 [4000, 7000]")
        if len(numbers) != 2 or numbers[0] is None or numbers[1] is None:
            raise ValueError(f"{name} 必须正好包含两个数字，例如 [4000, 7000]")
        low, high = float(numbers[0]), float(numbers[1])
        if low > high:
            raise ValueError(f"{name} 的下限必须 <= 上限")
        return (low, high)

    def _canonical_region_id(self, value: Any) -> str:
        text = str(value).strip()
        if text in self.query_region_by_id:
            return text
        short = text.replace("区域", "").strip().lower()
        for region_id in self.query_region_by_id:
            if str(region_id).replace("区域", "").strip().lower() == short:
                return region_id
        raise ValueError(f"未知区域 {value!r}，可选: {sorted(self.query_region_by_id)}")

    def _parse_time(self, value: Any, name: str) -> str | None:
        if value is None or value == "":
            return None
        parsed = _parse_iso(value)
        if parsed is None:
            raise ValueError(f"{name} 必须是 ISO 8601 UTC 时间，例如 2025-01-01T00:00:00Z")
        return _iso(parsed)

    def search_tracks(
        self,
        ir: dict[str, Any] | None = None,
        limit: int = 20,
        count_only: bool = False,
    ) -> dict[str, Any]:
        sample = self.context.get("sample", {})
        if not sample or self.context.get("task") != "task2":
            return _err("BAD_CONTEXT", "search_tracks is only available for task2 samples")
        if not isinstance(ir, dict):
            return _err(
                "BAD_ARGS",
                "ir 必须是一个 JSON 对象",
                '示例: {"tool":"search_tracks","args":{"ir":{"positive_regions":["区域 A"]},"limit":20}}',
            )
        allowed = {
            "time_start_utc",
            "time_end_utc",
            "altitude_range_m",
            "speed_range_mps",
            "positive_regions",
            "negative_regions",
            "behavior_sequence",
            "flags",
            "thresholds",
        }
        unknown = set(ir) - allowed
        if unknown:
            return _err("BAD_ARGS", f"ir 里有不支持的字段: {sorted(unknown)}", f"允许的字段: {sorted(allowed)}")

        try:
            limits = {"low_speed_max_mps": 80.0, "very_high_altitude_min_m": 8000.0}
            raw_thresholds = ir.get("thresholds") or {}
            if not isinstance(raw_thresholds, dict):
                raise ValueError("thresholds 必须是对象")
            for key in limits:
                if key in raw_thresholds:
                    number = self._to_number(raw_thresholds[key])
                    if number is None:
                        raise ValueError(f"thresholds.{key} 必须是数字")
                    limits[key] = number

            parsed = parse_query(
                {
                    "sample_id": sample.get("sample_id"),
                    "text_query": sample.get("text_query", ""),
                    "top_k": int(sample.get("top_k", 10)),
                    "regions": sample.get("regions", []),
                }
            )
            parsed["top_k"] = int(sample.get("top_k", 10))
            parsed["time_start_utc"] = self._parse_time(ir.get("time_start_utc"), "time_start_utc")
            parsed["time_end_utc"] = self._parse_time(ir.get("time_end_utc"), "time_end_utc")
            parsed["altitude_range_m"] = self._parse_range(ir.get("altitude_range_m"), "altitude_range_m")
            parsed["speed_range_mps"] = self._parse_range(ir.get("speed_range_mps"), "speed_range_mps")
            parsed["positive_regions"] = [
                self._canonical_region_id(value) for value in (ir.get("positive_regions") or [])
            ]
            parsed["negative_regions"] = [
                self._canonical_region_id(value) for value in (ir.get("negative_regions") or [])
            ]
            behaviors: list[dict[str, Any]] = []
            raw_behaviors = ir.get("behavior_sequence") or []
            if not isinstance(raw_behaviors, list):
                raise ValueError("behavior_sequence 必须是数组，例如 [{\"name\": \"orbit\"}]")
            for item in raw_behaviors:
                raw_name = item.get("name") if isinstance(item, dict) else str(item)
                name = str(raw_name).strip().lower()
                canonical = _BEHAVIOR_ALIASES.get(name)
                if canonical is None:
                    for candidate in sorted(BEHAVIOR_NAMES, key=len, reverse=True):
                        if name.endswith(candidate):
                            canonical = candidate
                            break
                if canonical is None:
                    raise ValueError(f"未知行为 {raw_name!r}，可选: {list(BEHAVIOR_NAMES)}")
                behaviors.append({"name": canonical, "term": str(raw_name), "char_position": None})
            parsed["behavior_sequence"] = behaviors
            flags = {"low_speed": False, "very_high_altitude": False, "negation": False}
            raw_flags = ir.get("flags") or {}
            if not isinstance(raw_flags, dict):
                raise ValueError("flags 必须是对象，例如 {\"low_speed\": true}")
            for key, value in raw_flags.items():
                if key in flags:
                    flags[key] = bool(value)
            parsed["flags"] = flags
            if flags.get("low_speed"):
                if parsed["speed_range_mps"] is None:
                    parsed["speed_range_mps"] = (float("-inf"), limits["low_speed_max_mps"])
            if flags.get("very_high_altitude"):
                if parsed["altitude_range_m"] is None:
                    parsed["altitude_range_m"] = (limits["very_high_altitude_min_m"], float("inf"))
            limit = max(1, min(int(limit), 50))
            count_only = bool(count_only)
        except (TypeError, ValueError) as exc:
            return _err(
                "BAD_ARGS",
                str(exc),
                hint=(
                    '修正后重试，示例: {"tool":"search_tracks","args":{"ir":'
                    '{"time_start_utc":"2025-01-01T00:00:00Z","time_end_utc":"2025-01-01T06:00:00Z",'
                    '"altitude_range_m":[4000,7000],"speed_range_mps":[0,100],'
                    '"positive_regions":["区域 A"],"negative_regions":["区域 E"],'
                    '"behavior_sequence":[{"name":"orbit"}],"flags":{"low_speed":false}},"limit":20}}'
                ),
            )

        candidates: list[dict[str, Any]] = []
        for fact in self.facts:
            trajectory_id = fact["trajectory_id"]
            features = self.feature_by_id[trajectory_id]
            passed, reasons = hard_filter(
                fact, parsed, self.region_hits_by_id.get(trajectory_id, {}), features
            )
            if not passed:
                continue
            score, matched = behavior_score(features, parsed)
            failed = [f"behavior:{item['name']}" for item in matched if not item["matched"]]
            candidates.append(
                {
                    "id": trajectory_id,
                    "score": round(score, 4),
                    "ok": not failed,
                    "failed": failed,
                    "start": fact.get("start_time_utc"),
                    "alt_m": round(fact["summary"]["altitude_m"]["mean"], 1),
                    "spd_mps": round(fact["summary"]["speed_mps"]["mean"], 1),
                }
            )
        candidates.sort(key=lambda item: (-item["score"], item["id"]))
        limit = max(1, min(int(limit), 50))
        returned = candidates[:limit]
        self.context.setdefault("allowed_ids", set()).update(item["id"] for item in returned)
        self.context["search_calls"] = int(self.context.get("search_calls", 0)) + 1
        payload = {
            "constraints_echo": {
                "time_start_utc": parsed["time_start_utc"],
                "time_end_utc": parsed["time_end_utc"],
                "altitude_range_m": list(parsed["altitude_range_m"]) if ir.get("altitude_range_m") else None,
                "speed_range_mps": list(parsed["speed_range_mps"]) if ir.get("speed_range_mps") else None,
                "positive_regions": parsed["positive_regions"],
                "negative_regions": parsed["negative_regions"],
                "behavior_sequence": [item["name"] for item in behaviors],
                "flags": flags,
                "top_k": parsed["top_k"],
            },
            "thresholds_used": limits,
            "candidate_count": len(candidates),
        }
        if not count_only:
            payload["returned_count"] = len(returned)
            payload["results"] = returned
        return _ok(payload)

    def track_digest(self, trajectory_id: str) -> dict[str, Any]:
        if trajectory_id not in self.tracks_by_id:
            return _err("NOT_FOUND", f"unknown trajectory_id: {trajectory_id}", "ids must come from search_tracks")
        fact = self.facts_by_id[trajectory_id]
        features = self.feature_by_id[trajectory_id]
        flags = [name for name, value in features["behavior_flags"].items() if value]
        return _ok(
            {
                "trajectory_id": trajectory_id,
                "start_time_utc": fact.get("start_time_utc"),
                "end_time_utc": fact.get("end_time_utc"),
                "point_count": fact["summary"].get("point_count"),
                "duration_seconds": fact["summary"].get("duration_seconds"),
                "altitude_m": fact["summary"].get("altitude_m"),
                "speed_mps": fact["summary"].get("speed_mps"),
                "behavior_flags": flags,
                "stages": features.get("stages"),
                "quality_issue": bool(fact.get("quality", {}).get("has_quality_issue", False)),
            }
        )

    def region_hit(self, trajectory_id: str, region_id: str) -> dict[str, Any]:
        if trajectory_id not in self.tracks_by_id:
            return _err("NOT_FOUND", f"unknown trajectory_id: {trajectory_id}")
        region = self.query_region_by_id.get(str(region_id))
        if region is None:
            return _err("NOT_FOUND", f"unknown region_id: {region_id}", f"allowed: {sorted(self.query_region_by_id)}")
        summary = region_event_summary(self.tracks_by_id[trajectory_id], region)
        return _ok(
            {
                "trajectory_id": trajectory_id,
                "region_id": str(region_id),
                "hit": bool(
                    summary["start_inside"]
                    or summary["end_inside"]
                    or summary["entry_events"]
                    or summary["exit_events"]
                ),
                "start_inside": summary["start_inside"],
                "end_inside": summary["end_inside"],
                "entry_events": summary["entry_events"],
                "exit_events": summary["exit_events"],
                "total_dwell_seconds": round(summary["total_dwell_seconds"], 3),
            }
        )

    def behavior_check(self, trajectory_id: str, behavior: str) -> dict[str, Any]:
        if trajectory_id not in self.tracks_by_id:
            return _err("NOT_FOUND", f"unknown trajectory_id: {trajectory_id}")
        if behavior not in BEHAVIOR_NAMES:
            return _err("BAD_ARGS", f"unknown behavior: {behavior!r}", f"allowed: {list(BEHAVIOR_NAMES)}")
        features = self.feature_by_id[trajectory_id]
        evidence = {
            key: features[key]
            for key in (
                "altitude_mean_m",
                "altitude_min_m",
                "altitude_max_m",
                "speed_mean_mps",
                "speed_median_mps",
                "speed_std_mps",
                "altitude_std_m",
                "circularity",
                "total_turn_deg",
            )
        }
        evidence["altitude_window_deltas_m"] = [round(value, 1) for value in features["altitude_window_deltas_m"]]
        evidence["speed_window_deltas_mps"] = [round(value, 1) for value in features["speed_window_deltas_mps"]]
        return _ok(
            {
                "trajectory_id": trajectory_id,
                "behavior": behavior,
                "matched": bool(features["behavior_flags"].get(behavior, False)),
                "evidence": evidence,
            }
        )

    def validate_submission(self, sample_id: str, answer: dict[str, Any]) -> dict[str, Any]:
        sample = self.context.get("sample", {})
        if sample.get("sample_id") != sample_id:
            return _err("BAD_CONTEXT", f"sample_id mismatch: {sample_id}")
        task = self.context.get("task")
        errors: list[str] = []
        checks: dict[str, Any] = {}
        if task in {"task1_segment", "task1_region"}:
            choice = (answer or {}).get("choice")
            checks["format"] = isinstance(choice, str) and choice.upper() in {"A", "B", "C", "D"}
            if not checks["format"]:
                errors.append("choice must be one of A/B/C/D")
        elif task == "task2":
            results = (answer or {}).get("results")
            ids = results if isinstance(results, list) else []
            top_k = int(sample.get("top_k", 10))
            checks["format"] = isinstance(ids, list) and all(isinstance(item, str) for item in ids)
            checks["count_ok"] = 1 <= len(ids) <= top_k
            checks["unique_ids"] = len(ids) == len(set(ids))
            checks["ids_in_library"] = all(item in self.library_ids for item in ids)
            allowed = self.context.get("allowed_ids") or set()
            checks["ids_in_candidates"] = all(item in allowed for item in ids) if allowed else False
            checks["top_k"] = top_k
            checks["returned"] = len(ids)
            if not checks["format"]:
                errors.append("results 必须是字符串数组")
            if not checks["count_ok"]:
                errors.append(f"返回 {len(ids)} 条，但 top_k={top_k}，必须为 1..{top_k} 条")
            if not checks["unique_ids"]:
                errors.append("results 里有重复的航迹编号")
            if not checks["ids_in_library"]:
                errors.append("results 里有不在航迹库中的编号")
            if not checks["ids_in_candidates"]:
                errors.append("results 里有不在本次 search_tracks 候选中的编号")
        else:
            return _err("BAD_CONTEXT", f"unsupported task: {task}")
        return _ok({"sample_id": sample_id, "ok": not errors, "errors": errors, "checks": checks})

    # ------------------------------------------------------------------ dispatch
    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, name, None)
        if handler is None or name.startswith("_") or name in {
            "set_context",
            "call",
            "context",
        }:
            return _err("UNKNOWN_TOOL", f"unknown tool: {name}")
        if not isinstance(args, dict):
            return _err("BAD_ARGS", "args must be a JSON object")
        try:
            return handler(**args)
        except TypeError as exc:
            return _err("BAD_ARGS", f"invalid tool arguments: {exc}")
        except Exception as exc:  # noqa: BLE001 - tools must never crash the agent loop
            return _err("TOOL_ERROR", f"{type(exc).__name__}: {exc}")
