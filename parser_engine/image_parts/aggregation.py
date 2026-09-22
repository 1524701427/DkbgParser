from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Any


class BoreholeAggregationMixin:
    """钻孔识别结果标准化、深度校验与跨孔聚合职责。"""

    @staticmethod
    def _normalize_result(
        result: dict[str, Any],
        page: int,
        image_path: Path,
    ) -> list[dict[str, Any]]:
        borehole_id = result.get("borehole_id")
        records = []
        for layer in result.get("layers", []):
            if not isinstance(layer, dict) or not layer.get("layer_code"):
                continue
            code = re.sub(r"\s+", "", str(layer["layer_code"]))
            records.append(
                {
                    "layer_code": code,
                    "main_layer_code": BoreholeAggregationMixin._main_layer_code(code),
                    "layer_name": layer.get("layer_name"),
                    "bottom_elevation": BoreholeAggregationMixin._number(
                        layer.get("bottom_elevation")
                    ),
                    "bottom_depth": BoreholeAggregationMixin._number(
                        layer.get("bottom_depth")
                    ),
                    "image_thickness": BoreholeAggregationMixin._number(
                        layer.get("thickness")
                    ),
                    "description": layer.get("description"),
                    "confidence": BoreholeAggregationMixin._number(
                        layer.get("confidence")
                    ),
                    "borehole_id": borehole_id,
                    "evidence": {
                        "source_type": "image",
                        "page": page,
                        "image_path": str(image_path),
                        "ocr_evidence": layer.get("ocr_evidence"),
                    },
                }
            )
        return records

    @staticmethod
    def _validate_observations(
        records: list[dict[str, Any]], tolerance: float
    ) -> None:
        groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            evidence = record.get("evidence", {})
            group_key = record.get("borehole_id") or f"page:{evidence.get('page')}"
            groups[group_key].append(record)
        for layers in groups.values():
            previous_depth = 0.0
            for layer in layers:
                bottom_depth = layer.get("bottom_depth")
                thickness = layer.get("image_thickness")
                if bottom_depth is None:
                    layer["depth_validation"] = None
                    continue
                calculated = round(bottom_depth - previous_depth, 3)
                if thickness is None and calculated > 0:
                    layer["image_thickness"] = calculated
                    layer["thickness_source"] = "calculated_from_bottom_depth"
                    layer["depth_validation"] = True
                elif thickness is not None:
                    layer["depth_validation"] = abs(calculated - thickness) <= tolerance
                previous_depth = bottom_depth

    @staticmethod
    def _aggregate(
        records: list[dict[str, Any]], operator: str
    ) -> list[dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            groups[str(record["layer_code"])].append(record)
        result = []
        for code, observations in groups.items():
            valid_observations = [
                item
                for item in observations
                if item.get("image_thickness") is not None
                and item.get("depth_validation") is not False
            ]
            values = [item["image_thickness"] for item in valid_observations]
            if not values:
                continue
            if operator == "minimum":
                aggregated = min(values)
            elif operator == "maximum":
                aggregated = max(values)
            else:
                aggregated = sum(values) / len(values)
            first = valid_observations[0]
            layer_name = next(
                (
                    item.get("layer_name")
                    for item in observations
                    if item.get("layer_name")
                ),
                None,
            )
            borehole_ids = sorted(
                {
                    str(item["borehole_id"])
                    for item in valid_observations
                    if item.get("borehole_id")
                },
                key=BoreholeAggregationMixin._borehole_sort_key,
            )
            governing_borehole_ids = sorted(
                {
                    str(item["borehole_id"])
                    for item in valid_observations
                    if item.get("borehole_id")
                    and item.get("image_thickness") == aggregated
                },
                key=BoreholeAggregationMixin._borehole_sort_key,
            )
            result.append(
                {
                    "layer_code": code,
                    "main_layer_code": first["main_layer_code"],
                    "layer_name": layer_name,
                    "image_average": round(aggregated, 3),
                    "image_thickness": (
                        first.get("image_thickness")
                        if len(observations) == 1
                        else None
                    ),
                    "observation_count": len(values),
                    "borehole_ids": borehole_ids,
                    "governing_borehole_ids": (
                        governing_borehole_ids
                        if operator in {"minimum", "maximum"}
                        else []
                    ),
                    "observations": observations,
                    "evidence": first["evidence"],
                }
            )
        return result

    @staticmethod
    def _borehole_sort_key(value: str) -> tuple[str, int, str]:
        match = re.fullmatch(r"([A-Za-z]+)0*(\d+)", value.strip())
        if not match:
            return value.upper(), -1, value
        return match.group(1).upper(), int(match.group(2)), value

    @staticmethod
    def _main_layer_code(code: str) -> str:
        match = re.match(
            r"([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|\d+)",
            code,
        )
        return match.group(1) if match else code

    @staticmethod
    def _number(value: Any) -> float | None:
        if value is None:
            return None
        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        return float(match.group(0)) if match else None

    @staticmethod
    def _default_prompt() -> str:
        return """识别图片中的钻孔柱状图，只返回一个 JSON 对象，不要返回 Markdown。
JSON 格式：
{"borehole_id":"钻孔编号", "layers":[{"layer_code":"层号", "layer_name":"岩土名称", "bottom_elevation":数值或null, "bottom_depth":数值或null, "thickness":数值或null, "description":"岩性描述", "confidence":0到1}]}
必须按照图片中从上到下的岩土层顺序输出。无法确认的字段使用 null，禁止猜测。数字不要包含单位。"""
