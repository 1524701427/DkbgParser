from __future__ import annotations

from typing import Any, Iterable


def merge_first_cultivated_soil_layer(
    layers: Any,
    excluded_names: Iterable[str] = ("耕土",),
) -> list[dict[str, Any]]:
    """删除首层耕土，并将其厚度合并到下一层业务记录。"""
    copied_layers = [dict(layer) for layer in layers if isinstance(layer, dict)]
    if len(copied_layers) < 2:
        return copied_layers

    first_name = str(copied_layers[0].get("layer_name") or "")
    names = [str(name) for name in excluded_names]
    if not any(name in first_name for name in names):
        return copied_layers

    first = copied_layers[0]
    following = copied_layers[1]

    def business_thickness(record: dict[str, Any]) -> float | None:
        """按既有业务优先级读取一条土层当前可用厚度。"""
        for field in (
            "final_value",
            "effective_value",
            "thickness",
            "thickness_average",
            "maximum_exposed",
            "maximum_exposed_thickness",
            "thickness_exact",
            "thickness_max",
        ):
            value = record.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        return None

    first_thickness = business_thickness(first)
    following_thickness = business_thickness(following)
    if first_thickness is not None and following_thickness is not None:
        merged_thickness = round(first_thickness + following_thickness, 6)
        updated = False
        for field in ("effective_value", "final_value", "thickness"):
            if isinstance(following.get(field), (int, float)):
                following[field] = merged_thickness
                updated = True
        if not updated:
            following["final_value"] = merged_thickness

        for field in ("thickness_min", "thickness_max", "thickness_average"):
            first_value = first.get(field)
            following_value = following.get(field)
            if isinstance(first_value, (int, float)) and isinstance(
                following_value, (int, float)
            ):
                following[field] = round(
                    float(first_value) + float(following_value), 6
                )
        following["merged_cultivated_soil"] = {
            "layer_code": first.get("layer_code"),
            "layer_name": first.get("layer_name"),
            "thickness": first_thickness,
        }
    return copied_layers[1:]


def infer_foundation_type(layer_name: str) -> str:
    """按当前土层名称判断该层采用的桩型。"""
    name = str(layer_name or "")
    return "cast_in_place" if "岩" in name or "石" in name else "precast"


def select_foundation_parameter(
    layer_name: str,
    *,
    cast_in_place: Any,
    precast: Any,
) -> Any:
    """按照当前土层名称选择灌注桩或预制桩参数。"""
    if infer_foundation_type(layer_name) == "cast_in_place":
        return cast_in_place
    return precast
