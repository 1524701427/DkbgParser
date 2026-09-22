from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

from .extraction import merge_first_cultivated_soil_layer, select_foundation_parameter


DEFAULT_REVERSE_GEOLOGY_URL = (
    "http://172.16.14.71:10004/rpc-api/reverse-callback/parse-reverse-geology"
)


# 顶层字段映射：键是接口字段，值是 result.json 中的来源路径。
# 此字典既用于实际转换，也可以直接作为接口字段对照表查看。
CALLBACK_FIELD_MAPPING = {
    "bearingStratumInfo": "key_data.bearing_layer_description",  # 持力层描述
    "conclusion": "conclusion_and_evaluation",  # 报告结论与评价正文
    "earthquakeGroup": "seismic_parameters.design_earthquake_group",  # 设计地震分组
    "epa": "seismic_parameters.peak_ground_acceleration_g",  # 地震动峰值加速度
    "evaluation": "site_geological_conditions_and_evaluation.evaluation",  # 场地评价
    "geologyId": "$config.geology_id",  # 后端已有地质数据ID，来自 main.py 配置
    "geologyRockSoilsReq": "geotechnical_layer_parameters",  # 岩土层参数列表
    "groundwaterDepth": "key_data.groundwater_depth_m",  # 地下水埋深或埋深区间
    "handleKeyword": "key_data.foundation_treatment",  # 岩溶、湿陷性黄土等处理关键字
    "ld": "seismic_parameters.basic_seismic_intensity_degree",  # 场址基本烈度
    "projectId": "$config.project_id",  # 项目ID，来自 main.py 配置
    "regionalGeologyConditionEvaluation": (
        "site_geological_conditions_and_evaluation.evaluation"
    ),  # 地质条件评价
    "regionalGeologyInfo": "draft_content.[区域地质相关真实章节]",  # 区域地质正文
    "regionalHydrologyInfo": "regional_hydrology",  # 区域水文正文
    "siteClassification": "seismic_parameters.site_category",  # 场地类别
    "tg": "seismic_parameters.response_spectrum_characteristic_period_s",  # 特征周期
    "waterSoilErosion": "key_data.water_soil_corrosion",  # 水土腐蚀性编码
}

# 岩土层字段映射：每个 geotechnical_layer_parameters 元素转换为一个接口土层。
CALLBACK_LAYER_FIELD_MAPPING = {
    "angleFriction": "friction_angle",  # 摩擦角 Φ，单位：度
    "bearingCapacity": "bearing_capacity_fak",  # 承载力特征值 fak，单位：kPa
    "cohesion": "cohesion",  # 黏聚力/内聚力 C，单位：kPa
    "compressionModulus": "compression_modulus_es1_2",  # 压缩模量 Es1-2，单位：MPa
    "correctionFactorDeepBearCapacity": "depth_bearing_coefficient_eta_d",  # 深度修正系数
    "correctionSeismicBearCapacityFoundation": (
        "seismic_bearing_coefficient_zeta_a"
    ),  # 地基抗震承载力调整系数
    "endEffectCoefficientLateralResistance": "tip_resistance_size_effect",  # 端阻尺寸效应系数
    "geologyId": "$config.geology_id",  # 父级地质数据ID
    "gravityDensity": "gravity_density",  # 重力密度 γ，单位：kN/m³
    "horizontalResistanceRatioCoefficient": (
        "horizontal_resistance_coefficient"
    ),  # result 已按每层名称自动选择水平抗力比例系数
    "id": "$config.layer_ids",  # 已有土层ID，按层号或层名匹配
    "liquefactionFactor": "liquefaction_reduction_coefficient",  # 液化折减系数
    "name": "layer_code + layer_name",  # 岩土层完整名称，例如“②-1粉质黏土”
    "negativeFrictionResistanceCoefficient": (
        "negative_friction_coefficient"
    ),  # result 已按每层名称自动选择负摩擦阻力系数
    "poissonRatio": "poisson_ratio",  # 泊松比；接口要求字符串
    "pullOutCoefficient": "uplift_coefficient",  # 抗拔系数
    "sizeEffectCoefficientLateralResistance": "side_resistance_size_effect",  # 侧阻尺寸效应系数
    "standardPileEndResistance": "pile_tip_resistance",  # 按每层名称自动选择的桩端阻力
    "standardPileSideResistance": "pile_side_resistance",  # 按每层名称自动选择的桩侧阻力
    "thickness": "thickness",  # 工程采用厚度，单位：m
    "widthBearCapacityCorrectionFactor": "width_bearing_coefficient_eta_b",  # 宽度修正系数
}


_CORROSION_CODES = {
    "微腐蚀": 0,
    "微腐蚀性": 0,
    "弱腐蚀": 1,
    "弱腐蚀性": 1,
    "中腐蚀": 2,
    "中腐蚀性": 2,
    "强腐蚀": 3,
    "强腐蚀性": 3,
}


def build_reverse_geology_payload(
    result: Mapping[str, Any],
    *,
    project_id: int,
    geology_id: int = 0,
    layer_ids: Mapping[str, int] | None = None,
    handle_keyword_codes: Mapping[str, int] | None = None,
    ambiguous_corrosion_code: int | None = None,
    omit_missing: bool = False,
    strict_enums: bool = False,
) -> dict[str, Any]:
    """把精简抽取结果转换为逆向更新地质数据接口的请求体。

    Args:
        result: ``<报告名>.json`` 对应的精简业务结果。
        project_id: 接口要求的项目ID。
        geology_id: 已存在的地质数据ID；新增或未知时可传 ``0``。
        layer_ids: 可选的岩土层数据库ID映射。键可使用层号、层名或
            ``层号+层名``，例如 ``{"②": 101}``。
        handle_keyword_codes: 接口 ``handleKeyword`` 的业务枚举映射，例如
            ``{"岩溶": 1, "湿陷性黄土": 2}``。接口文档未定义编码，因此
            检出处理关键字时必须由调用方明确提供。
        ambiguous_corrosion_code: 当前结果为“中强腐蚀性”时采用的接口编码。
            接口把中腐蚀和强腐蚀分为2、3，无法从合并值判断时必须明确指定。
        omit_missing: 是否删除值为 ``None`` 的可选字段。默认不删除，保证接口
            始终返回完整映射结构；没有抽取到或没有映射上的字段返回 JSON ``null``。
            旧调用方如仍要求省略空字段，可显式传入 ``True``。
        strict_enums: 枚举无法确定时是否抛出异常。默认不抛出，并把无法映射的
            枚举字段保留为 ``None``；接口联调校验时可设为 ``True``。

    Returns:
        可直接作为 ``/rpc-api/reverse-callback/parse-reverse-geology`` 请求体
        ``condition`` 内容的字典。

    Raises:
        ValueError: 基础形式无效，或遇到接口文档未定义的枚举值。
    """
    layer_ids = layer_ids or {}
    seismic = _mapping(result.get("seismic_parameters"))
    key_data = _mapping(result.get("key_data"))
    site_data = _mapping(result.get("site_geological_conditions_and_evaluation"))
    draft_content = _mapping(result.get("draft_content"))

    treatment_values = _as_text_list(key_data.get("foundation_treatment"))
    handle_keyword = _handle_keyword_code(
        treatment_values,
        handle_keyword_codes,
        strict=strict_enums,
    )
    corrosion_code = _corrosion_code(
        key_data.get("water_soil_corrosion"),
        ambiguous_corrosion_code=ambiguous_corrosion_code,
        strict=strict_enums,
    )

    # 接口用于基础计算：首层为耕土时不单独回写，而是把其厚度并入下一层。
    # 这里只整理接口副本，不修改 result.json 中用于追溯的原始地层清单。
    interface_layers = merge_first_cultivated_soil_layer(
        result.get("geotechnical_layer_parameters", [])
    )
    rock_soils = [
        _map_layer(
            layer,
            geology_id=geology_id,
            layer_ids=layer_ids,
            omit_missing=omit_missing,
        )
        for layer in interface_layers
        if isinstance(layer, Mapping)
    ]

    # 接口同时存在 evaluation 与 regionalGeologyConditionEvaluation，但当前
    # 业务结果只有一份明确的场区地质评价，因此两个字段使用同一可靠来源。
    evaluation = _text(site_data.get("evaluation"))
    # 普通字段严格按照 CALLBACK_FIELD_MAPPING 取值；需要格式转换或外部配置的
    # 字段在取值后按接口类型统一处理。
    payload = {
        target: _get_path(result, source)
        for target, source in CALLBACK_FIELD_MAPPING.items()
        if not source.startswith("$config") and "[" not in source
    }
    payload.update(
        {
            "bearingStratumInfo": _join_text(payload.get("bearingStratumInfo")),
            "conclusion": _text(payload.get("conclusion")),
            "earthquakeGroup": _text(payload.get("earthquakeGroup")),
            "epa": _number(payload.get("epa")),
            "evaluation": evaluation,
            "geologyId": int(geology_id),
            "geologyRockSoilsReq": rock_soils,
            "groundwaterDepth": _format_groundwater_depth(payload.get("groundwaterDepth")),
            "handleKeyword": handle_keyword,
            "ld": _integer(payload.get("ld")),
            "projectId": int(project_id),
            "regionalGeologyConditionEvaluation": evaluation,
            "regionalGeologyInfo": _find_draft_section(
                draft_content,
                ("区域地质构造", "构造地质条件", "地质构造", "区域地质", "区域稳定"),
            ),
            "regionalHydrologyInfo": _text(payload.get("regionalHydrologyInfo")),
            "siteClassification": _join_text(
                payload.get("siteClassification"), separator="、"
            ),
            "tg": _number(payload.get("tg")),
            "waterSoilErosion": corrosion_code,
        }
    )
    return _omit_none(payload) if omit_missing else payload


def post_reverse_geology_payload(
    payload: Mapping[str, Any],
    *,
    api_url: str = DEFAULT_REVERSE_GEOLOGY_URL,
    timeout: float = 30.0,
) -> Any:
    """把映射后的地质结果作为 JSON 直接 POST 到逆向回调接口。

    Args:
        payload: ``build_reverse_geology_payload()`` 生成的完整接口请求体。
        api_url: 逆向地质回调地址。
        timeout: HTTP 请求超时时间，单位为秒。

    Returns:
        接口响应。响应为 JSON 时返回解析后的对象；普通文本按字符串返回；
        空响应返回 ``None``。

    Raises:
        RuntimeError: HTTP 状态异常、网络不可达或请求超时。
        ValueError: timeout 非正数或 api_url 为空。
    """
    url = str(api_url or "").strip()
    if not url:
        raise ValueError("逆向地质回调地址不能为空")
    if timeout <= 0:
        raise ValueError("接口请求超时时间必须大于0")

    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            status = getattr(response, "status", None) or response.getcode()
            response_text = response.read().decode("utf-8", errors="replace").strip()
    except urllib.error.HTTPError as exc:
        try:
            error_text = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            error_text = ""
        detail = f": {error_text}" if error_text else ""
        raise RuntimeError(
            f"逆向地质接口请求失败，HTTP {exc.code}{detail}"
        ) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"逆向地质接口请求失败: {reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("逆向地质接口请求超时") from exc

    if int(status) < 200 or int(status) >= 300:
        raise RuntimeError(
            f"逆向地质接口请求失败，HTTP {status}: {response_text}"
        )
    if not response_text:
        return None
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return response_text


def write_reverse_geology_payload(
    result_path: str | Path,
    output_path: str | Path,
    **mapping_options: Any,
) -> dict[str, Any]:
    """读取已生成的 result.json，并写出接口请求 JSON。

    Args:
        result_path: 主流程生成的精简结果 JSON 路径。
        output_path: 接口请求 JSON 输出路径。
        **mapping_options: 传给 :func:`build_reverse_geology_payload` 的项目ID、
            地质ID、基础形式和枚举映射等参数。

    Returns:
        已写入文件的接口请求体字典。

    Raises:
        FileNotFoundError: 精简结果文件不存在。
        ValueError: JSON 或映射配置不合法。
    """
    source = Path(result_path)
    result = json.loads(source.read_text(encoding="utf-8"))
    payload = build_reverse_geology_payload(result, **mapping_options)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _map_layer(
    layer: Mapping[str, Any],
    *,
    geology_id: int,
    layer_ids: Mapping[str, int],
    omit_missing: bool,
) -> dict[str, Any]:
    """映射一条土层结果，并按基础形式选择桩基参数。

    Args:
        layer: 单条精简土层结果。
        geology_id: 父级地质数据ID。
        layer_ids: 土层编号或名称到数据库ID的映射。
        omit_missing: 是否删除缺失字段。

    Returns:
        ``GeologyRockSoilReqDTO`` 对应字典。
    """
    layer_code = _text(layer.get("layer_code")) or ""
    layer_name = _text(layer.get("layer_name")) or ""
    display_name = f"{layer_code}{layer_name}" or None
    layer_id = _lookup_layer_id(layer_ids, layer_code, layer_name, display_name)
    # 兼容旧 result.json 中保留两套参数的结构；新结果会直接提供已经自动选择的
    # pile_side_resistance 和 pile_tip_resistance。
    side_resistance = _foundation_parameter(
        layer,
        generic="pile_side_resistance",
        cast_in_place="cast_in_place_side_resistance",
        precast="precast_side_resistance",
    )
    end_resistance = _foundation_parameter(
        layer,
        generic="pile_tip_resistance",
        cast_in_place="cast_in_place_tip_resistance",
        precast="precast_tip_resistance",
    )

    # 先按公开映射字典复制普通数值字段，再补充需要组合或选择基础形式的字段。
    special_targets = {
        "geologyId",
        "horizontalResistanceRatioCoefficient",
        "id",
        "name",
        "negativeFrictionResistanceCoefficient",
        "poissonRatio",
        "standardPileEndResistance",
        "standardPileSideResistance",
    }
    mapped = {
        target: _number(layer.get(source))
        for target, source in CALLBACK_LAYER_FIELD_MAPPING.items()
        if target not in special_targets
    }
    mapped.update(
        {
            "geologyId": int(geology_id),
            "horizontalResistanceRatioCoefficient": _number(
                _select_foundation_value(
                    layer.get("horizontal_resistance_coefficient"), layer_name
                )
            ),
            "id": layer_id,
            "name": display_name,
            "negativeFrictionResistanceCoefficient": _number(
                _select_foundation_value(
                    layer.get("negative_friction_coefficient"), layer_name
                )
            ),
            # 接口把泊松比声明为字符串，避免反序列化时发生类型不一致。
            "poissonRatio": _number_text(layer.get("poisson_ratio")),
            "standardPileEndResistance": _number(end_resistance),
            "standardPileSideResistance": _number(side_resistance),
        }
    )
    return _omit_none(mapped) if omit_missing else mapped


def _foundation_parameter(
    layer: Mapping[str, Any],
    *,
    generic: str,
    cast_in_place: str,
    precast: str,
) -> Any:
    """优先读取已选择的通用桩参数，否则读取指定基础形式字段。"""
    if layer.get(generic) is not None:
        return layer[generic]
    return select_foundation_parameter(
        str(layer.get("layer_name") or ""),
        cast_in_place=layer.get(cast_in_place),
        precast=layer.get(precast),
    )


def _select_foundation_value(value: Any, layer_name: str) -> Any:
    """按公共土层判断规则从灌注桩、预制桩组合值中选择参数。"""
    if isinstance(value, Mapping):
        return select_foundation_parameter(
            layer_name,
            cast_in_place=value.get("cast_in_place"),
            precast=value.get("precast"),
        )
    return value


def _lookup_layer_id(
    layer_ids: Mapping[str, int], layer_code: str, layer_name: str, display_name: str | None
) -> int | None:
    """依次按完整名称、层号和层名查找已有土层ID；未映射时返回空值。"""
    for key in (display_name, layer_code, layer_name):
        if key and key in layer_ids:
            return int(layer_ids[key])
    return None


def _handle_keyword_code(
    treatment_values: list[str], codes: Mapping[str, int] | None, *, strict: bool
) -> int | None:
    """将地基处理关键字映射为接口枚举，未检出时返回0。

    接口只允许一个整数，因此同时命中多个不同枚举时不能静默取第一项，
    需要调用方先确定本次回写采用哪个业务状态。
    """
    if not treatment_values:
        return 0
    if not codes:
        if strict:
            raise ValueError(
                "抽取结果包含地基处理关键字，但接口文档未定义 handleKeyword 编码；"
                "请传入 handle_keyword_codes"
            )
        return None
    matched_codes: set[int] = set()
    for treatment in treatment_values:
        for keyword, code in codes.items():
            if str(keyword) in treatment:
                matched_codes.add(int(code))
    if not matched_codes:
        if strict:
            raise ValueError(f"地基处理关键字没有对应接口编码: {treatment_values}")
        return None
    if len(matched_codes) > 1:
        if strict:
            raise ValueError(
                "抽取结果同时命中多个 handleKeyword 编码，但接口只接收一个整数："
                f"{sorted(matched_codes)}"
            )
        return None
    return matched_codes.pop()


def _corrosion_code(
    value: Any, *, ambiguous_corrosion_code: int | None, strict: bool
) -> int | None:
    """把腐蚀性文本映射为接口0～3编码。"""
    text = _text(value)
    if not text:
        return None
    if "中强" in text:
        if ambiguous_corrosion_code not in {2, 3}:
            if strict:
                raise ValueError(
                    "当前结果为“中强腐蚀性”，接口却区分中腐蚀2和强腐蚀3；"
                    "请把 ambiguous_corrosion_code 明确设为2或3"
                )
            return None
        return ambiguous_corrosion_code
    for label, code in _CORROSION_CODES.items():
        if label in text:
            return code
    if strict:
        raise ValueError(f"无法映射水土腐蚀性: {text}")
    return None


def _get_path(data: Mapping[str, Any], path: str) -> Any:
    """按照点分路径读取嵌套结果字段。

    Args:
        data: 精简结果字典。
        path: 例如 ``seismic_parameters.site_category`` 的来源路径。

    Returns:
        路径对应的值；中间字段不存在时返回 ``None``。
    """
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _format_groundwater_depth(value: Any) -> str | None:
    """把地下水埋深数值或区间转换为接口字符串。"""
    if isinstance(value, Mapping):
        start = _number(value.get("start"))
        end = _number(value.get("end"))
        if start is None and end is None:
            return None
        if start == end or end is None:
            return f"{_plain_number(start)}m" if start is not None else None
        if start is None:
            return f"{_plain_number(end)}m"
        return f"{_plain_number(start)}~{_plain_number(end)}m"
    number = _number(value)
    return f"{_plain_number(number)}m" if number is not None else _text(value)


def _find_draft_section(draft: Mapping[str, Any], keywords: tuple[str, ...]) -> str | None:
    """按真实章节标题查找区域地质正文。"""
    for title, text in draft.items():
        compact_title = str(title).replace(" ", "")
        if any(keyword in compact_title for keyword in keywords):
            return _text(text)
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    """返回映射值；其他类型按空映射处理。"""
    return value if isinstance(value, Mapping) else {}


def _as_text_list(value: Any) -> list[str]:
    """把标量或数组统一成非空文本列表。"""
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _join_text(value: Any, *, separator: str = "\n") -> str | None:
    """连接字符串数组，标量保持原样。"""
    values = _as_text_list(value)
    return separator.join(values) if values else None


def _text(value: Any) -> str | None:
    """将非空值转换为字符串。"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _number(value: Any) -> int | float | None:
    """保留有效数值并拒绝布尔值。"""
    if isinstance(value, bool) or value is None:
        return None
    return value if isinstance(value, (int, float)) else None


def _integer(value: Any) -> int | None:
    """将有效数字转换为整数。"""
    number = _number(value)
    return int(number) if number is not None else None


def _number_text(value: Any) -> str | None:
    """把数字转换为不带多余零的字符串。"""
    number = _number(value)
    return _plain_number(number) if number is not None else _text(value)


def _plain_number(value: int | float) -> str:
    """输出适合接口的紧凑数字文本。"""
    return str(int(value)) if float(value).is_integer() else str(value)


def _omit_none(values: dict[str, Any]) -> dict[str, Any]:
    """删除缺失字段，但保留0、空数组及空字符串等有效接口值。"""
    return {key: value for key, value in values.items() if value is not None}
