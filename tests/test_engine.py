from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest

from parser_engine.engine import DocumentParser, ParserConfig
from parser_engine.extraction import (
    ExtractionEngine,
    _compact_result,
    infer_foundation_type,
    merge_first_cultivated_soil_layer,
    select_foundation_parameter,
)
from parser_engine.callback_mapping import (
    build_reverse_geology_payload,
    write_reverse_geology_payload,
)
from parser_engine.image_recognition import (
    BoreholeImageRecognizer,
    OpenAICompatibleVisionClient,
    RapidOCRClient,
)
from parser_engine.aspose_runtime import load_aspose
from parser_engine.exceptions import (
    BackendUnavailableError,
    ScannedPdfNotSupportedError,
    UnsupportedFormatError,
)
from parser_engine.models import (
    DocumentBlock,
    DocumentModel,
    PageInfo,
    ParagraphStyle,
    TableCell,
    TableData,
    TextSpan,
    TextStyle,
)
from parser_engine.loaders.opendataloader import OpenDataLoaderPdfLoader


def test_build_reverse_geology_payload_maps_business_result():
    """验证精简结果能够按基础形式映射成接口请求体。"""
    result = {
        "geotechnical_layer_parameters": [
            {
                "layer_code": "②-1",
                "layer_name": "粉质黏土",
                "thickness": 1.79,
                "gravity_density": 18.2,
                "cohesion": 38.4,
                "friction_angle": 12.3,
                "compression_modulus_es1_2": 5.12,
                "bearing_capacity_fak": 100,
                "poisson_ratio": 0.35,
                "width_bearing_coefficient_eta_b": 0.3,
                "depth_bearing_coefficient_eta_d": 1.6,
                "seismic_bearing_coefficient_zeta_a": 1.1,
                "liquefaction_reduction_coefficient": 1.0,
                "uplift_coefficient": 0.7,
                "side_resistance_size_effect": 1.0,
                "tip_resistance_size_effect": 1.0,
                "horizontal_resistance_coefficient": {
                    "precast": 8000,
                    "cast_in_place": 20000,
                },
                "negative_friction_coefficient": {
                    "precast": 0.1,
                    "cast_in_place": 0.2,
                },
                "cast_in_place_side_resistance": 50,
                "cast_in_place_tip_resistance": 1000,
                "precast_side_resistance": 52,
                "precast_tip_resistance": 2600,
            }
        ],
        "seismic_parameters": {
            "peak_ground_acceleration_g": 0.1,
            "basic_seismic_intensity_degree": 7,
            "response_spectrum_characteristic_period_s": 0.65,
            "site_category": "Ⅲ类",
            "design_earthquake_group": "第三组",
        },
        "key_data": {
            "water_soil_corrosion": "微腐蚀性",
            "foundation_treatment": ["岩溶处理"],
            "bearing_layer_description": ["建议以⑨层作为桩端持力层"],
            "groundwater_depth_m": {"start": 3.16, "end": 5.29},
        },
        "draft_content": {"3.2 构造地质条件": "区域构造正文"},
        "site_geological_conditions_and_evaluation": {"evaluation": "场地稳定"},
        "regional_hydrology": "区域水文正文",
        "conclusion_and_evaluation": "报告结论正文",
    }

    payload = build_reverse_geology_payload(
        result,
        project_id=10,
        geology_id=20,
        layer_ids={"②-1": 30},
        handle_keyword_codes={"岩溶": 1, "湿陷性黄土": 2},
    )

    assert payload["projectId"] == 10
    assert payload["geologyId"] == 20
    assert payload["epa"] == 0.1
    assert payload["ld"] == 7
    assert payload["tg"] == 0.65
    assert payload["groundwaterDepth"] == "3.16~5.29m"
    assert payload["handleKeyword"] == 1
    assert payload["waterSoilErosion"] == 0
    assert payload["regionalGeologyInfo"] == "区域构造正文"
    layer = payload["geologyRockSoilsReq"][0]
    assert layer["id"] == 30
    assert layer["name"] == "②-1粉质黏土"
    assert layer["poissonRatio"] == "0.35"
    # “粉质黏土”不含“岩/石”，应逐层自动采用预制桩参数。
    assert layer["horizontalResistanceRatioCoefficient"] == 8000
    assert layer["negativeFrictionResistanceCoefficient"] == 0.1
    assert layer["standardPileSideResistance"] == 52
    assert layer["standardPileEndResistance"] == 2600


def test_build_reverse_geology_payload_requires_ambiguous_enum_choices():
    """验证接口文档未定义或语义含混的枚举不会被静默猜测。"""
    result = {
        "geotechnical_layer_parameters": [],
        "key_data": {"water_soil_corrosion": "中强腐蚀性"},
    }

    with pytest.raises(ValueError, match="ambiguous_corrosion_code"):
        build_reverse_geology_payload(
            result,
            project_id=1,
            strict_enums=True,
        )

    payload = build_reverse_geology_payload(
        result,
        project_id=1,
        ambiguous_corrosion_code=3,
    )
    assert payload["waterSoilErosion"] == 3
    assert payload["handleKeyword"] == 0


def test_build_reverse_geology_payload_rejects_multiple_handle_keyword_codes():
    """验证单值接口不会静默丢弃多个地基处理类型。"""
    result = {
        "geotechnical_layer_parameters": [],
        "key_data": {"foundation_treatment": ["岩溶", "湿陷性黄土"]},
    }

    with pytest.raises(ValueError, match="只接收一个整数"):
        build_reverse_geology_payload(
            result,
            project_id=1,
            handle_keyword_codes={"岩溶": 1, "湿陷性黄土": 2},
            strict_enums=True,
        )


def test_build_reverse_geology_payload_omits_missing_values():
    """验证缺失结果不使用零值覆盖接口中的已有业务数据。"""
    payload = build_reverse_geology_payload(
        {"geotechnical_layer_parameters": []},
        project_id=8,
    )

    assert payload["projectId"] == 8
    assert payload["handleKeyword"] == 0
    assert "groundwaterDepth" not in payload
    assert "epa" not in payload


def test_write_reverse_geology_payload_reads_result_file(tmp_path: Path):
    """验证主流程可在 result.json 生成后直接写出接口映射文件。"""
    result_path = tmp_path / "report.json"
    callback_path = tmp_path / "report_callback.json"
    result_path.write_text(
        json.dumps(
            {
                "geotechnical_layer_parameters": [],
                "seismic_parameters": {"basic_seismic_intensity_degree": 7},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    payload = write_reverse_geology_payload(
        result_path,
        callback_path,
        project_id=99,
    )

    assert payload["projectId"] == 99
    assert payload["ld"] == 7
    assert json.loads(callback_path.read_text(encoding="utf-8")) == payload


def test_foundation_parameters_are_selected_per_layer_name():
    """验证所有桩型赋值共用逐层判断规则。"""
    assert infer_foundation_type("粉质黏土") == "precast"
    assert infer_foundation_type("卵石") == "cast_in_place"
    assert infer_foundation_type("砾砂") == "precast"
    assert infer_foundation_type("强风化砂砾岩") == "cast_in_place"
    assert select_foundation_parameter(
        "粉土", cast_in_place=50, precast=52
    ) == 52
    assert select_foundation_parameter(
        "中风化灰岩", cast_in_place=1000, precast=2600
    ) == 1000


def test_callback_mapping_selects_cast_in_place_for_rock_layer():
    """验证旧结果同时保留两套参数时，岩层接口值自动选择灌注桩。"""
    result = {
        "geotechnical_layer_parameters": [
            {
                "layer_code": "③",
                "layer_name": "中风化灰岩",
                "cast_in_place_side_resistance": 80,
                "cast_in_place_tip_resistance": 1500,
                "precast_side_resistance": 90,
                "precast_tip_resistance": 3000,
                "horizontal_resistance_coefficient": {
                    "cast_in_place": 20000,
                    "precast": 8000,
                },
            }
        ]
    }

    payload = build_reverse_geology_payload(result, project_id=1)
    layer = payload["geologyRockSoilsReq"][0]
    assert layer["standardPileSideResistance"] == 80
    assert layer["standardPileEndResistance"] == 1500
    assert layer["horizontalResistanceRatioCoefficient"] == 20000


def test_callback_merges_first_cultivated_soil_into_next_layer():
    """验证接口删除首层耕土，并将其厚度累加到下一层。"""
    result = {
        "geotechnical_layer_parameters": [
            {"layer_code": "①", "layer_name": "耕土", "thickness": 0.5},
            {
                "layer_code": "②",
                "layer_name": "砂砾岩(全风化)",
                "thickness": 9.1,
            },
            {
                "layer_code": "③",
                "layer_name": "砂砾岩(强风化)",
                "thickness": 27.5,
            },
        ]
    }

    payload = build_reverse_geology_payload(result, project_id=1)
    layers = payload["geologyRockSoilsReq"]

    assert [layer["name"] for layer in layers] == [
        "②砂砾岩(全风化)",
        "③砂砾岩(强风化)",
    ]
    assert layers[0]["thickness"] == 9.6
    # 接口整理不能反向修改原始 result 数据。
    assert result["geotechnical_layer_parameters"][1]["thickness"] == 9.1


def test_cultivated_soil_merge_requires_a_following_layer():
    """验证只有耕土而没有下一层时不删除数据。"""
    layers = [{"layer_code": "①", "layer_name": "耕土", "thickness": 0.5}]

    assert merge_first_cultivated_soil_layer(layers) == layers


def test_compact_result_selects_foundation_values_for_each_layer():
    """验证精简结果也逐层使用公共桩型规则，不保留两套重复参数。"""
    full_result = {
        "tasks": {
            "layer_thickness": {
                "mode": "layer_records",
                "selected_records": [
                    {
                        "layer_code": "①",
                        "layer_name": "粉质黏土",
                        "precast_side_resistance": 52,
                        "precast_tip_resistance": 2600,
                        "cast_in_place_side_resistance": 50,
                        "cast_in_place_tip_resistance": 1000,
                    },
                    {
                        "layer_code": "②",
                        "layer_name": "强风化砂砾岩",
                        "precast_side_resistance": 90,
                        "precast_tip_resistance": 3000,
                        "cast_in_place_side_resistance": 80,
                        "cast_in_place_tip_resistance": 1500,
                    },
                ],
            }
        }
    }

    layers = _compact_result(full_result)["geotechnical_layer_parameters"]
    assert layers[0]["pile_side_resistance"] == 52
    assert layers[0]["pile_tip_resistance"] == 2600
    assert layers[1]["pile_side_resistance"] == 80
    assert layers[1]["pile_tip_resistance"] == 1500
    assert "precast_side_resistance" not in layers[0]
    assert "cast_in_place_side_resistance" not in layers[1]


@pytest.mark.parametrize(
    ("record", "expected_precast", "expected_cast"),
    [
        ({"layer_name": "淤泥"}, 3500, 4000),
        ({"layer_name": "饱和湿陷性黄土"}, 3500, 4000),
        ({"layer_name": "素填土"}, 5000, 8000),
        ({"layer_name": "一般黏性土", "liquid_index": 0.8}, 5000, 8000),
        ({"layer_name": "湿陷性黄土", "liquid_index": 0.5}, 8000, 20000),
        ({"layer_name": "粉质黏土", "liquid_index": 0.2}, 15000, 50000),
        ({"layer_name": "粉土", "void_ratio": 0.95}, 5000, 8000),
        ({"layer_name": "粉土", "void_ratio": 0.8}, 8000, 20000),
        ({"layer_name": "粉土", "void_ratio": 0.7}, 15000, 50000),
        ({"layer_name": "细砂", "standard_penetration_n": 8}, 5000, 8000),
        ({"layer_name": "细砂", "standard_penetration_n": 12}, 8000, 20000),
        ({"layer_name": "细砂", "standard_penetration_n": 18}, 15000, 50000),
        ({"layer_name": "碎石土", "standard_penetration_n": 20}, None, 250000),
    ],
)
def test_horizontal_resistance_rules_cover_all_layer_branches(
    record: dict,
    expected_precast: float | None,
    expected_cast: float,
):
    """验证水平抗力比例系数各分档均能得到两套候选值。"""
    engine = ExtractionEngine.from_files("configs/layer_thickness.yaml")
    engine._apply_derived_fields([record], engine.configs[0]["derived_fields"])

    values = record["horizontal_resistance_coefficient"]
    assert values["precast"] == expected_precast
    assert values["cast_in_place"] == expected_cast


def test_pile_defaults_preserve_report_values_and_fill_missing_mapping_items():
    """验证桩参数默认值不覆盖报告值，并能补齐组合参数缺失项。"""
    engine = ExtractionEngine.from_files("configs/layer_thickness.yaml")
    record = {
        "layer_name": "淤泥",
        "uplift_coefficient": 0.85,
        "side_resistance_size_effect": 0.9,
        "tip_resistance_size_effect": 0.8,
        "negative_friction_coefficient": {"precast": 0.18},
    }

    engine._apply_derived_fields([record], engine.configs[0]["derived_fields"])

    assert record["uplift_coefficient"] == 0.85
    assert record["side_resistance_size_effect"] == 0.9
    assert record["tip_resistance_size_effect"] == 0.8
    assert record["negative_friction_coefficient"] == {
        "precast": 0.18,
        "cast_in_place": 0.25,
    }


def test_model_serializes_and_exposes_text(tmp_path: Path):
    """验证统一模型能够拼接纯文本并输出包含样式的 JSON。

    Args:
        tmp_path: pytest 提供的临时目录。
    """
    model = DocumentModel(
        source_path="example.docx",
        source_format="docx",
        parser_backend="test",
        blocks=[
            DocumentBlock("b1", "heading", "标题", [TextSpan("标题", TextStyle(bold=True))]),
            DocumentBlock("b2", "paragraph", "正文"),
        ],
    )
    # 输出目录不存在时，模型应自动创建目录。
    output = tmp_path / "nested" / "result.json"
    model.write_json(output)
    assert model.text == "标题\n正文"
    assert '"bold": true' in output.read_text(encoding="utf-8")


def test_compact_result_removes_query_details():
    """验证精简结果不会携带原文证据和全部候选数据。"""
    full_result = {
        "document": {"source_path": "report.pdf"},
        "tasks": {
            "layer_thickness": {
                "mode": "layer_records",
                "status": "success",
                "warnings": [],
                "records": [
                    {
                        "layer_code": "①",
                        "thickness_min": 0.4,
                        "thickness_max": 0.6,
                        "thickness_average": 0.54,
                        "evidence": {"text": "很长的原文"},
                    }
                ],
                "selected_records": [
                    {
                        "layer_code": "①",
                        "final_value": 0.54,
                        "thickness_min": 0.4,
                        "thickness_max": 0.6,
                        "thickness_average": 0.54,
                        "gravity_density": 19.1,
                        "cohesion": 8.9,
                        "friction_angle": 19.5,
                        "compression_modulus": 6.03,
                        "side_friction": 51.0,
                        "pile_tip_resistance": 8.9,
                        "poisson_ratio": 0.35,
                        "bearing_capacity": 120.0,
                        "evidence": {"text": "很长的原文"},
                        "table_fields": {"gravity_density": {}},
                    }
                ],
            },
            "report_sections": {
                "selected_records": [
                    {
                        "section": "hydro_meteorology",
                        "output_title": "水文气象",
                        "outline_path": [2, 1, 1],
                        "title": "气候、气象",
                        "source_title": "2.2 气候、气象 本区属暖温带季风型大陆性气候",
                        "sources": [
                            {
                                "title": "2.2 气候、气象 本区属暖温带季风型大陆性气候",
                                "matched_alias": "气候、气象",
                                "text": "水文气象正文",
                            }
                        ],
                        "text": "水文气象正文",
                    }
                ]
            },
        },
    }

    compact = _compact_result(full_result)
    task = compact["geotechnical_layer_parameters"]

    assert len(task) == 1
    expected = {
        "layer_code": "①",
        "thickness_range": {"min": 0.4, "max": 0.6, "unit": "m"},
        "average_thickness": 0.54,
        "thickness": 0.54,
        "unit": "m",
        "gravity_density": 19.1,
        "cohesion": 8.9,
        "friction_angle": 19.5,
        "compression_modulus_es1_2": 6.03,
        "side_friction_fs": 51.0,
        "pile_tip_resistance_rho_c": 8.9,
        "poisson_ratio": 0.35,
        "bearing_capacity_fak": 120.0,
    }
    assert task[0] == expected
    assert "evidence" not in task[0]
    assert "table_fields" not in task[0]
    assert "parameter_sources" not in task[0]
    assert "penetration_blow_count" not in task[0]
    assert compact["draft_content"] == {"2.2 气候、气象": "水文气象正文"}
    assert compact["regional_hydrology"] == "水文气象正文"
    assert set(compact) == {
        "geotechnical_layer_parameters",
        "seismic_parameters",
        "key_data",
        "draft_content",
        "site_geological_conditions_and_evaluation",
        "regional_hydrology",
        "conclusion_and_evaluation",
    }


def test_compact_result_uses_borehole_as_primary_profile_structure():
    """验证柱状图结果按钻孔组织，同时保留按层汇总结果。"""
    full_result = {
        "tasks": {
            "layer_thickness": {
                "mode": "layer_records",
                "selected_records": [
                    {
                        "layer_code": "①",
                        "layer_name": "粉质黏土",
                        "final_value": 2.0,
                        "observations": [
                            {
                                "layer_code": "①",
                                "layer_name": "粉质黏土",
                                "image_thickness": 1.8,
                                "bottom_depth": 1.8,
                                "bottom_elevation": 281.0,
                                "borehole_id": "F03",
                                "confidence": 0.96,
                                "evidence": {"page": 19},
                            },
                            {
                                "layer_code": "①",
                                "layer_name": "粉质黏土",
                                "image_thickness": 2.2,
                                "bottom_depth": 2.2,
                                "borehole_id": "F01",
                                "confidence": 0.94,
                                "evidence": {"page": 17},
                            },
                        ],
                    }
                ],
            }
        }
    }

    compact = _compact_result(full_result)

    layer = compact["geotechnical_layer_parameters"][0]
    assert [item["borehole_id"] for item in layer["boreholes"]] == ["F01", "F03"]
    assert layer["boreholes"][1]["thickness"] == 1.8
    assert "source_page" not in layer["boreholes"][1]
    assert "confidence" not in layer["boreholes"][1]
    assert "borehole_ids" not in layer
    assert "governing_borehole_ids" not in layer
    assert layer["thickness"] == 2.0


def test_compact_draft_content_splits_multiple_real_source_sections():
    """验证汇总目标在精简文字稿中按真实来源标题分别输出。"""
    full_result = {
        "tasks": {
            "report_sections": {
                "selected_records": [
                    {
                        "section": "seismic_action",
                        "output_title": "地震作用",
                        "text": "液化正文\n稳定性正文",
                        "sources": [
                            {
                                "title": "5.4.4 地震液化 原文首句",
                                "matched_alias": "地震液化",
                                "text": "液化正文",
                            },
                            {
                                "title": "5.4.5 地震稳定性评价 原文首句",
                                "matched_alias": "地震稳定性评价",
                                "text": "稳定性正文",
                            },
                        ],
                    }
                ]
            }
        }
    }

    compact = _compact_result(full_result)

    assert compact["draft_content"] == {
        "5.4.4 地震液化": "液化正文",
        "5.4.5 地震稳定性评价": "稳定性正文",
    }


def test_section_match_prefers_earlier_plain_numbered_conclusion():
    """验证“8 结论”不会被附件中较晚的“5. 结论”覆盖。"""
    document = DocumentModel(
        source_path="report.pdf",
        source_format="pdf",
        parser_backend="test",
        blocks=[
            DocumentBlock("main", "heading", "8 结论与建议", page=15),
            DocumentBlock("main_text", "paragraph", "主体报告结论。", page=15),
            DocumentBlock("appendix", "heading", "附件1", page=16),
            DocumentBlock("appendix_conclusion", "heading", "5. 结论与建议", page=20),
            DocumentBlock("appendix_text", "paragraph", "附件结论。", page=20),
        ],
    )

    task = ExtractionEngine.from_files("configs/report_sections.yaml").extract_all(document)["tasks"][
        "report_sections"
    ]
    conclusion = next(item for item in task["selected_records"] if item["section"] == "conclusion")

    assert conclusion["source_title"] == "8 结论与建议"
    assert conclusion["text"] == "主体报告结论。"


def test_parent_section_includes_numbered_children_with_same_heading_level():
    """验证 Word 错标相同标题级别时，父章节仍聚合编号子章节正文。"""
    document = DocumentModel(
        source_path="report.doc",
        source_format="doc",
        parser_backend="test",
        blocks=[
            DocumentBlock(
                "parent",
                "heading",
                "3.场地条件",
                page=4,
                metadata={"heading_level": 2},
            ),
            DocumentBlock(
                "child_1",
                "heading",
                "3.1水文、气象",
                page=4,
                metadata={"heading_level": 2},
            ),
            DocumentBlock("body_1", "paragraph", "水文气象正文。", page=4),
            DocumentBlock(
                "child_2",
                "heading",
                "3.2场地位置、地形及地貌",
                page=4,
                metadata={"heading_level": 2},
            ),
            DocumentBlock("body_2", "paragraph", "地形地貌正文。", page=4),
            DocumentBlock(
                "next",
                "heading",
                "4.岩土工程分析与评价",
                page=6,
                metadata={"heading_level": 2},
            ),
            DocumentBlock("next_body", "paragraph", "下一章正文。", page=6),
        ],
    )

    task = ExtractionEngine.from_files("configs/report_sections.yaml").extract_all(document)[
        "tasks"
    ]["report_sections"]
    record = next(
        item
        for item in task["selected_records"]
        if item["section"] == "site_geological_conditions"
    )

    assert "水文气象正文" in record["text"]
    assert "地形地貌正文" in record["text"]
    assert "下一章正文" not in record["text"]


def test_compact_result_applies_last_layer_adjustment():
    """验证末层保留原始揭露厚度，并输出 PRD 规定的工程厚度。"""
    full_result = {
        "tasks": {
            "layer_thickness": {
                "selected_records": [
                    {
                        "layer_code": "⑩-1",
                        "layer_name": "粉砂",
                        "geological_age": "Q3 al",
                        "maximum_exposed": 4.1,
                        "effective_value": 4.1,
                        "adjustment": 20.0,
                        "final_value": 24.1,
                    }
                ]
            }
        }
    }

    compact = _compact_result(full_result)

    selected = compact["geotechnical_layer_parameters"][0]
    assert selected["thickness"] == 24.1
    assert "thickness_range" not in selected
    assert selected["maximum_exposed_thickness"] == 4.1


def test_layer_rules_support_layer_thickness_and_unhyphenated_sublayers():
    """验证“层厚”表述和“②1”式亚层编号都能完整识别。"""
    document = DocumentModel(
        source_path="report.pdf",
        source_format="pdf",
        parser_backend="test",
        blocks=[
            DocumentBlock("h1", "heading", "2.2 地层结构和岩性特征", page=8),
            DocumentBlock(
                "l1",
                "paragraph",
                "①粉细砂(Q4el+dl)：含植物根系。层厚一般0.20～0.80m。",
                page=8,
            ),
            DocumentBlock(
                "l2",
                # 模拟 PDF 解析器把加粗的土层行误判为标题。
                "heading",
                "②1 粉土：稍密。层厚一般0.30～4.60m。",
                page=9,
            ),
            DocumentBlock(
                "l3",
                "heading",
                "②2 粉质黏土：可塑。层厚为0.60m。",
                page=9,
            ),
            DocumentBlock(
                "l4",
                "paragraph",
                "③细砂：中密。层厚一般0.40～1.90m。",
                page=9,
            ),
            DocumentBlock(
                "l5",
                "paragraph",
                "④中风化灰岩(P1)：全场区均有分布，但层厚未明确。",
                page=9,
            ),
            DocumentBlock("h2", "heading", "2.3 地下水条件及水土腐蚀性评价", page=10),
            # 下一章节的序号不是土层，章节定位必须阻止它进入候选记录。
            DocumentBlock(
                "other",
                "paragraph",
                "①根据环境类型判定土对混凝土结构的腐蚀性。",
                page=10,
            ),
        ],
    )

    task = ExtractionEngine.from_files("configs/layer_thickness.yaml").extract_all(document)[
        "tasks"
    ]["layer_thickness"]

    selected = task["selected_records"]
    assert [record["layer_code"] for record in selected] == [
        "①",
        "②-1",
        "②-2",
        "③",
        "④",
    ]
    assert selected[0]["geological_age"] == "Q4el+dl"
    assert selected[0]["thickness_min"] == 0.2
    assert selected[0]["thickness_max"] == 0.8
    assert selected[1]["thickness_min"] == 0.3
    assert selected[1]["thickness_max"] == 4.6
    assert selected[2]["thickness_exact"] == 0.6
    assert selected[4]["geological_age"] == "P1"
    assert "effective_value" not in selected[4]
    assert all("腐蚀性" not in record["evidence"]["text"] for record in task["records"])


def test_word_auto_list_labels_are_restored_for_layer_extraction():
    """验证 Word 自动编号不在正文时仍能识别土层、中文括号和“米”单位。"""
    document = DocumentModel(
        source_path="report.doc",
        source_format="doc",
        parser_backend="test",
        blocks=[
            DocumentBlock("task", "paragraph", "勘察任务为查明场区地层结构。", page=2),
            DocumentBlock("h", "heading", "3.4 地层", page=5),
            DocumentBlock(
                "l1",
                "list_item",
                "耕土：褐黄色，稍湿，分布连续，层厚0.3-0.5米。",
                page=5,
                paragraph_style=ParagraphStyle(list_level=0, list_label="①"),
            ),
            DocumentBlock(
                "l2",
                "list_item",
                "砂砾岩（全风化）：黄褐色，分布连续，层厚7.1-9.1米。",
                page=5,
                paragraph_style=ParagraphStyle(list_level=0, list_label="②"),
            ),
            DocumentBlock(
                "l3",
                "list_item",
                "砂砾岩（强风化）：浅红色，钻孔揭露最大厚度7.5米。",
                page=5,
                paragraph_style=ParagraphStyle(list_level=0, list_label="③"),
            ),
            DocumentBlock("next", "heading", "3.5 水文地质条件", page=5),
        ],
    )

    task = ExtractionEngine.from_files("configs/layer_thickness.yaml").extract_all(document)["tasks"][
        "layer_thickness"
    ]
    records = {record["layer_code"]: record for record in task["selected_records"]}

    assert list(records) == ["②", "③"]
    assert records["②"]["layer_name"] == "砂砾岩(全风化)"
    assert (records["②"]["thickness_min"], records["②"]["thickness_max"]) == (7.4, 9.6)
    assert records["②"]["final_value"] == 9.6
    assert records["③"]["layer_name"] == "砂砾岩(强风化)"
    assert records["③"]["maximum_exposed"] == 7.5


def test_hydro_meteorology_matches_real_punctuated_title():
    """验证“气象、水文”真实标题可输出完整区域水文正文。"""
    document = DocumentModel(
        source_path="report.docx",
        source_format="docx",
        parser_backend="test",
        blocks=[
            DocumentBlock("h1", "paragraph", "2、气象、水文", page=5),
            DocumentBlock(
                "p1",
                "paragraph",
                "南川区属于亚热带湿润季风气候，常年平均气温16.5℃。",
                page=5,
            ),
            DocumentBlock(
                "p2",
                "paragraph",
                "南川多年平均降雨量为1147mm，降水主要集中在5至10月。",
                page=5,
            ),
            DocumentBlock(
                "p3",
                "paragraph",
                "南川区境内溪河有91条，主要河流有大溪河、柏枝溪。",
                page=5,
            ),
            DocumentBlock("h2", "paragraph", "3、区域地质构造及地震活动特征", page=6),
        ],
    )

    full_result = ExtractionEngine.from_files("configs/report_sections.yaml").extract_all(document)
    compact = _compact_result(full_result)

    assert compact["regional_hydrology"].startswith("南川区属于亚热带湿润季风气候")
    assert "多年平均降雨量为1147mm" in compact["regional_hydrology"]
    assert "境内溪河有91条" in compact["regional_hydrology"]
    assert "区域地质构造" not in compact["regional_hydrology"]


def test_unsupported_format(tmp_path: Path):
    """验证未注册的文件格式会返回明确异常。

    Args:
        tmp_path: pytest 提供的临时目录。
    """
    source = tmp_path / "input.txt"
    source.write_text("hello", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        DocumentParser().parse(source)


def test_real_aspose_word_round_trip(tmp_path: Path):
    """使用真实 Aspose 生成并解析带标题和粗体的 Word 文档。

    Args:
        tmp_path: pytest 提供的临时目录。
    """
    try:
        aw = load_aspose("words")
    except BackendUnavailableError as exc:
        pytest.skip(str(exc))

    source = tmp_path / "styled.docx"
    native = aw.Document()
    builder = aw.DocumentBuilder(native)
    builder.ParagraphFormat.StyleIdentifier = aw.StyleIdentifier.Heading1
    builder.Writeln("Parser heading")
    builder.ParagraphFormat.StyleIdentifier = aw.StyleIdentifier.Normal
    builder.Font.Bold = True
    builder.Write("Bold")
    builder.Font.Bold = False
    builder.Writeln(" body")
    native.Save(str(source))

    parser = DocumentParser()
    word = parser.parse(source)

    assert [block.kind for block in word.blocks] == ["heading", "paragraph"]
    assert word.blocks[1].spans[0].style.bold is True


def test_opendataloader_ignores_temporary_cleanup_failure(tmp_path: Path, monkeypatch):
    """验证临时目录清理失败不会覆盖成功的解析结果。

    Args:
        tmp_path: pytest 提供的临时目录。
        monkeypatch: pytest 提供的运行时替换工具。
    """
    source = tmp_path / "input.pdf"
    source.write_bytes(b"%PDF-test")

    def fake_convert(*, input_path, output_dir, format, image_output, quiet):
        """生成最小 OpenDataLoader JSON 测试结果。

        Args:
            input_path: 输入 PDF 路径列表。
            output_dir: JSON 输出目录。
            format: 请求的输出格式。
            image_output: 图片输出方式。
            quiet: 是否关闭后端冗余日志。

        Returns:
            生成的 JSON 文件路径。
        """
        assert input_path == [str(source)]
        assert format == "json"
        assert image_output == "off"
        assert quiet is True
        output = Path(output_dir) / "result.json"
        output.write_text(
            json.dumps(
                {
                    "number of pages": 1,
                    "kids": [
                        {
                            "type": "heading",
                            "content": "标题",
                            "page number": 1,
                            "heading level": 1,
                            "font": "SimHei-Bold",
                            "font size": 18,
                            "bounding box": [10, 20, 100, 40],
                        },
                        {
                            "type": "paragraph",
                            "content": "正文",
                            "page number": 1,
                            "font": "SimSun",
                            "font size": 12,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return str(output)

    output_dir = tmp_path / "odl-output"
    output_dir.mkdir()

    def fail_cleanup(path):
        """模拟 Java 仍占用图片文件时的清理异常。

        Args:
            path: 待删除的临时目录。

        Raises:
            PermissionError: 始终模拟 Windows 文件占用错误。
        """
        assert path == output_dir
        raise PermissionError(32, "文件正由另一进程使用")

    monkeypatch.setitem(sys.modules, "opendataloader_pdf", SimpleNamespace(convert=fake_convert))
    monkeypatch.setattr("parser_engine.loaders.opendataloader.tempfile.mkdtemp", lambda **_: str(output_dir))
    monkeypatch.setattr("parser_engine.loaders.opendataloader.shutil.rmtree", fail_cleanup)

    # 本用例只验证临时目录清理，关闭与主题无关的扫描件拒绝检查。
    document = OpenDataLoaderPdfLoader(reject_scanned=False).load(source)

    assert document.text == "标题\n正文"
    assert document.pages[0].number == 1
    assert document.blocks[0].kind == "heading"
    assert document.blocks[0].spans[0].style.bold is True
    assert document.blocks[0].bbox.width == 90


def test_config_driven_layer_thickness_extraction():
    """验证公共引擎保留每个完整层号并计算有效厚度。"""
    document = DocumentModel(
        source_path="report.pdf",
        source_format="pdf",
        parser_backend="test",
        blocks=[
            DocumentBlock("b1", "heading", "场地地层结构及岩土物理力学性质", page=3),
            DocumentBlock("b2", "paragraph", "①层耕土(Q4 pd)：厚度：0.40～0.60m，平均0.54m；", page=3),
            DocumentBlock("b3", "paragraph", "②层粉土(Q4 al)：厚度1.30～4.80m，平均3.17m；", page=3),
            DocumentBlock("b4", "paragraph", "②-1层粉质黏土(Q4 al)：厚度0.80～3.20m，平均1.79m；", page=4),
            DocumentBlock("b5", "paragraph", "⑩层粉质黏土(Q3 al)：最大揭露厚度8.40m；", page=5),
            DocumentBlock("b6", "paragraph", "⑩-1层粉砂(Q3 al)：最大揭露厚度4.10m。", page=5),
        ],
    )

    result = ExtractionEngine.from_files("configs/layer_thickness.yaml").extract_all(document)
    task = result["tasks"]["layer_thickness"]

    assert task["status"] == "success"
    assert len(task["records"]) == 5
    assert [item["layer_code"] for item in task["selected_records"]] == [
        "②",
        "②-1",
        "⑩",
        "⑩-1",
    ]
    assert task["selected_records"][0]["thickness_min"] == 1.7
    assert task["selected_records"][0]["thickness_max"] == 5.4
    assert task["selected_records"][0]["final_value"] == 3.71
    assert task["selected_records"][1]["effective_value"] == 1.79
    assert task["selected_records"][-1]["final_value"] == 24.1


def test_layer_codes_above_ten_and_complete_last_layer_are_supported():
    """验证⑪以上层号可识别，完整末层厚度不会错误增加20m。"""
    document = DocumentModel(
        source_path="report.pdf",
        source_format="pdf",
        parser_backend="test",
        blocks=[
            DocumentBlock("h", "heading", "地层岩性", page=1),
            DocumentBlock("l10", "paragraph", "⑩层粉砂：层厚2.0m。", page=1),
            DocumentBlock("l11", "paragraph", "⑪层中砂：层厚3.0m。", page=1),
        ],
    )

    task = ExtractionEngine.from_files("configs/layer_thickness.yaml").extract_all(document)[
        "tasks"
    ]["layer_thickness"]

    assert [item["layer_code"] for item in task["selected_records"]] == ["⑩", "⑪"]
    assert task["selected_records"][-1]["final_value"] == 3.0
    assert "adjustment" not in task["selected_records"][-1]


def test_layer_table_average_values_and_density_conversion():
    """验证统计表按层绑定、读取平均值并完成密度单位换算。"""
    table = TableData(
        rows=8,
        columns=4,
        cells=[
            TableCell(0, 0, "项目"),
            TableCell(0, 1, "最小值"),
            TableCell(0, 2, "最大值"),
            TableCell(0, 3, "平均值 Xm"),
            TableCell(1, 0, "天然密度 ρ (g/cm³)"),
            TableCell(1, 3, "1.95"),
            TableCell(2, 0, "内聚力 C（kPa）"),
            TableCell(2, 3, "8.9"),
            TableCell(3, 0, "内摩擦角 Φ（度）"),
            TableCell(3, 3, "19.5"),
            TableCell(4, 0, "Es1-2（MPa）"),
            TableCell(4, 3, "6.03"),
            TableCell(5, 0, "fs（kPa）"),
            TableCell(5, 3, "51"),
            TableCell(6, 0, "qc（MPa）"),
            TableCell(6, 3, "3.354"),
            TableCell(7, 0, "ρc"),
            TableCell(7, 3, "8.9"),
        ],
    )
    document = DocumentModel(
        source_path="report.pdf",
        source_format="pdf",
        parser_backend="test",
        blocks=[
            DocumentBlock("toc", "list_item", "4.3 场地地层结构及岩土物理力学性质……14", page=3),
            DocumentBlock("title", "list_item", "4.3 场地地层结构及岩土物理力学性质", page=14),
            DocumentBlock("prefix1", "paragraph", "pd)：灰褐色", page=14),
            DocumentBlock("layer1", "paragraph", "①层耕土(Q4", page=14),
            DocumentBlock("depth1", "paragraph", "厚度0.40～0.60m，平均0.54m", page=14),
            DocumentBlock("prefix2", "paragraph", "al)：灰黄色", page=14),
            DocumentBlock("layer2", "paragraph", "②层粉土(Q4", page=14),
            DocumentBlock("depth2", "paragraph", "厚度1.30～4.80m，平均3.17m", page=14),
            DocumentBlock("table2", "table", table=table, page=15),
            DocumentBlock("next", "list_item", "4.4 地下水", page=24),
        ],
    )

    task = ExtractionEngine.from_files("configs/layer_thickness.yaml").extract_all(document)["tasks"][
        "layer_thickness"
    ]

    assert [record["layer_code"] for record in task["records"]] == ["①", "②"]
    assert task["records"][0].get("gravity_density") is None
    second = task["records"][1]
    assert second["gravity_density"] == 19.11
    assert second["cohesion"] == 8.9
    assert second["friction_angle"] == 19.5
    assert second["compression_modulus"] == 6.03
    assert second["side_friction"] == 51.0
    assert second["cone_tip_resistance"] == 3.354
    assert second["clay_content"] == 8.9
    assert second["pile_tip_resistance"] == 8.9
    assert second["evidence"]["table_fields"]["gravity_density"]["multiplier"] == 9.8


def test_physical_table_prefers_average_values_with_compact_unit_labels():
    """验证 Cq、φq、Es 等紧凑表头仍读取平均值而非推荐值。"""
    table = TableData(
        rows=5,
        columns=3,
        cells=[
            TableCell(0, 0, "指标名称"),
            TableCell(0, 1, "平均值"),
            TableCell(0, 2, "最大值"),
            TableCell(1, 0, "天然密度ρg/cm3"),
            TableCell(1, 1, "1.94"),
            TableCell(2, 0, "压缩模量Es1-2MPa"),
            TableCell(2, 1, "14.5"),
            TableCell(3, 0, "黏聚力CqkPa"),
            TableCell(3, 1, "12.9"),
            TableCell(4, 0, "内摩擦角φq度"),
            TableCell(4, 1, "49"),
        ],
    )
    document = DocumentModel(
        source_path="report.pdf",
        source_format="pdf",
        parser_backend="test",
        blocks=[
            DocumentBlock("h", "heading", "2.3 地层岩性分布特征", page=8),
            DocumentBlock("layer", "paragraph", "层③粉质黏土：硬塑。", page=9),
            DocumentBlock("table", "table", table=table, page=9),
        ],
    )

    record = ExtractionEngine.from_files("configs/layer_thickness.yaml").extract_all(document)[
        "tasks"
    ]["layer_thickness"]["records"][0]

    assert record["gravity_density"] == pytest.approx(19.012)
    assert record["compression_modulus"] == 14.5
    assert record["cohesion"] == 12.9
    assert record["friction_angle"] == 49.0


def test_split_interlayer_number_is_restored():
    """验证解析器拆开的“层②”与下一块“1”恢复成②-1层。"""
    document = DocumentModel(
        source_path="report.pdf",
        source_format="pdf",
        parser_backend="test",
        blocks=[
            DocumentBlock("h", "heading", "2.3 地层岩性分布特征", page=8),
            DocumentBlock("code", "paragraph", "层②", page=9),
            DocumentBlock("suffix", "paragraph", "1", page=9),
            DocumentBlock("title", "paragraph", "表2.3-3 层②1卵石 原位测试指标统计成果表", page=9),
        ],
    )

    record = ExtractionEngine.from_files("configs/layer_thickness.yaml").extract_all(document)[
        "tasks"
    ]["layer_thickness"]["records"][0]

    assert record["layer_code"] == "②-1"
    assert record["layer_name"] == "卵石"


def test_keyword_fields_prefer_conclusion_and_apply_selection():
    """验证普通字段优先搜索结论章节并执行最大值和严重程度规则。"""
    document = DocumentModel(
        source_path="report.docx",
        source_format="docx",
        parser_backend="test",
        blocks=[
            DocumentBlock("body", "paragraph", "场地地震动峰值加速度为0.30g。", page=3),
            DocumentBlock("h1", "heading", "9 结论与建议", page=20),
            DocumentBlock(
                "p1",
                "paragraph",
                "场地类别为Ⅱ类，设计地震分组为第二组，地震动峰值加速度为0.15g，反应谱特征周期为0.45s。",
                page=20,
            ),
            DocumentBlock("p2", "paragraph", "地下水埋深2.1～5.2m，另一处地下水埋深3.6～4.8m。", page=20),
            DocumentBlock("p3", "paragraph", "水对混凝土为弱腐蚀，土对钢筋为强腐蚀。", page=20),
            DocumentBlock("next", "heading", "附表", page=21),
        ],
    )

    task = ExtractionEngine.from_files("configs/report_fields.yaml").extract_all(document)["tasks"][
        "report_fields"
    ]

    assert task["values"]["seismic_peak_acceleration"] == 0.15
    assert task["values"]["site_category"] == "Ⅱ"
    assert task["values"]["earthquake_group"] == "第二组"
    assert task["values"]["corrosion"] == "中强腐蚀性"
    assert task["values"]["groundwater_depth"] == {"start": 3.6, "end": 4.8}


def test_keyword_all_selection_removes_contained_sentences():
    """验证持力层等多值字段会去掉被完整句包含的重复残句。"""
    records = [
        {"field": "description", "value": "第⑨层粉质黏土，建议做为桩端持力层。"},
        {"field": "description", "value": "建议作为桩端持力层。"},
    ]
    selected = ExtractionEngine._select_keyword_records(
        records,
        {"description": {"selection": "all", "deduplicate_contained": True}},
    )

    assert [item["value"] for item in selected] == ["第⑨层粉质黏土，建议做为桩端持力层。"]


def test_foundation_treatment_ignores_standard_title():
    """验证规范引用不会被误报为场地存在湿陷性黄土。"""
    document = DocumentModel(
        source_path="report.doc",
        source_format="doc",
        parser_backend="test",
        blocks=[
            DocumentBlock(
                "reference",
                "paragraph",
                "（7）《湿陷性黄土地区建筑规范》GB 50025-2004；",
                page=3,
            )
        ],
    )

    task = ExtractionEngine.from_files("configs/report_fields.yaml").extract_all(document)[
        "tasks"
    ]["report_fields"]

    assert "foundation_treatment" not in task["values"]


def test_section_content_keeps_structure_and_stops_at_next_heading():
    """验证章节内容抽取保留块证据且不会越过下一个同级标题。"""
    document = DocumentModel(
        source_path="report.docx",
        source_format="docx",
        parser_backend="test",
        blocks=[
            DocumentBlock("h1", "heading", "3.1 区域地质构造", page=4),
            DocumentBlock("p1", "paragraph", "区域内构造稳定。", page=4),
            DocumentBlock("h2", "heading", "3.2 地下水条件", page=5),
            DocumentBlock("p2", "paragraph", "地下水埋深较大。", page=5),
        ],
    )

    task = ExtractionEngine.from_files("configs/report_sections.yaml").extract_all(document)["tasks"][
        "report_sections"
    ]

    regional = next(
        item for item in task["records"] if item["section"] == "regional_geological_structure"
    )
    groundwater = next(item for item in task["records"] if item["section"] == "groundwater_conditions")
    meteorology = next(item for item in task["records"] if item["section"] == "hydro_meteorology")
    assert regional["text"] == "区域内构造稳定。"
    assert regional["content"][0]["block_id"] == "p1"
    assert regional["outline_path"] == [2, 1, 3]
    assert regional["source_outline"][-1]["title"] == "3.1 区域地质构造"
    assert groundwater["text"] == "地下水埋深较大。"
    assert meteorology["source"] == "placeholder"
    assert meteorology["placeholder"]["provider"] == "baidu"


def test_section_content_prefers_direct_subheading_over_parent_title():
    """验证上级标题包含别名时，优先抽取真正的同名小节。"""
    document = DocumentModel(
        source_path="report.pdf",
        source_format="pdf",
        parser_backend="test",
        blocks=[
            DocumentBlock("h1", "heading", "2、自然地理与气候、气象", page=1),
            DocumentBlock("p1", "paragraph", "2.1 自然地理正文", page=1),
            DocumentBlock("h2", "list_item", "2.2 气候、气象 气候正文首句", page=1),
            DocumentBlock("p2", "paragraph", "气候正文续句", page=1),
            DocumentBlock("h3", "list_item", "2.3 地形地貌", page=1),
        ],
    )
    config = {
        "name": "sections",
        "mode": "section_content",
        "sections": [
            {
                "key": "hydro_meteorology",
                "output_title": "水文气象",
                "outline_path": [2, 1, 1],
                "aliases": ["气候、气象"],
            }
        ],
    }

    result = ExtractionEngine([config]).extract_all(document)
    selected = result["tasks"]["sections"]["selected_records"][0]

    assert selected["source_title"].startswith("2.2")
    assert selected["text"] == "气候正文首句\n气候正文续句"
    assert "自然地理" not in selected["text"]


def test_section_content_combines_configured_alias_groups():
    """验证一个目标章节可按配置合并多个独立来源小节。"""
    document = DocumentModel(
        source_path="report.pdf",
        source_format="pdf",
        parser_backend="test",
        blocks=[
            DocumentBlock("h1", "list_item", "5.4.1 建筑场地抗震设防烈度 7 度", page=1),
            DocumentBlock("p1", "paragraph", "设计地震分组为第三组。", page=1),
            DocumentBlock("h2", "list_item", "5.4.2 场地类别 Ⅲ类", page=1),
            DocumentBlock("h3", "list_item", "5.4.3 场地特征周期", page=2),
            DocumentBlock("p3", "paragraph", "调整后特征周期为 0.65s。", page=2),
            DocumentBlock("h4", "list_item", "5.4.4 地震液化", page=2),
        ],
    )
    config = {
        "name": "sections",
        "mode": "section_content",
        "sections": [
            {
                "key": "seismic_site_division",
                "output_title": "抗震地段划分",
                "alias_groups": [
                    ["建筑场地抗震设防烈度", "抗震设防烈度"],
                    ["场地特征周期"],
                ],
            }
        ],
    }

    selected = ExtractionEngine([config]).extract_all(document)["tasks"]["sections"][
        "selected_records"
    ][0]

    assert selected["source_titles"] == [
        "5.4.1 建筑场地抗震设防烈度 7 度",
        "5.4.3 场地特征周期",
    ]
    assert "7 度" in selected["text"]
    assert "0.65s" in selected["text"]
    assert "场地类别" not in selected["text"]


def test_section_boundary_handles_missing_parent_heading():
    """验证下一父节标题缺失时，编号仍能阻止章节正文越界。"""
    document = DocumentModel(
        source_path="report.pdf",
        source_format="pdf",
        parser_backend="test",
        blocks=[
            DocumentBlock("h1", "heading", "5.3 不良地质作用及特殊性岩土", page=1),
            DocumentBlock("p1", "paragraph", "本场地无不良地质作用。", page=1),
            DocumentBlock("h2", "list_item", "5.3.2 特殊性岩土 软土", page=1),
            DocumentBlock("h3", "list_item", "5.4.1 建筑场地抗震设防烈度 7 度", page=2),
        ],
    )
    config = {
        "name": "sections",
        "mode": "section_content",
        "sections": [
            {"key": "adverse", "aliases": ["不良地质作用及特殊性岩土"]}
        ],
    }

    selected = ExtractionEngine([config]).extract_all(document)["tasks"]["sections"][
        "selected_records"
    ][0]

    assert "软土" in selected["text"]
    assert "抗震设防烈度" not in selected["text"]


def test_section_number_does_not_treat_decimal_value_as_heading():
    """验证段首高程小数不会被误判成章节编号。"""
    document = DocumentModel(
        source_path="report.pdf",
        source_format="pdf",
        parser_backend="test",
        blocks=[
            DocumentBlock("h1", "list_item", "4.1 地形、地貌 地面标高最大值 39.34m，最小值", page=1),
            DocumentBlock("p1", "paragraph", "36.21m，地表相对高差 3.13m。", page=1),
            DocumentBlock("h2", "list_item", "4.2 地下水特征", page=1),
        ],
    )
    config = {
        "name": "sections",
        "mode": "section_content",
        "sections": [{"key": "terrain", "aliases": ["地形、地貌"]}],
    }

    selected = ExtractionEngine([config]).extract_all(document)["tasks"]["sections"][
        "selected_records"
    ][0]

    assert selected["text"] == "地面标高最大值 39.34m，最小值\n36.21m，地表相对高差 3.13m。"


def test_section_boundary_does_not_treat_chart_rows_as_headings():
    """验证月份图表中的整数行不会提前截断章节正文。"""
    document = DocumentModel(
        source_path="report.pdf",
        source_format="pdf",
        parser_backend="test",
        blocks=[
            DocumentBlock("h1", "list_item", "2.2 气候、气象 气候正文", page=1),
            DocumentBlock("p1", "list_item", "1 35.2 5", page=1),
            DocumentBlock("p2", "list_item", "2 52.7 6.2", page=1),
            DocumentBlock("p3", "list_item", "3 103.4 12.7", page=1),
            DocumentBlock("p4", "paragraph", "全年平均气温 13.5℃。", page=1),
            DocumentBlock("h2", "heading", "3、区域地质条件", page=2),
        ],
    )
    config = {
        "name": "sections",
        "mode": "section_content",
        "sections": [{"key": "climate", "aliases": ["气候、气象"]}],
    }

    selected = ExtractionEngine([config]).extract_all(document)["tasks"]["sections"][
        "selected_records"
    ][0]

    assert "1 35.2 5" in selected["text"]
    assert "3 103.4 12.7" in selected["text"]
    assert "全年平均气温 13.5℃" in selected["text"]
    assert "区域地质条件" not in selected["text"]


def test_section_content_trims_embedded_next_parent_heading():
    """验证粘在末段后的下一父节标题会被裁掉，末段正文仍保留。"""
    document = DocumentModel(
        source_path="report.pdf",
        source_format="pdf",
        parser_backend="test",
        blocks=[
            DocumentBlock("h1", "heading", "5.3 不良地质作用及特殊性岩土", page=1),
            DocumentBlock(
                "p1",
                "paragraph",
                "该层不得作为持力层。 5.4 场地和地基的地震效应",
                page=1,
            ),
            DocumentBlock("h2", "list_item", "5.4.1 建筑场地抗震设防烈度", page=1),
        ],
    )
    config = {
        "name": "sections",
        "mode": "section_content",
        "sections": [
            {"key": "adverse", "aliases": ["不良地质作用及特殊性岩土"]}
        ],
    }

    selected = ExtractionEngine([config]).extract_all(document)["tasks"]["sections"][
        "selected_records"
    ][0]

    assert selected["text"] == "该层不得作为持力层。"


def test_matrix_table_merges_bearing_capacity_and_derived_values():
    """验证推荐值矩阵表按层号合并，并由配置计算派生参数。"""
    table = TableData(
        rows=3,
        columns=5,
        cells=[
            TableCell(0, 0, "层号"),
            TableCell(0, 1, "岩土名称"),
            TableCell(0, 2, "承载力特征值"),
            TableCell(0, 3, "压缩模量建议值"),
            TableCell(0, 4, "泊松比"),
            TableCell(1, 2, "fak(kPa)"),
            TableCell(1, 3, "Es1-2(MPa)"),
            TableCell(2, 0, "②"),
            TableCell(2, 1, "粉土"),
            TableCell(2, 2, "170"),
            TableCell(2, 3, "8.4"),
            TableCell(2, 4, ""),
        ],
    )
    document = DocumentModel(
        source_path="report.pdf",
        source_format="pdf",
        parser_backend="test",
        blocks=[
            DocumentBlock("h1", "heading", "4.3 场地地层结构及岩土物理力学性质", page=4),
            DocumentBlock("l1", "paragraph", "②层粉土(Q4 al)：厚度1.3～4.8m，平均3.17m。", page=4),
            DocumentBlock("h2", "heading", "4.4 地基承载力", page=5),
            DocumentBlock("t1", "table", "各层建议值 承载力特征值", table=table, page=5),
        ],
    )

    task = ExtractionEngine.from_files("configs/layer_thickness.yaml").extract_all(document)["tasks"][
        "layer_thickness"
    ]
    record = task["selected_records"][0]

    assert record["bearing_capacity"] == 170.0
    assert record["compression_modulus_recommended"] == 8.4
    assert record["poisson_ratio"] == 0.35
    assert record["width_bearing_coefficient"] == 0.3
    assert record["depth_bearing_coefficient"] == 1.5
    assert record["seismic_bearing_coefficient"] == 1.3


def test_matrix_table_supports_soil_layer_number_and_deformation_modulus():
    """验证“土层编号”和 ``Eo=数值`` 格式可以合并到对应岩土层。"""
    table = TableData(
        rows=4,
        columns=7,
        cells=[
            TableCell(0, 0, "土层编号"),
            TableCell(0, 1, "岩土名称"),
            TableCell(0, 2, "重力密度(kN/m3)"),
            TableCell(0, 3, "压缩模量 Es(MPa)（变形模量 Eo(MPa)）"),
            TableCell(0, 4, "内聚力 C(kPa)"),
            TableCell(0, 5, "摩擦角 ψ(°)"),
            TableCell(0, 6, "承载力特征值 fak(kPa)"),
            TableCell(1, 0, "①"),
            TableCell(1, 1, "耕土"),
            TableCell(1, 2, "-"),
            TableCell(1, 3, "-"),
            TableCell(1, 4, "-"),
            TableCell(1, 5, "-"),
            TableCell(1, 6, "-"),
            TableCell(2, 0, "②"),
            TableCell(2, 1, "砂砾岩（全风化）"),
            TableCell(2, 2, "20.0"),
            TableCell(2, 3, "Eo=21.0"),
            TableCell(2, 4, "-"),
            TableCell(2, 5, "-"),
            TableCell(2, 6, "230"),
            TableCell(3, 0, "③"),
            TableCell(3, 1, "砂砾岩（强风化）"),
            TableCell(3, 2, "23.0"),
            TableCell(3, 3, "Eo=35.0"),
            TableCell(3, 4, "-"),
            TableCell(3, 5, "-"),
            TableCell(3, 6, "380"),
        ],
    )
    document = DocumentModel(
        source_path="report.doc",
        source_format="doc",
        parser_backend="test",
        blocks=[
            DocumentBlock("h1", "heading", "3.2 地层岩性", page=5),
            DocumentBlock("l1", "paragraph", "①耕土：层厚0.3～0.5米。", page=5),
            DocumentBlock("l2", "paragraph", "②砂砾岩（全风化）：层厚7.1～9.1米。", page=5),
            DocumentBlock("l3", "paragraph", "③砂砾岩（强风化）：最大揭露厚度7.5米。", page=5),
            DocumentBlock("caption", "paragraph", "承载力特征值 fak 及变形参数如下表", page=6),
            DocumentBlock("table", "table", table=table, page=6),
        ],
    )

    selected = ExtractionEngine.from_files("configs/layer_thickness.yaml").extract_all(document)[
        "tasks"
    ]["layer_thickness"]["selected_records"]
    by_code = {record["layer_code"]: record for record in selected}

    # 首层耕土会按业务规则并入下一层，但矩阵表参数仍应绑定在正确层号上。
    assert set(by_code) == {"②", "③"}
    assert by_code["②"]["gravity_density_recommended"] == 20.0
    assert by_code["②"]["compression_modulus_recommended"] == 21.0
    assert by_code["②"]["bearing_capacity"] == 230.0
    assert by_code["③"]["gravity_density_recommended"] == 23.0
    assert by_code["③"]["compression_modulus_recommended"] == 35.0
    assert by_code["③"]["bearing_capacity"] == 380.0


@pytest.mark.parametrize(
    ("layer_name", "expected_cohesion", "expected_angle", "expected_modulus"),
    [
        ("一般粘性土", 30, 18, 8),
        ("淤泥质黏土", 10, 7, 3),
        ("红黏土", 55, 8, 15),
        ("粗砂", 2, 35, 30),
        ("中砂", 2, 35, 30),
        ("细砂", 6, 32, 15),
        ("粉砂", 6, 32, 15),
        ("全风化花岗岩", 28, 20, 6),
        ("强风化花岗岩", 30, 25, 30),
        ("中风化灰岩", 100, 30, 1000),
    ],
)
def test_missing_strength_parameters_use_layer_type_defaults(
    layer_name: str,
    expected_cohesion: float,
    expected_angle: float,
    expected_modulus: float,
):
    """验证缺少强度参数时按配置中的土类型规则补值。

    Args:
        layer_name: 用于匹配规则的土层名称。
        expected_cohesion: 预期黏聚力，单位为 kPa。
        expected_angle: 预期内摩擦角，单位为度。
        expected_modulus: 预期压缩模量，单位为 MPa。
    """
    engine = ExtractionEngine.from_files("configs/layer_thickness.yaml")
    record = {"layer_name": layer_name}

    engine._apply_derived_fields([record], engine.configs[0]["derived_fields"])

    assert record["cohesion"] == expected_cohesion
    assert record["friction_angle"] == expected_angle
    assert record["compression_modulus"] == expected_modulus
    assert record["derived_field_sources"] == {
        "cohesion": "土类型缺省规则",
        "friction_angle": "土类型缺省规则",
        "compression_modulus": "土类型缺省规则",
    }


def test_liquefied_fine_sand_uses_prd_bearing_correction_coefficients():
    """验证液化粉砂/细砂按 PRD 取 ηb=0、ηd=1。"""
    engine = ExtractionEngine.from_files("configs/layer_thickness.yaml")
    record = {
        "layer_name": "粉砂",
        "liquefaction_is_liquefied": True,
        "liquefaction_reduction_coefficient": 0.666667,
    }

    engine._apply_derived_fields([record], engine.configs[0]["derived_fields"])

    assert record["width_bearing_coefficient"] == 0
    assert record["depth_bearing_coefficient"] == 1.0


def test_liquefied_fine_sand_with_reduction_factor_one_still_uses_zero_one():
    """验证已液化但折减系数恰好为1时，ηb/ηd仍按液化规则取0/1。"""
    engine = ExtractionEngine.from_files("configs/layer_thickness.yaml")
    record = {
        "layer_name": "细砂",
        "liquefaction_is_liquefied": True,
        "liquefaction_reduction_coefficient": 1.0,
    }

    engine._apply_derived_fields([record], engine.configs[0]["derived_fields"])

    assert record["width_bearing_coefficient"] == 0
    assert record["depth_bearing_coefficient"] == 1.0


def test_non_liquefied_fine_sand_keeps_original_bearing_correction_coefficients():
    """验证无液化粉砂/细砂仍保持 ηb=2、ηd=3。"""
    engine = ExtractionEngine.from_files("configs/layer_thickness.yaml")
    record = {
        "layer_name": "细砂",
        "liquefaction_is_liquefied": False,
        "liquefaction_reduction_coefficient": 1.0,
    }

    engine._apply_derived_fields([record], engine.configs[0]["derived_fields"])

    assert record["width_bearing_coefficient"] == 2.0
    assert record["depth_bearing_coefficient"] == 3.0


def test_layer_type_defaults_do_not_override_report_or_recommended_values():
    """验证原文平均值和推荐值均优先于土类型缺省规则。"""
    engine = ExtractionEngine.from_files("configs/layer_thickness.yaml")
    extracted = {
        "layer_name": "淤泥质黏土",
        "cohesion": 21.1,
        "friction_angle": 9.2,
        "compression_modulus": 3.01,
    }
    recommended = {
        "layer_name": "淤泥质黏土",
        "cohesion_recommended": 18.0,
        "friction_angle_recommended": 8.5,
        "compression_modulus_recommended": 2.8,
    }

    engine._apply_derived_fields(
        [extracted, recommended], engine.configs[0]["derived_fields"]
    )

    assert extracted["cohesion"] == 21.1
    assert extracted["friction_angle"] == 9.2
    assert extracted["compression_modulus"] == 3.01
    assert "cohesion" not in recommended
    assert "friction_angle" not in recommended
    assert "compression_modulus" not in recommended


def test_unspecified_gravel_rock_does_not_use_sand_defaults():
    """验证未在业务表中定义的砂砾岩不会套用粗砂或细砂缺省参数。"""
    engine = ExtractionEngine.from_files("configs/layer_thickness.yaml")
    record = {"layer_name": "砂砾岩"}

    engine._apply_derived_fields([record], engine.configs[0]["derived_fields"])

    assert "cohesion" not in record
    assert "friction_angle" not in record
    assert "compression_modulus" not in record


def test_recommendation_table_uses_caption_context_and_combined_layer_column():
    """验证表名在前置段落、层号名称同列时仍能合并推荐指标。"""
    table = TableData(
        rows=3,
        columns=14,
        cells=[
            TableCell(0, 0, "指标名称\n层号 名称", row_span=2),
            TableCell(0, 2, "天然密度ρ(g/cm3)", row_span=2),
            TableCell(0, 8, "压缩模量 Es1-2(MPa)", row_span=2),
            TableCell(0, 11, "直接快剪", column_span=2),
            TableCell(1, 11, "黏聚力 ck(kPa)"),
            TableCell(1, 12, "内摩擦角 φk(°)"),
            TableCell(0, 13, "承载力特征值 fak(kPa)", row_span=2),
            TableCell(2, 0, "层①粉质黏土"),
            TableCell(2, 2, "1.93"),
            TableCell(2, 8, "7.0"),
            TableCell(2, 11, "20"),
            TableCell(2, 12, "12"),
            TableCell(2, 13, "155"),
        ],
    )
    document = DocumentModel(
        source_path="report.pdf",
        source_format="pdf",
        parser_backend="test",
        blocks=[
            DocumentBlock("h1", "heading", "2.3 地层岩性分布特征", page=8),
            DocumentBlock("l1", "paragraph", "层①粉质黏土：厚度1.0～2.0m，平均1.5m。", page=8),
            DocumentBlock("caption", "paragraph", "表4.2 主要物理及工程特性指标推荐值", page=13),
            DocumentBlock("t1", "table", table=table, page=13),
        ],
    )

    task = ExtractionEngine.from_files("configs/layer_thickness.yaml").extract_all(document)[
        "tasks"
    ]["layer_thickness"]
    selected = task["selected_records"][0]

    assert selected["layer_code"] == "①"
    assert selected["layer_name"] == "粉质黏土"
    assert selected["gravity_density_recommended"] == pytest.approx(18.914)
    assert selected["cohesion_recommended"] == 20.0
    assert selected["friction_angle_recommended"] == 12.0
    assert selected["compression_modulus_recommended"] == 7.0
    assert selected["bearing_capacity"] == 155.0


def test_one_row_parameter_header_extracts_all_prd_layer_fields():
    """验证“土层指标”单行表头能够提取五项主要物理力学参数。"""
    table = TableData(
        rows=2,
        columns=6,
        cells=[
            TableCell(0, 0, "土层 指标"),
            TableCell(0, 1, "天然重度 γ（kN m3）"),
            TableCell(0, 2, "压缩模量 Es1-2（Mpa）"),
            TableCell(0, 3, "黏聚力 C（kPa）"),
            TableCell(0, 4, "内摩擦角 Φ（度）"),
            TableCell(0, 5, "承载力特征值 fak(kPa)"),
            TableCell(1, 0, "②1层粉土"),
            TableCell(1, 1, "17.1"),
            TableCell(1, 2, "5.62"),
            TableCell(1, 3, "6.9"),
            TableCell(1, 4, "19.0"),
            TableCell(1, 5, "130"),
        ],
    )
    document = DocumentModel(
        source_path="report.pdf",
        source_format="pdf",
        parser_backend="test",
        blocks=[
            DocumentBlock("h1", "heading", "2.2 地层结构和岩性特征", page=8),
            DocumentBlock(
                "l1",
                "paragraph",
                "②1粉土：稍密。层厚一般0.30～4.60m。",
                page=9,
            ),
            DocumentBlock("h2", "heading", "3、岩土工程评价", page=14),
            DocumentBlock(
                "caption",
                "paragraph",
                "地基土主要物理力学性质指标参考值及承载力特征值",
                page=14,
            ),
            DocumentBlock("table", "table", table=table, page=14),
        ],
    )

    selected = ExtractionEngine.from_files("configs/layer_thickness.yaml").extract_all(document)[
        "tasks"
    ]["layer_thickness"]["selected_records"][0]

    assert selected["layer_code"] == "②-1"
    assert selected["gravity_density_recommended"] == 17.1
    assert selected["compression_modulus_recommended"] == 5.62
    assert selected["cohesion_recommended"] == 6.9
    assert selected["friction_angle_recommended"] == 19.0
    assert selected["bearing_capacity"] == 130.0


def test_selected_layer_evidence_is_json_serializable():
    """验证图片厚度和表格指标合并后证据不会形成循环引用。"""
    selected = ExtractionEngine._select_records(
        [
            {
                "layer_code": "①",
                "thickness_average": 1.5,
                "evidence": {"source_type": "image_ocr"},
            },
            {
                "layer_code": "①",
                "gravity_density": 19.0,
                "evidence": {"source_type": "table"},
            },
        ],
        {
            "group_by": "layer_code",
            "value_priority": ["thickness_average"],
            "operator": "minimum",
        },
    )

    assert selected[0]["gravity_density"] == 19.0
    json.dumps(selected, ensure_ascii=False)


def test_report_fields_keep_multiple_site_categories_and_liquefaction_conclusion():
    """验证分区场地类别不丢值，并识别“不存在地震液化问题”。"""
    document = DocumentModel(
        source_path="report.pdf",
        source_format="pdf",
        parser_backend="test",
        blocks=[
            DocumentBlock("h", "heading", "8 结论与建议", page=15),
            DocumentBlock(
                "p",
                "paragraph",
                "F01建筑场地类别为I0类，F02建筑场地类别为II类，场地不存在地震液化问题。",
                page=15,
            ),
        ],
    )

    task = ExtractionEngine.from_files("configs/report_fields.yaml").extract_all(document)["tasks"][
        "report_fields"
    ]

    assert task["values"]["site_category"] == ["Ⅰ0", "Ⅱ"]
    assert task["values"]["liquefaction_status"] == "no_liquefaction"


def test_report_fields_handle_combined_seismic_wording_without_false_positive():
    """验证综合地震章节优先取基本值，并识别烈度、多个类别和否定液化结论。"""
    document = DocumentModel(
        source_path="report.docx",
        source_format="docx",
        parser_backend="test",
        blocks=[
            DocumentBlock("h1", "heading", "6.1 地震效应", page=20),
            DocumentBlock(
                "p1",
                "paragraph",
                "峰值加速度为0.05g，基本地震动加速度反应谱特征周期0.35s。",
                page=20,
            ),
            DocumentBlock(
                "p2",
                "paragraph",
                "建筑场地类别分别为I0、I1类、Ⅱ类，可不考虑地震液化和震陷问题。",
                page=20,
            ),
            DocumentBlock("h2", "heading", "6.2 场地稳定性", page=21),
            DocumentBlock(
                "p3",
                "paragraph",
                "地震动峰值加速度小于0.09g，本场地地震烈度为6度。",
                page=21,
            ),
        ],
    )

    task = ExtractionEngine.from_files("configs/report_fields.yaml").extract_all(document)["tasks"][
        "report_fields"
    ]

    assert task["values"]["seismic_peak_acceleration"] == 0.05
    assert task["values"]["seismic_intensity"] == 6
    assert task["values"]["characteristic_period"] == 0.35
    assert task["values"]["site_category"] == ["Ⅰ0", "Ⅰ1", "Ⅱ"]
    assert task["values"]["liquefaction_status"] == "no_liquefaction"


def test_report_sections_support_combined_and_alternative_real_titles():
    """验证真实章节异名及合并的“地震效应”章节均能映射到 PRD 目标。"""
    document = DocumentModel(
        source_path="report.docx",
        source_format="docx",
        parser_backend="test",
        blocks=[
            DocumentBlock("h4", "heading", "4 场地环境及工程地质条件", page=12),
            DocumentBlock("p4", "paragraph", "场地工程地质正文。", page=12),
            DocumentBlock("h44", "heading", "4.4 水文地质条件", page=15),
            DocumentBlock("p44", "paragraph", "各钻孔均未见稳定水位。", page=15),
            DocumentBlock("h5", "heading", "5 岩土参数统计", page=18),
            DocumentBlock("p5", "paragraph", "岩土参数正文。", page=18),
            DocumentBlock("h6", "heading", "6 场地稳定性、适宜性评价", page=20),
            DocumentBlock("p6", "paragraph", "场地基本稳定。", page=20),
            DocumentBlock("h61", "heading", "6.1 地震效应", page=20),
            DocumentBlock("p61", "paragraph", "峰值加速度、场地类别及液化结论。", page=20),
            DocumentBlock("h74", "heading", "7.4 水、土壤对建筑材料腐蚀性评价", page=22),
            DocumentBlock("p74", "paragraph", "场地土具有微腐蚀性。", page=22),
            DocumentBlock("h8", "heading", "8 风机基础岩土工程分析评价", page=23),
            DocumentBlock("p8", "paragraph", "天然地基与桩基础评价。", page=23),
            DocumentBlock("h9", "heading", "9 场内道路工程地质条件", page=27),
        ],
    )

    task = ExtractionEngine.from_files("configs/report_sections.yaml").extract_all(document)["tasks"][
        "report_sections"
    ]
    records = {record["section"]: record for record in task["selected_records"]}

    for key in (
        "groundwater_conditions",
        "soil_corrosion_evaluation",
        "geotechnical_parameters",
        "site_stability_evaluation",
        "foundation_evaluation",
        "seismic_site_division",
        "soil_and_site_category",
        "seismic_action",
        "site_geological_conditions",
    ):
        assert records[key]["status"] == "extracted"
    assert records["seismic_site_division"]["source_title"] == "6.1 地震效应"
    assert records["soil_and_site_category"]["source_title"] == "6.1 地震效应"
    assert records["seismic_action"]["source_title"] == "6.1 地震效应"


def test_matrix_table_supports_merged_pile_headers():
    """验证公共表格方法可展开灌注桩和预制桩的合并表头。"""
    table = TableData(
        rows=3,
        columns=6,
        cells=[
            TableCell(0, 0, "层号", row_span=2),
            TableCell(0, 1, "岩土名称", row_span=2),
            TableCell(0, 2, "钻孔灌注桩", column_span=2),
            TableCell(0, 4, "预制桩", column_span=2),
            TableCell(1, 2, "侧阻力标准值qsik"),
            TableCell(1, 3, "端阻力标准值qpk"),
            TableCell(1, 4, "侧阻力标准值qsik"),
            TableCell(1, 5, "端阻力标准值qpk"),
            TableCell(2, 0, "⑦"),
            TableCell(2, 1, "粉质黏土"),
            TableCell(2, 2, "76"),
            TableCell(2, 3, "1000"),
            TableCell(2, 4, "78"),
            TableCell(2, 5, "2600"),
        ],
    )
    config = ExtractionEngine.from_files("configs/layer_thickness.yaml").configs[0]
    pile_config = config["matrix_tables"][1]
    engine = ExtractionEngine([config])

    records = engine.extract_table_records(
        DocumentBlock("pile", "table", "钻孔灌注桩 预制桩", table=table, page=11),
        pile_config,
    )

    assert records[0]["cast_in_place_side_resistance"] == 76.0
    assert records[0]["cast_in_place_tip_resistance"] == 1000.0
    assert records[0]["precast_side_resistance"] == 78.0


def test_matrix_table_supports_layer_name_header_spanning_code_and_name_columns():
    """合并的“土层名称”表头应仍能区分层号列和名称列。"""
    config = ExtractionEngine.from_files("configs/layer_thickness.yaml").configs[0]
    pile_config = config["matrix_tables"][1]
    block = DocumentBlock(
        id="pile-table",
        kind="table",
        page=1,
        table=TableData(
            rows=3,
            columns=6,
            cells=[
                TableCell(0, 0, "土层名称", column_span=2),
                TableCell(0, 2, "钻孔灌注桩", column_span=2),
                TableCell(0, 4, "预制桩", column_span=2),
                TableCell(1, 0, "土层名称", column_span=2),
                TableCell(1, 2, "桩的侧阻力标准值 qsik（kPa）"),
                TableCell(1, 3, "桩的端阻力标准值 qpk（kPa）"),
                TableCell(1, 4, "桩的侧阻力标准值 qsik（kPa）"),
                TableCell(1, 5, "桩的端阻力标准值 qpk（kPa）"),
                TableCell(2, 0, "⑦"),
                TableCell(2, 1, "粉质黏土"),
                TableCell(2, 2, "76"),
                TableCell(2, 3, "1000"),
                TableCell(2, 4, "78"),
                TableCell(2, 5, "2600"),
            ],
        ),
    )

    records = ExtractionEngine([config]).extract_table_records(block, pile_config)

    assert records[0]["layer_code"] == "⑦"
    assert records[0]["layer_name"] == "粉质黏土"
    assert records[0]["precast_side_resistance"] == 78.0
    assert records[0]["precast_tip_resistance"] == 2600.0


def test_derived_rule_uses_recommended_value_when_statistical_value_is_missing():
    """水平抗力等派生规则应兼容推荐表字段。"""
    records = [
        {
            "layer_code": "②",
            "layer_name": "粉土",
            "void_ratio_recommended": 0.8,
        }
    ]
    config = ExtractionEngine.from_files("configs/layer_thickness.yaml").configs[0]

    ExtractionEngine([config])._apply_derived_fields(records, config["derived_fields"])

    assert records[0]["horizontal_resistance_coefficient"] == {
        "precast": 8000,
        "cast_in_place": 20000,
    }


def test_image_fallback_uses_common_selection_rules(tmp_path: Path, monkeypatch):
    """验证图片结果可以复用公共分组、选择和最后一层调整规则。

    Args:
        tmp_path: pytest 提供的临时目录。
        monkeypatch: pytest 提供的运行时替换工具。
    """
    class FakeVisionClient:
        """返回固定钻孔柱状图结果的测试视觉客户端。"""

        def recognize(self, image_path: Path, prompt: str):
            """返回两层模拟识别数据。

            Args:
                image_path: 模拟页面图片路径。
                prompt: 配置中的识别提示词。

            Returns:
                模拟的结构化钻孔识别结果。
            """
            assert image_path.is_file()
            assert "钻孔柱状图" in prompt
            return {
                "borehole_id": "F01",
                "layers": [
                    {
                        "layer_code": "①",
                        "layer_name": "粉质黏土",
                        "bottom_depth": 1.7,
                        "thickness": 1.7,
                        "confidence": 0.98,
                    },
                    {
                        "layer_code": "②",
                        "layer_name": "砂砾",
                        "bottom_depth": 14.0,
                        "thickness": 12.3,
                        "confidence": 0.96,
                    },
                ],
            }

    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-test")
    page_image = tmp_path / "page_0002.png"
    page_image.write_bytes(b"png")
    document = DocumentModel(
        source_path=str(source),
        source_format="pdf",
        parser_backend="test",
        blocks=[DocumentBlock("b1", "image", page=2)],
    )
    recognizer = BoreholeImageRecognizer(FakeVisionClient(), tmp_path / "assets")
    monkeypatch.setattr(
        recognizer,
        "_render_pages",
        lambda *_args: [(2, page_image)],
    )

    result = ExtractionEngine.from_files(
        "configs/layer_thickness.yaml",
        image_recognizer=recognizer,
    ).extract_all(document)
    selected = result["tasks"]["layer_thickness"]["selected_records"]

    assert [item["layer_code"] for item in selected] == ["①", "②"]
    assert selected[0]["borehole_ids"] == ["F01"]
    assert selected[1]["effective_field"] == "image_average"
    assert selected[1]["final_value"] == 32.3
    assert selected[1]["observations"][0]["depth_validation"] is True


def test_rapidocr_client_builds_layers_from_coordinates(tmp_path: Path):
    """验证免费 OCR 客户端能按文字框位置组织层号和厚度。

    Args:
        tmp_path: pytest 提供的临时目录。
    """

    def box(x: float, y: float, width: float = 40, height: float = 20):
        """创建模拟 OCR 四点坐标框。

        Args:
            x: 文字框中心横坐标。
            y: 文字框中心纵坐标。
            width: 文字框宽度。
            height: 文字框高度。

        Returns:
            RapidOCR 使用的四点坐标列表。
        """
        return [
            [x - width / 2, y - height / 2],
            [x + width / 2, y - height / 2],
            [x + width / 2, y + height / 2],
            [x - width / 2, y + height / 2],
        ]

    class FakeImage:
        """仅提供图片尺寸的测试对象。"""

        shape = (1200, 1000, 3)

    output = SimpleNamespace(
        img=FakeImage(),
        boxes=[
            box(500, 50),
            box(330, 100),
            box(290, 100),
            box(150, 300),
            box(500, 330, 180),
            box(330, 400),
            box(290, 400),
            box(150, 500),
            box(500, 530, 180),
            box(330, 600),
            box(290, 600),
        ],
        txts=("F01", "分层厚度", "层底深度", "①", "杂填土", "1.70", "1.70", "②", "卵石", "12.30", "14.00"),
        scores=(0.99,) * 11,
    )

    class FakeOCREngine:
        """返回固定坐标识别结果的 OCR 引擎。"""

        def __call__(self, image_path: str):
            """返回模拟 RapidOCR 输出。

            Args:
                image_path: 待识别图片路径。

            Returns:
                模拟的 RapidOCR 输出对象。
            """
            assert Path(image_path).name == "page.png"
            return output

    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"png")
    result = RapidOCRClient(engine=FakeOCREngine()).recognize(image_path, "unused")

    assert result["borehole_id"] == "F01"
    assert [layer["layer_code"] for layer in result["layers"]] == ["①", "②"]
    assert [layer["thickness"] for layer in result["layers"]] == [1.7, 12.3]
    assert [layer["bottom_depth"] for layer in result["layers"]] == [1.7, 14.0]
    assert [layer["layer_name"] for layer in result["layers"]] == ["杂填土", "卵石"]


def test_borehole_aggregation_keeps_source_and_governing_ids():
    """验证多钻孔汇总后保留来源钻孔和控制厚度钻孔编号。"""
    records = [
        {
            "layer_code": "①",
            "main_layer_code": "①",
            "layer_name": "粉质黏土",
            "image_thickness": 2.5,
            "borehole_id": "F10",
            "evidence": {"page": 20},
        },
        {
            "layer_code": "①",
            "main_layer_code": "①",
            "layer_name": "粉质黏土",
            "image_thickness": 1.8,
            "borehole_id": "F03",
            "evidence": {"page": 17},
        },
    ]

    result = BoreholeImageRecognizer._aggregate(records, "minimum")[0]

    assert result["borehole_ids"] == ["F03", "F10"]
    assert result["governing_borehole_ids"] == ["F03"]


def test_borehole_aggregation_excludes_depth_validation_failures():
    """验证深度校验失败的 OCR 厚度保留追溯，但不参与平均值。"""
    records = [
        {
            "layer_code": "①",
            "main_layer_code": "①",
            "layer_name": "粉质黏土",
            "image_thickness": 2.0,
            "depth_validation": True,
            "borehole_id": "F01",
            "evidence": {"page": 10},
        },
        {
            "layer_code": "①",
            "main_layer_code": "①",
            "layer_name": "粉质黏土",
            "image_thickness": 20.0,
            "depth_validation": False,
            "borehole_id": "F02",
            "evidence": {"page": 11},
        },
    ]

    result = BoreholeImageRecognizer._aggregate(records, "average")[0]

    assert result["image_average"] == 2.0
    assert result["observation_count"] == 1
    assert result["borehole_ids"] == ["F01"]
    assert len(result["observations"]) == 2


@pytest.mark.parametrize(("operator", "expected"), [("minimum", 2.0), ("maximum", 3.0)])
def test_borehole_aggregation_excludes_invalid_values_for_min_and_max(
    operator: str,
    expected: float,
):
    """验证深度校验失败值不会污染最小值或最大值。"""
    records = [
        {
            "layer_code": "①",
            "main_layer_code": "①",
            "image_thickness": 2.0,
            "depth_validation": True,
            "borehole_id": "F01",
            "evidence": {"page": 10},
        },
        {
            "layer_code": "①",
            "main_layer_code": "①",
            "image_thickness": 3.0,
            "depth_validation": True,
            "borehole_id": "F02",
            "evidence": {"page": 11},
        },
        {
            "layer_code": "①",
            "main_layer_code": "①",
            "image_thickness": 99.0 if operator == "maximum" else 0.1,
            "depth_validation": False,
            "borehole_id": "F03",
            "evidence": {"page": 12},
        },
    ]

    result = BoreholeImageRecognizer._aggregate(records, operator)[0]

    assert result["image_average"] == expected
    assert result["observation_count"] == 2
    assert "F03" not in result["borehole_ids"]


def test_image_fallback_detects_partial_layer_results():
    """验证文本层数少于原文声明数量时会触发图片补漏。"""
    blocks = [
        DocumentBlock(
            "p1",
            "paragraph",
            "根据勘察结果，将主层划分为 2 层，夹层 1 层。",
            page=1,
        )
    ]
    config = {
        "enabled": True,
        "completeness": {
            "expected_count_patterns": [
                {
                    "pattern": r"主层划分为\s*(?P<main>\d+)\s*层.*?夹层\s*(?P<interlayer>\d+)\s*层",
                    "sum_groups": ["main", "interlayer"],
                }
            ]
        },
    }

    assert ExtractionEngine._needs_image_fallback(blocks, [{}, {}], config) is True
    assert ExtractionEngine._needs_image_fallback(blocks, [{}, {}, {}], config) is False


def test_image_fallback_detects_diagram_after_partial_text_results():
    """验证未声明总层数时，柱状图关键词仍会触发 OCR 补漏。"""
    blocks = [DocumentBlock("layer", "paragraph", "①层耕土：层厚0.5m。", page=2)]
    document = DocumentModel(
        source_path="report.pdf",
        source_format="pdf",
        parser_backend="test",
        blocks=[
            *blocks,
            DocumentBlock("diagram", "paragraph", "附图：钻孔柱状图", page=10),
        ],
        pages=[PageInfo(number=2, text_characters=20), PageInfo(number=10, text_characters=8)],
    )
    config = {
        "enabled": True,
        "fallback_when_diagram_pages_exist": True,
        "page_keywords": ["钻孔柱状图"],
    }

    assert ExtractionEngine._needs_image_fallback(
        blocks,
        [{"layer_code": "①"}],
        config,
        document=document,
    ) is True


def test_missing_image_recognizer_marks_partial_layer_task():
    """验证需要图片补漏但未提供识别器时，任务状态不是成功。"""
    document = DocumentModel(
        source_path="report.pdf",
        source_format="pdf",
        parser_backend="test",
        blocks=[
            DocumentBlock("h", "heading", "地层岩性", page=1),
            DocumentBlock("layer", "paragraph", "①层耕土：层厚0.5m。", page=1),
            DocumentBlock("diagram", "paragraph", "附图：钻孔柱状图", page=10),
        ],
        pages=[PageInfo(number=1, text_characters=30), PageInfo(number=10, text_characters=8)],
    )

    task = ExtractionEngine.from_files("configs/layer_thickness.yaml").extract_all(document)[
        "tasks"
    ]["layer_thickness"]

    assert task["status"] == "partial"
    assert any("没有传入图片识别器" in warning for warning in task["warnings"])


def test_opendataloader_bounding_box_ignores_invalid_coordinates():
    """验证第三方 PDF 后端坐标异常时只丢弃坐标，不中断文档解析。"""
    assert OpenDataLoaderPdfLoader._bounding_box(["10", "20", "30", "40"]) is not None
    assert OpenDataLoaderPdfLoader._bounding_box(["10", "", "30", "40"]) is None
    assert OpenDataLoaderPdfLoader._bounding_box(["10", None, "30", "40"]) is None


def test_vision_client_extracts_first_valid_json_object(tmp_path: Path, monkeypatch):
    """验证视觉服务带说明文字或非法花括号片段时仍能取得合法 JSON。"""

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            body = {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "识别说明 {这不是JSON}。\n"
                                "```json\n"
                                '{"borehole_id":"F01","layers":[]}\n'
                                "```"
                            )
                        }
                    }
                ]
            }
            return json.dumps(body, ensure_ascii=False).encode("utf-8")

    monkeypatch.setattr(
        "parser_engine.image_recognition.urllib.request.urlopen",
        lambda *_args, **_kwargs: FakeResponse(),
    )

    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"png")
    client = OpenAICompatibleVisionClient(
        "http://vision.test/v1/chat/completions",
        "test-model",
    )

    result = client.recognize(image_path, "识别钻孔柱状图")

    assert result == {"borehole_id": "F01", "layers": []}


def test_opendataloader_scanned_pdf_policy_is_enforced():
    """验证 OpenDataLoader 与 Aspose 使用相同的扫描件拒绝策略。"""
    document = DocumentModel(
        source_path="scan.pdf",
        source_format="pdf",
        parser_backend="opendataloader.pdf",
        pages=[PageInfo(number=1, text_characters=0), PageInfo(number=2, text_characters=3)],
    )

    with pytest.raises(ScannedPdfNotSupportedError):
        OpenDataLoaderPdfLoader()._check_scanned_document(document)


def test_section_fallback_is_partial_not_success():
    """验证配置兜底文字不能被计为原文章节抽取成功。"""
    engine = ExtractionEngine(
        [
            {
                "name": "sections",
                "mode": "section_content",
                "sections": [
                    {
                        "key": "safety",
                        "output_title": "稳定性评价",
                        "aliases": ["稳定性评价"],
                        "fallback_text": "场地稳定。",
                    }
                ],
            }
        ]
    )
    document = DocumentModel("report.pdf", "pdf", "test")

    task = engine.extract_all(document)["tasks"]["sections"]

    assert task["status"] == "partial"
    assert task["selected_records"][0]["status"] == "defaulted"
    assert any("不属于原文抽取" in warning for warning in task["warnings"])


def test_foundation_treatment_ignores_conditional_survey_requirement():
    """验证任务书中条件性列举的湿陷性黄土不会被认定为实际场地问题。"""
    document = DocumentModel(
        source_path="report.pdf",
        source_format="pdf",
        parser_backend="test",
        blocks=[
            DocumentBlock(
                "requirement",
                "list_item",
                "查清场地有无滑坡、泥石流等不良地质作用，",
                page=2,
            ),
            DocumentBlock(
                "requirement_continued",
                "paragraph",
                "包括湿陷性黄土、膨胀土，如有应提出防治建议。",
                page=2,
            ),
            DocumentBlock(
                "conclusion",
                "paragraph",
                "拟建场区内无特殊性岩土地层分布。",
                page=20,
            ),
        ],
    )

    task = ExtractionEngine.from_files("configs/report_fields.yaml").extract_all(document)[
        "tasks"
    ]["report_fields"]

    assert "foundation_treatment" not in task["values"]


def test_config_validation_rejects_invalid_alias_groups_and_regex():
    """验证错误的章节分组和正则会在加载配置时立即报错。"""
    with pytest.raises(ValueError, match="alias_groups"):
        ExtractionEngine(
            [
                {
                    "name": "bad_sections",
                    "mode": "section_content",
                    "sections": [{"key": "x", "alias_groups": ["不是二维列表"]}],
                }
            ]
        )

    with pytest.raises(ValueError, match="正则"):
        ExtractionEngine(
            [
                {
                    "name": "bad_fields",
                    "mode": "keyword_fields",
                    "fields": {"x": {"patterns": ["("]}},
                }
            ]
        )


def test_rapidocr_soil_names_can_be_overridden_by_config():
    """验证 OCR 岩土名称规则可以由配置覆盖。"""
    client = RapidOCRClient(engine=object())

    client.configure({"soil_name_patterns": ["冻土", "盐渍土"]})

    assert client._soil_pattern.search("岩性为冻土") is not None
    assert client._soil_pattern.search("岩性为粉土") is None


def test_borehole_pages_include_low_text_image_attachments():
    """验证图片数量缺失时仍能发现低文本的柱状图附件页。"""
    document = DocumentModel(
        source_path="report.pdf",
        source_format="pdf",
        parser_backend="test",
        blocks=[DocumentBlock("p10", "paragraph", "详见钻孔柱状图", page=10)],
        pages=[
            PageInfo(10, text_characters=500, image_count=0),
            PageInfo(17, text_characters=66, image_count=0),
            PageInfo(18, text_characters=105, image_count=0),
        ],
    )

    pages = BoreholeImageRecognizer._candidate_pages(
        document,
        {
            "page_keywords": ["钻孔柱状图"],
            "use_image_blocks": True,
            "image_page_max_text": 150,
        },
    )

    assert pages == [10, 17, 18]


def test_page_ocr_text_reuses_common_layer_extraction(tmp_path: Path, monkeypatch):
    """验证编码异常 PDF 的页面 OCR 仍复用公共土层规则。"""

    class FakeTextClient:
        """返回带 Unicode 下标层号的普通页面 OCR 结果。"""

        def configure(self, config):
            """接受公共 OCR 配置。

            Args:
                config: OCR 配置字典。
            """
            assert isinstance(config, dict)

        def read_lines(self, image_path):
            """返回模拟的两行跨行土层描述。

            Args:
                image_path: 模拟页面图片路径。

            Returns:
                OCR 行及页面宽度。
            """
            assert Path(image_path).is_file()
            return (
                [
                    {"text": "②₁粉土：稍密。层厚一般0.30～4.60m。", "confidence": 0.99},
                    {"text": "层底深度一般1.00～5.60m。", "confidence": 0.98},
                ],
                1000.0,
            )

    source = tmp_path / "encoded.pdf"
    source.write_bytes(b"%PDF-test")
    page_image = tmp_path / "page_0009.png"
    page_image.write_bytes(b"png")
    document = DocumentModel(
        source_path=str(source),
        source_format="pdf",
        parser_backend="test",
        pages=[PageInfo(number=9, text_characters=20)],
    )
    recognizer = BoreholeImageRecognizer(FakeTextClient(), tmp_path / "assets")
    monkeypatch.setattr(recognizer, "_render_pages", lambda *_args: [(9, page_image)])

    blocks = recognizer.recognize_text_blocks(
        document,
        {"pages": [9], "ocr": {}, "render": {"dpi": 350}},
    )
    engine = ExtractionEngine.from_files("configs/layer_thickness.yaml")
    records = engine._extract_records(blocks, engine.configs[0])

    assert blocks[0].text.startswith("②1粉土")
    assert records[0]["layer_code"] == "②-1"
    assert records[0]["layer_name"] == "粉土"
    assert records[0]["thickness_min"] == 0.3
    assert records[0]["thickness_max"] == 4.6


def test_layer_catalog_corrects_image_sublayers_and_rock_layers():
    """验证正文目录能修正柱状图中遗漏的亚层下标和岩层主编号。"""
    document = DocumentModel(
        source_path="report.pdf",
        source_format="pdf",
        parser_backend="test",
        blocks=[
            DocumentBlock(
                "catalog",
                "paragraph",
                "层②粉质黏土；表 3 层②1卵石；层④中等风化砂岩；层④1强风化砂岩。",
            )
        ],
    )
    catalog = BoreholeImageRecognizer._extract_layer_catalog(document)
    records = [
        {"layer_code": "②", "main_layer_code": "②", "layer_name": "卵石", "description": ""},
        {"layer_code": "③", "main_layer_code": "③", "layer_name": "中等风化砂岩", "description": ""},
    ]

    BoreholeImageRecognizer._apply_layer_catalog(records, catalog)

    assert catalog["②-1"] == "卵石"
    assert records[0]["layer_code"] == "②-1"
    assert records[1]["layer_code"] == "④"


def test_layer_sequence_repairs_a_missing_main_layer():
    """验证上下层之间唯一缺失的同名主层可以被顺序规则恢复。"""
    catalog = {
        "②": "粉质黏土",
        "③": "粉质黏土",
        "④-1": "强风化砂岩",
    }
    records = [
        {"layer_code": "②", "main_layer_code": "②", "layer_name": "粉质黏土"},
        {"layer_code": "②", "main_layer_code": "②", "layer_name": "粉质黏土"},
        {"layer_code": "④-1", "main_layer_code": "④", "layer_name": "强风化砂岩"},
    ]

    BoreholeImageRecognizer._repair_layer_sequence(records, catalog)

    assert records[1]["layer_code"] == "③"


def test_compact_result_discovers_tasks_by_mode_instead_of_fixed_name():
    """验证精简结果能读取任意配置名称的公共抽取任务。"""
    full_result = {
        "tasks": {
            "custom_layers": {
                "mode": "layer_records",
                "selected_records": [{"layer_code": "①", "final_value": 1.2}],
            },
            "custom_fields": {
                "mode": "keyword_fields",
                "values": {"site_category": "Ⅱ类"},
            },
            "custom_sections": {
                "mode": "section_content",
                "selected_records": [],
            },
        }
    }

    compact = _compact_result(full_result)

    assert compact["geotechnical_layer_parameters"][0]["layer_code"] == "①"
    assert compact["seismic_parameters"]["site_category"] == "Ⅱ类"


def test_real_report_regression_for_layers_and_section_boundaries():
    """使用仓库内真实报告验证完整土层和关键章节边界。"""
    project_root = Path(__file__).resolve().parents[1]
    reports = list((project_root / "地勘报告案例").glob("*140MW*.pdf"))
    if not reports:
        pytest.skip("仓库中没有真实回归报告")
    pytest.importorskip("opendataloader_pdf")

    document = DocumentParser(ParserConfig(pdf_backend="opendataloader")).parse(reports[0])
    result = ExtractionEngine.from_files(
        [
            project_root / "configs" / "layer_thickness.yaml",
            project_root / "configs" / "report_fields.yaml",
            project_root / "configs" / "report_sections.yaml",
        ]
    ).extract_all(document)
    layers = result["tasks"]["layer_thickness"]["selected_records"]
    sections = {
        item["section"]: item
        for item in result["tasks"]["report_sections"]["selected_records"]
    }

    assert [item["layer_code"] for item in layers] == [
        "②",
        "②-1",
        "③",
        "③-1",
        "③-2",
        "④",
        "④-1",
        "⑤",
        "⑥",
        "⑥-1",
        "⑥-2",
        "⑦",
        "⑧",
        "⑧-1",
        "⑨",
        "⑩",
        "⑩-1",
    ]
    first = layers[0]
    assert (first["thickness_min"], first["thickness_max"], first["final_value"]) == (
        1.7,
        5.4,
        3.71,
    )
    assert layers[-1]["maximum_exposed"] == 4.1
    assert layers[-1]["final_value"] == 24.1
    assert sections["regional_geological_structure"]["source_title"].startswith(
        "3.2 构造地质条件"
    )
    assert "水文地质条件" not in sections["regional_geological_structure"]["text"]
    assert sections["soil_corrosion_evaluation"]["source_title"].startswith(
        "5.2.2 地基土腐蚀性评价"
    )
    assert "5.4 场地和地基的地震效应" not in sections["adverse_geology"]["text"]
    assert len(sections["seismic_site_division"]["sources"]) == 2
    # “地基均匀性/稳定性评价”不是“土类型及场地类别”的同义章节，
    # 这里只应采用真实的“场地类别”来源。
    assert len(sections["soil_and_site_category"]["sources"]) == 1
    assert len(sections["seismic_action"]["sources"]) == 2

    # 最终文字稿按报告真实标题输出；逐节检查代表章节末尾的内容，防止
    # 图表数字、粘连标题或别名优先级导致正文被提前截断或串入下一节。
    draft = _compact_result(result)["draft_content"]
    expected_section_content = {
        "2.2 气候、气象": "年平均日照总时数为 2506 小时",
        "4.1 地形、地貌": "地下埋藏物",
        "3.2 构造地质条件": "区域稳定",
        "4.3 场地地层结构及岩土物理力学性质": "N63.5 标贯及双桥静力触探统计表 表 18",
        "4.2 地下水特征": "分层捣实回填封孔",
        "5.2.2 地基土腐蚀性评价": "中等腐蚀性",
        "5.3 不良地质作用及特殊性岩土": "不得作为地基持力层",
        "5.1 场地的稳定性及适宜性评价": "各拟建建筑物",
        "5.6 地基基础方案分析": "钻芯法检测",
        "5.4.1 建筑场地抗震设防烈度": "第三组",
        "5.4.3 场地特征周期": "0.65s",
        "5.4.2 场地类别": "Ⅲ 类",
        "5.4.4 地震液化": "不具液化性",
        "5.4.5 地震稳定性评价": "无影响",
        "6、结论与建议": "重新对场地进行勘察",
    }
    for title, expected_text in expected_section_content.items():
        assert title in draft
        assert expected_text in draft[title]
    # 5.6 是父章节，子节内容应归入真实父标题，不再伪造多个来源标题。
    assert "均匀地基" in draft["5.6 地基基础方案分析"]
    assert "稳定性较差" in draft["5.6 地基基础方案分析"]
