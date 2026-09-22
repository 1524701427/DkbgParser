from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from parser_engine import (
    BoreholeImageRecognizer,
    DocumentParser,
    ParserConfig,
    ParserError,
    RapidOCRClient,
    extract_document,
    write_reverse_geology_payload,
)


# 项目根目录。所有相对路径均以该目录为基准解析。
PROJECT_ROOT = Path(__file__).resolve().parent

# 项目固定业务配置。保持原有配置文件及加载顺序不变。
CONFIG_PATHS = (
    PROJECT_ROOT / "configs/layer_thickness.yaml",
    PROJECT_ROOT / "configs/report_fields.yaml",
    PROJECT_ROOT / "configs/report_sections.yaml",
)

# PRD 2.2.3：地震参数字段及其日志展示名称。
SEISMIC_FIELDS = {
    "seismic_peak_acceleration": "地震动峰值加速度",
    "seismic_intensity": "地震烈度",
    "characteristic_period": "反应谱特征周期",
    "site_category": "场地类别",
    "earthquake_group": "设计地震分组",
}

# PRD 2.2.4：关键业务字段及其日志展示名称。
KEY_FIELDS = {
    "corrosion": "水土腐蚀性",
    "foundation_treatment": "地基处理关键字",
    "bearing_layer_description": "持力层描述",
    "groundwater_depth": "地下水埋深",
}

# PRD 2.2.5：需要统计的文字稿目标章节。
DRAFT_SECTIONS = (
    "hydro_meteorology",
    "regional_terrain",
    "regional_geological_structure",
    "stratigraphic_lithology",
    "groundwater_conditions",
    "soil_corrosion_evaluation",
    "adverse_geology",
    "geotechnical_parameters",
    "site_stability_evaluation",
    "foundation_evaluation",
    "seismic_site_division",
    "soil_and_site_category",
    "seismic_action",
    "conclusion",
)

logger = logging.getLogger(__name__)


def _has_result_value(value: object) -> bool:
    """判断抽取字段是否包含有效业务值。

    Args:
        value: 待检查的字段值。

    Returns:
        字段包含有效值时返回 ``True``，否则返回 ``False``。
    """
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


def _resolve_project_path(path: str | Path) -> Path:
    """将路径解析为绝对路径，相对路径统一以项目根目录为基准。

    Args:
        path: 待解析的文件或目录路径。

    Returns:
        规范化后的绝对路径。
    """
    resolved_path = Path(path)
    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path
    return resolved_path.resolve()


def _configure_logging() -> None:
    """初始化控制台日志，并尽量统一标准输出编码为 UTF-8。

    Windows 在输出重定向场景下可能使用 GBK，因此保留原逻辑，对支持
    ``reconfigure`` 的标准流显式设置 UTF-8，避免中文日志乱码。
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def _build_output_paths(input_path: Path, output_dir: Path) -> tuple[Path, Path, Path, Path]:
    """根据输入文件名生成四类固定输出文件路径。

    Args:
        input_path: 已解析为绝对路径的输入报告。
        output_dir: 已解析为绝对路径的输出目录。

    Returns:
        依次返回精简结果、查询明细、字段说明和接口映射文件路径。
    """
    stem = input_path.stem
    return (
        output_dir / f"{stem}.json",
        output_dir / f"{stem}_details.json",
        output_dir / f"{stem}_fields.json",
        output_dir / f"{stem}_callback.json",
    )


def _tasks_for_mode(
    tasks: dict[str, Any],
    mode: str,
    legacy_name: str,
) -> list[dict]:
    """按公共抽取模式获取任务，并兼容旧任务名称。

    优先按 ``mode`` 查找全部同类任务；若当前数据仍使用早期固定任务名，
    则回退到 ``legacy_name``。该行为与原实现一致。

    Args:
        tasks: 完整任务字典。
        mode: 当前公共抽取模式名称。
        legacy_name: 早期版本使用的固定任务名称。

    Returns:
        匹配当前模式的任务列表。
    """
    matched = [
        task
        for task in tasks.values()
        if isinstance(task, dict) and task.get("mode") == mode
    ]
    if matched:
        return matched

    legacy = tasks.get(legacy_name)
    return [legacy] if isinstance(legacy, dict) else []


def _log_prd_summary(result: dict) -> None:
    """按 PRD 2.2.2～2.2.8 输出业务抽取汇总。

    Args:
        result: 完整抽取结果，包含土层、字段和章节任务。
    """
    tasks = result.get("tasks", {})

    # 日志按公共 mode 汇总，不绑定某一个 YAML 文件名。这样即使后续拆分或
    # 新增同类型配置文件，汇总逻辑仍能自动覆盖对应任务。
    layer_tasks = _tasks_for_mode(tasks, "layer_records", "layer_thickness")
    field_tasks = _tasks_for_mode(tasks, "keyword_fields", "report_fields")
    section_tasks = _tasks_for_mode(tasks, "section_content", "report_sections")

    # 多个 keyword_fields 任务的 values 按原有顺序合并；后出现的同名字段
    # 覆盖前面的值，保持原实现中的 dict.update 行为不变。
    field_values: dict = {}
    for task in field_tasks:
        field_values.update(task.get("values", {}))

    # 章节以 section 作为唯一键。若多个任务命中同一 section，后出现的记录
    # 覆盖前面的记录，与原字典推导行为一致。
    section_records = {
        str(item.get("section")): item
        for task in section_tasks
        for item in task.get("selected_records", [])
    }

    def section_available(key: str) -> bool:
        """判断指定章节是否确实从报告原文抽取成功。

        Args:
            key: 章节配置键。

        Returns:
            状态为 ``extracted`` 且正文包含有效值时返回 ``True``。
        """
        record = section_records.get(key, {})

        # 只有真实来自报告原文的章节才计入完成数。配置兜底和待复核内容
        # 均不能计为抽取成功。
        return (
            record.get("status") == "extracted"
            and _has_result_value(record.get("text"))
        )

    # 土层数统计最终选择记录；钻孔数从逐孔 observations 中去重，避免将
    # “候选命中数量”误报为最终业务结果数量。
    layer_count = sum(
        len(task.get("selected_records", []))
        for task in layer_tasks
    )
    borehole_ids = {
        str(observation["borehole_id"])
        for task in layer_tasks
        for record in task.get("selected_records", [])
        for observation in record.get("observations", [])
        if observation.get("borehole_id")
    }

    missing_seismic_fields = [
        label
        for key, label in SEISMIC_FIELDS.items()
        if not _has_result_value(field_values.get(key))
    ]
    seismic_count = len(SEISMIC_FIELDS) - len(missing_seismic_fields)

    missing_key_fields = [
        label
        for key, label in KEY_FIELDS.items()
        if not _has_result_value(field_values.get(key))
    ]
    key_count = len(KEY_FIELDS) - len(missing_key_fields)

    draft_count = sum(section_available(key) for key in DRAFT_SECTIONS)
    site_count = sum(
        section_available(key)
        for key in ("site_geological_conditions", "site_stability_evaluation")
    )

    logger.info("抽取业务汇总（按 PRD）：")
    logger.info(
        "  2.2.2 岩土层厚度及承载力：%d 个钻孔，%d 个汇总层",
        len(borehole_ids),
        layer_count,
    )

    if missing_seismic_fields:
        logger.info(
            "  2.2.3 地震参数：%d/%d 项，原文未明确：%s",
            seismic_count,
            len(SEISMIC_FIELDS),
            "、".join(missing_seismic_fields),
        )
    else:
        logger.info(
            "  2.2.3 地震参数：%d/%d 项",
            seismic_count,
            len(SEISMIC_FIELDS),
        )

    if missing_key_fields:
        logger.info(
            "  2.2.4 关键数据：%d/%d 项，原文未给出有效值：%s",
            key_count,
            len(KEY_FIELDS),
            "、".join(missing_key_fields),
        )
    else:
        logger.info(
            "  2.2.4 关键数据：%d/%d 项",
            key_count,
            len(KEY_FIELDS),
        )

    # 一个真实章节可能覆盖多个 PRD 目标，例如“地震效应”同时包含场地类别、
    # 抗震地段和液化结论，因此这里统计的是目标项，不是唯一章节标题数量。
    logger.info(
        "  2.2.5 文字稿内容：%d/%d 个目标项",
        draft_count,
        len(DRAFT_SECTIONS),
    )
    logger.info("  2.2.6 场区地质条件与评价：%d/2 项", site_count)
    logger.info(
        "  2.2.7 区域水文：%s",
        "已识别" if section_available("hydro_meteorology") else "未识别",
    )
    logger.info(
        "  2.2.8 结论与评价：%s",
        "已识别" if section_available("conclusion") else "未识别",
    )


def _build_image_recognizer(
    enable_ocr: bool,
    output_dir: Path,
) -> BoreholeImageRecognizer | None:
    """按开关创建钻孔附图 OCR 识别器。

    Args:
        enable_ocr: 是否启用 OCR 兜底识别。
        output_dir: 当前业务输出目录。

    Returns:
        启用 OCR 时返回识别器实例，否则返回 ``None``。
    """
    if not enable_ocr:
        return None

    return BoreholeImageRecognizer(
        RapidOCRClient(),
        output_dir=output_dir / "assets",
    )


def _log_mapping_warnings(output_path: Path, callback_payload: dict) -> None:
    """检查接口映射结果，并输出需要人工补充编码的提示。

    Args:
        output_path: 精简业务结果 JSON 路径。
        callback_payload: 已生成的接口请求体。
    """
    key_data = json.loads(output_path.read_text(encoding="utf-8")).get("key_data", {})

    if (
        key_data.get("foundation_treatment")
        and callback_payload.get("handleKeyword") is None
    ):
        logger.warning("接口映射 handleKeyword 为空：请传入实际枚举编码")

    if (
        "中强" in str(key_data.get("water_soil_corrosion") or "")
        and callback_payload.get("waterSoilErosion") is None
    ):
        logger.warning("接口映射 waterSoilErosion 为空：请确认中腐蚀2或强腐蚀3")


def parse_document(
    input_path: str | Path,
    output_path: str | Path | None = None,
    pdf_backend: str = "opendataloader",
    allow_scanned: bool = False,
) -> dict:
    """解析 Word 或文本型 PDF，并返回稳定的 JSON 数据结构。

    Args:
        input_path: 输入文件路径，支持 .doc、.docx、.pdf。
        output_path: JSON 输出路径；为 None 时仅返回解析结果，不写文件。
        pdf_backend: PDF 解析后端，可选 "aspose" 或 "opendataloader"。
        allow_scanned: 是否允许扫描型 PDF。

    Returns:
        解析后的字典数据。

    Raises:
        ParserError: 文档解析失败。
        FileNotFoundError: 输入文件不存在。
        ValueError: 参数或文件格式不合法。
    """
    input_path = Path(input_path)

    # 文档解析与业务抽取保持解耦。该函数只负责生成统一文档模型，既便于
    # 单独调试不同 PDF 后端，也便于上层复用统一解析结果。
    config = ParserConfig(
        pdf_backend=pdf_backend,
        reject_scanned_pdf=not allow_scanned,
    )
    document = DocumentParser(config).parse(input_path)
    result = document.to_dict()

    # 保持原有行为：仅在调用方明确提供输出路径时写文件。
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return result


def main(
    input_file: str | Path,
    output_directory: str | Path = "output",
    *,
    pdf_backend: str = "opendataloader",
    enable_ocr: bool = True,
    project_id: int = 0,
    geology_id: int = 0,
    layer_ids: dict[str, int] | None = None,
    handle_keyword_codes: dict[str, int] | None = None,
    ambiguous_corrosion_code: int | None = None,
) -> dict:
    """解析地勘报告，并生成业务结果、查询明细和接口映射 JSON。

    Args:
        input_file: 输入报告路径，支持 DOC、DOCX 和文本型 PDF。相对路径以项目
            根目录为基准。
        output_directory: 输出目录。相对路径以项目根目录为基准。
        pdf_backend: PDF 解析后端，可选 ``opendataloader`` 或 ``aspose``。
        enable_ocr: 正文或表格结果不足时，是否使用本地 RapidOCR 识别附图。
        project_id: 业务系统项目ID；不是报告抽取字段。
        geology_id: 业务系统地质数据ID；新增数据可传0。
        layer_ids: 已有土层ID映射，键可用完整层名、层号或层名。
        handle_keyword_codes: 地基处理关键字到接口枚举的映射。接口文档没有
            给出编码，确认后传入，例如 ``{"岩溶": 1}``。
        ambiguous_corrosion_code: “中强腐蚀性”对应的接口编码，只能传2或3。

    Returns:
        已写入 ``*_callback.json`` 的接口请求体。

    Raises:
        ParserError: 文档解析或抽取失败。
        FileNotFoundError: 输入文件或配置文件不存在。
        ValueError: 接口映射参数不合法。
        RuntimeError: OCR 或底层解析组件执行失败。
    """
    # 相对路径仍以项目根目录为基准，保持原有路径解析规则不变。
    input_path = _resolve_project_path(input_file)
    output_dir = _resolve_project_path(output_directory)

    (
        output_path,
        details_output_path,
        fields_output_path,
        callback_output_path,
    ) = _build_output_paths(input_path, output_dir)

    _configure_logging()

    try:
        logger.info("程序启动")
        logger.info("输入文件：%s", input_path)
        logger.info(
            "PDF 后端：%s；OCR 兜底：%s",
            pdf_backend,
            "开启" if enable_ocr else "关闭",
        )

        # OCR 仅作为兜底方案。正文和表格已有结果时，底层业务逻辑仍会避免
        # 重复识别图片；这里只负责按原开关决定是否构造识别器。
        recognizer = _build_image_recognizer(enable_ocr, output_dir)

        result = extract_document(
            input_path=input_path,
            config_paths=list(CONFIG_PATHS),
            pdf_backend=pdf_backend,
            output_path=output_path,
            details_output_path=details_output_path,
            field_descriptions_output_path=fields_output_path,
            image_recognizer=recognizer,
        )

        # 精简业务结果与查询明细保持职责分离。接口映射继续读取 result.json，
        # 避免误用 extract_document 返回的完整查询明细结构。
        callback_payload = write_reverse_geology_payload(
            output_path,
            callback_output_path,
            project_id=project_id,
            geology_id=geology_id,
            layer_ids=layer_ids,
            handle_keyword_codes=handle_keyword_codes,
            ambiguous_corrosion_code=ambiguous_corrosion_code,
        )

        _log_prd_summary(result)
        logger.info("精简结果：%s", output_path)
        logger.info("查询明细：%s", details_output_path)
        logger.info("字段说明：%s", fields_output_path)
        logger.info("接口映射结果：%s", callback_output_path)

        _log_mapping_warnings(output_path, callback_payload)
        return callback_payload

    except (ParserError, FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("解析失败：%s", exc)

        # main() 保持可复用函数语义：记录错误后继续抛出原异常，使调用方能够
        # 按原异常类型处理；脚本直接运行时也会自然产生非零退出码。
        raise


if __name__ == "__main__":
    # 直接运行时只需修改 input_file；其余参数有需要时再显式传入。
    main(input_file=r"地勘报告案例\安徽宣城沈村镇50MW风电项目.doc")
