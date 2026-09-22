from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import yaml

from .engine import DocumentParser, ParserConfig
from .models import DocumentBlock, DocumentModel


logger = logging.getLogger(__name__)


class ExtractionEngine:
    """读取 YAML 配置并对统一文档模型执行公共抽取流程。"""

    # 公共抽取模式集中维护，避免校验逻辑中重复硬编码。
    _SUPPORTED_MODES = frozenset({"layer_records", "keyword_fields", "section_content"})

    def __init__(self, configs: list[dict[str, Any]], image_recognizer: Any | None = None) -> None:
        """初始化抽取引擎。

        Args:
            configs: 已加载并通过基础校验的抽取配置列表。
            image_recognizer: 可选的钻孔柱状图识别器。
        """
        for config in configs:
            self._validate_config(config)
        self.configs = configs
        self.image_recognizer = image_recognizer

    @classmethod
    def _validate_config(cls, config: dict[str, Any]) -> None:
        """校验抽取配置的公共结构和正则表达式。

        Args:
            config: 待校验的单个抽取配置。

        Raises:
            ValueError: 配置名称、模式、必填结构或正则表达式不合法。
        """
        # 先检查公共任务外壳，再按 mode 检查各自必需结构，错误会在启动阶段
        # 暴露，而不是等到处理完整份报告后才失败。
        name = str(config.get("name") or "").strip()
        if not name:
            raise ValueError("抽取配置缺少 name")
        mode = str(config.get("mode", "layer_records"))
        if mode not in cls._SUPPORTED_MODES:
            raise ValueError(f"配置 {name} 使用了未知抽取模式: {mode}")
        if mode == "layer_records" and not isinstance(config.get("fields"), dict):
            raise ValueError(f"配置 {name} 的 layer_records 模式缺少 fields 字典")
        if mode == "keyword_fields" and not isinstance(config.get("fields"), dict):
            raise ValueError(f"配置 {name} 的 keyword_fields 模式缺少 fields 字典")
        if mode == "section_content":
            sections = config.get("sections")
            if not isinstance(sections, list) or not sections:
                raise ValueError(f"配置 {name} 的 section_content 模式缺少 sections 列表")
            for index, section in enumerate(sections, start=1):
                if not isinstance(section, dict) or not section.get("key"):
                    raise ValueError(f"配置 {name} 的第 {index} 个章节缺少 key")
                groups = section.get("alias_groups")
                if groups is not None and (
                    not isinstance(groups, list)
                    or not groups
                    or any(not isinstance(group, list) or not group for group in groups)
                ):
                    raise ValueError(
                        f"配置 {name} 的章节 {section['key']} 的 alias_groups 必须是非空二维列表"
                    )
                has_match_rule = bool(section.get("aliases") or groups)
                if not has_match_rule and not section.get("fallback_text") and not section.get(
                    "placeholder"
                ):
                    raise ValueError(
                        f"配置 {name} 的章节 {section['key']} 缺少标题别名或兜底规则"
                    )
        # 正则可能深藏在字段、章节或表格配置中，因此统一递归预编译。
        cls._validate_regex_values(config, name)

    @classmethod
    def _validate_regex_values(cls, value: Any, config_name: str, path: str = "") -> None:
        """递归检查配置中以 pattern 或 patterns 命名的正则项。

        Args:
            value: 当前待遍历的配置值。
            config_name: 配置名称，用于生成错误信息。
            path: 当前值在配置中的点分路径。

        Raises:
            ValueError: 某个正则表达式无法编译。
        """
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if key == "pattern" and isinstance(child, str):
                    cls._compile_config_pattern(child, config_name, child_path)
                elif key.endswith("patterns") and isinstance(child, list):
                    for index, pattern in enumerate(child):
                        if isinstance(pattern, str):
                            cls._compile_config_pattern(
                                pattern, config_name, f"{child_path}[{index}]"
                            )
                cls._validate_regex_values(child, config_name, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                cls._validate_regex_values(child, config_name, f"{path}[{index}]")

    @staticmethod
    def _compile_config_pattern(pattern: str, config_name: str, path: str) -> None:
        """编译单个配置正则并转换为清晰的配置错误。

        Args:
            pattern: 正则表达式文本。
            config_name: 配置名称。
            path: 正则表达式所在配置路径。

        Raises:
            ValueError: 正则表达式语法错误。
        """
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"配置 {config_name} 的正则 {path} 不合法: {exc}") from exc

    @classmethod
    def from_files(
        cls,
        config_paths: str | Path | Iterable[str | Path],
        *,
        image_recognizer: Any | None = None,
    ) -> "ExtractionEngine":
        """从一个或多个 YAML 文件创建抽取引擎。

        Args:
            config_paths: 单个配置路径，或者配置路径集合。
            image_recognizer: 可选的钻孔柱状图识别器。

        Returns:
            可以执行全部配置的抽取引擎。

        Raises:
            ValueError: 配置为空、缺少名称或任务名称重复。
        """
        if isinstance(config_paths, (str, Path)):
            paths = [Path(config_paths)]
        else:
            paths = [Path(path) for path in config_paths]

        # 多配置共享同一份 DocumentModel；这里仅加载和校验配置，不重复解析文档。
        configs: list[dict[str, Any]] = []
        names: set[str] = set()
        for path in paths:
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(config, dict) or not config.get("name"):
                raise ValueError(f"抽取配置缺少 name: {path}")
            name = str(config["name"])
            if name in names:
                raise ValueError(f"抽取任务名称重复: {name}")
            config["_config_path"] = str(path.resolve())
            names.add(name)
            configs.append(config)
        if not configs:
            raise ValueError("至少需要一个抽取配置文件")
        return cls(configs, image_recognizer=image_recognizer)

    def extract_all(self, document: DocumentModel) -> dict[str, Any]:
        """在同一份解析结果上执行全部配置。

        Args:
            document: Word 或 PDF 解析得到的统一文档模型。

        Returns:
            按任务名称组织的抽取结果字典。
        """
        tasks = {str(config["name"]): self._extract_one(document, config) for config in self.configs}
        return {
            "document": {
                "source_path": document.source_path,
                "source_format": document.source_format,
                "parser_backend": document.parser_backend,
            },
            "tasks": tasks,
        }

    def extract_keyword_fields(
        self, document: DocumentModel, config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """执行可复用的关键词字段抽取。

        Args:
            document: 已解析的统一文档模型。
            config: ``keyword_fields`` 模式配置。

        Returns:
            完成候选值选择后的字段记录。
        """
        records = self._extract_keyword_records(document.blocks, config)
        return self._select_keyword_records(records, config.get("fields", {}))

    def extract_section_content(
        self, document: DocumentModel, config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """执行可复用的章节内容抽取。

        Args:
            document: 已解析的统一文档模型。
            config: ``section_content`` 模式配置。

        Returns:
            命中的章节结构化记录。
        """
        return self._extract_sections(document.blocks, config)

    def extract_table_records(
        self, block: DocumentBlock, config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """使用配置化表头规则抽取普通矩阵表。

        Args:
            block: 统一模型中的表格块。
            config: 表头行数及列字段规则。

        Returns:
            每个数据行对应的一条记录。
        """
        return self._extract_matrix_table(block, config)

    def _extract_one(self, document: DocumentModel, config: dict[str, Any]) -> dict[str, Any]:
        """执行一个配置定义的抽取任务。

        Args:
            document: 统一文档模型。
            config: 当前任务配置。

        Returns:
            原始候选、选择结果和任务状态。
        """
        # 三种 mode 共用任务状态、证据和输出协议，差异只集中在候选生成及选择。
        mode = str(config.get("mode", "layer_records"))
        blocks, section_found = self._locate_section(document.blocks, config.get("section", {}))
        if mode == "keyword_fields":
            records = self._extract_keyword_records(document.blocks, config)
            selected = self._select_keyword_records(records, config.get("fields", {}))
        elif mode == "section_content":
            records = self._extract_sections(document.blocks, config)
            selected = list(records)
        elif mode == "layer_records":
            records = self._extract_records(blocks, config)
            self._merge_matrix_tables(document.blocks, records, config.get("matrix_tables", []))
            selected = self._select_records(records, config.get("selection", {}))
        else:
            raise ValueError(f"未知抽取模式: {mode}")
        warnings = []
        if not section_found:
            warnings.append("未找到目标章节，已按配置决定是否扫描全文")
        # OCR 是文本结果不足时的兜底，不会无条件扫描所有 PDF 页面。
        image_config = config.get("image_fallback", {})
        needs_image = mode == "layer_records" and self._needs_image_fallback(
            blocks, selected, image_config, document=document
        )
        image_fallback_incomplete = False
        if needs_image:
            if self.image_recognizer is None:
                warnings.append("文本和表格结果不完整，但没有传入图片识别器")
                image_fallback_incomplete = True
            else:
                ocr_text_records: list[dict[str, Any]] = []
                try:
                    # 某些 PDF 的可见中文正常，但内嵌文本编码已损坏。先把候选页
                    # OCR 成公共内容块，再复用同一套土层正则；钻孔图仍走后面的
                    # 坐标分栏识别，两种来源最终按层号统一归并。
                    recognize_text_blocks = getattr(
                        self.image_recognizer, "recognize_text_blocks", None
                    )
                    if callable(recognize_text_blocks):
                        ocr_blocks = recognize_text_blocks(document, image_config)
                        ocr_text_records = self._extract_records(ocr_blocks, config)
                        self._merge_matrix_tables(
                            document.blocks,
                            ocr_text_records,
                            config.get("matrix_tables", []),
                        )
                    image_records = self.image_recognizer.recognize_document(document, image_config)
                    # 图片层厚是在正文表格合并之后加入的，因此必须让新增记录再次经过
                    # 同一套推荐值表、桩基参数表和派生字段流程。
                    self._merge_matrix_tables(
                        document.blocks,
                        image_records,
                        config.get("matrix_tables", []),
                    )
                    records.extend(ocr_text_records)
                    records.extend(image_records)
                    selected = self._select_records(records, config.get("selection", {}))
                except Exception as exc:
                    # 普通页面 OCR 已补出土层时，钻孔分栏识别失败不应丢弃这些
                    # 有效结果；仅记录提示。两种 OCR 均无结果时才标记任务不完整。
                    if ocr_text_records:
                        records.extend(ocr_text_records)
                        selected = self._select_records(records, config.get("selection", {}))
                        warnings.append(f"钻孔图识别未补出更多结果: {exc}")
                    else:
                        warnings.append(f"图片识别失败: {exc}")
                        image_fallback_incomplete = True
        if mode == "layer_records":
            # 耕土合并必须发生在所有派生参数和后处理之前。此后 selected、
            # result.json 和 callback.json 都使用同一份合并后的业务土层。
            excluded_names = config.get("postprocess", {}).get(
                "foundation_exclude_names", ["耕土"]
            )
            selected = merge_first_cultivated_soil_layer(selected, excluded_names)
            # 候选合并后再计算缺省/派生参数，确保正文平均值和报告推荐值
            # 始终优先于按土类型生成的缺省值。
            self._apply_derived_fields(selected, config.get("derived_fields", {}))
            self._postprocess(selected, config.get("postprocess", {}), records)
        pending_sections = (
            [item for item in selected if item.get("status") == "pending"]
            if mode == "section_content"
            else []
        )
        defaulted_sections = (
            [item for item in selected if item.get("status") == "defaulted"]
            if mode == "section_content"
            else []
        )
        if pending_sections:
            warnings.append(
                "以下文字稿章节尚未从报告提取，已保留占位："
                + "、".join(str(item.get("output_title") or item["section"]) for item in pending_sections)
            )
        if defaulted_sections:
            warnings.append(
                "以下文字稿章节使用了配置兜底文本，不属于原文抽取："
                + "、".join(
                    str(item.get("output_title") or item["section"])
                    for item in defaulted_sections
                )
            )
        status = (
            "partial"
            if pending_sections or defaulted_sections or image_fallback_incomplete
            else ("success" if selected else ("partial" if records else "no_data"))
        )
        result = {
            "status": status,
            "mode": mode,
            "config_version": str(config.get("version", "1.0")),
            "config_path": config.get("_config_path"),
            "records": records,
            "selected_records": selected,
            "warnings": warnings,
        }
        if mode == "keyword_fields":
            values: dict[str, Any] = {}
            for item in selected:
                field_name = str(item["field"])
                selection_mode = config.get("fields", {}).get(field_name, {}).get("selection")
                if selection_mode in {"all", "all_or_single"}:
                    values.setdefault(field_name, []).append(item["value"])
                else:
                    values[field_name] = item["value"]
            # 同一报告可能按风机分成多个场地类别；只有一个唯一值时仍保持标量，
            # 多个值时输出列表，兼顾普通报告和分区报告。
            for field_name, field_config in config.get("fields", {}).items():
                if field_config.get("selection") != "all_or_single":
                    continue
                values_for_field = values.get(field_name)
                if isinstance(values_for_field, list) and len(values_for_field) == 1:
                    values[field_name] = values_for_field[0]
            result["values"] = values
        elif mode == "section_content":
            result["values"] = {item["section"]: item["text"] for item in selected}
        return result


    @staticmethod
    def _needs_image_fallback(
        blocks: list[DocumentBlock],
        selected: list[dict[str, Any]],
        image_config: dict[str, Any],
        *,
        document: DocumentModel | None = None,
    ) -> bool:
        """判断当前文本结果是否需要使用图片识别补充。

        Args:
            blocks: 已定位到的目标章节内容块。
            selected: 文本和表格已经选出的记录。
            image_config: 图片兜底配置。
            document: 可选的完整文档，用于判断是否存在柱状图候选页。

        Returns:
            图片兜底已启用且结果为空或少于原文声明数量时返回 ``True``。
        """
        # 未命中任何文本层时直接回退；已有结果时再用报告声明层数判断完整性。
        if not image_config.get("enabled"):
            return False
        if not selected:
            return True
        completeness = image_config.get("completeness", {})
        patterns = completeness.get("expected_count_patterns", [])
        text = "\n".join(block.text for block in blocks)
        for item in patterns:
            if not isinstance(item, dict) or not item.get("pattern"):
                continue
            match = re.search(str(item["pattern"]), text, re.DOTALL)
            if not match:
                continue
            groups = item.get("sum_groups", [])
            try:
                expected = sum(int(match.group(str(group))) for group in groups)
            except (IndexError, TypeError, ValueError):
                continue
            return len(selected) < expected
        minimum_count = completeness.get("minimum_count")
        if minimum_count is not None:
            return len(selected) < int(minimum_count)

        # 很多报告不会声明总层数，正文只描述部分土层，完整数据放在钻孔
        # 柱状图中。只在确实存在柱状图关键词或低文本图片页时继续 OCR，
        # 避免普通纯文本报告被无条件扫描。
        if document is None or not image_config.get("fallback_when_diagram_pages_exist", False):
            return False
        page_keywords = [
            str(value).replace(" ", "") for value in image_config.get("page_keywords", [])
        ]
        has_keyword_page = any(
            page_keywords
            and any(keyword in block.text.replace(" ", "") for keyword in page_keywords)
            for block in document.blocks
        )
        if has_keyword_page:
            return True
        if not image_config.get("use_image_blocks", False):
            return False
        maximum_text = int(image_config.get("image_page_max_text", 150))
        return any(
            page.image_count > 0 and page.text_characters < maximum_text
            for page in document.pages
        )

    @staticmethod
    def _locate_section(
        blocks: list[DocumentBlock], section_config: dict[str, Any]
    ) -> tuple[list[DocumentBlock], bool]:
        """根据标题别名截取目标章节。

        Args:
            blocks: 文档的全部内容块。
            section_config: 章节别名、结束方式和全文兜底配置。

        Returns:
            目标章节内容块及是否成功找到章节。
        """
        aliases = [str(alias).replace(" ", "") for alias in section_config.get("aliases", [])]
        if not aliases:
            return blocks, True

        # 同一关键词可能先出现在勘察任务、目录或正文说明中。收集全部候选后
        # 优先选择真正的标题，其次选择带章节编号的段落，最后才用普通正文。
        candidates: list[tuple[int, int, int, int]] = []
        for index, block in enumerate(blocks):
            compact_text = block.text.replace(" ", "")
            # 目录中的标题通常带有点线和页码，不能把它当成正文标题。
            is_toc_item = bool(
                re.search(
                    r"(?:\.{3,}|…{2,})\s*\d+\s*$|(?:PAGEREF|HYPERLINK)",
                    compact_text,
                    re.IGNORECASE,
                )
            )
            if is_toc_item:
                continue
            matched_aliases = [alias for alias in aliases if alias in compact_text]
            if not matched_aliases:
                continue
            number = ExtractionEngine._section_number(block.text)
            heading_rank = 3 if block.kind == "heading" else (2 if number else 1)
            stripped = re.sub(r"^\s*\d+(?:\.\d+)*[、．.]?\s*", "", compact_text)
            direct_rank = int(any(stripped.startswith(alias) for alias in matched_aliases))
            candidates.append((heading_rank, direct_rank, max(map(len, matched_aliases)), -index))

        start_index = None
        start_level = None
        start_number = None
        if candidates:
            _, _, _, negative_index = max(candidates)
            start_index = -negative_index
            start_block = blocks[start_index]
            start_level = ExtractionEngine._heading_level(start_block)
            start_number = ExtractionEngine._section_number(start_block.text)

        if start_index is None:
            fallback = bool(section_config.get("fallback_whole_document", False))
            return (blocks if fallback else []), False

        end_index = len(blocks)
        if section_config.get("stop_at_next_heading", True):
            for index in range(start_index + 1, len(blocks)):
                block = blocks[index]
                level = ExtractionEngine._heading_level(block)
                number = ExtractionEngine._section_number(block.text)
                is_next_number = bool(
                    start_number
                    and number
                    and number > start_number
                    and len(number) <= len(start_number)
                )
                # PDF 版面解析器可能把“②细砂”“②1粉土”等加粗土层行误标为
                # heading。起始标题已有 2.2 这类可靠编号时，应以 2.3/3 等章节
                # 编号作为边界，不能遇到任意 heading 就提前截断。
                is_next_heading = block.kind == "heading" and (
                    (start_number is None and start_level is None)
                    or (
                        start_level is not None
                        and level is not None
                        and level <= start_level
                    )
                )
                if is_next_number or is_next_heading:
                    end_index = index
                    break
        return blocks[start_index + 1 : end_index], True

    @staticmethod
    def _section_number(text: str) -> tuple[int, ...] | None:
        """读取段首的章节编号。

        Args:
            text: 标题或段落文本。

        Returns:
            章节编号元组，例如 ``4.3`` 返回 ``(4, 3)``；无编号时返回
            ``None``。
        """
        # 编号后应接空白、顿号或中文标题。这样不会把“36.21m”之类的
        # 小数误判为第 36.21 节，导致正文被提前截断。
        match = re.match(
            r"\s*(\d+(?:\.\d+)*)(?:[\.．])?(?=\s+|[、]|[\u4e00-\u9fff])",
            text,
        )
        if not match:
            return None
        return tuple(int(part) for part in match.group(1).split("."))

    @staticmethod
    def _heading_level(block: DocumentBlock) -> int | None:
        """读取内容块的标题层级。

        Args:
            block: 可能为标题的内容块。

        Returns:
            标题层级；不存在时返回 ``None``。
        """
        if block.paragraph_style and block.paragraph_style.outline_level:
            return block.paragraph_style.outline_level
        value = block.metadata.get("heading_level") or block.metadata.get("heading level")
        return int(value) if value else None

    def _extract_records(
        self, blocks: list[DocumentBlock], config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """从章节内容块中抽取候选记录。

        Args:
            blocks: 已定位的章节内容块。
            config: 当前任务配置。

        Returns:
            带原文证据的候选记录列表。
        """
        # 地层描述允许跨段落/跨表格时走专门的合并流程；简单任务仍可按分隔符
        # 逐段匹配，公共引擎无需为每类报告新增 Python 方法。
        record_config = config.get("record", {})
        if record_config.get("merge_blocks"):
            return self._extract_layer_records(blocks, config)

        split_pattern = record_config.get("split_pattern", r"[\n；;]+")
        match_patterns = record_config.get("match_patterns", [])
        records = []
        for block in blocks:
            if block.kind == "image" or not block.text.strip():
                continue
            for segment in re.split(split_pattern, block.text):
                original_text = segment.strip()
                if not original_text:
                    continue
                normalized_text = self._normalize_text(original_text)
                if match_patterns and not any(
                    re.search(pattern, normalized_text, re.IGNORECASE) for pattern in match_patterns
                ):
                    continue
                record = self._extract_fields(normalized_text, config.get("fields", {}))
                if not any(value is not None for value in record.values()):
                    continue
                record["evidence"] = {
                    "block_id": block.id,
                    "page": block.page,
                    "source_type": block.kind,
                    "text": original_text,
                }
                records.append(record)
        return records

    def _extract_layer_records(
        self, blocks: list[DocumentBlock], config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """合并被解析器拆开的层名、描述、厚度和统计表。

        OpenDataLoader 可能把 ``Q4 al`` 的上下标、层名和厚度拆成多个内容块，
        甚至把同一行右侧文字放到层号之前。本方法以层号块为锚点，并绑定其后
        紧随其后的统计表及跨页续表，使正文结构差异不会进入具体字段规则。

        Args:
            blocks: 已定位到目标章节的内容块。
            config: 当前任务配置。

        Returns:
            每个岩土层对应的一条候选记录。
        """
        record_config = config.get("record", {})
        start_patterns = record_config.get("start_patterns") or record_config.get(
            "match_patterns", []
        )
        anchors = [
            index
            for index, block in enumerate(blocks)
            if block.kind != "table"
            and any(
                re.search(pattern, self._layer_source_text(block), re.IGNORECASE)
                for pattern in start_patterns
            )
        ]

        records: list[dict[str, Any]] = []
        for anchor_position, block_index in enumerate(anchors):
            next_anchor = anchors[anchor_position + 1] if anchor_position + 1 < len(anchors) else len(blocks)
            anchor = blocks[block_index]
            group_blocks = [anchor]

            # 同行的地质年代或描述片段有时会被排在层号块之前，只回看一块可避免
            # 把上一层的厚度误并入当前层。
            if block_index > 0:
                previous = blocks[block_index - 1]
                normalized_previous = self._layer_source_text(previous)
                previous_is_anchor = any(
                    re.search(pattern, normalized_previous, re.IGNORECASE)
                    for pattern in start_patterns
                )
                # 上一块若已经出现层厚或层底数据，通常是上一土层的续行，不能
                # 拼给当前层；否则会让“②细砂”错误继承“①粉细砂”的厚度。
                previous_has_layer_measurement = bool(
                    re.search(r"(?:层厚|厚度|层底(?:深度|标高))", normalized_previous)
                )
                # 某些 PDF 的阅读顺序会把“al)：……厚度……”放到“②-1层…(Q4”
                # 之前。此时两块可由未闭合的地质年代明确拼接，仍应保留上一块。
                previous_completes_geological_age = bool(
                    re.search(r"\(Q\d+\s*$", self._layer_source_text(anchor), re.IGNORECASE)
                    and re.match(r"^[a-z]{1,4}\s*\)", normalized_previous, re.IGNORECASE)
                )
                if (
                    previous.kind not in {"table", "heading", "image", "page_break"}
                    and not previous_is_anchor
                    and (
                        not previous_has_layer_measurement
                        or previous_completes_geological_age
                    )
                ):
                    group_blocks.append(previous)

            table_blocks: list[DocumentBlock] = []
            for candidate in blocks[block_index + 1 : next_anchor]:
                if candidate.kind == "table" and candidate.table:
                    table_blocks.append(candidate)
                    group_blocks.append(candidate)
                elif candidate.kind not in {"image", "page_break"}:
                    group_blocks.append(candidate)

            text_blocks = [item for item in group_blocks if item.kind != "table" and item.text.strip()]
            merged_text = self._normalize_text(
                " ".join(self._layer_source_text(item) for item in text_blocks)
            )
            # 某些 PDF 把“层②1”拆成“层②”和下一块“1”，先恢复亚层编号。
            merged_text = re.sub(
                r"层\s*([①②③④⑤⑥⑦⑧⑨⑩])\s+(\d+)",
                r"层\1-\2",
                merged_text,
            )
            record = self._extract_fields(merged_text, config.get("fields", {}))
            if not record.get("layer_code"):
                continue

            evidence: dict[str, Any] = {
                "block_ids": [item.id for item in group_blocks],
                "pages": sorted({item.page for item in group_blocks if item.page is not None}),
                "source_type": "merged_blocks",
                "text": "\n".join(self._layer_source_text(item) for item in text_blocks),
            }
            combined_table_evidence: dict[str, Any] = {}
            for table_block in table_blocks:
                table_values, table_evidence = self._extract_table_fields(
                    table_block,
                    config.get("table_fields", {}),
                    config.get("table_options", {}),
                )
                record.update(
                    {name: value for name, value in table_values.items() if value is not None}
                )
                if table_evidence:
                    combined_table_evidence.update(table_evidence)
            if combined_table_evidence:
                evidence["table_fields"] = combined_table_evidence
            record["evidence"] = evidence
            records.append(record)
        return records

    @staticmethod
    def _layer_source_text(block: DocumentBlock) -> str:
        """返回用于土层识别的文本，并恢复 Word 自动列表编号。

        Word 的 ``①``、``②`` 等编号可能不在正文字符中，而是保存在段落的
        ``list_label`` 属性里。这里只为土层抽取临时拼回编号，不修改统一
        文档模型，避免影响普通编号列表和其他章节输出。

        Args:
            block: 待读取的统一内容块。

        Returns:
            规范化后的土层源文本；存在独立列表编号时将其前置。
        """
        text = ExtractionEngine._normalize_text(block.text)
        style = block.paragraph_style
        label = str(style.list_label).strip() if style and style.list_label else ""
        if label and not text.startswith(label):
            return ExtractionEngine._normalize_text(f"{label}{text}")
        return text

    def _extract_table_fields(
        self,
        block: DocumentBlock,
        table_fields: dict[str, Any],
        table_options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, float | None], dict[str, Any]]:
        """从岩土物理力学统计表的“平均值”列抽取配置字段。

        Args:
            block: 带结构化单元格的表格内容块。
            table_fields: 指标行别名、单位和换算规则配置。
            table_options: 平均值表头别名和无表头续表的回退列配置。

        Returns:
            字段值字典及对应的表格证据。
        """
        if not block.table or not table_fields:
            return {}, {}

        rows: dict[int, dict[int, str]] = {}
        for cell in block.table.cells:
            rows.setdefault(cell.row, {})[cell.column] = cell.text.strip()

        table_options = table_options or {}
        average_aliases = table_options.get("average_aliases", ["平均值", "平均", "均值"])
        # 统计表优先根据“平均值”表头定位列，而不是假设固定列号。
        average_column = None
        header_row = None
        for row_number in sorted(rows)[:3]:
            for column, value in rows[row_number].items():
                compact = self._compact_label(value)
                if any(self._compact_label(alias) in compact for alias in average_aliases):
                    average_column = column
                    header_row = row_number
                    break
            if average_column is not None:
                break
        if average_column is None:
            # 跨页续表经常省略表头，仅在配置显式声明回退列时使用固定列号。
            fallback_column = table_options.get("fallback_average_column")
            if fallback_column is None:
                return {}, {}
            average_column = int(fallback_column)
            first_row = min(rows) if rows else 0
            first_label = self._compact_label(rows.get(first_row, {}).get(0, ""))
            first_is_metric = any(
                re.search(pattern, first_label, re.IGNORECASE)
                for field_config in table_fields.values()
                for pattern in field_config.get("row_patterns", [])
            )
            header_row = -1 if first_is_metric else first_row
        # 找到平均值列时一定同时记录了表头行，这里显式收窄可空类型。
        assert header_row is not None

        values: dict[str, float | None] = {}
        evidence: dict[str, Any] = {}
        for field_name, field_config in table_fields.items():
            values[field_name] = None
            patterns = field_config.get("row_patterns", [])
            for row_number in sorted(rows):
                if row_number == header_row:
                    continue
                label = rows[row_number].get(0, "")
                compact_label = self._compact_label(label)
                if not any(re.search(pattern, compact_label, re.IGNORECASE) for pattern in patterns):
                    continue
                raw_text = rows[row_number].get(average_column, "")
                raw_value = self._first_number(raw_text)
                if raw_value is None:
                    continue
                # 换算规则由 YAML 决定，例如天然密度 g/cm³ 乘 9.8 后统一为
                # PRD 要求的重力密度 kN/m³。
                multiplier = 1.0
                matched_unit = field_config.get("source_unit")
                for rule in field_config.get("unit_rules", []):
                    if re.search(str(rule["pattern"]), compact_label, re.IGNORECASE):
                        multiplier = float(rule.get("multiplier", 1))
                        matched_unit = rule.get("source_unit", matched_unit)
                        break
                converted = round(raw_value * multiplier, 6)
                values[field_name] = converted
                evidence[field_name] = {
                    "row_label": label,
                    "average_column": (
                        rows.get(header_row, {}).get(average_column) or f"配置列{average_column}"
                    ),
                    "raw_value": raw_value,
                    "source_unit": matched_unit,
                    "target_unit": field_config.get("target_unit"),
                    "multiplier": multiplier,
                }
                break
        return values, evidence

    @staticmethod
    def _compact_label(text: str) -> str:
        """清理表格指标名称，保留单位相关符号。

        Args:
            text: 单元格原文。

        Returns:
            去掉空白并统一常见上下标字符后的文本。
        """
        replacements = {"³": "3", "²": "2", "₁": "1", "₂": "2", "－": "-", "—": "-"}
        for source, target in replacements.items():
            text = text.replace(source, target)
        return re.sub(r"\s+", "", text)

    @staticmethod
    def _first_number(text: str) -> float | None:
        """读取单元格中的首个数值。

        Args:
            text: 单元格文本。

        Returns:
            浮点数；单元格无有效数值时返回 ``None``。
        """
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
        return float(match.group(0)) if match else None

    def _extract_keyword_records(
        self, blocks: list[DocumentBlock], config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """使用同一套关键词、正则和选择规则抽取普通字段。

        字段可以配置优先章节。优先章节存在有效结果时不再扫描全文，符合 PRD
        中“先查结论与建议，找不到再查全文”的要求。

        Args:
            blocks: 文档全部内容块。
            config: ``keyword_fields`` 模式配置。

        Returns:
            每个命中值对应的一条带证据候选记录。
        """
        search_config = config.get("search", {})
        preferred_aliases = search_config.get("priority_section_aliases", [])
        preferred_blocks = self._collect_sections_by_aliases(blocks, preferred_aliases)
        records: list[dict[str, Any]] = []
        for field_name, field_config in config.get("fields", {}).items():
            candidates = self._find_field_candidates(preferred_blocks, field_name, field_config)
            if not candidates and search_config.get("fallback_whole_document", True):
                candidates = self._find_field_candidates(blocks, field_name, field_config)
            records.extend(candidates)
        return records

    def _find_field_candidates(
        self,
        blocks: list[DocumentBlock],
        field_name: str,
        field_config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """在给定内容块中查找一个字段的全部候选值。

        Args:
            blocks: 本轮搜索范围。
            field_name: 输出字段名称。
            field_config: 关键词、正则、类型及校验配置。

        Returns:
            当前字段的候选记录列表。
        """
        keywords = [self._normalize_text(str(item)) for item in field_config.get("keywords", [])]
        negative_words = [str(item) for item in field_config.get("negative_prefixes", [])]
        negative_window = int(field_config.get("negative_window", 5))
        candidates: list[dict[str, Any]] = []
        for block_index, block in enumerate(blocks):
            if block.kind in {"image", "page_break"} or not block.text.strip():
                continue
            allowed_kinds = field_config.get("block_kinds")
            if allowed_kinds and block.kind not in set(allowed_kinds):
                continue
            text = self._normalize_text(block.text)
            keyword_positions = [text.find(keyword) for keyword in keywords if keyword in text]
            if keywords and not keyword_positions:
                continue
            if keyword_positions:
                position = min(value for value in keyword_positions if value >= 0)
                prefix = text[max(0, position - negative_window) : position]
                if any(word in prefix for word in negative_words):
                    continue
            # PDF/Word 可能把同一句任务要求拆成相邻内容块，例如上一块是
            # “查清有无……”，下一块才出现“湿陷性黄土”。字段可通过配置声明
            # 回看若干块，防止把条件性调查事项误判为报告结论。
            previous_count = max(0, int(field_config.get("exclude_previous_blocks", 0)))
            exclusion_text = self._normalize_text(
                " ".join(
                    item.text
                    for item in blocks[max(0, block_index - previous_count) : block_index + 1]
                    if item.kind not in {"image", "page_break"}
                )
            )
            if any(
                re.search(pattern, exclusion_text, re.IGNORECASE)
                for pattern in field_config.get("exclude_patterns", [])
            ):
                continue
            for pattern in field_config.get("patterns", []):
                pattern_matched = False
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    raw_value = match.groupdict().get("value", match.group(0))
                    if field_config.get("require_terminal_punctuation") and not re.search(
                        r"[。；;.!！]$", str(raw_value).strip()
                    ):
                        continue
                    value = self._convert_keyword_value(raw_value, field_config)
                    if not self._value_in_range(value, field_config):
                        continue
                    candidates.append(
                        {
                            "field": field_name,
                            "value": value,
                            "raw_value": raw_value,
                            "evidence": {
                                "block_id": block.id,
                                "page": block.page,
                                "source_type": block.kind,
                                "text": block.text,
                            },
                        }
                    )
                    pattern_matched = True
                if pattern_matched and field_config.get("first_matching_pattern"):
                    break
        return candidates

    @staticmethod
    def _convert_keyword_value(value: Any, field_config: dict[str, Any]) -> Any:
        """按照关键词字段配置转换捕获值。

        Args:
            value: 正则捕获的原始内容。
            field_config: 类型和映射配置。

        Returns:
            数字、区间、映射字符串或清理后的原文。
        """
        text = str(value).strip()
        value_map = field_config.get("value_map", {})
        if text in value_map:
            converted: Any = value_map[text]
        elif field_config.get("type", "string") == "number":
            match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
            converted = float(match.group(0)) if match else None
        elif field_config.get("type", "string") == "range":
            numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
            if not numbers:
                return None
            converted = {"start": float(numbers[0]), "end": float(numbers[-1])}
        else:
            if field_config.get("strip_numbering"):
                text = ExtractionEngine._strip_numbering(text)
            converted = re.sub(r"\s+", "", text) if field_config.get("strip_spaces") else text
        transforms = field_config.get("transforms", [])
        if isinstance(transforms, str):
            transforms = [transforms]
        for transform in transforms:
            converted = ExtractionEngine._apply_transform(converted, transform)
        return converted

    @staticmethod
    def _value_in_range(value: Any, field_config: dict[str, Any]) -> bool:
        """校验数字候选是否位于业务允许范围。

        Args:
            value: 已转换的候选值。
            field_config: 可选的 ``minimum`` 和 ``maximum`` 配置。

        Returns:
            候选值是否有效。
        """
        if value is None:
            return False
        if not isinstance(value, (int, float)):
            return True
        minimum = field_config.get("minimum")
        maximum = field_config.get("maximum")
        return (minimum is None or value >= float(minimum)) and (
            maximum is None or value <= float(maximum)
        )

    @staticmethod
    def _keyword_candidate_marker(value: Any) -> str:
        """生成关键词候选值的稳定去重标记。

        字符串直接使用原值；字典、列表等结构化值序列化为稳定 JSON。随后统一
        去除空白并归一化“做为/作为”，保持原有候选去重规则不变。

        Args:
            value: 候选字段值。

        Returns:
            用于候选值比较和去重的规范化字符串。
        """
        marker = (
            value
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, sort_keys=True)
        )
        return re.sub(r"\s+", "", marker).replace("做为", "作为")

    @staticmethod
    def _select_keyword_records(
        records: list[dict[str, Any]], fields: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """按字段配置从候选值中选择最终值。

        Args:
            records: 全部关键词候选记录。
            fields: 各字段的选择方式及严重程度顺序。

        Returns:
            每个字段的一条最终记录；``all`` 类型字段可返回多条。
        """
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            groups.setdefault(str(record["field"]), []).append(record)
        selected: list[dict[str, Any]] = []
        for field_name, candidates in groups.items():
            field_config = fields.get(field_name, {})
            preferred_context = [
                ExtractionEngine._normalize_text(str(value))
                for value in field_config.get("prefer_context_keywords", [])
            ]
            if preferred_context:
                preferred_candidates = [
                    candidate
                    for candidate in candidates
                    if any(
                        keyword
                        in ExtractionEngine._normalize_text(
                            str(candidate.get("evidence", {}).get("text", ""))
                        )
                        for keyword in preferred_context
                    )
                ]
                if preferred_candidates:
                    candidates = preferred_candidates
            operator = field_config.get("selection", "first")
            if operator in {"all", "all_or_single"}:
                unique_candidates: list[dict[str, Any]] = []
                for candidate in candidates:
                    marker = ExtractionEngine._keyword_candidate_marker(candidate["value"])
                    duplicate_index = None
                    for index, existing in enumerate(unique_candidates):
                        existing_marker = ExtractionEngine._keyword_candidate_marker(existing["value"])
                        if marker == existing_marker or (
                            field_config.get("deduplicate_contained")
                            and (marker in existing_marker or existing_marker in marker)
                        ):
                            duplicate_index = index
                            if len(marker) > len(existing_marker):
                                unique_candidates[index] = candidate
                            break
                    if duplicate_index is None:
                        unique_candidates.append(candidate)
                selected.extend(unique_candidates)
                continue
            if operator == "maximum":
                item = max(candidates, key=lambda value: float(value["value"]))
            elif operator == "start_maximum":
                item = max(candidates, key=lambda value: float(value["value"]["start"]))
            elif operator == "longest":
                item = max(candidates, key=lambda value: len(str(value["value"])))
            elif operator == "severity_maximum":
                order = field_config.get("severity_order", [])
                ranking = {str(value): index for index, value in enumerate(order)}
                item = max(candidates, key=lambda value: ranking.get(str(value["value"]), -1))
            else:
                item = candidates[0]
            selected.append(item)
        return selected

    def _collect_sections_by_aliases(
        self, blocks: list[DocumentBlock], aliases: Iterable[str]
    ) -> list[DocumentBlock]:
        """收集标题命中任一别名的完整章节内容。

        Args:
            blocks: 文档全部内容块。
            aliases: 可接受的章节标题别名。

        Returns:
            去重后保持原阅读顺序的内容块。
        """
        compact_aliases = [str(alias).replace(" ", "") for alias in aliases]
        if not compact_aliases:
            return []
        indexes: set[int] = set()
        for start, block in enumerate(blocks):
            compact = block.text.replace(" ", "")
            is_heading_like = block.kind in {"heading", "list_item"} or bool(
                self._section_number(block.text)
            )
            if not is_heading_like:
                continue
            if not any(alias in compact for alias in compact_aliases):
                continue
            if re.search(r"(?:\.{3,}|…{2,}|PAGEREF)\s*", compact, re.IGNORECASE):
                continue
            start_level = self._heading_level(block)
            start_number = self._section_number(block.text)
            end = self._next_section_index(blocks, start, start_level, start_number)
            indexes.update(range(start, end))
        return [block for index, block in enumerate(blocks) if index in indexes]

    def _extract_sections(
        self, blocks: list[DocumentBlock], config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """按配置抽取章节原文、表格和图片引用。

        Args:
            blocks: 文档全部内容块。
            config: ``section_content`` 模式配置。

        Returns:
            每个目标章节对应的一条结构化记录。
        """
        results: list[dict[str, Any]] = []
        for section in config.get("sections", []):
            configured_groups = section.get("alias_groups")
            alias_groups = configured_groups or [section.get("aliases", [])]
            selected_matches: list[tuple[int, str]] = []
            for group in alias_groups:
                match = self._best_section_match(blocks, section, [str(value) for value in group])
                if match is not None and match not in selected_matches:
                    selected_matches.append(match)
            # 一些报告把 PRD 要求的多个小节合并为一个“地震效应”等综合章节。
            # 仅当分组标题全部未命中时才启用回退别名，避免与正常分节重复抽取。
            if not selected_matches and section.get("fallback_aliases"):
                match = self._best_section_match(
                    blocks,
                    section,
                    [str(value) for value in section.get("fallback_aliases", [])],
                )
                if match is not None:
                    selected_matches.append(match)
            selected_matches.sort(key=lambda item: item[0])
            if not selected_matches:
                fallback = section.get("fallback_text")
                if fallback:
                    results.append(
                        {
                            "section": section["key"],
                            "title": section.get("output_title"),
                            "output_title": section.get("output_title"),
                            "outline_path": section.get("outline_path"),
                            "include_in_draft": section.get("include_in_draft", True),
                            "level": None,
                            "page": None,
                            "text": str(fallback),
                            "content": [],
                            "source": "fallback",
                            "status": "defaulted",
                        }
                    )
                elif section.get("placeholder"):
                    placeholder = dict(section["placeholder"])
                    results.append(
                        {
                            "section": section["key"],
                            "title": section.get("output_title"),
                            "output_title": section.get("output_title"),
                            "outline_path": section.get("outline_path"),
                            "include_in_draft": section.get("include_in_draft", True),
                            "level": None,
                            "page": None,
                            "text": "",
                            "content": [],
                            "source": "placeholder",
                            "status": "pending",
                            "placeholder": placeholder,
                        }
                    )
                continue
            stop_keywords = [str(value) for value in section.get("stop_keywords", [])]
            content: list[dict[str, Any]] = []
            sources: list[dict[str, Any]] = []
            for start, matched_title in selected_matches:
                heading = blocks[start]
                end = self._next_section_index(
                    blocks, start, self._heading_level(heading), self._section_number(heading.text)
                )
                source_content = self._section_content(
                    blocks, start, end, matched_title, section, stop_keywords
                )
                content.extend(source_content)
                sources.append(
                    {
                        "title": heading.text,
                        "matched_alias": matched_title,
                        "page": heading.page,
                        "block_id": heading.id,
                        "text": "\n".join(
                            str(item["text"]) for item in source_content if item["text"]
                        ),
                        "source_outline": self._source_outline(blocks, start),
                    }
                )
            first_start, first_title = selected_matches[0]
            heading = blocks[first_start]
            results.append(
                {
                    "section": section["key"],
                    "title": first_title,
                    "output_title": section.get("output_title", first_title),
                    "outline_path": section.get("outline_path"),
                    "include_in_draft": section.get("include_in_draft", True),
                    "source_title": heading.text,
                    "source_titles": [source["title"] for source in sources],
                    "sources": sources,
                    "level": self._heading_level(heading),
                    "source_outline": self._source_outline(blocks, first_start),
                    "page": heading.page,
                    "text": "\n".join(str(item["text"]) for item in content if item["text"]),
                    "content": content,
                    "source": "document",
                    "status": "extracted",
                }
            )
        return results

    def _best_section_match(
        self,
        blocks: list[DocumentBlock],
        section: dict[str, Any],
        aliases: list[str],
    ) -> tuple[int, str] | None:
        """为一组同义标题选择最准确的原文章节。

        Args:
            blocks: 文档全部内容块。
            section: 当前章节抽取配置。
            aliases: 表示同一来源章节的一组标题别名。

        Returns:
            最佳内容块下标和命中的原始别名；没有命中时返回 ``None``。
        """
        candidates: list[tuple[int, int, int, int, str]] = []
        allowed_kinds = set(section.get("heading_kinds", ["heading"]))
        for index, block in enumerate(blocks):
            if block.kind not in allowed_kinds and not self._is_numbered_heading_candidate(block):
                continue
            compact = block.text.replace(" ", "")
            if re.search(r"(?:\.{3,}|…{2,}|PAGEREF)", compact, re.IGNORECASE):
                continue
            stripped = self._strip_numbering(block.text).replace(" ", "")
            for alias in aliases:
                compact_alias = alias.replace(" ", "")
                if compact_alias not in compact:
                    continue
                direct_score = 2 if stripped.startswith(compact_alias) else 1
                # 普通编号正文可能在句中提到“场地类别”“不良地质作用”等
                # 关键词，不能把整句误当章节标题。非 heading 块只接受编号后
                # 直接以别名开头的标题形式。
                if block.kind not in allowed_kinds and direct_score < 2:
                    continue
                # Word 转换后偶尔会连续生成两个同名标题块。前一个通常只是
                # 空壳，优先选择其后确有正文范围的标题，同时仍保持正常报告
                # 中“主体结论优先于附件结论”的先后顺序。
                end = self._next_section_index(
                    blocks,
                    index,
                    self._heading_level(block),
                    self._section_number(block.text),
                )
                has_following_content = int(end > index + 1)
                # 直接标题优先；同等级下，较长的具体别名比短泛称更可靠。
                candidates.append(
                    (direct_score, len(compact_alias), has_following_content, -index, alias)
                )
        if not candidates:
            return None
        _, _, _, negative_index, alias = max(candidates)
        return -negative_index, alias

    def _section_content(
        self,
        blocks: list[DocumentBlock],
        start: int,
        end: int,
        matched_title: str,
        section: dict[str, Any],
        stop_keywords: list[str],
    ) -> list[dict[str, Any]]:
        """读取单个来源章节范围内的正文和结构化内容。

        Args:
            blocks: 文档全部内容块。
            start: 来源章节标题下标。
            end: 来源章节结束下标，不包含该位置。
            matched_title: 命中的标题别名。
            section: 当前章节抽取配置。
            stop_keywords: 遇到后提前停止的关键词。

        Returns:
            可写入明细 JSON 的内容块字典列表。
        """
        heading = blocks[start]
        content: list[dict[str, Any]] = []
        heading_body = self._heading_body(heading.text, matched_title)
        if heading_body:
            # 部分 PDF 会把“标题 + 首段正文”合并为一个块，正文不能丢失。
            content.append(
                {
                    "block_id": heading.id,
                    "kind": "paragraph",
                    "page": heading.page,
                    "text": heading_body,
                    "table": None,
                    "bbox": asdict(heading.bbox) if heading.bbox is not None else None,
                    "metadata": {**heading.metadata, "split_from_heading": True},
                }
            )
        for item in blocks[start + 1 : end]:
            if stop_keywords and any(keyword in item.text for keyword in stop_keywords):
                break
            if item.kind == "image" and not section.get("include_images", True):
                continue
            if item.kind == "table" and not section.get("include_tables", True):
                continue
            text = item.text.strip()
            text, reached_next_section = self._trim_embedded_next_heading(
                text, self._section_number(heading.text)
            )
            if section.get("strip_numbering"):
                text = self._strip_numbering(text)
            content.append(
                {
                    "block_id": item.id,
                    "kind": item.kind,
                    "page": item.page,
                    "text": text,
                    "table": asdict(item.table) if item.table is not None else None,
                    "bbox": asdict(item.bbox) if item.bbox is not None else None,
                    "metadata": item.metadata,
                }
            )
            if reached_next_section:
                break
        return content

    @staticmethod
    def _trim_embedded_next_heading(
        text: str, start_number: tuple[int, ...] | None
    ) -> tuple[str, bool]:
        """裁掉被解析器粘在上一段末尾的下一章节标题。

        Args:
            text: 当前内容块文本。
            start_number: 正在抽取的来源章节编号。

        Returns:
            清理后的文本，以及是否已遇到下一章节。
        """
        if not start_number:
            return text, False
        pattern = re.compile(r"(?<=[。！？])\s+(\d+(?:\.\d+)+)\s+(?=[\u4e00-\u9fff])")
        for match in pattern.finditer(text):
            number = tuple(int(part) for part in match.group(1).split("."))
            if number > start_number and number[: len(start_number)] != start_number:
                return text[: match.start()].rstrip(), True
        return text, False

    @staticmethod
    def _heading_body(text: str, matched_title: str) -> str:
        """从被 PDF 合并的标题块中分离标题后的首段正文。

        Args:
            text: 原始标题块文本，可能同时包含正文。
            matched_title: 配置命中的标题别名。

        Returns:
            标题后面的正文；标题块没有正文时返回空字符串。
        """
        without_number = ExtractionEngine._strip_numbering(text)
        title_position = without_number.find(matched_title)
        if title_position < 0:
            return ""
        body = without_number[title_position + len(matched_title) :].strip(" ：:、，,\t\r\n")
        return body if len(body) >= 2 else ""

    @classmethod
    def _source_outline(
        cls, blocks: list[DocumentBlock], target_index: int
    ) -> list[dict[str, Any]]:
        """构建目标章节在原报告中的标题层级路径。

        Args:
            blocks: 文档全部内容块。
            target_index: 目标标题在内容块列表中的下标。

        Returns:
            从上级到目标标题排列的标题层级及原文。
        """
        stack: list[dict[str, Any]] = []
        for block in blocks[: target_index + 1]:
            level = cls._heading_level(block)
            number = cls._section_number(block.text)
            # PDF 后端可能只识别出标题文本、没有标题样式，此时用编号深度兜底。
            if level is None and number is not None:
                level = len(number)
            if level is None:
                continue
            # 普通正文即使以数字开头，也不能进入章节路径。
            if not cls._is_numbered_heading_candidate(block):
                continue
            while stack and int(stack[-1]["level"]) >= level:
                stack.pop()
            stack.append(
                {
                    "level": level,
                    "title": block.text,
                    "page": block.page,
                    "block_id": block.id,
                }
            )
        return stack

    @staticmethod
    def _next_section_index(
        blocks: list[DocumentBlock],
        start: int,
        start_level: int | None,
        start_number: tuple[int, ...] | None,
    ) -> int:
        """查找当前章节结束位置。

        Args:
            blocks: 文档全部内容块。
            start: 当前标题下标。
            start_level: 当前标题样式层级。
            start_number: 当前标题编号。

        Returns:
            下一个同级或上级标题的下标，找不到时返回文档长度。
        """
        for index in range(start + 1, len(blocks)):
            block = blocks[index]
            level = ExtractionEngine._heading_level(block)
            number = ExtractionEngine._section_number(block.text)
            # Word 中父、子标题有时被错误地映射成相同 heading_level。
            # 章节编号仍能可靠表明 4.2.1 属于 4.2，因此父章节必须继续包含
            # 这些子章节，不能仅因样式级别相同就在 4.2.1 前截断。
            is_numbered_descendant = bool(
                start_number
                and number
                and len(number) > len(start_number)
                and number[: len(start_number)] == start_number
            )
            by_level = block.kind == "heading" and (
                start_level is None or level is None or level <= start_level
            ) and not is_numbered_descendant
            by_number = bool(
                start_number
                and number
                and ExtractionEngine._is_numbered_heading_candidate(block)
                and number > start_number
                # 正常子章节以当前编号为前缀；若父标题漏识别，遇到不属于
                # 当前编号树的新编号时也必须结束，避免正文串到下一节。
                and number[: len(start_number)] != start_number
            )
            if by_level or by_number:
                return index
        return len(blocks)

    @staticmethod
    def _is_numbered_heading_candidate(block: DocumentBlock) -> bool:
        """判断内容块是否像带编号的真实章节标题。

        Args:
            block: 待判断的统一文档内容块。

        Returns:
            标题，或以“编号 + 中文标题”开头的列表项、段落返回 ``True``；
            图表数字、月份和普通数值段返回 ``False``。
        """
        if block.kind == "heading":
            return True
        if block.kind not in {"list_item", "paragraph"}:
            return False
        # OpenDataLoader 会把章节标题和图表中的数字行都标成 list_item，
        # 因此不能只依赖块类型，还要确认编号后面确实是中文标题文字。
        return bool(
            re.match(
                r"^\s*\d+(?:\.\d+)*(?:(?:\s+|[、．])(?=[\u4e00-\u9fff])|(?=[\u4e00-\u9fff]))",
                block.text,
            )
        )

    @staticmethod
    def _strip_numbering(text: str) -> str:
        """去掉段首常见章节编号但保留正文内部编号。

        Args:
            text: 原始标题或段落。

        Returns:
            清理后的文本。
        """
        return re.sub(
            # 多级编号必须先匹配，否则“2.2”会被单级规则只删除前半个“2.”。
            r"^\s*(?:(?:\d+(?:\.\d+)+)|(?:\d+(?=\s))|(?:\(?[a-zA-Z0-9一二三四五六七八九十]+\)?[、.)）]))\s*",
            "",
            text,
        ).strip()

    def _merge_matrix_tables(
        self,
        blocks: list[DocumentBlock],
        layer_records: list[dict[str, Any]],
        table_configs: list[dict[str, Any]],
    ) -> None:
        """把承载力推荐表等按行组织的表格字段合并到土层记录。

        Args:
            blocks: 文档全部内容块。
            layer_records: 已从正文识别的土层记录。
            table_configs: 矩阵表关键词、表头别名和字段类型配置。
        """
        for table_config in table_configs:
            keywords = [str(value) for value in table_config.get("table_keywords", [])]
            for block_index, block in enumerate(blocks):
                if block.kind != "table" or not block.table:
                    continue
                # 表名通常是表格前一个独立段落，不能只检查 table 块自身文本。
                context_start = max(0, block_index - int(table_config.get("context_blocks", 3)))
                table_context = "\n".join(
                    item.text for item in blocks[context_start : block_index + 1]
                )
                if keywords and not any(keyword in table_context for keyword in keywords):
                    continue
                for table_record in self._extract_matrix_table(block, table_config):
                    target = self._match_layer_record(layer_records, table_record)
                    if target is None:
                        continue
                    for name, value in table_record.items():
                        if name not in {"layer_code", "layer_name", "evidence"} and value is not None:
                            operator = table_config.get("merge_operators", {}).get(name)
                            old_value = target.get(name)
                            if operator == "minimum" and old_value is not None:
                                target[name] = min(old_value, value)
                            elif operator == "maximum" and old_value is not None:
                                target[name] = max(old_value, value)
                            else:
                                target[name] = value
                    target.setdefault("evidence", {}).setdefault("matrix_tables", []).append(
                        table_record["evidence"]
                    )

    def _extract_matrix_table(
        self, block: DocumentBlock, config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """按表头别名从普通行列型表格抽取记录。

        Args:
            block: 结构化表格内容块。
            config: 表头行数、主键列和字段列规则。

        Returns:
            每个数据行对应的一条记录。
        """
        if not block.table:
            return []
        rows: dict[int, dict[int, str]] = {}
        for cell in block.table.cells:
            # 将合并表头文字传播到覆盖列，后续即可把“灌注桩 + 端阻力”组合成
            # 完整列名，而不依赖某个解析后端如何表示合并单元格。
            for row_offset in range(max(1, cell.row_span)):
                for column_offset in range(max(1, cell.column_span)):
                    row = rows.setdefault(cell.row + row_offset, {})
                    column = cell.column + column_offset
                    old = row.get(column, "")
                    cell_value = cell.text.strip()
                    row[column] = f"{old}{cell_value}" if cell_value not in old else old
        configured_header_rows = config.get("header_rows", 2)
        header_row_options = (
            [int(value) for value in configured_header_rows]
            if isinstance(configured_header_rows, list)
            else [int(configured_header_rows)]
        )
        headers: dict[int, str] = {}
        columns: dict[str, int] = {}
        header_rows = header_row_options[0]
        # 同一类指标表可能使用一行或两行表头。逐个尝试配置允许的表头高度，
        # 选择能识别出最多字段的方案，避免把第一页数据行当成第二行表头。
        for candidate_header_rows in header_row_options:
            candidate_headers = {
                column: self._compact_label(
                    "".join(
                        rows.get(row, {}).get(column, "")
                        for row in range(candidate_header_rows)
                    )
                )
                for column in range(block.table.columns)
            }
            candidate_columns: dict[str, int] = {}
            for field_name, field_config in config.get("columns", {}).items():
                # 合并表头有时会把“土层名称”横跨层号、名称两列。此时两列的
                # 表头文本完全相同，单靠正则无法区分。配置可显式声明列序号，
                # 其余普通表格仍继续使用表头别名自动识别。
                configured_column = field_config.get("column_index")
                if isinstance(configured_column, int) and (
                    0 <= configured_column < block.table.columns
                ):
                    candidate_columns[field_name] = configured_column
                    continue
                for column, header in candidate_headers.items():
                    if any(
                        re.search(pattern, header, re.IGNORECASE)
                        for pattern in field_config.get("header_patterns", [])
                    ):
                        candidate_columns[field_name] = column
                        break
            if len(candidate_columns) > len(columns):
                header_rows = candidate_header_rows
                headers = candidate_headers
                columns = candidate_columns
        if not columns:
            return []

        result = []
        for row_number in sorted(rows):
            if row_number < header_rows:
                continue
            record: dict[str, Any] = {}
            for field_name, column in columns.items():
                field_config = config["columns"][field_name]
                raw_value = rows[row_number].get(column, "")
                cleaned = re.sub(r"[*（）()\s]", "", raw_value)
                parsed_value: Any
                if field_config.get("type") == "number":
                    parsed_value = self._first_number(cleaned)
                    if parsed_value is not None:
                        multiplier = float(field_config.get("multiplier", 1.0))
                        header = headers.get(column, "")
                        for unit_rule in field_config.get("unit_rules", []):
                            if re.search(unit_rule["pattern"], header, re.IGNORECASE):
                                multiplier = float(unit_rule["multiplier"])
                                break
                        parsed_value = round(parsed_value * multiplier, 6)
                else:
                    parsed_value = cleaned or None
                    transforms = field_config.get("transforms", [])
                    if isinstance(transforms, str):
                        transforms = [transforms]
                    for transform in transforms:
                        parsed_value = self._apply_transform(parsed_value, transform)
                record[field_name] = parsed_value
            if not record.get("layer_code") and not record.get("layer_name"):
                continue
            record["evidence"] = {
                "block_id": block.id,
                "page": block.page,
                "row": row_number,
                "source_type": "table",
                "headers": headers,
            }
            self._apply_derived_fields([record], config.get("derived_fields", {}))
            for output_name, source_fields in config.get("compose_fields", {}).items():
                composed = {
                    key: record.get(source_name)
                    for key, source_name in source_fields.items()
                }
                if any(value is not None for value in composed.values()):
                    record[output_name] = composed
            result.append(record)
        return result

    @staticmethod
    def _match_layer_record(
        records: list[dict[str, Any]], table_record: dict[str, Any]
    ) -> dict[str, Any] | None:
        """按层号优先、岩土名称其次匹配已有土层。

        Args:
            records: 正文土层记录。
            table_record: 表格中的当前行。

        Returns:
            匹配的土层记录；无法匹配时返回 ``None``。
        """
        code = re.sub(r"\s+", "", str(table_record.get("layer_code") or ""))
        if code:
            matched = next((item for item in records if item.get("layer_code") == code), None)
            if matched is not None:
                return matched
        name = re.sub(r"\s+", "", str(table_record.get("layer_name") or ""))
        return next((item for item in records if item.get("layer_name") == name), None) if name else None

    def _apply_derived_fields(
        self, records: list[dict[str, Any]], derived_fields: dict[str, Any]
    ) -> None:
        """使用配置中的首条命中规则计算土层派生字段。

        Args:
            records: 岩土层记录。
            derived_fields: 默认值及条件规则配置。
        """
        # 规则按配置顺序执行并采用首条命中项，YAML 中更具体的条件应放在前面。
        for record in records:
            for field_name, field_config in derived_fields.items():
                existing_value = record.get(field_name)
                only_when_missing = bool(field_config.get("only_when_missing"))
                # 标量已有报告值时完全保留。组合参数可能只给出一种桩型，仍需
                # 继续计算规则值，以便只补齐字典中缺失的另一种桩型。
                if only_when_missing and existing_value is not None and not isinstance(
                    existing_value, dict
                ):
                    continue
                # 某些业务默认值只有在“统计平均值、推荐值等所有来源都缺失”时
                # 才能启用，避免默认值覆盖报告表格中的推荐参数。
                missing_fields = [
                    str(value) for value in field_config.get("only_when_all_missing", [])
                ]
                if missing_fields and any(record.get(name) is not None for name in missing_fields):
                    continue
                formula = field_config.get("formula")
                value = self._evaluate_formula(record, formula) if formula else field_config.get("default")
                for rule in field_config.get("rules", []):
                    if self._condition_matches(record, rule.get("when", {})):
                        value = rule.get("value")
                        break
                if value is not None:
                    if (
                        only_when_missing
                        and isinstance(existing_value, dict)
                        and isinstance(value, dict)
                    ):
                        # 报告明确值优先；只有不存在或为 None 的桩型才使用规则值。
                        merged_value = dict(value)
                        merged_value.update(
                            {
                                key: item
                                for key, item in existing_value.items()
                                if item is not None
                            }
                        )
                        value = merged_value
                    record[field_name] = value
                    source_label = field_config.get("source_label")
                    if source_label:
                        record.setdefault("derived_field_sources", {})[field_name] = str(
                            source_label
                        )

    @staticmethod
    def _evaluate_formula(record: dict[str, Any], formula: dict[str, Any]) -> float | None:
        """计算配置声明的简单两字段算式。

        Args:
            record: 当前岩土层数据。
            formula: 运算符、左右字段名和小数位配置。

        Returns:
            计算结果；输入缺失或除数为零时返回 ``None``。
        """
        # 仅开放四则运算，不执行 eval，避免配置文件获得任意代码执行能力。
        left_field = formula.get("left")
        right_field = formula.get("right")
        if not isinstance(left_field, str) or not isinstance(right_field, str):
            return None
        left = record.get(left_field)
        right = record.get(right_field)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return None
        operator = formula.get("operator")
        if operator == "divide" and right != 0:
            value = left / right
        elif operator == "multiply":
            value = left * right
        elif operator == "add":
            value = left + right
        elif operator == "subtract":
            value = left - right
        else:
            return None
        return round(value, int(formula.get("precision", 4)))

    @staticmethod
    def _condition_matches(record: dict[str, Any], condition: dict[str, Any]) -> bool:
        """判断一条简单的配置化条件是否命中。

        Args:
            record: 当前岩土层数据。
            condition: contains、数值比较或 all/any 组合条件。

        Returns:
            条件是否成立。
        """
        # all/any 允许组合条件，叶子节点只支持白名单内的字符串和数值比较。
        if "all" in condition:
            return all(ExtractionEngine._condition_matches(record, item) for item in condition["all"])
        if "any" in condition:
            return any(ExtractionEngine._condition_matches(record, item) for item in condition["any"])
        field = condition.get("field")
        if not isinstance(field, str):
            return False
        value = record.get(field)
        # 同一个业务指标可能来自统计表或“建议值/推荐值”表。派生规则优先使用
        # 统计值，缺失时自动回退到 *_recommended，避免在 YAML 中重复两套规则。
        if value is None:
            value = record.get(f"{field}_recommended")
        if "contains_any" in condition:
            return any(word in str(value or "") for word in condition["contains_any"])
        if "not_contains_any" in condition:
            return not any(word in str(value or "") for word in condition["not_contains_any"])
        if value is None:
            return False
        if "lt" in condition:
            return value < condition["lt"]
        if "lte" in condition:
            return value <= condition["lte"]
        if "gt" in condition:
            return value > condition["gt"]
        if "gte" in condition:
            return value >= condition["gte"]
        if "eq" in condition:
            return value == condition["eq"]
        return bool(value)

    def _extract_fields(self, text: str, fields: dict[str, Any]) -> dict[str, Any]:
        """按照字段配置从一段文本中提取值。

        Args:
            text: 已完成基础清洗的候选文本。
            fields: 字段正则、类型和转换配置。

        Returns:
            当前候选记录的字段字典。
        """
        record: dict[str, Any] = {}
        # 先提取直接来自文本的字段，再计算依赖其他字段的派生字段。
        for name, field_config in fields.items():
            if "source" in field_config:
                continue
            value = None
            for pattern in field_config.get("patterns", []):
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = match.groupdict().get("value", match.group(0))
                    break
            record[name] = self._convert_value(value, field_config)

        for name, field_config in fields.items():
            if "source" not in field_config:
                continue
            value = record.get(field_config["source"])
            record[name] = self._apply_transform(value, field_config.get("transform"))
        return record

    @staticmethod
    def _normalize_text(text: str) -> str:
        """统一常见全角符号、区间符号和空白。

        Args:
            text: 原始候选文本。

        Returns:
            适合正则匹配的规范文本。
        """
        replacements = {"（": "(", "）": ")", "：": ":", "，": ",", "～": "~", "—": "-", "－": "-", "ｍ": "m"}
        for source, target in replacements.items():
            text = text.replace(source, target)
        # PDF 按视觉行提取时可能在中文词语中插入空格，例如“反 应谱”。
        text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def _convert_value(self, value: Any, field_config: dict[str, Any]) -> Any:
        """按字段类型和转换器处理提取值。

        Args:
            value: 正则提取到的原始值。
            field_config: 字段类型与转换配置。

        Returns:
            转换后的字段值。
        """
        if value is None:
            return None
        if field_config.get("type") == "number":
            value = float(value)
        transforms = field_config.get("transforms", [])
        if isinstance(transforms, str):
            transforms = [transforms]
        for transform in transforms:
            value = self._apply_transform(value, transform)
        return value

    @staticmethod
    def _apply_transform(value: Any, transform: str | None) -> Any:
        """执行内置、安全的字段转换。

        Args:
            value: 待转换字段值。
            transform: 配置中声明的转换器名称。

        Returns:
            转换后的字段值。

        Raises:
            ValueError: 配置引用了未注册的转换器。
        """
        if value is None or not transform:
            return value
        if transform == "strip_spaces":
            return re.sub(r"\s+", "", str(value))
        if transform == "extract_layer_code":
            match = re.search(
                r"(?:第|层)?([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|\d+)"
                r"(?:[-－]?([0-9]+))?",
                re.sub(r"\s+", "", str(value)),
            )
            if not match:
                return None
            code = match.group(1) + (f"-{match.group(2)}" if match.group(2) else "")
            return ExtractionEngine._apply_transform(code, "normalize_layer_code")
        if transform == "extract_layer_name":
            text = re.sub(r"\s+", "", str(value))
            text = re.sub(
                r"^(?:第|层)?(?:[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|\d+)"
                r"(?:[-－]?[0-9]+)?(?:层)?",
                "",
                text,
            )
            return text or None
        if transform == "main_layer_code":
            match = re.match(r"([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|\d+)", str(value))
            return match.group(1) if match else value
        if transform == "normalize_layer_code":
            text = re.sub(r"\s+", "", str(value)).replace("－", "-")
            match = re.fullmatch(
                r"(?:第|层)?([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|\d+)"
                r"(?:-?(\d+))?(?:层)?",
                text,
            )
            if not match:
                return text
            main, child = match.groups()
            # ``str.isdigit`` 对①这类带圈数字也返回 True，只转换 ASCII 层号。
            if main.isascii() and main.isdigit():
                circled = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
                number = int(main)
                main = circled[number - 1] if 1 <= number <= len(circled) else main
            return main + (f"-{child}" if child else "")
        if transform == "normalize_foundation_treatment":
            text = re.sub(r"\s+", "", str(value))
            if "岩溶" in text or "溶岩" in text:
                return "岩溶"
            depth = re.search(
                r"(?:埋深|深度)[^0-9]{0,5}(\d+(?:\.\d+)?)\s*(?:m|米)?",
                text,
                re.IGNORECASE,
            )
            # 原文没有给出埋深时保留定性结论，不能制造“x米”占位值。
            return f"湿陷性黄土埋深{depth.group(1)}米" if depth else "湿陷性黄土"
        raise ValueError(f"未知字段转换器: {transform}")

    @staticmethod
    def _select_records(records: list[dict[str, Any]], selection: dict[str, Any]) -> list[dict[str, Any]]:
        """按配置分组并选择有效数值最小或最大的记录。

        Args:
            records: 全部候选记录。
            selection: 分组字段、数值优先级和选择操作配置。

        Returns:
            每组选择出的最终候选记录。
        """
        group_by = selection.get("group_by")
        value_priority = selection.get("value_priority", [])
        operator = selection.get("operator", "minimum")
        if not group_by:
            return records

        groups: dict[Any, list[dict[str, Any]]] = {}
        for record in records:
            group = record.get(group_by)
            if group is None:
                continue
            groups.setdefault(group, []).append(record)

        selected = []
        for group, candidates in groups.items():
            # 优先级作用于整组：只要组内存在平均厚度，就不会拿最大揭露厚度
            # 与平均厚度直接比较。
            effective_field = next(
                (
                    field
                    for field in value_priority
                    if any(candidate.get(field) is not None for candidate in candidates)
                ),
                None,
            )
            if effective_field is None:
                # 岩层可能只有名称和描述、没有可量化厚度。完整地层清单仍应
                # 保留该层，不能因为缺少厚度而从最终结果中删除。
                item = dict(candidates[0])
            else:
                effective_candidates = []
                for candidate in candidates:
                    if candidate.get(effective_field) is None:
                        continue
                    effective_item = dict(candidate)
                    effective_item["effective_field"] = effective_field
                    effective_item["effective_value"] = candidate[effective_field]
                    effective_candidates.append(effective_item)
                chooser = max if operator == "maximum" else min
                item = dict(
                    chooser(
                        effective_candidates,
                        key=lambda candidate: candidate["effective_value"],
                    )
                )
            # 浅拷贝会让所选候选与最终记录共享 evidence 字典；随后把候选证据
            # 放入 merged_sources 时会形成自引用，导致 details JSON 无法序列化。
            item["evidence"] = dict(item.get("evidence") or {})
            # 厚度可能来自图片，而物理力学参数来自正文统计表或推荐值表。
            # 最终记录按层号合并非空字段，避免因选择厚度候选而丢失其他业务结果。
            merged_sources = []
            for candidate in candidates:
                for field_name, field_value in candidate.items():
                    if field_name == "evidence" or field_value is None:
                        continue
                    if item.get(field_name) is None:
                        item[field_name] = field_value
                if candidate.get("evidence"):
                    merged_sources.append(candidate["evidence"])
            if merged_sources:
                item.setdefault("evidence", {})["merged_sources"] = merged_sources
            if effective_field is None:
                item["selection_rule"] = f"按 {group_by}={group} 分组，无厚度值但保留完整土层"
            else:
                item["selection_rule"] = f"按 {group_by}={group} 分组，选择{operator}有效值"
                item["final_value"] = item["effective_value"]
            selected.append(item)
        if selection.get("sort") == "layer_code":
            selected.sort(key=lambda item: ExtractionEngine._layer_sort_key(item.get(group_by)))
        return selected

    @staticmethod
    def _layer_sort_key(value: Any) -> tuple[int, int, str]:
        """把圈号或数字层号转换为稳定的地质顺序键。

        Args:
            value: 例如 ``②-1``、``⑩`` 或 ``10-1`` 的完整层号。

        Returns:
            主层序号、夹层序号和原始文本组成的排序键。
        """
        text = re.sub(r"\s+", "", str(value or ""))
        circled = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
        match = re.match(rf"([{circled}]|\d+)(?:-(\d+))?", text)
        if not match:
            return 10_000, 10_000, text
        main_text, child_text = match.groups()
        main = circled.index(main_text) + 1 if main_text in circled else int(main_text)
        child = int(child_text) if child_text is not None else 0
        return main, child, text

    @staticmethod
    def _postprocess(
        records: list[dict[str, Any]],
        postprocess: dict[str, Any],
        all_records: list[dict[str, Any]] | None = None,
    ) -> None:
        """执行简单的结果后处理规则。

        Args:
            records: 已完成分组选择的记录列表。
            postprocess: 最后一组数值调整配置。
            all_records: 全部土层记录，用于确认被调整的确实是地质顺序最后一层。
        """
        adjustment = postprocess.get("last_group_add")
        last_group = records[-1].get("main_layer_code") if records else None
        document_last_group = None
        if all_records:
            ordered = [record for record in all_records if record.get("layer_code") is not None]
            if ordered:
                document_last_group = max(
                    ordered,
                    key=lambda item: ExtractionEngine._layer_sort_key(item.get("layer_code")),
                ).get("main_layer_code")
        is_document_last = document_last_group is None or last_group == document_last_group
        last_effective_value = records[-1].get("effective_value") if records else None
        allowed_fields = [str(value) for value in postprocess.get("last_group_add_fields", [])]
        last_effective_field = records[-1].get("effective_field") if records else None
        field_allows_adjustment = not allowed_fields or last_effective_field in allowed_fields
        if (
            records
            and adjustment is not None
            and is_document_last
            and field_allows_adjustment
            and isinstance(last_effective_value, (int, float))
        ):
            records[-1]["adjustment"] = float(adjustment)
            records[-1]["final_value"] = last_effective_value + float(adjustment)
            records[-1]["adjustment_reason"] = f"最后一层厚度增加 {adjustment}m"


def merge_first_cultivated_soil_layer(
    layers: Any,
    excluded_names: Iterable[str] = ("耕土",),
) -> list[dict[str, Any]]:
    """删除首层耕土，并将其厚度合并到下一层业务记录。

    Args:
        layers: 已按地质顺序排列的土层记录。
        excluded_names: 需要从首层排除的土层名称关键字，默认仅为“耕土”。

    Returns:
        复制后的业务土层列表。满足条件时删除首层，并将首层厚度累加到下一层
        的有效厚度、最终厚度和厚度统计字段。

    Notes:
        原始候选记录不会被修改；抽取流程从本方法返回后，所有派生计算均使用
        合并后的新列表。
    """
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
        """按照业务厚度优先级读取一条土层的当前厚度。"""
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
        # selection 记录使用 effective/final，精简 result 使用 thickness；分别更新
        # 实际存在的字段，保证任何后续入口读取到的都是合并后厚度。
        updated = False
        for field in ("effective_value", "final_value", "thickness"):
            if isinstance(following.get(field), (int, float)):
                following[field] = merged_thickness
                updated = True
        if not updated:
            following["final_value"] = merged_thickness

        # 两层都有明确统计值时同步合并，避免范围与最终厚度互相矛盾。
        for field in ("thickness_min", "thickness_max", "thickness_average"):
            first_value = first.get(field)
            following_value = following.get(field)
            if isinstance(first_value, (int, float)) and isinstance(
                following_value, (int, float)
            ):
                following[field] = round(float(first_value) + float(following_value), 6)
        following["merged_cultivated_soil"] = {
            "layer_code": first.get("layer_code"),
            "layer_name": first.get("layer_name"),
            "thickness": first_thickness,
        }
    return copied_layers[1:]


def _write_json_file(path: str | Path, data: Any) -> Path:
    """将数据以 UTF-8 格式写入 JSON 文件并返回目标路径。

    该辅助函数只收敛目录创建、JSON 序列化和文件写入，不改变任何业务数据。

    Args:
        path: JSON 输出路径。
        data: 可被 ``json.dumps`` 序列化的数据。

    Returns:
        已完成写入的 ``Path`` 对象。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target


def extract_document(
    input_path: str | Path,
    config_paths: str | Path | Iterable[str | Path],
    *,
    pdf_backend: str = "opendataloader",
    output_path: str | Path | None = None,
    details_output_path: str | Path | None = None,
    field_descriptions_output_path: str | Path | None = None,
    image_recognizer: Any | None = None,
) -> dict[str, Any]:
    """解析文档并执行一个或多个配置化抽取任务。

    Args:
        input_path: Word 或 PDF 文件路径。
        config_paths: 一个或多个 YAML 抽取配置路径。
        pdf_backend: PDF 解析后端名称。
        output_path: 可选的精简结果 JSON 输出路径。
        details_output_path: 可选的完整查询明细 JSON 输出路径。
        field_descriptions_output_path: 可选的结果字段中文说明 JSON 输出路径。
        image_recognizer: 可选的钻孔柱状图识别器。

    Returns:
        包含文档信息和各配置任务结果的字典。

    Raises:
        ValueError: 抽取配置不合法。
    """
    started_at = time.perf_counter()
    logger.info("开始解析文档：%s", Path(input_path))
    # 输入文件只解析一次，所有 YAML 任务复用同一个统一文档模型。
    document = DocumentParser(ParserConfig(pdf_backend=pdf_backend)).parse(input_path)
    logger.info(
        "文档解析完成：后端=%s，页数=%d，内容块=%d，耗时=%.2f 秒",
        document.parser_backend,
        len(document.pages),
        len(document.blocks),
        time.perf_counter() - started_at,
    )

    engine = ExtractionEngine.from_files(
        config_paths,
        image_recognizer=image_recognizer,
    )
    logger.info("已加载 %d 个抽取配置，开始执行抽取", len(engine.configs))
    extraction_started_at = time.perf_counter()
    result = engine.extract_all(document)
    for name, task in result["tasks"].items():
        recognized_count = len(task.get("records", []))
        selected_count = len(task.get("selected_records", []))
        warning_count = len(task.get("warnings", []))
        if task.get("mode") == "layer_records":
            logger.info(
                "抽取完成：任务=%s，状态=%s，原文识别=%d层，有效结果=%d层，警告=%d",
                name,
                task["status"],
                recognized_count,
                selected_count,
                warning_count,
            )
        else:
            logger.info(
                "抽取完成：任务=%s，状态=%s，候选命中=%d处，有效结果=%d项，警告=%d",
                name,
                task["status"],
                recognized_count,
                selected_count,
                warning_count,
            )
        for warning in task.get("warnings", []):
            logger.warning("任务 %s：%s", name, warning)
    logger.info("全部抽取完成，耗时=%.2f 秒", time.perf_counter() - extraction_started_at)
    if output_path is not None:
        # result.json 面向业务使用；只保留 PRD 结果，不携带候选与原文证据。
        compact_result = _compact_result(result)
        target = _write_json_file(output_path, compact_result)
        logger.info("精简结果 JSON 已写入：%s", target.resolve())
    if details_output_path is not None:
        # details.json 保留完整任务结构，用于定位页码、OCR 置信度和选值过程。
        details_target = _write_json_file(details_output_path, result)
        logger.info("查询明细 JSON 已写入：%s", details_target.resolve())
    if field_descriptions_output_path is not None:
        # 字段说明与报告数据分开存放，业务系统读取 result.json 时无需过滤说明信息。
        descriptions = _result_field_descriptions(
            Path(output_path).name if output_path is not None else None
        )
        descriptions_target = _write_json_file(field_descriptions_output_path, descriptions)
        logger.info("字段说明 JSON 已写入：%s", descriptions_target.resolve())
    return result


def _result_field_descriptions(result_filename: str | None = None) -> dict[str, Any]:
    """生成精简结果 JSON 的中文字段说明。

    Args:
        result_filename: 字段说明所对应的精简结果文件名。

    Returns:
        包含顶层字段、土层字段和嵌套对象字段含义的说明字典。
    """
    layer_fields = {
        "layer_code": {"type": "string", "description": "规范化土层编号，例如②-1。"},
        "layer_name": {"type": "string", "description": "岩土名称，例如粉土、粉质黏土。"},
        "geological_age": {"type": "string", "description": "地质年代或成因代号，例如Q4 al。"},
        "boreholes": {
            "type": "array<object>",
            "description": "该土层在不同钻孔中的实际观测数据；报告没有钻孔柱状图时不输出。",
            "item_fields": {
                "borehole_id": {"type": "string", "description": "钻孔编号。"},
                "thickness": {"type": "number", "unit": "m", "description": "该钻孔中本层厚度。"},
                "bottom_depth": {"type": "number", "unit": "m", "description": "该钻孔中本层层底深度。"},
                "bottom_elevation": {"type": "number", "unit": "m", "description": "该钻孔中本层层底标高。"},
            },
        },
        "thickness_range": {
            "type": "object",
            "description": "正文给出的层厚范围。",
            "fields": {
                "min": {"type": "number", "unit": "m", "description": "最小层厚。"},
                "max": {"type": "number", "unit": "m", "description": "最大层厚。"},
                "unit": {"type": "string", "description": "厚度单位，当前为m。"},
            },
        },
        "average_thickness": {"type": "number", "unit": "m", "description": "原文明确给出的平均层厚。"},
        "maximum_exposed_thickness": {"type": "number", "unit": "m", "description": "未揭穿土层的最大揭露厚度。"},
        "thickness": {"type": "number", "unit": "m", "description": "按需求规则得到的最终业务厚度；末层可能已增加20m。"},
        "unit": {"type": "string", "description": "当前土层厚度字段的单位。"},
        "gravity_density": {"type": "number", "unit": "kN/m³", "description": "天然重度或重力密度γ。"},
        "cohesion": {"type": "number", "unit": "kPa", "description": "黏聚力或内聚力C。"},
        "friction_angle": {"type": "number", "unit": "度", "description": "内摩擦角或摩擦角Φ。"},
        "compression_modulus_es1_2": {"type": "number", "unit": "MPa", "description": "压缩模量Es1-2。"},
        "side_friction_fs": {"type": "number", "unit": "kPa", "description": "侧摩阻力fs平均值。"},
        "pile_tip_resistance_rho_c": {"type": "number", "description": "需求定义的桩端阻力ρc平均值。"},
        "poisson_ratio": {"type": "number", "description": "泊松比；原文缺失时按需求规则赋值。"},
        "bearing_capacity_fak": {"type": "number", "unit": "kPa", "description": "地基承载力特征值fak。"},
        "width_bearing_coefficient_eta_b": {"type": "number", "description": "地基承载力宽度修正系数ηb。"},
        "depth_bearing_coefficient_eta_d": {"type": "number", "description": "地基承载力深度修正系数ηd。"},
        "seismic_bearing_coefficient_zeta_a": {"type": "number", "description": "地基抗震承载力调整系数ζa。"},
        "liquefaction_reduction_coefficient": {"type": "number", "description": "液化土层承载力折减系数；无液化时为1。"},
        "horizontal_resistance_coefficient": {"type": "number", "description": "按土层名称自动选择的水平抗力比例系数m。"},
        "uplift_coefficient": {"type": "number", "description": "抗拔系数。"},
        "side_resistance_size_effect": {"type": "number", "description": "桩侧阻力尺寸效应系数。"},
        "tip_resistance_size_effect": {"type": "number", "description": "桩端阻力尺寸效应系数。"},
        "negative_friction_coefficient": {"type": "number", "description": "按土层名称自动选择的负摩阻力系数。"},
        "pile_side_resistance": {"type": "number", "unit": "kPa", "description": "按该层自动判断的基础形式得到的桩侧阻力。"},
        "pile_tip_resistance": {"type": "number", "unit": "kPa", "description": "按该层自动判断的基础形式得到的桩端阻力。"},
    }
    return _without_empty_values(
        {
            "schema_version": "1.0",
            "result_file": result_filename,
            "description": "精简业务结果字段说明。result.json不输出空值，候选、页码和置信度请查看details.json。",
            "fields": {
                "geotechnical_layer_parameters": {
                    "type": "array<object>",
                    "description": "岩土层厚度、物理力学指标和承载力参数。",
                    "item_fields": layer_fields,
                },
                "seismic_parameters": {
                    "type": "object",
                    "description": "地震作用及场地类别参数。",
                    "fields": {
                        "peak_ground_acceleration_g": {"type": "number", "unit": "g", "description": "设计地震动峰值加速度。"},
                        "basic_seismic_intensity_degree": {"type": "number", "unit": "度", "description": "抗震设防烈度或地震基本烈度。"},
                        "response_spectrum_characteristic_period_s": {"type": "number", "unit": "s", "description": "地震动加速度反应谱特征周期。"},
                        "site_category": {"type": "string|array<string>", "description": "建筑场地类别。"},
                        "design_earthquake_group": {"type": "string", "description": "设计地震分组。"},
                        "liquefaction_status": {"type": "string", "description": "场地液化判定结论。"},
                    },
                },
                "key_data": {
                    "type": "object",
                    "description": "水土腐蚀性、地基处理、持力层和地下水等关键数据。",
                    "fields": {
                        "water_soil_corrosion": {"type": "string", "description": "地下水或地基土腐蚀性结论。"},
                        "foundation_treatment": {"type": "array<string>", "description": "地基处理相关结论。"},
                        "bearing_layer_description": {"type": "array<string>", "description": "建议持力层及相关描述。"},
                        "groundwater_depth_m": {"type": "number|object", "unit": "m", "description": "地下水稳定水位埋深或埋深范围。"},
                    },
                },
                "draft_content": {"type": "object<string,string|null>", "description": "以源报告真实章节标题为key、章节正文为value的文字稿内容。"},
                "site_geological_conditions_and_evaluation": {
                    "type": "object",
                    "description": "场区工程地质条件及稳定性评价。",
                    "fields": {
                        "conditions": {"type": "string", "description": "场区工程地质条件正文。"},
                        "evaluation": {"type": "string", "description": "场地稳定性或适宜性评价正文。"},
                    },
                },
                "regional_hydrology": {"type": "string", "description": "区域水文、水系或水文气象正文。"},
                "conclusion_and_evaluation": {"type": "string", "description": "报告结论与评价正文。"},
            },
        }
    )


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    """从完整抽取数据生成便于业务使用的精简结果。

    Args:
        result: 包含候选、证据和最终选择结果的完整抽取数据。

    Returns:
        去掉候选记录、原文证据、表格结构和图片坐标后的结果。
    """
    tasks = result.get("tasks", {})
    layer_tasks = _tasks_for_mode(tasks, "layer_records", "layer_thickness")
    field_tasks = _tasks_for_mode(tasks, "keyword_fields", "report_fields")
    section_tasks = _tasks_for_mode(tasks, "section_content", "report_sections")
    selected_layer_records = [
        record for task in layer_tasks for record in task.get("selected_records", [])
    ]
    # 兼容旧 details 数据或外部直接调用：统一公共方法是幂等的，已经合并过的
    # 结果不会再次增加厚度。
    selected_layer_records = merge_first_cultivated_soil_layer(
        selected_layer_records, ["耕土"]
    )
    field_values: dict[str, Any] = {}
    for task in field_tasks:
        field_values.update(task.get("values", {}))
    section_records = {
        str(record.get("section")): record
        for task in section_tasks
        for record in task.get("selected_records", [])
    }

    def section_value(key: str, *, numbered_title: bool = False) -> dict[str, Any] | None:
        """返回指定章节的精简结果。

        Args:
            key: 章节配置键。
            numbered_title: 是否把大纲路径组合到输出标题前。

        Returns:
            精简章节记录；章节未配置时返回 ``None``。
        """
        record = section_records.get(key)
        return (
            _compact_record(record, "section_content", numbered_title=numbered_title)
            if record
            else None
        )

    draft_content: dict[str, str | None] = {}
    for key, record in section_records.items():
        if record.get("include_in_draft", True) is False:
            continue
        value = section_value(key)
        if value is not None and record is not None:
            sources = record.get("sources")
            if isinstance(sources, list) and sources:
                # 汇总型目标拆成真实的来源小节，避免把多个标题拼成一个键。
                for source in sources:
                    if not isinstance(source, dict):
                        continue
                    title = _real_source_title(source)
                    if title:
                        draft_content.setdefault(title, source.get("text"))
                continue
            # 缺少真实来源（例如待外部搜索）时，才使用配置中的占位标题。
            title = str(value.get("output_title") or key)
            text = None if value.get("status") == "pending" else value.get("text")
            draft_content.setdefault(title, text)

    def section_text(key: str) -> str | None:
        """返回指定章节的业务正文。

        Args:
            key: 章节配置键。

        Returns:
            已抽取的正文；章节缺失或尚待处理时返回 ``None``。
        """
        record = section_records.get(key)
        if not record or record.get("status") == "pending":
            return None
        text = record.get("text")
        return str(text) if text is not None else None

    key_data = {
        "water_soil_corrosion": field_values.get("corrosion"),
        "foundation_treatment": field_values.get("foundation_treatment", []),
        "bearing_layer_description": field_values.get("bearing_layer_description", []),
        "groundwater_depth_m": field_values.get("groundwater_depth"),
    }
    known_field_names = {
        "seismic_peak_acceleration",
        "seismic_intensity",
        "characteristic_period",
        "site_category",
        "earthquake_group",
        "corrosion",
        "foundation_treatment",
        "bearing_layer_description",
        "groundwater_depth",
        "liquefaction_status",
    }
    # 新增关键词配置时自动进入精简业务结果，不依赖修改 Python 固定字段表。
    key_data.update(
        {name: value for name, value in field_values.items() if name not in known_field_names}
    )

    layer_parameters = [
        _compact_record(record, "layer_records")
        for record in selected_layer_records
    ]
    if field_values.get("liquefaction_status") == "no_liquefaction":
        for layer in layer_parameters:
            if layer.get("liquefaction_reduction_coefficient") is None:
                layer["liquefaction_reduction_coefficient"] = 1.0

    # 顶层业务分类保持稳定，分类内部则删除空值，避免 result.json 被大量
    # ``null``、空数组和空对象占满。完整候选与证据仍保存在 details.json。
    return {
        "geotechnical_layer_parameters": layer_parameters,
        "seismic_parameters": _without_empty_values(
            {
                "peak_ground_acceleration_g": field_values.get("seismic_peak_acceleration"),
                "basic_seismic_intensity_degree": field_values.get("seismic_intensity"),
                "response_spectrum_characteristic_period_s": field_values.get("characteristic_period"),
                "site_category": field_values.get("site_category"),
                "design_earthquake_group": field_values.get("earthquake_group"),
                "liquefaction_status": field_values.get("liquefaction_status"),
            }
        ),
        "key_data": _without_empty_values(key_data),
        "draft_content": draft_content,
        "site_geological_conditions_and_evaluation": _without_empty_values(
            {
                "conditions": section_text("site_geological_conditions"),
                "evaluation": section_text("site_stability_evaluation"),
            }
        ),
        "regional_hydrology": section_text("hydro_meteorology"),
        "conclusion_and_evaluation": section_text("conclusion"),
    }


def _borehole_sort_key(value: str) -> tuple[str, int, str]:
    """生成钻孔编号的自然排序键。

    Args:
        value: 例如 ``F03``、``ZK12`` 的钻孔编号。

    Returns:
        字母前缀、数字序号和原值组成的排序键。
    """
    match = re.fullmatch(r"([A-Za-z]+)0*(\d+)", value.strip())
    if not match:
        return value.upper(), -1, value
    return match.group(1).upper(), int(match.group(2)), value


def _tasks_for_mode(
    tasks: dict[str, Any], mode: str, legacy_name: str
) -> list[dict[str, Any]]:
    """按抽取模式收集任务，兼容早期没有 mode 字段的结果。

    Args:
        tasks: 按配置名称组织的任务结果。
        mode: 需要收集的公共抽取模式。
        legacy_name: 旧版结果使用的固定任务名称。

    Returns:
        匹配该模式的全部任务；没有模式信息时返回旧名称对应任务。
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


def _real_source_title(source: dict[str, Any]) -> str | None:
    """从单个来源记录中生成不含正文的真实章节标题。

    Args:
        source: 来源章节记录。

    Returns:
        真实章节标题；来源信息不足时返回 ``None``。
    """
    raw_title = str(source.get("title") or "").strip()
    alias = str(source.get("matched_alias") or "").strip()
    if not raw_title or not alias:
        return None
    alias_position = raw_title.find(alias)
    if alias_position >= 0:
        title = raw_title[: alias_position + len(alias)]
    else:
        number = ExtractionEngine._section_number(raw_title)
        prefix = ".".join(str(value) for value in number) if number else ""
        title = f"{prefix} {alias}" if prefix else alias
    return re.sub(r"\s+", " ", title).strip() or None


def _compact_record(
    record: dict[str, Any],
    mode: str,
    *,
    numbered_title: bool = False,
) -> dict[str, Any]:
    """按任务类型清理单条最终记录中的查询辅助字段。

    Args:
        record: 完整的最终记录。
        mode: 抽取任务模式。
        numbered_title: 是否输出带大纲编号的章节标题。

    Returns:
        只包含业务结果字段的记录。
    """
    if mode == "section_content":
        compact = {
            key: record[key]
            for key in ("output_title", "outline_path", "text")
            if key in record
        }
        if numbered_title and compact.get("output_title"):
            outline_path = compact.get("outline_path")
            if isinstance(outline_path, list) and outline_path:
                section_number = ".".join(str(value) for value in outline_path)
                compact["output_title"] = f"{section_number} {compact['output_title']}"
        # 未完成的外部补充属于业务结果，保留占位状态和后续查询模板。
        if record.get("status") == "pending":
            compact["status"] = "pending"
            compact["placeholder"] = record.get("placeholder")
        return compact

    # 最后一层先使用 PRD 规定的 +20m 工程厚度，同时保留原始揭露厚度。
    reported_thickness = record.get("final_value")
    if reported_thickness is None:
        reported_thickness = record.get("effective_value")
    if reported_thickness is None:
        reported_thickness = record.get("thickness_average")
    if reported_thickness is None:
        reported_thickness = record.get("maximum_exposed")
    layer_name = str(record.get("layer_name") or "")
    selected_side_resistance = select_foundation_parameter(
        layer_name,
        cast_in_place=record.get("cast_in_place_side_resistance"),
        precast=record.get("precast_side_resistance"),
    )
    selected_tip_resistance = select_foundation_parameter(
        layer_name,
        cast_in_place=record.get("cast_in_place_tip_resistance"),
        precast=record.get("precast_tip_resistance"),
    )
    horizontal_resistance = record.get("horizontal_resistance_coefficient")
    negative_friction = record.get("negative_friction_coefficient")
    if isinstance(horizontal_resistance, dict):
        horizontal_resistance = select_foundation_parameter(
            layer_name,
            cast_in_place=horizontal_resistance.get("cast_in_place"),
            precast=horizontal_resistance.get("precast"),
        )
    if isinstance(negative_friction, dict):
        negative_friction = select_foundation_parameter(
            layer_name,
            cast_in_place=negative_friction.get("cast_in_place"),
            precast=negative_friction.get("precast"),
        )

    def parameter_value(primary: str, recommended: str) -> Any:
        """优先返回统计平均值，缺失时再采用推荐值。

        Args:
            primary: 统计平均值字段名。
            recommended: 推荐值字段名。

        Returns:
            参数值；两个来源都没有值时返回 ``None``。
        """
        if record.get(primary) is not None:
            return record[primary]
        if record.get(recommended) is not None:
            return record[recommended]
        return None

    gravity_density = parameter_value(
        "gravity_density", "gravity_density_recommended"
    )
    cohesion = parameter_value("cohesion", "cohesion_recommended")
    friction_angle = parameter_value(
        "friction_angle", "friction_angle_recommended"
    )
    compression_modulus = parameter_value(
        "compression_modulus", "compression_modulus_recommended"
    )

    # 这里列出的都是 PRD 2.2.2 的最终业务指标。标准贯入击数、候选来源、
    # 控制钻孔编号等只是计算或追溯依据，因此仅保留在 details.json。
    compact = {
        "layer_code": record.get("layer_code"),
        "layer_name": record.get("layer_name"),
        "geological_age": record.get("geological_age"),
        # 以土层为主，并在该层下列出所有钻孔中的实际观测值。
        "boreholes": _compact_layer_boreholes(record),
        "thickness_range": (
            {
                "min": record.get("thickness_min"),
                "max": record.get("thickness_max"),
                "unit": "m",
            }
            if record.get("thickness_min") is not None
            or record.get("thickness_max") is not None
            else None
        ),
        "average_thickness": record.get("thickness_average"),
        "maximum_exposed_thickness": record.get("maximum_exposed"),
        # thickness 为业务最终厚度；末层已按 PRD 加 20m。
        "thickness": reported_thickness,
        "unit": (
            "m"
            if reported_thickness is not None
            or record.get("thickness_min") is not None
            or record.get("thickness_max") is not None
            or record.get("maximum_exposed") is not None
            or record.get("observations")
            else None
        ),
        "gravity_density": gravity_density,
        "cohesion": cohesion,
        "friction_angle": friction_angle,
        "compression_modulus_es1_2": compression_modulus,
        "side_friction_fs": record.get("side_friction"),
        "pile_tip_resistance_rho_c": record.get("pile_tip_resistance"),
        "poisson_ratio": record.get("poisson_ratio"),
        "bearing_capacity_fak": record.get("bearing_capacity"),
        "width_bearing_coefficient_eta_b": record.get("width_bearing_coefficient"),
        "depth_bearing_coefficient_eta_d": record.get("depth_bearing_coefficient"),
        "seismic_bearing_coefficient_zeta_a": record.get("seismic_bearing_coefficient"),
        "liquefaction_reduction_coefficient": record.get(
            "liquefaction_reduction_coefficient"
        ),
        "horizontal_resistance_coefficient": horizontal_resistance,
        "uplift_coefficient": record.get("uplift_coefficient"),
        "side_resistance_size_effect": record.get("side_resistance_size_effect"),
        "tip_resistance_size_effect": record.get("tip_resistance_size_effect"),
        "negative_friction_coefficient": negative_friction,
        "pile_side_resistance": selected_side_resistance,
        "pile_tip_resistance": selected_tip_resistance,
    }
    return _without_empty_values(compact)


def infer_foundation_type(layer_name: str) -> str:
    """根据岩土层名称判断该层采用的桩型。

    Args:
        layer_name: 岩土层名称，例如“粉质黏土”“强风化砂砾岩”“卵石”。

    Returns:
        名称包含“岩”或“石”时返回 ``cast_in_place``（灌注桩），其余返回
        ``precast``（预制桩）。
    """
    name = str(layer_name or "")
    return "cast_in_place" if "岩" in name or "石" in name else "precast"


def select_foundation_parameter(
    layer_name: str,
    *,
    cast_in_place: Any,
    precast: Any,
) -> Any:
    """按照土层名称公共规则选择灌注桩或预制桩参数。

    Args:
        layer_name: 岩土层名称。
        cast_in_place: 灌注桩对应参数值。
        precast: 预制桩对应参数值。

    Returns:
        当前土层按规则选中的参数值。
    """
    if infer_foundation_type(layer_name) == "cast_in_place":
        return cast_in_place
    return precast


def _compact_layer_boreholes(record: dict[str, Any]) -> list[dict[str, Any]]:
    """整理一个土层在不同钻孔中的观测数据。

    Args:
        record: 带逐孔 ``observations`` 的最终土层记录。

    Returns:
        按钻孔编号排序的观测列表，每个钻孔最多保留一条最完整记录。
    """
    observations: dict[str, dict[str, Any]] = {}
    for item in record.get("observations", []):
        borehole_id = item.get("borehole_id")
        if not borehole_id:
            continue
        identifier = str(borehole_id)
        # 结果文件只表达“哪个钻孔有什么数据”；页码、OCR 置信度和坐标属于
        # 追溯信息，完整保留在 details.json 的 observations 中。
        candidate = {
            "borehole_id": identifier,
            "thickness": item.get("image_thickness"),
            "bottom_depth": item.get("bottom_depth"),
            "bottom_elevation": item.get("bottom_elevation"),
        }
        current = observations.get(identifier)
        # 同一钻孔同一土层偶尔会被 OCR 重复识别：优先选择数值更完整的记录，
        # 完整程度相同再使用置信度打破平局，但置信度不写入业务结果。
        candidate_score = sum(value is not None for value in candidate.values())
        candidate_confidence = float(item.get("confidence") or 0)
        current_score = (
            sum(
                value is not None
                for key, value in current.items()
                if key != "_confidence"
            )
            if current
            else -1
        )
        current_confidence = float(current.get("_confidence") or 0) if current else -1
        candidate["_confidence"] = candidate_confidence
        if current is None or (candidate_score, candidate_confidence) > (
            current_score,
            current_confidence,
        ):
            observations[identifier] = candidate
    return [
        _without_empty_values(
            {
                key: value
                for key, value in observations[identifier].items()
                if key != "_confidence"
            }
        )
        for identifier in sorted(observations, key=_borehole_sort_key)
    ]


def _without_empty_values(values: dict[str, Any]) -> dict[str, Any]:
    """删除最终业务对象中的空字段，同时保留零值和布尔值。

    Args:
        values: 待清理的单层字典。

    Returns:
        不含 ``None``、空字符串、空列表和空字典的新字典。

    Notes:
        本函数只用于 ``result.json`` 的展示层。``details.json`` 不调用该函数，
        因而不会丢失候选字段、来源信息或空值诊断线索。
    """
    empty_values = (None, "", [], {})
    return {key: value for key, value in values.items() if value not in empty_values}
