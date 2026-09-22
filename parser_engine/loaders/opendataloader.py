from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Iterator, cast

from ..exceptions import BackendUnavailableError, ParserError, ScannedPdfNotSupportedError
from ..models import (
    BlockKind,
    BoundingBox,
    DocumentBlock,
    DocumentModel,
    PageInfo,
    ParagraphStyle,
    TableCell,
    TableData,
    TextSpan,
    TextStyle,
)
from .base import DocumentLoader


logger = logging.getLogger(__name__)


class OpenDataLoaderPdfLoader(DocumentLoader):
    """可选的 OpenDataLoader PDF 后端，将原生 JSON 归一化为统一模型。"""

    name = "opendataloader.pdf"
    extensions = frozenset({".pdf"})

    def __init__(
        self,
        *,
        reject_scanned: bool = True,
        scanned_page_ratio: float = 0.8,
        minimum_text_characters: int = 8,
    ) -> None:
        """初始化 OpenDataLoader PDF 加载器。

        Args:
            reject_scanned: 是否拒绝主要由图片构成的扫描版 PDF。
            scanned_page_ratio: 低文本页达到该比例时判定为扫描件。
            minimum_text_characters: 页面低于该字符数时视为低文本页。
        """
        self.reject_scanned = reject_scanned
        self.scanned_page_ratio = scanned_page_ratio
        self.minimum_text_characters = minimum_text_characters

    def load(self, path: Path) -> DocumentModel:
        """调用 OpenDataLoader 生成 JSON，并将结果载入内存。

        Args:
            path: 待解析 PDF 文件的绝对路径。

        Returns:
            由 OpenDataLoader 原生结果归一化得到的文档模型。

        Raises:
            BackendUnavailableError: 未安装 OpenDataLoader PDF 包。
            ParserError: 后端执行完成但没有生成 JSON 文件。
        """
        self._configure_java_encoding()
        try:
            import opendataloader_pdf
        except ImportError as exc:
            raise BackendUnavailableError(
                "OpenDataLoader backend is optional. Install it with: pip install opendataloader-pdf"
            ) from exc

        # OpenDataLoader 底层由 Java/PDFBox 执行。Windows 下转换方法返回后，后台文件句柄
        # 可能尚未完全释放，因此不能让临时文件清理异常覆盖已经成功读取的解析结果。
        output_dir = Path(tempfile.mkdtemp(prefix="dkbg_odl_"))
        try:
            try:
                result = opendataloader_pdf.convert(
                    input_path=[str(path)],
                    output_dir=str(output_dir),
                    format="json",
                    # OCR 会直接从原 PDF 渲染目标页，不需要导出全部原图。
                    # 关闭图片导出可减少临时文件，并规避 Windows 文件占用。
                    image_output="off",
                    quiet=True,
                )
            except Exception as exc:
                raise ParserError(f"OpenDataLoader 解析 PDF 失败: {path.name}: {exc}") from exc
            json_path = self._find_json(result, output_dir)
            if json_path is None:
                raise ParserError("OpenDataLoader completed but did not produce a JSON result")
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            document = self._normalize(path, payload)
            self._check_scanned_document(document)
        finally:
            # Python 3.10 的 TemporaryDirectory 即使配置 ignore_cleanup_errors，仍可能在
            # chmod 阶段抛出 PermissionError。直接捕获整个 rmtree 才能兼容被 Java 占用的文件。
            try:
                shutil.rmtree(output_dir)
            except OSError as exc:
                logger.warning("OpenDataLoader 临时目录暂时无法清理，将由系统后续回收：%s；%s", output_dir, exc)
        return document

    def _check_scanned_document(self, document: DocumentModel) -> None:
        """根据页面文字量统一执行扫描件判定。

        OpenDataLoader 关闭图片导出后无法可靠提供 ``image_count``，因此这里
        使用低文本页比例判断。该规则与 Aspose 后端共享相同阈值配置。

        Args:
            document: 已完成归一化的 PDF 文档模型。

        Raises:
            ScannedPdfNotSupportedError: 文档达到扫描件阈值且配置为拒绝。
        """
        if not document.pages:
            return
        candidates = [
            page.number
            for page in document.pages
            if page.text_characters < self.minimum_text_characters
        ]
        if len(candidates) / len(document.pages) < self.scanned_page_ratio:
            return
        message = (
            "PDF appears to be scanned/image-only; OCR is not enabled. "
            f"Candidate pages: {candidates}"
        )
        if self.reject_scanned:
            raise ScannedPdfNotSupportedError(message, pages=candidates)
        document.warnings.append(message)

    @staticmethod
    def _configure_java_encoding() -> None:
        """让 OpenDataLoader 的 Java 日志统一使用 UTF-8。

        Windows 中文环境中的 Java 默认可能按 GBK 输出，而调用方按 UTF-8
        接收，最终显示成乱码。该设置必须在首次导入并启动 Java 虚拟机前完成。
        """
        encoding_options = (
            "-Dfile.encoding=UTF-8 "
            "-Dsun.stdout.encoding=UTF-8 "
            "-Dsun.stderr.encoding=UTF-8"
        )
        current = os.environ.get("JAVA_TOOL_OPTIONS", "").strip()
        if "-Dfile.encoding=" not in current:
            os.environ["JAVA_TOOL_OPTIONS"] = f"{current} {encoding_options}".strip()

    @staticmethod
    def _find_json(result, output_dir: Path) -> Path | None:
        """从库返回值或临时输出目录中定位生成的 JSON 文件。

        Args:
            result: OpenDataLoader 的原始返回值。
            output_dir: 本次转换使用的临时输出目录。

        Returns:
            找到的 JSON 文件路径；没有结果时返回 ``None``。
        """
        if isinstance(result, (str, Path)) and Path(result).suffix.lower() == ".json":
            candidate = Path(result)
            if candidate.is_file():
                return candidate
        return next(output_dir.rglob("*.json"), None)

    def _normalize(self, path: Path, payload) -> DocumentModel:
        """将常见 OpenDataLoader 元素字段投影到稳定文档模型。

        Args:
            path: 原始 PDF 文件路径。
            payload: OpenDataLoader 输出的原生 JSON 数据。

        Returns:
            可供下游统一消费的文档模型。
        """
        document_metadata = {
            key: value
            for key, value in payload.items()
            if key != "kids"
        } if isinstance(payload, dict) else {"native_result": payload}
        model = DocumentModel(str(path), "pdf", self.name, metadata=document_metadata)
        # 顶层文档属性保存在 metadata 中，同时把内容元素映射到稳定模型，
        # 方便下游统一消费不同版本、不同解析后端的结果。
        elements: list[dict] = []
        if isinstance(payload, dict):
            # 当前版本使用 kids；同时兼容早期或其他版本可能使用的 elements/content。
            elements = payload.get("kids") or payload.get("elements") or payload.get("content") or []
        if isinstance(elements, list):
            for index, element in enumerate(self._iter_elements(elements), 1):
                model.blocks.append(self._element_to_block(element, index))

        declared_pages = payload.get("number of pages", 0) if isinstance(payload, dict) else 0
        detected_pages = max((block.page or 0 for block in model.blocks), default=0)
        page_count = max(int(declared_pages or 0), detected_pages)
        for page_number in range(1, page_count + 1):
            page_blocks = [block for block in model.blocks if block.page == page_number]
            model.pages.append(
                PageInfo(
                    number=page_number,
                    text_characters=sum(len(block.text) for block in page_blocks),
                    image_count=sum(block.kind == "image" for block in page_blocks),
                )
            )
        return model

    @classmethod
    def _iter_elements(cls, elements: list) -> Iterator[dict]:
        """按阅读顺序展开 OpenDataLoader 的列表容器。

        Args:
            elements: OpenDataLoader 顶层或嵌套元素列表。

        Yields:
            可直接转换为统一内容块的元素；列表容器本身不会产生重复文本块。
        """
        for element in elements:
            if not isinstance(element, dict):
                continue
            element_type = str(element.get("type", "")).lower()
            if element_type == "list":
                yield from cls._iter_elements(element.get("list items", []))
                continue
            # text block 是版面分组容器，真实内容位于 kids 中；输出容器会产生空块，
            # 还会遗漏其中的跨页表格，因此与 list 一样只展开其子元素。
            if element_type == "text block" and element.get("kids"):
                yield from cls._iter_elements(element["kids"])
                continue
            yield element
            if element_type in {"list item", "list_item"}:
                yield from cls._iter_elements(element.get("kids", []))

    def _element_to_block(self, element: dict, index: int) -> DocumentBlock:
        """把单个 OpenDataLoader 元素转换为统一内容块。

        Args:
            element: OpenDataLoader 元素字典。
            index: 元素在展开后阅读顺序中的序号。

        Returns:
            转换后的标题、段落、列表项、表格或图片块。
        """
        kind = self._normalize_kind(element.get("type"))
        text = str(element.get("text") or element.get("content") or "")
        page_value = element.get("page") or element.get("page_number") or element.get("page number")
        bbox = self._bounding_box(
            element.get("bbox") or element.get("bounding_box") or element.get("bounding box")
        )
        style = self._text_style(element)
        spans = [TextSpan(text=text, style=style, bbox=bbox)] if text else []
        heading_level = element.get("heading level") if kind == "heading" else None
        metadata = {
            key: value
            for key, value in element.items()
            if key not in {"kids", "list items", "rows"}
        }
        table = self._table_data(element, index) if kind == "table" else None
        if table is not None:
            text = "\n".join(
                "\t".join(cell.text for cell in table.cells if cell.row == row)
                for row in range(table.rows)
            )
        return DocumentBlock(
            id=f"b{index}",
            kind=kind,
            text=text,
            spans=spans,
            paragraph_style=ParagraphStyle(outline_level=int(heading_level)) if heading_level else None,
            table=table,
            bbox=bbox,
            page=int(page_value) if page_value else None,
            metadata=metadata,
        )

    def _table_data(self, element: dict, block_index: int) -> TableData:
        """把 OpenDataLoader 表格行列转换为统一表格结构。

        Args:
            element: 类型为 ``table`` 的原生元素。
            block_index: 表格在文档内容块中的序号。

        Returns:
            包含单元格文本、合并信息和内部段落的表格数据。
        """
        cells: list[TableCell] = []
        rows = element.get("rows", [])
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            for column_index, cell in enumerate(row.get("cells", [])):
                if not isinstance(cell, dict):
                    continue
                paragraphs = [
                    self._element_to_block(child, block_index * 100000 + row_index * 1000 + column_index * 100 + child_index)
                    for child_index, child in enumerate(cell.get("kids", []), 1)
                    if isinstance(child, dict)
                ]
                text = "\n".join(paragraph.text for paragraph in paragraphs if paragraph.text)
                cells.append(
                    TableCell(
                        row=int(cell.get("row number", row_index + 1)) - 1,
                        column=int(cell.get("column number", column_index + 1)) - 1,
                        text=text,
                        paragraphs=paragraphs,
                        row_span=int(cell.get("row span", 1)),
                        column_span=int(cell.get("column span", 1)),
                    )
                )
        row_count = int(element.get("number of rows", len(rows)))
        column_count = int(element.get("number of columns", 0))
        if not column_count:
            column_count = max((cell.column + cell.column_span for cell in cells), default=0)
        return TableData(rows=row_count, columns=column_count, cells=cells)

    @staticmethod
    def _normalize_kind(value) -> BlockKind:
        """将 OpenDataLoader 元素类型映射为统一内容块类型。

        Args:
            value: 原生元素类型。

        Returns:
            统一模型支持的内容块类型。
        """
        normalized = str(value or "paragraph").lower().replace(" ", "_")
        if normalized == "list":
            return "list_item"
        if normalized in {"heading", "paragraph", "list_item", "table", "image"}:
            return cast(BlockKind, normalized)
        return "paragraph"

    @staticmethod
    def _bounding_box(values) -> BoundingBox | None:
        """将四元坐标转换为统一边界框。

        Args:
            values: ``[x1, y1, x2, y2]`` 格式的坐标列表。

        Returns:
            转换后的边界框；坐标不合法时返回 ``None``。
        """
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            return None
        x1, y1, x2, y2 = map(float, values)
        return BoundingBox(left=x1, top=y1, width=x2 - x1, height=y2 - y1)

    @staticmethod
    def _text_style(element: dict) -> TextStyle:
        """从 OpenDataLoader 元素提取字符样式。

        Args:
            element: 包含字体、字号和颜色信息的原生元素。

        Returns:
            统一的字符级样式。
        """
        font = str(element.get("font") or "") or None
        font_lower = font.lower() if font else ""
        color = element.get("text color")
        foreground = None
        if color is not None:
            components = re.findall(r"\d+(?:\.\d+)?", str(color))
            if len(components) >= 3:
                rgb = [round(float(component) * 255) for component in components[:3]]
                foreground = "#" + "".join(f"{max(0, min(255, value)):02X}" for value in rgb)
        font_size = element.get("font size")
        return TextStyle(
            font_family=font,
            font_size=float(font_size) if font_size is not None else None,
            bold="bold" in font_lower,
            italic="italic" in font_lower or "oblique" in font_lower,
            foreground=foreground,
        )
