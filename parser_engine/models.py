from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


BlockKind = Literal["heading", "paragraph", "list_item", "table", "image", "page_break"]


@dataclass(slots=True)
class BoundingBox:
    """元素在页面中的边界框，坐标和尺寸统一使用点（point）。"""
    left: float
    top: float
    width: float
    height: float
    page_width: float | None = None
    page_height: float | None = None


@dataclass(slots=True)
class TextStyle:
    """文本片段的字符级样式。"""
    font_family: str | None = None
    font_size: float | None = None
    bold: bool | None = None
    italic: bool | None = None
    underline: bool | None = None
    strike: bool | None = None
    foreground: str | None = None
    background: str | None = None


@dataclass(slots=True)
class TextSpan:
    """具有相同样式的一段连续文本。"""
    text: str
    style: TextStyle = field(default_factory=TextStyle)
    bbox: BoundingBox | None = None


@dataclass(slots=True)
class ParagraphStyle:
    """段落级版式信息，包括对齐、缩进、间距和列表层级。"""
    name: str | None = None
    alignment: str | None = None
    left_indent: float | None = None
    right_indent: float | None = None
    first_line_indent: float | None = None
    space_before: float | None = None
    space_after: float | None = None
    line_spacing: float | None = None
    outline_level: int | None = None
    list_level: int | None = None
    list_label: str | None = None


@dataclass(slots=True)
class TableCell:
    """表格单元格及其内部段落。"""
    row: int
    column: int
    text: str
    paragraphs: list["DocumentBlock"] = field(default_factory=list)
    row_span: int = 1
    column_span: int = 1


@dataclass(slots=True)
class TableData:
    """表格的行列规模与单元格集合。"""
    rows: int
    columns: int
    cells: list[TableCell] = field(default_factory=list)


@dataclass(slots=True)
class DocumentBlock:
    """统一文档中的内容块，例如标题、段落、列表项或表格。"""
    id: str
    kind: BlockKind
    text: str = ""
    spans: list[TextSpan] = field(default_factory=list)
    paragraph_style: ParagraphStyle | None = None
    table: TableData | None = None
    bbox: BoundingBox | None = None
    page: int | None = None
    section: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PageInfo:
    """页面尺寸以及用于扫描件判断的基础统计信息。"""
    number: int
    width: float | None = None
    height: float | None = None
    text_characters: int = 0
    image_count: int = 0


@dataclass(slots=True)
class DocumentModel:
    """解析后的稳定中间模型，供结构恢复和语义抽取继续处理。"""
    source_path: str
    source_format: str
    parser_backend: str
    blocks: list[DocumentBlock] = field(default_factory=list)
    pages: list[PageInfo] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    schema_version: str = "1.0"

    @property
    def text(self) -> str:
        """按内容块顺序拼接文档的纯文本。

        Returns:
            使用换行符连接的非空内容块文本。
        """
        return "\n".join(block.text for block in self.blocks if block.text)

    def to_dict(self) -> dict[str, Any]:
        """将完整文档模型递归转换为字典。

        Returns:
            可直接进行 JSON 序列化的文档字典。
        """
        return asdict(self)

    def write_json(self, output_path: str | Path, *, indent: int = 2) -> None:
        """将文档模型以 UTF-8 JSON 写入指定路径。

        Args:
            output_path: JSON 输出文件路径。
            indent: JSON 缩进空格数。
        """
        import json

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=indent),
            encoding="utf-8",
        )
