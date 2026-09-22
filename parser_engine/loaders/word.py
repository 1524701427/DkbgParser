from __future__ import annotations

import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from ..aspose_runtime import load_aspose
from ..models import (
    BlockKind,
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


def _enum(value) -> str | None:
    """将 .NET 枚举安全转换为便于序列化的小写字符串。

    Args:
        value: .NET 枚举值或 ``None``。

    Returns:
        枚举成员的小写名称；输入为空时返回 ``None``。
    """
    if value is None:
        return None
    return str(value).split(".")[-1].lower()


def _rgb(color) -> str | None:
    """将 ``System.Drawing.Color`` 转换为十六进制 RGB 字符串。

    Args:
        color: Aspose 返回的 .NET 颜色对象。

    Returns:
        ``#RRGGBB`` 格式的颜色；颜色为空或读取失败时返回 ``None``。
    """
    try:
        if color is None or color.IsEmpty:
            return None
        return f"#{int(color.R):02X}{int(color.G):02X}{int(color.B):02X}"
    except Exception:
        return None


class AsposeWordsLoader(DocumentLoader):
    """使用 Aspose.Words 解析 Word 及兼容的富文本文档。"""
    name = "aspose.words"
    extensions = frozenset({".doc", ".docx", ".docm", ".dot", ".dotx", ".rtf", ".odt"})

    def load(self, path: Path) -> DocumentModel:
        """解析 Word 文档，按正文顺序提取段落、标题、列表和表格。

        Args:
            path: 待解析 Word 文档的绝对路径。

        Returns:
            包含正文结构、字符样式和页面信息的统一文档模型。

        Raises:
            BackendUnavailableError: Aspose.Words 或其运行时依赖无法加载。
        """
        aw = load_aspose("words")
        # Word/WPS 打开文档时，Aspose 可能因独占读取方式而报文件占用。
        # 先复制一份只读快照，既兼容已打开的文件，也避免解析期间源文件变化。
        with tempfile.TemporaryDirectory(prefix="dkbg_word_") as temp_dir:
            snapshot_path = Path(temp_dir) / path.name
            shutil.copy2(path, snapshot_path)
            document = aw.Document(str(snapshot_path))
        collector = aw.Layout.LayoutCollector(document)
        model = DocumentModel(
            source_path=str(path),
            source_format=path.suffix.lower().lstrip("."),
            parser_backend=self.name,
            metadata=self._document_metadata(document),
        )

        block_number = 0
        # 按节遍历正文的直接子节点，避免把表格内段落重复解析成顶层块。
        for section_index in range(document.Sections.Count):
            section = document.Sections[section_index]
            body = section.Body
            child_nodes = body.GetChildNodes(aw.NodeType.Any, False)
            for node_index in range(child_nodes.Count):
                node = child_nodes[node_index]
                if node.NodeType == aw.NodeType.Paragraph:
                    block_number += 1
                    block = self._paragraph(aw, node, collector, block_number, section_index + 1)
                    if block.text or block.metadata:
                        model.blocks.append(block)
                elif node.NodeType == aw.NodeType.Table:
                    block_number += 1
                    model.blocks.append(self._table(aw, node, collector, block_number, section_index + 1))

        # 记录内嵌图片所在页，供钻孔柱状图 OCR 判断是否需要补漏。
        image_counts: dict[int, int] = defaultdict(int)
        shapes = document.GetChildNodes(aw.NodeType.Shape, True)
        for shape_index in range(shapes.Count):
            shape = shapes[shape_index]
            try:
                if bool(shape.HasImage):
                    image_counts[int(collector.GetStartPageIndex(shape))] += 1
            except Exception:
                # 个别矢量对象不暴露 HasImage，忽略即可，不影响正文解析。
                continue

        for page_number in range(1, document.PageCount + 1):
            setup = document.Sections[0].PageSetup if document.Sections.Count else None
            model.pages.append(
                PageInfo(
                    number=page_number,
                    width=float(setup.PageWidth) if setup else None,
                    height=float(setup.PageHeight) if setup else None,
                    text_characters=sum(len(b.text) for b in model.blocks if b.page == page_number),
                    image_count=image_counts.get(page_number, 0),
                )
            )
        return model

    def _paragraph(self, aw, paragraph, collector, number: int, section: int) -> DocumentBlock:
        """把 Aspose 段落转换为统一内容块，并保留字符级样式。

        Args:
            aw: ``Aspose.Words`` CLR 模块。
            paragraph: Aspose 段落节点。
            collector: 用于查询节点页码的版式收集器。
            number: 当前顶层内容块序号。
            section: 当前节编号。

        Returns:
            转换后的段落、标题或列表内容块。
        """
        spans: list[TextSpan] = []
        for run_index in range(paragraph.Runs.Count):
            run = paragraph.Runs[run_index]
            text = str(run.Text).replace("\r", "").replace("\x07", "")
            if not text:
                continue
            font = run.Font
            underline = _enum(font.Underline)
            spans.append(
                TextSpan(
                    text=text,
                    style=TextStyle(
                        font_family=str(font.Name) if font.Name else None,
                        font_size=float(font.Size) if float(font.Size) > 0 else None,
                        bold=bool(font.Bold),
                        italic=bool(font.Italic),
                        underline=underline not in (None, "none", "0"),
                        strike=bool(font.StrikeThrough),
                        foreground=_rgb(font.Color),
                        background=_rgb(font.HighlightColor),
                    ),
                )
            )
        text = "".join(span.text for span in spans).strip()
        fmt = paragraph.ParagraphFormat
        style_name = str(fmt.StyleName) if fmt.StyleName else None
        outline = _enum(fmt.OutlineLevel)
        # Word 标题优先依据命名样式判断，必要时再参考大纲级别。
        heading_level = self._heading_level(style_name, outline)
        is_list = bool(paragraph.IsListItem)
        list_level = int(paragraph.ListFormat.ListLevelNumber) if is_list else None
        try:
            list_label = str(paragraph.ListLabel.LabelString) if is_list else None
        except Exception:
            list_label = None
        kind: BlockKind = "heading" if heading_level else ("list_item" if is_list else "paragraph")
        page = int(collector.GetStartPageIndex(paragraph)) or None
        return DocumentBlock(
            id=f"b{number}",
            kind=kind,
            text=text,
            spans=spans,
            page=page,
            section=section,
            paragraph_style=ParagraphStyle(
                name=style_name,
                alignment=_enum(fmt.Alignment),
                left_indent=float(fmt.LeftIndent),
                right_indent=float(fmt.RightIndent),
                first_line_indent=float(fmt.FirstLineIndent),
                space_before=float(fmt.SpaceBefore),
                space_after=float(fmt.SpaceAfter),
                line_spacing=float(fmt.LineSpacing),
                outline_level=heading_level,
                list_level=list_level,
                list_label=list_label,
            ),
            metadata={"heading_level": heading_level} if heading_level else {},
        )

    def _table(self, aw, table, collector, number: int, section: int) -> DocumentBlock:
        """解析表格及每个单元格内的段落内容。

        Args:
            aw: ``Aspose.Words`` CLR 模块。
            table: Aspose 表格节点。
            collector: 用于查询节点页码的版式收集器。
            number: 当前顶层内容块序号。
            section: 当前节编号。

        Returns:
            包含行列、单元格和内部段落的表格内容块。
        """
        cells: list[TableCell] = []
        max_columns = 0
        for row_index in range(table.Rows.Count):
            row = table.Rows[row_index]
            max_columns = max(max_columns, row.Cells.Count)
            for column_index in range(row.Cells.Count):
                cell = row.Cells[column_index]
                paragraphs = []
                for paragraph_index in range(cell.Paragraphs.Count):
                    paragraphs.append(
                        self._paragraph(
                            aw,
                            cell.Paragraphs[paragraph_index],
                            collector,
                            number * 100000 + row_index * 1000 + column_index * 100 + paragraph_index,
                            section,
                        )
                    )
                text = "\n".join(p.text for p in paragraphs if p.text)
                cells.append(TableCell(row=row_index, column=column_index, text=text, paragraphs=paragraphs))
        return DocumentBlock(
            id=f"b{number}",
            kind="table",
            text="\n".join(cell.text for cell in cells if cell.text),
            table=TableData(rows=table.Rows.Count, columns=max_columns, cells=cells),
            page=int(collector.GetStartPageIndex(table)) or None,
            section=section,
        )

    @staticmethod
    def _heading_level(style_name: str | None, outline: str | None) -> int | None:
        """根据中英文标题样式名或大纲级别推断标题层级。

        Args:
            style_name: Word 段落样式名称。
            outline: 已转换为字符串的大纲级别。

        Returns:
            1～9 的标题层级；无法识别时返回 ``None``。
        """
        if style_name:
            match = re.search(r"(?:heading|标题)\s*([1-9])", style_name, re.IGNORECASE)
            if match:
                return int(match.group(1))
        if outline:
            match = re.search(r"level[_ ]?([1-9])", outline, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    @staticmethod
    def _document_metadata(document) -> dict:
        """提取 Word 内置属性中适合下游使用的文档元数据。

        Args:
            document: Aspose Word 文档对象。

        Returns:
            包含标题、主题、作者、关键词和备注等非空属性的字典。
        """
        props = document.BuiltInDocumentProperties
        result = {}
        for key, attribute in (
            ("title", "Title"),
            ("subject", "Subject"),
            ("author", "Author"),
            ("keywords", "Keywords"),
            ("comments", "Comments"),
        ):
            value = getattr(props, attribute, None)
            if value:
                result[key] = str(value)
        return result
