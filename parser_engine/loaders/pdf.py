from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ..aspose_runtime import load_aspose
from ..exceptions import ScannedPdfNotSupportedError
from ..models import BoundingBox, DocumentBlock, DocumentModel, PageInfo, ParagraphStyle, TextSpan, TextStyle
from .base import DocumentLoader


def _safe(getter, default=None):
    """读取可能因 PDF 缺少属性而失败的 Aspose 值。

    Args:
        getter: 无参数的延迟读取函数。
        default: 读取失败时使用的默认值。

    Returns:
        成功读取的属性值，或调用方提供的默认值。
    """
    try:
        return getter()
    except Exception:
        return default


def _pdf_color(color) -> str | None:
    """将 Aspose.PDF 颜色分量转换为十六进制 RGB 字符串。

    Args:
        color: Aspose.PDF 颜色对象或 ``None``。

    Returns:
        ``#RRGGBB`` 格式的颜色；无法读取时返回 ``None``。
    """
    if color is None:
        return None
    # Aspose.PDF 的颜色分量通常是 0～1 的浮点数，也兼容已经是 0～255 的值。
    components = _safe(lambda: (float(color.R), float(color.G), float(color.B)))
    if components is None:
        return None
    values = [round(value * 255 if value <= 1 else value) for value in components]
    return "#" + "".join(f"{max(0, min(255, value)):02X}" for value in values)


class AsposePdfLoader(DocumentLoader):
    """使用 Aspose.PDF 解析包含文本层的 PDF，并保留页面坐标。"""
    name = "aspose.pdf"
    extensions = frozenset({".pdf"})

    def __init__(self, *, reject_scanned: bool = True, scanned_page_ratio: float = 0.8, minimum_text_characters: int = 8) -> None:
        """初始化 PDF 加载器。

        Args:
            reject_scanned: 是否拒绝疑似扫描版 PDF。
            scanned_page_ratio: 判定整份文档为扫描件所需的疑似扫描页比例。
            minimum_text_characters: 页面不被视为低文本页所需的最少字符数。
        """
        self.reject_scanned = reject_scanned
        self.scanned_page_ratio = scanned_page_ratio
        self.minimum_text_characters = minimum_text_characters

    def load(self, path: Path) -> DocumentModel:
        """逐页提取文本片段，并转换为统一文档模型。

        Args:
            path: 待解析 PDF 文件的绝对路径。

        Returns:
            包含文本样式、页面位置和页面统计的统一文档模型。

        Raises:
            BackendUnavailableError: Aspose.PDF 或其运行时依赖无法加载。
            ScannedPdfNotSupportedError: 文档达到扫描件判定阈值且配置为拒绝。
        """
        ap = load_aspose("pdf")
        document = ap.Document(str(path))
        model = DocumentModel(
            source_path=str(path),
            source_format="pdf",
            parser_backend=self.name,
            metadata={"page_count": int(document.Pages.Count)},
        )
        scanned_candidates: list[int] = []
        block_number = 0

        for page_number in range(1, document.Pages.Count + 1):
            page = document.Pages[page_number]
            absorber = ap.Text.TextFragmentAbsorber()
            page.Accept(absorber)
            fragments = [absorber.TextFragments[index] for index in range(1, absorber.TextFragments.Count + 1)]
            character_count = sum(len(str(fragment.Text).strip()) for fragment in fragments)
            image_count = int(_safe(lambda: page.Resources.Images.Count, 0))
            rect = page.Rect
            page_width, page_height = float(rect.Width), float(rect.Height)
            model.pages.append(
                PageInfo(
                    number=page_number,
                    width=page_width,
                    height=page_height,
                    text_characters=character_count,
                    image_count=image_count,
                )
            )
            # 同时满足“几乎无文字”和“页面含图片”才视为扫描页，避免误判空白页。
            if character_count < self.minimum_text_characters and image_count > 0:
                scanned_candidates.append(page_number)

            for line in self._group_lines(fragments):
                block_number += 1
                model.blocks.append(self._line_block(line, block_number, page_number, page_width, page_height))

        if document.Pages.Count and len(scanned_candidates) / document.Pages.Count >= self.scanned_page_ratio:
            message = f"PDF appears to be scanned/image-only; OCR is not enabled. Candidate pages: {scanned_candidates}"
            if self.reject_scanned:
                raise ScannedPdfNotSupportedError(message, pages=scanned_candidates)
            model.warnings.append(message)
        return model

    @staticmethod
    def _group_lines(fragments) -> list[list]:
        """按近似基线聚合 PDF 文本片段。

        Args:
            fragments: 当前页面的 Aspose 文本片段集合。

        Returns:
            按从上到下排序的视觉行，每行内部按从左到右排序。
        """
        rows: dict[float, list] = defaultdict(list)
        for fragment in fragments:
            y = float(_safe(lambda: fragment.Position.YIndent, 0.0))
            # 允许约 2pt 的纵向误差，解决同一视觉行中文字基线略有偏移的问题。
            key = round(y / 2.0) * 2.0
            rows[key].append(fragment)
        lines = []
        for y in sorted(rows, reverse=True):
            lines.append(sorted(rows[y], key=lambda item: float(_safe(lambda: item.Position.XIndent, 0.0))))
        return lines

    def _line_block(self, fragments, number: int, page: int, page_width: float, page_height: float) -> DocumentBlock:
        """将同一视觉行的片段合并为段落块。

        Args:
            fragments: 已按横坐标排序的同一行文本片段。
            number: 当前内容块序号。
            page: 当前页码。
            page_width: 页面宽度，单位为点。
            page_height: 页面高度，单位为点。

        Returns:
            保留文本样式和片段坐标的段落内容块。
        """
        spans: list[TextSpan] = []
        lefts, bottoms, rights, tops = [], [], [], []
        for fragment in fragments:
            text = str(fragment.Text)
            rect = fragment.Rectangle
            left, bottom, right, top = float(rect.LLX), float(rect.LLY), float(rect.URX), float(rect.URY)
            lefts.append(left)
            bottoms.append(bottom)
            rights.append(right)
            tops.append(top)
            state = fragment.TextState
            font_name = _safe(lambda: str(state.Font.FontName))
            style_text = str(_safe(lambda: state.FontStyle, "")).lower()
            spans.append(
                TextSpan(
                    text=text,
                    style=TextStyle(
                        font_family=font_name,
                        font_size=float(_safe(lambda: state.FontSize, 0.0)) or None,
                        bold="bold" in style_text,
                        italic="italic" in style_text,
                        foreground=_pdf_color(_safe(lambda: state.ForegroundColor)),
                        background=_pdf_color(_safe(lambda: state.BackgroundColor)),
                    ),
                    bbox=BoundingBox(
                        left=left,
                        top=page_height - top,
                        width=right - left,
                        height=top - bottom,
                        page_width=page_width,
                        page_height=page_height,
                    ),
                )
            )
        joined_parts: list[str] = []
        previous_right: float | None = None
        for span in spans:
            # PDF 文本绘制指令不一定包含空格，根据片段间距补回单词或单元格间隔。
            if previous_right is not None and span.bbox is not None:
                gap = span.bbox.left - previous_right
                threshold = (span.style.font_size or 10.0) * 0.35
                if gap > threshold and joined_parts and not joined_parts[-1].endswith((" ", "\t")):
                    joined_parts.append(" ")
            joined_parts.append(span.text)
            if span.bbox is not None:
                previous_right = span.bbox.left + span.bbox.width
        joined = "".join(joined_parts).strip()
        left, bottom, right, top = min(lefts), min(bottoms), max(rights), max(tops)
        return DocumentBlock(
            id=f"b{number}",
            kind="paragraph",
            text=joined,
            spans=spans,
            paragraph_style=ParagraphStyle(),
            bbox=BoundingBox(
                left=left,
                top=page_height - top,
                width=right - left,
                height=top - bottom,
                page_width=page_width,
                page_height=page_height,
            ),
            page=page,
        )
