from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .exceptions import UnsupportedFormatError
from .loaders.base import DocumentLoader
from .models import DocumentModel


@dataclass(slots=True)
class ParserConfig:
    """文档解析器运行配置。

    Attributes:
        pdf_backend: PDF 解析后端名称，支持 ``aspose`` 和 ``opendataloader``。
        reject_scanned_pdf: 是否拒绝扫描型 PDF。
        scanned_page_ratio: 判断扫描型 PDF 时允许的扫描页比例阈值。
        minimum_text_characters: 判断页面是否包含有效文本时的最小字符数。
    """

    pdf_backend: str = "aspose"
    reject_scanned_pdf: bool = True
    scanned_page_ratio: float = 0.8
    minimum_text_characters: int = 8


class DocumentParser:
    """统一文档解析入口。

    根据文件扩展名自动选择对应的 ``DocumentLoader``，
    将 Word、PDF 等不同格式解析为统一文档模型。

    PDF 后端由 ``ParserConfig.pdf_backend`` 决定。
    """

    def __init__(self, config: ParserConfig | None = None) -> None:
        """初始化文档解析器并注册默认加载器。

        Args:
            config: 文档解析配置。为 ``None`` 时使用默认配置。

        Raises:
            ValueError: ``pdf_backend`` 不是支持的 PDF 后端。
        """
        # 延迟导入具体解析器，避免模块加载时初始化不必要的第三方依赖。
        from .loaders.pdf import AsposePdfLoader
        from .loaders.word import AsposeWordsLoader

        self.config = config or ParserConfig()

        # 文件扩展名到 Loader 的映射。
        self._loaders: dict[str, DocumentLoader] = {}

        # Word 固定使用 Aspose 解析。
        self.register(AsposeWordsLoader())

        # 两种 PDF Loader 使用相同配置参数。
        pdf_loader_kwargs = {
            "reject_scanned": self.config.reject_scanned_pdf,
            "scanned_page_ratio": self.config.scanned_page_ratio,
            "minimum_text_characters": self.config.minimum_text_characters,
        }

        if self.config.pdf_backend == "aspose":
            pdf_loader = AsposePdfLoader(**pdf_loader_kwargs)

        elif self.config.pdf_backend == "opendataloader":
            # 只有真正使用 OpenDataLoader 时才导入对应依赖。
            from .loaders.opendataloader import OpenDataLoaderPdfLoader

            pdf_loader = OpenDataLoaderPdfLoader(**pdf_loader_kwargs)

        else:
            raise ValueError(
                f"Unknown PDF backend: {self.config.pdf_backend}"
            )

        self.register(pdf_loader)

    def register(self, loader: DocumentLoader) -> None:
        """注册文档加载器。

        Loader 声明的所有扩展名都会统一转换为小写后注册。

        如果同一个扩展名被重复注册，后注册的 Loader 会覆盖之前的映射，
        与原有行为保持一致。

        Args:
            loader: 实现 ``DocumentLoader`` 协议的加载器。
        """
        for extension in loader.extensions:
            self._loaders[extension.lower()] = loader

    def parse(self, source: str | Path) -> DocumentModel:
        """解析单个本地文档。

        Args:
            source: 待解析文件路径。

        Returns:
            与具体解析后端无关的统一文档模型。

        Raises:
            FileNotFoundError: 输入路径不存在或不是普通文件。
            UnsupportedFormatError: 当前没有 Loader 支持该文件格式。
            ParserError: 底层解析器初始化或文档解析失败。
        """
        # 支持 ~/xxx，并统一转换为绝对路径。
        path = Path(source).expanduser().resolve()

        # 输入必须存在且是普通文件。
        if not path.is_file():
            raise FileNotFoundError(path)

        # 扩展名统一转小写，兼容 .PDF、.DOCX 等情况。
        extension = path.suffix.lower()
        loader = self._loaders.get(extension)

        if loader is None:
            # 根据当前已注册 Loader 动态生成支持格式列表。
            supported = ", ".join(sorted(self._loaders))
            raise UnsupportedFormatError(
                f"Unsupported file type '{path.suffix}'. "
                f"Supported: {supported}"
            )

        # DocumentParser 只负责路由，具体解析逻辑由 Loader 完成。
        return loader.load(path)