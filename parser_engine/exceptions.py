class ParserError(RuntimeError):
    """所有文档解析异常的基类。"""


class UnsupportedFormatError(ParserError):
    """输入文件格式没有对应的已注册加载器。"""


class ScannedPdfNotSupportedError(ParserError):
    """PDF 主要由图片构成，而当前版本尚未启用 OCR。"""

    def __init__(self, message: str, *, pages: list[int] | None = None) -> None:
        """初始化扫描版 PDF 异常。

        Args:
            message: 面向调用方的错误说明。
            pages: 疑似扫描页的页码列表。
        """
        super().__init__(message)
        self.pages = pages or []


class BackendUnavailableError(ParserError):
    """配置的解析后端或其运行时依赖无法加载。"""
