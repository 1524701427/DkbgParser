from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..models import DocumentModel


class DocumentLoader(ABC):
    """解析后端的统一协议，新增后端只需实现该接口。"""

    name: str
    extensions: frozenset[str]

    def supports(self, path: Path) -> bool:
        """判断当前加载器是否支持给定文件的扩展名。

        Args:
            path: 待判断的文件路径。

        Returns:
            扩展名受支持时返回 ``True``，否则返回 ``False``。
        """
        return path.suffix.lower() in self.extensions

    @abstractmethod
    def load(self, path: Path) -> DocumentModel:
        """解析文件并返回统一文档模型。

        Args:
            path: 待解析文件的绝对路径。

        Returns:
            解析完成的统一文档模型。
        """
        raise NotImplementedError
