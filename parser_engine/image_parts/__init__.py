"""图片识别内部职责模块。"""

from .aggregation import BoreholeAggregationMixin
from .vision import OpenAICompatibleVisionClient, VisionClient

__all__ = [
    "BoreholeAggregationMixin",
    "OpenAICompatibleVisionClient",
    "VisionClient",
]
