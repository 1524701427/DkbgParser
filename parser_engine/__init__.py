"""可扩展的文档解析与配置化抽取公共入口。"""

from .engine import DocumentParser, ParserConfig
from .extraction import (
    ExtractionEngine,
    extract_document,
    infer_foundation_type,
    merge_first_cultivated_soil_layer,
    select_foundation_parameter,
)
from .callback_mapping import (
    CALLBACK_FIELD_MAPPING,
    CALLBACK_LAYER_FIELD_MAPPING,
    DEFAULT_REVERSE_GEOLOGY_URL,
    build_reverse_geology_payload,
    post_reverse_geology_payload,
    write_reverse_geology_payload,
)
from .image_recognition import (
    BoreholeImageRecognizer,
    OpenAICompatibleVisionClient,
    RapidOCRClient,
    VisionClient,
)
from .exceptions import (
    ParserError,
    ScannedPdfNotSupportedError,
    UnsupportedFormatError,
)
from .models import DocumentModel

__all__ = [
    "DocumentModel",
    "DocumentParser",
    "ExtractionEngine",
    "BoreholeImageRecognizer",
    "CALLBACK_FIELD_MAPPING",
    "CALLBACK_LAYER_FIELD_MAPPING",
    "DEFAULT_REVERSE_GEOLOGY_URL",
    "build_reverse_geology_payload",
    "post_reverse_geology_payload",
    "OpenAICompatibleVisionClient",
    "RapidOCRClient",
    "ParserConfig",
    "ParserError",
    "ScannedPdfNotSupportedError",
    "UnsupportedFormatError",
    "VisionClient",
    "extract_document",
    "infer_foundation_type",
    "merge_first_cultivated_soil_layer",
    "select_foundation_parameter",
    "write_reverse_geology_payload",
]
