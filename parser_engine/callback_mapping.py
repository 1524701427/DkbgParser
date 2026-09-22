"""兼容旧导入路径。

新代码请优先从 :mod:`parser_engine.callback` 导入。该模块保留原有公开 API，
避免已有调用方因结构重构而修改业务代码。
"""

from .callback import (
    CALLBACK_FIELD_MAPPING,
    CALLBACK_LAYER_FIELD_MAPPING,
    DEFAULT_REVERSE_GEOLOGY_URL,
    build_reverse_geology_payload,
    post_reverse_geology_payload,
    write_reverse_geology_payload,
)

__all__ = [
    "CALLBACK_FIELD_MAPPING",
    "CALLBACK_LAYER_FIELD_MAPPING",
    "DEFAULT_REVERSE_GEOLOGY_URL",
    "build_reverse_geology_payload",
    "post_reverse_geology_payload",
    "write_reverse_geology_payload",
]
