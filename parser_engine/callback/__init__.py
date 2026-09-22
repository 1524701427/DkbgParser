"""逆向地质接口映射与发送。"""

from .client import DEFAULT_REVERSE_GEOLOGY_URL, post_reverse_geology_payload
from .mapping import (
    CALLBACK_FIELD_MAPPING,
    CALLBACK_LAYER_FIELD_MAPPING,
    build_reverse_geology_payload,
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
