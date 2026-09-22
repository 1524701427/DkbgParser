"""ExtractionEngine 的内部职责拆分。

这些模块只承载通用配置、规则执行和公共业务函数；对外稳定入口仍是
`parser_engine.extraction`。
"""

from .business import (
    infer_foundation_type,
    merge_first_cultivated_soil_layer,
    select_foundation_parameter,
)
from .config import ExtractionConfigMixin
from .rules import DerivedRulesMixin

__all__ = [
    "DerivedRulesMixin",
    "ExtractionConfigMixin",
    "infer_foundation_type",
    "merge_first_cultivated_soil_layer",
    "select_foundation_parameter",
]
