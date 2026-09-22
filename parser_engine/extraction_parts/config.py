from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import yaml


class ExtractionConfigMixin:
    """ExtractionEngine 的配置加载与校验职责。"""

    _SUPPORTED_MODES = frozenset({"layer_records", "keyword_fields", "section_content"})

    @classmethod
    def _validate_config(cls, config: dict[str, Any]) -> None:
        name = str(config.get("name") or "").strip()
        if not name:
            raise ValueError("抽取配置缺少 name")
        mode = str(config.get("mode", "layer_records"))
        if mode not in cls._SUPPORTED_MODES:
            raise ValueError(f"配置 {name} 使用了未知抽取模式: {mode}")
        if mode == "layer_records" and not isinstance(config.get("fields"), dict):
            raise ValueError(f"配置 {name} 的 layer_records 模式缺少 fields 字典")
        if mode == "keyword_fields" and not isinstance(config.get("fields"), dict):
            raise ValueError(f"配置 {name} 的 keyword_fields 模式缺少 fields 字典")
        if mode == "section_content":
            sections = config.get("sections")
            if not isinstance(sections, list) or not sections:
                raise ValueError(f"配置 {name} 的 section_content 模式缺少 sections 列表")
            for index, section in enumerate(sections, start=1):
                if not isinstance(section, dict) or not section.get("key"):
                    raise ValueError(f"配置 {name} 的第 {index} 个章节缺少 key")
                groups = section.get("alias_groups")
                if groups is not None and (
                    not isinstance(groups, list)
                    or not groups
                    or any(not isinstance(group, list) or not group for group in groups)
                ):
                    raise ValueError(
                        f"配置 {name} 的章节 {section['key']} 的 alias_groups 必须是非空二维列表"
                    )
                has_match_rule = bool(section.get("aliases") or groups)
                if (
                    not has_match_rule
                    and not section.get("fallback_text")
                    and not section.get("placeholder")
                ):
                    raise ValueError(
                        f"配置 {name} 的章节 {section['key']} 缺少标题别名或兜底规则"
                    )
        cls._validate_regex_values(config, name)

    @classmethod
    def _validate_regex_values(
        cls, value: Any, config_name: str, path: str = ""
    ) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if key == "pattern" and isinstance(child, str):
                    cls._compile_config_pattern(child, config_name, child_path)
                elif key.endswith("patterns") and isinstance(child, list):
                    for index, pattern in enumerate(child):
                        if isinstance(pattern, str):
                            cls._compile_config_pattern(
                                pattern, config_name, f"{child_path}[{index}]"
                            )
                cls._validate_regex_values(child, config_name, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                cls._validate_regex_values(child, config_name, f"{path}[{index}]")

    @staticmethod
    def _compile_config_pattern(pattern: str, config_name: str, path: str) -> None:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(
                f"配置 {config_name} 的正则 {path} 不合法: {exc}"
            ) from exc

    @classmethod
    def from_files(
        cls,
        config_paths: str | Path | Iterable[str | Path],
        *,
        image_recognizer: Any | None = None,
    ):
        """从一个或多个 YAML 文件创建抽取引擎。"""
        if isinstance(config_paths, (str, Path)):
            paths = [Path(config_paths)]
        else:
            paths = [Path(path) for path in config_paths]

        configs: list[dict[str, Any]] = []
        names: set[str] = set()
        for path in paths:
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(config, dict) or not config.get("name"):
                raise ValueError(f"抽取配置缺少 name: {path}")
            name = str(config["name"])
            if name in names:
                raise ValueError(f"抽取任务名称重复: {name}")
            config["_config_path"] = str(path.resolve())
            names.add(name)
            configs.append(config)
        if not configs:
            raise ValueError("至少需要一个抽取配置文件")
        return cls(configs, image_recognizer=image_recognizer)
