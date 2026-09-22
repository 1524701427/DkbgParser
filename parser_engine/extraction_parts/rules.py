from __future__ import annotations

from typing import Any


class DerivedRulesMixin:
    """ExtractionEngine 的配置化派生规则执行职责。"""

    def _apply_derived_fields(
        self, records: list[dict[str, Any]], derived_fields: dict[str, Any]
    ) -> None:
        """按 YAML 声明的顺序为每条土层记录计算派生字段。

        Args:
            records: 待补充派生参数的土层记录。
            derived_fields: derived_fields 配置字典。

        Notes:
            规则按配置顺序执行并采用首条命中项；顺序本身就是业务优先级。
            only_when_missing 与 only_when_all_missing 用于保护报告原值。
        """
        # 规则按配置顺序执行并采用首条命中项；顺序就是业务优先级。
        for record in records:
            for field_name, field_config in derived_fields.items():
                existing_value = record.get(field_name)
                only_when_missing = bool(field_config.get("only_when_missing"))
                if (
                    only_when_missing
                    and existing_value is not None
                    and not isinstance(existing_value, dict)
                ):
                    continue

                missing_fields = [
                    str(value)
                    for value in field_config.get("only_when_all_missing", [])
                ]
                if missing_fields and any(
                    record.get(name) is not None for name in missing_fields
                ):
                    continue

                formula = field_config.get("formula")
                value = (
                    self._evaluate_formula(record, formula)
                    if formula
                    else field_config.get("default")
                )
                for rule in field_config.get("rules", []):
                    if self._condition_matches(record, rule.get("when", {})):
                        value = rule.get("value")
                        break

                if value is not None:
                    if (
                        only_when_missing
                        and isinstance(existing_value, dict)
                        and isinstance(value, dict)
                    ):
                        merged_value = dict(value)
                        merged_value.update(
                            {
                                key: item
                                for key, item in existing_value.items()
                                if item is not None
                            }
                        )
                        value = merged_value
                    record[field_name] = value
                    source_label = field_config.get("source_label")
                    if source_label:
                        record.setdefault("derived_field_sources", {})[
                            field_name
                        ] = str(source_label)

    @staticmethod
    def _evaluate_formula(
        record: dict[str, Any], formula: dict[str, Any]
    ) -> float | None:
        """计算配置声明的简单两字段四则运算。

        Args:
            record: 当前土层记录。
            formula: 包含 left、right、operator、precision 的公式配置。

        Returns:
            计算后的数值；输入缺失、类型不合法或除零时返回 None。
        """
        left_field = formula.get("left")
        right_field = formula.get("right")
        if not isinstance(left_field, str) or not isinstance(right_field, str):
            return None
        left = record.get(left_field)
        right = record.get(right_field)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return None

        operator = formula.get("operator")
        if operator == "divide" and right != 0:
            value = left / right
        elif operator == "multiply":
            value = left * right
        elif operator == "add":
            value = left + right
        elif operator == "subtract":
            value = left - right
        else:
            return None
        return round(value, int(formula.get("precision", 4)))

    @staticmethod
    def _condition_matches(
        record: dict[str, Any], condition: dict[str, Any]
    ) -> bool:
        """判断当前土层记录是否命中一条配置化条件。

        Args:
            record: 当前土层记录。
            condition: 支持 all/any、contains_any、not_contains_any 及数值比较。

        Returns:
            条件命中返回 True，否则返回 False。

        Notes:
            主字段为空时会按既有逻辑回退读取同名 *_recommended 字段。
        """
        if "all" in condition:
            return all(
                DerivedRulesMixin._condition_matches(record, item)
                for item in condition["all"]
            )
        if "any" in condition:
            return any(
                DerivedRulesMixin._condition_matches(record, item)
                for item in condition["any"]
            )

        field = condition.get("field")
        if not isinstance(field, str):
            return False
        value = record.get(field)
        if value is None:
            value = record.get(f"{field}_recommended")

        if "contains_any" in condition:
            return any(
                word in str(value or "") for word in condition["contains_any"]
            )
        if "not_contains_any" in condition:
            return not any(
                word in str(value or "") for word in condition["not_contains_any"]
            )
        if value is None:
            return False
        if "lt" in condition:
            return value < condition["lt"]
        if "lte" in condition:
            return value <= condition["lte"]
        if "gt" in condition:
            return value > condition["gt"]
        if "gte" in condition:
            return value >= condition["gte"]
        if "eq" in condition:
            return value == condition["eq"]
        return bool(value)
