from __future__ import annotations

import base64
import json
import os
import re
import shutil
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol

from .aspose_runtime import load_aspose
from .models import DocumentBlock, DocumentModel


class RapidOCRClient:
    """使用免费的 RapidOCR 在本地识别钻孔柱状图。

    该客户端不调用网络接口，也不需要 API Key。它根据 OCR 返回的文字坐标，
    从钻孔柱状图的层号列、层底深度列和分层厚度列中组织结构化结果。
    """

    _CIRCLED_NUMBERS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
    _DEFAULT_SOIL_NAME_PATTERNS = (
        r"杂填土|素填土|填土|耕土",
        r"粉质黏土|粉质粘土|黏土|粘土|粉土|淤泥质黏土|淤泥质粘土",
        r"粉砂|细砂|中砂|粗砂|砾砂|圆砾|角砾|卵石|碎石",
        r"(?:全|强|中等|中|微)风化[^，,；;：: ]*",
    )

    def __init__(
        self,
        *,
        minimum_confidence: float = 0.45,
        layer_code_max_x_ratio: float = 0.22,
        bottom_depth_x_ratio: float = 0.315,
        thickness_x_ratio: float = 0.355,
        column_tolerance_ratio: float = 0.035,
        soil_name_patterns: list[str] | tuple[str, ...] | None = None,
        engine: Any | None = None,
    ) -> None:
        """初始化本地 OCR 客户端。

        Args:
            minimum_confidence: 接受 OCR 文本行的最低置信度。
            layer_code_max_x_ratio: 层号列允许的最大横坐标比例。
            bottom_depth_x_ratio: 找不到表头时，层底深度列的默认横坐标比例。
            thickness_x_ratio: 找不到表头时，分层厚度列的默认横坐标比例。
            column_tolerance_ratio: 数值中心与目标列中心的最大横向误差比例。
            soil_name_patterns: 可选的岩土名称正则列表；为空时使用内置常用名称。
            engine: 可选的 RapidOCR 实例，主要用于测试或复用模型。
        """
        self.minimum_confidence = minimum_confidence
        self.layer_code_max_x_ratio = layer_code_max_x_ratio
        self.bottom_depth_x_ratio = bottom_depth_x_ratio
        self.thickness_x_ratio = thickness_x_ratio
        self.column_tolerance_ratio = column_tolerance_ratio
        self.soil_name_patterns = tuple(soil_name_patterns or self._DEFAULT_SOIL_NAME_PATTERNS)
        self._soil_pattern = self._compile_soil_patterns(self.soil_name_patterns)
        self._engine = engine
        # 同一候选页可能先用于正文 OCR，再用于钻孔分栏识别。缓存整页结果，
        # 避免一次抽取中对相同高清图片重复执行耗时的模型推理。
        self._line_cache: dict[str, tuple[list[dict[str, Any]], float]] = {}

    def configure(self, config: dict[str, Any]) -> None:
        """使用 YAML 中的 OCR 参数更新当前客户端。

        Args:
            config: ``image_fallback.ocr`` 下的 OCR 参数。
        """
        numeric_options = (
            "minimum_confidence",
            "layer_code_max_x_ratio",
            "bottom_depth_x_ratio",
            "thickness_x_ratio",
            "column_tolerance_ratio",
        )
        for name in numeric_options:
            if name in config:
                setattr(self, name, float(config[name]))
        patterns = config.get("soil_name_patterns")
        if isinstance(patterns, list) and patterns:
            self.soil_name_patterns = tuple(str(pattern) for pattern in patterns)
            self._soil_pattern = self._compile_soil_patterns(self.soil_name_patterns)

    @staticmethod
    def _compile_soil_patterns(patterns: tuple[str, ...]) -> re.Pattern[str]:
        """把多条岩土名称规则合并为一个正则表达式。

        Args:
            patterns: 岩土名称正则列表。

        Returns:
            可用于岩性描述搜索的已编译正则表达式。

        Raises:
            ValueError: 岩土名称规则为空。
        """
        if not patterns:
            raise ValueError("soil_name_patterns 不能为空")
        return re.compile("|".join(f"(?:{pattern})" for pattern in patterns))

    def recognize(self, image_path: Path, prompt: str) -> dict[str, Any]:
        """识别一张钻孔柱状图并返回公共结构。

        Args:
            image_path: 待识别的 PNG 页面图片。
            prompt: 公共视觉接口要求的提示词；本地 OCR 不使用该参数。

        Returns:
            包含钻孔编号、岩土层及 OCR 原始证据的字典。

        Raises:
            RuntimeError: RapidOCR 未安装，或者图片中没有识别到文字。
        """
        del prompt
        lines, width = self.read_lines(image_path)
        if not lines:
            raise RuntimeError(f"RapidOCR 未在图片中识别到文字: {image_path}")

        borehole_id = self._find_borehole_id(lines)
        if borehole_id is None:
            # 普通正文插图不会包含 F01、ZK01 等钻孔编号，无需继续执行较慢的分栏 OCR。
            return {"borehole_id": None, "layers": [], "ocr_line_count": len(lines)}
        thickness_x = self._find_column_x(lines, ("分层厚度", "层厚", "厚度"), width)
        depth_x = self._find_column_x(lines, ("层底深度", "底深度"), width)
        if thickness_x is None:
            thickness_x = width * self.thickness_x_ratio
        if depth_x is None:
            depth_x = width * self.bottom_depth_x_ratio

        height = self._image_height(image_path)
        header_y = max(
            (
                float(line["y"])
                for line in lines
                if any(
                    keyword in re.sub(r"\s+", "", line["text"])
                    for keyword in ("地层编号", "层底深度", "分层厚度")
                )
            ),
            default=0.0,
        )
        # 整页 OCR 容易漏掉窄列中的圈号、下标和 0.20m 等薄层数值，
        # 对关键列做一次局部放大识别，再映射回原页坐标。
        focused_lines = self._read_focused_columns(
            image_path,
            width,
            height,
            header_y,
            thickness_x,
        )
        lines = self._merge_lines(lines, focused_lines)
        layer_items = self._find_layer_items(lines, width)
        thickness_items = self._find_number_items(lines, thickness_x, width, header_y)
        depth_items = self._find_number_items(lines, depth_x, width, header_y)
        layers = self._build_layers(
            layer_items,
            thickness_items,
            depth_items,
            lines,
            width,
            header_y,
        )
        return {
            "borehole_id": borehole_id,
            "layers": layers,
            "ocr_line_count": len(lines),
        }

    def read_lines(self, image_path: str | Path) -> tuple[list[dict[str, Any]], float]:
        """使用公共 OCR 能力读取图片文字及坐标。

        该方法不包含钻孔柱状图业务规则，可供普通图片文字、图片表格预处理等
        其他抽取任务复用。

        Args:
            image_path: 待识别图片路径。

        Returns:
            按阅读顺序排列的文字行，以及图片宽度。
        """
        resolved_path = str(Path(image_path).resolve())
        cached = self._line_cache.get(resolved_path)
        if cached is not None:
            # 调用方会追加局部 OCR 行，因此返回副本，不能暴露缓存内部列表。
            lines, width = cached
            return [dict(line) for line in lines], width
        output = self._get_engine()(resolved_path)
        lines, width = self._read_lines(output)
        self._line_cache[resolved_path] = ([dict(line) for line in lines], width)
        return lines, width

    @staticmethod
    def _image_height(image_path: Path) -> float:
        """读取图片高度，失败时返回零以保留旧版兼容行为。

        Args:
            image_path: 页面图片路径。

        Returns:
            图片像素高度；图片不可读时返回 ``0.0``。
        """
        try:
            from PIL import Image

            with Image.open(image_path) as image:
                return float(image.height)
        except (ImportError, OSError):
            return 0.0

    def _read_focused_columns(
        self,
        image_path: Path,
        width: float,
        height: float,
        header_y: float,
        thickness_x: float,
    ) -> list[dict[str, Any]]:
        """局部放大层号、层底深度和层厚列并执行第二次 OCR。

        Args:
            image_path: 完整页面图片路径。
            width: 完整页面宽度。
            height: 完整页面高度。
            header_y: 柱状图表头底部纵坐标。
            thickness_x: 分层厚度列中心横坐标。

        Returns:
            已换算回完整页面坐标的 OCR 文字行。
        """
        if height <= 0:
            return []
        # 柱状图正文通常位于页面中下部，只裁剪层号列和厚度列，可显著减少
        # 表格线、岩性描述文字对数字 OCR 的干扰。
        start_y = max(header_y, height * 0.25)
        end_y = height * 0.92
        crops = [
            (width * 0.15, width * (self.layer_code_max_x_ratio + 0.015), start_y, end_y, 3),
            (thickness_x - width * 0.018, thickness_x + width * 0.018, start_y, end_y, 4),
        ]
        result: list[dict[str, Any]] = []
        for left, right, top, bottom, scale in crops:
            result.extend(
                self._read_crop_lines(
                    image_path,
                    left=max(0.0, left),
                    top=max(0.0, top),
                    right=min(width, right),
                    bottom=min(height, bottom),
                    scale=scale,
                )
            )
        return result

    def _read_crop_lines(
        self,
        image_path: Path,
        *,
        left: float,
        top: float,
        right: float,
        bottom: float,
        scale: int,
    ) -> list[dict[str, Any]]:
        """识别一个页面裁剪区域，并把文字框坐标还原到完整页面。

        Args:
            image_path: 完整页面图片路径。
            left: 裁剪区域左边界。
            top: 裁剪区域上边界。
            right: 裁剪区域右边界。
            bottom: 裁剪区域下边界。
            scale: OCR 前的整数放大倍数。

        Returns:
            使用完整页面坐标的 OCR 文字行；裁剪失败时返回空列表。
        """
        if right <= left or bottom <= top:
            return []
        try:
            import numpy as np
            from PIL import Image, ImageOps

            with Image.open(image_path) as source:
                # 灰度化、自动对比度和放大共同改善细小竖排数字的识别率。
                crop = source.crop((int(left), int(top), int(right), int(bottom)))
                crop = ImageOps.autocontrast(crop.convert("L"))
                crop = crop.resize(
                    (crop.width * scale, crop.height * scale),
                    Image.Resampling.LANCZOS,
                ).convert("RGB")
            lines, _crop_width = self._read_lines(self._get_engine()(np.asarray(crop)))
        except (ImportError, OSError, ValueError):
            return []

        restored = []
        for line in lines:
            item = dict(line)
            x1, y1, x2, y2 = item["bbox"]
            item["bbox"] = [
                x1 / scale + left,
                y1 / scale + top,
                x2 / scale + left,
                y2 / scale + top,
            ]
            item["x"] = float(item["x"]) / scale + left
            item["y"] = float(item["y"]) / scale + top
            restored.append(item)
        return restored

    @staticmethod
    def _merge_lines(
        original: list[dict[str, Any]], focused: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """合并整页与局部 OCR 结果并去除同位置重复文字。

        Args:
            original: 整页 OCR 文字行。
            focused: 局部放大 OCR 文字行。

        Returns:
            按页面坐标排序的去重文字行。
        """
        # 局部 OCR 与整页 OCR 会重复命中同一文字框，按文本和坐标去重并保留
        # 置信度更高的一条。
        merged = list(original)
        for item in focused:
            duplicate = next(
                (
                    old
                    for old in merged
                    if re.sub(r"\s+", "", str(old["text"]))
                    == re.sub(r"\s+", "", str(item["text"]))
                    and abs(float(old["x"]) - float(item["x"])) <= 8
                    and abs(float(old["y"]) - float(item["y"])) <= 8
                ),
                None,
            )
            if duplicate is None:
                merged.append(item)
            elif float(item["confidence"]) > float(duplicate["confidence"]):
                merged[merged.index(duplicate)] = item
        return sorted(merged, key=lambda value: (value["y"], value["x"]))

    def read_text(self, image_path: str | Path) -> str:
        """识别图片并拼接为纯文本。

        Args:
            image_path: 待识别图片路径。

        Returns:
            使用换行符连接的 OCR 文字。
        """
        lines, _width = self.read_lines(image_path)
        return "\n".join(str(line["text"]) for line in lines)

    def _get_engine(self):
        """延迟创建 RapidOCR 引擎，避免普通文本解析加载模型。

        Returns:
            RapidOCR 推理实例。

        Raises:
            RuntimeError: 没有安装 RapidOCR 或 ONNX Runtime。
        """
        if self._engine is not None:
            return self._engine
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                '免费 OCR 依赖未安装，请执行: python -m pip install -e ".[opendataloader,ocr]"'
            ) from exc
        self._engine = RapidOCR()
        return self._engine

    def _read_lines(self, output: Any) -> tuple[list[dict[str, Any]], float]:
        """兼容新版和旧版 RapidOCR 输出并统一文字行坐标。

        Args:
            output: RapidOCR 原始返回值。

        Returns:
            OCR 文字行列表和图片宽度。
        """
        boxes = getattr(output, "boxes", None)
        texts = getattr(output, "txts", None)
        scores = getattr(output, "scores", None)
        image = getattr(output, "img", None)

        # 兼容旧版 ``(result, elapsed)`` 和 ``[box, text, score]`` 返回格式。
        if boxes is None and isinstance(output, tuple) and output:
            legacy = output[0]
            if isinstance(legacy, list):
                boxes = [item[0] for item in legacy]
                texts = [item[1] for item in legacy]
                scores = [item[2] for item in legacy]

        if boxes is None or texts is None:
            return [], 1.0
        width = float(image.shape[1]) if image is not None and hasattr(image, "shape") else 0.0
        lines = []
        for index, (box, text) in enumerate(zip(boxes, texts)):
            score = float(scores[index]) if scores is not None else 1.0
            if score < self.minimum_confidence or not str(text).strip():
                continue
            points = [[float(value) for value in point] for point in box]
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            width = max(width, max(xs, default=0.0))
            lines.append(
                {
                    "text": str(text).strip(),
                    "confidence": score,
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "x": sum(xs) / len(xs),
                    "y": sum(ys) / len(ys),
                }
            )
        return sorted(lines, key=lambda item: (item["y"], item["x"])), max(width, 1.0)

    @staticmethod
    def _find_borehole_id(lines: list[dict[str, Any]]) -> str | None:
        """从 OCR 文字中查找 F01、ZK01 等钻孔编号。

        Args:
            lines: 已规范化的 OCR 文字行。

        Returns:
            钻孔编号；没有识别到时返回 ``None``。
        """
        for line in lines:
            match = re.search(r"\b([A-Za-z]{1,4})\s*[-－]?\s*(\d{1,4})\b", line["text"])
            if match:
                return f"{match.group(1).upper()}{match.group(2)}"
        return None

    @staticmethod
    def _find_column_x(
        lines: list[dict[str, Any]], keywords: tuple[str, ...], width: float
    ) -> float | None:
        """根据横排或竖排表头确定目标数值列中心。

        Args:
            lines: 已规范化的 OCR 文字行。
            keywords: 可接受的表头关键词。
            width: 图片宽度。

        Returns:
            表头中心横坐标；无法定位时返回 ``None``。
        """
        for line in lines:
            compact = re.sub(r"\s+", "", line["text"])
            if any(keyword in compact for keyword in keywords):
                return float(line["x"])

        # 柱状图表头经常竖排，OCR 可能将每个汉字识别为一个独立文字框。
        columns: list[list[dict[str, Any]]] = []
        tolerance = width * 0.015
        for line in sorted(lines, key=lambda item: item["x"]):
            column = next(
                (items for items in columns if abs(items[0]["x"] - line["x"]) <= tolerance),
                None,
            )
            if column is None:
                columns.append([line])
            else:
                column.append(line)
        for column in columns:
            text = "".join(item["text"] for item in sorted(column, key=lambda item: item["y"]))
            if any(keyword in re.sub(r"\s+", "", text) for keyword in keywords):
                return sum(float(item["x"]) for item in column) / len(column)
        return None

    def _find_layer_items(
        self, lines: list[dict[str, Any]], width: float
    ) -> list[dict[str, Any]]:
        """识别位于图表左侧的岩土层编号。

        Args:
            lines: 已规范化的 OCR 文字行。
            width: 图片宽度。

        Returns:
            按纵坐标排序的层号文字行。
        """
        result = []
        for line in lines:
            text = re.sub(r"\s+", "", line["text"])
            match = re.search(f"([{self._CIRCLED_NUMBERS}](?:[-－]?\\d+)?)", text)
            if match is None and line["x"] <= width * self.layer_code_max_x_ratio:
                match = re.search(r"(?:第|层)(\d+(?:-\d+)?)(?:层)?", text)
            if match is None and line["x"] <= width * self.layer_code_max_x_ratio:
                match = re.fullmatch(r"(\d{1,2}(?:-\d+)?)", text)
            if match is None:
                continue
            item = dict(line)
            item["layer_code"] = self._normalize_layer_code(match.group(1))
            result.append(item)

        # 相同纵向位置出现重复层号时只保留置信度最高的一条。
        unique: list[dict[str, Any]] = []
        for item in sorted(result, key=lambda value: value["y"]):
            duplicate = next(
                (
                    old
                    for old in unique
                    if old["layer_code"] == item["layer_code"]
                    and abs(old["y"] - item["y"]) <= 8
                ),
                None,
            )
            if duplicate is None:
                unique.append(item)
            elif item["confidence"] > duplicate["confidence"]:
                unique[unique.index(duplicate)] = item
        return unique

    def _find_number_items(
        self,
        lines: list[dict[str, Any]],
        column_x: float,
        width: float,
        minimum_y: float = 0.0,
    ) -> list[dict[str, Any]]:
        """提取靠近目标列中心的数值文字框。

        Args:
            lines: 已规范化的 OCR 文字行。
            column_x: 目标列中心横坐标。
            width: 图片宽度。
            minimum_y: 表格数据区域的最小纵坐标。

        Returns:
            按纵坐标排序的数值文字行。
        """
        tolerance = width * self.column_tolerance_ratio
        result = []
        for line in lines:
            if float(line["y"]) <= minimum_y or abs(float(line["x"]) - column_x) > tolerance:
                continue
            text = line["text"].strip().replace("，", ".").replace(",", ".")
            match = re.fullmatch(r"[-+]?(\d+(?:\.\d+)?)\s*(?:m|米)?", text, re.IGNORECASE)
            if match:
                item = dict(line)
                item["value"] = float(match.group(1))
                result.append(item)

        # 相邻数值列较窄时，同一行可能同时落入容差范围；保留最靠近目标列的一项。
        unique: list[dict[str, Any]] = []
        for item in sorted(result, key=lambda value: value["y"]):
            duplicate = next((old for old in unique if abs(old["y"] - item["y"]) <= 5), None)
            if duplicate is None:
                unique.append(item)
            elif abs(item["x"] - column_x) < abs(duplicate["x"] - column_x):
                unique[unique.index(duplicate)] = item
        return unique

    def _build_layers(
        self,
        layer_items: list[dict[str, Any]],
        thickness_items: list[dict[str, Any]],
        depth_items: list[dict[str, Any]],
        lines: list[dict[str, Any]],
        width: float,
        header_y: float = 0.0,
    ) -> list[dict[str, Any]]:
        """把层号、厚度、深度和岩性描述按纵向位置组成岩土层。

        Args:
            layer_items: 层号文字行。
            thickness_items: 分层厚度数值行。
            depth_items: 层底深度数值行。
            lines: 全部 OCR 文字行。
            width: 图片宽度。
            header_y: 数据区域开始位置，用于排除页面表头。

        Returns:
            从上到下排列的结构化岩土层列表。
        """
        layers = []
        previous_boundary_y = header_y
        for thickness in thickness_items:
            boundary_y = float(thickness["y"])
            interval_center = (previous_boundary_y + boundary_y) / 2
            candidates = [
                item
                for item in layer_items
                if previous_boundary_y <= float(item["y"]) <= boundary_y
            ]
            if candidates:
                layer = min(candidates, key=lambda item: abs(float(item["y"]) - interval_center))
            elif layer_items:
                # OCR 可能漏掉较小的圈号或下标，此时使用距离该层区间最近的已识别层号。
                layer = min(layer_items, key=lambda item: abs(float(item["y"]) - boundary_y))
            else:
                previous_boundary_y = boundary_y
                continue
            depth = min(
                depth_items,
                key=lambda item: abs(float(item["y"]) - boundary_y),
                default=None,
            )
            if depth is not None and abs(float(depth["y"]) - boundary_y) > 8:
                depth = None
            related = [
                item
                for item in lines
                if previous_boundary_y <= float(item["y"]) <= boundary_y
                and float(item["x"]) >= width * 0.40
            ]
            description = " ".join(item["text"] for item in related).strip() or None
            # 岩性标题常跨越很高的合并单元格，不能只按上下边界截取文本。
            # 使用最靠近当前层中心的“岩土名称：”文字行确定本层名称。
            name_lines = [
                item
                for item in lines
                if float(item["x"]) >= width * 0.40
                and self._soil_pattern.search(str(item["text"]))
            ]
            nearest_name_line = min(
                name_lines,
                key=lambda item: abs(float(item["y"]) - interval_center),
                default=None,
            )
            name_match = self._soil_pattern.search(
                str(nearest_name_line["text"]) if nearest_name_line else description or ""
            )
            numeric_scores = [
                item["confidence"] for item in (thickness, depth) if item is not None
            ]
            confidence = min([float(layer["confidence"]), *numeric_scores])
            layers.append(
                {
                    "layer_code": layer["layer_code"],
                    "layer_name": name_match.group(0) if name_match else None,
                    "bottom_elevation": None,
                    "bottom_depth": depth["value"] if depth else None,
                    "thickness": thickness["value"] if thickness else None,
                    "description": description,
                    "confidence": round(confidence, 4),
                    "ocr_evidence": {
                        "layer_code_bbox": layer["bbox"],
                        "thickness_bbox": thickness["bbox"] if thickness else None,
                        "bottom_depth_bbox": depth["bbox"] if depth else None,
                    },
                }
            )
            previous_boundary_y = boundary_y
        return layers

    @classmethod
    def _normalize_layer_code(cls, value: str) -> str:
        """统一阿拉伯数字和带圈数字形式的层号。

        Args:
            value: OCR 识别到的层号。

        Returns:
            规范后的层号，1～20 转换为带圈数字。
        """
        value = value.strip()
        circled_match = re.fullmatch(
            f"([{cls._CIRCLED_NUMBERS}])[-－]?(\\d+)?",
            value,
        )
        if circled_match:
            suffix = circled_match.group(2)
            return circled_match.group(1) + (f"-{suffix}" if suffix else "")
        # 地勘柱状图不存在“第0层”，OCR 常把带圈的①识别成 0。
        if value == "0":
            return "①"
        match = re.fullmatch(r"(\d+)(-\d+)?", value)
        if not match:
            return value
        number = int(match.group(1))
        main = cls._CIRCLED_NUMBERS[number - 1] if 1 <= number <= len(cls._CIRCLED_NUMBERS) else str(number)
        return main + (match.group(2) or "")


class VisionClient(Protocol):
    """视觉模型客户端需要实现的最小公共接口。"""

    def recognize(self, image_path: Path, prompt: str) -> dict[str, Any]:
        """识别单张图片并返回结构化结果。

        Args:
            image_path: 待识别的图片路径。
            prompt: 图片识别要求。

        Returns:
            包含钻孔编号和岩土层列表的字典。
        """
        ...


class OpenAICompatibleVisionClient:
    """调用兼容多模态 Chat Completions 格式的视觉模型服务。"""

    def __init__(
        self,
        api_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: int = 120,
    ) -> None:
        """初始化视觉模型客户端。

        Args:
            api_url: 多模态接口完整地址。
            model: 服务端使用的视觉模型名称。
            api_key: 接口密钥；为空时读取 ``VISION_API_KEY``。
            timeout: 单页请求超时时间，单位为秒。
        """
        self.api_url = api_url
        self.model = model
        self.api_key = api_key or os.getenv("VISION_API_KEY")
        self.timeout = timeout

    def recognize(self, image_path: Path, prompt: str) -> dict[str, Any]:
        """把页面图片发送给视觉模型，并解析模型返回的 JSON。

        Args:
            image_path: 待识别的 PNG 页面图片。
            prompt: 钻孔柱状图识别提示词。

        Returns:
            视觉模型返回的结构化字典。

        Raises:
            RuntimeError: 服务响应中没有合法的 JSON 对象。
        """
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"},
                        },
                    ],
                }
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        if isinstance(content, dict):
            return content
        match = re.search(r"\{.*\}", str(content), re.DOTALL)
        if not match:
            raise RuntimeError("视觉模型没有返回合法 JSON")
        return json.loads(match.group(0))


class BoreholeImageRecognizer:
    """渲染并识别 PDF 中的钻孔柱状图。"""

    _CIRCLED_NUMBERS = RapidOCRClient._CIRCLED_NUMBERS
    _CATALOG_NAME_PATTERN = re.compile(
        r"杂填土|素填土|填土|耕土|粉质黏土|粉质粘土|淤泥质黏土|淤泥质粘土|"
        r"黏土|粘土|粉土|粉砂|细砂|中砂|粗砂|砾砂|圆砾|角砾|卵石|碎石|"
        r"(?:全|强|中等|中|微)风化[\u4e00-\u9fff]{0,8}岩"
    )
    # RapidOCR 可能把亚层编号中的普通数字识别为 Unicode 下标数字。
    # 统一在生成 DocumentBlock 前恢复，避免影响后续公共层号正则。
    _SUBSCRIPT_TRANSLATION = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

    def __init__(self, client: VisionClient, output_dir: str | Path = "output/assets") -> None:
        """初始化钻孔柱状图识别器。

        Args:
            client: 实现 ``VisionClient`` 协议的视觉模型客户端。
            output_dir: 渲染页面的持久化输出目录。
        """
        self.client = client
        self.output_dir = Path(output_dir)

    def _configure_client(self, config: dict[str, Any]) -> None:
        """把 OCR 子配置传递给支持动态配置的视觉客户端。

        ``VisionClient`` 协议只要求实现 ``recognize``，因此这里继续使用
        能力探测方式调用 ``configure``，保持对视觉大模型客户端的兼容。

        Args:
            config: ``image_fallback`` 完整配置。
        """
        configure = getattr(self.client, "configure", None)
        if callable(configure):
            configure(config.get("ocr", {}))

    def _prepare_images(
        self,
        document: DocumentModel,
        config: dict[str, Any],
        *,
        empty_pages_message: str,
    ) -> list[tuple[int, Path]]:
        """确定候选页并统一渲染为待识别图片。

        该方法只收敛 ``recognize_document`` 和 ``recognize_text_blocks``
        共有的候选页、输出目录和渲染流程，不增加新的业务判断。

        Args:
            document: 已解析的统一文档模型。
            config: ``image_fallback`` 配置。
            empty_pages_message: 没有候选页时使用的原始错误信息。

        Returns:
            ``(页码, 图片路径)`` 列表。

        Raises:
            ValueError: 没有找到候选页面。
            RuntimeError: 页面渲染依赖不可用或文档转 PDF 失败。
        """
        pages = self._candidate_pages(document, config)
        if not pages:
            raise ValueError(empty_pages_message)

        task_dir = self.output_dir / Path(document.source_path).stem
        task_dir.mkdir(parents=True, exist_ok=True)

        return self._render_pages(
            Path(document.source_path),
            pages,
            task_dir,
            int(config.get("render", {}).get("dpi", 350)),
            document.source_format,
        )

    def recognize_document(
        self,
        document: DocumentModel,
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """定位、渲染并识别文档中的钻孔柱状图页面。

        Args:
            document: 已解析的 PDF 统一文档模型。
            config: YAML 中的 ``image_fallback`` 配置。

        Returns:
            按岩土层聚合后的图片候选记录。

        Raises:
            ValueError: 输入文档不是 PDF，或没有可识别页面。
            RuntimeError: 页面渲染依赖不可用或视觉接口调用失败。
        """
        if document.source_format.lower() not in {"pdf", "doc", "docx", "docm", "rtf", "odt"}:
            raise ValueError("图片兜底仅支持 PDF 和 Word 文档")
        self._configure_client(config)
        images = self._prepare_images(
            document,
            config,
            empty_pages_message="没有找到钻孔柱状图候选页面",
        )
        prompt = str(config.get("prompt") or self._default_prompt())
        layer_catalog = self._extract_layer_catalog(document)
        observations: list[dict[str, Any]] = []
        failures: list[str] = []
        for page, image_path in images:
            try:
                result = self.client.recognize(image_path, prompt)
            except (RuntimeError, OSError, ValueError) as exc:
                # 候选附件中可能混有普通插图或无文字页面，单页失败不应中断其他钻孔。
                failures.append(f"第 {page} 页: {exc}")
                continue
            if not self._is_borehole_result(result):
                continue
            page_records = self._normalize_result(result, page, image_path)
            self._apply_layer_catalog(page_records, layer_catalog)
            self._repair_layer_sequence(page_records, layer_catalog)
            observations.extend(page_records)
        if not observations:
            message = "候选页面中没有识别到有效钻孔柱状图"
            if failures:
                message += "；" + "；".join(failures[:3])
            raise RuntimeError(message)
        self._validate_observations(observations, float(config.get("validation", {}).get("depth_tolerance", 0.05)))
        return self._aggregate(observations, str(config.get("aggregate", "average")))

    def recognize_text_blocks(
        self,
        document: DocumentModel,
        config: dict[str, Any],
    ) -> list[DocumentBlock]:
        """把候选图片页的普通 OCR 文字转换为公共内容块。

        该方法处理“PDF 可见文字正常、内嵌文本编码乱码”的报告。返回的内容块
        继续交给公共土层正则解析，不在 OCR 模块中重复实现厚度和层名规则。

        Args:
            document: 已解析的统一文档模型。
            config: ``image_fallback`` 配置。

        Returns:
            按页码和视觉阅读顺序排列的 OCR 段落块。

        Raises:
            ValueError: 没有找到候选页面。
        """
        read_lines = getattr(self.client, "read_lines", None)
        if not callable(read_lines):
            # 视觉大模型客户端通常只提供结构化 recognize 接口；这种情况下
            # 跳过普通正文 OCR，仍由原有钻孔图识别流程继续处理。
            return []
        self._configure_client(config)
        images = self._prepare_images(
            document,
            config,
            empty_pages_message="没有找到 OCR 文字候选页面",
        )
        blocks: list[DocumentBlock] = []
        for page, image_path in images:
            lines, _width = read_lines(image_path)
            for line_number, line in enumerate(lines, start=1):
                text = str(line.get("text") or "").translate(self._SUBSCRIPT_TRANSLATION).strip()
                if not text:
                    continue
                blocks.append(
                    DocumentBlock(
                        id=f"ocr_p{page}_l{line_number}",
                        kind="paragraph",
                        text=text,
                        page=page,
                        metadata={
                            "source_type": "image_ocr",
                            "confidence": line.get("confidence"),
                            "image_path": str(image_path),
                        },
                    )
                )
        return blocks

    @staticmethod
    def _candidate_pages(document: DocumentModel, config: dict[str, Any]) -> list[int]:
        """根据显式页码、关键词、图片块和尾页策略确定候选页。

        Args:
            document: 统一文档模型。
            config: 图片识别配置。

        Returns:
            去重并升序排列的候选页码。
        """
        explicit_pages = config.get("pages")
        if explicit_pages:
            return sorted({int(page) for page in explicit_pages})

        keywords = [str(value).replace(" ", "") for value in config.get("page_keywords", [])]
        pages = {
            block.page
            for block in document.blocks
            if block.page
            and keywords
            and any(keyword in block.text.replace(" ", "") for keyword in keywords)
        }
        if config.get("use_image_blocks", False):
            # 关键词所在页可能只是正文中的“详见钻孔柱状图”。整页图片附件必须一并扫描，
            # 最终再根据钻孔编号和逐层厚度判断是否是真正的柱状图。
            pages.update(
                block.page
                for block in document.blocks
                if block.kind == "image" and block.page
            )
            pages.update(
                page.number
                for page in document.pages
                if (
                    (page.image_count > 0 and page.text_characters < 200)
                    # OpenDataLoader 对整页扫描图有时不会登记 image_count，
                    # 但这类附件页通常只有极少量可提取文字，仍需交给 OCR 二次确认。
                    or page.text_characters < int(config.get("image_page_max_text", 150))
                )
            )
        if not pages:
            last_pages = max(0, int(config.get("fallback_last_pages", 5)))
            page_count = max(
                len(document.pages),
                max((block.page or 0 for block in document.blocks), default=0),
            )
            pages = set(range(max(1, page_count - last_pages + 1), page_count + 1))
        return sorted(int(page) for page in pages)

    @staticmethod
    def _is_borehole_result(result: dict[str, Any]) -> bool:
        """判断视觉结果是否来自真正的钻孔柱状图。

        Args:
            result: 单页视觉识别结果。

        Returns:
            同时包含钻孔编号和至少一条有效层厚时返回 ``True``。
        """
        if not result.get("borehole_id"):
            return False
        return any(
            isinstance(layer, dict)
            and layer.get("layer_code")
            and BoreholeImageRecognizer._number(layer.get("thickness")) is not None
            for layer in result.get("layers", [])
        )

    @classmethod
    def _extract_layer_catalog(cls, document: DocumentModel) -> dict[str, str]:
        """从正文和表格中建立层号到标准岩土名称的目录。

        柱状图负责提供厚度，正文负责提供更可靠的层号含义。两种来源融合后，
        可以修正 OCR 把 ``④-1`` 识别成 ``④`` 等常见错误。

        Args:
            document: 已解析的统一文档模型。

        Returns:
            规范层号到岩土名称的映射。
        """
        code_pattern = re.compile(
            rf"(?:第|层)?(?P<main>[{cls._CIRCLED_NUMBERS}])"
            rf"[-－]?(?P<suffix>\d+)?(?P<gap>层)?\s*"
            rf"(?P<name>{cls._CATALOG_NAME_PATTERN.pattern})"
        )
        catalog: dict[str, str] = {}
        for block in document.blocks:
            compact = re.sub(r"\s+", "", block.text or "")
            for match in code_pattern.finditer(compact):
                suffix = match.group("suffix")
                code = match.group("main") + (f"-{suffix}" if suffix else "")
                # 保留正文中首次出现的名称，避免后续推荐表的简写覆盖地层描述。
                catalog.setdefault(code, match.group("name"))
        return catalog

    @classmethod
    def _apply_layer_catalog(
        cls,
        records: list[dict[str, Any]],
        catalog: dict[str, str],
    ) -> None:
        """用正文土层目录校正单页 OCR 的层号和名称。

        Args:
            records: 单张柱状图的逐层观测记录。
            catalog: 正文提取出的标准层号和岩土名称。
        """
        # OCR 的层号容易漏读“-1/-2”；正文层目录同时提供层号和岩土名称，
        # 只有候选唯一时才校正，避免同名土层之间被错误合并。
        for record in records:
            code = str(record.get("layer_code") or "")
            main_code = cls._main_layer_code(code)
            context = re.sub(
                r"\s+",
                "",
                f"{record.get('layer_name') or ''}{record.get('description') or ''}",
            )
            recognized_name = re.sub(r"\s+", "", str(record.get("layer_name") or ""))
            global_candidates = [
                (candidate_code, name)
                for candidate_code, name in catalog.items()
                if recognized_name
                and (name in recognized_name or recognized_name in name)
            ]
            recognized_same_main = [
                item
                for item in global_candidates
                if cls._main_layer_code(item[0]) == main_code
            ]
            global_context_candidates = [
                (candidate_code, name)
                for candidate_code, name in catalog.items()
                if name in context
            ]
            same_main_candidates = [
                (candidate_code, name)
                for candidate_code, name in catalog.items()
                if cls._main_layer_code(candidate_code) == main_code and name in context
            ]
            if len(global_candidates) == 1:
                candidates = global_candidates
            elif len(recognized_same_main) == 1:
                candidates = recognized_same_main
            elif len(global_context_candidates) == 1:
                candidates = global_context_candidates
            else:
                candidates = same_main_candidates
            if len(candidates) == 1:
                code, name = candidates[0]
                record["layer_code"] = code
                record["main_layer_code"] = cls._main_layer_code(code)
                record["layer_name"] = name
            elif code in catalog:
                record["layer_name"] = catalog[code]

    @classmethod
    def _repair_layer_sequence(
        cls,
        records: list[dict[str, Any]],
        catalog: dict[str, str],
    ) -> None:
        """依据同一钻孔的上下层顺序修复被漏读的主层圈号。

        该规则只处理“连续两个同名同号层，且后方已出现更大主层”的明确缺号情形。
        例如 F12 中 OCR 把位于②层和④-1层之间的③层误读成了第二个②层。

        Args:
            records: 已完成正文目录校正的单孔逐层记录。
            catalog: 正文层号到标准岩土名称的映射。
        """
        # 该补偿只处理可由上下层顺序唯一确定的漏号，不对模糊结果强行猜测。
        main_order = {
            value: index + 1 for index, value in enumerate(cls._CIRCLED_NUMBERS)
        }
        for index in range(1, len(records) - 1):
            previous = records[index - 1]
            current = records[index]
            previous_main = cls._main_layer_code(str(previous.get("layer_code") or ""))
            current_main = cls._main_layer_code(str(current.get("layer_code") or ""))
            if previous_main != current_main or previous.get("layer_name") != current.get("layer_name"):
                continue
            later_mains = [
                cls._main_layer_code(str(item.get("layer_code") or ""))
                for item in records[index + 1 :]
            ]
            current_number = main_order.get(current_main)
            later_numbers = [main_order[value] for value in later_mains if value in main_order]
            if current_number is None or not later_numbers:
                continue
            next_number = min(later_numbers)
            missing_codes = [
                code
                for code, name in catalog.items()
                if code == cls._main_layer_code(code)
                and current_number < main_order.get(code, 0) < next_number
                and name == current.get("layer_name")
            ]
            if len(missing_codes) == 1:
                repaired_code = missing_codes[0]
                current["layer_code"] = repaired_code
                current["main_layer_code"] = repaired_code

    @staticmethod
    def _render_pages(
        source_path: Path,
        pages: list[int],
        output_dir: Path,
        dpi: int,
        source_format: str = "pdf",
    ) -> list[tuple[int, Path]]:
        """将候选 PDF 或 Word 页面渲染为高清 PNG。

        Args:
            source_path: 原始 PDF 或 Word 路径。
            pages: 一基页码列表。
            output_dir: PNG 输出目录。
            dpi: 页面渲染分辨率。
            source_format: 输入文档格式。

        Returns:
            页码和对应 PNG 路径的列表。

        Raises:
            RuntimeError: 没有安装 PyMuPDF。
        """
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("图片识别需要安装 PyMuPDF: pip install PyMuPDF") from exc

        render_source = source_path
        if source_format.lower() != "pdf":
            # Aspose.Words 先生成版式一致的 PDF，再复用统一的页面渲染流程。
            render_source = output_dir / "_word_render_source.pdf"
            try:
                # 与 Word 正文解析保持一致：先复制只读快照。Word/WPS 正在打开
                # 原文件时，Aspose 直接读取源路径可能因共享锁失败。
                with tempfile.TemporaryDirectory(prefix="dkbg_word_ocr_") as temp_dir:
                    snapshot_path = Path(temp_dir) / source_path.name
                    shutil.copy2(source_path, snapshot_path)
                    word_document = load_aspose("words").Document(str(snapshot_path))
                    word_document.Save(str(render_source))
            except Exception as exc:
                raise RuntimeError(f"Word 页面转 PDF 失败: {exc}") from exc

        rendered = []
        scale = dpi / 72.0
        with fitz.open(render_source) as pdf:
            for page_number in pages:
                if page_number < 1 or page_number > pdf.page_count:
                    continue
                output_path = output_dir / f"page_{page_number:04d}.png"
                pixmap = pdf[page_number - 1].get_pixmap(
                    matrix=fitz.Matrix(scale, scale),
                    alpha=False,
                )
                pixmap.save(output_path)
                rendered.append((page_number, output_path))
        return rendered

    @staticmethod
    def _normalize_result(
        result: dict[str, Any],
        page: int,
        image_path: Path,
    ) -> list[dict[str, Any]]:
        """将单页视觉识别结果转换为公共候选字段。

        Args:
            result: 视觉模型返回的字典。
            page: 原 PDF 页码。
            image_path: 本次识别使用的页面图片。

        Returns:
            当前页面的岩土层观测记录。
        """
        # 所有识别后端都转换成同一 observation 结构，后续聚合逻辑不依赖
        # RapidOCR 或视觉模型的私有返回格式。
        borehole_id = result.get("borehole_id")
        records = []
        for layer in result.get("layers", []):
            if not isinstance(layer, dict) or not layer.get("layer_code"):
                continue
            code = re.sub(r"\s+", "", str(layer["layer_code"]))
            records.append(
                {
                    "layer_code": code,
                    "main_layer_code": BoreholeImageRecognizer._main_layer_code(code),
                    "layer_name": layer.get("layer_name"),
                    "bottom_elevation": BoreholeImageRecognizer._number(layer.get("bottom_elevation")),
                    "bottom_depth": BoreholeImageRecognizer._number(layer.get("bottom_depth")),
                    "image_thickness": BoreholeImageRecognizer._number(layer.get("thickness")),
                    "description": layer.get("description"),
                    "confidence": BoreholeImageRecognizer._number(layer.get("confidence")),
                    "borehole_id": borehole_id,
                    "evidence": {
                        "source_type": "image",
                        "page": page,
                        "image_path": str(image_path),
                        "ocr_evidence": layer.get("ocr_evidence"),
                    },
                }
            )
        return records

    @staticmethod
    def _validate_observations(records: list[dict[str, Any]], tolerance: float) -> None:
        """用相邻层底深度差校验或补充分层厚度。

        Args:
            records: 多个钻孔的逐层观测记录。
            tolerance: 深度计算值与识别厚度的允许误差。
        """
        # 相邻层底深度必须在同一个钻孔内相减，绝不能跨孔连续计算层厚。
        groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            evidence = record.get("evidence", {})
            group_key = record.get("borehole_id") or f"page:{evidence.get('page')}"
            groups[group_key].append(record)
        for layers in groups.values():
            previous_depth = 0.0
            for layer in layers:
                bottom_depth = layer.get("bottom_depth")
                thickness = layer.get("image_thickness")
                if bottom_depth is None:
                    layer["depth_validation"] = None
                    continue
                calculated = round(bottom_depth - previous_depth, 3)
                if thickness is None and calculated > 0:
                    layer["image_thickness"] = calculated
                    layer["thickness_source"] = "calculated_from_bottom_depth"
                    layer["depth_validation"] = True
                elif thickness is not None:
                    layer["depth_validation"] = abs(calculated - thickness) <= tolerance
                previous_depth = bottom_depth

    @staticmethod
    def _aggregate(records: list[dict[str, Any]], operator: str) -> list[dict[str, Any]]:
        """按完整层号汇总多个钻孔中的层厚观测值。

        Args:
            records: 所有钻孔的逐层观测记录。
            operator: ``average``、``minimum`` 或 ``maximum``。

        Returns:
            带图片统计厚度和全部观测证据的候选记录。
        """
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            groups[str(record["layer_code"])].append(record)
        result = []
        for code, observations in groups.items():
            values = [item["image_thickness"] for item in observations if item.get("image_thickness") is not None]
            if not values:
                continue
            if operator == "minimum":
                aggregated = min(values)
            elif operator == "maximum":
                aggregated = max(values)
            else:
                aggregated = sum(values) / len(values)
            first = observations[0]
            layer_name = next(
                (item.get("layer_name") for item in observations if item.get("layer_name")),
                None,
            )
            borehole_ids = sorted(
                {
                    str(item["borehole_id"])
                    for item in observations
                    if item.get("borehole_id")
                },
                key=BoreholeImageRecognizer._borehole_sort_key,
            )
            governing_borehole_ids = sorted(
                {
                    str(item["borehole_id"])
                    for item in observations
                    if item.get("borehole_id")
                    and item.get("image_thickness") == aggregated
                },
                key=BoreholeImageRecognizer._borehole_sort_key,
            )
            result.append(
                {
                    "layer_code": code,
                    "main_layer_code": first["main_layer_code"],
                    "layer_name": layer_name,
                    "image_average": round(aggregated, 3),
                    "image_thickness": first.get("image_thickness") if len(observations) == 1 else None,
                    "observation_count": len(values),
                    "borehole_ids": borehole_ids,
                    # 最小值或最大值汇总时，记录最终控制厚度来自哪些钻孔。
                    "governing_borehole_ids": (
                        governing_borehole_ids if operator in {"minimum", "maximum"} else []
                    ),
                    "observations": observations,
                    "evidence": first["evidence"],
                }
            )
        return result

    @staticmethod
    def _borehole_sort_key(value: str) -> tuple[str, int, str]:
        """生成钻孔编号的自然排序键。

        Args:
            value: 例如 ``F03``、``ZK12`` 的钻孔编号。

        Returns:
            字母前缀、数字序号和原值组成的排序键。
        """
        match = re.fullmatch(r"([A-Za-z]+)0*(\d+)", value.strip())
        if not match:
            return value.upper(), -1, value
        return match.group(1).upper(), int(match.group(2)), value

    @staticmethod
    def _main_layer_code(code: str) -> str:
        """从完整层号中提取主层号。

        Args:
            code: 完整岩土层编号。

        Returns:
            去掉亚层后缀的主层号。
        """
        match = re.match(r"([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|\d+)", code)
        return match.group(1) if match else code

    @staticmethod
    def _number(value: Any) -> float | None:
        """把模型返回的数字或带单位字符串转换为浮点数。

        Args:
            value: 原始数字、字符串或空值。

        Returns:
            浮点数；无法识别时返回 ``None``。
        """
        if value is None:
            return None
        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        return float(match.group(0)) if match else None

    @staticmethod
    def _default_prompt() -> str:
        """返回默认钻孔柱状图识别提示词。

        Returns:
            要求模型只返回结构化 JSON 的中文提示词。
        """
        return """识别图片中的钻孔柱状图，只返回一个 JSON 对象，不要返回 Markdown。
JSON 格式：
{"borehole_id":"钻孔编号", "layers":[{"layer_code":"层号", "layer_name":"岩土名称", "bottom_elevation":数值或null, "bottom_depth":数值或null, "thickness":数值或null, "description":"岩性描述", "confidence":0到1}]}
必须按照图片中从上到下的岩土层顺序输出。无法确认的字段使用 null，禁止猜测。数字不要包含单位。"""
