from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Protocol
import urllib.request


class VisionClient(Protocol):
    """视觉模型客户端需要实现的最小公共接口。"""

    def recognize(self, image_path: Path, prompt: str) -> dict[str, Any]:
        """识别单张图片并返回结构化结果。"""
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
        """初始化兼容 Chat Completions 的视觉客户端。

        Args:
            api_url: 多模态接口完整地址。
            model: 服务端模型名称。
            api_key: 可选 API Key；为空时读取 VISION_API_KEY。
            timeout: 单张图片请求超时时间，单位为秒。
        """
        self.api_url = api_url
        self.model = model
        self.api_key = api_key or os.getenv("VISION_API_KEY")
        self.timeout = timeout

    def recognize(self, image_path: Path, prompt: str) -> dict[str, Any]:
        """把页面图片发送给视觉模型，并解析模型返回的 JSON。

        Args:
            image_path: 待识别图片路径。
            prompt: 发送给视觉模型的识别提示词。

        Returns:
            模型返回的结构化 JSON 对象。

        Raises:
            RuntimeError: 响应内容中找不到合法 JSON 对象。
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

        content_text = str(content)
        decoder = json.JSONDecoder()
        for index, char in enumerate(content_text):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(content_text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        raise RuntimeError("视觉模型没有返回合法 JSON")
