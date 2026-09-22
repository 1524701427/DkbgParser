from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any
import urllib.error
import urllib.request


DEFAULT_REVERSE_GEOLOGY_URL = (
    "http://172.16.14.71:10004/rpc-api/reverse-callback/parse-reverse-geology"
)


def post_reverse_geology_payload(
    payload: Mapping[str, Any],
    *,
    api_url: str = DEFAULT_REVERSE_GEOLOGY_URL,
    timeout: float = 30.0,
) -> Any:
    """把映射后的地质结果作为 JSON 直接 POST 到逆向回调接口。"""
    url = str(api_url or "").strip()
    if not url:
        raise ValueError("逆向地质回调地址不能为空")
    if timeout <= 0:
        raise ValueError("接口请求超时时间必须大于0")

    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            status = getattr(response, "status", None) or response.getcode()
            response_text = response.read().decode("utf-8", errors="replace").strip()
    except urllib.error.HTTPError as exc:
        try:
            error_text = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            error_text = ""
        detail = f": {error_text}" if error_text else ""
        raise RuntimeError(
            f"逆向地质接口请求失败，HTTP {exc.code}{detail}"
        ) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"逆向地质接口请求失败: {reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("逆向地质接口请求超时") from exc

    if int(status) < 200 or int(status) >= 300:
        raise RuntimeError(
            f"逆向地质接口请求失败，HTTP {status}: {response_text}"
        )
    if not response_text:
        return None
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return response_text
