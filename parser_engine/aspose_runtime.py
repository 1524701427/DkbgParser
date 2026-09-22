from __future__ import annotations

import os
import sys
from pathlib import Path
from threading import Lock
from typing import Literal

from .exceptions import BackendUnavailableError

_LOCK = Lock()
_LOADED: set[str] = set()


def _runtime_dir() -> Path:
    """返回 Aspose DLL 所在目录。

    Returns:
        Aspose DLL 目录。环境变量配置优先于项目默认目录。
    """
    configured = os.getenv("ASPOSE_DLL_DIR") or os.getenv("ASPOSE_WORDS_DLL_DIR")
    return Path(configured).expanduser().resolve() if configured else Path(__file__).parents[1] / "runtime"


def load_aspose(product: Literal["words", "pdf"]):
    """通过 pythonnet 加载指定 Aspose 程序集，并应用本地许可证。

    Args:
        product: Aspose 产品名称，可选 ``words`` 或 ``pdf``。

    Returns:
        已加载的 ``Aspose.Words`` 或 ``Aspose.Pdf`` CLR 模块。

    Raises:
        BackendUnavailableError: 程序集不存在、运行时不可用或程序集加载失败。
    """
    assembly_name = "Aspose.Words.dll" if product == "words" else "Aspose.PDF.dll"
    namespace = "Aspose.Words" if product == "words" else "Aspose.Pdf"
    runtime_dir = _runtime_dir()
    assembly_path = runtime_dir / assembly_name
    if not assembly_path.is_file():
        raise BackendUnavailableError(f"Aspose assembly not found: {assembly_path}")

    try:
        if "clr" not in sys.modules:
            from pythonnet import load

            # 显式指定 runtimeconfig，确保 Aspose.PDF 能找到桌面运行时依赖。
            runtime_config = Path(__file__).parents[1] / "runtime" / "DkbgParser.runtimeconfig.json"
            if runtime_config.is_file():
                load("coreclr", runtime_config=str(runtime_config))
            else:
                load("coreclr")
        import clr

        with _LOCK:
            if product not in _LOADED:
                clr.AddReference(str(assembly_path))
                _LOADED.add(product)
        module = __import__(namespace, fromlist=[namespace.rsplit(".", 1)[-1]])
        _apply_license(module, runtime_dir)
        return module
    except BackendUnavailableError:
        raise
    except Exception as exc:
        raise BackendUnavailableError(
            f"Unable to load {assembly_name}. Install pythonnet and a compatible .NET runtime: {exc}"
        ) from exc


def _apply_license(module, runtime_dir: Path) -> None:
    """为 Aspose 模块设置许可证，同一模块在进程内只执行一次。

    Args:
        module: 已加载的 Aspose CLR 模块。
        runtime_dir: 默认许可证文件所在的运行时目录。
    """
    marker = f"license:{module.__name__}"
    if marker in _LOADED:
        return
    configured = os.getenv("ASPOSE_LICENSE_PATH") or os.getenv("ASPOSE_WORDS_LICENSE_PATH")
    # 允许部署环境把许可证放在项目目录之外。
    license_path = Path(configured).expanduser() if configured else runtime_dir / "Aspose.Total.NET.lic"
    if license_path.is_file():
        module.License().SetLicense(str(license_path.resolve()))
    _LOADED.add(marker)
