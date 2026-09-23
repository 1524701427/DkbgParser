from __future__ import annotations

from contextlib import contextmanager
import logging
from pathlib import Path
import sys
from time import perf_counter
from typing import Iterator


def configure_logging(log_path: str | Path | None = None) -> None:
    """同时配置控制台日志和可选的 UTF-8 文件日志。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers: list[logging.Handler] = []
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    handlers.append(console)

    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        force=True,
    )


def close_file_handlers_under(directory: str | Path) -> None:
    """关闭指定目录下由日志系统持有的文件句柄。

    FastAPI 请求使用临时输出目录。在 Windows 中，如果 ``FileHandler`` 没有
    显式关闭，临时目录清理会因日志文件仍被占用而失败。

    Args:
        directory: 即将被删除的临时目录或其父目录。
    """
    root = Path(directory).resolve()
    loggers: list[logging.Logger] = [logging.getLogger()]
    loggers.extend(
        item
        for item in logging.Logger.manager.loggerDict.values()
        if isinstance(item, logging.Logger)
    )

    # 同一个处理器可能同时挂在多个 logger 上，用 id 去重后只关闭一次。
    closed_handlers: set[int] = set()
    for current_logger in loggers:
        for handler in list(current_logger.handlers):
            if not isinstance(handler, logging.FileHandler):
                continue
            try:
                log_path = Path(handler.baseFilename).resolve()
                belongs_to_directory = log_path.is_relative_to(root)
            except (OSError, ValueError):
                belongs_to_directory = False
            if not belongs_to_directory:
                continue
            current_logger.removeHandler(handler)
            if id(handler) not in closed_handlers:
                handler.close()
                closed_handlers.add(id(handler))


@contextmanager
def log_stage(logger: logging.Logger, stage_name: str) -> Iterator[None]:
    """记录一个主流程阶段的开始、完成和耗时，异常时同样记录耗时。"""
    started = perf_counter()
    logger.info("阶段开始：%s", stage_name)
    try:
        yield
    except Exception:
        elapsed = perf_counter() - started
        logger.exception("阶段失败：%s，耗时 %.3fs", stage_name, elapsed)
        raise
    else:
        elapsed = perf_counter() - started
        logger.info("阶段完成：%s，耗时 %.3fs", stage_name, elapsed)
