"""ロギングのセットアップユーティリティ.

アプリケーション全体で統一されたフォーマットのロガーを生成する.
各モジュールで `setup_logging(__name__)` のように呼び出して使用する.

出力フォーマット例:
    2024-01-15 12:34:56 - ply.ply - INFO - Successfully loaded and preprocessed ply file: ...
"""

import logging
import sys
from logging import Logger
from typing import TextIO


_HANDLER_MARKER = "_core_setup_logging_handler"
_CORE_LOGGER_NAME = "core"


def _create_handler(stream: TextIO, level: int) -> logging.StreamHandler[TextIO]:
    """core 用の標準フォーマット済みハンドラを生成する."""
    handler = logging.StreamHandler(stream)
    setattr(handler, _HANDLER_MARKER, True)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    return handler


def setup_logging(
    name: str,
    *,
    stream: TextIO = sys.stderr,
    level: int = logging.INFO,
) -> Logger:
    """指定された名前のロガーを作成・設定して返す.

    INFOレベル以上のログをコンソール(stderr)に出力するハンドラを設定する.

    既にハンドラが設定済みの場合はハンドラの重複追加を防止する.

    Args:
        name: ロガー名. 通常は `__name__` を渡してモジュール名を設定する

    Returns:
        設定済みのLoggerインスタンス.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    handlers = [
        handler
        for handler in logger.handlers
        if getattr(handler, _HANDLER_MARKER, False)
    ]
    if not handlers:
        logger.addHandler(_create_handler(stream, level))
    else:
        for handler in handlers:
            handler.setLevel(level)
            if isinstance(handler, logging.StreamHandler):
                handler.setStream(stream)

    return logger


def configure_core_logging(*, stream: TextIO, level: int = logging.INFO) -> None:
    """初期化済みの core logger を CLI 実行時の出力先へ統一する.

    CLI のオプションはモジュール import 後に評価されるため、既に作成済みの
    logger もここで再設定する。未初期化の logger は対象にしない。
    """
    logger_dict = logging.Logger.manager.loggerDict
    for name, value in logger_dict.items():
        if name != _CORE_LOGGER_NAME and not name.startswith(f"{_CORE_LOGGER_NAME}."):
            continue
        if not isinstance(value, Logger):
            continue

        value.setLevel(level)
        value.propagate = False
        handlers = [
            handler
            for handler in value.handlers
            if getattr(handler, _HANDLER_MARKER, False)
        ]
        if not handlers:
            value.addHandler(_create_handler(stream, level))
            continue
        for handler in handlers:
            handler.setLevel(level)
            if isinstance(handler, logging.StreamHandler):
                handler.setStream(stream)
