"""ロギングのセットアップユーティリティ。

アプリケーション全体で統一されたフォーマットのロガーを生成する。
各モジュールで `setup_logging(__name__)` のように呼び出して使用する。

出力フォーマット例:
    2024-01-15 12:34:56 - ply.ply - INFO - Successfully loaded and preprocessed ply file: ...
"""

import logging
from logging import Logger


def setup_logging(name: str) -> Logger:
    """指定された名前のロガーを作成・設定して返す。

    INFOレベル以上のログをコンソール（stderr）に出力するハンドラを設定する。
    既にハンドラが設定済みの場合はハンドラの重複追加を防止する。

    Args:
        name: ロガー名。通常は `__name__` を渡してモジュール名を設定する

    Returns:
        設定済みのLoggerインスタンス
    """
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)

    # ハンドラの重複追加を防止（モジュールが複数回インポートされた場合など）
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger
