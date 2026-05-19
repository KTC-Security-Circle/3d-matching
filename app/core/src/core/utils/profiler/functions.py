from collections.abc import Callable, Generator
from contextlib import contextmanager

from core.utils.profiler.profiler import Profiler


@contextmanager
def profile_block(name: str, *, track_memory: bool = False) -> Generator[None, None, None]:
    """コードブロックをプロファイリングするコンテキストマネージャー.

    Args:
        name: プロファイリングブロックの名前
        track_memory: メモリ使用量を追跡するかどうか

    Yields:
        None

    Example:
        with profile_block("data_loading"):
            data = load_large_dataset()
    """
    with Profiler(name, track_memory=track_memory):
        yield


# 便利な関数型インターフェース
def profile[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """関数をプロファイリングするデコレーター(簡易版).

    Args:
        func: プロファイリング対象の関数

    Returns:
        ラップされた関数

    Example:
        @profile
        def compute_ransac():
            # ... 処理 ...
    """
    return Profiler.profile(func)
