import functools
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Self

import psutil

from core.utils.profiler.timing_stats import TimingStats
from core.utils.setup_logging import setup_logging

logger = setup_logging(__name__)


class Profiler:
    """パフォーマンスプロファイリング用のクラス.

    使用方法:

    1. コンテキストマネージャーとして:
        with Profiler("my_operation") as prof:
            # ... 処理 ...

    2. デコレーターとして:
        @Profiler.profile
        def my_function():
            # ... 処理 ...

    3. グローバルレポート:
        Profiler.report()  # 全ての測定結果を表示
        Profiler.reset()   # 統計をリセット
    """

    # グローバルな統計情報を保存
    _stats: ClassVar[dict[str, TimingStats]] = {}
    _memory_snapshots: ClassVar[list[tuple]] = []

    def __init__(self, name: str, *, track_memory: bool = False) -> None:
        """プロファイラーを初期化する.

        Args:
            name: 測定対象の名前
            track_memory: メモリ使用量を追跡するかどうか
        """
        self.name = name
        self.track_memory = track_memory
        self.start_time: float | None = None
        self.start_memory: float | None = None

    def __enter__(self) -> Self:
        """コンテキストマネージャーのエントリーポイント."""
        self.start_time = time.perf_counter()
        if self.track_memory:
            process = psutil.Process()
            self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """コンテキストマネージャーの終了処理."""
        _ = exc_type, exc_val, exc_tb
        if self.start_time is None:
            msg = "Profiler was not properly started."
            raise RuntimeError(msg)
        elapsed = time.perf_counter() - self.start_time

        # 統計を更新
        if self.name not in self._stats:
            self._stats[self.name] = TimingStats(name=self.name)
        self._stats[self.name].add_time(elapsed)

        # メモリ使用量を記録
        if self.track_memory and self.start_memory is not None:
            process = psutil.Process()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = end_memory - self.start_memory
            self._memory_snapshots.append((self.name, memory_delta))

    @staticmethod
    def profile[**P, R](func: Callable[P, R]) -> Callable[P, R]:
        """関数をプロファイリングするデコレーター.

        Args:
            func: プロファイリング対象の関数

        Returns:
            ラップされた関数
        """
        func_name = getattr(func, "__name__", type(func).__name__)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with Profiler(func_name):
                return func(*args, **kwargs)

        return wrapper

    @classmethod
    def get_stats(cls, name: str) -> TimingStats | None:
        """特定の名前の統計情報を取得する.

        Args:
            name: 統計情報の名前

        Returns:
            TimingStats オブジェクト、存在しない場合は None
        """
        return cls._stats.get(name)

    @classmethod
    def report(cls, sort_by: str = "total") -> str:
        """プロファイリング結果のレポートを生成する.

        Args:
            sort_by: ソート基準 ("total", "avg", "calls", "name")

        Returns:
            フォーマットされたレポート文字列
        """
        if not cls._stats:
            return "No profiling data available."

        # ソート
        stats_list = list(cls._stats.values())
        if sort_by == "total":
            stats_list.sort(key=lambda s: s.total_time, reverse=True)
        elif sort_by == "avg":
            stats_list.sort(key=lambda s: s.avg_time, reverse=True)
        elif sort_by == "calls":
            stats_list.sort(key=lambda s: s.call_count, reverse=True)
        elif sort_by == "name":
            stats_list.sort(key=lambda s: s.name)

        # レポート生成
        lines = []
        lines.append("=" * 100)
        lines.append("PROFILING REPORT")
        lines.append("=" * 100)
        lines.append(
            f"{'Name':<40} {'Calls':>8} {'Total (s)':>12} {'Avg (ms)':>12}"
            f"{'Median (ms)':>12} {'Min (ms)':>12} {'Max (ms)':>12}",
        )
        lines.append("-" * 100)

        lines.extend(
            [
                f"{stat.name:<40} {stat.call_count:>8} {stat.total_time:>12.4f} "
                f"{stat.avg_time * 1000:>12.2f} {stat.median_time * 1000:>12.2f}"
                f"{stat.min_time * 1000:>12.2f} {stat.max_time * 1000:>12.2f}"
                for stat in stats_list
            ],
        )

        # 合計
        total_time = sum(s.total_time for s in stats_list)
        lines.append("-" * 100)
        lines.append(f"{'TOTAL':<40} {'':<8} {total_time:>12.4f}")

        # メモリ情報
        if cls._memory_snapshots:
            lines.append("")
            lines.append("=" * 100)
            lines.append("MEMORY USAGE")
            lines.append("=" * 100)
            lines.append(f"{'Name':<40} {'Delta (MB)':>20}")
            lines.append("-" * 100)

            memory_by_name = defaultdict(list)
            for name, delta in cls._memory_snapshots:
                memory_by_name[name].append(delta)

            for name, deltas in sorted(memory_by_name.items()):
                avg_delta = sum(deltas) / len(deltas)
                lines.append(f"{name:<40} {avg_delta:>20.2f}")

        lines.append("=" * 100)
        return "\n".join(lines)

    @classmethod
    def print_report(cls, sort_by: str = "total") -> None:
        """プロファイリング結果を標準出力に表示する.

        Args:
            sort_by: ソート基準 ("total", "avg", "calls", "name")
        """
        logger.info(cls.report(sort_by=sort_by))

    @classmethod
    def reset(cls) -> None:
        """全ての統計情報をリセットする."""
        cls._stats.clear()
        cls._memory_snapshots.clear()

    @classmethod
    def save_report(cls, filepath: str, sort_by: str = "total") -> None:
        """プロファイリング結果をファイルに保存する.

        Args:
            filepath: 保存先のファイルパス
            sort_by: ソート基準 ("total", "avg", "calls", "name")
        """
        with Path(filepath).open("w", encoding="utf-8") as f:
            f.write(cls.report(sort_by=sort_by))
