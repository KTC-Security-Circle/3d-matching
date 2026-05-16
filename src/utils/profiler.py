"""Performance profiling utilities for RANSAC optimization.

このモジュールは、RANSACアルゴリズムの性能を測定・分析するためのプロファイリングツールを提供します。
関数レベルのタイミング測定、メモリ使用量の追跡、統計レポートの生成が可能です。
"""

import functools
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class TimingStats:
    """関数の実行時間統計を保存するクラス。"""

    name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    times: List[float] = field(default_factory=list)

    def add_time(self, elapsed: float) -> None:
        """実行時間の記録を追加する。

        Args:
            elapsed: 実行時間（秒）
        """
        self.call_count += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        self.times.append(elapsed)

    @property
    def avg_time(self) -> float:
        """平均実行時間を返す（秒）。"""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0

    @property
    def median_time(self) -> float:
        """中央値の実行時間を返す（秒）。"""
        if not self.times:
            return 0.0
        sorted_times = sorted(self.times)
        n = len(sorted_times)
        if n % 2 == 0:
            return (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2
        else:
            return sorted_times[n // 2]


class Profiler:
    """パフォーマンスプロファイリング用のクラス。

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
    _stats: Dict[str, TimingStats] = {}
    _memory_snapshots: List[tuple] = []

    def __init__(self, name: str, track_memory: bool = False):
        """プロファイラーを初期化する。

        Args:
            name: 測定対象の名前
            track_memory: メモリ使用量を追跡するかどうか
        """
        self.name = name
        self.track_memory = track_memory
        self.start_time: Optional[float] = None
        self.start_memory: Optional[float] = None

    def __enter__(self):
        """コンテキストマネージャーのエントリーポイント。"""
        self.start_time = time.perf_counter()
        if self.track_memory and HAS_PSUTIL:
            process = psutil.Process()
            self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了処理。"""
        elapsed = time.perf_counter() - self.start_time

        # 統計を更新
        if self.name not in self._stats:
            self._stats[self.name] = TimingStats(name=self.name)
        self._stats[self.name].add_time(elapsed)

        # メモリ使用量を記録
        if self.track_memory and HAS_PSUTIL and self.start_memory is not None:
            process = psutil.Process()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = end_memory - self.start_memory
            self._memory_snapshots.append((self.name, memory_delta))

    @staticmethod
    def profile(func: Callable) -> Callable:
        """関数をプロファイリングするデコレーター。

        Args:
            func: プロファイリング対象の関数

        Returns:
            ラップされた関数
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with Profiler(func.__name__):
                return func(*args, **kwargs)

        return wrapper

    @classmethod
    def get_stats(cls, name: str) -> Optional[TimingStats]:
        """特定の名前の統計情報を取得する。

        Args:
            name: 統計情報の名前

        Returns:
            TimingStats オブジェクト、存在しない場合は None
        """
        return cls._stats.get(name)

    @classmethod
    def report(cls, sort_by: str = "total") -> str:
        """プロファイリング結果のレポートを生成する。

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
            f"{'Name':<40} {'Calls':>8} {'Total (s)':>12} {'Avg (ms)':>12} {'Median (ms)':>12} {'Min (ms)':>12} {'Max (ms)':>12}"
        )
        lines.append("-" * 100)

        for stat in stats_list:
            lines.append(
                f"{stat.name:<40} {stat.call_count:>8} {stat.total_time:>12.4f} "
                f"{stat.avg_time * 1000:>12.2f} {stat.median_time * 1000:>12.2f} "
                f"{stat.min_time * 1000:>12.2f} {stat.max_time * 1000:>12.2f}"
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
        """プロファイリング結果を標準出力に表示する。

        Args:
            sort_by: ソート基準 ("total", "avg", "calls", "name")
        """
        print(cls.report(sort_by=sort_by))

    @classmethod
    def reset(cls) -> None:
        """全ての統計情報をリセットする。"""
        cls._stats.clear()
        cls._memory_snapshots.clear()

    @classmethod
    def save_report(cls, filepath: str, sort_by: str = "total") -> None:
        """プロファイリング結果をファイルに保存する。

        Args:
            filepath: 保存先のファイルパス
            sort_by: ソート基準 ("total", "avg", "calls", "name")
        """
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(cls.report(sort_by=sort_by))


@contextmanager
def profile_block(name: str, track_memory: bool = False):
    """コードブロックをプロファイリングするコンテキストマネージャー。

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
def profile(func: Callable) -> Callable:
    """関数をプロファイリングするデコレーター（簡易版）。

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
