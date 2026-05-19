from dataclasses import dataclass, field
from typing import Final


@dataclass
class TimingStats:
    """関数の実行時間統計を保存するクラス."""

    name: Final[str]
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    times: list[float] = field(default_factory=list)

    def add_time(self, elapsed: float) -> None:
        """実行時間の記録を追加する.

        Args:
            elapsed: 実行時間(秒)

        """
        self.call_count += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        self.times.append(elapsed)

    @property
    def avg_time(self) -> float:
        """平均実行時間を返す(秒)."""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0

    @property
    def median_time(self) -> float:
        """中央値の実行時間を返す(秒)."""
        if not self.times:
            return 0.0
        sorted_times = sorted(self.times)
        n = len(sorted_times)
        if n % 2 == 0:
            return (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2
        return sorted_times[n // 2]
