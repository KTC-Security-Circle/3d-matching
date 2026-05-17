"""RANSAC performance benchmarking suite.

このスクリプトは、RANSACアルゴリズムの性能を測定し、ボトルネックを特定するためのベンチマークを実行します。
最適化前のベースライン性能を記録し、最適化後の改善を定量的に評価できます。
"""

import argparse
import copy
import time
from pathlib import Path
from typing import Dict

import numpy as np

from matcher.ransac import (
    compute_feature_correspondences,
    compute_step_transformation,
    evaluate_inlier_ratio,
    global_registration,
)
from ply import Ply
from src.utils.profiler import Profiler, profile_block
from utils.setup_logging import setup_logging

logger = setup_logging(__name__)

#  プロジェクトルート直下の 3d_data/ ディレクトリを参照
DATA_DIRECTORY = (Path(__file__).parent / "3d_data").resolve()


def benchmark_preprocessing(src_path: Path, tgt_path: Path) -> Dict[str, float]:
    """前処理（PLY読み込み、ダウンサンプル、FPFH計算）のベンチマーク。

    Args:
        src_path: ソースPLYファイルのパス
        tgt_path: ターゲットPLYファイルのパス

    Returns:
        各処理の実行時間を含む辞書
    """
    logger.info("=== Benchmarking Preprocessing ===")
    timings = {}

    # PLY読み込み
    with profile_block("ply_loading"):
        src_ply = Ply(src_path)
        tgt_ply = Ply(tgt_path)

    timings["ply_loading"] = Profiler.get_stats("ply_loading").total_time

    # 点数を記録
    src_points = len(src_ply.pcd.points)
    tgt_points = len(tgt_ply.pcd.points)
    src_down_points = len(src_ply.pcd_down.points)
    tgt_down_points = len(tgt_ply.pcd_down.points)

    logger.info(f"  Source points: {src_points:,} → {src_down_points:,} (downsampled)")
    logger.info(f"  Target points: {tgt_points:,} → {tgt_down_points:,} (downsampled)")

    return timings, src_ply, tgt_ply


def benchmark_correspondence_computation(src_ply: Ply, tgt_ply: Ply, noise_ratio: float = 0.0) -> Dict[str, float]:
    """対応点計算のベンチマーク。

    Args:
        src_ply: ソース点群
        tgt_ply: ターゲット点群
        noise_ratio: ノイズ比率

    Returns:
        実行時間と対応点数を含む辞書
    """
    logger.info("=== Benchmarking Correspondence Computation ===")

    with profile_block("correspondence_computation"):
        corres = compute_feature_correspondences(src_ply, tgt_ply, noise_ratio=noise_ratio)

    timings = {"correspondence_computation": Profiler.get_stats("correspondence_computation").total_time}

    logger.info(f"  Correspondences: {len(corres):,}")
    logger.info(f"  Time: {timings['correspondence_computation']:.4f}s")

    return timings, corres


def benchmark_ransac_iteration(
    src_ply: Ply, tgt_ply: Ply, correspondences, voxel_size: float, num_iterations: int = 100
) -> Dict[str, float]:
    """RANSACイテレーションのベンチマーク。

    Args:
        src_ply: ソース点群
        tgt_ply: ターゲット点群
        correspondences: 対応点リスト
        voxel_size: ボクセルサイズ
        num_iterations: 実行するイテレーション数

    Returns:
        各処理の実行時間を含む辞書
    """
    logger.info(f"=== Benchmarking RANSAC Iterations (n={num_iterations}) ===")

    # 個別処理のプロファイリング
    for i in range(num_iterations):
        with profile_block("ransac_iteration"):
            # Step 1: ランダムサンプリングと変換推定
            with profile_block("compute_transformation"):
                result = compute_step_transformation(src_ply, tgt_ply, correspondences)

            # Step 2: インライア率評価
            with profile_block("evaluate_inliers"):
                fitness = evaluate_inlier_ratio(src_ply, tgt_ply, correspondences, result.transformation, voxel_size)

    timings = {
        "ransac_iteration": Profiler.get_stats("ransac_iteration").avg_time,
        "compute_transformation": Profiler.get_stats("compute_transformation").avg_time,
        "evaluate_inliers": Profiler.get_stats("evaluate_inliers").avg_time,
    }

    logger.info(f"  Average iteration time: {timings['ransac_iteration'] * 1000:.2f}ms")
    logger.info(f"    - Transformation: {timings['compute_transformation'] * 1000:.2f}ms")
    logger.info(f"    - Inlier evaluation: {timings['evaluate_inliers'] * 1000:.2f}ms")

    return timings


def benchmark_deep_copy(src_ply: Ply, num_copies: int = 100) -> Dict[str, float]:
    """deep copyのベンチマーク。

    Args:
        src_ply: コピー対象の点群
        num_copies: コピー回数

    Returns:
        実行時間を含む辞書
    """
    logger.info(f"=== Benchmarking Deep Copy (n={num_copies}) ===")

    for i in range(num_copies):
        with profile_block("deep_copy"):
            _ = copy.deepcopy(src_ply.pcd)

    timings = {"deep_copy": Profiler.get_stats("deep_copy").avg_time}

    logger.info(f"  Average deep copy time: {timings['deep_copy'] * 1000:.2f}ms")

    return timings


def benchmark_sleep(sleep_time: float = 0.03, num_sleeps: int = 100) -> Dict[str, float]:
    """sleepのベンチマーク。

    Args:
        sleep_time: スリープ時間（秒）
        num_sleeps: スリープ回数

    Returns:
        実行時間を含む辞書
    """
    logger.info(f"=== Benchmarking Sleep (time={sleep_time}s, n={num_sleeps}) ===")

    total_time = 0.0
    for i in range(num_sleeps):
        start = time.perf_counter()
        time.sleep(sleep_time)
        total_time += time.perf_counter() - start

    avg_time = total_time / num_sleeps

    logger.info(f"  Average sleep time: {avg_time * 1000:.2f}ms (target: {sleep_time * 1000:.2f}ms)")
    logger.info(f"  Total sleep time: {total_time:.2f}s")

    return {"sleep": avg_time}


def benchmark_full_ransac(
    src_ply: Ply, tgt_ply: Ply, voxel_size: float = 0.3, num_iterations: int = 30
) -> Dict[str, float]:
    """フルRANSACパイプラインのベンチマーク（Open3D組み込み版）。

    Args:
        src_ply: ソース点群
        tgt_ply: ターゲット点群
        voxel_size: ボクセルサイズ
        num_iterations: RANSACイテレーション数

    Returns:
        実行時間を含む辞書
    """
    logger.info(f"=== Benchmarking Full RANSAC Pipeline (iterations={num_iterations}) ===")

    with profile_block("full_ransac"):
        result = global_registration(src_ply, tgt_ply, voxel_size=voxel_size, iteration=num_iterations)

    timings = {"full_ransac": Profiler.get_stats("full_ransac").total_time}

    logger.info(f"  Total time: {timings['full_ransac']:.4f}s")
    logger.info(f"  Fitness: {result.fitness:.4f}")
    logger.info(f"  RMSE: {result.inlier_rmse:.4f}")

    return timings


def estimate_10k_iteration_time(avg_iteration_time: float, sleep_time: float = 0.03) -> None:
    """10,000イテレーションRANSACの推定実行時間を計算。

    Args:
        avg_iteration_time: 1イテレーションあたりの平均実行時間（秒）
        sleep_time: スリープ時間（秒）
    """
    compute_time_10k = avg_iteration_time * 10000
    sleep_time_10k = sleep_time * 10000
    total_time_10k = compute_time_10k + sleep_time_10k

    logger.info("\n=== Estimated Time for 10,000 Iterations ===")
    logger.info(f"  Compute time: {compute_time_10k:.2f}s ({compute_time_10k / 60:.1f}min)")
    logger.info(f"  Sleep time: {sleep_time_10k:.2f}s ({sleep_time_10k / 60:.1f}min)")
    logger.info(f"  Total time: {total_time_10k:.2f}s ({total_time_10k / 60:.1f}min)")
    logger.info(f"  Sleep overhead: {(sleep_time_10k / total_time_10k) * 100:.1f}%")


def run_comprehensive_benchmark(
    src_path: Path,
    tgt_path: Path,
    voxel_size: float = 0.3,
    noise_ratio: float = 0.0,
    test_iterations: int = 100,
    full_ransac_iterations: int = 30,
) -> None:
    """包括的なベンチマークスイートを実行。

    Args:
        src_path: ソースPLYファイルのパス
        tgt_path: ターゲットPLYファイルのパス
        voxel_size: ボクセルサイズ
        noise_ratio: ノイズ比率
        test_iterations: テスト用イテレーション数
        full_ransac_iterations: フルRANSACのイテレーション数
    """
    logger.info("=" * 100)
    logger.info("RANSAC PERFORMANCE BENCHMARK SUITE")
    logger.info("=" * 100)
    logger.info(f"Source: {src_path.name}")
    logger.info(f"Target: {tgt_path.name}")
    logger.info(f"Voxel size: {voxel_size}")
    logger.info(f"Noise ratio: {noise_ratio}")
    logger.info("=" * 100)

    Profiler.reset()

    # Phase 1: 前処理
    _, src_ply, tgt_ply = benchmark_preprocessing(src_path, tgt_path)

    # Phase 2: 対応点計算
    _, corres = benchmark_correspondence_computation(src_ply, tgt_ply, noise_ratio)

    # Phase 3: RANSACイテレーション
    iter_timings = benchmark_ransac_iteration(src_ply, tgt_ply, corres, voxel_size, test_iterations)

    # Phase 4: Deep copy
    copy_timings = benchmark_deep_copy(src_ply, test_iterations)

    # Phase 5: Sleep
    sleep_timings = benchmark_sleep(0.03, test_iterations)

    # Phase 6: フルRANSACパイプライン
    full_timings = benchmark_full_ransac(src_ply, tgt_ply, voxel_size, full_ransac_iterations)

    # 10,000イテレーションの推定時間
    estimate_10k_iteration_time(iter_timings["ransac_iteration"], sleep_timings["sleep"])

    # 完全なレポートを表示
    logger.info("\n")
    Profiler.print_report(sort_by="total")

    # レポートをファイルに保存
    report_path = Path(__file__).parent / "benchmark_results.txt"
    Profiler.save_report(report_path)
    logger.info(f"\nBenchmark report saved to: {report_path}")


def main():
    """ベンチマークスイートのエントリーポイント。"""
    parser = argparse.ArgumentParser(description="RANSAC performance benchmark suite")
    parser.add_argument(
        "--source",
        type=str,
        default="sample.ply",
        help="Source PLY file name (default: sample.ply)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="target.ply",
        help="Target PLY file name (default: target.ply)",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.3,
        help="Voxel size for downsampling (default: 0.3)",
    )
    parser.add_argument(
        "--noise-ratio",
        type=float,
        default=0.0,
        help="Noise ratio for correspondence (default: 0.0)",
    )
    parser.add_argument(
        "--test-iterations",
        type=int,
        default=100,
        help="Number of iterations for testing (default: 100)",
    )
    parser.add_argument(
        "--ransac-iterations",
        type=int,
        default=30,
        help="Number of RANSAC iterations for full pipeline (default: 30)",
    )

    args = parser.parse_args()

    src_path = DATA_DIRECTORY / args.source
    tgt_path = DATA_DIRECTORY / args.target

    if not src_path.exists():
        logger.error(f"Source file not found: {src_path}")
        return

    if not tgt_path.exists():
        logger.error(f"Target file not found: {tgt_path}")
        return

    run_comprehensive_benchmark(
        src_path,
        tgt_path,
        args.voxel_size,
        args.noise_ratio,
        args.test_iterations,
        args.ransac_iterations,
    )


if __name__ == "__main__":
    main()
