"""RANSAC crash test suite for edge cases and numerical stability.

このスクリプトは、RANSACアルゴリズムが極端な条件下でもクラッシュせず、
数値的に安定していることを検証するテストスイートです。
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d

from matcher.ransac import (
    compute_feature_correspondences,
    compute_step_transformation,
    evaluate_inlier_ratio,
)
from ply import Ply
from utils.setup_logging import setup_logging

logger = setup_logging(__name__)

DATA_DIRECTORY = (Path(__file__).parent / "3d_data").resolve()


def create_minimal_point_cloud(num_points: int = 10) -> o3d.geometry.PointCloud:
    """最小限の点群を生成する。

    Args:
        num_points: 点数

    Returns:
        PointCloud オブジェクト
    """
    points = np.random.rand(num_points, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def create_collinear_point_cloud() -> o3d.geometry.PointCloud:
    """共線点群を生成する（全ての点が一直線上）。

    Returns:
        PointCloud オブジェクト
    """
    # Z軸上に配置された点
    points = np.array([[0, 0, i] for i in range(10)], dtype=float)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def create_coplanar_point_cloud() -> o3d.geometry.PointCloud:
    """共面点群を生成する（全ての点が一平面上）。

    Returns:
        PointCloud オブジェクト
    """
    # XY平面上に配置された点
    points = np.random.rand(10, 3)
    points[:, 2] = 0.0  # Z座標を全て0に
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def create_duplicate_point_cloud() -> o3d.geometry.PointCloud:
    """重複点を含む点群を生成する。

    Returns:
        PointCloud オブジェクト
    """
    # 同じ点を複数回含む
    points = np.array([[1, 1, 1]] * 10, dtype=float)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def test_minimal_correspondences():
    """最小限の対応点数でのテスト。"""
    logger.info("=== Test: Minimal Correspondences ===")

    try:
        # 3点のみの点群
        src_pcd = create_minimal_point_cloud(3)
        tgt_pcd = create_minimal_point_cloud(3)

        # Ply風のラッパーを作成
        class MockPly:
            def __init__(self, pcd):
                self.pcd = pcd
                self.pcd_down = pcd
                self.pcd_fpfh = None

        src = MockPly(src_pcd)
        tgt = MockPly(tgt_pcd)

        # 対応点を手動作成（最小の3ペア）
        correspondences = o3d.utility.Vector2iVector(np.array([[0, 0], [1, 1], [2, 2]]))

        # 変換行列を計算
        result = compute_step_transformation(src, tgt, correspondences)

        logger.info(f"  ✓ Success: Transformation computed for 3 correspondences")
        logger.info(f"    Fitness: {result.fitness}")

    except Exception as e:
        logger.error(f"  ✗ Failed: {type(e).__name__}: {e}")


def test_collinear_points():
    """共線点のテスト。"""
    logger.info("=== Test: Collinear Points ===")

    try:
        src_pcd = create_collinear_point_cloud()
        tgt_pcd = create_collinear_point_cloud()

        class MockPly:
            def __init__(self, pcd):
                self.pcd = pcd
                self.pcd_down = pcd
                self.pcd_fpfh = None

        src = MockPly(src_pcd)
        tgt = MockPly(tgt_pcd)

        correspondences = o3d.utility.Vector2iVector(np.array([[0, 0], [1, 1], [2, 2]]))

        result = compute_step_transformation(src, tgt, correspondences)

        logger.info(f"  ✓ Success: Handled collinear points")
        logger.info(f"    Fitness: {result.fitness}")

    except Exception as e:
        logger.error(f"  ✗ Failed: {type(e).__name__}: {e}")


def test_coplanar_points():
    """共面点のテスト。"""
    logger.info("=== Test: Coplanar Points ===")

    try:
        src_pcd = create_coplanar_point_cloud()
        tgt_pcd = create_coplanar_point_cloud()

        class MockPly:
            def __init__(self, pcd):
                self.pcd = pcd
                self.pcd_down = pcd
                self.pcd_fpfh = None

        src = MockPly(src_pcd)
        tgt = MockPly(tgt_pcd)

        correspondences = o3d.utility.Vector2iVector(np.array([[0, 0], [1, 1], [2, 2]]))

        result = compute_step_transformation(src, tgt, correspondences)

        logger.info(f"  ✓ Success: Handled coplanar points")
        logger.info(f"    Fitness: {result.fitness}")

    except Exception as e:
        logger.error(f"  ✗ Failed: {type(e).__name__}: {e}")


def test_duplicate_points():
    """重複点のテスト。"""
    logger.info("=== Test: Duplicate Points ===")

    try:
        src_pcd = create_duplicate_point_cloud()
        tgt_pcd = create_duplicate_point_cloud()

        class MockPly:
            def __init__(self, pcd):
                self.pcd = pcd
                self.pcd_down = pcd
                self.pcd_fpfh = None

        src = MockPly(src_pcd)
        tgt = MockPly(tgt_pcd)

        correspondences = o3d.utility.Vector2iVector(np.array([[0, 0], [1, 1], [2, 2]]))

        result = compute_step_transformation(src, tgt, correspondences)

        logger.info(f"  ✓ Success: Handled duplicate points")
        logger.info(f"    Fitness: {result.fitness}")

    except Exception as e:
        logger.error(f"  ✗ Failed: {type(e).__name__}: {e}")


def test_zero_correspondences():
    """対応点が0個の場合のテスト。"""
    logger.info("=== Test: Zero Correspondences ===")

    try:
        src_pcd = create_minimal_point_cloud(10)
        tgt_pcd = create_minimal_point_cloud(10)

        class MockPly:
            def __init__(self, pcd):
                self.pcd = pcd
                self.pcd_down = pcd
                self.pcd_fpfh = None

        src = MockPly(src_pcd)
        tgt = MockPly(tgt_pcd)

        correspondences = o3d.utility.Vector2iVector(np.array([]).reshape(0, 2))

        # インライア評価（0個の対応点）
        fitness = evaluate_inlier_ratio(src, tgt, correspondences, np.eye(4), 0.3)

        logger.info(f"  ✓ Success: Handled zero correspondences")
        logger.info(f"    Fitness: {fitness}")

    except Exception as e:
        logger.error(f"  ✗ Failed: {type(e).__name__}: {e}")


def test_extreme_noise_ratio(src_ply: Ply, tgt_ply: Ply):
    """極端なノイズ比率のテスト。"""
    logger.info("=== Test: Extreme Noise Ratio ===")

    for noise_ratio in [0.0, 1.0, 5.0, 10.0, 100.0]:
        try:
            corres = compute_feature_correspondences(src_ply, tgt_ply, noise_ratio=noise_ratio)
            logger.info(f"  ✓ Noise ratio {noise_ratio}: {len(corres)} correspondences")
        except Exception as e:
            logger.error(f"  ✗ Noise ratio {noise_ratio} failed: {type(e).__name__}: {e}")


def test_numerical_stability(src_ply: Ply, tgt_ply: Ply):
    """数値的安定性のテスト。"""
    logger.info("=== Test: Numerical Stability ===")

    try:
        # 1000回のランダムサンプリングで安定性をテスト
        corres = compute_feature_correspondences(src_ply, tgt_ply, noise_ratio=0.0)

        success_count = 0
        failure_count = 0

        for i in range(1000):
            try:
                result = compute_step_transformation(src_ply, tgt_ply, corres)
                fitness = evaluate_inlier_ratio(
                    src_ply, tgt_ply, corres, result.transformation, 0.3
                )
                if not np.isnan(fitness) and not np.isinf(fitness):
                    success_count += 1
                else:
                    failure_count += 1
            except Exception as e:
                failure_count += 1

        success_rate = (success_count / 1000) * 100
        logger.info(f"  Success rate: {success_rate:.1f}% ({success_count}/1000)")

        if success_rate >= 95.0:
            logger.info("  ✓ Numerical stability: GOOD")
        elif success_rate >= 80.0:
            logger.info("  ⚠ Numerical stability: MODERATE")
        else:
            logger.error("  ✗ Numerical stability: POOR")

    except Exception as e:
        logger.error(f"  ✗ Failed: {type(e).__name__}: {e}")


def test_large_transformation(src_ply: Ply, tgt_ply: Ply):
    """大きな変換行列のテスト。"""
    logger.info("=== Test: Large Transformation ===")

    try:
        # 極端に大きな変換
        large_transform = np.eye(4)
        large_transform[:3, :3] *= 1000.0  # スケーリング
        large_transform[:3, 3] = [1000, 1000, 1000]  # 大きな平行移動

        corres = compute_feature_correspondences(src_ply, tgt_ply, noise_ratio=0.0)
        fitness = evaluate_inlier_ratio(src_ply, tgt_ply, corres, large_transform, 0.3)

        logger.info(f"  ✓ Success: Handled large transformation")
        logger.info(f"    Fitness: {fitness}")

    except Exception as e:
        logger.error(f"  ✗ Failed: {type(e).__name__}: {e}")


def run_all_crash_tests(src_path: Path, tgt_path: Path):
    """全てのクラッシュテストを実行。

    Args:
        src_path: ソースPLYファイルのパス
        tgt_path: ターゲットPLYファイルのパス
    """
    logger.info("=" * 100)
    logger.info("RANSAC CRASH TEST SUITE")
    logger.info("=" * 100)

    # 実データを使用するテスト
    logger.info("\n--- Loading Real Data ---")
    src_ply = Ply(src_path)
    tgt_ply = Ply(tgt_path)
    logger.info(f"Source: {src_path.name} ({len(src_ply.pcd.points)} points)")
    logger.info(f"Target: {tgt_path.name} ({len(tgt_ply.pcd.points)} points)")

    # 合成データを使用するテスト
    logger.info("\n--- Synthetic Data Tests ---")
    test_minimal_correspondences()
    test_collinear_points()
    test_coplanar_points()
    test_duplicate_points()
    test_zero_correspondences()

    # 実データを使用するテスト
    logger.info("\n--- Real Data Tests ---")
    test_extreme_noise_ratio(src_ply, tgt_ply)
    test_numerical_stability(src_ply, tgt_ply)
    test_large_transformation(src_ply, tgt_ply)

    logger.info("\n=" * 100)
    logger.info("CRASH TEST SUITE COMPLETED")
    logger.info("=" * 100)


def main():
    """クラッシュテストスイートのエントリーポイント。"""
    parser = argparse.ArgumentParser(description="RANSAC crash test suite")
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

    args = parser.parse_args()

    src_path = DATA_DIRECTORY / args.source
    tgt_path = DATA_DIRECTORY / args.target

    if not src_path.exists():
        logger.error(f"Source file not found: {src_path}")
        return

    if not tgt_path.exists():
        logger.error(f"Target file not found: {tgt_path}")
        return

    run_all_crash_tests(src_path, tgt_path)


if __name__ == "__main__":
    main()
