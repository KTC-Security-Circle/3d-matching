"""RANSACベースのグローバルレジストレーション モジュール.

FPFH特徴量を用いた対応点マッチングとRANSACアルゴリズムにより、
2つの点群間の粗い位置合わせ(グローバルレジストレーション)を行う.

主な機能:
    - global_registration: Open3DのRANSACパイプラインによるレジストレーション
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import open3d as o3d

if TYPE_CHECKING:
    from core.matcher.type import RegistrationResult
    from core.ply import Ply


def global_registration(
    src: Ply,
    tgt: Ply,
    voxel_size: float,
    iteration: int = 30,
) -> RegistrationResult:
    """Open3DのRANSACパイプラインを使用してグローバルレジストレーションを実行する.

    FPFH特徴量に基づく対応点マッチングとRANSACにより、
    ソース点群をターゲット点群に合わせるための4x4変換行列を推定する.

    Args:
        src: ソース点群(前処理済みのPlyオブジェクト)
        tgt: ターゲット点群(前処理済みのPlyオブジェクト)
        voxel_size: ボクセルサイズ. 距離閾値の算出基準に使用 (閾値 = voxel_size * 1.5)
        iteration: RANSACの最大イテレーション数(デフォルト: 30)

    Returns:
        RegistrationResult: 変換行列(transformation)とフィットネス値を含む結果.
    """
    # 対応点の距離閾値: ボクセルサイズの1.5倍をインライア判定基準とする
    dist_thresh = voxel_size * 1.5

    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src.pcd_down,
        tgt.pcd_down,
        src.pcd_fpfh,
        tgt.pcd_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist_thresh,
        # スケーリングなしのPoint-to-Point推定
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
        # RANSACで使用するサンプル数(3点で剛体変換を推定)
        ransac_n=3,
        checkers=[
            # 対応点間のエッジ長の整合性チェック(比率0.9以上)
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            # 対応点間の距離チェック
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
        ],
        # 収束条件: 最大イテレーション数と信頼度 0.999
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(iteration, 0.999),
    )
