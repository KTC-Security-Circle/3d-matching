import numpy as np
import open3d as o3d
from open3d import pipelines

from ply import Ply


def global_registration(
    src: Ply,
    tgt: Ply,
    voxel_size: float,
    iteration: int = 30,
) -> pipelines.registration.RegistrationResult:
    dist_thresh = voxel_size * 1.5
    return pipelines.registration.registration_ransac_based_on_feature_matching(
        src.pcd_down,
        tgt.pcd_down,
        src.pcd_fpfh,
        tgt.pcd_fpfh,
        False,
        dist_thresh,
        pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
        ],
        pipelines.registration.RANSACConvergenceCriteria(iteration, 0.999),
    )


# === 以下を追加してください ===


def compute_feature_correspondences(
    src: Ply, tgt: Ply, mutual_filter: bool = False, noise_ratio: float = 0.0
) -> o3d.utility.Vector2iVector:
    """特徴量マッチングを行い、対応点リストを生成する（ノイズ混入機能付き）"""

    # 1. 通常のマッチング
    corres = pipelines.registration.correspondences_from_features(src.pcd_fpfh, tgt.pcd_fpfh, mutual_filter)
    corres_np = np.asarray(corres)

    # 2. ノイズ（デタラメな対応）を混ぜる処理
    # noise_ratio=10.0 なら、正解の10倍のゴミを混ぜる
    if noise_ratio > 0:
        n_original = len(corres_np)
        n_noise = int(n_original * noise_ratio)

        if n_noise > 0:
            # ランダムな点のペアを作成
            src_indices = np.random.randint(0, len(src.pcd_down.points), n_noise)
            tgt_indices = np.random.randint(0, len(tgt.pcd_down.points), n_noise)
            noise_corres = np.stack((src_indices, tgt_indices), axis=1)

            # 元の対応と結合
            corres_np = np.vstack((corres_np, noise_corres))
            # シャッフル
            np.random.shuffle(corres_np)

    return o3d.utility.Vector2iVector(corres_np)


def run_ransac_step(
    src: Ply,
    tgt: Ply,
    correspondences: o3d.utility.Vector2iVector,
    voxel_size: float,
    max_iteration: int = 1,  # ★ここが重要：デフォルト1回
) -> pipelines.registration.RegistrationResult:
    """対応点リストを用いてRANSACを1回だけ実行する"""
    dist_thresh = voxel_size * 1.5

    # correspondenceベースの高速な関数を使用
    return pipelines.registration.registration_ransac_based_on_correspondence(
        src.pcd_down,
        tgt.pcd_down,
        correspondences,
        dist_thresh,
        pipelines.registration.TransformationEstimationPointToPoint(False),
        3,  # 3点をサンプリング
        [
            pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
        ],
        pipelines.registration.RANSACConvergenceCriteria(max_iteration, 0.999),
    )


def evaluate_inlier_ratio(
    src: Ply, tgt: Ply, correspondences: o3d.utility.Vector2iVector, transform: np.ndarray, voxel_size: float
) -> float:
    """インライア率 w を計算するヘルパー"""
    dist_thresh = voxel_size * 1.5
    corres = np.asarray(correspondences)
    if len(corres) == 0:
        return 0.0

    src_points = np.asarray(src.pcd_down.points)
    tgt_points = np.asarray(tgt.pcd_down.points)

    # 変換適用
    p_src = src_points[corres[:, 0]]
    p_tgt = tgt_points[corres[:, 1]]
    p_src_transformed = (transform[:3, :3] @ p_src.T).T + transform[:3, 3]

    # 閾値以内の割合を計算
    dists = np.linalg.norm(p_src_transformed - p_tgt, axis=1)
    return np.sum(dists < dist_thresh) / len(corres)
