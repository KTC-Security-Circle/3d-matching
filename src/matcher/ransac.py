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


def compute_feature_correspondences(
    src: Ply, tgt: Ply, mutual_filter: bool = False, noise_ratio: float = 0.0
) -> o3d.utility.Vector2iVector:
    """特徴量マッチングを行い、対応点リストを生成する（ノイズ混入機能付き）"""
    corres = pipelines.registration.correspondences_from_features(src.pcd_fpfh, tgt.pcd_fpfh, mutual_filter)
    corres_np = np.asarray(corres)

    if noise_ratio > 0:
        n_original = len(corres_np)
        n_noise = int(n_original * noise_ratio)
        if n_noise > 0:
            src_indices = np.random.randint(0, len(src.pcd_down.points), n_noise)
            tgt_indices = np.random.randint(0, len(tgt.pcd_down.points), n_noise)
            noise_corres = np.stack((src_indices, tgt_indices), axis=1)
            corres_np = np.vstack((corres_np, noise_corres))
            np.random.shuffle(corres_np)

    return o3d.utility.Vector2iVector(corres_np)


# ★修正: クラッシュを防ぐため、NumPyで安全に行列計算を行う版
def compute_step_transformation(
    src: Ply, tgt: Ply, correspondences: o3d.utility.Vector2iVector
) -> pipelines.registration.RegistrationResult:
    """
    対応点リストからランダムに3点を選び、変換行列を計算して返す。
    Open3Dの関数だと縮退ケースでクラッシュするため、NumPyで実装する。
    """
    corres_np = np.asarray(correspondences)
    n_corres = len(corres_np)

    # デフォルトのIdentity結果を用意
    res = pipelines.registration.RegistrationResult()
    res.transformation = np.eye(4)
    res.fitness = 0.0

    if n_corres < 3:
        return res

    # 重複なしで3つのインデックスを選ぶ
    idxs = np.random.choice(n_corres, 3, replace=False)
    sample = corres_np[idxs]

    # ソース点とターゲット点を取得
    src_points = np.asarray(src.pcd_down.points)[sample[:, 0]]
    tgt_points = np.asarray(tgt.pcd_down.points)[sample[:, 1]]

    # === NumPyによる Kabsch Algorithm 実装 ===
    try:
        # 重心計算
        centroid_src = np.mean(src_points, axis=0)
        centroid_tgt = np.mean(tgt_points, axis=0)

        # 重心を原点に移動
        p = src_points - centroid_src
        q = tgt_points - centroid_tgt

        # 共分散行列 H = P^T @ Q
        H = np.dot(p.T, q)

        # SVD特異値分解
        U, S, Vt = np.linalg.svd(H)

        # 回転行列 R = V @ U^T
        R = np.dot(Vt.T, U.T)

        # 反射の補正 (行列式が負の場合)
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)

        # 平行移動 t = centroid_tgt - R @ centroid_src
        t = centroid_tgt - np.dot(R, centroid_src)

        # 4x4 行列作成
        trans = np.eye(4)
        trans[:3, :3] = R
        trans[:3, 3] = t

        # NaNチェック (万が一計算が不安定だった場合)
        if np.isnan(trans).any() or np.isinf(trans).any():
            return res

        res.transformation = trans
        return res

    except Exception:
        # SVDが収束しない等のエラー時は Identity を返してクラッシュ回避
        return res


def evaluate_inlier_ratio(
    src: Ply, tgt: Ply, correspondences: o3d.utility.Vector2iVector, transform: np.ndarray, voxel_size: float
) -> float:
    """インライア率 w を計算する"""
    dist_thresh = voxel_size * 1.5
    corres = np.asarray(correspondences)
    if len(corres) == 0:
        return 0.0

    src_points = np.asarray(src.pcd_down.points)
    tgt_points = np.asarray(tgt.pcd_down.points)

    p_src = src_points[corres[:, 0]]
    p_tgt = tgt_points[corres[:, 1]]

    # 変換適用
    p_src_transformed = (transform[:3, :3] @ p_src.T).T + transform[:3, 3]

    # 距離計算
    dists = np.linalg.norm(p_src_transformed - p_tgt, axis=1)

    # インライア率
    return np.sum(dists < dist_thresh) / len(corres)
