"""RANSACベースのグローバルレジストレーション モジュール。

FPFH特徴量を用いた対応点マッチングとRANSACアルゴリズムにより、
2つの点群間の粗い位置合わせ（グローバルレジストレーション）を行う。

主な機能:
    - global_registration: Open3DのRANSACパイプラインによるレジストレーション
    - compute_feature_correspondences: FPFH特徴量による対応点計算（ノイズ混入機能付き）
    - compute_step_transformation: Kabschアルゴリズムによる変換行列推定（NumPy実装）
    - evaluate_inlier_ratio: 変換行列の品質をインライア率で評価
"""

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
    """Open3DのRANSACパイプラインを使用してグローバルレジストレーションを実行する。

    FPFH特徴量に基づく対応点マッチングとRANSACにより、
    ソース点群をターゲット点群に合わせるための4x4変換行列を推定する。

    Args:
        src: ソース点群（前処理済みのPlyオブジェクト）
        tgt: ターゲット点群（前処理済みのPlyオブジェクト）
        voxel_size: ボクセルサイズ。距離閾値の算出基準に使用 (閾値 = voxel_size * 1.5)
        iteration: RANSACの最大イテレーション数（デフォルト: 30）

    Returns:
        RegistrationResult: 変換行列（transformation）とフィットネス値を含む結果
    """
    # 対応点の距離閾値: ボクセルサイズの1.5倍をインライア判定基準とする
    dist_thresh = voxel_size * 1.5
    return pipelines.registration.registration_ransac_based_on_feature_matching(
        src.pcd_down,
        tgt.pcd_down,
        src.pcd_fpfh,
        tgt.pcd_fpfh,
        False,  # mutual_filter: 双方向フィルタを無効化
        dist_thresh,
        pipelines.registration.TransformationEstimationPointToPoint(False),  # スケーリングなしのPoint-to-Point推定
        3,  # RANSACで使用するサンプル数（3点で剛体変換を推定）
        [
            # 対応点間のエッジ長の整合性チェック（比率0.9以上）
            pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            # 対応点間の距離チェック（距離閾値以内）
            pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
        ],
        # 収束条件: 最大イテレーション数と信頼度 0.999
        pipelines.registration.RANSACConvergenceCriteria(iteration, 0.999),
    )


def compute_feature_correspondences(
    src: Ply,
    tgt: Ply,
    mutual_filter: bool = False,
    noise_ratio: float = 0.0,
) -> o3d.utility.Vector2iVector:
    """FPFH特徴量に基づく対応点リストを生成する（ノイズ混入機能付き）。

    特徴量空間での最近傍探索により対応点を求めた後、
    指定比率のランダムな偽対応点（ノイズ）を追加できる。
    ノイズ混入はRANSACのロバスト性テストに利用される。

    Args:
        src: ソース点群
        tgt: ターゲット点群
        mutual_filter: 双方向フィルタの有効化。Trueの場合、双方向で最近傍の対応のみ残す
        noise_ratio: ノイズ比率。元の対応点数に対する偽対応点の割合
                     (例: 2.0 → 元の2倍の偽対応点を追加)

    Returns:
        Vector2iVector: 対応点ペアのリスト。各行は [src_index, tgt_index]
    """
    # FPFH特徴量空間での最近傍探索による対応点計算
    corres = pipelines.registration.correspondences_from_features(src.pcd_fpfh, tgt.pcd_fpfh, mutual_filter)
    corres_np = np.asarray(corres)

    # ノイズ（ランダムな偽対応点）の混入
    if noise_ratio > 0:
        n_original = len(corres_np)
        n_noise = int(n_original * noise_ratio)
        if n_noise > 0:
            # ソース・ターゲットそれぞれからランダムなインデックスを生成
            src_indices = np.random.randint(0, len(src.pcd_down.points), n_noise)
            tgt_indices = np.random.randint(0, len(tgt.pcd_down.points), n_noise)
            noise_corres = np.stack((src_indices, tgt_indices), axis=1)
            # 元の対応点リストに偽対応点を結合し、シャッフルして混ぜる
            corres_np = np.vstack((corres_np, noise_corres))
            np.random.shuffle(corres_np)

    return o3d.utility.Vector2iVector(corres_np)


def compute_step_transformation(
    src: Ply,
    tgt: Ply,
    correspondences: o3d.utility.Vector2iVector,
) -> pipelines.registration.RegistrationResult:
    """対応点リストからランダムに3点を選び、Kabschアルゴリズムで変換行列を計算する。

    Open3Dの組み込み関数は縮退ケース（共線点など）でクラッシュする可能性があるため、
    NumPyのSVDを用いた安全な実装を使用している。

    Kabschアルゴリズムの手順:
        1. 3点を重複なしでランダムサンプリング
        2. 各点群の重心を計算し、重心を原点に移動
        3. 共分散行列 H = P^T @ Q を計算
        4. SVD分解から最適な回転行列 R を求める
        5. 平行移動ベクトル t = centroid_tgt - R @ centroid_src を計算

    Args:
        src: ソース点群
        tgt: ターゲット点群
        correspondences: 対応点ペアのリスト

    Returns:
        RegistrationResult: 推定された4x4変換行列を含む結果。
                           計算失敗時は単位行列（fitness=0.0）を返す。
    """
    corres_np = np.asarray(correspondences)
    n_corres = len(corres_np)

    # フォールバック用: 単位行列（変換なし）の結果を用意
    res = pipelines.registration.RegistrationResult()
    res.transformation = np.eye(4)
    res.fitness = 0.0

    # 対応点が3点未満の場合、変換を推定できないため単位行列を返す
    if n_corres < 3:
        return res

    # 重複なしで3つの対応点をランダム選択
    idxs = np.random.choice(n_corres, 3, replace=False)
    sample = corres_np[idxs]

    # 選択した対応点のソース座標・ターゲット座標を取得
    src_points = np.asarray(src.pcd_down.points)[sample[:, 0]]
    tgt_points = np.asarray(tgt.pcd_down.points)[sample[:, 1]]

    # === Kabschアルゴリズム（SVDベースの最適剛体変換推定） ===
    try:
        # 各点群の重心を計算
        centroid_src = np.mean(src_points, axis=0)
        centroid_tgt = np.mean(tgt_points, axis=0)

        # 重心を原点に移動（中心化）
        p = src_points - centroid_src
        q = tgt_points - centroid_tgt

        # 共分散行列: H = P^T @ Q （回転の推定に使用）
        H = np.dot(p.T, q)

        # 特異値分解 (SVD): H = U @ diag(S) @ Vt
        # 最適化: 3x3 行列には full_matrices=False で十分（メモリと計算量を削減）
        U, S, Vt = np.linalg.svd(H, full_matrices=False)

        # 最適回転行列: R = V @ U^T
        R = np.dot(Vt.T, U.T)

        # 反射の補正: det(R) < 0 の場合、鏡像反転が発生しているので修正
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)

        # 平行移動ベクトル: t = centroid_tgt - R @ centroid_src
        t = centroid_tgt - np.dot(R, centroid_src)

        # 4x4同次変換行列の組み立て
        trans = np.eye(4)
        trans[:3, :3] = R  # 左上3x3 = 回転行列
        trans[:3, 3] = t  # 右上3x1 = 平行移動ベクトル

        # 数値安定性チェック（NaN/Infが含まれる場合は単位行列にフォールバック）
        if np.isnan(trans).any() or np.isinf(trans).any():
            return res

        res.transformation = trans
        return res

    except Exception:
        # SVDが収束しない等のエラー時は単位行列を返してクラッシュを回避
        return res


def evaluate_inlier_ratio(
    src: Ply,
    tgt: Ply,
    correspondences: o3d.utility.Vector2iVector,
    transform: np.ndarray,
    voxel_size: float,
) -> float:
    """変換行列の品質をインライア率で評価する。

    ソース点群に変換行列を適用し、対応するターゲット点との距離が
    閾値（voxel_size * 1.5）以内となるペアの割合を計算する。
    インライア率が高いほど、変換行列の精度が良いことを示す。

    Args:
        src: ソース点群
        tgt: ターゲット点群
        correspondences: 対応点ペアのリスト
        transform: 評価対象の4x4変換行列
        voxel_size: ボクセルサイズ（距離閾値の算出に使用）

    Returns:
        float: インライア率（0.0〜1.0）。対応点がない場合は 0.0
    """
    dist_thresh = voxel_size * 1.5
    corres = np.asarray(correspondences)
    if len(corres) == 0:
        return 0.0

    src_points = np.asarray(src.pcd_down.points)
    tgt_points = np.asarray(tgt.pcd_down.points)

    p_src = src_points[corres[:, 0]]
    p_tgt = tgt_points[corres[:, 1]]

    # ソース点に変換行列を適用: p' = R @ p + t
    p_src_transformed = (transform[:3, :3] @ p_src.T).T + transform[:3, 3]

    # 変換後のソース点とターゲット点のユークリッド距離を計算
    dists = np.linalg.norm(p_src_transformed - p_tgt, axis=1)

    # インライア率 = 距離が閾値未満のペア数 / 全対応点数
    return np.sum(dists < dist_thresh) / len(corres)


def evaluate_inlier_ratio_fast(
    p_src: np.ndarray,
    p_tgt: np.ndarray,
    transform: np.ndarray,
    dist_thresh_sq: float,
) -> float:
    """最適化版: 変換行列の品質をインライア率で評価する（高速版）。

    事前に抽出された対応点を使用し、平方根計算を回避することで高速化。
    evaluate_inlier_ratio() と同じ結果を返すが、10,000回呼び出される場合に
    大幅な性能向上が期待できる。

    最適化ポイント:
    - 対応点の抽出を事前に1回だけ実行（ループ外）
    - 行列乗算を最適化: (R @ p.T).T → p @ R.T（転置削減）
    - 平方根計算を回避: norm(v) < thresh → sum(v^2) < thresh^2

    Args:
        p_src: 事前に抽出されたソース点の配列 (N, 3)
        p_tgt: 事前に抽出されたターゲット点の配列 (N, 3)
        transform: 評価対象の4x4変換行列
        dist_thresh_sq: 距離閾値の2乗（voxel_size * 1.5）^2

    Returns:
        float: インライア率（0.0〜1.0）。対応点がない場合は 0.0
    """
    if len(p_src) == 0:
        return 0.0

    # 最適化: p @ R.T + t は (R @ p.T).T + t と等価だが、転置が1回少ない
    R = transform[:3, :3]
    t = transform[:3, 3]
    p_src_transformed = p_src @ R.T + t

    # 最適化: 平方距離を使用して sqrt を回避
    dists_sq = np.sum((p_src_transformed - p_tgt) ** 2, axis=1)

    # インライア率 = 距離が閾値未満のペア数 / 全対応点数
    return np.sum(dists_sq < dist_thresh_sq) / len(p_src)
