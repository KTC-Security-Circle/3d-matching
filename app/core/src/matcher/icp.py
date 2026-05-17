"""ICP (Iterative Closest Point) によるレジストレーション リファインメント モジュール。

RANSACで得た粗い変換行列を初期値として、
Point-to-Plane距離メトリックを用いたICPアルゴリズムで精密な位置合わせを行う。

Note:
    ICPは局所最適化アルゴリズムのため、良好な初期変換行列（通常はRANSACの結果）が必要。
    初期値が悪いと局所解に陥る可能性がある。
"""

from numpy import ndarray
from open3d import pipelines

from ply import Ply


def refine_registration(
    src: Ply,
    tgt: Ply,
    init_trans: ndarray,
    voxel_size: float,
) -> pipelines.registration.RegistrationResult:
    """ICPアルゴリズムによるレジストレーションのリファインメントを行う。

    RANSACで得た初期変換行列を起点に、Point-to-Plane距離メトリックを用いて
    ソース点群とターゲット点群の位置合わせを反復的に精密化する。

    Point-to-Planeは、Point-to-Point ICPより収束が速く精度が良い。
    ただし、点群に法線情報が必要（Plyクラスの前処理で計算済み）。

    Args:
        src: ソース点群（法線推定済みのPlyオブジェクト）
        tgt: ターゲット点群（法線推定済みのPlyオブジェクト）
        init_trans: 初期変換行列（4x4）。通常はRANSACの出力を指定
        voxel_size: ボクセルサイズ。距離閾値の算出に使用 (閾値 = voxel_size * 0.4)

    Returns:
        RegistrationResult: 精密化された変換行列とフィットネス値を含む結果
    """
    # ICPの距離閾値: ボクセルサイズの0.4倍（RANSACの1.5倍より厳しい基準で精密化）
    dist_thresh = voxel_size * 0.4
    return pipelines.registration.registration_icp(
        src.pcd,  # フル解像度のソース点群を使用（ダウンサンプルではない）
        tgt.pcd,  # フル解像度のターゲット点群
        dist_thresh,
        init_trans,  # RANSACで得た初期変換行列
        pipelines.registration.TransformationEstimationPointToPlane(),  # Point-to-Plane距離メトリック
    )
