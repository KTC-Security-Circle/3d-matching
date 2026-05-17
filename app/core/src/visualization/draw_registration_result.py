"""レジストレーション結果の3D可視化モジュール。

ソース点群とターゲット点群を異なる色で表示し、
変換行列を適用した結果を視覚的に確認するためのユーティリティ。
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import open3d as o3d

if TYPE_CHECKING:
    from numpy import ndarray

    from ply import Ply


def draw_registration_result(source: Ply, target: Ply, transformation: ndarray) -> None:
    """レジストレーション結果を3Dビューアで可視化する。

    ソース点群に変換行列を適用し、ターゲット点群と重ねて表示する。
    色分けにより位置合わせの精度を視覚的に確認できる。

    Args:
        source: ソース点群（黄色で表示）
        target: ターゲット点群（シアンで表示）
        transformation: ソース点群に適用する4x4変換行列
    """
    # 元の点群を変更しないようディープコピーを作成
    source_temp = copy.deepcopy(source.pcd_down)
    target_temp = copy.deepcopy(target.pcd_down)

    # 点群に色を割り当てて区別しやすくする
    source_temp.paint_uniform_color([1, 0.706, 0])  # 黄色: ソース（変換対象）
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # シアン: ターゲット（基準）

    # ソース点群に変換行列を適用して位置合わせ結果を反映
    source_temp.transform(transformation)

    # Open3Dの3Dビューアで表示（カメラパラメータはプリセット値）
    o3d.visualization.draw_geometries(  # pyright: ignore[reportAttributeAccessIssue]
        [source_temp, target_temp],
        zoom=0.4559,
        front=[0.6452, -0.3036, -0.7011],
        lookat=[1.9892, 2.0208, 1.8945],
        up=[-0.2779, -0.9482, 0.1556],
    )
