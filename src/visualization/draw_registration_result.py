from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import open3d as o3d

if TYPE_CHECKING:
    from numpy import ndarray

    from ply import Ply


def draw_registration_result(source: Ply, target: Ply, transformation: ndarray) -> None:
    source_temp = copy.deepcopy(source.pcd_down)
    target_temp = copy.deepcopy(target.pcd_down)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries(  # pyright: ignore[reportAttributeAccessIssue]
        [source_temp, target_temp],
        zoom=0.4559,
        front=[0.6452, -0.3036, -0.7011],
        lookat=[1.9892, 2.0208, 1.8945],
        up=[-0.2779, -0.9482, 0.1556],
    )
