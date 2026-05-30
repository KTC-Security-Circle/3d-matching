from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Protocol, Self

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from core.type import O3dColor


class KDTreeSearchParam:
    def __init__(
        self,
        *,
        radius: float | None = None,
        max_nn: int | None = None,
    ) -> None: ...


class PointCloud(Protocol):
    points: Annotated[npt.NDArray[np.float64], "Shape: (num_points, 3)"]

    def has_points(self) -> bool: ...

    def voxel_down_sample(self, voxel_size: float) -> Self: ...

    def estimate_normals(
        self,
        search_param: KDTreeSearchParam,
        *,
        fast_normal_computation: bool = True,
    ) -> None: ...

    def paint_uniform_color(self, color: O3dColor) -> None: ...

    def transform(self, arg0: Annotated[npt.NDArray[np.float64], "Shape: (4, 4)"]) -> Self: ...


class Feature(Protocol):
    pass
