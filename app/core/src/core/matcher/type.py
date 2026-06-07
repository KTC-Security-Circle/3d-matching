from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Protocol

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


class RegistrationResult(Protocol):
    correspondence_set: Annotated[npt.NDArray[np.integer], "Shape: (N, 2)"]
    fitness: float
    inlier_rmse: float
    transformation: Annotated[npt.NDArray[np.float64], "Shape: (4, 4)"]
