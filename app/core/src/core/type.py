from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

type O3dColor = Annotated[npt.NDArray[np.float64], "Shape: (3, 1)"]
