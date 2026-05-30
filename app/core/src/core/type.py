from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

type O3dColor = np.ndarray[np.float64[3, 1]]
