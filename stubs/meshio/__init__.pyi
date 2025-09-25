# Minimal meshio stub for MFG_PDE strategic typing
# Focus on functions actually used in geometry modules

from typing import Any

import numpy as np
from numpy.typing import NDArray

# Core meshio classes
class Mesh:
    def __init__(self, points: NDArray[np.floating], cells: list[Any], **kwargs: Any) -> None: ...
    def write(self, filename: str, **kwargs: Any) -> None: ...

# Mesh I/O functions
def read(filename: str, **kwargs: Any) -> Mesh: ...
def write(filename: str, mesh: Mesh, **kwargs: Any) -> None: ...

# Catch-all for any missing meshio attributes
def __getattr__(name: str) -> Any: ...
