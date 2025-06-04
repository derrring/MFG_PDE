# mfg_pde/utils/aux_func.py
import numpy as np
from typing import Union

# Original definition was -np.minimum(x,0), which is equivalent to np.maximum(-x,0)
# For consistency with ppart(x) = max(x,0), npart(x) = max(-x,0)
# If p_fwd = (U[i+1]-U[i])/Dx, then npart(p_fwd) is for flow to the right if p_fwd < 0.
# If p_bwd = (U[i]-U[i-1])/Dx, then ppart(p_bwd) is for flow to the left if p_bwd > 0.


def ppart(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Positive part of x.
    Works element-wise for NumPy arrays.
    """
    return np.maximum(x, 0.0)


def npart(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Negative part of x, defined as max(0, -x).
    This means (npart(x))^2 is equivalent to (x^-)^2 where x^- = min(0, x).
    Works element-wise for NumPy arrays.
    """
    return np.maximum(-x, 0.0)


if __name__ == "__main__":
    # Example usage:

    # Single numeric
    print(f"ppart(5.0) = {ppart(5.0)}")  # Expected: 5.0
    print(f"ppart(-3.0) = {ppart(-3.0)}")  # Expected: 0.0
    print(f"npart(5.0) = {npart(5.0)}")  # Expected: 0.0
    print(f"npart(-3.0) = {npart(-3.0)}")  # Expected: 3.0

    # NumPy array
    arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    print(f"\narr = {arr}")
    print(f"ppart(arr) = {ppart(arr)}")  # Expected: [0. 0. 0. 1. 2.]
    print(f"npart(arr) = {npart(arr)}")  # Expected: [2. 1. 0. 0. 0.]

    # Check square equivalence for npart
    # (x^-)^2
    x_minus_sq = np.minimum(arr, 0.0) ** 2
    # (npart(x))^2
    npart_x_sq = npart(arr) ** 2
    print(f"(min(arr,0))^2 = {x_minus_sq}")
    print(f"(npart(arr))^2 = {npart_x_sq}")
    assert np.allclose(x_minus_sq, npart_x_sq)

    # (x^+)^2
    x_plus_sq = np.maximum(arr, 0.0) ** 2
    # (ppart(x))^2
    ppart_x_sq = ppart(arr) ** 2
    print(f"(max(arr,0))^2 = {x_plus_sq}")
    print(f"(ppart(arr))^2 = {ppart_x_sq}")
    assert np.allclose(x_plus_sq, ppart_x_sq)
