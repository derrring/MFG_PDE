"""
Gradient notation compatibility layer.

This module provides utilities for converting between different gradient
notation formats used in MFG computations:
- Tuple notation: (dx, dy) derivatives
- p-values notation: p1, p2 values for 1D problems
- Gradient arrays: numpy arrays

.. deprecated:: 0.17.0
    All gradient_notation conversion functions are deprecated. Use
    ``DerivativeTensors`` from ``mfg_pde.types`` instead. The non-deprecated
    utilities (``check_derivs_format``, ``ensure_tuple_notation``) remain for
    legacy compatibility but have no production call sites.
"""

__all__ = [
    "check_derivs_format",
    "derivs_to_gradient_array",
    "derivs_to_p_values_1d",
    "ensure_tuple_notation",
    "gradient_array_to_derivs",
    "p_values_to_derivs_1d",
]

# Gradient notation utilities
from .gradient_notation import (
    check_derivs_format,
    derivs_to_gradient_array,
    derivs_to_p_values_1d,
    ensure_tuple_notation,
    gradient_array_to_derivs,
    p_values_to_derivs_1d,
)
