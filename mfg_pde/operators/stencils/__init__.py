"""
Finite Difference Stencils for MFG_PDE.

This module provides low-level stencil implementations with fixed coefficients:
    - Central differences (2nd-order)
    - Upwind schemes (1st-order, directional)
    - One-sided differences (boundary handling)

Stencils are the building blocks used by differential operators.
They provide fixed-coefficient differentiation formulas.

Note: For adaptive reconstruction strategies (WENO, ENO), see the
`mfg_pde.operators.reconstruction` module.

TODO: Extract stencils from tensor_calculus.py into this module.
"""

__all__: list[str] = []
