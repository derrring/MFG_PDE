from .base_hjb import BaseHJBSolver
from .hjb_fdm import HJBFDMSolver
from .hjb_gfdm import HJBGFDMSolver
from .hjb_semi_lagrangian import HJBSemiLagrangianSolver
from .hjb_weno import HJBWenoSolver

# Legacy compatibility - deprecated, use HJBWenoSolver with weno_variant="weno5"
from .hjb_weno5 import HJBWeno5Solver

__all__ = [
    "BaseHJBSolver",
    "HJBFDMSolver",
    "HJBGFDMSolver",
    "HJBSemiLagrangianSolver",
    "HJBWenoSolver",
    # Legacy - deprecated
    "HJBWeno5Solver",
]
