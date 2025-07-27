from .base_hjb import BaseHJBSolver
from .hjb_fdm import HJBFDMSolver
from .hjb_semi_lagrangian import HJBSemiLagrangianSolver
from .hjb_gfdm import HJBGFDMSolver
from .hjb_gfdm_optimized import HJBGFDMOptimizedSolver
from .hjb_gfdm_smart_qp import (
    HJBGFDMQPSolver,
    HJBGFDMSmartQPSolver,
)  # New name + backward compatibility
from .hjb_gfdm_tuned_smart_qp import (
    HJBGFDMTunedQPSolver,
    HJBGFDMTunedSmartQPSolver,
)  # New name + backward compatibility

__all__ = [
    "BaseHJBSolver",
    "HJBFDMSolver",
    "HJBSemiLagrangianSolver",
    "HJBGFDMSolver",
    "HJBGFDMOptimizedSolver",
    # New standardized names
    "HJBGFDMQPSolver",
    "HJBGFDMTunedQPSolver",
    # Backward compatibility (deprecated)
    "HJBGFDMSmartQPSolver",
    "HJBGFDMTunedSmartQPSolver",
]
