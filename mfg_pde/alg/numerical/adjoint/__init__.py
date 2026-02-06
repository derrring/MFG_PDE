"""
Adjoint consistency utilities for numerical PDE solvers.

IMPORTANT (Issue #706): The matrix transpose approach (A_FP = A_HJB^T) is
mathematically incorrect for the gradient matrix. The correct approach is
**scheme pairing**:
- HJB: gradient_upwind → FP: divergence_upwind
- HJB: gradient_centered → FP: divergence_centered

The divergence schemes implement the correct Jacobian transpose structure
from Achdou's structure-preserving discretization.

This module provides:
1. **BC coupling**: State-dependent BC for HJB at reflecting boundaries (Issue #574)
2. **Scheme validation**: Verify correct HJB-FP scheme pairing
3. **Operators**: Matrix building utilities for custom solvers
4. **Verification**: Tools to check discrete relationships (updated criteria)

Usage:
------
    from mfg_pde.alg.numerical.adjoint import (
        # Scheme pairing validation (RECOMMENDED)
        validate_scheme_pairing,

        # BC coupling (for adjoint-consistent HJB BC at reflecting boundaries)
        create_adjoint_consistent_bc_1d,
        compute_adjoint_consistent_bc_values,

        # Verification
        verify_discrete_adjoint,
        verify_duality,

        # Diagnostics
        diagnose_adjoint_error,
        AdjointDiagnosticReport,
    )

    # Correct approach: validate scheme pairing
    is_valid, error = validate_scheme_pairing(hjb_solver, fp_solver)
    if not is_valid:
        raise ValueError(error)

References:
-----------
- Issue #574: Adjoint-consistent BC for reflecting boundaries
- Issue #704: Adjoint module redesign
- Issue #706: Deprecation of incorrect transpose approach
- docs/theory/adjoint_discretization_mfg.md: Mathematical foundations
"""

# BC coupling (state-dependent BC for boundary adjoint correction)
from .bc_coupling import (
    compute_adjoint_consistent_bc_values,
    compute_boundary_log_density_gradient_1d,
    # Backward compatibility alias
    compute_coupled_hjb_bc_values,
    create_adjoint_consistent_bc_1d,
)

# Diagnostics (error localization and recommendations)
from .diagnostics import (
    AdjointDiagnosticReport,
    BoundaryErrorInfo,
    ErrorSeverity,
    ErrorSource,
    diagnose_adjoint_error,
)

# Operators (adjoint-consistent operator construction)
from .operators import (
    OperatorConfig,
    OperatorGeometry,
    # Primitive implementations (for direct use)
    build_advection_matrix_1d,
    # Geometry-aware factories (preferred API)
    build_advection_matrix_from_geometry,
    # BC-aware adjoint
    build_bc_aware_adjoint_matrix,
    build_diffusion_matrix,
    build_diffusion_matrix_1d,
    build_diffusion_matrix_2d,
    build_diffusion_matrix_from_geometry,
    # Verification utilities
    check_operator_adjoint,
    get_boundary_indices_from_geometry,
    make_operator_adjoint,
    verify_operator_splitting_adjoint,
)

# Protocols (for solver integration)
from .protocols import (
    # DEPRECATED - kept for backward compatibility
    AdjointCapableFPSolver,
    AdjointCapableHJBSolver,
    validate_adjoint_capability,
    # NEW - correct approach
    validate_scheme_pairing,
)

# Verification (adjoint checking)
from .verification import (
    AdjointVerificationResult,
    DualityVerificationResult,
    compute_adjoint_error_profile,
    verify_discrete_adjoint,
    verify_duality,
)

__all__ = [
    # BC coupling
    "create_adjoint_consistent_bc_1d",
    "compute_adjoint_consistent_bc_values",
    "compute_boundary_log_density_gradient_1d",
    "compute_coupled_hjb_bc_values",  # Backward compat
    # Operators - Geometry-aware (preferred)
    "build_diffusion_matrix_from_geometry",
    "build_advection_matrix_from_geometry",
    "get_boundary_indices_from_geometry",
    "OperatorGeometry",
    # Operators - Primitive implementations
    "build_diffusion_matrix",
    "build_diffusion_matrix_1d",
    "build_diffusion_matrix_2d",
    "build_advection_matrix_1d",
    "check_operator_adjoint",
    "make_operator_adjoint",
    "verify_operator_splitting_adjoint",
    "OperatorConfig",
    "build_bc_aware_adjoint_matrix",
    # Verification
    "verify_discrete_adjoint",
    "verify_duality",
    "compute_adjoint_error_profile",
    "AdjointVerificationResult",
    "DualityVerificationResult",
    # Diagnostics
    "diagnose_adjoint_error",
    "AdjointDiagnosticReport",
    "BoundaryErrorInfo",
    "ErrorSource",
    "ErrorSeverity",
    # Protocols (deprecated, kept for backward compatibility)
    "AdjointCapableHJBSolver",
    "AdjointCapableFPSolver",
    "validate_adjoint_capability",
    # Protocols (new - correct approach)
    "validate_scheme_pairing",
]
