"""
Validation utilities for MFG_PDE.

This module provides comprehensive validation for MFG problem inputs,
custom functions, arrays, and runtime solver outputs.

Modules:
    protocol: Core types (ValidationResult, ValidationError, ValidationIssue)
    components: IC/BC validation (m_initial, u_final)
    functions: Custom function validation (Hamiltonian, drift, running_cost)
    arrays: Array validation (dtype, shape, dimension)
    runtime: Runtime checks (NaN/Inf, bounds)

Usage:
    from mfg_pde.utils.validation import (
        validate_components,
        validate_custom_functions,
        check_finite,
    )

    # Validate components at problem construction
    result = validate_components(components, geometry)
    if not result.is_valid:
        raise ValidationError(result)

Note: For convergence/divergence monitoring during solving, use:
    from mfg_pde.utils.convergence import DistributionConvergenceMonitor

Issue #689: Validation module infrastructure
Parent: #685 (Comprehensive Input Validation Initiative)
"""

# Array validation (Issue #687)
from mfg_pde.utils.validation.arrays import (
    validate_array_dtype,
    validate_array_shape,
    validate_field_dimension,
    validate_field_shape,
    validate_finite,
    validate_non_negative,
)

# Components validation (Issue #679, #681-#684)
from mfg_pde.utils.validation.components import (
    detect_callable_signature,
    validate_components,
    validate_m_initial,
    validate_mass_normalization,
    validate_u_final,
)

# Custom function validation (Issue #686)
from mfg_pde.utils.validation.functions import (
    validate_custom_functions,
    validate_drift,
    validate_hamiltonian,
    validate_hamiltonian_consistency,
    validate_hamiltonian_derivative,
    validate_running_cost,
)
from mfg_pde.utils.validation.protocol import (
    ValidationError,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    Validator,
)

# Runtime validation (Issue #688)
# Note: For convergence/divergence monitors, use mfg_pde.utils.convergence
from mfg_pde.utils.validation.runtime import (
    check_bounds,
    check_finite,
    validate_solver_output,
)

__all__ = [
    # Protocol
    "ValidationResult",
    "ValidationError",
    "ValidationIssue",
    "ValidationSeverity",
    "Validator",
    # Components
    "validate_components",
    "validate_m_initial",
    "validate_u_final",
    "validate_mass_normalization",
    "detect_callable_signature",
    # Functions
    "validate_custom_functions",
    "validate_hamiltonian",
    "validate_hamiltonian_derivative",
    "validate_hamiltonian_consistency",
    "validate_drift",
    "validate_running_cost",
    # Arrays
    "validate_array_dtype",
    "validate_array_shape",
    "validate_field_shape",
    "validate_field_dimension",
    "validate_finite",
    "validate_non_negative",
    # Runtime
    "check_finite",
    "check_bounds",
    "validate_solver_output",
]
