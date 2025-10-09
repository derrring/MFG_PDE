"""
Unit tests for 1D Boundary Conditions.

Tests the BoundaryConditions class and factory functions for 1D MFG problems,
including initialization, validation, type checking, matrix sizing, and edge cases.
"""

import pytest

from mfg_pde.geometry.boundary_conditions_1d import (
    BoundaryConditions,
    dirichlet_bc,
    neumann_bc,
    no_flux_bc,
    periodic_bc,
    robin_bc,
)

# ============================================================================
# Test: BoundaryConditions Class - Initialization
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_periodic_initialization():
    """Test periodic boundary condition initialization."""
    bc = BoundaryConditions(type="periodic")
    assert bc.type == "periodic"
    assert bc.left_value is None
    assert bc.right_value is None


@pytest.mark.unit
@pytest.mark.fast
def test_dirichlet_initialization():
    """Test Dirichlet boundary condition initialization."""
    bc = BoundaryConditions(type="dirichlet", left_value=0.0, right_value=1.0)
    assert bc.type == "dirichlet"
    assert bc.left_value == 0.0
    assert bc.right_value == 1.0


@pytest.mark.unit
@pytest.mark.fast
def test_neumann_initialization():
    """Test Neumann boundary condition initialization."""
    bc = BoundaryConditions(type="neumann", left_value=0.5, right_value=-0.5)
    assert bc.type == "neumann"
    assert bc.left_value == 0.5
    assert bc.right_value == -0.5


@pytest.mark.unit
@pytest.mark.fast
def test_no_flux_initialization():
    """Test no-flux boundary condition initialization."""
    bc = BoundaryConditions(type="no_flux")
    assert bc.type == "no_flux"
    assert bc.left_value is None
    assert bc.right_value is None


@pytest.mark.unit
@pytest.mark.fast
def test_robin_initialization():
    """Test Robin boundary condition initialization."""
    bc = BoundaryConditions(
        type="robin",
        left_alpha=1.0,
        left_beta=0.5,
        left_value=0.0,
        right_alpha=1.0,
        right_beta=0.5,
        right_value=1.0,
    )
    assert bc.type == "robin"
    assert bc.left_alpha == 1.0
    assert bc.left_beta == 0.5
    assert bc.left_value == 0.0
    assert bc.right_alpha == 1.0
    assert bc.right_beta == 0.5
    assert bc.right_value == 1.0


@pytest.mark.unit
@pytest.mark.fast
def test_missing_robin_coefficients_raises():
    """Test Robin BC without required coefficients raises ValueError."""
    with pytest.raises(ValueError, match="Robin boundary conditions require alpha and beta"):
        BoundaryConditions(type="robin", left_value=0.0, right_value=1.0)

    with pytest.raises(ValueError, match="Robin boundary conditions require alpha and beta"):
        BoundaryConditions(type="robin", left_alpha=1.0, right_alpha=1.0)


@pytest.mark.unit
@pytest.mark.fast
def test_default_values():
    """Test default values for optional parameters."""
    bc = BoundaryConditions(type="periodic")
    assert bc.left_value is None
    assert bc.right_value is None
    assert bc.left_alpha is None
    assert bc.left_beta is None
    assert bc.right_alpha is None
    assert bc.right_beta is None


# ============================================================================
# Test: Type Checking Methods
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_is_periodic():
    """Test is_periodic() method."""
    bc_periodic = BoundaryConditions(type="periodic")
    bc_dirichlet = BoundaryConditions(type="dirichlet", left_value=0.0, right_value=1.0)

    assert bc_periodic.is_periodic() is True
    assert bc_dirichlet.is_periodic() is False


@pytest.mark.unit
@pytest.mark.fast
def test_is_dirichlet():
    """Test is_dirichlet() method."""
    bc_dirichlet = BoundaryConditions(type="dirichlet", left_value=0.0, right_value=1.0)
    bc_periodic = BoundaryConditions(type="periodic")

    assert bc_dirichlet.is_dirichlet() is True
    assert bc_periodic.is_dirichlet() is False


@pytest.mark.unit
@pytest.mark.fast
def test_is_neumann():
    """Test is_neumann() method."""
    bc_neumann = BoundaryConditions(type="neumann", left_value=0.0, right_value=0.0)
    bc_no_flux = BoundaryConditions(type="no_flux")

    assert bc_neumann.is_neumann() is True
    assert bc_no_flux.is_neumann() is False


@pytest.mark.unit
@pytest.mark.fast
def test_is_no_flux():
    """Test is_no_flux() method."""
    bc_no_flux = BoundaryConditions(type="no_flux")
    bc_neumann = BoundaryConditions(type="neumann", left_value=0.0, right_value=0.0)

    assert bc_no_flux.is_no_flux() is True
    assert bc_neumann.is_no_flux() is False


@pytest.mark.unit
@pytest.mark.fast
def test_is_robin():
    """Test is_robin() method."""
    bc_robin = BoundaryConditions(
        type="robin",
        left_alpha=1.0,
        left_beta=0.5,
        left_value=0.0,
        right_alpha=1.0,
        right_beta=0.5,
        right_value=1.0,
    )
    bc_dirichlet = BoundaryConditions(type="dirichlet", left_value=0.0, right_value=1.0)

    assert bc_robin.is_robin() is True
    assert bc_dirichlet.is_robin() is False


# ============================================================================
# Test: Matrix Size Computation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_get_matrix_size_periodic():
    """Test matrix size for periodic BC."""
    bc = BoundaryConditions(type="periodic")
    # Periodic: M × M
    assert bc.get_matrix_size(10) == 10
    assert bc.get_matrix_size(100) == 100


@pytest.mark.unit
@pytest.mark.fast
def test_get_matrix_size_dirichlet():
    """Test matrix size for Dirichlet BC."""
    bc = BoundaryConditions(type="dirichlet", left_value=0.0, right_value=1.0)
    # Dirichlet: (M-1) × (M-1)
    assert bc.get_matrix_size(10) == 9
    assert bc.get_matrix_size(100) == 99


@pytest.mark.unit
@pytest.mark.fast
def test_get_matrix_size_neumann():
    """Test matrix size for Neumann BC."""
    bc = BoundaryConditions(type="neumann", left_value=0.0, right_value=0.0)
    # Neumann: (M+1) × (M+1)
    assert bc.get_matrix_size(10) == 11
    assert bc.get_matrix_size(100) == 101


@pytest.mark.unit
@pytest.mark.fast
def test_get_matrix_size_no_flux():
    """Test matrix size for no-flux BC."""
    bc = BoundaryConditions(type="no_flux")
    # No-flux: M × M
    assert bc.get_matrix_size(10) == 10
    assert bc.get_matrix_size(100) == 100


@pytest.mark.unit
@pytest.mark.fast
def test_get_matrix_size_robin():
    """Test matrix size for Robin BC."""
    bc = BoundaryConditions(
        type="robin",
        left_alpha=1.0,
        left_beta=0.5,
        left_value=0.0,
        right_alpha=1.0,
        right_beta=0.5,
        right_value=1.0,
    )
    # Robin: (M+1) × (M+1)
    assert bc.get_matrix_size(10) == 11
    assert bc.get_matrix_size(100) == 101


@pytest.mark.unit
@pytest.mark.fast
def test_get_matrix_size_invalid_type_raises():
    """Test matrix size with invalid BC type raises ValueError."""
    bc = BoundaryConditions(type="invalid_type")
    with pytest.raises(ValueError, match="Unknown boundary condition type"):
        bc.get_matrix_size(10)


# ============================================================================
# Test: Value Validation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_validate_values_periodic_no_requirements():
    """Test validation for periodic BC (no values required)."""
    bc = BoundaryConditions(type="periodic")
    # Should not raise
    bc.validate_values()


@pytest.mark.unit
@pytest.mark.fast
def test_validate_values_no_flux_no_requirements():
    """Test validation for no-flux BC (no values required)."""
    bc = BoundaryConditions(type="no_flux")
    # Should not raise
    bc.validate_values()


@pytest.mark.unit
@pytest.mark.fast
def test_validate_values_dirichlet_missing_left_raises():
    """Test Dirichlet validation fails when left_value is missing."""
    bc = BoundaryConditions(type="dirichlet", right_value=1.0)
    with pytest.raises(ValueError, match="Dirichlet boundary conditions require left_value and right_value"):
        bc.validate_values()


@pytest.mark.unit
@pytest.mark.fast
def test_validate_values_dirichlet_missing_right_raises():
    """Test Dirichlet validation fails when right_value is missing."""
    bc = BoundaryConditions(type="dirichlet", left_value=0.0)
    with pytest.raises(ValueError, match="Dirichlet boundary conditions require left_value and right_value"):
        bc.validate_values()


@pytest.mark.unit
@pytest.mark.fast
def test_validate_values_dirichlet_complete():
    """Test Dirichlet validation passes with all values."""
    bc = BoundaryConditions(type="dirichlet", left_value=0.0, right_value=1.0)
    # Should not raise
    bc.validate_values()


@pytest.mark.unit
@pytest.mark.fast
def test_validate_values_neumann_missing_left_raises():
    """Test Neumann validation fails when left_value is missing."""
    bc = BoundaryConditions(type="neumann", right_value=0.5)
    with pytest.raises(ValueError, match="Neumann boundary conditions require left_value and right_value"):
        bc.validate_values()


@pytest.mark.unit
@pytest.mark.fast
def test_validate_values_neumann_missing_right_raises():
    """Test Neumann validation fails when right_value is missing."""
    bc = BoundaryConditions(type="neumann", left_value=0.5)
    with pytest.raises(ValueError, match="Neumann boundary conditions require left_value and right_value"):
        bc.validate_values()


@pytest.mark.unit
@pytest.mark.fast
def test_validate_values_neumann_complete():
    """Test Neumann validation passes with all values."""
    bc = BoundaryConditions(type="neumann", left_value=0.5, right_value=-0.5)
    # Should not raise
    bc.validate_values()


@pytest.mark.unit
@pytest.mark.fast
def test_validate_values_robin_missing_params_raises():
    """Test Robin validation fails when parameters are missing during __post_init__."""
    # Missing alpha coefficients - should fail in __post_init__
    with pytest.raises(ValueError, match="Robin boundary conditions require alpha and beta"):
        BoundaryConditions(
            type="robin",
            left_beta=0.5,
            left_value=0.0,
            right_beta=0.5,
            right_value=1.0,
        )


@pytest.mark.unit
@pytest.mark.fast
def test_validate_values_robin_complete():
    """Test Robin validation passes with all parameters."""
    bc = BoundaryConditions(
        type="robin",
        left_alpha=1.0,
        left_beta=0.5,
        left_value=0.0,
        right_alpha=1.0,
        right_beta=0.5,
        right_value=1.0,
    )
    # Should not raise
    bc.validate_values()


# ============================================================================
# Test: String Representation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_str_periodic():
    """Test string representation for periodic BC."""
    bc = BoundaryConditions(type="periodic")
    assert str(bc) == "Periodic"


@pytest.mark.unit
@pytest.mark.fast
def test_str_dirichlet():
    """Test string representation for Dirichlet BC."""
    bc = BoundaryConditions(type="dirichlet", left_value=0.0, right_value=1.0)
    assert str(bc) == "Dirichlet(left=0.0, right=1.0)"


@pytest.mark.unit
@pytest.mark.fast
def test_str_neumann():
    """Test string representation for Neumann BC."""
    bc = BoundaryConditions(type="neumann", left_value=0.5, right_value=-0.5)
    assert str(bc) == "Neumann(left=0.5, right=-0.5)"


@pytest.mark.unit
@pytest.mark.fast
def test_str_no_flux():
    """Test string representation for no-flux BC."""
    bc = BoundaryConditions(type="no_flux")
    assert str(bc) == "No-flux"


@pytest.mark.unit
@pytest.mark.fast
def test_str_robin():
    """Test string representation for Robin BC."""
    bc = BoundaryConditions(
        type="robin",
        left_alpha=1.0,
        left_beta=0.5,
        left_value=0.0,
        right_alpha=2.0,
        right_beta=0.3,
        right_value=1.0,
    )
    expected = "Robin(left: 1.0u + 0.5u' = 0.0, right: 2.0u + 0.3u' = 1.0)"
    assert str(bc) == expected


@pytest.mark.unit
@pytest.mark.fast
def test_str_unknown_type():
    """Test string representation for unknown BC type."""
    bc = BoundaryConditions(type="unknown_type")
    assert str(bc) == "Unknown(unknown_type)"


# ============================================================================
# Test: Factory Functions
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_factory_periodic_bc():
    """Test periodic_bc() factory function."""
    bc = periodic_bc()
    assert bc.type == "periodic"
    assert bc.is_periodic()


@pytest.mark.unit
@pytest.mark.fast
def test_factory_dirichlet_bc():
    """Test dirichlet_bc() factory function."""
    bc = dirichlet_bc(left_value=0.0, right_value=1.0)
    assert bc.type == "dirichlet"
    assert bc.is_dirichlet()
    assert bc.left_value == 0.0
    assert bc.right_value == 1.0


@pytest.mark.unit
@pytest.mark.fast
def test_factory_neumann_bc():
    """Test neumann_bc() factory function."""
    bc = neumann_bc(left_gradient=0.5, right_gradient=-0.5)
    assert bc.type == "neumann"
    assert bc.is_neumann()
    assert bc.left_value == 0.5
    assert bc.right_value == -0.5


@pytest.mark.unit
@pytest.mark.fast
def test_factory_no_flux_bc():
    """Test no_flux_bc() factory function."""
    bc = no_flux_bc()
    assert bc.type == "no_flux"
    assert bc.is_no_flux()


@pytest.mark.unit
@pytest.mark.fast
def test_factory_robin_bc():
    """Test robin_bc() factory function."""
    bc = robin_bc(
        left_alpha=1.0,
        left_beta=0.5,
        left_value=0.0,
        right_alpha=1.0,
        right_beta=0.5,
        right_value=1.0,
    )
    assert bc.type == "robin"
    assert bc.is_robin()
    assert bc.left_alpha == 1.0
    assert bc.left_beta == 0.5
    assert bc.left_value == 0.0


@pytest.mark.unit
@pytest.mark.fast
def test_factory_dirichlet_values_stored():
    """Test dirichlet_bc() correctly stores boundary values."""
    bc = dirichlet_bc(left_value=2.5, right_value=-1.3)
    assert bc.left_value == 2.5
    assert bc.right_value == -1.3


@pytest.mark.unit
@pytest.mark.fast
def test_factory_neumann_values_stored():
    """Test neumann_bc() correctly stores gradient values."""
    bc = neumann_bc(left_gradient=1.2, right_gradient=3.4)
    assert bc.left_value == 1.2
    assert bc.right_value == 3.4


@pytest.mark.unit
@pytest.mark.fast
def test_factory_robin_all_params_stored():
    """Test robin_bc() correctly stores all parameters."""
    bc = robin_bc(
        left_alpha=2.0,
        left_beta=0.7,
        left_value=1.5,
        right_alpha=3.0,
        right_beta=0.9,
        right_value=2.5,
    )
    assert bc.left_alpha == 2.0
    assert bc.left_beta == 0.7
    assert bc.left_value == 1.5
    assert bc.right_alpha == 3.0
    assert bc.right_beta == 0.9
    assert bc.right_value == 2.5


@pytest.mark.unit
@pytest.mark.fast
def test_factory_functions_return_correct_type():
    """Test factory functions return BoundaryConditions instances."""
    assert isinstance(periodic_bc(), BoundaryConditions)
    assert isinstance(dirichlet_bc(0.0, 1.0), BoundaryConditions)
    assert isinstance(neumann_bc(0.0, 0.0), BoundaryConditions)
    assert isinstance(no_flux_bc(), BoundaryConditions)
    assert isinstance(robin_bc(1.0, 0.5, 0.0, 1.0, 0.5, 1.0), BoundaryConditions)


@pytest.mark.unit
@pytest.mark.fast
def test_factory_validation_on_creation():
    """Test factory functions trigger validation on creation."""
    # Robin BC without all params should fail in __post_init__
    with pytest.raises(ValueError):
        BoundaryConditions(type="robin", left_alpha=1.0)


# ============================================================================
# Test: Edge Cases and Integration
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_multiple_bc_instances_independent():
    """Test multiple BC instances are independent."""
    bc1 = dirichlet_bc(0.0, 1.0)
    bc2 = dirichlet_bc(2.0, 3.0)

    assert bc1.left_value == 0.0
    assert bc2.left_value == 2.0
    # Modifying bc2 should not affect bc1
    assert bc1.right_value == 1.0


@pytest.mark.unit
@pytest.mark.fast
def test_bc_with_negative_values():
    """Test BC with negative boundary values."""
    bc_dirichlet = dirichlet_bc(-1.0, -2.0)
    bc_neumann = neumann_bc(-0.5, -1.5)

    assert bc_dirichlet.left_value == -1.0
    assert bc_neumann.right_value == -1.5


@pytest.mark.unit
@pytest.mark.fast
def test_bc_with_zero_values():
    """Test BC with zero boundary values."""
    bc = dirichlet_bc(0.0, 0.0)
    assert bc.left_value == 0.0
    assert bc.right_value == 0.0


@pytest.mark.unit
@pytest.mark.fast
def test_bc_with_large_values():
    """Test BC with large boundary values."""
    bc = dirichlet_bc(1e10, 1e15)
    assert bc.left_value == 1e10
    assert bc.right_value == 1e15


@pytest.mark.unit
@pytest.mark.fast
def test_bc_with_small_values():
    """Test BC with very small boundary values."""
    bc = dirichlet_bc(1e-10, 1e-15)
    assert bc.left_value == 1e-10
    assert bc.right_value == 1e-15


@pytest.mark.unit
@pytest.mark.fast
def test_bc_large_matrix_sizes():
    """Test matrix size computation with large grids."""
    bc_periodic = BoundaryConditions(type="periodic")
    bc_dirichlet = dirichlet_bc(0.0, 1.0)

    assert bc_periodic.get_matrix_size(10000) == 10000
    assert bc_dirichlet.get_matrix_size(10000) == 9999


@pytest.mark.unit
@pytest.mark.fast
def test_bc_minimal_matrix_sizes():
    """Test matrix size computation with minimal grids."""
    bc_periodic = BoundaryConditions(type="periodic")
    bc_dirichlet = dirichlet_bc(0.0, 1.0)

    assert bc_periodic.get_matrix_size(2) == 2
    assert bc_dirichlet.get_matrix_size(2) == 1


@pytest.mark.unit
@pytest.mark.fast
def test_robin_bc_equal_coefficients():
    """Test Robin BC with equal alpha and beta coefficients."""
    bc = robin_bc(
        left_alpha=1.0,
        left_beta=1.0,
        left_value=0.0,
        right_alpha=1.0,
        right_beta=1.0,
        right_value=1.0,
    )
    assert bc.left_alpha == bc.left_beta
    assert bc.right_alpha == bc.right_beta


@pytest.mark.unit
@pytest.mark.fast
def test_robin_bc_asymmetric_coefficients():
    """Test Robin BC with different left and right coefficients."""
    bc = robin_bc(
        left_alpha=2.0,
        left_beta=0.5,
        left_value=0.0,
        right_alpha=0.5,
        right_beta=2.0,
        right_value=1.0,
    )
    assert bc.left_alpha != bc.right_alpha
    assert bc.left_beta != bc.right_beta


@pytest.mark.unit
@pytest.mark.fast
def test_validation_called_independently():
    """Test validate_values() can be called multiple times."""
    bc = dirichlet_bc(0.0, 1.0)
    # Should not raise on repeated calls
    bc.validate_values()
    bc.validate_values()
    bc.validate_values()


@pytest.mark.unit
@pytest.mark.fast
def test_bc_type_string_case_sensitive():
    """Test BC type strings are case-sensitive."""
    bc_lower = BoundaryConditions(type="periodic")
    assert bc_lower.is_periodic()

    # Different case should not match
    bc_upper = BoundaryConditions(type="PERIODIC")
    assert not bc_upper.is_periodic()
