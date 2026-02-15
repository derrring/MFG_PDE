#!/usr/bin/env python3
"""
Unit tests for array/tensor validation (Issue #687).

Tests that:
- Array dtype validation catches incompatible types
- Field shape validation catches spatial/spatiotemporal mismatches
- Finite value validation detects NaN/Inf with location
- Non-negative validation catches negative density values
- MFGProblem construction rejects NaN diffusion/drift arrays
- MFGProblem construction rejects wrong-shape diffusion arrays
- MFGProblem construction rejects NaN-producing u_final/m_initial

Follows the pattern of test_custom_function_validation.py.
"""

import pytest

import numpy as np

from mfg_pde.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfg_pde.core.mfg_components import MFGComponents
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary.conditions import no_flux_bc
from mfg_pde.utils.validation import (
    ValidationError,
    validate_array_dtype,
    validate_field_shape,
    validate_finite,
    validate_non_negative,
)

# ===========================================================================
# Test Helpers
# ===========================================================================

Nx = 11


def _geometry(Nx_points=Nx, dimension=1):
    """Create a test geometry."""
    bounds = [(0.0, 1.0)] * dimension
    nx = [Nx_points] * dimension
    return TensorProductGrid(
        bounds=bounds,
        Nx_points=nx,
        boundary_conditions=no_flux_bc(dimension=dimension),
    )


def _hamiltonian():
    """Create a standard test Hamiltonian."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: -(m**2),
        coupling_dm=lambda m: -2 * m,
    )


def _problem(m_initial, u_terminal, hamiltonian=None, Nx_points=Nx, **kwargs):
    """Create a test MFGProblem."""
    geom = _geometry(Nx_points=Nx_points)
    components = MFGComponents(
        hamiltonian=hamiltonian or _hamiltonian(),
        m_initial=m_initial,
        u_terminal=u_terminal,
    )
    return MFGProblem(geometry=geom, components=components, **kwargs)


# ===========================================================================
# Standalone: validate_array_dtype
# ===========================================================================


@pytest.mark.unit
def test_validate_array_dtype_float64_passes():
    """float64 array should pass dtype validation."""
    arr = np.ones(10, dtype=np.float64)
    result = validate_array_dtype(arr, "test_arr")
    assert result.is_valid


@pytest.mark.unit
def test_validate_array_dtype_float32_warns():
    """float32 array should produce warning, not error."""
    arr = np.ones(10, dtype=np.float32)
    result = validate_array_dtype(arr, "test_arr")
    # float32 is compatible floating type: warning, but still valid
    assert result.is_valid
    assert len(result.issues) > 0
    assert any("float32" in str(issue) for issue in result.issues)


@pytest.mark.unit
def test_validate_array_dtype_complex_errors():
    """Complex dtype should fail validation."""
    arr = np.ones(10, dtype=np.complex128)
    result = validate_array_dtype(arr, "test_arr")
    assert not result.is_valid
    assert any("incompatible" in str(issue).lower() for issue in result.issues)


# ===========================================================================
# Standalone: validate_field_shape
# ===========================================================================


@pytest.mark.unit
def test_validate_field_shape_spatial_passes():
    """Array matching spatial shape should pass."""
    arr = np.ones((Nx,))
    result = validate_field_shape(arr, (Nx,), "test_field")
    assert result.is_valid
    assert result.context.get("is_temporal") is False


@pytest.mark.unit
def test_validate_field_shape_spatiotemporal_passes():
    """Array with (Nt, *spatial) shape should pass when temporal allowed."""
    Nt = 20
    arr = np.ones((Nt, Nx))
    result = validate_field_shape(arr, (Nx,), "test_field")
    assert result.is_valid
    assert result.context.get("is_temporal") is True
    assert result.context.get("n_timesteps") == Nt


@pytest.mark.unit
def test_validate_field_shape_wrong_errors():
    """Array with mismatched shape should fail."""
    arr = np.ones((Nx + 5,))  # Wrong spatial size
    result = validate_field_shape(arr, (Nx,), "test_field")
    assert not result.is_valid
    assert any("shape" in str(issue).lower() for issue in result.issues)


# ===========================================================================
# Standalone: validate_finite
# ===========================================================================


@pytest.mark.unit
def test_validate_finite_passes():
    """Clean array should pass finiteness check."""
    arr = np.linspace(0.0, 1.0, Nx)
    result = validate_finite(arr, "test_arr")
    assert result.is_valid


@pytest.mark.unit
def test_validate_finite_nan_errors():
    """Array with NaN should fail with location info."""
    arr = np.ones(Nx)
    arr[5] = np.nan
    result = validate_finite(arr, "test_arr")
    assert not result.is_valid
    assert any("NaN" in str(issue) for issue in result.issues)
    assert result.context.get("n_nan") == 1


@pytest.mark.unit
def test_validate_finite_inf_errors():
    """Array with Inf should fail with location info."""
    arr = np.ones(Nx)
    arr[3] = np.inf
    result = validate_finite(arr, "test_arr")
    assert not result.is_valid
    assert any("Inf" in str(issue) for issue in result.issues)
    assert result.context.get("n_inf") == 1


# ===========================================================================
# Standalone: validate_non_negative
# ===========================================================================


@pytest.mark.unit
def test_validate_non_negative_passes():
    """Non-negative array should pass."""
    arr = np.abs(np.random.randn(Nx))
    result = validate_non_negative(arr, "density")
    assert result.is_valid


@pytest.mark.unit
def test_validate_non_negative_errors():
    """Array with negative values should fail."""
    arr = np.ones(Nx)
    arr[2] = -0.5
    result = validate_non_negative(arr, "density")
    assert not result.is_valid
    assert any("negative" in str(issue).lower() for issue in result.issues)
    assert result.context.get("min_value") == pytest.approx(-0.5)


# ===========================================================================
# Integration: MFGProblem with array diffusion
# ===========================================================================


@pytest.mark.unit
def test_mfg_problem_valid_diffusion_array_accepted():
    """MFGProblem with valid ndarray diffusion should construct."""
    sigma_array = np.linspace(0.1, 1.0, Nx)
    problem = _problem(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: x**2,
        sigma=sigma_array,
    )
    assert problem is not None
    assert isinstance(problem.diffusion_field, np.ndarray)


@pytest.mark.unit
def test_mfg_problem_nan_diffusion_array_rejected():
    """MFGProblem with NaN in diffusion array should raise ValidationError."""
    sigma_array = np.ones(Nx)
    sigma_array[5] = np.nan

    with pytest.raises(ValidationError, match="NaN"):
        _problem(
            m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
            u_terminal=lambda x: x**2,
            sigma=sigma_array,
        )


@pytest.mark.unit
def test_mfg_problem_wrong_shape_diffusion_rejected():
    """MFGProblem with wrong-shape diffusion array should raise ValidationError."""
    wrong_shape = np.ones(Nx + 10)  # 21 points, but grid has 11

    with pytest.raises(ValidationError, match="shape"):
        _problem(
            m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
            u_terminal=lambda x: x**2,
            diffusion=wrong_shape,
        )


# ===========================================================================
# Integration: MFGProblem NaN-producing callables
# ===========================================================================


@pytest.mark.unit
def test_mfg_problem_nan_u_final_rejected():
    """MFGProblem with NaN-producing u_final should raise ValidationError."""

    def nan_u_final(x):
        return float("nan")

    with pytest.raises(ValidationError, match=r"non-finite|NaN"):
        _problem(
            m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
            u_terminal=nan_u_final,
        )


@pytest.mark.unit
def test_mfg_problem_nan_m_initial_rejected():
    """MFGProblem with NaN-producing m_initial should raise ValidationError."""

    def nan_m_initial(x):
        return float("nan")

    with pytest.raises(ValidationError, match=r"non-finite|NaN"):
        _problem(
            m_initial=nan_m_initial,
            u_terminal=lambda x: x**2,
        )
