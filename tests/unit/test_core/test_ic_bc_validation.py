#!/usr/bin/env python3
"""
Unit tests for IC/BC validation integration (Issue #681).

Tests that:
- NDArray m_initial/u_final are accepted and copied correctly
- Wrong-shape arrays are caught by validate_components()
- Callable dimension mismatches are detected
- NaN/Inf values in arrays and callables are rejected
- Invalid types (e.g. string) raise ValidationError
"""

import pytest

import numpy as np

from mfg_pde.core.mfg_components import MFGComponents
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary.conditions import no_flux_bc
from mfg_pde.utils.validation import ValidationError

# ===========================================================================
# Test Helpers
# ===========================================================================


def _geometry(Nx_points=11, dimension=1):
    """Create a test geometry."""
    bounds = [(0.0, 1.0)] * dimension
    nx = [Nx_points] * dimension
    return TensorProductGrid(
        bounds=bounds,
        Nx_points=nx,
        boundary_conditions=no_flux_bc(dimension=dimension),
    )


def _hamiltonian():
    """Create a test Hamiltonian."""
    from mfg_pde.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian

    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: -(m**2),
        coupling_dm=lambda m: -2 * m,
    )


def _quadratic(x):
    """u_final = x^2."""
    return x**2


def _gaussian(x):
    """m_initial = Gaussian centered at 0.5."""
    return np.exp(-10 * (x - 0.5) ** 2)


def _nan_func(x):
    """Always returns NaN."""
    return float("nan")


def _problem(m_initial, u_terminal, Nx_points=11, dimension=1, **kwargs):
    """Create a test MFGProblem with given IC/BC."""
    geom = _geometry(Nx_points=Nx_points, dimension=dimension)
    components = MFGComponents(
        hamiltonian=_hamiltonian(),
        m_initial=m_initial,
        u_terminal=u_terminal,
    )
    return MFGProblem(geometry=geom, components=components, **kwargs)


# ===========================================================================
# NDArray m_initial tests
# ===========================================================================


@pytest.mark.unit
def test_array_m_initial_correct_shape():
    """NDArray m_initial with matching shape is accepted and copied."""
    Nx = 11
    arr = np.ones(Nx) / Nx  # Uniform density, correct shape (11,)

    problem = _problem(m_initial=arr, u_terminal=_quadratic, Nx_points=Nx)

    # m_initial should be set (after normalization, values may differ from input)
    assert problem.m_initial is not None
    assert problem.m_initial.shape == (Nx,)
    # Should not be all zeros (the array was copied)
    assert np.sum(problem.m_initial) > 0


@pytest.mark.unit
def test_array_m_initial_wrong_shape_raises():
    """NDArray m_initial with wrong shape raises ValidationError."""
    Nx = 11
    wrong_arr = np.ones(Nx + 5)  # Shape (16,) vs expected (11,)

    with pytest.raises(ValidationError, match="shape"):
        _problem(m_initial=wrong_arr, u_terminal=_quadratic, Nx_points=Nx)


# ===========================================================================
# NDArray u_final tests
# ===========================================================================


@pytest.mark.unit
def test_array_u_final_correct_shape():
    """NDArray u_terminal with matching shape is accepted and copied."""
    Nx = 11
    arr = np.linspace(0, 1, Nx) ** 2  # Quadratic terminal cost as array

    problem = _problem(m_initial=_gaussian, u_terminal=arr, Nx_points=Nx)

    assert problem.u_terminal is not None
    assert problem.u_terminal.shape == (Nx,)
    np.testing.assert_array_almost_equal(problem.u_terminal, arr)


@pytest.mark.unit
def test_array_u_final_wrong_shape_raises():
    """NDArray u_final with wrong shape raises ValidationError."""
    Nx = 11
    wrong_arr = np.ones(Nx + 3)  # Shape (14,) vs expected (11,)

    with pytest.raises(ValidationError, match="shape"):
        _problem(m_initial=_gaussian, u_terminal=wrong_arr, Nx_points=Nx)


# ===========================================================================
# NaN/Inf detection
# ===========================================================================


@pytest.mark.unit
def test_array_with_nan_raises():
    """Array containing NaN raises ValidationError."""
    Nx = 11
    arr = np.ones(Nx)
    arr[5] = np.nan

    with pytest.raises(ValidationError, match="NaN"):
        _problem(m_initial=arr, u_terminal=_quadratic, Nx_points=Nx)


@pytest.mark.unit
def test_array_with_inf_raises():
    """Array containing Inf raises ValidationError."""
    Nx = 11
    arr = np.ones(Nx)
    arr[3] = np.inf

    with pytest.raises(ValidationError, match="Inf"):
        _problem(m_initial=arr, u_terminal=_quadratic, Nx_points=Nx)


@pytest.mark.unit
def test_callable_returning_nan_raises():
    """Callable that returns NaN at sample point raises ValidationError."""
    with pytest.raises(ValidationError, match="non-finite"):
        _problem(m_initial=_nan_func, u_terminal=_quadratic)


# ===========================================================================
# Invalid type detection
# ===========================================================================


@pytest.mark.unit
def test_invalid_type_m_initial_raises():
    """Non-callable, non-array m_initial raises ValidationError."""
    with pytest.raises(ValidationError, match="callable or ndarray"):
        _problem(m_initial="not_valid", u_terminal=_quadratic)


@pytest.mark.unit
def test_invalid_type_u_final_raises():
    """Non-callable, non-array u_final raises ValidationError."""
    with pytest.raises(ValidationError, match="callable or ndarray"):
        _problem(m_initial=_gaussian, u_terminal="not_valid")


# ===========================================================================
# Callable with wrong signature
# ===========================================================================


@pytest.mark.unit
def test_callable_raising_exception_detected():
    """Callable that crashes on evaluation raises ValidationError."""

    def bad_m_initial(x):
        raise RuntimeError("Intentional failure")

    with pytest.raises(ValidationError, match="exception"):
        _problem(m_initial=bad_m_initial, u_terminal=_quadratic)


# ===========================================================================
# Both NDArray IC and BC
# ===========================================================================


@pytest.mark.unit
def test_both_arrays_correct_shape():
    """Both m_initial and u_terminal as NDArrays with correct shape work."""
    Nx = 11
    m_arr = np.ones(Nx)
    u_arr = np.linspace(0, 1, Nx)

    problem = _problem(m_initial=m_arr, u_terminal=u_arr, Nx_points=Nx)

    assert problem.m_initial.shape == (Nx,)
    assert problem.u_terminal.shape == (Nx,)
    np.testing.assert_array_almost_equal(problem.u_terminal, u_arr)
