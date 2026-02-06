#!/usr/bin/env python3
"""
Unit tests for custom function validation (Issue #686).

Tests that:
- Valid HamiltonianBase instances pass validation
- NaN-producing Hamiltonians are caught
- Hamiltonian derivative consistency checking works
- Drift functions with correct/wrong signatures are handled
- Running cost functions with correct/wrong signatures are handled

Follows the pattern of test_ic_bc_validation.py.
"""

import pytest

import numpy as np

from mfg_pde.core.hamiltonian import (
    HamiltonianBase,
    QuadraticControlCost,
    SeparableHamiltonian,
)
from mfg_pde.core.mfg_components import MFGComponents
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary.conditions import no_flux_bc
from mfg_pde.utils.validation import (
    ValidationError,
    validate_custom_functions,
    validate_drift,
    validate_hamiltonian,
    validate_hamiltonian_consistency,
    validate_running_cost,
)

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
    """Create a standard test Hamiltonian."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: -(m**2),
        coupling_dm=lambda m: -2 * m,
    )


def _problem(m_initial, u_terminal, hamiltonian=None, Nx_points=11, **kwargs):
    """Create a test MFGProblem."""
    geom = _geometry(Nx_points=Nx_points)
    components = MFGComponents(
        hamiltonian=hamiltonian or _hamiltonian(),
        m_initial=m_initial,
        u_terminal=u_terminal,
    )
    return MFGProblem(geometry=geom, components=components, **kwargs)


# ===========================================================================
# Hamiltonian validation (standalone)
# ===========================================================================


@pytest.mark.unit
def test_valid_hamiltonian_passes():
    """A standard SeparableHamiltonian should pass validation."""
    H = _hamiltonian()
    geom = _geometry()
    result = validate_hamiltonian(H, geom)
    assert result.is_valid, f"Unexpected issues: {result.issues}"


@pytest.mark.unit
def test_hamiltonian_returning_nan_raises():
    """A Hamiltonian that returns NaN should fail validation."""

    class NaNHamiltonian(HamiltonianBase):
        @property
        def dimension(self):
            return 1

        def __call__(self, x, m, p, t=0.0):
            return float("nan")

    H = NaNHamiltonian()
    geom = _geometry()
    result = validate_hamiltonian(H, geom)
    assert not result.is_valid
    assert any("NaN" in str(issue) for issue in result.issues)


@pytest.mark.unit
def test_hamiltonian_wrong_signature_raises():
    """An object that doesn't accept (x, m, p, t) should fail validation."""

    class BadHamiltonian:
        def __call__(self, x):
            return 0.0

    geom = _geometry()
    result = validate_hamiltonian(BadHamiltonian(), geom)
    assert not result.is_valid
    assert any("signature" in str(issue).lower() for issue in result.issues)


# ===========================================================================
# Hamiltonian consistency check
# ===========================================================================


@pytest.mark.unit
def test_hamiltonian_consistency_passes():
    """SeparableHamiltonian with correct coupling_dm should pass consistency."""
    H = _hamiltonian()
    geom = _geometry()
    result = validate_hamiltonian_consistency(H, H.dm, geom)
    # Should have no warnings about inconsistency
    inconsistent_warnings = [i for i in result.issues if "inconsistent" in str(i).lower()]
    assert len(inconsistent_warnings) == 0


@pytest.mark.unit
def test_hamiltonian_consistency_warning():
    """Inconsistent dH_dm should produce a warning."""
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: -(m**2),
        coupling_dm=lambda m: -2 * m,
    )
    geom = _geometry()

    # Provide a wrong derivative: constant 42.0 instead of -2m
    def wrong_dm(x, m, p, t=0.0):
        return 42.0

    result = validate_hamiltonian_consistency(H, wrong_dm, geom)
    warnings = [i for i in result.issues if "inconsistent" in str(i).lower()]
    assert len(warnings) > 0


@pytest.mark.unit
def test_hamiltonian_dp_consistency_passes():
    """Correct dH_dp should pass consistency check."""
    H = _hamiltonian()
    geom = _geometry()
    result = validate_hamiltonian_consistency(H, H.dm, geom, dH_dp=H.dp)
    inconsistent_warnings = [i for i in result.issues if "inconsistent" in str(i).lower()]
    assert len(inconsistent_warnings) == 0


@pytest.mark.unit
def test_hamiltonian_dp_consistency_warning():
    """Inconsistent dH_dp should produce a warning."""
    H = _hamiltonian()
    geom = _geometry()

    # Provide a wrong dp: constant 99.0 instead of p/control_cost
    def wrong_dp(x, m, p, t=0.0):
        return np.array([99.0])

    result = validate_hamiltonian_consistency(H, H.dm, geom, dH_dp=wrong_dp)
    warnings = [i for i in result.issues if "dH_dp" in str(i)]
    assert len(warnings) > 0


# ===========================================================================
# validate_custom_functions (aggregate)
# ===========================================================================


@pytest.mark.unit
def test_validate_custom_functions_all_valid():
    """All functions valid should produce no errors."""
    H = _hamiltonian()
    geom = _geometry()
    result = validate_custom_functions(
        hamiltonian=H,
        dH_dm=H.dm,
        dH_dp=H.dp,
        geometry=geom,
    )
    assert result.is_valid


@pytest.mark.unit
def test_validate_custom_functions_with_consistency():
    """Consistency check enabled with correct derivatives should pass."""
    H = _hamiltonian()
    geom = _geometry()
    result = validate_custom_functions(
        hamiltonian=H,
        dH_dm=H.dm,
        dH_dp=H.dp,
        geometry=geom,
        check_consistency=True,
    )
    assert result.is_valid


# ===========================================================================
# Drift validation
# ===========================================================================


@pytest.mark.unit
def test_valid_drift_passes():
    """A drift with signature drift(x, m) should pass."""
    geom = _geometry()

    def my_drift(x, m):
        return -x

    result = validate_drift(my_drift, geom)
    assert result.is_valid


@pytest.mark.unit
def test_drift_wrong_signature_raises():
    """A drift with wrong arity should fail."""
    geom = _geometry()

    def bad_drift():
        return 0.0

    result = validate_drift(bad_drift, geom)
    assert not result.is_valid
    assert any("signature" in str(issue).lower() for issue in result.issues)


# ===========================================================================
# Running cost validation
# ===========================================================================


@pytest.mark.unit
def test_valid_running_cost_passes():
    """A running cost with signature f(x, m) should pass."""
    geom = _geometry()

    def my_cost(x, m):
        return float(np.sum(x**2)) + m

    result = validate_running_cost(my_cost, geom)
    assert result.is_valid


@pytest.mark.unit
def test_running_cost_wrong_signature_raises():
    """A running cost with wrong arity should fail."""
    geom = _geometry()

    def bad_cost():
        return 0.0

    result = validate_running_cost(bad_cost, geom)
    assert not result.is_valid
    assert any("signature" in str(issue).lower() for issue in result.issues)


# ===========================================================================
# Integration: MFGProblem construction triggers validation
# ===========================================================================


@pytest.mark.unit
def test_mfg_problem_valid_hamiltonian_accepted():
    """MFGProblem with valid Hamiltonian should construct without error."""
    problem = _problem(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_terminal=lambda x: x**2,
    )
    assert problem is not None


@pytest.mark.unit
def test_mfg_problem_nan_hamiltonian_rejected():
    """MFGProblem with NaN-producing Hamiltonian should raise ValidationError."""

    class NaNHamiltonian(HamiltonianBase):
        @property
        def dimension(self):
            return 1

        def __call__(self, x, m, p, t=0.0):
            return float("nan")

    with pytest.raises(ValidationError, match="NaN"):
        _problem(
            m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
            u_terminal=lambda x: x**2,
            hamiltonian=NaNHamiltonian(),
        )
