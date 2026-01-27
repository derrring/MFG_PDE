"""
Test validation of Hamiltonian derivative sign.

Default Hamiltonian: H = 0.5*c*|p|^2 - V(x) - m^2
Correct derivative: dH/dm = -2m
"""

import pytest

import numpy as np

from mfg_pde import MFGProblem
from mfg_pde.core.derivatives import DerivativeTensors
from mfg_pde.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfg_pde.core.mfg_components import MFGComponents
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import no_flux_bc


def _default_hamiltonian():
    """Default Hamiltonian for testing dH/dm = -2m (Issue #670: explicit specification required).

    H = 0.5*c*|p|^2 - V(x) - m^2 => coupling(m) = -m^2, coupling_dm(m) = -2m
    """
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: -(m**2),
        coupling_dm=lambda m: -2 * m,
    )


def _default_components():
    """Default MFGComponents for testing (Issue #670: explicit specification required)."""
    return MFGComponents(
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),
        u_final=lambda x: 0.0,
        hamiltonian=_default_hamiltonian(),
    )


@pytest.fixture
def simple_problem():
    """Create a simple 1D MFG problem for testing."""
    geometry = TensorProductGrid(bounds=[(0, 1)], Nx=[10], boundary_conditions=no_flux_bc(dimension=1))
    return MFGProblem(geometry=geometry, T=1.0, Nt=10, diffusion=0.1, components=_default_components())


@pytest.fixture
def dummy_derivs():
    """Create dummy derivatives required by the dH_dm API."""
    return DerivativeTensors.from_arrays(grad=np.array([0.0]))


class TestHamiltonianDerivative:
    """Tests for Hamiltonian derivative dH/dm correctness."""

    @pytest.mark.parametrize("m_value", [0.0, 0.5, 1.0, 2.0, -1.0])
    def test_dH_dm_returns_correct_value(self, simple_problem, dummy_derivs, m_value):
        """Verify dH/dm = -2m for default Hamiltonian H = 0.5*c*|p|^2 - V(x) - m^2."""
        dH_dm = simple_problem.dH_dm(x_idx=0, m_at_x=m_value, derivs=dummy_derivs)
        expected = -2.0 * m_value

        assert abs(dH_dm - expected) < 1e-10, f"dH/dm incorrect at m={m_value}: got {dH_dm}, expected {expected}"

    def test_dH_dm_sign_is_negative_for_positive_m(self, simple_problem, dummy_derivs):
        """Verify dH/dm is negative when m > 0."""
        dH_dm = simple_problem.dH_dm(x_idx=0, m_at_x=1.0, derivs=dummy_derivs)
        assert dH_dm < 0, f"dH/dm should be negative for m=1.0, got {dH_dm}"

    def test_dH_dm_sign_is_positive_for_negative_m(self, simple_problem, dummy_derivs):
        """Verify dH/dm is positive when m < 0."""
        dH_dm = simple_problem.dH_dm(x_idx=0, m_at_x=-1.0, derivs=dummy_derivs)
        assert dH_dm > 0, f"dH/dm should be positive for m=-1.0, got {dH_dm}"

    def test_dH_dm_is_zero_at_m_zero(self, simple_problem, dummy_derivs):
        """Verify dH/dm = 0 when m = 0."""
        dH_dm = simple_problem.dH_dm(x_idx=0, m_at_x=0.0, derivs=dummy_derivs)
        assert abs(dH_dm) < 1e-10, f"dH/dm should be zero at m=0, got {dH_dm}"
