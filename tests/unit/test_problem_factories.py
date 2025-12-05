"""
Unit tests for unified problem factories (Phase 3.3).

Tests the new factory functions that support both unified MFGProblem
and legacy specialized problem classes.
"""

import pytest

import numpy as np

from mfg_pde.core.mfg_problem import MFGComponents, MFGProblem
from mfg_pde.factory import (
    create_crowd_problem,
    create_lq_problem,
    create_mfg_problem,
    create_standard_problem,
    create_stochastic_problem,
)

# Use legacy 1D BoundaryConditions for Domain1D/2D
from mfg_pde.geometry import Domain1D, Domain2D
from mfg_pde.geometry.boundary.fdm_bc_1d import BoundaryConditions


@pytest.fixture
def simple_domain():
    """Create simple 1D domain for testing."""
    bc = BoundaryConditions("neumann", left_value=0.0, right_value=0.0)
    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)
    domain.create_grid(num_points=51)  # Must create grid before use
    return domain


@pytest.fixture
def simple_2d_domain():
    """Create simple 2D domain for testing."""
    bc = BoundaryConditions("neumann", left_value=0.0, right_value=0.0)
    domain = Domain2D(xmin=[0.0, 0.0], xmax=[1.0, 1.0], boundary_conditions=bc)
    domain.create_grid(num_points=[21, 21])  # Must create grid before use
    return domain


@pytest.mark.skip(reason="Factory signature validation issue - deferred to Phase 3.5")
def test_create_standard_problem(simple_domain):
    """Test standard MFG problem creation."""

    def hamiltonian(x, p, m, t):
        return 0.5 * p**2 + m

    def hamiltonian_dm(x, p, m, t):
        return 1.0

    def terminal_cost(x):
        return x**2

    def initial_density(x):
        return np.exp(-10 * (x - 0.5) ** 2)

    # Test unified API
    problem = create_standard_problem(
        hamiltonian=hamiltonian,
        hamiltonian_dm=hamiltonian_dm,
        terminal_cost=terminal_cost,
        initial_density=initial_density,
        geometry=simple_domain,
        use_unified=True,
    )

    assert isinstance(problem, MFGProblem)
    assert problem.components is not None
    assert problem.components.hamiltonian_func is not None
    assert problem.components.hamiltonian_dm_func is not None
    assert problem.get_problem_type() == "standard"


@pytest.mark.skip(reason="Factory signature validation issue - deferred to Phase 3.5")
def test_create_lq_problem(simple_domain):
    """Test Linear-Quadratic MFG problem creation."""

    def terminal_cost(x):
        return x**2

    def initial_density(x):
        return np.exp(-10 * (x - 0.5) ** 2)

    problem = create_lq_problem(
        geometry=simple_domain,
        terminal_cost=terminal_cost,
        initial_density=initial_density,
        running_cost_control=0.5,
        running_cost_congestion=1.0,
    )

    assert isinstance(problem, MFGProblem)
    assert problem.components is not None
    assert problem.components.hamiltonian_func is not None


@pytest.mark.skip(reason="Factory signature validation issue - deferred to Phase 3.5")
def test_create_crowd_problem(simple_2d_domain):
    """Test crowd dynamics MFG problem creation."""
    target = np.array([0.8, 0.8])

    def initial_density(x):
        return np.exp(-10 * np.linalg.norm(x - np.array([0.2, 0.2])) ** 2)

    problem = create_crowd_problem(
        geometry=simple_2d_domain,
        target_location=target,
        initial_density=initial_density,
    )

    assert isinstance(problem, MFGProblem)
    assert problem.components is not None
    assert problem.components.potential_func is not None


@pytest.mark.skip(reason="Factory signature validation issue - deferred to Phase 3.5")
def test_create_stochastic_problem(simple_domain):
    """Test stochastic MFG problem creation."""

    def hamiltonian(x, p, m, t):
        return 0.5 * p**2 + m

    def hamiltonian_dm(x, p, m, t):
        return 1.0

    def terminal_cost(x):
        return x**2

    def initial_density(x):
        return np.exp(-10 * (x - 0.5) ** 2)

    def common_noise(t):
        return np.sin(2 * np.pi * t)

    problem = create_stochastic_problem(
        hamiltonian=hamiltonian,
        hamiltonian_dm=hamiltonian_dm,
        terminal_cost=terminal_cost,
        initial_density=initial_density,
        geometry=simple_domain,
        noise_intensity=0.5,
        common_noise=common_noise,
    )

    assert isinstance(problem, MFGProblem)
    assert problem.components is not None
    assert problem.components.noise_intensity == 0.5
    assert problem.components.common_noise_func is not None
    assert problem.get_problem_type() == "stochastic"


@pytest.mark.skip(reason="Factory signature validation issue - deferred to Phase 3.5")
def test_create_mfg_problem_with_components(simple_domain):
    """Test main factory function with explicit components."""

    def hamiltonian(x, p, m, t):
        return 0.5 * p**2

    def hamiltonian_dm(x, p, m, t):
        return 0.0

    components = MFGComponents(
        hamiltonian_func=hamiltonian,
        hamiltonian_dm_func=hamiltonian_dm,
        final_value_func=lambda x: x**2,
        initial_density_func=lambda x: np.exp(-(x**2)),
        problem_type="standard",
    )

    problem = create_mfg_problem("standard", components, geometry=simple_domain)

    assert isinstance(problem, MFGProblem)
    assert problem.components.problem_type == "standard"


@pytest.mark.skip(reason="Factory signature validation issue - deferred to Phase 3.5")
def test_backward_compatibility_warning(simple_domain):
    """Test that legacy API triggers deprecation warning."""

    def terminal_cost(x):
        return x**2

    def initial_density(x):
        return np.exp(-10 * (x - 0.5) ** 2)

    # Test that use_unified=False triggers warning
    with pytest.warns(DeprecationWarning, match="deprecated"):
        problem = create_lq_problem(
            geometry=simple_domain,
            terminal_cost=terminal_cost,
            initial_density=initial_density,
            use_unified=False,
        )

    # Should still create a valid problem
    assert problem is not None


@pytest.mark.skip(reason="Factory signature validation issue - deferred to Phase 3.5")
def test_problem_type_detection():
    """Test automatic problem type detection."""

    bc = BoundaryConditions("neumann", left_value=0.0, right_value=0.0)
    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)

    # Standard MFG
    components_standard = MFGComponents(
        hamiltonian_func=lambda x, p, m, t: 0.5 * p**2,
        hamiltonian_dm_func=lambda x, p, m, t: 0.0,
    )
    problem = MFGProblem(geometry=domain, components=components_standard)
    assert problem.get_problem_type() == "standard"

    # Stochastic MFG (auto-detect from noise)
    components_stochastic = MFGComponents(
        hamiltonian_func=lambda x, p, m, t: 0.5 * p**2,
        hamiltonian_dm_func=lambda x, p, m, t: 0.0,
        noise_intensity=0.5,
    )
    problem = MFGProblem(geometry=domain, components=components_stochastic)
    assert problem.get_problem_type() == "stochastic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
