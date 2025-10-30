#!/usr/bin/env python3
"""
Unit tests for mfg_pde/core/mfg_problem.py

Tests MFG problem base class including:
- MFGComponents dataclass
- MFGProblem initialization and defaults
- Default mathematical functions
- Custom components handling
- Hamiltonian methods
- Boundary conditions
- Getter methods
- MFGProblemBuilder pattern
"""

import pytest

import numpy as np

from mfg_pde.core.mfg_problem import MFGComponents, MFGProblem, MFGProblemBuilder
from mfg_pde.geometry import BoundaryConditions

# ===================================================================
# Test MFGComponents Dataclass
# ===================================================================


@pytest.mark.unit
def test_mfg_components_defaults():
    """Test MFGComponents has correct default values."""
    components = MFGComponents()

    assert components.hamiltonian_func is None
    assert components.hamiltonian_dm_func is None
    assert components.hamiltonian_jacobian_func is None
    assert components.potential_func is None
    assert components.initial_density_func is None
    assert components.final_value_func is None
    assert components.boundary_conditions is None
    assert components.coupling_func is None
    assert components.parameters == {}
    assert components.description == "MFG Problem"
    assert components.problem_type == "mfg"


@pytest.mark.unit
def test_mfg_components_custom_values():
    """Test MFGComponents with custom values."""

    def custom_h(x, m, p, t):
        return 0.5 * p**2

    def custom_dh(x, m, p, t):
        return 0.0

    components = MFGComponents(
        hamiltonian_func=custom_h,
        hamiltonian_dm_func=custom_dh,
        parameters={"param1": 1.0},
        description="Custom Problem",
        problem_type="custom",
    )

    assert components.hamiltonian_func == custom_h
    assert components.hamiltonian_dm_func == custom_dh
    assert components.parameters == {"param1": 1.0}
    assert components.description == "Custom Problem"
    assert components.problem_type == "custom"


# ===================================================================
# Test MFGProblem Initialization
# ===================================================================


@pytest.mark.unit
def test_mfg_problem_default_initialization():
    """Test MFGProblem with default parameters."""
    problem = MFGProblem()

    # Domain parameters
    assert problem.xmin == 0.0
    assert problem.xmax == 1.0
    assert problem.Lx == 1.0
    assert problem.Nx == 51
    assert problem.Dx == pytest.approx(1.0 / 51)

    # Time parameters
    assert problem.T == 1.0
    assert problem.Nt == 51
    assert problem.Dt == pytest.approx(1.0 / 51)

    # Physical parameters
    assert problem.sigma == 1.0
    assert problem.coefCT == 0.5

    # Custom components
    assert problem.components is None
    assert problem.is_custom is False


@pytest.mark.unit
def test_mfg_problem_custom_domain():
    """Test MFGProblem with custom domain parameters."""
    problem = MFGProblem(xmin=-1.0, xmax=2.0, Nx=100)

    assert problem.xmin == -1.0
    assert problem.xmax == 2.0
    assert problem.Lx == 3.0
    assert problem.Nx == 100
    assert problem.Dx == pytest.approx(3.0 / 100)
    assert len(problem.xSpace) == 101  # Nx + 1


@pytest.mark.unit
def test_mfg_problem_custom_time():
    """Test MFGProblem with custom time parameters."""
    problem = MFGProblem(T=2.0, Nt=100)

    assert problem.T == 2.0
    assert problem.Nt == 100
    assert problem.Dt == pytest.approx(2.0 / 100)
    assert len(problem.tSpace) == 101  # Nt + 1


@pytest.mark.unit
def test_mfg_problem_custom_coefficients():
    """Test MFGProblem with custom coefficients."""
    problem = MFGProblem(sigma=0.5, coefCT=0.8)

    assert problem.sigma == 0.5
    assert problem.coefCT == 0.8


@pytest.mark.unit
def test_mfg_problem_spatial_grid():
    """Test MFGProblem spatial grid generation."""
    problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=10)

    assert len(problem.xSpace) == 11  # Nx + 1 points
    assert problem.xSpace[0] == 0.0
    assert problem.xSpace[-1] == 1.0
    # Check uniform spacing
    spacing = np.diff(problem.xSpace)
    assert np.allclose(spacing, spacing[0])


@pytest.mark.unit
def test_mfg_problem_temporal_grid():
    """Test MFGProblem temporal grid generation."""
    problem = MFGProblem(T=1.0, Nt=10)

    assert len(problem.tSpace) == 11  # Nt + 1 points
    assert problem.tSpace[0] == 0.0
    assert problem.tSpace[-1] == 1.0
    # Check uniform spacing
    spacing = np.diff(problem.tSpace)
    assert np.allclose(spacing, spacing[0])


# ===================================================================
# Test Default Functions
# ===================================================================


@pytest.mark.unit
def test_mfg_problem_default_potential():
    """Test default potential function is initialized."""
    problem = MFGProblem(Nx=10)

    assert hasattr(problem, "f_potential")
    assert isinstance(problem.f_potential, np.ndarray)
    assert len(problem.f_potential) == 11  # Nx + 1
    # Check potential is non-zero (has spatial variation)
    assert not np.allclose(problem.f_potential, 0.0)


@pytest.mark.unit
def test_mfg_problem_default_final_value():
    """Test default final value function is initialized."""
    problem = MFGProblem(Nx=10)

    assert hasattr(problem, "u_fin")
    assert isinstance(problem.u_fin, np.ndarray)
    assert len(problem.u_fin) == 11  # Nx + 1
    # Check final value is non-zero
    assert not np.allclose(problem.u_fin, 0.0)


@pytest.mark.unit
def test_mfg_problem_default_initial_density():
    """Test default initial density function is initialized."""
    problem = MFGProblem(Nx=10)

    assert hasattr(problem, "m_init")
    assert isinstance(problem.m_init, np.ndarray)
    assert len(problem.m_init) == 11  # Nx + 1
    # Check initial density is non-negative
    assert np.all(problem.m_init >= 0.0)
    # Check it has some mass
    assert np.sum(problem.m_init) > 0.0


# ===================================================================
# Test Custom Components
# ===================================================================


@pytest.mark.unit
def test_mfg_problem_with_custom_potential():
    """Test MFGProblem with custom potential function."""

    def custom_potential(x, t):
        return x**2

    components = MFGComponents(potential_func=custom_potential)
    problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=10, components=components)

    assert problem.is_custom is True
    assert problem.components is not None
    # Check potential was set using custom function
    expected = problem.xSpace**2
    assert np.allclose(problem.f_potential, expected)


@pytest.mark.unit
def test_mfg_problem_with_custom_initial_density():
    """Test MFGProblem with custom initial density function."""

    def custom_initial(x):
        return np.exp(-10 * (x - 0.5) ** 2)

    components = MFGComponents(initial_density_func=custom_initial)
    problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=10, components=components)

    assert problem.is_custom is True
    # Check initial density was set using custom function and normalized
    expected_unnormalized = np.exp(-10 * (problem.xSpace - 0.5) ** 2)
    # Normalize expected (same way as in MFGProblem.__init__)
    integral = np.sum(expected_unnormalized) * problem.Dx
    expected = expected_unnormalized / integral
    assert np.allclose(problem.m_init, expected)


@pytest.mark.unit
def test_mfg_problem_with_custom_final_value():
    """Test MFGProblem with custom final value function."""

    def custom_final(x):
        return np.sin(x * np.pi)

    components = MFGComponents(final_value_func=custom_final)
    problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=10, components=components)

    assert problem.is_custom is True
    # Check final value was set using custom function
    expected = np.sin(problem.xSpace * np.pi)
    assert np.allclose(problem.u_fin, expected)


# ===================================================================
# Test Hamiltonian Methods
# ===================================================================


@pytest.mark.unit
def test_hamiltonian_h_default():
    """Test default Hamiltonian H computation."""
    problem = MFGProblem(Nx=10, sigma=1.0, coefCT=0.5)

    x_idx = 5
    m_at_x = 1.0
    p_values = {"forward": 0.1, "backward": 0.1}
    t_idx = 5

    H_val = problem.H(x_idx=x_idx, m_at_x=m_at_x, p_values=p_values, t_idx=t_idx)

    # Should return a finite value
    assert np.isfinite(H_val)
    assert isinstance(H_val, (float, np.floating))


@pytest.mark.unit
def test_hamiltonian_dh_dm_default():
    """Test default Hamiltonian derivative dH/dm."""
    problem = MFGProblem(Nx=10)

    x_idx = 5
    m_at_x = 1.0
    p_values = {"forward": 0.1, "backward": 0.1}
    t_idx = 5

    dH_val = problem.dH_dm(x_idx=x_idx, m_at_x=m_at_x, p_values=p_values, t_idx=t_idx)

    # Should return a finite value
    assert np.isfinite(dH_val)
    assert isinstance(dH_val, (float, np.floating))


@pytest.mark.unit
def test_mfg_problem_custom_hamiltonian():
    """Test MFGProblem with custom Hamiltonian."""

    def custom_H(x_idx, m_at_x, p_values, t_idx, **kwargs):
        """Custom Hamiltonian function with correct signature."""
        p_forward = p_values.get("forward", 0.0)
        p_backward = p_values.get("backward", 0.0)
        return p_forward**2 + p_backward**2 + m_at_x**2

    def custom_dH_dm(x_idx, m_at_x, p_values, t_idx, **kwargs):
        """Custom Hamiltonian derivative with correct signature."""
        return 2.0 * m_at_x

    components = MFGComponents(hamiltonian_func=custom_H, hamiltonian_dm_func=custom_dH_dm)
    problem = MFGProblem(components=components)

    # Should use custom Hamiltonian
    assert problem.is_custom is True
    # Use symmetric p_values (conversion to tuple notation preserves symmetry)
    H_value = problem.H(x_idx=10, m_at_x=0.5, p_values={"forward": 0.15, "backward": 0.15}, t_idx=0)
    assert isinstance(H_value, float)
    assert H_value == pytest.approx(0.15**2 + 0.15**2 + 0.5**2)  # = 0.295


# ===================================================================
# Test Boundary Conditions
# ===================================================================


@pytest.mark.unit
def test_get_boundary_conditions_default():
    """Test get_boundary_conditions returns default periodic BC."""
    problem = MFGProblem()

    bc = problem.get_boundary_conditions()

    assert isinstance(bc, BoundaryConditions)
    assert bc.type == "periodic"


@pytest.mark.unit
def test_get_boundary_conditions_custom():
    """Test get_boundary_conditions with custom BC."""
    custom_bc = BoundaryConditions(type="dirichlet", left_value=0.0, right_value=0.0)
    components = MFGComponents(boundary_conditions=custom_bc)
    problem = MFGProblem(components=components)

    bc = problem.get_boundary_conditions()

    assert isinstance(bc, BoundaryConditions)
    assert bc.type == "dirichlet"


# ===================================================================
# Test Getter Methods
# ===================================================================


@pytest.mark.unit
def test_get_potential_at_time():
    """Test get_potential_at_time returns array."""
    problem = MFGProblem(Nx=10, Nt=20)

    potential = problem.get_potential_at_time(t_idx=5)

    assert isinstance(potential, np.ndarray)
    assert len(potential) == 11  # Nx + 1
    # Should match stored potential
    assert np.allclose(potential, problem.f_potential)


@pytest.mark.unit
def test_get_final_u():
    """Test get_final_u returns final value function."""
    problem = MFGProblem(Nx=10)

    u_final = problem.get_final_u()

    assert isinstance(u_final, np.ndarray)
    assert len(u_final) == 11  # Nx + 1
    assert np.allclose(u_final, problem.u_fin)


@pytest.mark.unit
def test_get_initial_m():
    """Test get_initial_m returns initial density."""
    problem = MFGProblem(Nx=10)

    m_initial = problem.get_initial_m()

    assert isinstance(m_initial, np.ndarray)
    assert len(m_initial) == 11  # Nx + 1
    assert np.allclose(m_initial, problem.m_init)


@pytest.mark.unit
def test_get_problem_info():
    """Test get_problem_info returns comprehensive info dict."""
    problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=100, sigma=0.5, coefCT=0.8)

    info = problem.get_problem_info()

    assert isinstance(info, dict)
    # Check key fields are present
    assert "domain" in info
    assert "time" in info
    assert "coefficients" in info
    assert "problem_type" in info

    # Check values
    assert info["domain"]["xmin"] == 0.0
    assert info["domain"]["xmax"] == 1.0
    assert info["domain"]["Nx"] == 50
    assert info["time"]["T"] == 1.0
    assert info["time"]["Nt"] == 100
    assert info["coefficients"]["sigma"] == 0.5
    assert info["coefficients"]["coefCT"] == 0.8


# ===================================================================
# Test MFGProblemBuilder
# ===================================================================


@pytest.mark.unit
def test_problem_builder_basic():
    """Test MFGProblemBuilder basic usage without Hamiltonian (default problem)."""
    builder = MFGProblemBuilder()
    problem = builder.domain(0.0, 1.0, 50).time(1.0, 100).coefficients(sigma=0.5).build()

    assert isinstance(problem, MFGProblem)
    assert problem.xmin == 0.0
    assert problem.xmax == 1.0
    assert problem.Nx == 50
    assert problem.T == 1.0
    assert problem.Nt == 100
    assert problem.sigma == 0.5
    # Default problem (no custom components) - should work
    assert problem.is_custom is False


@pytest.mark.unit
def test_problem_builder_hamiltonian():
    """Test MFGProblemBuilder with custom Hamiltonian."""

    def custom_h(x_idx, m_at_x, p_values, t_idx, **kwargs):
        p_forward = p_values.get("forward", 0.0)
        p_backward = p_values.get("backward", 0.0)
        return p_forward**2 + p_backward**2

    def custom_dh(x_idx, m_at_x, p_values, t_idx, **kwargs):
        return 0.0

    builder = MFGProblemBuilder()
    problem = builder.hamiltonian(custom_h, custom_dh).build()

    assert isinstance(problem, MFGProblem)
    assert problem.is_custom is True
    assert problem.components.hamiltonian_func == custom_h
    assert problem.components.hamiltonian_dm_func == custom_dh


@pytest.mark.unit
def test_problem_builder_potential():
    """Test MFGProblemBuilder with custom potential (no Hamiltonian needed)."""

    def custom_potential(x, t):
        return x**2

    builder = MFGProblemBuilder()
    problem = builder.potential(custom_potential).build()

    assert isinstance(problem, MFGProblem)
    # Custom potential makes it a custom problem, but without Hamiltonian it uses defaults
    assert problem.is_custom is True
    assert problem.components.potential_func == custom_potential


@pytest.mark.unit
def test_problem_builder_chaining():
    """Test MFGProblemBuilder method chaining (without Hamiltonian uses defaults)."""

    def custom_initial(x):
        return np.exp(-10 * x**2)

    def custom_final(x):
        return np.sin(x)

    builder = MFGProblemBuilder()
    problem = (
        builder.domain(-1.0, 1.0, 100)
        .time(2.0, 200)
        .coefficients(sigma=0.8, coefCT=0.3)
        .initial_density(custom_initial)
        .final_value(custom_final)
        .description("Test Problem", "test")
        .build()
    )

    assert isinstance(problem, MFGProblem)
    assert problem.xmin == -1.0
    assert problem.xmax == 1.0
    assert problem.Nx == 100
    assert problem.T == 2.0
    assert problem.Nt == 200
    assert problem.sigma == 0.8
    assert problem.coefCT == 0.3
    assert problem.is_custom is True
    assert problem.components.description == "Test Problem"


@pytest.mark.unit
def test_problem_builder_boundary_conditions():
    """Test MFGProblemBuilder with custom boundary conditions (no Hamiltonian needed)."""
    custom_bc = BoundaryConditions(type="neumann", left_value=0.0, right_value=0.0)
    builder = MFGProblemBuilder()
    problem = builder.boundary_conditions(custom_bc).build()

    assert isinstance(problem, MFGProblem)
    assert problem.is_custom is True
    bc = problem.get_boundary_conditions()
    assert bc.type == "neumann"


@pytest.mark.unit
def test_problem_builder_parameters():
    """Test MFGProblemBuilder with custom parameters."""
    builder = MFGProblemBuilder()
    problem = builder.parameters(alpha=1.5, beta=0.3).build()

    assert isinstance(problem, MFGProblem)
    assert problem.is_custom is True
    assert "alpha" in problem.components.parameters
    assert problem.components.parameters["alpha"] == 1.5
    assert problem.components.parameters["beta"] == 0.3


# ===================================================================
# Test Edge Cases
# ===================================================================


@pytest.mark.unit
def test_mfg_problem_zero_nx():
    """Test MFGProblem handles Nx=0 gracefully."""
    problem = MFGProblem(Nx=0)

    assert problem.Nx == 0
    assert problem.Dx == 0.0
    assert len(problem.xSpace) == 1


@pytest.mark.unit
def test_mfg_problem_zero_nt():
    """Test MFGProblem handles Nt=0 gracefully."""
    problem = MFGProblem(Nt=0)

    assert problem.Nt == 0
    assert problem.Dt == 0.0
    assert len(problem.tSpace) == 1


# ===================================================================
# Test Module Exports
# ===================================================================


@pytest.mark.unit
def test_module_exports_classes():
    """Test module exports main classes."""
    from mfg_pde.core import mfg_problem

    assert hasattr(mfg_problem, "MFGComponents")
    assert hasattr(mfg_problem, "MFGProblem")
    assert hasattr(mfg_problem, "MFGProblemBuilder")


@pytest.mark.unit
def test_module_exports_are_classes():
    """Test exported objects are actually classes."""
    assert isinstance(MFGComponents, type)
    assert isinstance(MFGProblem, type)
    assert isinstance(MFGProblemBuilder, type)
