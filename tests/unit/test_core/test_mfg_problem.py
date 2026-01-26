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
"""

import pytest

import numpy as np

from mfg_pde.core.mfg_problem import MFGComponents, MFGProblem
from mfg_pde.geometry import TensorProductGrid

# Unified BC from conditions.py (current API)
from mfg_pde.geometry.boundary.conditions import BoundaryConditions, no_flux_bc

# Legacy 1D BC: testing compatibility with 1D MFG problems (deprecated in v0.14, remove in v1.0)
from mfg_pde.geometry.boundary.fdm_bc_1d import BoundaryConditions as LegacyBoundaryConditions

# ===========================================================================
# Test Helpers - Issue #670: m_initial/u_final now required in MFGComponents
# ===========================================================================


def default_geometry(bounds=None, Nx_points=None, dimension=1):
    """Create a default geometry with explicit boundary conditions."""
    if bounds is None:
        bounds = [(0.0, 1.0)] if dimension == 1 else [(0.0, 1.0)] * dimension
    if Nx_points is None:
        Nx_points = [11] if dimension == 1 else [11] * dimension
    return TensorProductGrid(
        bounds=bounds,
        Nx_points=Nx_points,
        boundary_conditions=no_flux_bc(dimension=dimension),
    )


def default_hamiltonian():
    """Create a default class-based Hamiltonian for testing.

    Issue #673: Hamiltonian required (no default in MFGComponents)

    Returns:
        SeparableHamiltonian: H = ½|p|²/λ - m²
    """
    from mfg_pde.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian

    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: -(m**2),
        coupling_dm=lambda m: -2 * m,
    )


def default_components():
    """Create default MFGComponents with Hamiltonian, m_initial, and u_final.

    Issue #670: m_initial/u_final required
    Issue #673: Hamiltonian required (no default)
    """
    return MFGComponents(
        hamiltonian=default_hamiltonian(),
        m_initial=lambda x: np.exp(-10 * (x - 0.5) ** 2),  # Gaussian at center
        u_final=lambda x: x**2,  # Quadratic terminal cost
    )


def create_test_problem(**kwargs):
    """Create a test MFGProblem with required m_initial/u_final (Issue #670).

    Uses default geometry and components if not provided.
    """
    if "geometry" not in kwargs:
        kwargs["geometry"] = default_geometry()
    if "components" not in kwargs:
        kwargs["components"] = default_components()
    return MFGProblem(**kwargs)


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
    assert components.m_initial is None
    assert components.u_final is None
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
    """Test MFGProblem with default parameters (Issue #670: requires m_initial/u_final)."""
    problem = create_test_problem()

    # Domain parameters (default_geometry uses Nx=11, so 10 intervals)
    assert problem.xmin == 0.0
    assert problem.xmax == 1.0
    assert problem.Lx == 1.0
    assert problem.geometry.get_grid_shape()[0] == 11  # 11 points

    # Time parameters
    assert problem.T == 1.0
    assert problem.Nt == 51
    assert problem.dt == pytest.approx(1.0 / 51)

    # Physical parameters (None → 0 for diffusion/drift)
    assert problem.sigma == 0.0
    assert problem.diffusion_field == 0.0
    assert problem.drift_field == 0.0
    assert problem.coupling_coefficient == 0.5

    # Custom components (now required)
    assert problem.components is not None  # Must have components now
    assert problem.is_custom is True  # Has custom m_initial/u_final


@pytest.mark.unit
def test_mfg_problem_custom_domain():
    """Test MFGProblem with custom domain parameters."""
    geometry = default_geometry(bounds=[(-1.0, 2.0)], Nx_points=[101])  # Nx=100 intervals
    problem = create_test_problem(geometry=geometry)

    assert problem.xmin == -1.0
    assert problem.xmax == 2.0
    assert problem.Lx == 3.0
    assert problem.geometry.get_grid_shape()[0] - 1 == 100  # Nx intervals
    assert problem.geometry.get_grid_spacing()[0] == pytest.approx(3.0 / 100)
    assert problem.geometry.get_grid_shape()[0] == 101  # Nx+1 points


@pytest.mark.unit
def test_mfg_problem_custom_time():
    """Test MFGProblem with custom time parameters."""
    problem = create_test_problem(T=2.0, Nt=100)

    assert problem.T == 2.0
    assert problem.Nt == 100
    assert problem.dt == pytest.approx(2.0 / 100)
    assert len(problem.tSpace) == 101  # Nt + 1


@pytest.mark.unit
def test_mfg_problem_custom_coefficients():
    """Test MFGProblem with custom coefficients."""
    problem = create_test_problem(diffusion=0.5, coupling_coefficient=0.8)

    assert problem.sigma == 0.5
    assert problem.coupling_coefficient == 0.8


@pytest.mark.unit
def test_mfg_problem_spatial_grid():
    """Test MFGProblem spatial grid generation."""
    geometry = default_geometry(bounds=[(0.0, 1.0)], Nx_points=[11])  # Nx=10 intervals
    problem = create_test_problem(geometry=geometry)

    assert problem.geometry.get_grid_shape()[0] == 11  # Nx+1 points
    assert problem.xSpace[0] == 0.0
    assert problem.xSpace[-1] == 1.0
    # Check uniform spacing
    spacing = np.diff(problem.xSpace)
    assert np.allclose(spacing, spacing[0])


@pytest.mark.unit
def test_mfg_problem_temporal_grid():
    """Test MFGProblem temporal grid generation."""
    problem = create_test_problem(T=1.0, Nt=10)

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
    """Test default potential function is initialized (Issue #670: requires m_initial/u_final)."""
    # Issue #671 will address: potential defaults to zero if not provided
    problem = create_test_problem()

    assert hasattr(problem, "f_potential")
    assert isinstance(problem.f_potential, np.ndarray)
    assert len(problem.f_potential) == problem.geometry.get_grid_shape()[0]  # Nx+1 points
    # Potential defaults to zero when not provided (Issue #671 will change this)
    assert np.allclose(problem.f_potential, 0.0)


@pytest.mark.unit
def test_mfg_problem_default_final_value():
    """Test final value function is initialized (Issue #670: must come from MFGComponents)."""
    problem = create_test_problem()

    assert hasattr(problem, "u_final")
    assert isinstance(problem.u_final, np.ndarray)
    assert len(problem.u_final) == problem.geometry.get_grid_shape()[0]  # Nx+1 points
    # Check final value is non-zero (set from default_components: x**2)
    assert not np.allclose(problem.u_final, 0.0)


@pytest.mark.unit
def test_mfg_problem_default_initial_density():
    """Test initial density function is initialized (Issue #670: must come from MFGComponents)."""
    problem = create_test_problem()

    assert hasattr(problem, "m_initial")
    assert isinstance(problem.m_initial, np.ndarray)
    assert len(problem.m_initial) == problem.geometry.get_grid_shape()[0]  # Nx+1 points
    # Check initial density is non-negative
    assert np.all(problem.m_initial >= 0.0)
    # Check it has some mass
    assert np.sum(problem.m_initial) > 0.0


# ===================================================================
# Test Custom Components
# ===================================================================


@pytest.mark.unit
def test_mfg_problem_with_custom_potential():
    """Test MFGProblem with custom potential function."""

    def custom_potential(x, t):
        return x**2

    geometry = default_geometry(bounds=[(0.0, 1.0)], Nx_points=[11])
    # Issue #670: must provide m_initial and u_final
    # Issue #673: Hamiltonian required
    components = MFGComponents(
        hamiltonian=default_hamiltonian(),
        potential_func=custom_potential,
        m_initial=lambda x: 1.0,  # Uniform
        u_final=lambda x: 0.0,  # Zero terminal cost
    )
    problem = MFGProblem(geometry=geometry, components=components)

    assert problem.is_custom is True
    assert problem.components is not None
    # Check potential was set using custom function
    expected = problem.xSpace**2
    # Flatten both arrays for comparison (problem stores as 2D column vector)
    assert np.allclose(np.ravel(problem.f_potential), np.ravel(expected))


@pytest.mark.unit
def test_mfg_problem_with_custom_initial_density():
    """Test MFGProblem with custom initial density function."""

    def custom_initial(x):
        return np.exp(-10 * (x - 0.5) ** 2)

    geometry = default_geometry(bounds=[(0.0, 1.0)], Nx_points=[11])
    # Issue #670: must provide both m_initial and u_final
    # Issue #673: Hamiltonian required
    components = MFGComponents(
        hamiltonian=default_hamiltonian(),
        m_initial=custom_initial,
        u_final=lambda x: 0.0,  # Zero terminal cost
    )
    problem = MFGProblem(geometry=geometry, components=components)

    assert problem.is_custom is True
    # Check initial density was set using custom function and normalized
    expected_unnormalized = np.exp(-10 * (problem.xSpace - 0.5) ** 2)
    # Normalize expected (same way as in MFGProblem.__init__)
    integral = np.sum(expected_unnormalized) * problem.geometry.get_grid_spacing()[0]
    expected = expected_unnormalized / integral
    # Flatten both arrays for comparison (problem stores as 2D column vector)
    assert np.allclose(np.ravel(problem.m_initial), np.ravel(expected))


@pytest.mark.unit
def test_mfg_problem_with_custom_final_value():
    """Test MFGProblem with custom final value function."""

    def custom_final(x):
        return np.sin(x * np.pi)

    geometry = default_geometry(bounds=[(0.0, 1.0)], Nx_points=[11])
    # Issue #670: must provide both m_initial and u_final
    # Issue #673: Hamiltonian required
    components = MFGComponents(
        hamiltonian=default_hamiltonian(),
        u_final=custom_final,
        m_initial=lambda x: 1.0,  # Uniform
    )
    problem = MFGProblem(geometry=geometry, components=components)

    assert problem.is_custom is True
    # Check final value was set using custom function
    expected = np.sin(problem.xSpace * np.pi)
    # Flatten both arrays for comparison (problem stores as 2D column vector)
    assert np.allclose(np.ravel(problem.u_final), np.ravel(expected))


@pytest.mark.unit
def test_mfg_problem_validates_negative_m_initial():
    """Test that negative m_initial raises ValueError (Issue #672: Fail Fast)."""
    geometry = default_geometry()
    # Invalid: negative density
    components = MFGComponents(
        m_initial=lambda x: x - 0.5,  # Negative for x < 0.5
        u_final=lambda x: 0.0,
    )

    with pytest.raises(ValueError, match="m_initial contains negative values"):
        MFGProblem(geometry=geometry, components=components)


@pytest.mark.unit
def test_mfg_problem_validates_zero_mass_m_initial():
    """Test that zero-mass m_initial raises ValueError (Issue #672: Fail Fast)."""
    geometry = default_geometry()
    # Invalid: zero everywhere
    components = MFGComponents(
        m_initial=lambda x: 0.0,  # Zero mass
        u_final=lambda x: 0.0,
    )

    with pytest.raises(ValueError, match="m_initial has zero or negligible total mass"):
        MFGProblem(geometry=geometry, components=components)


# ===================================================================
# Test Hamiltonian Methods
# ===================================================================


@pytest.mark.unit
def test_hamiltonian_h_default():
    """Test default Hamiltonian H computation."""
    problem = create_test_problem(diffusion=1.0, coupling_coefficient=0.5)

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
    problem = create_test_problem()

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

    # Issue #670: must provide m_initial and u_final
    components = MFGComponents(
        hamiltonian_func=custom_H,
        hamiltonian_dm_func=custom_dH_dm,
        m_initial=lambda x: 1.0,
        u_final=lambda x: 0.0,
    )
    problem = create_test_problem(components=components)

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
    """Test get_boundary_conditions returns default no_flux BC from geometry."""
    problem = create_test_problem()

    bc = problem.get_boundary_conditions()

    # Default BC from geometry is no_flux (mass-conserving)
    assert isinstance(bc, BoundaryConditions)
    assert bc.type == "no_flux"


@pytest.mark.unit
def test_get_boundary_conditions_custom():
    """Test get_boundary_conditions: geometry BC takes priority (Issue #674 SSOT).

    Note: With geometry-first API (Issue #674), the geometry's BC takes priority
    over components BC. This test verifies this priority order.
    """
    # Uses legacy 1D BC in components (lower priority)
    custom_bc = LegacyBoundaryConditions(type="dirichlet", left_value=0.0, right_value=0.0)
    # Issue #670: must provide m_initial and u_final
    # Issue #673: Hamiltonian required
    components = MFGComponents(
        hamiltonian=default_hamiltonian(),
        boundary_conditions=custom_bc,
        m_initial=lambda x: 1.0,
        u_final=lambda x: 0.0,
    )
    # create_test_problem uses default_geometry which has no_flux_bc
    problem = create_test_problem(components=components)

    bc = problem.get_boundary_conditions()

    # Issue #674: Geometry BC takes priority over components BC
    # The geometry has no_flux BC (from default_geometry), so that's what we get
    assert isinstance(bc, BoundaryConditions)
    assert bc.type == "no_flux"


# ===================================================================
# Test Getter Methods
# ===================================================================


@pytest.mark.unit
def test_get_potential_at_time():
    """Test get_potential_at_time returns array."""
    problem = create_test_problem(Nt=20)

    potential = problem.get_potential_at_time(t_idx=5)

    assert isinstance(potential, np.ndarray)
    assert len(potential) == problem.geometry.get_grid_shape()[0]  # Nx+1 points
    # Should match stored potential
    assert np.allclose(potential, problem.f_potential)


@pytest.mark.unit
def test_get_final_u():
    """Test get_final_u returns final value function."""
    problem = create_test_problem()

    u_final = problem.get_final_u()

    assert isinstance(u_final, np.ndarray)
    assert len(u_final) == problem.geometry.get_grid_shape()[0]  # Nx+1 points
    assert np.allclose(u_final, problem.u_final)


@pytest.mark.unit
def test_get_initial_m():
    """Test get_initial_m returns initial density."""
    problem = create_test_problem()

    m_initial = problem.get_initial_m()

    assert isinstance(m_initial, np.ndarray)
    assert len(m_initial) == problem.geometry.get_grid_shape()[0]  # Nx+1 points
    assert np.allclose(m_initial, problem.m_initial)


@pytest.mark.unit
def test_get_problem_info():
    """Test get_problem_info returns comprehensive info dict."""
    geometry = default_geometry(bounds=[(0.0, 1.0)], Nx_points=[51])  # Nx=50 intervals
    problem = create_test_problem(geometry=geometry, T=1.0, Nt=100, diffusion=0.5, coupling_coefficient=0.8)

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
    assert info["domain"]["Nx"] == 50  # Intervals, from legacy info
    assert info["time"]["T"] == 1.0
    assert info["time"]["Nt"] == 100
    assert info["coefficients"]["sigma"] == 0.5
    assert info["coefficients"]["coupling_coefficient"] == 0.8


# ===================================================================
# Test Edge Cases
# ===================================================================


@pytest.mark.unit
@pytest.mark.skip(
    reason="Test deprecated behavior. New geometry validation (TensorProductGrid) correctly "
    "rejects Nx=0 because at least 2 points are needed to define a grid with spacing. "
    "Legacy API behavior is no longer supported."
)
def test_mfg_problem_zero_nx():
    """Test MFGProblem handles Nx=0 gracefully."""
    geometry = default_geometry(bounds=[(0.0, 1.0)], Nx_points=[1])
    problem = create_test_problem(geometry=geometry)

    assert problem.Nx == 0
    assert problem.dx == 0.0
    assert len(problem.xSpace) == 1


@pytest.mark.unit
def test_mfg_problem_zero_nt():
    """Test MFGProblem handles Nt=0 gracefully."""
    problem = create_test_problem(Nt=0)

    assert problem.Nt == 0
    assert problem.dt == 0.0
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


@pytest.mark.unit
def test_module_exports_are_classes():
    """Test exported objects are actually classes."""
    assert isinstance(MFGComponents, type)
    assert isinstance(MFGProblem, type)


# ===================================================================
# Test Dual Geometry Support (Issue #257 Phase 3)
# ===================================================================


@pytest.mark.unit
def test_dual_geometry_specification():
    """Test MFGProblem with separate HJB and FP geometries (Issue #257)."""
    # Create two different 2D grids with BC
    hjb_grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[51, 51],
        boundary_conditions=no_flux_bc(dimension=2),
    )
    fp_grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[21, 21],
        boundary_conditions=no_flux_bc(dimension=2),
    )

    # Create 2D components
    components = MFGComponents(
        hamiltonian=default_hamiltonian(),
        m_initial=lambda x: 1.0,
        u_final=lambda x: 0.0,
    )
    # Create problem with dual geometries
    problem = MFGProblem(
        hjb_geometry=hjb_grid,
        fp_geometry=fp_grid,
        time_domain=(1.0, 50),
        diffusion=0.1,
        components=components,
    )

    # Check that both geometries are stored
    assert problem.hjb_geometry is hjb_grid
    assert problem.fp_geometry is fp_grid

    # Check that geometry projector was created
    assert problem.geometry_projector is not None
    assert problem.geometry_projector.hjb_geometry is hjb_grid
    assert problem.geometry_projector.fp_geometry is fp_grid


@pytest.mark.unit
def test_dual_geometry_backward_compatibility():
    """Test that unified geometry mode still works (backward compatibility)."""
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[31, 31],
        boundary_conditions=no_flux_bc(dimension=2),
    )

    components = MFGComponents(
        hamiltonian=default_hamiltonian(),
        m_initial=lambda x: 1.0,
        u_final=lambda x: 0.0,
    )
    # Create problem with unified geometry (old API)
    problem = MFGProblem(geometry=grid, time_domain=(1.0, 50), diffusion=0.1, components=components)

    # Check that both hjb_geometry and fp_geometry point to the same geometry
    assert problem.hjb_geometry is grid
    assert problem.fp_geometry is grid

    # Check that no projector is created for unified mode
    assert problem.geometry_projector is None


@pytest.mark.unit
def test_dual_geometry_error_on_partial_specification():
    """Test that specifying only one of hjb_geometry/fp_geometry raises error."""
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[31, 31],
        boundary_conditions=no_flux_bc(dimension=2),
    )

    components = MFGComponents(hamiltonian=default_hamiltonian(), m_initial=lambda x: 1.0, u_final=lambda x: 0.0)
    # Test with only hjb_geometry
    with pytest.raises(ValueError, match="both 'hjb_geometry' AND 'fp_geometry' must be specified"):
        MFGProblem(hjb_geometry=grid, time_domain=(1.0, 50), components=components)

    # Test with only fp_geometry
    with pytest.raises(ValueError, match="both 'hjb_geometry' AND 'fp_geometry' must be specified"):
        MFGProblem(fp_geometry=grid, time_domain=(1.0, 50), components=components)


@pytest.mark.unit
def test_dual_geometry_error_on_conflict():
    """Test that specifying both geometry and dual geometries raises error."""
    grid1 = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[31, 31],
        boundary_conditions=no_flux_bc(dimension=2),
    )
    grid2 = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[21, 21],
        boundary_conditions=no_flux_bc(dimension=2),
    )
    grid3 = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[11, 11],
        boundary_conditions=no_flux_bc(dimension=2),
    )

    components = MFGComponents(hamiltonian=default_hamiltonian(), m_initial=lambda x: 1.0, u_final=lambda x: 0.0)
    # Test conflict: can't specify both geometry and dual geometries
    with pytest.raises(ValueError, match=r"Specify EITHER 'geometry'.*OR.*'hjb_geometry', 'fp_geometry'"):
        MFGProblem(geometry=grid1, hjb_geometry=grid2, fp_geometry=grid3, time_domain=(1.0, 50), components=components)


@pytest.mark.unit
def test_dual_geometry_projector_attributes():
    """Test that geometry projector has correct attributes."""
    hjb_grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[51, 51],
        boundary_conditions=no_flux_bc(dimension=2),
    )
    fp_grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[31, 31],
        boundary_conditions=no_flux_bc(dimension=2),
    )

    components = MFGComponents(hamiltonian=default_hamiltonian(), m_initial=lambda x: 1.0, u_final=lambda x: 0.0)
    problem = MFGProblem(hjb_geometry=hjb_grid, fp_geometry=fp_grid, time_domain=(1.0, 50), components=components)

    projector = problem.geometry_projector

    # Check that projector has the right geometries
    assert projector.hjb_geometry is hjb_grid
    assert projector.fp_geometry is fp_grid

    # Check that projector has selected appropriate methods
    assert projector.hjb_to_fp_method in ["grid_interpolation", "interpolation", "registry"]
    assert projector.fp_to_hjb_method in ["grid_restriction", "nearest", "registry"]


@pytest.mark.unit
def test_dual_geometry_with_1d_grids():
    """Test dual geometry with 1D grids."""
    # Create two 1D grids with different resolutions
    hjb_grid = default_geometry(bounds=[(0.0, 1.0)], Nx_points=[101])  # Fine grid
    fp_grid = default_geometry(bounds=[(0.0, 1.0)], Nx_points=[51])  # Coarse grid

    components = MFGComponents(hamiltonian=default_hamiltonian(), m_initial=lambda x: 1.0, u_final=lambda x: 0.0)
    problem = MFGProblem(
        hjb_geometry=hjb_grid, fp_geometry=fp_grid, time_domain=(1.0, 50), diffusion=0.1, components=components
    )

    # Verify dual geometry setup
    assert problem.hjb_geometry is hjb_grid
    assert problem.fp_geometry is fp_grid
    assert problem.geometry_projector is not None


@pytest.mark.unit
def test_dual_geometry_legacy_mode_compatibility():
    """Test that legacy 1D mode sets hjb_geometry and fp_geometry correctly."""
    # Legacy mode creates its own grid internally
    geometry = default_geometry(bounds=[(0.0, 1.0)], Nx_points=[101])  # Nx=100 intervals
    problem = create_test_problem(geometry=geometry, T=1.0, Nt=50)

    # Check that hjb_geometry and fp_geometry are set (to the same unified geometry)
    assert problem.hjb_geometry is not None
    assert problem.fp_geometry is not None
    assert problem.hjb_geometry is problem.fp_geometry  # Unified mode
    assert problem.geometry_projector is None  # No projector for unified mode


# ===================================================================
# Test DiffusionField Support (Feature: diffusion-field-support)
# ===================================================================


@pytest.mark.unit
def test_diffusion_field_none():
    """Test MFGProblem with no diffusion (deterministic). None → 0."""
    problem = create_test_problem(diffusion=None)

    assert problem.diffusion_field == 0.0
    assert problem.sigma == 0.0
    assert not problem.has_state_dependent_coefficients()


@pytest.mark.unit
def test_diffusion_field_scalar():
    """Test MFGProblem with scalar diffusion coefficient."""
    problem = create_test_problem(diffusion=0.5)

    assert problem.sigma == 0.5
    assert problem.diffusion_field == 0.5
    assert not problem.has_state_dependent_coefficients()


@pytest.mark.unit
def test_diffusion_field_array():
    """Test MFGProblem with array diffusion coefficient (spatially varying)."""
    # Create a spatially varying diffusion array (11 points from default_geometry)
    sigma_array = np.linspace(0.1, 1.0, 11)

    problem = create_test_problem(sigma=sigma_array)

    # Array should be stored in diffusion_field
    assert isinstance(problem.diffusion_field, np.ndarray)
    assert np.array_equal(problem.diffusion_field, sigma_array)

    # Scalar sigma should be the mean for backward compatibility
    assert problem.sigma == pytest.approx(np.mean(sigma_array))
    assert not problem.has_state_dependent_coefficients()


@pytest.mark.unit
def test_diffusion_field_callable():
    """Test MFGProblem with callable diffusion coefficient (state-dependent)."""

    def sigma_func(t, x, m):
        """State-dependent diffusion: higher diffusion in high-density regions."""
        return 0.1 + 0.5 * m

    problem = create_test_problem(sigma=sigma_func)

    # Callable should be stored in diffusion_field
    assert callable(problem.diffusion_field)
    assert problem.diffusion_field is sigma_func

    # Scalar sigma should default to 1.0 for callable
    assert problem.sigma == 1.0
    assert problem.has_state_dependent_coefficients()


@pytest.mark.unit
def test_diffusion_primary_parameter():
    """Test that 'diffusion' is the primary parameter."""
    problem = create_test_problem(diffusion=0.3)

    assert problem.sigma == 0.3
    assert problem.diffusion_field == 0.3


@pytest.mark.unit
def test_sigma_deprecated_alias():
    """Test that 'sigma' parameter is deprecated alias for 'diffusion'."""
    import warnings

    # Using sigma= should emit a deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        problem = create_test_problem(sigma=0.4)  # Deprecated parameter

        # Check deprecation warning was raised for using sigma=
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1, "Expected deprecation warning when using sigma= parameter"
        # At least one warning should mention sigma or diffusion
        warning_messages = [str(w.message) for w in deprecation_warnings]
        assert any("sigma" in msg or "diffusion" in msg for msg in warning_messages)

    # Value should still work via backward compat
    assert problem.sigma == 0.4
    assert problem.diffusion_field == 0.4


@pytest.mark.unit
def test_drift_field_none():
    """Test MFGProblem with no drift field (default). None → 0."""
    problem = create_test_problem()

    assert problem.drift_field == 0.0
    assert not problem.has_state_dependent_coefficients()


@pytest.mark.unit
def test_drift_field_array():
    """Test MFGProblem with array drift field."""
    # Create a drift field array (default_geometry has 11 grid points)
    drift_array = np.ones(11) * 0.1  # Constant drift

    problem = create_test_problem(drift=drift_array)

    assert isinstance(problem.drift_field, np.ndarray)
    assert np.array_equal(problem.drift_field, drift_array)
    assert not problem.has_state_dependent_coefficients()


@pytest.mark.unit
def test_drift_field_scalar():
    """Test MFGProblem with scalar drift (constant drift)."""
    problem = create_test_problem(drift=0.05)

    assert problem.drift_field == 0.05
    assert not problem.has_state_dependent_coefficients()


@pytest.mark.unit
def test_drift_field_callable():
    """Test MFGProblem with callable drift field."""

    def drift_func(t, x, m):
        """Drift towards high-density regions."""
        return -0.1 * x

    problem = create_test_problem(drift=drift_func)

    assert callable(problem.drift_field)
    assert problem.drift_field is drift_func
    assert problem.has_state_dependent_coefficients()


@pytest.mark.unit
def test_get_diffusion_coefficient_field():
    """Test get_diffusion_coefficient_field returns CoefficientField wrapper."""
    problem = create_test_problem(diffusion=0.5)

    coeff_field = problem.get_diffusion_coefficient_field()

    # Should return a CoefficientField instance
    from mfg_pde.utils.pde_coefficients import CoefficientField

    assert isinstance(coeff_field, CoefficientField)
    assert coeff_field.name == "diffusion"


@pytest.mark.unit
def test_get_drift_coefficient_field():
    """Test get_drift_coefficient_field returns CoefficientField wrapper."""
    # default_geometry has 11 grid points
    drift_array = np.ones(11) * 0.1
    problem = create_test_problem(drift=drift_array)

    coeff_field = problem.get_drift_coefficient_field()

    # Should return a CoefficientField instance
    from mfg_pde.utils.pde_coefficients import CoefficientField

    assert isinstance(coeff_field, CoefficientField)
    assert coeff_field.name == "drift"


@pytest.mark.unit
def test_has_state_dependent_coefficients_mixed():
    """Test has_state_dependent_coefficients with mixed coefficient types."""

    def sigma_func(t, x, m):
        return 0.1 + 0.5 * m

    # Scalar drift, callable diffusion (default_geometry has 11 grid points)
    problem1 = create_test_problem(diffusion=sigma_func, drift=np.zeros(11))
    assert problem1.has_state_dependent_coefficients()

    # Callable drift, scalar diffusion
    def drift_func(t, x, m):
        return -0.1 * x

    problem2 = create_test_problem(diffusion=0.5, drift=drift_func)
    assert problem2.has_state_dependent_coefficients()

    # Both scalar
    problem3 = create_test_problem(diffusion=0.5, drift=np.zeros(11))
    assert not problem3.has_state_dependent_coefficients()


@pytest.mark.unit
def test_diffusion_field_with_geometry():
    """Test DiffusionField support works with geometry-based API."""
    grid = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[21, 21],
        boundary_conditions=no_flux_bc(dimension=2),
    )

    def sigma_func(t, x, m):
        """2D state-dependent diffusion."""
        return 0.1 + 0.1 * np.sum(x**2)

    # Issue #670: must provide m_initial and u_final
    components = MFGComponents(
        hamiltonian=default_hamiltonian(),
        m_initial=lambda x: 1.0,
        u_final=lambda x: 0.0,
    )
    problem = MFGProblem(geometry=grid, time_domain=(1.0, 50), diffusion=sigma_func, components=components)

    assert callable(problem.diffusion_field)
    assert problem.has_state_dependent_coefficients()
    assert problem.dimension == 2
