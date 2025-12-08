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

# Unified BC from conditions.py (current API)
from mfg_pde.geometry.boundary.conditions import BoundaryConditions

# Legacy 1D BC: testing compatibility with 1D MFG problems (deprecated in v0.14, remove in v1.0)
from mfg_pde.geometry.boundary.fdm_bc_1d import BoundaryConditions as LegacyBoundaryConditions

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
    assert problem.dx == pytest.approx(1.0 / 51)

    # Time parameters
    assert problem.T == 1.0
    assert problem.Nt == 51
    assert problem.dt == pytest.approx(1.0 / 51)

    # Physical parameters
    assert problem.sigma == 1.0
    assert problem.coupling_coefficient == 0.5

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
    assert problem.dx == pytest.approx(3.0 / 100)
    assert len(problem.xSpace) == 101  # Nx + 1


@pytest.mark.unit
def test_mfg_problem_custom_time():
    """Test MFGProblem with custom time parameters."""
    problem = MFGProblem(T=2.0, Nt=100)

    assert problem.T == 2.0
    assert problem.Nt == 100
    assert problem.dt == pytest.approx(2.0 / 100)
    assert len(problem.tSpace) == 101  # Nt + 1


@pytest.mark.unit
def test_mfg_problem_custom_coefficients():
    """Test MFGProblem with custom coefficients."""
    problem = MFGProblem(sigma=0.5, coupling_coefficient=0.8)

    assert problem.sigma == 0.5
    assert problem.coupling_coefficient == 0.8


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
    integral = np.sum(expected_unnormalized) * problem.dx
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
    problem = MFGProblem(Nx=10, sigma=1.0, coupling_coefficient=0.5)

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
    """Test get_boundary_conditions with custom BC (legacy 1D BC backward compat)."""
    # Uses legacy 1D BC to test backward compatibility
    custom_bc = LegacyBoundaryConditions(type="dirichlet", left_value=0.0, right_value=0.0)
    components = MFGComponents(boundary_conditions=custom_bc)
    problem = MFGProblem(components=components)

    bc = problem.get_boundary_conditions()

    # Custom BC is passed through as-is (legacy type)
    assert isinstance(bc, LegacyBoundaryConditions)
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
    problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=100, sigma=0.5, coupling_coefficient=0.8)

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
    assert info["coefficients"]["coupling_coefficient"] == 0.8


# ===================================================================
# Test Edge Cases
# ===================================================================


@pytest.mark.unit
@pytest.mark.skip(
    reason="Test deprecated behavior. New geometry validation (SimpleGrid1D) correctly "
    "rejects Nx=0 because at least 2 points are needed to define a grid with spacing. "
    "Legacy API behavior is no longer supported."
)
def test_mfg_problem_zero_nx():
    """Test MFGProblem handles Nx=0 gracefully."""
    problem = MFGProblem(Nx=0)

    assert problem.Nx == 0
    assert problem.dx == 0.0
    assert len(problem.xSpace) == 1


@pytest.mark.unit
def test_mfg_problem_zero_nt():
    """Test MFGProblem handles Nt=0 gracefully."""
    problem = MFGProblem(Nt=0)

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
    from mfg_pde.geometry import SimpleGrid2D

    # Create two different grids
    hjb_grid = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(50, 50))
    fp_grid = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(20, 20))

    # Create problem with dual geometries
    problem = MFGProblem(hjb_geometry=hjb_grid, fp_geometry=fp_grid, time_domain=(1.0, 50), sigma=0.1)

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
    from mfg_pde.geometry import SimpleGrid2D

    grid = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(30, 30))

    # Create problem with unified geometry (old API)
    problem = MFGProblem(geometry=grid, time_domain=(1.0, 50), sigma=0.1)

    # Check that both hjb_geometry and fp_geometry point to the same geometry
    assert problem.hjb_geometry is grid
    assert problem.fp_geometry is grid

    # Check that no projector is created for unified mode
    assert problem.geometry_projector is None


@pytest.mark.unit
def test_dual_geometry_error_on_partial_specification():
    """Test that specifying only one of hjb_geometry/fp_geometry raises error."""
    from mfg_pde.geometry import SimpleGrid2D

    grid = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(30, 30))

    # Test with only hjb_geometry
    with pytest.raises(ValueError, match="both 'hjb_geometry' AND 'fp_geometry' must be specified"):
        MFGProblem(hjb_geometry=grid, time_domain=(1.0, 50))

    # Test with only fp_geometry
    with pytest.raises(ValueError, match="both 'hjb_geometry' AND 'fp_geometry' must be specified"):
        MFGProblem(fp_geometry=grid, time_domain=(1.0, 50))


@pytest.mark.unit
def test_dual_geometry_error_on_conflict():
    """Test that specifying both geometry and dual geometries raises error."""
    from mfg_pde.geometry import SimpleGrid2D

    grid1 = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(30, 30))
    grid2 = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(20, 20))
    grid3 = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(10, 10))

    # Test conflict: can't specify both geometry and dual geometries
    with pytest.raises(ValueError, match=r"Specify EITHER 'geometry'.*OR.*'hjb_geometry', 'fp_geometry'"):
        MFGProblem(geometry=grid1, hjb_geometry=grid2, fp_geometry=grid3, time_domain=(1.0, 50))


@pytest.mark.unit
def test_dual_geometry_projector_attributes():
    """Test that geometry projector has correct attributes."""
    from mfg_pde.geometry import SimpleGrid2D

    hjb_grid = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(50, 50))
    fp_grid = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(30, 30))

    problem = MFGProblem(hjb_geometry=hjb_grid, fp_geometry=fp_grid, time_domain=(1.0, 50))

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
    from mfg_pde.geometry import SimpleGrid1D
    from mfg_pde.geometry.boundary.fdm_bc_1d import BoundaryConditions

    bc = BoundaryConditions(type="periodic")

    # Create two 1D grids with different resolutions
    hjb_grid = SimpleGrid1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)
    hjb_grid.create_grid(num_points=101)  # Fine grid

    fp_grid = SimpleGrid1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)
    fp_grid.create_grid(num_points=51)  # Coarse grid

    problem = MFGProblem(hjb_geometry=hjb_grid, fp_geometry=fp_grid, time_domain=(1.0, 50), sigma=0.1)

    # Verify dual geometry setup
    assert problem.hjb_geometry is hjb_grid
    assert problem.fp_geometry is fp_grid
    assert problem.geometry_projector is not None


@pytest.mark.unit
def test_dual_geometry_legacy_mode_compatibility():
    """Test that legacy 1D mode sets hjb_geometry and fp_geometry correctly."""
    # Legacy mode creates its own grid internally
    problem = MFGProblem(Nx=100, xmin=0.0, xmax=1.0, T=1.0, Nt=50)

    # Check that hjb_geometry and fp_geometry are set (to the same unified geometry)
    assert problem.hjb_geometry is not None
    assert problem.fp_geometry is not None
    assert problem.hjb_geometry is problem.fp_geometry  # Unified mode
    assert problem.geometry_projector is None  # No projector for unified mode
