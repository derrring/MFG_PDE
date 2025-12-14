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

    # Physical parameters (None → 0 for diffusion/drift)
    assert problem.sigma == 0.0
    assert problem.diffusion_field == 0.0
    assert problem.drift_field == 0.0
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
    # Flatten both arrays for comparison (problem stores as 2D column vector)
    assert np.allclose(np.ravel(problem.f_potential), np.ravel(expected))


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
    # Flatten both arrays for comparison (problem stores as 2D column vector)
    assert np.allclose(np.ravel(problem.m_init), np.ravel(expected))


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
    # Flatten both arrays for comparison (problem stores as 2D column vector)
    assert np.allclose(np.ravel(problem.u_fin), np.ravel(expected))


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
    reason="Test deprecated behavior. New geometry validation (TensorProductGrid) correctly "
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
    from mfg_pde.geometry import TensorProductGrid

    # Create two different grids
    hjb_grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[51, 51])
    fp_grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[21, 21])

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
    from mfg_pde.geometry import TensorProductGrid

    grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[31, 31])

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
    from mfg_pde.geometry import TensorProductGrid

    grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[31, 31])

    # Test with only hjb_geometry
    with pytest.raises(ValueError, match="both 'hjb_geometry' AND 'fp_geometry' must be specified"):
        MFGProblem(hjb_geometry=grid, time_domain=(1.0, 50))

    # Test with only fp_geometry
    with pytest.raises(ValueError, match="both 'hjb_geometry' AND 'fp_geometry' must be specified"):
        MFGProblem(fp_geometry=grid, time_domain=(1.0, 50))


@pytest.mark.unit
def test_dual_geometry_error_on_conflict():
    """Test that specifying both geometry and dual geometries raises error."""
    from mfg_pde.geometry import TensorProductGrid

    grid1 = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[31, 31])
    grid2 = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[21, 21])
    grid3 = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11])

    # Test conflict: can't specify both geometry and dual geometries
    with pytest.raises(ValueError, match=r"Specify EITHER 'geometry'.*OR.*'hjb_geometry', 'fp_geometry'"):
        MFGProblem(geometry=grid1, hjb_geometry=grid2, fp_geometry=grid3, time_domain=(1.0, 50))


@pytest.mark.unit
def test_dual_geometry_projector_attributes():
    """Test that geometry projector has correct attributes."""
    from mfg_pde.geometry import TensorProductGrid

    hjb_grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[51, 51])
    fp_grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[31, 31])

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
    from mfg_pde.geometry import TensorProductGrid

    # Create two 1D grids with different resolutions
    hjb_grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[101])  # Fine grid
    fp_grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[51])  # Coarse grid

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


# ===================================================================
# Test DiffusionField Support (Feature: diffusion-field-support)
# ===================================================================


@pytest.mark.unit
def test_diffusion_field_none():
    """Test MFGProblem with no diffusion (deterministic). None → 0."""
    problem = MFGProblem(diffusion=None)

    assert problem.diffusion_field == 0.0
    assert problem.sigma == 0.0
    assert not problem.has_state_dependent_coefficients()


@pytest.mark.unit
def test_diffusion_field_scalar():
    """Test MFGProblem with scalar diffusion coefficient."""
    problem = MFGProblem(diffusion=0.5)

    assert problem.sigma == 0.5
    assert problem.diffusion_field == 0.5
    assert not problem.has_state_dependent_coefficients()


@pytest.mark.unit
def test_diffusion_field_array():
    """Test MFGProblem with array diffusion coefficient (spatially varying)."""
    # Create a spatially varying diffusion array
    sigma_array = np.linspace(0.1, 1.0, 52)  # Nx+1 = 52 for Nx=51

    problem = MFGProblem(sigma=sigma_array)

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

    problem = MFGProblem(sigma=sigma_func)

    # Callable should be stored in diffusion_field
    assert callable(problem.diffusion_field)
    assert problem.diffusion_field is sigma_func

    # Scalar sigma should default to 1.0 for callable
    assert problem.sigma == 1.0
    assert problem.has_state_dependent_coefficients()


@pytest.mark.unit
def test_diffusion_primary_parameter():
    """Test that 'diffusion' is the primary parameter."""
    problem = MFGProblem(diffusion=0.3)

    assert problem.sigma == 0.3
    assert problem.diffusion_field == 0.3


@pytest.mark.unit
def test_sigma_deprecated_alias():
    """Test that 'sigma' is deprecated alias for 'diffusion'."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        problem = MFGProblem(sigma=0.4)

        # Check deprecation warning was raised
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        assert "sigma" in str(deprecation_warnings[0].message)
        assert "diffusion" in str(deprecation_warnings[0].message)

    # Value should still work
    assert problem.sigma == 0.4
    assert problem.diffusion_field == 0.4


@pytest.mark.unit
def test_drift_field_none():
    """Test MFGProblem with no drift field (default). None → 0."""
    problem = MFGProblem()

    assert problem.drift_field == 0.0
    assert not problem.has_state_dependent_coefficients()


@pytest.mark.unit
def test_drift_field_array():
    """Test MFGProblem with array drift field."""
    # Create a drift field array
    drift_array = np.ones(52) * 0.1  # Constant drift

    problem = MFGProblem(drift=drift_array)

    assert isinstance(problem.drift_field, np.ndarray)
    assert np.array_equal(problem.drift_field, drift_array)
    assert not problem.has_state_dependent_coefficients()


@pytest.mark.unit
def test_drift_field_scalar():
    """Test MFGProblem with scalar drift (constant drift)."""
    problem = MFGProblem(drift=0.05)

    assert problem.drift_field == 0.05
    assert not problem.has_state_dependent_coefficients()


@pytest.mark.unit
def test_drift_field_callable():
    """Test MFGProblem with callable drift field."""

    def drift_func(t, x, m):
        """Drift towards high-density regions."""
        return -0.1 * x

    problem = MFGProblem(drift=drift_func)

    assert callable(problem.drift_field)
    assert problem.drift_field is drift_func
    assert problem.has_state_dependent_coefficients()


@pytest.mark.unit
def test_get_diffusion_coefficient_field():
    """Test get_diffusion_coefficient_field returns CoefficientField wrapper."""
    problem = MFGProblem(sigma=0.5)

    coeff_field = problem.get_diffusion_coefficient_field()

    # Should return a CoefficientField instance
    from mfg_pde.utils.pde_coefficients import CoefficientField

    assert isinstance(coeff_field, CoefficientField)
    assert coeff_field.name == "diffusion"


@pytest.mark.unit
def test_get_drift_coefficient_field():
    """Test get_drift_coefficient_field returns CoefficientField wrapper."""
    drift_array = np.ones(52) * 0.1
    problem = MFGProblem(drift=drift_array)

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

    # Scalar drift, callable diffusion
    problem1 = MFGProblem(sigma=sigma_func, drift=np.zeros(52))
    assert problem1.has_state_dependent_coefficients()

    # Callable drift, scalar diffusion
    def drift_func(t, x, m):
        return -0.1 * x

    problem2 = MFGProblem(sigma=0.5, drift=drift_func)
    assert problem2.has_state_dependent_coefficients()

    # Both scalar
    problem3 = MFGProblem(sigma=0.5, drift=np.zeros(52))
    assert not problem3.has_state_dependent_coefficients()


@pytest.mark.unit
def test_diffusion_field_with_geometry():
    """Test DiffusionField support works with geometry-based API."""
    from mfg_pde.geometry import TensorProductGrid

    grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[21, 21])

    def sigma_func(t, x, m):
        """2D state-dependent diffusion."""
        return 0.1 + 0.1 * np.sum(x**2)

    problem = MFGProblem(geometry=grid, time_domain=(1.0, 50), sigma=sigma_func)

    assert callable(problem.diffusion_field)
    assert problem.has_state_dependent_coefficients()
    assert problem.dimension == 2
