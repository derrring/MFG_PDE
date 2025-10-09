"""
Unit tests for 1D Domain.

Tests the Domain1D class for 1D MFG problems, including initialization,
grid generation, boundary condition integration, and edge cases.
"""

import pytest

from mfg_pde.geometry.boundary_conditions_1d import (
    dirichlet_bc,
    neumann_bc,
    no_flux_bc,
    periodic_bc,
)
from mfg_pde.geometry.domain_1d import Domain1D

# ============================================================================
# Test: Domain Initialization
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_domain_default_initialization():
    """Test Domain1D initialization with basic parameters."""
    bc = periodic_bc()
    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)

    assert domain.xmin == 0.0
    assert domain.xmax == 1.0
    assert domain.length == 1.0
    assert domain.boundary_conditions.is_periodic()


@pytest.mark.unit
@pytest.mark.fast
def test_domain_custom_bounds():
    """Test Domain1D with custom domain bounds."""
    bc = dirichlet_bc(0.0, 1.0)
    domain = Domain1D(xmin=-2.0, xmax=5.0, boundary_conditions=bc)

    assert domain.xmin == -2.0
    assert domain.xmax == 5.0
    assert domain.length == 7.0


@pytest.mark.unit
@pytest.mark.fast
def test_domain_invalid_bounds_raises():
    """Test Domain1D raises ValueError when xmax <= xmin."""
    bc = periodic_bc()

    with pytest.raises(ValueError, match="xmax must be greater than xmin"):
        Domain1D(xmin=1.0, xmax=1.0, boundary_conditions=bc)

    with pytest.raises(ValueError, match="xmax must be greater than xmin"):
        Domain1D(xmin=2.0, xmax=1.0, boundary_conditions=bc)


@pytest.mark.unit
@pytest.mark.fast
def test_domain_boundary_condition_integration():
    """Test domain correctly stores boundary conditions."""
    bc = no_flux_bc()
    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)

    assert domain.boundary_conditions is bc
    assert domain.boundary_conditions.is_no_flux()


@pytest.mark.unit
@pytest.mark.fast
def test_domain_validates_boundary_conditions():
    """Test domain validates boundary conditions on initialization."""
    # Create invalid Dirichlet BC (missing values)
    from mfg_pde.geometry.boundary_conditions_1d import BoundaryConditions

    bc = BoundaryConditions(type="dirichlet")  # Missing left/right values

    with pytest.raises(ValueError, match="Dirichlet boundary conditions require"):
        Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)


# ============================================================================
# Test: Domain Properties
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_domain_length_computation():
    """Test domain length is correctly computed."""
    bc = periodic_bc()

    domain1 = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)
    assert domain1.length == 1.0

    domain2 = Domain1D(xmin=-1.0, xmax=3.0, boundary_conditions=bc)
    assert domain2.length == 4.0

    domain3 = Domain1D(xmin=2.5, xmax=7.5, boundary_conditions=bc)
    assert domain3.length == 5.0


@pytest.mark.unit
@pytest.mark.fast
def test_domain_negative_bounds():
    """Test domain with negative bounds."""
    bc = periodic_bc()
    domain = Domain1D(xmin=-5.0, xmax=-2.0, boundary_conditions=bc)

    assert domain.xmin == -5.0
    assert domain.xmax == -2.0
    assert domain.length == 3.0


@pytest.mark.unit
@pytest.mark.fast
def test_domain_large_bounds():
    """Test domain with large coordinate values."""
    bc = periodic_bc()
    domain = Domain1D(xmin=1e6, xmax=2e6, boundary_conditions=bc)

    assert domain.xmin == 1e6
    assert domain.xmax == 2e6
    assert domain.length == 1e6


# ============================================================================
# Test: Grid Generation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_create_grid_basic():
    """Test basic grid generation."""
    bc = periodic_bc()
    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)

    dx, x_points = domain.create_grid(num_points=11)

    assert dx == pytest.approx(0.1)
    assert len(x_points) == 11
    assert x_points[0] == pytest.approx(0.0)
    assert x_points[-1] == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.fast
def test_create_grid_spacing():
    """Test grid spacing is uniform."""
    bc = dirichlet_bc(0.0, 1.0)
    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)

    dx, x_points = domain.create_grid(num_points=21)

    # Check uniform spacing
    for i in range(len(x_points) - 1):
        spacing = x_points[i + 1] - x_points[i]
        assert spacing == pytest.approx(dx)


@pytest.mark.unit
@pytest.mark.fast
def test_create_grid_different_sizes():
    """Test grid generation with different numbers of points."""
    bc = periodic_bc()
    domain = Domain1D(xmin=0.0, xmax=2.0, boundary_conditions=bc)

    # 5 points
    dx5, x5 = domain.create_grid(num_points=5)
    assert len(x5) == 5
    assert dx5 == pytest.approx(0.5)

    # 11 points
    dx11, x11 = domain.create_grid(num_points=11)
    assert len(x11) == 11
    assert dx11 == pytest.approx(0.2)

    # 101 points
    dx101, x101 = domain.create_grid(num_points=101)
    assert len(x101) == 101
    assert dx101 == pytest.approx(0.02)


@pytest.mark.unit
@pytest.mark.fast
def test_create_grid_custom_domain():
    """Test grid generation on custom domain."""
    bc = neumann_bc(0.0, 0.0)
    domain = Domain1D(xmin=-1.0, xmax=3.0, boundary_conditions=bc)

    dx, x_points = domain.create_grid(num_points=9)

    assert dx == pytest.approx(0.5)
    assert len(x_points) == 9
    assert x_points[0] == pytest.approx(-1.0)
    assert x_points[-1] == pytest.approx(3.0)


@pytest.mark.unit
@pytest.mark.fast
def test_create_grid_minimal_points():
    """Test grid generation with minimum number of points."""
    bc = periodic_bc()
    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)

    dx, x_points = domain.create_grid(num_points=2)

    assert len(x_points) == 2
    assert x_points[0] == pytest.approx(0.0)
    assert x_points[1] == pytest.approx(1.0)
    assert dx == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.fast
def test_create_grid_invalid_num_points():
    """Test grid generation raises ValueError for invalid num_points."""
    bc = periodic_bc()
    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)

    with pytest.raises(ValueError, match="num_points must be at least 2"):
        domain.create_grid(num_points=1)

    with pytest.raises(ValueError, match="num_points must be at least 2"):
        domain.create_grid(num_points=0)


# ============================================================================
# Test: Matrix Size Computation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_get_matrix_size_periodic():
    """Test matrix size computation for periodic BC."""
    bc = periodic_bc()
    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)

    assert domain.get_matrix_size(10) == 10
    assert domain.get_matrix_size(100) == 100


@pytest.mark.unit
@pytest.mark.fast
def test_get_matrix_size_dirichlet():
    """Test matrix size computation for Dirichlet BC."""
    bc = dirichlet_bc(0.0, 1.0)
    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)

    assert domain.get_matrix_size(10) == 9
    assert domain.get_matrix_size(100) == 99


@pytest.mark.unit
@pytest.mark.fast
def test_get_matrix_size_neumann():
    """Test matrix size computation for Neumann BC."""
    bc = neumann_bc(0.0, 0.0)
    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)

    assert domain.get_matrix_size(10) == 11
    assert domain.get_matrix_size(100) == 101


@pytest.mark.unit
@pytest.mark.fast
def test_get_matrix_size_no_flux():
    """Test matrix size computation for no-flux BC."""
    bc = no_flux_bc()
    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)

    assert domain.get_matrix_size(10) == 10
    assert domain.get_matrix_size(100) == 100


# ============================================================================
# Test: Integration with Different BC Types
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_domain_with_periodic_bc():
    """Test domain with periodic boundary conditions."""
    bc = periodic_bc()
    domain = Domain1D(xmin=0.0, xmax=2.0, boundary_conditions=bc)

    assert domain.boundary_conditions.is_periodic()
    assert domain.get_matrix_size(50) == 50


@pytest.mark.unit
@pytest.mark.fast
def test_domain_with_dirichlet_bc():
    """Test domain with Dirichlet boundary conditions."""
    bc = dirichlet_bc(left_value=0.5, right_value=1.5)
    domain = Domain1D(xmin=-1.0, xmax=1.0, boundary_conditions=bc)

    assert domain.boundary_conditions.is_dirichlet()
    assert domain.boundary_conditions.left_value == 0.5
    assert domain.boundary_conditions.right_value == 1.5


@pytest.mark.unit
@pytest.mark.fast
def test_domain_with_neumann_bc():
    """Test domain with Neumann boundary conditions."""
    bc = neumann_bc(left_gradient=0.2, right_gradient=-0.3)
    domain = Domain1D(xmin=0.0, xmax=5.0, boundary_conditions=bc)

    assert domain.boundary_conditions.is_neumann()
    assert domain.boundary_conditions.left_value == 0.2
    assert domain.boundary_conditions.right_value == -0.3


@pytest.mark.unit
@pytest.mark.fast
def test_domain_with_no_flux_bc():
    """Test domain with no-flux boundary conditions."""
    bc = no_flux_bc()
    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)

    assert domain.boundary_conditions.is_no_flux()


# ============================================================================
# Test: String Representation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_domain_str_periodic():
    """Test string representation with periodic BC."""
    bc = periodic_bc()
    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)

    s = str(domain)
    assert "Domain1D" in s
    assert "[0.0, 1.0]" in s
    assert "Periodic" in s


@pytest.mark.unit
@pytest.mark.fast
def test_domain_str_dirichlet():
    """Test string representation with Dirichlet BC."""
    bc = dirichlet_bc(0.0, 1.0)
    domain = Domain1D(xmin=-1.0, xmax=2.0, boundary_conditions=bc)

    s = str(domain)
    assert "Domain1D" in s
    assert "[-1.0, 2.0]" in s
    assert "Dirichlet" in s


@pytest.mark.unit
@pytest.mark.fast
def test_domain_repr():
    """Test detailed representation of domain."""
    bc = periodic_bc()
    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)

    r = repr(domain)
    assert "Domain1D" in r
    assert "xmin=0.0" in r
    assert "xmax=1.0" in r
    assert "length=1.0" in r
    assert "bc=" in r


# ============================================================================
# Test: Edge Cases
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_domain_very_small_length():
    """Test domain with very small length."""
    bc = periodic_bc()
    domain = Domain1D(xmin=0.0, xmax=1e-10, boundary_conditions=bc)

    assert domain.length == pytest.approx(1e-10)

    dx, _ = domain.create_grid(num_points=3)
    assert dx == pytest.approx(5e-11)


@pytest.mark.unit
@pytest.mark.fast
def test_domain_very_large_length():
    """Test domain with very large length."""
    bc = periodic_bc()
    domain = Domain1D(xmin=0.0, xmax=1e10, boundary_conditions=bc)

    assert domain.length == pytest.approx(1e10)

    dx, _ = domain.create_grid(num_points=11)
    assert dx == pytest.approx(1e9)


@pytest.mark.unit
@pytest.mark.fast
def test_domain_many_grid_points():
    """Test grid generation with many points."""
    bc = periodic_bc()
    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)

    dx, x_points = domain.create_grid(num_points=10001)

    assert len(x_points) == 10001
    assert x_points[0] == pytest.approx(0.0)
    assert x_points[-1] == pytest.approx(1.0)
    assert dx == pytest.approx(0.0001)


@pytest.mark.unit
@pytest.mark.fast
def test_domain_grid_endpoints():
    """Test grid always includes domain endpoints."""
    bc = dirichlet_bc(0.0, 1.0)
    domain = Domain1D(xmin=-2.5, xmax=7.3, boundary_conditions=bc)

    _, x_points = domain.create_grid(num_points=50)

    assert x_points[0] == pytest.approx(domain.xmin)
    assert x_points[-1] == pytest.approx(domain.xmax)


@pytest.mark.unit
@pytest.mark.fast
def test_multiple_domains_independent():
    """Test multiple domain instances are independent."""
    bc1 = periodic_bc()
    bc2 = dirichlet_bc(0.0, 1.0)

    domain1 = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=bc1)
    domain2 = Domain1D(xmin=2.0, xmax=5.0, boundary_conditions=bc2)

    assert domain1.xmin == 0.0
    assert domain2.xmin == 2.0
    assert domain1.length == 1.0
    assert domain2.length == 3.0


@pytest.mark.unit
@pytest.mark.fast
def test_domain_grid_reproducibility():
    """Test grid generation is reproducible."""
    bc = periodic_bc()
    domain = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)

    dx1, x_points1 = domain.create_grid(num_points=11)
    dx2, x_points2 = domain.create_grid(num_points=11)

    assert dx1 == dx2
    assert x_points1 == x_points2


@pytest.mark.unit
@pytest.mark.fast
def test_domain_fractional_bounds():
    """Test domain with fractional bounds."""
    bc = periodic_bc()
    domain = Domain1D(xmin=0.3, xmax=0.7, boundary_conditions=bc)

    assert domain.length == pytest.approx(0.4)

    dx, _ = domain.create_grid(num_points=5)
    assert dx == pytest.approx(0.1)
