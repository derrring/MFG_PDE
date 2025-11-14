#!/usr/bin/env python3
"""
Unit tests for mfg_pde/geometry/boundary_manager.py

Tests boundary condition management for complex geometry domains including:
- GeometricBoundaryCondition initialization and validation
- BoundaryCondition evaluation (constant and function values)
- BoundaryManager initialization and region extraction
- Adding and managing multiple boundary conditions
- Applying Dirichlet and Neumann conditions to linear systems
- Legacy boundary condition conversion
- Summary generation
"""

import pytest

import numpy as np

from mfg_pde.geometry.boundary_conditions_1d import BoundaryConditions
from mfg_pde.geometry.boundary_manager import BoundaryManager, GeometricBoundaryCondition
from mfg_pde.geometry.meshes.mesh_data import MeshData

# ===================================================================
# Test GeometricBoundaryCondition Initialization and Validation
# ===================================================================


@pytest.mark.unit
def test_geometric_bc_dirichlet_constant():
    """Test Dirichlet BC with constant value."""
    bc = GeometricBoundaryCondition(region_id=1, bc_type="dirichlet", value=1.0)

    assert bc.region_id == 1
    assert bc.bc_type == "dirichlet"
    assert bc.value == 1.0


@pytest.mark.unit
def test_geometric_bc_dirichlet_function():
    """Test Dirichlet BC with function value."""

    def bc_func(coords):
        return np.sin(coords[:, 0])

    bc = GeometricBoundaryCondition(region_id=2, bc_type="dirichlet", value=bc_func)

    assert bc.region_id == 2
    assert callable(bc.value)


@pytest.mark.unit
def test_geometric_bc_neumann_constant():
    """Test Neumann BC with constant gradient."""
    bc = GeometricBoundaryCondition(region_id=1, bc_type="neumann", gradient_value=0.5)

    assert bc.bc_type == "neumann"
    assert bc.gradient_value == 0.5


@pytest.mark.unit
def test_geometric_bc_robin():
    """Test Robin BC with alpha and beta coefficients."""
    bc = GeometricBoundaryCondition(region_id=1, bc_type="robin", alpha=1.0, beta=2.0, value=0.0)

    assert bc.bc_type == "robin"
    assert bc.alpha == 1.0
    assert bc.beta == 2.0


@pytest.mark.unit
def test_geometric_bc_no_flux():
    """Test no-flux BC (special case of Neumann)."""
    bc = GeometricBoundaryCondition(region_id=1, bc_type="no_flux", gradient_value=0.0)

    assert bc.bc_type == "no_flux"
    assert bc.gradient_value == 0.0


@pytest.mark.unit
def test_geometric_bc_time_dependent():
    """Test time-dependent BC with spatial function."""

    def spatial_func(coords, t):
        return np.sin(coords[:, 0] + t)

    bc = GeometricBoundaryCondition(
        region_id=1, bc_type="dirichlet", value=1.0, time_dependent=True, spatial_function=spatial_func
    )

    assert bc.time_dependent is True
    assert bc.spatial_function is not None


@pytest.mark.unit
def test_geometric_bc_with_metadata():
    """Test BC with metadata."""
    metadata = {"author": "test", "version": "1.0"}
    bc = GeometricBoundaryCondition(
        region_id=1, bc_type="dirichlet", value=1.0, description="Test BC", metadata=metadata
    )

    assert bc.description == "Test BC"
    assert bc.metadata["author"] == "test"
    assert bc.metadata["version"] == "1.0"


# ===================================================================
# Test GeometricBoundaryCondition Validation
# ===================================================================


@pytest.mark.unit
def test_geometric_bc_robin_missing_alpha_raises():
    """Test Robin BC without alpha raises error."""
    with pytest.raises(ValueError, match="Robin boundary conditions require alpha and beta"):
        GeometricBoundaryCondition(region_id=1, bc_type="robin", beta=2.0, value=0.0)


@pytest.mark.unit
def test_geometric_bc_robin_missing_beta_raises():
    """Test Robin BC without beta raises error."""
    with pytest.raises(ValueError, match="Robin boundary conditions require alpha and beta"):
        GeometricBoundaryCondition(region_id=1, bc_type="robin", alpha=1.0, value=0.0)


@pytest.mark.unit
def test_geometric_bc_dirichlet_missing_value_raises():
    """Test Dirichlet BC without value raises error."""
    with pytest.raises(ValueError, match="Dirichlet boundary condition requires value"):
        GeometricBoundaryCondition(region_id=1, bc_type="dirichlet")


@pytest.mark.unit
def test_geometric_bc_neumann_missing_value_raises():
    """Test Neumann BC without gradient_value or value raises error."""
    with pytest.raises(ValueError, match="Neumann boundary condition requires gradient_value or value"):
        GeometricBoundaryCondition(region_id=1, bc_type="neumann")


# ===================================================================
# Test GeometricBoundaryCondition Evaluation
# ===================================================================


@pytest.mark.unit
def test_geometric_bc_evaluate_constant():
    """Test evaluation of constant BC."""
    bc = GeometricBoundaryCondition(region_id=1, bc_type="dirichlet", value=2.5)

    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    values = bc.evaluate(coords)

    assert values.shape == (3,)
    assert np.allclose(values, 2.5)


@pytest.mark.unit
def test_geometric_bc_evaluate_function():
    """Test evaluation of function-based BC."""

    def bc_func(coords):
        return coords[:, 0] ** 2

    bc = GeometricBoundaryCondition(region_id=1, bc_type="dirichlet", value=bc_func)

    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    values = bc.evaluate(coords)

    expected = np.array([0.0, 1.0, 4.0])
    assert np.allclose(values, expected)


@pytest.mark.unit
def test_geometric_bc_evaluate_time_dependent():
    """Test evaluation of time-dependent BC."""

    def spatial_func(coords, t):
        return np.sin(coords[:, 0] + t)

    bc = GeometricBoundaryCondition(
        region_id=1, bc_type="dirichlet", value=1.0, time_dependent=True, spatial_function=spatial_func
    )

    coords = np.array([[0.0, 0.0], [np.pi / 2, 0.0]])
    values_t0 = bc.evaluate(coords, time=0.0)
    values_t1 = bc.evaluate(coords, time=1.0)

    # Values should change with time
    assert not np.allclose(values_t0, values_t1)


@pytest.mark.unit
def test_geometric_bc_evaluate_gradient_constant():
    """Test gradient evaluation with constant value."""
    bc = GeometricBoundaryCondition(region_id=1, bc_type="neumann", gradient_value=1.5)

    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    gradients = bc.evaluate_gradient(coords)

    assert np.allclose(gradients, 1.5)


@pytest.mark.unit
def test_geometric_bc_evaluate_gradient_function():
    """Test gradient evaluation with function value."""

    def grad_func(coords):
        return 2.0 * coords[:, 0]

    bc = GeometricBoundaryCondition(region_id=1, bc_type="neumann", value=0.0, gradient_value=grad_func)

    coords = np.array([[1.0, 0.0], [2.0, 0.0]])
    gradients = bc.evaluate_gradient(coords)

    expected = np.array([2.0, 4.0])
    assert np.allclose(gradients, expected)


# ===================================================================
# Test BoundaryManager Initialization
# ===================================================================


def create_simple_mesh_data():
    """Create simple 2D mesh data for testing."""
    # Simple 2x2 grid: 9 vertices, 4 triangles
    vertices = np.array(
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0, 0.5], [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]]
    )

    # Triangular elements
    elements = np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4], [3, 4, 6], [4, 7, 6], [4, 5, 7], [5, 8, 7]])

    # Boundary tags: region 1 = left, region 2 = right, region 3 = top, region 4 = bottom
    boundary_tags = np.array([1, 0, 2, 1, 0, 2, 1, 0, 2])

    # Boundary faces (edges)
    boundary_faces = np.array([[0, 3], [3, 6], [2, 5], [5, 8], [0, 1], [1, 2], [6, 7], [7, 8]])

    return MeshData(
        vertices=vertices,
        elements=elements,
        element_type="triangle",
        boundary_tags=boundary_tags,
        element_tags=np.zeros(len(elements), dtype=int),
        boundary_faces=boundary_faces,
        dimension=2,
    )


@pytest.mark.unit
def test_boundary_manager_initialization():
    """Test BoundaryManager initialization with mesh data."""
    mesh_data = create_simple_mesh_data()
    manager = BoundaryManager(mesh_data)

    assert manager.mesh_data is mesh_data
    assert isinstance(manager.boundary_conditions, dict)
    assert isinstance(manager.boundary_nodes, dict)
    assert isinstance(manager.boundary_faces, dict)


@pytest.mark.unit
def test_boundary_manager_extract_regions():
    """Test extraction of boundary regions from mesh."""
    mesh_data = create_simple_mesh_data()
    manager = BoundaryManager(mesh_data)

    # Should have regions 1 and 2 (skipping region 0 = interior)
    assert 1 in manager.boundary_nodes
    assert 2 in manager.boundary_nodes
    assert 0 not in manager.boundary_nodes  # Interior nodes excluded


@pytest.mark.unit
def test_boundary_manager_get_boundary_nodes():
    """Test getting boundary nodes for a region."""
    mesh_data = create_simple_mesh_data()
    manager = BoundaryManager(mesh_data)

    nodes_region1 = manager.get_boundary_nodes(1)
    assert len(nodes_region1) > 0
    assert isinstance(nodes_region1, np.ndarray)


@pytest.mark.unit
def test_boundary_manager_get_boundary_nodes_nonexistent():
    """Test getting boundary nodes for nonexistent region returns empty array."""
    mesh_data = create_simple_mesh_data()
    manager = BoundaryManager(mesh_data)

    nodes = manager.get_boundary_nodes(999)
    assert len(nodes) == 0


@pytest.mark.unit
def test_boundary_manager_get_boundary_coordinates():
    """Test getting boundary coordinates for a region."""
    mesh_data = create_simple_mesh_data()
    manager = BoundaryManager(mesh_data)

    coords = manager.get_boundary_coordinates(1)
    assert coords.shape[1] == 2  # 2D coordinates
    assert len(coords) > 0


# ===================================================================
# Test Adding Boundary Conditions
# ===================================================================


@pytest.mark.unit
def test_add_boundary_condition_dirichlet():
    """Test adding Dirichlet boundary condition."""
    mesh_data = create_simple_mesh_data()
    manager = BoundaryManager(mesh_data)

    bc = manager.add_boundary_condition(region_id=1, bc_type="dirichlet", value=1.0)

    assert isinstance(bc, GeometricBoundaryCondition)
    assert bc.region_id == 1
    assert bc.bc_type == "dirichlet"
    assert 1 in manager.boundary_conditions


@pytest.mark.unit
def test_add_boundary_condition_neumann():
    """Test adding Neumann boundary condition."""
    mesh_data = create_simple_mesh_data()
    manager = BoundaryManager(mesh_data)

    bc = manager.add_boundary_condition(region_id=2, bc_type="neumann", gradient_value=0.5)

    assert bc.bc_type == "neumann"
    assert bc.gradient_value == 0.5


@pytest.mark.unit
def test_add_multiple_boundary_conditions():
    """Test adding boundary conditions to multiple regions."""
    mesh_data = create_simple_mesh_data()
    manager = BoundaryManager(mesh_data)

    manager.add_boundary_condition(region_id=1, bc_type="dirichlet", value=0.0)
    manager.add_boundary_condition(region_id=2, bc_type="dirichlet", value=1.0)

    assert len(manager.boundary_conditions) == 2
    assert 1 in manager.boundary_conditions
    assert 2 in manager.boundary_conditions


@pytest.mark.unit
def test_add_boundary_condition_with_function():
    """Test adding boundary condition with function value."""
    mesh_data = create_simple_mesh_data()
    manager = BoundaryManager(mesh_data)

    def bc_func(coords):
        return coords[:, 0]

    bc = manager.add_boundary_condition(region_id=1, bc_type="dirichlet", value=bc_func)

    assert callable(bc.value)


# ===================================================================
# Test Evaluating Boundary Conditions
# ===================================================================


@pytest.mark.unit
def test_evaluate_boundary_condition():
    """Test evaluating boundary condition for a region."""
    mesh_data = create_simple_mesh_data()
    manager = BoundaryManager(mesh_data)

    manager.add_boundary_condition(region_id=1, bc_type="dirichlet", value=2.5)

    values = manager.evaluate_boundary_condition(region_id=1)

    assert len(values) > 0
    assert np.allclose(values, 2.5)


@pytest.mark.unit
def test_evaluate_boundary_condition_time_dependent():
    """Test evaluating time-dependent boundary condition."""
    mesh_data = create_simple_mesh_data()
    manager = BoundaryManager(mesh_data)

    def spatial_func(coords, t):
        return t * np.ones(len(coords))

    manager.add_boundary_condition(
        region_id=1, bc_type="dirichlet", value=0.0, time_dependent=True, spatial_function=spatial_func
    )

    values_t0 = manager.evaluate_boundary_condition(region_id=1, time=0.0)
    values_t1 = manager.evaluate_boundary_condition(region_id=1, time=1.0)

    assert np.allclose(values_t0, 0.0)
    assert np.allclose(values_t1, 1.0)


@pytest.mark.unit
def test_evaluate_nonexistent_region_raises():
    """Test evaluating boundary condition for nonexistent region raises error."""
    mesh_data = create_simple_mesh_data()
    manager = BoundaryManager(mesh_data)

    with pytest.raises(ValueError, match="No boundary condition defined for region"):
        manager.evaluate_boundary_condition(region_id=999)


# ===================================================================
# Test Applying Boundary Conditions to Linear Systems
# ===================================================================


@pytest.mark.unit
def test_apply_dirichlet_conditions():
    """Test applying Dirichlet conditions to linear system."""
    mesh_data = create_simple_mesh_data()
    manager = BoundaryManager(mesh_data)

    # Add Dirichlet BC
    manager.add_boundary_condition(region_id=1, bc_type="dirichlet", value=1.0)

    # Create simple linear system
    n = len(mesh_data.vertices)
    A = np.eye(n)
    b = np.zeros(n)

    A_new, b_new = manager.apply_dirichlet_conditions(A, b)

    # Check that boundary nodes have correct RHS values
    boundary_nodes = manager.get_boundary_nodes(1)
    for node in boundary_nodes:
        assert b_new[node] == pytest.approx(1.0)
        assert A_new[node, node] == pytest.approx(1.0)


@pytest.mark.unit
def test_apply_dirichlet_modifies_matrix_rows():
    """Test that Dirichlet conditions zero out matrix rows and set diagonal."""
    mesh_data = create_simple_mesh_data()
    manager = BoundaryManager(mesh_data)

    manager.add_boundary_condition(region_id=1, bc_type="dirichlet", value=2.0)

    n = len(mesh_data.vertices)
    A = np.ones((n, n))  # Full matrix
    b = np.zeros(n)

    A_new, _ = manager.apply_dirichlet_conditions(A, b)

    boundary_nodes = manager.get_boundary_nodes(1)
    for node in boundary_nodes:
        # Row should be zero except diagonal
        assert A_new[node, node] == pytest.approx(1.0)
        # Check off-diagonal elements are zero
        row_sum = np.sum(A_new[node, :]) - A_new[node, node]
        assert row_sum == pytest.approx(0.0)


@pytest.mark.unit
def test_apply_neumann_conditions():
    """Test applying Neumann conditions to RHS vector."""
    mesh_data = create_simple_mesh_data()
    manager = BoundaryManager(mesh_data)

    # Add Neumann BC
    manager.add_boundary_condition(region_id=1, bc_type="neumann", gradient_value=0.5)

    n = len(mesh_data.vertices)
    b = np.zeros(n)

    b_new = manager.apply_neumann_conditions(b)

    # RHS should be modified at boundary nodes
    boundary_nodes = manager.get_boundary_nodes(1)
    for node in boundary_nodes:
        assert b_new[node] != 0.0


@pytest.mark.unit
def test_apply_multiple_boundary_conditions():
    """Test applying multiple different boundary conditions."""
    mesh_data = create_simple_mesh_data()
    manager = BoundaryManager(mesh_data)

    # Region 1: Dirichlet, Region 2: Neumann
    manager.add_boundary_condition(region_id=1, bc_type="dirichlet", value=1.0)
    manager.add_boundary_condition(region_id=2, bc_type="neumann", gradient_value=0.5)

    n = len(mesh_data.vertices)
    A = np.eye(n)
    b = np.zeros(n)

    _, b_new = manager.apply_dirichlet_conditions(A, b)
    b_new = manager.apply_neumann_conditions(b_new)

    # Check Dirichlet region
    nodes_1 = manager.get_boundary_nodes(1)
    for node in nodes_1:
        assert b_new[node] == pytest.approx(1.0)

    # Check Neumann region (should have non-zero RHS)
    nodes_2 = manager.get_boundary_nodes(2)
    assert len(nodes_2) > 0


# ===================================================================
# Test Legacy Boundary Condition Conversion
# ===================================================================


@pytest.mark.unit
def test_create_legacy_boundary_conditions():
    """Test conversion to legacy BoundaryConditions format."""
    mesh_data = create_simple_mesh_data()
    manager = BoundaryManager(mesh_data)

    # Add boundary conditions (region 1 = left, region 2 = right)
    manager.add_boundary_condition(region_id=1, bc_type="dirichlet", value=0.0)
    manager.add_boundary_condition(region_id=2, bc_type="dirichlet", value=1.0)

    legacy_bc = manager.create_legacy_boundary_conditions()

    assert isinstance(legacy_bc, BoundaryConditions)
    assert legacy_bc.type == "dirichlet"
    assert legacy_bc.left_value == 0.0
    assert legacy_bc.right_value == 1.0


@pytest.mark.unit
def test_create_legacy_boundary_conditions_default():
    """Test legacy conversion with no conditions defaults to periodic."""
    mesh_data = create_simple_mesh_data()
    manager = BoundaryManager(mesh_data)

    legacy_bc = manager.create_legacy_boundary_conditions()

    assert legacy_bc.type == "periodic"


# ===================================================================
# Test Summary Generation
# ===================================================================


@pytest.mark.unit
def test_get_summary():
    """Test generating summary of boundary condition setup."""
    mesh_data = create_simple_mesh_data()
    manager = BoundaryManager(mesh_data)

    manager.add_boundary_condition(region_id=1, bc_type="dirichlet", value=0.0, description="Left wall")
    manager.add_boundary_condition(region_id=2, bc_type="neumann", gradient_value=0.5, description="Right wall")

    summary = manager.get_summary()

    assert "num_regions" in summary
    assert summary["num_regions"] == 2
    assert "regions" in summary
    assert 1 in summary["regions"]
    assert 2 in summary["regions"]
    assert summary["regions"][1]["type"] == "dirichlet"
    assert summary["regions"][2]["type"] == "neumann"


@pytest.mark.unit
def test_get_summary_includes_metadata():
    """Test summary includes boundary condition metadata."""
    mesh_data = create_simple_mesh_data()
    manager = BoundaryManager(mesh_data)

    manager.add_boundary_condition(region_id=1, bc_type="dirichlet", value=1.0, description="Test boundary")

    summary = manager.get_summary()

    assert summary["regions"][1]["description"] == "Test boundary"
    assert "num_nodes" in summary["regions"][1]
    assert "time_dependent" in summary["regions"][1]


@pytest.mark.unit
def test_get_summary_empty_manager():
    """Test summary generation for manager with no BCs."""
    mesh_data = create_simple_mesh_data()
    manager = BoundaryManager(mesh_data)

    summary = manager.get_summary()

    assert summary["num_regions"] == 0
    assert len(summary["regions"]) == 0


# ===================================================================
# Test Edge Cases
# ===================================================================


@pytest.mark.unit
def test_boundary_manager_empty_boundary():
    """Test manager behavior with region having no boundary nodes."""
    mesh_data = create_simple_mesh_data()
    manager = BoundaryManager(mesh_data)

    # Add BC for nonexistent region
    manager.add_boundary_condition(region_id=999, bc_type="dirichlet", value=1.0)

    # Should not raise error, but applying conditions should have no effect
    n = len(mesh_data.vertices)
    A = np.eye(n)
    b = np.zeros(n)

    A_new, b_new = manager.apply_dirichlet_conditions(A, b)

    # Matrix and RHS should be unchanged
    assert np.allclose(A_new, A)
    assert np.allclose(b_new, b)


@pytest.mark.unit
def test_geometric_bc_complex_function():
    """Test BC with complex spatial function."""

    def complex_func(coords):
        x, y = coords[:, 0], coords[:, 1]
        return np.sin(x) * np.cos(y) + x * y

    bc = GeometricBoundaryCondition(region_id=1, bc_type="dirichlet", value=complex_func)

    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    values = bc.evaluate(coords)

    assert len(values) == 3
    assert not np.allclose(values, values[0])  # Should have different values


@pytest.mark.unit
def test_robin_bc_with_function_coefficients():
    """Test Robin BC with function-based alpha and beta."""

    def alpha_func(coords):
        return coords[:, 0]

    def beta_func(coords):
        return 1.0 + coords[:, 1]

    bc = GeometricBoundaryCondition(region_id=1, bc_type="robin", alpha=alpha_func, beta=beta_func, value=0.0)

    assert callable(bc.alpha)
    assert callable(bc.beta)


@pytest.mark.unit
def test_apply_conditions_preserves_interior_nodes():
    """Test that applying BCs only affects boundary nodes."""
    mesh_data = create_simple_mesh_data()
    manager = BoundaryManager(mesh_data)

    manager.add_boundary_condition(region_id=1, bc_type="dirichlet", value=5.0)

    n = len(mesh_data.vertices)
    A_original = np.random.rand(n, n)
    b_original = np.random.rand(n)

    A = A_original.copy()
    b = b_original.copy()

    A_new, b_new = manager.apply_dirichlet_conditions(A, b)

    # Interior nodes (tag = 0) should be unchanged
    interior_mask = mesh_data.boundary_tags == 0
    interior_nodes = np.where(interior_mask)[0]

    for node in interior_nodes:
        assert np.allclose(A_new[node, :], A_original[node, :])
        assert b_new[node] == pytest.approx(b_original[node])
