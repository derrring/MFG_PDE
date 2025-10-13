"""
Unit tests for 3D boundary condition classes.

Tests comprehensive functionality of BoundaryCondition3D abstract base class
and its concrete implementations: DirichletBC3D, NeumannBC3D, RobinBC3D,
plus BoundaryConditionManager3D and MFGBoundaryHandler3D.
"""

from typing import Any

import pytest

import numpy as np
from scipy.sparse import csr_matrix, eye

from mfg_pde.geometry.base_geometry import MeshData
from mfg_pde.geometry.boundary_conditions_3d import (
    BoundaryCondition3D,
    BoundaryConditionManager3D,
    DirichletBC3D,
    MFGBoundaryHandler3D,
    NeumannBC3D,
    RobinBC3D,
    create_box_boundary_conditions,
    create_sphere_boundary_conditions,
)

# ============================================================================
# Test Fixtures - Create reusable test meshes
# ============================================================================


@pytest.fixture
def simple_3d_mesh() -> MeshData:
    """Create a simple 3D box mesh for testing."""
    # 2x2x2 grid of vertices
    x = np.linspace(0.0, 1.0, 2)
    y = np.linspace(0.0, 1.0, 2)
    z = np.linspace(0.0, 1.0, 2)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # Tetrahedral elements (simple subdivision of cube)
    elements = np.array([[0, 1, 2, 4], [1, 3, 2, 7], [1, 5, 4, 7], [2, 6, 4, 7], [1, 2, 4, 7]])

    # Boundary tags (0 = interior, 1 = boundary)
    boundary_tags = np.zeros(len(vertices), dtype=int)
    for i in range(len(vertices)):
        x_val, y_val, z_val = vertices[i]
        if (
            np.isclose(x_val, 0.0)
            or np.isclose(x_val, 1.0)
            or np.isclose(y_val, 0.0)
            or np.isclose(y_val, 1.0)
            or np.isclose(z_val, 0.0)
            or np.isclose(z_val, 1.0)
        ):
            boundary_tags[i] = 1

    # Element tags (all interior)
    element_tags = np.zeros(len(elements), dtype=int)

    # Boundary faces (triangular faces in 3D)
    boundary_faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=int)

    mesh = MeshData(
        vertices=vertices,
        elements=elements,
        element_type="tetrahedron",
        boundary_tags=boundary_tags,
        element_tags=element_tags,
        boundary_faces=boundary_faces,
        dimension=3,
    )
    mesh.boundary_markers = boundary_tags

    return mesh


@pytest.fixture
def simple_matrix_and_rhs(simple_3d_mesh: MeshData) -> tuple[csr_matrix, np.ndarray]:
    """Create simple matrix and RHS for testing."""
    n = len(simple_3d_mesh.vertices)
    matrix = eye(n, format="csr")
    rhs = np.ones(n)
    return matrix, rhs


# ============================================================================
# Test BoundaryCondition3D Abstract Base Class
# ============================================================================


class TestBoundaryCondition3DBase:
    """Test abstract base class properties and interface."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that BoundaryCondition3D cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            BoundaryCondition3D("test", region_id=0)  # type: ignore

    def test_abstract_methods_required(self) -> None:
        """Test that subclasses must implement abstract methods."""

        class IncompleteBoundaryCondition(BoundaryCondition3D):
            """Incomplete implementation missing abstract methods."""

        with pytest.raises(TypeError, match="abstract"):
            IncompleteBoundaryCondition("incomplete", region_id=0)  # type: ignore


# ============================================================================
# Test DirichletBC3D
# ============================================================================


class TestDirichletBC3D:
    """Test Dirichlet boundary condition: u = g(x,y,z,t) on boundary."""

    def test_initialization_constant_value(self) -> None:
        """Test initialization with constant value."""
        bc = DirichletBC3D(value_function=1.5, name="Test Dirichlet")

        assert bc.name == "Test Dirichlet"
        assert bc.region_id is None
        assert bc.value_function(0.0, 0.0, 0.0, 0.0) == 1.5
        assert bc.value_function(1.0, 1.0, 1.0, 1.0) == 1.5

    def test_initialization_function_value(self) -> None:
        """Test initialization with function value."""

        def value_func(x: float, y: float, z: float, t: float) -> float:
            return x + y + z + t

        bc = DirichletBC3D(value_function=value_func, name="Function Dirichlet")

        assert bc.value_function(1.0, 2.0, 3.0, 4.0) == 10.0
        assert bc.value_function(0.5, 0.5, 0.5, 0.0) == 1.5

    def test_apply_to_matrix(self, simple_3d_mesh: MeshData, simple_matrix_and_rhs: tuple[Any, Any]) -> None:
        """Test applying Dirichlet condition to matrix."""
        matrix, _ = simple_matrix_and_rhs
        bc = DirichletBC3D(0.0)

        # Apply to boundary vertices
        boundary_indices = np.array([0, 1, 2])

        matrix_mod = bc.apply_to_matrix(matrix, simple_3d_mesh, boundary_indices)

        # Check that boundary rows are set to identity
        for idx in boundary_indices:
            row = matrix_mod.getrow(idx).toarray()[0]
            assert row[idx] == 1.0
            assert np.sum(np.abs(row)) == 1.0

    def test_apply_to_rhs_constant(self, simple_3d_mesh: MeshData, simple_matrix_and_rhs: tuple[Any, Any]) -> None:
        """Test applying constant Dirichlet value to RHS."""
        _, rhs = simple_matrix_and_rhs
        bc = DirichletBC3D(2.5)

        boundary_indices = np.array([0, 1, 2])
        rhs_mod = bc.apply_to_rhs(rhs, simple_3d_mesh, boundary_indices, time=0.0)

        for idx in boundary_indices:
            assert rhs_mod[idx] == 2.5

    def test_apply_to_rhs_function(self, simple_3d_mesh: MeshData, simple_matrix_and_rhs: tuple[Any, Any]) -> None:
        """Test applying function-based Dirichlet value to RHS."""
        _, rhs = simple_matrix_and_rhs

        def value_func(x: float, y: float, z: float, t: float) -> float:
            return x**2 + y**2 + z**2

        bc = DirichletBC3D(value_func)

        boundary_indices = np.array([0, 1, 2])
        rhs_mod = bc.apply_to_rhs(rhs, simple_3d_mesh, boundary_indices, time=0.0)

        for idx in boundary_indices:
            x, y, z = simple_3d_mesh.vertices[idx]
            expected = x**2 + y**2 + z**2
            assert np.isclose(rhs_mod[idx], expected)

    def test_validate_mesh_compatibility(self, simple_3d_mesh: MeshData) -> None:
        """Test mesh compatibility validation."""
        bc = DirichletBC3D(0.0)

        assert bc.validate_mesh_compatibility(simple_3d_mesh) is True

        # Test with mesh lacking boundary_markers
        mesh_no_markers = MeshData(
            vertices=simple_3d_mesh.vertices,
            elements=simple_3d_mesh.elements,
            element_type="tetrahedron",
            boundary_tags=np.zeros(len(simple_3d_mesh.vertices), dtype=int),
            element_tags=np.zeros(len(simple_3d_mesh.elements), dtype=int),
            boundary_faces=np.array([[0, 1, 2]], dtype=int),
            dimension=3,
        )
        assert bc.validate_mesh_compatibility(mesh_no_markers) is False


# ============================================================================
# Test NeumannBC3D
# ============================================================================


class TestNeumannBC3D:
    """Test Neumann boundary condition: ∇u·n = g(x,y,z,t) on boundary."""

    def test_initialization_constant_flux(self) -> None:
        """Test initialization with constant flux."""
        bc = NeumannBC3D(flux_function=0.5, name="No Flux")

        assert bc.name == "No Flux"
        assert bc.flux_function(0.0, 0.0, 0.0, 0.0) == 0.5

    def test_initialization_function_flux(self) -> None:
        """Test initialization with function flux."""

        def flux_func(x: float, y: float, z: float, t: float) -> float:
            return x * y * z

        bc = NeumannBC3D(flux_function=flux_func)

        assert bc.flux_function(2.0, 3.0, 4.0, 0.0) == 24.0

    def test_apply_to_matrix(self, simple_3d_mesh: MeshData, simple_matrix_and_rhs: tuple[Any, Any]) -> None:
        """Test that Neumann condition doesn't modify matrix."""
        matrix, _ = simple_matrix_and_rhs
        bc = NeumannBC3D(0.0)

        boundary_indices = np.array([0, 1, 2])
        matrix_mod = bc.apply_to_matrix(matrix, simple_3d_mesh, boundary_indices)

        assert np.allclose(matrix_mod.toarray(), matrix.toarray())

    def test_apply_to_rhs_zero_flux(self, simple_3d_mesh: MeshData, simple_matrix_and_rhs: tuple[Any, Any]) -> None:
        """Test applying zero flux to RHS."""
        _, rhs = simple_matrix_and_rhs
        bc = NeumannBC3D(0.0, name="No Flux")

        boundary_indices = np.array([0, 1, 2])
        rhs_original = rhs.copy()
        rhs_mod = bc.apply_to_rhs(rhs, simple_3d_mesh, boundary_indices, time=0.0)

        assert np.allclose(rhs_mod, rhs_original)

    def test_validate_mesh_compatibility(self, simple_3d_mesh: MeshData) -> None:
        """Test mesh compatibility validation."""
        bc = NeumannBC3D(0.0)

        result = bc.validate_mesh_compatibility(simple_3d_mesh)
        assert isinstance(result, bool)


# ============================================================================
# Test RobinBC3D
# ============================================================================


class TestRobinBC3D:
    """Test Robin boundary condition: α·u + β·∇u·n = g(x,y,z,t)."""

    def test_initialization_constant_coefficients(self) -> None:
        """Test initialization with constant coefficients."""
        bc = RobinBC3D(alpha=1.0, beta=2.0, value_function=3.0, name="Robin Test")

        assert bc.name == "Robin Test"
        assert bc.alpha(0.0, 0.0, 0.0, 0.0) == 1.0
        assert bc.beta(0.0, 0.0, 0.0, 0.0) == 2.0
        assert bc.value_function(0.0, 0.0, 0.0, 0.0) == 3.0

    def test_initialization_function_coefficients(self) -> None:
        """Test initialization with function coefficients."""

        def alpha_func(x: float, y: float, z: float, t: float) -> float:
            return x + 1.0

        def beta_func(x: float, y: float, z: float, t: float) -> float:
            return y + 2.0

        bc = RobinBC3D(alpha=alpha_func, beta=beta_func, value_function=0.0)

        assert bc.alpha(1.0, 0.0, 0.0, 0.0) == 2.0
        assert bc.beta(0.0, 3.0, 0.0, 0.0) == 5.0

    def test_apply_to_matrix(self, simple_3d_mesh: MeshData, simple_matrix_and_rhs: tuple[Any, Any]) -> None:
        """Test applying Robin condition to matrix."""
        matrix, _ = simple_matrix_and_rhs
        bc = RobinBC3D(alpha=1.0, beta=0.0, value_function=0.0)

        boundary_indices = np.array([0, 1, 2])
        matrix_mod = bc.apply_to_matrix(matrix, simple_3d_mesh, boundary_indices)

        for idx in boundary_indices:
            assert matrix_mod[idx, idx] > matrix[idx, idx]

    def test_apply_to_rhs(self, simple_3d_mesh: MeshData, simple_matrix_and_rhs: tuple[Any, Any]) -> None:
        """Test applying Robin condition to RHS."""
        _, rhs = simple_matrix_and_rhs
        bc = RobinBC3D(alpha=1.0, beta=1.0, value_function=2.5)

        boundary_indices = np.array([0, 1, 2])
        rhs_mod = bc.apply_to_rhs(rhs, simple_3d_mesh, boundary_indices, time=0.0)

        for idx in boundary_indices:
            assert rhs_mod[idx] == rhs[idx] + 2.5

    def test_validate_mesh_compatibility(self, simple_3d_mesh: MeshData) -> None:
        """Test mesh compatibility validation."""
        bc = RobinBC3D(alpha=1.0, beta=1.0, value_function=0.0)

        assert bc.validate_mesh_compatibility(simple_3d_mesh) is True


# ============================================================================
# Test BoundaryConditionManager3D
# ============================================================================


class TestBoundaryConditionManager3D:
    """Test boundary condition manager for 3D domains."""

    def test_initialization(self) -> None:
        """Test manager initialization."""
        manager = BoundaryConditionManager3D()

        assert len(manager.conditions) == 0
        assert len(manager.region_map) == 0

    def test_add_condition_with_region_id(self) -> None:
        """Test adding condition with integer region ID."""
        manager = BoundaryConditionManager3D()
        bc = DirichletBC3D(0.0, name="Test")

        manager.add_condition(bc, boundary_region=0)

        assert len(manager.conditions) == 1
        assert 0 in manager.region_map
        assert bc in manager.region_map[0]

    def test_add_condition_with_direct_vertices(self) -> None:
        """Test adding condition with direct vertex indices."""
        manager = BoundaryConditionManager3D()
        bc = DirichletBC3D(0.0)

        vertices = np.array([0, 1, 2])
        manager.add_condition(bc, boundary_region=vertices)

        assert len(manager.conditions) == 1
        assert np.array_equal(bc._direct_vertices, vertices)

    def test_identify_boundary_vertices(self, simple_3d_mesh: MeshData) -> None:
        """Test boundary vertex identification."""
        manager = BoundaryConditionManager3D()

        boundary_vertices = manager.identify_boundary_vertices(simple_3d_mesh)

        assert "x_min" in boundary_vertices
        assert "x_max" in boundary_vertices
        assert "y_min" in boundary_vertices
        assert "y_max" in boundary_vertices
        assert "z_min" in boundary_vertices
        assert "z_max" in boundary_vertices

    def test_validate_all_conditions(self, simple_3d_mesh: MeshData) -> None:
        """Test validation of all conditions."""
        manager = BoundaryConditionManager3D()

        bc1 = DirichletBC3D(0.0)
        bc2 = RobinBC3D(1.0, 1.0, 0.0)

        manager.add_condition(bc1, boundary_region=0)
        manager.add_condition(bc2, boundary_region=1)

        assert manager.validate_all_conditions(simple_3d_mesh) is True


# ============================================================================
# Test MFGBoundaryHandler3D
# ============================================================================


class TestMFGBoundaryHandler3D:
    """Test MFG-specific boundary condition handler for 3D."""

    def test_initialization(self) -> None:
        """Test handler initialization."""
        handler = MFGBoundaryHandler3D()

        assert isinstance(handler.hjb_manager, BoundaryConditionManager3D)
        assert isinstance(handler.fp_manager, BoundaryConditionManager3D)

    def test_add_state_constraint(self) -> None:
        """Test adding state constraint for HJB equation."""
        handler = MFGBoundaryHandler3D()

        def constraint(x: float, y: float, z: float, t: float) -> float:
            return 0.0

        handler.add_state_constraint(region=0, constraint_function=constraint)

        assert len(handler.hjb_manager.conditions) == 1
        assert isinstance(handler.hjb_manager.conditions[0], DirichletBC3D)

    def test_add_no_flux_condition(self) -> None:
        """Test adding no-flux condition for FP equation."""
        handler = MFGBoundaryHandler3D()

        handler.add_no_flux_condition(region=0)

        assert len(handler.fp_manager.conditions) == 1
        assert isinstance(handler.fp_manager.conditions[0], NeumannBC3D)

    def test_validate_mfg_compatibility(self, simple_3d_mesh: MeshData) -> None:
        """Test MFG compatibility validation."""
        handler = MFGBoundaryHandler3D()

        handler.add_state_constraint(region=0, constraint_function=lambda x, y, z, t: 0.0)
        handler.add_no_flux_condition(region=0)

        result = handler.validate_mfg_compatibility(simple_3d_mesh)
        assert isinstance(result, bool)


# ============================================================================
# Test Factory Functions
# ============================================================================


class TestFactoryFunctions:
    """Test factory functions for common 3D boundary condition scenarios."""

    def test_create_box_boundary_conditions_dirichlet_zero(self) -> None:
        """Test creating zero Dirichlet conditions for box."""
        bounds = (0.0, 1.0, 0.0, 2.0, 0.0, 3.0)
        manager = create_box_boundary_conditions(bounds, condition_type="dirichlet_zero")

        assert len(manager.conditions) == 6  # 6 faces of box
        assert all(isinstance(bc, DirichletBC3D) for bc in manager.conditions)

    def test_create_box_boundary_conditions_neumann_zero(self) -> None:
        """Test creating zero Neumann conditions for box."""
        bounds = (0.0, 1.0, 0.0, 2.0, 0.0, 3.0)
        manager = create_box_boundary_conditions(bounds, condition_type="neumann_zero")

        assert len(manager.conditions) == 6
        assert all(isinstance(bc, NeumannBC3D) for bc in manager.conditions)

    def test_create_sphere_boundary_conditions_dirichlet(self) -> None:
        """Test creating Dirichlet conditions for sphere."""
        center = (0.5, 0.5, 0.5)
        radius = 0.3
        manager = create_sphere_boundary_conditions(center, radius, condition_type="dirichlet_zero")

        assert len(manager.conditions) == 1
        assert isinstance(manager.conditions[0], DirichletBC3D)

    def test_create_sphere_boundary_conditions_neumann(self) -> None:
        """Test creating Neumann conditions for sphere."""
        center = (0.5, 0.5, 0.5)
        radius = 0.3
        manager = create_sphere_boundary_conditions(center, radius, condition_type="neumann_zero")

        assert len(manager.conditions) == 1
        assert isinstance(manager.conditions[0], NeumannBC3D)


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_boundary_indices(self, simple_3d_mesh: MeshData, simple_matrix_and_rhs: tuple[Any, Any]) -> None:
        """Test applying conditions with empty boundary indices."""
        matrix, rhs = simple_matrix_and_rhs
        bc = DirichletBC3D(0.0)

        boundary_indices = np.array([], dtype=int)

        matrix_mod = bc.apply_to_matrix(matrix, simple_3d_mesh, boundary_indices)
        rhs_mod = bc.apply_to_rhs(rhs, simple_3d_mesh, boundary_indices)

        assert np.allclose(matrix_mod.toarray(), matrix.toarray())
        assert np.allclose(rhs_mod, rhs)

    def test_time_dependent_boundary_condition(
        self, simple_3d_mesh: MeshData, simple_matrix_and_rhs: tuple[Any, Any]
    ) -> None:
        """Test time-dependent boundary condition."""
        _, rhs = simple_matrix_and_rhs

        def time_dependent_value(x: float, y: float, z: float, t: float) -> float:
            return t * (x + y + z)

        bc = DirichletBC3D(time_dependent_value)

        boundary_indices = np.array([0, 1])

        rhs_t0 = bc.apply_to_rhs(rhs, simple_3d_mesh, boundary_indices, time=0.0)
        rhs_t1 = bc.apply_to_rhs(rhs, simple_3d_mesh, boundary_indices, time=1.0)

        assert not np.allclose(rhs_t0[boundary_indices], rhs_t1[boundary_indices])

    def test_large_coefficient_values(self, simple_3d_mesh: MeshData, simple_matrix_and_rhs: tuple[Any, Any]) -> None:
        """Test Robin BC with large coefficient values."""
        matrix, _ = simple_matrix_and_rhs
        bc = RobinBC3D(alpha=1e6, beta=1e-6, value_function=0.0)

        boundary_indices = np.array([0])

        matrix_mod = bc.apply_to_matrix(matrix, simple_3d_mesh, boundary_indices)

        assert np.isfinite(matrix_mod.toarray()).all()

    def test_validation_with_incompatible_mesh(self) -> None:
        """Test validation with incompatible mesh."""
        manager = BoundaryConditionManager3D()
        bc = NeumannBC3D(0.0)
        manager.add_condition(bc, boundary_region=0)

        mesh = MeshData(
            vertices=np.array([[0.0, 0.0, 0.0]]),
            elements=np.array([[0]]),
            element_type="tetrahedron",
            boundary_tags=np.array([0], dtype=int),
            element_tags=np.array([0], dtype=int),
            boundary_faces=np.array([[0, 0, 0]], dtype=int),
            dimension=3,
        )

        with pytest.warns(UserWarning, match="incompatible with mesh"):
            result = manager.validate_all_conditions(mesh)

        assert result is False
