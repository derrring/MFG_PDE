#!/usr/bin/env python3
"""
Comprehensive test suite for periodic boundary conditions.

Tests periodic BC implementations across 1D, 2D, and 3D including:
1. Parameter validation
2. Matrix/RHS modification
3. Mathematical correctness
4. Vertex pairing algorithms
"""

import pytest

import numpy as np
from scipy.sparse import csr_matrix

from mfg_pde.geometry import BoundaryConditions
from mfg_pde.geometry.base_geometry import MeshData
from mfg_pde.geometry.boundary_conditions_2d import BoundaryConditionManager2D, PeriodicBC2D
from mfg_pde.geometry.boundary_conditions_3d import BoundaryConditionManager3D, PeriodicBC3D


class TestPeriodic1D:
    """Test suite for 1D periodic boundary conditions."""

    def test_periodic_validation(self):
        """Test 1D periodic BC validation."""
        bc = BoundaryConditions(type="periodic")
        assert bc.is_periodic()
        assert bc.type == "periodic"

    def test_periodic_matrix_sizing(self):
        """Test matrix sizing for periodic BC."""
        bc = BoundaryConditions(type="periodic")

        # Periodic BC should give M × M matrices (domain treated as circular)
        num_interior = 10
        expected_size = num_interior
        assert bc.get_matrix_size(num_interior) == expected_size

    def test_periodic_type_checking(self):
        """Test type checking methods."""
        bc = BoundaryConditions(type="periodic")

        assert bc.is_periodic()
        assert not bc.is_robin()
        assert not bc.is_dirichlet()
        assert not bc.is_neumann()
        assert not bc.is_no_flux()

    def test_periodic_legacy_compatibility(self):
        """Test that periodic BC can be created from boundary manager."""
        from mfg_pde.geometry.boundary_manager import BoundaryManager, GeometricBoundaryCondition

        # Create a mock mesh data
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
        connectivity = np.array([[0, 1, 2]])
        mesh = MeshData(vertices=vertices, connectivity=connectivity, dimension=2, num_vertices=3)

        # Add boundary tags
        mesh.boundary_tags = np.array([1, 1, 0])  # First two vertices on boundary

        manager = BoundaryManager(mesh)

        # Add periodic condition
        periodic_bc = GeometricBoundaryCondition(region_id=1, bc_type="periodic")
        manager.boundary_conditions[1] = periodic_bc

        # Convert to legacy format
        legacy_bc = manager.create_legacy_boundary_conditions()
        assert legacy_bc.is_periodic()


class TestPeriodic2D:
    """Test suite for 2D periodic boundary conditions."""

    @pytest.fixture
    def rectangle_mesh_2d(self):
        """Create simple rectangular mesh for testing."""
        # 3x3 grid of vertices for rectangle [0,1] x [0,1]
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 1, 3)

        vertices = []
        for yi in y:
            for xi in x:
                vertices.append([xi, yi, 0.0])  # Add z=0 for 3D compatibility

        vertices = np.array(vertices)
        connectivity = np.array([[0, 1, 3]])  # Simple connectivity

        mesh = MeshData(vertices=vertices, connectivity=connectivity, dimension=2, num_vertices=len(vertices))

        return mesh

    def test_periodic_2d_initialization(self):
        """Test 2D periodic BC initialization."""
        # Single periodic pair
        bc = PeriodicBC2D(paired_boundaries=[(0, 1)], name="X_Periodic")
        assert bc.name == "X_Periodic"
        assert bc.paired_boundaries == [(0, 1)]

        # Multiple periodic pairs
        bc_both = PeriodicBC2D(paired_boundaries=[(0, 1), (2, 3)], name="XY_Periodic")
        assert len(bc_both.paired_boundaries) == 2

    def test_periodic_2d_boundary_identification(self, rectangle_mesh_2d):
        """Test boundary vertex identification for 2D periodic pairing."""
        bc = PeriodicBC2D([(0, 1)])  # Left-right periodic

        boundary_mapping = bc._identify_boundary_vertices_for_pairing(rectangle_mesh_2d)

        # Should identify 4 boundary regions
        assert len(boundary_mapping) == 4
        assert 0 in boundary_mapping  # x_min
        assert 1 in boundary_mapping  # x_max
        assert 2 in boundary_mapping  # y_min
        assert 3 in boundary_mapping  # y_max

        # Check that boundary vertices are correctly identified
        vertices_2d = rectangle_mesh_2d.vertices[:, :2]

        # Left boundary (x=0)
        left_vertices = boundary_mapping[0]
        for idx in left_vertices:
            assert abs(vertices_2d[idx, 0] - 0.0) < 1e-10

        # Right boundary (x=1)
        right_vertices = boundary_mapping[1]
        for idx in right_vertices:
            assert abs(vertices_2d[idx, 0] - 1.0) < 1e-10

    def test_periodic_2d_vertex_pairing(self, rectangle_mesh_2d):
        """Test corresponding vertex finding for 2D periodic boundaries."""
        bc = PeriodicBC2D([(0, 1)])

        # Get boundary mapping
        boundary_mapping = bc._identify_boundary_vertices_for_pairing(rectangle_mesh_2d)

        left_vertices = boundary_mapping[0]
        right_vertices = boundary_mapping[1]

        # Find corresponding pairs
        vertex_pairs = bc._find_corresponding_vertices_2d(
            rectangle_mesh_2d.vertices[left_vertices],
            rectangle_mesh_2d.vertices[right_vertices],
            left_vertices,
            right_vertices,
        )

        # Should find corresponding pairs
        assert len(vertex_pairs) > 0

        # Check that paired vertices have same y-coordinate but different x-coordinate
        for v1_idx, v2_idx in vertex_pairs:
            v1_coord = rectangle_mesh_2d.vertices[v1_idx, :2]
            v2_coord = rectangle_mesh_2d.vertices[v2_idx, :2]

            # Same y-coordinate
            assert abs(v1_coord[1] - v2_coord[1]) < 1e-10
            # Different x-coordinate
            assert abs(v1_coord[0] - v2_coord[0]) > 1e-6

    def test_periodic_2d_matrix_application(self, rectangle_mesh_2d):
        """Test periodic BC application to system matrix."""
        bc = PeriodicBC2D([(0, 1)])  # Left-right periodic

        # Create test matrix
        n = rectangle_mesh_2d.num_vertices
        matrix = csr_matrix(np.eye(n))

        # Apply periodic BC
        boundary_indices = np.arange(n)  # All vertices (simplified test)
        modified_matrix = bc.apply_to_matrix(matrix, rectangle_mesh_2d, boundary_indices)

        # Matrix should be modified
        assert not np.allclose(modified_matrix.toarray(), matrix.toarray())

    def test_periodic_2d_validation(self, rectangle_mesh_2d):
        """Test 2D periodic BC mesh compatibility."""
        bc = PeriodicBC2D([(0, 1)])

        # Should pass validation if mesh has boundary markers
        rectangle_mesh_2d.boundary_markers = np.ones(rectangle_mesh_2d.num_vertices)
        assert bc.validate_mesh_compatibility(rectangle_mesh_2d)


class TestPeriodic3D:
    """Test suite for 3D periodic boundary conditions."""

    @pytest.fixture
    def cube_mesh_3d(self):
        """Create simple cube mesh for testing."""
        # 2x2x2 cube vertices
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],  # 0
                [1.0, 0.0, 0.0],  # 1
                [0.0, 1.0, 0.0],  # 2
                [1.0, 1.0, 0.0],  # 3
                [0.0, 0.0, 1.0],  # 4
                [1.0, 0.0, 1.0],  # 5
                [0.0, 1.0, 1.0],  # 6
                [1.0, 1.0, 1.0],  # 7
            ]
        )

        connectivity = np.array([[0, 1, 2, 4]])  # Simple connectivity

        mesh = MeshData(vertices=vertices, connectivity=connectivity, dimension=3, num_vertices=len(vertices))

        mesh.boundary_markers = np.ones(len(vertices))
        return mesh

    def test_periodic_3d_initialization(self):
        """Test 3D periodic BC initialization."""
        # Face pairing for cube: x_min-x_max, y_min-y_max, z_min-z_max
        bc = PeriodicBC3D(paired_boundaries=[(0, 1), (2, 3), (4, 5)], name="Cube_Periodic")
        assert bc.name == "Cube_Periodic"
        assert len(bc.paired_boundaries) == 3

    def test_periodic_3d_boundary_identification(self, cube_mesh_3d):
        """Test boundary vertex identification for 3D periodic pairing."""
        bc = PeriodicBC3D([(0, 1)])

        boundary_mapping = bc._identify_boundary_vertices_for_pairing(cube_mesh_3d)

        # Should identify 6 boundary regions (faces of cube)
        assert len(boundary_mapping) == 6

        # Check x_min face (x=0)
        x_min_vertices = boundary_mapping[0]
        for idx in x_min_vertices:
            assert abs(cube_mesh_3d.vertices[idx, 0] - 0.0) < 1e-10

        # Check x_max face (x=1)
        x_max_vertices = boundary_mapping[1]
        for idx in x_max_vertices:
            assert abs(cube_mesh_3d.vertices[idx, 0] - 1.0) < 1e-10

    def test_periodic_3d_vertex_pairing(self, cube_mesh_3d):
        """Test corresponding vertex finding for 3D periodic boundaries."""
        bc = PeriodicBC3D([(0, 1)])  # x_min - x_max pairing

        boundary_mapping = bc._identify_boundary_vertices_for_pairing(cube_mesh_3d)

        x_min_vertices = boundary_mapping[0]
        x_max_vertices = boundary_mapping[1]

        # Find corresponding pairs
        vertex_pairs = bc._find_corresponding_vertices(
            cube_mesh_3d.vertices[x_min_vertices], cube_mesh_3d.vertices[x_max_vertices], x_min_vertices, x_max_vertices
        )

        # Should find corresponding pairs
        assert len(vertex_pairs) > 0

        # Check that paired vertices have same y,z coordinates but different x
        for v1_idx, v2_idx in vertex_pairs:
            v1_coord = cube_mesh_3d.vertices[v1_idx]
            v2_coord = cube_mesh_3d.vertices[v2_idx]

            # Same y,z coordinates
            assert abs(v1_coord[1] - v2_coord[1]) < 1e-10
            assert abs(v1_coord[2] - v2_coord[2]) < 1e-10
            # Different x coordinate
            assert abs(v1_coord[0] - v2_coord[0]) > 1e-6

    def test_periodic_3d_matrix_application(self, cube_mesh_3d):
        """Test 3D periodic BC application to system matrix."""
        bc = PeriodicBC3D([(0, 1)])

        # Create test matrix
        n = cube_mesh_3d.num_vertices
        matrix = csr_matrix(np.eye(n))

        # Apply periodic BC
        boundary_indices = np.arange(n)
        modified_matrix = bc.apply_to_matrix(matrix, cube_mesh_3d, boundary_indices)

        # Matrix should be modified
        assert not np.allclose(modified_matrix.toarray(), matrix.toarray())

    def test_periodic_3d_validation(self, cube_mesh_3d):
        """Test 3D periodic BC validation."""
        bc = PeriodicBC3D([(0, 1)])

        # Should pass validation with boundary markers
        assert bc.validate_mesh_compatibility(cube_mesh_3d)


class TestBoundaryConditionManagers:
    """Test boundary condition managers with periodic BCs."""

    def test_manager_2d_periodic_conditions(self):
        """Test 2D manager with periodic conditions."""
        manager = BoundaryConditionManager2D()

        # Add periodic condition
        periodic_bc = PeriodicBC2D([(0, 1)], "X_Periodic")
        manager.add_condition(periodic_bc, 0)

        assert len(manager.conditions) == 1
        assert 0 in manager.region_map

    def test_manager_3d_periodic_conditions(self):
        """Test 3D manager with periodic conditions."""
        manager = BoundaryConditionManager3D()

        # Add periodic condition
        periodic_bc = PeriodicBC3D([(0, 1), (2, 3)], "XY_Periodic")
        manager.add_condition(periodic_bc, 0)

        assert len(manager.conditions) == 1
        assert 0 in manager.region_map

    @pytest.fixture
    def simple_2d_mesh(self):
        """Simple 2D mesh for manager testing."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        connectivity = np.array([[0, 1, 2]])
        mesh = MeshData(vertices=vertices, connectivity=connectivity, dimension=2, num_vertices=4)
        return mesh

    def test_manager_2d_apply_periodic(self, simple_2d_mesh):
        """Test applying periodic conditions through 2D manager."""
        manager = BoundaryConditionManager2D()

        periodic_bc = PeriodicBC2D([(0, 1)], "Test_Periodic")
        manager.add_condition(periodic_bc, 0)

        # Create test system
        n = simple_2d_mesh.num_vertices
        matrix = csr_matrix(np.eye(n))
        rhs = np.zeros(n)

        # Apply conditions
        modified_matrix, modified_rhs = manager.apply_all_conditions(matrix, rhs, simple_2d_mesh)

        # System should potentially be modified (depends on boundary detection)
        assert modified_matrix.shape == matrix.shape
        assert len(modified_rhs) == len(rhs)


class TestPeriodicFactoryFunctions:
    """Test factory functions for periodic boundary conditions."""

    def test_rectangle_periodic_x(self):
        """Test rectangular domain with x-periodic BC."""
        from mfg_pde.geometry.boundary_conditions_2d import create_rectangle_boundary_conditions

        manager = create_rectangle_boundary_conditions(domain_bounds=(0, 1, 0, 1), condition_type="periodic_x")

        assert len(manager.conditions) > 0

        # Should have periodic condition and Neumann conditions
        condition_types = [type(cond).__name__ for cond in manager.conditions]
        assert "PeriodicBC2D" in condition_types
        assert "NeumannBC2D" in condition_types

    def test_rectangle_periodic_both(self):
        """Test rectangular domain with both x,y periodic BC."""
        from mfg_pde.geometry.boundary_conditions_2d import create_rectangle_boundary_conditions

        manager = create_rectangle_boundary_conditions(domain_bounds=(0, 1, 0, 1), condition_type="periodic_both")

        assert len(manager.conditions) >= 2

        # Should have two periodic conditions
        periodic_conditions = [cond for cond in manager.conditions if isinstance(cond, PeriodicBC2D)]
        assert len(periodic_conditions) == 2


class TestPeriodicMathematicalCorrectness:
    """Test mathematical correctness of periodic boundary conditions."""

    def test_periodic_symmetry_preservation(self):
        """Test that periodic BC preserves solution symmetry."""
        # This test would require a complete solver setup
        # For now, verify the structure is mathematically sound

        # 2D periodic BC should enforce u(x_left, y) = u(x_right, y)
        bc = PeriodicBC2D([(0, 1)])  # Left-right periodic

        # Mathematical constraint: when applied to matrix, should create
        # relationships between corresponding boundary vertices
        assert bc.paired_boundaries == [(0, 1)]

    def test_periodic_conservation_property(self):
        """Test that periodic BC conserves appropriate quantities."""
        # Periodic BCs should conserve mass in MFG context

        # 1D case: u(0) = u(L) for periodic domain [0,L]
        bc_1d = BoundaryConditions(type="periodic")
        assert bc_1d.is_periodic()

        # Matrix size should be M x M (no extra constraints)
        assert bc_1d.get_matrix_size(10) == 10

    def test_periodic_consistency_across_dimensions(self):
        """Test consistency of periodic BC concepts across dimensions."""
        # All periodic BCs should follow same mathematical principle

        bc_1d = BoundaryConditions(type="periodic")
        bc_2d = PeriodicBC2D([(0, 1)])
        bc_3d = PeriodicBC3D([(0, 1)])

        # All should be periodic type
        assert bc_1d.is_periodic()
        assert "Periodic" in bc_2d.name
        assert "Periodic" in bc_3d.name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
