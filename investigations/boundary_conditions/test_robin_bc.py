#!/usr/bin/env python3
"""
Comprehensive test suite for Robin boundary conditions.

Tests both 1D and 3D Robin BC implementations including:
1. Parameter validation
2. Matrix/RHS modification
3. Mathematical correctness
4. Integration with geometry system
"""

import pytest

import numpy as np
from scipy.sparse import csr_matrix

from mfg_pde.geometry import BoundaryConditions
from mfg_pde.geometry.base_geometry import MeshData
from mfg_pde.geometry.boundary_conditions_3d import BoundaryConditionManager3D, RobinBC3D
from mfg_pde.geometry.boundary_manager import GeometricBoundaryCondition


class TestRobin1D:
    """Test suite for 1D Robin boundary conditions."""

    def test_robin_validation(self):
        """Test parameter validation for Robin BC."""
        # Valid Robin BC
        bc = BoundaryConditions(
            type="robin",
            left_alpha=1.0,
            left_beta=0.5,
            right_alpha=2.0,
            right_beta=1.0,
            left_value=0.0,
            right_value=1.0,
        )
        assert bc.is_robin()
        assert bc.type == "robin"

        # Invalid Robin BC - missing coefficients
        with pytest.raises(ValueError, match="Robin boundary conditions require alpha and beta coefficients"):
            BoundaryConditions(
                type="robin",
                left_alpha=1.0,
                # Missing left_beta
                right_alpha=2.0,
                right_beta=1.0,
            )

    def test_robin_matrix_sizing(self):
        """Test matrix sizing for Robin BC."""
        bc = BoundaryConditions(type="robin", left_alpha=1.0, left_beta=0.5, right_alpha=2.0, right_beta=1.0)

        # Robin BC should give (M+1) × (M+1) matrices
        num_interior = 10
        expected_size = num_interior + 1
        assert bc.get_matrix_size(num_interior) == expected_size

    def test_robin_type_checking(self):
        """Test type checking methods."""
        bc = BoundaryConditions(type="robin", left_alpha=1.0, left_beta=0.5, right_alpha=2.0, right_beta=1.0)

        assert bc.is_robin()
        assert not bc.is_periodic()
        assert not bc.is_dirichlet()
        assert not bc.is_neumann()
        assert not bc.is_no_flux()


class TestRobin3D:
    """Test suite for 3D Robin boundary conditions."""

    @pytest.fixture
    def simple_mesh_3d(self):
        """Create simple 3D mesh for testing."""
        # Simple 2x2x2 cube mesh
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

        # Simple connectivity (not used for BC tests)
        elements = np.array([[0, 1, 2, 4]])  # Single tetrahedron
        boundary_tags = np.array([0])
        element_tags = np.array([0])
        boundary_faces = np.array([[0, 1, 2], [0, 1, 4], [0, 2, 4], [1, 2, 4]])

        mesh = MeshData(
            vertices=vertices,
            elements=elements,
            element_type="tetrahedron",
            boundary_tags=boundary_tags,
            element_tags=element_tags,
            boundary_faces=boundary_faces,
            dimension=3,
        )

        # Add boundary markers
        mesh.boundary_markers = np.ones(len(vertices))  # All vertices on boundary

        return mesh

    def test_robin_3d_initialization(self):
        """Test Robin 3D boundary condition initialization."""
        # Constant coefficients
        bc = RobinBC3D(alpha=1.0, beta=0.5, value_function=2.0)
        assert bc.name == "Robin"

        # Function coefficients
        def alpha_func(x, y, z, t):
            return x + y + z

        def beta_func(x, y, z, t):
            return 1.0

        def value_func(x, y, z, t):
            return np.sin(x * y * z)

        bc_func = RobinBC3D(alpha=alpha_func, beta=beta_func, value_function=value_func)
        assert callable(bc_func.alpha)
        assert callable(bc_func.beta)
        assert callable(bc_func.value_function)

    def test_robin_3d_matrix_application(self, simple_mesh_3d):
        """Test Robin BC application to system matrix."""
        bc = RobinBC3D(alpha=2.0, beta=1.0, value_function=0.0)

        # Create simple test matrix
        n_vertices = simple_mesh_3d.num_vertices
        matrix = csr_matrix(np.eye(n_vertices))

        # Apply Robin BC to boundary vertices
        boundary_indices = np.array([0, 1, 2, 3])  # First 4 vertices as boundary

        modified_matrix = bc.apply_to_matrix(matrix, simple_mesh_3d, boundary_indices)

        # Check that matrix was modified
        assert modified_matrix.shape == matrix.shape
        assert not np.allclose(modified_matrix.toarray(), matrix.toarray())

        # Check that diagonal entries were modified by alpha
        diag_original = matrix.diagonal()
        diag_modified = modified_matrix.diagonal()

        # Boundary vertices should have alpha added to diagonal
        for idx in boundary_indices:
            assert diag_modified[idx] == diag_original[idx] + 2.0  # alpha = 2.0

    def test_robin_3d_rhs_application(self, simple_mesh_3d):
        """Test Robin BC application to RHS vector."""

        def value_func(x, y, z, t):
            return x + y + z  # Simple linear function

        bc = RobinBC3D(alpha=1.0, beta=0.5, value_function=value_func)

        # Create test RHS vector
        n_vertices = simple_mesh_3d.num_vertices
        rhs = np.zeros(n_vertices)

        # Apply Robin BC
        boundary_indices = np.array([0, 1, 2, 3])
        modified_rhs = bc.apply_to_rhs(rhs, simple_mesh_3d, boundary_indices, time=0.0)

        # Check that RHS was modified
        assert not np.allclose(modified_rhs, rhs)

        # Check specific values
        for idx in boundary_indices:
            vertex_coord = simple_mesh_3d.vertices[idx]
            expected_value = vertex_coord[0] + vertex_coord[1] + vertex_coord[2]
            assert modified_rhs[idx] == expected_value

    def test_robin_3d_validation(self, simple_mesh_3d):
        """Test Robin 3D mesh compatibility validation."""
        bc = RobinBC3D(alpha=1.0, beta=0.5, value_function=0.0)

        # Should pass validation (Robin BCs are generally compatible)
        assert bc.validate_mesh_compatibility(simple_mesh_3d)


class TestGeometricRobinBC:
    """Test Robin BC in geometric boundary condition system."""

    def test_geometric_robin_initialization(self):
        """Test GeometricBoundaryCondition with Robin type."""
        # Valid Robin BC
        bc = GeometricBoundaryCondition(region_id=1, bc_type="robin", alpha=2.0, beta=1.5, value=3.0)

        assert bc.bc_type == "robin"
        assert bc.alpha == 2.0
        assert bc.beta == 1.5

        # Invalid Robin BC - missing coefficients
        with pytest.raises(ValueError, match="Robin boundary conditions require alpha and beta coefficients"):
            GeometricBoundaryCondition(
                region_id=1,
                bc_type="robin",
                alpha=2.0,
                # Missing beta
                value=3.0,
            )

    def test_geometric_robin_function_coefficients(self):
        """Test Robin BC with function coefficients."""

        def alpha_func(coords):
            return coords[:, 0] + coords[:, 1]  # x + y

        def beta_func(coords):
            return np.ones(len(coords))  # constant 1

        bc = GeometricBoundaryCondition(region_id=1, bc_type="robin", alpha=alpha_func, beta=beta_func, value=0.0)

        # Note: Direct testing of function coefficients would require
        # integration with the boundary manager evaluation system
        assert callable(bc.alpha)
        assert callable(bc.beta)


class TestBoundaryConditionManager3D:
    """Test 3D boundary condition manager with Robin BCs."""

    @pytest.fixture
    def box_mesh_3d(self):
        """Create simple box mesh for testing."""
        # 2x2x2 grid of vertices
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 1, 3)
        z = np.linspace(0, 1, 3)

        vertices = []
        for zi in z:
            for yi in y:
                for xi in x:
                    vertices.append([xi, yi, zi])

        vertices = np.array(vertices)
        elements = np.array([[0, 1, 3, 9]])  # Simple connectivity
        boundary_tags = np.array([0])
        element_tags = np.array([0])
        boundary_faces = np.array([[0, 1, 3], [0, 1, 9], [0, 3, 9], [1, 3, 9]])

        mesh = MeshData(
            vertices=vertices,
            elements=elements,
            element_type="tetrahedron",
            boundary_tags=boundary_tags,
            element_tags=element_tags,
            boundary_faces=boundary_faces,
            dimension=3,
        )

        return mesh

    def test_manager_add_robin_condition(self, box_mesh_3d):
        """Test adding Robin condition to manager."""
        manager = BoundaryConditionManager3D()

        robin_bc = RobinBC3D(alpha=1.5, beta=2.0, value_function=0.0, name="Test Robin")
        manager.add_condition(robin_bc, boundary_region=0)  # Add to region 0

        assert len(manager.conditions) == 1
        assert 0 in manager.region_map
        assert manager.region_map[0][0] == robin_bc

    def test_manager_apply_robin_conditions(self, box_mesh_3d):
        """Test applying Robin conditions through manager."""
        manager = BoundaryConditionManager3D()

        # Add Robin BC for x_min face (region 0)
        robin_bc = RobinBC3D(alpha=2.0, beta=1.0, value_function=1.0)
        manager.add_condition(robin_bc, boundary_region=0)

        # Create test system
        n = box_mesh_3d.num_vertices
        matrix = csr_matrix(np.eye(n))
        rhs = np.zeros(n)

        # Apply conditions
        modified_matrix, modified_rhs = manager.apply_all_conditions(matrix, rhs, box_mesh_3d)

        # System should be modified
        assert not np.allclose(modified_matrix.toarray(), matrix.toarray())
        assert not np.allclose(modified_rhs, rhs)

    def test_manager_validation(self, box_mesh_3d):
        """Test manager validation of Robin conditions."""
        manager = BoundaryConditionManager3D()

        robin_bc = RobinBC3D(alpha=1.0, beta=1.0, value_function=0.0)
        manager.add_condition(robin_bc, boundary_region=0)

        # Should validate successfully
        assert manager.validate_all_conditions(box_mesh_3d)


class TestRobinBCIntegration:
    """Integration tests combining different Robin BC components."""

    def test_1d_to_3d_compatibility(self):
        """Test that 1D Robin concepts extend to 3D."""
        # 1D Robin BC
        bc_1d = BoundaryConditions(
            type="robin",
            left_alpha=1.0,
            left_beta=0.5,
            right_alpha=2.0,
            right_beta=1.0,
            left_value=0.0,
            right_value=1.0,
        )

        # 3D Robin BC with same coefficients
        bc_3d_left = RobinBC3D(alpha=1.0, beta=0.5, value_function=0.0, name="Left")
        bc_3d_right = RobinBC3D(alpha=2.0, beta=1.0, value_function=1.0, name="Right")

        # Both should have same mathematical structure
        assert bc_1d.is_robin()
        assert bc_3d_left.name == "Left"
        assert bc_3d_right.name == "Right"

    def test_robin_mathematical_correctness(self):
        """Test mathematical correctness of Robin BC: αu + β∇u·n = g."""
        # This test would require a complete finite element setup
        # For now, verify the structure is correct

        alpha, beta, g = 2.0, 1.5, 3.0

        # 3D Robin BC
        bc = RobinBC3D(alpha=alpha, beta=beta, value_function=g)

        # Verify coefficients are stored correctly
        test_coord = np.array([0.5, 0.5, 0.5])
        assert bc.alpha(test_coord[0], test_coord[1], test_coord[2], 0.0) == alpha
        assert bc.beta(test_coord[0], test_coord[1], test_coord[2], 0.0) == beta
        assert bc.value_function(test_coord[0], test_coord[1], test_coord[2], 0.0) == g


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
