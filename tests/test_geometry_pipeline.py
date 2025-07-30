"""
Basic tests for the geometry pipeline implementation.

These tests verify that the core mesh generation pipeline works correctly
without requiring external dependencies like Gmsh.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add MFG_PDE to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mfg_pde.geometry import BaseGeometry, MeshData, Domain2D, BoundaryManager


class TestMeshData:
    """Test the MeshData container class."""
    
    def test_mesh_data_creation(self):
        """Test basic MeshData creation and validation."""
        vertices = np.array([[0, 0], [1, 0], [0, 1]])
        elements = np.array([[0, 1, 2]])
        
        mesh_data = MeshData(
            vertices=vertices,
            elements=elements,
            element_type="triangle",
            boundary_tags=np.array([1, 1, 1]),
            element_tags=np.array([1]),
            boundary_faces=np.array([[0, 1], [1, 2], [2, 0]]),
            dimension=2
        )
        
        assert mesh_data.num_vertices == 3
        assert mesh_data.num_elements == 1
        assert mesh_data.dimension == 2
        assert mesh_data.element_type == "triangle"
    
    def test_mesh_data_bounds(self):
        """Test bounding box computation."""
        vertices = np.array([[0, 0], [2, 0], [1, 3]])
        elements = np.array([[0, 1, 2]])
        
        mesh_data = MeshData(
            vertices=vertices,
            elements=elements,
            element_type="triangle",
            boundary_tags=np.array([1, 1, 1]),
            element_tags=np.array([1]),
            boundary_faces=np.array([[0, 1], [1, 2], [2, 0]]),
            dimension=2
        )
        
        min_coords, max_coords = mesh_data.bounds
        np.testing.assert_array_equal(min_coords, [0, 0])
        np.testing.assert_array_equal(max_coords, [2, 3])
    
    def test_triangle_area_computation(self):
        """Test triangle area computation."""
        # Simple right triangle with base=1, height=1, area=0.5
        vertices = np.array([[0, 0], [1, 0], [0, 1]])
        elements = np.array([[0, 1, 2]])
        
        mesh_data = MeshData(
            vertices=vertices,
            elements=elements,
            element_type="triangle",
            boundary_tags=np.array([1, 1, 1]),
            element_tags=np.array([1]),
            boundary_faces=np.array([[0, 1], [1, 2], [2, 0]]),
            dimension=2
        )
        
        areas = mesh_data.compute_element_volumes()
        assert len(areas) == 1
        assert abs(areas[0] - 0.5) < 1e-10
    
    def test_invalid_dimension(self):
        """Test validation of invalid dimensions."""
        vertices = np.array([[0, 0], [1, 0], [0, 1]])
        elements = np.array([[0, 1, 2]])
        
        with pytest.raises(ValueError, match="Dimension must be 2 or 3"):
            MeshData(
                vertices=vertices,
                elements=elements,
                element_type="triangle",
                boundary_tags=np.array([1, 1, 1]),
                element_tags=np.array([1]),
                boundary_faces=np.array([[0, 1], [1, 2], [2, 0]]),
                dimension=4  # Invalid dimension
            )


class TestDomain2D:
    """Test the Domain2D geometry class."""
    
    def test_rectangle_domain_creation(self):
        """Test rectangular domain creation."""
        domain = Domain2D(
            domain_type="rectangle",
            bounds=(0.0, 2.0, 0.0, 1.0),
            mesh_size=0.5
        )
        
        assert domain.domain_type == "rectangle"
        assert domain.dimension == 2
        assert domain.xmin == 0.0
        assert domain.xmax == 2.0
        assert domain.ymin == 0.0
        assert domain.ymax == 1.0
        assert domain.mesh_size == 0.5
    
    def test_circle_domain_creation(self):
        """Test circular domain creation."""
        domain = Domain2D(
            domain_type="circle",
            center=(0.5, 0.5),
            radius=0.4,
            mesh_size=0.1
        )
        
        assert domain.domain_type == "circle"
        assert domain.center == (0.5, 0.5)
        assert domain.radius == 0.4
    
    def test_polygon_domain_creation(self):
        """Test polygonal domain creation."""
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        
        domain = Domain2D(
            domain_type="polygon",
            vertices=vertices,
            mesh_size=0.2
        )
        
        assert domain.domain_type == "polygon"
        assert domain.vertices == vertices
    
    def test_polygon_validation(self):
        """Test polygon validation with insufficient vertices."""
        with pytest.raises(ValueError, match="at least 3 vertices"):
            Domain2D(
                domain_type="polygon",
                vertices=[(0, 0), (1, 0)],  # Only 2 vertices
                mesh_size=0.2
            )
    
    def test_rectangle_bounds(self):
        """Test bounds computation for rectangular domain."""
        domain = Domain2D(
            domain_type="rectangle",
            bounds=(1.0, 3.0, 2.0, 4.0),
            mesh_size=0.1
        )
        
        min_coords, max_coords = domain.bounds
        np.testing.assert_array_equal(min_coords, [1.0, 2.0])
        np.testing.assert_array_equal(max_coords, [3.0, 4.0])
    
    def test_circle_bounds(self):
        """Test bounds computation for circular domain."""
        domain = Domain2D(
            domain_type="circle",
            center=(1.0, 2.0),
            radius=0.5,
            mesh_size=0.1
        )
        
        min_coords, max_coords = domain.bounds
        np.testing.assert_array_equal(min_coords, [0.5, 1.5])
        np.testing.assert_array_equal(max_coords, [1.5, 2.5])
    
    def test_polygon_bounds(self):
        """Test bounds computation for polygonal domain."""
        vertices = [(0, 1), (2, 0), (3, 3), (1, 2)]
        
        domain = Domain2D(
            domain_type="polygon",
            vertices=vertices,
            mesh_size=0.1
        )
        
        min_coords, max_coords = domain.bounds
        np.testing.assert_array_equal(min_coords, [0, 0])
        np.testing.assert_array_equal(max_coords, [3, 3])


class TestBoundaryManager:
    """Test the BoundaryManager class."""
    
    def create_simple_mesh_data(self):
        """Create simple mesh data for testing."""
        vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        elements = np.array([[0, 1, 2], [0, 2, 3]])
        boundary_tags = np.array([1, 2, 3, 4])  # Different regions
        
        return MeshData(
            vertices=vertices,
            elements=elements,
            element_type="triangle",
            boundary_tags=boundary_tags,
            element_tags=np.array([1, 1]),
            boundary_faces=np.array([[0, 1], [1, 2], [2, 3], [3, 0]]),
            dimension=2
        )
    
    def test_boundary_manager_creation(self):
        """Test BoundaryManager creation and boundary extraction."""
        mesh_data = self.create_simple_mesh_data()
        boundary_manager = BoundaryManager(mesh_data)
        
        assert boundary_manager.mesh_data is mesh_data
        assert len(boundary_manager.boundary_nodes) == 4  # 4 boundary regions
    
    def test_add_boundary_condition(self):
        """Test adding boundary conditions."""
        mesh_data = self.create_simple_mesh_data()
        boundary_manager = BoundaryManager(mesh_data)
        
        # Add Dirichlet boundary condition
        bc = boundary_manager.add_boundary_condition(
            region_id=1,
            bc_type="dirichlet",
            value=1.0,
            description="Bottom boundary"
        )
        
        assert bc.region_id == 1
        assert bc.bc_type == "dirichlet"
        assert bc.value == 1.0
        assert bc.description == "Bottom boundary"
        assert 1 in boundary_manager.boundary_conditions
    
    def test_boundary_condition_evaluation(self):
        """Test boundary condition evaluation."""
        mesh_data = self.create_simple_mesh_data()
        boundary_manager = BoundaryManager(mesh_data)
        
        # Add constant Dirichlet BC
        boundary_manager.add_boundary_condition(
            region_id=1,
            bc_type="dirichlet",
            value=2.5
        )
        
        # Add function Dirichlet BC
        boundary_manager.add_boundary_condition(
            region_id=2,
            bc_type="dirichlet",
            value=lambda coords: coords[:, 0] + coords[:, 1]  # x + y
        )
        
        # Evaluate constant BC
        values1 = boundary_manager.evaluate_boundary_condition(region_id=1)
        assert np.all(values1 == 2.5)
        
        # Evaluate function BC (this test assumes node 1 is in region 2)
        values2 = boundary_manager.evaluate_boundary_condition(region_id=2)
        assert len(values2) > 0
    
    def test_summary_generation(self):
        """Test boundary condition summary generation."""
        mesh_data = self.create_simple_mesh_data()
        boundary_manager = BoundaryManager(mesh_data)
        
        boundary_manager.add_boundary_condition(
            region_id=1,
            bc_type="dirichlet",
            value=0.0,
            description="Zero Dirichlet"
        )
        
        boundary_manager.add_boundary_condition(
            region_id=2,
            bc_type="neumann",
            gradient_value=1.0,
            description="Unit flux"
        )
        
        summary = boundary_manager.get_summary()
        
        assert summary["num_regions"] == 2
        assert 1 in summary["regions"]
        assert 2 in summary["regions"]
        assert summary["regions"][1]["type"] == "dirichlet"
        assert summary["regions"][2]["type"] == "neumann"


def test_geometry_package_import():
    """Test that the geometry package imports correctly."""
    from mfg_pde.geometry import BaseGeometry, MeshData, Domain2D, BoundaryManager
    
    # Test that classes are available
    assert BaseGeometry is not None
    assert MeshData is not None
    assert Domain2D is not None
    assert BoundaryManager is not None


def test_main_package_geometry_import():
    """Test that geometry system is available from main package."""
    import mfg_pde
    
    # Check that the flag is set correctly
    if hasattr(mfg_pde, 'GEOMETRY_SYSTEM_AVAILABLE'):
        # If geometry system is available, test imports
        if mfg_pde.GEOMETRY_SYSTEM_AVAILABLE:
            assert hasattr(mfg_pde, 'Domain2D')
            assert hasattr(mfg_pde, 'MeshData')
        else:
            # If not available, that's okay too (missing optional dependencies)
            pass
    

if __name__ == "__main__":
    pytest.main([__file__])