#!/usr/bin/env python3
"""
Unit tests for Simple Grid Geometry

Tests the SimpleGrid2D and SimpleGrid3D classes for mesh generation
without external dependencies.
"""

import tempfile
from pathlib import Path

import pytest

import numpy as np

from mfg_pde.geometry.simple_grid import SimpleGrid2D, SimpleGrid3D

# ============================================================================
# Test: SimpleGrid2D - Initialization
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_simple_grid_2d_initialization_default():
    """Test SimpleGrid2D initialization with defaults."""
    grid = SimpleGrid2D()

    assert grid.dimension == 2
    assert grid.xmin == 0.0
    assert grid.xmax == 1.0
    assert grid.ymin == 0.0
    assert grid.ymax == 1.0
    assert grid.nx == 32
    assert grid.ny == 32


@pytest.mark.unit
@pytest.mark.fast
def test_simple_grid_2d_initialization_custom():
    """Test SimpleGrid2D initialization with custom parameters."""
    bounds = (-1.0, 2.0, -0.5, 1.5)
    resolution = (16, 24)

    grid = SimpleGrid2D(bounds=bounds, resolution=resolution)

    assert grid.xmin == -1.0
    assert grid.xmax == 2.0
    assert grid.ymin == -0.5
    assert grid.ymax == 1.5
    assert grid.nx == 16
    assert grid.ny == 24


@pytest.mark.unit
@pytest.mark.fast
def test_simple_grid_2d_bounds_property():
    """Test bounds property returns correct min/max coordinates."""
    grid = SimpleGrid2D(bounds=(0.0, 2.0, 1.0, 3.0), resolution=(10, 10))

    min_coords, max_coords = grid.bounds

    assert np.allclose(min_coords, [0.0, 1.0])
    assert np.allclose(max_coords, [2.0, 3.0])


# ============================================================================
# Test: SimpleGrid2D - Mesh Generation
# ============================================================================


@pytest.mark.unit
def test_simple_grid_2d_generate_mesh():
    """Test mesh generation produces valid MeshData."""
    grid = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(4, 4))

    mesh = grid.generate_mesh()

    # Check mesh structure
    assert mesh.vertices is not None
    assert mesh.elements is not None
    assert mesh.boundary_faces is not None
    assert mesh.boundary_tags is not None

    # Check dimensions
    assert mesh.vertices.shape[1] == 2  # 2D vertices
    assert mesh.elements.shape[1] == 3  # Triangular elements

    # Check vertex count: (nx+1) * (ny+1)
    expected_vertices = (4 + 1) * (4 + 1)
    assert mesh.vertices.shape[0] == expected_vertices


@pytest.mark.unit
def test_simple_grid_2d_mesh_vertices_correct():
    """Test mesh vertices are correctly positioned."""
    grid = SimpleGrid2D(bounds=(0.0, 2.0, 0.0, 1.0), resolution=(2, 2))

    mesh = grid.generate_mesh()

    # Check corner vertices exist
    corners = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [2.0, 1.0]])

    for corner in corners:
        distances = np.linalg.norm(mesh.vertices - corner, axis=1)
        assert np.min(distances) < 1e-10, f"Corner {corner} not found in mesh"


@pytest.mark.unit
def test_simple_grid_2d_element_count():
    """Test correct number of triangular elements."""
    grid = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(3, 3))

    mesh = grid.generate_mesh()

    # Each grid cell has 2 triangles
    expected_elements = 3 * 3 * 2
    assert mesh.elements.shape[0] == expected_elements


@pytest.mark.unit
def test_simple_grid_2d_boundary_faces():
    """Test boundary faces are correctly identified."""
    grid = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(2, 2))

    mesh = grid.generate_mesh()

    # Should have boundary faces
    assert len(mesh.boundary_faces) > 0
    assert len(mesh.boundary_tags) == len(mesh.boundary_faces)

    # All boundary faces should have 2 vertices (edges in 2D)
    assert all(len(face) == 2 for face in mesh.boundary_faces)


@pytest.mark.unit
def test_simple_grid_2d_different_resolutions():
    """Test mesh generation works with different resolutions."""
    resolutions = [(5, 5), (10, 5), (5, 10), (8, 12)]

    for nx, ny in resolutions:
        grid = SimpleGrid2D(resolution=(nx, ny))
        mesh = grid.generate_mesh()

        expected_vertices = (nx + 1) * (ny + 1)
        expected_elements = nx * ny * 2

        assert mesh.vertices.shape[0] == expected_vertices
        assert mesh.elements.shape[0] == expected_elements


# ============================================================================
# Test: SimpleGrid2D - Export
# ============================================================================


@pytest.mark.unit
def test_simple_grid_2d_export_txt():
    """Test exporting mesh to text format."""
    grid = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(3, 3))

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_mesh"
        grid.export_mesh("txt", str(filepath))

        # Check files were created
        assert (Path(tmpdir) / "test_mesh_vertices.txt").exists()
        assert (Path(tmpdir) / "test_mesh_elements.txt").exists()

        # Check vertices can be loaded
        vertices = np.loadtxt(Path(tmpdir) / "test_mesh_vertices.txt")
        assert vertices.shape[1] == 2


@pytest.mark.unit
def test_simple_grid_2d_export_numpy():
    """Test exporting mesh to numpy format."""
    grid = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(2, 2))

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_mesh"
        grid.export_mesh("numpy", str(filepath))

        assert (Path(tmpdir) / "test_mesh_vertices.txt").exists()
        assert (Path(tmpdir) / "test_mesh_elements.txt").exists()


@pytest.mark.unit
@pytest.mark.fast
def test_simple_grid_2d_export_unsupported_format():
    """Test exporting to unsupported format raises error."""
    grid = SimpleGrid2D()

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_mesh"

        with pytest.raises(NotImplementedError, match="not supported"):
            grid.export_mesh("vtk", str(filepath))


# ============================================================================
# Test: SimpleGrid2D - Gmsh Methods
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_simple_grid_2d_create_gmsh_not_implemented():
    """Test that create_gmsh_geometry raises NotImplementedError."""
    grid = SimpleGrid2D()

    with pytest.raises(NotImplementedError, match="does not use Gmsh"):
        grid.create_gmsh_geometry()


@pytest.mark.unit
@pytest.mark.fast
def test_simple_grid_2d_set_mesh_parameters_noop():
    """Test set_mesh_parameters is a no-op (doesn't crash)."""
    grid = SimpleGrid2D()

    # Should not raise error
    grid.set_mesh_parameters(some_param=10, other_param=20)


# ============================================================================
# Test: SimpleGrid2D - Quality Metrics
# ============================================================================


@pytest.mark.unit
def test_simple_grid_2d_quality_metrics():
    """Test quality metrics computation."""
    grid = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(4, 4))

    mesh = grid.generate_mesh()
    metrics = grid._compute_quality_metrics(mesh)

    assert isinstance(metrics, dict)
    # Should have some quality metrics
    assert len(metrics) > 0


# ============================================================================
# Test: SimpleGrid3D - Initialization
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_simple_grid_3d_initialization_default():
    """Test SimpleGrid3D initialization with defaults."""
    grid = SimpleGrid3D()

    assert grid.dimension == 3
    assert grid.xmin == 0.0
    assert grid.xmax == 1.0
    assert grid.ymin == 0.0
    assert grid.ymax == 1.0
    assert grid.zmin == 0.0
    assert grid.zmax == 1.0
    assert grid.nx == 16
    assert grid.ny == 16
    assert grid.nz == 16


@pytest.mark.unit
@pytest.mark.fast
def test_simple_grid_3d_initialization_custom():
    """Test SimpleGrid3D initialization with custom parameters."""
    bounds = (-1.0, 2.0, -0.5, 1.5, 0.0, 3.0)
    resolution = (8, 10, 12)

    grid = SimpleGrid3D(bounds=bounds, resolution=resolution)

    assert grid.xmin == -1.0
    assert grid.xmax == 2.0
    assert grid.ymin == -0.5
    assert grid.ymax == 1.5
    assert grid.zmin == 0.0
    assert grid.zmax == 3.0
    assert grid.nx == 8
    assert grid.ny == 10
    assert grid.nz == 12


@pytest.mark.unit
@pytest.mark.fast
def test_simple_grid_3d_bounds_property():
    """Test 3D bounds property returns correct min/max coordinates."""
    grid = SimpleGrid3D(bounds=(0.0, 2.0, 1.0, 3.0, 0.5, 2.5), resolution=(4, 4, 4))

    min_coords, max_coords = grid.bounds

    assert np.allclose(min_coords, [0.0, 1.0, 0.5])
    assert np.allclose(max_coords, [2.0, 3.0, 2.5])


# ============================================================================
# Test: SimpleGrid3D - Mesh Generation
# ============================================================================


@pytest.mark.unit
def test_simple_grid_3d_generate_mesh():
    """Test 3D mesh generation produces valid MeshData."""
    grid = SimpleGrid3D(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(3, 3, 3))

    mesh = grid.generate_mesh()

    # Check mesh structure
    assert mesh.vertices is not None
    assert mesh.elements is not None
    assert mesh.boundary_faces is not None

    # Check dimensions
    assert mesh.vertices.shape[1] == 3  # 3D vertices
    assert mesh.elements.shape[1] == 4  # Tetrahedral elements

    # Check vertex count: (nx+1) * (ny+1) * (nz+1)
    expected_vertices = (3 + 1) * (3 + 1) * (3 + 1)
    assert mesh.vertices.shape[0] == expected_vertices


@pytest.mark.unit
def test_simple_grid_3d_mesh_vertices_correct():
    """Test 3D mesh vertices are correctly positioned."""
    grid = SimpleGrid3D(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(2, 2, 2))

    mesh = grid.generate_mesh()

    # Check corner vertices exist
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    for corner in corners:
        distances = np.linalg.norm(mesh.vertices - corner, axis=1)
        assert np.min(distances) < 1e-10, f"Corner {corner} not found in mesh"


@pytest.mark.unit
def test_simple_grid_3d_element_count():
    """Test correct number of tetrahedral elements."""
    grid = SimpleGrid3D(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(2, 2, 2))

    mesh = grid.generate_mesh()

    # Each grid cell has 6 tetrahedra (implementation uses 6-tet decomposition)
    expected_elements = 2 * 2 * 2 * 6
    assert mesh.elements.shape[0] == expected_elements


@pytest.mark.unit
def test_simple_grid_3d_boundary_faces():
    """Test 3D boundary faces structure exists."""
    grid = SimpleGrid3D(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(2, 2, 2))

    mesh = grid.generate_mesh()

    # Should have boundary_faces attribute
    assert hasattr(mesh, "boundary_faces")
    assert mesh.boundary_faces is not None

    # If boundary faces are populated, they should have 3 vertices (triangles in 3D)
    if len(mesh.boundary_faces) > 0:
        assert all(len(face) == 3 for face in mesh.boundary_faces)


# ============================================================================
# Test: SimpleGrid3D - Export
# ============================================================================


@pytest.mark.unit
def test_simple_grid_3d_export_txt():
    """Test exporting 3D mesh to text format."""
    grid = SimpleGrid3D(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(2, 2, 2))

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_mesh_3d"
        grid.export_mesh("txt", str(filepath))

        # Check files were created
        assert (Path(tmpdir) / "test_mesh_3d_vertices.txt").exists()
        assert (Path(tmpdir) / "test_mesh_3d_elements.txt").exists()

        # Check vertices can be loaded and are 3D
        vertices = np.loadtxt(Path(tmpdir) / "test_mesh_3d_vertices.txt")
        assert vertices.shape[1] == 3


@pytest.mark.unit
@pytest.mark.fast
def test_simple_grid_3d_create_gmsh_not_implemented():
    """Test that 3D create_gmsh_geometry raises NotImplementedError."""
    grid = SimpleGrid3D()

    with pytest.raises(NotImplementedError, match="does not use Gmsh"):
        grid.create_gmsh_geometry()


# ============================================================================
# Test: Edge Cases and Validation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_simple_grid_2d_minimal_resolution():
    """Test grid with minimal resolution (1x1)."""
    grid = SimpleGrid2D(resolution=(1, 1))

    mesh = grid.generate_mesh()

    # Should have 4 vertices (2x2 grid)
    assert mesh.vertices.shape[0] == 4
    # Should have 2 triangles
    assert mesh.elements.shape[0] == 2


@pytest.mark.unit
@pytest.mark.fast
def test_simple_grid_3d_minimal_resolution():
    """Test 3D grid with minimal resolution (1x1x1)."""
    grid = SimpleGrid3D(resolution=(1, 1, 1))

    mesh = grid.generate_mesh()

    # Should have 8 vertices (2x2x2 grid)
    assert mesh.vertices.shape[0] == 8
    # Should have 6 tetrahedra (1 cell * 6 tets/cell)
    assert mesh.elements.shape[0] == 6


@pytest.mark.unit
def test_simple_grid_2d_non_square_domain():
    """Test grid on non-square domain."""
    grid = SimpleGrid2D(bounds=(0.0, 3.0, 0.0, 1.0), resolution=(9, 3))

    mesh = grid.generate_mesh()

    # Check domain dimensions are respected
    assert mesh.vertices[:, 0].min() == pytest.approx(0.0)
    assert mesh.vertices[:, 0].max() == pytest.approx(3.0)
    assert mesh.vertices[:, 1].min() == pytest.approx(0.0)
    assert mesh.vertices[:, 1].max() == pytest.approx(1.0)


@pytest.mark.unit
def test_simple_grid_3d_non_cubic_domain():
    """Test 3D grid on non-cubic domain."""
    grid = SimpleGrid3D(bounds=(0.0, 2.0, 0.0, 1.0, 0.0, 0.5), resolution=(4, 2, 1))

    mesh = grid.generate_mesh()

    # Check domain dimensions are respected
    assert mesh.vertices[:, 0].min() == pytest.approx(0.0)
    assert mesh.vertices[:, 0].max() == pytest.approx(2.0)
    assert mesh.vertices[:, 1].min() == pytest.approx(0.0)
    assert mesh.vertices[:, 1].max() == pytest.approx(1.0)
    assert mesh.vertices[:, 2].min() == pytest.approx(0.0)
    assert mesh.vertices[:, 2].max() == pytest.approx(0.5)


# ============================================================================
# Test: Element Connectivity
# ============================================================================


@pytest.mark.unit
def test_simple_grid_2d_element_indices_valid():
    """Test all element indices are valid vertex references."""
    grid = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(3, 3))

    mesh = grid.generate_mesh()

    # All element indices should be within vertex range
    max_index = mesh.elements.max()
    num_vertices = mesh.vertices.shape[0]

    assert max_index < num_vertices, "Element references non-existent vertex"


@pytest.mark.unit
def test_simple_grid_3d_element_indices_valid():
    """Test all 3D element indices are valid vertex references."""
    grid = SimpleGrid3D(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(2, 2, 2))

    mesh = grid.generate_mesh()

    # All element indices should be within vertex range
    max_index = mesh.elements.max()
    num_vertices = mesh.vertices.shape[0]

    assert max_index < num_vertices, "Element references non-existent vertex"


# ============================================================================
# Test: Mesh Regeneration
# ============================================================================


@pytest.mark.unit
def test_simple_grid_2d_mesh_regeneration():
    """Test mesh can be regenerated with consistent results."""
    grid = SimpleGrid2D(bounds=(0.0, 1.0, 0.0, 1.0), resolution=(3, 3))

    mesh1 = grid.generate_mesh()
    mesh2 = grid.generate_mesh()

    # Should produce identical meshes
    assert np.allclose(mesh1.vertices, mesh2.vertices)
    assert np.array_equal(mesh1.elements, mesh2.elements)


@pytest.mark.unit
def test_simple_grid_3d_mesh_regeneration():
    """Test 3D mesh can be regenerated with consistent results."""
    grid = SimpleGrid3D(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(2, 2, 2))

    mesh1 = grid.generate_mesh()
    mesh2 = grid.generate_mesh()

    # Should produce identical meshes
    assert np.allclose(mesh1.vertices, mesh2.vertices)
    assert np.array_equal(mesh1.elements, mesh2.elements)
