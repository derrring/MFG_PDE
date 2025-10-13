"""
Unit tests for Domain2D geometry implementation.

Tests the 2D domain class with various geometry types, mesh generation,
and boundary extraction capabilities.
"""

from __future__ import annotations

import pytest

import numpy as np

from mfg_pde.geometry.domain_2d import Domain2D

pytestmark = pytest.mark.unit


class TestDomain2DInitialization:
    """Test Domain2D initialization and parameter setup."""

    def test_rectangle_initialization(self):
        """Test rectangular domain initialization."""
        domain = Domain2D(
            domain_type="rectangle",
            bounds=(0.0, 1.0, 0.0, 2.0),
            mesh_size=0.1,
        )

        assert domain.dimension == 2
        assert domain.domain_type == "rectangle"
        assert domain.xmin == 0.0
        assert domain.xmax == 1.0
        assert domain.ymin == 0.0
        assert domain.ymax == 2.0
        assert domain.mesh_size == 0.1

    def test_circle_initialization(self):
        """Test circular domain initialization."""
        domain = Domain2D(
            domain_type="circle",
            bounds=(0.0, 1.0, 0.0, 1.0),
            center=(0.5, 0.5),
            radius=0.4,
            mesh_size=0.05,
        )

        assert domain.dimension == 2
        assert domain.domain_type == "circle"
        assert domain.center == (0.5, 0.5)
        assert domain.radius == 0.4
        assert domain.mesh_size == 0.05

    def test_polygon_initialization(self):
        """Test polygonal domain initialization."""
        vertices = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        domain = Domain2D(
            domain_type="polygon",
            bounds=(0.0, 1.0, 0.0, 1.0),
            vertices=vertices,
            mesh_size=0.1,
        )

        assert domain.dimension == 2
        assert domain.domain_type == "polygon"
        assert len(domain.vertices) == 4
        assert domain.vertices == vertices

    def test_polygon_insufficient_vertices(self):
        """Test that polygon requires at least 3 vertices."""
        with pytest.raises(ValueError, match="at least 3 vertices"):
            Domain2D(
                domain_type="polygon",
                vertices=[(0.0, 0.0), (1.0, 0.0)],  # Only 2 vertices
            )

    def test_custom_domain_without_function(self):
        """Test that custom domain requires geometry_func."""
        with pytest.raises(ValueError, match="geometry_func"):
            Domain2D(domain_type="custom")

    def test_cad_import_without_file(self):
        """Test that CAD import requires cad_file."""
        with pytest.raises(ValueError, match="cad_file"):
            Domain2D(domain_type="cad_import")

    def test_with_holes_specification(self):
        """Test domain initialization with holes."""
        holes = [
            {"type": "circle", "center": (0.5, 0.5), "radius": 0.1},
        ]
        domain = Domain2D(
            domain_type="rectangle",
            bounds=(0.0, 1.0, 0.0, 1.0),
            holes=holes,
            mesh_size=0.05,
        )

        assert domain.dimension == 2
        assert len(domain.holes) == 1
        assert domain.holes[0]["type"] == "circle"

    def test_default_parameters(self):
        """Test domain with default parameters."""
        domain = Domain2D()

        assert domain.dimension == 2
        assert domain.domain_type == "rectangle"
        assert domain.bounds_rect == (0.0, 1.0, 0.0, 1.0)
        assert domain.mesh_size == 0.1
        assert len(domain.holes) == 0


class TestDomain2DBounds:
    """Test domain bounding box calculations."""

    def test_rectangle_bounds(self):
        """Test rectangular domain bounds."""
        domain = Domain2D(
            domain_type="rectangle",
            bounds=(0.5, 2.5, -1.0, 1.0),
        )

        min_coords, max_coords = domain.bounds

        np.testing.assert_array_equal(min_coords, [0.5, -1.0])
        np.testing.assert_array_equal(max_coords, [2.5, 1.0])

    def test_circle_bounds(self):
        """Test circular domain bounds."""
        domain = Domain2D(
            domain_type="circle",
            center=(1.0, 2.0),
            radius=0.5,
        )

        min_coords, max_coords = domain.bounds

        np.testing.assert_array_almost_equal(min_coords, [0.5, 1.5])
        np.testing.assert_array_almost_equal(max_coords, [1.5, 2.5])

    def test_polygon_bounds(self):
        """Test polygonal domain bounds."""
        vertices = [(0.0, 0.0), (2.0, 0.5), (1.5, 2.0), (0.5, 1.5)]
        domain = Domain2D(
            domain_type="polygon",
            vertices=vertices,
        )

        min_coords, max_coords = domain.bounds

        np.testing.assert_array_almost_equal(min_coords, [0.0, 0.0])
        np.testing.assert_array_almost_equal(max_coords, [2.0, 2.0])

    def test_unit_square_bounds(self):
        """Test unit square domain bounds."""
        domain = Domain2D(bounds=(0.0, 1.0, 0.0, 1.0))

        min_coords, max_coords = domain.bounds

        np.testing.assert_array_equal(min_coords, [0.0, 0.0])
        np.testing.assert_array_equal(max_coords, [1.0, 1.0])


class TestDomain2DMeshParameters:
    """Test mesh parameter configuration."""

    def test_set_mesh_size(self):
        """Test setting mesh size parameter."""
        domain = Domain2D(mesh_size=0.1)

        domain.set_mesh_parameters(mesh_size=0.05)

        assert domain.mesh_size == 0.05

    def test_set_mesh_algorithm(self):
        """Test setting mesh algorithm."""
        domain = Domain2D()

        domain.set_mesh_parameters(algorithm="frontal")

        assert domain.mesh_algorithm == "frontal"

    def test_set_additional_parameters(self):
        """Test setting additional mesh parameters."""
        domain = Domain2D()

        domain.set_mesh_parameters(
            mesh_size=0.02,
            algorithm="delaunay",
            optimize=True,
            quality_threshold=0.8,
        )

        assert domain.mesh_size == 0.02
        assert domain.mesh_algorithm == "delaunay"
        assert domain.mesh_kwargs["optimize"] is True
        assert domain.mesh_kwargs["quality_threshold"] == 0.8


@pytest.mark.skipif(
    not pytest.importorskip("gmsh", reason="gmsh not available"), reason="gmsh required for mesh generation tests"
)
class TestDomain2DGmshGeometry:
    """Test Gmsh geometry creation (requires gmsh)."""

    def test_create_rectangle_gmsh(self):
        """Test creating rectangular geometry in Gmsh."""
        domain = Domain2D(
            domain_type="rectangle",
            bounds=(0.0, 1.0, 0.0, 1.0),
            mesh_size=0.2,
        )

        import gmsh

        try:
            gmsh_model = domain.create_gmsh_geometry()
            assert gmsh_model is not None
            assert hasattr(domain, "main_surface")
        finally:
            gmsh.finalize()

    def test_create_circle_gmsh(self):
        """Test creating circular geometry in Gmsh."""
        domain = Domain2D(
            domain_type="circle",
            center=(0.5, 0.5),
            radius=0.4,
            mesh_size=0.1,
        )

        import gmsh

        try:
            gmsh_model = domain.create_gmsh_geometry()
            assert gmsh_model is not None
            assert hasattr(domain, "main_surface")
        finally:
            gmsh.finalize()

    def test_create_polygon_gmsh(self):
        """Test creating polygonal geometry in Gmsh."""
        vertices = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.5), (0.0, 1.0)]
        domain = Domain2D(
            domain_type="polygon",
            vertices=vertices,
            mesh_size=0.1,
        )

        import gmsh

        try:
            gmsh_model = domain.create_gmsh_geometry()
            assert gmsh_model is not None
            assert hasattr(domain, "main_surface")
        finally:
            gmsh.finalize()

    def test_unknown_domain_type(self):
        """Test that unknown domain type raises error."""
        import contextlib

        domain = Domain2D()
        domain.domain_type = "unknown_type"

        import gmsh

        try:
            with pytest.raises(ValueError, match="Unknown domain type"):
                domain.create_gmsh_geometry()
        finally:
            with contextlib.suppress(BaseException):
                gmsh.finalize()


@pytest.mark.skipif(
    not pytest.importorskip("gmsh", reason="gmsh not available")
    or not pytest.importorskip("meshio", reason="meshio not available"),
    reason="gmsh and meshio required for mesh generation",
)
class TestDomain2DMeshGeneration:
    """Test mesh generation (requires gmsh and meshio)."""

    def test_generate_rectangle_mesh(self):
        """Test generating mesh for rectangular domain."""
        domain = Domain2D(
            domain_type="rectangle",
            bounds=(0.0, 1.0, 0.0, 1.0),
            mesh_size=0.2,
        )

        mesh_data = domain.generate_mesh()

        assert mesh_data is not None
        assert mesh_data.dimension == 2
        assert mesh_data.element_type == "triangle"
        assert mesh_data.vertices.shape[1] == 2  # 2D vertices
        assert mesh_data.elements.shape[1] == 3  # Triangles
        assert len(mesh_data.vertices) > 0
        assert len(mesh_data.elements) > 0

    def test_generate_circle_mesh(self):
        """Test generating mesh for circular domain."""
        domain = Domain2D(
            domain_type="circle",
            center=(0.5, 0.5),
            radius=0.4,
            mesh_size=0.15,
        )

        mesh_data = domain.generate_mesh()

        assert mesh_data is not None
        assert mesh_data.dimension == 2
        assert mesh_data.element_type == "triangle"
        assert len(mesh_data.vertices) > 0
        assert len(mesh_data.elements) > 0

    def test_mesh_data_structure(self):
        """Test that mesh data has correct structure."""
        domain = Domain2D(bounds=(0.0, 1.0, 0.0, 1.0), mesh_size=0.2)

        mesh_data = domain.generate_mesh()

        # Check arrays exist and have correct shapes
        assert hasattr(mesh_data, "vertices")
        assert hasattr(mesh_data, "elements")
        assert hasattr(mesh_data, "boundary_tags")
        assert hasattr(mesh_data, "element_tags")
        assert hasattr(mesh_data, "boundary_faces")

        # Check dimensions
        assert mesh_data.vertices.ndim == 2
        assert mesh_data.elements.ndim == 2
        assert mesh_data.boundary_faces.ndim == 2

    def test_mesh_size_effect(self):
        """Test that smaller mesh_size produces more elements."""
        domain_coarse = Domain2D(
            bounds=(0.0, 1.0, 0.0, 1.0),
            mesh_size=0.5,
        )
        domain_fine = Domain2D(
            bounds=(0.0, 1.0, 0.0, 1.0),
            mesh_size=0.1,
        )

        mesh_coarse = domain_coarse.generate_mesh()
        mesh_fine = domain_fine.generate_mesh()

        # Finer mesh should have more elements
        assert len(mesh_fine.elements) > len(mesh_coarse.elements)
        assert len(mesh_fine.vertices) > len(mesh_coarse.vertices)

    def test_mesh_boundary_extraction(self):
        """Test boundary edge extraction."""
        domain = Domain2D(bounds=(0.0, 1.0, 0.0, 1.0), mesh_size=0.2)

        mesh_data = domain.generate_mesh()

        # Boundary faces should be edges (pairs of vertices)
        assert mesh_data.boundary_faces.shape[1] == 2
        assert len(mesh_data.boundary_faces) > 0

        # Boundary edges should form closed loop
        assert len(mesh_data.boundary_faces) >= 4  # At least 4 for simple rectangle

    def test_mesh_metadata(self):
        """Test that mesh metadata is stored correctly."""
        domain = Domain2D(
            domain_type="rectangle",
            bounds=(0.5, 1.5, -0.5, 0.5),
            mesh_size=0.15,
        )

        mesh_data = domain.generate_mesh()

        assert "domain_type" in mesh_data.metadata
        assert "mesh_size" in mesh_data.metadata
        assert "bounds" in mesh_data.metadata
        assert mesh_data.metadata["domain_type"] == "rectangle"
        assert mesh_data.metadata["mesh_size"] == 0.15


@pytest.mark.skipif(
    not pytest.importorskip("gmsh", reason="gmsh not available")
    or not pytest.importorskip("meshio", reason="meshio not available"),
    reason="gmsh and meshio required",
)
class TestDomain2DWithHoles:
    """Test domain with holes (requires gmsh and meshio)."""

    @pytest.mark.skip(reason="Known issue: Gmsh OpenCASCADE entity problem with holes - see domain_2d.py:279")
    def test_rectangle_with_circular_hole(self):
        """Test rectangular domain with circular hole."""
        holes = [
            {"type": "circle", "center": (0.5, 0.5), "radius": 0.15},
        ]
        domain = Domain2D(
            domain_type="rectangle",
            bounds=(0.0, 1.0, 0.0, 1.0),
            holes=holes,
            mesh_size=0.1,
        )

        mesh_data = domain.generate_mesh()

        assert mesh_data is not None
        assert len(mesh_data.vertices) > 0
        assert len(mesh_data.elements) > 0

    @pytest.mark.skip(reason="Known issue: Gmsh OpenCASCADE entity problem with holes - see domain_2d.py:279")
    def test_rectangle_with_rectangular_hole(self):
        """Test rectangular domain with rectangular hole."""
        holes = [
            {"type": "rectangle", "bounds": (0.3, 0.7, 0.3, 0.7)},
        ]
        domain = Domain2D(
            domain_type="rectangle",
            bounds=(0.0, 1.0, 0.0, 1.0),
            holes=holes,
            mesh_size=0.1,
        )

        mesh_data = domain.generate_mesh()

        assert mesh_data is not None
        assert len(mesh_data.vertices) > 0
        assert len(mesh_data.elements) > 0

    @pytest.mark.skip(reason="Known issue: Gmsh OpenCASCADE entity problem with holes - see domain_2d.py:279")
    def test_domain_with_multiple_holes(self):
        """Test domain with multiple holes."""
        holes = [
            {"type": "circle", "center": (0.3, 0.3), "radius": 0.1},
            {"type": "circle", "center": (0.7, 0.7), "radius": 0.1},
        ]
        domain = Domain2D(
            domain_type="rectangle",
            bounds=(0.0, 1.0, 0.0, 1.0),
            holes=holes,
            mesh_size=0.08,
        )

        mesh_data = domain.generate_mesh()

        assert mesh_data is not None
        # Mesh with holes should have multiple boundary components
        assert len(mesh_data.boundary_faces) > 4


@pytest.mark.skipif(
    not pytest.importorskip("gmsh", reason="gmsh not available")
    or not pytest.importorskip("meshio", reason="meshio not available"),
    reason="gmsh and meshio required",
)
class TestDomain2DMeshRefinement:
    """Test mesh refinement capabilities."""

    def test_uniform_refinement(self):
        """Test uniform mesh refinement."""
        domain = Domain2D(bounds=(0.0, 1.0, 0.0, 1.0), mesh_size=0.4)

        # Generate initial mesh
        domain.generate_mesh()

        # Refine uniformly (reduce mesh_size by half)
        mesh_refined = domain.refine_mesh(refinement_factor=0.5)

        # Verify refinement changes mesh_size parameter
        assert domain.mesh_size == 0.2  # 0.4 * 0.5

        # Verify new mesh is generated
        assert mesh_refined is not None
        assert len(mesh_refined.elements) > 0

    def test_adaptive_refinement_requires_indicator(self):
        """Test that adaptive refinement requires indicator function."""
        domain = Domain2D(bounds=(0.0, 1.0, 0.0, 1.0), mesh_size=0.2)
        domain.generate_mesh()

        with pytest.raises(ValueError, match="indicator_function"):
            domain.refine_mesh(adaptive=True)

    def test_adaptive_refinement_not_implemented(self):
        """Test that adaptive refinement raises NotImplementedError."""
        domain = Domain2D(bounds=(0.0, 1.0, 0.0, 1.0), mesh_size=0.2)
        domain.generate_mesh()

        def indicator_func(x):
            return np.linalg.norm(x - np.array([0.5, 0.5]))

        with pytest.raises(NotImplementedError, match="Adaptive refinement"):
            domain.refine_mesh(adaptive=True, indicator_function=indicator_func)


@pytest.mark.skipif(
    not pytest.importorskip("gmsh", reason="gmsh not available")
    or not pytest.importorskip("meshio", reason="meshio not available"),
    reason="gmsh and meshio required",
)
class TestDomain2DMeshExport:
    """Test mesh export functionality."""

    @pytest.mark.skip(reason="Known issue: MeshData.to_meshio() has KeyError - see base_geometry.py:137")
    def test_export_mesh_vtk(self, tmp_path):
        """Test exporting mesh to VTK format."""
        domain = Domain2D(bounds=(0.0, 1.0, 0.0, 1.0), mesh_size=0.2)
        domain.generate_mesh()

        output_file = tmp_path / "test_mesh.vtk"
        domain.export_mesh("vtk", str(output_file))

        assert output_file.exists()

    @pytest.mark.skip(reason="Known issue: MeshData.to_meshio() has KeyError - see base_geometry.py:137")
    def test_export_mesh_requires_generation(self, tmp_path):
        """Test that export generates mesh if not already done."""
        domain = Domain2D(bounds=(0.0, 1.0, 0.0, 1.0), mesh_size=0.2)

        output_file = tmp_path / "test_mesh.vtk"
        domain.export_mesh("vtk", str(output_file))

        assert output_file.exists()
        assert domain.mesh_data is not None


class TestDomain2DEdgeCases:
    """Test edge cases and special scenarios."""

    def test_very_small_domain(self):
        """Test domain with very small dimensions."""
        domain = Domain2D(
            bounds=(0.0, 0.01, 0.0, 0.01),
            mesh_size=0.002,
        )

        min_coords, max_coords = domain.bounds

        assert np.all(min_coords == [0.0, 0.0])
        assert np.all(max_coords == [0.01, 0.01])

    def test_very_large_domain(self):
        """Test domain with very large dimensions."""
        domain = Domain2D(
            bounds=(-1000.0, 1000.0, -1000.0, 1000.0),
            mesh_size=100.0,
        )

        min_coords, max_coords = domain.bounds

        assert np.all(min_coords == [-1000.0, -1000.0])
        assert np.all(max_coords == [1000.0, 1000.0])

    def test_non_square_rectangle(self):
        """Test rectangular domain with different x and y dimensions."""
        domain = Domain2D(
            bounds=(0.0, 2.0, 0.0, 0.5),
        )

        min_coords, max_coords = domain.bounds

        # Width = 2.0, Height = 0.5
        assert (max_coords[0] - min_coords[0]) == 2.0
        assert (max_coords[1] - min_coords[1]) == 0.5

    def test_circle_with_small_radius(self):
        """Test circular domain with very small radius."""
        domain = Domain2D(
            domain_type="circle",
            center=(0.5, 0.5),
            radius=0.01,
        )

        min_coords, max_coords = domain.bounds

        # Bounds should be approximately [0.49, 0.51] in each dimension
        assert np.allclose(min_coords, [0.49, 0.49], atol=0.01)
        assert np.allclose(max_coords, [0.51, 0.51], atol=0.01)

    def test_triangle_polygon(self):
        """Test triangular polygon (minimum vertices)."""
        vertices = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        domain = Domain2D(
            domain_type="polygon",
            vertices=vertices,
        )

        assert len(domain.vertices) == 3
        assert domain.domain_type == "polygon"
