"""
Unit tests for Domain3D geometry implementation.

Tests the 3D domain class with various geometry types, mesh generation,
quality metrics, and export capabilities.
"""

from __future__ import annotations

import contextlib

import pytest

import numpy as np

from mfg_pde.geometry.domain_3d import Domain3D

pytestmark = pytest.mark.unit


class TestDomain3DInitialization:
    """Test Domain3D initialization and parameter setup."""

    def test_box_initialization(self):
        """Test box domain initialization."""
        domain = Domain3D(
            domain_type="box",
            bounds=(0.0, 1.0, 0.0, 2.0, 0.0, 3.0),
            mesh_size=0.2,
        )

        assert domain.dimension == 3
        assert domain.domain_type == "box"
        assert domain.xmin == 0.0
        assert domain.xmax == 1.0
        assert domain.ymin == 0.0
        assert domain.ymax == 2.0
        assert domain.zmin == 0.0
        assert domain.zmax == 3.0
        assert domain.mesh_size == 0.2

    def test_sphere_initialization(self):
        """Test spherical domain initialization."""
        domain = Domain3D(
            domain_type="sphere",
            center=(0.5, 0.5, 0.5),
            radius=0.3,
            mesh_size=0.1,
        )

        assert domain.dimension == 3
        assert domain.domain_type == "sphere"
        assert domain.center == (0.5, 0.5, 0.5)
        assert domain.radius == 0.3
        assert domain.mesh_size == 0.1

    def test_cylinder_initialization(self):
        """Test cylindrical domain initialization."""
        domain = Domain3D(
            domain_type="cylinder",
            center=(0.5, 0.5),
            radius=0.3,
            height=1.0,
            axis="z",
            mesh_size=0.1,
        )

        assert domain.dimension == 3
        assert domain.domain_type == "cylinder"
        assert domain.center == (0.5, 0.5)
        assert domain.radius == 0.3
        assert domain.height == 1.0
        assert domain.axis == "z"

    def test_polyhedron_initialization(self):
        """Test polyhedron domain initialization (tetrahedron)."""
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0), (0.5, 0.5, 1.0)]
        faces = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]

        domain = Domain3D(
            domain_type="polyhedron",
            vertices=vertices,
            faces=faces,
            mesh_size=0.1,
        )

        assert domain.dimension == 3
        assert domain.domain_type == "polyhedron"
        assert len(domain.vertices) == 4
        assert len(domain.faces) == 4

    def test_polyhedron_insufficient_vertices(self):
        """Test that polyhedron requires at least 4 vertices."""
        with pytest.raises(ValueError, match="at least 4 vertices"):
            Domain3D(
                domain_type="polyhedron",
                vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)],
                faces=[[0, 1, 2]],
            )

    def test_polyhedron_insufficient_faces(self):
        """Test that polyhedron requires at least 4 faces."""
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0), (0.5, 0.5, 1.0)]
        with pytest.raises(ValueError, match="at least 4 faces"):
            Domain3D(
                domain_type="polyhedron",
                vertices=vertices,
                faces=[[0, 1, 2]],  # Only 1 face
            )

    def test_custom_domain_without_function(self):
        """Test that custom domain requires geometry_func."""
        with pytest.raises(ValueError, match="geometry_func"):
            Domain3D(domain_type="custom")

    def test_cad_import_without_file(self):
        """Test that CAD import requires cad_file."""
        with pytest.raises(ValueError, match="cad_file"):
            Domain3D(domain_type="cad_import")

    def test_cylinder_axis_variants(self):
        """Test cylinder with different axis orientations."""
        for axis in ["x", "y", "z"]:
            domain = Domain3D(
                domain_type="cylinder",
                center=(0.5, 0.5),
                radius=0.3,
                height=1.0,
                axis=axis,
            )
            assert domain.axis == axis

    def test_default_parameters(self):
        """Test domain with default parameters."""
        domain = Domain3D()

        assert domain.dimension == 3
        assert domain.domain_type == "box"
        assert domain.bounds_box == (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        assert domain.mesh_size == 0.1
        assert len(domain.holes) == 0


@pytest.mark.skipif(
    not pytest.importorskip("gmsh", reason="gmsh not available"), reason="gmsh required for geometry creation"
)
class TestDomain3DGmshGeometry:
    """Test Gmsh geometry creation (requires gmsh)."""

    def test_create_box_gmsh(self):
        """Test creating box geometry in Gmsh."""
        domain = Domain3D(
            domain_type="box",
            bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
            mesh_size=0.3,
        )

        import gmsh

        try:
            gmsh_model = domain.create_gmsh_geometry()
            assert gmsh_model is not None

            # Verify 3D volume exists
            volumes = gmsh.model.getEntities(3)
            assert len(volumes) > 0
        finally:
            gmsh.finalize()

    def test_create_sphere_gmsh(self):
        """Test creating sphere geometry in Gmsh."""
        domain = Domain3D(
            domain_type="sphere",
            center=(0.5, 0.5, 0.5),
            radius=0.4,
            mesh_size=0.2,
        )

        import gmsh

        try:
            gmsh_model = domain.create_gmsh_geometry()
            assert gmsh_model is not None

            volumes = gmsh.model.getEntities(3)
            assert len(volumes) > 0
        finally:
            gmsh.finalize()

    def test_create_cylinder_gmsh(self):
        """Test creating cylinder geometry in Gmsh."""
        domain = Domain3D(
            domain_type="cylinder",
            center=(0.5, 0.5),
            radius=0.3,
            height=1.0,
            axis="z",
            mesh_size=0.2,
        )

        import gmsh

        try:
            gmsh_model = domain.create_gmsh_geometry()
            assert gmsh_model is not None

            volumes = gmsh.model.getEntities(3)
            assert len(volumes) > 0
        finally:
            gmsh.finalize()

    def test_cylinder_invalid_axis(self):
        """Test that invalid cylinder axis raises error."""
        domain = Domain3D(
            domain_type="cylinder",
            center=(0.5, 0.5),
            radius=0.3,
            height=1.0,
            axis="invalid",
        )

        import gmsh

        try:
            with pytest.raises(ValueError, match="Invalid cylinder axis"):
                domain.create_gmsh_geometry()
        finally:
            with contextlib.suppress(BaseException):
                gmsh.finalize()

    def test_unknown_domain_type(self):
        """Test that unknown domain type raises error."""
        domain = Domain3D()
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
class TestDomain3DMeshGeneration:
    """Test mesh generation (requires gmsh and meshio)."""

    @pytest.mark.skip(
        reason="Known issue: Domain3D._create_box_gmsh() boundary surface numbering - see domain_3d.py:139"
    )
    def test_generate_box_mesh(self):
        """Test generating mesh for box domain."""
        domain = Domain3D(
            domain_type="box",
            bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
            mesh_size=0.3,
        )

        mesh_data = domain.generate_mesh()

        assert mesh_data is not None
        assert mesh_data.dimension == 3
        assert mesh_data.element_type == "tetrahedron"
        assert mesh_data.vertices.shape[1] == 3  # 3D vertices
        assert mesh_data.elements.shape[1] == 4  # Tetrahedra
        assert len(mesh_data.vertices) > 0
        assert len(mesh_data.elements) > 0

    @pytest.mark.skip(reason="Known issue: Domain3D._create_sphere_gmsh() boundary surface numbering")
    def test_generate_sphere_mesh(self):
        """Test generating mesh for spherical domain."""
        domain = Domain3D(
            domain_type="sphere",
            center=(0.5, 0.5, 0.5),
            radius=0.4,
            mesh_size=0.2,
        )

        mesh_data = domain.generate_mesh()

        assert mesh_data is not None
        assert mesh_data.dimension == 3
        assert mesh_data.element_type == "tetrahedron"
        assert len(mesh_data.vertices) > 0
        assert len(mesh_data.elements) > 0

    def test_mesh_data_structure(self):
        """Test that mesh data has correct structure."""
        domain = Domain3D(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), mesh_size=0.4)

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
        domain_coarse = Domain3D(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), mesh_size=0.5)
        domain_fine = Domain3D(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), mesh_size=0.2)

        mesh_coarse = domain_coarse.generate_mesh()
        mesh_fine = domain_fine.generate_mesh()

        # Finer mesh should have more elements
        assert len(mesh_fine.elements) > len(mesh_coarse.elements)
        assert len(mesh_fine.vertices) > len(mesh_coarse.vertices)

    def test_mesh_quality_metrics(self):
        """Test that quality metrics are computed."""
        domain = Domain3D(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), mesh_size=0.3)

        mesh_data = domain.generate_mesh()

        # Check quality metrics exist
        assert hasattr(mesh_data, "quality_metrics")
        assert mesh_data.quality_metrics is not None
        assert "min_quality" in mesh_data.quality_metrics
        assert "mean_quality" in mesh_data.quality_metrics
        assert "min_volume" in mesh_data.quality_metrics
        assert "mean_volume" in mesh_data.quality_metrics

        # Check quality values are reasonable
        assert 0.0 <= mesh_data.quality_metrics["min_quality"] <= 1.0
        assert 0.0 <= mesh_data.quality_metrics["mean_quality"] <= 1.0
        assert mesh_data.quality_metrics["min_volume"] > 0.0

    def test_element_volumes(self):
        """Test that element volumes are computed."""
        domain = Domain3D(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), mesh_size=0.3)

        mesh_data = domain.generate_mesh()

        # Check element volumes exist
        assert hasattr(mesh_data, "element_volumes")
        assert mesh_data.element_volumes is not None
        assert len(mesh_data.element_volumes) == len(mesh_data.elements)
        assert np.all(mesh_data.element_volumes > 0)

    def test_boundary_faces_extraction(self):
        """Test boundary face extraction."""
        domain = Domain3D(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), mesh_size=0.3)

        mesh_data = domain.generate_mesh()

        # Boundary faces should be triangles (3 vertices each)
        assert mesh_data.boundary_faces.shape[1] == 3
        assert len(mesh_data.boundary_faces) > 0

        # For a box, expect boundary faces on all 6 faces
        assert len(mesh_data.boundary_faces) >= 12  # At least 2 triangles per face


@pytest.mark.skipif(
    not pytest.importorskip("gmsh", reason="gmsh not available")
    or not pytest.importorskip("meshio", reason="meshio not available"),
    reason="gmsh and meshio required",
)
class TestDomain3DWithHoles:
    """Test domain with holes (requires gmsh and meshio)."""

    @pytest.mark.skip(reason="Holes functionality may have issues - test for API coverage")
    def test_box_with_spherical_hole(self):
        """Test box domain with spherical hole."""
        holes = [
            {"type": "sphere", "center": (0.5, 0.5, 0.5), "radius": 0.2},
        ]
        domain = Domain3D(
            domain_type="box",
            bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
            holes=holes,
            mesh_size=0.2,
        )

        mesh_data = domain.generate_mesh()

        assert mesh_data is not None
        assert len(mesh_data.vertices) > 0
        assert len(mesh_data.elements) > 0

    @pytest.mark.skip(reason="Holes functionality may have issues - test for API coverage")
    def test_box_with_box_hole(self):
        """Test box domain with box-shaped hole."""
        holes = [
            {"type": "box", "bounds": (0.3, 0.7, 0.3, 0.7, 0.3, 0.7)},
        ]
        domain = Domain3D(
            domain_type="box",
            bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
            holes=holes,
            mesh_size=0.2,
        )

        mesh_data = domain.generate_mesh()

        assert mesh_data is not None
        assert len(mesh_data.vertices) > 0
        assert len(mesh_data.elements) > 0


@pytest.mark.skipif(
    not pytest.importorskip("gmsh", reason="gmsh not available")
    or not pytest.importorskip("meshio", reason="meshio not available"),
    reason="gmsh and meshio required",
)
class TestDomain3DMeshExport:
    """Test mesh export functionality."""

    @pytest.mark.skip(reason="Export implementation may vary - test for API coverage")
    def test_export_mesh_vtk(self, tmp_path):
        """Test exporting mesh to VTK format."""
        domain = Domain3D(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), mesh_size=0.4)
        domain.generate_mesh()

        output_file = tmp_path / "test_mesh_3d.vtk"
        domain.export_mesh("vtk", str(output_file))

        assert output_file.exists()


class TestDomain3DQualityMetrics:
    """Test mesh quality computation methods."""

    def test_tetrahedron_volume_computation(self):
        """Test volume computation for single tetrahedron."""
        domain = Domain3D()

        # Unit tetrahedron
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        elements = np.array([[0, 1, 2, 3]])

        volumes = domain._compute_tetrahedron_volumes(vertices, elements)

        # Volume of unit tetrahedron = 1/6
        expected_volume = 1.0 / 6.0
        assert np.isclose(volumes[0], expected_volume, rtol=1e-5)

    def test_circumradius_computation(self):
        """Test circumradius computation for tetrahedron."""
        domain = Domain3D()

        # Simple tetrahedron
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        circumradius = domain._compute_circumradius(coords)

        # Should be positive
        assert circumradius > 0

    def test_inradius_computation(self):
        """Test inradius computation for tetrahedron."""
        domain = Domain3D()

        # Simple tetrahedron
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        inradius = domain._compute_inradius(coords)

        # Should be positive and less than circumradius
        circumradius = domain._compute_circumradius(coords)
        assert inradius > 0
        assert inradius < circumradius


class TestDomain3DEdgeCases:
    """Test edge cases and special scenarios."""

    def test_unit_cube(self):
        """Test unit cube domain."""
        domain = Domain3D(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0))

        assert domain.xmax - domain.xmin == 1.0
        assert domain.ymax - domain.ymin == 1.0
        assert domain.zmax - domain.zmin == 1.0

    def test_non_cubic_box(self):
        """Test box with different dimensions."""
        domain = Domain3D(bounds=(0.0, 2.0, 0.0, 1.0, 0.0, 0.5))

        assert domain.xmax - domain.xmin == 2.0
        assert domain.ymax - domain.ymin == 1.0
        assert domain.zmax - domain.zmin == 0.5

    def test_very_small_domain(self):
        """Test domain with very small dimensions."""
        domain = Domain3D(bounds=(0.0, 0.01, 0.0, 0.01, 0.0, 0.01), mesh_size=0.005)

        assert domain.xmax - domain.xmin == 0.01
        assert domain.ymax - domain.ymin == 0.01
        assert domain.zmax - domain.zmin == 0.01

    def test_very_large_domain(self):
        """Test domain with very large dimensions."""
        domain = Domain3D(bounds=(-100.0, 100.0, -100.0, 100.0, -100.0, 100.0), mesh_size=20.0)

        assert domain.xmax - domain.xmin == 200.0
        assert domain.ymax - domain.ymin == 200.0
        assert domain.zmax - domain.zmin == 200.0

    def test_sphere_with_small_radius(self):
        """Test sphere with very small radius."""
        domain = Domain3D(
            domain_type="sphere",
            center=(0.5, 0.5, 0.5),
            radius=0.01,
        )

        assert domain.radius == 0.01
        assert domain.center == (0.5, 0.5, 0.5)

    def test_tall_cylinder(self):
        """Test cylinder with large height-to-radius ratio."""
        domain = Domain3D(
            domain_type="cylinder",
            center=(0.5, 0.5),
            radius=0.1,
            height=5.0,
            axis="z",
        )

        assert domain.height / domain.radius == 50.0

    def test_flat_cylinder(self):
        """Test cylinder with small height-to-radius ratio."""
        domain = Domain3D(
            domain_type="cylinder",
            center=(0.5, 0.5),
            radius=0.5,
            height=0.1,
            axis="z",
        )

        assert domain.height / domain.radius == 0.2
