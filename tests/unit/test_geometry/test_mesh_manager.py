#!/usr/bin/env python3
"""
Unit tests for mfg_pde/geometry/mesh_manager.py

Tests mesh management and pipeline orchestration including:
- MeshPipeline initialization and configuration
- Geometry creation from dictionary config
- Pipeline execution and stage tracking
- Mesh generation, quality analysis, and export
- MeshManager high-level interface
- Batch mesh generation and quality comparison
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

import numpy as np

from mfg_pde.geometry.base_geometry import MeshData
from mfg_pde.geometry.domain_2d import Domain2D
from mfg_pde.geometry.mesh_manager import MeshManager, MeshPipeline

# ===================================================================
# Test MeshPipeline Initialization
# ===================================================================


@pytest.mark.unit
def test_mesh_pipeline_initialization_with_geometry():
    """Test MeshPipeline initialization with geometry object."""
    geometry = Domain2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = MeshPipeline(geometry, output_dir=tmpdir, verbose=True)

        assert pipeline.geometry is geometry
        assert pipeline.output_dir == Path(tmpdir)
        assert pipeline.verbose is True
        assert pipeline.mesh_data is None


@pytest.mark.unit
def test_mesh_pipeline_initialization_with_config():
    """Test MeshPipeline initialization with configuration dictionary."""
    config = {
        "type": "rectangle",
        "dimension": 2,
        "bounds": (0.0, 2.0, 0.0, 1.0),
        "mesh_size": 0.2,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = MeshPipeline(config, output_dir=tmpdir)

        assert isinstance(pipeline.geometry, Domain2D)
        assert pipeline.geometry.domain_type == "rectangle"


@pytest.mark.unit
def test_mesh_pipeline_default_output_dir():
    """Test MeshPipeline creates default output directory."""
    geometry = Domain2D(domain_type="rectangle")

    pipeline = MeshPipeline(geometry, output_dir=None)

    assert pipeline.output_dir == Path("./mesh_output")
    # Clean up created directory
    pipeline.output_dir.rmdir()


@pytest.mark.unit
def test_mesh_pipeline_stages_tracking():
    """Test pipeline stages tracking initialization."""
    geometry = Domain2D(domain_type="rectangle")

    pipeline = MeshPipeline(geometry)

    assert "geometry_created" in pipeline.stages_completed
    assert "mesh_generated" in pipeline.stages_completed
    assert "quality_analyzed" in pipeline.stages_completed
    assert "visualization_ready" in pipeline.stages_completed
    assert all(not v for v in pipeline.stages_completed.values())


# ===================================================================
# Test Geometry Creation from Config
# ===================================================================


@pytest.mark.unit
def test_create_geometry_from_config_rectangle():
    """Test creating rectangle geometry from config."""
    config = {
        "type": "rectangle",
        "dimension": 2,
        "bounds": (0.0, 1.0, 0.0, 1.0),
        "mesh_size": 0.1,
    }

    geometry = Domain2D(domain_type="rectangle", bounds=config["bounds"], mesh_size=config["mesh_size"])
    pipeline = MeshPipeline(geometry)

    assert pipeline.geometry.domain_type == "rectangle"


@pytest.mark.unit
def test_create_geometry_from_config_circle():
    """Test creating circle geometry from config."""
    config = {
        "type": "circle",
        "dimension": 2,
        "bounds": (0.5, 0.5, 0.3),  # center_x, center_y, radius
        "mesh_size": 0.1,
    }

    geometry = Domain2D(domain_type="circle", bounds=config["bounds"], mesh_size=config["mesh_size"])
    pipeline = MeshPipeline(geometry)

    assert pipeline.geometry.domain_type == "circle"


@pytest.mark.unit
def test_create_geometry_from_config_with_holes():
    """Test creating geometry with holes from config."""
    config = {
        "type": "rectangle",
        "dimension": 2,
        "bounds": (0.0, 1.0, 0.0, 1.0),
        "holes": [{"type": "circle", "center": (0.5, 0.5), "radius": 0.2}],
        "mesh_size": 0.1,
    }

    geometry = Domain2D(
        domain_type=config["type"], bounds=config["bounds"], holes=config["holes"], mesh_size=config["mesh_size"]
    )
    pipeline = MeshPipeline(geometry)

    assert len(pipeline.geometry.holes) == 1


@pytest.mark.unit
def test_create_geometry_unsupported_dimension_raises():
    """Test that unsupported dimension raises NotImplementedError."""
    config = {"dimension": 3, "type": "box"}

    geometry = Domain2D(domain_type="rectangle")  # Use 2D for test
    pipeline = MeshPipeline(geometry)

    # Direct test of the method with invalid config
    with pytest.raises(NotImplementedError, match="Dimension 3 not yet supported"):
        pipeline._create_geometry_from_config(config)


# ===================================================================
# Test Pipeline Execution with Mocking
# ===================================================================


def create_mock_mesh_data():
    """Create mock MeshData for testing."""
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    elements = np.array([[0, 1, 2], [1, 3, 2]])
    return MeshData(
        vertices=vertices,
        elements=elements,
        element_type="triangle",
        boundary_tags=np.array([1, 1, 1, 1]),
        element_tags=np.zeros(2, dtype=int),
        boundary_faces=np.array([[0, 1], [1, 3], [3, 2], [2, 0]]),
        dimension=2,
    )


@pytest.mark.unit
def test_execute_pipeline_generate_stage():
    """Test pipeline execution with generate stage only."""
    geometry = Mock(spec=Domain2D)
    geometry.generate_mesh = Mock(return_value=create_mock_mesh_data())
    geometry.compute_mesh_quality = Mock(return_value={"aspect_ratio": 1.0})

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = MeshPipeline(geometry, output_dir=tmpdir)

        mesh_data = pipeline.execute_pipeline(stages=["generate"])

        assert pipeline.stages_completed["mesh_generated"] is True
        assert pipeline.stages_completed["quality_analyzed"] is False
        assert mesh_data is not None
        geometry.generate_mesh.assert_called_once()


@pytest.mark.unit
def test_execute_pipeline_multiple_stages():
    """Test pipeline execution with multiple stages."""
    geometry = Mock(spec=Domain2D)
    geometry.generate_mesh = Mock(return_value=create_mock_mesh_data())
    geometry.compute_mesh_quality = Mock(return_value={"aspect_ratio": 1.0, "min_angle": 30.0})

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = MeshPipeline(geometry, output_dir=tmpdir)

        mesh_data = pipeline.execute_pipeline(stages=["generate", "analyze"])

        assert pipeline.stages_completed["mesh_generated"] is True
        assert pipeline.stages_completed["quality_analyzed"] is True
        assert mesh_data is not None


@pytest.mark.unit
def test_execute_pipeline_default_stages():
    """Test pipeline execution with default stages."""
    mock_mesh = create_mock_mesh_data()
    mock_mesh.to_pyvista = Mock(return_value=MagicMock())

    geometry = Mock(spec=Domain2D)
    geometry.generate_mesh = Mock(return_value=mock_mesh)
    geometry.compute_mesh_quality = Mock(return_value={})

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = MeshPipeline(geometry, output_dir=tmpdir, verbose=False)

        pipeline.execute_pipeline()

        # Default stages are ["generate", "analyze", "visualize"]
        assert pipeline.stages_completed["mesh_generated"] is True
        assert pipeline.stages_completed["quality_analyzed"] is True
        assert pipeline.stages_completed["visualization_ready"] is True


@pytest.mark.unit
def test_execute_pipeline_export_stage():
    """Test pipeline execution with export stage."""
    geometry = Mock(spec=Domain2D)
    geometry.generate_mesh = Mock(return_value=create_mock_mesh_data())
    geometry.export_mesh = Mock()

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = MeshPipeline(geometry, output_dir=tmpdir)

        pipeline.execute_pipeline(stages=["generate", "export"], export_formats=["msh", "vtk"])

        # Check export was called for each format
        assert geometry.export_mesh.call_count == 2


# ===================================================================
# Test Individual Pipeline Stages
# ===================================================================


@pytest.mark.unit
def test_stage_generate_mesh():
    """Test mesh generation stage."""
    mock_mesh = create_mock_mesh_data()
    geometry = Mock(spec=Domain2D)
    geometry.generate_mesh = Mock(return_value=mock_mesh)

    pipeline = MeshPipeline(geometry)

    result = pipeline._stage_generate_mesh()

    assert result is mock_mesh
    assert result.num_vertices == 4
    assert result.num_elements == 2


@pytest.mark.unit
def test_stage_generate_mesh_failure():
    """Test mesh generation stage handles failures."""
    geometry = Mock(spec=Domain2D)
    geometry.generate_mesh = Mock(side_effect=RuntimeError("Mesh generation failed"))

    pipeline = MeshPipeline(geometry)

    with pytest.raises(RuntimeError, match="Mesh generation failed"):
        pipeline._stage_generate_mesh()


@pytest.mark.unit
def test_stage_analyze_quality():
    """Test quality analysis stage."""
    geometry = Mock(spec=Domain2D)
    geometry.compute_mesh_quality = Mock(return_value={"aspect_ratio": 1.2, "min_angle": 35.0})

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = MeshPipeline(geometry, output_dir=tmpdir, verbose=True)
        pipeline.mesh_data = create_mock_mesh_data()

        pipeline._stage_analyze_quality()

        # Check that quality report was saved
        report_file = pipeline.output_dir / "quality_report.txt"
        assert report_file.exists()


@pytest.mark.unit
def test_stage_prepare_visualization():
    """Test visualization preparation stage."""
    mock_mesh = create_mock_mesh_data()
    mock_pyvista_mesh = MagicMock()
    mock_mesh.to_pyvista = Mock(return_value=mock_pyvista_mesh)

    geometry = Mock(spec=Domain2D)
    pipeline = MeshPipeline(geometry)
    pipeline.mesh_data = mock_mesh

    pipeline._stage_prepare_visualization()

    mock_mesh.to_pyvista.assert_called_once()


@pytest.mark.unit
def test_stage_prepare_visualization_no_mesh_raises():
    """Test visualization preparation fails without mesh data."""
    geometry = Mock(spec=Domain2D)
    pipeline = MeshPipeline(geometry)
    pipeline.mesh_data = None

    with pytest.raises(RuntimeError, match="Mesh data is None"):
        pipeline._stage_prepare_visualization()


@pytest.mark.unit
def test_stage_export_mesh():
    """Test mesh export stage."""
    geometry = Mock(spec=Domain2D)
    geometry.export_mesh = Mock()

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = MeshPipeline(geometry, output_dir=tmpdir)
        pipeline.mesh_data = create_mock_mesh_data()

        pipeline._stage_export_mesh(["msh", "vtk", "xdmf"])

        assert geometry.export_mesh.call_count == 3


# ===================================================================
# Test Quality Report Generation
# ===================================================================


@pytest.mark.unit
def test_save_quality_report():
    """Test saving quality report to file."""
    geometry = Mock(spec=Domain2D)
    quality_metrics = {"aspect_ratio": 1.15, "min_angle": 32.5, "max_angle": 87.3}

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = MeshPipeline(geometry, output_dir=tmpdir)
        pipeline.mesh_data = create_mock_mesh_data()

        pipeline._save_quality_report(quality_metrics)

        report_file = pipeline.output_dir / "quality_report.txt"
        assert report_file.exists()

        # Check report contents
        content = report_file.read_text()
        assert "Mesh Quality Analysis Report" in content
        assert "aspect_ratio: 1.150000" in content
        assert "Vertices: 4" in content
        assert "Elements: 2" in content


@pytest.mark.unit
def test_save_quality_report_no_mesh_raises():
    """Test saving quality report without mesh data raises error."""
    geometry = Mock(spec=Domain2D)

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = MeshPipeline(geometry, output_dir=tmpdir)
        pipeline.mesh_data = None

        with pytest.raises(RuntimeError, match="Mesh data is None"):
            pipeline._save_quality_report({"metric": 1.0})


# ===================================================================
# Test Pipeline Summary
# ===================================================================


@pytest.mark.unit
def test_get_pipeline_summary_no_mesh():
    """Test getting pipeline summary before mesh generation."""
    geometry = Domain2D(domain_type="rectangle")

    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = MeshPipeline(geometry, output_dir=tmpdir)

        summary = pipeline.get_pipeline_summary()

        assert summary["geometry_type"] == "Domain2D"
        assert summary["mesh_info"]["vertices"] == 0
        assert summary["mesh_info"]["elements"] == 0
        assert "stages_completed" in summary


@pytest.mark.unit
def test_get_pipeline_summary_with_mesh():
    """Test getting pipeline summary after mesh generation."""
    # Create a real Domain2D instance instead of Mock
    geometry = Domain2D(domain_type="rectangle", bounds=(0.0, 1.0, 0.0, 1.0))

    # Mock the generate_mesh method
    with (
        patch.object(geometry, "generate_mesh", return_value=create_mock_mesh_data()),
        tempfile.TemporaryDirectory() as tmpdir,
    ):
        pipeline = MeshPipeline(geometry, output_dir=tmpdir)
        pipeline.execute_pipeline(stages=["generate"])

        summary = pipeline.get_pipeline_summary()

        assert summary["geometry_type"] == "Domain2D"
        assert summary["mesh_info"]["vertices"] == 4
        assert summary["mesh_info"]["elements"] == 2
        assert summary["mesh_info"]["element_type"] == "triangle"
        assert summary["mesh_info"]["dimension"] == 2
        assert summary["stages_completed"]["mesh_generated"] is True


# ===================================================================
# Test MeshManager Initialization
# ===================================================================


@pytest.mark.unit
def test_mesh_manager_initialization_empty():
    """Test MeshManager initialization without config."""
    manager = MeshManager()

    assert manager.config == {}
    assert len(manager.geometries) == 0
    assert len(manager.pipelines) == 0


@pytest.mark.unit
def test_mesh_manager_initialization_with_config():
    """Test MeshManager initialization with config."""
    config = {"default_mesh_size": 0.1, "verbose": True}
    manager = MeshManager(config)

    assert manager.config == config


# ===================================================================
# Test MeshManager Geometry Creation
# ===================================================================


@pytest.mark.unit
def test_mesh_manager_create_geometry():
    """Test creating geometry with MeshManager."""
    manager = MeshManager()

    config = {
        "type": "rectangle",
        "dimension": 2,
        "bounds": (0.0, 1.0, 0.0, 1.0),
        "mesh_size": 0.1,
    }

    geometry = manager.create_geometry("test_geom", config)

    assert "test_geom" in manager.geometries
    assert isinstance(geometry, Domain2D)
    assert geometry.domain_type == "rectangle"


@pytest.mark.unit
def test_mesh_manager_create_multiple_geometries():
    """Test creating multiple geometries."""
    manager = MeshManager()

    config1 = {"type": "rectangle", "dimension": 2, "bounds": (0.0, 1.0, 0.0, 1.0)}
    config2 = {"type": "circle", "dimension": 2, "bounds": (0.5, 0.5, 0.3)}

    manager.create_geometry("rect", config1)
    manager.create_geometry("circle", config2)

    assert len(manager.geometries) == 2
    assert "rect" in manager.geometries
    assert "circle" in manager.geometries


@pytest.mark.unit
def test_mesh_manager_create_geometry_unsupported_dimension():
    """Test creating geometry with unsupported dimension raises error."""
    manager = MeshManager()

    config = {"type": "box", "dimension": 3}

    with pytest.raises(NotImplementedError, match="3D geometries not yet implemented"):
        manager.create_geometry("box3d", config)


# ===================================================================
# Test MeshManager Pipeline Creation
# ===================================================================


@pytest.mark.unit
def test_mesh_manager_create_pipeline():
    """Test creating pipeline with MeshManager."""
    manager = MeshManager()

    # First create geometry
    geom_config = {"type": "rectangle", "dimension": 2}
    manager.create_geometry("test_geom", geom_config)

    # Then create pipeline
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = manager.create_pipeline("test_pipe", "test_geom", output_dir=tmpdir)

        assert "test_pipe" in manager.pipelines
        assert isinstance(pipeline, MeshPipeline)
        assert pipeline.geometry is manager.geometries["test_geom"]


@pytest.mark.unit
def test_mesh_manager_create_pipeline_nonexistent_geometry():
    """Test creating pipeline with nonexistent geometry raises error."""
    manager = MeshManager()

    with pytest.raises(ValueError, match="Geometry 'nonexistent' not found"):
        manager.create_pipeline("pipe", "nonexistent")


# ===================================================================
# Test Batch Mesh Generation
# ===================================================================


@pytest.mark.unit
def test_batch_generate_meshes():
    """Test batch mesh generation."""
    manager = MeshManager()

    # Create geometries with mocked generate_mesh
    for i in range(3):
        geometry = Mock(spec=Domain2D)
        geometry.generate_mesh = Mock(return_value=create_mock_mesh_data())
        geometry.compute_mesh_quality = Mock(return_value={})
        manager.geometries[f"geom{i}"] = geometry

    # Create pipelines
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            pipeline = MeshPipeline(manager.geometries[f"geom{i}"], output_dir=tmpdir)
            manager.pipelines[f"pipe{i}"] = pipeline

        # Batch generate
        results = manager.batch_generate_meshes(["pipe0", "pipe1", "pipe2"], stages=["generate"])

        assert len(results) == 3
        assert "pipe0" in results
        assert "pipe1" in results
        assert "pipe2" in results


@pytest.mark.unit
def test_batch_generate_meshes_with_missing_pipeline():
    """Test batch generation handles missing pipelines gracefully."""
    manager = MeshManager()

    # Create only one pipeline
    geometry = Mock(spec=Domain2D)
    geometry.generate_mesh = Mock(return_value=create_mock_mesh_data())
    geometry.compute_mesh_quality = Mock(return_value={})
    manager.geometries["geom1"] = geometry

    with tempfile.TemporaryDirectory() as tmpdir:
        manager.pipelines["pipe1"] = MeshPipeline(geometry, output_dir=tmpdir)

        # Request batch with nonexistent pipeline
        results = manager.batch_generate_meshes(["pipe1", "pipe_nonexistent"], stages=["generate"])

        assert len(results) == 1  # Only pipe1 should succeed
        assert "pipe1" in results


@pytest.mark.unit
def test_batch_generate_meshes_handles_errors():
    """Test batch generation continues despite individual failures."""
    manager = MeshManager()

    # Create pipelines with one that fails
    geom1 = Mock(spec=Domain2D)
    geom1.generate_mesh = Mock(return_value=create_mock_mesh_data())
    geom1.compute_mesh_quality = Mock(return_value={})

    geom2 = Mock(spec=Domain2D)
    geom2.generate_mesh = Mock(side_effect=RuntimeError("Generation failed"))

    manager.geometries["geom1"] = geom1
    manager.geometries["geom2"] = geom2

    with tempfile.TemporaryDirectory() as tmpdir:
        manager.pipelines["pipe1"] = MeshPipeline(geom1, output_dir=tmpdir)
        manager.pipelines["pipe2"] = MeshPipeline(geom2, output_dir=tmpdir)

        results = manager.batch_generate_meshes(["pipe1", "pipe2"], stages=["generate"])

        # Only successful pipeline should be in results
        assert "pipe1" in results
        assert "pipe2" not in results


# ===================================================================
# Test Quality Comparison
# ===================================================================


@pytest.mark.unit
def test_compare_mesh_quality():
    """Test comparing quality metrics across multiple meshes."""
    manager = MeshManager()

    # Create pipelines with quality metrics
    for i in range(2):
        geometry = Mock(spec=Domain2D)
        geometry.generate_mesh = Mock(return_value=create_mock_mesh_data())
        geometry.compute_mesh_quality = Mock(return_value={"aspect_ratio": 1.0 + i * 0.1, "min_angle": 30.0 + i * 5.0})

        manager.geometries[f"geom{i}"] = geometry

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = MeshPipeline(geometry, output_dir=tmpdir)
            pipeline.execute_pipeline(stages=["generate"])
            manager.pipelines[f"pipe{i}"] = pipeline

    comparison = manager.compare_mesh_quality(["pipe0", "pipe1"])

    assert len(comparison) == 2
    assert "pipe0" in comparison
    assert "pipe1" in comparison
    assert "aspect_ratio" in comparison["pipe0"]
    assert "min_angle" in comparison["pipe0"]


@pytest.mark.unit
def test_compare_mesh_quality_no_mesh_data():
    """Test quality comparison skips pipelines without mesh data."""
    manager = MeshManager()

    geometry = Mock(spec=Domain2D)
    pipeline = MeshPipeline(geometry)
    pipeline.mesh_data = None
    manager.pipelines["pipe1"] = pipeline

    comparison = manager.compare_mesh_quality(["pipe1"])

    # Should not include pipe1 since it has no mesh data
    assert len(comparison) == 0


@pytest.mark.unit
def test_compare_mesh_quality_nonexistent_pipeline():
    """Test quality comparison ignores nonexistent pipelines."""
    manager = MeshManager()

    comparison = manager.compare_mesh_quality(["nonexistent"])

    assert len(comparison) == 0
