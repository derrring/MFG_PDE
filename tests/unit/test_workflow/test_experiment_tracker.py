"""
Unit tests for Experiment Tracker.

Tests the experiment lifecycle management system including experiment creation,
execution tracking, result storage, and comparative analysis.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from mfg_pde.workflow.experiment_tracker import (
    Experiment,
    ExperimentMetadata,
    ExperimentResult,
    ExperimentStatus,
    ExperimentTracker,
)

# ============================================================================
# Test: ExperimentStatus Enum
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_experiment_status_values():
    """Test ExperimentStatus enum values."""
    assert ExperimentStatus.CREATED.value == "created"
    assert ExperimentStatus.RUNNING.value == "running"
    assert ExperimentStatus.COMPLETED.value == "completed"
    assert ExperimentStatus.FAILED.value == "failed"
    assert ExperimentStatus.CANCELLED.value == "cancelled"


@pytest.mark.unit
@pytest.mark.fast
def test_experiment_status_comparison():
    """Test ExperimentStatus can be compared."""
    assert ExperimentStatus.CREATED == ExperimentStatus.CREATED
    assert ExperimentStatus.RUNNING != ExperimentStatus.COMPLETED


# ============================================================================
# Test: ExperimentMetadata
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_experiment_metadata_creation():
    """Test ExperimentMetadata initialization."""
    metadata = ExperimentMetadata(
        id="test_id",
        name="test_experiment",
        description="Test description",
    )

    assert metadata.id == "test_id"
    assert metadata.name == "test_experiment"
    assert metadata.description == "Test description"
    assert metadata.status == ExperimentStatus.CREATED
    assert isinstance(metadata.created_time, datetime)


@pytest.mark.unit
@pytest.mark.fast
def test_experiment_metadata_with_tags():
    """Test ExperimentMetadata with tags."""
    metadata = ExperimentMetadata(
        id="test_id",
        name="test",
        description="desc",
        tags=["ml", "research"],
    )

    assert metadata.tags == ["ml", "research"]


@pytest.mark.unit
@pytest.mark.fast
def test_experiment_metadata_to_dict():
    """Test ExperimentMetadata serialization to dict."""
    metadata = ExperimentMetadata(
        id="test_id",
        name="test",
        description="desc",
    )

    data = metadata.to_dict()

    assert data["id"] == "test_id"
    assert data["name"] == "test"
    assert data["status"] == "created"
    assert "created_time" in data


# ============================================================================
# Test: ExperimentResult
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_experiment_result_creation():
    """Test ExperimentResult initialization."""
    result = ExperimentResult(
        experiment_id="exp_123",
        name="accuracy",
        value=0.95,
    )

    assert result.experiment_id == "exp_123"
    assert result.name == "accuracy"
    assert result.value == 0.95
    assert isinstance(result.timestamp, datetime)


@pytest.mark.unit
@pytest.mark.fast
def test_experiment_result_to_dict():
    """Test ExperimentResult serialization."""
    result = ExperimentResult(
        experiment_id="exp_123",
        name="loss",
        value=0.05,
    )

    data = result.to_dict()

    assert data["experiment_id"] == "exp_123"
    assert data["name"] == "loss"
    assert data["value"] == 0.05
    assert "timestamp" in data


@pytest.mark.unit
@pytest.mark.fast
def test_experiment_result_with_metadata():
    """Test ExperimentResult with metadata."""
    result = ExperimentResult(
        experiment_id="exp_123",
        name="score",
        value=100,
        metadata={"units": "points", "max": 100},
    )

    assert result.metadata["units"] == "points"
    assert result.metadata["max"] == 100


# ============================================================================
# Test: Experiment Initialization
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_experiment_creation():
    """Test basic experiment creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(
            name="test_exp",
            description="Test experiment",
            workspace_path=Path(tmpdir),
        )

        assert exp.metadata.name == "test_exp"
        assert exp.metadata.description == "Test experiment"
        assert exp.metadata.status == ExperimentStatus.CREATED
        assert exp.experiment_dir.exists()


@pytest.mark.unit
@pytest.mark.fast
def test_experiment_with_tags():
    """Test experiment creation with tags."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(
            name="test",
            workspace_path=Path(tmpdir),
            tags=["neural", "optimization"],
        )

        assert exp.metadata.tags == ["neural", "optimization"]


@pytest.mark.unit
@pytest.mark.fast
def test_experiment_generates_unique_id():
    """Test each experiment gets unique ID."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp1 = Experiment(name="exp1", workspace_path=Path(tmpdir))
        exp2 = Experiment(name="exp2", workspace_path=Path(tmpdir))

        assert exp1.metadata.id != exp2.metadata.id


# ============================================================================
# Test: Experiment Lifecycle
# ============================================================================


@pytest.mark.unit
def test_experiment_start():
    """Test starting an experiment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test", workspace_path=Path(tmpdir))

        exp.start()

        assert exp.metadata.status == ExperimentStatus.RUNNING
        assert exp.metadata.started_time is not None


@pytest.mark.unit
def test_experiment_complete():
    """Test completing an experiment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test", workspace_path=Path(tmpdir))

        exp.start()
        exp.complete()

        assert exp.metadata.status == ExperimentStatus.COMPLETED
        assert exp.metadata.completed_time is not None
        assert exp.execution_time is not None


@pytest.mark.unit
def test_experiment_fail():
    """Test failing an experiment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test", workspace_path=Path(tmpdir))

        exp.start()
        exp.fail("Test error message")

        assert exp.metadata.status == ExperimentStatus.FAILED
        assert exp.error_message == "Test error message"


@pytest.mark.unit
def test_experiment_cancel():
    """Test cancelling an experiment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test", workspace_path=Path(tmpdir))

        exp.start()
        exp.cancel()

        assert exp.metadata.status == ExperimentStatus.CANCELLED


# ============================================================================
# Test: Result Management
# ============================================================================


@pytest.mark.unit
def test_add_result():
    """Test adding results to experiment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test", workspace_path=Path(tmpdir))

        exp.add_result("accuracy", 0.95)

        assert "accuracy" in exp.results
        assert exp.results["accuracy"].value == 0.95


@pytest.mark.unit
def test_add_multiple_results():
    """Test adding multiple results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test", workspace_path=Path(tmpdir))

        exp.add_result("loss", 0.1)
        exp.add_result("accuracy", 0.9)
        exp.add_result("f1_score", 0.85)

        assert len(exp.results) == 3
        assert exp.results["loss"].value == 0.1
        assert exp.results["accuracy"].value == 0.9


@pytest.mark.unit
def test_get_result():
    """Test retrieving a specific result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test", workspace_path=Path(tmpdir))

        exp.add_result("metric", 42)
        result = exp.get_result("metric")

        assert result == 42  # get_result returns value directly


@pytest.mark.unit
def test_get_nonexistent_result():
    """Test getting non-existent result returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test", workspace_path=Path(tmpdir))

        result = exp.get_result("nonexistent")

        assert result is None


# ============================================================================
# Test: Parameter Management
# ============================================================================


@pytest.mark.unit
def test_set_parameter():
    """Test setting experiment parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test", workspace_path=Path(tmpdir))

        exp.set_parameter("learning_rate", 0.001)

        assert exp.metadata.parameters["learning_rate"] == 0.001


@pytest.mark.unit
def test_set_multiple_parameters():
    """Test setting multiple parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test", workspace_path=Path(tmpdir))

        exp.set_parameter("lr", 0.01)
        exp.set_parameter("batch_size", 32)
        exp.set_parameter("epochs", 100)

        assert len(exp.metadata.parameters) == 3


# ============================================================================
# Test: ExperimentTracker
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_tracker_creation():
    """Test ExperimentTracker initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(workspace_path=Path(tmpdir))

        assert tracker.workspace_path.exists()
        assert len(tracker.experiments) == 0


@pytest.mark.unit
def test_create_experiment_via_tracker():
    """Test creating experiment through tracker."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(workspace_path=Path(tmpdir))

        exp = tracker.create_experiment(
            name="test_exp",
            description="Test",
            tags=["test"],
        )

        assert exp.metadata.name == "test_exp"
        assert exp.metadata.id in tracker.experiments


@pytest.mark.unit
def test_tracker_get_experiment():
    """Test retrieving experiment from tracker."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(workspace_path=Path(tmpdir))

        exp = tracker.create_experiment(name="test")
        retrieved = tracker.get_experiment(exp.metadata.id)

        assert retrieved == exp


@pytest.mark.unit
def test_tracker_get_nonexistent_experiment():
    """Test getting non-existent experiment returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(workspace_path=Path(tmpdir))

        result = tracker.get_experiment("nonexistent_id")

        assert result is None


@pytest.mark.unit
def test_tracker_list_experiments():
    """Test listing all experiments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(workspace_path=Path(tmpdir))

        exp1 = tracker.create_experiment(name="exp1")
        exp2 = tracker.create_experiment(name="exp2")

        experiments = tracker.list_experiments()

        assert len(experiments) == 2
        # list_experiments returns dicts, not Experiment objects
        exp_ids = [e["id"] for e in experiments]
        assert exp1.metadata.id in exp_ids
        assert exp2.metadata.id in exp_ids


# ============================================================================
# Test: Edge Cases
# ============================================================================


@pytest.mark.unit
def test_experiment_complete_without_start():
    """Test completing experiment without starting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test", workspace_path=Path(tmpdir))

        exp.complete()

        # Should still complete, but may have None started_time
        assert exp.metadata.status == ExperimentStatus.COMPLETED


@pytest.mark.unit
def test_result_overwrite():
    """Test overwriting existing result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test", workspace_path=Path(tmpdir))

        exp.add_result("metric", 10)
        exp.add_result("metric", 20)

        assert exp.results["metric"].value == 20
