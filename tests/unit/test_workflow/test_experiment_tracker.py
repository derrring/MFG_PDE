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


# ============================================================================
# Test: Numpy Array Results
# ============================================================================


@pytest.mark.unit
def test_add_numpy_array_result():
    """Test adding numpy array as result."""
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test", workspace_path=Path(tmpdir))

        arr = np.array([1.0, 2.0, 3.0])
        exp.add_result("array_result", arr)

        assert "array_result" in exp.results
        # Array should be saved separately
        array_file = exp.experiment_dir / "arrays" / "result_array_result.npy"
        assert array_file.exists()


@pytest.mark.unit
def test_get_numpy_array_result():
    """Test retrieving numpy array result."""
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test", workspace_path=Path(tmpdir))

        original = np.array([[1, 2], [3, 4]])
        exp.add_result("matrix", original)

        # The result value should be serialized info, but get_result loads the array
        loaded = exp.get_result("matrix")
        np.testing.assert_array_equal(loaded, original)


@pytest.mark.unit
def test_experiment_result_serialize_numpy():
    """Test ExperimentResult serialization with numpy array."""
    import numpy as np

    result = ExperimentResult(
        experiment_id="exp_123",
        name="array_data",
        value=np.array([1, 2, 3]),
    )

    serialized = result._serialize_value(result.value)

    assert serialized["type"] == "numpy_array"
    assert serialized["shape"] == (3,)
    assert serialized["dtype"] == "int64"


@pytest.mark.unit
def test_experiment_result_serialize_object_with_to_dict():
    """Test ExperimentResult serialization with object having to_dict."""

    class MockResult:
        def to_dict(self):
            return {"key": "value", "number": 42}

    result = ExperimentResult(
        experiment_id="exp_123",
        name="custom",
        value=MockResult(),
    )

    serialized = result._serialize_value(result.value)
    assert serialized == {"key": "value", "number": 42}


@pytest.mark.unit
def test_experiment_result_serialize_unsupported_type():
    """Test ExperimentResult serialization with unsupported type."""

    class CustomObject:
        pass

    result = ExperimentResult(
        experiment_id="exp_123",
        name="custom",
        value=CustomObject(),
    )

    serialized = result._serialize_value(result.value)
    # Should convert to string representation
    assert isinstance(serialized, str)


# ============================================================================
# Test: Set Parameters (batch)
# ============================================================================


@pytest.mark.unit
def test_set_parameters_batch():
    """Test setting multiple parameters at once."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test", workspace_path=Path(tmpdir))

        params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
        }
        exp.set_parameters(params)

        assert exp.metadata.parameters["learning_rate"] == 0.001
        assert exp.metadata.parameters["batch_size"] == 32
        assert exp.metadata.parameters["epochs"] == 100


# ============================================================================
# Test: Artifact Management
# ============================================================================


@pytest.mark.unit
def test_add_artifact():
    """Test adding file artifact to experiment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        exp = Experiment(name="test", workspace_path=tmpdir)

        # Create a test file
        test_file = tmpdir / "test_artifact.txt"
        test_file.write_text("test content")

        exp.add_artifact("test_file", test_file)

        assert "test_file" in exp.artifacts
        artifact_path = exp.artifacts["test_file"]
        assert artifact_path.exists()
        assert artifact_path.read_text() == "test content"


@pytest.mark.unit
def test_add_artifact_nonexistent_file():
    """Test adding non-existent file as artifact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test", workspace_path=Path(tmpdir))

        # Try to add non-existent file
        exp.add_artifact("missing", Path("/nonexistent/file.txt"))

        # Should not be added
        assert "missing" not in exp.artifacts


# ============================================================================
# Test: Logging
# ============================================================================


@pytest.mark.unit
def test_log_message():
    """Test logging messages to experiment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test", workspace_path=Path(tmpdir))

        exp.log_message("info", "Test message", extra_data=42)

        assert len(exp.logs) == 1
        log_entry = exp.logs[0]
        assert log_entry["level"] == "info"
        assert log_entry["message"] == "Test message"
        assert log_entry["data"]["extra_data"] == 42


@pytest.mark.unit
def test_log_multiple_messages():
    """Test logging multiple messages."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test", workspace_path=Path(tmpdir))

        exp.log_message("info", "Starting")
        exp.log_message("debug", "Processing")
        exp.log_message("warning", "Almost done")

        assert len(exp.logs) == 3


# ============================================================================
# Test: Save and Load
# ============================================================================


@pytest.mark.unit
def test_experiment_save():
    """Test saving experiment to disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="save_test", workspace_path=Path(tmpdir))

        exp.set_parameter("param1", 100)
        exp.add_result("result1", 0.95)
        exp.start()
        exp.complete()

        exp.save()

        # Check files exist
        assert (exp.experiment_dir / "metadata.json").exists()
        assert (exp.experiment_dir / "results.json").exists()
        assert (exp.experiment_dir / "logs.json").exists()
        assert (exp.experiment_dir / "experiment.pkl").exists()


@pytest.mark.unit
def test_experiment_load():
    """Test loading experiment from disk."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create and save an experiment
        exp1 = Experiment(name="load_test", workspace_path=tmpdir)
        exp1.set_parameter("test_param", 42)
        exp1.add_result("accuracy", 0.99)
        exp1.start()
        exp1.complete()
        exp1.save()

        # Load into new experiment
        exp2 = Experiment(name="temp", workspace_path=tmpdir)
        exp2.load(exp1.experiment_dir)

        assert exp2.metadata.name == "load_test"
        assert exp2.metadata.parameters["test_param"] == 42
        assert exp2.metadata.status == ExperimentStatus.COMPLETED


@pytest.mark.unit
def test_experiment_save_load_with_arrays():
    """Test saving and loading experiment with numpy arrays."""
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create experiment with array result
        exp1 = Experiment(name="array_test", workspace_path=tmpdir)
        original_array = np.array([1.0, 2.0, 3.0, 4.0])
        exp1.add_result("data", original_array)
        exp1.save()

        # Load and verify array is recoverable
        exp2 = Experiment(name="temp", workspace_path=tmpdir)
        exp2.load(exp1.experiment_dir)

        loaded_array = exp2.get_result("data")
        np.testing.assert_array_equal(loaded_array, original_array)


# ============================================================================
# Test: Experiment Comparison
# ============================================================================


@pytest.mark.unit
def test_compare_experiments():
    """Test comparing two experiments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        exp1 = Experiment(name="exp1", workspace_path=tmpdir)
        exp1.set_parameter("lr", 0.01)
        exp1.add_result("accuracy", 0.9)
        exp1.start()
        exp1.complete()

        exp2 = Experiment(name="exp2", workspace_path=tmpdir)
        exp2.set_parameter("lr", 0.001)
        exp2.add_result("accuracy", 0.95)
        exp2.start()
        exp2.complete()

        comparison = exp1.compare_with(exp2)

        assert "experiments" in comparison
        assert "parameter_differences" in comparison
        assert "result_differences" in comparison
        assert "lr" in comparison["parameter_differences"]
        assert "accuracy" in comparison["result_differences"]


@pytest.mark.unit
def test_compare_experiments_with_arrays():
    """Test comparing experiments with array results."""
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        exp1 = Experiment(name="exp1", workspace_path=tmpdir)
        exp1.add_result("data", np.array([1.0, 2.0, 3.0]))
        exp1.start()
        exp1.complete()

        exp2 = Experiment(name="exp2", workspace_path=tmpdir)
        exp2.add_result("data", np.array([1.1, 2.0, 3.0]))
        exp2.start()
        exp2.complete()

        comparison = exp1.compare_with(exp2)

        # Should have array comparison metrics
        assert "data" in comparison["result_differences"]
        diff = comparison["result_differences"]["data"]
        assert "max_absolute_difference" in diff
        assert "mean_absolute_difference" in diff


@pytest.mark.unit
def test_compare_experiments_missing_result():
    """Test comparing experiments with missing results in one."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        exp1 = Experiment(name="exp1", workspace_path=tmpdir)
        exp1.add_result("metric1", 10)

        exp2 = Experiment(name="exp2", workspace_path=tmpdir)
        exp2.add_result("metric2", 20)

        comparison = exp1.compare_with(exp2)

        # Should note missing results
        assert comparison["result_differences"]["metric1"]["missing_in"] == "other"
        assert comparison["result_differences"]["metric2"]["missing_in"] == "this"


# ============================================================================
# Test: Tracker Delete Experiment
# ============================================================================


@pytest.mark.unit
def test_tracker_delete_experiment():
    """Test deleting experiment from tracker."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(workspace_path=Path(tmpdir))

        exp = tracker.create_experiment(name="to_delete")
        exp_id = exp.metadata.id
        exp_dir = exp.experiment_dir

        assert exp_id in tracker.experiments
        assert exp_dir.exists()

        result = tracker.delete_experiment(exp_id)

        assert result is True
        assert exp_id not in tracker.experiments
        assert not exp_dir.exists()


@pytest.mark.unit
def test_tracker_delete_nonexistent_experiment():
    """Test deleting non-existent experiment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(workspace_path=Path(tmpdir))

        result = tracker.delete_experiment("nonexistent_id")

        assert result is False


# ============================================================================
# Test: Tracker List with Filters
# ============================================================================


@pytest.mark.unit
def test_tracker_list_experiments_by_tag():
    """Test listing experiments filtered by tag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(workspace_path=Path(tmpdir))

        tracker.create_experiment(name="exp1", tags=["neural"])
        tracker.create_experiment(name="exp2", tags=["optimization"])
        tracker.create_experiment(name="exp3", tags=["neural", "optimization"])

        neural_exps = tracker.list_experiments(tags=["neural"])

        assert len(neural_exps) == 2
        names = [e["name"] for e in neural_exps]
        assert "exp1" in names
        assert "exp3" in names


@pytest.mark.unit
def test_tracker_list_experiments_by_status():
    """Test listing experiments filtered by status."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(workspace_path=Path(tmpdir))

        exp1 = tracker.create_experiment(name="exp1")
        exp1.start()
        exp1.complete()

        exp2 = tracker.create_experiment(name="exp2")
        exp2.start()
        exp2.fail("error")

        tracker.create_experiment(name="exp3")  # Still CREATED

        completed = tracker.list_experiments(status=ExperimentStatus.COMPLETED)
        failed = tracker.list_experiments(status=ExperimentStatus.FAILED)
        created = tracker.list_experiments(status=ExperimentStatus.CREATED)

        assert len(completed) == 1
        assert completed[0]["name"] == "exp1"
        assert len(failed) == 1
        assert len(created) == 1


# ============================================================================
# Test: Tracker Compare Multiple Experiments
# ============================================================================


@pytest.mark.unit
def test_tracker_compare_experiments():
    """Test comparing multiple experiments through tracker."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(workspace_path=Path(tmpdir))

        exp1 = tracker.create_experiment(name="exp1")
        exp1.set_parameter("lr", 0.01)
        exp1.add_result("score", 85)
        exp1.start()
        exp1.complete()

        exp2 = tracker.create_experiment(name="exp2")
        exp2.set_parameter("lr", 0.001)
        exp2.add_result("score", 90)
        exp2.start()
        exp2.complete()

        comparison = tracker.compare_experiments([exp1.metadata.id, exp2.metadata.id])

        assert comparison["experiment_count"] == 2
        assert "pairwise_comparisons" in comparison
        assert "aggregate_analysis" in comparison


@pytest.mark.unit
def test_tracker_compare_experiments_insufficient():
    """Test comparing with less than 2 experiments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(workspace_path=Path(tmpdir))

        exp = tracker.create_experiment(name="solo")

        result = tracker.compare_experiments([exp.metadata.id])

        assert "error" in result


# ============================================================================
# Test: Find Similar Experiments
# ============================================================================


@pytest.mark.unit
def test_tracker_find_similar_experiments():
    """Test finding similar experiments."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(workspace_path=Path(tmpdir))

        # Create experiments with similar parameters
        exp1 = tracker.create_experiment(name="base", tags=["neural"])
        exp1.set_parameters({"lr": 0.01, "batch": 32, "epochs": 100})

        exp2 = tracker.create_experiment(name="similar", tags=["neural"])
        exp2.set_parameters({"lr": 0.01, "batch": 32, "epochs": 100})

        exp3 = tracker.create_experiment(name="different", tags=["other"])
        exp3.set_parameters({"lr": 0.1, "batch": 64, "epochs": 50})

        similar = tracker.find_similar_experiments(exp1.metadata.id, similarity_threshold=0.5)

        # exp2 should be most similar
        assert len(similar) >= 1
        similar_ids = [s["experiment"]["id"] for s in similar]
        assert exp2.metadata.id in similar_ids


@pytest.mark.unit
def test_tracker_find_similar_nonexistent():
    """Test finding similar experiments for non-existent experiment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(workspace_path=Path(tmpdir))

        result = tracker.find_similar_experiments("nonexistent")

        assert result == []


# ============================================================================
# Test: Export Experiments
# ============================================================================


@pytest.mark.unit
def test_tracker_export_experiments_json():
    """Test exporting experiments to JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(workspace_path=Path(tmpdir))

        exp = tracker.create_experiment(name="export_test")
        exp.set_parameter("param", 123)
        exp.add_result("result", 0.99)

        export_path = tracker.export_experiments([exp.metadata.id], export_format="json")

        assert Path(export_path).exists()
        assert export_path.endswith(".json")

        # Verify content
        import json

        with open(export_path) as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["metadata"]["name"] == "export_test"


@pytest.mark.unit
def test_tracker_export_experiments_csv():
    """Test exporting experiments to CSV."""
    pytest.importorskip("pandas")

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(workspace_path=Path(tmpdir))

        exp = tracker.create_experiment(name="export_csv")
        exp.set_parameter("param", 456)
        exp.add_result("metric", 0.88)

        export_path = tracker.export_experiments([exp.metadata.id], export_format="csv")

        assert Path(export_path).exists()
        assert export_path.endswith(".csv")


@pytest.mark.unit
def test_tracker_export_invalid_format():
    """Test exporting with invalid format raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(workspace_path=Path(tmpdir))

        exp = tracker.create_experiment(name="test")

        with pytest.raises(ValueError, match="Unsupported export format"):
            tracker.export_experiments([exp.metadata.id], export_format="invalid")


# ============================================================================
# Test: Tracker Load Existing Experiments
# ============================================================================


@pytest.mark.unit
def test_tracker_loads_existing_experiments():
    """Test tracker loads experiments from disk on init."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create and save an experiment
        tracker1 = ExperimentTracker(workspace_path=tmpdir)
        exp = tracker1.create_experiment(name="persistent")
        exp.set_parameter("key", "value")
        exp.save()

        # Create new tracker - should load existing
        tracker2 = ExperimentTracker(workspace_path=tmpdir)

        assert len(tracker2.experiments) == 1
        loaded_exp = next(iter(tracker2.experiments.values()))
        assert loaded_exp.metadata.name == "persistent"


# ============================================================================
# Test: Experiment Fail Tracking
# ============================================================================


@pytest.mark.unit
def test_experiment_fail_with_execution_time():
    """Test experiment fail records execution time."""
    import time

    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test", workspace_path=Path(tmpdir))

        exp.start()
        time.sleep(0.01)  # Small delay
        exp.fail("Test failure")

        assert exp.execution_time is not None
        assert exp.execution_time > 0


# ============================================================================
# Test: New Improvements
# ============================================================================


@pytest.mark.unit
def test_experiment_from_path():
    """Test loading experiment via from_path classmethod."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create and save an experiment
        exp1 = Experiment(name="from_path_test", workspace_path=tmpdir)
        exp1.set_parameter("key", "value")
        exp1.add_result("metric", 42)
        exp1.start()
        exp1.complete()
        exp1.save()

        # Load via from_path (should not create temp directory)
        exp2 = Experiment.from_path(exp1.experiment_dir)

        assert exp2.metadata.name == "from_path_test"
        assert exp2.metadata.parameters["key"] == "value"
        assert exp2.metadata.status == ExperimentStatus.COMPLETED


@pytest.mark.unit
def test_experiment_from_path_not_found():
    """Test from_path raises error for non-existent directory."""
    with pytest.raises(FileNotFoundError):
        Experiment.from_path(Path("/nonexistent/path"))


@pytest.mark.unit
def test_experiment_from_path_invalid_directory():
    """Test from_path raises error for invalid experiment directory."""
    with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(ValueError, match="Not a valid experiment directory"):
        # Directory exists but has no metadata.json
        Experiment.from_path(Path(tmpdir))


@pytest.mark.unit
def test_log_message_invalid_level():
    """Test log_message raises error for invalid level."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test", workspace_path=Path(tmpdir))

        with pytest.raises(ValueError, match="Invalid log level"):
            exp.log_message("invalid_level", "test message")


@pytest.mark.unit
def test_log_message_case_insensitive():
    """Test log_message accepts different cases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp = Experiment(name="test", workspace_path=Path(tmpdir))

        # These should all work
        exp.log_message("INFO", "uppercase")
        exp.log_message("Info", "mixed case")
        exp.log_message("info", "lowercase")

        assert len(exp.logs) == 3
        # All should be normalized to lowercase
        assert all(log["level"] == "info" for log in exp.logs)


@pytest.mark.unit
def test_compare_experiments_shape_mismatch():
    """Test comparing experiments with different shaped arrays."""
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        exp1 = Experiment(name="exp1", workspace_path=tmpdir)
        exp1.add_result("data", np.array([1.0, 2.0, 3.0]))

        exp2 = Experiment(name="exp2", workspace_path=tmpdir)
        exp2.add_result("data", np.array([[1.0, 2.0], [3.0, 4.0]]))

        comparison = exp1.compare_with(exp2)

        # Should report shape mismatch
        assert "data" in comparison["result_differences"]
        assert "shape_mismatch" in comparison["result_differences"]["data"]
        assert comparison["result_differences"]["data"]["shape_mismatch"]["this"] == (3,)
        assert comparison["result_differences"]["data"]["shape_mismatch"]["other"] == (2, 2)
