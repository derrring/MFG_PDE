"""
Unit tests for PerformanceTracker

Tests the performance tracking system including:
- BenchmarkResult creation and serialization
- PerformanceTracker storage and retrieval
- Regression detection
- Statistical summaries
- Git integration
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import numpy as np

from benchmarks.performance_tracker import BenchmarkResult, PerformanceTracker


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_benchmark_result_creation(self):
        """Test creating a BenchmarkResult instance."""
        result = BenchmarkResult(
            timestamp="2025-10-11T10:00:00",
            commit_hash="abc123",
            branch="main",
            solver_name="HJB-FDM",
            problem_name="LQ-MFG-Small",
            problem_size={"Nx": 50, "Nt": 50},
            execution_time=1.23,
            peak_memory_mb=150.5,
            converged=True,
            iterations=50,
            final_error=1.2e-6,
            python_version="3.10.0",
            numpy_version="1.24.0",
            platform="darwin",
            metadata={"note": "test"},
        )

        assert result.solver_name == "HJB-FDM"
        assert result.problem_name == "LQ-MFG-Small"
        assert result.execution_time == 1.23
        assert result.converged is True
        assert result.metadata["note"] == "test"

    def test_benchmark_result_to_dict(self):
        """Test converting BenchmarkResult to dictionary."""
        result = BenchmarkResult(
            timestamp="2025-10-11T10:00:00",
            commit_hash="abc123",
            branch="main",
            solver_name="HJB-FDM",
            problem_name="LQ-MFG-Small",
            problem_size={"Nx": 50, "Nt": 50},
            execution_time=1.23,
            peak_memory_mb=150.5,
            converged=True,
            iterations=50,
            final_error=1.2e-6,
            python_version="3.10.0",
            numpy_version="1.24.0",
            platform="darwin",
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["solver_name"] == "HJB-FDM"
        assert result_dict["execution_time"] == 1.23
        assert result_dict["problem_size"]["Nx"] == 50

    def test_benchmark_result_from_dict(self):
        """Test creating BenchmarkResult from dictionary."""
        data = {
            "timestamp": "2025-10-11T10:00:00",
            "commit_hash": "abc123",
            "branch": "main",
            "solver_name": "HJB-FDM",
            "problem_name": "LQ-MFG-Small",
            "problem_size": {"Nx": 50, "Nt": 50},
            "execution_time": 1.23,
            "peak_memory_mb": 150.5,
            "converged": True,
            "iterations": 50,
            "final_error": 1.2e-6,
            "python_version": "3.10.0",
            "numpy_version": "1.24.0",
            "platform": "darwin",
            "metadata": None,
        }

        result = BenchmarkResult.from_dict(data)
        assert result.solver_name == "HJB-FDM"
        assert result.execution_time == 1.23
        assert result.problem_size["Nx"] == 50

    def test_benchmark_result_roundtrip(self):
        """Test that to_dict() and from_dict() are inverses."""
        original = BenchmarkResult(
            timestamp="2025-10-11T10:00:00",
            commit_hash="abc123",
            branch="main",
            solver_name="HJB-FDM",
            problem_name="LQ-MFG-Small",
            problem_size={"Nx": 50, "Nt": 50},
            execution_time=1.23,
            peak_memory_mb=150.5,
            converged=True,
            iterations=50,
            final_error=1.2e-6,
            python_version="3.10.0",
            numpy_version="1.24.0",
            platform="darwin",
            metadata={"test": "value"},
        )

        # Roundtrip
        result_dict = original.to_dict()
        restored = BenchmarkResult.from_dict(result_dict)

        # Check all fields match
        assert restored.solver_name == original.solver_name
        assert restored.execution_time == original.execution_time
        assert restored.problem_size == original.problem_size
        assert restored.metadata == original.metadata


class TestPerformanceTracker:
    """Tests for PerformanceTracker class."""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Create a PerformanceTracker instance with temporary directory."""
        return PerformanceTracker(history_dir=tmp_path / "history")

    @pytest.fixture
    def mock_git_info(self):
        """Mock git subprocess calls."""
        with patch("benchmarks.performance_tracker.subprocess.run") as mock_run:
            # Configure mock to return git info
            mock_run.return_value = MagicMock(
                stdout="abc123\n",
                returncode=0,
            )
            yield mock_run

    def test_tracker_initialization(self, tmp_path):
        """Test that tracker creates history directory."""
        history_dir = tmp_path / "history"
        tracker = PerformanceTracker(history_dir=history_dir)

        assert tracker.history_dir == history_dir
        assert history_dir.exists()
        assert history_dir.is_dir()

    def test_track_solver_creates_result(self, tracker, mock_git_info):
        """Test that track_solver() creates a BenchmarkResult."""
        result = tracker.track_solver(
            solver_name="HJB-FDM",
            problem_name="LQ-MFG-Small",
            problem_size={"Nx": 50, "Nt": 50},
            execution_time=1.23,
            converged=True,
            iterations=50,
            final_error=1.2e-6,
            peak_memory_mb=150.5,
        )

        assert isinstance(result, BenchmarkResult)
        assert result.solver_name == "HJB-FDM"
        assert result.problem_name == "LQ-MFG-Small"
        assert result.execution_time == 1.23
        assert result.converged is True

    def test_track_solver_saves_to_file(self, tracker, mock_git_info):
        """Test that track_solver() saves result to JSON file."""
        tracker.track_solver(
            solver_name="HJB-FDM",
            problem_name="LQ-MFG-Small",
            problem_size={"Nx": 50, "Nt": 50},
            execution_time=1.23,
            converged=True,
            iterations=50,
            final_error=1.2e-6,
        )

        # Check that JSON file was created
        expected_file = tracker.history_dir / "HJB-FDM_LQ-MFG-Small.json"
        assert expected_file.exists()

        # Check file contents
        with open(expected_file) as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["solver_name"] == "HJB-FDM"
        assert data[0]["execution_time"] == 1.23

    def test_track_solver_appends_to_existing_file(self, tracker, mock_git_info):
        """Test that multiple track_solver() calls append to same file."""
        # Track first result
        tracker.track_solver(
            solver_name="HJB-FDM",
            problem_name="LQ-MFG-Small",
            problem_size={"Nx": 50, "Nt": 50},
            execution_time=1.23,
            converged=True,
            iterations=50,
            final_error=1.2e-6,
        )

        # Track second result
        tracker.track_solver(
            solver_name="HJB-FDM",
            problem_name="LQ-MFG-Small",
            problem_size={"Nx": 50, "Nt": 50},
            execution_time=1.45,
            converged=True,
            iterations=55,
            final_error=1.5e-6,
        )

        # Check that both results are in file
        expected_file = tracker.history_dir / "HJB-FDM_LQ-MFG-Small.json"
        with open(expected_file) as f:
            data = json.load(f)

        assert len(data) == 2
        assert data[0]["execution_time"] == 1.23
        assert data[1]["execution_time"] == 1.45

    def test_load_history_returns_results(self, tracker, mock_git_info):
        """Test that load_history() returns BenchmarkResult objects."""
        # Track some results
        tracker.track_solver(
            solver_name="HJB-FDM",
            problem_name="LQ-MFG-Small",
            problem_size={"Nx": 50, "Nt": 50},
            execution_time=1.23,
            converged=True,
            iterations=50,
            final_error=1.2e-6,
        )

        # Load history
        history = tracker.load_history("HJB-FDM", "LQ-MFG-Small")

        assert len(history) == 1
        assert isinstance(history[0], BenchmarkResult)
        assert history[0].solver_name == "HJB-FDM"
        assert history[0].execution_time == 1.23

    def test_load_history_sorts_by_timestamp(self, tracker, tmp_path):
        """Test that load_history() sorts results by timestamp."""
        # Manually create JSON file with out-of-order timestamps
        results = [
            {
                "timestamp": "2025-10-11T12:00:00",
                "commit_hash": "abc",
                "branch": "main",
                "solver_name": "HJB-FDM",
                "problem_name": "Test",
                "problem_size": {"Nx": 50},
                "execution_time": 2.0,
                "peak_memory_mb": None,
                "converged": True,
                "iterations": 50,
                "final_error": 1e-6,
                "python_version": "3.10.0",
                "numpy_version": "1.24.0",
                "platform": "darwin",
                "metadata": None,
            },
            {
                "timestamp": "2025-10-11T10:00:00",
                "commit_hash": "abc",
                "branch": "main",
                "solver_name": "HJB-FDM",
                "problem_name": "Test",
                "problem_size": {"Nx": 50},
                "execution_time": 1.0,
                "peak_memory_mb": None,
                "converged": True,
                "iterations": 50,
                "final_error": 1e-6,
                "python_version": "3.10.0",
                "numpy_version": "1.24.0",
                "platform": "darwin",
                "metadata": None,
            },
        ]

        filepath = tracker.history_dir / "HJB-FDM_Test.json"
        with open(filepath, "w") as f:
            json.dump(results, f)

        # Load and verify sorting
        history = tracker.load_history("HJB-FDM", "Test")

        assert len(history) == 2
        assert history[0].timestamp == "2025-10-11T10:00:00"
        assert history[1].timestamp == "2025-10-11T12:00:00"

    def test_load_history_empty_for_nonexistent(self, tracker):
        """Test that load_history() returns empty list for nonexistent file."""
        history = tracker.load_history("NonExistent", "Solver")
        assert history == []

    def test_check_regression_insufficient_history(self, tracker, mock_git_info):
        """Test regression detection with insufficient history."""
        # Track only 2 results (need 5 for baseline)
        tracker.track_solver(
            solver_name="HJB-FDM",
            problem_name="Test",
            problem_size={"Nx": 50},
            execution_time=1.0,
            converged=True,
            iterations=50,
            final_error=1e-6,
        )

        result = tracker.track_solver(
            solver_name="HJB-FDM",
            problem_name="Test",
            problem_size={"Nx": 50},
            execution_time=2.0,
            converged=True,
            iterations=50,
            final_error=1e-6,
        )

        # Should not detect regression (insufficient history)
        is_regression, pct_change = tracker.check_regression(result)
        assert is_regression is False
        assert pct_change == 0.0

    def test_check_regression_detects_slowdown(self, tracker, mock_git_info):
        """Test that regression detection identifies performance degradation."""
        # Track 5 fast results
        for _ in range(5):
            tracker.track_solver(
                solver_name="HJB-FDM",
                problem_name="Test",
                problem_size={"Nx": 50},
                execution_time=1.0,
                converged=True,
                iterations=50,
                final_error=1e-6,
            )

        # Track slow result (30% slower, exceeds 20% threshold)
        slow_result = tracker.track_solver(
            solver_name="HJB-FDM",
            problem_name="Test",
            problem_size={"Nx": 50},
            execution_time=1.3,
            converged=True,
            iterations=50,
            final_error=1e-6,
        )

        is_regression, pct_change = tracker.check_regression(slow_result, threshold=0.2)
        assert is_regression is True
        assert pct_change > 0.2  # More than 20% slower

    def test_check_regression_no_false_positive(self, tracker, mock_git_info):
        """Test that regression detection doesn't trigger on small variations."""
        # Track 5 results with ~1.0s execution time
        for _ in range(5):
            tracker.track_solver(
                solver_name="HJB-FDM",
                problem_name="Test",
                problem_size={"Nx": 50},
                execution_time=1.0,
                converged=True,
                iterations=50,
                final_error=1e-6,
            )

        # Track slightly slower result (10% slower, under 20% threshold)
        result = tracker.track_solver(
            solver_name="HJB-FDM",
            problem_name="Test",
            problem_size={"Nx": 50},
            execution_time=1.1,
            converged=True,
            iterations=50,
            final_error=1e-6,
        )

        is_regression, pct_change = tracker.check_regression(result, threshold=0.2)
        assert is_regression is False
        assert 0.0 < pct_change < 0.2

    def test_get_statistics_empty_history(self, tracker):
        """Test statistics for nonexistent history."""
        stats = tracker.get_statistics("NonExistent", "Solver")
        assert stats == {"count": 0}

    def test_get_statistics_computes_correctly(self, tracker, mock_git_info):
        """Test that statistics are computed correctly."""
        # Track results with known values
        execution_times = [1.0, 1.1, 1.2, 1.3, 1.4]
        for time in execution_times:
            tracker.track_solver(
                solver_name="HJB-FDM",
                problem_name="Test",
                problem_size={"Nx": 50},
                execution_time=time,
                converged=True,
                iterations=50,
                final_error=1e-6,
            )

        stats = tracker.get_statistics("HJB-FDM", "Test")

        assert stats["count"] == 5
        assert np.isclose(stats["time_mean"], np.mean(execution_times))
        assert np.isclose(stats["time_std"], np.std(execution_times))
        assert stats["time_min"] == 1.0
        assert stats["time_max"] == 1.4
        assert stats["convergence_rate"] == 1.0  # All converged

    def test_get_statistics_convergence_rate(self, tracker, mock_git_info):
        """Test convergence rate calculation."""
        # Track 5 results, 3 converged, 2 failed
        convergence_flags = [True, True, False, True, False]
        for converged in convergence_flags:
            tracker.track_solver(
                solver_name="HJB-FDM",
                problem_name="Test",
                problem_size={"Nx": 50},
                execution_time=1.0,
                converged=converged,
                iterations=50,
                final_error=1e-6,
            )

        stats = tracker.get_statistics("HJB-FDM", "Test")
        assert stats["convergence_rate"] == 0.6  # 3/5 = 0.6

    def test_git_commit_extraction(self):
        """Test that git commit hash is extracted correctly."""
        with patch("benchmarks.performance_tracker.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="abc123def456\n",
                returncode=0,
            )

            commit = PerformanceTracker._get_git_commit()
            assert commit == "abc123def456"

    def test_git_commit_fallback_on_error(self):
        """Test that git commit extraction falls back to 'unknown' on error."""
        with patch("benchmarks.performance_tracker.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            commit = PerformanceTracker._get_git_commit()
            assert commit == "unknown"

    def test_git_branch_extraction(self):
        """Test that git branch name is extracted correctly."""
        with patch("benchmarks.performance_tracker.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="feature/test-branch\n",
                returncode=0,
            )

            branch = PerformanceTracker._get_git_branch()
            assert branch == "feature/test-branch"

    def test_memory_tracking_with_psutil(self, tracker, mock_git_info):
        """Test memory tracking when psutil is available."""
        pytest.importorskip("psutil")  # Skip if psutil not available

        result = tracker.track_solver(
            solver_name="HJB-FDM",
            problem_name="Test",
            problem_size={"Nx": 50},
            execution_time=1.0,
            converged=True,
            iterations=50,
            final_error=1e-6,
        )

        # Should have auto-detected memory
        assert result.peak_memory_mb is not None
        assert result.peak_memory_mb > 0

    def test_memory_tracking_manual_override(self, tracker, mock_git_info):
        """Test that manual memory value overrides auto-detection."""
        result = tracker.track_solver(
            solver_name="HJB-FDM",
            problem_name="Test",
            problem_size={"Nx": 50},
            execution_time=1.0,
            converged=True,
            iterations=50,
            final_error=1e-6,
            peak_memory_mb=250.5,
        )

        assert result.peak_memory_mb == 250.5

    def test_metadata_storage(self, tracker, mock_git_info):
        """Test that custom metadata is stored correctly."""
        metadata = {
            "solver_config": "fast",
            "notes": "test run",
            "tags": ["regression", "benchmark"],
        }

        result = tracker.track_solver(
            solver_name="HJB-FDM",
            problem_name="Test",
            problem_size={"Nx": 50},
            execution_time=1.0,
            converged=True,
            iterations=50,
            final_error=1e-6,
            metadata=metadata,
        )

        assert result.metadata == metadata

        # Verify it's saved to file
        history = tracker.load_history("HJB-FDM", "Test")
        assert history[0].metadata == metadata
