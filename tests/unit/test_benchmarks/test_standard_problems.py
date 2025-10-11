"""
Unit tests for standard_problems module

Tests the benchmark problem specifications and registry.
"""

from __future__ import annotations

import pytest

from benchmarks.standard_problems import (
    BenchmarkProblem,
    get_all_problems,
    get_problem_by_name,
    get_problem_info,
    get_problems_by_category,
)


class TestBenchmarkProblem:
    """Tests for BenchmarkProblem dataclass."""

    def test_benchmark_problem_creation(self):
        """Test creating a BenchmarkProblem instance."""
        problem = BenchmarkProblem(
            name="Test-Problem",
            category="small",
            description="Test problem for unit tests",
            solver_type="hjb_fdm",
            problem_config={"Nx": 50, "Nt": 50},
            solver_config={"max_iterations": 100, "tolerance": 1e-6},
            expected_time_range=(1.0, 5.0),
            convergence_threshold=1e-6,
            success_criteria={"must_converge": True},
        )

        assert problem.name == "Test-Problem"
        assert problem.category == "small"
        assert problem.solver_type == "hjb_fdm"
        assert problem.problem_config["Nx"] == 50
        assert problem.expected_time_range == (1.0, 5.0)


class TestProblemRegistry:
    """Tests for problem registry functions."""

    def test_get_all_problems_returns_list(self):
        """Test that get_all_problems() returns a list."""
        problems = get_all_problems()
        assert isinstance(problems, list)
        assert len(problems) > 0
        assert all(isinstance(p, BenchmarkProblem) for p in problems)

    def test_get_all_problems_has_expected_count(self):
        """Test that we have the expected number of standard problems."""
        problems = get_all_problems()
        # Should have: 2 small + 2 medium + 1 large = 5 problems
        assert len(problems) == 5

    def test_get_all_problems_unique_names(self):
        """Test that all problem names are unique."""
        problems = get_all_problems()
        names = [p.name for p in problems]
        assert len(names) == len(set(names))  # No duplicates

    def test_get_problems_by_category_small(self):
        """Test filtering by 'small' category."""
        problems = get_problems_by_category("small")
        assert len(problems) == 2
        assert all(p.category == "small" for p in problems)

    def test_get_problems_by_category_medium(self):
        """Test filtering by 'medium' category."""
        problems = get_problems_by_category("medium")
        assert len(problems) == 2
        assert all(p.category == "medium" for p in problems)

    def test_get_problems_by_category_large(self):
        """Test filtering by 'large' category."""
        problems = get_problems_by_category("large")
        assert len(problems) == 1
        assert all(p.category == "large" for p in problems)

    def test_get_problems_by_category_all(self):
        """Test filtering with 'all' category."""
        problems = get_problems_by_category("all")
        assert len(problems) == 5

    def test_get_problem_by_name_valid(self):
        """Test retrieving problem by valid name."""
        problem = get_problem_by_name("LQ-MFG-Small")
        assert problem.name == "LQ-MFG-Small"
        assert problem.category == "small"
        assert isinstance(problem, BenchmarkProblem)

    def test_get_problem_by_name_invalid(self):
        """Test that invalid name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown problem"):
            get_problem_by_name("NonExistent-Problem")

    def test_get_problem_info_returns_string(self):
        """Test that get_problem_info() returns formatted string."""
        info = get_problem_info()
        assert isinstance(info, str)
        assert len(info) > 0
        assert "Standard Benchmark Problems" in info


class TestSmallProblems:
    """Tests for small benchmark problems."""

    @pytest.fixture
    def lq_small(self):
        """Get LQ-MFG-Small problem."""
        return get_problem_by_name("LQ-MFG-Small")

    @pytest.fixture
    def congestion_small(self):
        """Get Congestion-Small problem."""
        return get_problem_by_name("Congestion-Small")

    def test_lq_small_configuration(self, lq_small):
        """Test LQ-MFG-Small problem configuration."""
        assert lq_small.name == "LQ-MFG-Small"
        assert lq_small.category == "small"
        assert lq_small.solver_type == "hjb_fdm"
        assert lq_small.problem_config["Nx"] == 50
        assert lq_small.problem_config["Nt"] == 50
        assert lq_small.convergence_threshold == 1e-6

    def test_lq_small_expected_time(self, lq_small):
        """Test that small problems have reasonable time expectations."""
        min_time, max_time = lq_small.expected_time_range
        assert min_time < max_time
        assert max_time <= 5.0  # Small problems should be fast

    def test_congestion_small_configuration(self, congestion_small):
        """Test Congestion-Small problem configuration."""
        assert congestion_small.name == "Congestion-Small"
        assert congestion_small.category == "small"
        assert congestion_small.problem_config["Nx"] == 50
        assert congestion_small.problem_config["Nt"] == 50


class TestMediumProblems:
    """Tests for medium benchmark problems."""

    @pytest.fixture
    def lq_medium(self):
        """Get LQ-MFG-Medium problem."""
        return get_problem_by_name("LQ-MFG-Medium")

    @pytest.fixture
    def congestion_medium(self):
        """Get Congestion-Medium problem."""
        return get_problem_by_name("Congestion-Medium")

    def test_lq_medium_configuration(self, lq_medium):
        """Test LQ-MFG-Medium problem configuration."""
        assert lq_medium.name == "LQ-MFG-Medium"
        assert lq_medium.category == "medium"
        assert lq_medium.problem_config["Nx"] == 100
        assert lq_medium.problem_config["Nt"] == 100

    def test_medium_problems_larger_than_small(self, lq_medium):
        """Test that medium problems have larger grids than small."""
        lq_small = get_problem_by_name("LQ-MFG-Small")
        assert lq_medium.problem_config["Nx"] > lq_small.problem_config["Nx"]
        assert lq_medium.problem_config["Nt"] > lq_small.problem_config["Nt"]

    def test_medium_expected_time_longer_than_small(self, lq_medium):
        """Test that medium problems expect longer execution times."""
        lq_small = get_problem_by_name("LQ-MFG-Small")
        medium_min, _medium_max = lq_medium.expected_time_range
        _small_min, small_max = lq_small.expected_time_range
        assert medium_min >= small_max  # Medium should be slower


class TestLargeProblems:
    """Tests for large benchmark problems."""

    @pytest.fixture
    def traffic_2d(self):
        """Get Traffic-2D-Large problem."""
        return get_problem_by_name("Traffic-2D-Large")

    def test_traffic_2d_configuration(self, traffic_2d):
        """Test Traffic-2D-Large problem configuration."""
        assert traffic_2d.name == "Traffic-2D-Large"
        assert traffic_2d.category == "large"
        assert traffic_2d.problem_config["Nx"] == 50
        assert traffic_2d.problem_config["Ny"] == 50  # 2D problem
        assert traffic_2d.problem_config["Nt"] == 100

    def test_traffic_2d_is_2d(self, traffic_2d):
        """Test that Traffic-2D has 2D spatial grid."""
        assert "Ny" in traffic_2d.problem_config
        assert traffic_2d.problem_config["Nx"] > 0
        assert traffic_2d.problem_config["Ny"] > 0

    def test_large_expected_time_longest(self, traffic_2d):
        """Test that large problems expect longest execution times."""
        min_time, max_time = traffic_2d.expected_time_range
        assert min_time >= 30.0  # Large problems should be substantial
        assert max_time >= 60.0


class TestProblemConsistency:
    """Tests for consistency across all problems."""

    @pytest.fixture
    def all_problems(self):
        """Get all benchmark problems."""
        return get_all_problems()

    def test_all_have_required_fields(self, all_problems):
        """Test that all problems have required fields."""
        for problem in all_problems:
            assert problem.name
            assert problem.category in ["small", "medium", "large"]
            assert problem.description
            assert problem.solver_type
            assert problem.problem_config
            assert problem.solver_config
            assert problem.expected_time_range
            assert problem.convergence_threshold > 0
            assert problem.success_criteria

    def test_all_have_valid_time_ranges(self, all_problems):
        """Test that all time ranges are valid."""
        for problem in all_problems:
            min_time, max_time = problem.expected_time_range
            assert min_time > 0
            assert max_time > min_time

    def test_all_have_grid_sizes(self, all_problems):
        """Test that all problems specify grid sizes."""
        for problem in all_problems:
            assert "Nx" in problem.problem_config
            assert "Nt" in problem.problem_config
            assert problem.problem_config["Nx"] > 0
            assert problem.problem_config["Nt"] > 0

    def test_all_have_solver_config(self, all_problems):
        """Test that all problems have solver configuration."""
        for problem in all_problems:
            assert "max_iterations" in problem.solver_config
            assert "tolerance" in problem.solver_config
            assert problem.solver_config["max_iterations"] > 0
            assert problem.solver_config["tolerance"] > 0

    def test_all_have_success_criteria(self, all_problems):
        """Test that all problems have success criteria."""
        for problem in all_problems:
            assert "must_converge" in problem.success_criteria
