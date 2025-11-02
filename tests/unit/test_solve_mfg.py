#!/usr/bin/env python3
"""
Unit Tests for solve_mfg() High-Level Interface

Tests for the solve_mfg() convenience function in mfg_pde/solve_mfg.py.

Coverage:
- Method presets (auto, fast, accurate, research)
- Custom parameter overrides (max_iterations, tolerance)
- Automatic resolution selection
- Result structure validation
- Error handling
"""

import pytest

from mfg_pde import ExampleMFGProblem, solve_mfg


class TestSolveMFGBasic:
    """Test basic solve_mfg() functionality."""

    def test_simplest_usage(self):
        """Test simplest usage with all defaults."""
        problem = ExampleMFGProblem()
        result = solve_mfg(problem, verbose=False, max_iterations=2)

        # Should return SolverResult
        assert hasattr(result, "U")
        assert hasattr(result, "M")
        assert hasattr(result, "iterations")
        assert hasattr(result, "converged")
        assert hasattr(result, "error_history_U")
        assert hasattr(result, "error_history_M")

        # Should have run iterations
        assert result.iterations >= 1
        assert len(result.error_history_U) >= 1
        assert len(result.error_history_M) >= 1

    def test_method_auto(self):
        """Test method='auto' preset."""
        problem = ExampleMFGProblem()
        result = solve_mfg(problem, method="auto", verbose=False, max_iterations=2)

        assert result.iterations >= 1
        assert result.U.shape == result.M.shape

    def test_method_fast(self):
        """Test method='fast' preset."""
        problem = ExampleMFGProblem()
        result = solve_mfg(problem, method="fast", verbose=False, max_iterations=2)

        assert result.iterations >= 1
        assert result.U.shape == result.M.shape

    def test_method_accurate(self):
        """Test method='accurate' preset."""
        problem = ExampleMFGProblem()
        result = solve_mfg(problem, method="accurate", verbose=False, max_iterations=2)

        assert result.iterations >= 1
        assert result.U.shape == result.M.shape

    def test_method_research(self):
        """Test method='research' preset."""
        problem = ExampleMFGProblem()
        result = solve_mfg(problem, method="research", verbose=False, max_iterations=2)

        assert result.iterations >= 1
        assert result.U.shape == result.M.shape


class TestSolveMFGParameterOverrides:
    """Test custom parameter overrides."""

    def test_custom_max_iterations(self):
        """Test max_iterations override."""
        problem = ExampleMFGProblem()
        result = solve_mfg(problem, method="fast", max_iterations=3, verbose=False)

        # Should stop at exactly 3 iterations (or converge earlier)
        assert result.iterations <= 3

    def test_custom_tolerance(self):
        """Test tolerance override."""
        problem = ExampleMFGProblem()
        # Note: Phase 3.2 removed 'convergence_tolerance' field
        # tolerance parameter affects picard.tolerance only
        result = solve_mfg(problem, method="fast", tolerance=1e-2, max_iterations=5, verbose=False)

        # Tolerance affects convergence criteria
        # Just verify it runs without error
        assert result.iterations >= 1

    def test_custom_max_iterations_and_tolerance(self):
        """Test both max_iterations and tolerance overrides."""
        problem = ExampleMFGProblem()
        # Note: Phase 3.2 removed 'convergence_tolerance' field
        # tolerance parameter affects picard.tolerance only
        result = solve_mfg(problem, method="accurate", max_iterations=4, tolerance=1e-4, verbose=False)

        assert result.iterations <= 4
        assert result.iterations >= 1

    def test_verbose_parameter(self):
        """Test verbose parameter."""
        problem = ExampleMFGProblem()

        # Should work with verbose=True
        result = solve_mfg(problem, method="fast", max_iterations=2, verbose=True)
        assert result.iterations >= 1

        # Should work with verbose=False
        result = solve_mfg(problem, method="fast", max_iterations=2, verbose=False)
        assert result.iterations >= 1


class TestSolveMFGErrorHandling:
    """Test error handling."""

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        problem = ExampleMFGProblem()

        # Phase 3.2/3.3: Error message changed from "Unknown method" to "Unknown config preset"
        with pytest.raises(ValueError, match="Unknown config preset"):
            solve_mfg(problem, method="invalid_method")

    def test_valid_methods_list(self):
        """Test all valid methods work."""
        problem = ExampleMFGProblem()
        valid_methods = ["auto", "fast", "accurate", "research"]

        for method in valid_methods:
            result = solve_mfg(problem, method=method, max_iterations=2, verbose=False)
            assert result.iterations >= 1, f"Method {method} failed"


class TestSolveMFGResultStructure:
    """Test result structure and attributes."""

    def test_result_has_required_attributes(self):
        """Test SolverResult has all required attributes."""
        problem = ExampleMFGProblem()
        result = solve_mfg(problem, method="fast", max_iterations=2, verbose=False)

        # Core arrays
        assert hasattr(result, "U")
        assert hasattr(result, "M")

        # Convergence info
        assert hasattr(result, "iterations")
        assert hasattr(result, "converged")

        # Error histories
        assert hasattr(result, "error_history_U")
        assert hasattr(result, "error_history_M")

        # Metadata
        assert hasattr(result, "solver_name")
        assert hasattr(result, "metadata")

    def test_result_array_shapes(self):
        """Test that U and M have consistent shapes."""
        problem = ExampleMFGProblem()
        result = solve_mfg(problem, method="fast", max_iterations=2, verbose=False)

        # U and M should have same shape
        assert result.U.shape == result.M.shape

        # Should be 2D arrays (Nt+1, Nx+1)
        assert result.U.ndim == 2
        assert result.M.ndim == 2

    def test_error_histories_length(self):
        """Test error history arrays have correct length."""
        problem = ExampleMFGProblem()
        result = solve_mfg(problem, method="fast", max_iterations=3, verbose=False)

        # Error histories should have length >= iterations
        # (may have initial error)
        assert len(result.error_history_U) >= result.iterations
        assert len(result.error_history_M) >= result.iterations


class TestSolveMFGKwargsPassthrough:
    """Test that **kwargs are passed through to solver."""

    def test_damping_factor_kwarg(self):
        """Test that damping_factor is passed through."""
        problem = ExampleMFGProblem()

        # Should accept damping_factor via **kwargs
        result = solve_mfg(problem, method="fast", damping_factor=0.3, max_iterations=2, verbose=False)

        assert result.iterations >= 1

    def test_backend_kwarg(self):
        """Test that backend parameter is passed through."""
        problem = ExampleMFGProblem()

        # Should accept backend string via **kwargs (converted to backend object internally)
        result = solve_mfg(problem, method="fast", backend="numpy", max_iterations=2, verbose=False)

        assert result.iterations >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
