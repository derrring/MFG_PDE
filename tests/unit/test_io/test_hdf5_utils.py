#!/usr/bin/env python3
"""
Unit tests for HDF5 utilities.

Tests save_solution, load_solution, save_checkpoint, load_checkpoint,
and SolverResult integration.
"""

import tempfile
from pathlib import Path

import pytest

import numpy as np

from mfg_pde.utils.solver_result import SolverResult

try:
    from mfg_pde.utils.io import HDF5_AVAILABLE
except ImportError:
    HDF5_AVAILABLE = False

if HDF5_AVAILABLE:
    from mfg_pde.utils.io.hdf5_utils import (
        get_hdf5_info,
        load_checkpoint,
        load_solution,
        save_checkpoint,
        save_solution,
    )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_solution():
    """Create sample solution arrays for testing."""
    Nt, Nx = 10, 20
    U = np.random.randn(Nt, Nx)
    M = np.random.rand(Nt, Nx)
    M = M / M.sum()  # Normalize to valid density
    return U, M


@pytest.fixture
def sample_metadata():
    """Create sample metadata dictionary."""
    return {
        "converged": True,
        "iterations": 47,
        "solver_name": "FixedPointIterator",
        "tolerance": 1e-6,
        "execution_time": 12.34,
    }


@pytest.fixture
def sample_grids():
    """Create sample spatial and temporal grids."""
    x_grid = np.linspace(0, 1, 20)
    t_grid = np.linspace(0, 1, 10)
    return x_grid, t_grid


@pytest.fixture
def temp_hdf5_file():
    """Create temporary HDF5 file path."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        filepath = Path(tmp.name)
    yield filepath
    # Cleanup after test
    if filepath.exists():
        filepath.unlink()


# ============================================================================
# Test: Basic Save/Load
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
def test_save_and_load_solution(sample_solution, sample_metadata, temp_hdf5_file):
    """Test saving and loading solution with metadata."""
    U, M = sample_solution

    # Save solution
    save_solution(U, M, sample_metadata, temp_hdf5_file)

    # Load solution
    U_loaded, M_loaded, metadata_loaded = load_solution(temp_hdf5_file)

    # Verify arrays
    np.testing.assert_array_equal(U, U_loaded)
    np.testing.assert_array_equal(M, M_loaded)

    # Verify metadata (basic fields)
    assert metadata_loaded["converged"] == sample_metadata["converged"]
    assert metadata_loaded["iterations"] == sample_metadata["iterations"]
    assert metadata_loaded["solver_name"] == sample_metadata["solver_name"]


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
def test_save_solution_with_grids(sample_solution, sample_metadata, sample_grids, temp_hdf5_file):
    """Test saving solution with spatial and temporal grids."""
    U, M = sample_solution
    x_grid, t_grid = sample_grids

    # Save with grids
    save_solution(U, M, sample_metadata, temp_hdf5_file, x_grid=x_grid, t_grid=t_grid)

    # Load and verify
    U_loaded, M_loaded, metadata_loaded = load_solution(temp_hdf5_file)

    np.testing.assert_array_equal(U, U_loaded)
    np.testing.assert_array_equal(M, M_loaded)
    np.testing.assert_array_equal(x_grid, metadata_loaded["x_grid"])
    np.testing.assert_array_equal(t_grid, metadata_loaded["t_grid"])


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
def test_save_solution_with_compression(sample_solution, sample_metadata, temp_hdf5_file):
    """Test saving with different compression settings."""
    U, M = sample_solution

    # Save with gzip compression
    save_solution(U, M, sample_metadata, temp_hdf5_file, compression="gzip", compression_opts=9)

    # Verify can load
    U_loaded, M_loaded, _ = load_solution(temp_hdf5_file)
    np.testing.assert_array_almost_equal(U, U_loaded)
    np.testing.assert_array_almost_equal(M, M_loaded)


# ============================================================================
# Test: Checkpoint Save/Load
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
def test_save_and_load_checkpoint(sample_solution, temp_hdf5_file):
    """Test checkpoint save and load for resuming computation."""
    U, M = sample_solution

    # Create solver state
    solver_state = {
        "U": U,
        "M": M,
        "iteration": 25,
        "residuals_u": np.array([1e-1, 1e-2, 1e-3]),
        "residuals_m": np.array([2e-1, 2e-2, 2e-3]),
        "config": {"tolerance": 1e-6, "max_iterations": 100},
    }

    # Save checkpoint
    save_checkpoint(solver_state, temp_hdf5_file)

    # Load checkpoint
    state_loaded = load_checkpoint(temp_hdf5_file)

    # Verify state
    np.testing.assert_array_equal(U, state_loaded["U"])
    np.testing.assert_array_equal(M, state_loaded["M"])
    assert state_loaded["iteration"] == 25
    np.testing.assert_array_equal(solver_state["residuals_u"], state_loaded["residuals_u"])
    np.testing.assert_array_equal(solver_state["residuals_m"], state_loaded["residuals_m"])
    assert state_loaded["config"]["tolerance"] == 1e-6


# ============================================================================
# Test: File Info
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
def test_get_hdf5_info(sample_solution, sample_metadata, temp_hdf5_file):
    """Test getting file information without loading data."""
    U, M = sample_solution
    save_solution(U, M, sample_metadata, temp_hdf5_file)

    # Get info
    info = get_hdf5_info(temp_hdf5_file)

    # Verify structure
    assert "U_shape" in info
    assert "M_shape" in info
    assert info["U_shape"] == U.shape
    assert info["M_shape"] == M.shape
    assert info["format_version"] == "1.0"


# ============================================================================
# Test: SolverResult Integration
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
def test_solver_result_save_hdf5(sample_solution, temp_hdf5_file):
    """Test SolverResult.save_hdf5() method."""
    U, M = sample_solution
    error_history_U = np.array([1e-1, 1e-2, 1e-3])
    error_history_M = np.array([2e-1, 2e-2, 2e-3])

    # Create result
    result = SolverResult(
        U=U,
        M=M,
        iterations=3,
        error_history_U=error_history_U,
        error_history_M=error_history_M,
        solver_name="TestSolver",
        converged=True,
        execution_time=10.5,
    )

    # Save via SolverResult method
    result.save_hdf5(temp_hdf5_file)

    # Verify file exists and can be loaded
    assert temp_hdf5_file.exists()
    U_loaded, M_loaded, metadata = load_solution(temp_hdf5_file)

    np.testing.assert_array_equal(U, U_loaded)
    np.testing.assert_array_equal(M, M_loaded)
    assert metadata["solver_name"] == "TestSolver"
    assert metadata["converged"]
    assert metadata["iterations"] == 3


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
def test_solver_result_load_hdf5(sample_solution, temp_hdf5_file):
    """Test SolverResult.load_hdf5() class method."""
    U, M = sample_solution
    error_history_U = np.array([1e-1, 1e-2, 1e-3])
    error_history_M = np.array([2e-1, 2e-2, 2e-3])

    # Create and save result
    result_original = SolverResult(
        U=U,
        M=M,
        iterations=3,
        error_history_U=error_history_U,
        error_history_M=error_history_M,
        solver_name="TestSolver",
        converged=True,
        execution_time=10.5,
    )
    result_original.save_hdf5(temp_hdf5_file)

    # Load via class method
    result_loaded = SolverResult.load_hdf5(temp_hdf5_file)

    # Verify reconstruction
    np.testing.assert_array_equal(result_original.U, result_loaded.U)
    np.testing.assert_array_equal(result_original.M, result_loaded.M)
    assert result_loaded.iterations == 3
    assert result_loaded.solver_name == "TestSolver"
    assert result_loaded.converged
    assert result_loaded.execution_time == 10.5
    np.testing.assert_array_equal(result_original.error_history_U, result_loaded.error_history_U)
    np.testing.assert_array_equal(result_original.error_history_M, result_loaded.error_history_M)


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
def test_solver_result_roundtrip_with_grids(sample_solution, sample_grids, temp_hdf5_file):
    """Test complete save/load roundtrip with grids."""
    U, M = sample_solution
    x_grid, t_grid = sample_grids
    error_history_U = np.array([1e-1, 1e-2])
    error_history_M = np.array([2e-1, 2e-2])

    # Create result
    result = SolverResult(
        U=U,
        M=M,
        iterations=2,
        error_history_U=error_history_U,
        error_history_M=error_history_M,
        solver_name="RoundtripTest",
        converged=False,
    )

    # Save with grids
    result.save_hdf5(temp_hdf5_file, x_grid=x_grid, t_grid=t_grid)

    # Load and verify
    result_loaded = SolverResult.load_hdf5(temp_hdf5_file)

    # Arrays match
    np.testing.assert_array_equal(result.U, result_loaded.U)
    np.testing.assert_array_equal(result.M, result_loaded.M)

    # Metadata includes grids
    assert "x_grid" in result_loaded.metadata
    assert "t_grid" in result_loaded.metadata
    np.testing.assert_array_equal(x_grid, result_loaded.metadata["x_grid"])
    np.testing.assert_array_equal(t_grid, result_loaded.metadata["t_grid"])


# ============================================================================
# Test: Error Handling
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
def test_save_solution_shape_mismatch(sample_solution, sample_metadata, temp_hdf5_file):
    """Test that shape mismatch raises error."""
    U, M = sample_solution
    M_wrong = M[:-1, :]  # Different shape

    with pytest.raises(ValueError, match="U and M must have same shape"):
        save_solution(U, M_wrong, sample_metadata, temp_hdf5_file)


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
def test_load_nonexistent_file():
    """Test loading nonexistent file raises error."""
    with pytest.raises(FileNotFoundError):
        load_solution("nonexistent_file.h5")


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
def test_save_solution_creates_parent_directories(sample_solution, sample_metadata):
    """Test that parent directories are created if needed."""
    U, M = sample_solution

    with tempfile.TemporaryDirectory() as tmpdir:
        nested_path = Path(tmpdir) / "nested" / "directories" / "solution.h5"

        # Should not raise error
        save_solution(U, M, sample_metadata, nested_path)

        # Verify file exists
        assert nested_path.exists()


# ============================================================================
# Test: Metadata Handling
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
def test_metadata_with_none_values(sample_solution, temp_hdf5_file):
    """Test handling of None values in metadata."""
    U, M = sample_solution
    metadata = {
        "converged": True,
        "optional_param": None,
        "another_param": "value",
    }

    save_solution(U, M, metadata, temp_hdf5_file)
    _, _, loaded_metadata = load_solution(temp_hdf5_file)

    assert loaded_metadata["optional_param"] is None
    assert loaded_metadata["another_param"] == "value"


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
def test_metadata_with_array_values(sample_solution, temp_hdf5_file):
    """Test handling of array values in metadata."""
    U, M = sample_solution
    convergence_history = np.array([1e-1, 1e-2, 1e-3, 1e-4])
    metadata = {
        "converged": True,
        "convergence_history": convergence_history,
    }

    save_solution(U, M, metadata, temp_hdf5_file)
    _, _, loaded_metadata = load_solution(temp_hdf5_file)

    np.testing.assert_array_equal(convergence_history, loaded_metadata["convergence_history"])


# ============================================================================
# Test: Import Error Handling
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_hdf5_unavailable_error_message():
    """Test that helpful error is raised when h5py not available."""
    if HDF5_AVAILABLE:
        pytest.skip("h5py is available, cannot test unavailable case")

    from mfg_pde.utils.io import save_solution

    with pytest.raises(ImportError, match="h5py is required"):
        save_solution(np.array([]), np.array([]), {}, "test.h5")
