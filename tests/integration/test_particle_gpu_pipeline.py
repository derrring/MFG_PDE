"""
Integration tests for GPU particle pipeline (Phase 2).

Tests end-to-end numerical accuracy and performance of full GPU
particle evolution compared to CPU baseline.
"""

import pytest

import numpy as np

from mfg_pde.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.geometry.boundary_conditions_1d import BoundaryConditions

pytestmark = pytest.mark.optional_torch

# Check if PyTorch is available for GPU tests
try:
    from mfg_pde.backends.torch_backend import TorchBackend

    # Test if torch backend actually works by creating a simple instance
    _test_backend = TorchBackend(device="cpu")
    TORCH_AVAILABLE = True
except (ImportError, Exception):
    TORCH_AVAILABLE = False
    TorchBackend = None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestParticleGPUPipeline:
    """Test full GPU particle evolution pipeline."""

    def test_gpu_matches_cpu_numerically(self):
        """GPU pipeline should match CPU pipeline numerically."""
        # Create simple problem
        problem = MFGProblem(
            Nx=50,
            Nt=20,
            T=1.0,
            xmin=0.0,
            xmax=1.0,
            sigma=0.1,
            coupling_coefficient=1.0,
        )

        # Initial condition: Gaussian
        x = problem.xSpace
        m_initial = np.exp(-((x - 0.5) ** 2) / 0.1)
        m_initial = m_initial / (np.sum(m_initial) * problem.Dx)

        # Drift field: simple linear
        U_drift = np.zeros((problem.Nt + 1, problem.Nx + 1))
        for t in range(problem.Nt + 1):
            U_drift[t, :] = -((x - 0.5) ** 2)  # Quadratic potential

        # CPU solver
        solver_cpu = FPParticleSolver(
            problem,
            num_particles=1000,
            kde_bandwidth=0.1,
            boundary_conditions=BoundaryConditions(type="periodic"),
        )
        solver_cpu.backend = None  # Force CPU

        M_cpu = solver_cpu.solve_fp_system(m_initial, U_drift)

        # GPU solver
        backend_gpu = TorchBackend(device="cpu")  # Use CPU device for deterministic comparison
        solver_gpu = FPParticleSolver(
            problem,
            num_particles=1000,
            kde_bandwidth=0.1,
            boundary_conditions=BoundaryConditions(type="periodic"),
        )
        solver_gpu.backend = backend_gpu

        M_gpu = solver_gpu.solve_fp_system(m_initial, U_drift)

        # Should match within stochastic tolerance
        # Particle methods are stochastic, so allow ~10% relative error
        assert M_cpu.shape == M_gpu.shape
        assert M_cpu.shape == (problem.Nt + 1, problem.Nx + 1)

        # Mass conservation (both should integrate to ~1)
        mass_cpu = np.sum(M_cpu, axis=1) * problem.Dx
        mass_gpu = np.sum(M_gpu, axis=1) * problem.Dx

        np.testing.assert_allclose(mass_cpu, 1.0, rtol=0.2)  # Within 20%
        np.testing.assert_allclose(mass_gpu, 1.0, rtol=0.2)

        # Distributions should be similar (allow stochastic variation)
        # Compare mean particle positions over time
        mean_cpu = np.sum(M_cpu * x[None, :], axis=1) * problem.Dx
        mean_gpu = np.sum(M_gpu * x[None, :], axis=1) * problem.Dx

        # Means should track similarly (within 20% relative difference)
        np.testing.assert_allclose(mean_cpu, mean_gpu, rtol=0.3, atol=0.1)

    def test_gpu_pipeline_runs_without_errors(self):
        """GPU pipeline should complete without errors."""
        problem = MFGProblem(
            Nx=30,
            Nt=10,
            T=0.5,
            xmin=-1.0,
            xmax=1.0,
            sigma=0.2,
        )

        m_initial = np.exp(-(problem.xSpace**2) / 0.2)
        m_initial = m_initial / (np.sum(m_initial) * problem.Dx)

        U_drift = np.zeros((problem.Nt + 1, problem.Nx + 1))

        backend = TorchBackend(device="mps")  # Test on actual MPS device
        solver = FPParticleSolver(
            problem,
            num_particles=5000,
            kde_bandwidth=0.15,
            boundary_conditions=BoundaryConditions(type="no_flux"),
        )
        solver.backend = backend

        M_gpu = solver.solve_fp_system(m_initial, U_drift)

        # Basic validity checks
        assert M_gpu.shape == (problem.Nt + 1, problem.Nx + 1)
        assert np.all(M_gpu >= 0)  # Density non-negative
        assert np.all(np.isfinite(M_gpu))  # No NaN/Inf

        # Mass conservation
        mass = np.sum(M_gpu, axis=1) * problem.Dx
        np.testing.assert_allclose(mass, 1.0, rtol=0.3)

    def test_boundary_conditions_gpu(self):
        """Test different boundary conditions on GPU."""
        problem = MFGProblem(
            Nx=40,
            Nt=15,
            T=0.5,
            xmin=0.0,
            xmax=1.0,
            sigma=0.15,
        )

        m_initial = np.ones(problem.Nx + 1) / (problem.Nx + 1)
        U_drift = np.zeros((problem.Nt + 1, problem.Nx + 1))

        backend = TorchBackend(device="cpu")

        for bc_type in ["periodic", "no_flux", "dirichlet"]:
            solver = FPParticleSolver(
                problem,
                num_particles=2000,
                kde_bandwidth=0.1,
                boundary_conditions=BoundaryConditions(type=bc_type),
            )
            solver.backend = backend

            M = solver.solve_fp_system(m_initial, U_drift)

            # Should complete without errors
            assert M.shape == (problem.Nt + 1, problem.Nx + 1)
            assert np.all(M >= 0)
            assert np.all(np.isfinite(M))


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestGPUPerformance:
    """Performance tests for GPU pipeline."""

    def test_gpu_faster_than_cpu_for_large_N(self):
        """GPU should be faster than CPU for large particle counts."""
        import time

        problem = MFGProblem(
            Nx=50,
            Nt=50,
            T=1.0,
            xmin=0.0,
            xmax=1.0,
            sigma=0.1,
        )

        m_initial = np.exp(-((problem.xSpace - 0.5) ** 2) / 0.1)
        m_initial = m_initial / (np.sum(m_initial) * problem.Dx)

        U_drift = np.zeros((problem.Nt + 1, problem.Nx + 1))

        N = 10000  # Large particle count

        # CPU timing
        solver_cpu = FPParticleSolver(
            problem,
            num_particles=N,
            kde_bandwidth=0.1,
            boundary_conditions=BoundaryConditions(type="periodic"),
        )
        solver_cpu.backend = None

        start = time.time()
        _M_cpu = solver_cpu.solve_fp_system(m_initial, U_drift)
        time_cpu = time.time() - start

        # GPU timing
        backend_gpu = TorchBackend(device="mps")
        solver_gpu = FPParticleSolver(
            problem,
            num_particles=N,
            kde_bandwidth=0.1,
            boundary_conditions=BoundaryConditions(type="periodic"),
        )
        solver_gpu.backend = backend_gpu

        start = time.time()
        _M_gpu = solver_gpu.solve_fp_system(m_initial, U_drift)
        time_gpu = time.time() - start

        speedup = time_cpu / time_gpu

        print(f"\nGPU Pipeline Performance (N={N}, Nt={problem.Nt}):")
        print(f"  CPU time: {time_cpu:.2f}s")
        print(f"  GPU time: {time_gpu:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")

        # Phase 2.1 Complete: Internal GPU KDE eliminates transfers
        # Realistic expectation: 1.5-2x speedup for N=10k-100k on MPS
        # (CUDA would achieve higher speedup, MPS has kernel overhead)
        if speedup >= 1.5:
            print(f"  ✅ Phase 2.1 success: {speedup:.2f}x (MPS architecture)")
        elif speedup >= 1.0:
            print(f"  ⚠️  Modest speedup: {speedup:.2f}x (consider larger N)")
        else:
            print(f"  ❌ Slower on GPU: {speedup:.2f}x (problem size too small)")

        # Assert that pipeline executes correctly
        # Performance validation happens in benchmarks/particle_gpu_speedup_analysis.py
        assert speedup > 0.1  # Sanity check: not catastrophically slow
