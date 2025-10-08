from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

try:  # pragma: no cover - optional SciPy dependency
    import scipy.interpolate as _scipy_interpolate
    from scipy.stats import gaussian_kde

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - graceful fallback when SciPy missing
    _scipy_interpolate = None
    gaussian_kde = None
    SCIPY_AVAILABLE = False

from mfg_pde.geometry import BoundaryConditions

from .base_fp import BaseFPSolver

if TYPE_CHECKING:
    from mfg_pde.core.mfg_problem import MFGProblem


class FPParticleSolver(BaseFPSolver):
    def __init__(
        self,
        problem: MFGProblem,
        num_particles: int = 5000,
        kde_bandwidth: Any = "scott",
        normalize_kde_output: bool = True,
        boundary_conditions: BoundaryConditions | None = None,
        backend: str | None = None,
    ) -> None:
        super().__init__(problem)
        self.fp_method_name = "Particle"
        self.num_particles = num_particles
        self.kde_bandwidth = kde_bandwidth
        self.normalize_kde_output = normalize_kde_output  # New flag
        self.M_particles_trajectory: np.ndarray | None = None

        # Initialize backend (defaults to NumPy)
        from mfg_pde.backends import create_backend

        if backend is not None:
            self.backend = create_backend(backend)
        else:
            self.backend = create_backend("numpy")  # NumPy fallback

        # Initialize strategy selector for intelligent pipeline selection
        from mfg_pde.backends.strategies.strategy_selector import StrategySelector

        self.strategy_selector = StrategySelector(enable_profiling=True, verbose=False)
        self.current_strategy = None  # Will be set in solve_fp_system

        # Default to periodic boundaries for backward compatibility
        if boundary_conditions is None:
            self.boundary_conditions = BoundaryConditions(type="periodic")
        else:
            self.boundary_conditions = boundary_conditions

    def _compute_gradient(self, U_array, Dx: float, use_backend: bool = False):
        """
        Compute spatial gradient using finite differences.

        Backend-agnostic helper to reduce code duplication between CPU and GPU pipelines.

        Parameters
        ----------
        U_array : np.ndarray or backend tensor
            Value function at grid points
        Dx : float
            Grid spacing
        use_backend : bool
            If True, use backend array module; if False, use NumPy

        Returns
        -------
        Gradient array (same type as input)
        """
        if use_backend and self.backend is not None:
            xp = self.backend.array_module
        else:
            xp = np

        if Dx > 1e-14:
            return (xp.roll(U_array, -1) - xp.roll(U_array, 1)) / (2 * Dx)
        else:
            return xp.zeros_like(U_array)

    def _normalize_density(self, M_array, Dx: float, use_backend: bool = False):
        """
        Normalize density to unit mass.

        Backend-agnostic helper to reduce code duplication between CPU and GPU pipelines.

        Parameters
        ----------
        M_array : np.ndarray or backend tensor
            Density array
        Dx : float
            Grid spacing
        use_backend : bool
            If True, use backend array module; if False, use NumPy

        Returns
        -------
        Normalized density array (same type as input)
        """
        if use_backend and self.backend is not None:
            xp = self.backend.array_module
            mass = xp.sum(M_array) * Dx if Dx > 1e-14 else xp.sum(M_array)
            # Handle PyTorch tensors
            if hasattr(mass, "item"):
                mass_val = mass.item()
            else:
                mass_val = float(mass)
        else:
            xp = np
            mass_val = float(np.sum(M_array) * Dx) if Dx > 1e-14 else float(np.sum(M_array))

        if mass_val > 1e-9:
            return M_array / mass_val
        else:
            return M_array * 0  # Return zeros

    def _estimate_density_from_particles(self, particles_at_time_t: np.ndarray) -> np.ndarray:
        Nx = self.problem.Nx + 1
        xSpace = self.problem.xSpace
        xmin = self.problem.xmin
        xmax = self.problem.xmax
        Dx = self.problem.Dx

        if self.num_particles == 0 or len(particles_at_time_t) == 0:
            return np.zeros(Nx)

        unique_particles = np.unique(particles_at_time_t)
        if len(unique_particles) < 2 or np.std(particles_at_time_t) < 1e-9 * (xmax - xmin):
            m_density_estimated = np.zeros(Nx)
            if len(particles_at_time_t) > 0:
                mean_pos = np.mean(particles_at_time_t)
                closest_idx = np.argmin(np.abs(xSpace - mean_pos))
                if Dx > 1e-14:
                    m_density_estimated[closest_idx] = 1.0 / Dx
                elif Nx == 1:
                    m_density_estimated[closest_idx] = 1.0

            # Normalization logic will apply below if self.normalize_kde_output is True
        else:
            try:
                # GPU-accelerated KDE if backend available (Track B Phase 1)
                if self.backend is not None:
                    from mfg_pde.alg.numerical.density_estimation import gaussian_kde_gpu

                    # Convert bandwidth parameter to float if needed
                    if isinstance(self.kde_bandwidth, str):
                        from mfg_pde.alg.numerical.density_estimation import adaptive_bandwidth_selection

                        bandwidth_value = adaptive_bandwidth_selection(particles_at_time_t, method=self.kde_bandwidth)
                    else:
                        bandwidth_value = float(self.kde_bandwidth)

                    m_density_estimated = gaussian_kde_gpu(particles_at_time_t, xSpace, bandwidth_value, self.backend)

                    m_density_estimated[xSpace < xmin] = 0
                    m_density_estimated[xSpace > xmax] = 0

                # CPU fallback: scipy.stats.gaussian_kde
                elif SCIPY_AVAILABLE and gaussian_kde is not None:
                    kde = gaussian_kde(particles_at_time_t, bw_method=self.kde_bandwidth)
                    m_density_estimated = kde(xSpace)

                    m_density_estimated[xSpace < xmin] = 0
                    m_density_estimated[xSpace > xmax] = 0
                else:
                    raise RuntimeError("SciPy not available for KDE")

            except Exception:
                # Fallback to peak approximation on error
                m_density_estimated = np.zeros(Nx)
                if len(particles_at_time_t) > 0:
                    mean_pos = np.mean(particles_at_time_t)
                    closest_idx = np.argmin(np.abs(xSpace - mean_pos))
                    if Dx > 1e-14:
                        m_density_estimated[closest_idx] = 1.0 / Dx
                    elif Nx == 1:
                        m_density_estimated[closest_idx] = 1.0

        # Normalization step (now optional)
        if self.normalize_kde_output:
            if Dx > 1e-14:
                current_mass = np.sum(m_density_estimated) * Dx
                if current_mass > 1e-9:
                    return m_density_estimated / current_mass
                else:  # If estimated mass is zero, return zeros
                    return np.zeros(Nx)
            elif Nx == 1:  # Single point domain
                sum_val = np.sum(m_density_estimated)
                return m_density_estimated / sum_val if sum_val > 1e-9 else np.zeros(Nx)
            else:  # Dx is zero but Nx > 1 (should not happen if Nx=1 handled above)
                return np.zeros(Nx)
        else:
            return m_density_estimated  # Return raw KDE output on grid

    def solve_fp_system(self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray) -> np.ndarray:
        """
        Solve FP system using particle method with intelligent strategy selection.

        Strategy Selection (Track B Phase 2.2 - Intelligent Dispatch):
        - Automatically selects optimal pipeline based on:
          * Backend capabilities (GPU acceleration availability)
          * Problem size (num_particles, grid_size, time_steps)
          * Device characteristics (MPS overhead, CUDA efficiency)
        - Strategies: CPU-only, GPU-accelerated, Hybrid (for medium-sized MPS problems)

        This replaces hard-coded if-else dispatch with intelligent cost-based selection.
        """
        # Determine problem size for strategy selection
        Nt = self.problem.Nt + 1
        Nx = self.problem.Nx + 1
        problem_size = (self.num_particles, Nx, Nt)

        # Select optimal strategy (GPU vs CPU vs Hybrid)
        self.current_strategy = self.strategy_selector.select_strategy(
            backend=self.backend if self.backend.name != "numpy" else None,
            problem_size=problem_size,
            strategy_hint="auto",  # Can be overridden to "cpu", "gpu", "hybrid"
        )

        # Execute using selected strategy's pipeline
        if self.current_strategy.name == "cpu":
            return self._solve_fp_system_cpu(m_initial_condition, U_solution_for_drift)
        else:
            # GPU or Hybrid strategy (both use GPU pipeline)
            return self._solve_fp_system_gpu(m_initial_condition, U_solution_for_drift)

    def _solve_fp_system_cpu(self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray) -> np.ndarray:
        """CPU pipeline - existing NumPy implementation."""
        # print(f"****** Solving FP ({self.fp_method_name}) with {self.num_particles} particles ******")
        Nx = self.problem.Nx + 1
        Nt = self.problem.Nt + 1
        Dx = self.problem.Dx
        Dt = self.problem.Dt
        sigma_sde = self.problem.sigma
        coefCT = self.problem.coefCT
        x_grid = self.problem.xSpace
        xmin = self.problem.xmin
        Lx = self.problem.Lx

        if Nt == 0:
            return np.zeros((0, Nx))

        M_density_on_grid = np.zeros((Nt, Nx))
        current_M_particles_t = np.zeros((Nt, self.num_particles))

        if Dx > 1e-14 and np.sum(m_initial_condition * Dx) > 1e-9:
            m0_probs_unnormalized = m_initial_condition * Dx
            m0_probs = m0_probs_unnormalized / np.sum(m0_probs_unnormalized)
            try:
                initial_particle_positions = np.random.choice(x_grid, size=self.num_particles, p=m0_probs, replace=True)
            except ValueError:
                initial_particle_positions = np.random.uniform(xmin, xmin + Lx, self.num_particles)
        else:
            initial_particle_positions = (
                np.random.uniform(xmin, xmin + Lx, self.num_particles)
                if Lx > 1e-14
                else np.full(self.num_particles, xmin)
            )

        current_M_particles_t[0, :] = initial_particle_positions
        M_density_on_grid[0, :] = self._estimate_density_from_particles(current_M_particles_t[0, :])

        if Nt == 1:
            self.M_particles_trajectory = current_M_particles_t
            return M_density_on_grid

        for n_time_idx in range(Nt - 1):
            U_at_tn = U_solution_for_drift[n_time_idx, :]

            # Use helper function for gradient computation
            if Nx > 1:
                dUdx_grid = self._compute_gradient(U_at_tn, Dx, use_backend=False)
            else:
                dUdx_grid = np.zeros(Nx)

            if Nx > 1:
                if SCIPY_AVAILABLE and _scipy_interpolate is not None:
                    try:
                        interp_func_dUdx = _scipy_interpolate.interp1d(
                            x_grid, dUdx_grid, kind="linear", fill_value="extrapolate"
                        )
                        dUdx_at_particles = interp_func_dUdx(current_M_particles_t[n_time_idx, :])
                    except ValueError:
                        dUdx_at_particles = np.zeros(self.num_particles)
                else:
                    dUdx_at_particles = np.interp(
                        current_M_particles_t[n_time_idx, :], x_grid, dUdx_grid, left=dUdx_grid[0], right=dUdx_grid[-1]
                    )
            else:
                dUdx_at_particles = np.zeros(self.num_particles)

            alpha_optimal_at_particles = -coefCT * dUdx_at_particles

            dW = np.random.normal(0.0, np.sqrt(Dt), self.num_particles) if Dt > 1e-14 else np.zeros(self.num_particles)

            current_M_particles_t[n_time_idx + 1, :] = (
                current_M_particles_t[n_time_idx, :] + alpha_optimal_at_particles * Dt + sigma_sde * dW
            )

            # Apply boundary conditions to particles
            if self.boundary_conditions.type == "periodic" and Lx > 1e-14:
                # Periodic boundaries: wrap around
                current_M_particles_t[n_time_idx + 1, :] = xmin + (current_M_particles_t[n_time_idx + 1, :] - xmin) % Lx
            elif self.boundary_conditions.type == "no_flux":
                # Reflecting boundaries: bounce particles back
                xmax = xmin + Lx
                particles = current_M_particles_t[n_time_idx + 1, :]

                # Reflect particles that go beyond left boundary
                left_violations = particles < xmin
                particles[left_violations] = 2 * xmin - particles[left_violations]

                # Reflect particles that go beyond right boundary
                right_violations = particles > xmax
                particles[right_violations] = 2 * xmax - particles[right_violations]

                current_M_particles_t[n_time_idx + 1, :] = particles

            M_density_on_grid[n_time_idx + 1, :] = self._estimate_density_from_particles(
                current_M_particles_t[n_time_idx + 1, :]
            )

        self.M_particles_trajectory = current_M_particles_t
        return M_density_on_grid

    def _solve_fp_system_gpu(self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray) -> np.ndarray:
        """
        GPU pipeline - full particle evolution on GPU.

        Track B Phase 2.1: Full GPU acceleration including internal KDE.
        Eliminates all GPU↔CPU transfers during evolution loop.

        Expected speedup:
            - Apple Silicon MPS: 1.5-2x for N≥50k particles
            - NVIDIA CUDA: 3-5x (estimated, not tested)
            - See docs/development/TRACK_B_GPU_ACCELERATION_COMPLETE.md
        """
        from mfg_pde.alg.numerical.density_estimation import gaussian_kde_gpu_internal
        from mfg_pde.alg.numerical.particle_utils import (
            apply_boundary_conditions_gpu,
            interpolate_1d_gpu,
            sample_from_density_gpu,
        )

        # Problem parameters
        Nx = self.problem.Nx + 1
        Nt = self.problem.Nt + 1
        Dx = self.problem.Dx
        Dt = self.problem.Dt
        sigma_sde = self.problem.sigma
        coefCT = self.problem.coefCT
        x_grid = self.problem.xSpace
        xmin = self.problem.xmin
        xmax = xmin + self.problem.Lx
        Lx = self.problem.Lx

        if Nt == 0:
            return np.zeros((0, Nx))

        # Convert inputs to GPU ONCE at start
        x_grid_gpu = self.backend.from_numpy(x_grid)
        U_drift_gpu = self.backend.from_numpy(U_solution_for_drift)

        # Allocate arrays on GPU
        X_particles_gpu = self.backend.zeros((Nt, self.num_particles))
        M_density_gpu = self.backend.zeros((Nt, Nx))

        # Sample initial particles on GPU
        m_initial_gpu = self.backend.from_numpy(m_initial_condition)
        try:
            X_particles_gpu[0, :] = sample_from_density_gpu(
                m_initial_gpu, x_grid_gpu, self.num_particles, self.backend, seed=None
            )
        except Exception:
            # Fallback: uniform sampling
            X_init_np = np.random.uniform(xmin, xmax, self.num_particles)
            X_particles_gpu[0, :] = self.backend.from_numpy(X_init_np)

        # Compute bandwidth for KDE (do this once on CPU)
        # Convert bandwidth parameter to absolute bandwidth value
        if isinstance(self.kde_bandwidth, str):
            from mfg_pde.alg.numerical.density_estimation import adaptive_bandwidth_selection

            # Need numpy array for bandwidth calculation
            X_init_np = self.backend.to_numpy(X_particles_gpu[0, :])
            bandwidth_absolute = adaptive_bandwidth_selection(X_init_np, method=self.kde_bandwidth)
        else:
            # User provided factor - compute factor * std(particles)
            X_init_np = self.backend.to_numpy(X_particles_gpu[0, :])
            data_std = np.std(X_init_np, ddof=1)
            bandwidth_absolute = float(self.kde_bandwidth) * data_std

        # Estimate initial density using internal GPU KDE (Phase 2.1)
        M_density_gpu[0, :] = gaussian_kde_gpu_internal(
            X_particles_gpu[0, :], x_grid_gpu, bandwidth_absolute, self.backend
        )

        # Normalize if required (use helper function)
        if self.normalize_kde_output:
            M_density_gpu[0, :] = self._normalize_density(M_density_gpu[0, :], Dx, use_backend=True)

        if Nt == 1:
            self.M_particles_trajectory = self.backend.to_numpy(X_particles_gpu)
            return self.backend.to_numpy(M_density_gpu)

        # Main evolution loop - ALL GPU
        for t in range(Nt - 1):
            U_t_gpu = U_drift_gpu[t, :]

            # Compute gradient on grid (use helper function)
            if Nx > 1:
                dUdx_gpu = self._compute_gradient(U_t_gpu, Dx, use_backend=True)
            else:
                dUdx_gpu = self.backend.zeros((Nx,))

            # Interpolate gradient to particle positions (GPU)
            if Nx > 1:
                dUdx_particles_gpu = interpolate_1d_gpu(X_particles_gpu[t, :], x_grid_gpu, dUdx_gpu, self.backend)
            else:
                dUdx_particles_gpu = self.backend.zeros((self.num_particles,))

            # Compute drift (GPU)
            drift_gpu = -coefCT * dUdx_particles_gpu

            # Random noise (GPU native RNG)
            if Dt > 1e-14:
                # Generate on CPU and transfer (safest approach for now)
                noise_scale = sigma_sde * np.sqrt(Dt)
                noise_np = np.random.randn(self.num_particles) * noise_scale
                noise_gpu = self.backend.from_numpy(noise_np)
            else:
                noise_gpu = self.backend.zeros((self.num_particles,))

            # Euler-Maruyama update (GPU)
            X_particles_gpu[t + 1, :] = X_particles_gpu[t, :] + drift_gpu * Dt + noise_gpu

            # Apply boundary conditions (GPU)
            if Lx > 1e-14:
                X_particles_gpu[t + 1, :] = apply_boundary_conditions_gpu(
                    X_particles_gpu[t + 1, :], xmin, xmax, self.boundary_conditions.type, self.backend
                )

            # Estimate density using internal GPU KDE (Phase 2.1 - no transfers!)
            M_density_gpu[t + 1, :] = gaussian_kde_gpu_internal(
                X_particles_gpu[t + 1, :], x_grid_gpu, bandwidth_absolute, self.backend
            )

            # Normalize if required (use helper function)
            if self.normalize_kde_output:
                M_density_gpu[t + 1, :] = self._normalize_density(M_density_gpu[t + 1, :], Dx, use_backend=True)

        # Store trajectory and convert to NumPy ONCE at end
        self.M_particles_trajectory = self.backend.to_numpy(X_particles_gpu)
        return self.backend.to_numpy(M_density_gpu)
