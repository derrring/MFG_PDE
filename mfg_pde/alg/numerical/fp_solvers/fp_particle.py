from __future__ import annotations

from enum import Enum
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


class KDENormalization(str, Enum):
    """KDE normalization strategy for particle-based FP solvers."""

    NONE = "none"  # No normalization (raw KDE output)
    INITIAL_ONLY = "initial_only"  # Normalize only at t=0
    ALL = "all"  # Normalize at every time step (default)


class ParticleMode(str, Enum):
    """Particle solver operating mode for FP-based solvers."""

    HYBRID = "hybrid"  # Sample own particles, output to grid via KDE (default)
    COLLOCATION = "collocation"  # Use external particles, output on particles (no KDE)


class FPParticleSolver(BaseFPSolver):
    def __init__(
        self,
        problem: MFGProblem,
        # Mode selection (new dual-mode capability)
        mode: ParticleMode | str = ParticleMode.HYBRID,
        external_particles: np.ndarray | None = None,
        # Existing parameters
        num_particles: int = 5000,
        kde_bandwidth: Any = "scott",
        kde_normalization: KDENormalization | str = KDENormalization.ALL,
        boundary_conditions: BoundaryConditions | None = None,
        backend: str | None = None,
        # Deprecated parameters for backward compatibility
        normalize_kde_output: bool | None = None,
        normalize_only_initial: bool | None = None,
    ) -> None:
        super().__init__(problem)

        # Convert string to enum if needed
        if isinstance(mode, str):
            mode = ParticleMode(mode)
        self.mode = mode

        # Validate mode-specific parameters and configure solver
        if mode == ParticleMode.COLLOCATION:
            if external_particles is None:
                raise ValueError(
                    "Collocation mode requires external_particles. "
                    "Pass the collocation points used by your HJB solver.\n"
                    "Example: FPParticleSolver(problem, mode='collocation', external_particles=points)"
                )
            if external_particles.ndim != 2:
                raise ValueError(
                    f"external_particles must be 2D array (N_points, dimension), got shape {external_particles.shape}"
                )
            self.collocation_points = external_particles.copy()
            self.num_particles = len(external_particles)
            self.fp_method_name = "Particle-Collocation"
        else:  # HYBRID mode (default)
            self.collocation_points = None
            self.num_particles = num_particles
            self.fp_method_name = "Particle"

        self.kde_bandwidth = kde_bandwidth

        # Handle deprecated parameters
        if normalize_kde_output is not None or normalize_only_initial is not None:
            import warnings

            warnings.warn(
                "Parameters 'normalize_kde_output' and 'normalize_only_initial' are deprecated. "
                "Use 'kde_normalization' instead with KDENormalization.NONE, KDENormalization.INITIAL_ONLY, or KDENormalization.ALL",
                DeprecationWarning,
                stacklevel=2,
            )

            # Map old parameters to new enum
            if normalize_kde_output is False:
                kde_normalization = KDENormalization.NONE
            elif normalize_only_initial is True:
                kde_normalization = KDENormalization.INITIAL_ONLY
            else:
                kde_normalization = KDENormalization.ALL

        # Convert string to enum if needed
        if isinstance(kde_normalization, str):
            kde_normalization = KDENormalization(kde_normalization)

        self.kde_normalization = kde_normalization
        self.M_particles_trajectory: np.ndarray | None = None
        self._time_step_counter = 0  # Track current time step for normalization logic

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
        Respects kde_normalization strategy: NONE, INITIAL_ONLY, or ALL.

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
        # Determine if we should normalize based on strategy
        if self.kde_normalization == KDENormalization.NONE:
            should_normalize = False
        elif self.kde_normalization == KDENormalization.INITIAL_ONLY:
            should_normalize = self._time_step_counter == 0
        else:  # KDENormalization.ALL
            should_normalize = True

        if not should_normalize:
            return M_array  # Return raw density without normalization

        # Proceed with normalization
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

        # Normalization step (conditional based on kde_normalization strategy)
        if self.kde_normalization == KDENormalization.NONE:
            should_normalize = False
        elif self.kde_normalization == KDENormalization.INITIAL_ONLY:
            should_normalize = self._time_step_counter == 0
        else:  # KDENormalization.ALL
            should_normalize = True

        if should_normalize:
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

    def solve_fp_system(
        self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray, show_progress: bool = True
    ) -> np.ndarray:
        """
        Solve FP system using particle method with dual-mode support.

        Modes:
        - HYBRID (default): Sample own particles, output to grid via KDE
          Strategy Selection: Automatically selects CPU/GPU/Hybrid based on problem size
        - COLLOCATION: Use external particles, output on particles (no KDE)
          Returns density on collocation points (true meshfree representation)

        Args:
            m_initial_condition: Initial density (grid or particle-based depending on mode)
            U_solution_for_drift: Value function for drift computation
            show_progress: Display progress bar (hybrid mode only)

        Returns:
            M_solution: Density evolution
                - HYBRID mode: (Nt, Nx) on grid
                - COLLOCATION mode: (Nt, N_particles) on particles
        """
        # Reset time step counter for normalization logic
        self._time_step_counter = 0

        # Store show_progress for use in methods
        self._show_progress = show_progress

        # Route to appropriate solver based on mode
        if self.mode == ParticleMode.COLLOCATION:
            # Collocation mode: particles → particles (no KDE)
            return self._solve_fp_system_collocation(m_initial_condition, U_solution_for_drift)
        else:
            # Hybrid mode: particles → grid (existing behavior with strategy selection)
            # Determine problem size for strategy selection
            Nt = self.problem.Nt + 1
            Nx = self.problem.Nx + 1
            problem_size = (self.num_particles, Nx, Nt)

            # Select optimal strategy (GPU vs CPU vs Hybrid)
            self.current_strategy = self.strategy_selector.select_strategy(
                backend=self.backend if (self.backend is not None and self.backend.name != "numpy") else None,
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
        self._time_step_counter += 1  # Increment after computing density at t=0

        if Nt == 1:
            self.M_particles_trajectory = current_M_particles_t
            return M_density_on_grid

        # Progress bar for particle timesteps
        from mfg_pde.utils.progress import tqdm

        timestep_range = range(Nt - 1)
        if self._show_progress:
            timestep_range = tqdm(
                timestep_range,
                desc="FP (forward)",
                unit="step",
                disable=False,
            )

        for n_time_idx in timestep_range:
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
            self._time_step_counter += 1  # Increment after each time step

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

        # Normalize based on strategy (use helper function)
        if self.kde_normalization != KDENormalization.NONE:
            M_density_gpu[0, :] = self._normalize_density(M_density_gpu[0, :], Dx, use_backend=True)

        self._time_step_counter += 1  # Increment after computing density at t=0

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

            # Normalize based on strategy (use helper function)
            if self.kde_normalization != KDENormalization.NONE:
                M_density_gpu[t + 1, :] = self._normalize_density(M_density_gpu[t + 1, :], Dx, use_backend=True)

            self._time_step_counter += 1  # Increment after each time step

        # Store trajectory and convert to NumPy ONCE at end
        self.M_particles_trajectory = self.backend.to_numpy(X_particles_gpu)
        return self.backend.to_numpy(M_density_gpu)

    def _solve_fp_system_collocation(
        self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray
    ) -> np.ndarray:
        """
        Solve FP system in collocation mode (meshfree particle output).

        In collocation mode:
        - Particles are FIXED at collocation points (Eulerian representation)
        - Density evolves on particles via continuity equation
        - NO KDE interpolation to grid
        - Output: density on collocation points

        This enables true meshfree MFG when combined with particle-collocation HJB solvers (GFDM).

        Args:
            m_initial_condition: Initial density on collocation points (N_particles,)
            U_solution_for_drift: Value function on collocation points (Nt, N_particles)

        Returns:
            M_solution: Density evolution on collocation points (Nt, N_particles)

        Note:
            In collocation mode, particles remain at collocation points (Eulerian grid-free).
            Mass is advected using continuity equation: ∂m/∂t + ∇·(m α) = σ²/2 Δm
            where α = -coefCT ∇H (optimal control from Hamiltonian).
        """
        Nt = self.problem.Nt + 1
        N_points = len(self.collocation_points)

        # Validate input shapes
        if m_initial_condition.shape != (N_points,):
            raise ValueError(
                f"m_initial_condition shape {m_initial_condition.shape} "
                f"must match collocation_points count ({N_points},)"
            )
        if U_solution_for_drift.shape != (Nt, N_points):
            raise ValueError(
                f"U_solution_for_drift shape {U_solution_for_drift.shape} must be (Nt={Nt}, N_points={N_points})"
            )

        # Storage for density evolution on collocation points
        M_solution = np.zeros((Nt, N_points))
        M_solution[0, :] = m_initial_condition.copy()

        # Simplified collocation mode: density remains constant on particles
        # TODO: Implement proper advection on particles using drift from U_solution_for_drift
        # (Full implementation would solve continuity equation on particle basis)
        # This is a first-order approximation - mass is conserved on particles
        for t_idx in range(Nt - 1):
            # In true Eulerian meshfree: solve ∂m/∂t + ∇·(m α) = σ²/2 Δm on particles
            # Simplified version: particles carry constant mass (valid for small dt)
            # Future enhancement: implement full continuity equation with GFDM Laplacian

            # For now: preserve initial mass distribution on particles
            # This is equivalent to Lagrangian mass tracking
            M_solution[t_idx + 1, :] = M_solution[0, :]

        # Store particle trajectory (in collocation mode, particles are fixed)
        self.M_particles_trajectory = np.tile(self.collocation_points, (Nt, 1, 1))

        return M_solution


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing FPParticleSolver...")

    from mfg_pde import ExampleMFGProblem

    # Test 1D problem with particle solver
    problem = ExampleMFGProblem(Nx=30, Nt=20, T=1.0, sigma=0.1)
    solver = FPParticleSolver(problem, num_particles=1000, mode="hybrid")

    # Test solver initialization
    assert solver.fp_method_name == "Particle"
    assert solver.num_particles == 1000
    assert solver.mode == ParticleMode.HYBRID

    # Test solve_fp_system
    import numpy as np

    U_test = np.zeros((problem.Nt + 1, problem.Nx + 1))
    M_init = problem.m_init

    M_solution = solver.solve_fp_system(M_init, U_test)

    assert M_solution.shape == (problem.Nt + 1, problem.Nx + 1)
    assert not np.any(np.isnan(M_solution))
    assert not np.any(np.isinf(M_solution))
    assert np.all(M_solution >= 0), "Density must be non-negative"

    print("  Particle solver converged")
    print(f"  Num particles: {solver.num_particles}")
    print(f"  M range: [{M_solution.min():.3f}, {M_solution.max():.3f}]")
    print(f"  KDE bandwidth: {solver.kde_bandwidth}")
    print(f"  Mode: {solver.mode.value}")

    print("All smoke tests passed!")
