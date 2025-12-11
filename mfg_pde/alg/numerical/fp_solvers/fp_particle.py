from __future__ import annotations

import warnings
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.geometry import BoundaryConditions

try:  # pragma: no cover - optional SciPy dependency
    import scipy.interpolate as _scipy_interpolate
    from scipy.stats import gaussian_kde

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - graceful fallback when SciPy missing
    _scipy_interpolate = None
    gaussian_kde = None
    SCIPY_AVAILABLE = False

from mfg_pde.geometry.boundary.conditions import periodic_bc

from .base_fp import BaseFPSolver


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

            # WARN: Incomplete implementation
            import warnings

            warnings.warn(
                "Collocation mode has an INCOMPLETE IMPLEMENTATION. "
                "Density is frozen at the initial condition instead of evolving via the continuity equation. "
                "This is physically incorrect for non-equilibrium problems. "
                "Use mode='default' (KDE-based) for correct density evolution. "
                "See Issue #240 for implementation status. "
                "Expected completion: v1.0.0",
                UserWarning,
                stacklevel=2,
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

        # Boundary condition resolution hierarchy:
        # 1. Explicit boundary_conditions parameter (highest priority)
        # 2. Grid geometry boundary handler (if available)
        # 3. Default periodic BC (fallback for backward compatibility)
        if boundary_conditions is not None:
            self.boundary_conditions = boundary_conditions
        elif hasattr(problem, "geometry") and hasattr(problem.geometry, "get_boundary_handler"):
            # Try to get BC from grid geometry (Phase 2 integration)
            try:
                self.boundary_conditions = problem.geometry.get_boundary_handler()
            except Exception:
                # Fallback if geometry BC retrieval fails
                self.boundary_conditions = periodic_bc(dimension=1)
        else:
            # Default to periodic boundaries for backward compatibility
            self.boundary_conditions = periodic_bc(dimension=1)

    def _get_grid_params(self) -> dict:
        """
        Extract grid parameters from geometry (preferred) or legacy problem API.

        Returns dict with nD-aware parameters:
            - dimension: int, spatial dimension (1, 2, 3, ...)
            - grid_shape: tuple, shape of spatial grid (Nx+1,) or (Nx+1, Ny+1, ...)
            - spacings: list[float], grid spacing per dimension [Dx, Dy, ...]
            - bounds: list[tuple], bounds per dimension [(xmin, xmax), (ymin, ymax), ...]
            - coordinates: list[np.ndarray], 1D coordinate arrays per dimension
            - total_points: int, total number of grid points
            - Nt, Dt, sigma, coupling_coefficient: time/physics parameters

        For 1D backward compatibility, also includes:
            - Nx, Dx, xmin, xmax, Lx, xSpace (aliased from nD params)
        """
        # Try geometry-first API
        if hasattr(self.problem, "geometry") and self.problem.geometry is not None:
            geom = self.problem.geometry
            grid_shape = tuple(geom.get_grid_shape())
            dimension = len(grid_shape)

            # Get spacing per dimension
            spacing = geom.get_grid_spacing()
            spacings = list(spacing) if spacing else [0.0] * dimension

            # Get bounds per dimension
            if hasattr(geom, "bounds") and geom.bounds is not None:
                bounds = list(geom.bounds)
            elif hasattr(geom, "xmin") and hasattr(geom, "xmax"):
                # Legacy 1D geometry
                bounds = [(geom.xmin, geom.xmax)]
            elif hasattr(geom, "coordinates") and len(geom.coordinates) > 0:
                bounds = [(coords[0], coords[-1]) for coords in geom.coordinates]
            else:
                bounds = [(0.0, 1.0)] * dimension

            # Get coordinate arrays per dimension
            if hasattr(geom, "coordinates") and len(geom.coordinates) > 0:
                coordinates = [np.array(c) for c in geom.coordinates]
            else:
                coordinates = [np.linspace(bounds[d][0], bounds[d][1], grid_shape[d]) for d in range(dimension)]

        # Fallback to legacy 1D API
        elif self.problem.Nx is not None:
            dimension = 1
            Nx = self.problem.Nx + 1
            grid_shape = (Nx,)
            Dx = self.problem.dx if self.problem.dx is not None else 0.0
            spacings = [Dx]
            xmin = self.problem.xmin if self.problem.xmin is not None else 0.0
            xmax = self.problem.xmax if self.problem.xmax is not None else 1.0
            bounds = [(xmin, xmax)]
            xSpace = self.problem.xSpace if self.problem.xSpace is not None else np.linspace(xmin, xmax, Nx)
            coordinates = [xSpace]
        else:
            raise ValueError(
                "FPParticleSolver requires either a geometry object or legacy problem.Nx. "
                "Create MFGProblem with geometry=TensorProductGrid(...) or with Nx=... parameter."
            )

        # Compute derived quantities
        total_points = int(np.prod(grid_shape))
        domain_lengths = [b[1] - b[0] for b in bounds]

        # Time parameters (always from problem)
        # n_time_points = problem.Nt + 1 (number of time knots including t=0 and t=T)
        # problem.Nt = number of time intervals
        n_time_points = self.problem.Nt + 1
        Dt = (
            self.problem.dt
            if self.problem.dt is not None
            else (self.problem.T / self.problem.Nt if self.problem.Nt > 0 else 0.0)
        )
        sigma = self.problem.sigma if self.problem.sigma is not None else 0.1
        coupling_coefficient = (
            self.problem.coupling_coefficient if self.problem.coupling_coefficient is not None else 1.0
        )

        result = {
            # nD parameters
            "dimension": dimension,
            "grid_shape": grid_shape,
            "spacings": spacings,
            "bounds": bounds,
            "coordinates": coordinates,
            "total_points": total_points,
            "domain_lengths": domain_lengths,
            # Time/physics parameters
            "n_time_points": n_time_points,  # Nt + 1 (number of knots)
            "Nt": n_time_points,  # Backward compatible alias (deprecated)
            "Dt": Dt,
            "sigma": sigma,
            "coupling_coefficient": coupling_coefficient,
        }

        # 1D backward compatibility aliases
        if dimension == 1:
            result["Nx"] = grid_shape[0]
            result["Dx"] = spacings[0]
            result["xmin"] = bounds[0][0]
            result["xmax"] = bounds[0][1]
            result["Lx"] = domain_lengths[0]
            result["xSpace"] = coordinates[0]

        return result

    def _compute_gradient(self, U_array, Dx: float, use_backend: bool = False):
        """
        Compute spatial gradient using finite differences (1D backward-compatible).

        Backend-agnostic helper to reduce code duplication between CPU and GPU pipelines.

        Parameters
        ----------
        U_array : np.ndarray or backend tensor
            Value function at grid points (1D array)
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

    def _compute_gradient_nd(
        self,
        U_array: np.ndarray,
        spacings: list[float],
        use_backend: bool = False,
    ) -> list:
        """
        Compute spatial gradient in each dimension using central differences.

        Parameters
        ----------
        U_array : np.ndarray or backend tensor
            Value function at grid points, shape (N1, N2, ..., Nd)
        spacings : list[float]
            Grid spacing per dimension [Dx, Dy, ...]
        use_backend : bool
            If True, use backend array module; if False, use NumPy

        Returns
        -------
        list of gradient arrays, one per dimension
            Each gradient[d] has the same shape as U_array
        """
        if use_backend and self.backend is not None:
            xp = self.backend.array_module
        else:
            xp = np

        dimension = len(spacings)
        gradients = []

        for d in range(dimension):
            if spacings[d] > 1e-14:
                # Central difference along axis d
                grad_d = (xp.roll(U_array, -1, axis=d) - xp.roll(U_array, 1, axis=d)) / (2 * spacings[d])
            else:
                grad_d = xp.zeros_like(U_array)
            gradients.append(grad_d)

        return gradients

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

    # =========================================================================
    # nD Helper Methods
    # =========================================================================

    def _sample_particles_from_density_nd(
        self,
        M_initial: np.ndarray,
        coordinates: list[np.ndarray],
        num_particles: int,
    ) -> np.ndarray:
        """
        Sample particles from nD density distribution.

        Parameters
        ----------
        M_initial : np.ndarray
            Initial density on grid, shape (N1, N2, ..., Nd)
        coordinates : list[np.ndarray]
            List of 1D coordinate arrays per dimension
        num_particles : int
            Number of particles to sample

        Returns
        -------
        particles : np.ndarray
            Particle positions, shape (num_particles, dimension)
        """
        dimension = len(coordinates)
        grid_shape = tuple(len(c) for c in coordinates)

        # Flatten density and normalize to probability
        M_flat = M_initial.ravel()
        total_mass = np.sum(M_flat)

        if total_mass < 1e-14:
            # Uniform fallback if density is zero (vectorized)
            bounds = np.array([[c[0], c[-1]] for c in coordinates])  # shape: (d, 2)
            particles = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(num_particles, dimension))
            return particles

        probs = M_flat / total_mass

        # Sample flat indices according to probability
        try:
            flat_indices = np.random.choice(len(M_flat), size=num_particles, p=probs, replace=True)
        except ValueError:
            # Fallback to uniform if probability is degenerate (vectorized)
            bounds = np.array([[c[0], c[-1]] for c in coordinates])  # shape: (d, 2)
            particles = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(num_particles, dimension))
            return particles

        # Convert flat indices to multi-indices
        multi_indices = np.unravel_index(flat_indices, grid_shape)

        # Get coordinates with sub-grid jitter for smoothness (vectorized)
        # Stack multi-indices into array for vectorized coordinate lookup
        particles = np.column_stack([coordinates[d][multi_indices[d]] for d in range(dimension)])

        # Compute grid spacings and add uniform jitter (vectorized)
        spacings = np.array([c[1] - c[0] if len(c) > 1 else 0.0 for c in coordinates])
        jitter = np.random.uniform(-0.5, 0.5, size=(num_particles, dimension)) * spacings
        particles += jitter

        return particles

    def _generate_brownian_increment_nd(
        self,
        num_particles: int,
        dimension: int,
        Dt: float,
        sigma: float,
    ) -> np.ndarray:
        """
        Generate d-dimensional Brownian increment for SDE evolution.

        Parameters
        ----------
        num_particles : int
            Number of particles
        dimension : int
            Spatial dimension
        Dt : float
            Time step size
        sigma : float
            Diffusion coefficient

        Returns
        -------
        dW : np.ndarray
            Brownian increments, shape (num_particles, dimension)
        """
        if Dt < 1e-14:
            return np.zeros((num_particles, dimension))

        # Independent Brownian motion in each dimension
        # dX = sigma * dW where dW ~ N(0, sqrt(dt))
        return sigma * np.random.normal(0, np.sqrt(Dt), (num_particles, dimension))

    def _apply_boundary_conditions_nd(
        self,
        particles: np.ndarray,
        bounds: list[tuple[float, float]],
        bc_type: str,
    ) -> np.ndarray:
        """
        Apply boundary conditions per dimension.

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape (num_particles, dimension)
        bounds : list[tuple[float, float]]
            Bounds per dimension [(xmin, xmax), (ymin, ymax), ...]
        bc_type : str
            Boundary condition type: "periodic" or "no_flux"

        Returns
        -------
        particles : np.ndarray
            Updated particle positions (modified in place)
        """
        dimension = particles.shape[1] if particles.ndim > 1 else 1

        for d in range(dimension):
            xmin, xmax = bounds[d]
            Lx = xmax - xmin

            if Lx < 1e-14:
                continue  # Skip degenerate dimension

            if bc_type == "periodic":
                # Wrap around boundaries
                particles[:, d] = xmin + (particles[:, d] - xmin) % Lx

            elif bc_type == "no_flux":
                # Reflecting boundaries: use modular reflection for arbitrary displacement
                # This handles particles that travel multiple domain widths in one step
                # Uses "fold" reflection: position bounces back and forth within domain
                shifted = particles[:, d] - xmin
                period = 2 * Lx
                # Position within one period [0, 2*Lx)
                pos_in_period = shifted % period
                # If in second half of period, reflect back
                in_second_half = pos_in_period > Lx
                pos_in_period[in_second_half] = period - pos_in_period[in_second_half]
                # Shift back to original domain
                particles[:, d] = xmin + pos_in_period

        return particles

    def _create_nd_interpolator(
        self,
        coordinates: list[np.ndarray],
        values: np.ndarray,
    ):
        """
        Create nD interpolator for grid values.

        Parameters
        ----------
        coordinates : list[np.ndarray]
            List of 1D coordinate arrays per dimension
        values : np.ndarray
            Values on grid, shape (N1, N2, ..., Nd)

        Returns
        -------
        interpolator : RegularGridInterpolator
            Scipy interpolator for nD data
        """
        if not SCIPY_AVAILABLE or _scipy_interpolate is None:
            raise RuntimeError("SciPy required for nD interpolation")

        return _scipy_interpolate.RegularGridInterpolator(
            tuple(coordinates),
            values,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

    def _estimate_density_from_particles_nd(
        self,
        particles: np.ndarray,
        coordinates: list[np.ndarray],
        bounds: list[tuple[float, float]],
    ) -> np.ndarray:
        """
        Estimate density from particles using KDE on nD grid.

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape (num_particles, dimension)
        coordinates : list[np.ndarray]
            List of 1D coordinate arrays per dimension
        bounds : list[tuple[float, float]]
            Bounds per dimension

        Returns
        -------
        density : np.ndarray
            Density on grid, shape (N1, N2, ..., Nd)
        """
        dimension = len(coordinates)
        grid_shape = tuple(len(c) for c in coordinates)

        if self.num_particles == 0 or len(particles) == 0:
            return np.zeros(grid_shape)

        # Check for degenerate particle distribution
        if len(np.unique(particles, axis=0)) < 2:
            # All particles at same location - delta function
            density = np.zeros(grid_shape)
            mean_pos = np.mean(particles, axis=0)
            # Find closest grid point
            indices = []
            for d in range(dimension):
                idx = np.argmin(np.abs(coordinates[d] - mean_pos[d]))
                indices.append(idx)
            density[tuple(indices)] = self.num_particles
            return density

        try:
            # scipy.stats.gaussian_kde handles nD
            if SCIPY_AVAILABLE and gaussian_kde is not None:
                # KDE expects shape (d, N) not (N, d)
                kde = gaussian_kde(particles.T, bw_method=self.kde_bandwidth)

                # Create evaluation grid
                mesh_grids = np.meshgrid(*coordinates, indexing="ij")
                eval_points = np.vstack([g.ravel() for g in mesh_grids])

                # Evaluate KDE
                density_flat = kde(eval_points)
                density = density_flat.reshape(grid_shape)

                # Zero out points outside bounds
                for d in range(dimension):
                    mask = (mesh_grids[d] < bounds[d][0]) | (mesh_grids[d] > bounds[d][1])
                    density[mask] = 0.0

                return density

            else:
                raise RuntimeError("SciPy not available for KDE")

        except Exception as e:
            warnings.warn(f"KDE failed in nD: {e}. Returning histogram estimate.")
            # Fallback to histogram
            density, _ = np.histogramdd(
                particles,
                bins=[len(c) for c in coordinates],
                range=bounds,
                density=True,
            )
            return density

    def _estimate_density_from_particles(self, particles_at_time_t: np.ndarray) -> np.ndarray:
        # Use geometry-aware parameter extraction
        params = self._get_grid_params()
        Nx = params["Nx"]
        xSpace = params["xSpace"]
        xmin = params["xmin"]
        xmax = params["xmax"]
        Dx = params["Dx"]

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

            except Exception as e:
                error_msg = (
                    f"KDE density estimation failed in FPParticleSolver: {e}\n"
                    f"Number of particles: {len(particles_at_time_t)}\n"
                    f"Grid size: {Nx}\n"
                    f"Bandwidth: {self.kde_bandwidth}\n"
                    "Possible causes:\n"
                    "  1. Too few particles for reliable KDE (need at least 10-20)\n"
                    "  2. Bandwidth selection failed (try fixed bandwidth like 0.1)\n"
                    "  3. Particles outside domain bounds\n"
                    "  4. GPU/SciPy library issues\n"
                    "Suggestions:\n"
                    "  - Increase number of particles (Np > 100 recommended)\n"
                    "  - Use fixed bandwidth: kde_bandwidth=0.1\n"
                    "  - Check particle initialization and drift/diffusion"
                )
                raise RuntimeError(error_msg) from e

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
        self,
        M_initial: np.ndarray | None = None,
        drift_field: np.ndarray | Callable | None = None,
        diffusion_field: float | np.ndarray | Callable | None = None,
        show_progress: bool = True,
        # Deprecated parameter name for backward compatibility
        m_initial_condition: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Solve FP system using particle method with unified API.

        Modes:
        - HYBRID (default): Sample own particles, output to grid via KDE
          Strategy Selection: Automatically selects CPU/GPU/Hybrid based on problem size
        - COLLOCATION: Use external particles, output on particles (no KDE)
          Returns density on collocation points (true meshfree representation)

        Args:
            M_initial: Initial density m₀(x) (grid or particle-based depending on mode)
            m_initial_condition: DEPRECATED, use M_initial
            drift_field: Drift field specification (optional):
                - None: Zero drift (pure diffusion)
                - np.ndarray: Precomputed drift (e.g., -∇U/λ for MFG)
                - Callable: Function α(t, x, m) -> drift (Phase 2)
            diffusion_field: Diffusion specification (optional):
                - None: Use problem.sigma (backward compatible)
                - float: Constant isotropic diffusion
                - np.ndarray/Callable: Phase 2
            show_progress: Display progress bar (hybrid mode only)

        Returns:
            M_solution: Density evolution
                - HYBRID mode: (Nt, Nx) on grid
                - COLLOCATION mode: (Nt, N_particles) on particles
        """
        # Handle deprecated parameter name
        if m_initial_condition is not None:
            if M_initial is not None:
                raise ValueError(
                    "Cannot specify both M_initial and m_initial_condition. "
                    "Use M_initial (m_initial_condition is deprecated)."
                )
            warnings.warn(
                "Parameter 'm_initial_condition' is deprecated. Use 'M_initial' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            M_initial = m_initial_condition

        # Validate required parameter
        if M_initial is None:
            raise ValueError("M_initial is required")

        # Handle drift_field parameter
        if drift_field is None:
            # Zero drift (pure diffusion): create zero U field for internal use
            params = self._get_grid_params()
            Nt = params["Nt"]
            if self.mode == ParticleMode.COLLOCATION:
                N_points = len(self.collocation_points)
                effective_U = np.zeros((Nt, N_points))
            else:
                grid_shape = params["grid_shape"]
                effective_U = np.zeros((Nt, *grid_shape))
        elif isinstance(drift_field, np.ndarray):
            # Precomputed drift field (including MFG drift = -∇U/λ)
            effective_U = drift_field
        elif callable(drift_field):
            # Custom drift function - Phase 2
            raise NotImplementedError(
                "FPParticleSolver does not yet support callable drift_field. "
                "Pass precomputed drift as np.ndarray. "
                "Support for callable drift coming in Phase 2."
            )
        else:
            raise TypeError(f"drift_field must be None, np.ndarray, or Callable, got {type(drift_field)}")

        # Handle diffusion_field parameter
        if diffusion_field is None:
            # Use problem.sigma (backward compatible)
            effective_sigma = self.problem.sigma
        elif isinstance(diffusion_field, (int, float)):
            # Constant isotropic diffusion
            effective_sigma = float(diffusion_field)
        elif isinstance(diffusion_field, np.ndarray) or callable(diffusion_field):
            # Spatially varying or state-dependent - Phase 2
            raise NotImplementedError(
                "FPParticleSolver does not yet support spatially varying or callable diffusion_field. "
                "Pass constant diffusion as float or use problem.sigma. "
                "Support coming in Phase 2."
            )
        else:
            raise TypeError(
                f"diffusion_field must be None, float, np.ndarray, or Callable, got {type(diffusion_field)}"
            )

        # Temporarily override problem.sigma if custom diffusion provided
        original_sigma = self.problem.sigma
        if diffusion_field is not None:
            self.problem.sigma = effective_sigma

        # Reset time step counter for normalization logic
        self._time_step_counter = 0

        # Store show_progress for use in methods
        self._show_progress = show_progress

        try:
            # Route to appropriate solver based on mode
            if self.mode == ParticleMode.COLLOCATION:
                # Collocation mode: particles → particles (no KDE)
                return self._solve_fp_system_collocation(M_initial, effective_U)
            else:
                # Hybrid mode: particles → grid (existing behavior with strategy selection)
                # Determine problem size for strategy selection
                params = self._get_grid_params()
                Nt = params["Nt"]
                dimension = params["dimension"]
                grid_shape = params["grid_shape"]
                total_points = params["total_points"]
                problem_size = (self.num_particles, total_points, Nt)

                # Route based on dimension
                if dimension == 1:
                    # 1D: Use existing optimized solvers with strategy selection
                    self.current_strategy = self.strategy_selector.select_strategy(
                        backend=self.backend if (self.backend is not None and self.backend.name != "numpy") else None,
                        problem_size=problem_size,
                        strategy_hint="auto",
                    )

                    if self.current_strategy.name == "cpu":
                        return self._solve_fp_system_cpu(M_initial, effective_U)
                    else:
                        return self._solve_fp_system_gpu(M_initial, effective_U)
                else:
                    # nD (d >= 2): Use new nD CPU solver
                    # GPU nD solver not yet implemented
                    return self._solve_fp_system_cpu_nd(M_initial, effective_U)
        finally:
            # Restore original sigma
            self.problem.sigma = original_sigma

    def _solve_fp_system_cpu(self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray) -> np.ndarray:
        """CPU pipeline - existing NumPy implementation."""
        # Use geometry-aware parameter extraction
        params = self._get_grid_params()
        Nx = params["Nx"]
        Nt = params["Nt"]
        Dx = params["Dx"]
        Dt = params["Dt"]
        sigma = params["sigma"]
        coupling_coefficient = params["coupling_coefficient"]

        # SDE: dX = alpha*dt + sigma*dW
        # Convention: problem.sigma is the SDE noise coefficient directly
        sigma_sde = sigma
        x_grid = params["xSpace"]
        xmin = params["xmin"]
        Lx = params["Lx"]

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

        # Progress bar for forward particle timesteps
        # n_time_points - 1 steps to go from t=0 to t=T
        from mfg_pde.utils.progress import RichProgressBar

        timestep_range = range(Nt - 1)
        if self._show_progress:
            timestep_range = RichProgressBar(
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

            alpha_optimal_at_particles = -coupling_coefficient * dUdx_at_particles

            dW = np.random.normal(0.0, np.sqrt(Dt), self.num_particles) if Dt > 1e-14 else np.zeros(self.num_particles)

            current_M_particles_t[n_time_idx + 1, :] = (
                current_M_particles_t[n_time_idx, :] + alpha_optimal_at_particles * Dt + sigma_sde * dW
            )

            # Apply boundary conditions to particles
            if self.boundary_conditions.type == "periodic" and Lx > 1e-14:
                # Periodic boundaries: wrap around
                current_M_particles_t[n_time_idx + 1, :] = xmin + (current_M_particles_t[n_time_idx + 1, :] - xmin) % Lx
            elif self.boundary_conditions.type == "no_flux":
                # Reflecting boundaries: use modular reflection for arbitrary displacement
                # This handles particles that travel multiple domain widths in one step
                particles = current_M_particles_t[n_time_idx + 1, :].copy()

                if Lx > 1e-14:
                    # Normalize position relative to domain [0, 2*Lx] with reflection at Lx
                    # This uses the "fold" reflection: position bounces back and forth
                    # First shift to [0, ...], then fold within [0, 2*Lx], then map to [0, Lx]
                    shifted = particles - xmin
                    # Number of complete round-trips through domain
                    period = 2 * Lx
                    # Position within one period [0, 2*Lx)
                    pos_in_period = shifted % period
                    # If in second half of period, reflect back
                    in_second_half = pos_in_period > Lx
                    pos_in_period[in_second_half] = period - pos_in_period[in_second_half]
                    # Shift back to original domain
                    particles = xmin + pos_in_period

                current_M_particles_t[n_time_idx + 1, :] = particles

            M_density_on_grid[n_time_idx + 1, :] = self._estimate_density_from_particles(
                current_M_particles_t[n_time_idx + 1, :]
            )
            self._time_step_counter += 1  # Increment after each time step

        self.M_particles_trajectory = current_M_particles_t
        return M_density_on_grid

    def _solve_fp_system_cpu_nd(self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray) -> np.ndarray:
        """
        nD CPU pipeline - particle evolution for dimension >= 2.

        Uses the nD helper methods:
        - _sample_particles_from_density_nd() for initial sampling
        - _compute_gradient_nd() for gradient computation
        - _create_nd_interpolator() for drift interpolation
        - _generate_brownian_increment_nd() for vector Brownian motion
        - _apply_boundary_conditions_nd() for per-dimension boundary handling
        - _estimate_density_from_particles_nd() for KDE

        Args:
            m_initial_condition: Initial density on nD grid, shape (N1, N2, ..., Nd)
            U_solution_for_drift: Value function, shape (Nt, N1, N2, ..., Nd)

        Returns:
            Density evolution, shape (Nt, N1, N2, ..., Nd)
        """
        # Extract nD grid parameters
        params = self._get_grid_params()
        dimension = params["dimension"]
        grid_shape = params["grid_shape"]
        spacings = params["spacings"]
        bounds = params["bounds"]
        coordinates = params["coordinates"]
        Nt = params["Nt"]
        Dt = params["Dt"]
        sigma = params["sigma"]
        coupling_coefficient = params["coupling_coefficient"]

        if Nt == 0:
            return np.zeros((0, *tuple(grid_shape)))

        # SDE: dX = alpha*dt + sigma*dW
        # Convention: problem.sigma is the SDE noise coefficient directly
        sigma_sde = sigma

        # Allocate arrays
        M_density_on_grid = np.zeros((Nt, *tuple(grid_shape)))
        # Particle positions: (Nt, num_particles, dimension)
        current_particles = np.zeros((Nt, self.num_particles, dimension))

        # Sample initial particles from density
        current_particles[0] = self._sample_particles_from_density_nd(
            m_initial_condition, coordinates, self.num_particles
        )

        # Estimate initial density using KDE
        M_density_on_grid[0] = self._estimate_density_from_particles_nd(current_particles[0], coordinates, bounds)

        # Normalize if requested
        if self.kde_normalization != KDENormalization.NONE:
            M_density_on_grid[0] = self._normalize_density_nd(M_density_on_grid[0], spacings)

        self._time_step_counter += 1

        if Nt == 1:
            self.M_particles_trajectory = current_particles
            return M_density_on_grid

        # Progress bar for forward particle timesteps (consistent with 1D solver)
        # n_time_points - 1 steps to go from t=0 to t=T
        from mfg_pde.utils.progress import RichProgressBar

        timestep_range = range(Nt - 1)
        if self._show_progress:
            timestep_range = RichProgressBar(
                timestep_range,
                desc=f"FP {dimension}D (forward)",
                unit="step",
                disable=False,
            )

        # Main time evolution loop
        for t_idx in timestep_range:
            # Get value function at current time
            U_t = U_solution_for_drift[t_idx]

            # Compute gradient of U on the grid (list of d arrays, one per dimension)
            gradients = self._compute_gradient_nd(U_t, spacings, use_backend=False)

            # Interpolate gradients to particle positions
            particles_t = current_particles[t_idx]  # Shape: (num_particles, dimension)
            grad_at_particles = np.zeros((self.num_particles, dimension))

            for d in range(dimension):
                try:
                    interp = self._create_nd_interpolator(coordinates, gradients[d])
                    grad_at_particles[:, d] = interp(particles_t)
                except Exception:
                    # Fallback: zero gradient
                    grad_at_particles[:, d] = 0.0

            # Compute drift: alpha = -coupling_coefficient * grad(U)
            drift = -coupling_coefficient * grad_at_particles

            # Generate Brownian increments
            dW = self._generate_brownian_increment_nd(self.num_particles, dimension, Dt, sigma_sde)

            # Euler-Maruyama step: X_{t+1} = X_t + drift * dt + sigma * dW
            new_particles = particles_t + drift * Dt + dW

            # Apply boundary conditions
            # Handle different BC object types (1D has .type, 2D/3D has manager)
            if self.boundary_conditions is None:
                bc_type = "no_flux"
            elif hasattr(self.boundary_conditions, "type"):
                bc_type = self.boundary_conditions.type
            else:
                # For nD boundary condition managers, default to no_flux
                bc_type = "no_flux"
            new_particles = self._apply_boundary_conditions_nd(new_particles, bounds, bc_type)

            current_particles[t_idx + 1] = new_particles

            # Estimate density from particles
            M_density_on_grid[t_idx + 1] = self._estimate_density_from_particles_nd(new_particles, coordinates, bounds)

            # Normalize if requested
            if self.kde_normalization == KDENormalization.ALL:
                M_density_on_grid[t_idx + 1] = self._normalize_density_nd(M_density_on_grid[t_idx + 1], spacings)

            self._time_step_counter += 1

        self.M_particles_trajectory = current_particles
        return M_density_on_grid

    def _normalize_density_nd(self, density: np.ndarray, spacings: list[float]) -> np.ndarray:
        """
        Normalize density to integrate to 1 for nD grids.

        Args:
            density: Density array, shape (N1, N2, ..., Nd)
            spacings: Grid spacings [dx1, dx2, ..., dxd]

        Returns:
            Normalized density array
        """
        # Volume element = dx1 * dx2 * ... * dxd
        dV = np.prod(spacings)
        total_mass = np.sum(density) * dV
        if total_mass > 1e-14:
            return density / total_mass
        return density

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

        # Use geometry-aware parameter extraction
        params = self._get_grid_params()
        Nx = params["Nx"]
        Nt = params["Nt"]
        Dx = params["Dx"]
        Dt = params["Dt"]
        sigma_sde = params["sigma"]
        coupling_coefficient = params["coupling_coefficient"]
        x_grid = params["xSpace"]
        xmin = params["xmin"]
        xmax = params["xmax"]
        Lx = params["Lx"]

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
            drift_gpu = -coupling_coefficient * dUdx_particles_gpu

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

        WARNING: INCOMPLETE IMPLEMENTATION
            This method currently freezes density at the initial condition instead of
            solving the continuity equation. Density does NOT evolve in time.

            Current behavior (line 652): M[t+1] = M[0] for all t
            Expected behavior: M[t+1] = M[t] - Δt * (divergence_term - diffusion_term)

            See Issue #240 for implementation plan.
            Estimated completion: v1.0.0 (target: ~3 days development)

        Design Intent (not yet implemented):
        - Particles FIXED at collocation points (Eulerian representation)
        - Density should evolve via continuity equation: ∂m/∂t + ∇·(m α) = σ²/2 Δm
        - NO KDE interpolation to grid
        - Output: density evolution on collocation points

        Current Limitations:
        - Density frozen at initial condition (physically incorrect for non-equilibrium)
        - No GFDM spatial operators (divergence, Laplacian)
        - No time integration of continuity equation
        - Tests validate API only, not physical correctness

        Args:
            m_initial_condition: Initial density on collocation points (N_particles,)
            U_solution_for_drift: Value function on collocation points (Nt, N_particles)
                (Currently unused - will be needed for drift computation)

        Returns:
            M_solution: Density on collocation points (Nt, N_particles)
                WARNING: Currently returns M[0] repeated for all time steps

        Note:
            This incomplete implementation is sufficient for API testing but should NOT
            be used for production MFG simulations. Use mode="default" (KDE-based) for
            correct density evolution until Issue #240 is resolved.

            Tracking: https://github.com/your-org/MFG_PDE/issues/240
        """
        # n_time_points = problem.Nt + 1 (number of time knots including t=0 and t=T)
        # problem.Nt = number of time intervals
        n_time_points = self.problem.Nt + 1
        N_points = len(self.collocation_points)

        # Validate input shapes
        if m_initial_condition.shape != (N_points,):
            raise ValueError(
                f"m_initial_condition shape {m_initial_condition.shape} "
                f"must match collocation_points count ({N_points},)"
            )
        if U_solution_for_drift.shape != (n_time_points, N_points):
            raise ValueError(
                f"U_solution_for_drift shape {U_solution_for_drift.shape} must be "
                f"(n_time_points={n_time_points}, N_points={N_points})"
            )

        # Storage for density evolution on collocation points
        M_solution = np.zeros((n_time_points, N_points))
        M_solution[0, :] = m_initial_condition.copy()

        # Time step: dt = T / Nt (T divided by number of intervals)
        dt = self.problem.T / self.problem.Nt
        sigma = self.problem.sigma
        diffusion_coeff = 0.5 * sigma**2

        # Import GFDM operators for spatial derivatives
        from mfg_pde.utils.numerical.gfdm_operators import (
            compute_divergence_gfdm,
            compute_gradient_gfdm,
            compute_laplacian_gfdm,
        )

        # Solve continuity equation: ∂m/∂t + ∇·(m α) = σ²/2 Δm
        # Using GFDM operators for spatial derivatives on collocation points
        # Forward FP loop: (n_time_points - 1) steps = problem.Nt intervals
        for t_idx in range(n_time_points - 1):
            m_current = M_solution[t_idx, :]

            # Compute drift field α = -∇U at current time
            U_current = U_solution_for_drift[t_idx, :]

            # Compute gradient of value function using GFDM
            grad_U = compute_gradient_gfdm(U_current, self.collocation_points)

            # Drift field: α = -∇U
            # Shape: (N_points, dimension)
            drift_field = -grad_U

            # Advection term: ∇·(m α)
            divergence_term = compute_divergence_gfdm(drift_field, m_current, self.collocation_points)

            # Diffusion term: σ²/2 Δm
            laplacian_term = compute_laplacian_gfdm(m_current, self.collocation_points)
            diffusion_term = diffusion_coeff * laplacian_term

            # Forward Euler time stepping
            # ∂m/∂t = -∇·(m α) + σ²/2 Δm
            dm_dt = -divergence_term + diffusion_term

            # Update density
            M_solution[t_idx + 1, :] = m_current + dt * dm_dt

            # Enforce non-negativity (physical constraint)
            M_solution[t_idx + 1, :] = np.maximum(M_solution[t_idx + 1, :], 0.0)

            # Optional: Renormalize to conserve mass
            # (May need adjustment based on boundary conditions)
            mass_current = np.sum(M_solution[t_idx + 1, :])
            if mass_current > 0:
                M_solution[t_idx + 1, :] *= np.sum(m_initial_condition) / mass_current

        # Store particle trajectory (in collocation mode, particles are fixed)
        self.M_particles_trajectory = np.tile(self.collocation_points, (n_time_points, 1, 1))

        return M_solution


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing FPParticleSolver...")

    from mfg_pde import MFGProblem

    # Test 1D problem with particle solver
    problem = MFGProblem(Nx=30, Nt=20, T=1.0, sigma=0.1)
    solver = FPParticleSolver(problem, num_particles=1000, mode="hybrid")

    # Test solver initialization
    assert solver.fp_method_name == "Particle"
    assert solver.num_particles == 1000
    assert solver.mode == ParticleMode.HYBRID

    # Test solve_fp_system
    import numpy as np

    U_test = np.zeros((problem.Nt + 1, problem.Nx + 1))
    M_init = problem.m_init

    M_solution = solver.solve_fp_system(M_initial=M_init, drift_field=U_test)

    assert M_solution.shape == (problem.Nt + 1, problem.Nx + 1)
    assert not np.any(np.isnan(M_solution))
    assert not np.any(np.isinf(M_solution))
    assert np.all(M_solution >= 0), "Density must be non-negative"

    print("  Particle solver converged")
    print(f"  Num particles: {solver.num_particles}")
    print(f"  M range: [{M_solution.min():.3f}, {M_solution.max():.3f}]")
    print(f"  KDE bandwidth: {solver.kde_bandwidth}")
    print(f"  Mode: {solver.mode.value}")

    # Test 2D problem with particle solver (nD support)
    print("\nTesting 2D FPParticleSolver...")
    from mfg_pde.geometry import TensorProductGrid

    geometry_2d = TensorProductGrid(
        dimension=2,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[16, 16],
    )
    problem_2d = MFGProblem(geometry=geometry_2d, Nt=10, T=0.5, sigma=0.1)

    solver_2d = FPParticleSolver(problem_2d, num_particles=500, mode="hybrid")

    # Create 2D test arrays
    # U_test_2d has shape (n_time_points, *spatial) = (Nt + 1, *spatial)
    grid_shape_2d = problem_2d.geometry.get_grid_shape()  # (16, 16)
    U_test_2d = np.zeros((problem_2d.Nt + 1, *tuple(grid_shape_2d)))

    # Create 2D Gaussian initial density
    coords = problem_2d.geometry.coordinates
    X, Y = np.meshgrid(coords[0], coords[1], indexing="ij")
    M_init_2d = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.1)
    M_init_2d = M_init_2d / (np.sum(M_init_2d) * (1.0 / 15) ** 2)  # Normalize

    M_solution_2d = solver_2d.solve_fp_system(M_initial=M_init_2d, drift_field=U_test_2d)

    expected_shape_2d = (problem_2d.Nt + 1, *tuple(grid_shape_2d))  # Nt+1 for t=0...T
    assert M_solution_2d.shape == expected_shape_2d, f"Shape mismatch: {M_solution_2d.shape} vs {expected_shape_2d}"
    assert not np.any(np.isnan(M_solution_2d)), "NaN in 2D solution"
    assert not np.any(np.isinf(M_solution_2d)), "Inf in 2D solution"
    assert np.all(M_solution_2d >= 0), "2D density must be non-negative"

    # Check mass conservation (should be approximately preserved)
    initial_mass = np.sum(M_solution_2d[0]) * (1.0 / 15) ** 2
    final_mass = np.sum(M_solution_2d[-1]) * (1.0 / 15) ** 2
    mass_ratio = final_mass / initial_mass if initial_mass > 1e-10 else 1.0

    print("  2D Particle solver converged")
    print(f"  Grid shape: {grid_shape_2d}")
    print(f"  M shape: {M_solution_2d.shape}")
    print(f"  M range: [{M_solution_2d.min():.3f}, {M_solution_2d.max():.3f}]")
    print(f"  Mass ratio (final/initial): {mass_ratio:.4f}")

    # Test particle trajectory storage for 2D
    assert solver_2d.M_particles_trajectory is not None, "Particle trajectory not stored"
    assert solver_2d.M_particles_trajectory.shape[1] == solver_2d.num_particles, "Particle count mismatch"
    assert solver_2d.M_particles_trajectory.shape[2] == 2, "2D particles should have 2 coordinates"
    print(f"  Particle trajectory shape: {solver_2d.M_particles_trajectory.shape}")

    print("\nAll smoke tests passed!")
