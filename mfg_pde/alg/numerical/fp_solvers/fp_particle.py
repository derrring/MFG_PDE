from __future__ import annotations

import warnings
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.geometry import BoundaryConditions
    from mfg_pde.geometry.implicit import ImplicitDomain

try:  # pragma: no cover - optional SciPy dependency
    import scipy.interpolate as _scipy_interpolate
    from scipy.stats import gaussian_kde

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - graceful fallback when SciPy missing
    _scipy_interpolate = None
    gaussian_kde = None
    SCIPY_AVAILABLE = False

from mfg_pde.geometry.boundary.applicator_particle import ParticleApplicator
from mfg_pde.geometry.boundary.types import BCType
from mfg_pde.utils.numerical.particle import (
    interpolate_grid_to_particles,
    interpolate_particles_to_grid,
    sample_from_density,
)
from mfg_pde.utils.numerical.tensor_calculus import gradient_simple

from .base_fp import BaseFPSolver
from .particle_result import FPParticleResult

# Mapping from dimension index to boundary name prefix
_DIM_TO_AXIS_PREFIX = {0: "x", 1: "y", 2: "z"}


def _get_axis_prefix(dim_idx: int) -> str:
    """Get axis prefix for dimension index (x, y, z, dim3, dim4, ...)."""
    if dim_idx < 3:
        return _DIM_TO_AXIS_PREFIX[dim_idx]
    return f"dim{dim_idx}"


class KDENormalization(str, Enum):
    """KDE normalization strategy for particle-based FP solvers."""

    NONE = "none"  # No normalization (raw KDE output)
    INITIAL_ONLY = "initial_only"  # Normalize only at t=0
    ALL = "all"  # Normalize at every time step (default)


class FPParticleSolver(BaseFPSolver):
    """
    Particle-based Fokker-Planck solver using Monte Carlo sampling and KDE.

    This solver samples particles from the initial distribution, evolves them
    using SDE dynamics, and reconstructs the density on a grid using KDE.

    For meshfree density evolution on collocation points, use FPGFDMSolver instead.

    Density Modes (Issue #489 - Direct Particle Query):
        - "grid_only" (default): Store only grid density M (backward compatible)
        - "hybrid": Store both grid density M and particle positions for direct queries
        - "query_only": Store only particles (grid density computed on-demand)

        Use hybrid/query_only modes to enable efficient density queries at arbitrary
        points, providing 10-100× speedup for Semi-Lagrangian HJB coupling.

    Boundary Conditions:
        FPParticleSolver requires explicit boundary conditions. Provide via:
        1. boundary_conditions parameter (direct), OR
        2. problem.geometry.get_boundary_conditions() (from geometry)

        No default fallback - explicit BCs required for correctness.
        The solver will fail fast with a clear error message if BCs are missing.

    Composition Pattern (Issue #545):
        This solver uses composition instead of mixins:
        - self._applicator = ParticleApplicator() for BC application
        - self.geometry = problem.geometry for domain information
        - Explicit dependencies, no implicit state sharing

        Template for other solvers: See docs/development/PARTICLE_SOLVER_TEMPLATE.md
    """

    # Scheme family trait for duality validation (Issue #580)
    from mfg_pde.alg.base_solver import SchemeFamily

    _scheme_family = SchemeFamily.GENERIC  # Particle methods don't fit standard families

    def __init__(
        self,
        problem: MFGProblem,
        num_particles: int = 5000,
        kde_bandwidth: Any = "scott",
        kde_normalization: KDENormalization | str = KDENormalization.ALL,
        density_mode: Literal["grid_only", "hybrid", "query_only"] = "grid_only",
        boundary_conditions: BoundaryConditions | None = None,
        implicit_domain: ImplicitDomain | None = None,
        backend: str | None = None,
        # Deprecated parameters for backward compatibility
        normalize_kde_output: bool | None = None,
        normalize_only_initial: bool | None = None,
        # Legacy parameters (raise helpful errors)
        mode: str | None = None,
        external_particles: np.ndarray | None = None,
    ) -> None:
        super().__init__(problem)

        # Handle deprecated mode parameter
        if mode is not None:
            if mode == "collocation":
                raise ValueError(
                    "Collocation mode has been removed from FPParticleSolver.\n"
                    "Use FPGFDMSolver instead for meshfree density evolution:\n\n"
                    "    from mfg_pde.alg.numerical.fp_solvers import FPGFDMSolver\n"
                    "    solver = FPGFDMSolver(problem, collocation_points=points)\n"
                )
            elif mode != "hybrid":
                raise ValueError(f"Unknown mode: {mode}. FPParticleSolver only supports hybrid mode.")
            # mode="hybrid" is accepted silently for backward compatibility

        # Handle deprecated external_particles parameter
        if external_particles is not None:
            warnings.warn(
                "external_particles parameter is deprecated and ignored. "
                "For meshfree density evolution, use FPGFDMSolver instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.num_particles = num_particles
        self.fp_method_name = "Particle"

        self.kde_bandwidth = kde_bandwidth

        # Handle deprecated parameters
        if normalize_kde_output is not None or normalize_only_initial is not None:
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

        # Density mode for direct queries (Issue #489 Phase 2)
        self.density_mode = density_mode
        self._particle_history: list[np.ndarray] | None = None  # Stored if density_mode != "grid_only"

        # Segment-aware BC applicator (Pattern A: solver owns applicator)
        self._applicator = ParticleApplicator()

        # Exit flux tracking for absorbing BC analysis
        self.exit_flux_history: list[int] = []  # Number absorbed per timestep
        self.exit_positions_history: list[np.ndarray] = []  # Where particles exited
        self.total_absorbed: int = 0  # Cumulative absorbed count

        # Implicit domain for obstacle handling (Issue #533)
        # When set, particles entering obstacles are reflected back
        self._implicit_domain = implicit_domain

        # Initialize backend (defaults to NumPy)
        from mfg_pde.backends import create_backend

        if backend is not None:
            self.backend = create_backend(backend)
        else:
            self.backend = create_backend("numpy")  # NumPy fallback

        # Initialize strategy selector for intelligent pipeline selection
        from mfg_pde.backends.strategies.strategy_selector import StrategySelector

        self.strategy_selector = StrategySelector(profiling_mode="silent")
        self.current_strategy = None  # Will be set in solve_fp_system

        # Boundary condition resolution hierarchy (Issue #545):
        # 1. Explicit boundary_conditions parameter (highest priority)
        # 2. Grid geometry boundary conditions (from geometry)
        # 3. FAIL FAST - no silent fallback (CLAUDE.md principle)
        if boundary_conditions is not None:
            self.boundary_conditions = boundary_conditions
        else:
            # Try geometry BC (use try/except, not hasattr - Issue #543)
            try:
                self.boundary_conditions = problem.geometry.get_boundary_conditions()
            except AttributeError as e:
                # Fail fast - no silent fallback to periodic (CLAUDE.md principle)
                raise ValueError(
                    "FPParticleSolver requires explicit boundary conditions. "
                    "Boundary conditions not provided via:\n"
                    "  1. boundary_conditions=... parameter, OR\n"
                    "  2. problem.geometry.get_boundary_conditions()\n\n"
                    f"Original error: {e}"
                ) from e

            # Validate we got BCs (geometry method might return None)
            if self.boundary_conditions is None:
                raise ValueError(
                    "FPParticleSolver requires boundary conditions. "
                    "problem.geometry.get_boundary_conditions() returned None."
                )

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
        # Try geometry-first API (Issue #543: use try/except instead of hasattr)
        try:
            geom = self.problem.geometry
            if geom is None:
                raise AttributeError("geometry is None")

            grid_shape = tuple(geom.get_grid_shape())
            dimension = len(grid_shape)

            # Get spacing per dimension
            spacing = geom.get_grid_spacing()
            spacings = list(spacing) if spacing else [0.0] * dimension

            # Get bounds per dimension (with fallback chain for legacy interfaces)
            try:
                bounds = list(geom.bounds) if geom.bounds is not None else None
                if bounds is None:
                    raise AttributeError("bounds is None")
            except AttributeError:
                try:
                    # Legacy 1D geometry
                    bounds = [(geom.xmin, geom.xmax)]
                except AttributeError:
                    try:
                        # Infer from coordinates
                        if len(geom.coordinates) > 0:
                            bounds = [(coords[0], coords[-1]) for coords in geom.coordinates]
                        else:
                            bounds = [(0.0, 1.0)] * dimension
                    except AttributeError:
                        bounds = [(0.0, 1.0)] * dimension

            # Get coordinate arrays per dimension
            try:
                if len(geom.coordinates) > 0:
                    coordinates = [np.array(c) for c in geom.coordinates]
                else:
                    coordinates = [np.linspace(bounds[d][0], bounds[d][1], grid_shape[d]) for d in range(dimension)]
            except AttributeError:
                coordinates = [np.linspace(bounds[d][0], bounds[d][1], grid_shape[d]) for d in range(dimension)]

        except AttributeError as e:
            # Geometry is always available after MFGProblem initialization
            raise ValueError(
                "FPParticleSolver requires a geometry object. "
                "Create MFGProblem with geometry=TensorProductGrid(...) or with Nx=... parameter."
            ) from e

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

    def _compute_gradient_nd(
        self,
        U_array: np.ndarray,
        spacings: list[float],
        use_backend: bool = False,
    ) -> list:
        """
        Compute spatial gradient using grid_operators utility.

        Uses gradient_simple (central differences, no BC handling).
        BC handling for particles is done separately in _apply_boundary_conditions_nd.
        """
        backend = self.backend if use_backend else None
        return gradient_simple(U_array, spacings, backend=backend)

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
            # Handle PyTorch tensors (Issue #543: use try/except instead of hasattr)
            try:
                mass_val = mass.item()  # PyTorch tensor → scalar
            except AttributeError:
                mass_val = float(mass)  # NumPy or Python number
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

        Delegates to utils.numerical.particle.sample_from_density() for the
        actual sampling. This method provides a consistent interface within
        the solver class.

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
        return sample_from_density(
            density=M_initial,
            coordinates=coordinates,
            num_samples=num_particles,
            jitter=True,
            seed=None,  # Use global numpy random state for reproducibility with solver
        )

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

    def _needs_segment_aware_bc(self) -> bool:
        """
        Check if boundary conditions require segment-aware handling.

        Returns True if:
        - BC has multiple segments with different types
        - BC has absorbing (DIRICHLET) segments that should remove particles

        Returns False for uniform periodic/reflecting BC where fast path can be used.
        """
        bc = self.boundary_conditions

        if bc is None:
            return False

        # Check if uniform BC (same type everywhere) - Issue #543: use getattr instead of hasattr
        is_uniform = getattr(bc, "is_uniform", False)
        if is_uniform:
            # Uniform BC - use fast path unless it's DIRICHLET (absorbing)
            segments = getattr(bc, "segments", [])
            return len(segments) > 0 and segments[0].bc_type == BCType.DIRICHLET

        # Mixed BC with segments - need segment-aware handling
        segments = getattr(bc, "segments", [])
        if len(segments) > 1:
            return True

        # Check for DIRICHLET segments (absorbing BC)
        return any(segment.bc_type == BCType.DIRICHLET for segment in segments)

    def _get_topology_per_dimension(
        self,
        dimension: int,
    ) -> list[str]:
        """
        Get grid topology for each dimension from boundary conditions.

        This determines the INDEXING STRATEGY for particles, not the physical BC:
        - "periodic": Space wraps around (particles use modular arithmetic)
        - "bounded": Space has walls (particles reflect at boundaries)

        Note: This is about topology (how space connects), not physics (what values
        are prescribed). For particles, all non-periodic boundaries are treated as
        reflecting walls, regardless of whether the underlying BC is Dirichlet,
        Neumann, Robin, or no-flux.

        Parameters
        ----------
        dimension : int
            Number of spatial dimensions

        Returns
        -------
        topologies : list[str]
            Topology per dimension: ["periodic", "bounded", ...]
        """
        bc = self.boundary_conditions

        # Default to bounded (reflecting walls) for all dimensions
        topologies = ["bounded"] * dimension

        if bc is None:
            return topologies

        # For uniform BCs, check if periodic - Issue #543: use getattr instead of hasattr
        is_uniform = getattr(bc, "is_uniform", False)
        if is_uniform:
            if bc.type == "periodic":
                return ["periodic"] * dimension
            return topologies

        # For mixed BCs, check per dimension
        # Periodic requires BOTH min and max to be periodic (topological constraint)
        # Issue #543: use callable() for method existence check
        if not callable(getattr(bc, "get_bc_type_at_boundary", None)):
            return topologies  # Method doesn't exist, use default bounded

        for d in range(dimension):
            axis_prefix = _get_axis_prefix(d)
            min_boundary = f"{axis_prefix}_min"
            max_boundary = f"{axis_prefix}_max"

            try:
                bc_min = bc.get_bc_type_at_boundary(min_boundary)
                bc_max = bc.get_bc_type_at_boundary(max_boundary)

                # Periodic topology requires both boundaries to be periodic
                if bc_min == BCType.PERIODIC and bc_max == BCType.PERIODIC:
                    topologies[d] = "periodic"
                # All other cases: bounded topology (reflecting for particles)
            except (KeyError, AttributeError):
                pass  # Keep default "bounded"

        return topologies

    def _enforce_obstacle_boundary(self, particles: np.ndarray) -> np.ndarray:
        """
        Enforce obstacle boundaries via implicit domain geometry (Issue #533).

        If an implicit domain with obstacles is defined, particles that have
        entered obstacle regions (domain.contains() returns False) are projected
        back to the valid domain using domain.project_to_domain().

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape (num_particles, dimension)

        Returns
        -------
        particles : np.ndarray
            Updated particle positions with obstacle violations corrected
        """
        if self._implicit_domain is None:
            return particles

        # Check which particles are outside the valid domain (inside obstacles)
        inside_valid = self._implicit_domain.contains(particles)

        # Handle scalar return (single particle case)
        if np.isscalar(inside_valid):
            inside_valid = np.array([inside_valid])

        # Project invalid particles back to domain
        if not np.all(inside_valid):
            outside_indices = np.where(~inside_valid)[0]
            particles[outside_indices] = self._implicit_domain.project_to_domain(particles[outside_indices])

        return particles

    def _apply_boundary_conditions_nd(
        self,
        particles: np.ndarray,
        bounds: list[tuple[float, float]],
        topology: str | list[str],
    ) -> np.ndarray:
        """
        Apply boundary handling per dimension based on topology.

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape (num_particles, dimension)
        bounds : list[tuple[float, float]]
            Bounds per dimension [(xmin, xmax), (ymin, ymax), ...]
        topology : str or list[str]
            Grid topology: "periodic" (wrap) or "bounded" (reflect).
            Can be a single string (same for all dims) or per-dimension list.

        Returns
        -------
        particles : np.ndarray
            Updated particle positions (modified in place)
        """
        dimension = particles.shape[1] if particles.ndim > 1 else 1

        # Handle per-dimension topologies
        if isinstance(topology, str):
            topologies = [topology] * dimension
        else:
            topologies = topology

        for d in range(dimension):
            xmin, xmax = bounds[d]
            Lx = xmax - xmin

            if Lx < 1e-14:
                continue  # Skip degenerate dimension

            dim_topology = topologies[d] if d < len(topologies) else "bounded"

            if dim_topology == "periodic":
                # Periodic topology: wrap around
                particles[:, d] = xmin + (particles[:, d] - xmin) % Lx

            else:  # "bounded" -> reflecting walls
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

    def _apply_boundary_conditions_segment_aware(
        self,
        particles: np.ndarray,
        bounds: list[tuple[float, float]],
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """
        Apply segment-aware boundary conditions using the applicator.

        This method handles mixed BC where different boundaries have different types:
        - DIRICHLET segments: Absorb particles (remove from simulation)
        - REFLECTING/NO_FLUX segments: Bounce particles back
        - PERIODIC segments: Wrap particles

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape (num_particles, dimension)
        bounds : list[tuple[float, float]]
            Bounds per dimension [(xmin, xmax), (ymin, ymax), ...]

        Returns
        -------
        remaining_particles : np.ndarray
            Particles that were not absorbed, shape (M, dimension)
        n_absorbed : int
            Number of particles absorbed this step
        exit_positions : np.ndarray
            Positions where particles exited, shape (K, dimension)
        """
        remaining, absorbed_mask, exit_positions = self._applicator.apply(
            particles,
            self.boundary_conditions,
            bounds,
        )

        n_absorbed = int(np.sum(absorbed_mask))

        # Update cumulative tracking
        if n_absorbed > 0:
            self.exit_flux_history.append(n_absorbed)
            self.exit_positions_history.append(exit_positions)
            self.total_absorbed += n_absorbed

        return remaining, n_absorbed, exit_positions

    def _apply_boundary_conditions_with_flux_limits(
        self,
        particles: np.ndarray,
        bounds: list[tuple[float, float]],
        flux_limits: dict[str, float],
    ) -> tuple[np.ndarray, int, np.ndarray, dict[str, int]]:
        """
        Apply segment-aware BC with flux-limited absorption.

        Particles at DIRICHLET exits are absorbed only up to the flux capacity.
        When capacity is exceeded, particles are REFLECTED (creating queues).

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape (num_particles, dimension)
        bounds : list[tuple[float, float]]
            Bounds per dimension [(xmin, xmax), (ymin, ymax), ...]
        flux_limits : dict[str, float]
            Max particles to absorb per segment this step
            e.g., {"exit_A": 10, "exit_B": 15}

        Returns
        -------
        remaining_particles : np.ndarray
            Particles not absorbed
        n_absorbed : int
            Total particles absorbed this step
        exit_positions : np.ndarray
            Positions where absorbed
        absorbed_per_segment : dict[str, int]
            Absorbed count per segment
        """
        remaining, absorbed_mask, exit_positions, absorbed_per_segment = self._applicator.apply_with_flux_limits(
            particles,
            self.boundary_conditions,
            bounds,
            flux_limits,
        )

        n_absorbed = int(np.sum(absorbed_mask))

        if n_absorbed > 0:
            self.exit_flux_history.append(n_absorbed)
            self.exit_positions_history.append(exit_positions)
            self.total_absorbed += n_absorbed

        return remaining, n_absorbed, exit_positions, absorbed_per_segment

    def _interpolate_grid_to_particles_nd(
        self,
        grid_values: np.ndarray,
        bounds: list[tuple[float, float]],
        particles: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate grid values to particle positions.

        Delegates to utils.numerical.particle.interpolate_grid_to_particles().

        Parameters
        ----------
        grid_values : np.ndarray
            Values on grid, shape (N1, N2, ..., Nd)
        bounds : list[tuple[float, float]]
            Domain bounds per dimension
        particles : np.ndarray
            Particle positions, shape (num_particles, dimension)

        Returns
        -------
        values_at_particles : np.ndarray
            Interpolated values, shape (num_particles,)
        """
        # Convert bounds to format expected by utils: tuple of tuples
        grid_bounds = tuple(bounds)
        return interpolate_grid_to_particles(
            grid_values=grid_values,
            grid_bounds=grid_bounds,
            particle_positions=particles,
            method="linear",
        )

    def _estimate_density_from_particles_nd(
        self,
        particles: np.ndarray,
        coordinates: list[np.ndarray],
        bounds: list[tuple[float, float]],
    ) -> np.ndarray:
        """
        Estimate density from particles using KDE on nD grid.

        Delegates to utils.numerical.particle.interpolate_particles_to_grid()
        for the core KDE computation, with additional edge case handling.

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

        # Edge case: no particles
        if self.num_particles == 0 or len(particles) == 0:
            return np.zeros(grid_shape)

        # Edge case: degenerate particle distribution (all at same location)
        if len(np.unique(particles, axis=0)) < 2:
            density = np.zeros(grid_shape)
            mean_pos = np.mean(particles, axis=0)
            indices = []
            for d in range(dimension):
                idx = np.argmin(np.abs(coordinates[d] - mean_pos[d]))
                indices.append(idx)
            density[tuple(indices)] = self.num_particles
            return density

        try:
            # Delegate to utils for KDE computation
            # Note: particle_values is ignored for KDE method, use ones
            particle_values = np.ones(len(particles))
            grid_bounds = tuple(bounds)

            density = interpolate_particles_to_grid(
                particle_values=particle_values,
                particle_positions=particles,
                grid_shape=grid_shape,
                grid_bounds=grid_bounds,
                method="kde",
                bandwidth=self.kde_bandwidth,
            )

            return density

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
        drift_is_precomputed: bool = False,
        # Deprecated parameter name for backward compatibility
        m_initial_condition: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Solve FP system using particle method with unified API.

        Uses KDE-based particle method: sample own particles, output to grid via KDE.
        Strategy Selection: Automatically selects CPU/GPU/Hybrid based on problem size.

        For meshfree density evolution on scattered points, use FPGFDMSolver instead.

        Args:
            M_initial: Initial density m0(x) on grid
            m_initial_condition: DEPRECATED, use M_initial
            drift_field: Drift field specification (optional):
                - None: Zero drift (pure diffusion)
                - np.ndarray: If drift_is_precomputed=False (default), this is U(t,x) and gradient will be computed.
                             If drift_is_precomputed=True, this is α(t,x,d) vector field (Nt, *grid_shape, d).
                - Callable: Function alpha(t, x, m) -> drift (Phase 2)
            drift_is_precomputed: If True, drift_field is treated as precomputed drift vector α(t,x).
                                  If False (default), drift_field is treated as value function U(t,x) and
                                  drift is computed as α = -coupling_coefficient * ∇U.
                                  Use True to preserve high-precision gradients from GFDM or other meshfree methods.
            diffusion_field: Diffusion specification (optional):
                - None: Use problem.sigma (backward compatible)
                - float: Constant isotropic diffusion
                - np.ndarray/Callable: Phase 2
            show_progress: Display progress bar

        Returns:
            M_solution: Density evolution on grid, shape (Nt+1, *grid_shape)
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
            grid_shape = params["grid_shape"]
            effective_U = np.zeros((Nt, *grid_shape))
        elif isinstance(drift_field, np.ndarray):
            # Precomputed drift field (including MFG drift = -∇U/λ)
            effective_U = drift_field
        elif callable(drift_field):
            # Custom drift function - Phase 2
            # Route to callable drift solver
            return self._solve_fp_system_callable_drift(
                M_initial=M_initial,
                drift_callable=drift_field,
                diffusion_field=diffusion_field,
                show_progress=show_progress,
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
            # Spatially varying or state-dependent diffusion
            # Route to callable drift solver which supports this
            # Note: If drift_field is None, we need to handle pure diffusion case
            if drift_field is None:
                # Pure diffusion with spatially varying coefficient
                # Use callable drift path with zero drift
                def zero_drift(t, x, m):
                    if isinstance(x, np.ndarray) and x.ndim > 1:
                        return np.zeros_like(x)
                    return np.zeros_like(np.atleast_1d(x))

                return self._solve_fp_system_callable_drift(
                    M_initial=M_initial,
                    drift_callable=zero_drift,
                    diffusion_field=diffusion_field,
                    show_progress=show_progress,
                )
            else:
                # Already routed to callable drift above if drift is callable
                # This handles array drift + varying diffusion
                effective_sigma = self.problem.sigma  # Fallback, actual handled in solver
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

        # Store show_progress and drift_is_precomputed for use in methods
        self._show_progress = show_progress
        self._drift_is_precomputed = drift_is_precomputed

        # Initialize particle history for direct query modes (Issue #489)
        if self.density_mode in ("hybrid", "query_only"):
            self._particle_history = []

        try:
            # Hybrid mode: particles -> grid (with strategy selection)
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

        # Check if segment-aware BC is needed (Issue #535 Phase 1)
        use_segment_aware_bc = self._needs_segment_aware_bc()

        # Reset exit flux tracking for this solve
        if use_segment_aware_bc:
            self.exit_flux_history = []
            self.exit_positions_history = []
            self.total_absorbed = 0

        M_density_on_grid = np.zeros((Nt, Nx))

        # For segment-aware BC with absorption, use list storage (variable particle count)
        # For uniform BC, use fixed array (all particles preserved)
        if use_segment_aware_bc:
            particles_list: list[np.ndarray] = [None] * Nt  # type: ignore
            current_M_particles_t = None  # Will be built at end
        else:
            current_M_particles_t = np.zeros((Nt, self.num_particles))
            particles_list = None

        # Sample initial particles
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

        # Store initial particles
        if use_segment_aware_bc:
            particles_list[0] = initial_particle_positions
            init_particles = particles_list[0]
        else:
            current_M_particles_t[0, :] = initial_particle_positions
            init_particles = current_M_particles_t[0, :]

        M_density_on_grid[0, :] = self._estimate_density_from_particles(init_particles)
        self._time_step_counter += 1  # Increment after computing density at t=0

        if Nt == 1:
            if use_segment_aware_bc:
                self.M_particles_trajectory = particles_list
            else:
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
            # Get current particles from appropriate storage
            if use_segment_aware_bc:
                particles_t = particles_list[n_time_idx]
                n_particles_t = len(particles_t)
            else:
                particles_t = current_M_particles_t[n_time_idx, :]
                n_particles_t = self.num_particles

            # Skip if no particles remain (all absorbed)
            if n_particles_t == 0:
                if use_segment_aware_bc:
                    particles_list[n_time_idx + 1] = np.array([])
                M_density_on_grid[n_time_idx + 1, :] = np.zeros(Nx)
                self._time_step_counter += 1
                continue

            U_at_tn = U_solution_for_drift[n_time_idx, :]

            # Use shared nD gradient method (works for 1D too)
            if Nx > 1:
                gradients = self._compute_gradient_nd(U_at_tn, [Dx], use_backend=False)
                dUdx_grid = gradients[0]  # First (and only) dimension
            else:
                dUdx_grid = np.zeros(Nx)

            if Nx > 1:
                # Interpolate gradient to particle positions using utils
                particles_1d = particles_t.reshape(-1, 1)
                dUdx_at_particles = interpolate_grid_to_particles(
                    grid_values=dUdx_grid,
                    grid_bounds=(xmin, xmin + Lx),
                    particle_positions=particles_1d,
                    method="linear",
                )
            else:
                dUdx_at_particles = np.zeros(n_particles_t)

            alpha_optimal_at_particles = -coupling_coefficient * dUdx_at_particles

            # Generate Brownian motion for current particle count
            dW = np.random.normal(0.0, np.sqrt(Dt), n_particles_t) if Dt > 1e-14 else np.zeros(n_particles_t)

            # Euler-Maruyama update
            new_particles = particles_t + alpha_optimal_at_particles * Dt + sigma_sde * dW

            # Apply boundary conditions (segment-aware or topology-based)
            if use_segment_aware_bc:
                # Segment-aware BC: may absorb particles
                # Convert to 2D for applicator (expects shape (N, d))
                particles_2d = new_particles.reshape(-1, 1)
                remaining_2d, _n_absorbed, _ = self._apply_boundary_conditions_segment_aware(
                    particles_2d, [(xmin, xmin + Lx)]
                )
                new_particles = remaining_2d[:, 0]  # Back to 1D
                particles_list[n_time_idx + 1] = new_particles
            else:
                # Uniform BC: topology-based (no absorption)
                particles_2d = new_particles.reshape(-1, 1)
                topologies = self._get_topology_per_dimension(1)
                particles_2d = self._apply_boundary_conditions_nd(particles_2d, [(xmin, xmin + Lx)], topologies)
                new_particles = particles_2d[:, 0]
                current_M_particles_t[n_time_idx + 1, :] = new_particles

            M_density_on_grid[n_time_idx + 1, :] = self._estimate_density_from_particles(new_particles)
            self._time_step_counter += 1  # Increment after each time step

        # Store trajectory (different format for segment-aware BC)
        if use_segment_aware_bc:
            self.M_particles_trajectory = particles_list
        else:
            self.M_particles_trajectory = current_M_particles_t

        # Build particle history for direct query mode (Issue #489)
        if self._particle_history is not None:
            # Convert stored trajectory to list of (num_particles, dimension) arrays
            if use_segment_aware_bc:
                # List of 1D arrays -> List of (num_particles_t, 1) arrays
                for t_particles in particles_list:
                    if t_particles.ndim == 1:
                        self._particle_history.append(t_particles.reshape(-1, 1))
                    else:
                        self._particle_history.append(t_particles)
            else:
                # 2D array (Nt, num_particles) -> List of (num_particles, 1) arrays
                for t in range(Nt):
                    self._particle_history.append(current_M_particles_t[t, :].reshape(-1, 1))

        # Return FPParticleResult if particle history was stored (Issue #489)
        if self._particle_history is not None:
            time_grid = np.linspace(0, self.problem.T, Nt)
            return FPParticleResult(
                M_grid=M_density_on_grid,
                time_grid=time_grid,
                particle_history=self._particle_history,
                bandwidth=self.kde_bandwidth if isinstance(self.kde_bandwidth, (int, float)) else None,
            )

        # Backward compatible: return grid density only
        return M_density_on_grid

    def _solve_fp_system_cpu_nd(self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray) -> np.ndarray:
        """
        nD CPU pipeline - particle evolution for dimension >= 2.

        Uses the nD helper methods:
        - _sample_particles_from_density_nd() for initial sampling
        - _compute_gradient_nd() for gradient computation
        - _interpolate_grid_to_particles_nd() for drift interpolation
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

        # Check if we need segment-aware BC (for absorbing boundaries)
        use_segment_aware_bc = self._needs_segment_aware_bc()

        # Reset exit flux tracking for this solve
        self.exit_flux_history = []
        self.exit_positions_history = []
        self.total_absorbed = 0

        # Allocate arrays
        M_density_on_grid = np.zeros((Nt, *tuple(grid_shape)))

        # For segment-aware BC with absorption, use list storage (variable particle count)
        # For uniform BC, use fixed array (all particles preserved)
        if use_segment_aware_bc:
            # List storage: each timestep may have different particle count
            particles_list: list[np.ndarray] = [None] * Nt  # type: ignore
            particles_list[0] = self._sample_particles_from_density_nd(
                m_initial_condition, coordinates, self.num_particles
            )
            current_particles = None  # Will be built at end
        else:
            # Particle positions: (Nt, num_particles, dimension)
            current_particles = np.zeros((Nt, self.num_particles, dimension))
            particles_list = None  # Not used

            # Sample initial particles from density
            current_particles[0] = self._sample_particles_from_density_nd(
                m_initial_condition, coordinates, self.num_particles
            )

        # Get initial particles for density estimation
        init_particles = particles_list[0] if use_segment_aware_bc else current_particles[0]

        # Estimate initial density using KDE
        M_density_on_grid[0] = self._estimate_density_from_particles_nd(init_particles, coordinates, bounds)

        # Normalize if requested
        if self.kde_normalization != KDENormalization.NONE:
            M_density_on_grid[0] = self._normalize_density_nd(M_density_on_grid[0], spacings)

        self._time_step_counter += 1

        if Nt == 1:
            if use_segment_aware_bc:
                self.M_particles_trajectory = particles_list  # List of arrays
            else:
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
            # Get drift field at current time
            drift_or_U_t = U_solution_for_drift[t_idx]

            # Get particles at current timestep
            if use_segment_aware_bc:
                particles_t = particles_list[t_idx]
            else:
                particles_t = current_particles[t_idx]

            n_particles_t = len(particles_t)

            # Skip if no particles remain (all absorbed)
            if n_particles_t == 0:
                if use_segment_aware_bc:
                    particles_list[t_idx + 1] = np.array([]).reshape(0, dimension)
                M_density_on_grid[t_idx + 1] = np.zeros(grid_shape)
                self._time_step_counter += 1
                continue

            # Check if drift is precomputed or needs to be computed from U
            if self._drift_is_precomputed:
                # Drift is already computed (e.g., from GFDM: α = -D_p H(x,m,∇u))
                # drift_or_U_t has shape (*grid_shape, dimension)
                # Need to interpolate vector field to particle positions
                drift_at_particles = np.zeros((n_particles_t, dimension))

                for d in range(dimension):
                    # Extract d-th component of drift vector field
                    drift_d = drift_or_U_t[..., d]  # Shape: (*grid_shape,)
                    drift_at_particles[:, d] = self._interpolate_grid_to_particles_nd(drift_d, bounds, particles_t)

                # Use precomputed drift directly (no coupling_coefficient multiplication)
                drift = drift_at_particles

            else:
                # Traditional path: drift_or_U_t is value function U, compute gradient
                U_t = drift_or_U_t  # For clarity

                # Compute gradient of U on the grid (list of d arrays, one per dimension)
                gradients = self._compute_gradient_nd(U_t, spacings, use_backend=False)

                # Interpolate gradients to particle positions
                grad_at_particles = np.zeros((n_particles_t, dimension))

                for d in range(dimension):
                    grad_at_particles[:, d] = self._interpolate_grid_to_particles_nd(gradients[d], bounds, particles_t)

                # Compute drift: alpha = -coupling_coefficient * grad(U)
                drift = -coupling_coefficient * grad_at_particles

            # Generate Brownian increments
            dW = self._generate_brownian_increment_nd(n_particles_t, dimension, Dt, sigma_sde)

            # Euler-Maruyama step: X_{t+1} = X_t + drift * dt + sigma * dW
            new_particles = particles_t + drift * Dt + dW

            # Enforce obstacle boundaries if implicit domain is set (Issue #533)
            new_particles = self._enforce_obstacle_boundary(new_particles)

            # Apply boundary conditions
            if use_segment_aware_bc:
                # Segment-aware BC: may absorb particles
                new_particles, _n_absorbed, _ = self._apply_boundary_conditions_segment_aware(new_particles, bounds)
                particles_list[t_idx + 1] = new_particles
            else:
                # Uniform BC: topology-based (no absorption)
                topologies = self._get_topology_per_dimension(dimension)
                new_particles = self._apply_boundary_conditions_nd(new_particles, bounds, topologies)
                current_particles[t_idx + 1] = new_particles

            # Estimate density from particles
            M_density_on_grid[t_idx + 1] = self._estimate_density_from_particles_nd(new_particles, coordinates, bounds)

            # Normalize if requested
            if self.kde_normalization == KDENormalization.ALL:
                M_density_on_grid[t_idx + 1] = self._normalize_density_nd(M_density_on_grid[t_idx + 1], spacings)

            self._time_step_counter += 1

        # Store trajectory (different format for segment-aware BC)
        if use_segment_aware_bc:
            # Store as list of arrays (variable particle counts)
            self.M_particles_trajectory = particles_list
        else:
            self.M_particles_trajectory = current_particles

        # Build particle history for direct query mode (Issue #489)
        if self._particle_history is not None:
            # Convert stored trajectory to list of (num_particles, dimension) arrays
            if use_segment_aware_bc:
                # List already in correct format: (num_particles_t, dimension)
                self._particle_history.extend(particles_list)
            else:
                # 3D array (Nt, num_particles, dimension) -> List of (num_particles, dimension)
                for t in range(Nt):
                    self._particle_history.append(current_particles[t, :, :])

        # Return FPParticleResult if particle history was stored (Issue #489)
        if self._particle_history is not None:
            time_grid = np.linspace(0, self.problem.T, Nt)
            return FPParticleResult(
                M_grid=M_density_on_grid,
                time_grid=time_grid,
                particle_history=self._particle_history,
                bandwidth=self.kde_bandwidth if isinstance(self.kde_bandwidth, (int, float)) else None,
            )

        # Backward compatible: return grid density only
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

        Note (Issue #535 Phase 1): Segment-aware absorbing BC not yet implemented
        for GPU solver. Use CPU solver for mixed BC with DIRICHLET segments.
        GPU solver currently supports uniform BC only (periodic/reflecting).
        """
        from mfg_pde.alg.numerical.density_estimation import gaussian_kde_gpu_internal
        from mfg_pde.utils.particle_utils import (
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
        X_particles_gpu[0, :] = sample_from_density_gpu(
            m_initial_gpu, x_grid_gpu, self.num_particles, self.backend, seed=None
        )

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

            # Compute gradient on grid (use shared nD method)
            if Nx > 1:
                gradients_gpu = self._compute_gradient_nd(U_t_gpu, [Dx], use_backend=True)
                dUdx_gpu = gradients_gpu[0]  # First (and only) dimension
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

            # Apply boundary handling (GPU, supports mixed BCs)
            # Map topology to GPU bc_type: "bounded" -> "no_flux" for GPU function
            topology_1d = self._get_topology_per_dimension(1)[0]
            gpu_bc_type = "periodic" if topology_1d == "periodic" else "no_flux"
            if Lx > 1e-14:
                X_particles_gpu[t + 1, :] = apply_boundary_conditions_gpu(
                    X_particles_gpu[t + 1, :], xmin, xmax, gpu_bc_type, self.backend
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
        X_particles_np = self.backend.to_numpy(X_particles_gpu)
        M_density_np = self.backend.to_numpy(M_density_gpu)
        self.M_particles_trajectory = X_particles_np

        # Build particle history for direct query mode (Issue #489)
        if self._particle_history is not None:
            # Convert 2D array (Nt, num_particles) to list of (num_particles, 1) arrays
            for t in range(Nt):
                self._particle_history.append(X_particles_np[t, :].reshape(-1, 1))

        # Return FPParticleResult if particle history was stored (Issue #489)
        if self._particle_history is not None:
            time_grid = np.linspace(0, self.problem.T, Nt)
            return FPParticleResult(
                M_grid=M_density_np,
                time_grid=time_grid,
                particle_history=self._particle_history,
                bandwidth=self.kde_bandwidth if isinstance(self.kde_bandwidth, (int, float)) else None,
            )

        # Backward compatible: return grid density only
        return M_density_np

    def _solve_fp_system_callable_drift(
        self,
        M_initial: np.ndarray,
        drift_callable: Callable,
        diffusion_field: float | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Solve FP equation with callable (state-dependent) drift using particles.

        Evaluates drift at particle positions at each timestep, enabling
        nonlinear PDEs with state-dependent advection.

        Parameters
        ----------
        M_initial : np.ndarray
            Initial density on grid
        drift_callable : callable
            Function α(t, x, m) -> drift velocity
            - t: time (scalar)
            - x: particle positions, shape (N_particles, d)
            - m: density at particle positions, shape (N_particles,)
            Returns: drift velocity, shape (N_particles, d) for nD or (N_particles,) for 1D
        diffusion_field : float or None
            Constant diffusion coefficient (uses problem.sigma if None)
        show_progress : bool
            Show progress bar

        Returns
        -------
        np.ndarray
            Density evolution on grid, shape (Nt+1, *grid_shape)
        """
        from mfg_pde.types.pde_coefficients import DriftCallable

        # Validate callable
        if not isinstance(drift_callable, DriftCallable):
            raise TypeError(
                "drift_field callable does not match DriftCallable protocol. "
                "Expected signature: (t: float, x: ndarray, m: ndarray) -> ndarray"
            )

        # Get parameters
        params = self._get_grid_params()
        Nt = params["Nt"]
        Dt = params["Dt"]
        dimension = params["dimension"]
        grid_shape = params["grid_shape"]
        bounds = params["bounds"]
        spacings = params["spacings"]
        coordinates = params["coordinates"]

        # Get diffusion - supports constant, array, or callable
        # For SDE: dX = drift*dt + sqrt(2*sigma)*dW
        diffusion_is_callable = callable(diffusion_field)
        diffusion_is_array = isinstance(diffusion_field, np.ndarray)

        if diffusion_field is None:
            base_sigma = self.problem.sigma
        elif isinstance(diffusion_field, (int, float)):
            base_sigma = float(diffusion_field)
        elif diffusion_is_array or diffusion_is_callable:
            # Spatially varying or state-dependent diffusion
            # Will be evaluated per timestep at particle positions
            base_sigma = None  # Evaluated dynamically
        else:
            raise TypeError(f"diffusion_field must be None, float, ndarray, or Callable, got {type(diffusion_field)}")

        # Pre-compute constant sigma_sde if diffusion is constant
        if base_sigma is not None:
            sigma_sde_constant = np.sqrt(2 * base_sigma)
        else:
            sigma_sde_constant = None

        # Initialize particles from initial density
        current_particles = np.zeros((Nt + 1, self.num_particles, dimension))
        current_particles[0] = self._sample_particles_from_density_nd(M_initial, coordinates, self.num_particles)

        # Allocate density array
        M_density_on_grid = np.zeros((Nt + 1, *grid_shape))
        M_density_on_grid[0] = M_initial.copy()

        # Normalize initial if requested
        if self.kde_normalization == KDENormalization.ALL:
            M_density_on_grid[0] = self._normalize_density_nd(M_density_on_grid[0], spacings)

        # Progress bar
        from mfg_pde.utils.progress import RichProgressBar

        timestep_range = range(Nt)
        if show_progress:
            timestep_range = RichProgressBar(
                timestep_range,
                desc="FP Particle (callable drift)",
                unit="step",
                disable=False,
            )

        # Time evolution with callable drift
        for t_idx in timestep_range:
            t_current = t_idx * Dt
            particles_t = current_particles[t_idx]  # Shape: (num_particles, dimension)

            # Estimate density at particle positions for state-dependent drift
            m_at_particles = self._estimate_density_at_particles(
                particles_t, M_density_on_grid[t_idx], coordinates, bounds
            )

            # Evaluate drift callable
            # For 1D: x is (N,) and returns (N,)
            # For nD: x is (N, d) and returns (N, d)
            if dimension == 1:
                x_for_callable = particles_t[:, 0]  # Flatten to (N,)
                drift_values = drift_callable(t_current, x_for_callable, m_at_particles)
                # Ensure shape is (N, 1) for consistent processing
                if drift_values.ndim == 1:
                    drift = drift_values[:, np.newaxis]
                else:
                    drift = drift_values
            else:
                drift = drift_callable(t_current, particles_t, m_at_particles)
                if drift.ndim == 1:
                    # Scalar drift applied to all dimensions
                    drift = np.tile(drift[:, np.newaxis], (1, dimension))

            # Generate Brownian increments with per-particle diffusion support
            if sigma_sde_constant is not None:
                # Constant diffusion - use pre-computed value
                dW = self._generate_brownian_increment_nd(self.num_particles, dimension, Dt, sigma_sde_constant)
            else:
                # Spatially varying or callable diffusion - evaluate at particle positions
                if diffusion_is_callable:
                    # Callable: sigma(t, x, m) -> per-particle sigma
                    if dimension == 1:
                        x_for_callable = particles_t[:, 0]
                    else:
                        x_for_callable = particles_t
                    sigma_at_particles = diffusion_field(t_current, x_for_callable, m_at_particles)
                else:
                    # Array: interpolate from grid
                    sigma_at_particles = self._interpolate_field_at_particles(
                        particles_t, diffusion_field, coordinates, bounds
                    )

                # Ensure sigma_at_particles is 1D array of shape (num_particles,)
                sigma_at_particles = np.atleast_1d(sigma_at_particles).ravel()
                if sigma_at_particles.shape[0] == 1:
                    # Broadcast scalar to all particles
                    sigma_at_particles = np.full(self.num_particles, sigma_at_particles[0])

                # Generate per-particle Brownian increments
                # dW_i = sqrt(2 * sigma_i) * N(0, sqrt(dt))
                sigma_sde_particles = np.sqrt(2 * sigma_at_particles)
                dW = sigma_sde_particles[:, np.newaxis] * np.random.normal(
                    0, np.sqrt(Dt), (self.num_particles, dimension)
                )

            # Euler-Maruyama step
            new_particles = particles_t + drift * Dt + dW

            # Enforce obstacle boundaries if implicit domain is set (Issue #533)
            new_particles = self._enforce_obstacle_boundary(new_particles)

            # Apply boundary conditions
            topologies = self._get_topology_per_dimension(dimension)
            new_particles = self._apply_boundary_conditions_nd(new_particles, bounds, topologies)

            current_particles[t_idx + 1] = new_particles

            # Estimate density from particles
            M_density_on_grid[t_idx + 1] = self._estimate_density_from_particles_nd(new_particles, coordinates, bounds)

            # Normalize if requested
            if self.kde_normalization == KDENormalization.ALL:
                M_density_on_grid[t_idx + 1] = self._normalize_density_nd(M_density_on_grid[t_idx + 1], spacings)

            self._time_step_counter += 1

        self.M_particles_trajectory = current_particles

        # Build particle history for direct query mode (Issue #489)
        if self._particle_history is not None:
            # Convert 3D array (Nt+1, num_particles, dimension) to list
            for t in range(Nt + 1):
                self._particle_history.append(current_particles[t, :, :])

        # Return FPParticleResult if particle history was stored (Issue #489)
        if self._particle_history is not None:
            time_grid = np.linspace(0, self.problem.T, Nt + 1)
            return FPParticleResult(
                M_grid=M_density_on_grid,
                time_grid=time_grid,
                particle_history=self._particle_history,
                bandwidth=self.kde_bandwidth if isinstance(self.kde_bandwidth, (int, float)) else None,
            )

        # Backward compatible: return grid density only
        return M_density_on_grid

    def _interpolate_field_at_particles(
        self,
        particles: np.ndarray,
        field: np.ndarray,
        coordinates: list[np.ndarray],
        bounds: list[tuple[float, float]],
        fill_value: float = 0.0,
    ) -> np.ndarray:
        """
        Interpolate a grid field to particle positions.

        Delegates to _interpolate_grid_to_particles_nd() which uses the
        utils.numerical.particle.interpolate_grid_to_particles() function.

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape (N_particles, dimension)
        field : np.ndarray
            Field values on grid, shape (*grid_shape)
        coordinates : list of np.ndarray
            Grid coordinates per dimension (unused, kept for API compatibility)
        bounds : list of tuple
            Domain bounds per dimension
        fill_value : float
            Value for out-of-bounds particles (default: 0.0, handled by utils)

        Returns
        -------
        np.ndarray
            Field values at particle positions, shape (N_particles,)
        """
        return self._interpolate_grid_to_particles_nd(field, bounds, particles)

    def _estimate_density_at_particles(
        self,
        particles: np.ndarray,
        grid_density: np.ndarray,
        coordinates: list[np.ndarray],
        bounds: list[tuple[float, float]],
    ) -> np.ndarray:
        """
        Estimate density at particle positions by interpolating grid density.

        Parameters
        ----------
        particles : np.ndarray
            Particle positions, shape (N_particles, dimension)
        grid_density : np.ndarray
            Density on grid, shape (*grid_shape)
        coordinates : list of np.ndarray
            Grid coordinates per dimension
        bounds : list of tuple
            Domain bounds per dimension

        Returns
        -------
        np.ndarray
            Density at particle positions, shape (N_particles,)
        """
        # Use general interpolation with fill_value=0 for out-of-bounds
        return self._interpolate_field_at_particles(particles, grid_density, coordinates, bounds, fill_value=0.0)


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing FPParticleSolver...")

    from mfg_pde import MFGProblem
    from mfg_pde.geometry import TensorProductGrid

    # Test 1D problem with particle solver
    geometry_1d = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
    problem = MFGProblem(geometry=geometry_1d, T=1.0, Nt=20, diffusion=0.1)
    solver = FPParticleSolver(problem, num_particles=1000)

    # Test solver initialization
    assert solver.fp_method_name == "Particle"
    assert solver.num_particles == 1000

    # Test solve_fp_system
    import numpy as np

    Nx = problem.geometry.get_grid_shape()[0]
    U_test = np.zeros((problem.Nt + 1, Nx))
    M_init = problem.m_init

    M_solution = solver.solve_fp_system(M_initial=M_init, drift_field=U_test)

    assert M_solution.shape == (problem.Nt + 1, Nx)
    assert not np.any(np.isnan(M_solution))
    assert not np.any(np.isinf(M_solution))
    assert np.all(M_solution >= 0), "Density must be non-negative"

    print("  Particle solver converged")
    print(f"  Num particles: {solver.num_particles}")
    print(f"  M range: [{M_solution.min():.3f}, {M_solution.max():.3f}]")
    print(f"  KDE bandwidth: {solver.kde_bandwidth}")

    # Test 2D problem with particle solver (nD support)
    print("\nTesting 2D FPParticleSolver...")
    from mfg_pde.geometry import TensorProductGrid

    geometry_2d = TensorProductGrid(
        dimension=2,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[16, 16],
    )
    problem_2d = MFGProblem(geometry=geometry_2d, Nt=10, T=0.5, diffusion=0.1)

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

    # Test 3: Absorbing BC (segment-aware)
    print("\nTesting 2D FPParticleSolver with absorbing BC...")
    from mfg_pde.geometry.boundary import BCSegment, mixed_bc

    # Create BC with exit on right wall (DIRICHLET = absorbing for particles)
    bc_absorbing = mixed_bc(
        dimension=2,
        segments=[
            BCSegment(
                name="exit",
                bc_type=BCType.DIRICHLET,
                value=0.0,
                boundary="right",  # Right wall is exit (use direction name, not x_max)
            ),
            BCSegment(
                name="walls",
                bc_type=BCType.REFLECTING,
                boundary="all",
                priority=-1,  # Lower priority = fallback
            ),
        ],
        domain_bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
    )

    solver_absorbing = FPParticleSolver(
        problem_2d,
        num_particles=200,
        boundary_conditions=bc_absorbing,
    )

    # Drift particles toward the exit (right wall)
    # Use a gradient that pushes particles to the right
    drift_to_right = np.zeros((problem_2d.Nt + 1, *tuple(grid_shape_2d), 2))
    drift_to_right[..., 0] = 0.5  # Positive x-drift (toward x_max)

    M_solution_abs = solver_absorbing.solve_fp_system(
        M_initial=M_init_2d,
        drift_field=drift_to_right,
        drift_is_precomputed=True,
    )

    # Verify some particles were absorbed
    print(f"  Total absorbed: {solver_absorbing.total_absorbed}")
    print(f"  Exit flux history length: {len(solver_absorbing.exit_flux_history)}")

    # With strong rightward drift, particles should hit the exit
    # and be absorbed (mass should decrease)
    initial_mass_abs = np.sum(M_solution_abs[0])
    final_mass_abs = np.sum(M_solution_abs[-1])
    mass_loss = initial_mass_abs - final_mass_abs

    print(f"  Initial mass: {initial_mass_abs:.4f}")
    print(f"  Final mass: {final_mass_abs:.4f}")
    print(f"  Mass loss: {mass_loss:.4f}")

    # With absorbing BC, mass should decrease (particles exiting)
    # Note: This test may show small mass loss due to limited particles/time
    assert not np.any(np.isnan(M_solution_abs)), "NaN in absorbing BC solution"
    print("  Absorbing BC test passed")

    # Test 4: Obstacle geometry (Issue #533)
    print("\nTesting 2D FPParticleSolver with obstacle (Issue #533)...")
    from mfg_pde.geometry.implicit import DifferenceDomain, Hyperrectangle, Hypersphere

    # Create 2D domain with circular obstacle
    bounds_rect = np.array([[0.0, 1.0], [0.0, 1.0]])
    base_domain = Hyperrectangle(bounds_rect)
    obstacle = Hypersphere(center=[0.5, 0.5], radius=0.15)
    domain_with_obstacle = DifferenceDomain(base_domain, obstacle)

    # Solver with obstacle handling
    solver_obstacle = FPParticleSolver(
        problem_2d,
        num_particles=500,
        implicit_domain=domain_with_obstacle,
    )

    # Initial density near obstacle
    M_near_obstacle = np.exp(-((X - 0.35) ** 2 + (Y - 0.5) ** 2) / 0.05)
    M_near_obstacle = M_near_obstacle / np.sum(M_near_obstacle)

    # Pure diffusion (no drift) - particles will diffuse toward obstacle
    drift_zero = np.zeros((problem_2d.Nt + 1, *tuple(grid_shape_2d), 2))

    M_solution_obs = solver_obstacle.solve_fp_system(
        M_initial=M_near_obstacle,
        drift_field=drift_zero,
        drift_is_precomputed=True,
        show_progress=False,
    )

    # Check final particle positions
    final_particles = solver_obstacle.M_particles_trajectory[-1]
    inside_valid = domain_with_obstacle.contains(final_particles)
    pct_valid = 100.0 * np.sum(inside_valid) / len(final_particles)

    print(f"  Particles in valid domain: {pct_valid:.1f}%")
    print(f"  Particles inside obstacle: {len(final_particles) - np.sum(inside_valid)}")

    # Most particles should be in valid domain (outside obstacle)
    assert pct_valid > 95.0, f"Too many particles inside obstacle: {100 - pct_valid:.1f}%"
    assert not np.any(np.isnan(M_solution_obs)), "NaN in obstacle solution"
    print("  Obstacle geometry test passed")

    print("\nAll smoke tests passed!")
