from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

# Import MFGComponents and mixins from the dedicated module
from mfg_pde.core.mfg_components import (
    ConditionsMixin,
    HamiltonianMixin,
    MFGComponents,
)

# Use unified nD-capable BoundaryConditions from conditions.py

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfg_pde.geometry.protocol import GeometryProtocol
    from mfg_pde.types.pde_coefficients import DiffusionField, DriftField


# ============================================================================
# Unified MFG Problem Class
# ============================================================================


class MFGProblem(HamiltonianMixin, ConditionsMixin):
    """
    Unified MFG problem class that can handle both predefined and custom formulations.

    This class serves as the single constructor for all MFG problems:
    - Default usage: Uses built-in Hamiltonian (standard MFG formulation)
    - Custom usage: Accepts MFGComponents for full mathematical control

    Inherits from two mixins:

    HamiltonianMixin (mathematical Hamiltonian):
    - H(): Hamiltonian function
    - dH_dm(): Hamiltonian derivative w.r.t. density
    - get_hjb_hamiltonian_jacobian_contrib(): Jacobian for Newton methods
    - get_hjb_residual_m_coupling_term(): Coupling terms
    - get_potential_at_time(): Time-dependent potential accessor

    ConditionsMixin (problem setup):
    - get_boundary_conditions(): Boundary condition accessor
    - _setup_custom_initial_density(): Initial density setup
    - _setup_custom_final_value(): Final value setup
    """

    # Type annotations for geometry attributes (Phase 6 of Issue #435)
    # These are always non-None after __init__ completes
    geometry: GeometryProtocol
    hjb_geometry: GeometryProtocol | None
    fp_geometry: GeometryProtocol | None

    # Type annotations for PDE coefficient fields
    # sigma: float is the scalar diffusion for backward compatibility
    # diffusion_field: DiffusionField stores the full field (float, array, or callable)
    # drift_field: DriftField stores optional drift (float, array, or callable)
    sigma: float
    diffusion_field: DiffusionField
    drift_field: DriftField

    @staticmethod
    def _normalize_to_array(
        value: int | float | list[int] | list[float] | None,
        param_name: str = "parameter",
        warn: bool = True,
    ) -> list[int] | list[float] | None:
        """
        Convert scalar or array to array with optional deprecation warning.

        Args:
            value: Scalar or array value to normalize
            param_name: Parameter name for warning message
            warn: Whether to emit deprecation warning for scalar inputs

        Returns:
            Array form of the value, or None if input is None

        Examples:
            >>> MFGProblem._normalize_to_array(100, "Nx")  # Warns
            [100]
            >>> MFGProblem._normalize_to_array([100], "Nx")  # No warning
            [100]
            >>> MFGProblem._normalize_to_array(None, "Nx")
            None
        """
        import warnings

        if value is None:
            return None

        if isinstance(value, (int, float)):
            if warn:
                warnings.warn(
                    f"Passing scalar {param_name}={value} is deprecated. "
                    f"Use array notation {param_name}=[{value}] instead. "
                    f"Scalar support will be removed in v1.0.0. "
                    f"See docs/development/MATHEMATICAL_NOTATION_STANDARD.md for details.",
                    DeprecationWarning,
                    stacklevel=4,
                )
            return [value]

        # Already a list - return as-is
        return list(value)

    def __init__(
        self,
        # Legacy 1D parameters (backward compatible - scalars will be converted to arrays with deprecation warning)
        xmin: float | list[float] | None = None,
        xmax: float | list[float] | None = None,
        Nx: int | list[int] | None = None,
        Lx: float | None = None,  # Alternative to xmin/xmax
        # N-D grid parameters
        spatial_bounds: list[tuple[float, float]] | None = None,
        spatial_discretization: list[int] | None = None,
        # Complex geometry parameters (NEW)
        geometry: GeometryProtocol | None = None,
        obstacles: list | None = None,
        # Dual geometry parameters (Issue #257)
        hjb_geometry: GeometryProtocol | None = None,
        fp_geometry: GeometryProtocol | None = None,
        # Network parameters (NEW)
        network: Any | None = None,  # NetworkGraph
        # Time domain parameters
        T: float | None = None,
        Nt: int | None = None,
        time_domain: tuple[float, int] | None = None,  # Alternative to T/Nt
        # Physical parameters - support DiffusionField (float, array, or callable)
        diffusion: float | NDArray[np.floating] | Callable | None = None,  # Primary parameter
        sigma: float | NDArray[np.floating] | Callable | None = None,  # Legacy alias (deprecated)
        drift: float | NDArray[np.floating] | Callable | None = None,  # Optional drift field
        coupling_coefficient: float = 0.5,
        # MFG coupling parameters
        lambda_: float | None = None,  # Control cost (H uses |p|²/(2λ))
        gamma: float = 1.0,  # Density coupling strength (H uses -γm²)
        # Advanced
        components: MFGComponents | None = None,
        suppress_warnings: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize MFG problem with support for all spatial dimensions and domain types.

        Supports five initialization modes:
        1. Legacy 1D mode: Specify Nx, xmin, xmax
        2. N-D grid mode: Specify spatial_bounds, spatial_discretization
        3. Geometry mode: Specify geometry object (with optional obstacles)
        4. Network mode: Specify network graph
        5. Custom components: Full mathematical control via MFGComponents

        Args:
            xmin, xmax, Nx, Lx: Legacy 1D spatial domain parameters
            spatial_bounds: List of (min, max) tuples for each dimension
                           Example: [(0, 1), (0, 1)] for 2D unit square
            spatial_discretization: List of grid points per dimension
                                   Example: [50, 50] for 51×51 grid
            geometry: BaseGeometry object for complex domains (unified mode)
            obstacles: List of obstacle geometries
            hjb_geometry: Geometry for HJB solver (dual geometry mode, Issue #257)
            fp_geometry: Geometry for FP solver (dual geometry mode, Issue #257)
                        Note: Both hjb_geometry and fp_geometry must be specified together
            network: NetworkGraph for network MFG problems
            T, Nt, time_domain: Time domain parameters (T, Nt) or tuple (T, Nt)
            diffusion: Diffusion coefficient (primary parameter). None → 0 (deterministic).
                Supports:
                - None: No diffusion (deterministic dynamics)
                - float: Constant isotropic diffusion σ²
                - ndarray: Spatially/temporally varying diffusion
                - Callable: State-dependent σ(t, x, m) -> float | ndarray
            sigma: Legacy alias for diffusion (deprecated, use diffusion instead).
            drift: Drift field α(t, x, m) for FP equation. None → 0 (no drift).
                Supports:
                - None: No drift (no advection)
                - float: Constant drift (same in all directions)
                - ndarray: Precomputed drift array
                - Callable: State-dependent α(t, x, m) -> float | ndarray
            coupling_coefficient: Control cost coefficient
            components: Optional MFGComponents for custom problem definition
            suppress_warnings: Suppress computational feasibility warnings
            **kwargs: Additional parameters

        Examples:
            # Mode 1: 1D legacy (100% backward compatible)
            problem = MFGProblem(Nx=100, xmin=0.0, xmax=1.0, Nt=100)

            # Mode 2: N-D grid
            problem = MFGProblem(
                spatial_bounds=[(0, 1), (0, 1)],
                spatial_discretization=[50, 50],
                Nt=50
            )

            # Mode 3: Complex geometry with obstacles
            from mfg_pde.geometry import Hyperrectangle, Hypersphere
            domain = Hyperrectangle(bounds=[[0, 1], [0, 1]])
            obstacle = Hypersphere(center=[0.5, 0.5], radius=0.1)
            problem = MFGProblem(
                geometry=domain,
                obstacles=[obstacle],
                time_domain=(1.0, 50),
                diffusion=0.1
            )

            # Mode 4: Network MFG
            import networkx as nx
            graph = nx.grid_2d_graph(10, 10)
            problem = MFGProblem(network=graph, time_domain=(1.0, 100))

            # Mode 5: Custom components
            components = MFGComponents(hamiltonian_func=..., ...)
            problem = MFGProblem(
                spatial_bounds=[(0, 1)],
                spatial_discretization=[100],
                Nt=50,
                components=components
            )

            # Mode 6: Dual geometry (Issue #257) - Separate geometries for HJB and FP
            from mfg_pde.geometry import TensorProductGrid
            hjb_grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx_points=[51, 51])  # Fine grid for HJB
            fp_grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx_points=[21, 21])  # Coarse grid for FP
            problem = MFGProblem(
                hjb_geometry=hjb_grid,
                fp_geometry=fp_grid,
                time_domain=(1.0, 50),
                diffusion=0.1
            )
            # Automatically creates geometry_projector for mapping between geometries

            # Advanced: State-dependent diffusion (callable)
            def density_dependent_diffusion(t, x, m):
                return 0.1 * (1 + m)  # Higher diffusion in dense regions
            problem = MFGProblem(
                geometry=domain,
                sigma=density_dependent_diffusion,
                time_domain=(1.0, 50)
            )

            # Advanced: Spatially varying diffusion (array)
            sigma_array = np.ones((51, 51)) * 0.1  # Base diffusion
            sigma_array[20:30, 20:30] = 0.5  # Higher diffusion in center region
            problem = MFGProblem(
                geometry=domain,
                sigma=sigma_array,
                time_domain=(1.0, 50)
            )

            # Advanced: Custom drift field
            def crowd_avoidance_drift(t, x, m):
                grad_m = np.gradient(m)  # Density gradient
                return -np.stack(grad_m, axis=-1)  # Move down gradient
            problem = MFGProblem(
                geometry=domain,
                drift=crowd_avoidance_drift,
                time_domain=(1.0, 50)
            )
        """
        import warnings

        # Normalize parameter aliases
        if time_domain is not None:
            if T is not None or Nt is not None:
                raise ValueError("Specify EITHER (T, Nt) OR time_domain, not both")
            T, Nt = time_domain

        # Handle sigma as legacy alias for diffusion
        if sigma is not None:
            if diffusion is not None:
                raise ValueError("Specify EITHER diffusion OR sigma, not both")
            warnings.warn(
                "Parameter 'sigma' is deprecated. Use 'diffusion' instead. 'sigma' will be removed in v1.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            diffusion = sigma

        # Set defaults for T, Nt if not provided
        if T is None:
            T = 1.0
        if Nt is None:
            Nt = 51

        # Convert None to 0 for diffusion and drift (None = no diffusion/drift)
        if diffusion is None:
            diffusion = 0.0
        if drift is None:
            drift = 0.0

        # Store the full diffusion field (may be float, array, or callable)
        # self.sigma will be the scalar/default for backward compatibility
        # self.diffusion_field stores the full field for advanced solvers
        self.diffusion_field = diffusion
        self.drift_field = drift

        # Extract scalar sigma for backward compatibility
        # If diffusion is callable or array, use a representative scalar value
        if callable(diffusion):
            # Callable: store 1.0 as default, solvers should use diffusion_field
            sigma_scalar = 1.0
        elif isinstance(diffusion, np.ndarray):
            # Array: use mean value as representative scalar
            sigma_scalar = float(np.mean(diffusion))
        else:
            # Scalar: use directly
            sigma_scalar = float(diffusion)

        # Normalize spatial parameters to arrays (with deprecation warnings for scalars)
        # This enables dimension-agnostic code while maintaining backward compatibility
        if Nx is not None:
            Nx_normalized = self._normalize_to_array(Nx, "Nx")
        else:
            Nx_normalized = None

        if xmin is not None:
            xmin_normalized = self._normalize_to_array(xmin, "xmin")
        else:
            xmin_normalized = None

        if xmax is not None:
            xmax_normalized = self._normalize_to_array(xmax, "xmax")
        else:
            xmax_normalized = None

        # Handle dual geometry specification (Issue #257)
        self.geometry_projector = None  # Will be set if dual geometries provided

        if hjb_geometry is not None and fp_geometry is not None:
            # Dual geometry mode: separate geometries for HJB and FP
            if geometry is not None:
                raise ValueError(
                    "Specify EITHER 'geometry' (unified) OR ('hjb_geometry', 'fp_geometry') (dual), not both"
                )
            # Use dual geometries
            final_hjb_geometry = hjb_geometry
            final_fp_geometry = fp_geometry
            # Create projector for mapping between geometries
            from mfg_pde.geometry import GeometryProjector

            self.geometry_projector = GeometryProjector(
                hjb_geometry=hjb_geometry, fp_geometry=fp_geometry, projection_method="auto"
            )
        elif hjb_geometry is not None or fp_geometry is not None:
            # Partial dual geometry specification
            raise ValueError("If using dual geometries, both 'hjb_geometry' AND 'fp_geometry' must be specified")
        elif geometry is not None:
            # Unified geometry mode (backward compatible)
            final_hjb_geometry = geometry
            final_fp_geometry = geometry
        else:
            # No explicit geometry provided - will be handled by mode detection
            final_hjb_geometry = None
            final_fp_geometry = None

        # Detect initialization mode (use normalized Nx for detection)
        # For dual geometry, pass the hjb_geometry to mode detection
        geometry_for_detection = final_hjb_geometry if final_hjb_geometry is not None else geometry
        mode = self._detect_init_mode(
            Nx=Nx_normalized, spatial_bounds=spatial_bounds, geometry=geometry_for_detection, network=network
        )

        # Dispatch to appropriate initializer
        # Note: Pass sigma_scalar (the backward-compatible float value)
        if mode == "1d_legacy":
            # Mode 1: Legacy 1D
            if Lx is not None:
                # Use Lx to set xmin/xmax if provided
                if xmin_normalized is None:
                    xmin_normalized = [0.0]
                xmax_normalized = [xmin_normalized[0] + Lx]
            else:
                if xmin_normalized is None:
                    xmin_normalized = [0.0]
                if xmax_normalized is None:
                    xmax_normalized = [1.0]
            self._init_1d_legacy(
                xmin_normalized, xmax_normalized, Nx_normalized, T, Nt, sigma_scalar, coupling_coefficient
            )

        elif mode == "nd_grid":
            # Mode 2: N-dimensional grid
            self._init_nd(
                spatial_bounds, spatial_discretization, T, Nt, sigma_scalar, coupling_coefficient, suppress_warnings
            )

        elif mode == "geometry":
            # Mode 3: Complex geometry
            self._init_geometry(
                final_hjb_geometry,
                obstacles,
                T,
                Nt,
                sigma_scalar,
                coupling_coefficient,
                lambda_,
                gamma,
                suppress_warnings,
            )
            # For dual geometry mode, store both geometries explicitly
            if self.geometry_projector is not None:
                self.hjb_geometry = final_hjb_geometry
                self.fp_geometry = final_fp_geometry

        elif mode == "network":
            # Mode 4: Network MFG
            self._init_network(network, T, Nt, sigma_scalar, coupling_coefficient, lambda_, gamma)

        elif mode == "default":
            # Default: 1D with default parameters
            warnings.warn(
                "No spatial domain specified. Using default 1D domain: [0, 1] with 51 points.",
                UserWarning,
                stacklevel=2,
            )
            self._init_1d_legacy([0.0], [1.0], [51], T, Nt, sigma_scalar, coupling_coefficient)

        else:
            raise ValueError(f"Unknown initialization mode: {mode}")

        # Store dual geometries (Issue #257)
        # For unified mode, both point to self.geometry (set by init methods)
        # For dual mode, these were already computed above
        if not hasattr(self, "hjb_geometry"):
            self.hjb_geometry = getattr(self, "geometry", None)
            self.fp_geometry = getattr(self, "geometry", None)

        # Store custom components if provided
        self.components = components
        self.is_custom = components is not None

        # Merge parameters
        if self.is_custom and self.components is not None:
            all_params = {**self.components.parameters, **kwargs}
        else:
            all_params = kwargs

        # Initialize arrays
        self.f_potential: NDArray
        self.u_fin: NDArray
        self.m_init: NDArray

        # Initialize functions
        self._initialize_functions(**all_params)

        # Validate custom components if provided
        if self.is_custom:
            self._validate_hamiltonian_components()

        # Detect solver compatibility
        self._detect_solver_compatibility()

    def _init_1d_legacy(
        self,
        xmin: list[float],
        xmax: list[float],
        Nx: list[int],
        T: float,
        Nt: int,
        sigma: float,
        coupling_coefficient: float,
    ) -> None:
        """
        Initialize problem in legacy 1D mode (100% backward compatible).

        Args:
            xmin: Lower bound as array (e.g., [-2.0])
            xmax: Upper bound as array (e.g., [2.0])
            Nx: Grid points as array (e.g., [100])
            T: Terminal time
            Nt: Temporal grid points
            sigma: Diffusion coefficient
            coupling_coefficient: Control cost coefficient

        Note:
            This manual grid construction pattern is deprecated. Consider using
            the geometry-first API with TensorProductGrid instead.
            See migration guide: docs/migration/GEOMETRY_PARAMETER_MIGRATION.md
        """
        import warnings

        from mfg_pde.geometry import TensorProductGrid

        # Emit deprecation warning for manual grid construction pattern
        warnings.warn(
            "Manual grid construction in MFGProblem is deprecated and will be "
            "restricted in v1.0.0. Use the geometry-first API instead:\n\n"
            "  from mfg_pde.geometry import TensorProductGrid\n"
            f"  domain = TensorProductGrid(dimension=1, bounds=[({xmin[0]}, {xmax[0]})], Nx_points=[{Nx[0] + 1}])\n"
            f"  problem = MFGProblem(geometry=domain, T={T}, Nt={Nt})\n\n"
            "See docs/migration/GEOMETRY_PARAMETER_MIGRATION.md for details.",
            DeprecationWarning,
            stacklevel=4,  # Point to user's code, not internal calls
        )

        # Extract scalar values from arrays for backward compatibility
        xmin_scalar = xmin[0]
        xmax_scalar = xmax[0]
        Nx_scalar = Nx[0]

        # Create TensorProductGrid geometry object (unified internal representation)
        geometry = TensorProductGrid(
            dimension=1,
            bounds=[(xmin_scalar, xmax_scalar)],
            Nx_points=[Nx_scalar + 1],
        )

        # Store geometry for unified interface
        self.geometry = geometry

        # Set dimension from geometry
        self.dimension = geometry.dimension

        # Time domain
        self.T: float = T
        self.Nt: int = Nt
        self.dt: float = T / Nt if Nt > 0 else 0.0

        # Time grid
        self.tSpace: np.ndarray = np.linspace(0, T, Nt + 1, endpoint=True)

        # Coefficients
        self.sigma: float = sigma
        self.coupling_coefficient: float = coupling_coefficient

        # N-D attributes (derived from geometry)
        self.spatial_shape = geometry.get_grid_shape()
        self.spatial_bounds = [(xmin_scalar, xmax_scalar)]
        self.spatial_discretization = [Nx_scalar]

        # Set domain type
        self.domain_type = "grid"

    def _init_nd(
        self,
        spatial_bounds: list[tuple[float, float]],
        spatial_discretization: list[int] | None,
        T: float,
        Nt: int,
        sigma: float,
        coupling_coefficient: float,
        suppress_warnings: bool,
    ) -> None:
        """
        Initialize problem in n-dimensional mode.

        Note:
            This manual grid construction pattern is deprecated. Consider using
            the geometry-first API with TensorProductGrid instead.
            See migration guide: docs/migration/GEOMETRY_PARAMETER_MIGRATION.md
        """
        import warnings

        # Validate inputs
        if not spatial_bounds:
            raise ValueError("spatial_bounds must be a non-empty list of (min, max) tuples")

        dimension = len(spatial_bounds)

        # Emit deprecation warning for manual grid construction pattern
        warnings.warn(
            "Manual grid construction in MFGProblem is deprecated and will be "
            "restricted in v1.0.0. Use the geometry-first API instead:\n\n"
            "  from mfg_pde.geometry import TensorProductGrid\n"
            "  geometry = TensorProductGrid(\n"
            f"      dimension={dimension},\n"
            f"      bounds={spatial_bounds},\n"
            f"      Nx_points={spatial_discretization if spatial_discretization else [51] * dimension}\n"
            "  )\n"
            f"  problem = MFGProblem(geometry=geometry, T={T}, Nt={Nt})\n\n"
            "See docs/migration/GEOMETRY_PARAMETER_MIGRATION.md for details.",
            DeprecationWarning,
            stacklevel=4,  # Point to user's code, not internal calls
        )

        if spatial_discretization is None:
            # Default: 51 points per dimension
            spatial_discretization = [51] * dimension
        elif len(spatial_discretization) != dimension:
            raise ValueError(
                f"spatial_discretization must have {dimension} elements (one per dimension), "
                f"got {len(spatial_discretization)}"
            )

        # Create TensorProductGrid for all dimensions (unified approach)
        from mfg_pde.geometry import TensorProductGrid

        # Convert discretization to Nx_points (add 1 for point count vs intervals)
        Nx_points = [n + 1 for n in spatial_discretization]
        geometry = TensorProductGrid(dimension=dimension, bounds=spatial_bounds, Nx_points=Nx_points)

        # Store geometry for unified interface
        self.geometry = geometry

        # Set dimension from geometry
        self.dimension = geometry.dimension

        # Store n-D parameters
        self.spatial_bounds = spatial_bounds
        self.spatial_discretization = spatial_discretization

        # Spatial shape from geometry (actual grid size, not discretization)
        # For grids with resolution N, actual points = N+1 per dimension
        if dimension == 1:
            self.spatial_shape = (spatial_discretization[0] + 1,)
        elif dimension == 2 or dimension == 3:
            self.spatial_shape = tuple(n + 1 for n in spatial_discretization)
        else:
            # For TensorProductGrid, use num_spatial_points from geometry
            self.spatial_shape = (geometry.num_spatial_points,)

        # Time domain
        self.T: float = T
        self.Nt: int = Nt
        self.dt: float = T / Nt if Nt > 0 else 0.0
        self.tSpace: np.ndarray = np.linspace(0, T, Nt + 1, endpoint=True)

        # Coefficients
        self.sigma: float = sigma
        self.coupling_coefficient: float = coupling_coefficient

        # Set domain type
        self.domain_type = "grid"

        # Check computational feasibility and warn if needed
        if not suppress_warnings:
            self._check_computational_feasibility()

    def _check_computational_feasibility(self) -> None:
        """Warn about computational limits for high-dimensional problems."""
        import warnings

        MAX_PRACTICAL_DIMENSION = 4
        MAX_TOTAL_GRID_POINTS = 10_000_000  # 10 million

        # Calculate total grid points
        total_spatial_points = int(np.prod(self.spatial_shape))
        total_points = total_spatial_points * (self.Nt + 1)
        memory_mb = total_points * 8 / (1024**2)  # Assuming float64

        if self.dimension > MAX_PRACTICAL_DIMENSION:
            warnings.warn(
                f"\n{'=' * 80}\n"
                f"HIGH DIMENSION WARNING\n"
                f"{'=' * 80}\n"
                f"Problem dimension: {self.dimension}D\n"
                f"Practical limit for grid-based FDM: {MAX_PRACTICAL_DIMENSION}D\n"
                f"\n"
                f"Grid-based methods scale as O(N^d), becoming impractical for high dimensions.\n"
                f"Your problem will require:\n"
                f"  - Spatial points: {total_spatial_points:,}\n"
                f"  - Total points (space × time): {total_points:,}\n"
                f"  - Estimated memory: {memory_mb:,.1f} MB per array\n"
                f"\n"
                f"RECOMMENDATION:\n"
                f"For dimension > {MAX_PRACTICAL_DIMENSION}, consider alternative methods:\n"
                f"  - Particle-based collocation methods (algorithms/particle_collocation)\n"
                f"  - Network MFG formulations (for very high dimensions)\n"
                f"  - Dimension reduction techniques\n"
                f"\n"
                f"To suppress this warning: MFGProblem(..., suppress_warnings=True)\n"
                f"{'=' * 80}",
                UserWarning,
                stacklevel=3,
            )
        elif total_points > MAX_TOTAL_GRID_POINTS:
            warnings.warn(
                f"\n{'=' * 80}\n"
                f"MEMORY WARNING\n"
                f"{'=' * 80}\n"
                f"Problem requires {total_points:,} grid points ({memory_mb:,.1f} MB per array).\n"
                f"This may cause memory issues on typical machines.\n"
                f"\n"
                f"Consider:\n"
                f"  - Reducing spatial discretization\n"
                f"  - Reducing time steps\n"
                f"  - Using sparse storage methods\n"
                f"\n"
                f"To suppress this warning: MFGProblem(..., suppress_warnings=True)\n"
                f"{'=' * 80}",
                UserWarning,
                stacklevel=3,
            )

    def _detect_init_mode(
        self,
        Nx: list[int] | None,
        spatial_bounds: list[tuple[float, float]] | None,
        geometry: GeometryProtocol | None,
        network: Any | None,
    ) -> str:
        """
        Detect which initialization mode to use based on provided parameters.

        Args:
            Nx: Normalized array of grid points (or None)
            spatial_bounds: Spatial bounds (or None)
            geometry: Geometry object (or None)
            network: Network object (or None)

        Returns:
            mode: One of "1d_legacy", "nd_grid", "geometry", "network", "default"

        Raises:
            ValueError: If parameters are ambiguous or conflicting
        """
        # Count how many modes are specified
        mode_indicators = {
            "1d_legacy": Nx is not None,
            "nd_grid": spatial_bounds is not None,
            "geometry": geometry is not None,
            "network": network is not None,
        }

        num_modes = sum(mode_indicators.values())

        if num_modes == 0:
            return "default"
        elif num_modes > 1:
            specified = [k for k, v in mode_indicators.items() if v]
            raise ValueError(
                f"Ambiguous initialization: Multiple modes specified: {specified}\n"
                f"Provide ONLY ONE of:\n"
                f"  - Nx (for 1D legacy mode)\n"
                f"  - spatial_bounds (for n-D grid mode)\n"
                f"  - geometry (for complex geometry mode)\n"
                f"  - network (for network MFG mode)"
            )
        else:
            # Exactly one mode specified
            for mode, is_set in mode_indicators.items():
                if is_set:
                    return mode

        # Should never reach here
        return "default"

    def _init_geometry(
        self,
        geometry: GeometryProtocol,
        obstacles: list | None,
        T: float,
        Nt: int,
        sigma: float,
        coupling_coefficient: float,
        lambda_: float | None,
        gamma: float,
        suppress_warnings: bool,
    ) -> None:
        """
        Initialize problem with geometry object implementing GeometryProtocol.

        Accepts any geometry type: TensorProductGrid, BaseGeometry,
        ImplicitDomain, NetworkGeometry, etc.

        Args:
            geometry: Any object implementing GeometryProtocol
            obstacles: List of obstacle geometries (for domain geometries)
            T, Nt: Time domain parameters
            sigma, coupling_coefficient: Physical parameters
            lambda_, gamma: MFG coupling parameters
            suppress_warnings: Suppress warnings
        """
        # Import geometry protocol
        try:
            from mfg_pde.geometry import GeometryProtocol, validate_geometry
        except ImportError as err:
            raise ImportError(
                "Geometry mode requires geometry module. Install with: pip install mfg_pde[geometry]"
            ) from err

        # Validate geometry object implements GeometryProtocol
        if not isinstance(geometry, GeometryProtocol):
            raise TypeError(
                f"geometry must implement GeometryProtocol, got {type(geometry)}. "
                f"Use TensorProductGrid, BaseGeometry, ImplicitDomain, or NetworkGeometry."
            )

        # Validate geometry is properly implemented
        validate_geometry(geometry)

        # Store geometry
        self.geometry = geometry
        self.dimension = geometry.dimension
        self.obstacles = obstacles or []
        self.has_obstacles = len(self.obstacles) > 0

        # Time domain
        self.T = T
        self.Nt = Nt
        self.dt = T / Nt if Nt > 0 else 0.0  # Lowercase (official naming convention)
        self.tSpace = np.linspace(0, T, Nt + 1, endpoint=True)

        # Physical parameters
        self.sigma = sigma  # Already sigma_scalar from __init__ dispatch
        self.coupling_coefficient = coupling_coefficient

        # MFG coupling parameters (for custom Hamiltonians)
        self.lambda_ = lambda_
        self.gamma = gamma

        # Initialize spatial discretization based on geometry type
        from mfg_pde.geometry import GeometryType

        if geometry.geometry_type == GeometryType.CARTESIAN_GRID:
            # CARTESIAN_GRID: Can be TensorProductGrid or AMR mesh
            # Use polymorphic method to get configuration
            config = geometry.get_problem_config()

            # Apply configuration from geometry
            self.num_spatial_points = config["num_spatial_points"]
            self.spatial_shape = config["spatial_shape"]
            self.spatial_bounds = config["spatial_bounds"]
            self.spatial_discretization = config["spatial_discretization"]

            self.domain_type = "grid"

        elif geometry.geometry_type in (GeometryType.DOMAIN_2D, GeometryType.DOMAIN_3D):
            # BaseGeometry - unstructured mesh via Gmsh
            self.mesh_data = geometry.generate_mesh()
            self.collocation_points = self.mesh_data.vertices
            self.num_spatial_points = len(self.collocation_points)

            # Set spatial shape and bounds
            self.spatial_shape = (self.num_spatial_points,)  # Unstructured
            self.spatial_bounds = None  # Not a regular grid
            self.spatial_discretization = None

            self.domain_type = "mesh"

        elif geometry.geometry_type == GeometryType.IMPLICIT:
            # ImplicitDomain - point cloud from SDF
            self.num_spatial_points = geometry.num_spatial_points
            self.collocation_points = geometry.get_spatial_grid()
            self.spatial_shape = (self.num_spatial_points,)
            self.spatial_bounds = geometry.get_bounding_box()
            self.spatial_discretization = None

            self.domain_type = "implicit"

        elif geometry.geometry_type in (GeometryType.MAZE, GeometryType.NETWORK):
            # Graph-based geometries (mazes, networks)
            config = geometry.get_problem_config()
            self.num_spatial_points = config["num_spatial_points"]
            self.collocation_points = geometry.get_spatial_grid()
            self.spatial_shape = config["spatial_shape"]
            self.spatial_bounds = config.get("spatial_bounds")
            self.spatial_discretization = config.get("spatial_discretization")

            # Store graph-specific data if available
            if "graph_data" in config:
                self.graph_data = config["graph_data"]

            self.domain_type = str(geometry.geometry_type.value)

        else:
            # Generic GeometryProtocol object - use spatial grid
            self.num_spatial_points = geometry.num_spatial_points
            self.collocation_points = geometry.get_spatial_grid()
            self.spatial_shape = (self.num_spatial_points,)
            self.spatial_bounds = None
            self.spatial_discretization = None

            self.domain_type = str(geometry.geometry_type.value)

    def _init_network(
        self,
        network: Any,
        T: float,
        Nt: int,
        sigma: float,
        coupling_coefficient: float,
        lambda_: float | None,
        gamma: float,
    ) -> None:
        """
        Initialize problem on network/graph.

        Args:
            network: NetworkGraph or networkx.Graph
            T, Nt: Time domain parameters
            sigma, coupling_coefficient: Physical parameters
        """
        # Import CustomNetwork for geometry-first API
        from mfg_pde.geometry.graph import CustomNetwork

        # Store network
        self.network = network
        self.dimension = "network"  # Special dimension indicator
        self.domain_type = "network"

        # Create CustomNetwork geometry from the network
        try:
            import networkx as nx

            if isinstance(network, nx.Graph):
                # Create geometry from networkx graph
                geometry = CustomNetwork.from_networkx(network)
                self.num_nodes = network.number_of_nodes()
                self.adjacency_matrix = nx.adjacency_matrix(network).toarray()
            else:
                # Assume custom NetworkGraph type with adjacency_matrix attribute
                self.num_nodes = len(network.nodes)
                self.adjacency_matrix = network.adjacency_matrix
                # Create geometry from adjacency matrix
                geometry = CustomNetwork(network.adjacency_matrix)
        except ImportError:
            # NetworkX not available - assume custom type
            self.num_nodes = len(network.nodes)
            self.adjacency_matrix = network.adjacency_matrix
            # Create geometry from adjacency matrix
            geometry = CustomNetwork(network.adjacency_matrix)

        # Store geometry (geometry-first API: never None)
        self.geometry = geometry

        # Time domain
        self.T = T
        self.Nt = Nt
        self.dt = T / Nt if Nt > 0 else 0.0  # Lowercase (official naming convention)
        self.tSpace = np.linspace(0, T, Nt + 1, endpoint=True)

        # Physical parameters
        self.sigma = sigma  # Already sigma_scalar from __init__ dispatch
        self.coupling_coefficient = coupling_coefficient

        # MFG coupling parameters (for custom Hamiltonians)
        self.lambda_ = lambda_
        self.gamma = gamma

        # Spatial discretization (nodes)
        self.spatial_shape = (self.num_nodes,)
        self.num_spatial_points = self.num_nodes  # For networks, spatial points = nodes
        self.spatial_bounds = None
        self.spatial_discretization = None
        self.obstacles = None
        self.has_obstacles = False

    # =========================================================================
    # Geometry Type Helper Properties (Phase 2 of Issue #435)
    # =========================================================================

    @property
    def is_network(self) -> bool:
        """
        Check if this problem is defined on a network/graph domain.

        Returns:
            True if domain_type is "network", False otherwise.

        Example:
            >>> import networkx as nx
            >>> G = nx.grid_2d_graph(5, 5)
            >>> problem = MFGProblem(network=G, T=1.0, Nt=10)
            >>> problem.is_network
            True
        """
        return getattr(self, "domain_type", None) == "network"

    @property
    def is_cartesian(self) -> bool:
        """
        Check if this problem is defined on a Cartesian grid domain.

        Returns:
            True if domain_type is "grid", False otherwise.

        Example:
            >>> problem = MFGProblem(Nx=[50], xmin=[0.0], xmax=[1.0], T=1.0, Nt=10)
            >>> problem.is_cartesian
            True
        """
        return getattr(self, "domain_type", None) == "grid"

    @property
    def is_implicit(self) -> bool:
        """
        Check if this problem uses an implicit/complex geometry.

        Returns:
            True if domain_type is "implicit", False otherwise.

        Example:
            >>> from mfg_pde.geometry import ImplicitDomain
            >>> domain = ImplicitDomain(...)  # Complex geometry
            >>> problem = MFGProblem(geometry=domain, T=1.0, Nt=10)
            >>> problem.is_implicit
            True
        """
        return getattr(self, "domain_type", None) == "implicit"

    # =========================================================================
    # Time Grid Properties
    # =========================================================================

    @property
    def Nt_points(self) -> int:
        """
        Number of time grid points (Nt + 1).

        Nt is the number of time intervals, while Nt_points is the number
        of time grid points including both endpoints.

        Returns:
            Nt + 1 (number of time points)

        Example:
            >>> problem = MFGProblem(geometry=domain, T=1.0, Nt=10)
            >>> problem.Nt         # 10 intervals
            10
            >>> problem.Nt_points  # 11 points
            11
        """
        return self.Nt + 1

    # =========================================================================
    # Deprecated Legacy Attributes (Computed Properties)
    # Phase 7 of Issue #435: These are computed from geometry for backward
    # compatibility. Access emits DeprecationWarning.
    # =========================================================================

    @property
    def xmin(self) -> float | None:
        """
        DEPRECATED: Use problem.geometry.get_bounds() instead.

        Returns the minimum x-coordinate for 1D problems.
        """
        import warnings

        warnings.warn(
            "Accessing 'xmin' is deprecated. Use 'problem.geometry.get_bounds()[0][0]' instead. "
            "This attribute will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Check for stored value first (for backward compat with tests that set it)
        if hasattr(self, "_xmin_override"):
            return self._xmin_override
        if self.geometry is not None and self.dimension == 1:
            bounds = self.geometry.get_bounds()
            if bounds is not None:
                return float(bounds[0][0])
        return None

    @xmin.setter
    def xmin(self, value: float | None) -> None:
        """Allow setting for backward compatibility (with warning)."""
        import warnings

        warnings.warn(
            "Setting 'xmin' is deprecated. Use geometry-first API instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._xmin_override = value

    @property
    def xmax(self) -> float | None:
        """
        DEPRECATED: Use problem.geometry.get_bounds() instead.

        Returns the maximum x-coordinate for 1D problems.
        """
        import warnings

        warnings.warn(
            "Accessing 'xmax' is deprecated. Use 'problem.geometry.get_bounds()[1][0]' instead. "
            "This attribute will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        if hasattr(self, "_xmax_override"):
            return self._xmax_override
        if self.geometry is not None and self.dimension == 1:
            bounds = self.geometry.get_bounds()
            if bounds is not None:
                return float(bounds[1][0])
        return None

    @xmax.setter
    def xmax(self, value: float | None) -> None:
        """Allow setting for backward compatibility (with warning)."""
        import warnings

        warnings.warn(
            "Setting 'xmax' is deprecated. Use geometry-first API instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._xmax_override = value

    @property
    def Lx(self) -> float | None:
        """
        DEPRECATED: Compute from geometry bounds instead.

        Returns the domain length for 1D problems.
        """
        import warnings

        warnings.warn(
            "Accessing 'Lx' is deprecated. Compute from geometry bounds instead. "
            "This attribute will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        if hasattr(self, "_Lx_override"):
            return self._Lx_override
        if self.geometry is not None and self.dimension == 1:
            bounds = self.geometry.get_bounds()
            if bounds is not None:
                return float(bounds[1][0] - bounds[0][0])
        return None

    @Lx.setter
    def Lx(self, value: float | None) -> None:
        """Allow setting for backward compatibility (with warning)."""
        import warnings

        warnings.warn(
            "Setting 'Lx' is deprecated. Use geometry-first API instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._Lx_override = value

    @property
    def Nx(self) -> int | None:
        """
        DEPRECATED: Use problem.geometry.num_spatial_points instead.

        Returns the number of intervals (not points) for 1D problems.
        """
        import warnings

        warnings.warn(
            "Accessing 'Nx' is deprecated. Use 'problem.geometry.num_spatial_points - 1' for intervals. "
            "This attribute will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        if hasattr(self, "_Nx_override"):
            return self._Nx_override
        if self.geometry is not None and self.dimension == 1:
            # Nx is number of intervals, num_spatial_points is number of points
            return self.geometry.num_spatial_points - 1
        return None

    @Nx.setter
    def Nx(self, value: int | None) -> None:
        """Allow setting for backward compatibility (with warning)."""
        import warnings

        warnings.warn(
            "Setting 'Nx' is deprecated. Use geometry-first API instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._Nx_override = value

    @property
    def dx(self) -> float | None:
        """
        DEPRECATED: Compute from geometry bounds and num_points instead.

        Returns the grid spacing for 1D problems.
        """
        import warnings

        warnings.warn(
            "Accessing 'dx' is deprecated. Compute from geometry bounds and num_points instead. "
            "This attribute will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        if hasattr(self, "_dx_override"):
            return self._dx_override
        if self.geometry is not None and self.dimension == 1:
            from mfg_pde.geometry import TensorProductGrid

            if isinstance(self.geometry, TensorProductGrid):
                return float(self.geometry.spacing[0])
            else:
                # Compute from bounds
                bounds = self.geometry.get_bounds()
                if bounds is not None:
                    n_points = self.geometry.num_spatial_points
                    if n_points > 1:
                        return float((bounds[1][0] - bounds[0][0]) / (n_points - 1))
        return None

    @dx.setter
    def dx(self, value: float | None) -> None:
        """Allow setting for backward compatibility (with warning)."""
        import warnings

        warnings.warn(
            "Setting 'dx' is deprecated. Use geometry-first API instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._dx_override = value

    @property
    def xSpace(self) -> np.ndarray | None:
        """
        DEPRECATED: Use problem.geometry.get_spatial_grid() instead.

        Returns the spatial grid array.
        """
        import warnings

        warnings.warn(
            "Accessing 'xSpace' is deprecated. Use 'problem.geometry.get_spatial_grid()' instead. "
            "This attribute will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        if hasattr(self, "_xSpace_override"):
            return self._xSpace_override
        if self.geometry is not None:
            return self.geometry.get_spatial_grid()
        return None

    @xSpace.setter
    def xSpace(self, value: np.ndarray | None) -> None:
        """Allow setting for backward compatibility (with warning)."""
        import warnings

        warnings.warn(
            "Setting 'xSpace' is deprecated. Use geometry-first API instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._xSpace_override = value

    @property
    def _grid(self) -> Any:
        """
        DEPRECATED: Use problem.geometry instead.

        Returns the geometry object (for backward compatibility).
        """
        import warnings

        warnings.warn(
            "Accessing '_grid' is deprecated. Use 'problem.geometry' instead. "
            "This attribute will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        if hasattr(self, "_grid_override"):
            return self._grid_override
        return self.geometry

    @_grid.setter
    def _grid(self, value: Any) -> None:
        """Allow setting for backward compatibility (with warning)."""
        import warnings

        warnings.warn(
            "Setting '_grid' is deprecated. Use geometry-first API instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._grid_override = value

    # =========================================================================
    # Internal Geometry Helpers (no deprecation warnings)
    # These are for internal use only - external code should use geometry directly
    # =========================================================================

    def _get_domain_length(self) -> float | None:
        """Get domain length for 1D problems (internal use, no warning)."""
        if self.geometry is not None and self.dimension == 1:
            bounds = self.geometry.get_bounds()
            if bounds is not None:
                return float(bounds[1][0] - bounds[0][0])
        return None

    def _get_spacing(self) -> float | None:
        """Get grid spacing for 1D problems (internal use, no warning)."""
        if self.geometry is not None and self.dimension == 1:
            from mfg_pde.geometry import TensorProductGrid

            if isinstance(self.geometry, TensorProductGrid):
                return float(self.geometry.spacing[0])
            else:
                bounds = self.geometry.get_bounds()
                if bounds is not None:
                    n_points = self.geometry.num_spatial_points
                    if n_points > 1:
                        return float((bounds[1][0] - bounds[0][0]) / (n_points - 1))
        return None

    def _get_num_intervals(self) -> int | None:
        """Get number of intervals for 1D problems (internal use, no warning)."""
        if self.geometry is not None and self.dimension == 1:
            return self.geometry.num_spatial_points - 1
        return None

    def _get_spatial_grid_internal(self) -> np.ndarray | None:
        """Get spatial grid array (internal use, no warning)."""
        if self.geometry is not None:
            return self.geometry.get_spatial_grid()
        return None

    # =========================================================================
    # PDE Coefficient Field Helpers
    # =========================================================================

    def get_diffusion_coefficient_field(self) -> Any:
        """
        Get a CoefficientField wrapper for the diffusion coefficient.

        Returns a CoefficientField that handles scalar, array, and callable
        diffusion coefficients uniformly. Use this in solvers instead of
        directly accessing self.sigma.

        Returns:
            CoefficientField wrapping self.diffusion_field with self.sigma as default

        Example:
            >>> diffusion = problem.get_diffusion_coefficient_field()
            >>> sigma_at_t = diffusion.evaluate_at(
            ...     timestep_idx=5,
            ...     grid=x_coords,
            ...     density=m,
            ...     dt=problem.dt
            ... )
        """
        from mfg_pde.utils.pde_coefficients import CoefficientField

        return CoefficientField(
            field=self.diffusion_field,
            default_value=self.sigma,
            field_name="diffusion",
            dimension=self.dimension,
        )

    def get_drift_coefficient_field(self) -> Any:
        """
        Get a CoefficientField wrapper for the drift field.

        Returns a CoefficientField that handles array and callable drift
        coefficients uniformly. Use this in solvers instead of directly
        accessing self.drift_field.

        Returns:
            CoefficientField wrapping self.drift_field (default is zero drift)

        Example:
            >>> drift = problem.get_drift_coefficient_field()
            >>> alpha_at_t = drift.evaluate_at(
            ...     timestep_idx=5,
            ...     grid=x_coords,
            ...     density=m,
            ...     dt=problem.dt
            ... )
        """
        from mfg_pde.utils.pde_coefficients import CoefficientField

        # Default drift is zero
        default_drift = 0.0

        return CoefficientField(
            field=self.drift_field,
            default_value=default_drift,
            field_name="drift",
            dimension=self.dimension,
        )

    def has_state_dependent_coefficients(self) -> bool:
        """
        Check if problem has state-dependent (callable) PDE coefficients.

        Solvers may need to handle callable coefficients differently from
        constant/precomputed ones (e.g., re-evaluate at each timestep).

        Returns:
            True if diffusion_field or drift_field is callable
        """
        return callable(self.diffusion_field) or callable(self.drift_field)

    def __repr__(self) -> str:
        """
        Return string representation using geometry-first API.

        Avoids accessing deprecated attributes to prevent DeprecationWarning
        spam in Jupyter notebooks and debuggers.
        """
        # Use geometry for spatial info, not deprecated attrs
        geom_type = type(self.geometry).__name__ if self.geometry else "None"
        dim = self.dimension if hasattr(self, "dimension") else "?"

        return f"MFGProblem(geometry={geom_type}, dim={dim}, T={self.T}, Nt={self.Nt}, sigma={self.sigma})"

    def __getstate__(self) -> dict[str, Any]:
        """
        Get state for pickling.

        Returns the instance __dict__ for standard pickle behavior.
        """
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Restore state from pickle with legacy migration support.

        Handles legacy pickle files where geometry=None but legacy
        attributes (xmin, xmax, Nx) are present. Reconstructs geometry
        from these attributes for backward compatibility.
        """
        # Detect legacy format: geometry=None but has legacy 1D attrs
        if state.get("geometry") is None and state.get("xmin") is not None:
            try:
                from mfg_pde.geometry import TensorProductGrid

                # Reconstruct geometry from legacy attributes
                xmin = state.get("xmin")
                xmax = state.get("xmax")
                Nx = state.get("Nx")

                if xmin is None or xmax is None or Nx is None:
                    raise KeyError("Missing required legacy fields (xmin, xmax, Nx)")

                # Handle both scalar and list forms
                if isinstance(xmin, (int, float)):
                    bounds = [(float(xmin), float(xmax))]
                    Nx_points = [int(Nx) + 1]
                else:
                    bounds = list(zip(xmin, xmax, strict=True))
                    Nx_points = [n + 1 for n in Nx]

                state["geometry"] = TensorProductGrid(
                    dimension=len(bounds),
                    bounds=bounds,
                    Nx_points=Nx_points,
                )
            except (KeyError, ImportError) as e:
                import warnings

                warnings.warn(
                    f"Unable to migrate legacy pickle format: {e}. "
                    "This pickle file may be from an incompatible version. "
                    "Consider recreating the MFGProblem.",
                    UserWarning,
                    stacklevel=2,
                )

        self.__dict__.update(state)

    def _detect_solver_compatibility(self) -> None:
        """
        Detect which solver types are compatible with this problem.

        Sets:
            self.solver_compatible: List of compatible solver type strings
            self.solver_recommendations: Dict mapping use cases to solvers

        Called automatically after initialization.
        """
        compatible = []
        recommendations = {}

        # Get problem characteristics
        is_grid = self.domain_type == "grid"
        is_implicit = self.domain_type == "implicit"
        is_network = self.domain_type == "network"
        dim = self.dimension if isinstance(self.dimension, int) else None

        # FDM: Requires regular grid, no complex geometry, works best for dim <= 3
        if (is_grid and not hasattr(self, "has_obstacles")) or (
            hasattr(self, "has_obstacles") and not self.has_obstacles
        ):
            compatible.append("fdm")
            if dim and dim <= 2:
                recommendations["fast"] = "fdm"
                recommendations["accurate"] = "fdm"

        # Semi-Lagrangian: Works with grids, especially good for higher dimensions
        if is_grid:
            compatible.append("semi_lagrangian")
            if dim and dim >= 3:
                recommendations["fast"] = "semi_lagrangian"

        # GFDM: Works with grids and complex geometry (particle collocation)
        if is_grid or is_implicit:
            compatible.append("gfdm")
            if is_implicit or (hasattr(self, "has_obstacles") and self.has_obstacles):
                recommendations["obstacles"] = "gfdm"
                recommendations["complex_geometry"] = "gfdm"

        # Particle methods: Work with everything except pure networks
        if not is_network:
            compatible.append("particle")
            if dim and dim >= 4:
                recommendations["high_dimensional"] = "particle"
                recommendations["fast"] = "particle"

        # Network solver: Only for network problems
        if is_network:
            compatible.append("network_solver")
            recommendations["default"] = "network_solver"

        # DGM: Works with grids (experimental)
        if is_grid:
            compatible.append("dgm")

        # PINN: Works with everything (deep learning approach)
        compatible.append("pinn")
        if dim and dim >= 5:
            recommendations["very_high_dimensional"] = "pinn"

        # Set attributes
        self.solver_compatible = compatible
        self.solver_recommendations = recommendations

        # Set default recommendation
        if "default" not in recommendations:
            if is_grid and dim and dim <= 2:
                recommendations["default"] = "fdm"
            elif is_grid and dim and dim == 3:
                recommendations["default"] = "semi_lagrangian"
            elif is_implicit:
                recommendations["default"] = "gfdm"
            elif compatible:
                recommendations["default"] = compatible[0]

    def validate_solver_type(self, solver_type: str) -> None:
        """
        Validate that solver type is compatible with this problem.

        Args:
            solver_type: Solver type identifier (e.g., "fdm", "gfdm", "particle")

        Raises:
            ValueError: If solver type is incompatible with problem configuration

        Note:
            This method is called by solver constructors to provide early
            error detection with helpful messages.
        """
        if not hasattr(self, "solver_compatible"):
            # Compatibility not yet detected (shouldn't happen if __init__ called)
            self._detect_solver_compatibility()

        if solver_type not in self.solver_compatible:
            # Build helpful error message
            reason = self._get_incompatibility_reason(solver_type)
            suggestion = self._get_solver_suggestion()

            raise ValueError(
                f"Solver type '{solver_type}' is incompatible with this problem.\n\n"
                f"Problem Configuration:\n"
                f"  Domain type: {self.domain_type}\n"
                f"  Dimension: {self.dimension}\n"
                f"  Has obstacles: {getattr(self, 'has_obstacles', False)}\n\n"
                f"Reason: {reason}\n\n"
                f"Compatible solvers: {self.solver_compatible}\n\n"
                f"Suggestion: {suggestion}"
            )

    def _get_incompatibility_reason(self, solver_type: str) -> str:
        """Get human-readable reason why solver is incompatible."""
        reasons = {
            "fdm": {
                "implicit": "FDM requires regular grid, not implicit geometry",
                "network": "FDM requires spatial grid, not network structure",
                "obstacles": "FDM doesn't support obstacles (use GFDM instead)",
            },
            "semi_lagrangian": {
                "implicit": "Semi-Lagrangian requires regular grid",
                "network": "Semi-Lagrangian requires spatial grid",
            },
            "gfdm": {
                "network": "GFDM requires spatial coordinates, not network structure",
            },
            "particle": {
                "network": "Particle methods require spatial domain",
            },
            "network_solver": {
                "grid": "Network solver requires network structure, not spatial grid",
                "implicit": "Network solver requires network structure",
            },
        }

        domain_reasons = reasons.get(solver_type, {})
        return domain_reasons.get(self.domain_type, "Solver not compatible with problem configuration")

    def _get_solver_suggestion(self) -> str:
        """Get helpful suggestion for which solver to use."""
        if not self.solver_recommendations:
            if self.solver_compatible:
                return f"Try using: {self.solver_compatible[0]}"
            return "No compatible solvers found for this configuration"

        # Get default recommendation
        default_solver = self.solver_recommendations.get(
            "default", self.solver_compatible[0] if self.solver_compatible else None
        )

        if not default_solver:
            return "No solver recommendations available"

        # Build recommendation text
        suggestion = f"Use solver '{default_solver}' (recommended for this problem)"

        # Add context-specific recommendations
        additional_recs = []
        if "obstacles" in self.solver_recommendations:
            additional_recs.append(f"obstacles: {self.solver_recommendations['obstacles']}")
        if "fast" in self.solver_recommendations and self.solver_recommendations["fast"] != default_solver:
            additional_recs.append(f"fastest: {self.solver_recommendations['fast']}")
        if "accurate" in self.solver_recommendations and self.solver_recommendations["accurate"] != default_solver:
            additional_recs.append(f"most accurate: {self.solver_recommendations['accurate']}")

        if additional_recs:
            suggestion += f"\n  Alternative recommendations: {', '.join(additional_recs)}"

        suggestion += "\n  Or use create_fast_solver() for automatic selection"

        return suggestion

    def get_solver_info(self) -> dict[str, Any]:
        """
        Get comprehensive solver compatibility information.

        Returns:
            Dictionary with solver compatibility details:
            - compatible: List of compatible solver types
            - recommendations: Dict of use-case specific recommendations
            - dimension: Problem dimension
            - domain_type: Type of spatial domain
            - complexity: Estimated computational complexity
        """
        if not hasattr(self, "solver_compatible"):
            self._detect_solver_compatibility()

        return {
            "compatible": self.solver_compatible,
            "recommendations": self.solver_recommendations,
            "dimension": self.dimension,
            "domain_type": self.domain_type,
            "has_obstacles": getattr(self, "has_obstacles", False),
            "complexity": self._estimate_complexity(),
            "default_solver": self.solver_recommendations.get("default", None),
        }

    def _estimate_complexity(self) -> str:
        """Estimate computational complexity category."""
        if self.domain_type == "network":
            return "O(N_nodes × N_time)"

        if isinstance(self.dimension, int):
            if self.dimension == 1:
                return "O(Nx × Nt)"
            elif self.dimension == 2:
                return "O(Nx × Ny × Nt)"
            elif self.dimension == 3:
                return "O(Nx × Ny × Nz × Nt)"
            else:
                return f"O(N^{self.dimension} × Nt) - curse of dimensionality"

        return "Problem-dependent"

    def get_computational_cost_estimate(self) -> dict:
        """
        Get estimated computational cost for the problem.

        Returns:
            Dictionary with cost estimates:
            - total_spatial_points: Total spatial grid points
            - total_points: Total grid points (space × time)
            - memory_per_array_mb: Memory per solution array (MB)
            - estimated_memory_mb: Total estimated memory (MB)
            - is_feasible: Whether problem is computationally feasible
            - warnings: List of warnings about computational costs
        """
        total_spatial_points = int(np.prod(self.spatial_shape))
        total_points = total_spatial_points * (self.Nt + 1)
        memory_per_array_mb = total_points * 8 / (1024**2)
        estimated_total_mb = memory_per_array_mb * 10  # Rough estimate: ~10 arrays

        warnings_list = []
        is_feasible = True

        if self.dimension > 4:
            warnings_list.append(f"Dimension {self.dimension}D exceeds practical limit (4D)")
            is_feasible = False

        if total_points > 10_000_000:
            warnings_list.append(f"Total points ({total_points:,}) exceeds recommended limit (10M)")
            is_feasible = False

        if estimated_total_mb > 1000:
            warnings_list.append(f"Estimated memory ({estimated_total_mb:.1f} MB) may be excessive")

        return {
            "dimension": self.dimension,
            "spatial_shape": self.spatial_shape,
            "total_spatial_points": total_spatial_points,
            "total_points": total_points,
            "memory_per_array_mb": memory_per_array_mb,
            "estimated_memory_mb": estimated_total_mb,
            "is_feasible": is_feasible,
            "warnings": warnings_list,
        }

    def _potential(self, x: float) -> float:
        """Default potential function."""
        Lx = self._get_domain_length() or 1.0  # Fallback to 1.0 if not 1D
        return 50 * (
            0.1 * np.cos(x * 2 * np.pi / Lx) + 0.25 * np.sin(x * 2 * np.pi / Lx) + 0.1 * np.sin(x * 4 * np.pi / Lx)
        )

    def _u_final(self, x: float) -> float:
        """Default final value function."""
        Lx = self._get_domain_length() or 1.0  # Fallback to 1.0 if not 1D
        return 5 * (np.cos(x * 2 * np.pi / Lx) + 0.4 * np.sin(x * 4 * np.pi / Lx))

    def _m_initial(self, x: float) -> float:
        """Default initial density function."""
        return 2 * np.exp(-200 * (x - 0.2) ** 2) + np.exp(-200 * (x - 0.8) ** 2)

    def _initialize_functions(self, **kwargs: Any) -> None:
        """Initialize potential, initial density, and final value functions."""

        # Initialize arrays with correct shape for both 1D and n-D
        self.f_potential = np.zeros(self.spatial_shape)
        self.u_fin = np.zeros(self.spatial_shape)
        self.m_init = np.zeros(self.spatial_shape)

        # Handle custom vs default initialization
        if self.is_custom and self.components is not None:
            # Custom problem - use provided functions
            if self.components.potential_func is not None:
                self._setup_custom_potential()
            else:
                # Default potential for custom problem
                self.f_potential[:] = 0.0

            if self.components.final_value_func is not None:
                self._setup_custom_final_value()
            else:
                # Default final value for custom problem
                self.u_fin[:] = 0.0

            if self.components.initial_density_func is not None:
                self._setup_custom_initial_density()
            else:
                # Default initial density for custom problem
                self._setup_default_initial_density()
        else:
            # Default problem - use built-in functions
            if self.dimension == 1:
                # 1D default functions (original behavior)
                spatial_grid = self._get_spatial_grid_internal()
                for i in range(self.spatial_shape[0]):
                    # Extract scalar from grid point (grid has shape (Nx, 1) for 1D)
                    x_i = float(spatial_grid[i, 0])
                    self.f_potential[i] = self._potential(x_i)
                    self.u_fin[i] = self._u_final(x_i)
                    self.m_init[i] = self._m_initial(x_i)
            else:
                # n-D default functions (simple defaults)
                # Potential: zero (can be customized later)
                self.f_potential[:] = 0.0

                # Final value: zero (can be customized later)
                self.u_fin[:] = 0.0

                # Initial density: Gaussian at center
                self._setup_default_initial_density()

        # Normalize initial density
        if self.dimension == "network":
            # Network/graph: discrete probability mass, sum = 1
            # No cell volume - just normalize sum to 1
            integral_m_init = np.sum(self.m_init)
        elif self.dimension == 1:
            # 1D normalization (original)
            dx = self._get_spacing() or 1.0
            integral_m_init = np.sum(self.m_init) * dx
        elif self.spatial_bounds is not None and self.spatial_discretization is not None:
            # n-D normalization (integrate over all dimensions)
            # For tensor product grid: integral = sum(m) * prod(dx_i)
            dx_prod = np.prod(
                [
                    (bounds[1] - bounds[0]) / n
                    for bounds, n in zip(self.spatial_bounds, self.spatial_discretization, strict=False)
                ]
            )
            integral_m_init = np.sum(self.m_init) * dx_prod
        else:
            # For unstructured/implicit geometries: use uniform normalization
            # This is a rough approximation - for accurate integration, use proper
            # quadrature rules based on the geometry type
            integral_m_init = np.sum(self.m_init) / self.num_spatial_points

        if integral_m_init > 1e-10:
            self.m_init /= integral_m_init

    def _setup_default_initial_density(self) -> None:
        """Setup default initial density (Gaussian at center for n-D problems)."""
        if self.dimension == 1:
            # 1D: Use original default
            spatial_grid = self._get_spatial_grid_internal()
            for i in range(self.spatial_shape[0]):
                self.m_init[i] = self._m_initial(spatial_grid[i])
        elif self.dimension == "network":
            # Network/graph: uniform density on all nodes
            self.m_init[:] = 1.0 / self.num_nodes
        else:
            # n-D: Gaussian at center of domain
            # Use geometry interface instead of deprecated _grid
            if hasattr(self, "geometry") and self.geometry is not None:
                # Get spatial grid from geometry (works for all geometry types)
                spatial_grid = self.geometry.get_spatial_grid()

                # For CartesianGrid geometries, spatial_grid is already (N, d) array
                # For other geometries, it should also be (N, d)
                if spatial_grid.ndim == 1:
                    # 1D case or flattened - reshape if needed
                    all_points = spatial_grid.reshape(-1, 1)
                else:
                    all_points = spatial_grid

                # Center of domain
                center = np.array([(b[0] + b[1]) / 2 for b in self.spatial_bounds])

                # Gaussian: exp(-alpha * ||x - center||^2)
                alpha = 100.0  # Width parameter
                distances_sq = np.sum((all_points - center) ** 2, axis=1)
                density_flat = np.exp(-alpha * distances_sq)

                # Reshape to grid shape
                self.m_init = density_flat.reshape(self.spatial_shape)
            else:
                # Fallback: uniform density
                self.m_init[:] = 1.0

    # Methods inherited from HamiltonianMixin:
    # - H(), dH_dm(), get_hjb_hamiltonian_jacobian_contrib()
    # - get_hjb_residual_m_coupling_term(), get_potential_at_time()
    # - _setup_custom_potential(), _validate_hamiltonian_components()
    #
    # Methods inherited from ConditionsMixin:
    # - get_boundary_conditions()
    # - _setup_custom_initial_density(), _setup_custom_final_value()

    def get_u_fin(self) -> np.ndarray:
        """Get terminal condition u(T, x). Modern nD interface."""
        return self.u_fin.copy()

    def get_m_init(self) -> np.ndarray:
        """Get initial density m(0, x). Modern nD interface."""
        return self.m_init.copy()

    # Legacy aliases for backward compatibility
    def get_final_u(self) -> np.ndarray:
        """Legacy alias for get_u_fin()."""
        return self.get_u_fin()

    def get_initial_m(self) -> np.ndarray:
        """Legacy alias for get_m_init()."""
        return self.get_m_init()

    def get_problem_info(self) -> dict[str, Any]:
        """Get information about the problem."""
        # Get domain info from geometry (modern API)
        bounds = self.geometry.get_bounds() if self.geometry else None
        domain_info = {
            "dimension": self.dimension,
            "num_spatial_points": self.geometry.num_spatial_points if self.geometry else None,
        }
        if bounds is not None and self.dimension == 1:
            domain_info["xmin"] = float(bounds[0][0])
            domain_info["xmax"] = float(bounds[1][0])
            domain_info["Nx"] = self._get_num_intervals()

        if self.is_custom and self.components is not None:
            return {
                "description": self.components.description,
                "problem_type": self.components.problem_type,
                "is_custom": True,
                "has_custom_hamiltonian": True,
                "has_custom_potential": self.components.potential_func is not None,
                "has_custom_initial": self.components.initial_density_func is not None,
                "has_custom_final": self.components.final_value_func is not None,
                "has_jacobian": self.components.hamiltonian_jacobian_func is not None,
                "has_coupling": self.components.coupling_func is not None,
                "parameters": self.components.parameters,
                "domain": domain_info,
                "time": {"T": self.T, "Nt": self.Nt},
                "coefficients": {"sigma": self.sigma, "coupling_coefficient": self.coupling_coefficient},
            }
        else:
            return {
                "description": "Default MFG Problem",
                "problem_type": "example",
                "is_custom": False,
                "has_custom_hamiltonian": False,
                "has_custom_potential": False,
                "has_custom_initial": False,
                "has_custom_final": False,
                "has_jacobian": False,
                "has_coupling": False,
                "parameters": {},
                "domain": domain_info,
                "time": {"T": self.T, "Nt": self.Nt},
                "coefficients": {"sigma": self.sigma, "coupling_coefficient": self.coupling_coefficient},
            }

    # ============================================================================
    # Solve Method - Primary API for solving MFG problems
    # ============================================================================

    def solve(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
        config: Any | None = None,
    ) -> Any:
        """
        Solve this MFG problem.

        This is the primary API for solving MFG problems. The solver is
        automatically selected based on problem characteristics.

        Args:
            max_iterations: Maximum fixed-point iterations (default: 100)
            tolerance: Convergence tolerance (default: 1e-6)
            verbose: Show solver progress (default: True)
            config: Optional MFGSolverConfig for advanced configuration

        Returns:
            SolverResult with U (value function), M (density), convergence info

        Example:
            >>> problem = MFGProblem(Nx=50, Nt=20, T=1.0)
            >>> result = problem.solve()
            >>> print(f"Converged: {result.converged}")
            >>> U, M = result.U, result.M
        """
        import numpy as np

        from mfg_pde.alg.numerical.coupling import FixedPointIterator
        from mfg_pde.alg.numerical.fp_solvers import FPParticleSolver
        from mfg_pde.alg.numerical.hjb_solvers import HJBGFDMSolver
        from mfg_pde.config import MFGSolverConfig

        # Create or update config
        if config is None:
            config = MFGSolverConfig()

        # Override config with explicit parameters
        config.picard.max_iterations = max_iterations
        config.picard.tolerance = tolerance
        config.picard.verbose = verbose

        # Create collocation points from problem domain
        if hasattr(self, "geometry") and self.geometry is not None:
            # Use geometry grid if available
            if hasattr(self.geometry, "get_spatial_grid"):
                x = self.geometry.get_spatial_grid()
                collocation_points = np.atleast_2d(x).T if x.ndim == 1 else x
            elif hasattr(self.geometry, "interior_points"):
                collocation_points = self.geometry.interior_points
            else:
                # Fallback to grid-based points
                bounds = self.geometry.get_bounds()
                if bounds is not None:
                    x = np.linspace(bounds[0][0], bounds[1][0], self.geometry.num_spatial_points)
                    collocation_points = x.reshape(-1, 1)
                else:
                    raise ValueError("Cannot create collocation points: geometry has no bounds")
        else:
            raise ValueError("Cannot create collocation points: geometry is required")

        # Create component solvers
        hjb_solver = HJBGFDMSolver(self, collocation_points)
        fp_solver = FPParticleSolver(self)

        # Create fixed-point iterator
        solver = FixedPointIterator(
            problem=self,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            config=config,
        )

        return solver.solve(verbose=verbose)
