from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from mfg_pde.geometry import BoundaryConditions

# Import npart and ppart from the utils module
from mfg_pde.utils.aux_func import npart, ppart

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

# Define a limit for values before squaring to prevent overflow within H
VALUE_BEFORE_SQUARE_LIMIT = 1e150


# ============================================================================
# MFG Components for Custom Problem Definition
# ============================================================================


@dataclass
class MFGComponents:
    """
    Container for all components that define a custom MFG problem.

    This class holds all the mathematical components needed to fully specify
    an MFG problem, allowing users to provide custom implementations.
    """

    # Core Hamiltonian components
    hamiltonian_func: Callable | None = None  # H(x, m, p, t) -> float
    hamiltonian_dm_func: Callable | None = None  # dH/dm(x, m, p, t) -> float

    # Optional Jacobian for advanced solvers
    hamiltonian_jacobian_func: Callable | None = None  # Jacobian contribution

    # Potential function V(x, t)
    potential_func: Callable | None = None  # V(x, t) -> float

    # Initial and final conditions
    initial_density_func: Callable | None = None  # m_0(x) -> float
    final_value_func: Callable | None = None  # u_T(x) -> float

    # Boundary conditions
    boundary_conditions: BoundaryConditions | None = None

    # Coupling terms (for advanced MFG formulations)
    coupling_func: Callable | None = None  # Additional coupling terms

    # Problem parameters
    parameters: dict[str, Any] = field(default_factory=dict)

    # Metadata
    description: str = "MFG Problem"
    problem_type: str = "mfg"


# ============================================================================
# Unified MFG Problem Class
# ============================================================================


class MFGProblem:
    """
    Unified MFG problem class that can handle both predefined and custom formulations.

    This class serves as the single constructor for all MFG problems:
    - Default usage: Uses built-in Hamiltonian (equivalent to old ExampleMFGProblem)
    - Custom usage: Accepts MFGComponents for full mathematical control
    """

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
        geometry: Any | None = None,  # BaseGeometry
        obstacles: list | None = None,
        # Network parameters (NEW)
        network: Any | None = None,  # NetworkGraph
        # Time domain parameters
        T: float | None = None,
        Nt: int | None = None,
        time_domain: tuple[float, int] | None = None,  # Alternative to T/Nt
        # Physical parameters
        sigma: float | None = None,
        diffusion: float | None = None,  # Alias for sigma
        coupling_coefficient: float = 0.5,
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
            geometry: BaseGeometry object for complex domains
            obstacles: List of obstacle geometries
            network: NetworkGraph for network MFG problems
            T, Nt, time_domain: Time domain parameters (T, Nt) or tuple (T, Nt)
            sigma, diffusion: Diffusion coefficient (sigma is standard name)
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
        """
        import warnings

        # Normalize parameter aliases
        if time_domain is not None:
            if T is not None or Nt is not None:
                raise ValueError("Specify EITHER (T, Nt) OR time_domain, not both")
            T, Nt = time_domain

        if diffusion is not None:
            if sigma is not None:
                raise ValueError("Specify EITHER sigma OR diffusion, not both")
            sigma = diffusion

        # Set defaults for T, Nt, sigma if not provided
        if T is None:
            T = 1.0
        if Nt is None:
            Nt = 51
        if sigma is None:
            sigma = 1.0

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

        # Detect initialization mode (use normalized Nx for detection)
        mode = self._detect_init_mode(
            Nx=Nx_normalized, spatial_bounds=spatial_bounds, geometry=geometry, network=network
        )

        # Dispatch to appropriate initializer
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
            self._init_1d_legacy(xmin_normalized, xmax_normalized, Nx_normalized, T, Nt, sigma, coupling_coefficient)

        elif mode == "nd_grid":
            # Mode 2: N-dimensional grid
            self._init_nd(spatial_bounds, spatial_discretization, T, Nt, sigma, coupling_coefficient, suppress_warnings)

        elif mode == "geometry":
            # Mode 3: Complex geometry
            self._init_geometry(geometry, obstacles, T, Nt, sigma, coupling_coefficient, suppress_warnings)

        elif mode == "network":
            # Mode 4: Network MFG
            self._init_network(network, T, Nt, sigma, coupling_coefficient)

        elif mode == "default":
            # Default: 1D with default parameters
            warnings.warn(
                "No spatial domain specified. Using default 1D domain: [0, 1] with 51 points.",
                UserWarning,
                stacklevel=2,
            )
            self._init_1d_legacy([0.0], [1.0], [51], T, Nt, sigma, coupling_coefficient)

        else:
            raise ValueError(f"Unknown initialization mode: {mode}")

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
            self._validate_components()

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
            the geometry-first API with SimpleGrid1D or TensorProductGrid instead.
            See migration guide: docs/migration/GEOMETRY_PARAMETER_MIGRATION.md
        """
        import warnings

        from mfg_pde.geometry import SimpleGrid1D
        from mfg_pde.geometry.boundary_conditions_1d import BoundaryConditions

        # Emit deprecation warning for manual grid construction pattern
        warnings.warn(
            "Manual grid construction in MFGProblem is deprecated and will be "
            "restricted in v1.0.0. Use the geometry-first API instead:\n\n"
            "  from mfg_pde.geometry import SimpleGrid1D\n"
            f"  domain = SimpleGrid1D(xmin={xmin[0]}, xmax={xmax[0]}, boundary_conditions='periodic')\n"
            f"  domain.create_grid(num_points={Nx[0] + 1})\n"
            f"  problem = MFGProblem(geometry=domain, T={T}, Nt={Nt})\n\n"
            "See docs/migration/GEOMETRY_PARAMETER_MIGRATION.md for details.",
            DeprecationWarning,
            stacklevel=4,  # Point to user's code, not internal calls
        )

        # Extract scalar values from arrays for backward compatibility
        xmin_scalar = xmin[0]
        xmax_scalar = xmax[0]
        Nx_scalar = Nx[0]

        # Create SimpleGrid1D geometry object (unified internal representation)
        bc = BoundaryConditions(type="periodic")  # Default to periodic for backward compatibility
        geometry = SimpleGrid1D(xmin=xmin_scalar, xmax=xmax_scalar, boundary_conditions=bc)
        dx, _ = geometry.create_grid(num_points=Nx_scalar + 1)

        # Store geometry for unified interface
        self.geometry = geometry

        # Set dimension from geometry
        self.dimension = geometry.dimension

        # Legacy 1D attributes (scalars for backward compatibility)
        # Derived from geometry for consistency
        self.xmin: float = xmin_scalar
        self.xmax: float = xmax_scalar
        self.Lx: float = xmax_scalar - xmin_scalar
        self.Nx: int = Nx_scalar
        self.Dx: float = dx

        # Time domain
        self.T: float = T
        self.Nt: int = Nt
        self.Dt: float = T / Nt if Nt > 0 else 0.0

        # Grid arrays (from geometry)
        self.xSpace: np.ndarray = geometry.get_spatial_grid()
        self.tSpace: np.ndarray = np.linspace(0, T, Nt + 1, endpoint=True)

        # Coefficients
        self.sigma: float = sigma
        self.coupling_coefficient: float = coupling_coefficient

        # New n-D attributes for consistency (derived from geometry)
        self.spatial_shape = geometry.get_grid_shape()
        self.spatial_bounds = [(xmin_scalar, xmax_scalar)]
        self.spatial_discretization = [Nx_scalar]

        # Grid object (deprecated, use self.geometry instead)
        self._grid = None

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
            f"      num_points={spatial_discretization if spatial_discretization else [51] * dimension}\n"
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

        # Create appropriate geometry object based on dimension
        if dimension == 1:
            # 1D case: use SimpleGrid1D
            from mfg_pde.geometry import SimpleGrid1D
            from mfg_pde.geometry.boundary_conditions_1d import BoundaryConditions

            bc = BoundaryConditions(type="periodic")
            geometry = SimpleGrid1D(xmin=spatial_bounds[0][0], xmax=spatial_bounds[0][1], boundary_conditions=bc)
            dx, _ = geometry.create_grid(num_points=spatial_discretization[0] + 1)

            # Legacy 1D attributes
            self.xmin = spatial_bounds[0][0]
            self.xmax = spatial_bounds[0][1]
            self.Lx = self.xmax - self.xmin
            self.Nx = spatial_discretization[0]
            self.Dx = dx
            self.xSpace = geometry.get_spatial_grid()

        elif dimension == 2:
            # 2D case: use SimpleGrid2D
            from mfg_pde.geometry import SimpleGrid2D

            bounds_flat = (
                spatial_bounds[0][0],  # xmin
                spatial_bounds[0][1],  # xmax
                spatial_bounds[1][0],  # ymin
                spatial_bounds[1][1],  # ymax
            )
            resolution = tuple(spatial_discretization)
            geometry = SimpleGrid2D(bounds=bounds_flat, resolution=resolution)

            # No legacy 1D attributes for 2D
            self.xmin = None
            self.xmax = None
            self.Lx = None
            self.Nx = None
            self.Dx = None
            self.xSpace = None

        elif dimension == 3:
            # 3D case: use SimpleGrid3D
            from mfg_pde.geometry import SimpleGrid3D

            bounds_flat = (
                spatial_bounds[0][0],  # xmin
                spatial_bounds[0][1],  # xmax
                spatial_bounds[1][0],  # ymin
                spatial_bounds[1][1],  # ymax
                spatial_bounds[2][0],  # zmin
                spatial_bounds[2][1],  # zmax
            )
            resolution = tuple(spatial_discretization)
            geometry = SimpleGrid3D(bounds=bounds_flat, resolution=resolution)

            # No legacy 1D attributes for 3D
            self.xmin = None
            self.xmax = None
            self.Lx = None
            self.Nx = None
            self.Dx = None
            self.xSpace = None

        else:
            # 4D+: use TensorProductGrid (for now, until we have SimpleGridND)
            from mfg_pde.geometry import TensorProductGrid

            geometry = TensorProductGrid(dimension=dimension, bounds=spatial_bounds, num_points=spatial_discretization)

            # No legacy 1D attributes for nD
            self.xmin = None
            self.xmax = None
            self.Lx = None
            self.Nx = None
            self.Dx = None
            self.xSpace = None

        # Store geometry for unified interface
        self.geometry = geometry

        # Set dimension from geometry
        self.dimension = geometry.dimension

        # Store n-D parameters
        self.spatial_bounds = spatial_bounds
        self.spatial_discretization = spatial_discretization

        # Spatial shape from discretization
        self.spatial_shape = tuple(spatial_discretization)

        # Time domain
        self.T: float = T
        self.Nt: int = Nt
        self.Dt: float = T / Nt if Nt > 0 else 0.0
        self.tSpace: np.ndarray = np.linspace(0, T, Nt + 1, endpoint=True)

        # Coefficients
        self.sigma: float = sigma
        self.coupling_coefficient: float = coupling_coefficient

        # Grid object (deprecated, use self.geometry instead)
        self._grid = None

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
        geometry: Any | None,
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
        geometry: Any,
        obstacles: list | None,
        T: float,
        Nt: int,
        sigma: float,
        coupling_coefficient: float,
        suppress_warnings: bool,
    ) -> None:
        """
        Initialize problem with geometry object implementing GeometryProtocol.

        Accepts any geometry type: TensorProductGrid, SimpleGrid1D, BaseGeometry,
        ImplicitDomain, NetworkGeometry, etc.

        Args:
            geometry: Any object implementing GeometryProtocol
            obstacles: List of obstacle geometries (for domain geometries)
            T, Nt: Time domain parameters
            sigma, coupling_coefficient: Physical parameters
            suppress_warnings: Suppress warnings
        """
        # Import geometry protocol
        try:
            from mfg_pde.geometry.geometry_protocol import GeometryProtocol, validate_geometry
        except ImportError as err:
            raise ImportError(
                "Geometry mode requires geometry module. Install with: pip install mfg_pde[geometry]"
            ) from err

        # Validate geometry object implements GeometryProtocol
        if not isinstance(geometry, GeometryProtocol):
            raise TypeError(
                f"geometry must implement GeometryProtocol, got {type(geometry)}. "
                f"Use TensorProductGrid, SimpleGrid1D, BaseGeometry, ImplicitDomain, or NetworkGeometry."
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
        self.Dt = T / Nt if Nt > 0 else 0.0
        self.tSpace = np.linspace(0, T, Nt + 1, endpoint=True)

        # Physical parameters
        self.sigma = sigma
        self.coupling_coefficient = coupling_coefficient

        # Initialize spatial discretization based on geometry type
        from mfg_pde.geometry.geometry_protocol import GeometryType

        if geometry.geometry_type == GeometryType.CARTESIAN_GRID:
            # CARTESIAN_GRID: Can be TensorProductGrid or AMR mesh
            # Use polymorphic method to get configuration
            self._grid = geometry
            config = geometry.get_problem_config()

            # Apply configuration from geometry
            self.num_spatial_points = config["num_spatial_points"]
            self.spatial_shape = config["spatial_shape"]
            self.spatial_bounds = config["spatial_bounds"]
            self.spatial_discretization = config["spatial_discretization"]

            # Legacy 1D attributes (only if geometry provides them)
            if config["legacy_1d_attrs"] is not None:
                legacy = config["legacy_1d_attrs"]
                self.xmin = legacy["xmin"]
                self.xmax = legacy["xmax"]
                self.Lx = legacy["Lx"]
                self.Nx = legacy["Nx"]
                # Handle both "Dx" and "dx" for backward compatibility
                self.Dx = legacy.get("Dx") or legacy.get("dx")
                self.xSpace = legacy["xSpace"]
            else:
                # AMR or higher dimensional grids
                self.xmin = None
                self.xmax = None
                self.Lx = None
                self.Nx = None
                self.Dx = None
                self.xSpace = None

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
            self._grid = None

            # Legacy 1D attributes (None for unstructured mesh)
            self.xmin = None
            self.xmax = None
            self.Lx = None
            self.Nx = None
            self.Dx = None
            self.xSpace = None

            self.domain_type = "mesh"

        elif geometry.geometry_type == GeometryType.IMPLICIT:
            # ImplicitDomain - point cloud from SDF
            self.num_spatial_points = geometry.num_spatial_points
            self.collocation_points = geometry.get_spatial_grid()
            self.spatial_shape = (self.num_spatial_points,)
            self.spatial_bounds = geometry.get_bounding_box()
            self.spatial_discretization = None
            self._grid = None

            # Legacy 1D attributes (None for implicit domain)
            self.xmin = None
            self.xmax = None
            self.Lx = None
            self.Nx = None
            self.Dx = None
            self.xSpace = None

            self.domain_type = "implicit"

        else:
            # Generic GeometryProtocol object - use spatial grid
            self.num_spatial_points = geometry.num_spatial_points
            self.collocation_points = geometry.get_spatial_grid()
            self.spatial_shape = (self.num_spatial_points,)
            self.spatial_bounds = None
            self.spatial_discretization = None
            self._grid = None

            # Legacy 1D attributes (None)
            self.xmin = None
            self.xmax = None
            self.Lx = None
            self.Nx = None
            self.Dx = None
            self.xSpace = None

            self.domain_type = str(geometry.geometry_type.value)

    def _init_network(
        self,
        network: Any,
        T: float,
        Nt: int,
        sigma: float,
        coupling_coefficient: float,
    ) -> None:
        """
        Initialize problem on network/graph.

        Args:
            network: NetworkGraph or networkx.Graph
            T, Nt: Time domain parameters
            sigma, coupling_coefficient: Physical parameters
        """
        # Store network
        self.network = network
        self.dimension = "network"  # Special dimension indicator
        self.domain_type = "network"

        # Get number of nodes
        try:
            import networkx as nx

            if isinstance(network, nx.Graph):
                self.num_nodes = network.number_of_nodes()
                self.adjacency_matrix = nx.adjacency_matrix(network).toarray()
            else:
                # Assume custom NetworkGraph type
                self.num_nodes = len(network.nodes)
                self.adjacency_matrix = network.adjacency_matrix
        except ImportError:
            # Fallback: assume custom type
            self.num_nodes = len(network.nodes)
            self.adjacency_matrix = network.adjacency_matrix

        # Time domain
        self.T = T
        self.Nt = Nt
        self.Dt = T / Nt if Nt > 0 else 0.0
        self.tSpace = np.linspace(0, T, Nt + 1, endpoint=True)

        # Physical parameters
        self.sigma = sigma
        self.coupling_coefficient = coupling_coefficient

        # Spatial discretization (nodes)
        self.spatial_shape = (self.num_nodes,)
        self.spatial_bounds = None
        self.spatial_discretization = None

        # Legacy 1D attributes (None for network mode)
        self.xmin = None
        self.xmax = None
        self.Lx = None
        self.Nx = None
        self.Dx = None
        self.xSpace = None
        self._grid = None
        self.geometry = None
        self.obstacles = None
        self.has_obstacles = False

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
        return 50 * (
            0.1 * np.cos(x * 2 * np.pi / self.Lx)
            + 0.25 * np.sin(x * 2 * np.pi / self.Lx)
            + 0.1 * np.sin(x * 4 * np.pi / self.Lx)
        )

    def _u_final(self, x: float) -> float:
        """Default final value function."""
        return 5 * (np.cos(x * 2 * np.pi / self.Lx) + 0.4 * np.sin(x * 4 * np.pi / self.Lx))

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
                for i in range(self.spatial_shape[0]):
                    self.f_potential[i] = self._potential(self.xSpace[i])
                    self.u_fin[i] = self._u_final(self.xSpace[i])
                    self.m_init[i] = self._m_initial(self.xSpace[i])
            else:
                # n-D default functions (simple defaults)
                # Potential: zero (can be customized later)
                self.f_potential[:] = 0.0

                # Final value: zero (can be customized later)
                self.u_fin[:] = 0.0

                # Initial density: Gaussian at center
                self._setup_default_initial_density()

        # Normalize initial density
        if self.dimension == 1:
            # 1D normalization (original)
            integral_m_init = np.sum(self.m_init) * self.Dx
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
            for i in range(self.spatial_shape[0]):
                self.m_init[i] = self._m_initial(self.xSpace[i])
        else:
            # n-D: Gaussian at center of domain
            # Get grid coordinates
            if self._grid is not None:
                # Use TensorProductGrid
                all_points = self._grid.flatten()  # Shape: (N_total, dimension)

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

    def _validate_components(self):
        """Validate that required components are provided."""
        if self.components is None:
            raise ValueError("components is None but custom mode is enabled")

        # Only validate Hamiltonians if at least one is provided
        # If both are None, we'll use default implementations
        has_hamiltonian = self.components.hamiltonian_func is not None
        has_hamiltonian_dm = self.components.hamiltonian_dm_func is not None

        if has_hamiltonian and not has_hamiltonian_dm:
            raise ValueError("hamiltonian_dm_func is required when hamiltonian_func is provided")

        if has_hamiltonian_dm and not has_hamiltonian:
            raise ValueError("hamiltonian_func is required when hamiltonian_dm_func is provided")

        # Validate function signatures only if Hamiltonians are provided
        if has_hamiltonian:
            self._validate_function_signature(
                self.components.hamiltonian_func,
                "hamiltonian_func",
                ["x_idx", "m_at_x"],  # Base required params
                gradient_param_required=True,  # Must have EITHER derivs OR p_values
            )

        if has_hamiltonian_dm:
            self._validate_function_signature(
                self.components.hamiltonian_dm_func,
                "hamiltonian_dm_func",
                ["x_idx", "m_at_x"],  # Base required params
                gradient_param_required=True,  # Must have EITHER derivs OR p_values
            )

    def _validate_function_signature(
        self, func: Callable, name: str, expected_params: list, gradient_param_required: bool = False
    ) -> None:
        """
        Validate function signature has expected parameters.

        Args:
            func: Function to validate
            name: Name of the function (for error messages)
            expected_params: List of required parameter names
            gradient_param_required: If True, requires EITHER 'derivs' OR 'p_values'
        """
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Special handling for gradient notation migration
        if gradient_param_required:
            has_derivs = "derivs" in params
            has_p_values = "p_values" in params

            if not (has_derivs or has_p_values):
                raise ValueError(
                    f"{name} must accept either 'derivs' (tuple notation, preferred) "
                    f"or 'p_values' (legacy string-key format) parameter. "
                    f"Current parameters: {params}"
                )

        # Check remaining required parameters
        missing = [p for p in expected_params if p not in params]
        if missing:
            raise ValueError(f"{name} must accept parameters: {expected_params}. Missing: {missing}")

    def _setup_custom_potential(self):
        """Setup custom potential function."""
        if self.components is None or self.components.potential_func is None:
            return

        potential_func = self.components.potential_func

        for i in range(self.Nx + 1):
            x_i = self.xSpace[i]

            # Check if potential depends on time
            sig = inspect.signature(potential_func)
            if "t" in sig.parameters or "time" in sig.parameters:
                # Time-dependent potential - use t=0 for initialization
                self.f_potential[i] = potential_func(x_i, 0.0)
            else:
                # Time-independent potential
                self.f_potential[i] = potential_func(x_i)

    def _setup_custom_initial_density(self):
        """Setup custom initial density function."""
        if self.components is None or self.components.initial_density_func is None:
            return

        initial_func = self.components.initial_density_func

        for i in range(self.Nx + 1):
            x_i = self.xSpace[i]
            self.m_init[i] = max(initial_func(x_i), 0.0)

    def _setup_custom_final_value(self):
        """Setup custom final value function."""
        if self.components is None or self.components.final_value_func is None:
            return

        final_func = self.components.final_value_func

        for i in range(self.Nx + 1):
            x_i = self.xSpace[i]
            self.u_fin[i] = final_func(x_i)

    def H(
        self,
        x_idx: int,
        m_at_x: float,
        derivs: dict[tuple, float] | None = None,
        p_values: dict[str, float] | None = None,
        t_idx: int | None = None,
        x_position: float | None = None,
        current_time: float | None = None,
    ) -> float:
        """
        Hamiltonian function H(x, m, p, t).

        Supports both tuple notation (derivs) and legacy string-key (p_values) formats.

        Args:
            x_idx: Grid index (0 to Nx)
            m_at_x: Density at grid point x_idx
            derivs: Derivatives in tuple notation (NEW, preferred):
                    - 1D: {(0,): u, (1,): du/dx}
                    - 2D: {(0,0): u, (1,0): du/dx, (0,1): du/dy}
            p_values: Momentum dictionary (LEGACY, deprecated):
                      {"forward": p_forward, "backward": p_backward}
            t_idx: Time index (optional)
            x_position: Actual position coordinate (computed from x_idx if not provided)
            current_time: Actual time value (computed from t_idx if not provided)

        Returns:
            Hamiltonian value H(x, m, p, t)

        Note:
            Provide EITHER derivs OR p_values. If both provided, derivs takes precedence.
            p_values is deprecated and will be removed in a future version.
        """
        import warnings

        # Auto-detection and conversion
        if derivs is None and p_values is None:
            raise ValueError("Must provide either 'derivs' or 'p_values' to H()")

        if derivs is None:
            # Legacy mode: convert p_values to derivs
            warnings.warn(
                "p_values parameter is deprecated. Use derivs instead. "
                "See docs/gradient_notation_standard.md for migration guide.",
                DeprecationWarning,
                stacklevel=2,
            )
            from mfg_pde.compat.gradient_notation import ensure_tuple_notation

            derivs = ensure_tuple_notation(p_values, dimension=1, u_value=0.0)

        # Compute x_position and current_time if not provided
        if x_position is None:
            x_position = self.xSpace[x_idx]
        if current_time is None and t_idx is not None:
            current_time = self.tSpace[t_idx] if t_idx < len(self.tSpace) else 0.0

        # Use custom Hamiltonian if provided
        if self.is_custom and self.components is not None and self.components.hamiltonian_func is not None:
            try:
                # Check if custom function accepts 'derivs' or 'p_values'
                sig = inspect.signature(self.components.hamiltonian_func)
                params = list(sig.parameters.keys())

                if "derivs" in params:
                    # New-style custom Hamiltonian
                    result = self.components.hamiltonian_func(
                        x_idx=x_idx,
                        x_position=x_position,
                        m_at_x=m_at_x,
                        derivs=derivs,
                        t_idx=t_idx,
                        current_time=current_time,
                        problem=self,
                    )
                else:
                    # Legacy custom Hamiltonian - convert derivs to p_values
                    from mfg_pde.compat.gradient_notation import derivs_to_p_values_1d

                    p_values_legacy = derivs_to_p_values_1d(derivs)

                    result = self.components.hamiltonian_func(
                        x_idx=x_idx,
                        x_position=x_position,
                        m_at_x=m_at_x,
                        p_values=p_values_legacy,
                        t_idx=t_idx,
                        current_time=current_time,
                        problem=self,
                    )

                return result

            except Exception as e:
                # Log error but return NaN to maintain solver stability
                import logging

                logging.getLogger(__name__).warning(f"Custom Hamiltonian evaluation failed at x_idx={x_idx}: {e}")
                return np.nan

        # Default Hamiltonian implementation (uses tuple notation internally)
        p = derivs.get((1,), 0.0)  # Extract first derivative

        if np.isnan(p) or np.isinf(p) or np.isnan(m_at_x) or np.isinf(m_at_x):
            return np.nan

        # Use upwind scheme for default Hamiltonian
        npart_val = float(npart(p))
        ppart_val = float(ppart(p))

        if abs(npart_val) > VALUE_BEFORE_SQUARE_LIMIT or abs(ppart_val) > VALUE_BEFORE_SQUARE_LIMIT:
            return np.nan

        try:
            term_npart_sq = npart_val**2
            term_ppart_sq = ppart_val**2
        except OverflowError:
            return np.nan

        if np.isinf(term_npart_sq) or np.isnan(term_npart_sq) or np.isinf(term_ppart_sq) or np.isnan(term_ppart_sq):
            return np.nan

        hamiltonian_control_part = 0.5 * self.coupling_coefficient * (term_npart_sq + term_ppart_sq)

        if np.isinf(hamiltonian_control_part) or np.isnan(hamiltonian_control_part):
            return np.nan

        potential_cost_V_x = self.f_potential[x_idx]
        coupling_density_m_x = m_at_x**2

        if (
            np.isinf(potential_cost_V_x)
            or np.isnan(potential_cost_V_x)
            or np.isinf(coupling_density_m_x)
            or np.isnan(coupling_density_m_x)
        ):
            return np.nan

        result = hamiltonian_control_part - potential_cost_V_x - coupling_density_m_x

        if np.isinf(result) or np.isnan(result):
            return np.nan

        return result

    def dH_dm(
        self,
        x_idx: int,
        m_at_x: float,
        derivs: dict[tuple, float] | None = None,
        p_values: dict[str, float] | None = None,
        t_idx: int | None = None,
        x_position: float | None = None,
        current_time: float | None = None,
    ) -> float:
        """
        Hamiltonian derivative with respect to density dH/dm.

        Supports both tuple notation (derivs) and legacy string-key (p_values) formats.

        Args:
            x_idx: Grid index (0 to Nx)
            m_at_x: Density at grid point x_idx
            derivs: Derivatives in tuple notation (NEW, preferred)
            p_values: Momentum dictionary (LEGACY, deprecated)
            t_idx: Time index (optional)
            x_position: Actual position coordinate (computed from x_idx if not provided)
            current_time: Actual time value (computed from t_idx if not provided)

        Returns:
            Derivative dH/dm at (x, m, p, t)

        Note:
            Provide EITHER derivs OR p_values. If both provided, derivs takes precedence.
            p_values is deprecated and will be removed in a future version.
        """
        import warnings

        # Auto-detection and conversion (same as H())
        if derivs is None and p_values is None:
            raise ValueError("Must provide either 'derivs' or 'p_values' to dH_dm()")

        if derivs is None:
            # Legacy mode: convert p_values to derivs
            warnings.warn(
                "p_values parameter is deprecated. Use derivs instead. "
                "See docs/gradient_notation_standard.md for migration guide.",
                DeprecationWarning,
                stacklevel=2,
            )
            from mfg_pde.compat.gradient_notation import ensure_tuple_notation

            derivs = ensure_tuple_notation(p_values, dimension=1, u_value=0.0)

        # Compute x_position and current_time if not provided
        if x_position is None:
            x_position = self.xSpace[x_idx]
        if current_time is None and t_idx is not None:
            current_time = self.tSpace[t_idx] if t_idx < len(self.tSpace) else 0.0

        # Use custom derivative if provided
        if self.is_custom and self.components is not None and self.components.hamiltonian_dm_func is not None:
            try:
                # Check if custom function accepts 'derivs' or 'p_values'
                sig = inspect.signature(self.components.hamiltonian_dm_func)
                params = list(sig.parameters.keys())

                if "derivs" in params:
                    # New-style custom derivative
                    result = self.components.hamiltonian_dm_func(
                        x_idx=x_idx,
                        x_position=x_position,
                        m_at_x=m_at_x,
                        derivs=derivs,
                        t_idx=t_idx,
                        current_time=current_time,
                        problem=self,
                    )
                else:
                    # Legacy custom derivative - convert derivs to p_values
                    from mfg_pde.compat.gradient_notation import derivs_to_p_values_1d

                    p_values_legacy = derivs_to_p_values_1d(derivs)

                    result = self.components.hamiltonian_dm_func(
                        x_idx=x_idx,
                        x_position=x_position,
                        m_at_x=m_at_x,
                        p_values=p_values_legacy,
                        t_idx=t_idx,
                        current_time=current_time,
                        problem=self,
                    )

                return result

            except Exception as e:
                # Log error but return NaN
                import logging

                logging.getLogger(__name__).warning(f"Custom Hamiltonian derivative failed at x_idx={x_idx}: {e}")
                return np.nan

        # Default derivative implementation
        if np.isnan(m_at_x) or np.isinf(m_at_x):
            return np.nan
        return 2 * m_at_x

    def get_hjb_hamiltonian_jacobian_contrib(
        self,
        U_for_jacobian_terms: np.ndarray,
        t_idx_n: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Optional Jacobian contribution for advanced solvers.
        """
        # Use custom Jacobian if provided
        if self.is_custom and self.components is not None and self.components.hamiltonian_jacobian_func is not None:
            try:
                return self.components.hamiltonian_jacobian_func(
                    U_for_jacobian_terms=U_for_jacobian_terms,
                    t_idx_n=t_idx_n,
                    problem=self,
                )
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(f"Jacobian computation failed: {e}")

        # Default Jacobian implementation (only for non-custom problems)
        if not self.is_custom:
            Nx = self.Nx + 1
            Dx = self.Dx
            coupling_coefficient = self.coupling_coefficient

            J_D_H = np.zeros(Nx)
            J_L_H = np.zeros(Nx)
            J_U_H = np.zeros(Nx)

            if abs(Dx) < 1e-14 or Nx <= 1:
                return J_D_H, J_L_H, J_U_H

            U_curr = U_for_jacobian_terms

            for i in range(Nx):
                ip1 = (i + 1) % Nx
                im1 = (i - 1 + Nx) % Nx

                # Derivatives of U_curr
                p1_i = (U_curr[ip1] - U_curr[i]) / Dx
                p2_i = (U_curr[i] - U_curr[im1]) / Dx

                J_D_H[i] = coupling_coefficient * (npart(p1_i) + ppart(p2_i)) / (Dx**2)
                J_L_H[i] = -coupling_coefficient * ppart(p2_i) / (Dx**2)
                J_U_H[i] = -coupling_coefficient * npart(p1_i) / (Dx**2)

            return J_D_H, J_L_H, J_U_H

        return None

    def get_hjb_residual_m_coupling_term(
        self,
        M_density_at_n_plus_1: np.ndarray,
        U_n_current_guess_derivatives: dict[str, np.ndarray],
        x_idx: int,
        t_idx_n: int,
    ) -> float | None:
        """
        Optional coupling term for residual computation.
        """
        # Use custom coupling if provided
        if self.is_custom and self.components is not None and self.components.coupling_func is not None:
            try:
                return self.components.coupling_func(
                    M_density_at_n_plus_1=M_density_at_n_plus_1,
                    U_n_current_guess_derivatives=U_n_current_guess_derivatives,
                    x_idx=x_idx,
                    t_idx_n=t_idx_n,
                    problem=self,
                )
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(f"Coupling term computation failed: {e}")

        # Default coupling implementation (only for non-custom problems)
        if not self.is_custom:
            # Extract scalar value (works for both NumPy and PyTorch)
            m_val = M_density_at_n_plus_1[x_idx]
            m_val = m_val.item() if hasattr(m_val, "item") else float(m_val)
            if np.isnan(m_val) or np.isinf(m_val):
                return np.nan
            try:
                term = -2 * (m_val**2)
            except OverflowError:
                return np.nan
            if np.isinf(term) or np.isnan(term):
                return np.nan
            return term

        return None

    def get_boundary_conditions(self) -> BoundaryConditions:
        """Get boundary conditions for the problem."""
        if self.is_custom and self.components is not None and self.components.boundary_conditions is not None:
            return self.components.boundary_conditions
        else:
            # Default periodic boundary conditions
            return BoundaryConditions(type="periodic")

    def get_potential_at_time(self, t_idx: int) -> np.ndarray:
        """Get potential function at specific time (for time-dependent potentials)."""
        if self.is_custom and self.components is not None and self.components.potential_func is not None:
            # Check if potential is time-dependent
            sig = inspect.signature(self.components.potential_func)
            if "t" in sig.parameters or "time" in sig.parameters:
                # Recompute potential at current time
                current_time = self.tSpace[t_idx] if t_idx < len(self.tSpace) else 0.0
                potential_at_t = np.zeros_like(self.f_potential)

                for i in range(self.Nx + 1):
                    x_i = self.xSpace[i]
                    potential_at_t[i] = self.components.potential_func(x_i, current_time)

                return potential_at_t

        return self.f_potential.copy()

    def get_final_u(self) -> np.ndarray:
        return self.u_fin.copy()

    def get_initial_m(self) -> np.ndarray:
        return self.m_init.copy()

    def get_problem_info(self) -> dict[str, Any]:
        """Get information about the problem."""
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
                "domain": {"xmin": self.xmin, "xmax": self.xmax, "Nx": self.Nx},
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
                "domain": {"xmin": self.xmin, "xmax": self.xmax, "Nx": self.Nx},
                "time": {"T": self.T, "Nt": self.Nt},
                "coefficients": {"sigma": self.sigma, "coupling_coefficient": self.coupling_coefficient},
            }


# ============================================================================
# Builder Pattern for Easy Problem Construction
# ============================================================================


class MFGProblemBuilder:
    """
    Builder class for constructing MFG problems step by step.

    This class provides a fluent interface for building custom MFG problems.
    """

    def __init__(self):
        """Initialize empty builder."""
        self.components = MFGComponents()
        self.domain_params = {}
        self.time_params = {}
        self.solver_params = {}

    def hamiltonian(self, hamiltonian_func: Callable, hamiltonian_dm_func: Callable) -> MFGProblemBuilder:
        """
        Set custom Hamiltonian and its derivative.

        Args:
            hamiltonian_func: H(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem) -> float
            hamiltonian_dm_func: dH/dm(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem) -> float
        """
        self.components.hamiltonian_func = hamiltonian_func
        self.components.hamiltonian_dm_func = hamiltonian_dm_func
        return self

    def potential(self, potential_func: Callable) -> MFGProblemBuilder:
        """
        Set custom potential function.

        Args:
            potential_func: V(x, t=None) -> float
        """
        self.components.potential_func = potential_func
        return self

    def initial_density(self, initial_func: Callable) -> MFGProblemBuilder:
        """
        Set custom initial density function.

        Args:
            initial_func: m_0(x) -> float
        """
        self.components.initial_density_func = initial_func
        return self

    def final_value(self, final_func: Callable) -> MFGProblemBuilder:
        """
        Set custom final value function.

        Args:
            final_func: u_T(x) -> float
        """
        self.components.final_value_func = final_func
        return self

    def boundary_conditions(self, bc: BoundaryConditions) -> MFGProblemBuilder:
        """Set boundary conditions."""
        self.components.boundary_conditions = bc
        return self

    def jacobian(self, jacobian_func: Callable) -> MFGProblemBuilder:
        """Set optional Jacobian function for advanced solvers."""
        self.components.hamiltonian_jacobian_func = jacobian_func
        return self

    def coupling(self, coupling_func: Callable) -> MFGProblemBuilder:
        """Set optional coupling function."""
        self.components.coupling_func = coupling_func
        return self

    def domain(self, xmin: float, xmax: float, Nx: int) -> MFGProblemBuilder:
        """Set spatial domain parameters."""
        self.domain_params = {"xmin": xmin, "xmax": xmax, "Nx": Nx}
        return self

    def time(self, T: float, Nt: int) -> MFGProblemBuilder:
        """Set time domain parameters."""
        self.time_params = {"T": T, "Nt": Nt}
        return self

    def coefficients(self, sigma: float = 1.0, coupling_coefficient: float = 0.5) -> MFGProblemBuilder:
        """Set solver coefficients."""
        self.solver_params.update({"sigma": sigma, "coupling_coefficient": coupling_coefficient})
        return self

    def parameters(self, **params: Any) -> MFGProblemBuilder:
        """Set additional problem parameters."""
        self.components.parameters.update(params)
        return self

    def description(self, desc: str, problem_type: str = "custom") -> MFGProblemBuilder:
        """Set problem description and type."""
        self.components.description = desc
        self.components.problem_type = problem_type
        return self

    def build(self) -> MFGProblem:
        """Build the MFG problem."""
        # Validate Hamiltonians if both are provided
        has_hamiltonian = self.components.hamiltonian_func is not None
        has_hamiltonian_dm = self.components.hamiltonian_dm_func is not None

        if has_hamiltonian and not has_hamiltonian_dm:
            raise ValueError("Hamiltonian derivative function is required when Hamiltonian function is provided")
        if has_hamiltonian_dm and not has_hamiltonian:
            raise ValueError("Hamiltonian function is required when Hamiltonian derivative function is provided")

        # Set default domain if not specified
        if not self.domain_params:
            self.domain_params = {"xmin": 0.0, "xmax": 1.0, "Nx": 51}

        # Set default time domain if not specified
        if not self.time_params:
            self.time_params = {"T": 1.0, "Nt": 51}

        # Combine all parameters
        all_params = {**self.domain_params, **self.time_params, **self.solver_params}

        # Check if components are actually customized (any non-default value set)
        is_customized = (
            self.components.hamiltonian_func is not None
            or self.components.hamiltonian_dm_func is not None
            or self.components.hamiltonian_jacobian_func is not None
            or self.components.potential_func is not None
            or self.components.initial_density_func is not None
            or self.components.final_value_func is not None
            or self.components.boundary_conditions is not None
            or self.components.coupling_func is not None
            or len(self.components.parameters) > 0
            or self.components.description != "MFG Problem"
            or self.components.problem_type != "mfg"
        )

        # Create and return problem - with components only if customized
        if is_customized:
            return MFGProblem(components=self.components, **all_params)
        else:
            # No customization - create default problem without components
            return MFGProblem(**all_params)


# ============================================================================
# Convenience Functions and Backward Compatibility
# ============================================================================


def ExampleMFGProblem(**kwargs: Any) -> MFGProblem:
    """
    Create an MFG problem with default Hamiltonian (backward compatibility).

    This function provides backward compatibility for code that used
    the old ExampleMFGProblem class.
    """
    return MFGProblem(**kwargs)


def create_mfg_problem(hamiltonian_func: Callable, hamiltonian_dm_func: Callable, **kwargs: Any) -> MFGProblem:
    """
    Convenience function to create custom MFG problem.

    Args:
        hamiltonian_func: Custom Hamiltonian function
        hamiltonian_dm_func: Hamiltonian derivative function
        **kwargs: Domain, time, and solver parameters
    """
    # Extract configurations
    domain_config = {
        "xmin": kwargs.pop("xmin", 0.0),
        "xmax": kwargs.pop("xmax", 1.0),
        "Nx": kwargs.pop("Nx", 51),
    }

    time_config = {"T": kwargs.pop("T", 1.0), "Nt": kwargs.pop("Nt", 51)}

    solver_config = {
        "sigma": kwargs.pop("sigma", 1.0),
        "coupling_coefficient": kwargs.pop("coupling_coefficient", 0.5),
    }

    # Create components
    components = MFGComponents(
        hamiltonian_func=hamiltonian_func,
        hamiltonian_dm_func=hamiltonian_dm_func,
        **{k: v for k, v in kwargs.items() if k.endswith("_func")},
        parameters={k: v for k, v in kwargs.items() if not k.endswith("_func")},
    )

    return MFGProblem(components=components, **domain_config, **time_config, **solver_config)


# Test comment
# Test development config
