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

    def __init__(
        self,
        # Legacy 1D parameters (backward compatible)
        xmin: float | None = None,
        xmax: float | None = None,
        Nx: int | None = None,
        # New n-D parameters
        spatial_bounds: list[tuple[float, float]] | None = None,
        spatial_discretization: list[int] | None = None,
        # Common parameters
        T: float = 1.0,
        Nt: int = 51,
        sigma: float = 1.0,
        coefCT: float = 0.5,
        components: MFGComponents | None = None,
        suppress_warnings: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize MFG problem with support for arbitrary spatial dimensions.

        Supports two initialization modes:
        1. Legacy 1D mode (backward compatible): Specify Nx, xmin, xmax
        2. N-dimensional mode (new): Specify spatial_bounds, spatial_discretization

        Args:
            xmin, xmax, Nx: Legacy 1D spatial domain parameters
            spatial_bounds: List of (min, max) tuples for each dimension
                           Example: [(0, 1), (0, 1)] for 2D unit square
            spatial_discretization: List of grid points per dimension
                                   Example: [50, 50] for 51×51 grid
            T, Nt: Time domain parameters
            sigma: Diffusion coefficient
            coefCT: Control cost coefficient
            components: Optional MFGComponents for custom problem definition
            suppress_warnings: Suppress computational feasibility warnings
            **kwargs: Additional parameters

        Examples:
            # 1D (legacy API - 100% backward compatible)
            problem = MFGProblem(Nx=100, xmin=0.0, xmax=1.0, Nt=100)

            # 2D (new API)
            problem = MFGProblem(
                spatial_bounds=[(0, 1), (0, 1)],
                spatial_discretization=[50, 50],
                Nt=50
            )

            # 3D (new API)
            problem = MFGProblem(
                spatial_bounds=[(0, 1), (0, 1), (0, 1)],
                spatial_discretization=[30, 30, 30],
                Nt=30
            )
        """
        import warnings

        # Detect initialization mode
        if Nx is not None and spatial_bounds is None:
            # Mode 1: Legacy 1D initialization
            if xmin is None:
                xmin = 0.0
            if xmax is None:
                xmax = 1.0
            self._init_1d_legacy(xmin, xmax, Nx, T, Nt, sigma, coefCT)
        elif spatial_bounds is not None and Nx is None:
            # Mode 2: N-dimensional initialization
            self._init_nd(spatial_bounds, spatial_discretization, T, Nt, sigma, coefCT, suppress_warnings)
        elif Nx is None and spatial_bounds is None:
            # Default: 1D with default parameters
            warnings.warn(
                "No spatial domain specified. Using default 1D domain: [0, 1] with 51 points.",
                UserWarning,
                stacklevel=2,
            )
            self._init_1d_legacy(0.0, 1.0, 51, T, Nt, sigma, coefCT)
        else:
            raise ValueError(
                "Ambiguous initialization: Provide EITHER (Nx, xmin, xmax) for 1D "
                "OR (spatial_bounds, spatial_discretization) for n-D, but not both."
            )

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

    def _init_1d_legacy(
        self,
        xmin: float,
        xmax: float,
        Nx: int,
        T: float,
        Nt: int,
        sigma: float,
        coefCT: float,
    ) -> None:
        """Initialize problem in legacy 1D mode (100% backward compatible)."""
        # Set dimension
        self.dimension = 1

        # Legacy 1D attributes (exactly as before)
        self.xmin: float = xmin
        self.xmax: float = xmax
        self.Lx: float = xmax - xmin
        self.Nx: int = Nx
        self.Dx: float = (xmax - xmin) / Nx if Nx > 0 else 0.0

        # Time domain
        self.T: float = T
        self.Nt: int = Nt
        self.Dt: float = T / Nt if Nt > 0 else 0.0

        # Grid arrays
        self.xSpace: np.ndarray = np.linspace(xmin, xmax, Nx + 1, endpoint=True)
        self.tSpace: np.ndarray = np.linspace(0, T, Nt + 1, endpoint=True)

        # Coefficients
        self.sigma: float = sigma
        self.coefCT: float = coefCT

        # New n-D attributes for consistency
        self.spatial_shape = (Nx + 1,)  # 1D shape: (Nx+1,)
        self.spatial_bounds = [(xmin, xmax)]
        self.spatial_discretization = [Nx]

        # Grid object (None for 1D - not needed, kept for compatibility)
        self._grid = None

    def _init_nd(
        self,
        spatial_bounds: list[tuple[float, float]],
        spatial_discretization: list[int] | None,
        T: float,
        Nt: int,
        sigma: float,
        coefCT: float,
        suppress_warnings: bool,
    ) -> None:
        """Initialize problem in n-dimensional mode."""

        # Validate inputs
        if not spatial_bounds:
            raise ValueError("spatial_bounds must be a non-empty list of (min, max) tuples")

        dimension = len(spatial_bounds)

        if spatial_discretization is None:
            # Default: 51 points per dimension
            spatial_discretization = [51] * dimension
        elif len(spatial_discretization) != dimension:
            raise ValueError(
                f"spatial_discretization must have {dimension} elements (one per dimension), "
                f"got {len(spatial_discretization)}"
            )

        # Store dimension
        self.dimension = dimension

        # Store n-D parameters
        self.spatial_bounds = spatial_bounds
        self.spatial_discretization = spatial_discretization

        # For TensorProductGrid: num_points=[N] creates N points (not N+1)
        # So spatial_shape should match spatial_discretization directly
        self.spatial_shape = tuple(spatial_discretization)

        # Time domain
        self.T: float = T
        self.Nt: int = Nt
        self.Dt: float = T / Nt if Nt > 0 else 0.0
        self.tSpace: np.ndarray = np.linspace(0, T, Nt + 1, endpoint=True)

        # Coefficients
        self.sigma: float = sigma
        self.coefCT: float = coefCT

        # Legacy 1D attributes (set to None for n-D, dimension > 1)
        if dimension == 1:
            # Special case: 1D via n-D API (for consistency)
            self.xmin = spatial_bounds[0][0]
            self.xmax = spatial_bounds[0][1]
            self.Lx = self.xmax - self.xmin
            self.Nx = spatial_discretization[0]
            self.Dx = (self.xmax - self.xmin) / self.Nx if self.Nx > 0 else 0.0
            self.xSpace = np.linspace(self.xmin, self.xmax, self.Nx + 1, endpoint=True)
            self._grid = None
        else:
            # True n-D (dimension >= 2)
            self.xmin = None
            self.xmax = None
            self.Lx = None
            self.Nx = None
            self.Dx = None
            self.xSpace = None

            # Create TensorProductGrid
            from mfg_pde.geometry import TensorProductGrid

            self._grid = TensorProductGrid(
                dimension=dimension, bounds=spatial_bounds, num_points=spatial_discretization
            )

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
        else:
            # n-D normalization (integrate over all dimensions)
            # For tensor product grid: integral = sum(m) * prod(dx_i)
            dx_prod = np.prod(
                [
                    (bounds[1] - bounds[0]) / n
                    for bounds, n in zip(self.spatial_bounds, self.spatial_discretization, strict=False)
                ]
            )
            integral_m_init = np.sum(self.m_init) * dx_prod

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

        hamiltonian_control_part = 0.5 * self.coefCT * (term_npart_sq + term_ppart_sq)

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
            coefCT = self.coefCT

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

                J_D_H[i] = coefCT * (npart(p1_i) + ppart(p2_i)) / (Dx**2)
                J_L_H[i] = -coefCT * ppart(p2_i) / (Dx**2)
                J_U_H[i] = -coefCT * npart(p1_i) / (Dx**2)

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
                "coefficients": {"sigma": self.sigma, "coefCT": self.coefCT},
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
                "coefficients": {"sigma": self.sigma, "coefCT": self.coefCT},
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

    def coefficients(self, sigma: float = 1.0, coefCT: float = 0.5) -> MFGProblemBuilder:
        """Set solver coefficients."""
        self.solver_params.update({"sigma": sigma, "coefCT": coefCT})
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
        "coefCT": kwargs.pop("coefCT", 0.5),
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
