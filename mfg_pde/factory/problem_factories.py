"""
Unified problem factory supporting all MFG problem types.

This module provides factory functions for creating MFG problems with dual-output support:
- New unified MFGProblem (default, recommended)
- Old specialized classes (backward compatibility)

Supported Problem Types
-----------------------
- Standard HJB-FP (Hamiltonian-based)
- Network/Graph MFG (discrete domains)
- Variational/Lagrangian (optimization-based)
- Stochastic MFG (common noise)
- High-Dimensional MFG (n-D spatial domains)

IMPORTANT: Function Signature Requirements
-------------------------------------------
Hamiltonian and related functions must use the MFGProblem signature format:

def hamiltonian_func(x_idx, x_position, m_at_x, derivs, t_idx, current_time, problem):
    '''
    Args:
        x_idx: Grid index
        x_position: Physical position
        m_at_x: Density at position
        derivs: Dictionary with tuple keys:
            - (0,): u(x,t) function value
            - (1,): ∂u/∂x first derivative
        t_idx: Time index
        current_time: Physical time
        problem: MFGProblem reference
    '''
    du_dx = derivs.get((1,), 0.0)
    return 0.5 * du_dx**2 + m_at_x

See examples/basic/custom_hamiltonian_derivs_demo.py for complete examples.

Usage
-----
>>> # For simple examples, use ExampleMFGProblem
>>> from mfg_pde import ExampleMFGProblem
>>> problem = ExampleMFGProblem()
>>>
>>> # For custom problems, see custom_hamiltonian_derivs_demo.py
>>> # Factory functions are provided but require the complex signature above
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, overload

from mfg_pde.core.mfg_problem import MFGComponents, MFGProblem
from mfg_pde.utils.logging.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfg_pde.geometry import BoundaryConditions
    from mfg_pde.geometry.domain import Domain

logger = get_logger(__name__)

# =============================================================================
# Problem Type Constants
# =============================================================================

ProblemType = Literal["standard", "network", "variational", "stochastic", "highdim"]

# =============================================================================
# Main Factory Function
# =============================================================================


@overload
def create_mfg_problem(
    problem_type: ProblemType,
    components: MFGComponents,
    *,
    geometry: Domain,
    time_horizon: float = 1.0,
    num_timesteps: int = 100,
    use_unified: Literal[True] = True,
    **kwargs: Any,
) -> MFGProblem: ...


@overload
def create_mfg_problem(
    problem_type: Literal["network"],
    components: MFGComponents,
    *,
    geometry: Domain,
    time_horizon: float = 1.0,
    num_timesteps: int = 100,
    use_unified: Literal[False],
    **kwargs: Any,
) -> Any: ...  # NetworkMFGProblem


@overload
def create_mfg_problem(
    problem_type: Literal["variational"],
    components: MFGComponents,
    *,
    geometry: Domain,
    time_horizon: float = 1.0,
    num_timesteps: int = 100,
    use_unified: Literal[False],
    **kwargs: Any,
) -> Any: ...  # VariationalMFGProblem


@overload
def create_mfg_problem(
    problem_type: Literal["stochastic"],
    components: MFGComponents,
    *,
    geometry: Domain,
    time_horizon: float = 1.0,
    num_timesteps: int = 100,
    use_unified: Literal[False],
    **kwargs: Any,
) -> Any: ...  # StochasticMFGProblem


def create_mfg_problem(
    problem_type: ProblemType,
    components: MFGComponents,
    *,
    geometry: Domain,
    time_horizon: float = 1.0,
    num_timesteps: int = 100,
    use_unified: bool = True,
    **kwargs: Any,
) -> MFGProblem | Any:
    """
    Create MFG problem of specified type.

    This factory function provides a unified interface for creating any MFG problem
    type with support for both the new unified API and legacy specialized classes.

    Parameters
    ----------
    problem_type : {"standard", "network", "variational", "stochastic", "highdim"}
        Type of MFG problem to create
    components : MFGComponents
        Problem components (Hamiltonian, costs, densities, etc.)
    geometry : Domain
        Spatial domain
    time_horizon : float, default=1.0
        Final time T
    num_timesteps : int, default=100
        Number of time discretization points
    use_unified : bool, default=True
        If True, return unified MFGProblem (recommended)
        If False, return specialized class (backward compatibility)
    **kwargs : dict
        Additional problem-specific parameters

    Returns
    -------
    MFGProblem or specialized problem class
        If use_unified=True: MFGProblem instance
        If use_unified=False: Specialized class (NetworkMFGProblem, etc.)

    Examples
    --------
    >>> # Standard MFG problem (new unified API)
    >>> from mfg_pde.core import MFGComponents
    >>> from mfg_pde.factory import create_mfg_problem
    >>>
    >>> components = MFGComponents(
    ...     hamiltonian_func=H,
    ...     hamiltonian_dm_func=dH_dm,
    ...     terminal_cost_func=g,
    ...     initial_density_func=rho_0
    ... )
    >>> problem = create_mfg_problem("standard", components, geometry=domain)
    >>>
    >>> # Network MFG (new unified API)
    >>> components = MFGComponents(
    ...     network_geometry=graph,
    ...     node_interaction_func=node_cost,
    ...     edge_cost_func=edge_cost
    ... )
    >>> problem = create_mfg_problem("network", components, geometry=domain)
    >>>
    >>> # Backward compatibility
    >>> problem = create_mfg_problem("network", components, geometry=domain, use_unified=False)
    """
    # Set problem type in components
    if components.problem_type == "mfg":
        components.problem_type = problem_type

    if use_unified:
        # New unified API - create MFGProblem directly
        return MFGProblem(
            geometry=geometry,
            time_horizon=time_horizon,
            num_timesteps=num_timesteps,
            components=components,
            **kwargs,
        )
    else:
        # Legacy API - create specialized classes
        warnings.warn(
            f"Specialized {problem_type} problem classes are deprecated. "
            f"Use MFGProblem with use_unified=True (default).",
            DeprecationWarning,
            stacklevel=2,
        )

        if problem_type == "network":
            from mfg_pde.extensions.topology import NetworkMFGProblem

            return NetworkMFGProblem(
                geometry=geometry,
                time_horizon=time_horizon,
                num_timesteps=num_timesteps,
                components=components,
                **kwargs,
            )
        elif problem_type == "variational":
            from mfg_pde.solvers.variational import VariationalMFGProblem

            return VariationalMFGProblem(
                geometry=geometry,
                time_horizon=time_horizon,
                num_timesteps=num_timesteps,
                components=components,
                **kwargs,
            )
        elif problem_type == "stochastic":
            from mfg_pde.core.stochastic.stochastic_problem import StochasticMFGProblem

            return StochasticMFGProblem(
                geometry=geometry,
                time_horizon=time_horizon,
                num_timesteps=num_timesteps,
                components=components,
                **kwargs,
            )
        elif problem_type == "highdim":
            from mfg_pde.core.highdim_mfg_problem import GridBasedMFGProblem

            return GridBasedMFGProblem(
                geometry=geometry,
                time_horizon=time_horizon,
                num_timesteps=num_timesteps,
                components=components,
                **kwargs,
            )
        else:
            # Standard MFG
            return MFGProblem(
                geometry=geometry,
                time_horizon=time_horizon,
                num_timesteps=num_timesteps,
                components=components,
                **kwargs,
            )


# =============================================================================
# Convenience Factory Functions
# =============================================================================


def create_standard_problem(
    hamiltonian: Callable,
    hamiltonian_dm: Callable,
    terminal_cost: Callable,
    initial_density: Callable,
    geometry: Domain,
    *,
    potential: Callable | None = None,
    boundary_conditions: BoundaryConditions | None = None,
    time_horizon: float = 1.0,
    num_timesteps: int = 100,
    use_unified: bool = True,
    **kwargs: Any,
) -> MFGProblem | Any:
    """
    Create standard HJB-FP MFG problem.

    Parameters
    ----------
    hamiltonian : Callable
        Hamiltonian function H(x, p, m, t)
    hamiltonian_dm : Callable
        Derivative dH/dm(x, p, m, t)
    terminal_cost : Callable
        Terminal cost g(x)
    initial_density : Callable
        Initial density ρ₀(x)
    geometry : Domain
        Spatial domain
    potential : Callable, optional
        Potential function V(x)
    boundary_conditions : BoundaryConditions, optional
        Boundary conditions
    time_horizon : float, default=1.0
        Final time T
    num_timesteps : int, default=100
        Number of time steps
    use_unified : bool, default=True
        Use unified MFGProblem (True) or specialized class (False)

    Returns
    -------
    MFGProblem or specialized class
        Problem instance

    Examples
    --------
    >>> from mfg_pde.factory import create_standard_problem
    >>>
    >>> def H(x, p, m, t):
    ...     return 0.5 * p**2 + m
    >>>
    >>> def dH_dm(x, p, m, t):
    ...     return 1.0
    >>>
    >>> problem = create_standard_problem(
    ...     hamiltonian=H,
    ...     hamiltonian_dm=dH_dm,
    ...     terminal_cost=lambda x: x**2,
    ...     initial_density=lambda x: np.exp(-x**2),
    ...     geometry=domain
    ... )
    """
    components = MFGComponents(
        hamiltonian_func=hamiltonian,
        hamiltonian_dm_func=hamiltonian_dm,
        final_value_func=terminal_cost,
        initial_density_func=initial_density,
        potential_func=potential,
        boundary_conditions=boundary_conditions,
        problem_type="standard",
    )

    return create_mfg_problem(
        "standard",
        components,
        geometry=geometry,
        time_horizon=time_horizon,
        num_timesteps=num_timesteps,
        use_unified=use_unified,
        **kwargs,
    )


def create_network_problem(
    network_geometry: Any,
    node_interaction: Callable,
    edge_cost: Callable,
    initial_density: Callable | NDArray,
    geometry: Domain,
    *,
    edge_interaction: Callable | None = None,
    terminal_cost: Callable | None = None,
    time_horizon: float = 1.0,
    num_timesteps: int = 100,
    use_unified: bool = True,
    **kwargs: Any,
) -> MFGProblem | Any:
    """
    Create network/graph MFG problem.

    Parameters
    ----------
    network_geometry : NetworkGeometry or graph object
        Network/graph structure
    node_interaction : Callable
        Node interaction cost function
    edge_cost : Callable
        Edge traversal cost function
    initial_density : Callable or array
        Initial density on nodes
    geometry : Domain
        Domain (typically NetworkDomain)
    edge_interaction : Callable, optional
        Edge interaction function
    terminal_cost : Callable, optional
        Terminal cost on nodes
    time_horizon : float, default=1.0
        Final time T
    num_timesteps : int, default=100
        Number of time steps
    use_unified : bool, default=True
        Use unified MFGProblem (True) or NetworkMFGProblem (False)

    Returns
    -------
    MFGProblem or NetworkMFGProblem
        Problem instance

    Examples
    --------
    >>> from mfg_pde.factory import create_network_problem
    >>>
    >>> # Create network MFG on graph
    >>> problem = create_network_problem(
    ...     network_geometry=graph,
    ...     node_interaction=lambda node, m: m[node]**2,
    ...     edge_cost=lambda edge: 1.0,
    ...     initial_density=initial_m,
    ...     geometry=network_domain
    ... )
    """
    # Convert initial density to function if array
    if callable(initial_density):
        initial_density_func = initial_density
    else:
        initial_density_func = lambda x: initial_density  # noqa: E731

    components = MFGComponents(
        network_geometry=network_geometry,
        node_interaction_func=node_interaction,
        edge_cost_func=edge_cost,
        edge_interaction_func=edge_interaction,
        initial_density_func=initial_density_func,
        final_value_func=terminal_cost,
        problem_type="network",
    )

    return create_mfg_problem(
        "network",
        components,
        geometry=geometry,
        time_horizon=time_horizon,
        num_timesteps=num_timesteps,
        use_unified=use_unified,
        **kwargs,
    )


def create_variational_problem(
    lagrangian: Callable,
    lagrangian_dx: Callable,
    lagrangian_dv: Callable,
    lagrangian_dm: Callable,
    terminal_cost: Callable,
    initial_density: Callable,
    geometry: Domain,
    *,
    trajectory_cost: Callable | None = None,
    state_constraints: list[Callable] | None = None,
    velocity_constraints: list[Callable] | None = None,
    time_horizon: float = 1.0,
    num_timesteps: int = 100,
    use_unified: bool = True,
    **kwargs: Any,
) -> MFGProblem | Any:
    """
    Create variational/Lagrangian MFG problem.

    Parameters
    ----------
    lagrangian : Callable
        Lagrangian function L(x, v, m, t)
    lagrangian_dx : Callable
        Derivative ∂L/∂x
    lagrangian_dv : Callable
        Derivative ∂L/∂v
    lagrangian_dm : Callable
        Derivative ∂L/∂m
    terminal_cost : Callable
        Terminal cost Ψ(x)
    initial_density : Callable
        Initial density ρ₀(x)
    geometry : Domain
        Spatial domain
    trajectory_cost : Callable, optional
        Running cost along trajectory
    state_constraints : list of Callable, optional
        State constraints c(x) ≤ 0
    velocity_constraints : list of Callable, optional
        Velocity constraints h(v) ≤ 0
    time_horizon : float, default=1.0
        Final time T
    num_timesteps : int, default=100
        Number of time steps
    use_unified : bool, default=True
        Use unified MFGProblem (True) or VariationalMFGProblem (False)

    Returns
    -------
    MFGProblem or VariationalMFGProblem
        Problem instance

    Examples
    --------
    >>> from mfg_pde.factory import create_variational_problem
    >>>
    >>> def L(x, v, m, t):
    ...     return 0.5 * v**2 + m
    >>>
    >>> problem = create_variational_problem(
    ...     lagrangian=L,
    ...     lagrangian_dx=lambda x, v, m, t: 0,
    ...     lagrangian_dv=lambda x, v, m, t: v,
    ...     lagrangian_dm=lambda x, v, m, t: 1,
    ...     terminal_cost=lambda x: x**2,
    ...     initial_density=lambda x: np.exp(-x**2),
    ...     geometry=domain
    ... )
    """
    components = MFGComponents(
        lagrangian_func=lagrangian,
        lagrangian_dx_func=lagrangian_dx,
        lagrangian_dv_func=lagrangian_dv,
        lagrangian_dm_func=lagrangian_dm,
        terminal_cost_func=terminal_cost,
        initial_density_func=initial_density,
        trajectory_cost_func=trajectory_cost,
        state_constraints=state_constraints,
        velocity_constraints=velocity_constraints,
        problem_type="variational",
    )

    return create_mfg_problem(
        "variational",
        components,
        geometry=geometry,
        time_horizon=time_horizon,
        num_timesteps=num_timesteps,
        use_unified=use_unified,
        **kwargs,
    )


def create_stochastic_problem(
    hamiltonian: Callable,
    hamiltonian_dm: Callable,
    terminal_cost: Callable,
    initial_density: Callable,
    geometry: Domain,
    *,
    noise_intensity: float = 0.0,
    common_noise: Callable | None = None,
    idiosyncratic_noise: Callable | None = None,
    correlation_matrix: NDArray | None = None,
    potential: Callable | None = None,
    time_horizon: float = 1.0,
    num_timesteps: int = 100,
    use_unified: bool = True,
    **kwargs: Any,
) -> MFGProblem | Any:
    """
    Create stochastic MFG problem with common noise.

    Parameters
    ----------
    hamiltonian : Callable
        Hamiltonian function H(x, p, m, ξ, t)
    hamiltonian_dm : Callable
        Derivative dH/dm
    terminal_cost : Callable
        Terminal cost g(x, ξ)
    initial_density : Callable
        Initial density ρ₀(x)
    geometry : Domain
        Spatial domain
    noise_intensity : float, default=0.0
        Noise intensity σ
    common_noise : Callable, optional
        Common noise function ξ(t)
    idiosyncratic_noise : Callable, optional
        Idiosyncratic noise function
    correlation_matrix : ndarray, optional
        Noise correlation matrix
    potential : Callable, optional
        Potential function V(x)
    time_horizon : float, default=1.0
        Final time T
    num_timesteps : int, default=100
        Number of time steps
    use_unified : bool, default=True
        Use unified MFGProblem (True) or StochasticMFGProblem (False)

    Returns
    -------
    MFGProblem or StochasticMFGProblem
        Problem instance

    Examples
    --------
    >>> from mfg_pde.factory import create_stochastic_problem
    >>>
    >>> problem = create_stochastic_problem(
    ...     hamiltonian=H,
    ...     hamiltonian_dm=dH_dm,
    ...     terminal_cost=g,
    ...     initial_density=rho_0,
    ...     geometry=domain,
    ...     noise_intensity=0.5,
    ...     common_noise=lambda t: np.sin(t)
    ... )
    """
    components = MFGComponents(
        hamiltonian_func=hamiltonian,
        hamiltonian_dm_func=hamiltonian_dm,
        final_value_func=terminal_cost,
        initial_density_func=initial_density,
        potential_func=potential,
        noise_intensity=noise_intensity,
        common_noise_func=common_noise,
        idiosyncratic_noise_func=idiosyncratic_noise,
        correlation_matrix=correlation_matrix,
        problem_type="stochastic",
    )

    return create_mfg_problem(
        "stochastic",
        components,
        geometry=geometry,
        time_horizon=time_horizon,
        num_timesteps=num_timesteps,
        use_unified=use_unified,
        **kwargs,
    )


def create_highdim_problem(
    hamiltonian: Callable,
    hamiltonian_dm: Callable,
    terminal_cost: Callable,
    initial_density: Callable,
    geometry: Domain,
    *,
    dimension: int,
    potential: Callable | None = None,
    boundary_conditions: BoundaryConditions | None = None,
    time_horizon: float = 1.0,
    num_timesteps: int = 100,
    use_unified: bool = True,
    **kwargs: Any,
) -> MFGProblem | Any:
    """
    Create high-dimensional MFG problem (d > 3).

    Parameters
    ----------
    hamiltonian : Callable
        Hamiltonian function H(x, p, m, t) where x ∈ ℝᵈ
    hamiltonian_dm : Callable
        Derivative dH/dm
    terminal_cost : Callable
        Terminal cost g(x)
    initial_density : Callable
        Initial density ρ₀(x)
    geometry : Domain
        High-dimensional domain
    dimension : int
        Spatial dimension d
    potential : Callable, optional
        Potential function V(x)
    boundary_conditions : BoundaryConditions, optional
        Boundary conditions
    time_horizon : float, default=1.0
        Final time T
    num_timesteps : int, default=100
        Number of time steps
    use_unified : bool, default=True
        Use unified MFGProblem (True) or GridBasedMFGProblem (False)

    Returns
    -------
    MFGProblem or GridBasedMFGProblem
        Problem instance

    Examples
    --------
    >>> from mfg_pde.factory import create_highdim_problem
    >>>
    >>> # 5D MFG problem
    >>> problem = create_highdim_problem(
    ...     hamiltonian=H,
    ...     hamiltonian_dm=dH_dm,
    ...     terminal_cost=g,
    ...     initial_density=rho_0,
    ...     geometry=domain_5d,
    ...     dimension=5
    ... )
    """
    components = MFGComponents(
        hamiltonian_func=hamiltonian,
        hamiltonian_dm_func=hamiltonian_dm,
        final_value_func=terminal_cost,
        initial_density_func=initial_density,
        potential_func=potential,
        boundary_conditions=boundary_conditions,
        problem_type="highdim",
    )

    return create_mfg_problem(
        "highdim",
        components,
        geometry=geometry,
        time_horizon=time_horizon,
        num_timesteps=num_timesteps,
        use_unified=use_unified,
        dimension=dimension,
        **kwargs,
    )


# =============================================================================
# Legacy Compatibility Functions
# =============================================================================


def create_lq_problem(
    *,
    geometry: Domain,
    terminal_cost: Callable,
    initial_density: Callable,
    running_cost_control: float = 1.0,
    running_cost_congestion: float = 1.0,
    time_horizon: float = 1.0,
    num_timesteps: int = 100,
    use_unified: bool = True,
    **kwargs: Any,
) -> MFGProblem | Any:
    """
    Create Linear-Quadratic (LQ) MFG problem.

    Standard LQ-MFG with quadratic control cost and linear congestion.
    Uses Hamiltonian: H(x, p, m) = (α/2)|p|² + βm

    Parameters
    ----------
    geometry : Domain
        Spatial domain
    terminal_cost : Callable
        Terminal cost g(x)
    initial_density : Callable
        Initial density ρ₀(x)
    running_cost_control : float, default=1.0
        Control cost coefficient α
    running_cost_congestion : float, default=1.0
        Congestion cost coefficient β
    time_horizon : float, default=1.0
        Final time T
    num_timesteps : int, default=100
        Number of time steps
    use_unified : bool, default=True
        Use unified MFGProblem (True) or LQMFGProblem (False)

    Returns
    -------
    MFGProblem or LQMFGProblem
        Problem instance

    Examples
    --------
    >>> from mfg_pde.factory import create_lq_problem
    >>>
    >>> problem = create_lq_problem(
    ...     geometry=domain,
    ...     terminal_cost=lambda x: x**2,
    ...     initial_density=lambda x: np.exp(-x**2),
    ...     running_cost_control=0.5,
    ...     running_cost_congestion=1.0
    ... )
    """
    alpha = running_cost_control
    beta = running_cost_congestion

    def hamiltonian(x: Any, p: Any, m: Any, t: float) -> float:
        """LQ Hamiltonian: H = (α/2)|p|² + βm"""
        import numpy as np

        return 0.5 * alpha * np.sum(p**2) + beta * m

    def hamiltonian_dm(x: Any, p: Any, m: Any, t: float) -> float:
        """dH/dm = β"""
        return beta

    return create_standard_problem(
        hamiltonian=hamiltonian,
        hamiltonian_dm=hamiltonian_dm,
        terminal_cost=terminal_cost,
        initial_density=initial_density,
        geometry=geometry,
        time_horizon=time_horizon,
        num_timesteps=num_timesteps,
        use_unified=use_unified,
        **kwargs,
    )


def create_crowd_problem(
    *,
    geometry: Domain,
    target_location: NDArray | Callable,
    initial_density: Callable,
    running_cost_control: float = 1.0,
    congestion_sensitivity: float = 1.0,
    time_horizon: float = 1.0,
    num_timesteps: int = 100,
    use_unified: bool = True,
    **kwargs: Any,
) -> MFGProblem | Any:
    """
    Create crowd dynamics MFG problem.

    Models crowd motion toward target with congestion effects.
    Uses potential V(x) = |x - x_target|²/2

    Parameters
    ----------
    geometry : Domain
        Spatial domain
    target_location : array or Callable
        Target location x_target or function
    initial_density : Callable
        Initial crowd density ρ₀(x)
    running_cost_control : float, default=1.0
        Control cost coefficient
    congestion_sensitivity : float, default=1.0
        Congestion sensitivity
    time_horizon : float, default=1.0
        Final time T
    num_timesteps : int, default=100
        Number of time steps
    use_unified : bool, default=True
        Use unified MFGProblem (True) or CrowdDynamicsProblem (False)

    Returns
    -------
    MFGProblem or CrowdDynamicsProblem
        Problem instance

    Examples
    --------
    >>> from mfg_pde.factory import create_crowd_problem
    >>> import numpy as np
    >>>
    >>> problem = create_crowd_problem(
    ...     geometry=domain,
    ...     target_location=np.array([1.0, 1.0]),
    ...     initial_density=lambda x: np.exp(-np.linalg.norm(x)**2)
    ... )
    """
    import numpy as np

    alpha = running_cost_control
    beta = congestion_sensitivity

    # Create potential function
    if callable(target_location):
        target_func = target_location
    else:
        target = np.asarray(target_location)
        target_func = lambda x: 0.5 * np.sum((x - target) ** 2)  # noqa: E731

    def hamiltonian(x: Any, p: Any, m: Any, t: float) -> float:
        """Crowd Hamiltonian"""
        return 0.5 * alpha * np.sum(p**2) + beta * m

    def hamiltonian_dm(x: Any, p: Any, m: Any, t: float) -> float:
        """dH/dm"""
        return beta

    def terminal_cost(x: Any) -> float:
        """Terminal cost: distance to target"""
        return target_func(x)

    return create_standard_problem(
        hamiltonian=hamiltonian,
        hamiltonian_dm=hamiltonian_dm,
        terminal_cost=terminal_cost,
        initial_density=initial_density,
        potential=target_func,
        geometry=geometry,
        time_horizon=time_horizon,
        num_timesteps=num_timesteps,
        use_unified=use_unified,
        **kwargs,
    )
