"""
Core Type Protocols for MFG_PDE

These protocols define the clean, simple interfaces that users interact with.
They use duck typing - any object that implements these methods will work,
regardless of inheritance.
"""

from __future__ import annotations

from typing import Protocol, Any, runtime_checkable
from numpy.typing import NDArray


@runtime_checkable
class MFGProblem(Protocol):
    """
    Protocol for MFG problem objects.

    Any object that implements these methods can be used as an MFG problem,
    regardless of its actual class hierarchy.
    """

    def get_domain_bounds(self) -> tuple[float, float]:
        """Get spatial domain bounds (xmin, xmax)."""
        ...

    def get_time_horizon(self) -> float:
        """Get time horizon T."""
        ...

    def evaluate_hamiltonian(self, x: float, p: float, m: float, t: float) -> float:
        """Evaluate Hamiltonian H(x, p, m, t)."""
        ...

    def get_initial_density(self) -> NDArray:
        """Get initial density m_0(x)."""
        ...

    def get_initial_value_function(self) -> NDArray:
        """Get initial value function u_0(x) (or terminal condition)."""
        ...


@runtime_checkable
class MFGSolver(Protocol):
    """
    Protocol for MFG solver objects.

    Any object that implements solve() can be used as a solver.
    """

    def solve(self, problem: MFGProblem, **kwargs) -> MFGResult:
        """
        Solve the MFG problem.

        Args:
            problem: MFG problem to solve
            **kwargs: Additional solver parameters

        Returns:
            Solution result
        """
        ...


@runtime_checkable
class MFGResult(Protocol):
    """
    Protocol for MFG solution results.

    Provides clean access to solutions and analysis methods.
    """

    @property
    def u(self) -> NDArray:
        """Value function u(t, x)."""
        ...

    @property
    def m(self) -> NDArray:
        """Density function m(t, x)."""
        ...

    @property
    def converged(self) -> bool:
        """Whether the solver converged."""
        ...

    @property
    def iterations(self) -> int:
        """Number of iterations performed."""
        ...

    def plot_solution(self, **kwargs) -> None:
        """Plot the solution (u and m)."""
        ...

    def export_data(self, filename: str) -> None:
        """Export solution data to file."""
        ...


@runtime_checkable
class SolverConfig(Protocol):
    """
    Protocol for solver configuration objects.

    Provides a clean interface for solver parameters.
    """

    @property
    def max_iterations(self) -> int:
        """Maximum number of iterations."""
        ...

    @property
    def tolerance(self) -> float:
        """Convergence tolerance."""
        ...

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a configuration parameter."""
        ...


# Type aliases for commonly used types (simple, memorable names)
SolutionArray = NDArray  # For u(t,x) and m(t,x) arrays
SpatialGrid = NDArray   # For x-coordinates
TimeGrid = NDArray      # For t-coordinates