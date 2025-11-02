"""
Fundamental Nonlinear System Solvers.

This module provides generic, reusable solvers for nonlinear systems that
appear throughout MFG algorithms. These are the building blocks for value
iteration, policy iteration, and Newton-based methods.

Control Theory Context:
    - FixedPointSolver: Value iteration
    - NewtonSolver: Newton-Raphson dynamic programming
    - PolicyIterationSolver: Howard's policy iteration algorithm

All solvers are dimension-agnostic and preserve array shapes.

References:
    - Bertsekas (2012): Dynamic Programming and Optimal Control
    - Nocedal & Wright (2006): Numerical Optimization
    - Walker & Ni (2011): Anderson acceleration for fixed-point iterations
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SolverInfo:
    """
    Container for solver convergence information.

    Attributes:
        converged: Whether solver converged within tolerance
        iterations: Number of iterations performed
        residual: Final residual norm
        residual_history: Residual at each iteration
        solver_time: Total time in seconds (if tracked)
        extra: Additional solver-specific information
    """

    def __init__(
        self,
        converged: bool,
        iterations: int,
        residual: float,
        residual_history: list[float],
        solver_time: float = 0.0,
        **extra: Any,
    ):
        self.converged = converged
        self.iterations = iterations
        self.residual = residual
        self.residual_history = residual_history
        self.solver_time = solver_time
        self.extra = extra

    def __repr__(self) -> str:
        status = "converged" if self.converged else "not converged"
        return f"SolverInfo({status}, iterations={self.iterations}, residual={self.residual:.2e})"


class NonlinearSolver(ABC):
    """
    Abstract base class for nonlinear system solvers.

    All solvers follow the same interface:
        x_solution, info = solver.solve(F_or_G, x0)

    where F_or_G is either:
        - F: x → residual (for Newton: solve F(x) = 0)
        - G: x → next_x (for fixed-point: solve x = G(x))

    Shape preservation: Output x has same shape as input x0.
    """

    @abstractmethod
    def solve(
        self,
        func: Callable[[NDArray], NDArray],
        x0: NDArray,
        **kwargs: Any,
    ) -> tuple[NDArray, SolverInfo]:
        """
        Solve nonlinear system.

        Args:
            func: Function defining the system
            x0: Initial guess (any shape)
            **kwargs: Solver-specific options

        Returns:
            x: Solution (same shape as x0)
            info: Convergence information
        """


class FixedPointSolver(NonlinearSolver):
    """
    Damped fixed-point iteration solver.

    Solves: x = G(x) via iteration
        x^{k+1} = (1-ω)x^k + ω·G(x^k)

    where ω ∈ (0,1] is the damping factor:
        - ω = 1.0: Full update (no damping)
        - ω < 1.0: Damped update (more stable, slower)

    Control theory: This is value iteration with relaxation.

    Example:
        >>> G = lambda x: np.cos(x)  # Solve x = cos(x)
        >>> solver = FixedPointSolver(damping_factor=0.8, tol=1e-6)
        >>> x, info = solver.solve(G, x0=0.5)
        >>> print(f"Solution: {x}, converged: {info.converged}")

    Notes:
        - Works for any array shape (scalars, vectors, matrices, tensors)
        - Damping recommended for stability (ω ≈ 0.5-0.8)
        - Use Anderson acceleration for faster convergence
    """

    def __init__(
        self,
        damping_factor: float = 1.0,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        norm_type: Literal["relative", "absolute"] = "relative",
    ):
        """
        Initialize fixed-point solver.

        Args:
            damping_factor: Damping ω ∈ (0, 1]
                - 1.0 = full update (fast but may oscillate)
                - 0.5-0.8 = damped (stable, recommended)
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            norm_type: 'relative' or 'absolute' residual norm
        """
        if not 0 < damping_factor <= 1.0:
            raise ValueError(f"damping_factor must be in (0,1], got {damping_factor}")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be > 0, got {tolerance}")

        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.norm_type = norm_type

    def solve(
        self,
        G: Callable[[NDArray], NDArray],
        x0: NDArray,
        **kwargs: Any,
    ) -> tuple[NDArray, SolverInfo]:
        """
        Solve fixed-point problem x = G(x).

        Args:
            G: Fixed-point map G: x → x
            x0: Initial guess (any shape)

        Returns:
            x: Solution (same shape as x0)
            info: Convergence information
        """
        import time

        start_time = time.time()
        # Convert to numpy array to handle scalars
        x_current = np.asarray(x0, dtype=float).copy()
        original_shape = np.asarray(x0).shape
        is_scalar = original_shape == ()

        residual_history = []
        omega = self.damping_factor

        for iteration in range(self.max_iterations):
            # Evaluate fixed-point map
            x_new = G(x_current)

            # Apply damping
            x_updated = (1 - omega) * x_current + omega * x_new

            # Compute residual: ||x_updated - x_current||
            diff = x_updated - x_current
            residual_abs = np.linalg.norm(diff.flatten())

            if self.norm_type == "relative":
                residual = residual_abs / (np.linalg.norm(x_current.flatten()) + 1e-10)
            else:  # absolute
                residual = residual_abs

            residual_history.append(float(residual))

            # Check convergence
            if residual < self.tolerance:
                solver_time = time.time() - start_time
                result = x_updated.item() if is_scalar else x_updated
                return result, SolverInfo(
                    converged=True,
                    iterations=iteration + 1,
                    residual=residual,
                    residual_history=residual_history,
                    solver_time=solver_time,
                )

            x_current = x_updated

        # Maximum iterations reached
        solver_time = time.time() - start_time
        result = x_current.item() if is_scalar else x_current
        return result, SolverInfo(
            converged=False,
            iterations=self.max_iterations,
            residual=residual_history[-1] if residual_history else float("inf"),
            residual_history=residual_history,
            solver_time=solver_time,
        )


class NewtonSolver(NonlinearSolver):
    """
    Newton's method for nonlinear systems.

    Solves: F(x) = 0 via Newton iteration
        J(x^k) δx = -F(x^k)
        x^{k+1} = x^k + α·δx

    where:
        - J = ∂F/∂x is the Jacobian
        - α is the line search parameter (optional)

    Control theory: Newton-Raphson dynamic programming.

    Features:
        - Automatic Jacobian via finite differences
        - User-provided Jacobian (faster if available)
        - Sparse matrix support for large systems
        - Optional line search for robustness

    Example:
        >>> F = lambda x: x**2 - 4  # Solve x^2 = 4
        >>> solver = NewtonSolver(tol=1e-8)
        >>> x, info = solver.solve(F, x0=1.0)
        >>> print(f"Solution: {x}, iterations: {info.iterations}")

    Notes:
        - Quadratic convergence near solution
        - Requires good initial guess
        - May fail for poor x0 (use line search)
    """

    def __init__(
        self,
        max_iterations: int = 30,
        tolerance: float = 1e-6,
        jacobian: Callable[[NDArray], NDArray | sparse.spmatrix] | None = None,
        sparse: bool = True,
        line_search: bool = False,
        finite_diff_epsilon: float = 1e-7,
    ):
        """
        Initialize Newton solver.

        Args:
            max_iterations: Maximum Newton iterations
            tolerance: Convergence tolerance on ||F(x)||
            jacobian: Optional Jacobian function J: x → ∂F/∂x
                     If None, uses finite differences
            sparse: Use sparse linear solver (for large systems)
            line_search: Enable backtracking line search
            finite_diff_epsilon: Step size for finite differences
        """
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be > 0, got {tolerance}")
        if finite_diff_epsilon <= 0:
            raise ValueError(f"finite_diff_epsilon must be > 0, got {finite_diff_epsilon}")

        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.jacobian_func = jacobian
        self.use_sparse = sparse
        self.line_search = line_search
        self.epsilon = finite_diff_epsilon

    def solve(
        self,
        F: Callable[[NDArray], NDArray],
        x0: NDArray,
        jacobian: Callable[[NDArray], NDArray | sparse.spmatrix] | None = None,
        **kwargs: Any,
    ) -> tuple[NDArray, SolverInfo]:
        """
        Solve F(x) = 0 using Newton's method.

        Args:
            F: Residual function F: x → ℝ^n (same shape as x)
            x0: Initial guess (any shape)
            jacobian: Optional Jacobian (overrides self.jacobian_func)

        Returns:
            x: Solution (same shape as x0)
            info: Convergence information
        """
        import time

        start_time = time.time()
        # Convert to numpy array to handle scalars
        x_current = np.asarray(x0, dtype=float).copy()
        original_shape = np.asarray(x0).shape
        is_scalar = original_shape == ()

        residual_history = []
        jacobian_evals = 0

        # Use provided jacobian or default
        jac_func = jacobian or self.jacobian_func

        for iteration in range(self.max_iterations):
            # Evaluate residual
            F_current = F(x_current)
            residual_norm = np.linalg.norm(F_current.flatten())
            residual_history.append(float(residual_norm))

            # Check convergence
            if residual_norm < self.tolerance:
                solver_time = time.time() - start_time
                result = x_current.item() if is_scalar else x_current
                return result, SolverInfo(
                    converged=True,
                    iterations=iteration + 1,
                    residual=residual_norm,
                    residual_history=residual_history,
                    solver_time=solver_time,
                    jacobian_evals=jacobian_evals,
                )

            # Compute Jacobian
            if jac_func is not None:
                J = jac_func(x_current)
                jacobian_evals += 1
            else:
                # Automatic finite difference Jacobian
                J = self._finite_difference_jacobian(F, x_current, F_current)
                jacobian_evals += 1

            # Solve linear system: J δx = -F
            delta_x = self._solve_linear_system(J, -F_current.flatten(), original_shape)

            # Line search (optional)
            if self.line_search:
                alpha = self._backtracking_line_search(F, x_current, delta_x, F_current)
            else:
                alpha = 1.0

            # Update
            x_current = x_current + alpha * delta_x

        # Maximum iterations reached
        solver_time = time.time() - start_time
        F_final = F(x_current)
        residual_final = np.linalg.norm(F_final.flatten())

        result = x_current.item() if is_scalar else x_current
        return result, SolverInfo(
            converged=False,
            iterations=self.max_iterations,
            residual=float(residual_final),
            residual_history=residual_history,
            solver_time=solver_time,
            jacobian_evals=jacobian_evals,
        )

    def _finite_difference_jacobian(
        self,
        F: Callable[[NDArray], NDArray],
        x: NDArray,
        F_x: NDArray,
    ) -> NDArray | sparse.spmatrix:
        """
        Compute Jacobian via forward finite differences.

        J[i,j] ≈ (F(x + ε·e_j)[i] - F(x)[i]) / ε
        """
        x_flat = x.flatten()
        F_flat = F_x.flatten()
        n = len(x_flat)

        if self.use_sparse:
            # Build sparse Jacobian (assumes sparsity pattern)
            J = sparse.lil_matrix((n, n), dtype=np.float64)
        else:
            J = np.zeros((n, n), dtype=np.float64)

        for j in range(n):
            # Perturb j-th component
            x_pert = x_flat.copy()
            x_pert[j] += self.epsilon

            # Evaluate F at perturbed point
            F_pert = F(x_pert.reshape(x.shape)).flatten()

            # Finite difference
            J[:, j] = (F_pert - F_flat) / self.epsilon

        return J.tocsr() if self.use_sparse else J

    def _solve_linear_system(
        self,
        J: NDArray | sparse.spmatrix,
        rhs: NDArray,
        original_shape: tuple,
    ) -> NDArray:
        """Solve J·δx = rhs and reshape to original_shape."""
        if sparse.issparse(J):
            delta_x_flat = spsolve(J, rhs)
        else:
            delta_x_flat = np.linalg.solve(J, rhs)

        return delta_x_flat.reshape(original_shape)

    def _backtracking_line_search(
        self,
        F: Callable[[NDArray], NDArray],
        x: NDArray,
        delta_x: NDArray,
        F_x: NDArray,
        alpha_init: float = 1.0,
        rho: float = 0.5,
        c: float = 1e-4,
    ) -> float:
        """
        Backtracking line search (Armijo rule).

        Finds α such that: ||F(x + α·δx)|| < ||F(x)|| - c·α·||δx||²
        """
        alpha = alpha_init
        norm_F_x = np.linalg.norm(F_x.flatten())
        norm_delta = np.linalg.norm(delta_x.flatten())

        for _ in range(20):  # Max 20 backtracking steps
            x_trial = x + alpha * delta_x
            F_trial = F(x_trial)
            norm_F_trial = np.linalg.norm(F_trial.flatten())

            # Armijo condition
            if norm_F_trial <= norm_F_x - c * alpha * norm_delta**2:
                return alpha

            alpha *= rho

        # If line search fails, return small step
        return alpha


class PolicyIterationSolver(NonlinearSolver):
    """
    Policy iteration (Howard's algorithm) for control problems.

    Solves optimal control problems via:
        1. Policy evaluation: Solve value function V^π for fixed policy π
        2. Policy improvement: Update π' = argmax_a Q(s, a, V^π)
        3. Repeat until π = π'

    Control theory: This is Howard's policy iteration algorithm.
    MFG context: Alternates between solving HJB for fixed control and
                 improving control based on value function.

    Algorithm:
        Given: policy_eval(π) → V, policy_improve(V) → π'
        1. Initialize π_0
        2. Loop:
            a. V_k = policy_eval(π_k)
            b. π_{k+1} = policy_improve(V_k)
            c. If π_{k+1} = π_k, stop

    Example:
        >>> def eval_policy(policy):
        ...     # Solve linear system for value given policy
        ...     return value
        >>> def improve_policy(value):
        ...     # Compute optimal policy from value
        ...     return new_policy
        >>> solver = PolicyIterationSolver(max_iterations=20)
        >>> value, policy, info = solver.solve(eval_policy, improve_policy, policy0)

    Notes:
        - Typically faster than value iteration
        - Each step solves linear (not nonlinear) system
        - Natural for control problems with discrete actions
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ):
        """
        Initialize policy iteration solver.

        Args:
            max_iterations: Maximum policy iteration steps
            tolerance: Convergence tolerance on policy change
        """
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        if tolerance <= 0:
            raise ValueError(f"tolerance must be > 0, got {tolerance}")

        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def solve(
        self,
        policy_eval: Callable[[NDArray], NDArray],
        policy_improve: Callable[[NDArray], NDArray],
        policy0: NDArray,
        **kwargs: Any,
    ) -> tuple[NDArray, NDArray, SolverInfo]:
        """
        Solve optimal control via policy iteration.

        Args:
            policy_eval: Function π → V (evaluate value for policy)
            policy_improve: Function V → π' (improve policy from value)
            policy0: Initial policy guess

        Returns:
            value: Optimal value function
            policy: Optimal policy
            info: Convergence information
        """
        import time

        start_time = time.time()
        policy_current = policy0.copy()
        residual_history = []

        for iteration in range(self.max_iterations):
            # Step 1: Policy evaluation
            value_current = policy_eval(policy_current)

            # Step 2: Policy improvement
            policy_new = policy_improve(value_current)

            # Check convergence: ||π_new - π_current||
            policy_diff = policy_new - policy_current
            residual = np.linalg.norm(policy_diff.flatten())
            residual_history.append(float(residual))

            if residual < self.tolerance:
                solver_time = time.time() - start_time
                return (
                    value_current,
                    policy_new,
                    SolverInfo(
                        converged=True,
                        iterations=iteration + 1,
                        residual=residual,
                        residual_history=residual_history,
                        solver_time=solver_time,
                    ),
                )

            policy_current = policy_new

        # Maximum iterations reached
        solver_time = time.time() - start_time
        value_final = policy_eval(policy_current)

        return (
            value_final,
            policy_current,
            SolverInfo(
                converged=False,
                iterations=self.max_iterations,
                residual=residual_history[-1] if residual_history else float("inf"),
                residual_history=residual_history,
                solver_time=solver_time,
            ),
        )
