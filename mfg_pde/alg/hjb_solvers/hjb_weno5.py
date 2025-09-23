"""
WENO5 HJB Solver for Mean Field Games

This module implements a fifth-order WENO (Weighted Essentially Non-Oscillatory)
scheme for solving Hamilton-Jacobi-Bellman equations in Mean Field Games.

Mathematical Framework:
    ∂u/∂t + H(x, ∇u, m(t,x)) - (σ²/2)Δu = 0

The WENO5 scheme provides high-order spatial accuracy while maintaining stability
near discontinuities and steep gradients, making it ideal for benchmarking
against particle-collocation methods in academic publications.

Key Features:
- Fifth-order spatial accuracy in smooth regions
- Non-oscillatory behavior near discontinuities
- TVD-RK3 time integration for temporal accuracy
- Efficient implementation for 1D domains
- Integration with MFG framework architecture

References:
- Jiang & Shu (1996): Efficient Implementation of WENO Schemes
- Osher & Shu (1991): High-order essentially non-oscillatory schemes
- Shu & Osher (1988): Efficient implementation of essentially non-oscillatory shock-capturing schemes
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base_hjb import BaseHJBSolver

if TYPE_CHECKING:
    from mfg_pde.core.mfg_problem import MFGProblem


class HJBWeno5Solver(BaseHJBSolver):
    """
    WENO5 solver for Hamilton-Jacobi-Bellman equations in Mean Field Games.

    This solver implements a fifth-order WENO scheme with TVD-RK3 time integration
    for high-accuracy solutions to HJB equations. Designed for academic benchmarking
    and comparison with particle-collocation methods.

    Mathematical Approach:
    1. WENO5 spatial reconstruction for ∇u terms
    2. Central differences for diffusion term -(σ²/2)Δu
    3. TVD-RK3 for time integration
    4. Hamiltonian splitting for nonlinear terms

    Performance Characteristics:
    - Fifth-order convergence in smooth regions
    - Non-oscillatory near discontinuities
    - Stable CFL conditions for explicit time stepping
    - Efficient stencil operations for 1D problems
    """

    def __init__(
        self,
        problem: MFGProblem,
        cfl_number: float = 0.3,
        diffusion_stability_factor: float = 0.25,
        weno_epsilon: float = 1e-6,
        time_integration: str = "tvd_rk3",
    ):
        """
        Initialize WENO5 HJB solver.

        Args:
            problem: MFG problem instance
            cfl_number: CFL number for advection terms (typically 0.1-0.5)
            diffusion_stability_factor: Stability factor for diffusion (typically 0.25)
            weno_epsilon: WENO smoothness parameter (typically 1e-6)
            time_integration: Time integration scheme ("tvd_rk3", "explicit_euler")
        """
        super().__init__(problem)
        self.hjb_method_name = "WENO5"

        # WENO5 parameters
        self.cfl_number = cfl_number
        self.diffusion_stability_factor = diffusion_stability_factor
        self.weno_epsilon = weno_epsilon
        self.time_integration = time_integration

        # Validate parameters
        if not 0 < cfl_number <= 1.0:
            raise ValueError(f"CFL number must be in (0,1], got {cfl_number}")
        if not 0 < diffusion_stability_factor <= 0.5:
            raise ValueError(f"Diffusion stability factor must be in (0,0.5], got {diffusion_stability_factor}")
        if weno_epsilon <= 0:
            raise ValueError(f"WENO epsilon must be positive, got {weno_epsilon}")

        # WENO5 coefficients for 5-point stencil reconstruction
        self._setup_weno5_coefficients()

    def _setup_weno5_coefficients(self) -> None:
        """Setup WENO5 reconstruction coefficients."""
        # Optimal weights for 5th-order accuracy
        self.d = np.array([1 / 10, 6 / 10, 3 / 10])  # d0, d1, d2

        # Smoothness indicator coefficients
        # For polynomial reconstruction on 3-point sub-stencils
        self.beta_coeffs = {
            0: np.array([1 / 3, -7 / 6, 11 / 6]),  # β₀ coefficients
            1: np.array([-1 / 6, 5 / 6, 1 / 3]),  # β₁ coefficients
            2: np.array([1 / 3, 5 / 6, -1 / 6]),  # β₂ coefficients
        }

        # Reconstruction coefficients for each sub-stencil
        self.c = {
            0: np.array([1 / 3, -7 / 6, 11 / 6]),  # c₀ coefficients
            1: np.array([-1 / 6, 5 / 6, 1 / 3]),  # c₁ coefficients
            2: np.array([1 / 3, 5 / 6, -1 / 6]),  # c₂ coefficients
        }

    def _compute_weno5_weights(self, values: np.ndarray, i: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute WENO5 weights for reconstruction at point i.

        Args:
            values: Array of function values
            i: Grid point index

        Returns:
            (w_plus, w_minus): Weights for upwind and downwind reconstruction
        """
        n = len(values)

        # Handle boundary conditions with periodic extension
        def get_value(idx):
            return values[idx % n]

        # Extract 5-point stencil: [i-2, i-1, i, i+1, i+2]
        v = np.array([get_value(i - 2), get_value(i - 1), get_value(i), get_value(i + 1), get_value(i + 2)])

        # Compute smoothness indicators β₀, β₁, β₂
        beta = np.zeros(3)

        # β₀: stencil [i-2, i-1, i]
        s0 = v[0:3]
        beta[0] = 13 / 12 * (s0[0] - 2 * s0[1] + s0[2]) ** 2 + 1 / 4 * (s0[0] - 4 * s0[1] + 3 * s0[2]) ** 2

        # β₁: stencil [i-1, i, i+1]
        s1 = v[1:4]
        beta[1] = 13 / 12 * (s1[0] - 2 * s1[1] + s1[2]) ** 2 + 1 / 4 * (s1[0] - s1[2]) ** 2

        # β₂: stencil [i, i+1, i+2]
        s2 = v[2:5]
        beta[2] = 13 / 12 * (s2[0] - 2 * s2[1] + s2[2]) ** 2 + 1 / 4 * (3 * s2[0] - 4 * s2[1] + s2[2]) ** 2

        # Compute adaptive weights α = d / (ε + β)²
        alpha_plus = self.d / (self.weno_epsilon + beta) ** 2
        alpha_minus = self.d[::-1] / (self.weno_epsilon + beta[::-1]) ** 2  # Reverse for upwind

        # Normalize weights
        w_plus = alpha_plus / np.sum(alpha_plus)
        w_minus = alpha_minus / np.sum(alpha_minus)

        return w_plus, w_minus

    def _weno5_reconstruct(self, values: np.ndarray, i: int, direction: str = "upwind") -> float:
        """
        WENO5 reconstruction at grid point i.

        Args:
            values: Array of function values
            i: Grid point index
            direction: "upwind" or "central" reconstruction

        Returns:
            Reconstructed value with 5th-order accuracy
        """
        n = len(values)

        # Handle boundary conditions
        def get_value(idx):
            return values[idx % n]

        # 5-point stencil values
        v = np.array([get_value(i - 2), get_value(i - 1), get_value(i), get_value(i + 1), get_value(i + 2)])

        # Compute WENO weights
        w_plus, w_minus = self._compute_weno5_weights(values, i)

        # Reconstruct using weighted combination of 3rd-order polynomials
        if direction == "upwind":
            weights = w_minus
            # Upwind-biased reconstruction
            r0 = 1 / 3 * v[0] - 7 / 6 * v[1] + 11 / 6 * v[2]  # P₀(x_{i+1/2})
            r1 = -1 / 6 * v[1] + 5 / 6 * v[2] + 1 / 3 * v[3]  # P₁(x_{i+1/2})
            r2 = 1 / 3 * v[2] + 5 / 6 * v[3] - 1 / 6 * v[4]  # P₂(x_{i+1/2})
        else:  # direction == "downwind" or "central"
            weights = w_plus
            # Downwind-biased reconstruction
            r0 = 11 / 6 * v[0] - 7 / 6 * v[1] + 1 / 3 * v[2]  # P₀(x_{i-1/2})
            r1 = 1 / 3 * v[1] + 5 / 6 * v[2] - 1 / 6 * v[3]  # P₁(x_{i-1/2})
            r2 = -1 / 6 * v[2] + 5 / 6 * v[3] + 1 / 3 * v[4]  # P₂(x_{i-1/2})

        reconstructed = weights[0] * r0 + weights[1] * r1 + weights[2] * r2
        return reconstructed

    def _compute_weno5_derivatives(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute WENO5 spatial derivatives for Hamiltonian evaluation.

        Args:
            u: Solution array at current time

        Returns:
            (u_x_plus, u_x_minus): Upwind and downwind derivative approximations
        """
        n = len(u)
        dx = self.problem.Dx

        u_x_plus = np.zeros(n)
        u_x_minus = np.zeros(n)

        for i in range(n):
            # WENO5 reconstruction at cell interfaces
            u_right = self._weno5_reconstruct(u, i, "upwind")  # u_{i+1/2}^-
            u_left = self._weno5_reconstruct(u, i, "downwind")  # u_{i-1/2}^+

            # Compute derivatives using reconstructed interface values
            u_x_plus[i] = (u_right - u[i]) / dx  # Forward difference
            u_x_minus[i] = (u[i] - u_left) / dx  # Backward difference

        return u_x_plus, u_x_minus

    def _compute_diffusion_term(self, u: np.ndarray) -> np.ndarray:
        """
        Compute diffusion term -(σ²/2)Δu using central differences.

        Args:
            u: Solution array

        Returns:
            Diffusion contribution
        """
        n = len(u)
        dx = self.problem.Dx
        sigma = self.problem.sigma

        # Central difference for second derivative
        u_xx = np.zeros(n)
        for i in range(n):
            u_im1 = u[(i - 1) % n]
            u_ip1 = u[(i + 1) % n]
            u_xx[i] = (u_ip1 - 2 * u[i] + u_im1) / dx**2

        return -(sigma**2 / 2.0) * u_xx

    def _evaluate_hamiltonian_weno5(self, u: np.ndarray, m: np.ndarray, t_idx: int) -> np.ndarray:
        """
        Evaluate Hamiltonian using WENO5 spatial derivatives.

        Args:
            u: Current solution
            m: Density distribution
            t_idx: Time index

        Returns:
            Hamiltonian evaluation at all grid points
        """
        n = len(u)
        h_values = np.zeros(n)

        # Compute WENO5 derivatives
        u_x_plus, u_x_minus = self._compute_weno5_derivatives(u)

        for i in range(n):
            # Create p_values dict for Hamiltonian evaluation
            p_values = {"forward": u_x_plus[i], "backward": u_x_minus[i]}

            # Evaluate Hamiltonian
            h_values[i] = self.problem.H(x_idx=i, m_at_x=m[i], p_values=p_values, t_idx=t_idx)

        return h_values

    def _tvd_rk3_step(self, u: np.ndarray, m: np.ndarray, dt: float, t_idx: int) -> np.ndarray:
        """
        TVD-RK3 time integration step for HJB equation.

        TVD-RK3 scheme:
        u⁽¹⁾ = uⁿ + Δt L(uⁿ)
        u⁽²⁾ = 3/4 uⁿ + 1/4 u⁽¹⁾ + 1/4 Δt L(u⁽¹⁾)
        uⁿ⁺¹ = 1/3 uⁿ + 2/3 u⁽²⁾ + 2/3 Δt L(u⁽²⁾)

        Args:
            u: Solution at current time
            m: Density at current time
            dt: Time step
            t_idx: Time index

        Returns:
            Solution at next time step
        """

        def spatial_operator(u_state):
            # L(u) = -H(x, ∇u, m) + (σ²/2)Δu
            hamiltonian = self._evaluate_hamiltonian_weno5(u_state, m, t_idx)
            diffusion = self._compute_diffusion_term(u_state)
            return -hamiltonian + diffusion

        # Stage 1: u⁽¹⁾ = uⁿ + Δt L(uⁿ)
        L_u0 = spatial_operator(u)
        u1 = u + dt * L_u0

        # Stage 2: u⁽²⁾ = 3/4 uⁿ + 1/4 u⁽¹⁾ + 1/4 Δt L(u⁽¹⁾)
        L_u1 = spatial_operator(u1)
        u2 = 0.75 * u + 0.25 * u1 + 0.25 * dt * L_u1

        # Stage 3: uⁿ⁺¹ = 1/3 uⁿ + 2/3 u⁽²⁾ + 2/3 Δt L(u⁽²⁾)
        L_u2 = spatial_operator(u2)
        u_new = (1.0 / 3.0) * u + (2.0 / 3.0) * u2 + (2.0 / 3.0) * dt * L_u2

        return u_new

    def _compute_stable_timestep(self) -> float:
        """
        Compute stable time step based on CFL and diffusion stability conditions.

        Returns:
            Maximum stable time step
        """
        dx = self.problem.Dx
        sigma = self.problem.sigma

        # CFL condition for advection (estimated max |H_p|)
        max_velocity = 10.0  # Conservative estimate, could be problem-dependent
        dt_cfl = self.cfl_number * dx / max_velocity

        # Diffusion stability condition
        dt_diffusion = self.diffusion_stability_factor * dx**2 / sigma**2 if sigma > 0 else np.inf

        # Take minimum for stability
        dt_stable = min(dt_cfl, dt_diffusion)

        # Ensure reasonable bounds
        dt_stable = max(dt_stable, 1e-8)  # Minimum time step
        dt_stable = min(dt_stable, self.problem.Dt)  # Don't exceed problem time step

        return dt_stable

    def solve_hjb_system(
        self,
        M_density_evolution: np.ndarray,
        U_final_condition: np.ndarray,
        U_from_prev_picard: np.ndarray,
    ) -> np.ndarray:
        """
        Solve HJB system using WENO5 spatial discretization and TVD-RK3 time integration.

        This method implements explicit time stepping, which differs from the implicit
        Newton-based approach in base_hjb. For academic benchmarking, this provides
        a fundamentally different numerical approach to compare against particle methods.

        Args:
            M_density_evolution: Density evolution from FP equation (Nt, Nx)
            U_final_condition: Terminal condition U(T,x) (Nx,)
            U_from_prev_picard: Previous Picard iteration (Nt, Nx) - not used in explicit method

        Returns:
            Solution U(t,x) for all time steps (Nt, Nx)
        """
        Nt = self.problem.Nt + 1
        Nx = self.problem.Nx + 1

        # Initialize solution array
        U_solution = np.zeros((Nt, Nx))

        # Set terminal condition
        U_solution[Nt - 1, :] = U_final_condition.copy()

        # Compute stable time step for explicit method
        dt_stable = self._compute_stable_timestep()

        # Solve backward in time using explicit WENO5-TVD-RK3
        for n in range(Nt - 2, -1, -1):
            # Current time step
            dt_current = min(dt_stable, self.problem.Dt)

            # Density at current time
            m_current = M_density_evolution[n, :]

            if self.time_integration == "tvd_rk3":
                # TVD-RK3 time integration
                U_solution[n, :] = self._tvd_rk3_step(
                    U_solution[n + 1, :],  # Current solution
                    m_current,  # Density
                    dt_current,  # Time step
                    n,  # Time index
                )
            else:
                # Explicit Euler fallback
                hamiltonian = self._evaluate_hamiltonian_weno5(U_solution[n + 1, :], m_current, n)
                diffusion = self._compute_diffusion_term(U_solution[n + 1, :])

                # Explicit Euler: u^{n} = u^{n+1} + Δt[-H + (σ²/2)Δu]
                U_solution[n, :] = U_solution[n + 1, :] + dt_current * (-hamiltonian + diffusion)

        return U_solution
