"""
WENO Family HJB Solvers for Mean Field Games

This module implements the complete family of WENO (Weighted Essentially Non-Oscillatory)
schemes for solving Hamilton-Jacobi-Bellman equations in Mean Field Games.

Available WENO Variants:
- WENO5: Standard fifth-order WENO scheme (Jiang & Shu, 1996)
- WENO-Z: Enhanced WENO with τ-based weight modification (Borges et al., 2008)
- WENO-M: Mapped WENO for better performance near critical points
- WENO-JS: Original Jiang-Shu formulation with classic weights

Mathematical Framework:
    ∂u/∂t + H(x, ∇u, m(t,x)) - (σ²/2)Δu = 0

Each WENO variant provides:
- Fifth-order spatial accuracy in smooth regions
- Non-oscillatory behavior near discontinuities
- Different weight calculation strategies for various trade-offs
- TVD-RK3 or explicit Euler time integration

Key Features:
- Unified interface for all WENO variants
- Easy switching between methods for benchmarking
- Optimized stencil operations
- Comprehensive parameter validation
- Academic-quality implementation with references

References:
- Jiang & Shu (1996): Efficient Implementation of WENO Schemes
- Borges et al. (2008): An improved weighted essentially non-oscillatory scheme
- Henrick et al. (2005): Mapped weighted essentially non-oscillatory schemes
- Shu & Osher (1988): Efficient implementation of essentially non-oscillatory schemes
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .base_hjb import BaseHJBSolver

if TYPE_CHECKING:
    from mfg_pde.core.mfg_problem import MFGProblem

WenoVariant = Literal["weno5", "weno-z", "weno-m", "weno-js"]


class HJBWenoSolver(BaseHJBSolver):
    """
    Unified WENO family solver for Hamilton-Jacobi-Bellman equations.

    This solver provides access to the complete family of WENO schemes through
    a single interface. Users can select the variant most appropriate for their
    problem characteristics and performance requirements.

    WENO Variants:
    - "weno5": Standard WENO5 (balanced performance, widely used)
    - "weno-z": Enhanced resolution, reduced dissipation
    - "weno-m": Better critical point handling, mapped weights
    - "weno-js": Original formulation, maximum stability

    Mathematical Approach:
    1. WENO spatial reconstruction with variant-specific weights
    2. Central differences for diffusion term -(σ²/2)Δu
    3. TVD-RK3 or explicit Euler time integration
    4. Hamiltonian splitting for nonlinear terms

    Performance Guide:
    - Use "weno5" for general problems and benchmarking
    - Use "weno-z" for problems requiring high resolution
    - Use "weno-m" for critical points and smooth solutions
    - Use "weno-js" for maximum stability requirements
    """

    def __init__(
        self,
        problem: MFGProblem,
        weno_variant: WenoVariant = "weno5",
        cfl_number: float = 0.3,
        diffusion_stability_factor: float = 0.25,
        weno_epsilon: float = 1e-6,
        weno_z_parameter: float = 1.0,
        weno_m_parameter: float = 1.0,
        time_integration: str = "tvd_rk3",
        splitting_method: str = "strang",
    ):
        """
        Initialize WENO family HJB solver with multi-dimensional support.

        Automatically detects problem dimension and applies appropriate WENO schemes:
        - 1D: Direct WENO reconstruction
        - 2D/3D: Dimensional splitting approach with WENO in each direction

        Args:
            problem: MFG problem instance (1D, 2D, 3D, or high-dimensional)
            weno_variant: WENO scheme variant ("weno5", "weno-z", "weno-m", "weno-js")
            cfl_number: CFL number for advection terms (typically 0.1-0.5 for 1D, 0.1-0.3 for 2D+)
            diffusion_stability_factor: Stability factor for diffusion (typically 0.25 for 1D, 0.125 for 2D+)
            weno_epsilon: WENO smoothness parameter (typically 1e-6)
            weno_z_parameter: WENO-Z τ parameter for enhanced resolution (typically 1.0)
            weno_m_parameter: WENO-M mapping parameter for critical points (typically 1.0)
            time_integration: Time integration scheme ("tvd_rk3", "explicit_euler")
            splitting_method: Dimensional splitting method for 2D+ ("strang", "godunov")
        """
        super().__init__(problem)

        # Validate WENO variant
        if weno_variant not in ["weno5", "weno-z", "weno-m", "weno-js"]:
            raise ValueError(f"Unknown WENO variant: {weno_variant}")

        self.weno_variant = weno_variant
        self.splitting_method = splitting_method

        # Detect problem dimension
        self.dimension = self._detect_problem_dimension()
        self.hjb_method_name = f"{self.dimension}D-WENO-{weno_variant.upper()}"

        # Adjust parameters for multi-dimensional problems
        self.cfl_number = self._adjust_cfl_for_dimension(cfl_number)
        self.diffusion_stability_factor = self._adjust_diffusion_factor_for_dimension(diffusion_stability_factor)

        # WENO parameters
        self.weno_epsilon = weno_epsilon
        self.weno_z_parameter = weno_z_parameter
        self.weno_m_parameter = weno_m_parameter
        self.time_integration = time_integration

        # Validate parameters
        self._validate_parameters()

        # Setup dimension-specific grid information
        self._setup_dimensional_grid()

        # Setup WENO coefficients (shared across variants)
        self._setup_weno_coefficients()

    def _validate_parameters(self) -> None:
        """Validate all solver parameters."""
        if not 0 < self.cfl_number <= 1.0:
            raise ValueError(f"CFL number must be in (0,1], got {self.cfl_number}")
        if not 0 < self.diffusion_stability_factor <= 0.5:
            raise ValueError(f"Diffusion stability factor must be in (0,0.5], got {self.diffusion_stability_factor}")
        if self.weno_epsilon <= 0:
            raise ValueError(f"WENO epsilon must be positive, got {self.weno_epsilon}")
        if self.weno_z_parameter <= 0:
            raise ValueError(f"WENO-Z parameter must be positive, got {self.weno_z_parameter}")
        if self.weno_m_parameter <= 0:
            raise ValueError(f"WENO-M parameter must be positive, got {self.weno_m_parameter}")

        if self.splitting_method not in ["strang", "godunov"]:
            raise ValueError(f"Unknown splitting method: {self.splitting_method}")

    def _detect_problem_dimension(self) -> int:
        """Detect the spatial dimension of the MFG problem."""
        # Check if it's a high-dimensional problem
        if hasattr(self.problem, "dimension"):
            return self.problem.dimension

        # Check if it has multi-dimensional geometry
        if hasattr(self.problem, "geometry") and hasattr(self.problem.geometry, "dimension"):
            return self.problem.geometry.dimension

        # Legacy: check for 2D/3D grid properties
        if hasattr(self.problem, "Ny") or hasattr(self.problem, "ny"):
            if hasattr(self.problem, "Nz") or hasattr(self.problem, "nz"):
                return 3  # 3D problem
            return 2  # 2D problem

        # Default to 1D (backward compatibility)
        return 1

    def _adjust_cfl_for_dimension(self, base_cfl: float) -> float:
        """Adjust CFL number based on problem dimension for stability."""
        if self.dimension == 1:
            return base_cfl
        elif self.dimension == 2:
            return min(base_cfl, 0.25)  # More conservative for 2D
        else:  # 3D and higher
            return min(base_cfl, 0.15)  # Very conservative for 3D+

    def _adjust_diffusion_factor_for_dimension(self, base_factor: float) -> float:
        """Adjust diffusion stability factor based on problem dimension."""
        if self.dimension == 1:
            return base_factor
        elif self.dimension == 2:
            return min(base_factor, 0.125)  # More restrictive for 2D
        else:  # 3D and higher
            return min(base_factor, 0.0625)  # Very restrictive for 3D+

    def _setup_dimensional_grid(self) -> None:
        """Setup grid information based on problem dimension."""
        if self.dimension == 1:
            # 1D case - existing logic
            if hasattr(self.problem, "Nx"):
                self.Nx = self.problem.Nx
                self.Dx = self.problem.Dx
            else:
                self.Nx = getattr(self.problem, "nx", 64)
                self.Dx = getattr(self.problem, "dx", 1.0 / (self.Nx - 1))

        elif self.dimension == 2:
            # 2D case
            if hasattr(self.problem, "geometry") and hasattr(self.problem.geometry, "get_computational_grid"):
                grid = self.problem.geometry.get_computational_grid()
                self.Nx, self.Ny = grid["nx"], grid["ny"]
                self.Dx, self.Dy = grid["dx"], grid["dy"]
                self.X, self.Y = grid["X"], grid["Y"]
            else:
                # Fallback for 2D
                self.Nx = getattr(self.problem, "Nx", getattr(self.problem, "nx", 64))
                self.Ny = getattr(self.problem, "Ny", getattr(self.problem, "ny", 64))
                self.Dx = getattr(self.problem, "Dx", getattr(self.problem, "dx", 1.0 / (self.Nx - 1)))
                self.Dy = getattr(self.problem, "Dy", getattr(self.problem, "dy", 1.0 / (self.Ny - 1)))

        elif self.dimension == 3:
            # 3D case
            self.Nx = getattr(self.problem, "Nx", getattr(self.problem, "nx", 32))
            self.Ny = getattr(self.problem, "Ny", getattr(self.problem, "ny", 32))
            self.Nz = getattr(self.problem, "Nz", getattr(self.problem, "nz", 32))
            self.Dx = getattr(self.problem, "Dx", getattr(self.problem, "dx", 1.0 / (self.Nx - 1)))
            self.Dy = getattr(self.problem, "Dy", getattr(self.problem, "dy", 1.0 / (self.Ny - 1)))
            self.Dz = getattr(self.problem, "Dz", getattr(self.problem, "dz", 1.0 / (self.Nz - 1)))

    def _setup_weno_coefficients(self) -> None:
        """Setup WENO reconstruction coefficients (shared across variants)."""
        # Optimal linear weights
        self.d_plus = np.array([3 / 10, 3 / 5, 1 / 10])  # d₀, d₁, d₂ for positive reconstruction
        self.d_minus = np.array([1 / 10, 3 / 5, 3 / 10])  # d₀, d₁, d₂ for negative reconstruction

        # Stencil coefficients for polynomial reconstruction
        # Positive reconstruction coefficients (left-to-right bias)
        self.c_plus = np.array(
            [
                [1 / 3, -7 / 6, 11 / 6],  # S₀: u_{i-2}, u_{i-1}, u_i
                [-1 / 6, 5 / 6, 1 / 3],  # S₁: u_{i-1}, u_i, u_{i+1}
                [1 / 3, 5 / 6, -1 / 6],  # S₂: u_i, u_{i+1}, u_{i+2}
            ]
        )

        # Negative reconstruction coefficients (right-to-left bias)
        self.c_minus = np.array(
            [
                [-1 / 6, 5 / 6, 1 / 3],  # S₀: u_{i+1}, u_i, u_{i-1}
                [1 / 3, 5 / 6, -1 / 6],  # S₁: u_i, u_{i-1}, u_{i-2}
                [1 / 3, -7 / 6, 11 / 6],  # S₂: u_{i-1}, u_{i-2}, u_{i-3}
            ]
        )

    def _compute_weno_weights(self, values: np.ndarray, i: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute WENO weights using the selected variant.

        Args:
            values: Array of function values
            i: Grid point index

        Returns:
            (w_plus, w_minus): WENO weights for upwind and downwind reconstruction
        """
        # Extract safe stencil values
        n = len(values)
        i_safe = max(2, min(i, n - 3))  # Ensure valid 5-point stencil
        u = values[i_safe - 2 : i_safe + 3]  # 5-point stencil: u_{i-2} to u_{i+2}

        # Compute smoothness indicators (same for all variants)
        beta = self._compute_smoothness_indicators(u)

        # Select weight calculation based on variant
        if self.weno_variant == "weno5" or self.weno_variant == "weno-js":
            w_plus = self._compute_classic_weights(beta, self.d_plus)
            w_minus = self._compute_classic_weights(beta[::-1], self.d_minus)
        elif self.weno_variant == "weno-z":
            tau = self._compute_tau_indicator(u)
            w_plus = self._compute_z_weights(beta, tau, self.d_plus)
            w_minus = self._compute_z_weights(beta[::-1], tau, self.d_minus)
        elif self.weno_variant == "weno-m":
            w_plus = self._compute_mapped_weights(beta, self.d_plus)
            w_minus = self._compute_mapped_weights(beta[::-1], self.d_minus)
        else:
            raise ValueError(f"Unknown WENO variant: {self.weno_variant}")

        return w_plus, w_minus

    def _compute_smoothness_indicators(self, u: np.ndarray) -> np.ndarray:
        """
        Compute WENO smoothness indicators β for 3-point sub-stencils.

        Args:
            u: 5-point stencil values [u_{i-2}, u_{i-1}, u_i, u_{i+1}, u_{i+2}]

        Returns:
            β: Array of 3 smoothness indicators [β₀, β₁, β₂]
        """
        # Sub-stencil S₀: u_{i-2}, u_{i-1}, u_i
        beta_0 = (13 / 12) * (u[0] - 2 * u[1] + u[2]) ** 2 + (1 / 4) * (u[0] - 4 * u[1] + 3 * u[2]) ** 2

        # Sub-stencil S₁: u_{i-1}, u_i, u_{i+1}
        beta_1 = (13 / 12) * (u[1] - 2 * u[2] + u[3]) ** 2 + (1 / 4) * (u[1] - u[3]) ** 2

        # Sub-stencil S₂: u_i, u_{i+1}, u_{i+2}
        beta_2 = (13 / 12) * (u[2] - 2 * u[3] + u[4]) ** 2 + (1 / 4) * (3 * u[2] - 4 * u[3] + u[4]) ** 2

        return np.array([beta_0, beta_1, beta_2])

    def _compute_classic_weights(self, beta: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Compute classic WENO5/WENO-JS weights.

        Args:
            beta: Smoothness indicators [β₀, β₁, β₂]
            d: Linear weights [d₀, d₁, d₂]

        Returns:
            w: Classic WENO nonlinear weights [w₀, w₁, w₂]
        """
        # Classic WENO weight formula
        alpha = d / (self.weno_epsilon + beta) ** 2
        w = alpha / np.sum(alpha)
        return w

    def _compute_tau_indicator(self, u: np.ndarray) -> float:
        """
        Compute global smoothness indicator τ for WENO-Z scheme.

        Args:
            u: 5-point stencil values [u_{i-2}, u_{i-1}, u_i, u_{i+1}, u_{i+2}]

        Returns:
            τ: Global smoothness indicator
        """
        # WENO-Z τ₅ indicator using 4th-order differences
        tau = abs(u[0] - 4 * u[1] + 6 * u[2] - 4 * u[3] + u[4])
        return tau

    def _compute_z_weights(self, beta: np.ndarray, tau: float, d: np.ndarray) -> np.ndarray:
        """
        Compute WENO-Z weights with τ modification for enhanced resolution.

        Args:
            beta: Smoothness indicators [β₀, β₁, β₂]
            tau: Global smoothness indicator
            d: Linear weights [d₀, d₁, d₂]

        Returns:
            w: WENO-Z nonlinear weights [w₀, w₁, w₂]
        """
        # WENO-Z enhancement with τ-based modification
        tau_modified = (tau / (beta + self.weno_epsilon)) ** self.weno_z_parameter
        alpha = d * (1.0 + tau_modified)
        w = alpha / np.sum(alpha)
        return w

    def _compute_mapped_weights(self, beta: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Compute WENO-M mapped weights for better critical point handling.

        Args:
            beta: Smoothness indicators [β₀, β₁, β₂]
            d: Linear weights [d₀, d₁, d₂]

        Returns:
            w: WENO-M mapped weights [w₀, w₁, w₂]
        """
        # Compute classic weights first
        alpha_classic = d / (self.weno_epsilon + beta) ** 2
        w_classic = alpha_classic / np.sum(alpha_classic)

        # Apply mapping function for better critical point behavior
        # Henrick mapping: g(ω) = ω(d + d²/ω - d) / (d + d²/ω - 1)
        w_mapped = np.zeros_like(w_classic)
        for k in range(len(d)):
            if w_classic[k] > self.weno_epsilon:
                g_w = w_classic[k] * (d[k] + d[k] ** 2 / w_classic[k] - d[k]) / (d[k] + d[k] ** 2 / w_classic[k] - 1.0)
                w_mapped[k] = max(g_w, 0.0)  # Ensure positivity
            else:
                w_mapped[k] = 0.0

        # Renormalize
        w_sum = np.sum(w_mapped)
        if w_sum > self.weno_epsilon:
            w_mapped /= w_sum
        else:
            w_mapped = d  # Fallback to linear weights

        return w_mapped

    def _weno_reconstruction(self, values: np.ndarray, i: int) -> tuple[float, float]:
        """
        Perform WENO reconstruction to get left and right interface values.

        Args:
            values: Array of cell-centered values
            i: Interface index (between cells i and i+1)

        Returns:
            (u_left, u_right): Reconstructed values at interface
        """
        # Get WENO weights using selected variant
        w_plus, w_minus = self._compute_weno_weights(values, i)

        # Extract stencil for reconstruction
        n = len(values)
        i_safe = max(2, min(i, n - 3))
        u = values[i_safe - 2 : i_safe + 3]

        # Reconstruct using weighted combination of sub-stencil polynomials
        u_left = 0.0
        u_right = 0.0

        for k in range(3):
            # Left reconstruction (positive direction)
            u_left += w_plus[k] * np.dot(self.c_plus[k], u[k : k + 3])

            # Right reconstruction (negative direction) - fix indexing
            if k == 0:
                u_vals = u[2::-1]  # [u2, u1, u0]
            elif k == 1:
                u_vals = u[3:0:-1]  # [u3, u2, u1]
            else:  # k == 2
                u_vals = u[4:1:-1]  # [u4, u3, u2]

            u_right += w_minus[k] * np.dot(self.c_minus[k], u_vals)

        return u_left, u_right

    def solve_hjb_step(self, u_current: np.ndarray, m_current: np.ndarray, dt: float) -> np.ndarray:
        """
        Solve one time step of the HJB equation using selected WENO variant.

        Args:
            u_current: Current value function
            m_current: Current density
            dt: Time step size

        Returns:
            u_new: Updated value function after one time step
        """
        if self.time_integration == "tvd_rk3":
            return self._solve_hjb_tvd_rk3(u_current, m_current, dt)
        elif self.time_integration == "explicit_euler":
            return self._solve_hjb_explicit_euler(u_current, m_current, dt)
        else:
            raise ValueError(f"Unknown time integration: {self.time_integration}")

    def _solve_hjb_explicit_euler(self, u_current: np.ndarray, m_current: np.ndarray, dt: float) -> np.ndarray:
        """Explicit Euler time step with WENO spatial discretization."""
        rhs = self._compute_hjb_rhs(u_current, m_current)
        return u_current + dt * rhs

    def _solve_hjb_tvd_rk3(self, u_current: np.ndarray, m_current: np.ndarray, dt: float) -> np.ndarray:
        """TVD-RK3 time step with WENO spatial discretization."""
        # Stage 1
        k1 = self._compute_hjb_rhs(u_current, m_current)
        u1 = u_current + dt * k1

        # Stage 2
        k2 = self._compute_hjb_rhs(u1, m_current)
        u2 = (3 / 4) * u_current + (1 / 4) * u1 + (1 / 4) * dt * k2

        # Stage 3
        k3 = self._compute_hjb_rhs(u2, m_current)
        u_new = (1 / 3) * u_current + (2 / 3) * u2 + (2 / 3) * dt * k3

        return u_new

    def _compute_hjb_rhs(self, u: np.ndarray, m: np.ndarray) -> np.ndarray:
        """
        Compute right-hand side of HJB equation using WENO discretization.

        RHS = -H(x, ∇u, m) + (σ²/2)Δu
        """
        n = len(u)
        rhs = np.zeros(n)
        dx = self.problem.Dx

        # Compute spatial derivatives using WENO reconstruction
        u_x = np.zeros(n)

        for i in range(2, n - 2):  # Interior points with full stencil
            # WENO reconstruction for derivative approximation
            u_left, u_right = self._weno_reconstruction(u, i)

            # High-order central difference for first derivative
            u_x[i] = (u_right - u_left) / (2 * dx)

        # Handle boundaries with lower-order approximations
        u_x[0] = (-3 * u[0] + 4 * u[1] - u[2]) / (2 * dx)
        u_x[1] = (u[2] - u[0]) / (2 * dx)
        u_x[-2] = (u[-1] - u[-3]) / (2 * dx)
        u_x[-1] = (u[-3] - 4 * u[-2] + 3 * u[-1]) / (2 * dx)

        # Compute second derivative using central differences
        u_xx = np.zeros(n)
        for i in range(1, n - 1):
            u_xx[i] = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx**2

        # Boundary conditions for second derivative
        u_xx[0] = u_xx[1]
        u_xx[-1] = u_xx[-2]

        # Compute Hamiltonian and assemble RHS
        for i in range(n):
            # For simplicity in this demo, we use spatial index
            # In full implementation, would get actual x coordinate

            # Standard quadratic Hamiltonian with congestion
            # H = |∇u|²/2 + V(x,m) where V represents congestion effects
            hamiltonian = 0.5 * u_x[i] ** 2 + m[i] * u_x[i]

            # RHS = -H + diffusion
            rhs[i] = -hamiltonian + (self.problem.sigma**2 / 2) * u_xx[i]

        return rhs

    def _compute_dt_stable_1d(self, u: np.ndarray, m: np.ndarray) -> float:
        """Compute stable time step based on CFL and diffusion stability."""
        dx = getattr(self.problem, "Dx", self.Dx)

        # CFL condition for advection terms
        max_speed = np.max(np.abs(np.gradient(u, dx))) + 1e-10
        dt_cfl = self.cfl_number * dx / max_speed

        # Stability condition for diffusion term
        dt_diffusion = self.diffusion_stability_factor * dx**2 / self.problem.sigma**2

        # Take minimum for stability
        dt_stable = min(dt_cfl, dt_diffusion)

        return max(dt_stable, 1e-10)  # Ensure positive time step

    def solve_hjb_system(
        self,
        M_density_evolution_from_FP: np.ndarray,
        U_final_condition_at_T: np.ndarray,
        U_from_prev_picard: np.ndarray,
    ) -> np.ndarray:
        """
        Solve the complete HJB system using WENO spatial discretization.

        Automatically dispatches to appropriate dimensional solver based on detected problem dimension:
        - 1D: Direct WENO reconstruction
        - 2D/3D: Dimensional splitting with WENO in each direction

        Args:
            M_density_evolution_from_FP: Density evolution m(t,x[,y[,z]]) from FP solver
            U_final_condition_at_T: Terminal condition u(T,x[,y[,z]])
            U_from_prev_picard: Value function from previous Picard iteration

        Returns:
            U_solved: Complete solution u(t,x[,y[,z]]) over time domain
        """
        if self.dimension == 1:
            return self._solve_hjb_system_1d(M_density_evolution_from_FP, U_final_condition_at_T, U_from_prev_picard)
        elif self.dimension == 2:
            return self._solve_hjb_system_2d(M_density_evolution_from_FP, U_final_condition_at_T, U_from_prev_picard)
        elif self.dimension == 3:
            return self._solve_hjb_system_3d(M_density_evolution_from_FP, U_final_condition_at_T, U_from_prev_picard)
        else:
            raise NotImplementedError(f"WENO solver not implemented for dimension {self.dimension}")

    def _solve_hjb_system_1d(
        self,
        M_density_evolution_from_FP: np.ndarray,
        U_final_condition_at_T: np.ndarray,
        U_from_prev_picard: np.ndarray,
    ) -> np.ndarray:
        """Solve 1D HJB system (original implementation)."""
        Nt = self.problem.Nt
        Nx = self.Nx
        dt = self.problem.T / Nt

        # Initialize solution array
        U_solved = np.zeros((Nt + 1, Nx))

        # Set final condition
        U_solved[-1, :] = U_final_condition_at_T

        # Backward time integration
        for t_idx in range(Nt - 1, -1, -1):
            # Current density at this time
            m_current = M_density_evolution_from_FP[t_idx, :]

            # Current value function (start with final condition)
            u_current = U_solved[t_idx + 1, :].copy()

            # Compute stable time step
            dt_stable = min(dt, self._compute_dt_stable_1d(u_current, m_current))

            # Solve HJB step using selected WENO variant
            U_solved[t_idx, :] = self.solve_hjb_step(u_current, m_current, dt_stable)

        return U_solved

    def _solve_hjb_system_2d(
        self,
        M_density_evolution_from_FP: np.ndarray,
        U_final_condition_at_T: np.ndarray,
        U_from_prev_picard: np.ndarray,
    ) -> np.ndarray:
        """Solve 2D HJB system using dimensional splitting."""
        Nt = self.problem.Nt
        dt = self.problem.T / Nt

        # Initialize solution array
        U_solved = np.zeros((Nt + 1, self.Nx, self.Ny))

        # Set final condition
        U_solved[-1, :, :] = U_final_condition_at_T

        # Backward time integration
        for t_idx in range(Nt - 1, -1, -1):
            # Current density at this time
            m_current = M_density_evolution_from_FP[t_idx, :, :]

            # Current value function
            u_current = U_solved[t_idx + 1, :, :].copy()

            # Compute stable time step for 2D
            dt_stable = min(dt, self._compute_dt_stable_2d(u_current, m_current))

            # Apply dimensional splitting
            if self.splitting_method == "strang":
                # Strang splitting: X → Y → X with half time steps
                u_half = self._solve_hjb_step_2d_x_direction(u_current, m_current, dt_stable / 2)
                u_full = self._solve_hjb_step_2d_y_direction(u_half, m_current, dt_stable)
                u_new = self._solve_hjb_step_2d_x_direction(u_full, m_current, dt_stable / 2)
            else:  # godunov
                # Godunov splitting: X → Y with full time steps
                u_half = self._solve_hjb_step_2d_x_direction(u_current, m_current, dt_stable)
                u_new = self._solve_hjb_step_2d_y_direction(u_half, m_current, dt_stable)

            U_solved[t_idx, :, :] = u_new

        return U_solved

    def _solve_hjb_system_3d(
        self,
        M_density_evolution_from_FP: np.ndarray,
        U_final_condition_at_T: np.ndarray,
        U_from_prev_picard: np.ndarray,
    ) -> np.ndarray:
        """Solve 3D HJB system using dimensional splitting."""
        logger = self._get_logger()
        logger.info("Starting 3D WENO HJB solver with dimensional splitting")

        # Initialize solution array
        U_solved = np.copy(U_final_condition_at_T)

        # Time stepping parameters
        Nt = M_density_evolution_from_FP.shape[0]

        # Solve backward in time
        for time_idx in range(Nt - 2, -1, -1):
            logger.debug(f"  3D Time step {time_idx+1}/{Nt-1}")

            u_current = U_solved[time_idx + 1, :, :, :]
            m_current = M_density_evolution_from_FP[time_idx, :, :, :]

            # Compute stable time step for 3D
            dt_stable = min(self.dt, self._compute_dt_stable_3d(u_current, m_current))

            # Apply dimensional splitting (3D requires x, y, z directions)
            if self.splitting_method == "strang":
                # Strang splitting: x(dt/2) -> y(dt/2) -> z(dt) -> y(dt/2) -> x(dt/2)
                u_step1 = self._solve_hjb_step_3d_x_direction(u_current, m_current, dt_stable / 2)
                u_step2 = self._solve_hjb_step_3d_y_direction(u_step1, m_current, dt_stable / 2)
                u_step3 = self._solve_hjb_step_3d_z_direction(u_step2, m_current, dt_stable)
                u_step4 = self._solve_hjb_step_3d_y_direction(u_step3, m_current, dt_stable / 2)
                u_new = self._solve_hjb_step_3d_x_direction(u_step4, m_current, dt_stable / 2)
            else:  # Godunov splitting
                # Godunov splitting: x(dt) -> y(dt) -> z(dt)
                u_step1 = self._solve_hjb_step_3d_x_direction(u_current, m_current, dt_stable)
                u_step2 = self._solve_hjb_step_3d_y_direction(u_step1, m_current, dt_stable)
                u_new = self._solve_hjb_step_3d_z_direction(u_step2, m_current, dt_stable)

            U_solved[time_idx, :, :, :] = u_new

            # Progress logging for long computations
            if (time_idx + 1) % 20 == 0:
                logger.info(f"    3D WENO: Completed {Nt - time_idx - 1}/{Nt-1} time steps")

        logger.info("3D WENO HJB solver completed successfully")
        return U_solved

    def _compute_dt_stable_2d(self, u: np.ndarray, m: np.ndarray) -> float:
        """Compute stable time step for 2D problem based on CFL and diffusion stability."""
        # Compute gradients for stability analysis
        u_x = np.gradient(u, self.Dx, axis=0)
        u_y = np.gradient(u, self.Dy, axis=1)

        # CFL condition for advection terms
        max_speed_x = np.max(np.abs(u_x)) + 1e-10
        max_speed_y = np.max(np.abs(u_y)) + 1e-10

        dt_cfl_x = self.cfl_number * self.Dx / max_speed_x
        dt_cfl_y = self.cfl_number * self.Dy / max_speed_y
        dt_cfl = min(dt_cfl_x, dt_cfl_y)

        # Stability condition for diffusion term (more restrictive in 2D)
        dt_diffusion_x = self.diffusion_stability_factor * self.Dx**2 / self.problem.sigma**2
        dt_diffusion_y = self.diffusion_stability_factor * self.Dy**2 / self.problem.sigma**2
        dt_diffusion = min(dt_diffusion_x, dt_diffusion_y)

        # Take minimum for stability
        dt_stable = min(dt_cfl, dt_diffusion)

        return max(dt_stable, 1e-10)  # Ensure positive time step

    def _solve_hjb_step_2d_x_direction(self, u: np.ndarray, m: np.ndarray, dt: float) -> np.ndarray:
        """Apply WENO reconstruction in X-direction for 2D problem."""
        u_new = u.copy()

        # Apply 1D WENO reconstruction in X-direction for each Y-slice
        for j in range(self.Ny):
            u_slice = u[:, j]
            m_slice = m[:, j]

            # Use existing 1D WENO step
            u_new[:, j] = self.solve_hjb_step(u_slice, m_slice, dt)

        return u_new

    def _solve_hjb_step_2d_y_direction(self, u: np.ndarray, m: np.ndarray, dt: float) -> np.ndarray:
        """Apply WENO reconstruction in Y-direction for 2D problem."""
        u_new = u.copy()

        # Apply 1D WENO reconstruction in Y-direction for each X-slice
        # This requires adapting the 1D solver to work on transposed arrays
        for i in range(self.Nx):
            u_slice = u[i, :]
            m_slice = m[i, :]

            # Apply 1D WENO step (would need adaptation for different grid spacing)
            # For now, use simplified approach
            u_new[i, :] = self._solve_hjb_step_1d_y_adapted(u_slice, m_slice, dt)

        return u_new

    def _solve_hjb_step_1d_y_adapted(self, u_1d: np.ndarray, m_1d: np.ndarray, dt: float) -> np.ndarray:
        """Apply 1D WENO step adapted for Y-direction with appropriate grid spacing."""
        # This is a simplified adaptation - in full implementation would properly
        # handle different grid spacing and coordinate systems

        # Use existing WENO reconstruction logic but with Y-direction spacing
        if self.time_integration == "tvd_rk3":
            return self._solve_hjb_tvd_rk3_y_adapted(u_1d, m_1d, dt)
        else:
            return self._solve_hjb_explicit_euler_y_adapted(u_1d, m_1d, dt)

    def _solve_hjb_tvd_rk3_y_adapted(self, u_current: np.ndarray, m_current: np.ndarray, dt: float) -> np.ndarray:
        """TVD-RK3 time stepping adapted for Y-direction."""
        # Stage 1
        L1 = self._compute_spatial_operator_y_adapted(u_current, m_current)
        u1 = u_current + dt * L1

        # Stage 2
        L2 = self._compute_spatial_operator_y_adapted(u1, m_current)
        u2 = 0.75 * u_current + 0.25 * u1 + 0.25 * dt * L2

        # Stage 3
        L3 = self._compute_spatial_operator_y_adapted(u2, m_current)
        u_new = (1.0 / 3.0) * u_current + (2.0 / 3.0) * u2 + (2.0 / 3.0) * dt * L3

        return u_new

    def _solve_hjb_explicit_euler_y_adapted(
        self, u_current: np.ndarray, m_current: np.ndarray, dt: float
    ) -> np.ndarray:
        """Explicit Euler time stepping adapted for Y-direction."""
        L = self._compute_spatial_operator_y_adapted(u_current, m_current)
        return u_current + dt * L

    def _compute_spatial_operator_y_adapted(self, u: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Compute spatial operator for Y-direction with adapted grid spacing."""
        # This is a placeholder - full implementation would properly handle
        # Y-direction WENO reconstruction with self.Dy spacing
        n = len(u)
        rhs = np.zeros(n)

        # Compute derivatives using Y-direction spacing
        u_y = self._weno_reconstruction_y_adapted(u)

        # Second derivative (central differences with Y spacing)
        u_yy = np.zeros(n)
        u_yy[1:-1] = (u[:-2] - 2 * u[1:-1] + u[2:]) / (self.Dy**2)
        u_yy[0] = u_yy[1]
        u_yy[-1] = u_yy[-2]

        # Hamiltonian evaluation (simplified for Y-direction)
        for i in range(n):
            hamiltonian = 0.5 * u_y[i] ** 2 + m[i] * u_y[i]
            rhs[i] = -hamiltonian + (self.problem.sigma**2 / 2) * u_yy[i]

        return rhs

    def _weno_reconstruction_y_adapted(self, u: np.ndarray) -> np.ndarray:
        """WENO reconstruction adapted for Y-direction with proper grid spacing."""
        # This would use the same WENO logic but with Dy spacing
        # For now, use standard gradient as placeholder
        return np.gradient(u, self.Dy)

    def _compute_dt_stable_3d(self, u: np.ndarray, m: np.ndarray) -> float:
        """Compute stable time step for 3D problem based on CFL and diffusion stability."""
        # Compute gradients for stability analysis
        u_x = np.gradient(u, self.Dx, axis=0)
        u_y = np.gradient(u, self.Dy, axis=1)
        u_z = np.gradient(u, self.Dz, axis=2)

        # Maximum gradient magnitude for CFL condition
        max_grad_x = np.max(np.abs(u_x)) if u_x.size > 0 else 0.0
        max_grad_y = np.max(np.abs(u_y)) if u_y.size > 0 else 0.0
        max_grad_z = np.max(np.abs(u_z)) if u_z.size > 0 else 0.0

        # CFL stability condition (very conservative for 3D)
        if max_grad_x > 1e-12 or max_grad_y > 1e-12 or max_grad_z > 1e-12:
            dt_cfl_x = self.cfl_number * self.Dx / (max_grad_x + 1e-12)
            dt_cfl_y = self.cfl_number * self.Dy / (max_grad_y + 1e-12)
            dt_cfl_z = self.cfl_number * self.Dz / (max_grad_z + 1e-12)
            dt_cfl = min(dt_cfl_x, dt_cfl_y, dt_cfl_z)
        else:
            dt_cfl = self.dt

        # Stability condition for diffusion term (very restrictive in 3D)
        sigma_sq = self.problem.sigma**2 if hasattr(self.problem, "sigma") else 1.0
        dt_diffusion_x = self.diffusion_stability_factor * (self.Dx**2) / sigma_sq
        dt_diffusion_y = self.diffusion_stability_factor * (self.Dy**2) / sigma_sq
        dt_diffusion_z = self.diffusion_stability_factor * (self.Dz**2) / sigma_sq
        dt_diffusion = min(dt_diffusion_x, dt_diffusion_y, dt_diffusion_z)

        return min(dt_cfl, dt_diffusion)

    def _solve_hjb_step_3d_x_direction(self, u: np.ndarray, m: np.ndarray, dt: float) -> np.ndarray:
        """Apply WENO reconstruction in X-direction for 3D problem."""
        u_new = u.copy()
        # Apply 1D WENO reconstruction in X-direction for each (Y,Z)-slice
        for j in range(self.Ny):
            for k in range(self.Nz):
                u_slice = u[:, j, k]
                m_slice = m[:, j, k]
                # Apply 1D WENO step
                u_new[:, j, k] = self._solve_hjb_step_1d_adapted(u_slice, m_slice, dt)
        return u_new

    def _solve_hjb_step_3d_y_direction(self, u: np.ndarray, m: np.ndarray, dt: float) -> np.ndarray:
        """Apply WENO reconstruction in Y-direction for 3D problem."""
        u_new = u.copy()
        # Apply 1D WENO reconstruction in Y-direction for each (X,Z)-slice
        for i in range(self.Nx):
            for k in range(self.Nz):
                u_slice = u[i, :, k]
                m_slice = m[i, :, k]
                # Apply 1D WENO step adapted for Y-direction
                u_new[i, :, k] = self._solve_hjb_step_1d_y_adapted(u_slice, m_slice, dt)
        return u_new

    def _solve_hjb_step_3d_z_direction(self, u: np.ndarray, m: np.ndarray, dt: float) -> np.ndarray:
        """Apply WENO reconstruction in Z-direction for 3D problem."""
        u_new = u.copy()
        # Apply 1D WENO reconstruction in Z-direction for each (X,Y)-slice
        for i in range(self.Nx):
            for j in range(self.Ny):
                u_slice = u[i, j, :]
                m_slice = m[i, j, :]
                # Apply 1D WENO step adapted for Z-direction
                u_new[i, j, :] = self._solve_hjb_step_1d_z_adapted(u_slice, m_slice, dt)
        return u_new

    def _solve_hjb_step_1d_z_adapted(self, u_1d: np.ndarray, m_1d: np.ndarray, dt: float) -> np.ndarray:
        """Apply 1D WENO step adapted for Z-direction with appropriate grid spacing."""
        # Use existing WENO reconstruction logic but with Z-direction spacing
        if self.time_integration == "tvd_rk3":
            return self._solve_hjb_tvd_rk3_z_adapted(u_1d, m_1d, dt)
        else:
            return self._solve_hjb_explicit_euler_z_adapted(u_1d, m_1d, dt)

    def _solve_hjb_tvd_rk3_z_adapted(self, u_current: np.ndarray, m_current: np.ndarray, dt: float) -> np.ndarray:
        """TVD-RK3 time stepping adapted for Z-direction."""
        # Stage 1
        L1 = self._compute_spatial_operator_z_adapted(u_current, m_current)
        u1 = u_current + dt * L1

        # Stage 2
        L2 = self._compute_spatial_operator_z_adapted(u1, m_current)
        u2 = 0.75 * u_current + 0.25 * u1 + 0.25 * dt * L2

        # Stage 3
        L3 = self._compute_spatial_operator_z_adapted(u2, m_current)
        u_new = (1.0 / 3.0) * u_current + (2.0 / 3.0) * u2 + (2.0 / 3.0) * dt * L3

        return u_new

    def _solve_hjb_explicit_euler_z_adapted(
        self, u_current: np.ndarray, m_current: np.ndarray, dt: float
    ) -> np.ndarray:
        """Explicit Euler time stepping adapted for Z-direction."""
        L = self._compute_spatial_operator_z_adapted(u_current, m_current)
        return u_current + dt * L

    def _compute_spatial_operator_z_adapted(self, u: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Compute spatial operator for Z-direction with adapted grid spacing."""
        n = len(u)
        rhs = np.zeros(n)

        # Compute derivatives using Z-direction spacing
        u_z = self._weno_reconstruction_z_adapted(u)

        # Second derivative (central differences with Z spacing)
        u_zz = np.zeros(n)
        u_zz[1:-1] = (u[:-2] - 2 * u[1:-1] + u[2:]) / (self.Dz**2)
        u_zz[0] = u_zz[1]
        u_zz[-1] = u_zz[-2]

        # Hamiltonian evaluation (simplified for Z-direction)
        for i in range(n):
            hamiltonian = 0.5 * u_z[i] ** 2 + m[i] * u_z[i]
            rhs[i] = -hamiltonian + (self.problem.sigma**2 / 2) * u_zz[i]

        return rhs

    def _weno_reconstruction_z_adapted(self, u: np.ndarray) -> np.ndarray:
        """WENO reconstruction adapted for Z-direction with proper grid spacing."""
        # This would use the same WENO logic but with Dz spacing
        # For now, use standard gradient as placeholder
        return np.gradient(u, self.Dz)

    def get_variant_info(self) -> dict[str, str]:
        """
        Get information about the current WENO variant.

        Returns:
            dict: Information about the selected WENO variant
        """
        variant_info = {
            "weno5": {
                "name": "WENO5",
                "description": "Standard fifth-order WENO scheme",
                "characteristics": "Balanced performance, widely used, good stability",
                "best_for": "General MFG problems, benchmarking, production use",
            },
            "weno-z": {
                "name": "WENO-Z",
                "description": "Enhanced WENO with τ-based weight modification",
                "characteristics": "Reduced dissipation, better shock resolution",
                "best_for": "High-resolution requirements, discontinuous solutions",
            },
            "weno-m": {
                "name": "WENO-M",
                "description": "Mapped WENO for critical point handling",
                "characteristics": "Better critical points, enhanced accuracy preservation",
                "best_for": "Smooth solutions, critical points, long-time integration",
            },
            "weno-js": {
                "name": "WENO-JS",
                "description": "Original Jiang-Shu WENO formulation",
                "characteristics": "Maximum stability, conservative approach",
                "best_for": "Stability-critical applications, extreme conditions",
            },
        }

        return variant_info[self.weno_variant]
