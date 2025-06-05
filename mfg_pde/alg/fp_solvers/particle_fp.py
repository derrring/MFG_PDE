import numpy as np
import scipy.interpolate
from scipy.stats import gaussian_kde

from .base_fp import BaseFPSolver  # Assuming BaseFPSolver is in the same directory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mfg_pde.core.mfg_problem import MFGProblem


class ParticleFPSolver(BaseFPSolver):
    def __init__(
        self,
        problem: "MFGProblem",
        num_particles: int = 5000,
        kde_bandwidth: str = "scott",
    ):
        super().__init__(problem)
        self.fp_method_name = "Particle"
        self.num_particles = num_particles
        self.kde_bandwidth = kde_bandwidth
        self.M_particles_trajectory = None  # Optional: to store particle paths

    def _estimate_density_from_particles(
        self, particles_at_time_t: np.ndarray
    ) -> np.ndarray:
        """
        Estimates density m(t, x) on the grid from an array of particle positions
        at a specific time t, using Kernel Density Estimation (KDE).
        Adapted from MFG-FDM-particle2.py.
        """
        Nx = self.problem.Nx
        xSpace = self.problem.xSpace
        xmin = self.problem.xmin
        xmax = self.problem.xmax
        Dx = self.problem.Dx

        if len(particles_at_time_t) == 0:
            # print(f"Warning: KDE input has no particles at a given time.")
            return np.zeros(Nx)

        # Handle cases with all particles at the same spot or too few unique points (KDE fails or gives poor results)
        std_dev = np.std(particles_at_time_t)
        if (
            std_dev < 1e-9 * (xmax - xmin) or len(np.unique(particles_at_time_t)) < 2
        ):  # Check for unique points for KDE
            # print(f"Warning: KDE input problematic (stddev too small or too few unique points). Creating a narrow peak.")
            m_density_estimated = np.zeros(Nx)
            mean_pos = np.mean(particles_at_time_t)
            # Find closest grid point to the common particle position
            closest_idx = np.argmin(np.abs(xSpace - mean_pos))
            if Dx > 1e-14:
                m_density_estimated[closest_idx] = (
                    1.0 / Dx
                )  # Approximate delta function mass
            else:  # Single point case for space
                m_density_estimated[closest_idx] = 1.0
            return m_density_estimated

        try:
            kde = gaussian_kde(particles_at_time_t, bw_method=self.kde_bandwidth)
            m_density_estimated = kde(xSpace)

            # Ensure density is zero outside domain (KDE might spread a bit)
            m_density_estimated[xSpace < xmin] = 0
            m_density_estimated[xSpace > xmax] = 0

            # Normalize the density on the grid
            if Dx > 1e-14:
                current_mass = np.sum(m_density_estimated) * Dx
                if current_mass > 1e-9:
                    m_density_normalized = m_density_estimated / current_mass
                else:
                    # This can happen if all particles are outside [xmin, xmax] after KDE evaluation
                    # print(f"Warning: Estimated mass via KDE is near zero. Particles might be out of bounds.")
                    m_density_normalized = np.zeros(Nx)
            else:  # Single point case (Nx=1)
                m_density_normalized = (
                    m_density_estimated / np.sum(m_density_estimated)
                    if np.sum(m_density_estimated) > 1e-9
                    else np.zeros(Nx)
                )

            return m_density_normalized
        except Exception as e:
            # print(f"Error during KDE: {e}. Particles: {particles_at_time_t[:10]}") # Print some particle data
            return np.zeros(Nx)

    def solve_fp_system(
        self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray
    ) -> np.ndarray:
        """
        Solves the full FP system forward in time using a particle method.
        Args:
            m_initial_condition (np.ndarray): Initial density M(0,x) on the grid.
                                              Used to sample initial particle positions.
            U_solution_for_drift (np.ndarray): (Nt, Nx) array of value function U(t,x) used for drift.
        Returns:
            np.ndarray: M_density_on_grid (Nt, Nx) - estimated density on the grid.
        """
        # print(f"****** Solving FP ({self.fp_method_name}) with {self.num_particles} particles ******")
        Nx = self.problem.Nx
        Nt = self.problem.Nt
        Dx = self.problem.Dx
        Dt = self.problem.Dt
        sigma_sde = self.problem.sigma  # Diffusion for SDE
        coefCT = self.problem.coefCT  # Control cost coefficient
        x_grid = self.problem.xSpace
        xmin = self.problem.xmin
        Lx = self.problem.Lx

        if Nt == 0:
            return np.zeros((0, Nx))

        M_density_on_grid = np.zeros((Nt, Nx))
        current_M_particles_t = np.zeros((Nt, self.num_particles))

        # --- Sample initial particle positions from m_initial_condition ---
        if Dx > 1e-14 and np.sum(m_initial_condition * Dx) > 1e-9:
            m0_probs = m_initial_condition * Dx
            # Ensure probabilities sum to 1 for np.random.choice
            if not np.isclose(np.sum(m0_probs), 1.0):
                m0_probs = m0_probs / np.sum(m0_probs)
            try:
                initial_particle_positions = np.random.choice(
                    x_grid, size=self.num_particles, p=m0_probs, replace=True
                )
            except ValueError as e:  # If m0_probs doesn't sum to 1 or contains negative
                # print(f"Warning: Problem with m0_probs for particle sampling ({e}). Defaulting to uniform sampling.")
                initial_particle_positions = np.random.uniform(
                    xmin, xmin + Lx, self.num_particles
                )

        else:  # Uniform sampling if Dx is too small or initial mass is zero
            # print("Warning: Dx too small or zero initial mass. Uniformly sampling particles.")
            initial_particle_positions = np.random.uniform(
                xmin, xmin + Lx, self.num_particles
            )

        current_M_particles_t[0, :] = initial_particle_positions
        M_density_on_grid[0, :] = self._estimate_density_from_particles(
            current_M_particles_t[0, :]
        )

        if Nt == 1:  # Only initial condition
            self.M_particles_trajectory = current_M_particles_t
            return M_density_on_grid

        # --- Time stepping loop to evolve particles ---
        for n_time_idx in range(
            Nt - 1
        ):  # Loop from t_0 to t_{Nt-2} to compute up to t_{Nt-1}
            U_at_tn = U_solution_for_drift[n_time_idx, :]  # U at current time t_n

            # Estimate gradient dU/dx at time n_time_idx on the grid (periodic)
            if Nx > 1 and Dx > 1e-14:
                dUdx_grid = (np.roll(U_at_tn, -1) - np.roll(U_at_tn, 1)) / (2 * Dx)
            else:  # Single spatial point, gradient is conceptually zero
                dUdx_grid = np.zeros(Nx)

            # Interpolate gradient onto current particle positions X_t = current_M_particles_t[n_time_idx,:]
            if Nx > 1:
                try:
                    interp_func_dUdx = scipy.interpolate.interp1d(
                        x_grid, dUdx_grid, kind="linear", fill_value="extrapolate"
                    )
                    dUdx_at_particles = interp_func_dUdx(
                        current_M_particles_t[n_time_idx, :]
                    )
                except (
                    ValueError
                ) as ve:  # Can happen if particles are far out of bounds and extrapolate fails
                    # print(f"Interpolation error for dUdx at t_idx {n_time_idx}: {ve}. Using zero gradient.")
                    dUdx_at_particles = np.zeros(self.num_particles)
            else:  # Nx == 1
                dUdx_at_particles = np.zeros(self.num_particles)

            # Optimal control/drift alpha = -coefCT * dU/dx (example)
            # This should match the drift term implied by problem.H if possible
            alpha_optimal_at_particles = -coefCT * dUdx_at_particles

            # Evolve particles using Euler-Maruyama: dX = alpha*dt + sigma*dW
            if Dt > 1e-14:
                dW = np.random.normal(0.0, np.sqrt(Dt), self.num_particles)
            else:  # Dt is zero
                dW = np.zeros(self.num_particles)

            current_M_particles_t[n_time_idx + 1, :] = (
                current_M_particles_t[n_time_idx, :]
                + alpha_optimal_at_particles * Dt
                + sigma_sde * dW
            )

            # Apply periodic boundary conditions for particles
            if Lx > 1e-14:
                current_M_particles_t[n_time_idx + 1, :] = (
                    xmin + (current_M_particles_t[n_time_idx + 1, :] - xmin) % Lx
                )
            # Else: if Lx is zero (single point domain), particles don't move spatially due to PBC.

            # Estimate density on grid at this next time step t_{n+1}
            M_density_on_grid[n_time_idx + 1, :] = (
                self._estimate_density_from_particles(
                    current_M_particles_t[n_time_idx + 1, :]
                )
            )

        self.M_particles_trajectory = (
            current_M_particles_t  # Store for potential later access
        )
        return M_density_on_grid
