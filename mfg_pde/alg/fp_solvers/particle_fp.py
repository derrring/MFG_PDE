import numpy as np
import scipy.interpolate
from scipy.stats import gaussian_kde

from .base_fp import BaseFPSolver  # Assuming BaseFPSolver is in the same directory
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mfg_pde.core.mfg_problem import MFGProblem


class ParticleFPSolver(BaseFPSolver):
    def __init__(
        self,
        problem: "MFGProblem",
        num_particles: int = 5000,
        kde_bandwidth: Any = "scott",
        normalize_kde_output: bool = True,
    ):
        super().__init__(problem)
        self.fp_method_name = "Particle"
        self.num_particles = num_particles
        self.kde_bandwidth = kde_bandwidth
        self.normalize_kde_output = normalize_kde_output  # New flag
        self.M_particles_trajectory = None

    def _estimate_density_from_particles(
        self, particles_at_time_t: np.ndarray
    ) -> np.ndarray:
        Nx = self.problem.Nx + 1
        xSpace = self.problem.xSpace
        xmin = self.problem.xmin
        xmax = self.problem.xmax
        Dx = self.problem.Dx

        if self.num_particles == 0 or len(particles_at_time_t) == 0:
            return np.zeros(Nx)

        unique_particles = np.unique(particles_at_time_t)
        if len(unique_particles) < 2 or np.std(particles_at_time_t) < 1e-9 * (
            xmax - xmin
        ):
            m_density_estimated = np.zeros(Nx)
            if len(particles_at_time_t) > 0:
                mean_pos = np.mean(particles_at_time_t)
                closest_idx = np.argmin(np.abs(xSpace - mean_pos))
                if Dx > 1e-14:
                    m_density_estimated[closest_idx] = 1.0 / Dx
                elif Nx == 1:
                    m_density_estimated[closest_idx] = 1.0

            # Normalization logic will apply below if self.normalize_kde_output is True
            # If not normalizing, this peak might not integrate to 1 if Dx isn't "right" for one particle.
            # However, with many particles, this case becomes less likely.
        else:
            try:
                kde = gaussian_kde(particles_at_time_t, bw_method=self.kde_bandwidth)
                m_density_estimated = kde(xSpace)

                m_density_estimated[xSpace < xmin] = 0
                m_density_estimated[xSpace > xmax] = 0

            except Exception as e:
                # print(f"Error during KDE: {e}. Defaulting to peak approximation.")
                m_density_estimated = np.zeros(Nx)  # Fallback
                if len(particles_at_time_t) > 0:
                    mean_pos = np.mean(particles_at_time_t)
                    closest_idx = np.argmin(np.abs(xSpace - mean_pos))
                    if Dx > 1e-14:
                        m_density_estimated[closest_idx] = 1.0 / Dx
                    elif Nx == 1:
                        m_density_estimated[closest_idx] = 1.0

        # Normalization step (now optional)
        if self.normalize_kde_output:
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
        self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray
    ) -> np.ndarray:
        # (Rest of the solve_fp_system method remains the same as in particle_fp_py_v1)
        # print(f"****** Solving FP ({self.fp_method_name}) with {self.num_particles} particles ******")
        Nx = self.problem.Nx + 1
        Nt = self.problem.Nt + 1
        Dx = self.problem.Dx
        Dt = self.problem.Dt
        sigma_sde = self.problem.sigma
        coefCT = self.problem.coefCT
        x_grid = self.problem.xSpace
        xmin = self.problem.xmin
        Lx = self.problem.Lx

        if Nt == 0:
            return np.zeros((0, Nx))

        M_density_on_grid = np.zeros((Nt, Nx))
        current_M_particles_t = np.zeros((Nt, self.num_particles))

        if Dx > 1e-14 and np.sum(m_initial_condition * Dx) > 1e-9:
            m0_probs_unnormalized = m_initial_condition * Dx
            m0_probs = m0_probs_unnormalized / np.sum(m0_probs_unnormalized)
            try:
                initial_particle_positions = np.random.choice(
                    x_grid, size=self.num_particles, p=m0_probs, replace=True
                )
            except ValueError as e:
                initial_particle_positions = np.random.uniform(
                    xmin, xmin + Lx, self.num_particles
                )
        else:
            initial_particle_positions = (
                np.random.uniform(xmin, xmin + Lx, self.num_particles)
                if Lx > 1e-14
                else np.full(self.num_particles, xmin)
            )

        current_M_particles_t[0, :] = initial_particle_positions
        M_density_on_grid[0, :] = self._estimate_density_from_particles(
            current_M_particles_t[0, :]
        )

        if Nt == 1:
            self.M_particles_trajectory = current_M_particles_t
            return M_density_on_grid

        for n_time_idx in range(Nt - 1):
            U_at_tn = U_solution_for_drift[n_time_idx, :]

            if Nx > 1 and Dx > 1e-14:
                dUdx_grid = (np.roll(U_at_tn, -1) - np.roll(U_at_tn, 1)) / (2 * Dx)
            else:
                dUdx_grid = np.zeros(Nx)

            if Nx > 1:
                try:
                    interp_func_dUdx = scipy.interpolate.interp1d(
                        x_grid, dUdx_grid, kind="linear", fill_value="extrapolate"
                    )
                    dUdx_at_particles = interp_func_dUdx(
                        current_M_particles_t[n_time_idx, :]
                    )
                except ValueError as ve:
                    dUdx_at_particles = np.zeros(self.num_particles)
            else:
                dUdx_at_particles = np.zeros(self.num_particles)

            alpha_optimal_at_particles = -coefCT * dUdx_at_particles

            if Dt > 1e-14:
                dW = np.random.normal(0.0, np.sqrt(Dt), self.num_particles)
            else:
                dW = np.zeros(self.num_particles)

            current_M_particles_t[n_time_idx + 1, :] = (
                current_M_particles_t[n_time_idx, :]
                + alpha_optimal_at_particles * Dt
                + sigma_sde * dW
            )

            if Lx > 1e-14:
                current_M_particles_t[n_time_idx + 1, :] = (
                    xmin + (current_M_particles_t[n_time_idx + 1, :] - xmin) % Lx
                )

            M_density_on_grid[n_time_idx + 1, :] = (
                self._estimate_density_from_particles(
                    current_M_particles_t[n_time_idx + 1, :]
                )
            )

        self.M_particles_trajectory = current_M_particles_t
        return M_density_on_grid
