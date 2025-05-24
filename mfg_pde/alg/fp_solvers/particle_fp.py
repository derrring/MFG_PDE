import numpy as np
from scipy.stats import gaussian_kde
import scipy.interpolate as interpolate
from .base_fp import BaseFPSolver


class ParticleFPSolver(BaseFPSolver):
    def __init__(self, problem, num_particles, kde_bandwidth="scott"):
        super().__init__(problem)
        self.fp_method_name = "Particle"
        self.num_particles = num_particles
        self.kde_bandwidth = kde_bandwidth
        self.M_particles_trajectory = None # To store particle paths if needed

    def _estimate_density_from_particles(self, particles_at_time_t):
        # Copied directly from ParticleSolver._estimate_density_from_particles
        Nx = self.problem.Nx
        xSpace = self.problem.xSpace
        xmin = self.problem.xmin
        xmax = self.problem.xmax
        Dx = self.problem.Dx

        if len(particles_at_time_t) == 0:
            return np.zeros(Nx)
        if np.std(particles_at_time_t) < 1e-9 * (xmax - xmin):
            m_density_estimated = np.zeros(Nx)
            mean_pos = np.mean(particles_at_time_t)
            closest_idx = np.argmin(np.abs(xSpace - mean_pos))
            m_density_estimated[closest_idx] = 1.0 / Dx
            return m_density_estimated
        try:
            kde = gaussian_kde(particles_at_time_t, bw_method=self.kde_bandwidth)
            m_density_estimated = kde(xSpace)
            m_density_estimated[xSpace < xmin] = 0
            m_density_estimated[xSpace > xmax] = 0
            current_mass = np.sum(m_density_estimated) * Dx
            if current_mass > 1e-9:
                m_density_normalized = m_density_estimated / current_mass
            else:
                m_density_normalized = np.zeros(Nx)
            return m_density_normalized
        except Exception as e:
            print(f"Error during KDE: {e}. Returning zero density.")
            return np.zeros(Nx)

    def solve_fp_system(self, m_initial_condition_density, U_solution_for_drift):
        """
        Solves the full FP system forward in time using a particle method.
        Args:
            m_initial_condition_density (np.array): Initial density M(0,x) on the grid.
                                                    Used to sample initial particle positions.
            U_solution_for_drift (np.array): (Nt+1, Nx) array of value function U(t,x) used for drift.
        Returns:
            np.array: M_density_on_grid (Nt+1, Nx) - estimated density on the grid.
        """
        print(f"****** Solving FP ({self.fp_method_name} Evolution)")
        Nt = self.problem.Nt
        Nx = self.problem.Nx
        Dt = self.problem.Dt
        Dx = self.problem.Dx
        sigma_sde = self.problem.sigma
        coefCT = self.problem.coefCT
        x_grid = self.problem.xSpace

        # Sample initial particle positions from m_initial_condition_density
        m0_probs = m_initial_condition_density * Dx
        if not np.isclose(np.sum(m0_probs), 1.0) and np.sum(m0_probs) > 1e-9: # check sum > 0 before normalizing
            m0_probs = m0_probs / np.sum(m0_probs)
        elif np.sum(m0_probs) <= 1e-9: # Handle case of zero initial density
             m0_probs = np.ones(Nx) / Nx # Fallback to uniform if no density

        initial_particle_positions = np.random.choice(
            x_grid, size=self.num_particles, p=m0_probs
        )

        current_M_particles = np.zeros((Nt + 1, self.num_particles))
        current_M_particles[0, :] = initial_particle_positions

        M_density_on_grid = np.zeros((Nt + 1, Nx))
        M_density_on_grid[0, :] = self._estimate_density_from_particles(current_M_particles[0, :])

        for n_time_idx in range(Nt):
            U_at_tn = U_solution_for_drift[n_time_idx, :]
            dUdx_grid = (np.roll(U_at_tn, -1) - np.roll(U_at_tn, 1)) / (2 * Dx)
            try:
                interp_func_dUdx = interpolate.interp1d(
                    x_grid, dUdx_grid, kind="linear", fill_value="extrapolate"
                )
                dUdx_at_particles = interp_func_dUdx(current_M_particles[n_time_idx, :])
            except ValueError as ve:
                print(f"Interpolation error at t_idx {n_time_idx}: {ve}.")
                dUdx_at_particles = np.zeros(self.num_particles)

            alpha_optimal_at_particles = -coefCT * dUdx_at_particles
            dW = np.random.normal(0.0, np.sqrt(Dt), self.num_particles)
            current_M_particles[n_time_idx + 1, :] = (
                current_M_particles[n_time_idx, :]
                + alpha_optimal_at_particles * Dt
                + sigma_sde * dW
            )
            xmin = self.problem.xmin
            Lx = self.problem.Lx
            current_M_particles[n_time_idx + 1, :] = (
                xmin + (current_M_particles[n_time_idx + 1, :] - xmin) % Lx
            )

            # Estimate density on grid at this time step
            M_density_on_grid[n_time_idx + 1, :] = self._estimate_density_from_particles(
                current_M_particles[n_time_idx + 1, :]
            )

        self.M_particles_trajectory = current_M_particles # Store for potential later access
        return M_density_on_grid