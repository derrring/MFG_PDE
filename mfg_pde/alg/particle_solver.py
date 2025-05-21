import numpy as np
from scipy.stats import gaussian_kde
import scipy.interpolate as interpolate
import scipy.sparse as sparse
import scipy.sparse.linalg
import time
from ..env.base_solver import MFGSolver
from ..utils import hjb_utils


class ParticleSolver(MFGSolver):
    def __init__(
        self,
        problem,
        num_particles,
        particle_thetaUM=0.5,
        kde_bandwidth="scott",
        NiterNewton=30,
        l2errBoundNewton=1e-6,
    ):
        super().__init__(problem)
        self.num_particles = num_particles
        self.thetaUM = particle_thetaUM  # Damping factor for fixed-point iteration
        self.kde_bandwidth = kde_bandwidth
        self.NiterNewton = NiterNewton  # Retain for passing
        self.l2errBoundNewton = l2errBoundNewton  # Retain for passing

        self.U = None
        self.M_particles = None  # Stores particle positions (Nt+1, num_particles)
        self.M_density = None  # Stores estimated density on grid (Nt+1, Nx)
        self.l2distu = None
        self.l2distm_density = None
        self.l2disturel = None
        self.l2distmrel_density = None
        self.iterations_run = 0

    def _estimate_density_from_particles(self, particles_at_time_t):
        # This function remains unchanged.
        Nx = self.problem.Nx
        xSpace = self.problem.xSpace
        xmin = self.problem.xmin
        xmax = self.problem.xmax
        Dx = self.problem.Dx

        if len(particles_at_time_t) == 0:
            # print(f"Warning: KDE input has no particles.") # Optional warning
            return np.zeros(Nx)

        # Handle cases with all particles at the same spot (KDE fails or gives poor results)
        if np.std(particles_at_time_t) < 1e-9 * (xmax - xmin):  # Use relative tolerance
            # print(f"Warning: KDE input problematic at t (stddev too small). Creating a narrow peak.") # Optional warning
            # Create a density concentrated where particles are
            m_density_estimated = np.zeros(Nx)
            # Find closest grid point to the common particle position
            mean_pos = np.mean(particles_at_time_t)
            closest_idx = np.argmin(np.abs(xSpace - mean_pos))
            m_density_estimated[closest_idx] = 1.0 / Dx  # Approximate delta function
            return m_density_estimated

        try:
            kde = scipy.stats.gaussian_kde(
                particles_at_time_t, bw_method=self.kde_bandwidth
            )
            m_density_estimated = kde(xSpace)
            m_density_estimated[xSpace < xmin] = (
                0  # Ensure density is zero outside domain
            )
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

    def _solveFP_particle(self, m0_particles_initial_positions, U_solution):
        # This function (particle SDE solver) remains largely unchanged.
        # Ensure self.problem references are correct.
        print("****** Solving FP (Particle Evolution)")
        Nt = self.problem.Nt
        Dt = self.problem.Dt
        sigma_sde = (
            self.problem.sigma
        )  # Sigma for SDE, distinct from HJB sigma if needed
        Dx = self.problem.Dx
        coefCT = self.problem.coefCT  # For optimal control

        # M_particles_new will store trajectories: (Nt+1, num_particles)
        current_M_particles = np.zeros((Nt + 1, self.num_particles))
        current_M_particles[0, :] = m0_particles_initial_positions

        x_grid = self.problem.xSpace  # Grid for interpolating dU/dx

        for n_time_idx in range(Nt):  # Loop 0 to Nt-1
            U_at_tn = U_solution[n_time_idx, :]

            # Estimate gradient dU/dx at time n_time_idx on the grid (periodic)
            dUdx_grid = (np.roll(U_at_tn, -1) - np.roll(U_at_tn, 1)) / (2 * Dx)

            # Interpolate gradient onto current particle positions X_t = current_M_particles[n_time_idx,:]
            try:
                interp_func_dUdx = scipy.interpolate.interp1d(
                    x_grid, dUdx_grid, kind="linear", fill_value="extrapolate"
                )
                dUdx_at_particles = interp_func_dUdx(current_M_particles[n_time_idx, :])
            except ValueError as ve:
                print(
                    f"Interpolation error at t_idx {n_time_idx}: {ve}. Particles might be out of bounds."
                )
                # Handle particles out of bounds for interpolation if `fill_value="extrapolate"` is not robust enough
                # For now, this might lead to NaNs if particles are far out.
                # A robust solution would be to cap particle positions to domain before interpolation or use nearest fill.
                dUdx_at_particles = np.zeros(self.num_particles)  # Fallback

            # Optimal control alpha = -coefCT * dU/dx
            alpha_optimal_at_particles = -coefCT * dUdx_at_particles

            # Evolve particles using Euler-Maruyama: dX = alpha*dt + sigma*dW
            dW = np.random.normal(0.0, np.sqrt(Dt), self.num_particles)
            current_M_particles[n_time_idx + 1, :] = (
                current_M_particles[n_time_idx, :]
                + alpha_optimal_at_particles * Dt
                + sigma_sde * dW
            )

            # Apply periodic boundary conditions for particles
            xmin = self.problem.xmin
            Lx = self.problem.Lx  # Length of domain
            current_M_particles[n_time_idx + 1, :] = (
                xmin + (current_M_particles[n_time_idx + 1, :] - xmin) % Lx
            )

        return current_M_particles

    # REMOVE its own copies of _getPhi_U, _getJacobianU, _solveHJB_NewtonStep, _solveHJBTimeStep

    def _solveHJB_FDM(self, M_density_old_from_FP_iter):  # M_density_old is from KDE
        print("****** Solving HJB (FDM within Particle Method via hjb_utils)")
        U_new_solution = hjb_utils.solve_hjb_system_backward(
            M_density_evolution_from_FP=M_density_old_from_FP_iter,
            U_final_condition_at_T=self.problem.get_final_u(),
            problem=self.problem,
            NiterNewton=self.NiterNewton,
            l2errBoundNewton=self.l2errBoundNewton,
        )
        return U_new_solution

    def solve(self, Niter, l2errBoundPicard=1e-5):
        print(
            f"\n________________ Solving MFG with Hybrid Particle Method (T={self.problem.T}) _______________"
        )
        Nt = self.problem.Nt
        Nx = self.problem.Nx
        Dx = self.problem.Dx
        Dt = self.problem.Dt

        # Initialization
        self.U = np.zeros((Nt + 1, Nx))
        final_u_cost = self.problem.get_final_u()
        for n_idx in range(Nt + 1):
            self.U[n_idx] = final_u_cost  # Initialize U with final cost

        m0_density_on_grid = self.problem.get_initial_m()
        # Sample initial particle positions based on m0_density_on_grid
        # Ensure probabilities sum to 1 for np.random.choice
        m0_probs = m0_density_on_grid * Dx
        if not np.isclose(np.sum(m0_probs), 1.0):
            m0_probs = m0_probs / np.sum(m0_probs)  # Normalize if not already

        initial_particle_positions = np.random.choice(
            self.problem.xSpace, size=self.num_particles, p=m0_probs
        )

        self.M_particles = np.zeros((Nt + 1, self.num_particles))
        self.M_particles[0, :] = initial_particle_positions

        self.M_density = np.zeros((Nt + 1, Nx))
        self.M_density[0, :] = self._estimate_density_from_particles(
            self.M_particles[0, :]
        )

        # Initialize M_particles and M_density for t > 0 (as per original)
        for n_idx in range(1, Nt + 1):
            self.M_particles[n_idx, :] = self.M_particles[
                0, :
            ]  # Initialize with t=0 positions
            self.M_density[n_idx, :] = self.M_density[
                0, :
            ]  # Initialize with t=0 density estimate

        self.l2distu = np.ones(Niter)
        self.l2distm_density = np.ones(Niter)  # Convergence for M_density
        self.l2disturel = np.ones(Niter)
        self.l2distmrel_density = np.ones(Niter)
        self.iterations_run = 0

        for iiter in range(Niter):
            start_time_iter = time.time()
            print(
                f"\n******************** Hybrid Particle Iteration = {iiter + 1} / {Niter}"
            )

            U_old_iter = self.U.copy()
            M_density_old_iter = (
                self.M_density.copy()
            )  # Store previously estimated density

            # Solve HJB backward using the *estimated density* M_density_old_iter
            U_new_tmp_hjb = self._solveHJB_FDM(M_density_old_iter)

            # Apply damping to U update
            self.U = self.thetaUM * U_new_tmp_hjb + (1 - self.thetaUM) * U_old_iter

            # Evolve particles forward using the new U
            # Use the *initial* particle positions from this iteration's M_particles[0,:]
            # which should be consistent if M_particles isn't re-damped/changed in an unexpected way.
            # The original code passed M_particles_old[0,:], which implies initial positions don't change.
            self.M_particles = self._solveFP_particle(self.M_particles[0, :], self.U)

            # Estimate density on the grid from the new particle positions
            M_density_new_estimated_fp = np.zeros_like(self.M_density)
            for n_idx in range(Nt + 1):  # Estimate density at each time step
                M_density_new_estimated_fp[n_idx, :] = (
                    self._estimate_density_from_particles(self.M_particles[n_idx, :])
                )

            # Apply damping to M_density update (original code did not explicitly show M_density damping)
            # If damping M_density:
            # self.M_density = self.thetaUM * M_density_new_estimated_fp + (1 - self.thetaUM) * M_density_old_iter
            # If no damping for M_density (just use the new estimate):
            self.M_density = M_density_new_estimated_fp

            # Convergence Check
            self.l2distu[iiter] = np.linalg.norm(self.U - U_old_iter) * np.sqrt(Dx * Dt)
            norm_U_iter = np.linalg.norm(self.U) * np.sqrt(Dx * Dt)
            self.l2disturel[iiter] = (
                self.l2distu[iiter] / norm_U_iter
                if norm_U_iter > 1e-9
                else self.l2distu[iiter]
            )

            self.l2distm_density[iiter] = np.linalg.norm(
                self.M_density - M_density_old_iter
            ) * np.sqrt(Dx * Dt)
            norm_M_dens_iter = np.linalg.norm(self.M_density) * np.sqrt(Dx * Dt)
            self.l2distmrel_density[iiter] = (
                self.l2distm_density[iiter] / norm_M_dens_iter
                if norm_M_dens_iter > 1e-9
                else self.l2distm_density[iiter]
            )

            elapsed_time_iter = time.time() - start_time_iter
            print(
                f" === END Iteration {iiter+1}: ||u_new - u_old||_2 = {self.l2distu[iiter]:.4e} (rel: {self.l2disturel[iiter]:.4e})"
            )
            print(
                f" === END Iteration {iiter+1}: ||m_dens_new - m_dens_old||_2 = {self.l2distm_density[iiter]:.4e} (rel: {self.l2distmrel_density[iiter]:.4e})"
            )
            print(f" === Time for iteration = {elapsed_time_iter:.2f} s")

            self.iterations_run = iiter + 1
            if (
                self.l2disturel[iiter] < l2errBoundPicard
                and self.l2distmrel_density[iiter] < l2errBoundPicard
            ):
                print(f"Convergence reached after {iiter + 1} iterations.")
                break

        # Trim convergence arrays
        self.l2distu = self.l2distu[: self.iterations_run]
        self.l2distm_density = self.l2distm_density[: self.iterations_run]
        self.l2disturel = self.l2disturel[: self.iterations_run]
        self.l2distmrel_density = self.l2distmrel_density[: self.iterations_run]

        return (
            self.U,
            self.M_density,
            self.iterations_run,
            self.l2distu,
            self.l2distm_density,
        )

    def get_results(self):
        return self.U, self.M_density  # Return estimated density

    def get_convergence_data(self):
        return (
            self.iterations_run,
            self.l2distu,
            self.l2distm_density,
            self.l2disturel,
            self.l2distmrel_density,
        )
