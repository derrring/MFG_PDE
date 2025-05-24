from ..mfg_pde.core.mfg_problem import MFGProblem
from ..mfg_pde.alg.hjb_solvers.fdm_hjb import FdmHJBSolver 
from ..mfg_pde.alg.fp_solvers.particle_fp import ParticleFPSolver 
from ..mfg_pde.alg.damped_fixed_point_iterators import FixedPointIterator


print("--- Setting up MFG Problem ---")
# --- Define Problem Parameters ---
problem_params = {
    "xmin": 0.0, "xmax": 1.0, "Nx": 50,
    "T": 1, "Nt": 100, "sigma": 1.0, "coefCT": 0.5
}
mfg_problem = MFGProblem(**problem_params)

# --- Solver Parameters ---
Niter_max = 1
conv_threshold = 1e-5
fdm_damping = 0.5
particle_damping = 0.5
num_particles = 5000

# --- Execution Flags ---
run_fdm = True
run_particle = True

# --- Initialize results ---
U_fdm, M_fdm, U_particle, M_particle_density = None, None, None, None
time_fdm, time_particle = 0, 0
iters_fdm, iters_particle = 0, 0


if run_particle: # Assuming you want to run the HJB-FDM_FP-Particle combination
    print("\n--- Running HJB-FDM / FP-Particle Solver ---")

    # 1. Instantiate the HJB solver component
    hjb_solver_component = FdmHJBSolver(mfg_problem, 
                                        NiterNewton=NiterNewton, 
                                        l2errBoundNewton=l2errBoundNewton)

    # 2. Instantiate the FP solver component
    fp_solver_component = ParticleFPSolver(mfg_problem, 
                                           num_particles=num_particles, 
                                           kde_bandwidth=kde_bandwidth) # Pass relevant params

    # 3. Instantiate the FixedPointIterator with these components
    hybrid_solver = FixedPointIterator(mfg_problem,
                                       hjb_solver=hjb_solver_component,
                                       fp_solver=fp_solver_component,
                                       thetaUM=particle_damping) # Damping for the fixed point iteration

    solver_name_hybrid = hybrid_solver.name # Will be "HJB-FDM_FP-Particle"
    current_prefix_hybrid = f"exp1_{solver_name_hybrid}_sigma{mfg_problem.sigma}_Nx{mfg_problem.Nx}_Np{num_particles}_{timestamp}"

    start_time_hybrid = time.time()
    U_particle, M_particle_density, iters_particle, _, _ = hybrid_solver.solve(Niter_max, conv_threshold)
    time_hybrid = time.time() - start_time_hybrid
    print(f"--- {solver_name_hybrid} Solver Finished in {time_hybrid:.2f} seconds ---")

    # Assuming get_convergence_data provides relative errors as the 3rd and 4th error terms
    _, _, _, rel_distu_particle, rel_distm_particle_density = hybrid_solver.get_convergence_data()

    # Plot results
    if U_particle is not None:
        plot_results(mfg_problem, U_particle, M_particle_density, 
                     solver_name=solver_name_hybrid, prefix=current_prefix_hybrid)
        plot_convergence(iters_particle, rel_distu_particle, rel_distm_particle_density, 
                         solver_name=solver_name_hybrid, prefix=current_prefix_hybrid)