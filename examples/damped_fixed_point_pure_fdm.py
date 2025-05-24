import matplotlib.pyplot as plt
import numpy as np
import time
from ..mfg_pde.core import MFGProblem
from ..mfg_pde.alg import FDMSolver, ParticleSolver
from ..mfg_pde.utils import plot_results, plot_convergence

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

# --- FDM Solver Execution ---
if run_fdm:
    print("\n--- Running FDM Solver ---")
    fdm_solver = FDMSolver(mfg_problem, thetaUM=fdm_damping)
    start_time_fdm = time.time()
    U_fdm, M_fdm, iters_fdm, distu_fdm, distm_fdm = fdm_solver.solve(Niter_max, conv_threshold)
    time_fdm = time.time() - start_time_fdm
    print(f"--- FDM Solver Finished in {time_fdm:.2f} seconds ---")
    _, _, _, rel_distu_fdm, rel_distm_fdm = fdm_solver.get_convergence_data()

# --- Particle Solver Execution ---
if run_particle:
    print("\n--- Running Particle Solver ---")
    particle_solver = ParticleSolver(mfg_problem, num_particles=num_particles, particle_thetaUM=particle_damping)
    start_time_particle = time.time()
    U_particle, M_particle_density, iters_particle, distu_particle, distm_particle_density = particle_solver.solve(Niter_max, conv_threshold)
    time_particle = time.time() - start_time_particle
    print(f"--- Particle Solver Finished in {time_particle:.2f} seconds ---")
    _, _, _, rel_distu_particle, rel_distm_particle_density = particle_solver.get_convergence_data()


# --- Plotting and Comparison ---
print("\n--- Generating Plots and Comparisons ---")
prefix = f"comparison_sigma{mfg_problem.sigma}_T{mfg_problem.T}_Nx{mfg_problem.Nx}_Nt{mfg_problem.Nt}"

# Plot FDM Results
if run_fdm and U_fdm is not None:
    plot_results(prefix, "FDM", mfg_problem, U_fdm, M_fdm)
    plot_convergence(prefix, "FDM", iters_fdm, distu_fdm, distm_fdm, rel_distu_fdm, rel_distm_fdm)

# Plot Particle Method Results
if run_particle and U_particle is not None:
    plot_results(prefix, "Particle", mfg_problem, U_particle, M_particle_density)
    plot_convergence(prefix, "Particle", iters_particle, distu_particle, distm_particle_density, rel_distu_particle, rel_distm_particle_density)


# --- Direct Comparison Plots (only if both ran) ---
if run_fdm and run_particle and U_fdm is not None and U_particle is not None:
    # --- Comparison of Final Density and Initial Value ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(mfg_problem.xSpace, M_fdm[-1,:], label=f'FDM (Final - {iters_fdm} iters)')
    plt.plot(mfg_problem.xSpace, M_particle_density[-1,:], label=f'Particle (Final - {iters_particle} iters)', linestyle='--')
    plt.title('Comparison of Final Densities m(T,x)')
    plt.xlabel('x')
    plt.ylabel('m')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(mfg_problem.xSpace, U_fdm[0,:], label=f'FDM (Initial - {iters_fdm} iters)')
    plt.plot(mfg_problem.xSpace, U_particle[0,:], label=f'Particle (Initial - {iters_particle} iters)', linestyle='--')
    plt.title('Comparison of Initial Values U(0,x)')
    plt.xlabel('x')
    plt.ylabel('U')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    #plt.savefig(f'{prefix}_comparison_final.pdf')
    plt.show()


    # --- ADDED: Comparison of Total Mass Evolution ---
    plt.figure(figsize=(8, 5))
    mtot_fdm = np.sum(M_fdm * mfg_problem.Dx, axis=1)
    mtot_particle = np.sum(M_particle_density * mfg_problem.Dx, axis=1)
    plt.plot(mfg_problem.tSpace, mtot_fdm, label=f'FDM Total Mass ({iters_fdm} iters)')
    plt.plot(mfg_problem.tSpace, mtot_particle, label=f'Particle Total Mass ({iters_particle} iters)', linestyle='--')
    plt.title('Comparison of Total Mass Evolution $\\int m(t,x) dx$')
    plt.xlabel('Time (t)')
    plt.ylabel('Total Mass')
    plt.ylim(0.95, 1.05) # Zoom in around 1.0 for better visualization
    plt.legend()
    plt.grid(True)
    #plt.savefig(f'{prefix}_comparison_mass.pdf')
    plt.show()
    # --- End of Added Plot ---


    # --- Calculate and print difference metrics ---
    diff_M_final = np.linalg.norm(M_fdm[-1,:] - M_particle_density[-1,:]) * np.sqrt(mfg_problem.Dx)
    diff_U_initial = np.linalg.norm(U_fdm[0,:] - U_particle[0,:]) * np.sqrt(mfg_problem.Dx)
    # Difference in mass conservation (e.g., std dev from 1.0)
    mass_dev_fdm = np.std(mtot_fdm)
    mass_dev_particle = np.std(mtot_particle)


    print("\n--- Comparison Metrics ---")
    print(f"FDM Solver Time: {time_fdm:.2f} seconds ({iters_fdm} iterations)")
    print(f"Particle Solver Time: {time_particle:.2f} seconds ({iters_particle} iterations)")
    print(f"L2 Difference in final M: {diff_M_final:.4e}")
    print(f"L2 Difference in initial U: {diff_U_initial:.4e}")
    print(f"Std Dev of Total Mass (FDM): {mass_dev_fdm:.4e}")
    print(f"Std Dev of Total Mass (Particle): {mass_dev_particle:.4e}")


elif run_fdm:
    print(f"\n--- FDM Solver Finished ---")
    print(f"FDM Solver Time: {time_fdm:.2f} seconds ({iters_fdm} iterations)")
elif run_particle:
    print(f"\n--- Particle Solver Finished ---")
    print(f"Particle Solver Time: {time_particle:.2f} seconds ({iters_particle} iterations)")