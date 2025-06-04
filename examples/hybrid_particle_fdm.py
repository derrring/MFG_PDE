import numpy as np
import time

# Adjust these imports based on your package structure and where this script is located.
# This assumes the script is run from a location where 'mfg_pde' is in PYTHONPATH.
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.alg.hjb_solvers.fdm_hjb import FdmHJBSolver
from mfg_pde.alg.fp_solvers.particle_fp import ParticleFPSolver
from mfg_pde.alg.damped_fixed_point_iterator import FixedPointIterator
from mfg_pde.utils.plot_utils import plot_results, plot_convergence


def run_hybrid_fdm_particle_example():
    print("--- Setting up MFG Problem (Hybrid FDM-HJB / Particle-FP Example) ---")
    # --- Define Problem Parameters ---
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 51,  # Number of spatial points
        "T": 0.5,
        "Nt": 51,  # Number of time points
        "sigma": 0.1,
        "coefCT": 0.5,  # Diffusion and control coefficient
    }
    # Use the concrete ExampleMFGProblem
    mfg_problem = ExampleMFGProblem(**problem_params)

    # --- Solver Parameters ---
    Niter_max_picard = 50  # Max iterations for the fixed-point loop
    conv_threshold_picard = 1e-5  # Convergence tolerance for Picard iteration
    damping_factor = 0.5  # Damping for the fixed-point iteration (thetaUM)

    # Parameters for HJB Newton solver (used within FdmHJBSolver)
    NiterNewton = 30  # Max Newton iterations per HJB time step
    l2errBoundNewton = 1e-7  # Newton convergence tolerance for HJB

    # Parameters for Particle FP solver
    num_particles = 10000  # Number of particles for FP simulation
    kde_bandwidth = "scott"  # KDE bandwidth estimation method (e.g., 'scott', 'silverman', or a float)

    # --- Instantiate Solvers ---
    print("\n--- Running HJB-FDM / FP-Particle Solver (via FixedPointIterator) ---")

    # 1. Instantiate the HJB solver component (FDM)
    hjb_solver_component = FdmHJBSolver(
        mfg_problem, NiterNewton=NiterNewton, l2errBoundNewton=l2errBoundNewton
    )

    # 2. Instantiate the FP solver component (Particle)
    fp_solver_component = ParticleFPSolver(
        mfg_problem, num_particles=num_particles, kde_bandwidth=kde_bandwidth
    )

    # 3. Instantiate the FixedPointIterator with these components
    hybrid_iterator = FixedPointIterator(
        mfg_problem,
        hjb_solver=hjb_solver_component,
        fp_solver=fp_solver_component,
        thetaUM=damping_factor,
    )

    solver_name_hybrid = hybrid_iterator.name

    start_time_hybrid = time.time()
    # solve returns: U, M, iterations_run, l2disturel_u, l2disturel_m
    U_hybrid, M_hybrid_density, iters_hybrid, rel_distu_hybrid, rel_distm_hybrid = (
        hybrid_iterator.solve(Niter_max_picard, conv_threshold_picard)
    )
    time_hybrid_solve = time.time() - start_time_hybrid

    if iters_hybrid > 0:
        print(
            f"--- {solver_name_hybrid} Solver Finished in {time_hybrid_solve:.2f} seconds ({iters_hybrid} iterations) ---"
        )
        final_rel_err_U = (
            rel_distu_hybrid[iters_hybrid - 1] if iters_hybrid > 0 else float("nan")
        )
        final_rel_err_M = (
            rel_distm_hybrid[iters_hybrid - 1] if iters_hybrid > 0 else float("nan")
        )
        print(f"    Final relative error U: {final_rel_err_U:.2e}")
        print(f"    Final relative error M: {final_rel_err_M:.2e}")

        # Plot results
        if U_hybrid is not None and M_hybrid_density is not None:
            print("\n--- Plotting Hybrid Solver Results ---")
            plot_results(
                mfg_problem, U_hybrid, M_hybrid_density, solver_name=solver_name_hybrid
            )
            plot_convergence(
                iters_hybrid,
                rel_distu_hybrid,
                rel_distm_hybrid,
                solver_name=solver_name_hybrid,
            )
    else:
        print(f"--- {solver_name_hybrid} Solver did not run any iterations. ---")

    print("\n--- Hybrid FDM-Particle Example Script Finished ---")


if __name__ == "__main__":
    run_hybrid_fdm_particle_example()
