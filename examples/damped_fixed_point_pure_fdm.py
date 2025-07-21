import numpy as np
import time

# Adjust these imports based on your package structure and where this script is located
# Assuming this script is outside the mfg_pde package, and mfg_pde is installed
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.alg.hjb_solvers.fdm_hjb import FdmHJBSolver
from mfg_pde.alg.fp_solvers.fdm_fp import FdmFPSolver
from mfg_pde.alg.damped_fixed_point_iterator import FixedPointIterator
from mfg_pde.utils.plot_utils import plot_results, plot_convergence
from mfg_pde.core.boundaries import BoundaryConditions


def run_pure_fdm_example():
    print("--- Setting up MFG Problem (Pure FDM Example) ---")
    # --- Define Problem Parameters ---
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 5,
        "T": 1,
        "Nt": 50,  # Adjusted T for potentially faster run
        "sigma": 1,
        "coefCT": 0.5,
    }
    # Use the concrete ExampleMFGProblem
    mfg_problem = ExampleMFGProblem(**problem_params)

    # --- Solver Parameters ---
    Niter_max_picard = 100  # Max iterations for the fixed-point loop
    conv_threshold_picard = 1e-5  # Convergence tolerance for Picard iteration
    damping_factor = 0.5  # Damping for the fixed-point iteration (thetaUM)

    # Parameters for HJB Newton solver (used within FdmHJBSolver)
    NiterNewton = 30
    l2errBoundNewton = 1e-7

    # --- Instantiate Solvers for Pure FDM case ---
    print("\n--- Running Pure FDM Solver (via FixedPointIterator) ---")

    # 1. Instantiate the HJB solver component (FDM)
    hjb_solver_component = FdmHJBSolver(
        mfg_problem, NiterNewton=NiterNewton, l2errBoundNewton=l2errBoundNewton
    )

    # 2. Instantiate the FP solver component (FDM) with Dirichlet boundaries
    # Use homogeneous Dirichlet boundaries (m=0 at boundaries) to see mass loss
    dirichlet_bc = BoundaryConditions(type="dirichlet", left_value=0.0, right_value=0.0)
    fp_solver_component = FdmFPSolver(mfg_problem, boundary_conditions=dirichlet_bc)

    # 3. Instantiate the FixedPointIterator with these FDM components
    pure_fdm_iterator = FixedPointIterator(
        mfg_problem,
        hjb_solver=hjb_solver_component,
        fp_solver=fp_solver_component,
        thetaUM=damping_factor,
    )

    solver_name_fdm = pure_fdm_iterator.name

    start_time_fdm = time.time()
    # solve returns: U, M, iterations_run, l2disturel_u, l2disturel_m
    U_fdm, M_fdm, iters_fdm, rel_distu_fdm, rel_distm_fdm = pure_fdm_iterator.solve(
        Niter_max_picard, conv_threshold_picard
    )
    time_fdm_solve = time.time() - start_time_fdm
    print(
        f"--- {solver_name_fdm} Solver Finished in {time_fdm_solve:.2f} seconds ({iters_fdm} iterations) ---"
    )

    # Plot results
    if U_fdm is not None and M_fdm is not None and iters_fdm > 0:
        print("\n--- Plotting Pure FDM Solver Results ---")
        plot_results(mfg_problem, U_fdm, M_fdm, solver_name=solver_name_fdm)
        plot_convergence(
            iters_fdm, rel_distu_fdm, rel_distm_fdm, solver_name=solver_name_fdm
        )

    print("\n--- Pure FDM Example Script Finished ---")


if __name__ == "__main__":
    run_pure_fdm_example()
