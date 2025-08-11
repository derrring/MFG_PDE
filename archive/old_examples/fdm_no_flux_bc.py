import time

import numpy as np

from mfg_pde.alg.damped_fixed_point_iterator import FixedPointIterator
from mfg_pde.alg.fp_solvers import FPFDMSolver
from mfg_pde.alg.hjb_solvers import HJBFDMSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.utils.plot_utils import plot_convergence, plot_results


def run_no_flux_fdm_test():
    print("--- Testing FDM with No-Flux Boundary Conditions ---")
    print("--- This should show perfect mass conservation ---")

    # Problem parameters
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 50,
        "T": 1.0,
        "Nt": 50,
        "sigma": 1.0,
        "coefCT": 0.5,
    }
    mfg_problem = ExampleMFGProblem(**problem_params)

    # Solver parameters
    Niter_max_picard = 50
    conv_threshold_picard = 1e-5
    damping_factor = 0.5
    NiterNewton = 30
    l2errBoundNewton = 1e-7

    print("\n--- Instantiating Solvers with No-Flux Boundaries ---")

    # HJB solver (FDM)
    hjb_solver_component = HJBFDMSolver(mfg_problem, NiterNewton=NiterNewton, l2errBoundNewton=l2errBoundNewton)

    # FP solver (FDM) with no-flux boundaries
    no_flux_bc = BoundaryConditions(type="no_flux")
    fp_solver_component = FPFDMSolver(mfg_problem, boundary_conditions=no_flux_bc)

    # Fixed point iterator
    no_flux_iterator = FixedPointIterator(
        mfg_problem,
        hjb_solver=hjb_solver_component,
        fp_solver=fp_solver_component,
        thetaUM=damping_factor,
    )

    solver_name = no_flux_iterator.name

    print(f"\n--- Running {solver_name} with No-Flux Boundaries ---")
    start_time = time.time()

    U_solution, M_solution, iters_run, rel_distu, rel_distm = no_flux_iterator.solve(
        Niter_max_picard, conv_threshold_picard
    )

    solve_time = time.time() - start_time
    print(f"--- {solver_name} Finished in {solve_time:.2f} seconds ({iters_run} iterations) ---")

    # Analyze mass conservation
    if U_solution is not None and M_solution is not None and iters_run > 0:
        print("\n--- Mass Conservation Analysis ---")

        # Calculate total mass at each time step
        total_mass = np.sum(M_solution * mfg_problem.Dx, axis=1)

        print(f"Initial mass: {total_mass[0]:.10f}")
        print(f"Final mass: {total_mass[-1]:.10f}")
        print(f"Mass change: {(total_mass[-1] - total_mass[0]):.2e}")
        print(f"Relative mass change: {(total_mass[-1] - total_mass[0])/total_mass[0]*100:.6f}%")

        # Check mass conservation over time
        max_mass = np.max(total_mass)
        min_mass = np.min(total_mass)
        mass_variation = max_mass - min_mass
        print(f"Max mass variation: {mass_variation:.2e}")
        print(f"Relative mass variation: {mass_variation/total_mass[0]*100:.6f}%")

        # Test if mass is conserved within numerical precision
        if mass_variation < 1e-10:
            print("✓ EXCELLENT: Mass is conserved to machine precision!")
        elif mass_variation < 1e-6:
            print("✓ GOOD: Mass is well conserved within numerical tolerance")
        elif mass_variation < 1e-3:
            print("? FAIR: Some mass variation present")
        else:
            print("✗ POOR: Significant mass variation - check implementation")

        print("\n--- Plotting Results ---")
        plot_results(mfg_problem, U_solution, M_solution, solver_name=f"{solver_name}_NoFlux")
        plot_convergence(iters_run, rel_distu, rel_distm, solver_name=f"{solver_name}_NoFlux")

        # Additional mass conservation plot
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(mfg_problem.tSpace, total_mass, "b-", linewidth=2)
        plt.xlabel("Time t")
        plt.ylabel("Total Mass ∫m(t,x)dx")
        plt.title(f"Mass Conservation - No-Flux Boundaries ({solver_name})")
        plt.grid(True)
        # Set y-axis to show small variations around 1.0
        plt.ylim([min(0.999, min_mass * 0.9999), max(1.001, max_mass * 1.0001)])
        plt.show()

    print("\n--- No-Flux FDM Test Finished ---")


if __name__ == "__main__":
    run_no_flux_fdm_test()
