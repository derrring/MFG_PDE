import numpy as np
import time

from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.alg.hjb_solvers.fdm_hjb import FdmHJBSolver
from mfg_pde.alg.fp_solvers.particle_fp import ParticleFPSolver
from mfg_pde.alg.damped_fixed_point_iterator import FixedPointIterator
from mfg_pde.utils.plot_utils import plot_results, plot_convergence
from mfg_pde.core.boundaries import BoundaryConditions


def run_hybrid_no_flux_test():
    print("--- Testing Hybrid Particle-FDM with No-Flux Boundaries ---")
    print("--- Particles should reflect at boundaries ---")

    # Problem parameters
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 50,
        "T": 10.0,
        "Nt": 500,
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

    # Particle parameters
    num_particles = 500  # Smaller number for clearer visualization
    kde_bandwidth = "scott"

    print("\n--- Instantiating Hybrid Solvers with No-Flux Boundaries ---")

    # HJB solver (FDM)
    hjb_solver_component = FdmHJBSolver(
        mfg_problem, NiterNewton=NiterNewton, l2errBoundNewton=l2errBoundNewton
    )

    # FP solver (Particle) with no-flux boundaries
    no_flux_bc = BoundaryConditions(type="no_flux")
    fp_solver_component = ParticleFPSolver(
        mfg_problem,
        num_particles=num_particles,
        kde_bandwidth=kde_bandwidth,
        normalize_kde_output=False,  # No artificial normalization
        boundary_conditions=no_flux_bc,
    )

    # Fixed point iterator
    hybrid_iterator = FixedPointIterator(
        mfg_problem,
        hjb_solver=hjb_solver_component,
        fp_solver=fp_solver_component,
        thetaUM=damping_factor,
    )

    solver_name = hybrid_iterator.name

    print(f"\n--- Running {solver_name} with No-Flux Reflecting Boundaries ---")
    start_time = time.time()

    U_solution, M_solution, iters_run, rel_distu, rel_distm = hybrid_iterator.solve(
        Niter_max_picard, conv_threshold_picard
    )

    solve_time = time.time() - start_time
    print(
        f"--- {solver_name} Finished in {solve_time:.2f} seconds ({iters_run} iterations) ---"
    )

    # Analyze mass conservation and particle behavior
    if U_solution is not None and M_solution is not None and iters_run > 0:
        print("\n--- Mass Conservation Analysis ---")

        # Calculate total mass at each time step
        total_mass = np.sum(M_solution * mfg_problem.Dx, axis=1)

        print(f"Initial mass: {total_mass[0]:.10f}")
        print(f"Final mass: {total_mass[-1]:.10f}")
        print(f"Mass change: {(total_mass[-1] - total_mass[0]):.2e}")
        print(
            f"Relative mass change: {(total_mass[-1] - total_mass[0])/total_mass[0]*100:.6f}%"
        )

        # Check mass conservation over time
        max_mass = np.max(total_mass)
        min_mass = np.min(total_mass)
        mass_variation = max_mass - min_mass
        print(f"Max mass variation: {mass_variation:.2e}")
        print(f"Relative mass variation: {mass_variation/total_mass[0]*100:.6f}%")

        # Analyze particle behavior
        particles_trajectory = fp_solver_component.M_particles_trajectory
        if particles_trajectory is not None:
            print("\n--- Particle Boundary Analysis ---")
            # Check if particles stayed within bounds
            xmin, xmax = mfg_problem.xmin, mfg_problem.xmin + mfg_problem.Lx

            # Check final particle positions
            final_particles = particles_trajectory[-1, :]
            particles_in_bounds = np.all(
                (final_particles >= xmin) & (final_particles <= xmax)
            )

            print(
                f"All particles within bounds [{xmin:.2f}, {xmax:.2f}]: {particles_in_bounds}"
            )
            print(
                f"Final particle range: [{np.min(final_particles):.4f}, {np.max(final_particles):.4f}]"
            )
            print(f"Number of particles: {len(final_particles)}")

            # Count boundary violations over time
            total_violations = 0
            for t_step in range(particles_trajectory.shape[0]):
                step_particles = particles_trajectory[t_step, :]
                violations = np.sum((step_particles < xmin) | (step_particles > xmax))
                total_violations += violations

            print(f"Total boundary violations during evolution: {total_violations}")
            if total_violations == 0:
                print("✓ EXCELLENT: No particles escaped boundaries!")
            else:
                print(f"⚠ WARNING: {total_violations} boundary violations detected")

        # Test mass conservation assessment
        if mass_variation < 1e-6:
            print("✓ EXCELLENT: Mass is well conserved within tolerance")
        elif mass_variation < 1e-3:
            print("? FAIR: Some mass variation present")
        else:
            print("✗ POOR: Significant mass variation - check KDE estimation")

        print("\n--- Plotting Results ---")
        plot_results(
            mfg_problem, U_solution, M_solution, solver_name=f"{solver_name}_NoFlux"
        )
        plot_convergence(
            iters_run, rel_distu, rel_distm, solver_name=f"{solver_name}_NoFlux"
        )

        # Mass conservation plot
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(
            mfg_problem.tSpace,
            total_mass,
            "r-",
            linewidth=2,
            label="Hybrid Particle-FDM",
        )
        plt.xlabel("Time t")
        plt.ylabel("Total Mass ∫m(t,x)dx")
        plt.title(f"Mass Conservation - No-Flux Boundaries ({solver_name})")
        plt.grid(True)
        plt.legend()
        # Adaptive y-axis based on actual mass range
        y_margin = 0.1 * abs(max_mass - min_mass) if mass_variation > 1e-10 else 0.01
        plt.ylim([min_mass - y_margin, max_mass + y_margin])
        plt.show()

    print("\n--- Hybrid No-Flux Test Finished ---")


if __name__ == "__main__":
    run_hybrid_no_flux_test()
