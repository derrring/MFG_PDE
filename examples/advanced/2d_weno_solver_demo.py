#!/usr/bin/env python3
"""
2D WENO Solver Demonstration

This example demonstrates the 2D WENO solver capabilities for solving
Hamilton-Jacobi-Bellman equations in two spatial dimensions using
the complete WENO family of schemes.

Features Demonstrated:
1. 2D Domain setup with rectangular geometry
2. High-dimensional MFG problem configuration
3. All WENO variants in 2D: WENO5, WENO-Z, WENO-M, WENO-JS
4. Dimensional splitting approach for efficiency
5. 2D boundary condition handling
6. Performance comparison and visualization
7. Integration with 2D boundary condition framework

Mathematical Problem:
    ∂u/∂t + H(x, y, ∇u, ρ(t,x,y)) - (σ²/2)Δu = 0

Where H(x, y, p_x, p_y, ρ) = (1/2)(p_x² + p_y²) + interaction terms

This demonstration shows the power of extending 1D WENO methods to
multi-dimensional Mean Field Games using dimensional splitting.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import time as timer

# MFG_PDE imports
from mfg_pde.geometry import Domain2D
from mfg_pde.core.highdim_mfg_problem import HighDimMFGProblem
from mfg_pde.alg.hjb_solvers import HJBWenoSolver
from mfg_pde.alg.hjb_solvers.hjb_weno import WenoVariant
from mfg_pde.utils.logging import get_logger, configure_research_logging


class Demo2DMFGProblem(HighDimMFGProblem):
    """
    Demo 2D MFG problem with analytical properties for WENO validation.

    This problem includes:
    - Smooth initial conditions with known analytical properties
    - Controllable interaction terms for testing different WENO variants
    - Boundary conditions compatible with WENO schemes
    """

    def __init__(
        self,
        domain: Domain2D,
        time_domain: tuple[float, int] = (1.0, 100),
        diffusion_coeff: float = 0.1,
        interaction_strength: float = 1.0,
    ):
        """Initialize 2D demo MFG problem."""
        super().__init__(domain, time_domain, diffusion_coeff, dimension=2)
        self.interaction_strength = interaction_strength
        self.domain_2d = domain

    def create_initial_conditions(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Create initial conditions for 2D MFG system.

        Returns:
            (u0, rho0): Initial value function and density
        """
        # Get 2D grid from domain
        if hasattr(self.domain_2d, 'get_computational_grid'):
            grid = self.domain_2d.get_computational_grid()
            X, Y = grid['X'], grid['Y']
        else:
            # Fallback: create regular grid
            xmin, xmax, ymin, ymax = self.domain_2d.bounds_rect
            x = np.linspace(xmin, xmax, 64)
            y = np.linspace(ymin, ymax, 64)
            X, Y = np.meshgrid(x, y, indexing='ij')

        # Initial value function: smooth Gaussian-like function
        # u0(x,y) = exp(-α((x-x_c)² + (y-y_c)²))
        x_center, y_center = 0.3, 0.7
        alpha = 10.0
        u0 = np.exp(-alpha * ((X - x_center)**2 + (Y - y_center)**2))

        # Initial density: different center to create interesting dynamics
        # rho0(x,y) = normalized Gaussian
        rho_center_x, rho_center_y = 0.7, 0.3
        beta = 8.0
        rho0 = np.exp(-beta * ((X - rho_center_x)**2 + (Y - rho_center_y)**2))

        # Normalize density
        dx = X[1, 0] - X[0, 0]
        dy = Y[0, 1] - Y[0, 0]
        rho0 = rho0 / (np.sum(rho0) * dx * dy)

        return u0, rho0

    def create_density_trajectory(self, u0: np.ndarray, rho0: np.ndarray) -> np.ndarray:
        """
        Create simplified density trajectory for demo purposes.

        In a full MFG solver, this would be computed by solving the
        Fokker-Planck equation. For this demo, we create a plausible
        trajectory for testing the HJB solver.
        """
        nt = self.Nt
        nx, ny = u0.shape

        # Create time-evolving density with simple diffusion-like behavior
        rho_trajectory = np.zeros((nt + 1, nx, ny))
        rho_trajectory[0] = rho0.copy()

        # Simple time evolution: gradual spreading and drift
        dt = self.T / nt
        diffusion_rate = 0.1

        for t_idx in range(1, nt + 1):
            # Simple diffusion with slight drift
            prev_rho = rho_trajectory[t_idx - 1]

            # Add small random perturbation to simulate realistic evolution
            noise_scale = 0.01 * np.exp(-2 * t_idx * dt)  # Decreasing noise
            noise = noise_scale * np.random.randn(nx, ny)

            # Simple evolution rule with diffusion
            new_rho = prev_rho * (1 - diffusion_rate * dt) + noise
            new_rho = np.maximum(new_rho, 0)  # Ensure positivity

            # Renormalize
            total_mass = np.sum(new_rho)
            if total_mass > 0:
                new_rho = new_rho / total_mass

            rho_trajectory[t_idx] = new_rho

        return rho_trajectory


def setup_2d_demo_domain() -> Domain2D:
    """Setup 2D domain for WENO demonstration."""
    # Create rectangular domain [0,1] × [0,1]
    domain = Domain2D(
        domain_type="rectangle",
        bounds=(0.0, 1.0, 0.0, 1.0),  # (xmin, xmax, ymin, ymax)
        mesh_size=0.02,  # Fine mesh for high-order accuracy
    )

    # Add computational grid method if not present
    if not hasattr(domain, 'get_computational_grid'):
        def get_computational_grid():
            xmin, xmax, ymin, ymax = domain.bounds_rect
            nx, ny = 64, 64  # High resolution for WENO

            x_coords = np.linspace(xmin, xmax, nx)
            y_coords = np.linspace(ymin, ymax, ny)
            X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

            dx = x_coords[1] - x_coords[0]
            dy = y_coords[1] - y_coords[0]

            return {
                'nx': nx, 'ny': ny,
                'dx': dx, 'dy': dy,
                'x_coords': x_coords, 'y_coords': y_coords,
                'X': X, 'Y': Y
            }

        # Attach method to domain
        domain.get_computational_grid = get_computational_grid

    return domain


def run_2d_weno_variants_comparison():
    """
    Run comprehensive comparison of 2D WENO variants.

    This function tests all available WENO schemes on the same 2D problem
    and compares their performance and accuracy characteristics.
    """
    logger = get_logger(__name__)
    logger.info("Starting 2D WENO variants comparison")

    # Setup problem
    domain = setup_2d_demo_domain()
    problem = Demo2DMFGProblem(
        domain=domain,
        time_domain=(0.5, 50),  # Shorter time for demonstration
        diffusion_coeff=0.05,
        interaction_strength=0.5
    )

    # Create initial conditions
    u0, rho0 = problem.create_initial_conditions()
    rho_trajectory = problem.create_density_trajectory(u0, rho0)

    logger.info(f"Problem setup: {u0.shape} grid, {problem.Nt} time steps")

    # WENO variants to test
    weno_variants: list[WenoVariant] = ["weno5", "weno-z", "weno-m", "weno-js"]
    results = {}

    # Test each WENO variant
    for variant in weno_variants:
        logger.info(f"Testing 2D {variant.upper()} solver...")

        try:
            # Create solver with variant-specific parameters
            solver_params = _get_variant_parameters(variant)

            solver = HJBWenoSolver(
                problem=problem,
                weno_variant=variant,
                **solver_params
            )

            # Solve and time the computation
            start_time = timer.time()
            solution = solver.solve_hjb_system(
                u_initial=u0,
                rho_trajectory=rho_trajectory,
                max_iterations=problem.Nt,
                tolerance=1e-8
            )
            solve_time = timer.time() - start_time

            # Store results
            results[variant] = {
                'solution': solution,
                'solve_time': solve_time,
                'final_solution': solution[-1],
                'solver': solver
            }

            logger.info(f"2D {variant.upper()}: {solve_time:.2f}s, "
                       f"final max: {np.max(solution[-1]):.4f}, "
                       f"final min: {np.min(solution[-1]):.4f}")

        except Exception as e:
            logger.error(f"Error with 2D {variant}: {e}")
            continue

    if not results:
        logger.error("No WENO variants completed successfully")
        return

    # Analyze and visualize results
    _analyze_2d_weno_results(results, u0, rho0, domain)

    return results


def _get_variant_parameters(variant: WenoVariant) -> dict:
    """Get optimal parameters for each WENO variant in 2D."""
    base_params = {
        'cfl_number': 0.2,  # Conservative for 2D
        'diffusion_stability_factor': 0.1,  # Conservative for 2D
        'weno_epsilon': 1e-6,
        'time_integration': 'tvd_rk3',
        'splitting_method': 'strang'
    }

    variant_specific = {
        'weno5': {},  # Use base parameters
        'weno-z': {
            'weno_z_parameter': 2.0,  # Enhanced resolution
            'cfl_number': 0.15  # Slightly more conservative
        },
        'weno-m': {
            'weno_m_parameter': 1.5,  # Better critical point handling
            'cfl_number': 0.25  # Can be slightly less conservative
        },
        'weno-js': {
            'cfl_number': 0.1,  # Most conservative for stability
        }
    }

    params = base_params.copy()
    params.update(variant_specific.get(variant, {}))
    return params


def _analyze_2d_weno_results(results: dict, u0: np.ndarray, rho0: np.ndarray, domain: Domain2D):
    """Analyze and visualize 2D WENO comparison results."""
    logger = get_logger(__name__)

    # Get grid for plotting
    grid = domain.get_computational_grid()
    X, Y = grid['X'], grid['Y']

    # Create comprehensive comparison plots
    fig = plt.figure(figsize=(16, 12))

    # 1. Initial conditions
    plt.subplot(3, 4, 1)
    plt.contourf(X, Y, u0, levels=20, cmap='viridis')
    plt.colorbar(label='u₀(x,y)')
    plt.title('Initial Value Function')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(3, 4, 2)
    plt.contourf(X, Y, rho0, levels=20, cmap='plasma')
    plt.colorbar(label='ρ₀(x,y)')
    plt.title('Initial Density')
    plt.xlabel('x')
    plt.ylabel('y')

    # 2. Final solutions for each variant
    plot_idx = 3
    for variant, result in results.items():
        if plot_idx > 6:  # Limit to 4 variants
            break

        plt.subplot(3, 4, plot_idx)
        final_solution = result['final_solution']
        plt.contourf(X, Y, final_solution, levels=20, cmap='viridis')
        plt.colorbar(label=f'u(T,x,y)')
        plt.title(f'2D {variant.upper()} Final Solution')
        plt.xlabel('x')
        plt.ylabel('y')
        plot_idx += 1

    # 3. Cross-sectional comparison at y = 0.5
    plt.subplot(3, 4, 7)
    y_idx = len(grid['y_coords']) // 2  # Middle y-slice
    x_coords = grid['x_coords']

    plt.plot(x_coords, u0[:, y_idx], 'k--', label='Initial', alpha=0.7)
    for variant, result in results.items():
        final_solution = result['final_solution']
        plt.plot(x_coords, final_solution[:, y_idx],
                label=f'2D {variant.upper()}', linewidth=2)

    plt.xlabel('x')
    plt.ylabel('u(T, x, 0.5)')
    plt.title('Cross-section at y=0.5')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. Cross-sectional comparison at x = 0.5
    plt.subplot(3, 4, 8)
    x_idx = len(grid['x_coords']) // 2  # Middle x-slice
    y_coords = grid['y_coords']

    plt.plot(y_coords, u0[x_idx, :], 'k--', label='Initial', alpha=0.7)
    for variant, result in results.items():
        final_solution = result['final_solution']
        plt.plot(y_coords, final_solution[x_idx, :],
                label=f'2D {variant.upper()}', linewidth=2)

    plt.xlabel('y')
    plt.ylabel('u(T, 0.5, y)')
    plt.title('Cross-section at x=0.5')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 5. Performance comparison
    plt.subplot(3, 4, 9)
    variants = list(results.keys())
    solve_times = [results[v]['solve_time'] for v in variants]

    plt.bar(variants, solve_times, alpha=0.7, color=['blue', 'green', 'red', 'orange'][:len(variants)])
    plt.ylabel('Solve Time (s)')
    plt.title('2D WENO Performance Comparison')
    plt.xticks(rotation=45)

    # 6. Solution statistics
    plt.subplot(3, 4, 10)
    max_values = [np.max(results[v]['final_solution']) for v in variants]
    min_values = [np.min(results[v]['final_solution']) for v in variants]

    x_pos = np.arange(len(variants))
    plt.bar(x_pos - 0.2, max_values, 0.4, label='Max', alpha=0.7)
    plt.bar(x_pos + 0.2, min_values, 0.4, label='Min', alpha=0.7)
    plt.xticks(x_pos, variants, rotation=45)
    plt.ylabel('Solution Value')
    plt.title('Final Solution Statistics')
    plt.legend()

    # 7. L2 norm comparison (relative to first variant)
    plt.subplot(3, 4, 11)
    reference_solution = list(results.values())[0]['final_solution']
    l2_diffs = []

    for variant in variants:
        solution = results[variant]['final_solution']
        l2_diff = np.sqrt(np.sum((solution - reference_solution)**2))
        l2_diffs.append(l2_diff)

    plt.bar(variants, l2_diffs, alpha=0.7)
    plt.ylabel('L2 Difference from Reference')
    plt.title(f'Solution Differences (ref: {variants[0].upper()})')
    plt.xticks(rotation=45)

    # 8. Conservation properties
    plt.subplot(3, 4, 12)
    # Check mass conservation if density evolution was more sophisticated
    conservation_scores = []
    for variant in variants:
        # Placeholder conservation metric
        solution = results[variant]['final_solution']
        conservation_scores.append(np.sum(solution))

    plt.bar(variants, conservation_scores, alpha=0.7)
    plt.ylabel('Conservation Score')
    plt.title('Conservation Properties')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('2d_weno_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print summary
    logger.info("\n2D WENO Variants Performance Summary:")
    logger.info("=" * 50)
    for variant, result in results.items():
        solution = result['final_solution']
        logger.info(f"{variant.upper():>8}: {result['solve_time']:6.2f}s | "
                   f"Range: [{np.min(solution):8.4f}, {np.max(solution):8.4f}] | "
                   f"Mean: {np.mean(solution):8.4f}")


def demonstrate_2d_boundary_conditions():
    """Demonstrate 2D WENO solver with various boundary conditions."""
    logger = get_logger(__name__)
    logger.info("Demonstrating 2D WENO with boundary conditions")

    # This would integrate with the 2D boundary condition framework
    # created in the previous work

    domain = setup_2d_demo_domain()

    # Example: Periodic boundary conditions in both directions
    from mfg_pde.geometry.boundary_conditions_2d import (
        create_rectangle_boundary_conditions
    )

    bc_manager = create_rectangle_boundary_conditions(
        domain_bounds=(0.0, 1.0, 0.0, 1.0),
        condition_type="periodic_both"
    )

    logger.info(f"Created 2D boundary condition manager with {len(bc_manager.conditions)} conditions")

    # The 2D WENO solver would integrate with this BC framework
    # for proper boundary handling


def main():
    """Main demonstration function."""
    # Configure logging for research session
    configure_research_logging("2d_weno_demo", level="INFO")
    logger = get_logger(__name__)

    logger.info("🚀 Starting 2D WENO Solver Demonstration")
    logger.info("=" * 60)

    try:
        # Run comprehensive WENO variants comparison
        results = run_2d_weno_variants_comparison()

        if results:
            logger.info(f"✅ Successfully tested {len(results)} WENO variants in 2D")

            # Demonstrate boundary condition integration
            demonstrate_2d_boundary_conditions()

            logger.info("🎯 2D WENO demonstration completed successfully")
            logger.info("📊 Check '2d_weno_comparison.png' for detailed results")

        else:
            logger.error("❌ No WENO variants completed successfully")

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()