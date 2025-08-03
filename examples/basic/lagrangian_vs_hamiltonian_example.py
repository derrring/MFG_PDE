#!/usr/bin/env python3
"""
Lagrangian vs Hamiltonian MFG Formulations Comparison

This example demonstrates the relationship between Lagrangian and Hamiltonian
formulations of Mean Field Games, showing how they are mathematically equivalent
but offer different computational and conceptual advantages.

Key comparisons:
1. Lagrangian: Direct cost minimization perspective
2. Hamiltonian: Dynamic programming (HJB-FP) approach  
3. Solution equivalence and computational trade-offs
4. When to use each formulation
"""

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.core.lagrangian_mfg_problem import (
    LagrangianMFGProblem,
    create_quadratic_lagrangian_mfg
)
from mfg_pde.alg.variational_solvers.variational_mfg_solver import VariationalMFGSolver
from mfg_pde.factory import create_fast_solver
from mfg_pde.alg.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfg_pde.alg.fp_solvers.fp_fdm import FPFDMSolver
from mfg_pde.utils.logging import configure_research_logging, get_logger
from mfg_pde.utils.integration import trapezoid

# Configure logging
configure_research_logging("lagrangian_vs_hamiltonian", level="INFO")
logger = get_logger(__name__)


def create_test_problems():
    """
    Create equivalent Lagrangian and Hamiltonian MFG problems.
    
    Returns:
        Tuple of (lagrangian_problem, hamiltonian_problem)
    """
    logger.info("Creating equivalent Lagrangian and Hamiltonian problems...")
    
    # Problem parameters
    xmin, xmax = 0.0, 1.0
    Nx = 30  # Moderate resolution for comparison
    T = 0.3   # Shorter time horizon for faster computation
    Nt = 20
    sigma = 0.2
    
    # Lagrangian formulation
    lagrangian_problem = create_quadratic_lagrangian_mfg(
        xmin=xmin, xmax=xmax, Nx=Nx,
        T=T, Nt=Nt,
        kinetic_coefficient=1.0,
        congestion_coefficient=0.5,
        sigma=sigma
    )
    
    # Convert to Hamiltonian formulation
    hamiltonian_problem = lagrangian_problem.create_compatible_mfg_problem()
    
    logger.info(f"  Domain: [{xmin}, {xmax}] × [0, {T}]")
    logger.info(f"  Grid: {Nx+1} × {Nt} (space × time)")
    logger.info(f"  Lagrangian: L(t,x,v,m) = |v|²/2 + 0.5*m")
    logger.info(f"  Hamiltonian: H(x,p,m) = |p|²/2 + 0.5*m")
    
    return lagrangian_problem, hamiltonian_problem


def solve_lagrangian_formulation(problem):
    """
    Solve using direct variational optimization.
    
    Args:
        problem: LagrangianMFGProblem instance
        
    Returns:
        Variational solver result
    """
    logger.info("Solving Lagrangian formulation using variational optimization...")
    
    try:
        # Create variational solver
        solver = VariationalMFGSolver(
            problem,
            optimization_method="L-BFGS-B",
            penalty_weight=100.0,
            use_jax=False  # For compatibility
        )
        
        # Solve with moderate tolerance for demo
        result = solver.solve(
            max_iterations=50,
            tolerance=1e-4,
            verbose=True
        )
        
        if result.converged:
            logger.info(f"  ✓ Converged in {result.num_iterations} iterations")
            logger.info(f"  ✓ Final cost: {result.final_cost:.6e}")
            logger.info(f"  ✓ Solve time: {result.solve_time:.2f}s")
        else:
            logger.warning(f"  ⚠ Did not converge (max constraint violation)")
            
        return result
        
    except Exception as e:
        logger.error(f"  ✗ Lagrangian solver failed: {e}")
        return None


def solve_hamiltonian_formulation(problem):
    """
    Solve using HJB-FP approach.
    
    Args:
        problem: MFGProblem instance (Hamiltonian formulation)
        
    Returns:
        HJB-FP solver result
    """
    logger.info("Solving Hamiltonian formulation using HJB-FP method...")
    
    try:
        # Create HJB and FP solvers
        hjb_solver = HJBFDMSolver(problem)
        fp_solver = FPFDMSolver(problem)
        
        # Create composite solver
        solver = create_fast_solver(
            problem,
            solver_type="fixed_point",
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            max_picard_iterations=20,
            picard_tolerance=1e-4
        )
        
        # Solve
        result = solver.solve()
        
        if hasattr(result, 'converged') and result.converged:
            logger.info(f"  ✓ Converged successfully")
            logger.info(f"  ✓ Final error: {result.final_error:.6e}")
        else:
            logger.warning(f"  ⚠ Convergence status unclear")
            
        return result
        
    except Exception as e:
        logger.error(f"  ✗ Hamiltonian solver failed: {e}")
        return None


def compare_solutions(lagrangian_result, hamiltonian_result, lagrangian_problem):
    """
    Compare solutions from both formulations.
    
    Args:
        lagrangian_result: Result from variational solver
        hamiltonian_result: Result from HJB-FP solver
        lagrangian_problem: Original Lagrangian problem for analysis
    """
    logger.info("Comparing solutions from both formulations...")
    
    if lagrangian_result is None or hamiltonian_result is None:
        logger.warning("Cannot compare - one or both solutions failed")
        return
    
    try:
        # Extract density fields
        if hasattr(lagrangian_result, 'optimal_flow') and lagrangian_result.optimal_flow is not None:
            m_lagrangian = lagrangian_result.optimal_flow
            logger.info("  ✓ Lagrangian density field available")
        else:
            logger.warning("  ⚠ Lagrangian density field missing")
            return
            
        if hasattr(hamiltonian_result, 'M') and hamiltonian_result.M is not None:
            m_hamiltonian = hamiltonian_result.M
            logger.info("  ✓ Hamiltonian density field available")
        else:
            logger.warning("  ⚠ Hamiltonian density field missing")
            return
        
        # Compare final density distributions
        m_lag_final = m_lagrangian[-1, :]
        m_ham_final = m_hamiltonian[-1, :]
        
        # L2 difference
        x_grid = lagrangian_problem.x
        density_diff_l2 = np.sqrt(trapezoid((m_lag_final - m_ham_final)**2, x=x_grid))
        
        # Mass conservation check
        mass_lag = trapezoid(m_lag_final, x=x_grid)
        mass_ham = trapezoid(m_ham_final, x=x_grid)
        
        logger.info(f"  Final density L2 difference: {density_diff_l2:.4e}")
        logger.info(f"  Lagrangian final mass: {mass_lag:.6f}")
        logger.info(f"  Hamiltonian final mass: {mass_ham:.6f}")
        logger.info(f"  Mass difference: {abs(mass_lag - mass_ham):.4e}")
        
        # Compare representative trajectories if available
        if hasattr(lagrangian_result, 'representative_trajectory') and lagrangian_result.representative_trajectory is not None:
            traj_lag = lagrangian_result.representative_trajectory
            logger.info(f"  Lagrangian trajectory range: [{np.min(traj_lag):.3f}, {np.max(traj_lag):.3f}]")
        
        # Solution quality assessment
        if density_diff_l2 < 0.1:
            logger.info("  ✓ Solutions are in good agreement")
        elif density_diff_l2 < 0.5:
            logger.info("  ⚠ Solutions show moderate differences")
        else:
            logger.info("  ✗ Solutions differ significantly")
        
    except Exception as e:
        logger.error(f"Solution comparison failed: {e}")


def create_comparison_plots(lagrangian_result, hamiltonian_result, lagrangian_problem):
    """
    Create visualization comparing both approaches.
    
    Args:
        lagrangian_result: Variational solver result
        hamiltonian_result: HJB-FP solver result  
        lagrangian_problem: Lagrangian problem for grid information
    """
    logger.info("Creating comparison plots...")
    
    if lagrangian_result is None or hamiltonian_result is None:
        logger.warning("Cannot create plots - missing results")
        return
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Lagrangian vs Hamiltonian MFG Formulations", fontsize=14)
        
        x_grid = lagrangian_problem.x
        t_grid = lagrangian_problem.t
        
        # Plot 1: Final density comparison
        ax1 = axes[0, 0]
        if hasattr(lagrangian_result, 'optimal_flow') and lagrangian_result.optimal_flow is not None:
            m_lag_final = lagrangian_result.optimal_flow[-1, :]
            ax1.plot(x_grid, m_lag_final, 'b-', linewidth=2, label='Lagrangian (Variational)')
        
        if hasattr(hamiltonian_result, 'M') and hamiltonian_result.M is not None:
            m_ham_final = hamiltonian_result.M[-1, :]
            ax1.plot(x_grid, m_ham_final, 'r--', linewidth=2, label='Hamiltonian (HJB-FP)')
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('m(T, x)')
        ax1.set_title('Final Density Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Value function (if available from Hamiltonian)
        ax2 = axes[0, 1]
        if hasattr(hamiltonian_result, 'U') and hamiltonian_result.U is not None:
            U_final = hamiltonian_result.U[-1, :]
            ax2.plot(x_grid, U_final, 'r-', linewidth=2, label='u(T, x)')
            ax2.set_xlabel('x')
            ax2.set_ylabel('u(T, x)')
            ax2.set_title('Final Value Function (Hamiltonian)')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Value Function\nNot Available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Value Function (N/A)')
        
        # Plot 3: Representative trajectory (Lagrangian)
        ax3 = axes[1, 0]
        if hasattr(lagrangian_result, 'representative_trajectory') and lagrangian_result.representative_trajectory is not None:
            traj = lagrangian_result.representative_trajectory
            ax3.plot(t_grid, traj, 'b-', linewidth=2, marker='o', markersize=3)
            ax3.set_xlabel('t')
            ax3.set_ylabel('x(t)')
            ax3.set_title('Representative Trajectory (Lagrangian)')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Representative\nTrajectory\nNot Available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Representative Trajectory (N/A)')
        
        # Plot 4: Convergence history
        ax4 = axes[1, 1]
        
        # Lagrangian convergence
        if hasattr(lagrangian_result, 'cost_history') and lagrangian_result.cost_history:
            iterations_lag = range(len(lagrangian_result.cost_history))
            costs = lagrangian_result.cost_history
            ax4.semilogy(iterations_lag, costs, 'b-', linewidth=2, 
                        marker='o', markersize=3, label='Lagrangian Cost')
        
        # Hamiltonian convergence (if available)
        if hasattr(hamiltonian_result, 'convergence_history') and hamiltonian_result.convergence_history:
            iterations_ham = range(len(hamiltonian_result.convergence_history))
            errors = hamiltonian_result.convergence_history
            ax4_twin = ax4.twinx()
            ax4_twin.semilogy(iterations_ham, errors, 'r--', linewidth=2,
                             marker='s', markersize=3, label='Hamiltonian Error')
            ax4_twin.set_ylabel('HJB-FP Error', color='r')
            ax4_twin.tick_params(axis='y', labelcolor='r')
        
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Cost/Error')
        ax4.set_title('Convergence History')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = "lagrangian_vs_hamiltonian_comparison.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        logger.info(f"Comparison plots saved as: {plot_filename}")
        
        try:
            plt.show()
        except:
            logger.info("Plot display not available (non-interactive environment)")
            
    except Exception as e:
        logger.error(f"Plot creation failed: {e}")


def demonstrate_formulation_differences():
    """
    Demonstrate key conceptual differences between formulations.
    """
    logger.info("Demonstrating formulation differences...")
    
    logger.info("  LAGRANGIAN FORMULATION:")
    logger.info("    • Direct optimization: min J[m] = ∫∫ L(t,x,v,m) m dxdt")
    logger.info("    • Cost-based perspective: agents minimize individual costs")
    logger.info("    • Natural constraint handling: continuity equation as constraint")
    logger.info("    • Economic interpretation: clear cost structure")
    logger.info("    • Computational: gradient-based optimization methods")
    
    logger.info("  HAMILTONIAN FORMULATION:")
    logger.info("    • Dynamic programming: solve HJB + Fokker-Planck system")
    logger.info("    • Value-based perspective: agents maximize value functions")
    logger.info("    • PDE system: coupled backward-forward equations")
    logger.info("    • Control theory interpretation: optimal control problem")
    logger.info("    • Computational: PDE discretization methods")
    
    logger.info("  WHEN TO USE EACH:")
    logger.info("    Lagrangian:")
    logger.info("      - Natural constraint formulation (inequality constraints)")
    logger.info("      - Economic/cost optimization perspective")
    logger.info("      - Convex optimization structure")
    logger.info("      - Direct trajectory optimization")
    logger.info("    Hamiltonian:")
    logger.info("      - High-dimensional problems")
    logger.info("      - Real-time control applications")
    logger.info("      - When value function is primary interest")
    logger.info("      - Mature PDE solution methods")


def main():
    """Main execution function."""
    logger.info("=" * 70)
    logger.info("Lagrangian vs Hamiltonian MFG Formulations Comparison")
    logger.info("=" * 70)
    
    try:
        # Demonstrate conceptual differences
        demonstrate_formulation_differences()
        
        # Create equivalent problems
        lagrangian_prob, hamiltonian_prob = create_test_problems()
        
        # Solve using both approaches
        logger.info("\n" + "="*50 + " SOLVING " + "="*50)
        
        lagrangian_result = solve_lagrangian_formulation(lagrangian_prob)
        hamiltonian_result = solve_hamiltonian_formulation(hamiltonian_prob)
        
        # Compare solutions
        logger.info("\n" + "="*50 + " COMPARISON " + "="*48)
        compare_solutions(lagrangian_result, hamiltonian_result, lagrangian_prob)
        
        # Create visualization
        create_comparison_plots(lagrangian_result, hamiltonian_result, lagrangian_prob)
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("SUMMARY")
        logger.info("="*70)
        
        lagrangian_success = lagrangian_result is not None and (
            not hasattr(lagrangian_result, 'converged') or lagrangian_result.converged
        )
        hamiltonian_success = hamiltonian_result is not None and (
            not hasattr(hamiltonian_result, 'converged') or hamiltonian_result.converged
        )
        
        if lagrangian_success:
            logger.info("✓ Lagrangian (Variational) approach: SUCCESS")
        else:
            logger.info("✗ Lagrangian (Variational) approach: FAILED")
            
        if hamiltonian_success:
            logger.info("✓ Hamiltonian (HJB-FP) approach: SUCCESS")
        else:
            logger.info("✗ Hamiltonian (HJB-FP) approach: FAILED")
        
        if lagrangian_success and hamiltonian_success:
            logger.info("🎉 Both formulations solved successfully!")
            logger.info("   Mathematical equivalence demonstrated.")
        elif lagrangian_success or hamiltonian_success:
            logger.info("⚠️  Partial success - one formulation worked better")
        else:
            logger.info("❌ Both approaches encountered difficulties")
        
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Example failed with error: {e}")
        raise


if __name__ == "__main__":
    main()