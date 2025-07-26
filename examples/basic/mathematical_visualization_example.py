#!/usr/bin/env python3
"""
Mathematical Visualization Example

Demonstrates advanced mathematical visualization capabilities with comprehensive
LaTeX support for professional mathematical analysis and publication-quality
figures. Designed for mathematical researchers working with MFG systems.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import mfg_pde
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mfg_pde import MFGProblem, create_fast_solver
from mfg_pde.utils import configure_logging, get_logger
from mfg_pde.utils.mathematical_visualization import (
    MFGMathematicalVisualizer, quick_hjb_analysis, quick_fp_analysis,
    quick_phase_space_analysis
)


def create_mathematical_test_problem():
    """Create MFG problem with rich mathematical structure for visualization."""
    
    class MathematicalMFGProblem(MFGProblem):
        def __init__(self):
            super().__init__(T=1.0, Nt=40, xmin=0.0, xmax=1.0, Nx=80)
            
        def g(self, x):
            """Terminal cost: $g(x) = \\frac{1}{2}(x - \\frac{1}{2})^2 + \\frac{1}{10}\\sin(4\\pi x)$"""
            return 0.5 * (x - 0.5)**2 + 0.1 * np.sin(4 * np.pi * x)
        
        def rho0(self, x):
            """Initial density: $\\rho_0(x) = \\mathcal{N}(0.3, 0.05) + \\mathcal{N}(0.7, 0.05)$"""
            return (np.exp(-50 * (x - 0.3)**2) + np.exp(-50 * (x - 0.7)**2)) / np.sqrt(2 * np.pi * 0.02)
        
        def f(self, x, u, m):
            """Running cost: $f(x, u, m) = \\frac{\\lambda}{2}u^2 + \\nu m u + \\mu m^2$"""
            return 0.1 * u**2 + 0.05 * m * u + 0.02 * m**2
        
        def sigma(self, x):
            """Diffusion coefficient: $\\sigma(x) = \\sigma_0(1 + \\epsilon \\sin(2\\pi x))$"""
            return 0.2 * (1 + 0.3 * np.sin(2 * np.pi * x))
        
        def H(self, x, p, m):
            """Hamiltonian: $H(x, p, m) = \\frac{1}{2}p^2 + \\nu m p$"""
            return 0.5 * p**2 + 0.05 * m * p
        
        def dH_dm(self, x, p, m):
            """$\\frac{\\partial H}{\\partial m} = \\nu p$"""
            return 0.05 * p
    
    return MathematicalMFGProblem()


def generate_analytical_solution():
    """Generate analytical-like MFG solution for mathematical visualization."""
    logger = get_logger(__name__)
    logger.info("Generating analytical MFG solution with mathematical structure")
    
    # High-resolution grids for smooth visualization
    x_grid = np.linspace(0, 1, 100)
    t_grid = np.linspace(0, 1, 50)
    X, T = np.meshgrid(x_grid, t_grid)
    
    # Value function: $U(x,t) = e^{-\\alpha t}[\\sin(\\pi x) + \\beta \\sin(3\\pi x)] + \\gamma(x - \\frac{1}{2})^2$
    alpha, beta, gamma = 2.0, 0.3, 0.1
    U = (np.exp(-alpha * T) * (np.sin(np.pi * X) + beta * np.sin(3 * np.pi * X)) + 
         gamma * (X - 0.5)**2 * (1 - 0.5 * T))
    
    # Density: $m(x,t) = \\frac{1}{Z(t)}[\\exp(-\\kappa_1(x - \\mu_1(t))^2) + \\exp(-\\kappa_2(x - \\mu_2(t))^2)]$
    kappa1, kappa2 = 20, 25
    mu1 = lambda t: 0.3 + 0.1 * np.sin(np.pi * t)
    mu2 = lambda t: 0.7 - 0.1 * np.sin(np.pi * t)
    
    M = np.zeros_like(X)
    for i, t in enumerate(t_grid):
        m1 = np.exp(-kappa1 * (x_grid - mu1(t))**2)
        m2 = np.exp(-kappa2 * (x_grid - mu2(t))**2)
        m_unnormalized = m1 + m2
        M[i, :] = m_unnormalized / np.trapz(m_unnormalized, x_grid)
    
    return U.T, M.T, x_grid, t_grid


def demo_hjb_mathematical_analysis():
    """Demonstrate HJB equation analysis with LaTeX mathematical notation."""
    print("=" * 70)
    print("HJB MATHEMATICAL ANALYSIS DEMONSTRATION")
    print("=" * 70)
    
    logger = get_logger(__name__)
    U, M, x_grid, t_grid = generate_analytical_solution()
    
    # Compute gradients for comprehensive analysis
    gradients = {}
    gradients['du_dx'] = np.gradient(U, x_grid, axis=0)
    gradients['du_dt'] = np.gradient(U, t_grid, axis=1)
    gradients['d2u_dx2'] = np.gradient(gradients['du_dx'], x_grid, axis=0)
    
    logger.info("Creating comprehensive HJB analysis with LaTeX notation")
    
    output_dir = Path("mathematical_visualization_output")
    output_dir.mkdir(exist_ok=True)
    
    try:
        visualizer = MFGMathematicalVisualizer(backend="matplotlib", enable_latex=True)
        
        hjb_fig = visualizer.plot_hjb_analysis(
            U, x_grid, t_grid, gradients,
            title=r"Hamilton-Jacobi-Bellman Analysis: $-\frac{\partial U}{\partial t} + H\left(x, \frac{\partial U}{\partial x}, m\right) = 0$",
            save_path=output_dir / "hjb_mathematical_analysis.png",
            show=False
        )
        
        logger.info("HJB mathematical analysis completed")
        
        # Quick analysis function
        quick_hjb_fig = quick_hjb_analysis(
            U, x_grid, t_grid,
            save_path=output_dir / "hjb_quick_analysis.png"
        )
        
        logger.info("Quick HJB analysis completed")
        
    except Exception as e:
        logger.error(f"Error in HJB analysis: {e}")
    
    print()


def demo_fokker_planck_analysis():
    """Demonstrate Fokker-Planck equation analysis with mathematical rigor."""
    print("=" * 70)
    print("FOKKER-PLANCK MATHEMATICAL ANALYSIS DEMONSTRATION")  
    print("=" * 70)
    
    logger = get_logger(__name__)
    U, M, x_grid, t_grid = generate_analytical_solution()
    
    # Compute probability flux: $J(x,t) = m(x,t) \alpha^*(x,t) - \frac{\sigma^2}{2} \frac{\partial m}{\partial x}$
    flux = np.zeros_like(M)
    sigma_squared = 0.04  # σ² = 0.2²
    
    for t_idx in range(len(t_grid)):
        # Optimal control: α*(x,t) = -∂H/∂p = -∂U/∂x
        alpha_star = -np.gradient(U[:, t_idx], x_grid)
        dm_dx = np.gradient(M[:, t_idx], x_grid)
        
        # Probability flux
        flux[:, t_idx] = M[:, t_idx] * alpha_star - (sigma_squared / 2) * dm_dx
    
    output_dir = Path("mathematical_visualization_output")
    
    try:
        visualizer = MFGMathematicalVisualizer(backend="matplotlib", enable_latex=True)
        
        fp_fig = visualizer.plot_fokker_planck_analysis(
            M, x_grid, t_grid, flux,
            title=r"Fokker-Planck Analysis: $\frac{\partial m}{\partial t} + \nabla \cdot J = 0$, $J = m\alpha^* - \frac{\sigma^2}{2}\nabla m$",
            save_path=output_dir / "fp_mathematical_analysis.png",
            show=False
        )
        
        logger.info("Fokker-Planck mathematical analysis completed")
        
        # Quick analysis
        quick_fp_fig = quick_fp_analysis(
            M, x_grid, t_grid,
            save_path=output_dir / "fp_quick_analysis.png"
        )
        
        logger.info("Quick FP analysis completed")
        
    except Exception as e:
        logger.error(f"Error in FP analysis: {e}")
    
    print()


def demo_convergence_theory_analysis():
    """Demonstrate convergence theory with rigorous mathematical analysis."""
    print("=" * 70)
    print("CONVERGENCE THEORY ANALYSIS DEMONSTRATION")
    print("=" * 70)
    
    logger = get_logger(__name__)
    
    # Generate realistic convergence data with mathematical structure
    def generate_convergence_sequence(rho, n_iter, noise_level=0.1):
        """Generate convergence sequence: $e_k = e_0 \\rho^k (1 + \\xi_k)$ where $\\xi_k \\sim \\mathcal{N}(0, \\sigma^2)$"""
        errors = []
        e0 = 1.0
        for k in range(n_iter):
            noise = noise_level * np.random.normal()
            error = e0 * (rho ** k) * (1 + noise)
            errors.append(max(error, 1e-12))  # Avoid underflow
        return errors
    
    np.random.seed(42)  # Reproducible results
    
    convergence_data = {
        r'$\|U^k - U^*\|_{L^2(\Omega)}$': generate_convergence_sequence(0.82, 35, 0.08),
        r'$\|m^k - m^*\|_{L^1(\Omega)}$': generate_convergence_sequence(0.85, 35, 0.06),
        r'$\|\text{HJB residual}\|_{L^{\infty}}$': generate_convergence_sequence(0.78, 35, 0.12),
        r'$\|\text{FP residual}\|_{L^2}$': generate_convergence_sequence(0.80, 35, 0.10)
    }
    
    theoretical_rates = {
        r'$\|U^k - U^*\|_{L^2(\Omega)}$': 0.82,
        r'$\|m^k - m^*\|_{L^1(\Omega)}$': 0.85,
        r'$\|\text{HJB residual}\|_{L^{\infty}}$': 0.78,
        r'$\|\text{FP residual}\|_{L^2}$': 0.80
    }
    
    output_dir = Path("mathematical_visualization_output")
    
    try:
        visualizer = MFGMathematicalVisualizer(backend="matplotlib", enable_latex=True)
        
        conv_fig = visualizer.plot_convergence_theory(
            convergence_data, theoretical_rates,
            title=r"Convergence Theory: $\|e^k\| \leq C \rho^k \|e^0\|$ with rates $\rho \in (0,1)$",
            save_path=output_dir / "convergence_theory_analysis.png",
            show=False
        )
        
        logger.info("Convergence theory analysis completed")
        
    except Exception as e:
        logger.error(f"Error in convergence analysis: {e}")
    
    print()


def demo_phase_space_analysis():
    """Demonstrate phase space analysis of the MFG Hamiltonian system."""
    print("=" * 70)
    print("PHASE SPACE ANALYSIS DEMONSTRATION")
    print("=" * 70)
    
    logger = get_logger(__name__)
    U, M, x_grid, t_grid = generate_analytical_solution()
    
    output_dir = Path("mathematical_visualization_output")
    
    try:
        visualizer = MFGMathematicalVisualizer(backend="matplotlib", enable_latex=True)
        
        phase_fig = visualizer.plot_phase_space_analysis(
            U, M, x_grid, t_grid,
            title=r"Phase Space Analysis: Hamiltonian System $\dot{x} = \nabla_p H$, $\dot{p} = -\nabla_x H$",
            save_path=output_dir / "phase_space_analysis.png",
            show=False
        )
        
        logger.info("Phase space analysis completed")
        
        # Quick phase space analysis
        quick_phase_fig = quick_phase_space_analysis(
            U, M, x_grid, t_grid,
            save_path=output_dir / "quick_phase_space.png"
        )
        
        logger.info("Quick phase space analysis completed")
        
    except Exception as e:
        logger.error(f"Error in phase space analysis: {e}")
    
    print()


def demo_real_solver_mathematical_analysis():
    """Demonstrate mathematical analysis with real MFG solver."""
    print("=" * 70)
    print("REAL SOLVER MATHEMATICAL ANALYSIS DEMONSTRATION")
    print("=" * 70)
    
    logger = get_logger(__name__)
    
    try:
        # Create mathematical problem
        problem = create_mathematical_test_problem()
        x_coords = np.linspace(problem.xmin, problem.xmax, problem.Nx)
        collocation_points = x_coords.reshape(-1, 1)
        
        logger.info("Solving MFG problem for mathematical analysis")
        
        # Create and solve
        solver = create_fast_solver(
            problem=problem,
            solver_type="monitored_particle",
            collocation_points=collocation_points,
            num_particles=800
        )
        
        result = solver.solve(verbose=False)
        
        # Extract solution
        if hasattr(result, 'U') and hasattr(result, 'M'):
            U, M = result.U, result.M
        else:
            U, M = result[0], result[1]
        
        t_grid = np.linspace(0, problem.T, problem.Nt)
        
        output_dir = Path("mathematical_visualization_output")
        
        # Mathematical analysis of real solution
        visualizer = MFGMathematicalVisualizer(backend="matplotlib", enable_latex=True)
        
        # HJB analysis
        hjb_fig = visualizer.plot_hjb_analysis(
            U, x_coords, t_grid,
            title=r"Real Solver HJB Analysis: $-u_t + \frac{1}{2}|\nabla u|^2 + f(t, x, u, m) = 0$",
            save_path=output_dir / "real_solver_hjb.png",
            show=False
        )
        
        # FP analysis  
        fp_fig = visualizer.plot_fokker_planck_analysis(
            M, x_coords, t_grid,
            title=r"Real Solver FP Analysis: $\frac{\partial m}{\partial t} - \nabla \cdot(m \nabla u) - \frac{\sigma^2}{2}\Delta m = 0$",
            save_path=output_dir / "real_solver_fp.png",
            show=False
        )
        
        # Phase space analysis
        phase_fig = visualizer.plot_phase_space_analysis(
            U, M, x_coords, t_grid,
            title=r"Real Solver Phase Space: $(u, m)$ Dynamics",
            save_path=output_dir / "real_solver_phase.png",
            show=False
        )
        
        logger.info("Real solver mathematical analysis completed")
        
    except Exception as e:
        logger.error(f"Error in real solver analysis: {e}")
        logger.debug("This may be expected if solver dependencies are missing")
    
    print()


def run_comprehensive_mathematical_demo():
    """Run comprehensive mathematical visualization demonstration."""
    print("[DEMO] MFG_PDE MATHEMATICAL VISUALIZATION MODULE")
    print("=" * 80)
    print("Advanced mathematical visualization with comprehensive LaTeX support")
    print("for professional mathematical analysis and publication-quality figures.")
    print("Designed for mathematical researchers working with MFG systems.")
    print("=" * 80)
    print()
    
    # Configure logging
    configure_logging(level="INFO", use_colors=True)
    logger = get_logger(__name__)
    
    # Create output directory
    output_dir = Path("mathematical_visualization_output")
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    try:
        demo_hjb_mathematical_analysis()
        demo_fokker_planck_analysis()
        demo_convergence_theory_analysis()
        demo_phase_space_analysis()
        demo_real_solver_mathematical_analysis()
        
        print("=" * 80)
        print("[COMPLETED] MATHEMATICAL VISUALIZATION DEMONSTRATION")
        print("=" * 80)
        print()
        print("Mathematical Features Demonstrated:")
        print("• Hamilton-Jacobi-Bellman equation analysis with LaTeX notation")
        print("• Fokker-Planck equation analysis with probability flux computation")
        print("• Rigorous convergence theory with rate estimation") 
        print("• Phase space analysis of Hamiltonian MFG systems")
        print("• Professional LaTeX mathematical expressions throughout")
        print("• Publication-quality matplotlib figures with mathematical precision")
        print()
        print("LaTeX Mathematical Expressions Used:")
        print("• PDE notation: ∂U/∂t, ∇·J, Δm")
        print("• Norms: ‖U‖_{L²}, ‖m‖_{L¹}, ‖residual‖_{L^∞}")
        print("• Mathematical symbols: α*, σ², ρ, Ω, ∇_p H")
        print("• Statistical notation: 𝔼[X_t], Var[X_t], 𝒩(μ,σ²)")
        print()
        print("Output Files:")
        if output_dir.exists():
            output_files = list(output_dir.glob("*.png"))
            for file_path in sorted(output_files):
                print(f"  {file_path}")
        print()
        print("Requirements:")
        print("  For LaTeX rendering: LaTeX installation (optional)")
        print("  For mathematical plots: matplotlib with mathtext")
        print("  For professional typography: Computer Modern fonts")
        print()
        
    except Exception as e:
        logger.error(f"Error in mathematical demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_comprehensive_mathematical_demo()