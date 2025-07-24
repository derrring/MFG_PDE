#!/usr/bin/env python3
"""
Enhanced Particle Collocation Solver with Advanced Convergence Monitoring

This solver extends the standard ParticleCollocationSolver with robust convergence
criteria specifically designed for particle-based MFG methods, addressing:
- Statistical noise in particle-based distributions
- Oscillatory behavior in coupled HJB-FP systems
- Multi-criteria convergence assessment
"""

import numpy as np
from typing import TYPE_CHECKING, Optional, Dict, Tuple, Any
from .particle_collocation_solver import ParticleCollocationSolver
from ..utils.convergence import AdvancedConvergenceMonitor, create_default_monitor

if TYPE_CHECKING:
    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.core.boundaries import BoundaryConditions


class EnhancedParticleCollocationSolver(ParticleCollocationSolver):
    """
    Particle Collocation Solver with advanced convergence monitoring.
    
    Features:
    - Robust convergence criteria for noisy particle distributions
    - Oscillation stabilization detection for value functions
    - Wasserstein distance-based distribution comparison
    - Multi-criteria convergence validation
    - Detailed convergence diagnostics and visualization
    """
    
    def __init__(self,
                 problem: "MFGProblem",
                 collocation_points: np.ndarray,
                 num_particles: int = 5000,
                 delta: float = 0.1,
                 taylor_order: int = 2,
                 weight_function: str = "wendland",
                 NiterNewton: int = 30,
                 l2errBoundNewton: float = 1e-4,
                 kde_bandwidth: str = "scott",
                 normalize_kde_output: bool = False,
                 boundary_indices: Optional[np.ndarray] = None,
                 boundary_conditions: Optional["BoundaryConditions"] = None,
                 use_monotone_constraints: bool = False,
                 convergence_monitor: Optional[AdvancedConvergenceMonitor] = None,
                 **convergence_kwargs):
        """
        Initialize enhanced particle collocation solver.
        
        Args:
            convergence_monitor: Pre-configured convergence monitor
            **convergence_kwargs: Parameters for default monitor if none provided
            (All other args same as ParticleCollocationSolver)
        """
        super().__init__(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=num_particles,
            delta=delta,
            taylor_order=taylor_order,
            weight_function=weight_function,
            NiterNewton=NiterNewton,
            l2errBoundNewton=l2errBoundNewton,
            kde_bandwidth=kde_bandwidth,
            normalize_kde_output=normalize_kde_output,
            boundary_indices=boundary_indices,
            boundary_conditions=boundary_conditions,
            use_monotone_constraints=use_monotone_constraints
        )
        
        # Initialize convergence monitoring
        if convergence_monitor is not None:
            self.convergence_monitor = convergence_monitor
        else:
            self.convergence_monitor = create_default_monitor(**convergence_kwargs)
        
        # Enhanced convergence tracking
        self.detailed_convergence_history = []
        self.use_advanced_convergence = True
        
    def solve(self, 
              Niter: int = 20, 
              l2errBound: float = 1e-3,
              verbose: bool = True,
              plot_convergence: bool = False,
              save_convergence_plot: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Solve MFG system with enhanced convergence monitoring.
        
        Args:
            Niter: Maximum iterations
            l2errBound: Legacy L2 error bound (kept for compatibility)
            verbose: Print convergence progress
            plot_convergence: Generate convergence plots
            save_convergence_plot: Path to save convergence plot
            
        Returns:
            (U, M, enhanced_info) where enhanced_info includes detailed convergence data
        """
        if verbose:
            print("=" * 80)
            print("ENHANCED PARTICLE COLLOCATION SOLVER")
            print("=" * 80)
            print("Using advanced convergence criteria:")
            print(f"  - Wasserstein tolerance: {self.convergence_monitor.wasserstein_tol}")
            print(f"  - Value function magnitude tolerance: {self.convergence_monitor.u_magnitude_tol}")
            print(f"  - Stability tolerance: {self.convergence_monitor.u_stability_tol}")
            print()
        
        # Initialize
        problem = self.problem
        T, Nt = problem.T, problem.Nt
        xmin, xmax, Nx = problem.xmin, problem.xmax, problem.Nx
        
        dt = T / Nt
        times = np.linspace(0, T, Nt + 1)
        x_grid = np.linspace(xmin, xmax, Nx)
        
        # Initialize solution arrays
        U = np.zeros((Nt + 1, Nx))
        M = np.zeros((Nt + 1, Nx))
        
        # Set terminal condition for U
        U[-1, :] = problem.g(x_grid)
        
        # Set initial condition for M
        M[0, :] = problem.rho0(x_grid)
        
        # Fixed point iteration with advanced convergence monitoring
        U_prev = U.copy()
        converged = False
        iteration_info = []
        
        for iteration in range(Niter):
            if verbose:
                print(f"Iteration {iteration + 1}/{Niter}")
            
            # Store previous solution for convergence analysis
            U_old = U.copy()
            M_old = M.copy()
            
            # HJB step: solve backwards in time
            for n in range(Nt - 1, -1, -1):
                t = times[n]
                
                # Get current distribution for this time step
                current_m = M[n, :]
                
                # Solve HJB equation at this time step
                if n == Nt - 1:
                    U[n, :] = U[n + 1, :]  # Terminal condition
                else:
                    # Use HJB solver to compute U[n, :] given current_m and U[n+1, :]
                    self.hjb_solver.problem.current_time = t
                    self.hjb_solver.problem.current_density = current_m
                    
                    # Approximate spatial derivatives using GFDM
                    u_next = U[n + 1, :]
                    U[n, :] = self.hjb_solver.backward_step(u_next, current_m, dt)
            
            # FP step: solve forwards in time
            for n in range(Nt):
                t = times[n]
                
                # Get current value function for this time step
                current_u = U[n, :]
                
                # Solve FP equation at this time step
                if n == 0:
                    continue  # Initial condition already set
                else:
                    # Use FP solver to compute M[n, :] given current_u and M[n-1, :]
                    prev_m = M[n - 1, :]
                    M[n, :] = self.fp_solver.forward_step(prev_m, current_u, dt)
            
            # Advanced convergence analysis
            if iteration > 0:
                # Compute representative L2 errors
                u_l2_error = np.linalg.norm(U - U_old)
                m_l2_error = np.linalg.norm(M - M_old)
                
                # Use final time step for distribution convergence analysis
                convergence_diagnostics = self.convergence_monitor.update(
                    u_current=U[-1, :],  # Terminal value function
                    u_previous=U_prev[-1, :],
                    m_current=M[-1, :],   # Final distribution
                    x_grid=x_grid
                )
                
                # Enhanced iteration info
                iter_info = {
                    'iteration': iteration + 1,
                    'u_l2_error': u_l2_error,
                    'm_l2_error': m_l2_error,
                    'legacy_converged': u_l2_error < l2errBound,
                    'advanced_converged': convergence_diagnostics['converged'],
                    'convergence_diagnostics': convergence_diagnostics
                }
                
                iteration_info.append(iter_info)
                self.detailed_convergence_history.append(convergence_diagnostics)
                
                if verbose:
                    print(f"  L2 errors: U={u_l2_error:.2e}, M={m_l2_error:.2e}")
                    if 'wasserstein_distance' in convergence_diagnostics:
                        print(f"  Wasserstein distance: {convergence_diagnostics['wasserstein_distance']:.2e}")
                    if 'u_oscillation' in convergence_diagnostics:
                        osc = convergence_diagnostics['u_oscillation']
                        if 'mean_error' in osc:
                            print(f"  U oscillation: mean={osc['mean_error']:.2e}, std={osc['std_error']:.2e}")
                    
                    # Show convergence criteria status
                    criteria = convergence_diagnostics['convergence_criteria']
                    status_symbols = {True: "PASS", False: "FAIL"}
                    print(f"  Convergence: W={status_symbols.get(criteria.get('wasserstein', False))} " +
                          f"U_stab={status_symbols.get(criteria.get('u_stabilized', False))} " +
                          f"Overall={status_symbols.get(convergence_diagnostics['converged'])}")
                
                # Check for convergence
                if self.use_advanced_convergence:
                    if convergence_diagnostics['converged']:
                        if verbose:
                            print(f"\nAdvanced convergence achieved at iteration {iteration + 1}")
                        converged = True
                        break
                else:
                    # Fall back to legacy convergence
                    if u_l2_error < l2errBound:
                        if verbose:
                            print(f"\nLegacy convergence achieved at iteration {iteration + 1}")
                        converged = True
                        break
            
            U_prev = U.copy()
            
            if verbose:
                print()
        
        # Final convergence summary
        convergence_summary = self.convergence_monitor.get_convergence_summary()
        
        if verbose:
            print("=" * 80)
            print("CONVERGENCE SUMMARY")
            print("=" * 80)
            if converged:
                print(f"Converged in {iteration + 1} iterations")
            else:
                print(f"Did not converge in {Niter} iterations")
            
            print(f"Final L2 error: {convergence_summary.get('final_u_error', 'N/A'):.2e}")
            if convergence_summary.get('final_wasserstein') is not None:
                print(f"Final Wasserstein distance: {convergence_summary['final_wasserstein']:.2e}")
            print()
        
        # Generate convergence plots if requested
        if plot_convergence or save_convergence_plot:
            self.convergence_monitor.plot_convergence_history(save_convergence_plot)
        
        # Prepare enhanced info dictionary
        enhanced_info = {
            'converged': converged,
            'iterations': iteration + 1 if converged else Niter,
            'convergence_summary': convergence_summary,
            'iteration_info': iteration_info,
            'detailed_convergence': self.detailed_convergence_history,
            'method': 'Enhanced Particle Collocation',
            'convergence_criteria': 'Advanced (Wasserstein + Oscillation Stabilization)'
        }
        
        return U, M, enhanced_info
    
    def get_convergence_diagnostics(self) -> Dict[str, Any]:
        """
        Get detailed convergence diagnostics from the last solve.
        
        Returns:
            Dictionary with convergence analysis
        """
        if not self.detailed_convergence_history:
            return {'status': 'no_solve_history'}
        
        summary = self.convergence_monitor.get_convergence_summary()
        
        # Extract key metrics over time
        u_errors = [d['u_l2_error'] for d in self.detailed_convergence_history]
        wasserstein_dists = [d.get('wasserstein_distance', np.nan) 
                           for d in self.detailed_convergence_history]
        
        return {
            'summary': summary,
            'u_error_history': u_errors,
            'wasserstein_history': [w for w in wasserstein_dists if not np.isnan(w)],
            'convergence_achieved': summary.get('converged', False),
            'convergence_iteration': summary.get('convergence_iteration'),
            'total_iterations': len(self.detailed_convergence_history)
        }
    
    def reset_convergence_monitor(self, **kwargs):
        """Reset convergence monitor for new solve."""
        self.convergence_monitor = create_default_monitor(**kwargs)
        self.detailed_convergence_history = []


# Convenience function for creating enhanced solver
def create_enhanced_solver(problem: "MFGProblem", 
                         collocation_points: np.ndarray,
                         **kwargs) -> EnhancedParticleCollocationSolver:
    """
    Create enhanced particle collocation solver with optimized defaults.
    
    Args:
        problem: MFG problem instance
        collocation_points: Spatial collocation points
        **kwargs: Additional solver parameters
        
    Returns:
        Configured EnhancedParticleCollocationSolver
    """
    defaults = {
        'num_particles': 5000,
        'delta': 0.4,
        'taylor_order': 2,
        'weight_function': 'wendland',
        'use_monotone_constraints': True,
        'wasserstein_tol': 1e-4,
        'u_magnitude_tol': 1e-3,
        'u_stability_tol': 1e-4,
        'history_length': 10
    }
    defaults.update(kwargs)
    
    return EnhancedParticleCollocationSolver(problem, collocation_points, **defaults)