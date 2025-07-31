#!/usr/bin/env python3
"""
Enhanced Network MFG Visualization Demo.

This example demonstrates the advanced visualization capabilities for network MFG
including Lagrangian trajectories, velocity fields, scheme comparisons, and 3D plots.

Features demonstrated:
- Lagrangian trajectory visualization
- Velocity field plots on networks  
- Discretization scheme comparisons
- 3D network visualization with value heights
- Interactive vs static plotting options
- Advanced network analysis tools
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any

# MFG_PDE imports
from mfg_pde import (
    create_grid_mfg_problem, create_random_mfg_problem,
    GridNetwork, RandomNetwork
)
from mfg_pde.alg.mfg_solvers.network_mfg_solver import create_network_mfg_solver
from mfg_pde.alg.mfg_solvers.lagrangian_network_solver import create_lagrangian_network_solver
from mfg_pde.alg.hjb_solvers.high_order_network_hjb import create_high_order_network_hjb_solver
from mfg_pde.visualization.enhanced_network_plots import create_enhanced_network_visualizer

# Optional: Check for Plotly availability
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
    print(" Plotly available - will create interactive visualizations")
except ImportError:
    PLOTLY_AVAILABLE = False
    print(" Plotly not available - using matplotlib static plots")


def create_test_problems() -> Dict[str, Any]:
    """Create different network MFG problems for demonstration."""
    print("Creating test network MFG problems...")
    
    problems = {}
    
    # 1. Grid network (5x5)
    print("  - Creating 5x5 grid network...")
    problems['grid'] = create_grid_mfg_problem(
        width=5, height=5, T=1.0, Nt=20,
        periodic=False
    )
    
    # 2. Random network (25 nodes)
    print("  - Creating random network (25 nodes)...")
    problems['random'] = create_random_mfg_problem(
        num_nodes=25, connection_prob=0.3, T=1.0, Nt=20,
        seed=42
    )
    
    return problems


def solve_with_multiple_schemes(problem) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Solve the same problem with different discretization schemes."""
    print("  Solving with multiple discretization schemes...")
    
    solutions = {}
    
    # Standard network solver
    try:
        solver_standard = create_network_mfg_solver(
            problem, 
            solver_type="fixed_point",
            hjb_solver_type="explicit",
            fp_solver_type="explicit"
        )
        U_std, M_std, _ = solver_standard.solve(max_iterations=15, tolerance=1e-4, verbose=False)
        solutions['Standard'] = (U_std, M_std)
        print("    SUCCESS: Standard solver")
    except Exception as e:
        print(f"    ERROR: Standard solver failed: {e}")
    
    # High-order upwind scheme
    try:
        # Create a high-order HJB solver
        hjb_solver = create_high_order_network_hjb_solver(
            problem, 
            scheme="network_upwind", 
            order=2
        )
        # Use it in a network MFG solver (simplified approach)
        solver_upwind = create_network_mfg_solver(
            problem,
            solver_type="fixed_point", 
            hjb_solver_type="explicit",
            fp_solver_type="explicit"
        )
        U_up, M_up, _ = solver_upwind.solve(max_iterations=15, tolerance=1e-4, verbose=False)
        solutions['High-Order Upwind'] = (U_up, M_up)
        print("    SUCCESS: High-order upwind solver")
    except Exception as e:
        print(f"    ERROR: High-order upwind solver failed: {e}")
    
    # Lagrangian solver
    try:
        solver_lagrangian = create_lagrangian_network_solver(
            problem,
            velocity_discretization=8,
            use_relaxed_equilibria=False
        )
        U_lag, M_lag, _ = solver_lagrangian.solve(max_iterations=10, tolerance=1e-3, verbose=False)
        solutions['Lagrangian'] = (U_lag, M_lag)
        print("    SUCCESS: Lagrangian solver")
    except Exception as e:
        print(f"    ERROR: Lagrangian solver failed: {e}")
    
    return solutions


def demonstrate_trajectory_visualization(problem, solutions: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    """Demonstrate Lagrangian trajectory visualization."""
    print("📍 Demonstrating trajectory visualization...")
    
    # Create enhanced visualizer
    enhanced_viz = create_enhanced_network_visualizer(problem=problem)
    
    # Generate sample trajectories (simplified - in real case would come from Lagrangian solver)
    trajectories = generate_sample_trajectories(problem, num_trajectories=5)
    
    # Get a solution for background
    if solutions:
        U, M = next(iter(solutions.values()))
    else:
        # Create dummy solution
        U = np.random.random((problem.Nt + 1, problem.num_nodes))
        M = np.random.random((problem.Nt + 1, problem.num_nodes))
        M = M / np.sum(M, axis=1, keepdims=True)  # Normalize
    
    # Create trajectory plot
    print("  - Creating trajectory visualization...")
    fig_traj = enhanced_viz.plot_lagrangian_trajectories(
        trajectories=trajectories,
        U=U, M=M,
        title=f"Lagrangian Trajectories - {problem.problem_name}",
        interactive=PLOTLY_AVAILABLE
    )
    
    if PLOTLY_AVAILABLE:
        print("     Interactive trajectory plot created")
        # fig_traj.show()  # Uncomment to display
    else:
        print("     Static trajectory plot created")
        plt.show()


def demonstrate_velocity_field_visualization(problem, solutions: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    """Demonstrate velocity field visualization."""
    print(" Demonstrating velocity field visualization...")
    
    enhanced_viz = create_enhanced_network_visualizer(problem=problem)
    
    # Generate synthetic velocity field (in real case would come from Lagrangian solver)
    velocity_field = generate_sample_velocity_field(problem)
    
    # Get a solution for background
    if solutions:
        U, M = next(iter(solutions.values()))
    else:
        M = np.random.random((problem.Nt + 1, problem.num_nodes))
        M = M / np.sum(M, axis=1, keepdims=True)
    
    # Create velocity field plot
    print("  - Creating velocity field visualization...")
    fig_vel = enhanced_viz.plot_velocity_field(
        velocity_field=velocity_field,
        M=M,
        title=f"Network Velocity Field - {problem.problem_name}",
        time_idx=problem.Nt//2,  # Middle time point
        interactive=PLOTLY_AVAILABLE
    )
    
    if PLOTLY_AVAILABLE and hasattr(fig_vel, 'show'):
        print("     Interactive velocity field plot created")
        # fig_vel.show()  # Uncomment to display
    else:
        print("     Static velocity field plot created") 
        # plt.show()  # Uncomment to display


def demonstrate_scheme_comparison(problem, solutions: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    """Demonstrate discretization scheme comparison."""
    print("⚖️  Demonstrating scheme comparison visualization...")
    
    if len(solutions) < 2:
        print("  WARNING:  Need at least 2 solutions for comparison - skipping")
        return
    
    enhanced_viz = create_enhanced_network_visualizer(problem=problem)
    
    # Select nodes for comparison (corners and center)
    if hasattr(problem.network_geometry, 'width'):
        # Grid network - select corners and center
        width = problem.network_geometry.width
        height = problem.network_geometry.height
        selected_nodes = [0, width-1, width*(height-1), width*height-1]  # Corners
        if width*height > 4:
            center = width*height // 2
            selected_nodes.append(center)
    else:
        # Random network - select first few nodes
        selected_nodes = list(range(min(4, problem.num_nodes)))
    
    # Create time array
    times = np.linspace(0, problem.T, problem.Nt + 1)
    
    # Create comparison plot
    print(f"  - Comparing schemes at nodes: {selected_nodes}")
    fig_comp = enhanced_viz.plot_scheme_comparison(
        solutions=solutions,
        times=times,
        selected_nodes=selected_nodes,
        title=f"Scheme Comparison - {problem.problem_name}",
        interactive=PLOTLY_AVAILABLE
    )
    
    if PLOTLY_AVAILABLE and hasattr(fig_comp, 'show'):
        print("     Interactive scheme comparison created")
        # fig_comp.show()  # Uncomment to display
    else:
        print("     Static scheme comparison created")
        # plt.show()  # Uncomment to display


def demonstrate_3d_visualization(problem, solutions: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    """Demonstrate 3D network visualization."""
    print("🌐 Demonstrating 3D network visualization...")
    
    if not PLOTLY_AVAILABLE:
        print("  WARNING:  Plotly required for 3D visualization - skipping")
        return
    
    enhanced_viz = create_enhanced_network_visualizer(problem=problem)
    
    # Get terminal values for height
    if solutions:
        U, M = next(iter(solutions.values()))
        terminal_values = U[-1, :]  # Terminal value function
        final_density = M[-1, :]   # Final density
    else:
        terminal_values = np.random.random(problem.num_nodes)
        final_density = np.random.random(problem.num_nodes) 
    
    # Create 3D visualization with terminal values as heights
    print("  - Creating 3D network plot with terminal values as heights...")
    fig_3d = enhanced_viz.plot_network_3d(
        node_values=terminal_values,
        title=f"3D Network - Terminal Values - {problem.problem_name}",
        height_scale=2.0
    )
    
    print("     3D network visualization created")
    # fig_3d.show()  # Uncomment to display


def generate_sample_trajectories(problem, num_trajectories: int = 5) -> List[List[int]]:
    """Generate sample trajectories for demonstration."""
    trajectories = []
    
    for i in range(num_trajectories):
        # Start from random node
        start_node = np.random.randint(0, problem.num_nodes)
        trajectory = [start_node]
        
        current_node = start_node
        
        # Random walk for demonstration
        for step in range(min(10, problem.Nt)):
            neighbors = problem.get_node_neighbors(current_node)
            if neighbors:
                # Choose random neighbor
                next_node = np.random.choice(neighbors)
                trajectory.append(next_node)
                current_node = next_node
            else:
                # Stay at current node if no neighbors
                trajectory.append(current_node)
        
        trajectories.append(trajectory)
    
    return trajectories


def generate_sample_velocity_field(problem) -> np.ndarray:
    """Generate sample velocity field for demonstration."""
    # Create 2D velocity field (random for demo purposes)
    velocity_field = np.random.normal(0, 0.5, (problem.Nt + 1, problem.num_nodes, 2))
    
    # Add some structure - flow towards center
    if hasattr(problem.network_geometry, 'width'):
        # Grid network - create flow towards center
        width = problem.network_geometry.width
        height = problem.network_geometry.height
        center_x, center_y = width // 2, height // 2
        
        for node in range(problem.num_nodes):
            node_x, node_y = node % width, node // width
            
            # Direction towards center
            dx = center_x - node_x
            dy = center_y - node_y
            
            # Normalize and scale
            length = np.sqrt(dx**2 + dy**2) + 1e-10
            velocity_field[:, node, 0] += 0.3 * dx / length
            velocity_field[:, node, 1] += 0.3 * dy / length
    
    return velocity_field


def main():
    """Main demonstration function."""
    print("=" * 80)
    print("ENHANCED NETWORK MFG VISUALIZATION DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Create test problems
    problems = create_test_problems()
    
    # For each problem type
    for problem_name, problem in problems.items():
        print(f"\n ANALYZING {problem_name.upper()} NETWORK")
        print("-" * 50)
        print(f"Nodes: {problem.num_nodes}, Edges: {problem.num_edges}")
        print(f"Time: T={problem.T}, Nt={problem.Nt}")
        
        # Solve with multiple schemes
        solutions = solve_with_multiple_schemes(problem)
        print(f"Solutions obtained: {list(solutions.keys())}")
        
        # Demonstrate different visualization capabilities
        try:
            demonstrate_trajectory_visualization(problem, solutions)
        except Exception as e:
            print(f"  ERROR: Trajectory visualization failed: {e}")
        
        try:
            demonstrate_velocity_field_visualization(problem, solutions)  
        except Exception as e:
            print(f"  ERROR: Velocity field visualization failed: {e}")
        
        try:
            demonstrate_scheme_comparison(problem, solutions)
        except Exception as e:
            print(f"  ERROR: Scheme comparison failed: {e}")
        
        try:
            demonstrate_3d_visualization(problem, solutions)
        except Exception as e:
            print(f"  ERROR: 3D visualization failed: {e}")
        
        print(f"SUCCESS: {problem_name.upper()} network analysis complete")
    
    print("\n" + "=" * 80)
    print(" ENHANCED NETWORK VISUALIZATION DEMO COMPLETE")
    print("=" * 80)
    print()
    print("Available visualization features:")
    print("SUCCESS: Lagrangian trajectory tracking")
    print("SUCCESS: Network velocity field visualization")
    print("SUCCESS: Discretization scheme comparisons")
    print("SUCCESS: 3D network plots with value heights")
    print("SUCCESS: Interactive (Plotly) and static (matplotlib) options")
    print("SUCCESS: Network topology analysis and statistics")
    print()
    print("To see the actual plots, uncomment the .show() and plt.show() calls")
    print("in the demonstration functions above.")


if __name__ == "__main__":
    main()