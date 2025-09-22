#!/usr/bin/env python3
"""
Basic Network MFG Example.

This example demonstrates how to solve Mean Field Games on network structures
using the MFG_PDE package. We'll create a simple grid network and solve
a basic congestion game where agents try to reach certain target nodes
while avoiding overcrowding.

Mathematical Setup:
- Network: 5x5 grid with periodic boundary conditions
- Objective: Reach corner nodes while minimizing congestion
- Dynamics: Discrete movement with diffusion noise
- Coupling: Quadratic congestion penalties

Key concepts demonstrated:
- Network geometry creation
- Network MFG problem formulation
- Network MFG solver usage
- Basic visualization of results
"""

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.mfg_solvers.network_mfg_solver import create_network_mfg_solver

# MFG_PDE imports
from mfg_pde.core.network_mfg_problem import NetworkMFGComponents, NetworkMFGProblem, create_grid_mfg_problem
from mfg_pde.visualization.network_plots import create_network_visualizer


def create_simple_congestion_problem() -> NetworkMFGProblem:
    """
    Create a simple congestion game on a 5x5 grid network.

    Agents start uniformly distributed and try to reach corner nodes
    while avoiding congestion costs.
    """
    print("Creating 5x5 grid network MFG problem...")

    # Network parameters
    width, height = 5, 5
    T = 2.0  # Terminal time
    Nt = 50  # Time steps

    # Define problem components
    def terminal_value_func(node: int) -> float:
        """Terminal reward: high value at corner nodes."""
        # Convert node index to grid coordinates
        i, j = divmod(node, width)

        # Reward corner nodes
        if (i, j) in [(0, 0), (0, height - 1), (width - 1, 0), (width - 1, height - 1)]:
            return -5.0  # Negative because we minimize cost
        else:
            return 0.0

    def initial_density_func(node: int) -> float:
        """Initial density: uniform distribution."""
        return 1.0 / (width * height)

    def node_potential_func(node: int, t: float) -> float:
        """Node potential: small staying cost."""
        return 0.1

    def node_interaction_func(node: int, m: np.ndarray, t: float) -> float:
        """Congestion coupling: quadratic in local density."""
        return 2.0 * m[node] ** 2

    # Create network MFG components
    components = NetworkMFGComponents(
        terminal_node_value_func=terminal_value_func,
        initial_node_density_func=initial_density_func,
        node_potential_func=node_potential_func,
        node_interaction_func=node_interaction_func,
        diffusion_coefficient=0.2,
        drift_coefficient=1.0,
    )

    # Create grid MFG problem using factory function
    problem = create_grid_mfg_problem(
        width=width,
        height=height,
        T=T,
        Nt=Nt,
        periodic=False,
        # Pass components as keyword arguments
        terminal_node_value_func=terminal_value_func,
        initial_node_density_func=initial_density_func,
        node_potential_func=node_potential_func,
        node_interaction_func=node_interaction_func,
        diffusion_coefficient=0.2,
        drift_coefficient=1.0,
    )

    print("Created network MFG problem:")
    print(f"  Network: {width}x{height} grid")
    print(f"  Nodes: {problem.num_nodes}")
    print(f"  Edges: {problem.num_edges}")
    print(f"  Time horizon: T={T}, Nt={Nt}")
    print()

    return problem


def solve_network_mfg(problem: NetworkMFGProblem) -> tuple:
    """
    Solve the network MFG problem using fixed point iteration.

    Returns:
        (U, M, convergence_info) tuple with solution and convergence data
    """
    print("Solving network MFG problem...")

    # Create network MFG solver
    solver = create_network_mfg_solver(
        problem,
        solver_type="fixed_point",
        hjb_solver_type="explicit",
        fp_solver_type="explicit",
        damping_factor=0.6,
        hjb_kwargs={"cfl_factor": 0.3},
        fp_kwargs={"cfl_factor": 0.3, "enforce_mass_conservation": True},
    )

    # Solve the problem
    U, M, convergence_info = solver.solve(max_iterations=30, tolerance=1e-4, verbose=True)

    print("\nSolution completed:")
    print(f"  Converged: {convergence_info['converged']}")
    print(f"  Iterations: {convergence_info['iterations']}")
    print(f"  Final error: {convergence_info['final_error']:.2e}")
    print(f"  Execution time: {convergence_info['execution_time']:.2f}s")
    print()

    return U, M, convergence_info


def analyze_and_visualize_results(problem: NetworkMFGProblem, U: np.ndarray, M: np.ndarray, convergence_info: dict):
    """
    Analyze and visualize the network MFG solution.

    Args:
        problem: Network MFG problem
        U: Value function solution
        M: Density solution
        convergence_info: Convergence information
    """
    print("Analyzing and visualizing results...")

    # Create visualizer
    visualizer = create_network_visualizer(problem=problem)

    # 1. Network topology with initial density
    print("  - Plotting network topology...")
    initial_density = M[0, :]
    fig_topology = visualizer.plot_network_topology(
        node_values=initial_density,
        title="Network Topology with Initial Density",
        interactive=False,  # Use matplotlib for simplicity
    )
    plt.show()

    # 2. Terminal value function
    print("  - Plotting terminal value function...")
    terminal_values = U[-1, :]
    fig_terminal = visualizer.plot_network_topology(
        node_values=terminal_values, title="Terminal Value Function", interactive=False
    )
    plt.show()

    # 3. Density evolution for selected nodes
    print("  - Plotting density evolution...")
    times = np.linspace(0, problem.T, problem.Nt + 1)

    # Select corner nodes and center node for comparison
    corner_nodes = [0, 4, 20, 24]  # Corners of 5x5 grid
    center_node = [12]  # Center of 5x5 grid
    selected_nodes = corner_nodes + center_node

    fig_density = visualizer.plot_density_evolution(
        M, times, selected_nodes, title="Density Evolution: Corner vs Center Nodes", interactive=False
    )
    plt.show()

    # 4. Value function evolution
    print("  - Plotting value function evolution...")
    fig_value = visualizer.plot_value_function_evolution(
        U, times, selected_nodes, title="Value Function Evolution", interactive=False
    )
    plt.show()

    # 5. Analysis dashboard
    print("  - Creating analysis dashboard...")
    fig_dashboard = visualizer.plot_network_statistics_dashboard(convergence_info=convergence_info)
    if hasattr(fig_dashboard, "show"):
        fig_dashboard.show()  # Plotly figure
    else:
        plt.show()  # Matplotlib figure

    # 6. Analyze solution properties
    print("\nSolution Analysis:")
    print("-" * 40)

    # Mass conservation check
    mass_over_time = np.sum(M, axis=1)
    mass_variation = np.std(mass_over_time)
    print(f"Mass conservation (std): {mass_variation:.2e}")

    # Final density distribution
    final_density = M[-1, :]
    max_final_density = np.max(final_density)
    max_density_node = np.argmax(final_density)
    print(f"Max final density: {max_final_density:.3f} at node {max_density_node}")

    # Corner vs center density comparison
    corner_final_density = np.mean([final_density[node] for node in corner_nodes])
    center_final_density = final_density[center_node[0]]
    print(f"Average corner density: {corner_final_density:.3f}")
    print(f"Center density: {center_final_density:.3f}")
    print(f"Corner/Center ratio: {corner_final_density/center_final_density:.2f}")

    # Value function range
    final_values = U[-1, :]
    print(f"Value function range: [{np.min(final_values):.3f}, {np.max(final_values):.3f}]")

    # Network statistics
    network_stats = problem.get_network_statistics()
    print(f"Network density: {network_stats['density']:.3f}")
    print(f"Average degree: {network_stats['average_degree']:.2f}")


def main():
    """Main function demonstrating basic network MFG usage."""
    print("=" * 80)
    print("BASIC NETWORK MFG EXAMPLE")
    print("=" * 80)
    print("This example demonstrates Mean Field Games on a 5x5 grid network")
    print("with a simple congestion game where agents try to reach corner nodes.")
    print()

    try:
        # Step 1: Create network MFG problem
        problem = create_simple_congestion_problem()

        # Step 2: Solve the problem
        U, M, convergence_info = solve_network_mfg(problem)

        # Step 3: Analyze and visualize results
        analyze_and_visualize_results(problem, U, M, convergence_info)

        print("=" * 80)
        print("EXAMPLE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("Key takeaways:")
        print("- Network MFG problems can model discrete spatial games")
        print("- Agents balance between reaching targets and avoiding congestion")
        print("- The framework supports various network topologies")
        print("- Visualization tools help understand solution dynamics")

    except Exception as e:
        print(f"Error in network MFG example: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
