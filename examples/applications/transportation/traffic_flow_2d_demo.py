"""
2D Traffic Flow Control using Multi-Dimensional MFG Infrastructure.

This example demonstrates optimal traffic routing on a 2D spatial domain using:
- Tensor product grids for efficient 2D discretization
- Sparse matrix operations for large-scale problems
- Multi-dimensional visualization with 3D surface plots

Problem Description:
    Agents (vehicles) navigate a 2D road network [0,L]×[0,L] to reach a destination
    while minimizing travel time and avoiding congestion.

Mathematical Model:
    State: (x,y) ∈ [0,L]×[0,L] (vehicle position)
    Control: (vₓ, vᵧ) (velocity in x and y directions)

    HJB Equation:
        -∂u/∂t + H(∇u, m) + σ²Δu = 0
        u(T,x,y) = g(x,y)

    where H(p,m) = ½λ|p|² + f(x,y,m) (running cost)
          - λ|p|²/2: Control cost (fuel/energy)
          - f(x,y,m): Congestion cost (increases with density m)

    Fokker-Planck Equation:
        ∂m/∂t - ∇·(m∇ₚH) + σ²Δm = 0
        m(0,x,y) = m₀(x,y)

Application Context:
    - Urban traffic management
    - Autonomous vehicle routing
    - Crowd flow optimization
    - Supply chain logistics

References:
    - Achdou et al. (2020): Mean Field Games on Networks
    - Burger et al. (2021): Traffic Flow via MFG Models
"""

from pathlib import Path

import numpy as np

# Multi-dimensional infrastructure
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.utils import SparseMatrixBuilder, SparseSolver
from mfg_pde.visualization import MultiDimVisualizer


def create_traffic_problem():
    """
    Create 2D traffic flow MFG problem.

    Returns:
        Dictionary with problem parameters and grid
    """
    # Spatial domain: [0,10]×[0,10] km road network
    L = 10.0
    Nx = Ny = 51  # 51×51 spatial grid

    # Time horizon
    T = 1.0  # 1 hour
    Nt = 50

    # Create 2D tensor product grid
    grid = TensorProductGrid(bounds=[(0.0, L), (0.0, L)], num_points=[Nx, Ny])

    # Physical parameters
    sigma = 0.5  # Diffusion (traffic randomness)
    lambda_param = 1.0  # Control cost weight
    gamma = 2.0  # Congestion sensitivity

    # Destination (center of domain)
    x_dest = L / 2
    y_dest = L / 2

    return {
        "grid": grid,
        "T": T,
        "Nt": Nt,
        "sigma": sigma,
        "lambda_param": lambda_param,
        "gamma": gamma,
        "destination": (x_dest, y_dest),
    }


def initial_density(grid):
    """
    Initial vehicle distribution m₀(x,y).

    Vehicles start concentrated in bottom-left corner (residential area).
    """
    X, Y = grid.meshgrid(indexing="ij")

    # Gaussian distribution centered at (2,2)
    x0, y0 = 2.0, 2.0
    sigma_init = 1.0

    m0 = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma_init**2))

    # Normalize to probability distribution
    m0 = m0 / (np.sum(m0) * grid.volume_element())

    return m0


def terminal_cost(grid, destination):
    """
    Terminal cost g(x,y): Distance to destination.

    Vehicles want to reach the destination (city center).
    """
    X, Y = grid.meshgrid(indexing="ij")
    x_dest, y_dest = destination

    # Quadratic penalty for distance to destination
    g = 0.5 * ((X - x_dest) ** 2 + (Y - y_dest) ** 2)

    return g


def hamiltonian(px, py, m, lambda_param, gamma):
    """
    Hamiltonian H(p,m) for traffic flow.

    Args:
        px, py: Components of momentum ∇u
        m: Density
        lambda_param: Control cost weight
        gamma: Congestion sensitivity

    Returns:
        Hamiltonian value
    """
    # Control cost: ½λ|p|²
    control_cost = 0.5 * lambda_param * (px**2 + py**2)

    # Congestion cost: γm (increases with density)
    congestion_cost = gamma * m

    return control_cost + congestion_cost


def solve_traffic_mfg_simple(problem):
    """
    Solve 2D traffic MFG using simple iterative scheme.

    Simplified for demonstration - uses fixed-point iteration on (u,m).
    """
    grid = problem["grid"]
    T = problem["T"]
    Nt = problem["Nt"]
    sigma = problem["sigma"]
    lambda_param = problem["lambda_param"]
    gamma = problem["gamma"]

    dt = T / Nt
    Nx, Ny = grid.num_points

    # Initialize
    m0 = initial_density(grid)
    g = terminal_cost(grid, problem["destination"])

    # Build sparse Laplacian
    builder = SparseMatrixBuilder(grid, matrix_format="csr")
    L = builder.build_laplacian(boundary_conditions="neumann")

    # Initialize solution arrays
    u = np.zeros((Nt + 1, Nx, Ny))
    m = np.zeros((Nt + 1, Nx, Ny))

    # Terminal condition
    u[-1, :, :] = g
    m[0, :, :] = m0

    # Fixed-point iteration
    max_iter = 20
    tol = 1e-4

    print(f"Solving 2D Traffic MFG on {Nx}×{Ny} grid...")
    print(f"Total unknowns: {Nx * Ny} per time step")

    for iteration in range(max_iter):
        u_old = u.copy()
        m_old = m.copy()

        # Backward HJB solve (simplified implicit scheme)
        for n in range(Nt - 1, -1, -1):
            # Current density
            m_n = m[n, :, :].flatten()

            # Build gradient matrices
            Gx = builder.build_gradient(direction=0, order=2)
            Gy = builder.build_gradient(direction=1, order=2)

            # Compute gradients
            u_next = u[n + 1, :, :].flatten()
            px = (Gx @ u_next).reshape(Nx, Ny)
            py = (Gy @ u_next).reshape(Nx, Ny)

            # Hamiltonian
            H = hamiltonian(px, py, m[n, :, :], lambda_param, gamma)

            # Implicit time step for u (simplified)
            # u_n = u_{n+1} + dt * H - dt * σ²Δu_n
            identity_matrix = np.eye(Nx * Ny)
            A = identity_matrix + dt * sigma**2 * (-L)
            b = u_next + dt * H.flatten()

            # Solve linear system
            solver = SparseSolver(method="direct")
            u[n, :, :] = solver.solve(A, b).reshape(Nx, Ny)

        # Forward FP solve (simplified - diffusion only)
        for n in range(Nt):
            # Simplified forward Euler: m_{n+1} = m_n + dt * σ²Δm
            m_n = m[n, :, :].flatten()
            m[n + 1, :, :] = m[n, :, :] + dt * sigma**2 * (L @ m_n).reshape(Nx, Ny)

        # Check convergence
        u_change = np.max(np.abs(u - u_old))
        m_change = np.max(np.abs(m - m_old))

        print(f"Iteration {iteration + 1}: Δu = {u_change:.6f}, Δm = {m_change:.6f}")

        if u_change < tol and m_change < tol:
            print(f"✓ Converged in {iteration + 1} iterations")
            break

    return {"u": u, "m": m, "grid": grid}


def visualize_results(solution, output_dir="traffic_flow_2d"):
    """
    Create comprehensive visualizations of traffic flow solution.
    """
    u = solution["u"]
    m = solution["m"]
    grid = solution["grid"]

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\nCreating visualizations in {output_dir}/...")

    # Create visualizer
    viz = MultiDimVisualizer(grid, backend="plotly", colorscale="Viridis")

    # 1. Value function at final time
    print("  - Value function surface plot...")
    fig_u = viz.surface_plot(
        u[-1, :, :], title="Value Function u(x,y) at Final Time", xlabel="x (km)", ylabel="y (km)", zlabel="Cost-to-go"
    )
    viz.save(fig_u, output_path / "value_function.html")

    # 2. Density at initial time
    print("  - Initial density heatmap...")
    fig_m0 = viz.heatmap(m[0, :, :], title="Initial Vehicle Density m₀(x,y)", xlabel="x (km)", ylabel="y (km)")
    viz.save(fig_m0, output_path / "density_initial.html")

    # 3. Density at final time
    print("  - Final density heatmap...")
    fig_mT = viz.heatmap(m[-1, :, :], title="Final Vehicle Density m(T,x,y)", xlabel="x (km)", ylabel="y (km)")
    viz.save(fig_mT, output_path / "density_final.html")

    # 4. Contour plot of value function
    print("  - Value function contour plot...")
    fig_contour = viz.contour_plot(
        u[-1, :, :], title="Value Function Contours", xlabel="x (km)", ylabel="y (km)", levels=20
    )
    viz.save(fig_contour, output_path / "value_contours.html")

    # 5. Slice along y=5 (middle)
    print("  - Cross-section plot...")
    fig_slice = viz.slice_plot(u[-1, :, :], slice_dim=1, slice_index=25, title="Value Function along y=5 km")
    viz.save(fig_slice, output_path / "value_slice.html")

    # 6. Time evolution animation
    print("  - Time evolution animation...")
    fig_anim = viz.animation(
        m, title="Vehicle Density Evolution m(t,x,y)", xlabel="x (km)", ylabel="y (km)", zlabel="Density", fps=5
    )
    viz.save(fig_anim, output_path / "density_evolution.html")

    print(f"\n✓ Visualizations saved to {output_dir}/")
    print("\nGenerated files:")
    print("  - value_function.html: 3D surface plot of cost-to-go")
    print("  - density_initial.html: Initial vehicle distribution")
    print("  - density_final.html: Final vehicle distribution")
    print("  - value_contours.html: Level sets of value function")
    print("  - value_slice.html: Cross-sectional view")
    print("  - density_evolution.html: Interactive time animation")


def main():
    """
    Main execution: Solve 2D traffic MFG and visualize results.
    """
    print("=" * 60)
    print("2D Traffic Flow Control via Mean Field Games")
    print("=" * 60)
    print("\nProblem Setup:")
    print("  - Domain: [0,10] × [0,10] km road network")
    print("  - Vehicles start in bottom-left (residential)")
    print("  - Destination: City center (5,5)")
    print("  - Objective: Minimize travel time + avoid congestion")
    print()

    # Create problem
    problem = create_traffic_problem()

    print("Grid Information:")
    print(f"  - Spatial: {problem['grid'].num_points[0]}×{problem['grid'].num_points[1]} points")
    print(f"  - Temporal: {problem['Nt']} time steps")
    print(f"  - Total DOF: {problem['grid'].total_points() * problem['Nt']}")
    print()

    # Solve MFG
    solution = solve_traffic_mfg_simple(problem)

    # Visualize
    visualize_results(solution)

    # Analysis
    print("\n" + "=" * 60)
    print("Solution Analysis:")
    print("=" * 60)

    m_final = solution["m"][-1, :, :]
    u_final = solution["u"][-1, :, :]

    print("\nDensity Statistics:")
    print(f"  - Initial max density: {solution['m'][0, :, :].max():.4f}")
    print(f"  - Final max density: {m_final.max():.4f}")
    print(f"  - Mass conservation: {np.sum(m_final) * problem['grid'].volume_element():.4f}")

    print("\nValue Function:")
    print(f"  - Min cost-to-go: {u_final.min():.4f}")
    print(f"  - Max cost-to-go: {u_final.max():.4f}")

    print("\n" + "=" * 60)
    print("✓ Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    # Set matplotlib backend for non-interactive execution
    import matplotlib

    matplotlib.use("Agg")

    main()
