"""
2D Portfolio Optimization with Mean Field Games.

This example demonstrates optimal portfolio allocation in financial markets using:
- 2D state space: (wealth, stock allocation)
- Tensor product grids for efficient discretization
- Sparse matrix operations for large-scale financial models
- Multi-dimensional visualization for utility surfaces

Problem Description:
    Fund managers optimize portfolio allocation between risk-free bonds and risky stocks
    in the presence of market crowding effects from other investors.

Mathematical Model:
    State: (W, alpha) where
        - W: Wealth level (normalized)
        - alpha: Fraction of wealth in risky asset

    HJB Equation:
        -∂u/∂t + H(∇u, W, alpha, m) + sigma^2 Δu = 0
        u(T, W, alpha) = U(W)  (terminal utility)

    where H(p, W, alpha, m) models:
        - Optimal wealth accumulation
        - Portfolio rebalancing costs
        - Market impact from crowd behavior m

    Fokker-Planck Equation:
        ∂m/∂t - ∇·(m v*) + sigma^2 Δm = 0
        m(0, W, alpha) = m_0(W, alpha)

Application Context:
    - Hedge fund competition and systemic risk
    - Optimal investment with market impact
    - Crowding in quantitative strategies
    - Systemic financial stability

References:
    - Lasry & Lions (2007): Mean Field Games
    - Cardaliaguet & Lehalle (2018): Mean Field Games of Controls
    - Carmona & Delarue (2018): Probabilistic Theory of MFG (Vol. 2)
"""

from pathlib import Path

import numpy as np

# Multi-dimensional infrastructure
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.utils import SparseMatrixBuilder, SparseSolver
from mfg_pde.visualization import MultiDimVisualizer


def create_portfolio_problem():
    """
    Create 2D portfolio optimization MFG problem.

    Returns:
        Dictionary with problem parameters and grid
    """
    # State space: [W_min, W_max] × [alpha_min, alpha_max]
    W_min, W_max = 0.5, 2.0  # Wealth range (normalized, 1.0 = initial)
    alpha_min, alpha_max = 0.0, 1.0  # Portfolio allocation (0=bonds, 1=stocks)

    NW = 41  # Wealth grid points
    Nalpha = 31  # Allocation grid points

    # Time horizon
    T = 1.0  # 1 year investment horizon
    Nt = 50

    # Create 2D tensor product grid
    grid = TensorProductGrid(dimension=2, bounds=[(W_min, W_max), (alpha_min, alpha_max)], num_points=[NW, Nalpha])

    # Financial parameters
    r = 0.02  # Risk-free rate (bonds)
    mu = 0.08  # Expected stock return
    sigma_market = 0.2  # Stock volatility
    sigma_diffusion = 0.05  # Model diffusion

    # Utility and costs
    gamma_risk = 2.0  # Risk aversion coefficient
    lambda_rebalance = 0.1  # Portfolio rebalancing cost
    kappa_crowd = 0.5  # Market crowding penalty

    return {
        "grid": grid,
        "T": T,
        "Nt": Nt,
        "r": r,
        "mu": mu,
        "sigma_market": sigma_market,
        "sigma": sigma_diffusion,
        "gamma_risk": gamma_risk,
        "lambda_rebalance": lambda_rebalance,
        "kappa_crowd": kappa_crowd,
    }


def initial_distribution(grid):
    """
    Initial investor distribution m_0(W, alpha).

    Investors start with moderate wealth and diversified portfolios.
    """
    W, Alpha = grid.meshgrid(indexing="ij")

    # Initial wealth: concentrated around W=1.0 (normalized)
    W0 = 1.0
    sigma_W = 0.2

    # Initial allocation: moderate diversification around alpha=0.6 (60% stocks)
    alpha0 = 0.6
    sigma_alpha = 0.15

    # Joint Gaussian distribution
    m0 = np.exp(-(((W - W0) ** 2) / (2 * sigma_W**2) + (Alpha - alpha0) ** 2 / (2 * sigma_alpha**2)))

    # Normalize to probability distribution
    m0 = m0 / (np.sum(m0) * grid.volume_element())

    return m0


def terminal_utility(grid, gamma_risk):
    """
    Terminal utility function U(W).

    Power utility: U(W) = W^(1-gamma) / (1-gamma)
    Models risk aversion in wealth accumulation.
    """
    W, _ = grid.meshgrid(indexing="ij")

    if gamma_risk == 1.0:
        # Log utility (limit case)
        u_terminal = np.log(W)
    else:
        # Power utility
        u_terminal = (W ** (1 - gamma_risk)) / (1 - gamma_risk)

    return u_terminal


def hamiltonian(pW, palpha, W, alpha, m, params):
    """
    Hamiltonian for portfolio optimization MFG.

    Args:
        pW: Momentum in wealth direction (∂u/∂W)
        palpha: Momentum in allocation direction (∂u/∂alpha)
        W: Wealth state
        alpha: Portfolio allocation state
        m: Investor density
        params: Problem parameters

    Returns:
        Hamiltonian value
    """
    r = params["r"]
    mu = params["mu"]
    sigma_market = params["sigma_market"]
    lambda_rebalance = params["lambda_rebalance"]
    kappa_crowd = params["kappa_crowd"]

    # Wealth drift: dW/dt = W * (r + alpha*(mu - r))
    # This is the expected wealth growth rate
    wealth_drift = W * (r + alpha * (mu - r))

    # Running cost components:

    # 1. Wealth accumulation (negative = reward)
    wealth_reward = -pW * wealth_drift

    # 2. Portfolio variance cost (risk from stock volatility)
    variance_cost = 0.5 * (sigma_market * alpha * W) ** 2 / lambda_rebalance

    # 3. Rebalancing cost (proportional to momentum in allocation direction)
    rebalancing_cost = 0.5 * lambda_rebalance * palpha**2

    # 4. Crowding cost (higher when many investors cluster)
    crowding_cost = kappa_crowd * m

    return wealth_reward + variance_cost + rebalancing_cost + crowding_cost


def solve_portfolio_mfg_simple(problem):
    """
    Solve 2D portfolio MFG using simplified fixed-point iteration.

    This is a demonstration solver with simplified dynamics.
    Production solvers would use more sophisticated HJB/FP discretizations.
    """
    grid = problem["grid"]
    T = problem["T"]
    Nt = problem["Nt"]
    sigma = problem["sigma"]

    dt = T / Nt
    NW, Nalpha = grid.num_points

    # Initialize
    m0 = initial_distribution(grid)
    u_terminal = terminal_utility(grid, problem["gamma_risk"])

    # Build sparse Laplacian
    builder = SparseMatrixBuilder(grid, matrix_format="csr")
    L = builder.build_laplacian(boundary_conditions="neumann")

    # Initialize solution arrays
    u = np.zeros((Nt + 1, NW, Nalpha))
    m = np.zeros((Nt + 1, NW, Nalpha))

    # Boundary conditions
    u[-1, :, :] = u_terminal
    m[0, :, :] = m0

    # Fixed-point iteration
    max_iter = 15
    tol = 1e-4

    print(f"Solving 2D Portfolio MFG on {NW}×{Nalpha} grid...")
    print("State space: Wealth × Allocation")
    print(f"Total unknowns: {NW * Nalpha} per time step")

    for iteration in range(max_iter):
        u_old = u.copy()
        m_old = m.copy()

        # Backward HJB solve (simplified)
        for n in range(Nt - 1, -1, -1):
            # Current density
            m_n = m[n, :, :].flatten()

            # Build gradient matrices
            GW = builder.build_gradient(direction=0, order=2)
            Galpha = builder.build_gradient(direction=1, order=2)

            # Compute gradients
            u_next = u[n + 1, :, :].flatten()
            pW = (GW @ u_next).reshape(NW, Nalpha)
            palpha = (Galpha @ u_next).reshape(NW, Nalpha)

            # Meshgrid for state-dependent Hamiltonian
            W, Alpha = grid.meshgrid(indexing="ij")

            # Hamiltonian
            H = hamiltonian(pW, palpha, W, Alpha, m[n, :, :], problem)

            # Implicit time step: u_n = u_{n+1} + dt*H - dt*sigma^2*Δu_n
            identity_matrix = np.eye(NW * Nalpha)
            A = identity_matrix + dt * sigma**2 * (-L)
            b = u_next + dt * H.flatten()

            # Solve linear system
            solver = SparseSolver(method="direct")
            u[n, :, :] = solver.solve(A, b).reshape(NW, Nalpha)

        # Forward FP solve (simplified - diffusion only)
        for n in range(Nt):
            # Simplified forward Euler: m_{n+1} = m_n + dt * sigma^2 * Δm
            m_n = m[n, :, :].flatten()
            m[n + 1, :, :] = m[n, :, :] + dt * sigma**2 * (L @ m_n).reshape(NW, Nalpha)

        # Check convergence
        u_change = np.max(np.abs(u - u_old))
        m_change = np.max(np.abs(m - m_old))

        print(f"Iteration {iteration + 1}: Δu = {u_change:.6f}, Δm = {m_change:.6f}")

        if u_change < tol and m_change < tol:
            print(f"Converged in {iteration + 1} iterations")
            break

    return {"u": u, "m": m, "grid": grid}


def visualize_results(solution, problem, output_dir="portfolio_optimization_2d"):
    """
    Create comprehensive visualizations of portfolio optimization solution.
    """
    u = solution["u"]
    m = solution["m"]
    grid = solution["grid"]

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"\nCreating visualizations in {output_dir}/...")

    # Create visualizer
    viz = MultiDimVisualizer(grid, backend="plotly", colorscale="RdYlGn")

    # 1. Value function surface (optimal value-to-go)
    print("  - Value function surface...")
    fig_u = viz.surface_plot(
        u[-1, :, :],
        title="Portfolio Value Function u(W,alpha)",
        xlabel="Wealth W",
        ylabel="Stock Allocation alpha",
        zlabel="Value-to-go",
    )
    viz.save(fig_u, output_path / "value_function.html")

    # 2. Initial investor distribution
    print("  - Initial distribution heatmap...")
    fig_m0 = viz.heatmap(
        m[0, :, :],
        title="Initial Investor Distribution m_0(W,alpha)",
        xlabel="Wealth W",
        ylabel="Stock Allocation alpha",
    )
    viz.save(fig_m0, output_path / "distribution_initial.html")

    # 3. Final investor distribution
    print("  - Final distribution heatmap...")
    fig_mT = viz.heatmap(
        m[-1, :, :],
        title="Final Investor Distribution m(T,W,alpha)",
        xlabel="Wealth W",
        ylabel="Stock Allocation alpha",
    )
    viz.save(fig_mT, output_path / "distribution_final.html")

    # 4. Value function contours (indifference curves)
    print("  - Value function contours...")
    fig_contour = viz.contour_plot(
        u[-1, :, :],
        title="Portfolio Indifference Curves",
        xlabel="Wealth W",
        ylabel="Stock Allocation alpha",
        levels=25,
    )
    viz.save(fig_contour, output_path / "indifference_curves.html")

    # 5. Wealth slice (alpha = 0.6, typical diversified portfolio)
    print("  - Wealth slice plot...")
    alpha_idx = int(0.6 * grid.num_points[1])  # 60% stocks
    fig_slice = viz.slice_plot(
        u[-1, :, :],
        slice_dim=1,
        slice_index=alpha_idx,
        title="Value Function at alpha=0.6 (Diversified Portfolio)",
    )
    viz.save(fig_slice, output_path / "value_wealth_slice.html")

    # 6. Time evolution animation
    print("  - Time evolution animation...")
    fig_anim = viz.animation(
        m,
        title="Investor Distribution Evolution m(t,W,alpha)",
        xlabel="Wealth W",
        ylabel="Stock Allocation alpha",
        zlabel="Density",
        fps=5,
    )
    viz.save(fig_anim, output_path / "distribution_evolution.html")

    print(f"\nVisualizations saved to {output_dir}/")
    print("\nGenerated files:")
    print("  - value_function.html: 3D surface of optimal value-to-go")
    print("  - distribution_initial.html: Initial investor distribution")
    print("  - distribution_final.html: Final investor distribution")
    print("  - indifference_curves.html: Portfolio indifference curves")
    print("  - value_wealth_slice.html: Value function for diversified portfolio")
    print("  - distribution_evolution.html: Interactive time animation")


def main():
    """
    Main execution: Solve 2D portfolio MFG and visualize results.
    """
    print("=" * 70)
    print("2D Portfolio Optimization via Mean Field Games")
    print("=" * 70)
    print("\nProblem Setup:")
    print("  - State Space: (Wealth, Stock Allocation)")
    print("  - Wealth range: [0.5, 2.0] (normalized)")
    print("  - Allocation range: [0, 1] (0=bonds, 1=stocks)")
    print("  - Objective: Maximize utility with market crowding effects")
    print()

    # Create problem
    problem = create_portfolio_problem()

    print("Grid Information:")
    print(f"  - Wealth points: {problem['grid'].num_points[0]}")
    print(f"  - Allocation points: {problem['grid'].num_points[1]}")
    print(f"  - Temporal steps: {problem['Nt']}")
    print(f"  - Total DOF: {problem['grid'].total_points() * problem['Nt']}")
    print()

    print("Financial Parameters:")
    print(f"  - Risk-free rate: {problem['r']*100:.1f}%")
    print(f"  - Stock expected return: {problem['mu']*100:.1f}%")
    print(f"  - Stock volatility: {problem['sigma_market']*100:.1f}%")
    print(f"  - Risk aversion: {problem['gamma_risk']:.1f}")
    print()

    # Solve MFG
    solution = solve_portfolio_mfg_simple(problem)

    # Visualize
    visualize_results(solution, problem)

    # Analysis
    print("\n" + "=" * 70)
    print("Solution Analysis:")
    print("=" * 70)

    m_final = solution["m"][-1, :, :]
    u_final = solution["u"][-1, :, :]

    print("\nInvestor Distribution Statistics:")
    print(f"  - Initial max density: {solution['m'][0, :, :].max():.4f}")
    print(f"  - Final max density: {m_final.max():.4f}")
    print(f"  - Mass conservation: {np.sum(m_final) * problem['grid'].volume_element():.4f}")

    # Find optimal allocation region
    W_idx, alpha_idx = np.unravel_index(m_final.argmax(), m_final.shape)
    W_coords = problem["grid"].coordinates[0]
    alpha_coords = problem["grid"].coordinates[1]
    print("\nOptimal Region (highest final density):")
    print(f"  - Wealth: {W_coords[W_idx]:.2f}")
    print(f"  - Stock allocation: {alpha_coords[alpha_idx]:.1%}")

    print("\nValue Function:")
    print(f"  - Min value-to-go: {u_final.min():.4f}")
    print(f"  - Max value-to-go: {u_final.max():.4f}")

    print("\n" + "=" * 70)
    print("Demo Complete")
    print("=" * 70)


if __name__ == "__main__":
    # Set matplotlib backend for non-interactive execution
    import matplotlib

    matplotlib.use("Agg")

    main()
