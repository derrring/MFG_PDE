"""
Custom Hamiltonian with Tuple Notation (Phase 3 API).

This example demonstrates how to write custom Hamiltonians using the new
tuple multi-index notation for gradients. This is the recommended approach
for advanced users who need fine-grained control over the Hamiltonian function.

Mathematical Formulation:
    State: x ∈ [0, L]

    Custom Hamiltonian:
        H(x, ∇u, m) = (1/2)|∇u|² + V(x, m)

    where V(x, m) is a position and density-dependent potential.

Key Features:
    - Demonstrates new `derivs` parameter with tuple notation
    - Shows how to access derivatives: (0,) for u, (1,) for ∂u/∂x
    - Compares with simplified ExampleMFGProblem interface
    - Validates both approaches produce same results

Usage:
    python examples/basic/custom_hamiltonian_derivs_demo.py

See Also:
    - docs/migration_guides/phase3_gradient_notation_migration.md
    - docs/gradient_notation_standard.md
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde import ExampleMFGProblem
from mfg_pde.core.mfg_problem import MFGComponents, MFGProblem
from mfg_pde.factory import create_standard_solver
from mfg_pde.utils.mfg_logging import configure_research_logging, get_logger

# Configure logging
configure_research_logging("custom_hamiltonian_derivs_demo", level="INFO")
logger = get_logger(__name__)

# Output directory
EXAMPLE_DIR = Path(__file__).parent
OUTPUT_DIR = EXAMPLE_DIR.parent / "outputs" / "basic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Method 1: Advanced API with MFGProblem and MFGComponents (NEW derivs format)
# ==============================================================================


def create_custom_problem_advanced(
    L: float = 10.0,
    target_x: float = 5.0,
    control_cost: float = 0.5,
    congestion_cost: float = 0.2,
    Nx: int = 101,
    T: float = 3.0,
    Nt: int = 51,
    sigma: float = 0.3,
):
    """
    Create custom MFG problem using advanced API with tuple notation.

    This demonstrates the NEW recommended approach for custom Hamiltonians.

    Args:
        L: Domain length [0, L]
        target_x: Target position
        control_cost: Coefficient for control cost
        congestion_cost: Coefficient for congestion
        Nx: Number of spatial grid points
        T: Time horizon
        Nt: Number of time steps
        sigma: Diffusion coefficient

    Returns:
        MFGProblem with custom Hamiltonian using derivs parameter
    """

    def hamiltonian_func(
        x_idx: int,
        x_position: float,
        m_at_x: float,
        derivs: dict[tuple, float],  # NEW: Tuple notation
        t_idx: int | None,
        current_time: float,
        problem,
    ) -> float:
        """
        Custom Hamiltonian H(x, ∇u, m) with tuple notation for gradients.

        Args:
            x_idx: Spatial grid index
            x_position: Physical position x
            m_at_x: Density at this position
            derivs: Gradient dictionary with tuple keys:
                   - (0,): u(x, t)           (function value)
                   - (1,): ∂u/∂x             (first derivative)
                   Higher-order derivatives available for WENO solvers
            t_idx: Time index (optional)
            current_time: Physical time t
            problem: Reference to MFGProblem instance

        Returns:
            Hamiltonian value H(x, ∇u, m)
        """
        # Extract gradient using tuple notation
        # Note: derivs[(0,)] contains function value u (not needed for this Hamiltonian)
        du_dx = derivs.get((1,), 0.0)  # First derivative ∂u/∂x

        # Control cost: (1/2λ)|∇u|²
        kinetic_term = (du_dx**2) / (2.0 * control_cost)

        # Running cost: Distance to target
        distance_to_target = abs(x_position - target_x)
        potential_term = distance_to_target

        # Congestion cost: γ m(x, t)
        congestion_term = congestion_cost * m_at_x

        return kinetic_term + potential_term + congestion_term

    def hamiltonian_dm_func(
        x_idx: int,
        x_position: float,
        m_at_x: float,
        derivs: dict[tuple, float],  # NEW: Tuple notation
        t_idx: int | None,
        current_time: float,
        problem,
    ) -> float:
        """
        Derivative of Hamiltonian with respect to density: ∂H/∂m.

        For congestion cost γ m, we have ∂H/∂m = γ.
        """
        return congestion_cost

    # Create components with custom Hamiltonian
    components = MFGComponents(
        hamiltonian_func=hamiltonian_func,
        hamiltonian_dm_func=hamiltonian_dm_func,
    )

    # Initial distribution: Gaussian near x=0
    def initial_density_func(x):
        return np.exp(-((x - 0.5) ** 2) / 0.5)

    # Create problem
    problem = MFGProblem(
        xmin=0.0,
        xmax=L,
        Nx=Nx,
        T=T,
        Nt=Nt,
        sigma=sigma,
        components=components,
        m_initial=initial_density_func,
    )

    return problem


# ==============================================================================
# Method 2: Simplified API with ExampleMFGProblem (for comparison)
# ==============================================================================


def create_custom_problem_simple(
    L: float = 10.0,
    target_x: float = 5.0,
    control_cost: float = 0.5,
    congestion_cost: float = 0.2,
    Nx: int = 101,
    T: float = 3.0,
    Nt: int = 51,
    sigma: float = 0.3,
):
    """
    Create custom MFG problem using simplified API.

    This demonstrates the SIMPLIFIED approach suitable for most users.
    Note: hamiltonian signature is simpler: just (x, p, m).

    Returns:
        ExampleMFGProblem with custom Hamiltonian (simplified signature)
    """

    def hamiltonian(x, p, m):
        """
        Simplified Hamiltonian signature H(x, p, m).

        Args:
            x: Position (float)
            p: Momentum (float, equivalent to du/dx)
            m: Density (float)

        Returns:
            Hamiltonian value
        """
        # Control cost: (1/2λ)|p|²
        kinetic_term = (p**2) / (2.0 * control_cost)

        # Running cost: Distance to target
        distance_to_target = abs(x - target_x)
        potential_term = distance_to_target

        # Congestion cost: γ m
        congestion_term = congestion_cost * m

        return kinetic_term + potential_term + congestion_term

    # Initial distribution
    def initial_density_func(x):
        return np.exp(-((x - 0.5) ** 2) / 0.5)

    # Create problem with simplified API
    problem = ExampleMFGProblem(
        xmin=0.0,
        xmax=L,
        Nx=Nx,
        T=T,
        Nt=Nt,
        sigma=sigma,
        hamiltonian=hamiltonian,
        m_initial=initial_density_func,
    )

    return problem


# ==============================================================================
# Demonstration and Comparison
# ==============================================================================


def compare_approaches():
    """
    Compare advanced (derivs) vs simplified (ExampleMFGProblem) approaches.
    """
    logger.info("=" * 70)
    logger.info("Custom Hamiltonian Demo: Tuple Notation vs Simplified API")
    logger.info("=" * 70)

    # Problem parameters
    L = 10.0
    target_x = 5.0
    Nx = 101
    T = 3.0
    Nt = 51

    # =========================================================================
    # Method 1: Advanced API with tuple notation
    # =========================================================================
    logger.info("\n[1/4] Creating problem with advanced API (tuple notation)...")
    problem_advanced = create_custom_problem_advanced(L=L, target_x=target_x, Nx=Nx, T=T, Nt=Nt)
    logger.info("  Using: MFGProblem with MFGComponents")
    logger.info("  Hamiltonian signature: hamiltonian_func(..., derivs, ...)")
    logger.info("  Gradient access: derivs[(1,)] for ∂u/∂x")

    logger.info("\n  Solving with advanced API...")
    solver_advanced = create_standard_solver(problem_advanced)
    result_advanced = solver_advanced.solve(max_iterations=30, tolerance=1e-3)
    logger.info(f"  Converged: {result_advanced.converged}")
    logger.info(f"  Iterations: {result_advanced.iterations}")

    # =========================================================================
    # Method 2: Simplified API
    # =========================================================================
    logger.info("\n[2/4] Creating problem with simplified API...")
    problem_simple = create_custom_problem_simple(L=L, target_x=target_x, Nx=Nx, T=T, Nt=Nt)
    logger.info("  Using: ExampleMFGProblem")
    logger.info("  Hamiltonian signature: hamiltonian(x, p, m)")
    logger.info("  Gradient access: p directly (no dictionary)")

    logger.info("\n  Solving with simplified API...")
    solver_simple = create_standard_solver(problem_simple)
    result_simple = solver_simple.solve(max_iterations=30, tolerance=1e-3)
    logger.info(f"  Converged: {result_simple.converged}")
    logger.info(f"  Iterations: {result_simple.iterations}")

    # =========================================================================
    # Comparison
    # =========================================================================
    logger.info("\n[3/4] Comparing results...")

    # Compare final equilibrium distributions
    m_advanced = result_advanced.M[-1, :]
    m_simple = result_simple.M[-1, :]
    diff_m = np.linalg.norm(m_advanced - m_simple)

    logger.info(f"  Density difference (L2 norm): {diff_m:.6e}")

    if diff_m < 1e-2:
        logger.info("  ✓ Results match! Both APIs produce equivalent solutions.")
    else:
        logger.info(f"  ✗ Significant difference detected: {diff_m:.6e}")

    # =========================================================================
    # Visualization
    # =========================================================================
    logger.info("\n[4/4] Generating visualizations...")

    x_grid = np.linspace(problem_advanced.xmin, problem_advanced.xmax, Nx + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Final equilibrium comparison
    ax = axes[0, 0]
    ax.plot(x_grid, m_advanced, "b-", linewidth=2.5, label="Advanced API (derivs)")
    ax.plot(x_grid, m_simple, "r--", linewidth=2, label="Simplified API")
    ax.axvline(target_x, color="orange", linestyle=":", linewidth=2, label="Target")
    ax.set_xlabel("Position x", fontsize=11)
    ax.set_ylabel("Equilibrium density m(x)", fontsize=11)
    ax.set_title("Final Equilibrium Comparison", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Panel 2: Value function comparison
    ax = axes[0, 1]
    U_advanced = result_advanced.U[-1, :]
    U_simple = result_simple.U[-1, :]
    ax.plot(x_grid, U_advanced, "b-", linewidth=2.5, label="Advanced API")
    ax.plot(x_grid, U_simple, "r--", linewidth=2, label="Simplified API")
    ax.set_xlabel("Position x", fontsize=11)
    ax.set_ylabel("Value function u(x, T)", fontsize=11)
    ax.set_title("Value Function Comparison", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Panel 3: Convergence history (Advanced)
    ax = axes[1, 0]
    iters_advanced = range(1, result_advanced.iterations + 1)
    ax.semilogy(
        iters_advanced,
        result_advanced.error_history_M[: result_advanced.iterations],
        "b-",
        linewidth=2,
        label="Advanced API",
    )
    ax.semilogy(
        iters_advanced,
        result_simple.error_history_M[: result_simple.iterations],
        "r--",
        linewidth=2,
        label="Simplified API",
    )
    ax.set_xlabel("Picard iteration", fontsize=11)
    ax.set_ylabel("L² error", fontsize=11)
    ax.set_title("Convergence History", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Panel 4: Difference heatmap
    ax = axes[1, 1]
    diff = np.abs(m_advanced - m_simple)
    ax.plot(x_grid, diff, "k-", linewidth=2)
    ax.fill_between(x_grid, 0, diff, alpha=0.3, color="gray")
    ax.set_xlabel("Position x", fontsize=11)
    ax.set_ylabel("|m_advanced - m_simple|", fontsize=11)
    ax.set_title(f"Density Difference (L² = {diff_m:.2e})", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()

    output_path = OUTPUT_DIR / "custom_hamiltonian_derivs_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"  Saved: {output_path}")

    # Show plot
    plt.show()

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Demo Complete!")
    logger.info("=" * 70)
    logger.info("\nKey Takeaways:")
    logger.info("  1. Advanced API: Full control via MFGComponents and derivs parameter")
    logger.info("  2. Simplified API: Easy to use via ExampleMFGProblem")
    logger.info("  3. Both approaches produce equivalent results")
    logger.info("  4. Tuple notation (derivs) is dimension-agnostic and type-safe")
    logger.info("\nRecommendation:")
    logger.info("  - Use ExampleMFGProblem for most applications")
    logger.info("  - Use MFGProblem + derivs for advanced control or 2D/3D problems")
    logger.info("\nSee Also:")
    logger.info("  - docs/migration_guides/phase3_gradient_notation_migration.md")
    logger.info("  - docs/gradient_notation_standard.md")


def main():
    """Run the demonstration."""
    compare_approaches()


if __name__ == "__main__":
    main()
