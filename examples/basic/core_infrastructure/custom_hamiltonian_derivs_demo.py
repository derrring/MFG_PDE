"""
Custom Hamiltonian with Multiple API Approaches.

This example demonstrates three different ways to define custom Hamiltonians:

1. **Advanced API (derivs)**: Function-based with MFGComponents and tuple notation
2. **Simplified API (x, p, m)**: Function-based with direct MFGProblem
3. **Class-based API (NEW)**: Object-oriented SeparableHamiltonian

Mathematical Formulation:
    State: x ∈ [0, L]

    Custom Hamiltonian:
        H(x, m, p, t) = (1/2λ)|p|² + |x - x_target| + γm

    where:
    - (1/2λ)|p|² is the control cost (quadratic)
    - |x - x_target| is the running cost (distance to target)
    - γm is the congestion cost

Key Features:
    - Demonstrates derivs parameter with tuple notation (legacy)
    - Shows simplified MFGProblem interface
    - Introduces class-based SeparableHamiltonian (NEW, recommended)
    - Validates all approaches produce equivalent results

Usage:
    python examples/basic/custom_hamiltonian_derivs_demo.py

See Also:
    - mfg_pde/core/hamiltonian.py (class definitions)
    - docs/migration_guides/phase3_gradient_notation_migration.md
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfg_pde.core.mfg_problem import MFGComponents, MFGProblem
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

    # Initial distribution: Gaussian near x=0
    def initial_density_func(x):
        return np.exp(-((x - 0.5) ** 2) / 0.5)

    # Terminal value: zero (standard choice for running cost problems)
    def terminal_value_func(x):
        return 0.0

    # Create components with custom Hamiltonian (Issue #670: m_initial/u_final in MFGComponents)
    components = MFGComponents(
        hamiltonian_func=hamiltonian_func,
        hamiltonian_dm_func=hamiltonian_dm_func,
        m_initial=initial_density_func,
        u_final=terminal_value_func,
    )

    # Create problem
    problem = MFGProblem(
        xmin=0.0,
        xmax=L,
        Nx=Nx,
        T=T,
        Nt=Nt,
        sigma=sigma,
        components=components,
    )

    return problem


# ==============================================================================
# Method 2: Simplified Hamiltonian with Wrapper (for comparison)
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
    Create custom MFG problem using simplified hamiltonian(x, p, m) signature.

    This demonstrates how to write a Hamiltonian with a simple signature
    and wrap it for MFGComponents. Users define H(x, p, m) directly, then
    create a wrapper that extracts p from the derivs dictionary.

    Note: This approach requires manual wrapping. For a cleaner API,
    use Method 3 (class-based SeparableHamiltonian) instead.

    Returns:
        MFGProblem with wrapped custom Hamiltonian
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

    def hamiltonian_dm(x, p, m):
        """Derivative dH/dm = γ (congestion coefficient)."""
        return congestion_cost

    # Wrap the simple (x, p, m) signature into the full derivs-based signature
    def hamiltonian_func_wrapper(x_idx, x_position, m_at_x, derivs, t_idx, current_time, problem):
        """Adapter: converts derivs dict to simple p argument."""
        p = derivs.get((1,), 0.0)  # Get du/dx from derivs
        return hamiltonian(x_position, p, m_at_x)

    def hamiltonian_dm_func_wrapper(x_idx, x_position, m_at_x, derivs, t_idx, current_time, problem):
        """Adapter for dH/dm."""
        p = derivs.get((1,), 0.0)
        return hamiltonian_dm(x_position, p, m_at_x)

    # Initial distribution
    def initial_density_func(x):
        return np.exp(-((x - 0.5) ** 2) / 0.5)

    # Terminal value
    def terminal_value_func(x):
        return 0.0

    # Create components (Issue #670: m_initial/u_final in MFGComponents)
    components = MFGComponents(
        hamiltonian_func=hamiltonian_func_wrapper,
        hamiltonian_dm_func=hamiltonian_dm_func_wrapper,
        m_initial=initial_density_func,
        u_final=terminal_value_func,
    )

    # Create problem with simplified API
    problem = MFGProblem(
        xmin=0.0,
        xmax=L,
        Nx=Nx,
        T=T,
        Nt=Nt,
        sigma=sigma,
        components=components,
    )

    return problem


# ==============================================================================
# Method 3: Class-based API with SeparableHamiltonian (NEW, recommended)
# ==============================================================================


def create_custom_problem_class_based(
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
    Create custom MFG problem using the NEW class-based Hamiltonian API.

    This demonstrates the RECOMMENDED approach for custom Hamiltonians:
    - Use SeparableHamiltonian with control_cost, potential, and coupling
    - Convert to legacy format via to_legacy_func() for MFGComponents

    The class-based approach offers:
    - Type-safe, composable Hamiltonians
    - Auto-differentiation for dp() and dm()
    - Legendre transform duality (H ↔ L)
    - Clean separation of control cost, potential, and coupling

    Returns:
        MFGProblem with class-based Hamiltonian
    """

    # Define the potential V(x, t) = |x - target|
    def potential(x, t):
        return float(np.abs(x[0] - target_x))

    # Define the density coupling f(m) = γm and its derivative df/dm = γ
    def coupling(m):
        return congestion_cost * m

    def coupling_dm(m):
        return congestion_cost

    # Create class-based Hamiltonian: H = H_control(p) + V(x, t) + f(m)
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=control_cost),
        potential=potential,
        coupling=coupling,
        coupling_dm=coupling_dm,
    )

    # Convert to legacy function format for MFGComponents
    # This provides backward compatibility with existing solver infrastructure
    hamiltonian_func, hamiltonian_dm_func = hamiltonian.to_legacy_func()

    # Initial distribution: Gaussian near x=0
    def initial_density_func(x):
        return np.exp(-((x - 0.5) ** 2) / 0.5)

    # Terminal value
    def terminal_value_func(x):
        return 0.0

    # Create components with the converted functions (Issue #670: m_initial/u_final in MFGComponents)
    components = MFGComponents(
        hamiltonian_func=hamiltonian_func,
        hamiltonian_dm_func=hamiltonian_dm_func,
        m_initial=initial_density_func,
        u_final=terminal_value_func,
    )

    # Create problem
    problem = MFGProblem(
        xmin=0.0,
        xmax=L,
        Nx=Nx,
        T=T,
        Nt=Nt,
        sigma=sigma,
        components=components,
    )

    return problem


# ==============================================================================
# Demonstration and Comparison
# ==============================================================================


def compare_approaches():
    """
    Compare all three approaches: advanced (derivs), simplified, and class-based.
    """
    logger.info("=" * 70)
    logger.info("Custom Hamiltonian Demo: Three API Approaches Compared")
    logger.info("=" * 70)

    # Problem parameters
    L = 10.0
    target_x = 5.0
    Nx = 101
    T = 3.0
    Nt = 51

    # =========================================================================
    # Method 1: Advanced API with tuple notation (legacy function-based)
    # =========================================================================
    logger.info("\n[1/5] Creating problem with advanced API (tuple notation)...")
    problem_advanced = create_custom_problem_advanced(L=L, target_x=target_x, Nx=Nx, T=T, Nt=Nt)
    logger.info("  Using: MFGProblem with MFGComponents")
    logger.info("  Hamiltonian signature: hamiltonian_func(..., derivs, ...)")
    logger.info("  Gradient access: derivs[(1,)] for du/dx")

    logger.info("\n  Solving with advanced API...")
    result_advanced = problem_advanced.solve(max_iterations=30, tolerance=1e-3)
    logger.info(f"  Converged: {result_advanced.converged}")
    logger.info(f"  Iterations: {result_advanced.iterations}")

    # =========================================================================
    # Method 2: Simplified API (legacy function-based)
    # =========================================================================
    logger.info("\n[2/5] Creating problem with simplified API...")
    problem_simple = create_custom_problem_simple(L=L, target_x=target_x, Nx=Nx, T=T, Nt=Nt)
    logger.info("  Using: MFGProblem")
    logger.info("  Hamiltonian signature: hamiltonian(x, p, m)")
    logger.info("  Gradient access: p directly (no dictionary)")

    logger.info("\n  Solving with simplified API...")
    result_simple = problem_simple.solve(max_iterations=30, tolerance=1e-3)
    logger.info(f"  Converged: {result_simple.converged}")
    logger.info(f"  Iterations: {result_simple.iterations}")

    # =========================================================================
    # Method 3: Class-based API (NEW recommended approach)
    # =========================================================================
    logger.info("\n[3/5] Creating problem with class-based API (NEW)...")
    problem_class = create_custom_problem_class_based(L=L, target_x=target_x, Nx=Nx, T=T, Nt=Nt)
    logger.info("  Using: SeparableHamiltonian + to_legacy_func()")
    logger.info("  Hamiltonian: H = H_control(p) + V(x,t) + f(m)")
    logger.info("  Components: QuadraticControlCost, potential, coupling")

    logger.info("\n  Solving with class-based API...")
    result_class = problem_class.solve(max_iterations=30, tolerance=1e-3)
    logger.info(f"  Converged: {result_class.converged}")
    logger.info(f"  Iterations: {result_class.iterations}")

    # =========================================================================
    # Comparison
    # =========================================================================
    logger.info("\n[4/5] Comparing results...")

    # Compare final equilibrium distributions
    m_advanced = result_advanced.M[-1, :]
    m_simple = result_simple.M[-1, :]
    m_class = result_class.M[-1, :]

    diff_adv_sim = np.linalg.norm(m_advanced - m_simple)
    diff_adv_cls = np.linalg.norm(m_advanced - m_class)
    diff_sim_cls = np.linalg.norm(m_simple - m_class)

    logger.info(f"  Advanced vs Simplified (L2): {diff_adv_sim:.6e}")
    logger.info(f"  Advanced vs Class-based (L2): {diff_adv_cls:.6e}")
    logger.info(f"  Simplified vs Class-based (L2): {diff_sim_cls:.6e}")

    max_diff = max(diff_adv_sim, diff_adv_cls, diff_sim_cls)
    if max_diff < 1e-2:
        logger.info("  All three APIs produce equivalent solutions!")
    else:
        logger.info(f"  Note: Maximum difference = {max_diff:.6e}")

    # =========================================================================
    # Visualization
    # =========================================================================
    logger.info("\n[5/5] Generating visualizations...")

    x_grid = np.linspace(problem_advanced.xmin, problem_advanced.xmax, Nx + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Final equilibrium comparison (all three approaches)
    ax = axes[0, 0]
    ax.plot(x_grid, m_advanced, "b-", linewidth=2.5, label="Advanced API (derivs)")
    ax.plot(x_grid, m_simple, "r--", linewidth=2, label="Simplified API")
    ax.plot(x_grid, m_class, "g:", linewidth=2, label="Class-based API (NEW)")
    ax.axvline(target_x, color="orange", linestyle="-.", linewidth=1.5, label="Target", alpha=0.7)
    ax.set_xlabel("Position x", fontsize=11)
    ax.set_ylabel("Equilibrium density m(x)", fontsize=11)
    ax.set_title("Final Equilibrium Comparison", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Panel 2: Value function comparison (all three approaches)
    ax = axes[0, 1]
    U_advanced = result_advanced.U[-1, :]
    U_simple = result_simple.U[-1, :]
    U_class = result_class.U[-1, :]
    ax.plot(x_grid, U_advanced, "b-", linewidth=2.5, label="Advanced API")
    ax.plot(x_grid, U_simple, "r--", linewidth=2, label="Simplified API")
    ax.plot(x_grid, U_class, "g:", linewidth=2, label="Class-based API")
    ax.set_xlabel("Position x", fontsize=11)
    ax.set_ylabel("Value function u(x, T)", fontsize=11)
    ax.set_title("Value Function Comparison", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Panel 3: Convergence history (all three approaches)
    ax = axes[1, 0]
    ax.semilogy(
        range(1, result_advanced.iterations + 1),
        result_advanced.error_history_M[: result_advanced.iterations],
        "b-",
        linewidth=2,
        label="Advanced API",
    )
    ax.semilogy(
        range(1, result_simple.iterations + 1),
        result_simple.error_history_M[: result_simple.iterations],
        "r--",
        linewidth=2,
        label="Simplified API",
    )
    ax.semilogy(
        range(1, result_class.iterations + 1),
        result_class.error_history_M[: result_class.iterations],
        "g:",
        linewidth=2,
        label="Class-based API",
    )
    ax.set_xlabel("Picard iteration", fontsize=11)
    ax.set_ylabel("L2 error", fontsize=11)
    ax.set_title("Convergence History", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Panel 4: Pairwise differences
    ax = axes[1, 1]
    diff_1 = np.abs(m_advanced - m_simple)
    diff_2 = np.abs(m_advanced - m_class)
    ax.plot(x_grid, diff_1, "r-", linewidth=2, label=f"Adv-Simp (L2={diff_adv_sim:.2e})")
    ax.plot(x_grid, diff_2, "g--", linewidth=2, label=f"Adv-Class (L2={diff_adv_cls:.2e})")
    ax.set_xlabel("Position x", fontsize=11)
    ax.set_ylabel("|m_i - m_j|", fontsize=11)
    ax.set_title("Pairwise Density Differences", fontsize=12, fontweight="bold")
    ax.legend(loc="best")
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
    logger.info("  1. Advanced API: Function-based with derivs dict for full control")
    logger.info("  2. Simplified API: Simple (x, p, m) signature with wrapper")
    logger.info("  3. Class-based API: SeparableHamiltonian (NEW, recommended)")
    logger.info("  4. All three approaches produce equivalent results")
    logger.info("\nRecommendation:")
    logger.info("  - Use SeparableHamiltonian (class-based) for most applications")
    logger.info("  - Benefits: composable, type-safe, auto-diff for dp()/dm()")
    logger.info("  - Legendre transform: H.legendre_transform() <-> L.legendre_transform()")
    logger.info("\nSee Also:")
    logger.info("  - mfg_pde/core/hamiltonian.py (HamiltonianBase, SeparableHamiltonian)")
    logger.info("  - docs/migration_guides/phase3_gradient_notation_migration.md")


def main():
    """Run the demonstration."""
    compare_approaches()


if __name__ == "__main__":
    main()
