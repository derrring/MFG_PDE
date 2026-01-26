"""
Comprehensive demonstration of Physics-Informed Neural Networks for Mean Field Games.

This example showcases the complete PINN framework for solving MFG systems:
1. Setup of a 1D MFG problem with known analytical properties
2. Training separate HJB and FP PINN solvers
3. Training coupled MFG PINN solver
4. Comparison of results and analysis
5. Visualization and validation

The problem setup includes:
- Domain: x ∈ [0, 1], t ∈ [0, T]
- HJB: ∂u/∂t + (1/2)|∇u|² + log(m) = 0
- FP: ∂m/∂t - div(m∇u) - (σ²/2)Δm = 0
- Terminal condition: u(T, x) = (x - 0.5)²
- Initial density: m(0, x) = normalized Gaussian
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add the package root to the Python path for imports
package_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_root))

from mfg_pde.alg.neural.pinn_solvers import (  # noqa: E402
    FPPINNSolver,
    HJBPINNSolver,
    MFGPINNSolver,
    PINNConfig,
)
from mfg_pde.core.mfg_problem import MFGProblem  # noqa: E402
from mfg_pde.geometry import TensorProductGrid  # noqa: E402
from mfg_pde.geometry.boundary import no_flux_bc  # noqa: E402

try:
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None:
        TORCH_AVAILABLE = True
        print("PyTorch available for PINN demonstration")
    else:
        raise ImportError("PyTorch not found")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Please install PyTorch to run PINN examples.")
    sys.exit(1)


def create_mfg_problem() -> MFGProblem:
    """
    Create a 1D MFG problem for PINN demonstration.

    Returns:
        Configured MFG problem
    """

    def terminal_condition(x: np.ndarray) -> np.ndarray:
        """Terminal cost function: g(x) = (x - 0.5)²"""
        return (x - 0.5) ** 2

    def initial_density(x: np.ndarray) -> np.ndarray:
        """Initial density: normalized Gaussian centered at 0.3"""
        gaussian = np.exp(-50 * (x - 0.3) ** 2)
        # Normalize to ensure ∫m₀ dx = 1
        return gaussian / np.trapz(gaussian, x)

    # Create MFG problem with Geometry-First API
    geometry = TensorProductGrid(
        dimension=1, bounds=[(0.0, 1.0)], Nx_points=[65], boundary_conditions=no_flux_bc(dimension=1)
    )
    problem = MFGProblem(geometry=geometry, T=1.0, diffusion=0.1)

    # Set terminal and initial conditions
    problem.terminal_condition = terminal_condition
    problem.initial_density = initial_density

    return problem


def create_pinn_config(training_type: str = "standard") -> PINNConfig:
    """
    Create PINN configuration for different training scenarios.

    Args:
        training_type: "fast", "standard", or "thorough"

    Returns:
        PINN configuration
    """
    if training_type == "fast":
        # Quick training for testing
        return PINNConfig(
            hidden_layers=[32, 32],
            activation="tanh",
            learning_rate=1e-3,
            max_epochs=2000,
            n_interior_points=1000,
            n_boundary_points=100,
            n_initial_points=100,
            log_interval=200,
            convergence_tolerance=1e-4,
        )
    elif training_type == "standard":
        # Balanced training
        return PINNConfig(
            hidden_layers=[50, 50, 50],
            activation="tanh",
            learning_rate=1e-3,
            max_epochs=5000,
            n_interior_points=2000,
            n_boundary_points=200,
            n_initial_points=200,
            log_interval=500,
            convergence_tolerance=1e-5,
        )
    elif training_type == "thorough":
        # High-quality training
        return PINNConfig(
            hidden_layers=[100, 100, 100, 100],
            activation="tanh",
            learning_rate=5e-4,
            max_epochs=10000,
            n_interior_points=5000,
            n_boundary_points=500,
            n_initial_points=500,
            log_interval=500,
            convergence_tolerance=1e-6,
            early_stopping_patience=1000,
        )
    else:
        raise ValueError(f"Unknown training type: {training_type}")


def demonstrate_hjb_pinn(problem: MFGProblem, config: PINNConfig) -> dict:
    """
    Demonstrate HJB PINN solver.

    Args:
        problem: MFG problem
        config: PINN configuration

    Returns:
        HJB solution results
    """
    print("\n" + "=" * 60)
    print("HAMILTON-JACOBI-BELLMAN PINN DEMONSTRATION")
    print("=" * 60)

    # Create HJB solver
    hjb_solver = HJBPINNSolver(problem, config)

    print(f"Network architecture: {config.hidden_layers}")
    print(f"Total parameters: {sum(p.numel() for p in hjb_solver.u_net.parameters()):,}")

    # Solve HJB equation
    print("\nTraining HJB PINN...")
    hjb_results = hjb_solver.solve(verbose=True)

    # Evaluate solution quality
    quality_metrics = hjb_solver.evaluate_solution_quality()
    print("\nHJB Solution Quality Metrics:")
    for key, value in quality_metrics.items():
        print(f"  {key}: {value:.6f}")

    # Plot results
    print("\nPlotting HJB solution...")
    hjb_solver.plot_solution(save_path="hjb_pinn_solution.png")

    return hjb_results


def demonstrate_fp_pinn(problem: MFGProblem, config: PINNConfig) -> dict:
    """
    Demonstrate Fokker-Planck PINN solver.

    Args:
        problem: MFG problem
        config: PINN configuration

    Returns:
        FP solution results
    """
    print("\n" + "=" * 60)
    print("FOKKER-PLANCK PINN DEMONSTRATION")
    print("=" * 60)

    # Create FP solver
    fp_solver = FPPINNSolver(problem, config)

    print(f"Network architecture: {config.hidden_layers}")
    print(f"Total parameters: {sum(p.numel() for p in fp_solver.m_net.parameters()):,}")

    # Solve FP equation
    print("\nTraining FP PINN...")
    fp_results = fp_solver.solve(verbose=True)

    # Evaluate solution quality
    quality_metrics = fp_solver.evaluate_solution_quality()
    print("\nFP Solution Quality Metrics:")
    for key, value in quality_metrics.items():
        print(f"  {key}: {value:.6f}")

    # Plot results
    print("\nPlotting FP solution...")
    fp_solver.plot_solution(save_path="fp_pinn_solution.png")

    return fp_results


def demonstrate_coupled_mfg_pinn(problem: MFGProblem, config: PINNConfig, alternating_training: bool = False) -> dict:
    """
    Demonstrate coupled MFG PINN solver.

    Args:
        problem: MFG problem
        config: PINN configuration
        alternating_training: Whether to use alternating training

    Returns:
        MFG solution results
    """
    training_method = "Alternating" if alternating_training else "Joint"
    print("\n" + "=" * 60)
    print(f"COUPLED MFG PINN DEMONSTRATION ({training_method} Training)")
    print("=" * 60)

    # Create MFG solver
    mfg_solver = MFGPINNSolver(problem, config, alternating_training=alternating_training)

    u_params = sum(p.numel() for p in mfg_solver.u_net.parameters())
    m_params = sum(p.numel() for p in mfg_solver.m_net.parameters())
    print(f"u-network parameters: {u_params:,}")
    print(f"m-network parameters: {m_params:,}")
    print(f"Total parameters: {u_params + m_params:,}")

    # Solve coupled MFG system
    print(f"\nTraining coupled MFG PINN ({training_method})...")
    mfg_results = mfg_solver.solve(verbose=True)

    # Evaluate solution quality
    quality_metrics = mfg_solver.evaluate_mfg_quality()
    print("\nMFG Solution Quality Metrics:")
    for key, value in quality_metrics.items():
        print(f"  {key}: {value:.6f}")

    # Plot results
    print("\nPlotting MFG solution...")
    save_path = f"mfg_pinn_solution_{'alternating' if alternating_training else 'joint'}.png"
    mfg_solver.plot_solution(save_path=save_path)

    return mfg_results


def compare_pinn_approaches(
    hjb_results: dict, fp_results: dict, mfg_joint_results: dict, mfg_alt_results: dict | None = None
) -> None:
    """
    Compare different PINN approaches.

    Args:
        hjb_results: HJB PINN results
        fp_results: FP PINN results
        mfg_joint_results: Joint MFG PINN results
        mfg_alt_results: Alternating MFG PINN results (optional)
    """
    print("\n" + "=" * 60)
    print("PINN APPROACHES COMPARISON")
    print("=" * 60)

    # Training convergence comparison
    _fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # HJB training history
    ax = axes[0, 0]
    hjb_history = hjb_results["training_history"]
    ax.semilogy(hjb_history["total_loss"], label="Total Loss", color="blue")
    ax.semilogy(hjb_history["pde_loss"], label="PDE Loss", color="red", alpha=0.7)
    ax.set_title("HJB PINN Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    # FP training history
    ax = axes[0, 1]
    fp_history = fp_results["training_history"]
    ax.semilogy(fp_history["total_loss"], label="Total Loss", color="green")
    ax.semilogy(fp_history["pde_loss"], label="PDE Loss", color="orange", alpha=0.7)
    if "mass_conservation" in fp_history:
        ax.semilogy(fp_history["mass_conservation"], label="Mass Loss", color="purple", alpha=0.7)
    ax.set_title("FP PINN Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    # Joint MFG training history
    ax = axes[1, 0]
    mfg_history = mfg_joint_results["training_history"]
    ax.semilogy(mfg_history["total_loss"], label="Total Loss", color="black")
    if "hjb_loss" in mfg_history:
        ax.semilogy(mfg_history["hjb_loss"], label="HJB Loss", color="blue", alpha=0.7)
    if "fp_loss" in mfg_history:
        ax.semilogy(mfg_history["fp_loss"], label="FP Loss", color="red", alpha=0.7)
    ax.set_title("Joint MFG PINN Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    # Solution comparison
    ax = axes[1, 1]

    # Compare final loss values
    methods = ["HJB", "FP", "Joint MFG"]
    final_losses = [
        hjb_results["training_history"]["total_loss"][-1],
        fp_results["training_history"]["total_loss"][-1],
        mfg_joint_results["training_history"]["total_loss"][-1],
    ]

    if mfg_alt_results:
        methods.append("Alt MFG")
        final_losses.append(mfg_alt_results["training_history"]["total_loss"][-1])

    bars = ax.bar(methods, final_losses, color=["blue", "green", "red", "orange"][: len(methods)])
    ax.set_yscale("log")
    ax.set_title("Final Loss Comparison")
    ax.set_ylabel("Final Loss")

    # Add value labels on bars
    for bar, loss in zip(bars, final_losses, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.1,
            f"{loss:.2e}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig("pinn_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"HJB Final Loss: {final_losses[0]:.6f}")
    print(f"FP Final Loss: {final_losses[1]:.6f}")
    print(f"Joint MFG Final Loss: {final_losses[2]:.6f}")
    if len(final_losses) > 3:
        print(f"Alternating MFG Final Loss: {final_losses[3]:.6f}")

    print("\nTraining Epochs:")
    print(f"HJB: {len(hjb_results['training_history']['total_loss'])}")
    print(f"FP: {len(fp_results['training_history']['total_loss'])}")
    print(f"Joint MFG: {len(mfg_joint_results['training_history']['total_loss'])}")
    if mfg_alt_results:
        print(f"Alternating MFG: {len(mfg_alt_results['training_history']['total_loss'])}")


def main():
    """Main demonstration function."""
    print("Physics-Informed Neural Networks for Mean Field Games")
    print("=" * 60)

    # Create problem and configuration
    print("\nSetting up MFG problem...")
    problem = create_mfg_problem()
    config = create_pinn_config("fast")  # Use "standard" or "thorough" for better results

    bounds = problem.geometry.get_bounds()
    print(f"Problem domain: x ∈ [{bounds[0][0]}, {bounds[1][0]}], t ∈ [0, {problem.T}]")
    print(f"Diffusion coefficient: σ = {problem.diffusion}")
    print(f"PINN configuration: {config.max_epochs} epochs, {config.hidden_layers} architecture")

    # Demonstrate individual solvers
    print("\n" + "=" * 80)
    print("INDIVIDUAL PINN SOLVER DEMONSTRATIONS")
    print("=" * 80)

    try:
        # 1. HJB PINN Solver
        hjb_results = demonstrate_hjb_pinn(problem, config)

        # 2. FP PINN Solver
        fp_results = demonstrate_fp_pinn(problem, config)

        # 3. Coupled MFG PINN Solver (Joint Training)
        mfg_joint_results = demonstrate_coupled_mfg_pinn(problem, config, alternating_training=False)

        # 4. Coupled MFG PINN Solver (Alternating Training) - Optional for fast demo
        mfg_alt_results = None
        if config.max_epochs >= 4000:  # Only for longer training
            mfg_alt_results = demonstrate_coupled_mfg_pinn(problem, config, alternating_training=True)

        # Compare approaches
        compare_pinn_approaches(hjb_results, fp_results, mfg_joint_results, mfg_alt_results)

        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nGenerated files:")
        print("- hjb_pinn_solution.png: HJB solution visualization")
        print("- fp_pinn_solution.png: FP solution visualization")
        print("- mfg_pinn_solution_joint.png: Joint MFG solution")
        if mfg_alt_results:
            print("- mfg_pinn_solution_alternating.png: Alternating MFG solution")
        print("- pinn_comparison.png: Training comparison")

        print("\nKey Observations:")
        print("1. PINNs can successfully learn solutions to complex MFG systems")
        print("2. Joint training couples HJB and FP equations naturally")
        print("3. Alternating training can improve convergence for difficult problems")
        print("4. Mass conservation and PDE residuals are key quality metrics")
        print("5. Network architecture and sampling strategy affect solution quality")

    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        print("This may be due to:")
        print("1. PyTorch not properly installed")
        print("2. Insufficient computational resources")
        print("3. Numerical instabilities in training")
        print("\nTry reducing the problem size or adjusting PINN configuration.")
        raise


if __name__ == "__main__":
    main()
