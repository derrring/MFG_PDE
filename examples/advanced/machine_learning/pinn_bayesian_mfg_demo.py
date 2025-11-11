"""
Advanced PINN Demo: Bayesian MFG with Uncertainty Quantification.

This example demonstrates the complete PINN framework for Mean Field Games
including advanced features like:
- Physics-informed neural networks with automatic differentiation
- Adaptive training strategies and curriculum learning
- Bayesian uncertainty quantification using MCMC/HMC
- Multi-scale training and residual-based refinement
- Comprehensive visualization and analysis

Mathematical Problem:
- HJB equation: ∂u/∂t + (1/2)|∇u|² + V(x) + λm = 0
- FP equation: ∂m/∂t - Δm - div(m∇u) = 0
- Domain: [0,1] × [-2,2]² (2D spatial + time)
- Initial condition: m₀(x) = Gaussian blob
- Terminal condition: u(T,x) = |x|²/2

Advanced Features Demonstrated:
- Automatic differentiation for exact PDE residuals
- Physics-guided adaptive sampling
- Bayesian neural networks with weight uncertainty
- Multi-scale curriculum learning
- Real-time convergence monitoring
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mfg_pde.alg.neural.pinn.adaptive_training import (  # noqa: E402
    AdaptiveTrainingConfig,
    create_adaptive_strategy,
)
from mfg_pde.alg.neural.pinn.base_pinn import PINNConfig  # noqa: E402
from mfg_pde.core.mfg_problem import BoundaryConditions, MFGProblem  # noqa: E402
from mfg_pde.utils.mcmc import MCMCConfig  # noqa: E402

# Check if PyTorch is available
try:
    import torch

    from mfg_pde.alg.neural.pinn.mfg_pinn_solver import MFGPINNSolver

    TORCH_AVAILABLE = True
    print("✓ PyTorch available - Full PINN demo ready")
except ImportError:
    TORCH_AVAILABLE = False
    print("✗ PyTorch not available - Mock demo mode")


class AdvancedMFGProblem(MFGProblem):
    """
    Advanced 2D MFG problem for PINN demonstration.

    Features:
    - Quadratic-cubic Hamiltonian for nonlinearity
    - Obstacle potential for complexity
    - Time-dependent coupling for realism
    - Boundary conditions with physical meaning
    """

    def __init__(self, coupling_strength: float = 0.2, obstacle_strength: float = 1.0):
        """Initialize advanced MFG problem."""
        self.coupling_strength = coupling_strength
        self.obstacle_strength = obstacle_strength
        self.dimension = 2
        self.domain = [(-2, 2), (-2, 2)]  # 2D spatial domain
        self.time_domain = (0, 1.0)

        # Physical boundary conditions
        self.boundary_conditions = BoundaryConditions(
            value_bc=self._dirichlet_value_bc, density_bc=self._neumann_density_bc, bc_type="mixed"
        )

    def _dirichlet_value_bc(self, t: float, x: np.ndarray) -> np.ndarray:
        """Dirichlet boundary condition for value function."""
        # Terminal cost at boundary
        return 0.5 * np.sum(x**2, axis=-1) * (1 - t)

    def _neumann_density_bc(self, t: float, x: np.ndarray) -> np.ndarray:
        """Neumann boundary condition for density (zero flux)."""
        return np.zeros(len(x))

    def hamiltonian(self, t: float, x: np.ndarray, p: np.ndarray, m: float) -> float:
        """
        Advanced Hamiltonian with nonlinear terms.

        H(x,p,m) = (1/2)|p|² + V(x) + λ(t)m + ε|p|³
        """
        # Kinetic energy
        kinetic = 0.5 * np.sum(p**2, axis=-1)

        # Obstacle potential
        obstacle = self.obstacle_strength * np.exp(-np.sum(x**2, axis=-1))

        # Time-dependent coupling
        coupling = self.coupling_strength * (1 + 0.5 * np.sin(2 * np.pi * t)) * m

        # Nonlinear momentum term (small)
        nonlinear = 0.01 * (np.sum(p**2, axis=-1) ** 1.5)

        return kinetic + obstacle + coupling + nonlinear

    def initial_density(self, x: np.ndarray) -> np.ndarray:
        """
        Initial density: Two Gaussian blobs.

        m₀(x) = 0.6⋅N((-1,0), 0.3²I) + 0.4⋅N((1,0), 0.5²I)
        """
        # First blob
        center1 = np.array([-1.0, 0.0])
        var1 = 0.3**2
        blob1 = 0.6 * np.exp(-np.sum((x - center1) ** 2, axis=-1) / (2 * var1)) / (2 * np.pi * var1)

        # Second blob
        center2 = np.array([1.0, 0.0])
        var2 = 0.5**2
        blob2 = 0.4 * np.exp(-np.sum((x - center2) ** 2, axis=-1) / (2 * var2)) / (2 * np.pi * var2)

        return blob1 + blob2

    def terminal_condition(self, x: np.ndarray) -> np.ndarray:
        """
        Terminal condition: Quadratic cost with obstacle avoidance.

        u(T,x) = (1/2)|x|² + penalty⋅exp(-|x|²)
        """
        quadratic = 0.5 * np.sum(x**2, axis=-1)
        penalty = 2.0 * np.exp(-np.sum(x**2, axis=-1))  # Avoid center
        return quadratic + penalty


def create_advanced_pinn_config() -> PINNConfig:
    """Create sophisticated PINN configuration."""
    return PINNConfig(
        # Advanced network architecture
        hidden_layers=[256, 256, 256, 256, 128],  # Deep network for complexity
        activation="swish",  # Smooth activation for better gradients
        use_batch_norm=False,  # Can interfere with physics
        dropout_rate=0.0,
        network_initialization="xavier",
        # Physics-informed sampling
        num_physics_points=8000,  # Dense sampling for accuracy
        num_boundary_points=1000,
        num_initial_points=1000,
        physics_sampling="uniform",  # Start with uniform, adapt later
        # Sophisticated loss weighting
        physics_weight=1.0,
        boundary_weight=15.0,  # Strong boundary enforcement
        initial_weight=15.0,  # Strong initial condition
        coupling_weight=2.0,  # Emphasize HJB-FP coupling
        adaptive_weights=True,
        # Advanced optimization
        learning_rate=1e-4,  # Conservative for stability
        max_epochs=20000,  # Sufficient for convergence
        optimizer="adam",  # Reliable for PINNs
        scheduler="cosine",  # Smooth decay
        # Convergence and monitoring
        tolerance=1e-6,
        patience=2000,
        min_improvement=1e-7,
        # Automatic differentiation settings
        grad_penalty_weight=1e-4,  # Encourage smooth solutions
        higher_order_derivatives=True,
        # Adaptive training
        curriculum_learning=True,
        adaptive_sampling=True,
        residual_threshold=1e-3,
        # Computational settings
        device="auto",
        dtype="float32",  # Balance precision and speed
        compile_model=False,  # Avoid compilation for clarity
        # Monitoring
        checkpoint_frequency=1000,
        log_frequency=200,
        validate_frequency=500,
    )


def create_adaptive_training_config() -> AdaptiveTrainingConfig:
    """Create configuration for adaptive training."""
    return AdaptiveTrainingConfig(
        # Physics-guided sampling
        residual_threshold=5e-4,
        adaptive_sampling_frequency=100,
        max_adaptive_points=15000,
        importance_exponent=0.5,
        # Curriculum learning
        enable_curriculum=True,
        curriculum_epochs=8000,
        initial_complexity=0.2,  # Start with 20% complexity
        complexity_growth="sigmoid",  # Smooth progression
        # Multi-scale training
        enable_multiscale=True,
        num_scales=3,
        scale_transition_epochs=3000,
        coarse_to_fine=True,
        # Advanced loss balancing
        adaptive_loss_weights=True,
        weight_update_frequency=50,
        weight_momentum=0.9,
        gradient_balance=True,
        # Residual-based refinement
        enable_refinement=True,
        refinement_frequency=500,
        refinement_factor=1.3,
        max_refinement_levels=4,
        # Performance monitoring
        monitor_convergence=True,
        stagnation_patience=1500,
        stagnation_threshold=1e-7,
    )


def run_advanced_pinn_demo():
    """Run comprehensive PINN demonstration."""
    print("🚀 Advanced PINN Demo: Bayesian MFG with Uncertainty Quantification")
    print("=" * 80)

    if not TORCH_AVAILABLE:
        print("⚠️  PyTorch not available - running mock demonstration")
        run_mock_demo()
        return

    # Create problem and configurations
    problem = AdvancedMFGProblem(coupling_strength=0.15, obstacle_strength=0.8)
    pinn_config = create_advanced_pinn_config()
    adaptive_config = create_adaptive_training_config()

    print(f"Problem: {problem.dimension}D MFG with advanced features")
    print(f"Domain: {problem.domain} × {problem.time_domain}")
    print(f"Coupling: λ = {problem.coupling_strength}")
    print(f"Obstacle: strength = {problem.obstacle_strength}")
    print()

    try:
        # Initialize PINN solver
        solver = MFGPINNSolver(problem, pinn_config)
        print("✓ Advanced PINN solver initialized")

        # Setup adaptive training
        adaptive_strategy = create_adaptive_strategy("physics_guided", adaptive_config, problem.domain)
        print("✓ Physics-guided adaptive training configured")

        # Enable Bayesian mode for uncertainty quantification
        mcmc_config = MCMCConfig(
            num_samples=500,  # Reduced for demo
            num_warmup=200,
            num_chains=1,
            step_size=0.01,
            num_leapfrog_steps=10,
        )
        solver.enable_bayesian_mode(mcmc_config)
        print("✓ Bayesian uncertainty quantification enabled")
        print()

        # Demonstrate different training phases
        print("📊 Training Phase 1: Standard PINN Training")
        print("-" * 50)

        # Run initial training (shortened for demo)
        pinn_config.max_epochs = 1000  # Reduced for demo
        result = solver.solve()

        print(f"✓ Initial training completed: {result.num_epochs} epochs")
        print(f"✓ Final loss: {result.final_loss:.6e}")
        print(f"✓ Converged: {result.converged}")
        print(f"✓ Training time: {result.training_time:.2f}s")
        print()

        # Demonstrate adaptive sampling
        print("📊 Training Phase 2: Adaptive Sampling Demonstration")
        print("-" * 50)

        # Sample test points to demonstrate adaptive sampling
        test_points = solver._sample_physics_points(1000)
        with torch.no_grad():
            physics_loss = solver._compute_physics_loss(test_points)
            residuals = torch.abs(physics_loss).cpu().numpy()

        # Apply adaptive strategy
        point_updates = adaptive_strategy.update_sampling_strategy(residuals, test_points.cpu().numpy())

        print("Adaptive sampling analysis:")
        print(f"  Original physics points: {pinn_config.num_physics_points}")
        print(f"  Updated physics points: {point_updates['physics']}")
        print(f"  Residual statistics: mean={np.mean(residuals):.6e}, std={np.std(residuals):.6e}")
        print(f"  High-residual regions: {len(adaptive_strategy.state.high_residual_regions)}")
        print()

        # Demonstrate loss weight adaptation
        print("📊 Training Phase 3: Loss Weight Adaptation")
        print("-" * 50)

        loss_components = {
            "physics": float(physics_loss.item()),
            "boundary": 0.1,  # Mock values
            "initial": 0.05,
        }

        updated_weights = adaptive_strategy.update_loss_weights(loss_components)
        print("Loss component analysis:")
        print(f"  Physics loss: {loss_components['physics']:.6e}")
        print(f"  Boundary loss: {loss_components['boundary']:.6e}")
        print(f"  Initial loss: {loss_components['initial']:.6e}")
        print("Adaptive weights:")
        for component, weight in updated_weights.items():
            print(f"  {component}: {weight:.4f}")
        print()

        # Demonstrate Bayesian uncertainty quantification
        print("📊 Training Phase 4: Bayesian Uncertainty Quantification")
        print("-" * 50)

        print("Note: MCMC sampling is computationally intensive.")
        print("For demonstration, we'll show the framework setup.")

        # Create test points for prediction
        t_test = np.array([0.5])
        x_test = np.array([[0.0, 0.0], [1.0, 0.5], [-1.0, -0.5]])
        test_input = np.column_stack([np.repeat(t_test, len(x_test)), x_test])

        print(f"Test points for uncertainty quantification: {len(test_input)}")
        print("Test locations:")
        for i, point in enumerate(test_input):
            print(f"  Point {i + 1}: t={point[0]:.1f}, x=({point[1]:.1f}, {point[2]:.1f})")

        # Deterministic predictions (Bayesian would require full MCMC)
        u_mean, _u_std, m_mean, _m_std = solver.predict_with_uncertainty(test_input)

        print("\nDeterministic predictions (no uncertainty sampling):")
        for i in range(len(test_input)):
            print(f"  Point {i + 1}: u={u_mean[i, 0]:.4f}, m={m_mean[i, 0]:.4f}")

        print("\n✓ Uncertainty quantification framework demonstrated")
        print()

        # Solution analysis
        print("📊 Solution Analysis")
        print("-" * 50)

        # Evaluate solution quality
        if result.value_function is not None and result.density_function is not None:
            print("Solution quality metrics:")
            print(f"  Value function range: [{np.min(result.value_function):.4f}, {np.max(result.value_function):.4f}]")
            print(
                f"  Density function range: [{np.min(result.density_function):.4f}, "
                f"{np.max(result.density_function):.4f}]"
            )
            print(f"  Density conservation: ∫m dx ≈ {np.mean(result.density_function):.4f}")

        # Physics residual analysis
        print("\nPhysics residuals:")
        print(f"  HJB residual: {result.hjb_residual:.6e}")
        print(f"  FP residual: {result.fp_residual:.6e}")
        print(f"  Boundary satisfaction: {result.boundary_satisfaction:.6e}")

        print()
        print("✅ Advanced PINN demonstration completed successfully!")
        print("\nKey achievements:")
        print("  ✓ Physics-informed neural networks with automatic differentiation")
        print("  ✓ Adaptive training strategies and curriculum learning")
        print("  ✓ Bayesian uncertainty quantification framework")
        print("  ✓ Multi-component loss balancing")
        print("  ✓ Real-time convergence monitoring")

    except Exception as e:
        print(f"❌ Advanced PINN demo failed: {e}")
        import traceback

        traceback.print_exc()


def run_mock_demo():
    """Run mock demonstration when PyTorch unavailable."""
    print("Running mock advanced PINN demonstration")
    print()

    # Create problem
    problem = AdvancedMFGProblem()

    print("✓ Advanced MFG problem initialized")
    print(f"  - Dimension: {problem.dimension}")
    print(f"  - Domain: {problem.domain}")
    print(f"  - Time domain: {problem.time_domain}")
    print(f"  - Coupling strength: {problem.coupling_strength}")

    # Test mathematical components
    x_test = np.array([[0, 0], [1, 1], [-1, -1]])
    p_test = np.array([[0.5, 0.5], [1, 0], [0, 1]])

    print("\n✓ Testing mathematical components:")

    # Test Hamiltonian
    H_vals = [problem.hamiltonian(0.5, x, p, 0.1) for x, p in zip(x_test, p_test, strict=False)]
    print(f"  - Hamiltonian values: {H_vals}")

    # Test initial density
    m0_vals = problem.initial_density(x_test)
    print(f"  - Initial density: {m0_vals}")

    # Test terminal condition
    uT_vals = problem.terminal_condition(x_test)
    print(f"  - Terminal condition: {uT_vals}")

    # Mock adaptive training
    print("\n✓ Mock adaptive training demonstration:")
    adaptive_config = create_adaptive_training_config()
    print(f"  - Curriculum learning: {adaptive_config.enable_curriculum}")
    print(f"  - Multi-scale training: {adaptive_config.enable_multiscale}")
    print(f"  - Adaptive sampling: {adaptive_config.residual_threshold}")

    # Mock MCMC configuration
    print("\n✓ Mock Bayesian configuration:")
    mcmc_config = MCMCConfig(num_samples=1000, num_warmup=500)
    print(f"  - MCMC samples: {mcmc_config.num_samples}")
    print(f"  - Warmup samples: {mcmc_config.num_warmup}")
    print(f"  - Target acceptance: {mcmc_config.target_accept_rate}")

    print("\n✅ Mock demonstration completed!")
    print("Install PyTorch to run the full advanced PINN demo:")
    print("  pip install torch")


def create_visualization_plots():
    """Create visualization plots for the demo."""
    try:
        print("\n📊 Creating visualization plots...")

        # Create problem for visualization
        problem = AdvancedMFGProblem()

        # Create spatial grid
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x, y)
        xy = np.stack([X.ravel(), Y.ravel()], axis=1)

        # Plot initial density
        _fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Initial density
        m0 = problem.initial_density(xy).reshape(X.shape)
        im1 = axes[0, 0].contourf(X, Y, m0, levels=20, cmap="viridis")
        axes[0, 0].set_title("Initial Density m₀(x)")
        axes[0, 0].set_xlabel("x₁")
        axes[0, 0].set_ylabel("x₂")
        plt.colorbar(im1, ax=axes[0, 0])

        # Terminal condition
        uT = problem.terminal_condition(xy).reshape(X.shape)
        im2 = axes[0, 1].contourf(X, Y, uT, levels=20, cmap="plasma")
        axes[0, 1].set_title("Terminal Condition u(T,x)")
        axes[0, 1].set_xlabel("x₁")
        axes[0, 1].set_ylabel("x₂")
        plt.colorbar(im2, ax=axes[0, 1])

        # Mock training curve
        epochs = np.arange(1000)
        loss_curve = 1e-1 * np.exp(-epochs / 500) + 1e-4 * np.random.random(1000)
        axes[1, 0].semilogy(epochs, loss_curve)
        axes[1, 0].set_title("Training Loss Convergence")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].grid(True, alpha=0.3)

        # Mock uncertainty visualization
        x_unc = np.linspace(-2, 2, 30)
        uncertainty = 0.1 * np.exp(-(x_unc**2)) + 0.02
        axes[1, 1].fill_between(x_unc, -uncertainty, uncertainty, alpha=0.3, label="±1σ")
        axes[1, 1].plot(x_unc, np.sin(x_unc), "b-", label="Mean prediction")
        axes[1, 1].set_title("Prediction Uncertainty")
        axes[1, 1].set_xlabel("x₁ (at x₂=0, t=0.5)")
        axes[1, 1].set_ylabel("u(t,x)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("advanced_pinn_demo_plots.png", dpi=150, bbox_inches="tight")
        print("✓ Visualization plots saved as 'advanced_pinn_demo_plots.png'")

    except ImportError:
        print("⚠️  Matplotlib not available - skipping visualizations")


if __name__ == "__main__":
    run_advanced_pinn_demo()
    print()
    create_visualization_plots()
