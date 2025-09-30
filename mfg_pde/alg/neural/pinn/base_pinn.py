"""
Base Physics-Informed Neural Network (PINN) solver for MFG problems.

This module implements the abstract base class for PINN solvers, providing
the mathematical and computational framework for physics-informed learning
of Mean Field Games using automatic differentiation.

Mathematical Framework:
- Neural Approximation: u(t,x) ≈ U_θ(t,x), m(t,x) ≈ M_φ(t,x)
- Automatic Differentiation: ∇u = ∂U_θ/∂x, ∂u/∂t = ∂U_θ/∂t
- Physics Residual: R_HJB = ∂u/∂t + H(x,∇u,m), R_FP = ∂m/∂t - div(m∇H_p)
- Loss Function: L = Σᵢ wᵢ||Rᵢ||² + BC + IC penalties

Key Features:
- Automatic differentiation for exact gradient computation
- Multi-task learning for coupled HJB-FP system
- Adaptive loss weighting and curriculum learning
- Transfer learning and model checkpointing
- Advanced optimization strategies (L-BFGS, Adam variants)
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mfg_pde.alg.base_solver import BaseNeuralSolver

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.core.mfg_problem import MFGProblem

# Import with availability checking
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PINNConfig:
    """Configuration for Physics-Informed Neural Network solver."""

    # Network architecture
    hidden_layers: list[int] = None  # [128, 128, 128, 128] default
    activation: str = "tanh"  # "tanh" | "swish" | "sin" | "relu"
    use_batch_norm: bool = False
    dropout_rate: float = 0.0
    network_initialization: str = "xavier"  # "xavier" | "kaiming" | "normal"

    # Physics-informed training
    num_physics_points: int = 5000  # Interior domain points
    num_boundary_points: int = 1000  # Boundary condition points
    num_initial_points: int = 1000  # Initial condition points
    physics_sampling: str = "uniform"  # "uniform" | "sobol" | "adaptive"

    # Loss function weighting
    physics_weight: float = 1.0  # Weight for PDE residual loss
    boundary_weight: float = 10.0  # Weight for boundary condition loss
    initial_weight: float = 10.0  # Weight for initial condition loss
    coupling_weight: float = 1.0  # Weight for HJB-FP coupling
    adaptive_weights: bool = True  # Adaptive loss weighting

    # Training parameters
    learning_rate: float = 1e-3
    max_epochs: int = 50000
    batch_size: int | None = None  # Use all points if None
    optimizer: str = "adam"  # "adam" | "lbfgs" | "rmsprop"
    scheduler: str = "cosine"  # "cosine" | "step" | "none"

    # L-BFGS specific
    lbfgs_max_iter: int = 500
    lbfgs_history_size: int = 100
    lbfgs_tolerance: float = 1e-9

    # Convergence criteria
    tolerance: float = 1e-5
    patience: int = 1000  # Early stopping patience
    min_improvement: float = 1e-6

    # Automatic differentiation
    grad_penalty_weight: float = 0.0  # Gradient penalty regularization
    higher_order_derivatives: bool = True  # Enable second derivatives

    # Adaptive training
    curriculum_learning: bool = True  # Progressive difficulty
    adaptive_sampling: bool = True  # Physics-guided sampling
    residual_threshold: float = 1e-3  # Threshold for adaptive sampling

    # Computational settings
    device: str = "auto"  # "cpu" | "cuda" | "mps" | "auto"
    dtype: str = "float32"  # "float32" | "float64"
    compile_model: bool = False  # PyTorch 2.0 compilation

    # Monitoring and checkpointing
    checkpoint_frequency: int = 1000  # Save model every N epochs
    log_frequency: int = 100  # Log progress every N epochs
    validate_frequency: int = 500  # Validation every N epochs

    def __post_init__(self):
        """Set default hidden layers if not specified."""
        if self.hidden_layers is None:
            self.hidden_layers = [128, 128, 128, 128]


@dataclass
class PINNResult:
    """Result container for PINN solver."""

    # Trained networks
    value_network: Any = None  # Trained U_θ network
    density_network: Any = None  # Trained M_φ network

    # Solution evaluation
    value_function: NDArray | None = None  # u(t,x) on evaluation grid
    density_function: NDArray | None = None  # m(t,x) on evaluation grid

    # Training metrics
    loss_history: list[float] = None
    physics_loss_history: list[float] = None
    boundary_loss_history: list[float] = None
    initial_loss_history: list[float] = None

    # Physics residuals
    hjb_residual: float = np.inf
    fp_residual: float = np.inf
    coupling_residual: float = np.inf

    # Convergence information
    converged: bool = False
    final_loss: float = np.inf
    num_epochs: int = 0
    training_time: float = 0.0

    # PINN-specific metrics
    gradient_norm: float = 0.0
    physics_consistency: float = 0.0  # Physics loss / total loss ratio
    boundary_satisfaction: float = 0.0  # Boundary condition satisfaction

    # Model checkpoints
    best_model_path: str | None = None
    final_model_path: str | None = None


if TORCH_AVAILABLE:

    class BasePINNSolver(BaseNeuralSolver):
        """
        Abstract base class for Physics-Informed Neural Network solvers.

        This class provides the mathematical and computational framework for
        solving MFG problems using physics-informed neural networks with
        automatic differentiation for exact gradient computation.

        Mathematical Approach:
        1. Neural Approximation: u(t,x) ≈ U_θ(t,x), m(t,x) ≈ M_φ(t,x)
        2. Physics Residuals: R_HJB, R_FP computed via automatic differentiation
        3. Loss Function: L = Σᵢ wᵢ||Rᵢ||² + boundary + initial conditions
        4. Optimization: Gradient-based minimization with adaptive strategies

        Key Advantages:
        - Exact gradient computation via automatic differentiation
        - Mesh-free approach with flexible geometries
        - Natural handling of high-dimensional problems
        - Differentiable solutions for sensitivity analysis
        """

        def __init__(
            self,
            problem: MFGProblem,
            config: PINNConfig | None = None,
            **kwargs: Any,
        ):
            """
            Initialize PINN solver.

            Args:
                problem: MFG problem instance
                config: PINN solver configuration
                **kwargs: Additional solver arguments
            """
            super().__init__(problem, **kwargs)
            self.config = config or PINNConfig()
            self.logger = self._get_logger()

            # Initialize device
            self.device = self._get_device()
            self.dtype = getattr(torch, self.config.dtype)

            # Initialize networks
            self._setup_networks()

            # Initialize optimizers
            self._setup_optimizers()

            # Initialize adaptive training components
            self._setup_adaptive_training()

            # Training state
            self.epoch = 0
            self.best_loss = np.inf
            self.loss_history = []

        def _get_device(self) -> torch.device:
            """Get optimal device for computation."""
            if self.config.device == "auto":
                if torch.cuda.is_available():
                    return torch.device("cuda")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return torch.device("mps")
                else:
                    return torch.device("cpu")
            return torch.device(self.config.device)

        @abstractmethod
        def _setup_networks(self) -> None:
            """Setup neural networks for value and density functions."""

        @abstractmethod
        def _setup_optimizers(self) -> None:
            """Setup optimizers and learning rate schedulers."""

        @abstractmethod
        def _setup_adaptive_training(self) -> None:
            """Setup adaptive training components."""

        @abstractmethod
        def _sample_physics_points(self, num_points: int) -> torch.Tensor:
            """
            Sample points for physics loss computation.

            Args:
                num_points: Number of points to sample

            Returns:
                Physics points as torch.Tensor of shape (num_points, dim+1)
            """

        @abstractmethod
        def _sample_boundary_points(self, num_points: int) -> torch.Tensor:
            """
            Sample points for boundary condition loss.

            Args:
                num_points: Number of boundary points

            Returns:
                Boundary points as torch.Tensor
            """

        @abstractmethod
        def _sample_initial_points(self, num_points: int) -> torch.Tensor:
            """
            Sample points for initial condition loss.

            Args:
                num_points: Number of initial points

            Returns:
                Initial points as torch.Tensor
            """

        @abstractmethod
        def _compute_physics_loss(self, points: torch.Tensor) -> torch.Tensor:
            """
            Compute physics residual loss using automatic differentiation.

            Args:
                points: Sample points for physics loss

            Returns:
                Physics loss tensor
            """

        @abstractmethod
        def _compute_boundary_loss(self, points: torch.Tensor) -> torch.Tensor:
            """
            Compute boundary condition loss.

            Args:
                points: Boundary sample points

            Returns:
                Boundary loss tensor
            """

        @abstractmethod
        def _compute_initial_loss(self, points: torch.Tensor) -> torch.Tensor:
            """
            Compute initial condition loss.

            Args:
                points: Initial condition sample points

            Returns:
                Initial loss tensor
            """

        def solve(self) -> PINNResult:
            """
            Solve MFG problem using Physics-Informed Neural Networks.

            Returns:
                PINNResult with trained networks and solution analysis
            """
            self.logger.info(f"Starting PINN solver for {self.problem.dimension}D MFG problem")

            result = PINNResult()
            result.loss_history = []
            result.physics_loss_history = []
            result.boundary_loss_history = []
            result.initial_loss_history = []

            import time

            start_time = time.time()

            try:
                # Training loop
                for epoch in range(self.config.max_epochs):
                    self.epoch = epoch

                    # Sample points for this epoch
                    physics_points = self._sample_physics_points(self.config.num_physics_points)
                    boundary_points = self._sample_boundary_points(self.config.num_boundary_points)
                    initial_points = self._sample_initial_points(self.config.num_initial_points)

                    # Compute individual loss components
                    physics_loss = self._compute_physics_loss(physics_points)
                    boundary_loss = self._compute_boundary_loss(boundary_points)
                    initial_loss = self._compute_initial_loss(initial_points)

                    # Apply adaptive loss weighting if enabled
                    if self.config.adaptive_weights:
                        weights = self._compute_adaptive_weights(physics_loss, boundary_loss, initial_loss)
                        physics_weight, boundary_weight, initial_weight = weights
                    else:
                        physics_weight = self.config.physics_weight
                        boundary_weight = self.config.boundary_weight
                        initial_weight = self.config.initial_weight

                    # Total loss
                    total_loss = (
                        physics_weight * physics_loss + boundary_weight * boundary_loss + initial_weight * initial_loss
                    )

                    # Gradient penalty if enabled
                    if self.config.grad_penalty_weight > 0:
                        grad_penalty = self._compute_gradient_penalty(physics_points)
                        total_loss += self.config.grad_penalty_weight * grad_penalty

                    # Optimization step
                    self._optimization_step(total_loss)

                    # Record history
                    result.loss_history.append(float(total_loss.item()))
                    result.physics_loss_history.append(float(physics_loss.item()))
                    result.boundary_loss_history.append(float(boundary_loss.item()))
                    result.initial_loss_history.append(float(initial_loss.item()))

                    # Check convergence
                    if self._check_convergence(result.loss_history):
                        self.logger.info(f"PINN converged after {epoch + 1} epochs")
                        result.converged = True
                        break

                    # Adaptive sampling update
                    if self.config.adaptive_sampling and epoch % 500 == 0:
                        self._update_adaptive_sampling()

                    # Progress logging
                    if (epoch + 1) % self.config.log_frequency == 0:
                        self._log_training_progress(epoch, total_loss, physics_loss, boundary_loss, initial_loss)

                    # Model checkpointing
                    if (epoch + 1) % self.config.checkpoint_frequency == 0:
                        self._save_checkpoint(epoch, total_loss)

                # Finalize results
                result.num_epochs = len(result.loss_history)
                result.final_loss = result.loss_history[-1] if result.loss_history else np.inf
                result.training_time = time.time() - start_time

                # Extract trained networks
                result.value_network = self.value_network
                result.density_network = self.density_network

                # Evaluate solution on test grid
                result.value_function, result.density_function = self._evaluate_solution()

                # Compute final physics residuals
                result.hjb_residual, result.fp_residual = self._compute_final_residuals()

                self.logger.info(f"PINN training completed: Final loss = {result.final_loss:.6e}")

            except Exception as e:
                self.logger.error(f"PINN solver failed: {e}")
                raise

            return result

        @abstractmethod
        def _compute_adaptive_weights(
            self, physics_loss: torch.Tensor, boundary_loss: torch.Tensor, initial_loss: torch.Tensor
        ) -> tuple[float, float, float]:
            """Compute adaptive loss weights based on loss magnitudes."""

        @abstractmethod
        def _compute_gradient_penalty(self, points: torch.Tensor) -> torch.Tensor:
            """Compute gradient penalty for regularization."""

        @abstractmethod
        def _optimization_step(self, loss: torch.Tensor) -> None:
            """Perform single optimization step."""

        @abstractmethod
        def _update_adaptive_sampling(self) -> None:
            """Update adaptive sampling strategy."""

        @abstractmethod
        def _evaluate_solution(self) -> tuple[NDArray, NDArray]:
            """Evaluate trained networks on test grid."""

        @abstractmethod
        def _compute_final_residuals(self) -> tuple[float, float]:
            """Compute final physics residuals for validation."""

        def _check_convergence(self, loss_history: list[float]) -> bool:
            """Check convergence based on loss reduction."""
            if len(loss_history) < self.config.patience:
                return False

            recent_losses = loss_history[-self.config.patience :]
            improvement = recent_losses[0] - recent_losses[-1]

            return improvement < self.config.min_improvement

        def _log_training_progress(
            self,
            epoch: int,
            total_loss: torch.Tensor,
            physics_loss: torch.Tensor,
            boundary_loss: torch.Tensor,
            initial_loss: torch.Tensor,
        ) -> None:
            """Log training progress."""
            self.logger.info(
                f"Epoch {epoch + 1}: Total={total_loss.item():.6e}, "
                f"Physics={physics_loss.item():.6e}, "
                f"Boundary={boundary_loss.item():.6e}, "
                f"Initial={initial_loss.item():.6e}"
            )

        def _save_checkpoint(self, epoch: int, loss: torch.Tensor) -> None:
            """Save model checkpoint."""
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                # Implementation would save model state
                self.logger.debug(f"New best model saved at epoch {epoch + 1}")

else:
    # Placeholder when PyTorch unavailable
    class BasePINNSolver:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for PINN methods")
