"""
Base Deep Galerkin Method (DGM) solver for high-dimensional MFG problems.

This module implements the abstract base class for Deep Galerkin Methods,
providing the mathematical and computational framework for solving PDE systems
in high dimensions using neural network function approximation.

Mathematical Framework:
- Replace grid-based PDE methods with neural function approximation
- Use Monte Carlo sampling for high-dimensional domain integration
- Minimize physics residuals through gradient-based optimization
- Enable efficient solution in dimensions d > 5

Key Features:
- High-dimensional sampling strategies (Monte Carlo, Quasi-Monte Carlo)
- Variance reduction techniques for computational efficiency
- Adaptive sampling based on solution behavior and residual analysis
- Deep neural network architectures optimized for function approximation
"""

from __future__ import annotations

import logging
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from mfg_pde.alg.base_solver import BaseNeuralSolver

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.core.mfg_problem import MFGProblem

logger = logging.getLogger(__name__)


class VarianceReductionMethod(str, Enum):
    """
    Variance reduction technique for DGM Monte Carlo sampling.

    Determines which statistical methods are used to reduce variance in
    high-dimensional Monte Carlo integration for PDE residuals.

    Attributes:
        NONE: Standard Monte Carlo (no variance reduction)
        CONTROL_VARIATES: Control variates using baseline function
        IMPORTANCE_SAMPLING: Importance sampling based on residuals
        BOTH: Both control variates and importance sampling
    """

    NONE = "none"
    CONTROL_VARIATES = "control_variates"
    IMPORTANCE_SAMPLING = "importance_sampling"
    BOTH = "both"


@dataclass
class DGMConfig:
    """Configuration for Deep Galerkin Method solver."""

    # Network architecture
    hidden_layers: list[int] | None = None  # [128, 128, 128] for high-dimensional
    activation: str = "tanh"  # Smooth activation for PDE approximation
    use_residual_connections: bool = True
    batch_normalization: bool = False

    # Training parameters
    learning_rate: float = 1e-3
    max_epochs: int = 10000
    batch_size: int = 1024  # Large batches for Monte Carlo
    optimizer: str = "adam"
    scheduler: str = "cosine"  # Learning rate scheduling

    # High-dimensional sampling
    num_interior_points: int = 10000  # Interior domain sampling
    num_boundary_points: int = 2000  # Boundary condition sampling
    num_initial_points: int = 2000  # Initial condition sampling
    sampling_strategy: str = "monte_carlo"  # "monte_carlo" | "quasi_monte_carlo"

    # Variance reduction (replaces use_control_variates, use_importance_sampling)
    variance_reduction: VarianceReductionMethod = VarianceReductionMethod.CONTROL_VARIATES

    # Deprecated parameters (kept for backward compatibility)
    use_control_variates: bool | None = None  # Deprecated: use variance_reduction
    use_importance_sampling: bool | None = None  # Deprecated: use variance_reduction

    baseline_function: str | None = None  # Reference solution for variance reduction

    # Adaptive sampling
    adaptive_sampling: bool = True
    residual_threshold: float = 1e-3  # Threshold for adaptive point addition
    max_adaptive_points: int = 5000  # Maximum additional points per iteration

    # Physics constraints
    physics_weight: float = 1.0  # Weight for PDE residual loss
    boundary_weight: float = 10.0  # Weight for boundary condition loss
    initial_weight: float = 10.0  # Weight for initial condition loss
    coupling_weight: float = 1.0  # Weight for HJB-FP coupling

    # Convergence criteria
    tolerance: float = 1e-5
    patience: int = 500  # Early stopping patience
    min_improvement: float = 1e-6

    # Computational settings
    device: str = "auto"  # "cpu" | "cuda" | "mps" | "auto"
    dtype: str = "float32"  # "float32" | "float64"
    num_workers: int = 4  # Data loading workers

    def __post_init__(self) -> None:
        """Set default hidden layers and handle deprecated parameters."""
        # Handle deprecated variance reduction parameters
        if self.use_control_variates is not None or self.use_importance_sampling is not None:
            warnings.warn(
                "Parameters 'use_control_variates' and 'use_importance_sampling' are deprecated "
                "and will be removed in v1.0.0. Use 'variance_reduction' instead.\n\n"
                "Migration guide:\n"
                "  Old: DGMConfig(use_control_variates=True, use_importance_sampling=False)\n"
                "  New: DGMConfig(variance_reduction=VarianceReductionMethod.CONTROL_VARIATES)\n\n"
                "Available variance reduction methods:\n"
                "  - NONE: No variance reduction\n"
                "  - CONTROL_VARIATES: Control variates only\n"
                "  - IMPORTANCE_SAMPLING: Importance sampling only\n"
                "  - BOTH: Both techniques",
                DeprecationWarning,
                stacklevel=2,
            )

            # Map deprecated booleans to variance reduction enum
            if self.variance_reduction == VarianceReductionMethod.CONTROL_VARIATES:  # Default value
                cv = self.use_control_variates if self.use_control_variates is not None else True
                imp = self.use_importance_sampling if self.use_importance_sampling is not None else False

                if cv and imp:
                    self.variance_reduction = VarianceReductionMethod.BOTH
                elif cv and not imp:
                    self.variance_reduction = VarianceReductionMethod.CONTROL_VARIATES
                elif not cv and imp:
                    self.variance_reduction = VarianceReductionMethod.IMPORTANCE_SAMPLING
                else:
                    self.variance_reduction = VarianceReductionMethod.NONE

        # Set default hidden layers for high-dimensional problems
        if self.hidden_layers is None:
            # Default architecture for high-dimensional approximation
            self.hidden_layers = [256, 256, 256, 256]


@dataclass
class DGMResult:
    """Result container for Deep Galerkin Method solver."""

    # Learned neural network solutions
    value_network: Any = None  # Trained network for u(t,x)
    density_network: Any = None  # Trained network for m(t,x)
    value_function: NDArray | None = None  # u(t,x) on evaluation grid
    density_function: NDArray | None = None  # m(t,x) on evaluation grid

    # Training performance
    loss_history: list[float] | None = None
    physics_loss_history: list[float] | None = None
    boundary_loss_history: list[float] | None = None
    initial_loss_history: list[float] | None = None

    # High-dimensional metrics
    final_loss: float = np.inf
    physics_residual: float = np.inf
    boundary_error: float = np.inf
    initial_error: float = np.inf

    # Sampling statistics
    total_sample_points: int = 0
    adaptive_points_added: int = 0
    variance_reduction_factor: float = 1.0

    # Convergence information
    converged: bool = False
    num_epochs: int = 0
    training_time: float = 0.0

    # High-dimensional analysis
    dimension: int = 0
    effective_dimension: float = 0.0  # Effective dimension analysis
    approximation_quality: float = 0.0


class BaseDGMSolver(BaseNeuralSolver):
    """
    Abstract base class for Deep Galerkin Method solvers.

    This class provides the mathematical and computational framework for
    solving high-dimensional PDE systems using neural network function
    approximation and Monte Carlo sampling methods.

    Mathematical Approach:
    - Function Approximation: u(t,x) ≈ U_θ(t,x), m(t,x) ≈ M_φ(t,x)
    - Monte Carlo Integration: ∫_Ω f(x) dx ≈ (1/N) Σᵢ f(xᵢ), xᵢ ~ Uniform(Ω)
    - Physics Residual: L_physics = E[|PDE_residual(t,x)|²]
    - Boundary/Initial: L_BC = E[|BC_residual(x)|²], L_IC = E[|IC_residual(x)|²]

    Key Advantages:
    - No curse of dimensionality (scales as O(d) rather than O(h^d))
    - Natural handling of complex geometries
    - Differentiable solutions for sensitivity analysis
    - Mesh-free approach with adaptive sampling
    """

    def __init__(
        self,
        problem: MFGProblem,
        config: DGMConfig | None = None,
        **kwargs: Any,
    ):
        """
        Initialize DGM solver base class.

        Args:
            problem: High-dimensional MFG problem instance
            config: DGM solver configuration
            **kwargs: Additional solver arguments
        """
        self.config = config or DGMConfig()
        super().__init__(problem, self.config, **kwargs)
        self.logger = self._get_logger()

        # Validate high-dimensional problem
        self._validate_high_dimensional_problem()

        # Initialize neural networks
        self._setup_neural_networks()

        # Initialize sampling strategies
        self._setup_sampling()

        # Initialize variance reduction
        self._setup_variance_reduction()

    def _validate_high_dimensional_problem(self) -> None:
        """Validate that problem is suitable for DGM approach."""
        # Check dimension
        if hasattr(self.problem, "dimension"):
            self.dimension = self.problem.dimension
        else:
            # Infer from domain
            if hasattr(self.problem, "domain") and len(self.problem.domain) >= 2:
                self.dimension = len(self.problem.domain) // 2  # (min, max) pairs
            else:
                self.dimension = 1

        self.logger.info(f"DGM solver initialized for {self.dimension}D problem")

        # Warn if dimension is low (DGM overhead not justified)
        if self.dimension <= 2:
            self.logger.warning(
                f"DGM may not be optimal for {self.dimension}D problems. "
                f"Consider using numerical methods from the numerical paradigm."
            )

    @abstractmethod
    def _setup_neural_networks(self) -> None:
        """Setup neural networks for value and density function approximation."""

    @abstractmethod
    def _setup_sampling(self) -> None:
        """Setup high-dimensional sampling strategies."""

    @abstractmethod
    def _setup_variance_reduction(self) -> None:
        """Setup variance reduction techniques."""

    @abstractmethod
    def _sample_interior_points(self, num_points: int) -> NDArray:
        """
        Sample points in the interior domain for physics loss computation.

        Args:
            num_points: Number of points to sample

        Returns:
            Interior points of shape (num_points, dimension + 1) for (t, x)
        """

    @abstractmethod
    def _sample_boundary_points(self, num_points: int) -> NDArray:
        """
        Sample points on the boundary for boundary condition loss.

        Args:
            num_points: Number of points to sample

        Returns:
            Boundary points of shape (num_points, dimension + 1)
        """

    @abstractmethod
    def _sample_initial_points(self, num_points: int) -> NDArray:
        """
        Sample points at t=0 for initial condition loss.

        Args:
            num_points: Number of points to sample

        Returns:
            Initial points of shape (num_points, dimension)
        """

    @abstractmethod
    def _compute_physics_residual(self, points: NDArray) -> NDArray:
        """
        Compute physics residual at sampled points.

        Args:
            points: Sample points (t, x)

        Returns:
            Physics residual values
        """

    @abstractmethod
    def _compute_boundary_residual(self, points: NDArray) -> NDArray:
        """
        Compute boundary condition residual.

        Args:
            points: Boundary sample points

        Returns:
            Boundary residual values
        """

    @abstractmethod
    def _compute_initial_residual(self, points: NDArray) -> NDArray:
        """
        Compute initial condition residual.

        Args:
            points: Initial sample points

        Returns:
            Initial residual values
        """

    def solve(self) -> DGMResult:
        """
        Solve high-dimensional MFG problem using Deep Galerkin Method.

        Returns:
            DGMResult with trained networks and solution evaluation
        """
        self.logger.info(f"Starting DGM solver for {self.dimension}D MFG problem")

        result = DGMResult()
        result.dimension = self.dimension

        try:
            # Initialize training infrastructure
            self._initialize_training()

            # Training loop
            loss_history = []
            physics_history = []
            boundary_history = []
            initial_history = []

            for epoch in range(self.config.max_epochs):
                # Sample points for this epoch
                interior_points = self._sample_interior_points(self.config.num_interior_points)
                boundary_points = self._sample_boundary_points(self.config.num_boundary_points)
                initial_points = self._sample_initial_points(self.config.num_initial_points)

                # Compute losses
                physics_loss = self._compute_physics_loss(interior_points)
                boundary_loss = self._compute_boundary_loss(boundary_points)
                initial_loss = self._compute_initial_loss(initial_points)

                # Total loss with weighting
                total_loss = (
                    self.config.physics_weight * physics_loss
                    + self.config.boundary_weight * boundary_loss
                    + self.config.initial_weight * initial_loss
                )

                # Optimization step
                self._optimization_step(total_loss)

                # Record history
                loss_history.append(float(total_loss))
                physics_history.append(float(physics_loss))
                boundary_history.append(float(boundary_loss))
                initial_history.append(float(initial_loss))

                # Check convergence
                if self._check_convergence(loss_history):
                    self.logger.info(f"DGM converged after {epoch + 1} epochs")
                    result.converged = True
                    break

                # Adaptive sampling
                if self.config.adaptive_sampling and epoch % 100 == 0:
                    self._adaptive_sampling_update()

                # Progress logging
                if (epoch + 1) % 500 == 0:
                    self.logger.info(
                        f"Epoch {epoch + 1}: Loss={total_loss:.6e}, Physics={physics_loss:.6e}, BC={boundary_loss:.6e}"
                    )

            # Finalize results
            result.loss_history = loss_history
            result.physics_loss_history = physics_history
            result.boundary_loss_history = boundary_history
            result.initial_loss_history = initial_history
            result.final_loss = loss_history[-1] if loss_history else np.inf
            result.num_epochs = len(loss_history)

            # Extract trained networks
            result.value_network = self._extract_value_network()
            result.density_network = self._extract_density_network()

            # Evaluate on test grid
            result.value_function, result.density_function = self._evaluate_on_grid()

            self.logger.info(f"DGM training completed: Final loss = {result.final_loss:.6e}")

        except Exception as e:
            self.logger.error(f"DGM solver failed: {e}")
            raise

        return result

    @abstractmethod
    def _initialize_training(self) -> None:
        """Initialize training infrastructure (optimizers, schedulers)."""

    @abstractmethod
    def _compute_physics_loss(self, points: NDArray) -> float:
        """Compute physics residual loss at sample points."""

    @abstractmethod
    def _compute_boundary_loss(self, points: NDArray) -> float:
        """Compute boundary condition loss."""

    @abstractmethod
    def _compute_initial_loss(self, points: NDArray) -> float:
        """Compute initial condition loss."""

    @abstractmethod
    def _optimization_step(self, loss: float) -> None:
        """Perform single optimization step."""

    @abstractmethod
    def _adaptive_sampling_update(self) -> None:
        """Update sampling based on current solution quality."""

    @abstractmethod
    def _extract_value_network(self) -> Any:
        """Extract trained value function network."""

    @abstractmethod
    def _extract_density_network(self) -> Any:
        """Extract trained density function network."""

    @abstractmethod
    def _evaluate_on_grid(self) -> tuple[NDArray, NDArray]:
        """Evaluate trained networks on regular grid for visualization."""

    def _check_convergence(self, loss_history: list[float]) -> bool:
        """Check convergence based on loss reduction."""
        if len(loss_history) < self.config.patience:
            return False

        # Check for sufficient improvement
        recent_losses = loss_history[-self.config.patience :]
        improvement = recent_losses[0] - recent_losses[-1]

        return improvement < self.config.min_improvement

    def _get_device(self) -> str:
        """Get optimal device for computation."""
        if self.config.device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        return self.config.device

    def _log_training_setup(self) -> None:
        """Log training configuration for reproducibility."""
        self.logger.info("DGM Training Setup:")
        self.logger.info(f"  Dimension: {self.dimension}")
        self.logger.info(f"  Network: {self.config.hidden_layers}")
        self.logger.info(f"  Sampling: {self.config.num_interior_points} interior points")
        self.logger.info(f"  Device: {self._get_device()}")
        self.logger.info(f"  Variance reduction: {self.config.use_control_variates}")
