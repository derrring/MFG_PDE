"""
Adaptive training strategies for Physics-Informed Neural Networks.

This module implements sophisticated adaptive training techniques that
improve PINN convergence and solution quality by dynamically adjusting
training strategies based on physics residuals and solution behavior.

Key Strategies:
- Physics-Guided Sampling: Adaptive point generation based on residual analysis
- Curriculum Learning: Progressive difficulty in training
- Multi-Scale Training: Hierarchical resolution progression
- Loss Balancing: Dynamic weight adjustment for loss components
- Residual-Based Refinement: Targeted sampling in high-error regions

Mathematical Framework:
- Residual Analysis: R(x) = |PDE_residual(x)| for adaptive sampling
- Importance Sampling: p(x) ∝ |R(x)|^α for weighted point selection
- Curriculum Function: C(epoch) defining training complexity progression
- Multi-Scale Decomposition: L = Σᵢ wᵢ(epoch) L_scale_i

Applications:
- High-dimensional MFG problems with complex physics
- Multi-scale phenomena in crowd dynamics
- Singular perturbation problems in finance
- Problems with boundary layers and sharp gradients
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mfg_pde.utils.monte_carlo import (
    MCConfig,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

# Import with availability checking
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveTrainingConfig:
    """Configuration for adaptive training strategies."""

    # Physics-guided sampling
    residual_threshold: float = 1e-3  # Threshold for high-residual regions
    adaptive_sampling_frequency: int = 100  # Update sampling every N epochs
    max_adaptive_points: int = 10000  # Maximum points to add adaptively
    importance_exponent: float = 0.5  # p(x) ∝ |R(x)|^α

    # Curriculum learning
    enable_curriculum: bool = True
    curriculum_epochs: int = 5000  # Epochs for curriculum progression
    initial_complexity: float = 0.1  # Start with 10% of full problem
    complexity_growth: str = "linear"  # "linear" | "exponential" | "sigmoid"

    # Multi-scale training
    enable_multiscale: bool = True
    num_scales: int = 3  # Number of resolution scales
    scale_transition_epochs: int = 2000  # Epochs between scale transitions
    coarse_to_fine: bool = True  # Start coarse and refine

    # Loss balancing
    adaptive_loss_weights: bool = True
    weight_update_frequency: int = 50  # Update weights every N epochs
    weight_momentum: float = 0.9  # Momentum for weight updates
    gradient_balance: bool = True  # Balance gradients across loss terms

    # Residual-based refinement
    enable_refinement: bool = True
    refinement_frequency: int = 500  # Refine mesh every N epochs
    refinement_factor: float = 1.5  # Factor to increase points in high-residual regions
    max_refinement_levels: int = 5  # Maximum refinement levels

    # Performance monitoring
    monitor_convergence: bool = True
    stagnation_patience: int = 1000  # Epochs to wait before intervention
    stagnation_threshold: float = 1e-6  # Minimum improvement threshold


@dataclass
class TrainingState:
    """Current state of adaptive training."""

    epoch: int = 0
    current_complexity: float = 0.1
    current_scale: int = 0
    num_physics_points: int = 1000
    num_boundary_points: int = 200
    num_initial_points: int = 200

    # Loss weights
    physics_weight: float = 1.0
    boundary_weight: float = 10.0
    initial_weight: float = 10.0

    # Sampling state
    importance_function: Callable | None = None
    high_residual_regions: list[NDArray] = None

    # Performance tracking
    loss_history: list[float] = None
    residual_history: list[NDArray] = None
    convergence_rate: float = 0.0
    is_stagnating: bool = False

    def __post_init__(self):
        if self.loss_history is None:
            self.loss_history = []
        if self.residual_history is None:
            self.residual_history = []
        if self.high_residual_regions is None:
            self.high_residual_regions = []


class AdaptiveTrainingStrategy(ABC):
    """Abstract base class for adaptive training strategies."""

    def __init__(self, config: AdaptiveTrainingConfig):
        """Initialize adaptive training strategy."""
        self.config = config
        self.state = TrainingState()
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def update_sampling_strategy(self, residuals: NDArray, sample_points: NDArray) -> dict[str, int]:
        """
        Update sampling strategy based on residual analysis.

        Args:
            residuals: Current residuals at sample points
            sample_points: Current sample points

        Returns:
            Dictionary with updated point counts
        """

    @abstractmethod
    def update_loss_weights(self, loss_components: dict[str, float]) -> dict[str, float]:
        """
        Update loss component weights.

        Args:
            loss_components: Dictionary of current loss values

        Returns:
            Updated loss weights
        """

    @abstractmethod
    def should_advance_curriculum(self) -> bool:
        """Check if curriculum should advance to next stage."""

    def update_state(self, epoch: int, loss_value: float, residuals: NDArray | None = None):
        """Update training state."""
        self.state.epoch = epoch
        self.state.loss_history.append(loss_value)

        if residuals is not None:
            self.state.residual_history.append(residuals)

        # Update convergence rate
        if len(self.state.loss_history) >= 10:
            recent_losses = self.state.loss_history[-10:]
            self.state.convergence_rate = (recent_losses[0] - recent_losses[-1]) / 10

        # Check for stagnation
        if len(self.state.loss_history) >= self.config.stagnation_patience:
            recent_improvement = self.state.loss_history[-self.config.stagnation_patience] - self.state.loss_history[-1]
            self.state.is_stagnating = recent_improvement < self.config.stagnation_threshold


if TORCH_AVAILABLE:

    class PhysicsGuidedSampler(AdaptiveTrainingStrategy):
        """Physics-guided adaptive sampling strategy."""

        def __init__(self, config: AdaptiveTrainingConfig, domain_bounds: list[tuple[float, float]]):
            """
            Initialize physics-guided sampler.

            Args:
                config: Adaptive training configuration
                domain_bounds: Domain bounds for sampling
            """
            super().__init__(config)
            self.domain_bounds = domain_bounds
            self.dimension = len(domain_bounds)

            # Initialize Monte Carlo config
            self.mc_config = MCConfig(
                num_samples=config.max_adaptive_points,
                sampling_method="uniform",
                adaptive=True,
                error_tolerance=config.residual_threshold,
            )

        def update_sampling_strategy(self, residuals: NDArray, sample_points: NDArray) -> dict[str, int]:
            """Update sampling based on residual analysis."""
            if self.state.epoch % self.config.adaptive_sampling_frequency != 0:
                return self._get_current_point_counts()

            # Analyze residual distribution
            residual_magnitudes = np.abs(residuals.flatten())
            high_residual_threshold = np.percentile(residual_magnitudes, 90)

            # Identify high-residual regions
            high_residual_mask = residual_magnitudes > high_residual_threshold
            high_residual_points = sample_points[high_residual_mask]

            if len(high_residual_points) > 0:
                self.state.high_residual_regions.append(high_residual_points)

                # Create importance function
                self.state.importance_function = self._create_importance_function(sample_points, residual_magnitudes)

                # Increase sampling in high-residual regions
                additional_points = min(
                    int(len(high_residual_points) * self.config.refinement_factor), self.config.max_adaptive_points // 4
                )

                self.state.num_physics_points += additional_points
                self.logger.debug(f"Added {additional_points} points in high-residual regions")

            return self._get_current_point_counts()

        def _create_importance_function(self, points: NDArray, residuals: NDArray) -> Callable:
            """Create importance function based on residuals."""
            # Normalize residuals
            max_residual = np.max(residuals)
            normalized_residuals = residuals / max_residual if max_residual > 0 else residuals

            # Apply importance exponent
            importance_weights = normalized_residuals**self.config.importance_exponent

            def importance_func(query_points: NDArray) -> NDArray:
                """Evaluate importance function at query points."""
                if len(query_points) == 0:
                    return np.array([])

                # Simple nearest-neighbor interpolation
                distances = np.sum((query_points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=2)
                nearest_indices = np.argmin(distances, axis=1)
                return importance_weights[nearest_indices]

            return importance_func

        def update_loss_weights(self, loss_components: dict[str, float]) -> dict[str, float]:
            """Update loss weights based on component magnitudes."""
            if not self.config.adaptive_loss_weights:
                return {
                    "physics": self.state.physics_weight,
                    "boundary": self.state.boundary_weight,
                    "initial": self.state.initial_weight,
                }

            if self.state.epoch % self.config.weight_update_frequency != 0:
                return self._get_current_weights()

            # Compute adaptive weights based on loss magnitudes
            physics_loss = loss_components.get("physics", 1.0)
            boundary_loss = loss_components.get("boundary", 1.0)
            initial_loss = loss_components.get("initial", 1.0)

            # Inverse weighting with stabilization
            eps = 1e-8
            physics_weight = 1.0 / (physics_loss + eps)
            boundary_weight = 1.0 / (boundary_loss + eps)
            initial_weight = 1.0 / (initial_loss + eps)

            # Normalize weights
            total_weight = physics_weight + boundary_weight + initial_weight
            physics_weight /= total_weight
            boundary_weight /= total_weight
            initial_weight /= total_weight

            # Apply momentum
            momentum = self.config.weight_momentum
            self.state.physics_weight = momentum * self.state.physics_weight + (1 - momentum) * physics_weight
            self.state.boundary_weight = momentum * self.state.boundary_weight + (1 - momentum) * boundary_weight
            self.state.initial_weight = momentum * self.state.initial_weight + (1 - momentum) * initial_weight

            return self._get_current_weights()

        def should_advance_curriculum(self) -> bool:
            """Check if curriculum should advance."""
            if not self.config.enable_curriculum:
                return False

            curriculum_progress = self.state.epoch / self.config.curriculum_epochs
            target_complexity = self._compute_curriculum_complexity(curriculum_progress)

            return target_complexity > self.state.current_complexity

        def _compute_curriculum_complexity(self, progress: float) -> float:
            """Compute target complexity based on curriculum progress."""
            progress = min(1.0, max(0.0, progress))  # Clamp to [0, 1]

            if self.config.complexity_growth == "linear":
                complexity = self.config.initial_complexity + progress * (1.0 - self.config.initial_complexity)
            elif self.config.complexity_growth == "exponential":
                complexity = self.config.initial_complexity * (10**progress)
                complexity = min(1.0, complexity)
            elif self.config.complexity_growth == "sigmoid":
                # Sigmoid curve: smooth transition
                x = 12 * progress - 6  # Scale to [-6, 6]
                sigmoid = 1 / (1 + np.exp(-x))
                complexity = self.config.initial_complexity + sigmoid * (1.0 - self.config.initial_complexity)
            else:
                complexity = 1.0  # Default to full complexity

            return complexity

        def advance_curriculum(self):
            """Advance curriculum to next stage."""
            if not self.should_advance_curriculum():
                return

            curriculum_progress = self.state.epoch / self.config.curriculum_epochs
            new_complexity = self._compute_curriculum_complexity(curriculum_progress)

            self.state.current_complexity = new_complexity

            # Scale up problem based on complexity
            base_physics_points = 1000
            base_boundary_points = 200
            base_initial_points = 200

            self.state.num_physics_points = int(base_physics_points * new_complexity)
            self.state.num_boundary_points = int(base_boundary_points * new_complexity)
            self.state.num_initial_points = int(base_initial_points * new_complexity)

            self.logger.info(
                f"Advanced curriculum: complexity = {new_complexity:.3f}, "
                f"physics_points = {self.state.num_physics_points}"
            )

        def _get_current_point_counts(self) -> dict[str, int]:
            """Get current point counts."""
            return {
                "physics": self.state.num_physics_points,
                "boundary": self.state.num_boundary_points,
                "initial": self.state.num_initial_points,
            }

        def _get_current_weights(self) -> dict[str, float]:
            """Get current loss weights."""
            return {
                "physics": self.state.physics_weight,
                "boundary": self.state.boundary_weight,
                "initial": self.state.initial_weight,
            }

        def intervene_if_stagnating(self) -> bool:
            """Intervene if training is stagnating."""
            if not self.state.is_stagnating:
                return False

            self.logger.warning("Training stagnation detected - applying intervention")

            # Intervention strategies
            # 1. Increase sampling resolution
            self.state.num_physics_points = int(self.state.num_physics_points * 1.5)
            self.state.num_boundary_points = int(self.state.num_boundary_points * 1.2)

            # 2. Reset loss weights
            self.state.physics_weight = 1.0
            self.state.boundary_weight = 10.0
            self.state.initial_weight = 10.0

            # 3. Clear stagnation flag
            self.state.is_stagnating = False

            self.logger.info("Applied stagnation intervention: increased sampling resolution")
            return True

    class MultiScaleTrainingStrategy(AdaptiveTrainingStrategy):
        """Multi-scale training strategy for hierarchical learning."""

        def __init__(self, config: AdaptiveTrainingConfig, domain_bounds: list[tuple[float, float]]):
            """Initialize multi-scale training."""
            super().__init__(config)
            self.domain_bounds = domain_bounds
            self.scale_levels = self._create_scale_levels()

        def _create_scale_levels(self) -> list[dict[str, Any]]:
            """Create hierarchical scale levels."""
            scales = []

            for i in range(self.config.num_scales):
                # Coarse to fine progression
                scale_factor = 2**i if self.config.coarse_to_fine else 2 ** (self.config.num_scales - 1 - i)

                scale_info = {
                    "level": i,
                    "resolution_factor": scale_factor,
                    "physics_points": 500 * scale_factor,
                    "boundary_points": 100 * scale_factor,
                    "epochs": self.config.scale_transition_epochs,
                }
                scales.append(scale_info)

            return scales

        def update_sampling_strategy(self, residuals: NDArray, sample_points: NDArray) -> dict[str, int]:
            """Update sampling for multi-scale progression."""
            # Check if we should transition to next scale
            if self.should_advance_scale():
                self.advance_scale()

            current_scale = self.scale_levels[self.state.current_scale]
            return {
                "physics": current_scale["physics_points"],
                "boundary": current_scale["boundary_points"],
                "initial": current_scale["boundary_points"],  # Same as boundary
            }

        def should_advance_scale(self) -> bool:
            """Check if we should advance to next scale."""
            if not self.config.enable_multiscale:
                return False

            if self.state.current_scale >= len(self.scale_levels) - 1:
                return False  # Already at finest scale

            # Check if we've completed enough epochs at current scale
            epochs_at_scale = self.state.epoch - (self.state.current_scale * self.config.scale_transition_epochs)
            return epochs_at_scale >= self.config.scale_transition_epochs

        def advance_scale(self):
            """Advance to next scale level."""
            if self.state.current_scale < len(self.scale_levels) - 1:
                self.state.current_scale += 1
                current_scale = self.scale_levels[self.state.current_scale]

                self.logger.info(
                    f"Advanced to scale level {self.state.current_scale}: "
                    f"resolution_factor = {current_scale['resolution_factor']}"
                )

        def update_loss_weights(self, loss_components: dict[str, float]) -> dict[str, float]:
            """Update loss weights for multi-scale training."""
            # Use different weights at different scales
            current_scale = self.scale_levels[self.state.current_scale]
            current_scale["resolution_factor"]

            # At coarser scales, emphasize boundary and initial conditions
            # At finer scales, emphasize physics residual
            if self.state.current_scale < len(self.scale_levels) // 2:
                # Coarse scales
                physics_weight = 0.5
                boundary_weight = 15.0
                initial_weight = 15.0
            else:
                # Fine scales
                physics_weight = 2.0
                boundary_weight = 5.0
                initial_weight = 5.0

            return {"physics": physics_weight, "boundary": boundary_weight, "initial": initial_weight}

        def should_advance_curriculum(self) -> bool:
            """Multi-scale training has its own progression."""
            return False  # Handled by scale advancement

    class GradientBalancingStrategy(AdaptiveTrainingStrategy):
        """Strategy for balancing gradients across loss components."""

        def __init__(self, config: AdaptiveTrainingConfig):
            """Initialize gradient balancing."""
            super().__init__(config)
            self.gradient_history = []
            self.target_gradient_ratios = {"physics": 1.0, "boundary": 1.0, "initial": 1.0}

        def update_loss_weights(
            self, loss_components: dict[str, float], gradient_norms: dict[str, float] | None = None
        ) -> dict[str, float]:
            """Update weights based on gradient magnitudes."""
            if not self.config.gradient_balance or gradient_norms is None:
                return super().update_loss_weights(loss_components)

            # Store gradient history
            self.gradient_history.append(gradient_norms.copy())

            # Compute target gradient ratios
            if len(self.gradient_history) >= 10:
                # Use recent gradient statistics
                recent_grads = self.gradient_history[-10:]

                avg_physics_grad = np.mean([g.get("physics", 1.0) for g in recent_grads])
                avg_boundary_grad = np.mean([g.get("boundary", 1.0) for g in recent_grads])
                avg_initial_grad = np.mean([g.get("initial", 1.0) for g in recent_grads])

                # Balance gradients to have similar magnitudes
                total_grad = avg_physics_grad + avg_boundary_grad + avg_initial_grad
                if total_grad > 0:
                    physics_weight = total_grad / (3 * avg_physics_grad + 1e-8)
                    boundary_weight = total_grad / (3 * avg_boundary_grad + 1e-8)
                    initial_weight = total_grad / (3 * avg_initial_grad + 1e-8)

                    return {"physics": physics_weight, "boundary": boundary_weight, "initial": initial_weight}

            return self._get_current_weights()

        def update_sampling_strategy(self, residuals: NDArray, sample_points: NDArray) -> dict[str, int]:
            """Use base sampling strategy."""
            return self._get_current_point_counts()

        def should_advance_curriculum(self) -> bool:
            """No curriculum in gradient balancing."""
            return False

else:
    # Placeholder classes when PyTorch unavailable
    class PhysicsGuidedSampler:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for adaptive training strategies")

    class MultiScaleTrainingStrategy:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for adaptive training strategies")

    class GradientBalancingStrategy:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for adaptive training strategies")


# Utility functions for adaptive training
def create_adaptive_strategy(
    strategy_type: str, config: AdaptiveTrainingConfig, domain_bounds: list[tuple[float, float]]
) -> AdaptiveTrainingStrategy:
    """
    Create adaptive training strategy.

    Args:
        strategy_type: "physics_guided" | "multiscale" | "gradient_balance"
        config: Adaptive training configuration
        domain_bounds: Domain bounds for sampling

    Returns:
        Adaptive training strategy instance
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for adaptive training strategies")

    if strategy_type == "physics_guided":
        return PhysicsGuidedSampler(config, domain_bounds)
    elif strategy_type == "multiscale":
        return MultiScaleTrainingStrategy(config, domain_bounds)
    elif strategy_type == "gradient_balance":
        return GradientBalancingStrategy(config)
    else:
        raise ValueError(f"Unknown adaptive strategy: {strategy_type}")


def combine_strategies(strategies: list[AdaptiveTrainingStrategy]) -> AdaptiveTrainingStrategy:
    """
    Combine multiple adaptive training strategies.

    Args:
        strategies: List of strategies to combine

    Returns:
        Combined strategy
    """

    class CombinedStrategy(AdaptiveTrainingStrategy):
        def __init__(self, strategies):
            # Use config from first strategy
            super().__init__(strategies[0].config if strategies else AdaptiveTrainingConfig())
            self.strategies = strategies

        def update_sampling_strategy(self, residuals, sample_points):
            # Use first strategy that provides sampling updates
            for strategy in self.strategies:
                result = strategy.update_sampling_strategy(residuals, sample_points)
                if result != strategy._get_current_point_counts():
                    return result
            return self._get_current_point_counts()

        def update_loss_weights(self, loss_components):
            # Combine weights from all strategies
            all_weights = [strategy.update_loss_weights(loss_components) for strategy in self.strategies]

            # Average the weights
            combined_weights = {}
            for key in ["physics", "boundary", "initial"]:
                weights = [w.get(key, 1.0) for w in all_weights]
                combined_weights[key] = np.mean(weights)

            return combined_weights

        def should_advance_curriculum(self):
            # Advance if any strategy suggests advancement
            return any(strategy.should_advance_curriculum() for strategy in self.strategies)

    return CombinedStrategy(strategies)
