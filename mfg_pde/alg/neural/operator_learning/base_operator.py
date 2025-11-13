"""
Base Neural Operator Classes for MFG Problems.

This module provides the abstract base classes and configuration structures
for neural operator methods in Mean Field Games. Neural operators learn
mappings between parameter spaces and solution spaces, enabling rapid
evaluation without solving the underlying PDEs.

Mathematical Framework:
- Operator G: P → U maps parameters θ ∈ P to solutions u ∈ U
- Parameter Space P: Problem parameters (boundary conditions, initial data, coefficients)
- Solution Space U: MFG solutions (value function u(t,x), density m(t,x))
- Fast Evaluation: u(θ) = G(θ) computed in O(1) time after training

Key Advantages:
- Speed: 100-1000x faster than traditional PDE solvers
- Parameter Studies: Efficient exploration of parameter spaces
- Real-Time Applications: Enable control and optimization scenarios
- Uncertainty Quantification: Monte Carlo over parameter distributions
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np

# Import with availability checking
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from mfg_pde.utils.mfg_logging.logger import get_logger

NDArray = np.ndarray[Any, np.dtype[Any]]


@dataclass
class OperatorConfig:
    """Configuration for neural operator methods."""

    # Network Architecture
    input_dim: int = 64  # Parameter dimension
    output_dim: int = 1024  # Solution dimension (flattened)
    hidden_layers: list[int] = None  # Hidden layer sizes

    # Training Parameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 1000
    patience: int = 50  # Early stopping patience

    # Operator-Specific Parameters
    operator_type: str = "fno"  # "fno", "deeponet"

    # Fourier Neural Operator specific
    modes: int = 16  # Number of Fourier modes
    width: int = 64  # Channel width

    # DeepONet specific
    branch_depth: int = 4
    trunk_depth: int = 4

    # Regularization
    weight_decay: float = 1e-5
    dropout_rate: float = 0.1

    # Data Processing
    normalize_inputs: bool = True
    normalize_outputs: bool = True

    # Performance
    device: str = "auto"  # "cpu", "cuda", "mps", "auto"
    num_workers: int = 4  # DataLoader workers

    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_frequency: int = 100  # Save every N epochs

    def __post_init__(self):
        """Set default values and validate configuration."""
        if self.hidden_layers is None:
            self.hidden_layers = [256, 256, 256]


@dataclass
class OperatorResult:
    """Results from neural operator training and evaluation."""

    # Training Results
    training_loss: list[float] = None
    validation_loss: list[float] = None
    training_time: float = 0.0
    converged: bool = False
    num_epochs: int = 0

    # Evaluation Results
    test_error: float = np.inf
    relative_error: float = np.inf
    evaluation_time: float = 0.0
    throughput: float = 0.0  # Evaluations per second

    # Model Information
    model_parameters: int = 0
    model_size_mb: float = 0.0

    # Speedup Analysis
    traditional_solve_time: float = np.inf
    speedup_factor: float = 0.0

    def __post_init__(self):
        """Initialize default values."""
        if self.training_loss is None:
            self.training_loss = []
        if self.validation_loss is None:
            self.validation_loss = []


if TORCH_AVAILABLE:

    class BaseNeuralOperator(nn.Module, abc.ABC):
        """
        Abstract base class for neural operator methods in MFG.

        Neural operators learn mappings G: P → U where P is the parameter space
        and U is the solution space. This enables rapid evaluation of MFG solutions
        for different parameter values without solving the underlying PDEs.

        Mathematical Framework:
        - Input: θ ∈ P (problem parameters)
        - Output: u ∈ U (MFG solution)
        - Operator: u = G(θ) learned via neural networks

        Key Methods:
        - forward(): Neural network forward pass
        - train_operator(): Train the operator on parameter-solution pairs
        - predict(): Fast evaluation for new parameters
        - evaluate_performance(): Assess operator accuracy and speed
        """

        def __init__(self, config: OperatorConfig):
            """
            Initialize base neural operator.

            Args:
                config: Operator configuration
            """
            super().__init__()
            self.config = config
            self.logger = get_logger(__name__)

            # Set device
            if config.device == "auto":
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                elif torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                else:
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device(config.device)

            # Initialize normalization parameters
            self.input_mean = None
            self.input_std = None
            self.output_mean = None
            self.output_std = None

            # Training state
            self.is_trained = False
            self.training_history = OperatorResult()

        @abc.abstractmethod
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through the neural operator.

            Args:
                x: Input parameters [batch_size, input_dim]

            Returns:
                Predicted solutions [batch_size, output_dim]
            """

        @abc.abstractmethod
        def _build_network(self) -> None:
            """Build the neural network architecture."""

        def predict(self, parameters: torch.Tensor | NDArray) -> NDArray:
            """
            Fast prediction for new parameters.

            Args:
                parameters: Input parameters [batch_size, input_dim]

            Returns:
                Predicted solutions [batch_size, output_dim]
            """
            if not self.is_trained:
                raise RuntimeError("Operator must be trained before prediction")

            # Convert to tensor if needed
            if isinstance(parameters, np.ndarray):
                parameters = torch.from_numpy(parameters).float()

            parameters = parameters.to(self.device)

            # Normalize inputs if configured
            if self.config.normalize_inputs and self.input_mean is not None:
                parameters = (parameters - self.input_mean) / self.input_std

            # Forward pass
            self.eval()
            with torch.no_grad():
                predictions = self.forward(parameters)

            # Denormalize outputs if configured
            if self.config.normalize_outputs and self.output_mean is not None:
                predictions = predictions * self.output_std + self.output_mean

            return predictions.cpu().numpy()

        def train_operator(
            self,
            train_data: DataLoader | Dataset,
            val_data: DataLoader | Dataset | None = None,
            save_path: Path | None = None,
        ) -> OperatorResult:
            """
            Train the neural operator on parameter-solution pairs.

            Args:
                train_data: Training dataset or DataLoader
                val_data: Validation dataset or DataLoader
                save_path: Path to save trained model

            Returns:
                Training results and performance metrics
            """
            self.logger.info("Starting neural operator training")
            start_time = time.time()

            # Setup data loaders
            if isinstance(train_data, Dataset):
                train_loader = DataLoader(
                    train_data, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers
                )
            else:
                train_loader = train_data

            if val_data is not None and isinstance(val_data, Dataset):
                val_loader = DataLoader(
                    val_data, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers
                )
            else:
                val_loader = val_data

            # Setup optimizer and loss function
            optimizer = optim.Adam(
                self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
            )
            criterion = nn.MSELoss()

            # Training loop
            best_val_loss = float("inf")
            patience_counter = 0

            for epoch in range(self.config.max_epochs):
                # Training phase
                train_loss = self._train_epoch(train_loader, optimizer, criterion)
                self.training_history.training_loss.append(train_loss)

                # Validation phase
                if val_loader is not None:
                    val_loss = self._validate_epoch(val_loader, criterion)
                    self.training_history.validation_loss.append(val_loss)

                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        if save_path:
                            self._save_checkpoint(save_path / "best_model.pt")
                    else:
                        patience_counter += 1

                    if patience_counter >= self.config.patience:
                        self.logger.info(f"Early stopping at epoch {epoch}")
                        break

                # Progress logging
                if (epoch + 1) % 100 == 0:
                    self.logger.info(
                        f"Epoch {epoch + 1}: Train Loss = {train_loss:.6e}, Val Loss = {val_loss:.6e}"
                        if val_loader
                        else ""
                    )

            # Finalize training
            self.training_history.training_time = time.time() - start_time
            self.training_history.num_epochs = epoch + 1
            self.training_history.converged = patience_counter < self.config.patience
            self.is_trained = True

            self.logger.info(f"Training completed in {self.training_history.training_time:.2f}s")
            return self.training_history

        def _train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module) -> float:
            """Train for one epoch."""
            self.train()
            total_loss = 0.0
            num_batches = 0

            for batch_params, batch_solutions in train_loader:
                batch_params = batch_params.to(self.device)
                batch_solutions = batch_solutions.to(self.device)

                optimizer.zero_grad()
                predictions = self.forward(batch_params)
                loss = criterion(predictions, batch_solutions)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            return total_loss / num_batches

        def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> float:
            """Validate for one epoch."""
            self.eval()
            total_loss = 0.0
            num_batches = 0

            with torch.no_grad():
                for batch_params, batch_solutions in val_loader:
                    batch_params = batch_params.to(self.device)
                    batch_solutions = batch_solutions.to(self.device)

                    predictions = self.forward(batch_params)
                    loss = criterion(predictions, batch_solutions)

                    total_loss += loss.item()
                    num_batches += 1

            return total_loss / num_batches

        def _save_checkpoint(self, path: Path) -> None:
            """Save model checkpoint."""
            checkpoint = {
                "model_state_dict": self.state_dict(),
                "config": self.config,
                "normalization": {
                    "input_mean": self.input_mean,
                    "input_std": self.input_std,
                    "output_mean": self.output_mean,
                    "output_std": self.output_std,
                },
                "training_history": self.training_history,
            }
            torch.save(checkpoint, path)

        def load_checkpoint(self, path: Path) -> None:
            """Load model checkpoint."""
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint["model_state_dict"])

            # Restore normalization parameters
            norm_params = checkpoint["normalization"]
            self.input_mean = norm_params["input_mean"]
            self.input_std = norm_params["input_std"]
            self.output_mean = norm_params["output_mean"]
            self.output_std = norm_params["output_std"]

            self.training_history = checkpoint["training_history"]
            self.is_trained = True

else:
    # Placeholder classes when PyTorch is not available
    class BaseNeuralOperator:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("Neural operators require PyTorch")
