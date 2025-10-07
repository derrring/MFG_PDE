"""
Training Framework for Neural Operators in MFG.

This module provides comprehensive training infrastructure for neural operator
methods including data generation, training management, and performance evaluation
for Mean Field Games applications.

Key Components:
- OperatorDataset: Manages parameter-solution training data
- OperatorTrainingManager: Handles training workflows and optimization
- MFGDataGenerator: Generates training data from MFG problems
- PerformanceAnalyzer: Evaluates operator performance and speedup

Training Workflow:
1. Data Generation: Create parameter-solution pairs from MFG problems
2. Preprocessing: Normalize and augment training data
3. Training: Optimize neural operator on training data
4. Validation: Monitor performance on held-out data
5. Evaluation: Assess accuracy and speedup compared to traditional solvers

Data Management:
- Parameter Sampling: Generate diverse problem parameters
- Solution Computation: Solve MFG for each parameter set
- Data Augmentation: Enhance training with synthetic variations
- Caching: Store expensive computations for reuse
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

# Import with availability checking
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, random_split

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from mfg_pde.utils.logging.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

NDArray = np.ndarray[Any, np.dtype[Any]]


if TORCH_AVAILABLE:

    class OperatorDataset(Dataset):
        """
        Dataset for neural operator training.

        Manages parameter-solution pairs for training neural operators on
        Mean Field Games. Supports various data formats and preprocessing options.
        """

        def __init__(
            self,
            parameters: torch.Tensor | NDArray,
            solutions: torch.Tensor | NDArray,
            coordinates: torch.Tensor | NDArray | None = None,
            normalize: bool = True,
            device: str = "cpu",
        ):
            """
            Initialize operator dataset.

            Args:
                parameters: Problem parameters [num_samples, param_dim]
                solutions: Corresponding solutions [num_samples, solution_dim]
                coordinates: Spatial coordinates for DeepONet [num_samples, num_points, coord_dim]
                normalize: Whether to normalize data
                device: Device for tensors
            """
            self.device = device
            self.logger = get_logger(__name__)

            # Convert to tensors
            if isinstance(parameters, np.ndarray):
                parameters = torch.from_numpy(parameters).float()
            if isinstance(solutions, np.ndarray):
                solutions = torch.from_numpy(solutions).float()

            self.parameters = parameters.to(device)
            self.solutions = solutions.to(device)

            if coordinates is not None:
                if isinstance(coordinates, np.ndarray):
                    coordinates = torch.from_numpy(coordinates).float()
                self.coordinates = coordinates.to(device)
            else:
                self.coordinates = None

            # Normalization
            self.normalize = normalize
            if normalize:
                self._compute_normalization()
                self._apply_normalization()

        def _compute_normalization(self) -> None:
            """Compute normalization statistics."""
            # Parameter normalization
            self.param_mean = torch.mean(self.parameters, dim=0, keepdim=True)
            self.param_std = torch.std(self.parameters, dim=0, keepdim=True)
            self.param_std = torch.clamp(self.param_std, min=1e-8)  # Avoid division by zero

            # Solution normalization
            self.solution_mean = torch.mean(self.solutions, dim=0, keepdim=True)
            self.solution_std = torch.std(self.solutions, dim=0, keepdim=True)
            self.solution_std = torch.clamp(self.solution_std, min=1e-8)

            # Coordinate normalization (if applicable)
            if self.coordinates is not None:
                coord_flat = self.coordinates.view(-1, self.coordinates.size(-1))
                self.coord_mean = torch.mean(coord_flat, dim=0, keepdim=True)
                self.coord_std = torch.std(coord_flat, dim=0, keepdim=True)
                self.coord_std = torch.clamp(self.coord_std, min=1e-8)

        def _apply_normalization(self) -> None:
            """Apply normalization to data."""
            self.parameters = (self.parameters - self.param_mean) / self.param_std
            self.solutions = (self.solutions - self.solution_mean) / self.solution_std

            if self.coordinates is not None:
                original_shape = self.coordinates.shape
                coord_flat = self.coordinates.view(-1, self.coordinates.size(-1))
                coord_normalized = (coord_flat - self.coord_mean) / self.coord_std
                self.coordinates = coord_normalized.view(original_shape)

        def denormalize_solution(self, normalized_solution: torch.Tensor) -> torch.Tensor:
            """
            Denormalize solution predictions.

            Args:
                normalized_solution: Normalized solution

            Returns:
                Original scale solution
            """
            if not self.normalize:
                return normalized_solution

            return normalized_solution * self.solution_std + self.solution_mean

        def denormalize_parameters(self, normalized_params: torch.Tensor) -> torch.Tensor:
            """
            Denormalize parameters.

            Args:
                normalized_params: Normalized parameters

            Returns:
                Original scale parameters
            """
            if not self.normalize:
                return normalized_params

            return normalized_params * self.param_std + self.param_mean

        def __len__(self) -> int:
            """Return dataset size."""
            return len(self.parameters)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Get dataset item.

            Args:
                idx: Item index

            Returns:
                (parameters, solutions) or (combined_input, solutions) for DeepONet
            """
            if self.coordinates is not None:
                # DeepONet format: combine parameters and flattened coordinates
                coord_flat = self.coordinates[idx].view(-1)
                combined_input = torch.cat([self.parameters[idx], coord_flat])
                return combined_input, self.solutions[idx].view(-1)
            else:
                # Standard format for FNO
                return self.parameters[idx], self.solutions[idx]

        def save(self, path: Path) -> None:
            """Save dataset to file."""
            data = {
                "parameters": self.parameters.cpu(),
                "solutions": self.solutions.cpu(),
                "coordinates": self.coordinates.cpu() if self.coordinates is not None else None,
                "normalize": self.normalize,
            }

            if self.normalize:
                data.update(
                    {
                        "param_mean": self.param_mean.cpu(),
                        "param_std": self.param_std.cpu(),
                        "solution_mean": self.solution_mean.cpu(),
                        "solution_std": self.solution_std.cpu(),
                    }
                )

                if self.coordinates is not None:
                    data.update(
                        {
                            "coord_mean": self.coord_mean.cpu(),
                            "coord_std": self.coord_std.cpu(),
                        }
                    )

            torch.save(data, path)

        @classmethod
        def load(cls, path: Path, device: str = "cpu") -> OperatorDataset:
            """Load dataset from file."""
            data = torch.load(path, map_location=device)

            dataset = cls(
                parameters=data["parameters"],
                solutions=data["solutions"],
                coordinates=data["coordinates"],
                normalize=False,  # Data already normalized
                device=device,
            )

            if data["normalize"]:
                dataset.normalize = True
                dataset.param_mean = data["param_mean"].to(device)
                dataset.param_std = data["param_std"].to(device)
                dataset.solution_mean = data["solution_mean"].to(device)
                dataset.solution_std = data["solution_std"].to(device)

                if data["coordinates"] is not None:
                    dataset.coord_mean = data["coord_mean"].to(device)
                    dataset.coord_std = data["coord_std"].to(device)

            return dataset

    @dataclass
    class TrainingConfig:
        """Configuration for operator training."""

        # Training parameters
        max_epochs: int = 1000
        batch_size: int = 32
        learning_rate: float = 1e-3
        weight_decay: float = 1e-5

        # Validation
        validation_split: float = 0.2
        patience: int = 50  # Early stopping patience

        # Optimization
        optimizer: str = "adam"  # "adam", "sgd", "adamw"
        scheduler: str = "cosine"  # "cosine", "step", "plateau", "none"
        gradient_clip: float = 1.0

        # Loss function
        loss_type: str = "mse"  # "mse", "l1", "huber"
        loss_weights: dict[str, float] | None = None

        # Checkpointing
        save_checkpoints: bool = True
        checkpoint_frequency: int = 100
        save_best_only: bool = True

        # Logging
        log_frequency: int = 10
        verbose: bool = True

        # Device
        device: str = "auto"
        num_workers: int = 4

    class OperatorTrainingManager:
        """
        Training manager for neural operators.

        Handles the complete training workflow including data loading,
        optimization, validation, and checkpointing for neural operators.
        """

        def __init__(self, config: TrainingConfig):
            """
            Initialize training manager.

            Args:
                config: Training configuration
            """
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

            self.logger.info(f"Using device: {self.device}")

        def train_operator(
            self, operator: nn.Module, dataset: OperatorDataset, save_path: Path | None = None
        ) -> dict[str, Any]:
            """
            Train neural operator.

            Args:
                operator: Neural operator to train
                dataset: Training dataset
                save_path: Path to save trained model

            Returns:
                Training history and results
            """
            self.logger.info("Starting operator training")
            start_time = time.time()

            # Move operator to device
            operator = operator.to(self.device)

            # Split dataset
            val_size = int(len(dataset) * self.config.validation_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=self.device.type == "cuda",
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.device.type == "cuda",
            )

            # Setup optimizer
            optimizer = self._create_optimizer(operator)

            # Setup scheduler
            scheduler = self._create_scheduler(optimizer, len(train_loader))

            # Setup loss function
            criterion = self._create_loss_function()

            # Training loop
            train_losses = []
            val_losses = []
            best_val_loss = float("inf")
            patience_counter = 0

            for epoch in range(self.config.max_epochs):
                # Training phase
                train_loss = self._train_epoch(operator, train_loader, optimizer, criterion, scheduler)
                train_losses.append(train_loss)

                # Validation phase
                val_loss = self._validate_epoch(operator, val_loader, criterion)
                val_losses.append(val_loss)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0

                    # Save best model
                    if save_path and self.config.save_best_only:
                        self._save_checkpoint(operator, save_path / "best_model.pt", epoch, val_loss)
                else:
                    patience_counter += 1

                # Regular checkpoint
                if save_path and self.config.save_checkpoints and (epoch + 1) % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint(operator, save_path / f"checkpoint_epoch_{epoch}.pt", epoch, val_loss)

                # Early stopping
                if patience_counter >= self.config.patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

                # Logging
                if (epoch + 1) % self.config.log_frequency == 0 and self.config.verbose:
                    self.logger.info(
                        f"Epoch {epoch + 1}/{self.config.max_epochs}: "
                        f"Train Loss = {train_loss:.6e}, Val Loss = {val_loss:.6e}"
                    )

            # Training summary
            training_time = time.time() - start_time
            results = {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "best_val_loss": best_val_loss,
                "training_time": training_time,
                "num_epochs": epoch + 1,
                "converged": patience_counter < self.config.patience,
            }

            self.logger.info(f"Training completed in {training_time:.2f}s")
            return results

        def _create_optimizer(self, operator: nn.Module) -> optim.Optimizer:
            """Create optimizer."""
            if self.config.optimizer.lower() == "adam":
                return optim.Adam(
                    operator.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
                )
            elif self.config.optimizer.lower() == "adamw":
                return optim.AdamW(
                    operator.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
                )
            elif self.config.optimizer.lower() == "sgd":
                return optim.SGD(
                    operator.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    momentum=0.9,
                )
            else:
                raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        def _create_scheduler(
            self, optimizer: optim.Optimizer, steps_per_epoch: int
        ) -> optim.lr_scheduler.LRScheduler | None:
            """Create learning rate scheduler."""
            if self.config.scheduler.lower() == "cosine":
                return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs)
            elif self.config.scheduler.lower() == "step":
                return optim.lr_scheduler.StepLR(optimizer, step_size=self.config.max_epochs // 3, gamma=0.1)
            elif self.config.scheduler.lower() == "plateau":
                return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=self.config.patience // 2)
            elif self.config.scheduler.lower() == "none":
                return None
            else:
                raise ValueError(f"Unknown scheduler: {self.config.scheduler}")

        def _create_loss_function(self) -> nn.Module:
            """Create loss function."""
            if self.config.loss_type.lower() == "mse":
                return nn.MSELoss()
            elif self.config.loss_type.lower() == "l1":
                return nn.L1Loss()
            elif self.config.loss_type.lower() == "huber":
                return nn.HuberLoss()
            else:
                raise ValueError(f"Unknown loss type: {self.config.loss_type}")

        def _train_epoch(
            self,
            operator: nn.Module,
            train_loader: DataLoader,
            optimizer: optim.Optimizer,
            criterion: nn.Module,
            scheduler: optim.lr_scheduler.LRScheduler | None,
        ) -> float:
            """Train for one epoch."""
            operator.train()
            total_loss = 0.0
            num_batches = 0

            for batch_inputs, batch_targets in train_loader:
                batch_inputs = batch_inputs.to(self.device, non_blocking=True)
                batch_targets = batch_targets.to(self.device, non_blocking=True)

                # Forward pass
                optimizer.zero_grad()
                predictions = operator(batch_inputs)
                loss = criterion(predictions, batch_targets)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(operator.parameters(), self.config.gradient_clip)

                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            # Update scheduler (if not plateau-based)
            if scheduler and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            return total_loss / num_batches

        def _validate_epoch(self, operator: nn.Module, val_loader: DataLoader, criterion: nn.Module) -> float:
            """Validate for one epoch."""
            operator.eval()
            total_loss = 0.0
            num_batches = 0

            with torch.no_grad():
                for batch_inputs, batch_targets in val_loader:
                    batch_inputs = batch_inputs.to(self.device, non_blocking=True)
                    batch_targets = batch_targets.to(self.device, non_blocking=True)

                    predictions = operator(batch_inputs)
                    loss = criterion(predictions, batch_targets)

                    total_loss += loss.item()
                    num_batches += 1

            return total_loss / num_batches

        def _save_checkpoint(self, operator: nn.Module, path: Path, epoch: int, val_loss: float) -> None:
            """Save model checkpoint."""
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": operator.state_dict(),
                "val_loss": val_loss,
                "config": self.config,
            }
            torch.save(checkpoint, path)

    class MFGDataGenerator:
        """
        Data generator for MFG operator training.

        Generates parameter-solution pairs by sampling parameters and solving
        the corresponding MFG problems using traditional solvers.
        """

        def __init__(self, solver_factory: Callable, parameter_sampler: Callable):
            """
            Initialize data generator.

            Args:
                solver_factory: Function to create MFG solver given parameters
                parameter_sampler: Function to sample problem parameters
            """
            self.solver_factory = solver_factory
            self.parameter_sampler = parameter_sampler
            self.logger = get_logger(__name__)

        def generate_dataset(
            self, num_samples: int, save_path: Path | None = None, cache_solutions: bool = True
        ) -> OperatorDataset:
            """
            Generate training dataset.

            Args:
                num_samples: Number of parameter-solution pairs to generate
                save_path: Path to save generated dataset
                cache_solutions: Whether to cache solutions

            Returns:
                Generated operator dataset
            """
            self.logger.info(f"Generating {num_samples} parameter-solution pairs")

            parameters = []
            solutions = []

            for i in range(num_samples):
                # Sample parameters
                params = self.parameter_sampler()
                parameters.append(params)

                # Solve MFG problem
                solver = self.solver_factory(params)
                result = solver.solve()
                solutions.append(result.solution_vector)

                if (i + 1) % 10 == 0:
                    self.logger.info(f"Generated {i + 1}/{num_samples} samples")

            # Convert to arrays
            parameters = np.array(parameters)
            solutions = np.array(solutions)

            # Create dataset
            dataset = OperatorDataset(parameters, solutions)

            # Save if requested
            if save_path:
                dataset.save(save_path)
                self.logger.info(f"Dataset saved to {save_path}")

            return dataset

else:
    # Placeholder classes when PyTorch is not available
    class OperatorDataset:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("Operator training requires PyTorch")

    class OperatorTrainingManager:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("Operator training requires PyTorch")

    class MFGDataGenerator:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("Data generation requires PyTorch")
