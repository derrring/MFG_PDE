"""
Test cases for Neural Operator Methods.

This module tests the neural operator learning infrastructure including
Fourier Neural Operators (FNO), DeepONet, and the training framework
for Mean Field Games applications.
"""

import tempfile
from pathlib import Path

import pytest

import numpy as np

# Skip PyTorch tests if not available
pytorch_available = True
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset

    from mfg_pde.alg.neural.operator_learning import (
        BaseNeuralOperator,
        DeepONet,
        DeepONetConfig,
        FNOConfig,
        FourierNeuralOperator,
        OperatorConfig,
        OperatorDataset,
        OperatorResult,
        OperatorTrainingManager,
        TrainingConfig,
    )
except ImportError:
    pytorch_available = False


class TestOperatorConfig:
    """Test operator configuration classes."""

    def test_operator_config_defaults(self):
        """Test default configuration values."""
        config = OperatorConfig()

        assert config.input_dim == 64
        assert config.output_dim == 1024
        assert config.learning_rate == 1e-3
        assert config.batch_size == 32
        assert config.operator_type == "fno"

    def test_fno_config_defaults(self):
        """Test FNO-specific configuration."""
        if not pytorch_available:
            pytest.skip("PyTorch not available")

        config = FNOConfig()

        assert config.modes == 16
        assert config.width == 64
        assert config.num_layers == 4
        assert config.operator_type == "fno"

    def test_deeponet_config_defaults(self):
        """Test DeepONet-specific configuration."""
        if not pytorch_available:
            pytest.skip("PyTorch not available")

        config = DeepONetConfig()

        assert config.branch_depth == 6
        assert config.trunk_depth == 6
        assert config.latent_dim == 64
        assert config.operator_type == "deeponet"


class TestOperatorDataset:
    """Test operator dataset management."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        if not pytorch_available:
            pytest.skip("PyTorch not available")

        # Generate synthetic parameter-solution pairs
        num_samples = 100
        param_dim = 10
        solution_dim = 64

        parameters = torch.randn(num_samples, param_dim)
        solutions = torch.randn(num_samples, solution_dim)

        return parameters, solutions

    def test_dataset_initialization(self, sample_data):
        """Test dataset initialization."""
        if not pytorch_available:
            pytest.skip("PyTorch not available")

        parameters, solutions = sample_data

        dataset = OperatorDataset(parameters, solutions, normalize=False)

        assert len(dataset) == 100
        assert dataset.parameters.shape == (100, 10)
        assert dataset.solutions.shape == (100, 64)

    def test_dataset_normalization(self, sample_data):
        """Test data normalization."""
        if not pytorch_available:
            pytest.skip("PyTorch not available")

        parameters, solutions = sample_data

        dataset = OperatorDataset(parameters, solutions, normalize=True)

        # Check normalization statistics
        assert hasattr(dataset, "param_mean")
        assert hasattr(dataset, "param_std")
        assert hasattr(dataset, "solution_mean")
        assert hasattr(dataset, "solution_std")

        # Check that normalized data has approximately zero mean and unit std
        assert torch.abs(torch.mean(dataset.parameters)).item() < 1e-5
        assert torch.abs(torch.std(dataset.parameters) - 1.0).item() < 1e-1

    def test_dataset_getitem(self, sample_data):
        """Test dataset item access."""
        if not pytorch_available:
            pytest.skip("PyTorch not available")

        parameters, solutions = sample_data

        dataset = OperatorDataset(parameters, solutions, normalize=False)

        # Test single item access
        param, solution = dataset[0]
        assert param.shape == (10,)
        assert solution.shape == (64,)

    def test_dataset_save_load(self, sample_data):
        """Test dataset save and load functionality."""
        if not pytorch_available:
            pytest.skip("PyTorch not available")

        parameters, solutions = sample_data

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save dataset
            dataset = OperatorDataset(parameters, solutions, normalize=True)
            save_path = Path(temp_dir) / "test_dataset.pt"
            dataset.save(save_path)

            # Load dataset
            loaded_dataset = OperatorDataset.load(save_path)

            # Check that data is preserved
            assert len(loaded_dataset) == len(dataset)
            assert torch.allclose(loaded_dataset.parameters, dataset.parameters)
            assert torch.allclose(loaded_dataset.solutions, dataset.solutions)

    def test_dataset_with_coordinates(self):
        """Test dataset with coordinate data for DeepONet."""
        if not pytorch_available:
            pytest.skip("PyTorch not available")

        num_samples = 50
        sensor_points = 20
        num_eval_points = 30
        coord_dim = 2

        # Generate data
        parameters = torch.randn(num_samples, sensor_points)
        coordinates = torch.rand(num_samples, num_eval_points, coord_dim)
        solutions = torch.randn(num_samples, num_eval_points)

        dataset = OperatorDataset(parameters, solutions, coordinates, normalize=False)

        # Test item access
        combined_input, solution = dataset[0]
        expected_input_size = sensor_points + num_eval_points * coord_dim
        assert combined_input.shape == (expected_input_size,)
        assert solution.shape == (num_eval_points,)


@pytest.mark.skipif(not pytorch_available, reason="PyTorch not available")
class TestFourierNeuralOperator:
    """Test Fourier Neural Operator implementation."""

    def test_fno_initialization(self):
        """Test FNO initialization."""
        config = FNOConfig(input_dim=32, output_dim=256, modes=8, width=32, num_layers=2)

        fno = FourierNeuralOperator(config)

        assert fno.config.modes == 8
        assert fno.config.width == 32
        assert fno.config.num_layers == 2

    def test_fno_forward_pass_1d(self):
        """Test FNO forward pass for 1D problems."""
        config = FNOConfig(
            input_dim=16,
            output_dim=64,  # 1D output
            output_resolution=64,
            modes=4,
            width=16,
            num_layers=2,
        )

        fno = FourierNeuralOperator(config)
        fno.eval()

        # Test forward pass
        batch_size = 8
        input_params = torch.randn(batch_size, config.input_dim)

        with torch.no_grad():
            output = fno(input_params)

        assert output.shape == (batch_size, config.output_dim)
        assert not torch.isnan(output).any()

    def test_fno_forward_pass_2d(self):
        """Test FNO forward pass for 2D problems."""
        resolution = 32
        config = FNOConfig(
            input_dim=16,
            output_dim=resolution * resolution,  # 2D output
            output_resolution=resolution,
            modes=4,
            width=16,
            num_layers=2,
        )

        fno = FourierNeuralOperator(config)
        fno.eval()

        # Test forward pass
        batch_size = 4
        input_params = torch.randn(batch_size, config.input_dim)

        with torch.no_grad():
            output = fno(input_params)

        assert output.shape == (batch_size, resolution * resolution)
        assert not torch.isnan(output).any()

    def test_fno_spectral_conv_1d(self):
        """Test 1D spectral convolution layer."""
        from mfg_pde.alg.neural.operator_learning.fourier_neural_operator import SpectralConv1d

        in_channels = 8
        out_channels = 16
        modes = 4
        spatial_size = 32

        layer = SpectralConv1d(in_channels, out_channels, modes)

        # Test forward pass
        x = torch.randn(2, in_channels, spatial_size)
        output = layer(x)

        assert output.shape == (2, out_channels, spatial_size)
        assert not torch.isnan(output).any()

    def test_fno_spectral_conv_2d(self):
        """Test 2D spectral convolution layer."""
        from mfg_pde.alg.neural.operator_learning.fourier_neural_operator import SpectralConv2d

        in_channels = 8
        out_channels = 16
        modes = 4
        height, width = 16, 16

        layer = SpectralConv2d(in_channels, out_channels, modes, modes)

        # Test forward pass
        x = torch.randn(2, in_channels, height, width)
        output = layer(x)

        assert output.shape == (2, out_channels, height, width)
        assert not torch.isnan(output).any()


@pytest.mark.skipif(not pytorch_available, reason="PyTorch not available")
class TestDeepONet:
    """Test DeepONet implementation."""

    def test_deeponet_initialization(self):
        """Test DeepONet initialization."""
        config = DeepONetConfig(sensor_points=50, coordinate_dim=2, branch_depth=4, trunk_depth=4, latent_dim=32)

        deeponet = DeepONet(config)

        assert deeponet.config.sensor_points == 50
        assert deeponet.config.coordinate_dim == 2
        assert deeponet.config.latent_dim == 32

    def test_deeponet_branch_network(self):
        """Test branch network component."""
        from mfg_pde.alg.neural.operator_learning.deeponet import BranchNetwork

        config = DeepONetConfig(sensor_points=20, branch_depth=3, branch_width=64, latent_dim=16)

        branch_net = BranchNetwork(config)

        # Test forward pass
        batch_size = 4
        sensor_data = torch.randn(batch_size, config.sensor_points)
        output = branch_net(sensor_data)

        assert output.shape == (batch_size, config.latent_dim)
        assert not torch.isnan(output).any()

    def test_deeponet_trunk_network(self):
        """Test trunk network component."""
        from mfg_pde.alg.neural.operator_learning.deeponet import TrunkNetwork

        config = DeepONetConfig(coordinate_dim=2, trunk_depth=3, trunk_width=64, latent_dim=16)

        trunk_net = TrunkNetwork(config)

        # Test forward pass
        batch_size = 4
        num_points = 10
        coordinates = torch.randn(batch_size, num_points, config.coordinate_dim)
        output = trunk_net(coordinates)

        assert output.shape == (batch_size, num_points, config.latent_dim)
        assert not torch.isnan(output).any()

    def test_deeponet_forward_pass(self):
        """Test DeepONet forward pass."""
        config = DeepONetConfig(sensor_points=30, coordinate_dim=2, latent_dim=16, branch_depth=3, trunk_depth=3)

        deeponet = DeepONet(config)
        deeponet.eval()

        # Test forward pass
        batch_size = 4
        num_eval_points = 20

        branch_input = torch.randn(batch_size, config.sensor_points)
        trunk_input = torch.randn(batch_size, num_eval_points, config.coordinate_dim)

        with torch.no_grad():
            output = deeponet.forward_operator(branch_input, trunk_input)

        assert output.shape == (batch_size, num_eval_points)
        assert not torch.isnan(output).any()

    def test_deeponet_evaluate_at_coordinates(self):
        """Test DeepONet coordinate evaluation."""
        config = DeepONetConfig(sensor_points=25, coordinate_dim=1, latent_dim=12)

        deeponet = DeepONet(config)
        deeponet.eval()

        # Test evaluation
        batch_size = 3
        num_points = 15

        parameter_function = torch.randn(batch_size, config.sensor_points)
        coordinates = torch.linspace(0, 1, num_points).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1)

        output = deeponet.evaluate_at_coordinates(parameter_function, coordinates)

        assert output.shape == (batch_size, num_points)
        assert not torch.isnan(output).any()


@pytest.mark.skipif(not pytorch_available, reason="PyTorch not available")
class TestOperatorTraining:
    """Test operator training framework."""

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        num_samples = 50
        param_dim = 8
        solution_dim = 32

        parameters = torch.randn(num_samples, param_dim)
        solutions = torch.randn(num_samples, solution_dim)

        return OperatorDataset(parameters, solutions, normalize=True)

    def test_training_config(self):
        """Test training configuration."""
        config = TrainingConfig(max_epochs=100, batch_size=16, learning_rate=1e-4, patience=20)

        assert config.max_epochs == 100
        assert config.batch_size == 16
        assert config.learning_rate == 1e-4
        assert config.patience == 20

    def test_training_manager_initialization(self):
        """Test training manager initialization."""
        config = TrainingConfig(device="cpu")
        manager = OperatorTrainingManager(config)

        assert manager.config.device == "cpu"
        assert manager.device.type == "cpu"

    def test_operator_training_workflow(self, sample_training_data):
        """Test complete training workflow."""
        # Create simple operator
        config = FNOConfig(input_dim=8, output_dim=32, modes=4, width=16, num_layers=2)
        operator = FourierNeuralOperator(config)

        # Training configuration
        train_config = TrainingConfig(
            max_epochs=5,  # Short training for testing
            batch_size=8,
            learning_rate=1e-3,
            patience=10,
            device="cpu",
            verbose=False,
        )

        # Training manager
        manager = OperatorTrainingManager(train_config)

        # Train operator
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir)
            results = manager.train_operator(operator, sample_training_data, save_path)

            # Check results
            assert "train_losses" in results
            assert "val_losses" in results
            assert "training_time" in results
            assert len(results["train_losses"]) > 0
            assert len(results["val_losses"]) > 0

    def test_loss_function_creation(self):
        """Test loss function creation."""
        config = TrainingConfig(loss_type="mse")
        manager = OperatorTrainingManager(config)

        criterion = manager._create_loss_function()
        assert isinstance(criterion, nn.MSELoss)

        config.loss_type = "l1"
        criterion = manager._create_loss_function()
        assert isinstance(criterion, nn.L1Loss)

    def test_optimizer_creation(self):
        """Test optimizer creation."""
        config = TrainingConfig(optimizer="adam", learning_rate=1e-3)
        manager = OperatorTrainingManager(config)

        # Create dummy model
        model = nn.Linear(10, 5)
        optimizer = manager._create_optimizer(model)

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["lr"] == 1e-3


class TestOperatorFactory:
    """Test operator factory functions."""

    def test_create_mfg_operator_fno(self):
        """Test FNO creation via factory."""
        if not pytorch_available:
            pytest.skip("PyTorch not available")

        from mfg_pde.alg.neural.operator_learning import create_mfg_operator

        config = {"input_dim": 16, "output_dim": 64, "modes": 8, "width": 32}

        operator = create_mfg_operator("fno", config)
        assert isinstance(operator, FourierNeuralOperator)

    def test_create_mfg_operator_deeponet(self):
        """Test DeepONet creation via factory."""
        if not pytorch_available:
            pytest.skip("PyTorch not available")

        from mfg_pde.alg.neural.operator_learning import create_mfg_operator

        config = {"sensor_points": 30, "coordinate_dim": 2, "latent_dim": 16}

        operator = create_mfg_operator("deeponet", config)
        assert isinstance(operator, DeepONet)

    def test_create_mfg_operator_invalid(self):
        """Test factory with invalid operator type."""
        if not pytorch_available:
            pytest.skip("PyTorch not available")

        from mfg_pde.alg.neural.operator_learning import create_mfg_operator

        with pytest.raises(ValueError):
            create_mfg_operator("invalid_type", {})


class TestOperatorBenchmark:
    """Test operator benchmarking functionality."""

    def test_operator_benchmark(self):
        """Test operator performance benchmarking."""
        if not pytorch_available:
            pytest.skip("PyTorch not available")

        from mfg_pde.alg.neural.operator_learning import operator_benchmark

        # Create simple operator
        config = FNOConfig(input_dim=10, output_dim=20, modes=4, width=16)
        operator = FourierNeuralOperator(config)
        operator.eval()
        operator.is_trained = True  # Mark as trained

        # Test data
        test_data = np.random.randn(5, 10)

        # Benchmark
        metrics = operator_benchmark(operator, test_data)

        assert "evaluation_time" in metrics
        assert "throughput" in metrics
        assert metrics["evaluation_time"] > 0
        assert metrics["throughput"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
