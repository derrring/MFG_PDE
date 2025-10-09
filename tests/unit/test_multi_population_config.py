"""
Unit tests for multi-population configuration.

Tests PopulationConfig validation, coupling utilities, and configuration
validation for multi-population MFG systems.

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

import pytest

import numpy as np

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from mfg_pde.alg.reinforcement.multi_population.population_config import (
        PopulationConfig,
        create_asymmetric_coupling,
        create_symmetric_coupling,
        validate_population_set,
    )


class TestPopulationConfig:
    """Tests for PopulationConfig dataclass."""

    def test_valid_configuration(self):
        """Test creating a valid population configuration."""
        config = PopulationConfig(
            population_id="cars",
            state_dim=2,
            action_dim=1,
            action_bounds=(-3.0, 3.0),
            algorithm="ddpg",
            algorithm_config={"actor_lr": 1e-4},
            coupling_weights={"trucks": 0.5},
            population_size=100,
        )

        assert config.population_id == "cars"
        assert config.state_dim == 2
        assert config.action_dim == 1
        assert config.action_bounds == (-3.0, 3.0)
        assert config.algorithm == "ddpg"
        assert config.population_size == 100

    def test_invalid_state_dim(self):
        """Test that negative state_dim raises ValueError."""
        with pytest.raises(ValueError, match="state_dim must be positive"):
            PopulationConfig(
                population_id="test",
                state_dim=-1,
                action_dim=1,
                action_bounds=(0.0, 1.0),
                algorithm="ddpg",
            )

    def test_invalid_action_dim(self):
        """Test that negative action_dim raises ValueError."""
        with pytest.raises(ValueError, match="action_dim must be positive"):
            PopulationConfig(
                population_id="test",
                state_dim=2,
                action_dim=0,
                action_bounds=(0.0, 1.0),
                algorithm="ddpg",
            )

    def test_invalid_action_bounds(self):
        """Test that invalid action bounds raise ValueError."""
        with pytest.raises(ValueError, match="Invalid action_bounds"):
            PopulationConfig(
                population_id="test",
                state_dim=2,
                action_dim=1,
                action_bounds=(1.0, 0.0),  # min > max
                algorithm="ddpg",
            )

    def test_invalid_algorithm(self):
        """Test that invalid algorithm raises ValueError."""
        with pytest.raises(ValueError, match="algorithm must be one of"):
            PopulationConfig(
                population_id="test",
                state_dim=2,
                action_dim=1,
                action_bounds=(0.0, 1.0),
                algorithm="invalid",
            )

    def test_negative_coupling_weight(self):
        """Test that negative coupling weight raises ValueError."""
        with pytest.raises(ValueError, match=r"Coupling weight.*must be non-negative"):
            PopulationConfig(
                population_id="test",
                state_dim=2,
                action_dim=1,
                action_bounds=(0.0, 1.0),
                algorithm="ddpg",
                coupling_weights={"other": -0.5},
            )

    def test_action_scale_and_bias(self):
        """Test action scale and bias calculations."""
        config = PopulationConfig(
            population_id="test",
            state_dim=2,
            action_dim=1,
            action_bounds=(-3.0, 3.0),
            algorithm="ddpg",
        )

        assert config.get_action_scale() == 3.0  # (3 - (-3)) / 2
        assert config.get_action_bias() == 0.0  # (3 + (-3)) / 2

    def test_sample_initial_state(self):
        """Test sampling from initial distribution."""
        config = PopulationConfig(
            population_id="test",
            state_dim=2,
            action_dim=1,
            action_bounds=(0.0, 1.0),
            algorithm="ddpg",
            initial_distribution=lambda: np.array([1.0, 2.0]),
        )

        state = config.sample_initial_state()
        assert state.shape == (2,)
        assert np.allclose(state, [1.0, 2.0])

    def test_sample_initial_state_without_distribution(self):
        """Test that sampling without distribution raises error."""
        config = PopulationConfig(
            population_id="test",
            state_dim=2,
            action_dim=1,
            action_bounds=(0.0, 1.0),
            algorithm="ddpg",
        )

        with pytest.raises(RuntimeError, match="No initial distribution configured"):
            config.sample_initial_state()


class TestPopulationSetValidation:
    """Tests for validate_population_set function."""

    def test_valid_population_set(self):
        """Test validating a valid population set."""
        populations = {
            "cars": PopulationConfig(
                population_id="cars",
                state_dim=2,
                action_dim=1,
                action_bounds=(-3.0, 3.0),
                algorithm="ddpg",
                coupling_weights={"trucks": 0.5},
            ),
            "trucks": PopulationConfig(
                population_id="trucks",
                state_dim=2,
                action_dim=1,
                action_bounds=(-2.0, 2.0),
                algorithm="td3",
                coupling_weights={"cars": 0.3},
            ),
        }

        # Should not raise
        validate_population_set(populations)

    def test_too_few_populations(self):
        """Test that single population raises ValueError."""
        populations = {
            "cars": PopulationConfig(
                population_id="cars",
                state_dim=2,
                action_dim=1,
                action_bounds=(-3.0, 3.0),
                algorithm="ddpg",
            ),
        }

        with pytest.raises(ValueError, match="requires at least 2 populations"):
            validate_population_set(populations)

    def test_too_many_populations(self):
        """Test that >5 populations raises ValueError."""
        populations = {
            f"pop_{i}": PopulationConfig(
                population_id=f"pop_{i}",
                state_dim=2,
                action_dim=1,
                action_bounds=(0.0, 1.0),
                algorithm="ddpg",
            )
            for i in range(6)
        }

        with pytest.raises(ValueError, match="supports up to 5 populations"):
            validate_population_set(populations)

    def test_id_mismatch(self):
        """Test that ID mismatch raises ValueError."""
        populations = {
            "cars": PopulationConfig(
                population_id="trucks",  # Mismatch!
                state_dim=2,
                action_dim=1,
                action_bounds=(0.0, 1.0),
                algorithm="ddpg",
            ),
            "trucks": PopulationConfig(
                population_id="trucks",
                state_dim=2,
                action_dim=1,
                action_bounds=(0.0, 1.0),
                algorithm="td3",
            ),
        }

        with pytest.raises(ValueError, match="Population ID mismatch"):
            validate_population_set(populations)

    def test_invalid_coupling_reference(self):
        """Test that coupling to non-existent population raises ValueError."""
        populations = {
            "cars": PopulationConfig(
                population_id="cars",
                state_dim=2,
                action_dim=1,
                action_bounds=(0.0, 1.0),
                algorithm="ddpg",
                coupling_weights={"nonexistent": 0.5},
            ),
            "trucks": PopulationConfig(
                population_id="trucks",
                state_dim=2,
                action_dim=1,
                action_bounds=(0.0, 1.0),
                algorithm="td3",
            ),
        }

        with pytest.raises(ValueError, match="unknown population"):
            validate_population_set(populations)

    def test_self_coupling(self):
        """Test that self-coupling raises ValueError."""
        populations = {
            "cars": PopulationConfig(
                population_id="cars",
                state_dim=2,
                action_dim=1,
                action_bounds=(0.0, 1.0),
                algorithm="ddpg",
                coupling_weights={"cars": 0.5},  # Self-coupling!
            ),
            "trucks": PopulationConfig(
                population_id="trucks",
                state_dim=2,
                action_dim=1,
                action_bounds=(0.0, 1.0),
                algorithm="td3",
            ),
        }

        with pytest.raises(ValueError, match="cannot have self-coupling"):
            validate_population_set(populations)


class TestCouplingUtilities:
    """Tests for coupling weight utilities."""

    def test_symmetric_coupling(self):
        """Test creating symmetric coupling weights."""
        coupling = create_symmetric_coupling(["cars", "trucks"], weight=0.5)

        assert coupling == {
            "cars": {"trucks": 0.5},
            "trucks": {"cars": 0.5},
        }

    def test_symmetric_coupling_three_populations(self):
        """Test symmetric coupling with three populations."""
        coupling = create_symmetric_coupling(["cars", "trucks", "motorcycles"], weight=1.0)

        assert coupling == {
            "cars": {"trucks": 1.0, "motorcycles": 1.0},
            "trucks": {"cars": 1.0, "motorcycles": 1.0},
            "motorcycles": {"cars": 1.0, "trucks": 1.0},
        }

    def test_asymmetric_coupling(self):
        """Test creating asymmetric coupling weights."""
        weight_matrix = {
            "cars": {"trucks": 0.8},
            "trucks": {"cars": 0.5},
        }

        coupling = create_asymmetric_coupling(["cars", "trucks"], weight_matrix)

        assert coupling == weight_matrix

    def test_asymmetric_coupling_missing_population(self):
        """Test that missing population in weight matrix raises error."""
        weight_matrix = {
            "cars": {"trucks": 0.8},
            # Missing "trucks" entry
        }

        with pytest.raises(ValueError, match="don't match"):
            create_asymmetric_coupling(["cars", "trucks"], weight_matrix)

    def test_asymmetric_coupling_self_coupling(self):
        """Test that self-coupling in weight matrix raises error."""
        weight_matrix = {
            "cars": {"cars": 0.5, "trucks": 0.8},  # Self-coupling!
            "trucks": {"cars": 0.5},
        }

        with pytest.raises(ValueError, match="Self-coupling not allowed"):
            create_asymmetric_coupling(["cars", "trucks"], weight_matrix)
