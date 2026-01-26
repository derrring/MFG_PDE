"""Tests for nested MFGComponents structure with backward compatibility."""

import pytest

from mfg_pde.core.component_configs import (
    NeuralMFGConfig,
    RLMFGConfig,
    StandardMFGConfig,
)
from mfg_pde.core.mfg_components_nested import MFGComponents

# =============================================================================
# Test Nested Structure
# =============================================================================


def test_nested_structure_creation():
    """Nested structure can be created directly."""
    components = MFGComponents(
        standard=StandardMFGConfig(
            hamiltonian_func=lambda x, m, p, t: 0.5 * p**2 + m,
            potential_func=lambda x, t: x**2,
        ),
        neural=NeuralMFGConfig(
            neural_architecture={"layers": [64, 64, 64], "activation": "tanh"},
            loss_weights={"pde": 1.0, "ic": 10.0, "bc": 10.0},
        ),
    )

    assert components.standard is not None
    assert components.neural is not None
    assert components.standard.hamiltonian_func is not None
    assert components.neural.neural_architecture == {
        "layers": [64, 64, 64],
        "activation": "tanh",
    }


def test_nested_access():
    """Nested access pattern works correctly."""

    def my_hamiltonian(x, m, p, t):
        return 0.5 * p**2 + 2.0 * m

    components = MFGComponents(standard=StandardMFGConfig(hamiltonian_func=my_hamiltonian))

    # Nested access
    assert components.standard.hamiltonian_func is my_hamiltonian
    result = components.standard.hamiltonian_func(0, 1, 1, 0)
    assert result == 2.5


# =============================================================================
# Test Backward Compatibility (Flat Access)
# =============================================================================


def test_flat_structure_backward_compatibility():
    """Flat property access works for backward compatibility."""

    def my_hamiltonian(x, m, p, t):
        return 0.5 * p**2 + 2.0 * m

    # Create with nested structure
    components = MFGComponents(standard=StandardMFGConfig(hamiltonian_func=my_hamiltonian))

    # Access via flat property (backward compatible)
    assert components.hamiltonian_func is my_hamiltonian
    result = components.hamiltonian_func(0, 1, 1, 0)
    assert result == 2.5


def test_flat_setter_creates_nested():
    """Setting flat properties automatically creates nested configs."""
    components = MFGComponents()

    # Set via flat property
    components.hamiltonian_func = lambda x, m, p, t: 0.5 * p**2

    # Verify nested structure created
    assert components.standard is not None
    assert components.standard.hamiltonian_func is not None

    # Both access patterns work
    assert components.hamiltonian_func(0, 0, 2, 0) == 2.0
    assert components.standard.hamiltonian_func(0, 0, 2, 0) == 2.0


def test_flat_and_nested_access_consistency():
    """Flat and nested access return same values."""

    def my_H(x, m, p, t):
        return 0.5 * p**2 + m

    def my_V(x, t):
        return x**2

    components = MFGComponents(
        standard=StandardMFGConfig(hamiltonian_func=my_H, potential_func=my_V, m_initial=lambda x: 1.0)
    )

    # Both access patterns return same values
    assert components.hamiltonian_func is components.standard.hamiltonian_func
    assert components.potential_func is components.standard.potential_func
    assert components.m_initial is components.standard.m_initial


def test_lazy_config_creation():
    """Optional configs created on demand."""
    components = MFGComponents()

    # Network config not created yet
    assert components.network is None

    # Setting property creates config
    components.network_geometry = "some_geometry"

    # Now network config exists
    assert components.network is not None
    assert components.network.network_geometry == "some_geometry"


# =============================================================================
# Test Mixed Usage
# =============================================================================


def test_mixed_nested_and_flat():
    """Can mix nested creation with flat access."""
    components = MFGComponents(standard=StandardMFGConfig(hamiltonian_func=lambda x, m, p, t: 0.5 * p**2))

    # Add neural via flat property
    components.neural_architecture = {"layers": [128, 128]}

    # Both ways to access work
    assert components.neural is not None
    assert components.neural.neural_architecture == {"layers": [128, 128]}
    assert components.neural_architecture == {"layers": [128, 128]}


def test_multiple_formulations():
    """Can use multiple formulations together."""
    components = MFGComponents(
        standard=StandardMFGConfig(hamiltonian_func=lambda x, m, p, t: 0.5 * p**2),
        neural=NeuralMFGConfig(neural_architecture={"layers": [64, 64]}),
        rl=RLMFGConfig(reward_func=lambda s, a, m, t: -(a**2), action_space_bounds=[(-1.0, 1.0)]),
    )

    assert components.standard is not None
    assert components.neural is not None
    assert components.rl is not None

    # All accessible via nested and flat
    assert components.hamiltonian_func is not None
    assert components.neural_architecture is not None
    assert components.reward_func is not None


# =============================================================================
# Test Validation
# =============================================================================


def test_validation_catches_missing_hamiltonian():
    """Validation warns about neural without Hamiltonian."""
    components = MFGComponents(neural=NeuralMFGConfig(neural_architecture={"layers": [64, 64]}))

    warnings = components.validate()
    assert len(warnings) > 0
    assert any("hamiltonian" in w.lower() for w in warnings)


def test_validation_rl_missing_action_space():
    """Validation warns about RL without action space."""
    components = MFGComponents(rl=RLMFGConfig(reward_func=lambda s, a, m, t: -(a**2)))

    warnings = components.validate()
    assert len(warnings) > 0
    assert any("action_space" in w.lower() for w in warnings)


def test_validation_strict_mode():
    """Strict validation raises on warnings."""
    components = MFGComponents(neural=NeuralMFGConfig(neural_architecture={"layers": [64, 64]}))

    with pytest.raises(ValueError, match="validation failed"):
        components.validate(strict=True)


def test_validation_passes_for_valid_config():
    """No warnings for properly configured components."""
    components = MFGComponents(
        standard=StandardMFGConfig(
            hamiltonian_func=lambda x, m, p, t: 0.5 * p**2,
            m_initial=lambda x: 1.0,
        )
    )

    warnings = components.validate()
    assert len(warnings) == 0


# =============================================================================
# Test Real-World Usage Patterns
# =============================================================================


def test_standard_mfg_pattern():
    """Standard MFG usage pattern."""

    def H(x, m, p, t):
        return 0.5 * p**2 + 2.0 * m

    def m0(x):
        return 1.0 if -0.5 <= x <= 0.5 else 0.0

    # Nested style (recommended)
    components = MFGComponents(
        standard=StandardMFGConfig(hamiltonian_func=H, m_initial=m0, potential_func=lambda x, t: 0)
    )

    assert components.hamiltonian_func(0, 1, 1, 0) == 2.5
    assert components.m_initial(0) == 1.0


def test_neural_mfg_pattern():
    """Neural MFG usage pattern."""
    components = MFGComponents(
        standard=StandardMFGConfig(hamiltonian_func=lambda x, m, p, t: 0.5 * p**2 + m),
        neural=NeuralMFGConfig(
            neural_architecture={"layers": [128, 128, 128], "activation": "relu"},
            loss_weights={"pde": 1.0, "ic": 20.0, "bc": 20.0},
        ),
    )

    warnings = components.validate()
    assert len(warnings) == 0  # Should be valid


def test_composed_stochastic_mfg():
    """Composed MFG with stochasticity."""
    from mfg_pde.core.component_configs import StochasticMFGConfig

    components = MFGComponents(
        standard=StandardMFGConfig(hamiltonian_func=lambda x, m, p, t: 0.5 * p**2),
        stochastic=StochasticMFGConfig(noise_intensity=0.2, common_noise_func=lambda t: 0.1 * t),
    )

    assert components.noise_intensity == 0.2
    assert components.common_noise_func(5) == 0.5
