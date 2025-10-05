"""
Population configuration for multi-population Mean Field Games.

Defines the specification for a single population in a heterogeneous
multi-population MFG system, including state/action spaces, algorithm
choice, and coupling parameters.

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


@dataclass
class PopulationConfig:
    """
    Configuration for a single population in multi-population MFG.

    Each population can have:
    - Different state and action dimensions
    - Different action bounds
    - Different continuous control algorithm (DDPG, TD3, SAC)
    - Population-specific coupling weights to other populations
    - Custom initial distribution

    Mathematical Framework:
    - State space: S_i ⊆ ℝ^{state_dim}
    - Action space: A_i ⊆ [a_min, a_max]^{action_dim}
    - Population distribution: m_i(t,x) with support on S_i
    """

    # Population identification
    population_id: str

    # State and action spaces
    state_dim: int
    action_dim: int
    action_bounds: tuple[float, float]

    # Algorithm specification
    algorithm: Literal["ddpg", "td3", "sac"]
    algorithm_config: dict[str, Any] = field(default_factory=dict)

    # Population interaction parameters
    coupling_weights: dict[str, float] = field(default_factory=dict)
    """Weights for influence from other populations: {pop_id: weight}"""

    # Initial distribution
    initial_distribution: Callable[[], NDArray] | None = None
    """Function returning initial state sample for this population"""

    # Population size (for simulation)
    population_size: int = 1000

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate population configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate dimensions
        if self.state_dim <= 0:
            msg = f"state_dim must be positive, got {self.state_dim}"
            raise ValueError(msg)

        if self.action_dim <= 0:
            msg = f"action_dim must be positive, got {self.action_dim}"
            raise ValueError(msg)

        # Validate action bounds
        if self.action_bounds[0] >= self.action_bounds[1]:
            msg = f"Invalid action_bounds: {self.action_bounds}, must have min < max"
            raise ValueError(msg)

        # Validate algorithm choice
        valid_algorithms = ["ddpg", "td3", "sac"]
        if self.algorithm not in valid_algorithms:
            msg = f"algorithm must be one of {valid_algorithms}, got '{self.algorithm}'"
            raise ValueError(msg)

        # Validate coupling weights
        for pop_id, weight in self.coupling_weights.items():
            if weight < 0:
                msg = f"Coupling weight for '{pop_id}' must be non-negative, got {weight}"
                raise ValueError(msg)

        # Validate population size
        if self.population_size <= 0:
            msg = f"population_size must be positive, got {self.population_size}"
            raise ValueError(msg)

    def get_action_scale(self) -> float:
        """Get action scaling factor: (max - min) / 2."""
        return (self.action_bounds[1] - self.action_bounds[0]) / 2.0

    def get_action_bias(self) -> float:
        """Get action bias: (max + min) / 2."""
        return (self.action_bounds[1] + self.action_bounds[0]) / 2.0

    def sample_initial_state(self) -> NDArray:
        """
        Sample initial state for this population.

        Returns:
            Initial state array of shape (state_dim,)

        Raises:
            RuntimeError: If no initial distribution is configured
        """
        if self.initial_distribution is None:
            msg = f"No initial distribution configured for population '{self.population_id}'"
            raise RuntimeError(msg)

        return self.initial_distribution()

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"PopulationConfig("
            f"id='{self.population_id}', "
            f"state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, "
            f"action_bounds={self.action_bounds}, "
            f"algorithm='{self.algorithm}', "
            f"population_size={self.population_size})"
        )


def validate_population_set(populations: dict[str, PopulationConfig]) -> None:
    """
    Validate a set of populations for multi-population MFG.

    Checks:
    - Number of populations in valid range [2, 5]
    - Population IDs match dictionary keys
    - Coupling weights reference valid populations
    - No self-coupling

    Args:
        populations: Dictionary {pop_id: PopulationConfig}

    Raises:
        ValueError: If population set is invalid
    """
    num_pops = len(populations)

    # Check number of populations
    if num_pops < 2:
        msg = f"Multi-population MFG requires at least 2 populations, got {num_pops}"
        raise ValueError(msg)

    if num_pops > 5:
        msg = f"Current implementation supports up to 5 populations, got {num_pops}"
        raise ValueError(msg)

    # Check ID consistency
    for pop_id, config in populations.items():
        if config.population_id != pop_id:
            msg = f"Population ID mismatch: key='{pop_id}' but config.population_id='{config.population_id}'"
            raise ValueError(msg)

    # Check coupling weights reference valid populations
    all_pop_ids = set(populations.keys())
    for pop_id, config in populations.items():
        for coupled_pop_id in config.coupling_weights:
            if coupled_pop_id not in all_pop_ids:
                msg = f"Population '{pop_id}' has coupling weight for unknown population '{coupled_pop_id}'"
                raise ValueError(msg)

            if coupled_pop_id == pop_id:
                msg = f"Population '{pop_id}' cannot have self-coupling"
                raise ValueError(msg)


def create_symmetric_coupling(
    population_ids: list[str],
    weight: float = 1.0,
) -> dict[str, dict[str, float]]:
    """
    Create symmetric coupling weights for populations.

    All populations have equal influence on each other.

    Args:
        population_ids: List of population IDs
        weight: Coupling weight for all pairs

    Returns:
        Dictionary {pop_id: {other_pop_id: weight}}

    Example:
        >>> create_symmetric_coupling(["cars", "trucks"], weight=0.5)
        {'cars': {'trucks': 0.5}, 'trucks': {'cars': 0.5}}
    """
    coupling = {}
    for pop_id in population_ids:
        coupling[pop_id] = {other_id: weight for other_id in population_ids if other_id != pop_id}
    return coupling


def create_asymmetric_coupling(
    population_ids: list[str],
    weight_matrix: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """
    Create asymmetric coupling weights from explicit matrix.

    Args:
        population_ids: List of population IDs
        weight_matrix: {from_pop: {to_pop: weight}}

    Returns:
        Validated coupling weight dictionary

    Raises:
        ValueError: If weight matrix is invalid

    Example:
        >>> create_asymmetric_coupling(
        ...     ["cars", "trucks"],
        ...     {"cars": {"trucks": 0.8}, "trucks": {"cars": 0.5}}
        ... )
        {'cars': {'trucks': 0.8}, 'trucks': {'cars': 0.5}}
    """
    # Validate all populations are present
    all_pop_ids = set(population_ids)
    matrix_pop_ids = set(weight_matrix.keys())

    if matrix_pop_ids != all_pop_ids:
        msg = f"Weight matrix populations {matrix_pop_ids} don't match {all_pop_ids}"
        raise ValueError(msg)

    # Validate no self-coupling
    for from_pop, weights in weight_matrix.items():
        if from_pop in weights:
            msg = f"Self-coupling not allowed: '{from_pop}' → '{from_pop}'"
            raise ValueError(msg)

    return weight_matrix
