"""
Neural network architectures for multi-population Mean Field Games.

Implements:
- JointPopulationEncoder: Encodes multiple population distributions
- MultiPopulationActor: Actor for heterogeneous populations
- MultiPopulationCritic: Q-function for heterogeneous populations
- MultiPopulationStochasticActor: Stochastic policy (SAC)

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .population_config import PopulationConfig

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Normal

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    Normal = None  # type: ignore

# Constants for SAC
LOG_STD_MIN = -20
LOG_STD_MAX = 2


class JointPopulationEncoder(nn.Module):
    """
    Encoder for multiple population distributions.

    Encodes joint state (m_1, ..., m_N) of all populations into
    a compact representation for actor/critic networks.

    Architecture:
    1. Per-population encoders: m_i → h_i
    2. Optional cross-population attention
    3. Joint aggregation: [h_1, ..., h_N] → z
    """

    def __init__(
        self,
        population_configs: dict[str, PopulationConfig],
        hidden_dim: int = 128,
        use_attention: bool = False,
    ):
        """
        Initialize joint population encoder.

        Args:
            population_configs: {pop_id: PopulationConfig}
            hidden_dim: Hidden dimension for encodings
            use_attention: Enable cross-population attention
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        super().__init__()

        self.population_ids = sorted(population_configs.keys())
        self.use_attention = use_attention

        # Per-population encoders
        self.pop_encoders = nn.ModuleDict()
        for pop_id, config in population_configs.items():
            # Assume distributions are flattened histograms
            # For flexibility, use state_dim as proxy for distribution size
            dist_dim = config.state_dim * 10  # Assume 10-bin histogram per dimension

            self.pop_encoders[pop_id] = nn.Sequential(
                nn.Linear(dist_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        # Cross-population attention (optional)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                batch_first=True,
            )

        # Joint encoder
        num_pops = len(population_configs)
        self.joint_encoder = nn.Sequential(
            nn.Linear(num_pops * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.output_dim = hidden_dim // 2

    def forward(self, population_states: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode joint population state.

        Args:
            population_states: {pop_id: distribution [batch, dist_dim]}

        Returns:
            Joint encoding [batch, output_dim]
        """
        # Encode each population
        pop_features = []
        for pop_id in self.population_ids:
            h_i = self.pop_encoders[pop_id](population_states[pop_id])
            pop_features.append(h_i)

        # Stack: [batch, num_pops, hidden_dim]
        stacked_features = torch.stack(pop_features, dim=1)

        # Cross-population attention (optional)
        if self.use_attention:
            attended_features, _ = self.attention(stacked_features, stacked_features, stacked_features)
            joint_features = attended_features.flatten(start_dim=1)
        else:
            joint_features = stacked_features.flatten(start_dim=1)

        # Joint encoding
        return self.joint_encoder(joint_features)


class MultiPopulationActor(nn.Module):
    """
    Deterministic actor for multi-population MFG.

    Architecture:
    - State encoder: Process own state s_i
    - Joint population encoder: Process all distributions (m_1, ..., m_N)
    - Action head: Generate action a_i ∈ A_i
    """

    def __init__(
        self,
        pop_id: str,
        state_dim: int,
        action_dim: int,
        action_bounds: tuple[float, float],
        population_configs: dict[str, PopulationConfig],
        hidden_dims: list[int] | None = None,
    ):
        """
        Initialize multi-population actor.

        Args:
            pop_id: ID of this population
            state_dim: Own state dimension
            action_dim: Own action dimension
            action_bounds: Own action bounds (min, max)
            population_configs: All population configs
            hidden_dims: Hidden layer dimensions
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.pop_id = pop_id
        self.action_bounds = action_bounds

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )

        # Joint population encoder
        self.pop_encoder = JointPopulationEncoder(
            population_configs=population_configs,
            hidden_dim=hidden_dims[1],
            use_attention=False,
        )

        # Action head
        combined_dim = hidden_dims[1] + self.pop_encoder.output_dim
        self.action_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim),
            nn.Tanh(),
        )

        # Action scaling
        self.action_scale = (action_bounds[1] - action_bounds[0]) / 2.0
        self.action_bias = (action_bounds[1] + action_bounds[0]) / 2.0

    def forward(self, state: torch.Tensor, population_states: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Generate action.

        Args:
            state: Own state [batch, state_dim]
            population_states: {pop_id: distribution [batch, dist_dim]}

        Returns:
            Action [batch, action_dim]
        """
        state_feat = self.state_encoder(state)
        pop_feat = self.pop_encoder(population_states)

        combined = torch.cat([state_feat, pop_feat], dim=1)
        action_raw = self.action_head(combined)

        # Scale to bounds
        action = action_raw * self.action_scale + self.action_bias
        return action


class MultiPopulationCritic(nn.Module):
    """
    Q-function for multi-population MFG.

    Q_i(s_i, a_i, m_1, ..., m_N) for population i.
    """

    def __init__(
        self,
        pop_id: str,
        state_dim: int,
        action_dim: int,
        population_configs: dict[str, PopulationConfig],
        hidden_dims: list[int] | None = None,
    ):
        """
        Initialize multi-population critic.

        Args:
            pop_id: ID of this population
            state_dim: Own state dimension
            action_dim: Own action dimension
            population_configs: All population configs
            hidden_dims: Hidden layer dimensions
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.pop_id = pop_id

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
        )

        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
        )

        # Joint population encoder
        self.pop_encoder = JointPopulationEncoder(
            population_configs=population_configs,
            hidden_dim=hidden_dims[1],
            use_attention=False,
        )

        # Q-value head
        combined_dim = hidden_dims[0] + 64 + self.pop_encoder.output_dim
        self.q_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        population_states: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute Q-value.

        Args:
            state: Own state [batch, state_dim]
            action: Own action [batch, action_dim]
            population_states: {pop_id: distribution [batch, dist_dim]}

        Returns:
            Q-value [batch]
        """
        state_feat = self.state_encoder(state)
        action_feat = self.action_encoder(action)
        pop_feat = self.pop_encoder(population_states)

        combined = torch.cat([state_feat, action_feat, pop_feat], dim=1)
        return self.q_head(combined).squeeze(-1)


class MultiPopulationStochasticActor(nn.Module):
    """
    Stochastic actor for multi-population SAC.

    Outputs Gaussian policy: π(a|s,m_1,...,m_N) with tanh squashing.
    """

    def __init__(
        self,
        pop_id: str,
        state_dim: int,
        action_dim: int,
        action_bounds: tuple[float, float],
        population_configs: dict[str, PopulationConfig],
        hidden_dims: list[int] | None = None,
    ):
        """
        Initialize stochastic actor.

        Args:
            pop_id: ID of this population
            state_dim: Own state dimension
            action_dim: Own action dimension
            action_bounds: Own action bounds
            population_configs: All population configs
            hidden_dims: Hidden layer dimensions
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.pop_id = pop_id
        self.action_bounds = action_bounds

        # Shared encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
        )

        self.pop_encoder = JointPopulationEncoder(
            population_configs=population_configs,
            hidden_dim=hidden_dims[0],
            use_attention=False,
        )

        combined_dim = hidden_dims[0] + self.pop_encoder.output_dim
        self.encoder = nn.Sequential(
            nn.Linear(combined_dim, hidden_dims[1]),
            nn.ReLU(),
        )

        # Mean and log_std heads
        self.mean_head = nn.Linear(hidden_dims[1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[1], action_dim)

        # Action scaling
        self.action_scale = (action_bounds[1] - action_bounds[0]) / 2.0
        self.action_bias = (action_bounds[1] + action_bounds[0]) / 2.0

    def forward(
        self, state: torch.Tensor, population_states: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and log_std.

        Args:
            state: Own state [batch, state_dim]
            population_states: {pop_id: distribution [batch, dist_dim]}

        Returns:
            (mean, log_std) tuple
        """
        state_feat = self.state_encoder(state)
        pop_feat = self.pop_encoder(population_states)

        combined = torch.cat([state_feat, pop_feat], dim=1)
        features = self.encoder(combined)

        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def sample(
        self, state: torch.Tensor, population_states: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action with reparameterization trick.

        Args:
            state: Own state [batch, state_dim]
            population_states: {pop_id: distribution [batch, dist_dim]}

        Returns:
            (action, log_prob, mean_action) tuple
        """
        mean, log_std = self.forward(state, population_states)
        std = log_std.exp()

        # Reparameterization trick
        normal = Normal(mean, std)
        x_t = normal.rsample()

        # Squash with tanh
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        # Log probability with change of variables
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        # Mean action for evaluation
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean_action
