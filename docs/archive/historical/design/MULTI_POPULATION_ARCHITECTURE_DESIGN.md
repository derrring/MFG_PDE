# Multi-Population Continuous Control Architecture Design

**Status**: Phase 3.4 - Multi-Population MFG Framework
**Date**: 2025-10-05
**Issue**: #69

## 1. Executive Summary

This document specifies the architecture for extending MFG_PDE's continuous control algorithms (DDPG, TD3, SAC) to support **heterogeneous multi-population Mean Field Games** with 2-5 interacting populations.

**Key Innovation**: Enable populations with different state spaces, action spaces, and learning algorithms to interact through coupled mean field distributions, solving Nash equilibrium in heterogeneous multi-agent systems.

## 2. Mathematical Framework

### 2.1 Multi-Population MFG System

Consider $N$ populations indexed by $i \in \{1, \ldots, N\}$:

**State Evolution** (per population):
$$
ds_i^t = b_i(s_i^t, a_i^t, m_1^t, \ldots, m_N^t) dt + \sigma_i dW_i^t
$$

**Value Functions** (coupled through all populations):
$$
V_i(s_i, m_1, \ldots, m_N) = \sup_{a_i \in \mathcal{A}_i} Q_i(s_i, a_i, m_1, \ldots, m_N)
$$

**Nash Equilibrium Condition**:
$$
\forall i: \quad \pi_i^* \in \arg\max_{\pi_i} \mathbb{E}\left[\sum_{t=0}^T r_i(s_i^t, a_i^t, m_1^t, \ldots, m_N^t)\right]
$$

where $m_i^t = \text{Law}(s_i^t)$ is the population $i$ distribution at time $t$.

### 2.2 Heterogeneous Specifications

Each population $i$ has:
- **State space**: $\mathcal{S}_i \subseteq \mathbb{R}^{d_i^s}$
- **Action space**: $\mathcal{A}_i \subseteq \mathbb{R}^{d_i^a}$ with bounds $[a_i^{\min}, a_i^{\max}]$
- **Dynamics**: $f_i: \mathcal{S}_i \times \mathcal{A}_i \times \mathcal{M}_1 \times \cdots \times \mathcal{M}_N \to \mathcal{S}_i$
- **Reward**: $r_i: \mathcal{S}_i \times \mathcal{A}_i \times \mathcal{M}_1 \times \cdots \times \mathcal{M}_N \to \mathbb{R}$

**Coupling Mechanism**: Population $i$'s dynamics and rewards depend on **all** population distributions $(m_1, \ldots, m_N)$.

### 2.3 Algorithm Extensions

Each population can use a different continuous control algorithm:

| Algorithm | Policy Type | Q-Function | Key Feature |
|:----------|:------------|:-----------|:------------|
| **DDPG** | $\mu_i(s_i, m_{-i})$ deterministic | $Q_i(s_i, a_i, m_{-i})$ | OU noise exploration |
| **TD3** | $\mu_i(s_i, m_{-i})$ deterministic | $\min(Q_{i,1}, Q_{i,2})$ | Twin critics, delayed updates |
| **SAC** | $\pi_i(a_i \mid s_i, m_{-i})$ stochastic | Soft $Q_i + \alpha \mathcal{H}(\pi_i)$ | Entropy regularization |

where $m_{-i} = (m_1, \ldots, m_{i-1}, m_{i+1}, \ldots, m_N)$ denotes all other populations.

## 3. Core Architecture Components

### 3.1 Population Configuration

**File**: `mfg_pde/alg/reinforcement/multi_population/population_config.py`

```python
@dataclass
class PopulationConfig:
    """Configuration for a single population in multi-population MFG."""

    population_id: str
    state_dim: int
    action_dim: int
    action_bounds: tuple[float, float]

    # Algorithm specification
    algorithm: Literal["ddpg", "td3", "sac"]
    algorithm_config: dict[str, Any]

    # Population-specific parameters
    initial_distribution: Callable | None = None
    coupling_weights: dict[str, float] | None = None  # Weights for other populations

    def validate(self) -> None:
        """Validate configuration."""
        assert self.state_dim > 0
        assert self.action_dim > 0
        assert self.action_bounds[0] < self.action_bounds[1]
        assert self.algorithm in ["ddpg", "td3", "sac"]
```

### 3.2 Multi-Population Environment Base

**File**: `mfg_pde/alg/reinforcement/multi_population/base_environment.py`

```python
class MultiPopulationMFGEnvironment:
    """
    Base environment for multi-population Mean Field Games.

    Supports 2-5 heterogeneous populations with different state/action spaces.
    """

    def __init__(
        self,
        populations: dict[str, PopulationConfig],
        coupling_dynamics: Callable,
        domain: Domain1D | Domain2D,
    ):
        """
        Initialize multi-population environment.

        Args:
            populations: {pop_id: PopulationConfig}
            coupling_dynamics: Function computing coupled state evolution
            domain: Spatial domain for population distributions
        """
        assert 2 <= len(populations) <= 5, "Support 2-5 populations"

        self.populations = populations
        self.coupling_dynamics = coupling_dynamics
        self.domain = domain

        # Track population states and distributions
        self.population_states: dict[str, NDArray] = {}
        self.population_distributions: dict[str, NDArray] = {}

    def reset(self) -> tuple[dict[str, NDArray], dict[str, Any]]:
        """
        Reset environment.

        Returns:
            states: {pop_id: initial_state}
            info: {pop_id: metadata}
        """
        states = {}
        info = {}

        for pop_id, config in self.populations.items():
            if config.initial_distribution is not None:
                states[pop_id] = config.initial_distribution()
            else:
                states[pop_id] = np.zeros(config.state_dim)

            info[pop_id] = {"population_size": 1000}  # Default

        self.population_states = states
        self._update_distributions()

        return states, info

    def step(
        self,
        actions: dict[str, NDArray]
    ) -> tuple[dict[str, NDArray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, Any]]:
        """
        Execute one step for all populations.

        Args:
            actions: {pop_id: action_array}

        Returns:
            next_states: {pop_id: next_state}
            rewards: {pop_id: reward}
            terminated: {pop_id: done_flag}
            truncated: {pop_id: truncated_flag}
            info: {pop_id: metadata}
        """
        # Compute coupled dynamics
        next_states = self.coupling_dynamics(
            states=self.population_states,
            actions=actions,
            distributions=self.population_distributions,
        )

        # Compute rewards (population-specific)
        rewards = {}
        for pop_id in self.populations:
            rewards[pop_id] = self._compute_reward(
                pop_id=pop_id,
                state=self.population_states[pop_id],
                action=actions[pop_id],
                next_state=next_states[pop_id],
            )

        # Update state
        self.population_states = next_states
        self._update_distributions()

        # Termination (can be population-specific)
        terminated = {pop_id: self._is_terminated(pop_id) for pop_id in self.populations}
        truncated = {pop_id: False for pop_id in self.populations}
        info = {pop_id: {} for pop_id in self.populations}

        return next_states, rewards, terminated, truncated, info

    def get_population_state(self, pop_id: str) -> NDArray:
        """Get distribution for specific population."""
        return self.population_distributions[pop_id]

    def get_all_population_states(self) -> dict[str, NDArray]:
        """Get all population distributions."""
        return self.population_distributions

    def _update_distributions(self) -> None:
        """Update population distributions from states."""
        for pop_id, state in self.population_states.items():
            # Convert state to distribution (histogram on domain)
            self.population_distributions[pop_id] = self._state_to_distribution(state)

    def _state_to_distribution(self, state: NDArray) -> NDArray:
        """Convert state to spatial distribution."""
        # Implementation: binning on domain grid
        raise NotImplementedError("Implement in concrete environment")

    def _compute_reward(
        self, pop_id: str, state: NDArray, action: NDArray, next_state: NDArray
    ) -> float:
        """Compute population-specific reward."""
        raise NotImplementedError("Implement in concrete environment")

    def _is_terminated(self, pop_id: str) -> bool:
        """Check termination for specific population."""
        raise NotImplementedError("Implement in concrete environment")
```

### 3.3 Joint Population Encoder

**File**: `mfg_pde/alg/reinforcement/multi_population/networks.py`

```python
class JointPopulationEncoder(nn.Module):
    """
    Encodes multiple population distributions into joint representation.

    Architecture:
    1. Per-population encoders: m_i → h_i
    2. Attention mechanism: Cross-population interactions
    3. Joint encoding: [h_1, ..., h_N] → z
    """

    def __init__(
        self,
        population_configs: dict[str, PopulationConfig],
        hidden_dim: int = 128,
        use_attention: bool = True,
    ):
        """
        Initialize joint encoder.

        Args:
            population_configs: {pop_id: config}
            hidden_dim: Hidden dimension for encodings
            use_attention: Enable cross-population attention
        """
        super().__init__()

        self.population_ids = list(population_configs.keys())
        self.use_attention = use_attention

        # Per-population encoders
        self.pop_encoders = nn.ModuleDict({
            pop_id: nn.Sequential(
                nn.Linear(config.state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            for pop_id, config in population_configs.items()
        })

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
            population_states: {pop_id: distribution [batch, dim]}

        Returns:
            Joint encoding [batch, output_dim]
        """
        # Encode each population
        pop_features = []
        for pop_id in self.population_ids:
            h_i = self.pop_encoders[pop_id](population_states[pop_id])
            pop_features.append(h_i)

        # Stack for attention: [batch, num_pops, hidden_dim]
        stacked_features = torch.stack(pop_features, dim=1)

        # Cross-population attention
        if self.use_attention:
            attended_features, _ = self.attention(
                stacked_features, stacked_features, stacked_features
            )
            # Flatten: [batch, num_pops * hidden_dim]
            joint_features = attended_features.flatten(start_dim=1)
        else:
            joint_features = stacked_features.flatten(start_dim=1)

        # Joint encoding
        return self.joint_encoder(joint_features)
```

### 3.4 Multi-Population Actor

**File**: `mfg_pde/alg/reinforcement/multi_population/networks.py`

```python
class MultiPopulationActor(nn.Module):
    """
    Actor network for multi-population MFG.

    Architecture:
    1. State encoder: Process own state
    2. Joint population encoder: Process all distributions
    3. Action head: Generate population-specific action
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
            action_bounds: Own action bounds
            population_configs: All population configs
            hidden_dims: Hidden layer sizes
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.pop_id = pop_id
        self.action_bounds = action_bounds

        # State encoder (own state)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )

        # Joint population encoder (all distributions)
        self.pop_encoder = JointPopulationEncoder(
            population_configs=population_configs,
            hidden_dim=hidden_dims[1],
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

    def forward(
        self,
        state: torch.Tensor,
        population_states: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Generate action for this population.

        Args:
            state: Own state [batch, state_dim]
            population_states: {pop_id: distribution [batch, dim]}

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
```

### 3.5 Multi-Population Critic

**File**: `mfg_pde/alg/reinforcement/multi_population/networks.py`

```python
class MultiPopulationCritic(nn.Module):
    """
    Q-function network for multi-population MFG.

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
        """Initialize multi-population critic."""
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
            population_states: {pop_id: distribution [batch, dim]}

        Returns:
            Q-value [batch]
        """
        state_feat = self.state_encoder(state)
        action_feat = self.action_encoder(action)
        pop_feat = self.pop_encoder(population_states)

        combined = torch.cat([state_feat, action_feat, pop_feat], dim=1)
        return self.q_head(combined).squeeze(-1)
```

## 4. Multi-Population Algorithm Wrappers

### 4.1 Multi-Population DDPG

**File**: `mfg_pde/alg/reinforcement/multi_population/multi_ddpg.py`

Key modifications from single-population:
- Replace `DDPGActor` with `MultiPopulationActor`
- Replace `DDPGCritic` with `MultiPopulationCritic`
- Update `select_action()` to accept `dict[str, NDArray]` for all population states
- Update replay buffer to store joint population states

### 4.2 Multi-Population TD3

**File**: `mfg_pde/alg/reinforcement/multi_population/multi_td3.py`

Inherit twin critic architecture:
- Two `MultiPopulationCritic` instances
- Delayed policy updates (every `policy_delay` steps)
- Target policy smoothing with joint population states

### 4.3 Multi-Population SAC

**File**: `mfg_pde/alg/reinforcement/multi_population/multi_sac.py`

Stochastic policy with entropy:
- `MultiPopulationStochasticActor` with Gaussian outputs
- Twin soft critics with joint population encoding
- Automatic temperature tuning (per-population α_i)

## 5. Training Orchestrator

**File**: `mfg_pde/alg/reinforcement/multi_population/trainer.py`

```python
class MultiPopulationTrainer:
    """
    Orchestrates training for multi-population MFG.

    Manages:
    - Heterogeneous agents (DDPG, TD3, SAC)
    - Joint replay buffers
    - Nash equilibrium convergence monitoring
    """

    def __init__(
        self,
        env: MultiPopulationMFGEnvironment,
        agents: dict[str, MeanFieldDDPG | MeanFieldTD3 | MeanFieldSAC],
    ):
        """
        Initialize trainer.

        Args:
            env: Multi-population environment
            agents: {pop_id: algorithm_instance}
        """
        self.env = env
        self.agents = agents

        # Verify consistency
        assert set(env.populations.keys()) == set(agents.keys())

    def train(self, num_episodes: int = 1000) -> dict[str, Any]:
        """
        Train all populations simultaneously.

        Returns:
            Training statistics per population
        """
        stats = {pop_id: [] for pop_id in self.agents}

        for episode in range(num_episodes):
            states, _ = self.env.reset()
            population_states = self.env.get_all_population_states()

            done = {pop_id: False for pop_id in self.agents}

            while not all(done.values()):
                # Select actions for all populations
                actions = {}
                for pop_id, agent in self.agents.items():
                    if not done[pop_id]:
                        actions[pop_id] = agent.select_action(
                            state=states[pop_id],
                            population_state=self._flatten_population_states(
                                population_states, agent
                            ),
                            training=True,
                        )

                # Execute joint step
                next_states, rewards, terminated, truncated, _ = self.env.step(actions)
                next_population_states = self.env.get_all_population_states()

                # Store transitions and update each agent
                for pop_id, agent in self.agents.items():
                    if not done[pop_id]:
                        # Store transition
                        agent.replay_buffer.push(
                            state=states[pop_id],
                            action=actions[pop_id],
                            reward=rewards[pop_id],
                            next_state=next_states[pop_id],
                            population_state=self._flatten_population_states(
                                population_states, agent
                            ),
                            next_population_state=self._flatten_population_states(
                                next_population_states, agent
                            ),
                            done=terminated[pop_id] or truncated[pop_id],
                        )

                        # Update agent
                        loss = agent.update()
                        if loss is not None:
                            stats[pop_id].append(loss)

                states = next_states
                population_states = next_population_states
                done = {
                    pop_id: terminated[pop_id] or truncated[pop_id]
                    for pop_id in self.agents
                }

        return stats

    def _flatten_population_states(
        self,
        population_states: dict[str, NDArray],
        agent: Any,
    ) -> NDArray:
        """Flatten all population states for agent's encoder."""
        # Concatenate all distributions in consistent order
        return np.concatenate([
            population_states[pop_id]
            for pop_id in sorted(population_states.keys())
        ])
```

## 6. Example Application: Heterogeneous Traffic Flow

**File**: `examples/advanced/heterogeneous_traffic_multi_pop.py`

### 6.1 Problem Setup

Three vehicle populations:
1. **Cars**: Fast, small action bounds, DDPG
2. **Trucks**: Slow, large, constrained, TD3
3. **Motorcycles**: Very fast, agile, SAC

**Coupling**: Speed depends on density of all vehicle types:
$$
v_{\text{car}}(m_{\text{car}}, m_{\text{truck}}, m_{\text{moto}}) = v_{\max} \cdot (1 - 0.5 m_{\text{car}} - 0.8 m_{\text{truck}} - 0.3 m_{\text{moto}})
$$

### 6.2 Configuration

```python
populations = {
    "cars": PopulationConfig(
        population_id="cars",
        state_dim=2,  # (position, velocity)
        action_dim=1,  # acceleration
        action_bounds=(-3.0, 3.0),
        algorithm="ddpg",
        algorithm_config={"actor_lr": 1e-4, "critic_lr": 1e-3},
        coupling_weights={"trucks": 0.8, "motorcycles": 0.3},
    ),
    "trucks": PopulationConfig(
        population_id="trucks",
        state_dim=2,
        action_dim=1,
        action_bounds=(-2.0, 2.0),  # Lower acceleration
        algorithm="td3",
        algorithm_config={"policy_delay": 2},
        coupling_weights={"cars": 0.5, "motorcycles": 0.2},
    ),
    "motorcycles": PopulationConfig(
        population_id="motorcycles",
        state_dim=2,
        action_dim=1,
        action_bounds=(-4.0, 4.0),  # Higher acceleration
        algorithm="sac",
        algorithm_config={"target_entropy": -1.0},
        coupling_weights={"cars": 0.4, "trucks": 0.6},
    ),
}
```

## 7. Testing Strategy

### 7.1 Unit Tests (20 tests)

**File**: `tests/unit/test_multi_population.py`

- PopulationConfig validation (5 tests)
- JointPopulationEncoder forward/backward (5 tests)
- MultiPopulationActor/Critic architectures (5 tests)
- Environment reset/step logic (5 tests)

### 7.2 Integration Tests (15 tests)

**File**: `tests/integration/test_multi_population_training.py`

- Two-population training convergence (5 tests)
- Three-population Nash equilibrium (3 tests)
- Heterogeneous algorithms (DDPG+TD3+SAC) (4 tests)
- Replay buffer consistency (3 tests)

### 7.3 Application Tests (15 tests)

**File**: `tests/integration/test_heterogeneous_traffic.py`

- Traffic flow dynamics (5 tests)
- Coupled speed equations (3 tests)
- Multi-vehicle Nash equilibrium (4 tests)
- Visualization and metrics (3 tests)

**Total**: 50+ comprehensive tests

## 8. Implementation Roadmap

### Phase 1: Core Infrastructure (Days 1-2)
- [ ] PopulationConfig dataclass
- [ ] MultiPopulationMFGEnvironment base
- [ ] JointPopulationEncoder network
- [ ] Unit tests for components

### Phase 2: Network Architectures (Days 3-4)
- [ ] MultiPopulationActor
- [ ] MultiPopulationCritic
- [ ] MultiPopulationStochasticActor (SAC)
- [ ] Integration tests

### Phase 3: Algorithm Extensions (Days 5-6)
- [ ] Multi-Population DDPG
- [ ] Multi-Population TD3
- [ ] Multi-Population SAC
- [ ] MultiPopulationTrainer

### Phase 4: Application (Days 7-8)
- [ ] Heterogeneous traffic environment
- [ ] Training and visualization
- [ ] Performance benchmarks
- [ ] Application tests

### Phase 5: Documentation (Day 9)
- [ ] API documentation
- [ ] Theory documentation
- [ ] Tutorial notebook

## 9. Success Criteria

- ✅ Support 2-5 heterogeneous populations
- ✅ Different state/action dimensions per population
- ✅ Mix DDPG, TD3, SAC in single system
- ✅ Convergence to Nash equilibrium in traffic example
- ✅ 50+ passing tests
- ✅ Complete documentation with mathematical derivations

## 10. References

1. **Multi-Population MFG Theory**: Carmona & Delarue (2018) - Probabilistic Theory of Mean Field Games
2. **Heterogeneous Agents**: Achdou et al. (2022) - Income and wealth distribution in macroeconomics
3. **Continuous Control**: Haarnoja et al. (2018) - Soft Actor-Critic algorithms
4. **Traffic Applications**: Bayen et al. (2023) - Mixed autonomy traffic flow

---

**Next Steps**: Begin implementation with PopulationConfig and base environment (Phase 1).
