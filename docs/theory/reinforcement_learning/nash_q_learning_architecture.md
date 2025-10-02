# Nash Q-Learning Architecture Design for MFG_PDE

**Date**: October 2, 2025
**Status**: Architecture Design
**Context**: Phase 2.3 - Multi-Agent Extensions for MFG-RL

---

## Executive Summary

This document presents the architectural design for implementing Nash Q-Learning in MFG_PDE. Based on the theoretical formulation, we identify that:

1. **For symmetric MFG**: Nash Q-Learning ≈ Mean Field Q-Learning (already implemented)
2. **Extension needed**: Support for heterogeneous agents and multi-population games
3. **Architecture strategy**: Extend existing `MeanFieldQLearning` class rather than rewrite

---

## Current Architecture Analysis

### Existing Implementation (`mean_field_q_learning.py`)

**Key Components**:
```python
class MeanFieldQNetwork(nn.Module):
    """Q(s, m) -> [Q(s,a₁,m), ..., Q(s,aₙ,m)]"""
    - state_encoder: encodes individual state
    - population_encoder: encodes population state (mean field)
    - fusion_layers: combines features
    - q_head: outputs Q-values for all actions

class MeanFieldQLearning:
    """Main algorithm class"""
    - q_network: main Q-network
    - target_network: target Q-network (for stability)
    - replay_buffer: experience replay
    - select_action(): epsilon-greedy policy
    - train(): main training loop
    - _update_q_network(): Q-learning update
```

**Update Rule** (line 416-448):
```python
# Current Q-values
current_q = Q_network(s, m)[a]

# Target Q-values (Nash = max for symmetric MFG)
next_q = max_a Q_target(s', m')
target_q = r + γ * next_q

# Loss and optimization
loss = MSE(current_q, target_q)
```

**Key Insight**: This is already Nash Q-Learning for symmetric MFG! The `max` operation in line 437 computes the Nash equilibrium value for symmetric games.

---

## Nash Q-Learning Extensions

### Extension 1: Explicit Nash Equilibrium Computation

**When Needed**: Multi-population or competitive games where Nash ≠ max

**Architecture Addition**:
```python
class NashSolver:
    """
    Solve Nash equilibrium from Q-values.

    For symmetric MFG: Nash = argmax_a Q(s, a, m)
    For general games: Solve Nash equilibrium problem
    """

    def solve_nash_equilibrium(
        self,
        q_values: torch.Tensor,  # [batch, num_types, action_dim]
        game_type: str = "symmetric"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Nash equilibrium strategies and values.

        Args:
            q_values: Q-values for each agent type
            game_type: "symmetric", "zero_sum", "general"

        Returns:
            nash_strategies: [batch, num_types, action_dim] (probability distributions)
            nash_values: [batch, num_types] (expected values)
        """
        if game_type == "symmetric":
            # Deterministic Nash = max
            actions = q_values.argmax(dim=-1)  # [batch, num_types]
            nash_strategies = F.one_hot(actions, num_classes=q_values.size(-1)).float()
            nash_values = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        elif game_type == "zero_sum":
            # Zero-sum Nash via linear programming
            nash_strategies, nash_values = self._solve_zero_sum_nash(q_values)

        elif game_type == "general":
            # General Nash via iterative best response
            nash_strategies, nash_values = self._solve_general_nash(q_values)

        return nash_strategies, nash_values

    def _solve_zero_sum_nash(self, q_values: torch.Tensor):
        """Solve zero-sum Nash equilibrium (minimax)."""
        # For 2-player zero-sum: solve via LP
        # Player 1: max_π min_a' Q(π, a')
        # Player 2: min_π' max_a Q(a, π')

        batch_size = q_values.size(0)
        nash_strategies = []
        nash_values = []

        for b in range(batch_size):
            # Solve LP for this batch item
            strategy, value = self._minimax_lp(q_values[b])
            nash_strategies.append(strategy)
            nash_values.append(value)

        return torch.stack(nash_strategies), torch.stack(nash_values)

    def _solve_general_nash(self, q_values: torch.Tensor):
        """Solve general Nash equilibrium via best response dynamics."""
        # Iterative best response
        batch_size, num_types, action_dim = q_values.shape

        # Initialize with uniform strategies
        strategies = torch.ones(batch_size, num_types, action_dim) / action_dim

        # Best response iteration
        for _ in range(self.max_iterations):
            new_strategies = []
            for i in range(num_types):
                # Compute best response for type i given others
                br = self._best_response(q_values[:, i], strategies, i)
                new_strategies.append(br)

            new_strategies = torch.stack(new_strategies, dim=1)

            # Check convergence
            if torch.allclose(strategies, new_strategies, atol=self.tol):
                break

            strategies = new_strategies

        # Compute Nash values
        nash_values = self._compute_expected_values(q_values, strategies)

        return strategies, nash_values
```

### Extension 2: Heterogeneous Agent Types

**When Needed**: Multiple agent populations with different objectives

**Architecture Addition**:
```python
class HeterogeneousNashQNetwork(nn.Module):
    """
    Multi-type Nash Q-network.

    Each agent type has its own Q-function, but all depend on
    the full population distribution.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        population_dim: int,
        num_types: int = 1,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()

        self.num_types = num_types

        # Shared population encoder
        self.population_encoder = nn.Sequential(
            nn.Linear(population_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0] // 2),
            nn.ReLU(),
        )

        # Type-specific Q-networks
        self.type_networks = nn.ModuleList([
            MeanFieldQNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                population_dim=population_dim,
                hidden_dims=hidden_dims,
            )
            for _ in range(num_types)
        ])

    def forward(
        self,
        state: torch.Tensor,          # [batch, state_dim]
        population_state: torch.Tensor,  # [batch, population_dim]
        agent_types: torch.Tensor | None = None,  # [batch] (agent type indices)
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            If agent_types provided: [batch, action_dim] - Q-values for specific types
            If agent_types=None: [batch, num_types, action_dim] - Q-values for all types
        """
        if agent_types is not None:
            # Get Q-values for specific types
            batch_size = state.size(0)
            q_values = []

            for i in range(batch_size):
                type_idx = agent_types[i].item()
                q = self.type_networks[type_idx](
                    state[i:i+1],
                    population_state[i:i+1]
                )
                q_values.append(q)

            return torch.cat(q_values, dim=0)

        else:
            # Get Q-values for all types
            q_all_types = []
            for type_net in self.type_networks:
                q = type_net(state, population_state)
                q_all_types.append(q)

            return torch.stack(q_all_types, dim=1)  # [batch, num_types, action_dim]
```

### Extension 3: Multi-Population State Representation

**When Needed**: Tracking multiple interacting populations

**Architecture Addition**:
```python
class MultiPopulationStateEncoder(nn.Module):
    """
    Encode state of multiple populations.

    For example:
    - Population 1: Buyers (type 0)
    - Population 2: Sellers (type 1)

    Each population's distribution affects all agents.
    """

    def __init__(
        self,
        state_dim: int,
        num_populations: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.num_populations = num_populations

        # Encoder for each population
        self.population_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
            )
            for _ in range(num_populations)
        ])

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(num_populations * (hidden_dim // 2), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, population_states: list[torch.Tensor]) -> torch.Tensor:
        """
        Encode multiple population states.

        Args:
            population_states: List of [batch, state_dim] tensors (one per population)

        Returns:
            Joint population encoding [batch, hidden_dim]
        """
        encodings = []
        for i, pop_state in enumerate(population_states):
            enc = self.population_encoders[i](pop_state)
            encodings.append(enc)

        combined = torch.cat(encodings, dim=1)
        fused = self.fusion(combined)

        return fused

def _compute_multi_population_state(
    self,
    observations: np.ndarray,
    agent_types: np.ndarray
) -> list[np.ndarray]:
    """
    Compute population state for each agent type.

    Args:
        observations: [num_agents, state_dim]
        agent_types: [num_agents] - type index for each agent

    Returns:
        List of population states (one per type)
    """
    population_states = []

    for type_idx in range(self.num_types):
        # Get observations for this type
        type_mask = (agent_types == type_idx)
        type_obs = observations[type_mask]

        if len(type_obs) > 0:
            # Compute mean and std for this population
            mean_obs = np.mean(type_obs, axis=0)
            std_obs = np.std(type_obs, axis=0)
            pop_state = np.concatenate([mean_obs, std_obs])
        else:
            # No agents of this type - use zeros
            pop_state = np.zeros(self.population_dim)

        population_states.append(pop_state)

    return population_states
```

---

## Proposed Implementation Strategy

### Phase 1: Rename and Document Existing Implementation

**Goal**: Make explicit that current implementation is Nash Q-Learning for symmetric MFG

**Changes**:
1. Add docstring note in `MeanFieldQLearning` class:
   ```python
   """
   Mean Field Q-Learning algorithm for MFG problems.

   Note: For symmetric MFG, this is equivalent to Nash Q-Learning,
   since Nash equilibrium reduces to best response to mean field.
   The max operation in the target value computation (line 437)
   implements Nash equilibrium for symmetric games.
   """
   ```

2. Add method `compute_nash_value()`:
   ```python
   def compute_nash_value(
       self,
       state: torch.Tensor,
       population_state: torch.Tensor,
       game_type: str = "symmetric"
   ) -> torch.Tensor:
       """
       Compute Nash equilibrium value.

       For symmetric MFG: Nash value = max_a Q(s, a, m)
       For general games: Solve Nash equilibrium
       """
       with torch.no_grad():
           q_values = self.target_network(state, population_state)

           if game_type == "symmetric":
               nash_values = q_values.max(dim=1)[0]
           else:
               # General case would call NashSolver
               raise NotImplementedError("General Nash equilibrium not yet implemented")

       return nash_values
   ```

### Phase 2: Add Nash Solver Module (Optional)

**Goal**: Support non-symmetric games

**New File**: `mfg_pde/alg/reinforcement/nash_solver.py`

**Contents**:
- `NashSolver` class (from Extension 1 above)
- Support for zero-sum, general-sum games
- Integration with existing Q-learning update

### Phase 3: Add Heterogeneous Agent Support (Optional)

**Goal**: Support multiple agent types

**New File**: `mfg_pde/alg/reinforcement/heterogeneous_nash_q_learning.py`

**Contents**:
- `HeterogeneousNashQNetwork` (from Extension 2 above)
- `HeterogeneousNashQLearning` algorithm class
- Multi-population state computation

---

## Design Decisions

### Decision 1: Extend vs Rewrite

**Choice**: Extend existing `MeanFieldQLearning` class

**Rationale**:
- Current implementation already correct for symmetric MFG
- Extensions are modular (Nash solver, heterogeneous agents)
- Maintains backward compatibility
- Avoids code duplication

### Decision 2: Nash Solver as Separate Module

**Choice**: Create optional `NashSolver` module

**Rationale**:
- Most MFG applications use symmetric equilibrium (max)
- General Nash equilibrium is computationally expensive
- Modular design allows easy experimentation
- Users can choose solver based on problem structure

### Decision 3: Multi-Population as Extension

**Choice**: Create separate `HeterogeneousNashQLearning` class

**Rationale**:
- Heterogeneous agents add significant complexity
- Not all users need this feature
- Cleaner separation of concerns
- Easier to test and maintain

---

## API Design

### Symmetric MFG (Current Implementation)

```python
from mfg_pde.alg.reinforcement import create_mean_field_q_learning

# Create algorithm (already Nash Q-Learning for symmetric MFG)
algo = create_mean_field_q_learning(env, config)

# Train
results = algo.train(num_episodes=1000)

# Evaluate
actions = algo.predict(observations)
```

### Heterogeneous Agents (Proposed)

```python
from mfg_pde.alg.reinforcement import create_heterogeneous_nash_q_learning

# Create algorithm with multiple agent types
config = {
    "num_types": 2,  # Buyers and sellers
    "game_type": "general",  # Use general Nash solver
}
algo = create_heterogeneous_nash_q_learning(env, config)

# Train
results = algo.train(num_episodes=1000)

# Evaluate (returns actions for each agent type)
actions = algo.predict(observations, agent_types)
```

### Custom Nash Solver (Proposed)

```python
from mfg_pde.alg.reinforcement import MeanFieldQLearning
from mfg_pde.alg.reinforcement.nash_solver import NashSolver

# Create algorithm with custom Nash solver
algo = MeanFieldQLearning(env, state_dim, action_dim, population_dim)
algo.nash_solver = NashSolver(game_type="zero_sum")

# Modify update rule to use Nash solver
def custom_update(self):
    # ... standard update code ...

    # Compute Nash equilibrium instead of max
    nash_strategies, nash_values = self.nash_solver.solve_nash_equilibrium(
        next_q_values,
        game_type="zero_sum"
    )
    target_q = rewards + gamma * nash_values

    # ... rest of update ...
```

---

## Implementation Checklist

### Phase 1: Documentation and Clarification ✅
- [ ] Add Nash Q-Learning note to `MeanFieldQLearning` docstring
- [ ] Add `compute_nash_value()` method
- [ ] Update theoretical documentation with architecture details
- [ ] Add references to Nash Q-Learning literature

### Phase 2: Nash Solver Module (Optional)
- [ ] Create `nash_solver.py`
- [ ] Implement `NashSolver` class
- [ ] Add zero-sum Nash solver (minimax via LP)
- [ ] Add general Nash solver (best response dynamics)
- [ ] Add unit tests for Nash solver
- [ ] Add example: zero-sum game

### Phase 3: Heterogeneous Agents (Optional)
- [ ] Create `heterogeneous_nash_q_learning.py`
- [ ] Implement `HeterogeneousNashQNetwork`
- [ ] Implement `HeterogeneousNashQLearning`
- [ ] Add multi-population state computation
- [ ] Add unit tests
- [ ] Add example: buyer-seller market

---

## Testing Strategy

### Unit Tests

**Test 1: Nash Value Computation**
```python
def test_nash_value_computation():
    """Test that Nash value equals max for symmetric MFG."""
    algo = create_mean_field_q_learning(env)

    # Random state and population
    state = torch.randn(10, state_dim)
    pop_state = torch.randn(10, population_dim)

    # Compute Nash value
    nash_value = algo.compute_nash_value(state, pop_state, game_type="symmetric")

    # Should equal max Q-value
    q_values = algo.target_network(state, pop_state)
    max_q = q_values.max(dim=1)[0]

    assert torch.allclose(nash_value, max_q)
```

**Test 2: Zero-Sum Nash Equilibrium**
```python
def test_zero_sum_nash():
    """Test zero-sum Nash equilibrium solver."""
    solver = NashSolver(game_type="zero_sum")

    # Rock-paper-scissors payoff matrix
    q_values = torch.tensor([
        [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]  # batch=1, 2-player, 3-actions
    ])

    strategies, values = solver.solve_nash_equilibrium(q_values)

    # Nash equilibrium is uniform mixed strategy
    expected_strategy = torch.ones(1, 2, 3) / 3
    assert torch.allclose(strategies, expected_strategy, atol=0.1)
```

**Test 3: Heterogeneous Agents**
```python
def test_heterogeneous_agents():
    """Test heterogeneous Nash Q-Learning."""
    config = {"num_types": 2}
    algo = create_heterogeneous_nash_q_learning(env, config)

    # Run episode with mixed agent types
    obs, _ = env.reset()
    agent_types = np.array([0, 1, 0, 1])  # Alternating types

    actions = algo.predict(obs, agent_types)

    # Should have actions for all agents
    assert len(actions) == len(agent_types)
```

### Integration Tests

**Test 1: Convergence to Nash Equilibrium**
```python
def test_nash_convergence():
    """Test that algorithm converges to Nash equilibrium."""
    env = create_simple_congestion_game()
    algo = create_mean_field_q_learning(env)

    # Train
    results = algo.train(num_episodes=1000)

    # Evaluate Nash error
    nash_error = results["nash_errors"][-100:]  # Last 100 episodes
    avg_nash_error = np.mean(nash_error)

    # Should converge to near-zero Nash error
    assert avg_nash_error < 0.1
```

---

## Performance Considerations

### Computational Complexity

**Symmetric MFG** (current implementation):
- Forward pass: O(|S| · |A| · hidden_dim²)
- Nash computation: O(|A|) [just max operation]
- Overall: Same as standard Q-Learning

**Zero-Sum Nash**:
- Nash computation: O(|A|³) [LP solver]
- Becomes bottleneck for |A| > 20

**General Nash**:
- Nash computation: O(num_iterations · |A| · num_types²)
- Can be expensive for many agent types

**Recommendation**: Use symmetric Nash (max) when possible, only use general Nash solver when necessary.

### Memory Requirements

**Single Population**:
- Q-network parameters: O(hidden_dim²)
- Replay buffer: O(buffer_size · (state_dim + population_dim))

**Multiple Populations** (num_types):
- Q-network parameters: O(num_types · hidden_dim²)
- Replay buffer: O(buffer_size · num_types · (state_dim + population_dim))

**Scaling**: Linear in number of agent types

---

## Examples and Use Cases

### Use Case 1: Symmetric Crowd Navigation (Already Supported)

**Problem**: Agents navigate to goals while avoiding collisions

**Nash Equilibrium**: All agents follow same policy (symmetric)

**Implementation**: Use existing `MeanFieldQLearning`

### Use Case 2: Traffic Network with Congestion (Already Supported)

**Problem**: Drivers choose routes to minimize travel time

**Nash Equilibrium**: Wardrop equilibrium (symmetric)

**Implementation**: Use existing `MeanFieldQLearning`

### Use Case 3: Market with Buyers and Sellers (Requires Extension)

**Problem**: Buyers want low prices, sellers want high prices

**Nash Equilibrium**: Market clearing price (heterogeneous)

**Implementation**: Use `HeterogeneousNashQLearning` with 2 types

### Use Case 4: Zero-Sum Security Game (Requires Extension)

**Problem**: Attacker vs Defender on network

**Nash Equilibrium**: Minimax mixed strategy

**Implementation**: Use `NashSolver` with zero-sum mode

---

## References

**Nash Q-Learning**:
- Hu & Wellman (2003): "Nash Q-Learning for General-Sum Stochastic Games"
- Littman (1994): "Markov games as a framework for multi-agent reinforcement learning"

**Mean Field Games**:
- Lasry & Lions (2007): "Mean field games"
- Carmona & Delarue (2018): "Probabilistic Theory of Mean Field Games"

**Nash Equilibrium Computation**:
- Lemke-Howson (1964): "Equilibrium points of bimatrix games"
- Koller et al. (1996): "Efficient computation of equilibria for extensive two-person games"

**MFG-RL**:
- Yang et al. (2018): "Mean Field Multi-Agent Reinforcement Learning"
- Guo et al. (2019): "Learning Mean-Field Games"

---

**Status**: ✅ Architecture Design Complete
**Next**: Implement Phase 1 (documentation and clarification)
**Key Insight**: Our existing `MeanFieldQLearning` is already Nash Q-Learning for symmetric MFG. Extensions are modular and optional.
