# MADDPG Architecture Design for MFG_PDE

**Date**: October 2, 2025
**Status**: Architecture Design (Future Extension)
**Context**: Phase 2.3 - Multi-Agent Extensions for MFG-RL

---

## Executive Summary

This document presents the architecture design for implementing **Mean Field MADDPG (MF-MADDPG)** in MFG_PDE. MADDPG extends the current discrete-action RL paradigm to **continuous action spaces**, which is essential for many real-world MFG applications.

**Status**: This is a design document for future implementation. Current MFG_PDE focuses on discrete actions. Continuous action support requires:
1. New network architectures (action as input to critic)
2. Deterministic policy gradient updates
3. Exploration strategies for continuous spaces

See `continuous_action_mfg_theory.md` for the full 6-12 month roadmap.

---

## Design Philosophy

### Relationship to Existing Implementation

**Current**: Mean Field Q-Learning (discrete actions)
```python
Q_network(s, m) → [Q(s, a₁, m), Q(s, a₂, m), ..., Q(s, aₙ, m)]
```

**Proposed**: Mean Field MADDPG (continuous actions)
```python
actor(s, m) → a ∈ ℝᵈ
critic(s, a, m) → Q(s, a, m) ∈ ℝ
```

**Key Difference**: Action is an **input** to critic, not an index into output vector.

---

## Architecture Design

### File Structure

```
mfg_pde/alg/reinforcement/
├── algorithms/
│   ├── mean_field_q_learning.py       # Existing (discrete)
│   ├── mean_field_actor_critic.py      # Existing (discrete)
│   ├── mean_field_maddpg.py           # NEW (continuous) ⭐
│   └── __init__.py
├── networks/
│   ├── discrete_networks.py           # Q-networks for discrete actions
│   ├── continuous_networks.py         # NEW: Actor-critic for continuous ⭐
│   └── __init__.py
├── exploration/
│   ├── epsilon_greedy.py              # Discrete action exploration
│   ├── ou_noise.py                    # NEW: Ornstein-Uhlenbeck noise ⭐
│   └── __init__.py
└── core/
    └── base_rl.py                     # Base classes for all RL algorithms
```

---

## Network Architecture

### 1. Mean Field Actor (Deterministic Policy)

```python
class MeanFieldDeterministicActor(nn.Module):
    """
    Deterministic policy network for continuous actions in MFG.

    Maps (state, population_state) → continuous_action
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        population_dim: int,
        hidden_dims: list[int] | None = None,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.action_low = action_low
        self.action_high = action_high
        self.action_range = (action_high - action_low) / 2.0
        self.action_center = (action_high + action_low) / 2.0

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0] // 2),
            nn.ReLU(),
        )

        # Population encoder
        self.population_encoder = nn.Sequential(
            nn.Linear(population_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0] // 2),
            nn.ReLU(),
        )

        # Policy head (deterministic, continuous output)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim),
            nn.Tanh(),  # Output in [-1, 1]
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor, population_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: (s, m) → a

        Args:
            state: [batch, state_dim]
            population_state: [batch, population_dim]

        Returns:
            action: [batch, action_dim] scaled to [action_low, action_high]
        """
        state_feat = self.state_encoder(state)
        pop_feat = self.population_encoder(population_state)

        combined = torch.cat([state_feat, pop_feat], dim=1)
        action_tanh = self.policy_head(combined)  # [-1, 1]

        # Scale to [action_low, action_high]
        action = self.action_center + self.action_range * action_tanh

        return action
```

### 2. Mean Field Critic (Q-function for Continuous Actions)

```python
class MeanFieldContinuousCritic(nn.Module):
    """
    Critic network for continuous actions in MFG.

    Maps (state, action, population_state) → Q-value
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        population_dim: int,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
        )

        # Action encoder (NEW - action is input!)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dims[0]),
            nn.ReLU(),
        )

        # Population encoder
        self.population_encoder = nn.Sequential(
            nn.Linear(population_dim, hidden_dims[0]),
            nn.ReLU(),
        )

        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(3 * hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),  # Scalar Q-value
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        population_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: (s, a, m) → Q(s, a, m)

        Args:
            state: [batch, state_dim]
            action: [batch, action_dim] (CONTINUOUS!)
            population_state: [batch, population_dim]

        Returns:
            q_value: [batch, 1]
        """
        state_feat = self.state_encoder(state)
        action_feat = self.action_encoder(action)
        pop_feat = self.population_encoder(population_state)

        combined = torch.cat([state_feat, action_feat, pop_feat], dim=1)
        q_value = self.q_head(combined)

        return q_value
```

---

## Algorithm Implementation

### MeanFieldMADDPG Class

```python
class MeanFieldMADDPG:
    """
    Mean Field Multi-Agent DDPG for continuous action MFG.

    Combines:
    - MADDPG: Centralized critic, decentralized actor
    - Mean Field: Population state instead of all agent states
    - DDPG: Deterministic policy gradient for continuous actions
    """

    def __init__(
        self,
        env,
        state_dim: int,
        action_dim: int,
        population_dim: int,
        config: dict[str, Any] | None = None,
    ):
        # Configuration
        default_config = {
            "actor_lr": 1e-4,
            "critic_lr": 1e-3,
            "discount_factor": 0.99,
            "tau": 0.001,  # Soft update rate
            "batch_size": 128,
            "replay_buffer_size": 1000000,
            "hidden_dims": [256, 256],
            "action_low": -1.0,
            "action_high": 1.0,
            # Exploration noise
            "exploration_noise": 0.1,
            "noise_decay": 0.9999,
        }
        self.config = {**default_config, **(config or {})}

        # Networks
        self.actor = MeanFieldDeterministicActor(
            state_dim, action_dim, population_dim,
            hidden_dims=self.config["hidden_dims"],
            action_low=self.config["action_low"],
            action_high=self.config["action_high"],
        )

        self.critic = MeanFieldContinuousCritic(
            state_dim, action_dim, population_dim,
            hidden_dims=self.config["hidden_dims"],
        )

        # Target networks
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config["actor_lr"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config["critic_lr"])

        # Replay buffer
        self.replay_buffer = ContinuousActionReplayBuffer(self.config["replay_buffer_size"])

        # Exploration
        self.exploration_noise = self.config["exploration_noise"]

    def select_action(
        self,
        state: np.ndarray,
        population_state: np.ndarray,
        training: bool = True,
    ) -> np.ndarray:
        """Select action with exploration noise during training."""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            pop_t = torch.FloatTensor(population_state).unsqueeze(0)

            action = self.actor(state_t, pop_t).cpu().numpy()[0]

        if training:
            # Add Gaussian noise for exploration
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = np.clip(
                action + noise,
                self.config["action_low"],
                self.config["action_high"],
            )

        return action

    def update(self, batch_size: int) -> tuple[float, float]:
        """
        Update actor and critic networks.

        Returns:
            (critic_loss, actor_loss)
        """
        # Sample batch
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, pop_states, next_pop_states, dones = batch

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        pop_states = torch.FloatTensor(pop_states)
        next_pop_states = torch.FloatTensor(next_pop_states)
        dones = torch.BoolTensor(dones)

        # ===== Update Critic =====
        with torch.no_grad():
            # Target actions from target actor
            next_actions = self.target_actor(next_states, next_pop_states)

            # Target Q-values
            target_q = self.target_critic(next_states, next_actions, next_pop_states)
            target_q = rewards.unsqueeze(1) + self.config["discount_factor"] * target_q * (~dones).unsqueeze(1)

        # Current Q-values
        current_q = self.critic(states, actions, pop_states)

        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ===== Update Actor =====
        # Predicted actions
        predicted_actions = self.actor(states, pop_states)

        # Actor loss = -Q(s, π(s, m), m)
        actor_loss = -self.critic(states, predicted_actions, pop_states).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ===== Soft Update Target Networks =====
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)

        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, target_net, source_net):
        """Soft update: θ' ← τθ + (1-τ)θ'"""
        tau = self.config["tau"]
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
```

---

## Integration with MFG_PDE

### Factory Function

```python
def create_mf_maddpg(env, config: dict[str, Any] | None = None) -> MeanFieldMADDPG:
    """
    Factory function to create MF-MADDPG algorithm.

    Args:
        env: MFG environment with continuous action space
        config: Algorithm configuration

    Returns:
        Configured MF-MADDPG instance
    """
    # Extract dimensions from environment
    obs, _ = env.reset()
    obs_batch = np.atleast_2d(obs).astype(np.float32)

    state_dim = obs_batch.shape[1]

    # Check action space is continuous
    if not hasattr(env.action_space, 'shape'):
        raise ValueError("MF-MADDPG requires continuous action space (Box)")

    action_dim = env.action_space.shape[0]
    action_low = float(env.action_space.low[0])
    action_high = float(env.action_space.high[0])

    # Population dimension
    population_dim = state_dim * 2  # mean + std

    # Create config with action bounds
    maddpg_config = {
        "action_low": action_low,
        "action_high": action_high,
        **(config or {}),
    }

    return MeanFieldMADDPG(
        env=env,
        state_dim=state_dim,
        action_dim=action_dim,
        population_dim=population_dim,
        config=maddpg_config,
    )
```

---

## Testing Strategy

### Unit Tests

```python
def test_actor_output_shape():
    """Test actor produces correct action shape."""
    actor = MeanFieldDeterministicActor(
        state_dim=4, action_dim=2, population_dim=8
    )

    state = torch.randn(10, 4)
    pop_state = torch.randn(10, 8)

    action = actor(state, pop_state)

    assert action.shape == (10, 2)
    assert torch.all(action >= -1.0)
    assert torch.all(action <= 1.0)


def test_critic_continuous_action():
    """Test critic handles continuous actions."""
    critic = MeanFieldContinuousCritic(
        state_dim=4, action_dim=2, population_dim=8
    )

    state = torch.randn(10, 4)
    action = torch.randn(10, 2)  # Continuous actions
    pop_state = torch.randn(10, 8)

    q_value = critic(state, action, pop_state)

    assert q_value.shape == (10, 1)


def test_deterministic_policy_gradient():
    """Test that policy gradient flows through actor."""
    actor = MeanFieldDeterministicActor(
        state_dim=4, action_dim=2, population_dim=8
    )
    critic = MeanFieldContinuousCritic(
        state_dim=4, action_dim=2, population_dim=8
    )

    state = torch.randn(10, 4, requires_grad=True)
    pop_state = torch.randn(10, 8)

    action = actor(state, pop_state)
    q_value = critic(state, action, pop_state)

    loss = -q_value.mean()
    loss.backward()

    # Verify gradients exist
    assert actor.policy_head[-2].weight.grad is not None
```

---

## Comparison with Existing Algorithms

| Feature | MF Q-Learning | MF Actor-Critic | **MF-MADDPG** |
|---------|---------------|-----------------|---------------|
| **Action Space** | Discrete | Discrete (stochastic) | **Continuous (deterministic)** |
| **Policy** | Implicit (ε-greedy) | Stochastic (π(a\|s,m)) | **Deterministic (π(s,m))** |
| **Critic** | Q(s,m)→[Q(s,a,m)] | V(s,m) or Q(s,a,m) | **Q(s,a,m)** |
| **Update** | TD learning | Policy gradient + TD | **DPG + TD** |
| **Exploration** | ε-greedy | Entropy | **Action noise** |
| **Best For** | Discrete control | General | **Continuous control** |

---

## Implementation Roadmap

### Phase 1: Core MADDPG (4-6 weeks)
- [ ] Implement continuous action networks
- [ ] Implement MF-MADDPG algorithm
- [ ] Add Ornstein-Uhlenbeck noise
- [ ] Create unit tests

### Phase 2: Environment Integration (2-3 weeks)
- [ ] Create continuous action MFG environments
- [ ] Adapt maze environment for continuous actions
- [ ] Factory functions and configuration

### Phase 3: Advanced Features (3-4 weeks)
- [ ] Parameter space noise exploration
- [ ] Twin critics (TD3-style)
- [ ] Learned population embeddings
- [ ] Performance benchmarking

### Phase 4: Applications (4-6 weeks)
- [ ] Traffic flow control
- [ ] Financial market making
- [ ] Continuous crowd navigation
- [ ] Documentation and examples

**Total Estimated Time**: 3-4 months

---

## Current Status

**Implementation Status**: Design phase (not yet implemented)

**Blockers**:
1. Continuous action space environments needed
2. New network architectures required
3. Additional testing infrastructure

**Dependencies**:
- Current RL infrastructure (✅ Complete)
- PyTorch (✅ Available)
- Gymnasium with continuous spaces (✅ Available)

**Next Steps**:
1. Create simple continuous action MFG environment
2. Implement basic actor-critic networks
3. Test on simple continuous control task
4. Incrementally add MFG-specific features

---

## References

See `maddpg_for_mfg_formulation.md` for complete references.

---

**Status**: ✅ Architecture design complete
**Implementation**: Pending (future work)
**Estimated Effort**: 3-4 months full implementation
**Priority**: Medium (after discrete action algorithms mature)
