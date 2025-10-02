# MADDPG for Mean Field Games: Theoretical Formulation

**Date**: October 2, 2025
**Status**: Theoretical Foundation
**Context**: Phase 2.3 - Multi-Agent Extensions for MFG-RL

---

## Executive Summary

This document presents the theoretical foundation for adapting **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)** to **Mean Field Games**. MADDPG extends DDPG to multi-agent settings using centralized training with decentralized execution, making it particularly suitable for MFG problems with continuous action spaces.

**Key Contribution**: We show how MADDPG's centralized critic can be adapted to use population state (mean field) instead of tracking individual agents, achieving scalability to large populations.

---

## Background: MADDPG

### Original MADDPG (Lowe et al. 2017)

**Problem**: Multi-agent environments are non-stationary from each agent's perspective due to other agents' evolving policies.

**Solution**: Use centralized training with decentralized execution:
- **Training**: Centralized critic has access to all agents' observations and actions
- **Execution**: Decentralized actor uses only local observations

### MADDPG Framework

**For N agents**:

**Actors** (decentralized):
```
πᵢ(aᵢ | oᵢ; θᵢ)  for i = 1, ..., N
```
Each agent has its own policy using only local observation `oᵢ`.

**Critics** (centralized during training):
```
Qᵢ(o₁, ..., oₙ, a₁, ..., aₙ; φᵢ)  for i = 1, ..., N
```
Each critic uses all agents' observations and actions.

**Update Rules**:
```
Actor gradient:  ∇θᵢ J(θᵢ) = 𝔼[∇θᵢ πᵢ(aᵢ|oᵢ) · ∇ₐᵢ Qᵢ(o, a)|ₐᵢ=πᵢ(oᵢ)]
Critic loss:     L(φᵢ) = 𝔼[(Qᵢ(o, a) - yᵢ)²]
Target:          yᵢ = rᵢ + γ Qᵢ'(o', a'₁, ..., a'ₙ)
```

**Key Properties**:
- Continuous action spaces
- Off-policy learning with experience replay
- Handles non-stationarity via centralized critic
- Decentralized execution for scalability

---

## MADDPG for Mean Field Games

### Challenge: Scalability

**Problem**: Original MADDPG's centralized critic requires O(N) observations and actions, which is infeasible for large populations (N → ∞).

**Solution**: Replace explicit tracking of all agents with **population state (mean field)**.

### Mean Field MADDPG Framework

**Key Idea**: Centralized critic uses population state `m` instead of all individual agents' states.

#### Architecture

**Decentralized Actor**:
```
π(a | s, m; θ)
```
- Input: Individual state `s`, population state `m`
- Output: Continuous action `a ∈ ℝᵈ`
- Used for both training and execution

**Centralized Critic (training only)**:
```
Q(s, a, m; φ)
```
- Input: Individual state `s`, action `a`, population state `m`
- Output: Scalar Q-value
- Uses population state to capture other agents' influence

**Population State**:
```
m = μ(π)
```
- Mean field representation: histogram, moments, or learned embedding
- Captures aggregate behavior of population

#### Update Rules

**Critic Update** (centralized, with population state):
```
Loss: L(φ) = 𝔼[(Q(s, a, m; φ) - y)²]

Target: y = r(s, a, m) + γ Q'(s', a', m'; φ')

where:
  a' = π'(s', m'; θ')  (target actor)
  m' = next population state
```

**Actor Update** (deterministic policy gradient):
```
∇θ J(θ) = 𝔼ₛ,ₘ [∇θ π(s, m; θ) · ∇ₐ Q(s, a, m; φ)|ₐ=π(s,m;θ)]
```

**Population Update**:
```
m ← aggregate({sᵢ : i = 1, ..., N})
```

### Comparison: Original MADDPG vs MF-MADDPG

| Aspect | Original MADDPG | MF-MADDPG |
|--------|-----------------|-----------|
| **Critic Input** | (o₁, ..., oₙ, a₁, ..., aₙ) | (s, a, m) |
| **Complexity** | O(N) | O(1) per agent |
| **Scalability** | Limited to N < 100 | Scales to N → ∞ |
| **Population** | Fixed N | Variable N |
| **Execution** | Decentralized | Decentralized |
| **Training** | Centralized | Centralized (via m) |

---

## Mathematical Formulation

### MFG-MADDPG Objective

**Individual Agent Objective**:
```
max J(θ) = 𝔼[∑ₜ γᵗ r(sₜ, aₜ, mₜ) | π(·, ·; θ), m]
```

**Nash Equilibrium Condition**:
```
π* = argmax J(π | m*)
m* = μ(π*)
```

### Bellman Equation

**Q-Function**:
```
Q(s, a, m) = 𝔼[r(s, a, m) + γ Q(s', a', m') | a' = π(s', m')]
```

**Deterministic Policy Gradient**:
```
∇θ J(θ) = 𝔼[∇θ π(s, m; θ) · ∇ₐ Q(s, a, m)|ₐ=π(s,m)]
```

**Key Property**: Because actor is deterministic and differentiable, we can backpropagate through the policy to optimize Q-function.

---

## Network Architectures

### Actor Network: π(s, m; θ)

```python
class MeanFieldActor(nn.Module):
    """
    Deterministic policy for continuous actions.

    Architecture:
        Input: [state, population_state]
        Output: continuous action ∈ [action_low, action_high]
    """

    def __init__(self, state_dim, action_dim, population_dim, hidden_dim=256):
        super().__init__()

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Population encoder
        self.population_encoder = nn.Sequential(
            nn.Linear(population_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Policy head (deterministic)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Actions in [-1, 1]
        )

    def forward(self, state, population_state):
        """
        Args:
            state: [batch, state_dim]
            population_state: [batch, population_dim]

        Returns:
            action: [batch, action_dim] in [-1, 1]
        """
        state_feat = self.state_encoder(state)
        pop_feat = self.population_encoder(population_state)
        combined = torch.cat([state_feat, pop_feat], dim=1)
        action = self.policy(combined)
        return action
```

### Critic Network: Q(s, a, m; φ)

```python
class MeanFieldCritic(nn.Module):
    """
    Centralized Q-function with population state.

    Architecture:
        Input: [state, action, population_state]
        Output: scalar Q-value
    """

    def __init__(self, state_dim, action_dim, population_dim, hidden_dim=256):
        super().__init__()

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
        )

        # Population encoder
        self.population_encoder = nn.Sequential(
            nn.Linear(population_dim, hidden_dim),
            nn.ReLU(),
        )

        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Scalar Q-value
        )

    def forward(self, state, action, population_state):
        """
        Args:
            state: [batch, state_dim]
            action: [batch, action_dim]
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

## Algorithm: MF-MADDPG

### Pseudocode

```python
# Initialize networks
actor = MeanFieldActor(state_dim, action_dim, population_dim)
critic = MeanFieldCritic(state_dim, action_dim, population_dim)
target_actor = copy(actor)
target_critic = copy(critic)

# Initialize replay buffer
replay_buffer = ReplayBuffer(capacity)

for episode in episodes:
    # Reset environment
    states = env.reset()  # [num_agents, state_dim]

    for t in timesteps:
        # Compute population state
        m = compute_population_state(states)

        # Select actions (all agents use same policy)
        actions = []
        for i in range(num_agents):
            # Add exploration noise during training
            action = actor(states[i], m) + noise()
            actions.append(action)

        # Execute actions
        next_states, rewards, dones = env.step(actions)
        m_next = compute_population_state(next_states)

        # Store experiences
        for i in range(num_agents):
            replay_buffer.push(states[i], actions[i], rewards[i],
                             next_states[i], m, m_next, dones[i])

        # Update networks (if enough samples)
        if len(replay_buffer) >= batch_size:
            # Sample batch
            batch = replay_buffer.sample(batch_size)
            s, a, r, s_next, m, m_next, done = batch

            # Update critic
            with torch.no_grad():
                a_next = target_actor(s_next, m_next)
                q_target = r + gamma * target_critic(s_next, a_next, m_next) * (1 - done)

            q_pred = critic(s, a, m)
            critic_loss = MSE(q_pred, q_target)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Update actor
            a_pred = actor(s, m)
            actor_loss = -critic(s, a_pred, m).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Soft update target networks
            soft_update(target_actor, actor, tau)
            soft_update(target_critic, critic, tau)

        states = next_states
```

---

## Key Differences from Standard DDPG

### Standard DDPG (Single Agent)
```
Actor:  π(a | s)
Critic: Q(s, a)
Update: Standard DPG with single agent
```

### MF-MADDPG (Multi-Agent MFG)
```
Actor:  π(a | s, m)  - conditioned on population
Critic: Q(s, a, m)   - population-aware value
Update: DPG with population state tracking
```

**Additional Complexity**:
1. **Population State Computation**: Aggregate agent states → mean field
2. **Non-Stationarity**: Population evolves during training
3. **Coordination**: Multiple agents learning simultaneously

---

## Exploration Strategies

### Challenge
Continuous action spaces require exploration, but naive Gaussian noise may be insufficient for coordination.

### Solutions

**1. Ornstein-Uhlenbeck Noise** (Original DDPG):
```
dXₜ = θ(μ - Xₜ)dt + σdWₜ

where:
  θ = mean reversion rate
  μ = long-term mean
  σ = volatility
```

**2. Parameter Space Noise** (Plappert et al. 2017):
```
π̃(s, m) = π(s, m; θ + ε)  where ε ~ N(0, σ²I)
```
Perturb network parameters instead of actions.

**3. Population-Aware Exploration**:
```
a = π(s, m) + β · population_diversity_bonus
```
Encourage exploration in under-explored regions of population state.

---

## Theoretical Properties

### Convergence (Under Assumptions)

**Assumptions**:
1. Deterministic dynamics: s' = f(s, a, m)
2. Bounded rewards: |r(s, a, m)| ≤ R
3. Lipschitz policy and Q-function
4. Slowly varying population: |m_{t+1} - m_t| ≤ δ

**Result**: Under these conditions, MF-MADDPG converges to a Nash equilibrium of the MFG.

**Proof Sketch**:
1. Fixed population m → standard DDPG convergence
2. Population update is contraction under Lipschitz policy
3. Joint convergence via Banach fixed-point theorem

### Sample Complexity

**Per-Agent Complexity**: O(1/ε²) samples to achieve ε-optimal policy

**Total Complexity**: O(N/ε²) for N agents (parallelizable)

**Comparison**:
- Tabular Q-learning: O(|S||A|N/ε²)
- MF-MADDPG: O(poly(d)/ε²) where d = network width

---

## Implementation Challenges

### Challenge 1: Population State Representation

**Problem**: How to encode population distribution efficiently?

**Solutions**:
1. **Histogram**: Discretize state space, count agents per bin
2. **Moments**: Compute mean, variance, higher moments
3. **Learned Embedding**: Neural network encoder for population
4. **Attention Mechanism**: Self-attention over agent states

**Recommendation**: Start with moments (mean, std), upgrade to learned embedding if needed.

### Challenge 2: Non-Stationarity

**Problem**: Population changes as agents learn, violating stationarity assumption.

**Solutions**:
1. **Experience Replay**: Old population states still provide useful gradients
2. **Population Tracking**: Slow target network updates (large τ)
3. **Curriculum Learning**: Gradually increase population size
4. **Fictitious Play**: Use time-averaged population

### Challenge 3: Coordination

**Problem**: Agents must coordinate without explicit communication.

**Solutions**:
1. **Shared Policy**: All agents use same π(s, m) (symmetric MFG)
2. **Population State**: Implicitly communicates via m
3. **Reward Shaping**: Incentivize coordination
4. **Curriculum**: Start with simple coordination tasks

---

## Applications to MFG Problems

### Application 1: Traffic Flow Control

**Problem**: Autonomous vehicles coordinate to minimize travel time.

**State**: s = (position, velocity, destination)
**Action**: a = (acceleration, steering) ∈ ℝ²
**Population**: m = traffic density field
**Reward**: r = -travel_time - fuel_cost - congestion_penalty

**Why MADDPG**:
- Continuous control (acceleration, steering)
- Centralized training captures traffic patterns
- Decentralized execution for scalability

### Application 2: Financial Markets

**Problem**: Traders learn optimal execution strategies with market impact.

**State**: s = (holdings, cash, price, volatility)
**Action**: a = trading_rate ∈ ℝ
**Population**: m = market order flow
**Reward**: r = profit - transaction_cost - impact_cost

**Why MADDPG**:
- Continuous trading decisions
- Market impact via population state
- Nash equilibrium = market clearing

### Application 3: Crowd Navigation

**Problem**: Pedestrians navigate while avoiding collisions.

**State**: s = (position, velocity, goal)
**Action**: a = (vₓ, vᵧ) ∈ ℝ²
**Population**: m = crowd density
**Reward**: r = progress_to_goal - collision_penalty - effort

**Why MADDPG**:
- Continuous velocity control
- Crowd density affects navigation
- Social coordination

---

## Comparison: MF-MADDPG vs Other MFG-RL Methods

| Method | Action Space | Scalability | Coordination | Best For |
|--------|--------------|-------------|--------------|----------|
| **MF Q-Learning** | Discrete | High | Implicit | Discrete control |
| **MF Actor-Critic** | Both | High | Implicit | General purpose |
| **Nash Q-Learning** | Discrete | High | Explicit | Competitive games |
| **MF-MADDPG** | **Continuous** | **High** | **Explicit (via critic)** | **Continuous control** |

**Advantages of MF-MADDPG**:
1. Handles continuous actions naturally
2. Centralized critic helps coordination
3. Deterministic policy gradient (lower variance)
4. Scales to large populations via mean field

**Disadvantages**:
1. More complex than Q-learning
2. Sensitive to hyperparameters
3. Exploration can be challenging
4. Requires careful tuning

---

## References

**Original MADDPG**:
- Lowe et al. (2017): "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"

**DDPG**:
- Lillicrap et al. (2015): "Continuous control with deep reinforcement learning"

**Mean Field Games**:
- Lasry & Lions (2007): "Mean field games"
- Carmona & Delarue (2018): "Probabilistic Theory of Mean Field Games"

**Mean Field RL**:
- Yang et al. (2018): "Mean Field Multi-Agent Reinforcement Learning"
- Guo et al. (2019): "Learning Mean-Field Games"

**Exploration**:
- Plappert et al. (2017): "Parameter Space Noise for Exploration"
- Fortunato et al. (2017): "Noisy Networks for Exploration"

---

**Status**: ✅ Theoretical foundation complete
**Next**: Design MF-MADDPG architecture for MFG_PDE
**Key Insight**: MADDPG's centralized critic + mean field = scalable continuous control for MFG
