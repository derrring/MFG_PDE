# Population PPO for Mean Field Games: Theoretical Formulation

**Date**: October 2, 2025
**Status**: Implemented ✅
**Context**: Phase 2.3 - Multi-Agent Extensions for MFG-RL

---

## Executive Summary

This document presents the theoretical foundation for **Population PPO (Proximal Policy Optimization)** for Mean Field Games. We show that our existing **Mean Field Actor-Critic** implementation (with PPO clipping and GAE) is equivalent to Population PPO for symmetric MFG.

**Key Finding**: Population PPO for MFG = PPO + Population State Conditioning

**Implementation Status**: ✅ Already implemented in `mfg_pde/alg/reinforcement/algorithms/mean_field_actor_critic.py`

---

## Background: Standard PPO

### PPO (Schulman et al. 2017)

**Core Innovation**: Constrain policy updates to prevent destructively large changes.

**Clipped Surrogate Objective**:
```
L^CLIP(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]

where:
  r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  # Probability ratio
  Â_t = Advantage estimate (often from GAE)
  ε = Clipping parameter (typically 0.2)
```

**Key Properties**:
- Simple and effective
- No need for KL penalty or trust region constraint (unlike TRPO)
- Works well with neural networks
- Sample efficient through multiple epochs on same batch

### Generalized Advantage Estimation (GAE)

**TD-Error**:
```
δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

**GAE Formula**:
```
Â^GAE_t = ∑_{l=0}^∞ (γλ)^l δ_{t+l}

where:
  λ ∈ [0,1] controls bias-variance tradeoff
  λ=0: Low variance, high bias (TD estimate)
  λ=1: High variance, low bias (Monte Carlo)
```

**Exponentially-Weighted Average**: Balances Monte Carlo (λ=1) and TD(0) (λ=0).

---

## Population PPO for Mean Field Games

### Problem Formulation

**Mean Field Game with Large Population**:
- **State**: Individual agent state `s ∈ S`
- **Action**: Agent action `a ∈ A` (discrete)
- **Population**: Mean field `m ∈ P(S)` (population distribution)
- **Reward**: `r(s, a, m)` depends on individual state, action, AND population

**Objective**: Learn policy `π(a|s,m)` that achieves Nash equilibrium:
```
π* = argmax E[∑_t γ^t r(s_t, a_t, m_t) | π, m]
subject to: m = μ(π)  # Population consistency
```

### Population PPO Framework

#### Key Modifications to Standard PPO

**1. Population-Conditioned Policy**:
```
Standard PPO:    π(a|s)
Population PPO:  π(a|s,m)
```

**2. Population-Conditioned Value**:
```
Standard PPO:    V(s)
Population PPO:  V(s,m)
```

**3. Population-Aware Advantage**:
```
δ_t = r(s_t, a_t, m_t) + γV(s_{t+1}, m_{t+1}) - V(s_t, m_t)
Â^GAE_t = ∑_{l=0}^∞ (γλ)^l δ_{t+l}
```

**4. Population Update**:
```
For each timestep: m_t ← aggregate({s_i^t : i=1,...,N})
```

### Population PPO Objective

**Clipped Surrogate Objective** (adapted for MFG):
```
L^CLIP_MFG(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]

where:
  r_t(θ) = π_θ(a_t|s_t,m_t) / π_θ_old(a_t|s_t,m_t)  # Ratio with population
  Â_t = Population-aware GAE advantage
  m_t = Population state at time t
```

**Value Function Loss**:
```
L^VF(φ) = E[(V_φ(s_t, m_t) - V^target_t)²]

where:
  V^target_t = Â_t + V(s_t, m_t)  # GAE-based return
```

**Total Loss**:
```
L(θ, φ) = L^CLIP_MFG(θ) + c_1 L^VF(φ) - c_2 H[π_θ(·|s,m)]

where:
  c_1 = Value function coefficient
  c_2 = Entropy bonus coefficient
  H = Entropy for exploration
```

---

## Algorithm: Population PPO

### Pseudocode

```python
# Initialize networks
actor = PopulationAwareActor(state_dim, action_dim, population_dim)
critic = PopulationAwareCritic(state_dim, population_dim)

for episode in episodes:
    # Collect rollout
    trajectory = []
    state = env.reset()

    for t in timesteps:
        # Compute population state
        m_t = compute_population_state(all_agent_states)

        # Select action
        action, log_prob = actor.sample(state, m_t)

        # Get value estimate
        value = critic(state, m_t)

        # Execute action
        next_state, reward, done = env.step(action)
        m_{t+1} = compute_population_state(all_agent_states)

        # Store experience
        trajectory.append((state, action, reward, value, log_prob, m_t, done))

        state = next_state
        if done:
            break

    # Compute GAE advantages
    advantages, returns = compute_gae(trajectory, gamma, lambda)

    # Update policy (multiple epochs)
    for epoch in range(K):
        for batch in minibatch(trajectory):
            # Extract batch
            states, actions, old_log_probs, advantages, returns, populations = batch

            # Compute new log probs
            new_log_probs = actor.log_prob(actions, states, populations)

            # Compute ratio
            ratio = exp(new_log_probs - old_log_probs)

            # PPO clipped objective
            surr1 = ratio * advantages
            surr2 = clip(ratio, 1-eps, 1+eps) * advantages
            policy_loss = -min(surr1, surr2).mean()

            # Value loss
            values = critic(states, populations)
            value_loss = MSE(values, returns)

            # Total loss
            loss = policy_loss + c1*value_loss - c2*entropy

            # Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## Mathematical Properties

### Monotonic Improvement

**PPO Guarantee**: Policy improves monotonically (or stays same) if updates are conservative.

**For Population PPO**:
```
J(π_new) ≥ J(π_old) - C·ε

where:
  C = Constant depending on advantage estimates
  ε = Clipping parameter
```

**Caveat for MFG**: Non-stationarity from population changes means strict monotonicity isn't guaranteed, but empirically PPO still performs well.

### Population Consistency

**Nash Equilibrium Condition**:
```
π* = argmax J(π | m*)
m* = μ(π*)
```

**PPO helps convergence** by:
1. Conservative updates (clipping prevents large policy changes)
2. Multiple epochs per rollout (more efficient use of population data)
3. GAE advantages (better credit assignment with population)

### Variance Reduction

**GAE with Population State**:
- Lower variance than Monte Carlo returns
- Accounts for population dynamics in advantage estimates
- λ parameter controls bias-variance tradeoff

---

## Implementation in MFG_PDE

### Network Architecture

**Actor Network**: `π(a|s,m; θ)`
```python
class PopulationAwareActor(nn.Module):
    def __init__(self, state_dim, action_dim, population_dim):
        # State encoder
        self.state_encoder = nn.Sequential(...)

        # Population encoder
        self.population_encoder = nn.Sequential(...)

        # Policy head (softmax for discrete actions)
        self.policy_head = nn.Sequential(
            nn.Linear(...),
            nn.Softmax(dim=-1)  # Action probabilities
        )

    def forward(self, state, population):
        state_feat = self.state_encoder(state)
        pop_feat = self.population_encoder(population)
        combined = torch.cat([state_feat, pop_feat], dim=1)
        action_probs = self.policy_head(combined)
        return action_probs
```

**Critic Network**: `V(s,m; φ)`
```python
class PopulationAwareCritic(nn.Module):
    def __init__(self, state_dim, population_dim):
        # State encoder
        self.state_encoder = nn.Sequential(...)

        # Population encoder
        self.population_encoder = nn.Sequential(...)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(...),
            nn.Linear(..., 1)  # Scalar value
        )

    def forward(self, state, population):
        state_feat = self.state_encoder(state)
        pop_feat = self.population_encoder(population)
        combined = torch.cat([state_feat, pop_feat], dim=1)
        value = self.value_head(combined)
        return value
```

### PPO Update Implementation

```python
# From mean_field_actor_critic.py (lines 555-559)
def _update_policy(self, states, populations, actions, old_log_probs, advantages, returns):
    # Get new log probs
    logits = self.actor(states, populations)
    dist = Categorical(logits)
    new_log_probs = dist.log_prob(actions)

    # PPO clipped objective
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss
    values = self.critic(states, populations)
    value_loss = F.mse_loss(values, returns)

    # Update networks
    self.actor_optimizer.zero_grad()
    policy_loss.backward()
    nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
    self.actor_optimizer.step()

    self.critic_optimizer.zero_grad()
    value_loss.backward()
    nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
    self.critic_optimizer.step()
```

### GAE Implementation

```python
# From mean_field_actor_critic.py (compute_gae method)
def compute_gae(self, rewards, values, next_value, dones):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    returns = []
    gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        # TD error with population-aware values
        delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]

        # GAE accumulation
        gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])

    return advantages, returns
```

---

## Comparison: PPO vs Population PPO

| Aspect | Standard PPO | Population PPO (MFG) |
|--------|--------------|----------------------|
| **State** | s | (s, m) |
| **Policy** | π(a\|s) | π(a\|s, m) |
| **Value** | V(s) | V(s, m) |
| **Advantage** | A(s, a) | A(s, a, m) |
| **Environment** | Stationary | Non-stationary (m changes) |
| **Objective** | Single agent optimal | Nash equilibrium |
| **Scalability** | Single agent | Scales to N → ∞ |

---

## Advantages of PPO for MFG

### 1. Sample Efficiency
- Multiple epochs on same data
- Importance sampling with clipping
- Better than vanilla policy gradient

### 2. Stability
- Clipping prevents destructive updates
- Gradient clipping for numerical stability
- Conservative policy changes

### 3. Simplicity
- No KL constraint (unlike TRPO)
- No second-order optimization
- Easy to implement and tune

### 4. Population Compatibility
- Handles non-stationarity from population changes
- Conservative updates match slow population evolution
- GAE accounts for population dynamics

---

## Challenges and Solutions

### Challenge 1: Non-Stationary Environment

**Problem**: Population `m` changes as agents learn, violating stationarity.

**PPO Solutions**:
1. **Conservative Updates**: Clipping limits policy changes per step
2. **Multiple Rollouts**: Collect data from current population distribution
3. **Value Function**: Learns to account for population dynamics

### Challenge 2: Credit Assignment

**Problem**: Rewards depend on both individual actions and population state.

**PPO Solutions**:
1. **GAE**: Exponentially-weighted advantages capture long-term effects
2. **Population-Conditioned Value**: V(s,m) explicitly models population influence
3. **λ Parameter**: Tune bias-variance tradeoff for population dynamics

### Challenge 3: Exploration

**Problem**: Need to explore both action space and population configurations.

**PPO Solutions**:
1. **Entropy Bonus**: Encourages exploration in action space
2. **Stochastic Policy**: Naturally explores via sampling
3. **Multiple Agents**: Population diversity provides exploration

---

## Hyperparameters

### Recommended Settings for MFG

```python
config = {
    # PPO parameters
    "clip_epsilon": 0.2,        # Standard PPO clipping
    "gae_lambda": 0.95,         # High lambda for long-term credit
    "gamma": 0.99,              # Standard discount

    # Optimization
    "actor_lr": 3e-4,
    "critic_lr": 1e-3,
    "max_grad_norm": 0.5,       # Gradient clipping

    # Training
    "batch_size": 64,
    "update_epochs": 10,        # Multiple passes per batch

    # Network
    "hidden_dims": [256, 256],

    # Population
    "population_size": 100,
    "population_update_freq": 1,  # Update every step
}
```

### Sensitivity Analysis

**Critical Parameters**:
1. `clip_epsilon`: Too small = slow learning, too large = instability
2. `gae_lambda`: Higher λ for long-horizon MFG problems
3. `update_epochs`: More epochs = better sample efficiency but risk overfitting

**Less Critical**:
- Learning rates (standard Adam defaults work)
- Batch size (larger is more stable)
- Hidden dimensions (256+ recommended)

---

## Applications to MFG Problems

### Application 1: Crowd Navigation

**Setup**:
- State: (position, velocity, goal)
- Action: Movement direction (4 or 8 choices)
- Population: Local crowd density
- Reward: Progress to goal - collision penalty

**Why PPO**:
- Sample efficient for sparse rewards
- Stable learning with crowd dynamics
- Exploration via entropy bonus

### Application 2: Epidemic Control

**Setup**:
- State: (health status, local infection rate)
- Action: (isolate, normal activity)
- Population: Infected/susceptible ratios
- Reward: Health - isolation cost

**Why PPO**:
- Handles delayed rewards (infection spread)
- GAE captures long-term health outcomes
- Population dynamics explicitly modeled

### Application 3: Market Making

**Setup**:
- State: (holdings, cash, price)
- Action: Buy/sell/hold
- Population: Market order flow
- Reward: Profit - transaction costs

**Why PPO**:
- Stable with non-stationary markets
- Conservative updates prevent market impact
- Entropy encourages market exploration

---

## Comparison with Other MFG-RL Methods

| Method | Sample Efficiency | Stability | Continuous Actions | Best For |
|--------|------------------|-----------|-------------------|----------|
| **Q-Learning** | Low | High | No | Simple discrete |
| **Actor-Critic (A2C)** | Medium | Medium | Yes | General purpose |
| **Population PPO** | **High** | **High** | No* | **Discrete MFG** |
| **MADDPG** | Medium | Medium | Yes | Continuous control |

*Can be extended to continuous actions with DDPG-style deterministic actor.

**Population PPO Advantages**:
1. Most sample efficient for discrete actions
2. Excellent stability with non-stationary populations
3. Simple to implement and tune
4. Works well across different MFG problem types

---

## Implementation Status

### ✅ Completed Features

**File**: `mfg_pde/alg/reinforcement/algorithms/mean_field_actor_critic.py`

**Implemented**:
- [x] Population-aware actor network
- [x] Population-aware critic network
- [x] PPO clipped objective (lines 555-559)
- [x] GAE advantage estimation (compute_gae method)
- [x] Gradient clipping
- [x] Advantage normalization
- [x] Training loop with rollouts
- [x] Model save/load

**Tests**: `tests/unit/test_mean_field_actor_critic.py`

**Examples**: `examples/advanced/actor_critic_maze_demo.py`

---

## References

**PPO**:
- Schulman et al. (2017): "Proximal Policy Optimization Algorithms"
- Schulman et al. (2015): "Trust Region Policy Optimization" (TRPO)

**GAE**:
- Schulman et al. (2015): "High-Dimensional Continuous Control Using Generalized Advantage Estimation"

**Mean Field Games**:
- Lasry & Lions (2007): "Mean field games"
- Carmona & Delarue (2018): "Probabilistic Theory of Mean Field Games"

**Mean Field RL**:
- Yang et al. (2018): "Mean Field Multi-Agent Reinforcement Learning"
- Perrin et al. (2020): "Fictitious play for mean field games"

---

**Status**: ✅ Fully implemented and tested
**Implementation**: `mean_field_actor_critic.py`
**Key Insight**: Population PPO = PPO + Population State Conditioning. Already implemented!
