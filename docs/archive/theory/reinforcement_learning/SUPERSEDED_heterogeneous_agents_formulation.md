# Heterogeneous Agent Mean Field Games: Mathematical Formulation

**Date**: October 2025
**Status**: Phase 3.2.1 Implementation
**Context**: Multi-population MFG with different agent types and objectives

---

## Executive Summary

This document provides the complete mathematical framework for **heterogeneous agent Mean Field Games** where multiple agent types (or **populations**) coexist, each with distinct:
- **Objectives**: Different reward functions $r^k(s, a, \mathbf{m})$
- **Dynamics**: Different transition probabilities $P^k(s' | s, a, \mathbf{m})$
- **Capabilities**: Different action spaces $\mathcal{A}^k$
- **Interactions**: Cross-population coupling through multi-population state $\mathbf{m} = (m^1, m^2, \ldots, m^K)$

**Key Applications**:
- **Traffic**: Cars, trucks, motorcycles with different speeds and goals
- **Epidemiology**: Susceptible, infected, recovered populations
- **Economics**: Buyers, sellers, market makers with different strategies
- **Pedestrian dynamics**: Adults, children, elderly with different mobility

**Strategic Importance**: Heterogeneous agents are **fundamental to realistic MFG modeling** and bridge the gap between single-population theory and real-world multi-type scenarios.

---

## 1. Mathematical Framework

### 1.1 Notation and Definitions

**Agent Types**: $K$ distinct agent types indexed by $k \in \{1, 2, \ldots, K\}$

**Individual Agent State Space**:
- Type $k$ agent state: $s^k \in \mathcal{S}^k$
- May vary by type (e.g., $\mathcal{S}^{\text{car}} = \{positions\}$, $\mathcal{S}^{\text{pedestrian}} = \{sidewalks\}$)

**Action Spaces**:
- Type $k$ action space: $\mathcal{A}^k$ (may be discrete or continuous)
- Example: $\mathcal{A}^{\text{car}} = \{\text{FORWARD, LEFT, RIGHT}\}$, $\mathcal{A}^{\text{truck}} = \{\text{FORWARD, SLOW}\}$

**Population States**:
- Single population: $m^k \in \mathcal{P}(\mathcal{S}^k)$ (distribution of type $k$ agents)
- Multi-population state: $\mathbf{m} = (m^1, m^2, \ldots, m^K)$

**Notation Convention**:
- Superscript $k$ denotes agent type: $s^k, a^k, r^k, \pi^k$
- Bold $\mathbf{m}$ denotes multi-population state

### 1.2 Heterogeneous MFG System

**HJB Equations** (one per type):
$$
-\partial_t u^k(t,x) + H^k(x, \nabla_x u^k, \mathbf{m}) = f^k(x, \mathbf{m}), \quad x \in \mathcal{S}^k, \quad k = 1, \ldots, K
$$

**Fokker-Planck Equations** (one per type):
$$
\partial_t m^k(t,x) - \Delta m^k - \nabla \cdot (m^k \nabla_p H^k(x, \nabla_x u^k, \mathbf{m})) = 0, \quad x \in \mathcal{S}^k
$$

**Coupling**: Hamiltonians $H^k$ and costs $f^k$ depend on **all population states** $\mathbf{m} = (m^1, \ldots, m^K)$

**Key Difference from Single-Population MFG**:
- Single-population: $u(t,x)$ and $m(t,x)$ for one type
- Multi-population: $K$ value functions $u^1, \ldots, u^K$ and $K$ distributions $m^1, \ldots, m^K$

### 1.3 Nash Equilibrium Concept

**Definition**: A **heterogeneous MFG equilibrium** is a tuple $(\pi^1, \ldots, \pi^K, m^1, \ldots, m^K)$ such that:

1. **Best Response**: Each policy $\pi^k$ is optimal for type $k$ agents given other populations:
   $$
   \pi^k \in \arg\max_{\pi} \mathbb{E}_{\pi}\left[ \sum_{t=0}^T \gamma^t r^k(s_t^k, a_t^k, \mathbf{m}_t) \right]
   $$

2. **Population Consistency**: Each population distribution $m^k$ is induced by type $k$ policy $\pi^k$:
   $$
   m^k_t = \mu_t(\pi^k) \quad \text{(distribution of type $k$ agents following $\pi^k$)}
   $$

3. **Fixed Point**: No agent of any type can unilaterally improve by deviating

**Special Cases**:
- **Symmetric MFG**: $K = 1$ (all agents identical)
- **Two-Population MFG**: $K = 2$ (e.g., predator-prey, buyers-sellers)
- **Multi-Population MFG**: $K \geq 3$ (e.g., S-I-R-V epidemiology)

---

## 2. Reinforcement Learning Formulation

### 2.1 Multi-Population MDP

**State Space**:
- Individual agent state: $s^k \in \mathcal{S}^k$ (type-specific)
- Multi-population state: $\mathbf{m} = (m^1, \ldots, m^K) \in \mathcal{P}(\mathcal{S}^1) \times \cdots \times \mathcal{P}(\mathcal{S}^K)$

**Action Space**: $\mathcal{A}^k$ (type-specific, may vary significantly)

**Transition Dynamics**:
$$
P^k(s'^k | s^k, a^k, \mathbf{m}) = \Pr(s_{t+1}^k = s'^k | s_t^k = s^k, a_t^k = a^k, \mathbf{m}_t = \mathbf{m})
$$

**Reward Function** (type-specific):
$$
r^k(s^k, a^k, \mathbf{m}) = r^k(s^k, a^k, m^1, \ldots, m^K)
$$

**Key Features**:
- Different types may have **conflicting objectives** (e.g., cars want speed, pedestrians want safety)
- Reward coupling through $\mathbf{m}$ creates **strategic interactions**
- Population dynamics evolve based on all types' policies

### 2.2 Policy Representation

**Type-Specific Policies**:
$$
\pi^k(a^k | s^k, \mathbf{m}): \mathcal{S}^k \times \mathcal{P}(\mathcal{S}^1) \times \cdots \times \mathcal{P}(\mathcal{S}^K) \to \Delta(\mathcal{A}^k)
$$

**Parametric Policies**: Neural network with type-specific parameters $\theta^k$:
```python
def policy_k(state_k, population_states):
    """
    Type k policy network.

    Args:
        state_k: Individual state of type k agent [batch, state_dim_k]
        population_states: Dict {type_id: pop_state} for all types

    Returns:
        Action probabilities for type k [batch, action_dim_k]
    """
    # Encode individual state
    state_features = state_encoder_k(state_k)

    # Encode ALL population states (cross-type awareness)
    pop_features = []
    for type_id, pop_state in population_states.items():
        pop_features.append(population_encoder_k[type_id](pop_state))
    multi_pop_features = concat(pop_features)

    # Combine and output action distribution
    combined = concat([state_features, multi_pop_features])
    action_logits = policy_head_k(combined)
    return softmax(action_logits)
```

### 2.3 Value Functions

**Q-Function per Type**:
$$
Q^k(s^k, a^k, \mathbf{m}) = \mathbb{E}_{\pi^k}\left[ \sum_{t=0}^\infty \gamma^t r^k(s_t^k, a_t^k, \mathbf{m}_t) \mid s_0^k = s^k, a_0^k = a^k, \mathbf{m}_0 = \mathbf{m} \right]
$$

**Bellman Equation** (type $k$):
$$
Q^k(s^k, a^k, \mathbf{m}) = r^k(s^k, a^k, \mathbf{m}) + \gamma \sum_{s'^k} P^k(s'^k | s^k, a^k, \mathbf{m}) V^k(s'^k, \mathbf{m}')
$$

where $\mathbf{m}' = (m^{1'}, \ldots, m^{K'})$ is the next multi-population state.

**Value Function per Type**:
$$
V^k(s^k, \mathbf{m}) = \sum_{a^k \in \mathcal{A}^k} \pi^k(a^k | s^k, \mathbf{m}) Q^k(s^k, a^k, \mathbf{m})
$$

---

## 3. Algorithmic Approaches

### 3.1 Multi-Population Q-Learning

**Algorithm**: Extend Mean Field Q-Learning to $K$ types

**Q-Network per Type**:
```python
class MultiPopulationQNetwork(nn.Module):
    """Q-network for type k agents in heterogeneous MFG."""

    def __init__(self, state_dim_k, action_dim_k, population_dims):
        """
        Args:
            state_dim_k: Dimension of type k state
            action_dim_k: Number of type k actions
            population_dims: Dict {type_id: pop_dim} for all K types
        """
        self.state_encoder = nn.Sequential(...)

        # Separate encoders for each population type
        self.pop_encoders = nn.ModuleDict({
            str(type_id): nn.Sequential(...)
            for type_id in population_dims.keys()
        })

        self.q_head = nn.Sequential(...)

    def forward(self, state_k, population_states):
        """
        Args:
            state_k: [batch, state_dim_k]
            population_states: Dict {type_id: [batch, pop_dim]}

        Returns:
            Q-values: [batch, action_dim_k]
        """
        state_feat = self.state_encoder(state_k)

        # Encode all population states
        pop_feats = []
        for type_id, pop_state in population_states.items():
            pop_feats.append(self.pop_encoders[str(type_id)](pop_state))

        combined = torch.cat([state_feat] + pop_feats, dim=1)
        return self.q_head(combined)
```

**Update Rule** (for type $k$ agent):
$$
Q^k(s^k, a^k, \mathbf{m}) \leftarrow Q^k(s^k, a^k, \mathbf{m}) + \alpha \left[ r^k + \gamma \max_{a'^k} Q^k(s'^k, a'^k, \mathbf{m}') - Q^k(s^k, a^k, \mathbf{m}) \right]
$$

**Population State Update**: Compute $\mathbf{m}' = (m^{1'}, \ldots, m^{K'})$ by aggregating states of all agent types.

### 3.2 Multi-Population Actor-Critic

**Architecture**:

**Actor per Type** $k$:
$$
\pi^k(a^k | s^k, \mathbf{m}; \theta^k)
$$

**Critic per Type** $k$:
$$
V^k(s^k, \mathbf{m}; \phi^k) \quad \text{or} \quad Q^k(s^k, a^k, \mathbf{m}; \phi^k)
$$

**Policy Gradient** (for type $k$):
$$
\nabla_{\theta^k} J(\theta^k) = \mathbb{E}\left[ \nabla_{\theta^k} \log \pi^k(a^k | s^k, \mathbf{m}; \theta^k) \cdot A^k(s^k, a^k, \mathbf{m}) \right]
$$

where advantage:
$$
A^k(s^k, a^k, \mathbf{m}) = Q^k(s^k, a^k, \mathbf{m}) - V^k(s^k, \mathbf{m})
$$

**Training Loop**:
1. Each type $k$ agent selects action $a^k \sim \pi^k(\cdot | s^k, \mathbf{m})$
2. Environment transitions all agents: $(s^k, a^k) \to (s'^k, r^k)$
3. Update multi-population state: $\mathbf{m} \to \mathbf{m}'$ (aggregate all types)
4. Update each type's policy and critic using collected experience

### 3.3 Nash Q-Learning for Heterogeneous MFG

**Challenge**: Nash equilibrium computation is **non-trivial** when $K > 1$ and objectives differ.

**For Symmetric Populations** ($K = 1$):
$$
\text{Nash value} = \max_{a} Q(s, a, m)
$$

**For Heterogeneous Populations** ($K \geq 2$):
- Need to solve **coupled best-response equations**:
  $$
  \pi^k \in \arg\max_{\pi} J^k(\pi, \pi^{-k})
  $$
  where $\pi^{-k} = (\pi^1, \ldots, \pi^{k-1}, \pi^{k+1}, \ldots, \pi^K)$

**Solution Approaches**:
1. **Fictitious Play**: Iteratively update each type's policy assuming others are fixed
2. **Best Response Dynamics**: Each type plays best response to current population
3. **Nash-Q Iteration**: Extend Nash-Q to multi-population (see Section 4.3)

---

## 4. Implementation Challenges

### 4.1 Population State Representation

**Challenge**: Represent $\mathbf{m} = (m^1, \ldots, m^K)$ efficiently

**Options**:

**1. Concatenated Histograms**:
```python
# Each m^k is a histogram over S^k
population_state = {
    "type_1": histogram_1,  # [num_bins_1]
    "type_2": histogram_2,  # [num_bins_2]
    ...
}
# Concatenate for neural network input
pop_vector = torch.cat([histogram_1, histogram_2, ...])  # [sum(num_bins_k)]
```

**2. Separate Encoders** (recommended for heterogeneous types):
```python
# Learn separate embeddings for each population type
pop_embedding_1 = pop_encoder_1(histogram_1)  # [embedding_dim]
pop_embedding_2 = pop_encoder_2(histogram_2)  # [embedding_dim]
# Combine embeddings
multi_pop_embedding = torch.cat([pop_embedding_1, pop_embedding_2])
```

**3. Attention Mechanisms** (for large $K$):
```python
# Agent of type k attends to all population types
pop_embeddings = [pop_encoder_i(histogram_i) for i in range(K)]
attended_pop = attention_layer(query=state_k, keys=pop_embeddings)
```

### 4.2 Action Space Heterogeneity

**Challenge**: Different types may have different action spaces (discrete vs continuous, different dimensions)

**Solution**: Type-specific policy networks

```python
class HeterogeneousMFGSolver:
    def __init__(self, agent_types_config):
        self.policies = {}
        self.critics = {}

        for type_id, config in agent_types_config.items():
            if config["action_type"] == "discrete":
                self.policies[type_id] = DiscretePolicy(
                    state_dim=config["state_dim"],
                    action_dim=config["action_dim"],
                    population_dims=self._get_pop_dims()
                )
            elif config["action_type"] == "continuous":
                self.policies[type_id] = ContinuousPolicy(
                    state_dim=config["state_dim"],
                    action_dim=config["action_dim"],
                    population_dims=self._get_pop_dims()
                )
```

### 4.3 Nash Equilibrium Computation

**Fictitious Play Algorithm**:

```python
def fictitious_play_training(env, agent_types, num_iterations=1000):
    """
    Train heterogeneous agents using fictitious play.

    Each iteration:
    1. Fix policies π^{-k} for all types except k
    2. Train type k policy π^k to best respond
    3. Rotate through all types
    """
    policies = {k: initialize_policy(k) for k in agent_types}

    for iteration in range(num_iterations):
        for k in agent_types:
            # Fix all other policies
            frozen_policies = {j: policies[j] for j in agent_types if j != k}

            # Train type k to best respond
            policies[k] = train_best_response(
                env=env,
                type_id=k,
                other_policies=frozen_policies,
                num_episodes=100
            )

        # Check convergence (all types are mutual best responses)
        if check_nash_convergence(policies):
            break

    return policies
```

### 4.4 Sample Efficiency

**Challenge**: Need to collect experience for $K$ agent types simultaneously

**Solution 1: Parallel Rollouts**:
```python
# Sample trajectories for all types in parallel
trajectories = {k: [] for k in agent_types}

for episode in range(num_episodes):
    states = {k: env.reset_type(k) for k in agent_types}
    multi_pop_state = env.get_multi_population_state()

    while not done:
        # All types act simultaneously
        actions = {k: policies[k].select_action(states[k], multi_pop_state)
                   for k in agent_types}

        # Environment step (all types transition)
        next_states, rewards, dones = env.step(actions)

        # Store experience for each type
        for k in agent_types:
            trajectories[k].append((states[k], actions[k], rewards[k], next_states[k]))

        states = next_states
        multi_pop_state = env.get_multi_population_state()
```

**Solution 2: Prioritized Experience Replay per Type**:
- Maintain separate replay buffers for each type
- Sample proportionally to learning progress

---

## 5. Example Applications

### 5.1 Predator-Prey MFG

**Two Populations**: Predators ($K=1$) and Prey ($K=2$)

**State Spaces**:
- Both: $\mathcal{S}^1 = \mathcal{S}^2 = \{(x, y) : \text{grid positions}\}$

**Action Spaces**:
- Both: $\mathcal{A}^1 = \mathcal{A}^2 = \{\text{UP, DOWN, LEFT, RIGHT}\}$

**Rewards**:
- Predator: $r^{\text{pred}}(s, a, m^{\text{pred}}, m^{\text{prey}}) = +10$ if capture prey, $-0.1$ step cost
- Prey: $r^{\text{prey}}(s, a, m^{\text{pred}}, m^{\text{prey}}) = +10$ if reach safe zone, $-100$ if caught

**Coupling**: Predators seek high $m^{\text{prey}}$ regions, prey avoid high $m^{\text{pred}}$ regions

### 5.2 Traffic with Multiple Vehicle Types

**Three Populations**: Cars ($K=1$), Trucks ($K=2$), Motorcycles ($K=3$)

**Dynamics**:
- Cars: Fast acceleration, moderate speed
- Trucks: Slow acceleration, restricted to right lanes
- Motorcycles: Very fast, can use all lanes

**Rewards**:
- All types: Reach destination quickly
- Cars/Trucks: Penalty for congestion
- Motorcycles: Penalty for unsafe maneuvers

**Heterogeneity**: Different action spaces (lane changes allowed/forbidden), different speeds

### 5.3 Epidemic Model (S-I-R)

**Three Populations**: Susceptible ($S$), Infected ($I$), Recovered ($R$)

**State Transition**:
- $S \to I$: Infection based on contact with $I$ population
- $I \to R$: Recovery at fixed rate
- $R$: Immune, no further transitions

**Actions**:
- $S$: Social distancing level (affects infection probability)
- $I$: Isolation level (affects transmission rate)
- $R$: Normal behavior

**Rewards**:
- $S$: Minimize infection risk while maintaining social activity
- $I$: Minimize transmission while maintaining quality of life
- $R$: Maximize social activity

---

## 6. Theoretical Properties

### 6.1 Existence of Equilibrium

**Theorem** (Informal): Under mild conditions on rewards and dynamics, a heterogeneous MFG Nash equilibrium exists.

**Conditions**:
1. **Monotonicity**: $r^k(s, a, \mathbf{m})$ is monotone in each $m^j$
2. **Regularity**: Transition kernels $P^k$ are continuous in $\mathbf{m}$
3. **Compactness**: State and action spaces are compact

**Key Reference**: Carmona & Delarue (2018), Vol. II, Chapter 6 - Multi-population MFG

### 6.2 Uniqueness

**When is equilibrium unique?**

**Sufficient Condition**: Lasry-Lions monotonicity for **all types**:
$$
\sum_{k=1}^K \int_{\mathcal{S}^k} (m^k - \tilde{m}^k)(u^k[\tilde{\mathbf{m}}] - u^k[\mathbf{m}]) \, dx \geq \kappa \|\mathbf{m} - \tilde{\mathbf{m}}\|^2
$$

**Challenge**: Multi-population monotonicity is **much stronger** than single-population case.

**In Practice**: Uniqueness often fails for competitive heterogeneous MFG (e.g., predator-prey may have multiple equilibria).

### 6.3 Convergence of RL Algorithms

**Open Research Question**: Do multi-population RL algorithms converge to Nash equilibrium?

**Known Results**:
- **Fictitious Play**: Converges for zero-sum two-player games
- **Best Response Dynamics**: May cycle, not guaranteed to converge
- **Mean Field Q-Learning**: Convergence for symmetric MFG (Yang et al., 2018)

**Heterogeneous Case**: Limited theoretical guarantees, primarily empirical validation.

---

## 7. Implementation Roadmap

### Phase 1: Two-Population Baseline (2-3 weeks)
- [ ] **Environment**: Extend `MFGMazeEnvironment` to support two agent types
- [ ] **Population State**: Multi-population state representation
- [ ] **Q-Learning**: Implement two-population Q-learning with separate networks
- [ ] **Example**: Predator-prey maze navigation

### Phase 2: General K-Population Framework (3-4 weeks)
- [ ] **Architecture**: Generalize to arbitrary $K$ agent types
- [ ] **Config System**: YAML configuration for multi-population setups
- [ ] **Testing**: Validate with $K = 3$ (e.g., S-I-R model)
- [ ] **Documentation**: API docs and usage examples

### Phase 3: Advanced Algorithms (4-5 weeks)
- [ ] **Nash Equilibrium**: Fictitious play implementation
- [ ] **Actor-Critic**: Multi-population PPO
- [ ] **Benchmarking**: Compare convergence across algorithms
- [ ] **Applications**: Traffic, epidemiology examples

### Phase 4: Optimization & Integration (2-3 weeks)
- [ ] **Performance**: Efficient multi-population state updates
- [ ] **Visualization**: Multi-population density plots
- [ ] **Cross-paradigm**: Connect to numerical multi-population solvers
- [ ] **Documentation**: Theory docs and tutorials

---

## 8. References

### Theoretical Foundations
1. **Carmona & Delarue** (2018). *Probabilistic Theory of Mean Field Games*, Volume II, Chapter 6 - Multi-population games
2. **Lasry & Lions** (2007). *Mean field games*, Japanese Journal of Mathematics
3. **Guo et al.** (2019). *Learning Mean-Field Games*, NeurIPS

### Multi-Agent RL
4. **Mguni et al.** (2022). *Multi-Agent Reinforcement Learning in Games*, Cambridge University Press
5. **Yang et al.** (2018). *Mean Field Multi-Agent Reinforcement Learning*, ICML
6. **Subramanian & Mahajan** (2019). *Reinforcement Learning in Stationary Mean-field Games*, AAMAS

### Applications
7. **Lachapelle et al.** (2016). *Efficiency of the Price Formation Process*, Mathematics and Financial Economics
8. **Laguzet & Turinici** (2015). *Individual Vaccination as Nash Equilibrium*, Mathematical Biosciences
9. **Bauso et al.** (2016). *Mean-field Games for Traffic Flow Control*, Transportation Research Part B

---

**Status**: Ready for implementation (Phase 3.2.1)
**Next**: Implement two-population MFG environment and Q-learning algorithm
**Timeline**: 2-3 weeks for Phase 1, 10-15 weeks for complete implementation
