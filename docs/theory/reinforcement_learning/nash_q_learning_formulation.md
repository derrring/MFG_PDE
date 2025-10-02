# Nash Q-Learning for Mean Field Games

**Date**: October 2, 2025
**Status**: Theoretical Foundation
**Context**: Phase 2.3 - Multi-Agent Extensions for MFG-RL

---

## Mathematical Background

### Nash Q-Learning (Hu & Wellman, 2003)

Nash Q-Learning extends standard Q-learning to multi-agent settings by learning **Nash equilibrium policies** rather than single-agent optimal policies.

**Key Idea**: Each agent learns a Q-function that captures the value of joint actions, accounting for the strategic behavior of other agents.

---

## Standard Nash Q-Learning (Finite Agents)

### Problem Formulation

**N-Agent Stochastic Game**:
- **States**: $s \in \mathcal{S}$ (shared state space)
- **Actions**: $a^i \in \mathcal{A}^i$ for agent $i$, joint action $\mathbf{a} = (a^1, \ldots, a^N)$
- **Rewards**: $r^i(s, \mathbf{a})$ for agent $i$
- **Transitions**: $P(s' | s, \mathbf{a})$

**Objective**: Find Nash equilibrium policies $\pi^* = (\pi^{1*}, \ldots, \pi^{N*})$ such that:
$$
Q^i(s, \pi^*) \geq Q^i(s, \pi^{i'}, \pi^{-i*}) \quad \forall i, \pi^{i'}, s
$$

### Nash Q-Function

**Definition**: For agent $i$, the Nash Q-function is:
$$
Q^i(s, \mathbf{a}) = r^i(s, \mathbf{a}) + \gamma \sum_{s'} P(s' | s, \mathbf{a}) \cdot \text{Nash}_i(s')
$$

Where $\text{Nash}_i(s')$ is the expected value to agent $i$ under Nash equilibrium at state $s'$:
$$
\text{Nash}_i(s') = \sum_{\mathbf{a}'} \pi^*(a^1 | s') \cdots \pi^*(a^N | s') \cdot Q^i(s', \mathbf{a}')
$$

### Update Rule

**Bellman-style update**:
$$
Q^i(s, \mathbf{a}) \leftarrow (1 - \alpha) Q^i(s, \mathbf{a}) + \alpha \left[ r^i + \gamma \cdot \text{Nash}_i(s') \right]
$$

**Challenge**: Computing $\text{Nash}_i(s')$ requires solving a Nash equilibrium problem at each state.

---

## Mean Field Nash Q-Learning

### Motivation

For large populations ($N \to \infty$), tracking all agents is infeasible. **Mean Field Games** approximate the population effect via the **mean field** $m$.

**Key Insight**: Replace joint actions $\mathbf{a} = (a^1, \ldots, a^N)$ with:
- **Individual action**: $a \in \mathcal{A}$
- **Population distribution**: $m \in \mathcal{P}(\mathcal{S})$

### Mean Field Nash Q-Function

**For a representative agent**:
$$
Q(s, a, m) = r(s, a, m) + \gamma \mathbb{E}_{s' \sim P(\cdot | s, a, m)} \left[ V(s', m') \right]
$$

Where:
- $m$ is the current population distribution
- $m'$ is the next population distribution (induced by the population policy)
- $V(s, m) = \mathbb{E}_{a \sim \pi(\cdot | s, m)} [Q(s, a, m)]$

### Mean Field Nash Equilibrium

**Equilibrium condition**: The policy $\pi^*$ is a Nash equilibrium if:
1. **Best response**: $\pi^*(s, m)$ maximizes $Q(s, a, m)$ for each $s, m$
2. **Population consistency**: $m = \mu(\pi^*)$ (population distribution matches policy)

**Mathematical formulation**:
$$
\pi^*(s, m) \in \arg\max_a Q(s, a, m), \quad m = \mathbb{E}_{\tau \sim \pi^*}[\text{state distribution}]
$$

---

## Nash Q-Learning Algorithm for MFG

### Algorithm Structure

```python
# Pseudocode
Initialize Q(s, a, m) randomly
Initialize population m_0
for episode in episodes:
    # 1. Sample agent trajectories under current policy
    for agent in population:
        for t in timesteps:
            s_t = current_state
            m_t = estimate_population()
            a_t = epsilon_greedy(Q(s_t, ·, m_t))  # Exploration
            execute a_t, observe r_t, s_{t+1}
            m_{t+1} = estimate_population()

            # 2. Update Q-function (Nash Q-learning)
            nash_value = compute_nash_value(s_{t+1}, m_{t+1})
            Q(s_t, a_t, m_t) += alpha * (r_t + gamma * nash_value - Q(s_t, a_t, m_t))

    # 3. Update population distribution
    m = aggregate_agent_states()
```

### Computing Nash Value

**For discrete actions** (standard approach):
$$
\text{Nash}(s, m) = \text{solve Nash equilibrium from } Q(s, \cdot, m)
$$

**For Mean Field Games** (simplified):
Since we assume a **representative agent** and **symmetric equilibrium**, the Nash value simplifies to:
$$
V(s, m) = \max_a Q(s, a, m)
$$

This is because in symmetric MFG, all agents follow the same policy $\pi^*$, so the Nash equilibrium reduces to best response to the mean field.

---

## Differences from Standard Q-Learning

| Aspect | Standard Q-Learning | Nash Q-Learning | Mean Field Nash Q-Learning |
|--------|-------------------|----------------|--------------------------|
| **State** | $s$ | $s$ | $(s, m)$ |
| **Action** | $a$ | $\mathbf{a}$ (joint) | $a$ (individual) |
| **Q-function** | $Q(s, a)$ | $Q^i(s, \mathbf{a})$ | $Q(s, a, m)$ |
| **Target** | $\max_{a'} Q(s', a')$ | $\text{Nash}_i(s')$ | $\max_{a'} Q(s', a', m')$ |
| **Equilibrium** | Single agent optimal | Nash equilibrium | MFG Nash equilibrium |
| **Scalability** | O(1) | O($|\mathcal{A}|^N$) | O($|\mathcal{A}|$) |

**Key observation**: Mean Field Nash Q-Learning **scales like standard Q-learning** (O(|A|)), not exponentially in population size.

---

## Implementation Challenges

### 1. Population Estimation

**Problem**: How to estimate $m$ from finite agent samples?

**Solutions**:
- **Grid-based**: Discretize state space, count agents per cell
- **Kernel Density Estimation**: $m(s) = \frac{1}{N} \sum_{i=1}^N K_h(s - s_i)$
- **Histogram**: Binned distribution
- **Neural density**: Learn $m_\theta(s)$ via normalizing flow

### 2. Non-Stationarity

**Problem**: Population $m$ changes during learning, making environment non-stationary.

**Solutions**:
- **Fictitious Play**: Use historical average of populations
- **Experience Replay**: Store $(s, a, r, s', m, m')$ with old populations
- **Target Networks**: Stabilize Q-learning updates
- **Slow Population Updates**: Update $m$ less frequently than $Q$

### 3. Exploration-Exploitation

**Problem**: Need to explore both action space and population distribution.

**Solutions**:
- **$\epsilon$-greedy**: Standard exploration in action space
- **Population diversity**: Ensure diverse initial conditions
- **Entropy regularization**: Encourage policy diversity

### 4. Nash Computation (If needed)

**Problem**: Computing Nash equilibrium at each state is expensive.

**Simplification for MFG**:
- **Symmetric games**: Nash reduces to $\max_a Q(s, a, m)$
- **Best response dynamics**: Iterate best responses instead of exact Nash
- **Approximate Nash**: Use approximate solution methods

---

## Mean Field Nash Q-Learning vs. Mean Field Q-Learning

### Mean Field Q-Learning (Yang et al. 2018)
- **Assumption**: Agents cooperate (or are indistinguishable)
- **Objective**: Find policy maximizing $\mathbb{E}[\sum_t r(s_t, a_t, m_t)]$
- **Update**: Standard Q-learning with mean field as additional state

### Mean Field Nash Q-Learning
- **Assumption**: Agents are strategic (seek Nash equilibrium)
- **Objective**: Find Nash equilibrium policy
- **Update**: Nash Q-learning with mean field

**When to use**:
- **MF Q-Learning**: Cooperative or single-population games
- **MF Nash Q-Learning**: Competitive or multi-population games

**For our implementation**: Since most MFG problems assume **symmetric equilibrium** (all agents identical), Mean Field Q-Learning is often sufficient. Nash Q-Learning is needed when we have:
- **Heterogeneous agents** (multiple types)
- **Competitive interactions**
- **Explicit multi-agent dynamics**

---

## Implementation Design for MFG_PDE

### Architecture

```python
class MeanFieldNashQLearning:
    """
    Nash Q-Learning for Mean Field Games.

    For symmetric MFG, this simplifies to standard Q-learning with
    population state, since Nash equilibrium reduces to best response
    to mean field.
    """

    def __init__(self, state_dim, action_dim, population_dim):
        # Q-network: Q(s, a, m) -> R
        self.q_network = NashQNetwork(state_dim, action_dim, population_dim)
        self.target_q_network = NashQNetwork(state_dim, action_dim, population_dim)

        # Nash solver (optional, for heterogeneous agents)
        self.nash_solver = None  # SimplexProjection, LinearProgramming, etc.

    def compute_nash_value(self, state, population):
        """
        Compute Nash equilibrium value at state.

        For symmetric MFG, this is just max_a Q(s, a, m).
        For heterogeneous agents, solve Nash equilibrium.
        """
        if self.symmetric:
            # Symmetric MFG: Nash = max over actions
            q_values = self.target_q_network(state, population)  # [batch, action_dim]
            return q_values.max(dim=1, keepdim=True)  # [batch, 1]
        else:
            # General case: solve Nash equilibrium
            q_values = self.target_q_network(state, population)
            nash_strategy = self.nash_solver.solve(q_values)
            return (nash_strategy * q_values).sum(dim=1, keepdim=True)
```

### Network Architecture

**Option 1: Shared Q-Network** (our current approach):
```python
Q(s, a, m) = f(s, a, m)  # Single network for all agents
```

**Option 2: Type-Specific Q-Networks** (for heterogeneous agents):
```python
Q^i(s, a, m) = f_i(s, a, m)  # Separate network per agent type
```

---

## Theoretical Properties

### Convergence

**Theorem (Hu & Wellman, 2003)**: Under certain conditions, Nash Q-learning converges to a Nash equilibrium:
1. **Deterministic state transitions**
2. **Stage games have unique Nash equilibria**
3. **Diminishing learning rates**: $\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$

**For Mean Field Games**: Convergence is more subtle due to population dynamics. Theoretical guarantees exist for:
- **Contractive mean field operators** (Carmona & Delarue)
- **Monotone MFG** (Lasry & Lions)
- **Potential games** (Monderer & Shapley)

### Nash Equilibrium Characterization

**Fixed-Point Formulation**: Nash equilibrium satisfies:
$$
\pi^* = \text{BestResponse}(\mu(\pi^*))
$$

Where:
- $\pi^*$ is the equilibrium policy
- $\mu(\pi^*)$ is the population distribution induced by $\pi^*$
- BestResponse computes $\arg\max_a Q(s, a, m)$

---

## Example: Congestion Game

**Setup**: Agents choose routes in a network, with costs depending on congestion.

**State**: $s$ = current location
**Action**: $a$ = next location
**Population**: $m(s)$ = fraction of agents at each location
**Reward**: $r(s, a, m) = -\text{travel\_time}(a, m(a))$ (depends on congestion)

**Nash Equilibrium**: Each agent chooses route minimizing travel time given others' choices.

**Nash Q-Learning**: Learns route choices that account for congestion caused by population distribution.

---

## Implementation Roadmap

### Phase 1: Symmetric MFG Nash Q-Learning
- Simplification: Nash = best response to mean field
- Architecture: Q(s, a, m) network
- Algorithm: Standard Q-learning with population state
- **Status**: Essentially what we have with Mean Field Q-Learning

### Phase 2: Heterogeneous Agents
- Multiple agent types with different Q-functions
- Nash solver for computing equilibrium strategies
- Type-dependent population distributions

### Phase 3: General Nash Q-Learning
- Full Nash equilibrium computation
- Support for non-symmetric games
- Advanced equilibrium concepts (correlated, coarse)

---

## References

**Nash Q-Learning**:
- Hu & Wellman (2003): "Nash Q-Learning for General-Sum Stochastic Games"
- Littman (1994): "Markov games as a framework for multi-agent reinforcement learning"

**Mean Field Games**:
- Lasry & Lions (2007): "Mean field games"
- Carmona & Delarue (2018): "Probabilistic Theory of Mean Field Games"

**Mean Field RL**:
- Yang et al. (2018): "Mean Field Multi-Agent Reinforcement Learning"
- Guo et al. (2019): "Learning Mean-Field Games"
- Perrin et al. (2020): "Fictitious play for mean field games"

**Applications**:
- Traffic routing with congestion
- Economic market equilibria
- Multi-population dynamics
- Competitive resource allocation

---

**Status**: ✅ Theoretical foundation complete
**Next**: Design and implement Nash Q-Learning for MFG_PDE
**Key Insight**: For symmetric MFG, Nash Q-Learning simplifies to standard Q-learning with population state (which we already have!). The extension is mainly for heterogeneous/competitive scenarios.
