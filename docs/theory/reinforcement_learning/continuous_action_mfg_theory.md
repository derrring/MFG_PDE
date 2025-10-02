# Continuous Action Mean Field Games: Research Roadmap

**Date**: October 2025
**Status**: Research Planning Document
**Context**: Strategic roadmap for implementing continuous action spaces in MFG-RL algorithms

---

## Executive Summary

This document outlines a comprehensive research and implementation plan for extending MFG_PDE's reinforcement learning paradigm to **continuous action spaces**. Current implementations support discrete actions (|A| ≤ 20 optimal), but many real-world Mean Field Games require continuous control: crowd navigation with velocity control, price formation with continuous prices, resource allocation with continuous quantities.

**Strategic Goal**: Develop production-quality continuous action MFG-RL algorithms that maintain the same level of rigor, performance, and usability as our discrete implementations.

**Timeline**: 3-6 months for Phase 1-2, 6-12 months for complete implementation

**Expected Impact**:
- Unlock realistic MFG applications (continuous velocity, prices, resources)
- Enable high-dimensional action spaces (a ∈ ℝᵈ, d > 3)
- Bridge gap between classical MFG theory (continuous control) and RL practice

---

## 1. Theoretical Foundations

### 1.1 Mathematical Formulation

**Classical MFG with Continuous Actions**:

$$
\begin{cases}
-\partial_t u + H(x, \nabla_x u, m) = f(x, m) & \text{(HJB with continuous control)} \\
\partial_t m - \Delta m - \nabla \cdot (m \nabla_p H(x, \nabla_x u, m)) = 0 & \text{(Fokker-Planck)} \\
m(0, x) = m_0(x), \quad u(T, x) = g(x, m(T)) & \text{(Terminal conditions)}
\end{cases}
$$

**Hamiltonian with Continuous Control**:
$$
H(x, p, m) = \max_{a \in \mathcal{A}} \left\{ -p \cdot b(x, a, m) - L(x, a, m) \right\}
$$

where:
- $\mathcal{A} \subset \mathbb{R}^d$ is the continuous action space (typically compact, convex)
- $b(x, a, m)$ is the controlled drift
- $L(x, a, m)$ is the running cost
- Optimal control: $a^*(x, t) = \arg\max_a \{ -\nabla_x u(t,x) \cdot b(x,a,m) - L(x,a,m) \}$

**Key Theoretical Challenges**:
1. **Action space compactness**: How to enforce $a \in [a_{min}, a_{max}]$?
2. **Policy representation**: Deterministic vs stochastic policies?
3. **Exploration**: How to explore infinite action spaces efficiently?
4. **Convergence**: Do discrete-action convergence results extend to continuous?

### 1.2 Connection to Classical MFG Literature

**Continuous Control MFG Theory** (established results):

1. **Existence/Uniqueness**: Lasry-Lions (2007) - continuous control MFG with monotonicity
2. **Viscosity Solutions**: Cardaliaguet et al. (2015) - HJB-FP with continuous actions
3. **Optimal Transport**: Benamou-Brenier formulation with continuous velocities
4. **Price Formation**: Cardaliaguet & Lehalle (2018) - continuous price selection MFG

**Gap**: Classical theory assumes continuous actions, but **existing MFG-RL implementations use discrete actions**. This research bridges that gap.

### 1.3 Mean Field Reinforcement Learning with Continuous Actions

**Standard RL Formulation**:
- State: $s_t \in \mathcal{S}$
- Action: $a_t \in \mathcal{A} \subset \mathbb{R}^d$ (continuous)
- Population state: $m_t \in \mathcal{P}(\mathcal{S})$
- Reward: $r_t = r(s_t, a_t, m_t)$
- Transition: $s_{t+1} \sim P(\cdot | s_t, a_t, m_t)$

**Objective**: Find policy $\pi(a|s, m)$ maximizing
$$
J(\pi) = \mathbb{E}\left[ \sum_{t=0}^T \gamma^t r(s_t, a_t, m_t) \right], \quad m_t = \mu_t(\pi)
$$

**Mean Field Consistency**: Population distribution $m_t$ is induced by the representative agent's policy $\pi$.

---

## 2. Algorithmic Approaches

### 2.1 Deep Deterministic Policy Gradient (DDPG) for MFG

**Why DDPG**:
- Deterministic policy $\mu(s, m) \to a \in \mathbb{R}^d$ (direct continuous action output)
- Critic Q(s, a, m) with action as input (handles continuous actions naturally)
- Actor-Critic framework (aligns with MFG theory)
- Off-policy learning (efficient sample reuse)

**Architecture**:

```python
class MeanFieldDDPG:
    """
    Deep Deterministic Policy Gradient for Mean Field Games.

    Components:
    - Actor: μ(s, m) → a ∈ ℝᵈ (deterministic policy)
    - Critic: Q(s, a, m) → ℝ (action-value function)
    - Target networks: μ'(s, m), Q'(s, a, m)
    - Replay buffer: (s, a, r, s', m, m')
    """
```

**Key Differences from Standard DDPG**:
1. **Population state**: Both actor and critic take $m$ as input
2. **Population update**: After each episode, update $m$ based on agent trajectories
3. **Mean field consistency**: Iterate until $m = \mu(\pi)$ (population matches policy)

**Training Loop**:
1. Sample action: $a = \mu(s, m) + \mathcal{N}(0, \sigma)$ (exploration noise)
2. Execute action, observe $(s', r, m')$
3. Update critic: minimize $\mathcal{L}_Q = (Q(s,a,m) - y)^2$, $y = r + \gamma Q'(s', \mu'(s', m'), m')$
4. Update actor: maximize $Q(s, \mu(s, m), m)$ via policy gradient
5. Update population: $m \leftarrow \text{aggregate}(\{s_i^t\})$

**Challenges**:
- **Exploration**: Noise strategy (Gaussian, OU process, parameter space noise)
- **Population estimation**: How to aggregate agent states into $m$?
- **Convergence**: Coupled optimization (policy + population)

### 2.2 Twin Delayed DDPG (TD3) for MFG

**Improvements over DDPG**:
1. **Twin Critics**: Two Q-networks, use minimum for target (reduces overestimation)
2. **Delayed Policy Updates**: Update actor less frequently than critic (stability)
3. **Target Policy Smoothing**: Add noise to target actions (regularization)

**Relevance to MFG**:
- Better stability for coupled policy-population optimization
- Reduces Q-value overestimation (critical for MFG equilibrium)

**Implementation Priority**: ⭐⭐⭐⭐ High (should be implemented alongside DDPG)

### 2.3 Soft Actor-Critic (SAC) for MFG

**Why SAC**:
- **Stochastic policy**: $\pi(a|s, m) = \mathcal{N}(\mu(s,m), \sigma(s,m))$ (better exploration)
- **Maximum entropy**: Maximize $J(\pi) + \alpha \mathcal{H}(\pi)$ (robust policies)
- **Automatic temperature tuning**: Adaptive entropy coefficient $\alpha$
- **Off-policy**: Sample efficient

**Architecture**:

```python
class MeanFieldSAC:
    """
    Soft Actor-Critic for Mean Field Games.

    Components:
    - Actor: π(·|s, m) with Gaussian policy
      - Mean head: μ(s, m) → ℝᵈ
      - Log-std head: log σ(s, m) → ℝᵈ
    - Twin Critics: Q₁(s, a, m), Q₂(s, a, m)
    - Target Critics: Q₁'(s, a, m), Q₂'(s, a, m)
    - Entropy coefficient: α (learnable)
    """
```

**Training**:
1. Sample action: $a \sim \pi(\cdot|s, m)$ (reparameterization trick)
2. Update critics: $\mathcal{L}_Q = (Q(s,a,m) - y)^2$, $y = r + \gamma (V'(s', m') - \alpha \log \pi(a|s, m))$
3. Update actor: maximize $Q(s, a, m) - \alpha \log \pi(a|s, m)$
4. Update $\alpha$: $\mathcal{L}_\alpha = -\alpha (\log \pi(a|s,m) + \bar{\mathcal{H}})$

**Advantages for MFG**:
- Stochastic policy → better exploration of action space
- Entropy regularization → robust to population distribution uncertainty
- Automatic tuning → less hyperparameter sensitivity

**Implementation Priority**: ⭐⭐⭐⭐⭐ Very High (recommended primary algorithm)

### 2.4 Proximal Policy Optimization (PPO) for Continuous Actions

**Current Status**: MFG_PDE has discrete-action PPO-style Actor-Critic

**Extension to Continuous**:
- Replace categorical policy with Gaussian: $\pi(a|s,m) = \mathcal{N}(\mu(s,m), \sigma)$
- Clipped surrogate objective: $\mathcal{L}^{CLIP}(\theta) = \min(r_t(\theta) A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)$
- Advantage estimation: GAE (already implemented)

**Pros**:
- On-policy (simpler population consistency)
- Stable updates (clipping prevents large policy changes)
- Leverages existing MFG Actor-Critic codebase

**Cons**:
- On-policy → sample inefficient (need more environment interactions)
- Requires good population estimation each iteration

**Implementation Priority**: ⭐⭐⭐ Medium (good for comparison, less sample efficient)

---

## 3. Implementation Plan

### Phase 1: Core Infrastructure (2-3 weeks)

**Goal**: Build foundational components for continuous action support

#### 3.1.1 Continuous Action Networks

**Tasks**:
- [ ] Implement `MeanFieldContinuousQNetwork(state_dim, action_dim, population_dim)`
  - Action encoder: $a \to \text{features}$
  - Fusion layer: concat(state_feat, action_feat, pop_feat) → scalar Q-value
  - Support different architectures (MLP, residual, attention)

- [ ] Implement `MeanFieldContinuousActor(state_dim, action_dim, population_dim)`
  - Deterministic version: $\mu(s, m) \to a \in \mathbb{R}^d$
  - Stochastic version: $\mu(s, m), \sigma(s, m)$ for Gaussian policy
  - Action space bounds: tanh squashing for $a \in [a_{min}, a_{max}]$

**Files**:
- `mfg_pde/alg/reinforcement/networks/continuous_q_network.py`
- `mfg_pde/alg/reinforcement/networks/continuous_actor.py`

**Tests**:
- [ ] Unit tests: forward pass shapes, gradient flow
- [ ] Integration test: actor output within bounds
- [ ] Benchmark: inference speed vs discrete networks

#### 3.1.2 Continuous Action Environments

**Tasks**:
- [ ] Extend `MFGMazeEnvironment` to continuous velocity control
  - Action space: $a \in [-v_{max}, v_{max}]^2$ (2D velocity)
  - Dynamics: $x_{t+1} = x_t + \Delta t \cdot a_t$ (Euler integration)
  - Obstacles: collision detection and response

- [ ] Create `ContinuousPriceMFGEnvironment`
  - Action: price $p \in [p_{min}, p_{max}]$
  - Reward: profit based on price and population distribution
  - Population: distribution of agent prices

- [ ] Create `ResourceAllocationMFGEnvironment`
  - Action: resource allocation $a \in \Delta^n$ (simplex)
  - Reward: utility based on allocation and population strategy
  - Population: distribution of resource allocations

**Files**:
- `mfg_pde/alg/reinforcement/environments/continuous_maze_env.py`
- `mfg_pde/alg/reinforcement/environments/price_formation_env.py`
- `mfg_pde/alg/reinforcement/environments/resource_allocation_env.py`

**Tests**:
- [ ] Gymnasium API compliance (reset, step, render)
- [ ] Action space validation (bounds enforcement)
- [ ] Population state computation

#### 3.1.3 Population State Estimation

**Challenge**: How to represent population distribution $m$ from finite agent trajectories?

**Approaches**:

1. **Grid-based density** (current approach for discrete state spaces):
   - Discretize state space into grid
   - Count agents per cell: $m(x) \approx \frac{1}{N} \sum_{i=1}^N \mathbb{1}[x_i \in \text{cell}(x)]$
   - Normalize to probability distribution

2. **Kernel Density Estimation**:
   - Smooth density: $m(x) = \frac{1}{N} \sum_{i=1}^N K_h(x - x_i)$
   - Kernel $K_h$ (Gaussian, Epanechnikov)
   - Bandwidth $h$ (learnable or fixed)

3. **Neural Density Estimation**:
   - Train density network: $m_\phi(x)$ to fit agent distribution
   - Use normalizing flows or energy-based models
   - More flexible, handles high-dimensional states

**Recommendation**: Start with grid-based (simple), add KDE for continuous state spaces

**Tasks**:
- [ ] Implement `PopulationStateEstimator` base class
- [ ] Implement `GridDensityEstimator` (current method)
- [ ] Implement `KernelDensityEstimator` (for continuous states)
- [ ] Add population state to observation: $\text{obs} = (s, m)$

**Files**:
- `mfg_pde/alg/reinforcement/population/estimators.py`

### Phase 2: DDPG for MFG (3-4 weeks)

**Goal**: Implement production-quality DDPG algorithm for continuous action MFG

#### 3.2.1 Core DDPG Algorithm

**Tasks**:
- [ ] Implement `MeanFieldDDPG` algorithm class
  - Actor network: $\mu_\theta(s, m)$
  - Critic network: $Q_\phi(s, a, m)$
  - Target networks: $\mu_{\theta'}, Q_{\phi'}$ with soft updates
  - Replay buffer: $(s, a, r, s', m, m')$ transitions
  - Ornstein-Uhlenbeck noise for exploration

- [ ] Training loop:
  - Episode collection with noise
  - Critic update: minimize TD error
  - Actor update: policy gradient $\nabla_\theta J = \mathbb{E}[\nabla_a Q(s,a,m) \nabla_\theta \mu_\theta(s,m)]$
  - Target network updates: $\theta' \leftarrow \tau \theta + (1-\tau) \theta'$
  - Population update: aggregate agent states → $m$

- [ ] Mean field consistency loop:
  - Train policy with current population $m^{(k)}$
  - Update population: $m^{(k+1)} = \mu(\pi^{(k)})$
  - Check convergence: $\|m^{(k+1)} - m^{(k)}\| < \epsilon$

**Files**:
- `mfg_pde/alg/reinforcement/algorithms/mean_field_ddpg.py`

**Hyperparameters** (defaults):
```python
config = {
    "actor_lr": 1e-4,
    "critic_lr": 1e-3,
    "gamma": 0.99,
    "tau": 0.005,  # Soft update rate
    "buffer_size": 1e6,
    "batch_size": 256,
    "noise_std": 0.1,  # Exploration noise
    "noise_clip": 0.5,
    "population_update_freq": 100,  # Episodes between population updates
}
```

#### 3.2.2 TD3 Extensions

**Tasks**:
- [ ] Implement `MeanFieldTD3` (extends DDPG)
  - Twin critics: $Q_{\phi_1}(s,a,m), Q_{\phi_2}(s,a,m)$
  - Target policy smoothing: $a' = \mu'(s',m') + \text{clip}(\epsilon, -c, c)$
  - Delayed actor updates: update $\mu$ every $d$ critic updates

**Files**:
- `mfg_pde/alg/reinforcement/algorithms/mean_field_td3.py`

#### 3.2.3 Testing and Validation

**Tasks**:
- [ ] Unit tests: network shapes, gradient computation, target updates
- [ ] Integration tests: full training loop on simple environment
- [ ] Convergence test: verify mean field consistency $m = \mu(\pi)$
- [ ] Benchmark: compare DDPG vs TD3 on continuous maze

**Validation Criteria**:
- Training stability (no NaN, no divergence)
- Convergence to equilibrium (population consistency)
- Performance: achieve >80% success rate on continuous maze
- Speed: train 1000 episodes in <10 minutes (on CPU)

### Phase 3: SAC for MFG (3-4 weeks)

**Goal**: Implement Soft Actor-Critic for robust continuous action MFG

#### 3.3.1 Core SAC Algorithm

**Tasks**:
- [ ] Implement `MeanFieldSAC` algorithm class
  - Stochastic actor: Gaussian policy $\pi(\cdot|s,m)$
  - Twin critics: $Q_{\phi_1}(s,a,m), Q_{\phi_2}(s,a,m)$
  - Entropy coefficient: $\alpha$ (learnable via dual gradient descent)
  - Reparameterization trick for policy gradients

- [ ] Training loop:
  - Sample actions: $a \sim \pi(\cdot|s,m)$ using reparameterization
  - Critic update: $\mathcal{L}_Q = (Q(s,a,m) - (r + \gamma V'(s',m')))^2$
  - Actor update: $\nabla_\theta \mathbb{E}[Q(s, a_\theta(s,m), m) - \alpha \log \pi(a_\theta|s,m)]$
  - Temperature update: $\mathcal{L}_\alpha = -\alpha (\mathbb{E}[\log \pi(a|s,m)] + \bar{\mathcal{H}})$

**Files**:
- `mfg_pde/alg/reinforcement/algorithms/mean_field_sac.py`

**Hyperparameters**:
```python
config = {
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "alpha_lr": 3e-4,  # Entropy coefficient learning rate
    "gamma": 0.99,
    "tau": 0.005,
    "buffer_size": 1e6,
    "batch_size": 256,
    "target_entropy": -action_dim,  # Heuristic: -dim(A)
}
```

#### 3.3.2 Advanced Features

**Tasks**:
- [ ] Automatic entropy tuning (learnable $\alpha$)
- [ ] Squashed Gaussian policy (tanh squashing for bounded actions)
- [ ] Log-probability computation with tanh correction
- [ ] Population-aware value function: $V(s, m)$ as baseline

**References**:
- Haarnoja et al. (2018): "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
- Haarnoja et al. (2018): "Soft Actor-Critic Algorithms and Applications"

### Phase 4: Continuous PPO Extension (2-3 weeks)

**Goal**: Extend existing MFG Actor-Critic to continuous actions

#### 3.4.1 Gaussian Policy for PPO

**Tasks**:
- [ ] Modify `ActorNetwork` to output Gaussian parameters
  - Mean head: $\mu(s, m) \in \mathbb{R}^d$
  - Log-std head: $\log \sigma(s, m) \in \mathbb{R}^d$ OR fixed $\log \sigma$

- [ ] Update policy gradient computation:
  - Sample action: $a = \mu(s,m) + \sigma(s,m) \odot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$
  - Log-probability: $\log \pi(a|s,m) = -\frac{1}{2}\left(\frac{(a-\mu)^2}{\sigma^2} + \log(2\pi\sigma^2)\right)$
  - Clipped objective: $\mathcal{L}^{CLIP}(\theta) = \min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)$

**Files**:
- `mfg_pde/alg/reinforcement/algorithms/mean_field_ppo_continuous.py`

#### 3.4.2 Comparison with Discrete PPO

**Tasks**:
- [ ] Benchmark: continuous vs discrete PPO on maze navigation
- [ ] Ablation: fixed vs learned std, different clipping values
- [ ] Document: when to use PPO vs DDPG/SAC for MFG

### Phase 5: Applications and Examples (4-6 weeks)

**Goal**: Demonstrate continuous action MFG on realistic problems

#### 3.5.1 Crowd Navigation with Velocity Control

**Scenario**: Agents navigate 2D environment with continuous velocity

**Environment**:
- State: $(x, y, v_x, v_y, \text{goal}_x, \text{goal}_y)$
- Action: acceleration $(a_x, a_y) \in [-a_{max}, a_{max}]^2$
- Dynamics: $v_{t+1} = v_t + \Delta t \cdot a_t$, $x_{t+1} = x_t + \Delta t \cdot v_t$
- Reward: $r = -\|x - \text{goal}\| - \lambda \|v\|^2 - \mu \cdot \text{collision\_penalty}$
- Population: density $m(x, y)$ of other agents

**Tasks**:
- [ ] Implement `ContinuousCrowdNavigationEnv`
- [ ] Train SAC agent with population feedback
- [ ] Visualize: trajectories, velocity fields, density evolution
- [ ] Compare: continuous vs discrete (4-direction) navigation

**Expected Result**: Smoother trajectories, better obstacle avoidance

#### 3.5.2 Price Formation Game

**Scenario**: Sellers choose continuous prices in competitive market

**Environment**:
- State: $(cost, quality, market\_condition, \text{price\_history})$
- Action: price $p \in [p_{min}, p_{max}]$
- Reward: profit $r = (p - cost) \cdot demand(p, m)$
- Demand function: $demand(p, m) = D_0 \cdot e^{-\beta(p - \bar{p}(m))}$ (price-sensitive)
- Population: distribution $m(p)$ of competitor prices

**Tasks**:
- [ ] Implement `PriceFormationMFGEnv`
- [ ] Train DDPG/SAC agents
- [ ] Analyze: Nash equilibrium prices, market efficiency
- [ ] Compare: MFG equilibrium vs competitive equilibrium (economics)

**References**:
- Cardaliaguet & Lehalle (2018): "Mean Field Game of Controls and Model Uncertainty"
- Gomes & Saúde (2021): "Mean Field Games Models—A Brief Survey"

#### 3.5.3 Resource Allocation Game

**Scenario**: Agents allocate continuous resources across multiple options

**Environment**:
- State: $(resource\_level, option\_values, \text{history})$
- Action: allocation $a \in \Delta^n$ (n-dimensional simplex, $\sum a_i = 1$, $a_i \geq 0$)
- Reward: $r = \sum_i a_i \cdot value_i(m)$ (value depends on population allocation)
- Population: distribution $m(a)$ of other agents' allocations

**Tasks**:
- [ ] Implement `ResourceAllocationMFGEnv`
- [ ] Simplex projection layer for actor network
- [ ] Train SAC with simplex constraints
- [ ] Analyze: equilibrium allocations, social welfare

**Challenge**: Constrained action space (simplex) requires special handling

**Solution**:
- Softmax activation for actor output: $a = \text{softmax}(\mu(s, m))$
- OR projected gradient: $a = \Pi_{\Delta^n}(\mu(s, m))$

---

## 4. Technical Challenges and Solutions

### 4.1 Exploration in Continuous Action Spaces

**Problem**: Infinite action space → how to explore efficiently?

**Solutions**:

1. **Gaussian Noise** (DDPG):
   - $a = \mu(s, m) + \mathcal{N}(0, \sigma I)$
   - Simple, but uncorrelated noise
   - Decay $\sigma$ over time

2. **Ornstein-Uhlenbeck Noise** (DDPG):
   - Temporally correlated noise: $dn_t = \theta(\mu_n - n_t)dt + \sigma dW_t$
   - Better for physical systems (momentum)
   - Hyperparameters: $\theta$ (mean reversion), $\mu_n$ (long-term mean), $\sigma$ (volatility)

3. **Parameter Space Noise**:
   - Add noise to actor parameters, not actions
   - More consistent exploration
   - Computationally expensive

4. **Entropy Regularization** (SAC):
   - Maximize $J(\pi) + \alpha \mathcal{H}(\pi)$
   - Automatic exploration via stochastic policy
   - Adaptive entropy coefficient $\alpha$

**Recommendation**: SAC (entropy regularization) for best exploration-exploitation trade-off

### 4.2 Population State Estimation for Continuous States

**Problem**: How to represent $m(x)$ when $x \in \mathbb{R}^d$ (continuous state space)?

**Current Approach** (discrete states):
- Grid discretization: $m[i,j] = \#\text{agents in cell }(i,j)$
- Works for low-dimensional discrete states (e.g., maze grid)

**Extensions for Continuous States**:

1. **Kernel Density Estimation (KDE)**:
   ```python
   def estimate_population(agent_states, bandwidth=0.1):
       """
       KDE: m(x) = (1/N) Σᵢ K_h(x - xᵢ)
       """
       from scipy.stats import gaussian_kde
       kde = gaussian_kde(agent_states.T, bw_method=bandwidth)
       return kde  # Can evaluate at any x: m(x) = kde(x)
   ```
   - **Pros**: Smooth density, works for continuous states
   - **Cons**: Bandwidth selection, expensive for high-D

2. **Histogram with Adaptive Binning**:
   - Use adaptive bins (more bins in high-density regions)
   - OR use k-d tree partitioning
   - **Pros**: Faster than KDE
   - **Cons**: Still discretizes state space

3. **Neural Density Network**:
   - Train neural network $m_\phi(x)$ to match agent distribution
   - Use maximum likelihood: $\max_\phi \frac{1}{N} \sum_i \log m_\phi(x_i)$
   - Normalization: use normalizing flows or energy-based models
   - **Pros**: Flexible, handles high-D
   - **Cons**: Complex, requires training

4. **Finite Agent Representation**:
   - Don't estimate $m(x)$ explicitly
   - Input to network: states of $k$-nearest neighbors
   - **Pros**: Simple, exact for small populations
   - **Cons**: Scales poorly with population size

**Recommendation**:
- Start with KDE (simple, works for continuous states)
- For high-D or large populations, use k-nearest neighbors
- Long-term: neural density estimation (research direction)

**Implementation**:
```python
class PopulationStateEstimator(ABC):
    @abstractmethod
    def estimate(self, agent_states: np.ndarray) -> Callable:
        """Return density function m(x)."""
        pass

class KDEPopulationEstimator(PopulationStateEstimator):
    def __init__(self, bandwidth: float = 0.1):
        self.bandwidth = bandwidth

    def estimate(self, agent_states: np.ndarray) -> Callable:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(agent_states.T, bw_method=self.bandwidth)
        return lambda x: kde(x.T)

class KNNPopulationEstimator(PopulationStateEstimator):
    def __init__(self, k: int = 10):
        self.k = k

    def estimate(self, agent_states: np.ndarray) -> np.ndarray:
        """Return k-nearest neighbor states for each agent."""
        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=self.k)
        knn.fit(agent_states)
        _, indices = knn.kneighbors(agent_states)
        return agent_states[indices]  # [N, k, state_dim]
```

### 4.3 Action Space Constraints

**Problem**: How to enforce $a \in \mathcal{A}$ where $\mathcal{A}$ has constraints?

**Common Constraint Types**:

1. **Box Constraints**: $a \in [a_{min}, a_{max}]^d$
   - **Solution**: Tanh activation
   ```python
   def forward(self, state, population):
       features = self.encode(state, population)
       action_normalized = torch.tanh(self.policy_head(features))  # [-1, 1]
       action = action_min + (action_max - action_min) * (action_normalized + 1) / 2
       return action
   ```

2. **Simplex Constraints**: $a \in \Delta^n$ (sum to 1, non-negative)
   - **Solution**: Softmax activation
   ```python
   def forward(self, state, population):
       logits = self.policy_head(self.encode(state, population))
       action = F.softmax(logits, dim=-1)  # Σ aᵢ = 1, aᵢ ≥ 0
       return action
   ```

3. **Norm Constraints**: $\|a\| \leq c$
   - **Solution**: Normalize output
   ```python
   def forward(self, state, population):
       action_raw = self.policy_head(self.encode(state, population))
       action_norm = torch.norm(action_raw, dim=-1, keepdim=True)
       action = action_raw / action_norm * min(action_norm, c)  # Clip norm
       return action
   ```

4. **Linear Constraints**: $Aa \leq b$
   - **Solution**: Projected gradient descent
   ```python
   def project_to_constraint(action, A, b):
       """Project action onto constraint set {a : Aa ≤ b}."""
       # Use quadratic programming or projected gradient
       from scipy.optimize import minimize
       result = minimize(
           lambda a_proj: np.sum((a_proj - action.numpy())**2),
           x0=action.numpy(),
           constraints={'type': 'ineq', 'fun': lambda a: b - A @ a}
       )
       return torch.tensor(result.x)
   ```

**Recommendation**:
- Box constraints: Tanh activation (standard)
- Simplex: Softmax
- Complex constraints: Differentiable projection layers OR constrained optimization

### 4.4 Mean Field Consistency and Convergence

**Problem**: How to ensure population $m$ matches policy $\pi$?

**Iterative Fixed-Point Algorithm**:
1. Initialize population: $m^{(0)} = m_0$ (e.g., uniform)
2. For $k = 0, 1, 2, \ldots$:
   - Train policy: $\pi^{(k)} = \arg\max_\pi J(\pi | m^{(k)})$
   - Update population: $m^{(k+1)} = \mu(\pi^{(k)})$ (simulate agents)
   - Check convergence: if $\|m^{(k+1)} - m^{(k)}\| < \epsilon$, stop

**Convergence Issues**:
- **Oscillation**: Population and policy oscillate, never converge
- **Slow convergence**: Takes many iterations

**Solutions**:

1. **Damped Update**:
   ```python
   m_new = alpha * m_new + (1 - alpha) * m_old  # alpha ∈ (0, 1)
   ```
   - Slows down population changes
   - Improves stability

2. **Rolling Average**:
   ```python
   m_avg = (1 - beta) * m_avg + beta * m_new  # Exponential moving average
   ```

3. **Fictitious Play**:
   - Agents best-respond to historical average of population
   - $m^{(k)} = \frac{1}{k} \sum_{i=0}^{k-1} \mu(\pi^{(i)})$

4. **Online Learning**:
   - Update policy continuously with current population estimate
   - No separate population update step
   - Use replay buffer with recent population states

**Recommendation**: Start with damped update (simple, effective)

**Convergence Metrics**:
- Population distance: $\|m^{(k+1)} - m^{(k)}\|_1$ (L1 norm)
- Policy distance: $\|\pi^{(k+1)} - \pi^{(k)}\|$ (KL divergence)
- Reward stability: variance of episode rewards

**Implementation**:
```python
class MeanFieldConsistency:
    def __init__(self, damping: float = 0.1):
        self.damping = damping
        self.population_history = []

    def update_population(self, new_population):
        if len(self.population_history) == 0:
            self.current_population = new_population
        else:
            # Damped update
            self.current_population = (
                self.damping * new_population +
                (1 - self.damping) * self.current_population
            )
        self.population_history.append(self.current_population)

    def check_convergence(self, tolerance=1e-3):
        if len(self.population_history) < 2:
            return False
        diff = np.linalg.norm(
            self.population_history[-1] - self.population_history[-2], ord=1
        )
        return diff < tolerance
```

---

## 5. Testing and Validation Strategy

### 5.1 Unit Testing

**Network Tests**:
- [ ] Forward pass: correct output shapes
- [ ] Backward pass: gradients flow correctly
- [ ] Action bounds: outputs respect constraints
- [ ] Numerical stability: no NaN/Inf

**Algorithm Tests**:
- [ ] Replay buffer: correct sampling
- [ ] Target updates: soft update formula
- [ ] Population estimation: density integrates to 1
- [ ] Policy evaluation: Q-values consistent

### 5.2 Integration Testing

**End-to-End Training**:
- [ ] Train on simple environment (e.g., 1D navigation)
- [ ] Check convergence: rewards increase, losses decrease
- [ ] Population consistency: $\|m - \mu(\pi)\| \to 0$
- [ ] No crashes: training completes without errors

**Ablation Studies**:
- [ ] Compare: DDPG vs TD3 vs SAC vs PPO
- [ ] Hyperparameter sensitivity: learning rates, noise, batch size
- [ ] Population update frequency: every episode vs every N episodes
- [ ] Exploration strategy: Gaussian vs OU noise vs entropy

### 5.3 Benchmark Problems

**Test Suite**:

1. **1D Navigation** (sanity check):
   - State: $x \in [0, 1]$, goal at $x = 1$
   - Action: velocity $v \in [-v_{max}, v_{max}]$
   - Dynamics: $x_{t+1} = x_t + \Delta t \cdot v_t$
   - Reward: $r = -|x - 1|$
   - Population: density $m(x)$
   - **Success**: Reach goal in <100 steps

2. **2D Continuous Maze**:
   - State: $(x, y)$ position
   - Action: velocity $(v_x, v_y)$
   - Obstacles: collision penalties
   - **Success**: >80% agents reach goal

3. **Price Formation**:
   - Action: price $p \in [0, 10]$
   - Reward: profit based on demand
   - **Success**: Converge to Nash equilibrium price

4. **Crowd Avoidance**:
   - State: position + velocity
   - Action: acceleration
   - Reward: reach goal while avoiding others
   - **Success**: Low collision rate, smooth trajectories

### 5.4 Validation Metrics

**Performance**:
- Average episode reward
- Success rate (reach goal)
- Convergence speed (episodes to convergence)

**Quality**:
- Trajectory smoothness (for continuous control)
- Action variance (exploration vs exploitation)
- Q-value accuracy (TD error)

**Mean Field Consistency**:
- Population distance: $\|m^{(k)} - m^{(k-1)}\|$
- Equilibrium gap: $|J(\pi | m) - J(\pi | \mu(\pi))|$

**Computational**:
- Training time (wall-clock)
- Memory usage
- Inference speed (actions/second)

---

## 6. Documentation and Knowledge Transfer

### 6.1 Theoretical Documentation

**Create**:
- [ ] `docs/theory/continuous_action_mfg_formulation.md`
  - Mathematical foundations
  - HJB-FP with continuous control
  - Connection to classical MFG literature

- [ ] `docs/theory/continuous_action_rl_algorithms.md`
  - DDPG, TD3, SAC formulations
  - Mean field extensions
  - Convergence analysis

### 6.2 Implementation Documentation

**Create**:
- [ ] `docs/user/continuous_action_guide.md`
  - User guide: how to use continuous action algorithms
  - Environment creation
  - Training examples

- [ ] `docs/development/continuous_action_implementation.md`
  - Developer guide: architecture decisions
  - Network designs
  - Population estimation strategies

### 6.3 Examples and Tutorials

**Create**:
- [ ] `examples/basic/continuous_action_demo.py`
  - Simple 1D/2D navigation
  - Train DDPG/SAC
  - Visualize results

- [ ] `examples/advanced/continuous_crowd_navigation.py`
  - Full crowd navigation scenario
  - Multiple agents with velocity control
  - Population density visualization

- [ ] `examples/advanced/price_formation_game.py`
  - Price formation MFG
  - Nash equilibrium analysis
  - Economic interpretation

### 6.4 Jupyter Notebooks

**Create**:
- [ ] `examples/notebooks/continuous_action_intro.ipynb`
  - Interactive introduction
  - Compare discrete vs continuous
  - Visualize action spaces

- [ ] `examples/notebooks/sac_mfg_tutorial.ipynb`
  - Step-by-step SAC training
  - Hyperparameter tuning
  - Population consistency analysis

---

## 7. Research Directions and Extensions

### 7.1 High-Dimensional Action Spaces

**Challenge**: Current methods (DDPG, SAC) scale to ~20D actions, but some applications need 100+D

**Approaches**:

1. **Action Embeddings**:
   - Learn low-dimensional action representation
   - Policy operates in embedding space
   - Decoder maps embeddings → actual actions

2. **Hierarchical Actions**:
   - High-level policy: select action category
   - Low-level policy: continuous parameters
   - Example: navigation (discrete direction + continuous speed)

3. **Structured Actions**:
   - Exploit action space structure (e.g., symmetries, factorization)
   - Graph neural networks for structured action spaces

### 7.2 Multi-Modal Action Distributions

**Challenge**: Gaussian policies assume unimodal distributions, but some problems have multi-modal optimal actions

**Solutions**:

1. **Mixture of Gaussians**:
   - Policy: $\pi(a|s,m) = \sum_{i=1}^K w_i \mathcal{N}(\mu_i(s,m), \sigma_i(s,m))$
   - Mixture weights $w_i$ from softmax

2. **Normalizing Flows**:
   - Flexible distributions via invertible transformations
   - $\pi(a|s,m) = p_z(f^{-1}(a)) |\det \nabla_a f^{-1}(a)|$

3. **Energy-Based Policies**:
   - Implicit distribution: $\pi(a|s,m) \propto \exp(-E(a, s, m))$
   - Sampling via MCMC

### 7.3 Model-Based MFG with Continuous Actions

**Idea**: Learn dynamics model $\hat{P}(s'|s, a, m)$ and use planning

**Approaches**:

1. **Dyna-style**:
   - Learn model from real experience
   - Generate synthetic experience via model
   - Train policy on mixed real+synthetic data

2. **Model Predictive Control**:
   - Plan actions online via optimization: $\max_{a_0, \ldots, a_H} \sum_t r(s_t, a_t, m_t)$
   - Use learned model for predictions

3. **World Models**:
   - Learn latent dynamics: $z_{t+1} = f(z_t, a_t, m_t)$
   - Train policy in latent space

**Challenges**:
- Modeling population dynamics: $m_{t+1} = g(m_t, a_1, \ldots, a_N)$
- Compounding errors in long rollouts

### 7.4 Distributed Multi-Agent Training

**Challenge**: Training with $N$ agents is slow for large $N$

**Solutions**:

1. **Parallelization**:
   - Simulate agents in parallel (GPU vectorization)
   - Asynchronous updates (A3C-style)

2. **Population Sampling**:
   - Don't simulate all $N$ agents
   - Sample $M \ll N$ agents, estimate population from sample

3. **Replay Buffer Sharing**:
   - Share experiences across agents
   - Use importance sampling for off-policy correction

### 7.5 Continuous Time MFG

**Challenge**: Current implementation uses discrete time steps $\Delta t$

**Continuous Time Formulation**:
$$
\begin{cases}
-\partial_t u + \nu \Delta u + H(\nabla_x u, m) = f(x, m) \\
\partial_t m - \nu \Delta m - \nabla \cdot (m \nabla_p H) = 0
\end{cases}
$$

**RL Approach**:
- Neural ODE for policy: $\frac{da}{dt} = \pi(t, s, m)$
- Continuous-time Q-function: $Q(t, s, a, m)$
- Training via adjoint method (backpropagation through ODE solver)

**References**:
- Ruthotto et al. (2020): "A machine learning framework for solving high-dimensional mean field game and mean field control problems"

---

## 8. Success Criteria and Milestones

### Phase 1: Infrastructure (Weeks 1-3)

**Deliverables**:
- [ ] `MeanFieldContinuousQNetwork` class
- [ ] `MeanFieldContinuousActor` class
- [ ] `ContinuousMazeEnvironment`
- [ ] Population state estimation (grid or KDE)
- [ ] Unit tests passing

**Success Criteria**:
- Networks: correct shapes, gradients flow
- Environment: Gymnasium API compliant
- Tests: 100% passing

### Phase 2: DDPG (Weeks 4-7)

**Deliverables**:
- [ ] `MeanFieldDDPG` algorithm
- [ ] `MeanFieldTD3` algorithm
- [ ] Training scripts and configs
- [ ] Example: continuous maze navigation

**Success Criteria**:
- Training stability: no divergence, no NaN
- Performance: >70% success rate on maze
- Convergence: population consistency within 5%

### Phase 3: SAC (Weeks 8-11)

**Deliverables**:
- [ ] `MeanFieldSAC` algorithm
- [ ] Entropy tuning implementation
- [ ] Comparison benchmark (DDPG vs SAC)

**Success Criteria**:
- Performance: >80% success rate on maze
- Sample efficiency: better than DDPG
- Robustness: less sensitive to hyperparameters

### Phase 4: PPO (Weeks 12-14)

**Deliverables**:
- [ ] `MeanFieldPPOContinuous` algorithm
- [ ] Gaussian policy implementation
- [ ] Comparison with discrete PPO

**Success Criteria**:
- Training stability: clipping prevents divergence
- Performance: competitive with SAC
- On-policy consistency: population updates each iteration

### Phase 5: Applications (Weeks 15-20)

**Deliverables**:
- [ ] Crowd navigation example
- [ ] Price formation example
- [ ] Resource allocation example
- [ ] Documentation and tutorials

**Success Criteria**:
- Real-world relevance: examples solve practical problems
- Visualization: clear plots of policies, populations, trajectories
- Reproducibility: examples run out-of-box with good results

---

## 9. Long-Term Research Vision

### 9.1 Integration with Classical MFG Solvers

**Goal**: Bridge RL and PDE-based MFG solvers

**Approach**:
- Use classical solver (HJB-FP) to initialize RL policy
- Use RL to refine solution in complex domains
- Hybrid: RL for local decisions, PDE for global equilibrium

**Benefits**:
- Leverage theoretical guarantees of PDE methods
- Handle complex scenarios where PDE methods fail (high-D, discontinuous)

### 9.2 Scalability to Large Populations

**Current**: Train with $N \sim 100$ agents

**Goal**: Scale to $N \sim 10^6$ agents (realistic crowd simulations)

**Challenges**:
- Computational: simulating $10^6$ agents is expensive
- Memory: storing states, actions, rewards for all agents

**Solutions**:
- Mean field approximation: represent population as distribution, not individual agents
- Sampling: simulate subset of agents, estimate population from sample
- GPU acceleration: parallel simulation on GPU

### 9.3 Theoretical Convergence Guarantees

**Question**: Do our RL algorithms converge to MFG equilibrium?

**Current Status**: Empirical convergence observed, but no theoretical proof

**Research Directions**:
- Prove convergence of DDPG/SAC to Nash equilibrium in MFG setting
- Characterize convergence rate: polynomial, exponential?
- Identify conditions for convergence: Lipschitz, monotonicity, etc.

**Related Work**:
- Perrin et al. (2020): "Fictitious play for mean field games: Continuous time analysis and applications"
- Guo et al. (2019): "Learning Mean-Field Games"

### 9.4 Real-World Applications

**Target Domains**:

1. **Autonomous Vehicles**:
   - Continuous action: steering angle, acceleration
   - Population: distribution of vehicle positions/velocities
   - Objective: navigate safely and efficiently

2. **Energy Markets**:
   - Continuous action: power production level
   - Population: distribution of producer strategies
   - Objective: maximize profit given market clearing price

3. **Pandemic Control**:
   - Continuous action: social distancing level, mobility
   - Population: distribution of infection states
   - Objective: minimize infections + economic cost

4. **Finance**:
   - Continuous action: trading volume, position size
   - Population: distribution of trader strategies
   - Objective: maximize profit accounting for market impact

---

## 10. Resource Requirements

### 10.1 Personnel

**Roles**:
- Research Scientist (you): Theory, algorithm design, implementation
- Research Engineer: Coding, testing, optimization
- Domain Expert: Application-specific knowledge (economics, robotics, etc.)

**Time Commitment**:
- Phase 1-4: ~20 hours/week (part-time)
- Phase 5: ~10 hours/week (maintenance + applications)

### 10.2 Computational Resources

**Development**:
- CPU: Modern multi-core processor (8+ cores)
- RAM: 16 GB minimum, 32 GB recommended
- GPU: Not required for development, helpful for large-scale experiments

**Production/Benchmarking**:
- GPU: NVIDIA RTX 3090 or better (for large populations)
- Cluster: Optional, for distributed training

**Estimated Costs**:
- Development: $0 (use existing hardware)
- Cloud GPU (optional): ~$1-2/hour for RTX 3090

### 10.3 Software Dependencies

**Core**:
- PyTorch: Neural networks
- Gymnasium: RL environment API
- NumPy, SciPy: Numerical computing

**Visualization**:
- Matplotlib, Plotly: Plotting
- Seaborn: Statistical visualization

**Population Estimation**:
- scikit-learn: KDE, k-NN
- PyTorch (normalizing flows): Advanced density estimation

**Testing**:
- pytest: Unit testing
- hypothesis: Property-based testing

---

## 11. Risk Assessment and Mitigation

### 11.1 Technical Risks

**Risk 1: Convergence Instability**
- **Probability**: Medium
- **Impact**: High (algorithms don't converge → unusable)
- **Mitigation**:
  - Implement multiple algorithms (DDPG, TD3, SAC)
  - Use damped population updates
  - Extensive hyperparameter tuning

**Risk 2: Population Estimation Errors**
- **Probability**: Medium
- **Impact**: Medium (inaccurate population → suboptimal policies)
- **Mitigation**:
  - Test multiple estimators (grid, KDE, k-NN)
  - Validate: check density integrates to 1, visualize

**Risk 3: Scalability Issues**
- **Probability**: Low
- **Impact**: Medium (slow training → impractical)
- **Mitigation**:
  - Profile code, optimize bottlenecks
  - Use vectorized operations (PyTorch, NumPy)
  - GPU acceleration if needed

### 11.2 Research Risks

**Risk 1: Theoretical Gaps**
- **Probability**: High
- **Impact**: Low (empirical success sufficient for now)
- **Mitigation**:
  - Document assumptions clearly
  - Empirical validation on benchmarks
  - Future: collaborate with theorists

**Risk 2: Limited Applicability**
- **Probability**: Low
- **Impact**: Medium (only works for specific problems)
- **Mitigation**:
  - Test on diverse benchmarks
  - Modular design (easy to adapt)

### 11.3 Project Risks

**Risk 1: Scope Creep**
- **Probability**: High
- **Impact**: Medium (delays, incomplete work)
- **Mitigation**:
  - Stick to phased plan
  - Prioritize core algorithms (SAC) over extensions
  - Defer advanced features (multi-modal, model-based) to Phase 2

**Risk 2: Lack of Validation**
- **Probability**: Low
- **Impact**: High (bugs in production code)
- **Mitigation**:
  - Comprehensive testing (unit, integration, benchmarks)
  - Code review
  - Continuous integration

---

## 12. Conclusion and Next Steps

### Summary

This roadmap outlines a comprehensive plan to extend MFG_PDE's RL paradigm to **continuous action spaces**, addressing a critical gap between classical MFG theory (which assumes continuous control) and current RL implementations (which use discrete actions).

**Key Deliverables**:
1. **Algorithms**: DDPG, TD3, SAC, continuous PPO for MFG
2. **Infrastructure**: Continuous action networks, environments, population estimators
3. **Applications**: Crowd navigation, price formation, resource allocation
4. **Documentation**: Theory, implementation guides, tutorials

**Timeline**: 3-6 months for core implementation (Phase 1-4), additional 2-3 months for applications (Phase 5)

**Impact**: Enable realistic MFG applications requiring continuous control, bringing MFG_PDE closer to practical deployment in robotics, economics, and social systems.

### Immediate Next Steps

**Week 1-2**:
1. **Implement continuous action networks** (`MeanFieldContinuousQNetwork`, `MeanFieldContinuousActor`)
2. **Create simple test environment** (1D navigation with continuous velocity)
3. **Set up unit tests** (network shapes, gradients, bounds)

**Week 3-4**:
4. **Implement DDPG algorithm** (`MeanFieldDDPG`)
5. **Train on simple environment** (validate convergence)
6. **Document progress** (update this roadmap with findings)

**Week 5+**:
7. **Extend to 2D continuous maze**
8. **Implement TD3 and SAC**
9. **Benchmark and compare algorithms**

### Open Questions for Future Research

1. **Theoretical**: Can we prove convergence of SAC to MFG Nash equilibrium?
2. **Computational**: How to scale to $N > 10^6$ agents efficiently?
3. **Algorithmic**: What's the best population estimator for high-D continuous states?
4. **Applied**: Which real-world MFG problems benefit most from continuous action RL?

---

**Document Status**: ✅ Research Planning Document
**Next Review**: After Phase 1 completion (Week 3)
**Maintainer**: Primary researcher
**Related Documents**:
- `RL_ACTION_SPACE_SCALABILITY_ANALYSIS.md` (current limitations)
- `continuous_action_architecture_sketch.py` (code examples)
- `STRATEGIC_DEVELOPMENT_ROADMAP_2026.md` (overall package roadmap)

---

## References

**Continuous Control RL**:
- Lillicrap et al. (2015): "Continuous control with deep reinforcement learning" (DDPG)
- Fujimoto et al. (2018): "Addressing Function Approximation Error in Actor-Critic Methods" (TD3)
- Haarnoja et al. (2018): "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL" (SAC)
- Schulman et al. (2017): "Proximal Policy Optimization Algorithms" (PPO)

**Mean Field Games**:
- Lasry & Lions (2007): "Mean field games" (foundational theory)
- Cardaliaguet (2013): "Notes on Mean Field Games" (lecture notes)
- Carmona & Delarue (2018): "Probabilistic Theory of Mean Field Games" (comprehensive textbook)

**Mean Field RL**:
- Yang et al. (2018): "Mean Field Multi-Agent Reinforcement Learning"
- Guo et al. (2019): "Learning Mean-Field Games"
- Perrin et al. (2020): "Fictitious play for mean field games: Continuous time analysis and applications"
- Laurière et al. (2022): "Scalable Deep Reinforcement Learning Algorithms for Mean Field Games"

**MFG with Continuous Actions**:
- Cardaliaguet & Lehalle (2018): "Mean Field Game of Controls and Model Uncertainty"
- Gomes & Saúde (2021): "Mean Field Games Models—A Brief Survey"
- Ruthotto et al. (2020): "A machine learning framework for solving high-dimensional MFG and MFC problems"

**Applications**:
- Lachapelle et al. (2016): "Efficiency of the price formation process in presence of high frequency participants" (finance)
- Achdou & Capuzzo-Dolcetta (2010): "Mean field games: numerical methods" (numerical methods)
- Burger et al. (2013): "A mixed finite element method for nonlinear diffusion equations" (crowd motion)
