# Deep Deterministic Policy Gradient for Mean Field Games

**Algorithm**: Mean Field DDPG
**Action Space**: Continuous a ∈ ℝᵈ
**Date**: October 2025
**Status**: Phase 3.3.1 Implementation

---

## 1. Mathematical Formulation

### 1.1 Mean Field Game with Continuous Actions

**State Space**: $s \in \mathcal{S}$ (individual agent state)
**Action Space**: $a \in \mathcal{A} \subset \mathbb{R}^d$ (continuous, typically compact)
**Population State**: $m \in \mathcal{P}(\mathcal{S})$ (population distribution)

**MFG System**:
```math
\begin{cases}
-\partial_t u + H(s, \nabla_s u, m) = f(s, m) & \text{(HJB)} \\
\partial_t m - \Delta m - \nabla \cdot (m \nabla_p H(s, \nabla_s u, m)) = 0 & \text{(Fokker-Planck)} \\
m(0,s) = m_0(s), \quad u(T,s) = g(s, m(T)) & \text{(Boundary conditions)}
\end{cases}
```

**Hamiltonian**:
$$
H(s, p, m) = \max_{a \in \mathcal{A}} \left\{ -p \cdot b(s, a, m) - L(s, a, m) \right\}
$$

**Optimal Control**:
$$
a^*(s, t) = \arg\max_{a \in \mathcal{A}} \left\{ -\nabla_s u(t,s) \cdot b(s,a,m(t)) - L(s,a,m(t)) \right\}
$$

### 1.2 Reinforcement Learning Formulation

**Objective**: Find deterministic policy $\mu: \mathcal{S} \times \mathcal{P}(\mathcal{S}) \to \mathcal{A}$ that maximizes:
$$
J(\mu) = \mathbb{E}\left[ \sum_{t=0}^T \gamma^t r(s_t, \mu(s_t, m_t), m_t) \mid m_t = \mu_t(\mu) \right]
$$

**Mean Field Consistency**: Population distribution $m_t$ is induced by the policy $\mu$:
$$
m_t(s) = \mathbb{P}(s_t = s \mid a_0 = \mu(s_0, m_0), \ldots, a_{t-1} = \mu(s_{t-1}, m_{t-1}))
$$

**Nash Equilibrium**: Policy $\mu^*$ is a Nash equilibrium if:
$$
J(\mu^*) \geq J(\mu) \quad \forall \mu, \text{ when } m = \mu(\mu^*)
$$

---

## 2. DDPG Algorithm for MFG

### 2.1 Actor-Critic Architecture

**Actor Network** (Deterministic Policy):
$$
\mu_\theta: \mathcal{S} \times \mathcal{P}(\mathcal{S}) \to \mathcal{A}
$$
- Input: Individual state $s \in \mathbb{R}^{d_s}$, population state $m \in \mathbb{R}^{d_m}$
- Output: Continuous action $a \in \mathbb{R}^d$ (with bounded support via tanh)

**Critic Network** (Action-Value Function):
$$
Q_\phi: \mathcal{S} \times \mathcal{A} \times \mathcal{P}(\mathcal{S}) \to \mathbb{R}
$$
- Input: State $s$, action $a$, population state $m$
- Output: Q-value $Q_\phi(s, a, m)$

### 2.2 Bellman Equation

**Q-Function**:
$$
Q^\mu(s, a, m) = \mathbb{E}\left[ r(s, a, m) + \gamma Q^\mu(s', \mu(s', m'), m') \mid s, a, m \right]
$$

**Optimality**:
$$
Q^*(s, a, m) = \mathbb{E}\left[ r(s, a, m) + \gamma \max_{a'} Q^*(s', a', m') \mid s, a, m \right]
$$

For deterministic policy:
$$
Q^{\mu^*}(s, a, m) = \mathbb{E}\left[ r(s, a, m) + \gamma Q^{\mu^*}(s', \mu^*(s', m'), m') \mid s, a, m \right]
$$

### 2.3 Policy Gradient

**Deterministic Policy Gradient Theorem** (Silver et al., 2014):
$$
\nabla_\theta J(\mu_\theta) = \mathbb{E}_{s \sim \rho^\mu, m \sim \mu^\mu}\left[ \nabla_\theta \mu_\theta(s, m) \nabla_a Q^\mu(s, a, m)\Big|_{a=\mu_\theta(s,m)} \right]
$$

where:
- $\rho^\mu$ is state visitation distribution under policy $\mu$
- $m \sim \mu^\mu$ is population distribution induced by $\mu$

**Key Insight**: Gradient flows through the deterministic policy into the critic:
$$
\nabla_\theta J \approx \nabla_\theta \mu_\theta(s, m) \cdot \nabla_a Q_\phi(s, \mu_\theta(s, m), m)
$$

---

## 3. Algorithm Details

### 3.1 Critic Update (TD Learning)

**Loss Function**:
$$
L(\phi) = \mathbb{E}_{(s,a,r,s',m,m') \sim \mathcal{D}}\left[ \left( Q_\phi(s, a, m) - y \right)^2 \right]
$$

**Target**:
$$
y = r + \gamma Q_{\phi'}(s', \mu_{\theta'}(s', m'), m')
$$

where $\phi'$ and $\theta'$ are target network parameters.

**Gradient**:
$$
\nabla_\phi L(\phi) = \mathbb{E}\left[ \left( Q_\phi(s,a,m) - y \right) \nabla_\phi Q_\phi(s,a,m) \right]
$$

### 3.2 Actor Update (Policy Gradient)

**Objective**: Maximize $J(\theta) = \mathbb{E}_{s,m}[Q_\phi(s, \mu_\theta(s,m), m)]$

**Gradient**:
$$
\nabla_\theta J(\theta) = \mathbb{E}_{s,m}\left[ \nabla_\theta \mu_\theta(s,m) \nabla_a Q_\phi(s,a,m)\Big|_{a=\mu_\theta(s,m)} \right]
$$

**Update Rule**:
$$
\theta \leftarrow \theta + \alpha_\theta \nabla_\theta J(\theta)
$$

### 3.3 Target Networks (Soft Updates)

**Polyak Averaging**:
$$
\begin{align}
\phi' &\leftarrow \tau \phi + (1-\tau) \phi' \\
\theta' &\leftarrow \tau \theta + (1-\tau) \theta'
\end{align}
$$

Typically $\tau \ll 1$ (e.g., 0.001) for stability.

### 3.4 Exploration Noise

**Ornstein-Uhlenbeck Process**:
$$
dX_t = \theta_{OU}(\mu_{OU} - X_t)dt + \sigma_{OU} dW_t
$$

**Policy with Exploration**:
$$
a_t = \mu_\theta(s_t, m_t) + \mathcal{N}_t
$$

where $\mathcal{N}_t$ is OU noise or Gaussian noise.

---

## 4. Mean Field Consistency

### 4.1 Population State Tracking

**Density Estimation**:
$$
m_t(s) \approx \frac{1}{N} \sum_{i=1}^N \mathbb{1}_{s_t^i = s}
$$

For continuous states, use kernel density estimation:
$$
m_t(s) \approx \frac{1}{N} \sum_{i=1}^N K_h(s - s_t^i)
$$

where $K_h$ is a smoothing kernel (e.g., Gaussian).

### 4.2 Population State Representation

**Histogram Representation**:
- Discretize state space into bins
- Compute normalized histogram: $m \in \mathbb{R}^{d_m}$, $\sum_i m_i = 1$

**Feature Representation**:
- Mean, variance, higher moments
- Distribution embedding via neural network

### 4.3 Nash Equilibrium Property

At equilibrium, policy $\mu^*$ satisfies:
$$
\mu^*(s, m^*) = \arg\max_{a \in \mathcal{A}} Q^{\mu^*}(s, a, m^*)
$$

where $m^* = \mu(\mu^*)$ is the fixed-point population distribution.

**Verification**: Check $\epsilon$-Nash gap:
$$
\epsilon = J(\mu^*) - \max_{\mu \neq \mu^*} J(\mu \mid m^*)
$$

---

## 5. Implementation Architecture

### 5.1 Actor Network

```python
class DDPGActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, population_dim: int,
                 action_bounds: tuple[float, float]):
        super().__init__()

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Population encoder
        self.pop_encoder = nn.Sequential(
            nn.Linear(population_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),  # Bounded actions
        )

        self.action_scale = (action_bounds[1] - action_bounds[0]) / 2.0
        self.action_bias = (action_bounds[1] + action_bounds[0]) / 2.0

    def forward(self, state, population_state):
        state_feat = self.state_encoder(state)
        pop_feat = self.pop_encoder(population_state)
        combined = torch.cat([state_feat, pop_feat], dim=1)
        action = self.action_head(combined)
        return action * self.action_scale + self.action_bias
```

### 5.2 Critic Network

```python
class DDPGCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, population_dim: int):
        super().__init__()

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
        )

        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
        )

        # Population encoder
        self.pop_encoder = nn.Sequential(
            nn.Linear(population_dim, 128),
            nn.ReLU(),
        )

        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(256 + 64 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state, action, population_state):
        state_feat = self.state_encoder(state)
        action_feat = self.action_encoder(action)
        pop_feat = self.pop_encoder(population_state)
        combined = torch.cat([state_feat, action_feat, pop_feat], dim=1)
        return self.q_head(combined).squeeze(-1)
```

---

## 6. Convergence Analysis

### 6.1 Theoretical Guarantees

**Assumption 1**: Critic converges to true Q-function: $Q_\phi \to Q^{\mu_\theta}$

**Assumption 2**: Actor converges to greedy policy: $\mu_\theta \to \arg\max_a Q^{\mu_\theta}(s, a, m)$

**Assumption 3**: Population distribution converges: $m_t \to m^*$

**Theorem** (Informal): Under Assumptions 1-3, DDPG converges to Nash equilibrium of MFG.

### 6.2 Practical Convergence

**Convergence Criteria**:
1. **Policy Stability**: $\|\mu_\theta - \mu_{\theta_{old}}\| < \epsilon_\mu$
2. **Value Stability**: $\|Q_\phi - Q_{\phi_{old}}\| < \epsilon_Q$
3. **Population Stability**: $\|m_t - m_{t-1}\| < \epsilon_m$
4. **Nash Gap**: $J(\mu^*) - \max_\mu J(\mu \mid m^*) < \epsilon_{Nash}$

**Typical Values**: $\epsilon_\mu = 10^{-3}$, $\epsilon_Q = 10^{-2}$, $\epsilon_m = 10^{-3}$, $\epsilon_{Nash} = 10^{-2}$

### 6.3 Hyperparameters

**Learning Rates**:
- Actor: $\alpha_\theta = 10^{-4}$ (smaller than critic)
- Critic: $\alpha_\phi = 10^{-3}$

**Target Update**: $\tau = 0.001$ (soft update)

**Discount Factor**: $\gamma = 0.99$

**Exploration Noise**:
- OU process: $\theta_{OU} = 0.15$, $\sigma_{OU} = 0.2$
- Gaussian: $\sigma = 0.1$ (decaying)

**Batch Size**: 256-512 (larger than discrete RL)

**Replay Buffer**: $10^5$ - $10^6$ transitions

---

## 7. Comparison with Discrete Action Methods

| Aspect | Discrete (Q-Learning) | Continuous (DDPG) |
|--------|----------------------|-------------------|
| **Policy** | $\pi(a \mid s, m)$ (softmax) | $\mu(s, m)$ (deterministic) |
| **Action Selection** | $\arg\max_a Q(s,a,m)$ | Direct $\mu(s,m)$ |
| **Critic Input** | $Q(s,m)$ outputs $|\mathcal{A}|$ values | $Q(s,a,m)$ single value |
| **Exploration** | $\epsilon$-greedy | Additive noise |
| **Scalability** | Poor for $|\mathcal{A}| > 20$ | Excellent for $a \in \mathbb{R}^d$ |
| **Sample Efficiency** | Higher (off-policy Q-learning) | Lower (need many samples) |
| **Stability** | More stable | Needs target networks |

---

## 8. Applications

### 8.1 Crowd Navigation

**State**: $s = (x, y, v_x, v_y) \in \mathbb{R}^4$
**Action**: $a = (\Delta v_x, \Delta v_y) \in [-a_{max}, a_{max}]^2$
**Population**: $m(x,y)$ (crowd density)

**Reward**:
$$
r(s, a, m) = -\|s - s_{goal}\| - \lambda_1 \|a\| - \lambda_2 m(s)
$$

### 8.2 Price Formation

**State**: $s = (inventory, cash, market\_signal) \in \mathbb{R}^3$
**Action**: $a = price \in [p_{min}, p_{max}] \subset \mathbb{R}$
**Population**: $m(p)$ (price distribution)

**Reward**:
$$
r(s, a, m) = profit(s, a) - \lambda \cdot spread(a, m)
$$

### 8.3 Resource Allocation

**State**: $s = (demand, capacity, time) \in \mathbb{R}^3$
**Action**: $a = allocation \in [0, capacity] \subset \mathbb{R}$
**Population**: $m(allocation)$ (allocation distribution)

**Reward**:
$$
r(s, a, m) = utility(s, a) - cost(a) - congestion(a, m)
$$

---

## 9. Advantages and Limitations

### 9.1 Advantages

1. **Continuous Control**: Natural for MFG applications (velocity, price, allocation)
2. **High-Dimensional Actions**: Scales to $a \in \mathbb{R}^d$, $d > 10$
3. **Deterministic Policy**: Easier to interpret and deploy
4. **Off-Policy**: Sample efficient with replay buffer
5. **Theory-Aligned**: Matches classical MFG continuous control formulation

### 9.2 Limitations

1. **Exploration Challenge**: Additive noise may be insufficient for complex environments
2. **Overestimation Bias**: Critic can overestimate Q-values (mitigated by TD3/SAC)
3. **Sample Complexity**: Requires more samples than discrete methods
4. **Hyperparameter Sensitivity**: Sensitive to learning rates, noise levels
5. **Convergence**: No strong theoretical guarantees for nonlinear function approximation

---

## 10. Extensions

### 10.1 Multi-Population DDPG

**K Agent Types** with type-specific continuous actions:
$$
\mu^k: \mathcal{S}^k \times \mathcal{P}(\mathcal{S}^1) \times \cdots \times \mathcal{P}(\mathcal{S}^K) \to \mathcal{A}^k
$$

**Nash Equilibrium**: Each type plays best response:
$$
\mu^{k*} = \arg\max_{\mu^k} J^k(\mu^k \mid \mu^{-k*})
$$

### 10.2 TD3 for MFG (Twin Delayed DDPG)

**Twin Critics**: $Q_{\phi_1}(s,a,m)$, $Q_{\phi_2}(s,a,m)$

**Clipped Target**:
$$
y = r + \gamma \min_{i=1,2} Q_{\phi_i'}(s', \tilde{a}', m')
$$

where $\tilde{a}' = \mu_{\theta'}(s', m') + \epsilon$, $\epsilon \sim \text{clip}(\mathcal{N}(0, \sigma), -c, c)$

### 10.3 SAC for MFG (Soft Actor-Critic)

**Maximum Entropy Objective**:
$$
J(\pi) = \mathbb{E}\left[ \sum_t \gamma^t \left( r(s_t, a_t, m_t) + \alpha \mathcal{H}(\pi(\cdot | s_t, m_t)) \right) \right]
$$

**Stochastic Policy**: $\pi(a | s, m) = \text{tanh}(\mathcal{N}(\mu_\theta(s,m), \sigma_\theta(s,m)))$

---

## 11. References

**DDPG**:
- Lillicrap et al. (2015): "Continuous control with deep reinforcement learning"
- Silver et al. (2014): "Deterministic policy gradient algorithms"

**Mean Field Games**:
- Lasry & Lions (2007): "Mean field games"
- Cardaliaguet et al. (2015): "Viscosity solutions for MFG with continuous control"

**Mean Field RL**:
- Guo et al. (2019): "Mean field multi-agent reinforcement learning"
- Yang et al. (2018): "Mean field actor-critic"

**Extensions**:
- Fujimoto et al. (2018): "TD3" (Twin Delayed DDPG)
- Haarnoja et al. (2018): "SAC" (Soft Actor-Critic)

---

**Status**: ✅ Theory complete, ready for implementation
**Next**: Implementation in `mfg_pde/alg/reinforcement/algorithms/mean_field_ddpg.py`
