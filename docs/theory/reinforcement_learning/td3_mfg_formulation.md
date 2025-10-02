# Twin Delayed DDPG (TD3) for Mean Field Games

**Algorithm**: TD3 for MFG (Twin Delayed Deep Deterministic Policy Gradient)
**Improvements over DDPG**: Twin critics, delayed updates, target policy smoothing
**Date**: October 2025
**Status**: Phase 3.3.2 Implementation

---

## 1. Motivation: DDPG Limitations

### 1.1 Overestimation Bias in DDPG

**Problem**: DDPG's critic tends to overestimate Q-values due to maximization bias.

In standard DDPG:
$$
y = r + \gamma Q_{\phi'}(s', \mu_{\theta'}(s', m'), m')
$$

The critic $Q_\phi$ approximates the maximum, but approximation errors accumulate:
- Q-network has errors: $Q_\phi(s,a,m) \approx Q^*(s,a,m) + \epsilon$
- Target uses max over approximation: $\max_a Q_\phi(s,a,m) > Q^*(s,\arg\max_a Q^*(s,a,m),m)$
- Overestimation propagates through Bellman updates

**Impact on MFG**:
- Overoptimistic Q-values → suboptimal policies
- Nash equilibrium not achieved (agents believe rewards are higher than reality)
- Training instability and divergence

### 1.2 Actor-Critic Coupling Issues

**Problem**: In DDPG, actor and critic update at same frequency.

- **Critic variance**: Noisy Q-value estimates early in training
- **Actor instability**: Policy gradient based on unreliable Q-values
- **Oscillations**: Actor and critic chase each other without converging

---

## 2. TD3 Solutions

### 2.1 Twin Critics (Clipped Double Q-Learning)

**Idea**: Use two independent Q-networks, take minimum for target.

**Architecture**:
$$
\begin{align}
Q_{\phi_1}(s, a, m) &: \text{Critic 1} \\
Q_{\phi_2}(s, a, m) &: \text{Critic 2}
\end{align}
$$

**Target Computation**:
$$
y = r + \gamma \min_{i=1,2} Q_{\phi_i'}(s', \mu_{\theta'}(s', m'), m')
$$

**Key Insight**:
- Both critics have independent approximation errors
- Taking minimum provides **pessimistic** estimate (underestimation bias)
- Underestimation is less harmful than overestimation for policy learning

**Mathematical Justification**:
$$
\mathbb{E}[\min(Q_1, Q_2)] \leq \min(\mathbb{E}[Q_1], \mathbb{E}[Q_2]) \leq Q^*
$$

where $Q_1, Q_2$ are independent overestimators.

### 2.2 Delayed Policy Updates

**Idea**: Update actor less frequently than critics.

**Frequencies**:
- Critic update: **Every step**
- Actor update: **Every $d$ steps** (typically $d=2$)

**Rationale**:
- Allows critics to stabilize before policy changes
- Reduces variance in policy gradient estimates
- Prevents actor from exploiting transient Q-value errors

**Algorithm**:
```python
for iteration in range(total_iterations):
    # Update both critics
    update_critic(Q_φ1, Q_φ2)

    # Update actor only every d steps
    if iteration % d == 0:
        update_actor(μ_θ)
        update_targets(Q_φ1', Q_φ2', μ_θ')
```

### 2.3 Target Policy Smoothing

**Idea**: Add noise to target actions for regularization.

**Target with Smoothing**:
$$
\tilde{a}' = \mu_{\theta'}(s', m') + \epsilon, \quad \epsilon \sim \text{clip}(\mathcal{N}(0, \sigma), -c, c)
$$

**Clipped Target**:
$$
y = r + \gamma \min_{i=1,2} Q_{\phi_i'}(s', \tilde{a}', m')
$$

**Benefits**:
- Smooths Q-value estimates over similar actions
- Reduces sensitivity to single action choices
- Improves generalization to nearby state-action pairs

**Hyperparameters**:
- Noise std: $\sigma = 0.2$
- Clip range: $c = 0.5$ (prevents noise from dominating)

---

## 3. TD3 for Mean Field Games

### 3.1 Complete Algorithm

**State**: $s \in \mathcal{S}$ (individual agent state)
**Action**: $a \in \mathcal{A} \subset \mathbb{R}^d$ (continuous)
**Population**: $m \in \mathcal{P}(\mathcal{S})$ (density distribution)

**Components**:
1. **Actor**: $\mu_\theta: \mathcal{S} \times \mathcal{P}(\mathcal{S}) \to \mathcal{A}$
2. **Twin Critics**: $Q_{\phi_1}(s,a,m)$, $Q_{\phi_2}(s,a,m)$
3. **Target Networks**: $\mu_{\theta'}$, $Q_{\phi_1'}$, $Q_{\phi_2'}$

**Training Loop**:

```
For each episode:
    Reset environment, get initial state s, population m

    For each timestep:
        # Select action with exploration noise
        a = μ_θ(s, m) + ε,  ε ~ N(0, σ_explore)

        # Execute action, observe reward r, next state s', next population m'

        # Store transition (s, a, r, s', m, m') in replay buffer

        # Sample mini-batch from replay buffer

        # Compute target with clipped double Q-learning + target smoothing
        ã' = μ_θ'(s', m') + clip(N(0, σ_target), -c, c)
        y = r + γ · min(Q_φ1'(s', ã', m'), Q_φ2'(s', ã', m'))

        # Update both critics
        L_φ1 = MSE(Q_φ1(s,a,m), y)
        L_φ2 = MSE(Q_φ2(s,a,m), y)

        # Delayed policy update (every d steps)
        if iteration % d == 0:
            # Update actor
            L_θ = -E[Q_φ1(s, μ_θ(s,m), m)]

            # Soft update target networks
            φ1' ← τφ1 + (1-τ)φ1'
            φ2' ← τφ2 + (1-τ)φ2'
            θ' ← τθ + (1-τ)θ'
```

### 3.2 Nash Equilibrium with TD3

**Deterministic Nash Equilibrium**:
$$
\mu^*(s, m^*) = \arg\max_a Q^*(s, a, m^*)
$$

where $m^* = \mu(\mu^*)$ is the fixed-point population.

**TD3 Convergence** (informal):
1. Twin critics converge: $Q_{\phi_1}, Q_{\phi_2} \to Q^*$
2. Actor converges to greedy policy: $\mu_\theta \to \arg\max_a \min(Q_{\phi_1}, Q_{\phi_2})$
3. Population converges: $m \to m^*$

**Advantages for MFG**:
- **Stability**: Delayed updates prevent oscillations in Nash equilibrium search
- **Accuracy**: Twin critics give better Q-value estimates for best-response policies
- **Robustness**: Target smoothing handles population distribution changes

---

## 4. Implementation Details

### 4.1 Twin Critic Architecture

**Critic 1**:
```python
class TD3Critic1(nn.Module):
    def __init__(self, state_dim, action_dim, population_dim):
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU())
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, 64), nn.ReLU())
        self.pop_encoder = nn.Sequential(nn.Linear(population_dim, 128), nn.ReLU())

        combined_dim = 256 + 64 + 128
        self.q_head = nn.Sequential(
            nn.Linear(combined_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action, population_state):
        state_feat = self.state_encoder(state)
        action_feat = self.action_encoder(action)
        pop_feat = self.pop_encoder(population_state)
        combined = torch.cat([state_feat, action_feat, pop_feat], dim=1)
        return self.q_head(combined).squeeze(-1)
```

**Critic 2**: Identical architecture, independent parameters.

### 4.2 Target Computation

```python
def compute_td3_target(self, next_states, next_pop_states, rewards, dones):
    with torch.no_grad():
        # Target action with smoothing noise
        next_actions = self.actor_target(next_states, next_pop_states)
        noise = torch.randn_like(next_actions) * self.config['target_noise_std']
        noise = torch.clamp(noise, -self.config['target_noise_clip'],
                                   self.config['target_noise_clip'])
        next_actions = torch.clamp(next_actions + noise,
                                   self.action_bounds[0],
                                   self.action_bounds[1])

        # Clipped double Q-learning
        target_q1 = self.critic1_target(next_states, next_actions, next_pop_states)
        target_q2 = self.critic2_target(next_states, next_actions, next_pop_states)
        target_q = torch.min(target_q1, target_q2)

        # TD target
        return rewards + self.config['gamma'] * target_q * (~dones)
```

### 4.3 Delayed Actor Update

```python
def update(self):
    # Update critics every step
    batch = self.replay_buffer.sample(self.config['batch_size'])

    # Compute TD3 target
    target = self.compute_td3_target(...)

    # Update both critics
    critic1_loss = F.mse_loss(self.critic1(...), target)
    critic2_loss = F.mse_loss(self.critic2(...), target)

    self.critic1_optimizer.zero_grad()
    critic1_loss.backward()
    self.critic1_optimizer.step()

    self.critic2_optimizer.zero_grad()
    critic2_loss.backward()
    self.critic2_optimizer.step()

    # Delayed policy update
    if self.update_count % self.config['policy_delay'] == 0:
        # Actor update (use critic1 only)
        actor_loss = -self.critic1(states, self.actor(states, pop_states), pop_states).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update targets
        self._soft_update_targets()

    self.update_count += 1
```

---

## 5. Hyperparameters

### 5.1 TD3-Specific Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `policy_delay` | 2 | Actor update frequency |
| `target_noise_std` | 0.2 | Std of target smoothing noise |
| `target_noise_clip` | 0.5 | Clip range for smoothing noise |

### 5.2 Shared with DDPG

| Parameter | Value | Description |
|-----------|-------|-------------|
| `actor_lr` | 1e-4 | Actor learning rate |
| `critic_lr` | 1e-3 | Critic learning rate (higher) |
| `gamma` | 0.99 | Discount factor |
| `tau` | 0.001 | Soft update coefficient |
| `batch_size` | 256 | Mini-batch size |
| `buffer_size` | 1e6 | Replay buffer capacity |

---

## 6. Comparison: TD3 vs DDPG

### 6.1 Algorithmic Differences

| Aspect | DDPG | TD3 |
|--------|------|-----|
| **Critics** | Single Q(s,a,m) | Twin Q₁(s,a,m), Q₂(s,a,m) |
| **Target** | Q'(s',μ'(s',m'),m') | min(Q₁'(s',ã',m'), Q₂'(s',ã',m')) |
| **Policy Update** | Every step | Every d steps (delayed) |
| **Target Smoothing** | None | ã' = μ'(s',m') + clip(ε, -c, c) |
| **Bias** | Overestimation | Underestimation (safer) |

### 6.2 Performance Differences

**Stability**:
- DDPG: Prone to oscillations, divergence
- TD3: More stable, smoother convergence

**Sample Efficiency**:
- DDPG: Good when Q-values are accurate
- TD3: Better with noisy/approximate Q-values

**Final Performance**:
- DDPG: Can achieve high performance if stable
- TD3: More consistent, robust to hyperparameters

---

## 7. Mean Field Coupling

### 7.1 Population State Integration

TD3 inherits DDPG's mean field coupling:

**Actor**: $\mu_\theta(s, m)$ conditions on population $m$

**Twin Critics**: Both $Q_{\phi_1}(s,a,m)$ and $Q_{\phi_2}(s,a,m)$ encode $m$

**Population Update**:
$$
m_t(s) \approx \frac{1}{N} \sum_{i=1}^N K_h(s - s_t^i)
$$

### 7.2 Nash Equilibrium Properties

**Fixed Point**:
$$
\mu^*, m^* \text{ satisfy: } \mu^* = \arg\max_\mu J(\mu | m^*), \quad m^* = \mu(\mu^*)
$$

**TD3 Advantages for Nash Convergence**:
1. **Stable Q-values**: Twin critics → accurate best-response
2. **Delayed updates**: Population changes → actor adapts smoothly
3. **Target smoothing**: Robust to population distribution noise

---

## 8. Applications to MFG

### 8.1 Crowd Navigation

**State**: $s = (x, y, v_x, v_y) \in \mathbb{R}^4$
**Action**: $a = (\Delta v_x, \Delta v_y) \in \mathbb{R}^2$
**Population**: $m(x,y)$ (crowd density)

**Benefits of TD3**:
- Stable velocity control (no overestimation)
- Smooth congestion avoidance (target smoothing)
- Robust to varying crowd sizes

### 8.2 Price Formation

**State**: $s = (inventory, cash) \in \mathbb{R}^2$
**Action**: $a = price \in \mathbb{R}_+$
**Population**: $m(price)$ (price distribution)

**Benefits of TD3**:
- Accurate price estimation (twin critics)
- Stable market dynamics (delayed updates)
- Robust to market volatility (target smoothing)

### 8.3 Traffic Flow Control

**State**: $s = (position, velocity) \in \mathbb{R}^4$
**Action**: $a = (steering, acceleration) \in \mathbb{R}^2$
**Population**: $m(x,v)$ (traffic density)

**Benefits of TD3**:
- Safe control (underestimation bias → conservative)
- Smooth trajectories (target smoothing)
- Stable in congested traffic

---

## 9. Convergence Analysis

### 9.1 Theoretical Properties

**Assumption 1** (Twin Critic Convergence):
$$
Q_{\phi_1}, Q_{\phi_2} \to Q^*, \quad \min(Q_{\phi_1}, Q_{\phi_2}) \leq Q^*
$$

**Assumption 2** (Delayed Policy Convergence):
$$
\mu_\theta \to \arg\max_a Q^*(s, a, m) \text{ with } d\text{-step delay}
$$

**Assumption 3** (Population Fixed Point):
$$
m_t \to m^*, \quad m^* = \mu(\mu^*, m^*)
$$

**Theorem** (Informal): Under Assumptions 1-3, TD3 converges to Nash equilibrium of MFG.

### 9.2 Practical Convergence Criteria

1. **Q-value stability**: $|\min(Q_{\phi_1}, Q_{\phi_2}) - Q_{prev}| < \epsilon_Q$
2. **Policy stability**: $\|\mu_\theta - \mu_{prev}\| < \epsilon_\mu$
3. **Population stability**: $\|m_t - m_{t-d}\| < \epsilon_m$
4. **Nash gap**: $J(\mu^*) - \max_\mu J(\mu | m^*) < \epsilon_{Nash}$

---

## 10. Advantages and Limitations

### 10.1 Advantages

1. **Reduced Overestimation**: Twin critics mitigate value overestimation
2. **Training Stability**: Delayed updates reduce actor-critic oscillations
3. **Better Generalization**: Target smoothing improves robustness
4. **Hyperparameter Robustness**: Less sensitive than DDPG to learning rates
5. **State-of-the-Art Performance**: TD3 often outperforms DDPG empirically

### 10.2 Limitations

1. **Computational Cost**: 2× critic networks and updates
2. **Underestimation Risk**: Min operator can be too pessimistic
3. **Delayed Response**: Policy updates lag behind environment changes
4. **Hyperparameter Tuning**: Policy delay $d$ and noise parameters require tuning
5. **Still Deterministic**: No entropy regularization (unlike SAC)

---

## 11. Extensions

### 11.1 Multi-Population TD3

**K Agent Types** with type-specific continuous actions:

$$
\mu^k_{\theta_k}: \mathcal{S}^k \times \mathcal{P}(\mathcal{S}^1) \times \cdots \times \mathcal{P}(\mathcal{S}^K) \to \mathcal{A}^k
$$

**Twin Critics per Type**:
$$
Q^k_{\phi_{1,k}}(s^k, a^k, m^1, \ldots, m^K), \quad Q^k_{\phi_{2,k}}(s^k, a^k, m^1, \ldots, m^K)
$$

### 11.2 SAC for MFG (Next Phase)

TD3 is deterministic. For stochastic policies:

**Soft Actor-Critic**:
- Stochastic policy: $\pi_\theta(a|s,m)$
- Maximum entropy objective: $J = \mathbb{E}[r + \alpha \mathcal{H}(\pi)]$
- Automatic temperature tuning

---

## 12. References

**TD3**:
- Fujimoto et al. (2018): "Addressing Function Approximation Error in Actor-Critic Methods"
- Demonstrates TD3 > DDPG on MuJoCo benchmarks

**Mean Field Games**:
- Lasry & Lions (2007): Mean Field Games (continuous control)
- Cardaliaguet et al. (2015): Viscosity solutions for continuous action MFG

**Mean Field RL**:
- Guo et al. (2019): Mean Field Multi-Agent RL
- Yang et al. (2018): Mean Field Actor-Critic

---

**Status**: ✅ Theory complete, ready for implementation
**Next**: Implementation in `mfg_pde/alg/reinforcement/algorithms/mean_field_td3.py`
