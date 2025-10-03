# Soft Actor-Critic (SAC) for Mean Field Games

**Algorithm**: SAC for MFG (Soft Actor-Critic with Maximum Entropy)
**Key Innovation**: Stochastic policies with entropy regularization
**Date**: October 2025
**Status**: Phase 3.3.3 Implementation

---

## 1. Motivation: Beyond Deterministic Policies

### 1.1 Limitations of DDPG and TD3

**DDPG and TD3 are deterministic**:
- Policy: $\mu_\theta(s, m) \to a$ (single action)
- No exploration during evaluation
- Can get stuck in local optima
- Sensitive to initialization

**Problem for MFG**:
- Nash equilibrium may require **mixed strategies** (stochastic policies)
- Deterministic policies can't represent uncertainty
- Poor exploration in high-dimensional action spaces

### 1.2 Maximum Entropy Reinforcement Learning

**Idea**: Maximize both reward AND entropy.

**Objective**:
$$
J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[ \sum_{t=0}^T \gamma^t \left( r(s_t, a_t, m_t) + \alpha \mathcal{H}(\pi(\cdot | s_t, m_t)) \right) \right]
$$

where:
- $\mathcal{H}(\pi) = -\mathbb{E}_{a \sim \pi}[\log \pi(a | s, m)]$ is entropy
- $\alpha > 0$ is temperature parameter (controls exploration vs exploitation)

**Benefits**:
1. **Exploration**: High entropy → diverse actions
2. **Robustness**: Policy spreads probability over multiple good actions
3. **Transfer**: Exploration helps learn generalizable policies
4. **Multimodality**: Can represent multiple Nash equilibria

---

## 2. SAC Formulation for MFG

### 2.1 Stochastic Policy

**Gaussian Policy**:
$$
\pi_\theta(a | s, m) = \mathcal{N}(\mu_\theta(s, m), \sigma_\theta(s, m))
$$

**Squashed Gaussian** (for bounded actions):
$$
a = \tanh(\mu_\theta(s, m) + \sigma_\theta(s, m) \odot \epsilon), \quad \epsilon \sim \mathcal{N}(0, I)
$$

**Log-Probability** (with change of variables):
$$
\log \pi_\theta(a | s, m) = \log \mathcal{N}(\tilde{a} | \mu, \sigma) - \sum_i \log(1 - \tanh^2(\tilde{a}_i))
$$

where $\tilde{a} = \tanh^{-1}(a)$.

### 2.2 Soft Q-Function

**Standard Q-function** (without entropy):
$$
Q^{\pi}(s, a, m) = \mathbb{E}\left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t, m_t) \mid s_0=s, a_0=a \right]
$$

**Soft Q-function** (with entropy):
$$
Q^{\pi}_{\text{soft}}(s, a, m) = \mathbb{E}\left[ \sum_{t=0}^\infty \gamma^t \left( r(s_t, a_t, m_t) + \alpha \mathcal{H}(\pi(\cdot | s_t, m_t)) \right) \mid s_0=s, a_0=a \right]
$$

**Soft Bellman Equation**:
$$
Q^{\pi}_{\text{soft}}(s, a, m) = r(s, a, m) + \gamma \mathbb{E}_{s', m'}\left[ V^{\pi}_{\text{soft}}(s', m') \right]
$$

where **soft value function**:
$$
V^{\pi}_{\text{soft}}(s, m) = \mathbb{E}_{a \sim \pi}\left[ Q^{\pi}_{\text{soft}}(s, a, m) - \alpha \log \pi(a | s, m) \right]
$$

### 2.3 Policy Improvement

**Soft Policy Iteration**:

1. **Policy Evaluation**: Solve for $Q^{\pi}_{\text{soft}}$
2. **Policy Improvement**:
$$
\pi_{\text{new}} = \arg\max_{\pi'} \mathbb{E}_{s, m}\left[ \mathbb{E}_{a \sim \pi'}\left[ Q^{\pi}_{\text{soft}}(s, a, m) - \alpha \log \pi'(a | s, m) \right] \right]
$$

**Closed-form solution** (for Gaussian):
$$
\pi_{\text{new}}(a | s, m) \propto \exp\left( \frac{1}{\alpha} Q^{\pi}_{\text{soft}}(s, a, m) \right)
$$

---

## 3. SAC Algorithm for Mean Field Games

### 3.1 Components

**1. Stochastic Actor**: $\pi_\theta(a | s, m)$
- Outputs: mean $\mu_\theta(s, m)$ and std $\sigma_\theta(s, m)$
- Action: $a = \tanh(\mu + \sigma \odot \epsilon)$

**2. Twin Soft Q-Critics**: $Q_{\phi_1}(s, a, m)$, $Q_{\phi_2}(s, a, m)$
- Same architecture as TD3
- Trained to minimize soft Bellman error

**3. Target Soft V-Function**: $V_{\psi}(s, m)$
- Or use target Q-networks (SAC variant)

**4. Temperature Parameter**: $\alpha$ (automatic tuning)

### 3.2 Training Objectives

**Critic Loss** (soft Bellman residual):
$$
L_Q = \mathbb{E}\left[ \left( Q_\phi(s, a, m) - \left( r + \gamma V_{\bar{\psi}}(s', m') \right) \right)^2 \right]
$$

where target:
$$
V_{\bar{\psi}}(s', m') = \mathbb{E}_{a' \sim \pi_\theta}\left[ \min_{i=1,2} Q_{\bar{\phi}_i}(s', a', m') - \alpha \log \pi_\theta(a' | s', m') \right]
$$

**Actor Loss** (maximize soft Q-value):
$$
L_\pi = \mathbb{E}_{s, m}\left[ \mathbb{E}_{a \sim \pi_\theta}\left[ \alpha \log \pi_\theta(a | s, m) - Q_\phi(s, a, m) \right] \right]
$$

Equivalently:
$$
L_\pi = \mathbb{E}_{s, m, \epsilon}\left[ \alpha \log \pi_\theta(f_\theta(\epsilon; s, m) | s, m) - Q_\phi(s, f_\theta(\epsilon; s, m), m) \right]
$$

where $f_\theta(\epsilon; s, m) = \tanh(\mu_\theta(s,m) + \sigma_\theta(s,m) \odot \epsilon)$ is reparameterization trick.

**Temperature Loss** (automatic tuning):
$$
L_\alpha = \mathbb{E}_{s, m, a \sim \pi}\left[ -\alpha \left( \log \pi(a | s, m) + \bar{\mathcal{H}} \right) \right]
$$

where $\bar{\mathcal{H}}$ is target entropy (typically $-\dim(\mathcal{A})$).

### 3.3 Complete Algorithm

```
Initialize:
    - Actor π_θ(a|s,m) with parameters θ
    - Twin critics Q_φ1(s,a,m), Q_φ2(s,a,m)
    - Target critics Q_φ1', Q_φ2'
    - Temperature α (or log α as learnable parameter)
    - Replay buffer D

For each iteration:
    # Collect experience
    a ~ π_θ(·|s,m)
    (s', r, m') = env.step(a)
    D.push((s, a, r, s', m, m'))

    # Sample batch
    (s, a, r, s', m, m') ~ D

    # Update critics (twin Q-functions)
    a' ~ π_θ(·|s',m')
    y = r + γ(min(Q_φ1'(s',a',m'), Q_φ2'(s',a',m')) - α log π_θ(a'|s',m'))

    L_Q1 = MSE(Q_φ1(s,a,m), y)
    L_Q2 = MSE(Q_φ2(s,a,m), y)

    Update φ1, φ2

    # Update actor
    a_new ~ π_θ(·|s,m)  [via reparameterization]
    L_π = E[α log π_θ(a_new|s,m) - min(Q_φ1(s,a_new,m), Q_φ2(s,a_new,m))]

    Update θ

    # Update temperature (automatic)
    L_α = E[-α(log π_θ(a_new|s,m) + H_target)]

    Update α

    # Soft update targets
    φ1' ← τφ1 + (1-τ)φ1'
    φ2' ← τφ2 + (1-τ)φ2'
```

---

## 4. SAC for Mean Field Games: Specific Considerations

### 4.1 Population State Coupling

**Actor Network**:
$$
\mu_\theta(s, m), \sigma_\theta(s, m) = f_\theta(s, m)
$$

Both mean and variance depend on population state $m$.

**Soft Q-Network**:
$$
Q_\phi(s, a, m) = g_\phi(s, a, m)
$$

Standard critic with population encoding.

### 4.2 Nash Equilibrium with Maximum Entropy

**Soft Nash Equilibrium**:
$$
\pi^*(a | s, m^*) = \arg\max_\pi \mathbb{E}_{a \sim \pi}\left[ Q^*(s, a, m^*) - \alpha \log \pi(a | s, m^*) \right]
$$

where $m^* = \mu(\pi^*)$ is the fixed-point population.

**Properties**:
1. **Stochastic**: $\pi^*$ has non-zero variance
2. **Multimodal**: Can represent multiple equilibria
3. **Robust**: Spreads probability over good actions
4. **Exploration**: Natural exploration through entropy

### 4.3 Temperature Tuning for MFG

**Automatic Temperature Adjustment**:
$$
\alpha^* = \arg\min_\alpha \mathbb{E}_{s, m, a \sim \pi_{\alpha}}\left[ -\alpha \left( \log \pi_\alpha(a | s, m) + \bar{\mathcal{H}} \right) \right]
$$

**Target Entropy**:
- Standard: $\bar{\mathcal{H}} = -\dim(\mathcal{A})$
- For MFG: May need adjustment based on population coupling strength

**Effect on Population Dynamics**:
- High $\alpha$: More exploration → diverse population behaviors
- Low $\alpha$: More exploitation → concentrated population

---

## 5. Implementation Details

### 5.1 Stochastic Actor Network

```python
class SACActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, population_dim):
        # State + population encoder
        self.encoder = nn.Sequential(...)

        # Output mean and log_std
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # Action bounds
        self.action_scale = (action_max - action_min) / 2
        self.action_bias = (action_max + action_min) / 2

    def forward(self, state, population_state):
        features = self.encoder(torch.cat([state, population_state], dim=1))
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state, population_state):
        mean, log_std = self.forward(state, population_state)
        std = log_std.exp()

        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Sample with gradient

        # Squash to action bounds
        action = torch.tanh(x_t)
        action = action * self.action_scale + self.action_bias

        # Log probability with change of variables
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob
```

### 5.2 Automatic Temperature Tuning

```python
class SAC:
    def __init__(self, ...):
        # Learnable log(alpha)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        # Target entropy = -dim(A)
        self.target_entropy = -action_dim

    def update_temperature(self, log_probs):
        alpha = self.log_alpha.exp()

        # Temperature loss
        alpha_loss = -(alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return alpha.item(), alpha_loss.item()
```

---

## 6. Comparison: DDPG vs TD3 vs SAC

### 6.1 Algorithmic Differences

| Aspect | DDPG | TD3 | SAC |
|--------|------|-----|-----|
| **Policy** | Deterministic μ(s,m) | Deterministic μ(s,m) | Stochastic π(a\|s,m) |
| **Critics** | 1 Q-network | 2 Q-networks (twin) | 2 Q-networks (twin) |
| **Target** | Q'(s',μ'(s',m'),m') | min(Q'₁,Q'₂)(s',a',m') | min(Q'₁,Q'₂)(s',a',m') - α log π |
| **Exploration** | OU/Gaussian noise | Gaussian + smoothing | Entropy maximization |
| **Temperature** | Fixed | Fixed | Automatic tuning |
| **Entropy** | No | No | Yes (α𝓗) |
| **Update Freq** | Every step | Delayed (every d) | Every step |

### 6.2 Performance Characteristics

**Sample Efficiency**:
- DDPG: Good when stable
- TD3: Better (twin critics)
- **SAC: Best** (entropy exploration + twin critics)

**Stability**:
- DDPG: Can oscillate
- TD3: Stable (delayed updates)
- **SAC: Very stable** (entropy regularization)

**Exploration**:
- DDPG: Poor (additive noise)
- TD3: Moderate (target smoothing)
- **SAC: Excellent** (maximum entropy)

**Final Performance**:
- DDPG: Variable
- TD3: Good
- **SAC: State-of-the-art**

### 6.3 When to Use Each

**Use DDPG if**:
- Computational budget is limited (1 critic)
- Deterministic policy is required
- Problem is low-dimensional and simple

**Use TD3 if**:
- Need stability over DDPG
- Deterministic policy preferred
- Don't need automatic exploration

**Use SAC if**:
- Want state-of-the-art performance
- Need robust exploration
- Problem is complex/high-dimensional
- Stochastic policies acceptable

---

## 7. Applications to MFG

### 7.1 Crowd Navigation with Uncertainty

**State**: $s = (x, y, v_x, v_y)$
**Action**: $a = (\Delta v_x, \Delta v_y) \sim \pi(\cdot | s, m)$ (stochastic)
**Population**: $m(x, y)$

**Benefits**:
- Entropy → diverse navigation strategies
- Robust to congestion uncertainty
- Exploration → find alternative paths

### 7.2 Market Making with Stochastic Prices

**State**: $s = (inventory, market\_signal)$
**Action**: $a = (bid, ask) \sim \pi(\cdot | s, m)$
**Population**: $m(bid, ask)$ (order book distribution)

**Benefits**:
- Stochastic pricing → market exploration
- Entropy → diverse quotes (better liquidity)
- Automatic temperature → adaptive aggression

### 7.3 Multi-Agent Traffic Control

**State**: $s = (position, velocity)$
**Action**: $a = (steering, accel) \sim \pi(\cdot | s, m)$
**Population**: $m(x, v)$ (traffic density)

**Benefits**:
- Stochastic control → safe exploration
- Entropy → diverse maneuvers (avoids deadlocks)
- Robust to varying traffic patterns

---

## 8. Convergence and Theoretical Properties

### 8.1 Soft Policy Iteration Convergence

**Theorem** (Haarnoja et al., 2018):
Under mild assumptions, soft policy iteration converges to the optimal maximum entropy policy:
$$
\pi^* = \arg\max_\pi J_{\alpha}(\pi)
$$

**Proof Sketch**:
1. Soft Bellman operator is contraction
2. Policy improvement step increases soft Q-value
3. Converges to fixed point $\pi^*, Q^*$

### 8.2 Nash Equilibrium for MFG

**Soft Nash Equilibrium**:
$$
(\pi^*, m^*) \text{ such that } \pi^* = \arg\max_\pi J_\alpha(\pi | m^*), \quad m^* = \mu(\pi^*)
$$

**Advantages over Deterministic Nash**:
- **Robustness**: Stochastic policies less sensitive to perturbations
- **Multimodality**: Can represent mixed-strategy equilibria
- **Exploration**: Natural exploration through entropy

---

## 9. Advantages and Limitations

### 9.1 Advantages

1. **State-of-the-Art Performance**: Best empirical results on continuous control
2. **Sample Efficiency**: Entropy exploration + off-policy learning
3. **Stability**: Twin critics + entropy regularization
4. **Automatic Exploration**: No manual noise tuning
5. **Robustness**: Stochastic policies handle uncertainty well
6. **Generalization**: Entropy → diverse behaviors → transfer learning

### 9.2 Limitations

1. **Computational Cost**: 2 critics + entropy computation
2. **Hyperparameter**: Target entropy may need tuning for MFG
3. **Stochastic**: Not suitable when deterministic policy required
4. **Complexity**: More moving parts than DDPG/TD3
5. **Memory**: Requires large replay buffer for best performance

---

## 10. Implementation Checklist

**Core Components**:
- [ ] Stochastic actor with tanh-squashed Gaussian
- [ ] Twin soft Q-critics
- [ ] Target networks with soft updates
- [ ] Automatic temperature tuning
- [ ] Reparameterization trick for gradients
- [ ] Change-of-variables for log-probability

**MFG-Specific**:
- [ ] Population state encoding in actor/critics
- [ ] Mean field consistency tracking
- [ ] Soft Nash equilibrium verification
- [ ] Entropy impact on population dynamics

**Training**:
- [ ] Replay buffer with population states
- [ ] Simultaneous actor/critic/temperature updates
- [ ] Monitoring: α, entropy, Q-values, rewards

---

## 11. References

**SAC**:
- Haarnoja et al. (2018): "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
- Haarnoja et al. (2018): "Soft Actor-Critic Algorithms and Applications"

**Maximum Entropy RL**:
- Ziebart et al. (2008): "Maximum Entropy Inverse RL"
- Levine (2018): "Reinforcement Learning and Control as Probabilistic Inference"

**Mean Field Games**:
- Lasry & Lions (2007): Mean Field Games
- Guo et al. (2019): Mean Field Multi-Agent RL

---

**Status**: ✅ Theory complete, ready for implementation
**Next**: Implementation in `mfg_pde/alg/reinforcement/algorithms/mean_field_sac.py`
