"""
Sketch: Continuous Action Architecture for MFG-RL

This is a design sketch (not production code) showing how to extend
our MFG-RL algorithms to continuous action spaces.

CURRENT (Discrete):
    Q-Network: (s, m) → [Q(s,a₁,m), ..., Q(s,aₙ,m)]  # n outputs
    Actor: (s, m) → softmax([a₁, ..., aₙ])            # n outputs

PROPOSED (Continuous):
    Q-Network: (s, a, m) → Q(s,a,m)                   # 1 output, a is input
    Actor: (s, m) → μ(s,m) ∈ ℝᵈ                       # d outputs (action dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ============================================================================
# APPROACH 1: Continuous Q-Function (for DDPG/TD3)
# ============================================================================


class MeanFieldContinuousQNetwork(nn.Module):
    """
    Q(s, a, m) with continuous action as INPUT.

    Key difference from discrete Q-learning:
    - Action is an INPUT to the network (not an index)
    - Output is a SCALAR Q-value (not a vector)
    - Enables continuous action spaces a ∈ ℝᵈ
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,  # Dimension of continuous action space
        population_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Encode state
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Encode action (NEW - action is now an input!)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Encode population
        self.population_encoder = nn.Sequential(
            nn.Linear(population_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Fusion: combine all features → scalar Q-value
        fusion_dim = 3 * (hidden_dim // 2)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # SCALAR output
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor, population: torch.Tensor) -> torch.Tensor:
        """
        Compute Q(s, a, m).

        Args:
            state: [batch, state_dim]
            action: [batch, action_dim] - CONTINUOUS action
            population: [batch, population_dim]

        Returns:
            q_value: [batch, 1] - scalar Q-value for each (s,a,m)
        """
        state_feat = self.state_encoder(state)
        action_feat = self.action_encoder(action)  # Encode action
        pop_feat = self.population_encoder(population)

        # Concatenate all features
        combined = torch.cat([state_feat, action_feat, pop_feat], dim=1)

        # Fusion to scalar Q-value
        q_value = self.fusion(combined)

        return q_value


class MeanFieldContinuousActor(nn.Module):
    """
    Deterministic policy μ(s, m) for continuous actions.

    Used with DDPG/TD3 style algorithms.
    Output is the action itself (not probabilities).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,  # Dimension of continuous action
        population_dim: int,
        hidden_dim: int = 256,
        action_bound: float = 1.0,  # Action range [-bound, +bound]
    ):
        super().__init__()

        self.action_bound = action_bound

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

        # Policy head: outputs continuous action
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Bound action to [-1, 1], scale later
        )

    def forward(self, state: torch.Tensor, population: torch.Tensor) -> torch.Tensor:
        """
        Compute deterministic action μ(s, m).

        Args:
            state: [batch, state_dim]
            population: [batch, population_dim]

        Returns:
            action: [batch, action_dim] - continuous action in [-bound, +bound]
        """
        state_feat = self.state_encoder(state)
        pop_feat = self.population_encoder(population)

        combined = torch.cat([state_feat, pop_feat], dim=1)

        # Output continuous action
        action = self.policy_head(combined) * self.action_bound

        return action


# ============================================================================
# APPROACH 2: Stochastic Continuous Policy (for SAC/PPO)
# ============================================================================


class MeanFieldStochasticActor(nn.Module):
    """
    Stochastic policy π(·|s,m) for continuous actions.

    Outputs Gaussian distribution: N(μ(s,m), σ(s,m))
    Used with SAC, PPO-continuous style algorithms.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        population_dim: int,
        hidden_dim: int = 256,
    ):
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

        # Mean head
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Log std head (learnable per state)
        self.log_std_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor, population: torch.Tensor):
        """
        Compute Gaussian policy parameters μ(s,m) and σ(s,m).

        Args:
            state: [batch, state_dim]
            population: [batch, population_dim]

        Returns:
            mean: [batch, action_dim]
            std: [batch, action_dim]
        """
        state_feat = self.state_encoder(state)
        pop_feat = self.population_encoder(population)

        combined = torch.cat([state_feat, pop_feat], dim=1)

        # Gaussian parameters
        mean = self.mean_head(combined)
        log_std = self.log_std_head(combined)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Numerical stability
        std = torch.exp(log_std)

        return mean, std

    def sample(self, state: torch.Tensor, population: torch.Tensor):
        """Sample action from policy."""
        mean, std = self.forward(state, population)

        # Gaussian distribution
        dist = torch.distributions.Normal(mean, std)

        # Sample action with reparameterization trick
        action = dist.rsample()  # Reparameterized sample
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        # Optional: tanh squashing for bounded actions (SAC-style)
        # action = torch.tanh(action)
        # log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        return action, log_prob


# ============================================================================
# USAGE EXAMPLES
# ============================================================================


def example_continuous_q_learning():
    """Example: Training Q-network for continuous actions."""

    # Dimensions
    state_dim = 5  # e.g., (x, y, goal_x, goal_y, time)
    action_dim = 2  # e.g., (velocity_x, velocity_y) - CONTINUOUS
    population_dim = 49  # Local density patch

    # Create networks
    q_network = MeanFieldContinuousQNetwork(state_dim, action_dim, population_dim)
    actor = MeanFieldContinuousActor(state_dim, action_dim, population_dim, action_bound=2.0)  # Max velocity = 2
    # Note: target_q network would be used in DDPG training (not shown in this example)

    # Example forward pass
    batch_size = 32
    state = torch.randn(batch_size, state_dim)
    population = torch.randn(batch_size, population_dim)

    # Actor produces continuous action
    action = actor(state, population)  # [32, 2] - continuous velocities
    print(f"Action shape: {action.shape}")  # [32, 2]
    print(f"Action range: [{action.min():.2f}, {action.max():.2f}]")

    # Critic evaluates Q(s, a, m)
    q_value = q_network(state, action, population)  # [32, 1]
    print(f"Q-value shape: {q_value.shape}")  # [32, 1]

    # Training: Update critic
    # next_action = actor(next_state, next_population)  # From target actor
    # target_q_value = target_q(next_state, next_action, next_population)
    # critic_loss = F.mse_loss(q_value, reward + gamma * target_q_value)

    # Training: Update actor (policy gradient)
    # actor_loss = -q_network(state, actor(state, population), population).mean()


def example_stochastic_actor():
    """Example: Stochastic continuous policy."""

    state_dim, action_dim, population_dim = 5, 2, 49

    actor = MeanFieldStochasticActor(state_dim, action_dim, population_dim)

    state = torch.randn(32, state_dim)
    population = torch.randn(32, population_dim)

    # Sample action from Gaussian policy
    action, log_prob = actor.sample(state, population)

    print(f"Action shape: {action.shape}")  # [32, 2]
    print(f"Log prob shape: {log_prob.shape}")  # [32, 1]

    # PPO/SAC training uses log_prob for policy gradient


# ============================================================================
# KEY DIFFERENCES FROM DISCRETE IMPLEMENTATION
# ============================================================================

"""
DISCRETE (Current Implementation):
------------------------------------
1. Q-Network Output: Vector of size |A| (one Q-value per discrete action)
   - Q(s, m) → [Q(s,a₁,m), Q(s,a₂,m), ..., Q(s,aₙ,m)]
   - Action selection: argmax over output vector
   - Scales poorly: O(|A|) outputs, infeasible for |A| > 1000

2. Actor Output: Probability distribution over discrete actions
   - π(·|s, m) → softmax([logit₁, ..., logitₙ])
   - Sample: Categorical distribution
   - Scales poorly: softmax over |A| logits

CONTINUOUS (Proposed):
----------------------
1. Q-Network: Action is INPUT, output is SCALAR
   - Q(s, a, m) where a ∈ ℝᵈ is input
   - Output: Single Q-value
   - Action selection: Requires optimization (gradient ascent or actor network)
   - Scales well: O(1) output regardless of action space size

2. Actor: Direct action output or distribution parameters
   - Deterministic: μ(s, m) → a ∈ ℝᵈ
   - Stochastic: μ(s, m), σ(s, m) → Gaussian distribution
   - Scales well: O(d) outputs where d is action dimension

COMPUTATIONAL COMPLEXITY:
-------------------------
Discrete:
- Forward pass: O(|A|)
- Action selection: O(|A|) for argmax
- Total: O(|A|) - INFEASIBLE for large |A|

Continuous:
- Forward pass: O(d) where d is action dimension
- Action selection: O(1) for actor, O(optimization) for Q-based
- Total: O(d) - SCALES to high dimensions

EXAMPLE SCENARIOS:
------------------
✅ Discrete works: Maze navigation (4 directions), menu selection (10 items)
❌ Discrete fails: Price selection ($0.01-$100.00, 10000 values), continuous velocity

✅ Continuous works: Velocity control, price formation, resource allocation
"""

if __name__ == "__main__":
    print("=" * 70)
    print("Continuous Action MFG-RL Architecture Sketch")
    print("=" * 70)

    print("\n1. Continuous Q-Learning (DDPG-style):")
    print("-" * 70)
    example_continuous_q_learning()

    print("\n2. Stochastic Continuous Policy (SAC/PPO-style):")
    print("-" * 70)
    example_stochastic_actor()

    print("\n" + "=" * 70)
    print("Key Takeaway:")
    print("=" * 70)
    print(
        """
To support continuous actions, we need:
1. Q-function with action as INPUT: Q(s, a, m)
2. Actor that outputs continuous actions: μ(s, m) or π(·|s, m)
3. Different training procedure: DDPG/TD3/SAC instead of DQN

This is NOT a simple extension - requires NEW algorithm implementation.
Current discrete Q-learning and Actor-Critic CANNOT handle continuous actions.
"""
    )
