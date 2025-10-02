# Reinforcement Learning Examples for MFG_PDE

**Date**: October 2, 2025
**Status**: Phase 2 Complete ✅
**Purpose**: Guide to RL algorithms and examples for Mean Field Games

---

## Overview

This directory contains examples demonstrating the **Reinforcement Learning paradigm** for solving Mean Field Games. The RL paradigm provides agent-based approaches that learn equilibrium strategies through interaction and population feedback.

### Implemented Algorithms (Phase 2 ✅)

1. **Mean Field Q-Learning** - Value-based discrete action learning
2. **Mean Field Actor-Critic (PPO)** - Policy gradient with population state
3. **Nash Q-Learning** - Equilibrium learning (symmetric MFG)

### Future Algorithms (Phase 3+)

4. **MADDPG** - Continuous action control (requires continuous action spaces)
5. **Hierarchical RL** - Multi-scale policies
6. **Multi-Population** - Heterogeneous agents

---

## Quick Start

### 1. Basic Examples (`examples/basic/`)

**Nash Q-Learning Demo**:
```bash
python examples/basic/nash_q_learning_demo.py
```
- Visualizes Nash equilibrium policies
- Demonstrates Nash value = max Q-value for symmetric MFG
- Mathematical property verification

### 2. Advanced Examples (`examples/advanced/`)

**Actor-Critic Maze Demo**:
```bash
python examples/advanced/actor_critic_maze_demo.py
```
- Complete training example with Mean Field Actor-Critic (PPO)
- Maze navigation with population dynamics
- Learning curves and convergence visualization

**Comprehensive RL Demo**:
```bash
python examples/advanced/mfg_rl_comprehensive_demo.py
```
- Compares all RL algorithms
- Multiple maze configurations
- Performance benchmarking

**RL Experiment Suite**:
```bash
python examples/advanced/mfg_rl_experiment_suite.py
```
- Systematic algorithm comparison
- Hyperparameter sensitivity analysis
- Statistical significance testing

---

## Example Categories

### Algorithms

| Example | Algorithm | Difficulty | Description |
|---------|-----------|------------|-------------|
| `nash_q_learning_demo.py` | Nash Q-Learning | **Basic** | Nash equilibrium visualization |
| `actor_critic_maze_demo.py` | PPO | **Advanced** | Complete training example |
| `mfg_rl_comprehensive_demo.py` | All | **Advanced** | Algorithm comparison |

### Environments

| Example | Environment | Features |
|---------|-------------|----------|
| `mfg_maze_environment_demo.py` | MFG Maze | Population tracking, rewards |
| `quick_maze_demo.py` | Simple Maze | Quick testing |
| `hybrid_maze_demo.py` | Hybrid Maze | Complex layouts |

### Maze Generation

| Example | Algorithm | Description |
|---------|-----------|-------------|
| `perfect_maze_generator.py` | Cellular Automata | Perfect mazes |
| `voronoi_maze_demo.py` | Voronoi Diagram | Organic mazes |
| `maze_algorithm_assessment.py` | Comparison | Algorithm evaluation |

---

## Algorithm Usage Guide

### Mean Field Q-Learning

**When to use**:
- Discrete action spaces
- Tabular or function approximation
- Fast prototyping

**Example**:
```python
from mfg_pde.alg.reinforcement import create_mean_field_q_learning

# Create algorithm
algo = create_mean_field_q_learning(env, config={
    "learning_rate": 3e-4,
    "discount_factor": 0.99,
    "epsilon": 0.1,
    "batch_size": 64,
})

# Train
results = algo.train(num_episodes=1000)

# Evaluate
actions = algo.predict(observations)
```

### Mean Field Actor-Critic (PPO)

**When to use**:
- Sample efficiency needed
- Stochastic policies
- Stable training required

**Example**:
```python
from mfg_pde.alg.reinforcement.algorithms import MeanFieldActorCritic

# Create algorithm
algo = MeanFieldActorCritic(
    env=env,
    state_dim=4,
    action_dim=4,
    population_dim=8,
    config={
        "clip_epsilon": 0.2,
        "gae_lambda": 0.95,
        "gamma": 0.99,
    }
)

# Train
results = algo.train(num_episodes=1000)

# Evaluate
actions = algo.predict(observations)
```

### Nash Q-Learning

**Key Insight**: For symmetric MFG, Nash Q-Learning = Mean Field Q-Learning

**Example**:
```python
from mfg_pde.alg.reinforcement import create_mean_field_q_learning

# Create algorithm (already implements Nash Q-Learning for symmetric MFG)
algo = create_mean_field_q_learning(env)

# Compute Nash equilibrium value
nash_value = algo.compute_nash_value(states, populations, game_type="symmetric")
```

---

## Environment Configuration

### MFG Maze Environment

**Basic Configuration**:
```python
from mfg_pde.alg.reinforcement.environments import MFGMazeConfig, MFGMazeEnvironment

config = MFGMazeConfig(
    maze_array=maze,              # Numpy array (1=wall, 0=free)
    start_positions=[(1, 1)],     # Agent start positions
    goal_positions=[(8, 8)],      # Goal positions
    action_type=ActionType.FOUR_CONNECTED,  # 4 or 8 directions
    reward_type=RewardType.MFG_STANDARD,    # Reward structure
    num_agents=10,                # Population size
    max_episode_steps=100,        # Episode length
)

env = MFGMazeEnvironment(config)
```

**Reward Types**:
- `SPARSE`: Goal reward only
- `DENSE`: Distance-based rewards
- `MFG_STANDARD`: Goal + step penalty
- `CONGESTION`: Population density penalty

**Action Types**:
- `FOUR_CONNECTED`: Up, Down, Left, Right
- `EIGHT_CONNECTED`: 8 directions including diagonals

---

## Training Tips

### Hyperparameter Tuning

**Q-Learning**:
- `learning_rate`: 1e-4 to 1e-3 (higher for simple problems)
- `epsilon`: Start 1.0, decay to 0.01
- `batch_size`: 32-128 (larger for stability)

**Actor-Critic (PPO)**:
- `clip_epsilon`: 0.1-0.3 (0.2 standard)
- `gae_lambda`: 0.9-0.99 (higher for long-horizon)
- `gamma`: 0.95-0.99 (problem dependent)

### Debugging

**Common Issues**:
1. **No Learning**: Check reward scale, learning rate
2. **Unstable**: Reduce learning rate, increase batch size
3. **Slow Convergence**: Tune exploration, increase network size

**Diagnostic Tools**:
```python
# Enable detailed logging
from mfg_pde.utils.logging import configure_research_logging
configure_research_logging("rl_debug", level="DEBUG")

# Monitor training
import matplotlib.pyplot as plt
plt.plot(results["episode_rewards"])
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
```

---

## Performance Benchmarks

### Expected Results (10x10 Maze, 10 Agents)

| Algorithm | Episodes to Solve | Training Time | Final Reward |
|-----------|------------------|---------------|--------------|
| **Mean Field Q-Learning** | ~500 | 2-3 min | ~8.0 |
| **Mean Field Actor-Critic** | ~300 | 4-5 min | ~9.0 |
| **Nash Q-Learning** | ~500 | 2-3 min | ~8.0 |

*Results on M1 Mac, CPU only*

### Scalability

| Population Size | Memory | Time per Episode |
|----------------|---------|------------------|
| 10 agents | 100 MB | 0.2s |
| 100 agents | 500 MB | 0.5s |
| 1000 agents | 2 GB | 2.0s |

---

## Visualization

### Training Progress

```python
# Plot learning curves
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(results["episode_rewards"])
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")

plt.subplot(1, 2, 2)
plt.plot(results["episode_lengths"])
plt.title("Episode Lengths")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.show()
```

### Policy Visualization

```python
# Visualize learned policy
from mfg_pde.alg.reinforcement.environments import visualize_policy

visualize_policy(env, algo, save_path="policy.png")
```

---

## Theoretical References

### Documentation

**Theory**:
- `docs/theory/reinforcement_learning/nash_q_learning_formulation.md`
- `docs/theory/reinforcement_learning/population_ppo_formulation.md`
- `docs/theory/reinforcement_learning/maddpg_for_mfg_formulation.md`

**Architecture**:
- `docs/theory/reinforcement_learning/nash_q_learning_architecture.md`
- `docs/theory/reinforcement_learning/maddpg_architecture_design.md`

**Roadmap**:
- `docs/planning/roadmaps/REINFORCEMENT_LEARNING_ROADMAP.md`
- `docs/theory/reinforcement_learning/continuous_action_mfg_theory.md`

### Papers

**Mean Field Games**:
- Lasry & Lions (2007): "Mean field games"
- Carmona & Delarue (2018): "Probabilistic Theory of Mean Field Games"

**Mean Field RL**:
- Yang et al. (2018): "Mean Field Multi-Agent Reinforcement Learning"
- Guo et al. (2019): "Learning Mean-Field Games"

**Nash Q-Learning**:
- Hu & Wellman (2003): "Nash Q-Learning for General-Sum Stochastic Games"

**PPO**:
- Schulman et al. (2017): "Proximal Policy Optimization Algorithms"

---

## FAQ

**Q: Which algorithm should I use?**
A: Start with Mean Field Actor-Critic (PPO) - it's sample efficient and stable.

**Q: Can I use continuous actions?**
A: Not yet. See `continuous_action_mfg_theory.md` for roadmap (Phase 3).

**Q: How many agents do I need?**
A: 10-100 agents for mean field approximation. More improves accuracy.

**Q: What's the difference between Nash Q-Learning and Mean Field Q-Learning?**
A: For symmetric MFG, they're the same! Nash equilibrium = max Q-value.

**Q: Can I use GPU?**
A: Yes! Algorithms automatically use GPU if PyTorch detects one.

**Q: How do I save/load models?**
A:
```python
# Save
algo.save("model.pt")

# Load
algo.load("model.pt")
```

---

## Next Steps

### Phase 3: Advanced Features (Coming Soon)

- **Hierarchical RL**: Multi-scale policies
- **Multi-Population**: Heterogeneous agents
- **Continuous Actions**: DDPG/TD3/SAC

### Contributing

See `docs/planning/roadmaps/REINFORCEMENT_LEARNING_ROADMAP.md` for development priorities.

---

**Last Updated**: October 2, 2025
**Status**: Phase 2 Complete ✅
**Next Phase**: Advanced Features (Phase 3)
