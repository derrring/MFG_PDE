# Basic MFG Examples

This directory contains simple, single-concept examples demonstrating core Mean Field Game (MFG) capabilities in MFG_PDE. Each example focuses on one key concept and can be run standalone.

## Quick Start

All examples can be run directly:
```bash
python examples/basic/example_name.py
```

Output files (plots, logs) are saved to `examples/outputs/basic/`.

---

## Examples by Category

### 📚 Classic MFG Problems

#### `lq_mfg_demo.py` - Linear-Quadratic MFG
**Difficulty**: ⭐ Easiest
**Concepts**: Basic MFG setup, analytical validation
**Theory**: Tractable problem with closed-form solution

The simplest MFG problem with quadratic costs. Perfect for understanding:
- MFG problem setup
- Coupled HJB-FPK system
- Convergence monitoring
- Comparison with analytical solution

**Run**:
```bash
python examples/basic/lq_mfg_demo.py
```

---

#### `acceleration_comparison.py` - Solver Acceleration Techniques
**Difficulty**: ⭐⭐ Moderate
**Concepts**: Backend acceleration, Anderson acceleration, performance benchmarking
**Theory**: Computational acceleration for MFG solvers

Comprehensive comparison of acceleration methods:
- **Backend acceleration**: JAX (GPU/CUDA), PyTorch (MPS/CUDA), Numba (JIT)
- **Anderson acceleration**: Convergence speedup via extrapolation (memory depth m=5)
- **Performance analysis**: Timing, speedup metrics, iteration reduction

**Run**:
```bash
python examples/basic/acceleration_comparison.py
```

**Output**: Side-by-side performance visualization showing backend comparison and Anderson impact.

---

### 🎮 Coordination Games

#### `el_farol_bar_demo.py` - El Farol Bar Problem (Discrete)
**Difficulty**: ⭐⭐ Moderate
**Concepts**: Discrete state MFG, network formulation, coordination paradox
**Theory**: `docs/theory/coordination_games_mfg.md`

Binary choice coordination game where agents decide whether to attend a bar. Demonstrates:
- Discrete state space via 2-node network
- Threshold-based payoffs
- Coordination failure (no universally correct prediction)
- Phase portrait analysis

**Mathematical Formulation**:
- States: {home, bar}
- Threshold: 60% capacity
- Network MFG formulation

**Run**:
```bash
python examples/basic/el_farol_bar_demo.py
```

#### `santa_fe_bar_demo.py` - Santa Fe Bar (Continuous Preferences)
**Difficulty**: ⭐⭐ Moderate
**Concepts**: Continuous state MFG, preference evolution
**Theory**: `docs/theory/coordination_games_mfg.md` §3

Continuous preference evolution formulation of coordination game. Demonstrates:
- State space: preference θ ∈ [-θ_max, θ_max]
- Attendance: ∫_{θ>0} m(θ) dθ
- Preference diffusion dynamics
- Richer dynamics than discrete formulation

**Run**:
```bash
python examples/basic/santa_fe_bar_demo.py
```

---

### 🏖️ Spatial Competition

#### `towel_beach_demo.py` - Towel on the Beach
**Difficulty**: ⭐⭐ Moderate
**Concepts**: Spatial MFG, phase transitions, congestion
**Theory**: `docs/theory/spatial_competition_mfg.md`

Spatial competition where agents choose beach positions balancing proximity to ice cream stall vs crowding. Demonstrates:
- Continuous spatial state (position on beach)
- Proximity-congestion trade-off
- **Phase transitions** controlled by crowd aversion λ:
  - Low λ: Single peak at stall
  - High λ: Crater equilibrium (avoid stall)
- Running cost: |x - x_stall| + λ log(m(x)) + (1/2)|u|²

**Run**:
```bash
python examples/basic/towel_beach_demo.py
```

**Output**: Generates equilibrium plots for λ = 0.5, 1.0, 2.0 showing phase transition.

---

### 🔬 Stochastic MFG

#### `common_noise_lq_demo.py` - Common Noise MFG
**Difficulty**: ⭐⭐⭐ Advanced
**Concepts**: Stochastic MFG, common noise, Monte Carlo methods
**Theory**: `docs/theory/stochastic_mfg_common_noise.md`

Linear-Quadratic MFG with common noise (market volatility modeled as Ornstein-Uhlenbeck process). Demonstrates:
- Common noise affecting all agents
- Conditional HJB-FPK system
- Monte Carlo averaging over noise paths
- Risk-aversion depending on volatility

**Mathematical Setup**:
- Individual noise: dx_t = α_t dt + σ dW_t
- Common noise: dθ_t = κ(μ - θ_t) dt + σ_θ dB_t (OU process)
- Cost parameter: λ(θ_t) = λ_0(1 + β·θ_t)

**Run**:
```bash
python examples/basic/common_noise_lq_demo.py
```

---

### 🤖 Neural Methods (Deep Learning)

#### `adaptive_pinn_demo.py` - Physics-Informed Neural Networks
**Difficulty**: ⭐⭐⭐ Advanced
**Concepts**: PINN, mesh-free solving, adaptive training
**Paradigm**: Neural

Demonstrates solving MFG using Physics-Informed Neural Networks with:
- Mesh-free approach (no grid discretization)
- Adaptive point sampling (focus on high-error regions)
- Physics-guided loss functions
- Suitable for high-dimensional problems

**Run**:
```bash
python examples/basic/adaptive_pinn_demo.py
```

**Note**: Requires PyTorch (`pip install torch`)

#### `dgm_simple_validation.py` - Deep Galerkin Method
**Difficulty**: ⭐⭐⭐ Advanced
**Concepts**: DGM, variational formulation, neural network basis
**Paradigm**: Neural

Deep Galerkin Method validation against analytical solution:
- Variational formulation (weak form)
- More stable than standard PINN
- Theoretical convergence guarantees

**Run**:
```bash
python examples/basic/dgm_simple_validation.py
```

**Note**: Requires PyTorch

---

### 🎯 Reinforcement Learning

#### `rl_intro_comparison.py` - RL Introduction
**Difficulty**: ⭐⭐ Moderate
**Concepts**: Mean Field RL, PDE vs RL comparison
**Paradigm**: Reinforcement Learning

Introduction to RL paradigm with comparison to PDE methods:
- Simple LQ-MFG environment
- Mean Field Actor-Critic training
- Comparison with analytical solution
- Convergence analysis

**Run**:
```bash
python examples/basic/rl_intro_comparison.py
```

**Note**: Requires Gymnasium and Stable-Baselines3

#### `nash_q_learning_demo.py` - Nash Q-Learning
**Difficulty**: ⭐⭐⭐ Advanced
**Concepts**: Discrete action RL, Nash equilibrium, Q-learning
**Paradigm**: Reinforcement Learning

Nash Q-Learning for discrete action MFG:
- Discrete state and action spaces
- Q-learning with Nash equilibrium computation
- Value iteration for MFG

**Run**:
```bash
python examples/basic/nash_q_learning_demo.py
```

#### `continuous_action_ddpg_demo.py` - Deep Deterministic Policy Gradient
**Difficulty**: ⭐⭐⭐ Advanced
**Concepts**: Continuous control, actor-critic, DDPG
**Paradigm**: Reinforcement Learning

DDPG for continuous action MFG:
- Continuous action spaces
- Actor-critic architecture
- Off-policy learning with replay buffer
- Deterministic policy gradients

**Run**:
```bash
python examples/basic/continuous_action_ddpg_demo.py
```

---

### 🔀 Multi-Paradigm

#### `multi_paradigm_comparison.py` - Paradigm Comparison
**Difficulty**: ⭐⭐⭐ Advanced
**Concepts**: All paradigms, performance comparison
**Paradigms**: Numerical, Neural, Optimization, RL

Compares all four MFG paradigms on the same problem:
- **Numerical**: Finite difference methods
- **Neural**: PINN
- **Optimization**: JKO scheme
- **Reinforcement Learning**: Actor-Critic

Demonstrates when to use each paradigm.

**Run**:
```bash
python examples/basic/multi_paradigm_comparison.py
```

---

## Dependencies

### Core (Always Available)
- numpy, scipy, matplotlib
- mfg_pde (this package)

### Optional (Paradigm-Specific)
- **Neural**: `torch` (PyTorch)
- **Optimization**: `POT` (Python Optimal Transport)
- **RL**: `gymnasium`, `stable-baselines3`

### Installation
```bash
# All paradigms
pip install mfg_pde[all]

# Or individually
pip install mfg_pde[neural]      # Neural paradigm
pip install mfg_pde[optimization] # Optimization paradigm
pip install mfg_pde[rl]          # RL paradigm
```

---

## Example Difficulty Guide

- ⭐ **Easiest**: Basic MFG setup, good starting point
- ⭐⭐ **Moderate**: Introduces one new concept clearly
- ⭐⭐⭐ **Advanced**: Multiple concepts, requires paradigm-specific knowledge

---

## Learning Path

### Recommended Order for New Users:

1. **Start**: `lq_mfg_demo.py` - Understand basic MFG structure
2. **Classic Games**:
   - `el_farol_bar_demo.py` - Discrete coordination
   - `towel_beach_demo.py` - Spatial competition
3. **Choose Specialization**:
   - **Stochastic**: `common_noise_lq_demo.py`
   - **Neural**: `adaptive_pinn_demo.py`
   - **RL**: `rl_intro_comparison.py`
4. **Advanced**: `multi_paradigm_comparison.py` - See all paradigms together

---

## Output Files

All examples save output to `examples/outputs/basic/`:
- **Plots**: PNG files with visualizations
- **Logs**: Structured logging with solver progress
- **Data**: Convergence history (when applicable)

**Note**: Output directory is `.gitignored` by default. Use `examples/outputs/reference/` for tracked reference outputs.

---

## Theory References

Each example links to comprehensive theory documentation:
- **Coordination games**: `docs/theory/coordination_games_mfg.md`
- **Spatial competition**: `docs/theory/spatial_competition_mfg.md`
- **Stochastic MFG**: `docs/theory/stochastic_mfg_common_noise.md`
- **Mathematical foundations**: `docs/theory/mathematical_background.md`
- **Convergence criteria**: `docs/theory/convergence_criteria.md`

See `docs/DOCUMENTATION_INDEX.md` for complete documentation map.

---

## Contributing Examples

When adding new basic examples:

1. **Focus**: One concept per example
2. **Documentation**: Comprehensive docstring with theory references
3. **Simplicity**: Keep under 300 lines
4. **Output**: Save to `examples/outputs/basic/`
5. **README**: Update this file with example description

See `CLAUDE.md` for coding standards and example structure guidelines.

---

**Last Updated**: October 8, 2025
**Total Examples**: 12
**Coverage**: All 4 MFG paradigms (Numerical, Neural, Optimization, RL) + Acceleration techniques
