# Reinforcement Learning Approaches

**Status**: [PLANNED] Abstraction layer for future algorithm organization
**Purpose**: Base classes for mathematical approach categorization
**Current State**: Placeholder structure, not yet populated

## Overview

This directory is designed to contain **abstract base classes** that categorize reinforcement learning algorithms by their **mathematical approach** rather than specific implementation details.

## Design Philosophy

### Three Mathematical Approaches

Reinforcement learning algorithms can be classified into three fundamental approaches based on what they learn and optimize:

1. **Value-Based Methods** (`value_based/`)
   - **What they learn**: State-value function $V(s)$ or action-value function $Q(s,a)$
   - **Decision rule**: Greedy action selection from learned values
   - **Examples**: Q-learning, Nash-Q, DQN, value iteration
   - **MFG context**: Learn Nash equilibrium value functions

2. **Policy-Based Methods** (`policy_based/`)
   - **What they learn**: Policy $\pi(a|s)$ directly
   - **Optimization**: Gradient ascent on expected return
   - **Examples**: REINFORCE, Policy Gradient, Mean Field RL (Yang et al.)
   - **MFG context**: Direct policy optimization in mean field limit

3. **Actor-Critic Methods** (`actor_critic/`)
   - **What they learn**: Both policy (actor) and value function (critic)
   - **Synergy**: Critic reduces variance of policy gradient
   - **Examples**: DDPG, TD3, SAC, A2C/A3C
   - **MFG context**: Continuous control with mean field coupling

## Intended Structure

```
approaches/
├── value_based/
│   ├── __init__.py          # Export ValueBasedBase
│   └── base.py              # Abstract base class
├── policy_based/
│   ├── __init__.py          # Export PolicyGradientBase
│   └── base.py              # Abstract base class
└── actor_critic/
    ├── __init__.py          # Export ActorCriticBase
    └── base.py              # Abstract base class
```

## Intended Usage Pattern

### Abstract Base Classes (Future Implementation)

```python
# mfg_pde/alg/reinforcement/approaches/value_based/base.py
from abc import ABC, abstractmethod

class ValueBasedBase(ABC):
    """
    Base class for value-based RL methods.

    These methods learn value functions Q(s,a) or V(s) and derive
    policies by greedy action selection.
    """

    @abstractmethod
    def compute_q_values(self, state, population):
        """Compute Q(s,a) for all actions."""
        pass

    @abstractmethod
    def select_action(self, q_values, exploration_rate):
        """Select action using epsilon-greedy or similar."""
        pass

    @abstractmethod
    def update_q_function(self, transition, population):
        """Update Q-function from experience."""
        pass
```

### Algorithm Implementation (Future)

```python
# mfg_pde/alg/reinforcement/algorithms/nash_q.py
from ..approaches.value_based import ValueBasedBase

class NashQSolver(ValueBasedBase):
    """
    Nash Q-Learning for Mean Field Games.

    Inherits value-based interface and implements Nash equilibrium
    computation in finite-population setting.
    """

    def compute_q_values(self, state, population):
        # Implementation specific to Nash-Q
        return self.q_network(state, population)
```

## Current Algorithm Organization

**Algorithms are currently implemented directly in** `mfg_pde/alg/reinforcement/algorithms/`:

- `ddpg.py` - Deep Deterministic Policy Gradient (actor-critic)
- `td3.py` - Twin Delayed DDPG (actor-critic)
- `sac.py` - Soft Actor-Critic (actor-critic)

These algorithms do **not yet use** the `approaches/` abstraction layer.

## Migration Strategy

When `approaches/` is populated, existing algorithms can be refactored to inherit from approach base classes:

1. **Create base classes** in `approaches/*/base.py`
2. **Define common interfaces** for each approach category
3. **Refactor existing algorithms** to inherit from appropriate base
4. **Add new algorithms** using the established abstractions

## Benefits of This Structure

### For Users
- **Conceptual clarity**: Understand algorithm families by mathematical approach
- **Easy discovery**: Find algorithms by learning paradigm
- **Consistent interface**: All value-based methods share common API

### For Developers
- **Code reuse**: Common functionality in base classes
- **Enforced consistency**: Abstract methods ensure complete implementation
- **Clear extension points**: Adding new algorithms is straightforward

### For Research
- **Fair comparison**: Compare algorithms within same approach category
- **Hybrid methods**: Easier to identify opportunities for combining approaches
- **Educational value**: Structure reflects RL theory taxonomy

## Relationship to Algorithm Reorganization Plan

This structure is part of the broader **Algorithm Reorganization Plan** (see `docs/planning/roadmaps/ALGORITHM_REORGANIZATION_PLAN.md`), which organizes all MFG algorithms by mathematical paradigm:

```
mfg_pde/alg/
├── numerical/        # Classical numerical methods
├── optimization/     # Direct optimization (variational, optimal transport)
├── neural/           # Neural network methods (PINN, operators)
└── reinforcement/    # RL methods
    ├── core/         # Shared RL infrastructure
    ├── algorithms/   # Specific algorithm implementations
    ├── approaches/   # Mathematical approach abstractions (THIS DIRECTORY)
    └── environments/ # MFG-specific RL environments
```

## When to Populate This Directory

**Populate when**:
- Adding algorithms from multiple approach categories (e.g., both value-based and policy-based)
- Implementing common functionality that applies to all algorithms in a category
- Refactoring existing algorithms to share common interfaces

**Don't populate prematurely**:
- If only one approach category is used (e.g., only actor-critic methods currently)
- Before interface requirements are clear from actual algorithm implementations
- Just to have "complete" directory structure (YAGNI principle)

## Current Recommendation

**Keep empty for now**. The current Phase 3.x implementation focuses exclusively on actor-critic methods (DDPG, TD3, SAC). Populating `approaches/` becomes valuable when:

1. Value-based algorithms are added (Nash-Q, Q-learning variants)
2. Policy-based algorithms are added (REINFORCE, MFRL)
3. Clear common interfaces emerge from multiple implementations

## References

- **Algorithm Reorganization Plan**: `docs/planning/roadmaps/ALGORITHM_REORGANIZATION_PLAN.md`
- **RL Taxonomy**: Sutton & Barto, "Reinforcement Learning: An Introduction"
- **MFG-RL**: Yang et al., "Mean Field Multi-Agent Reinforcement Learning"

---

**Last Updated**: 2025-10-04
**Status**: Placeholder awaiting future population
