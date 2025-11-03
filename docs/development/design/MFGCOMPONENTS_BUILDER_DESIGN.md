# MFGComponents Builder Functions Design

**Date**: 2025-11-03
**Purpose**: Helper functions for common MFGComponents configurations
**Status**: Design specification (implementation pending)

---

## Motivation

With MFGComponents supporting 6+ formulations and 37+ optional fields, users need:

1. **Quick start**: Easy creation of common configurations
2. **Best practices**: Sensible defaults based on problem type
3. **Composability**: Combine multiple aspects (e.g., neural + stochastic)
4. **Discoverability**: Clear entry points for each formulation

---

## Design Principles

### **1. Formulation-Specific Builders**

Each MFG formulation gets a builder function:
- `standard_mfg_components()` - HJB-FP
- `neural_mfg_components()` - PINN, Deep BSDE
- `rl_mfg_components()` - RL-based MFG
- `network_mfg_components()` - Graph MFG
- `variational_mfg_components()` - Lagrangian
- `stochastic_mfg_components()` - Common noise

### **2. Composable Design**

Builders return MFGComponents that can be merged:
```python
base = standard_mfg_components(hamiltonian=my_H)
stochastic = stochastic_mfg_components(noise_intensity=0.1)
combined = merge_components(base, stochastic)
```

### **3. Sensible Defaults**

Builders provide best-practice defaults:
- Neural: Standard loss weights (PDE=1.0, IC=10.0, BC=10.0)
- RL: Reasonable action space bounds
- AMR: Typical refinement levels (3-5)

### **4. Clear Documentation**

Each builder explains:
- What it configures
- What parameters are required vs optional
- What numerical methods work best with this configuration

---

## Builder Functions

### **Standard HJB-FP Components**

```python
def standard_mfg_components(
    hamiltonian: Callable | None = None,
    hamiltonian_dm: Callable | None = None,
    potential: Callable | None = None,
    initial_density: Callable | None = None,
    final_value: Callable | None = None,
    boundary_conditions: BoundaryConditions | None = None,
    coupling: Callable | None = None,
) -> MFGComponents:
    """
    Create MFGComponents for standard HJB-FP formulation.

    This is the most common MFG formulation, using Hamiltonian H(x,m,p,t)
    to define agent dynamics and coupling through density m.

    Parameters
    ----------
    hamiltonian : Callable, optional
        H(x, m, p, t) -> float. If None, defaults to quadratic H = 0.5|p|^2.
    hamiltonian_dm : Callable, optional
        dH/dm(x, m, p, t) -> float. If None, numerical differentiation used.
    potential : Callable, optional
        V(x, t) -> float. External forces/obstacles.
    initial_density : Callable, optional
        m_0(x) -> float. Agent distribution at t=0.
    final_value : Callable, optional
        u_T(x) -> float. Terminal cost/reward.
    boundary_conditions : BoundaryConditions, optional
        Domain boundary behavior. If None, defaults to Neumann.
    coupling : Callable, optional
        Additional coupling terms F(x, m, t).

    Returns
    -------
    components : MFGComponents
        Configured for standard HJB-FP MFG.

    Examples
    --------
    >>> # Quadratic Hamiltonian with congestion
    >>> def H(x, m, p, t):
    ...     return 0.5 * np.sum(p**2) + 5.0 * m  # Congestion penalty
    >>>
    >>> components = standard_mfg_components(
    ...     hamiltonian=H,
    ...     initial_density=lambda x: np.exp(-np.sum(x**2))
    ... )

    Notes
    -----
    Best solved with:
    - HJB: HJBFDMSolver, HJBWENOSolver, HJBGFDMSolver
    - FP: FPFDMSolver, FPParticleSolver
    """
    return MFGComponents(
        hamiltonian_func=hamiltonian,
        hamiltonian_dm_func=hamiltonian_dm,
        potential_func=potential,
        initial_density_func=initial_density,
        final_value_func=final_value,
        boundary_conditions=boundary_conditions,
        coupling_func=coupling,
    )
```

### **Neural Network MFG Components**

```python
def neural_mfg_components(
    hamiltonian: Callable,
    architecture: dict[str, Any] | None = None,
    loss_weights: dict[str, float] | None = None,
    physics_loss: Callable | None = None,
    initial_density: Callable | None = None,
    final_value: Callable | None = None,
    boundary_conditions: BoundaryConditions | None = None,
) -> MFGComponents:
    """
    Create MFGComponents for neural network-based MFG solvers.

    Configures PINN (Physics-Informed Neural Networks) or Deep BSDE methods
    for solving MFG systems.

    Parameters
    ----------
    hamiltonian : Callable
        H(x, m, p, t) -> float. Required - defines PDE to learn.
    architecture : dict, optional
        Neural network architecture specification.
        Default: {'layers': [64, 64, 64], 'activation': 'tanh'}
    loss_weights : dict, optional
        Weights for loss components: {'pde': float, 'ic': float, 'bc': float}.
        Default: {'pde': 1.0, 'ic': 10.0, 'bc': 10.0}
    physics_loss : Callable, optional
        Custom physics-informed loss function.
    initial_density : Callable, optional
        m_0(x) -> float. Used in IC loss.
    final_value : Callable, optional
        u_T(x) -> float. Used in terminal condition loss.
    boundary_conditions : BoundaryConditions, optional
        Used in BC loss.

    Returns
    -------
    components : MFGComponents
        Configured for neural MFG solvers.

    Examples
    --------
    >>> def H(x, m, p, t):
    ...     return 0.5 * np.sum(p**2) + 2.0 * m
    >>>
    >>> components = neural_mfg_components(
    ...     hamiltonian=H,
    ...     architecture={'layers': [128, 128, 128], 'activation': 'relu'},
    ...     loss_weights={'pde': 1.0, 'ic': 20.0, 'bc': 20.0}
    ... )

    Notes
    -----
    Best solved with:
    - PINNSolver (physics-informed neural network)
    - DeepBSDESolver (backward SDE approach)

    The Hamiltonian is used to construct the PDE residual in the loss function.
    """
    # Default architecture
    if architecture is None:
        architecture = {
            'layers': [64, 64, 64],
            'activation': 'tanh',
            'initialization': 'xavier'
        }

    # Default loss weights (emphasize IC/BC)
    if loss_weights is None:
        loss_weights = {
            'pde': 1.0,
            'ic': 10.0,
            'bc': 10.0
        }

    return MFGComponents(
        hamiltonian_func=hamiltonian,
        neural_architecture=architecture,
        loss_weights=loss_weights,
        physics_loss_func=physics_loss,
        initial_density_func=initial_density,
        final_value_func=final_value,
        boundary_conditions=boundary_conditions,
    )
```

### **Reinforcement Learning MFG Components**

```python
def rl_mfg_components(
    reward: Callable,
    action_space_bounds: list[tuple[float, float]],
    terminal_reward: Callable | None = None,
    action_constraints: list[Callable] | None = None,
    observation_func: Callable | None = None,
    population_coupling: float = 0.0,
    initial_density: Callable | None = None,
) -> MFGComponents:
    """
    Create MFGComponents for RL-based MFG solvers.

    Configures reinforcement learning approaches (PPO, Actor-Critic) for
    solving MFG problems via multi-agent learning.

    Parameters
    ----------
    reward : Callable
        r(state, action, density, t) -> float. Agent reward function.
    action_space_bounds : list of tuples
        [(a_min, a_max), ...] for each action dimension.
    terminal_reward : Callable, optional
        r_T(state) -> float. Terminal reward.
    action_constraints : list of Callable, optional
        [g1, g2, ...] where gi(state, action) <= 0.
    observation_func : Callable, optional
        Maps full state to agent observation.
    population_coupling : float, default=0.0
        Strength of mean-field coupling term.
    initial_density : Callable, optional
        m_0(x) -> float. Initial agent distribution.

    Returns
    -------
    components : MFGComponents
        Configured for RL MFG solvers.

    Examples
    --------
    >>> # LQ MFG via RL
    >>> def reward(s, a, m, t):
    ...     return -(a**2 + 5.0 * m**2)  # Quadratic cost + congestion
    >>>
    >>> components = rl_mfg_components(
    ...     reward=reward,
    ...     action_space_bounds=[(-1.0, 1.0)],
    ...     population_coupling=5.0
    ... )

    Notes
    -----
    Best solved with:
    - PPOSolver (Proximal Policy Optimization)
    - ActorCriticSolver (A2C/A3C variants)

    The reward function is used to train agents to learn equilibrium strategies.
    """
    return MFGComponents(
        reward_func=reward,
        terminal_reward_func=terminal_reward,
        action_space_bounds=action_space_bounds,
        action_constraints=action_constraints,
        observation_func=observation_func,
        population_coupling_strength=population_coupling,
        initial_density_func=initial_density,
    )
```

### **Network/Graph MFG Components**

```python
def network_mfg_components(
    network_geometry: Any,
    node_interaction: Callable | None = None,
    edge_interaction: Callable | None = None,
    edge_cost: Callable | None = None,
    trajectory_cost: Callable | None = None,
    initial_density: Callable | None = None,
    terminal_cost: Callable | None = None,
) -> MFGComponents:
    """
    Create MFGComponents for network/graph MFG.

    Configures MFG on discrete domains (graphs, networks) where agents move
    between nodes along edges.

    Parameters
    ----------
    network_geometry : NetworkGeometry
        Graph structure defining nodes and edges.
    node_interaction : Callable, optional
        f_node(node_id, density, t) -> float. Interaction at nodes.
    edge_interaction : Callable, optional
        f_edge(edge_id, density, t) -> float. Interaction along edges.
    edge_cost : Callable, optional
        c(edge_id, density, t) -> float. Cost to traverse edge.
    trajectory_cost : Callable, optional
        L(path, density, t) -> float. Cost along entire path.
    initial_density : Callable, optional
        m_0(node_id) -> float. Initial node occupancy.
    terminal_cost : Callable, optional
        g(node_id) -> float. Terminal cost at nodes.

    Returns
    -------
    components : MFGComponents
        Configured for network MFG.

    Examples
    --------
    >>> from mfg_pde.geometry import NetworkGeometry
    >>>
    >>> # Simple 3-node network
    >>> network = NetworkGeometry(
    ...     num_nodes=3,
    ...     edges=[(0, 1), (1, 2), (2, 0)]
    ... )
    >>>
    >>> components = network_mfg_components(
    ...     network_geometry=network,
    ...     edge_cost=lambda edge_id, m, t: 1.0 + 2.0 * m  # Congestion
    ... )

    Notes
    -----
    Best solved with:
    - NetworkHJBSolver
    - NetworkFPSolver
    - TrajectoryBasedSolver (for trajectory costs)
    """
    return MFGComponents(
        network_geometry=network_geometry,
        node_interaction_func=node_interaction,
        edge_interaction_func=edge_interaction,
        edge_cost_func=edge_cost,
        trajectory_cost_func=trajectory_cost,
        initial_density_func=initial_density,
        terminal_cost_func=terminal_cost,
    )
```

### **Variational/Lagrangian MFG Components**

```python
def variational_mfg_components(
    lagrangian: Callable,
    lagrangian_dx: Callable | None = None,
    lagrangian_dv: Callable | None = None,
    lagrangian_dm: Callable | None = None,
    terminal_cost: Callable | None = None,
    terminal_cost_dx: Callable | None = None,
    state_constraints: list[Callable] | None = None,
    velocity_constraints: list[Callable] | None = None,
    initial_density: Callable | None = None,
) -> MFGComponents:
    """
    Create MFGComponents for variational/Lagrangian MFG.

    Configures optimization-based MFG formulation using Lagrangian
    L(t, x, v, m) instead of Hamiltonian.

    Parameters
    ----------
    lagrangian : Callable
        L(t, x, v, m) -> float. Running cost function.
    lagrangian_dx : Callable, optional
        ∂L/∂x. If None, numerical differentiation used.
    lagrangian_dv : Callable, optional
        ∂L/∂v. If None, numerical differentiation used.
    lagrangian_dm : Callable, optional
        ∂L/∂m. If None, numerical differentiation used.
    terminal_cost : Callable, optional
        g(x) -> float. Terminal cost.
    terminal_cost_dx : Callable, optional
        ∂g/∂x. If None, numerical differentiation used.
    state_constraints : list of Callable, optional
        [c1, c2, ...] where ci(t, x) <= 0.
    velocity_constraints : list of Callable, optional
        [h1, h2, ...] where hi(t, x, v) <= 0.
    initial_density : Callable, optional
        m_0(x) -> float.

    Returns
    -------
    components : MFGComponents
        Configured for variational MFG.

    Examples
    --------
    >>> # Quadratic Lagrangian
    >>> def L(t, x, v, m):
    ...     return 0.5 * np.sum(v**2) + 2.0 * m
    >>>
    >>> def L_dv(t, x, v, m):
    ...     return v  # Analytical derivative
    >>>
    >>> components = variational_mfg_components(
    ...     lagrangian=L,
    ...     lagrangian_dv=L_dv
    ... )

    Notes
    -----
    Best solved with:
    - VariationalSolver (direct optimization)
    - LagrangianRelaxationSolver
    - Can be converted to Hamiltonian form via Legendre transform
    """
    return MFGComponents(
        lagrangian_func=lagrangian,
        lagrangian_dx_func=lagrangian_dx,
        lagrangian_dv_func=lagrangian_dv,
        lagrangian_dm_func=lagrangian_dm,
        terminal_cost_func=terminal_cost,
        terminal_cost_dx_func=terminal_cost_dx,
        state_constraints=state_constraints,
        velocity_constraints=velocity_constraints,
        initial_density_func=initial_density,
    )
```

### **Stochastic MFG Components**

```python
def stochastic_mfg_components(
    base_components: MFGComponents,
    noise_intensity: float = 0.1,
    common_noise: Callable | None = None,
    idiosyncratic_noise: Callable | None = None,
    correlation_matrix: NDArray | None = None,
) -> MFGComponents:
    """
    Add stochastic noise to existing MFG components.

    Augments a base MFG formulation with common and/or idiosyncratic noise
    for stochastic MFG problems.

    Parameters
    ----------
    base_components : MFGComponents
        Base formulation (standard, network, variational, etc.)
    noise_intensity : float, default=0.1
        Diffusion coefficient σ.
    common_noise : Callable, optional
        W(t) -> float. Noise affecting all agents equally.
    idiosyncratic_noise : Callable, optional
        Z_i(t) -> float. Individual agent noise.
    correlation_matrix : NDArray, optional
        Correlations between noise dimensions.

    Returns
    -------
    components : MFGComponents
        Base components with stochastic noise added.

    Examples
    --------
    >>> # Start with standard MFG
    >>> base = standard_mfg_components(
    ...     hamiltonian=lambda x, m, p, t: 0.5 * p**2 + m
    ... )
    >>>
    >>> # Add common noise
    >>> stochastic = stochastic_mfg_components(
    ...     base_components=base,
    ...     noise_intensity=0.2,
    ...     common_noise=lambda t: np.sin(2 * np.pi * t)
    ... )

    Notes
    -----
    Stochastic MFG can be solved with:
    - Standard solvers with σ > 0 (adds diffusion term)
    - Particle methods (naturally handle stochasticity)
    - Common noise requires specialized coupling treatment
    """
    # Start with base components
    components_dict = vars(base_components).copy()

    # Add stochastic fields
    components_dict['noise_intensity'] = noise_intensity
    components_dict['common_noise_func'] = common_noise
    components_dict['idiosyncratic_noise_func'] = idiosyncratic_noise
    components_dict['correlation_matrix'] = correlation_matrix

    return MFGComponents(**components_dict)
```

---

## Composition Utilities

### **Component Merging**

```python
def merge_components(
    *components_list: MFGComponents,
    priority: str = 'last'
) -> MFGComponents:
    """
    Merge multiple MFGComponents into one.

    Combines fields from multiple component objects. Useful for composing
    different aspects (e.g., base formulation + neural + stochastic).

    Parameters
    ----------
    *components_list : MFGComponents
        Components to merge.
    priority : {'last', 'first'}, default='last'
        When fields conflict, use 'last' or 'first' non-None value.

    Returns
    -------
    merged : MFGComponents
        Combined components.

    Examples
    --------
    >>> base = standard_mfg_components(hamiltonian=my_H)
    >>> neural = neural_mfg_components(hamiltonian=my_H)
    >>> stochastic = stochastic_mfg_components(base, noise_intensity=0.1)
    >>>
    >>> # Combine all aspects
    >>> merged = merge_components(base, neural, stochastic)
    """
    merged_dict = {}

    if priority == 'last':
        # Later components override earlier
        for comp in components_list:
            for key, value in vars(comp).items():
                if value is not None:
                    merged_dict[key] = value

    elif priority == 'first':
        # Earlier components take precedence
        for comp in components_list:
            for key, value in vars(comp).items():
                if value is not None and key not in merged_dict:
                    merged_dict[key] = value

    else:
        raise ValueError(f"priority must be 'last' or 'first', got {priority}")

    return MFGComponents(**merged_dict)
```

---

## Usage Examples

### **Example 1: Standard MFG**

```python
from mfg_pde import MFGProblem, standard_mfg_components

# Configure environment
components = standard_mfg_components(
    hamiltonian=lambda x, m, p, t: 0.5 * p**2 + 2.0 * m,
    initial_density=lambda x: np.exp(-x**2)
)

# Create problem
problem = MFGProblem(
    xmin=0, xmax=1, Nx=100,
    T=1.0, Nt=50,
    components=components
)

# Solve with modular approach
from mfg_pde.solvers import HJBFDMSolver, FPParticleSolver, FixedPointIterator

hjb = HJBFDMSolver(problem)
fp = FPParticleSolver(problem, num_particles=5000)
solver = FixedPointIterator(problem, hjb, fp)

result = solver.solve()
```

### **Example 2: Neural MFG**

```python
from mfg_pde import MFGProblem, neural_mfg_components

# Configure neural environment
components = neural_mfg_components(
    hamiltonian=lambda x, m, p, t: 0.5 * np.sum(p**2) + m,
    architecture={'layers': [128, 128, 128], 'activation': 'relu'},
    loss_weights={'pde': 1.0, 'ic': 20.0, 'bc': 20.0}
)

problem = MFGProblem(
    xmin=0, xmax=1, Nx=100,
    T=1.0, Nt=50,
    components=components
)

# Solve with neural solver
from mfg_pde.solvers import PINNSolver

solver = PINNSolver(problem)
result = solver.solve(epochs=10000)
```

### **Example 3: Composed MFG**

```python
from mfg_pde import (
    MFGProblem,
    standard_mfg_components,
    stochastic_mfg_components,
    merge_components
)

# Base formulation
base = standard_mfg_components(
    hamiltonian=lambda x, m, p, t: 0.5 * p**2 + m
)

# Add stochasticity
stochastic = stochastic_mfg_components(
    base_components=base,
    noise_intensity=0.2,
    common_noise=lambda t: 0.1 * np.sin(2 * np.pi * t)
)

# Use combined components
problem = MFGProblem(
    xmin=0, xmax=1, Nx=100,
    T=1.0, Nt=50,
    components=stochastic
)
```

---

## Implementation Location

```
mfg_pde/
  core/
    mfg_problem.py         # MFGComponents dataclass (already exists)
    component_builders.py  # NEW: Builder functions
```

---

## Testing Strategy

```python
# tests/unit/test_component_builders.py

def test_standard_mfg_builder():
    """Standard builder creates valid components."""
    components = standard_mfg_components(
        hamiltonian=lambda x, m, p, t: 0.5 * p**2
    )
    assert components.hamiltonian_func is not None
    assert components.validate() == []  # No warnings


def test_neural_mfg_builder_defaults():
    """Neural builder provides sensible defaults."""
    components = neural_mfg_components(
        hamiltonian=lambda x, m, p, t: 0.5 * p**2
    )
    assert components.neural_architecture is not None
    assert components.loss_weights == {'pde': 1.0, 'ic': 10.0, 'bc': 10.0}


def test_rl_mfg_builder():
    """RL builder requires reward and action space."""
    components = rl_mfg_components(
        reward=lambda s, a, m, t: -a**2,
        action_space_bounds=[(-1.0, 1.0)]
    )
    assert components.reward_func is not None
    assert components.action_space_bounds == [(-1.0, 1.0)]
    assert components.validate() == []  # No warnings


def test_stochastic_composition():
    """Stochastic components compose with base."""
    base = standard_mfg_components(
        hamiltonian=lambda x, m, p, t: 0.5 * p**2
    )
    stochastic = stochastic_mfg_components(
        base_components=base,
        noise_intensity=0.2
    )
    assert stochastic.hamiltonian_func is not None  # From base
    assert stochastic.noise_intensity == 0.2  # Added


def test_merge_components_last_priority():
    """Merge with last priority uses later values."""
    comp1 = standard_mfg_components(hamiltonian=lambda: 1)
    comp2 = standard_mfg_components(hamiltonian=lambda: 2)
    merged = merge_components(comp1, comp2, priority='last')
    # Should use comp2's hamiltonian
    assert merged.hamiltonian_func() == 2
```

---

## Documentation Updates

After implementation, update:
1. **User guide**: Add "Using Component Builders" section
2. **Examples**: Show builder usage in basic examples
3. **API reference**: Document all builder functions
4. **Migration guide**: Show how builders simplify existing code

---

## Benefits

1. **Lower barrier to entry**: Builders make it easier to get started
2. **Best practices**: Encode expert knowledge in defaults
3. **Composability**: Mix and match formulations easily
4. **Discoverability**: Clear entry points for each MFG type
5. **Consistency**: Standardized way to create configurations

---

## Implementation Priority

**Phase 1: Core Builders** (Short-term)
- `standard_mfg_components()`
- `neural_mfg_components()`
- `rl_mfg_components()`
- `stochastic_mfg_components()`

**Phase 2: Advanced Builders** (Medium-term)
- `network_mfg_components()`
- `variational_mfg_components()`

**Phase 3: Composition Utilities** (Medium-term)
- `merge_components()`
- Component validation helpers

---

## Backward Compatibility

**Impact**: ✅ 100% backward compatible

- Builders are **convenience functions**, not required
- Direct MFGComponents construction still works
- Existing code unchanged

---

## Summary

**Builder Design**: Formulation-specific helper functions with sensible defaults

**Key Features**:
- ✅ Formulation-specific (standard, neural, RL, network, variational)
- ✅ Composable (combine multiple aspects)
- ✅ Best-practice defaults (loss weights, architectures, etc.)
- ✅ Well-documented (examples, recommendations, solver compatibility)

**Implementation**: Create `mfg_pde/core/component_builders.py` module

**Benefits**: Easier to get started, encodes expert knowledge, promotes best practices

---

**Last Updated**: 2025-11-03
**Status**: Design complete, ready for implementation
**Next Steps**: Implement Phase 1 builders (standard, neural, RL, stochastic)
