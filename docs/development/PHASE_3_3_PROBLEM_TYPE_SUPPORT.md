# Phase 3.3: Comprehensive Problem Type Support

**Date**: 2025-11-03
**Status**: Design Extension
**Related**: PHASE_3_3_FACTORY_INTEGRATION_DESIGN.md

---

## Problem Type Taxonomy

### Existing Problem Classes in Repository

1. **`MFGProblem`** (`mfg_pde/core/mfg_problem.py`)
   - Base unified problem class
   - Standard HJB-FP formulation
   - Supports n-D spatial domains
   - Uses `MFGComponents` for custom definitions

2. **`NetworkMFGProblem`** (`mfg_pde/core/network_mfg_problem.py`)
   - Extends `MFGProblem`
   - Discrete network/graph domains
   - Supports Lagrangian formulations (ArXiv 2207.10908v3)
   - Uses `NetworkMFGComponents`

3. **`VariationalMFGProblem`** (`mfg_pde/core/variational_mfg_problem.py`)
   - Standalone (not extending MFGProblem)
   - Lagrangian/variational formulation
   - Direct optimization perspective
   - Uses `VariationalMFGComponents`

4. **`StochasticMFGProblem`** (`mfg_pde/core/stochastic/stochastic_problem.py`)
   - Extends `MFGProblem`
   - Common noise formulation
   - Stochastic differential equations
   - Uses standard `MFGComponents` + noise

5. **`HighDimMFGProblem`** (`mfg_pde/core/highdim_mfg_problem.py`)
   - Abstract base class
   - High-dimensional problems (d > 3)
   - Grid-based and meshfree approaches
   - `GridBasedMFGProblem` concrete implementation

### Solution Methods (Not Problem Classes)

6. **Neural Network Methods** (`mfg_pde/alg/neural/`)
   - DGM (Deep Galerkin Method)
   - PINN (Physics-Informed Neural Networks)
   - Actor-Critic architectures
   - **Use standard problem classes**

7. **Reinforcement Learning Methods** (`mfg_pde/alg/reinforcement/`)
   - MFRL (Mean Field Reinforcement Learning)
   - Q-learning approaches
   - Policy gradient methods
   - **Use standard problem classes**

---

## Design Principle

### Problem vs Solution Method Separation

**Key Insight**: Neural and RL are **solution methods**, not problem types.

```python
# Problem: WHAT to solve (mathematical definition)
problem = MFGProblem(...)  # or NetworkMFGProblem, StochasticMFGProblem, etc.

# Solution Method: HOW to solve
config = ConfigBuilder()
    .solver_hjb(method="dgm")  # Deep Galerkin Method (neural)
    .solver_fp(method="actor_critic")  # Actor-Critic (RL)
    .build()

result = solve_mfg(problem, config=config)
```

**NOT**:
```python
# ❌ Wrong - mixing problem definition with solution method
problem = NeuralMFGProblem(...)  # No such thing!
```

---

## MFGProblem Extension Strategy

### Ensure MFGProblem Supports All Problem Types

`MFGProblem` already supports most types via `MFGComponents`:

```python
@dataclass
class MFGComponents:
    """Container for custom MFG problem definition."""

    # Core components (already supported)
    hamiltonian_func: Callable | None = None
    hamiltonian_dm_func: Callable | None = None
    hamiltonian_jacobian_func: Callable | None = None
    potential_func: Callable | None = None
    initial_density_func: Callable | None = None
    final_value_func: Callable | None = None
    boundary_conditions: BoundaryConditions | None = None
    coupling_func: Callable | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    description: str = "MFG Problem"
    problem_type: str = "mfg"
```

### Additions Needed for Complete Support

#### 1. Network Support in MFGProblem

Add network-specific components to `MFGComponents`:

```python
@dataclass
class MFGComponents:
    # ... existing fields ...

    # Network/graph support
    network_geometry: Any | None = None  # NetworkGeometry instance
    edge_cost_func: Callable | None = None  # Edge traversal costs
    node_interaction_func: Callable | None = None  # Node-based interactions

    # Lagrangian formulation (for NetworkMFGProblem compatibility)
    lagrangian_func: Callable | None = None  # L(node, velocity, m, t)
    trajectory_cost_func: Callable | None = None  # Cost along trajectories
```

#### 2. Variational/Lagrangian Support

Add Lagrangian components to `MFGComponents`:

```python
@dataclass
class MFGComponents:
    # ... existing fields ...

    # Variational/Lagrangian formulation
    lagrangian_func: Callable | None = None  # L(t, x, v, m)
    lagrangian_dx_func: Callable | None = None  # ∂L/∂x
    lagrangian_dv_func: Callable | None = None  # ∂L/∂v
    lagrangian_dm_func: Callable | None = None  # ∂L/∂m
    terminal_cost_func: Callable | None = None  # g(x)
    state_constraints: list[Callable] | None = None  # c(t, x) ≤ 0
    velocity_constraints: list[Callable] | None = None  # h(t, x, v) ≤ 0
```

#### 3. Stochastic/Common Noise Support

Add noise specification to `MFGComponents`:

```python
@dataclass
class MFGComponents:
    # ... existing fields ...

    # Stochastic formulation
    noise_intensity: float = 0.0  # Diffusion coefficient sigma
    common_noise_func: Callable | None = None  # Common noise W(t)
    idiosyncratic_noise_func: Callable | None = None  # Individual noise Z(t)
    correlation_matrix: NDArray | None = None  # Noise correlations
```

#### 4. High-Dimensional Support

`MFGProblem` already supports arbitrary dimensions via `spatial_bounds` and `spatial_discretization`.

---

## Factory Integration for All Types

### Unified Factory Structure

```python
# mfg_pde/factory/problem_factories.py

def create_mfg_problem(
    problem_type: Literal["standard", "network", "variational", "stochastic", "highdim"],
    components: MFGComponents | NetworkMFGComponents | VariationalMFGComponents,
    use_unified: bool = True,
    **kwargs
) -> MFGProblem | NetworkMFGProblem | VariationalMFGProblem | StochasticMFGProblem:
    """
    Unified factory for creating all types of MFG problems.

    Parameters
    ----------
    problem_type : str
        Type of MFG problem:
        - "standard": Basic HJB-FP formulation
        - "network": Network/graph MFG
        - "variational": Lagrangian formulation
        - "stochastic": Common noise MFG
        - "highdim": High-dimensional problems
    components : MFGComponents subclass
        Problem-specific components
    use_unified : bool
        If True, use unified MFGProblem with appropriate components
        If False, use specialized classes (backward compat)

    Returns
    -------
    MFGProblem or subclass
        Problem instance

    Examples
    --------
    # Standard MFG
    problem = create_mfg_problem(
        problem_type="standard",
        components=MFGComponents(...)
    )

    # Network MFG (unified)
    problem = create_mfg_problem(
        problem_type="network",
        components=NetworkMFGComponents(...),
        use_unified=True  # Returns MFGProblem with network components
    )

    # Network MFG (specialized)
    problem = create_mfg_problem(
        problem_type="network",
        components=NetworkMFGComponents(...),
        use_unified=False  # Returns NetworkMFGProblem
    )
    """
    if use_unified:
        # Use unified MFGProblem for all types
        return MFGProblem(components=components, **kwargs)
    else:
        # Use specialized classes (backward compat)
        if problem_type == "network":
            return NetworkMFGProblem(components=components, **kwargs)
        elif problem_type == "variational":
            return VariationalMFGProblem(components=components, **kwargs)
        elif problem_type == "stochastic":
            return StochasticMFGProblem(components=components, **kwargs)
        elif problem_type == "highdim":
            return GridBasedMFGProblem(components=components, **kwargs)
        else:
            return MFGProblem(components=components, **kwargs)
```

### Specialized Factories (Convenience)

```python
def create_network_problem(
    network_geometry: NetworkGeometry,
    components: NetworkMFGComponents,
    use_unified: bool = True,
    **kwargs
) -> MFGProblem | NetworkMFGProblem:
    """Create network MFG problem."""
    return create_mfg_problem(
        problem_type="network",
        components=components,
        network_geometry=network_geometry,
        use_unified=use_unified,
        **kwargs
    )

def create_variational_problem(
    components: VariationalMFGComponents,
    use_unified: bool = True,
    **kwargs
) -> MFGProblem | VariationalMFGProblem:
    """Create variational MFG problem."""
    return create_mfg_problem(
        problem_type="variational",
        components=components,
        use_unified=use_unified,
        **kwargs
    )

def create_stochastic_problem(
    components: MFGComponents,
    noise_intensity: float = 1.0,
    use_unified: bool = True,
    **kwargs
) -> MFGProblem | StochasticMFGProblem:
    """Create stochastic MFG problem with common noise."""
    components.noise_intensity = noise_intensity
    return create_mfg_problem(
        problem_type="stochastic",
        components=components,
        use_unified=use_unified,
        **kwargs
    )
```

---

## Problem Type Support Matrix

| Problem Type | MFGProblem Support | Specialized Class | Factory Support |
|:------------|:-------------------|:------------------|:----------------|
| Standard HJB-FP | ✅ Native | N/A | ✅ `create_mfg_problem()` |
| Network/Graph | ⚠️ Needs extension | `NetworkMFGProblem` | ✅ `create_network_problem()` |
| Variational | ⚠️ Needs extension | `VariationalMFGProblem` | ✅ `create_variational_problem()` |
| Stochastic | ⚠️ Needs extension | `StochasticMFGProblem` | ✅ `create_stochastic_problem()` |
| High-Dimensional | ✅ Native (via n-D) | `HighDimMFGProblem` (abstract) | ✅ `create_mfg_problem()` |
| Neural Methods | ✅ Via solver config | N/A (solution method) | N/A (use config) |
| RL Methods | ✅ Via solver config | N/A (solution method) | N/A (use config) |

**Legend**:
- ✅ Fully supported
- ⚠️ Partial support (needs MFGComponents extension)
- N/A Not applicable

---

## Implementation Tasks for Complete Support

### Task 1: Extend MFGComponents (1-2 hours)

Add fields to `MFGComponents` to support all problem types:

```python
# mfg_pde/core/mfg_problem.py

@dataclass
class MFGComponents:
    """Enhanced container for all MFG problem types."""

    # Core Hamiltonian components (existing)
    hamiltonian_func: Callable | None = None
    hamiltonian_dm_func: Callable | None = None
    hamiltonian_jacobian_func: Callable | None = None
    potential_func: Callable | None = None
    initial_density_func: Callable | None = None
    final_value_func: Callable | None = None
    boundary_conditions: BoundaryConditions | None = None
    coupling_func: Callable | None = None

    # Network/Graph support (NEW)
    network_geometry: Any | None = None
    edge_cost_func: Callable | None = None
    node_interaction_func: Callable | None = None
    edge_interaction_func: Callable | None = None

    # Variational/Lagrangian support (NEW)
    lagrangian_func: Callable | None = None
    lagrangian_dx_func: Callable | None = None
    lagrangian_dv_func: Callable | None = None
    lagrangian_dm_func: Callable | None = None
    terminal_cost_func: Callable | None = None
    terminal_cost_dx_func: Callable | None = None
    state_constraints: list[Callable] | None = None
    velocity_constraints: list[Callable] | None = None
    integral_constraints: list[Callable] | None = None

    # Stochastic support (NEW)
    noise_intensity: float = 0.0
    common_noise_func: Callable | None = None
    idiosyncratic_noise_func: Callable | None = None
    correlation_matrix: NDArray | None = None

    # Problem parameters (existing)
    parameters: dict[str, Any] = field(default_factory=dict)

    # Metadata (existing)
    description: str = "MFG Problem"
    problem_type: str = "mfg"  # Can be: "standard", "network", "variational", "stochastic", "highdim"
```

### Task 2: Add Problem Type Detection (30 min)

Add helper method to `MFGProblem`:

```python
class MFGProblem:
    # ... existing code ...

    def get_problem_type(self) -> str:
        """
        Detect problem type based on components.

        Returns
        -------
        str
            Problem type: "standard", "network", "variational", "stochastic", "highdim"
        """
        if self.components is None:
            return "standard"

        # Check for network
        if self.components.network_geometry is not None:
            return "network"

        # Check for variational
        if self.components.lagrangian_func is not None:
            return "variational"

        # Check for stochastic
        if self.components.noise_intensity > 0 or self.components.common_noise_func is not None:
            return "stochastic"

        # Check for high-dimensional
        if self.dimension > 3:
            return "highdim"

        return "standard"
```

### Task 3: Update Factory Functions (2-3 hours)

Implement the unified factory structure described above.

### Task 4: Add Validation (1 hour)

Add validation to ensure components are consistent with problem type:

```python
class MFGProblem:
    def __post_init__(self):
        """Validate components consistency."""
        if self.components is not None:
            problem_type = self.get_problem_type()

            if problem_type == "network" and self.components.network_geometry is None:
                warnings.warn(
                    "Detected network problem but network_geometry is None. "
                    "Provide network_geometry in components."
                )

            if problem_type == "variational" and self.components.lagrangian_func is None:
                raise ValueError(
                    "Variational problems require lagrangian_func in components."
                )
```

### Task 5: Documentation (1 hour)

Update documentation to show all problem types:

```python
"""
Examples
--------
# Standard MFG
problem = MFGProblem(...)

# Network MFG
components = MFGComponents(
    network_geometry=NetworkGeometry(...),
    edge_cost_func=edge_cost,
    problem_type="network"
)
problem = MFGProblem(components=components)

# Variational MFG
components = MFGComponents(
    lagrangian_func=L,
    terminal_cost_func=g,
    problem_type="variational"
)
problem = MFGProblem(components=components)

# Stochastic MFG
components = MFGComponents(
    noise_intensity=1.0,
    common_noise_func=W,
    problem_type="stochastic"
)
problem = MFGProblem(components=components)
"""
```

---

## Testing Strategy

### Unit Tests for Each Problem Type

```python
# tests/unit/test_mfg_problem_types.py

def test_standard_problem():
    """Test standard HJB-FP problem."""
    problem = MFGProblem(Nx=100, T=1.0)
    assert problem.get_problem_type() == "standard"

def test_network_problem():
    """Test network MFG problem."""
    components = MFGComponents(
        network_geometry=SimpleNetwork(...),
        problem_type="network"
    )
    problem = MFGProblem(components=components)
    assert problem.get_problem_type() == "network"

def test_variational_problem():
    """Test variational MFG problem."""
    components = MFGComponents(
        lagrangian_func=lambda t, x, v, m: 0.5 * v**2,
        problem_type="variational"
    )
    problem = MFGProblem(components=components)
    assert problem.get_problem_type() == "variational"

def test_stochastic_problem():
    """Test stochastic MFG problem."""
    components = MFGComponents(
        noise_intensity=1.0,
        problem_type="stochastic"
    )
    problem = MFGProblem(components=components)
    assert problem.get_problem_type() == "stochastic"

def test_highdim_problem():
    """Test high-dimensional problem."""
    problem = MFGProblem(
        spatial_bounds=[(0, 1)] * 5,  # 5D
        spatial_discretization=[10] * 5
    )
    assert problem.get_problem_type() == "highdim"
```

---

## Summary

To support all MFG problem types in the repository:

1. **Extend `MFGComponents`** with network, variational, and stochastic fields
2. **Add type detection** method to `MFGProblem`
3. **Create unified factory** with `problem_type` parameter
4. **Maintain backward compatibility** with specialized classes via `use_unified` flag
5. **Add validation** to ensure component consistency
6. **Update documentation** with examples for each type
7. **Add comprehensive tests** for all problem types

This ensures that `MFGProblem` can represent **all** problem types in the repository, while the factory provides a clean, unified interface for creating them.

**Timeline**: 1 day additional work on top of Phase 3.3.1

---

**Status**: Design Complete
**Next**: Implement during Phase 3.3.1 (Factory Integration)
