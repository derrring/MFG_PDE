# API Simplification - True 2-Level Architecture

**Date**: 2025-11-23
**Status**: ✅ IMPLEMENTED
**Impact**: Breaking changes (implemented immediately)

---

## Problem Statement

Current MFG_PDE API has **4 layers** of problem construction:

1. **Factory level**: `create_fast_solver()`, `create_accurate_solver()` (in `factory.py`)
2. **Problem factories**: `create_mfg_problem()` (redundant)
3. **Builder pattern**: `MFGProblemBuilder` (unnecessary abstraction)
4. **Direct construction**: `MFGProblem()`, `MFGComponents()` (expert level)

**Issue**: Layers 2 and 3 add complexity without clear benefit, violating the principle of **2-level architecture**:
- **Factory**: Pre-configured, batteries-included solutions
- **Expert**: Direct control, full flexibility

---

## Current State Analysis

### What Each API Serves

#### 1. `MFGComponents` ✅ KEEP
```python
@dataclass
class MFGComponents:
    hamiltonian_func: Callable | None = None
    hamiltonian_dm_func: Callable | None = None
    final_value_func: Callable | None = None
    # ...
```

**Purpose**: Clean data structure for passing custom functions to `MFGProblem`

**Usage** (Expert Level):
```python
components = MFGComponents(
    hamiltonian_func=my_H,
    hamiltonian_dm_func=my_dH_dm,
    final_value_func=my_terminal_condition,
)
problem = MFGProblem(spatial_bounds=..., components=components)
```

**Verdict**: ✅ **ESSENTIAL** - Clean separation of custom functions from grid/time parameters

---

#### 2. `MFGProblem` ✅ KEEP
```python
class MFGProblem:
    def __init__(
        self,
        spatial_bounds=None,
        spatial_discretization=None,
        T=None,
        Nt=None,
        components=None,
        # ...
    ):
```

**Purpose**: Core problem class for expert-level construction

**Usage** (Expert Level):
```python
problem = MFGProblem(
    spatial_bounds=[(0, 10), (0, 10)],
    spatial_discretization=[59, 59],
    T=2.0,
    Nt=40,
    sigma=0.5,
    components=components,
)
```

**Verdict**: ✅ **CORE CLASS** - Essential for expert-level control

---

#### 3. `MFGProblemBuilder` ❌ REMOVE
```python
class MFGProblemBuilder:
    def hamiltonian(self, H, dH_dm): ...
    def domain(self, xmin, xmax, Nx): ...
    def time(self, T, Nt): ...
    def build(self) -> MFGProblem: ...
```

**Purpose**: Fluent/builder pattern for "easier" construction

**Usage**:
```python
problem = (MFGProblemBuilder()
    .hamiltonian(my_H, my_dH_dm)
    .domain(0, 10, 100)
    .time(2.0, 40)
    .build())
```

**vs Direct**:
```python
components = MFGComponents(hamiltonian_func=my_H, hamiltonian_dm_func=my_dH_dm)
problem = MFGProblem(xmin=0, xmax=10, Nx=100, T=2.0, Nt=40, components=components)
```

**Analysis**:
- ❌ **No clear benefit**: Builder is ~140 lines but doesn't simplify anything
- ❌ **Method chaining is not simpler**: Same number of parameters, more indirection
- ❌ **Violates 2-level principle**: Creates middle layer between factory and expert
- ❌ **Rarely used**: Most users either use factory OR direct construction

**Verdict**: ❌ **REMOVE** - Unnecessary abstraction that adds cognitive load

---

#### 4. `create_mfg_problem()` ❌ REMOVE
```python
def create_mfg_problem(hamiltonian_func, hamiltonian_dm_func, **kwargs) -> MFGProblem:
    """Convenience function to create custom MFG problem."""
    components = MFGComponents(
        hamiltonian_func=hamiltonian_func,
        hamiltonian_dm_func=hamiltonian_dm_func,
    )
    return MFGProblem(components=components, **kwargs)
```

**Purpose**: "Convenience" wrapper around `MFGProblem`

**Usage**:
```python
problem = create_mfg_problem(
    hamiltonian_func=my_H,
    hamiltonian_dm_func=my_dH_dm,
    xmin=0, xmax=10, Nx=100, T=2.0, Nt=40
)
```

**vs Direct**:
```python
components = MFGComponents(hamiltonian_func=my_H, hamiltonian_dm_func=my_dH_dm)
problem = MFGProblem(xmin=0, xmax=10, Nx=100, T=2.0, Nt=40, components=components)
```

**Analysis**:
- ❌ **Redundant**: Does exactly what `MFGProblem` constructor does
- ❌ **Hides components concept**: Users should understand `MFGComponents`
- ❌ **Not actually more convenient**: One line longer but clearer
- ❌ **Violates 2-level principle**: Middle layer that shouldn't exist

**Verdict**: ❌ **REMOVE** - Redundant wrapper with no benefit

---

## Proposed 2-Level Architecture

### Level 1: Factory (Pre-configured Problems)

**Location**: `mfg_pde/factory/problems.py` (new module)

```python
# High-level factory functions for common problem types
from mfg_pde.factory.problems import (
    create_lq_problem,
    create_crowd_motion_problem,
    create_traffic_flow_problem,
)

# Example: Linear-Quadratic MFG
problem = create_lq_problem(
    domain_size=10.0,
    num_agents=1000,
    target_position=5.0,
    control_cost=1.0,
    congestion_weight=0.1,
)

# Example: 2D Crowd Motion
problem = create_crowd_motion_problem(
    room_size=(10.0, 10.0),
    exit_location=(10.0, 5.0),
    exit_width=1.5,
    num_agents=1000,
    diffusion=0.5,
)
```

**Characteristics**:
- ✅ Problem-specific parameters (domain_size, exit_width, etc.)
- ✅ Returns fully configured `MFGProblem`
- ✅ No need to understand `MFGComponents`, Hamiltonians, etc.
- ✅ "Just works" for common use cases

**Target Users**: Applied researchers, students, rapid prototyping

---

### Level 2: Expert (Direct Construction)

**Location**: `mfg_pde` (top-level imports)

```python
from mfg_pde import MFGProblem, MFGComponents

# Step 1: Define custom functions
def my_hamiltonian(x_idx, x_position, m_at_x, derivs, **kwargs):
    px = derivs.get((1, 0), 0.0)
    py = derivs.get((0, 1), 0.0)
    return 0.5 * (px**2 + py**2)

def my_dH_dm(x_idx, x_position, m_at_x, derivs, **kwargs):
    return 2.0 * m_at_x  # Quadratic congestion

def my_terminal_condition(x):
    return (x[0] - 10.0)**2 + (x[1] - 5.0)**2

# Step 2: Create components
components = MFGComponents(
    hamiltonian_func=my_hamiltonian,
    hamiltonian_dm_func=my_dH_dm,
    final_value_func=my_terminal_condition,
    description="Custom 2D Navigation Problem",
)

# Step 3: Create problem with full control
problem = MFGProblem(
    spatial_bounds=[(0.0, 10.0), (0.0, 10.0)],
    spatial_discretization=[59, 59],
    T=2.0,
    Nt=40,
    sigma=0.5,
    coupling_coefficient=0.1,
    components=components,
)
```

**Characteristics**:
- ✅ Full control over Hamiltonians, terminal conditions, BC
- ✅ Direct access to all `MFGProblem` parameters
- ✅ Clean separation: `MFGComponents` (custom functions) vs `MFGProblem` (grid/time)
- ✅ No hidden magic, explicit and debuggable

**Target Users**: Expert researchers, custom problems, publications

---

## Proposed Changes

### Files to Modify

1. **`mfg_pde/core/mfg_problem.py`**
   - ❌ Remove `MFGProblemBuilder` class (lines ~2007-2146)
   - ❌ Remove `create_mfg_problem()` function (lines ~2161-2175)
   - ✅ Keep `MFGProblem` class
   - ✅ Keep `MFGComponents` dataclass

2. **`mfg_pde/__init__.py`**
   - Remove `MFGProblemBuilder` from imports
   - Remove `create_mfg_problem` from imports
   - Remove from `__all__` list

3. **`mfg_pde/core/__init__.py`**
   - Remove `MFGProblemBuilder` from imports
   - Remove `create_mfg_problem` from imports

4. **Create `mfg_pde/factory/problems.py`** (new file)
   - Implement `create_lq_problem()`
   - Implement `create_crowd_motion_problem()`
   - Implement `create_traffic_flow_problem()`
   - Add to `mfg_pde/factory/__init__.py` exports

### Migration Guide

**Old Code** (using Builder):
```python
from mfg_pde import MFGProblemBuilder

problem = (MFGProblemBuilder()
    .hamiltonian(my_H, my_dH_dm)
    .domain(0, 10, 100)
    .time(2.0, 40)
    .build())
```

**New Code** (Direct):
```python
from mfg_pde import MFGProblem, MFGComponents

components = MFGComponents(
    hamiltonian_func=my_H,
    hamiltonian_dm_func=my_dH_dm,
)
problem = MFGProblem(
    xmin=0, xmax=10, Nx=100,
    T=2.0, Nt=40,
    components=components,
)
```

**Old Code** (using `create_mfg_problem`):
```python
from mfg_pde import create_mfg_problem

problem = create_mfg_problem(my_H, my_dH_dm, xmin=0, xmax=10, Nx=100, T=2.0, Nt=40)
```

**New Code**:
```python
from mfg_pde import MFGProblem, MFGComponents

components = MFGComponents(hamiltonian_func=my_H, hamiltonian_dm_func=my_dH_dm)
problem = MFGProblem(xmin=0, xmax=10, Nx=100, T=2.0, Nt=40, components=components)
```

---

## Impact Analysis

### Breaking Changes

1. **Removed APIs**:
   - `MFGProblemBuilder` - Users must migrate to direct `MFGProblem` construction
   - `create_mfg_problem()` - Users must migrate to `MFGComponents` + `MFGProblem`

2. **Affected Users**:
   - **Internal codebase**: Search for usages and update
   - **External users**: Provide migration guide in CHANGELOG
   - **Estimated impact**: Low (Builder/create_mfg_problem rarely used based on code search)

### Benefits

1. **Cognitive Load**:
   - ✅ Clear 2-level API: Factory OR Expert
   - ✅ No confusion about which API to use
   - ✅ Less documentation needed

2. **Maintenance**:
   - ✅ ~200 lines of code removed
   - ✅ Fewer APIs to maintain
   - ✅ Simpler test matrix

3. **User Experience**:
   - ✅ **Factory users**: Unchanged (already use `create_*_solver()`)
   - ✅ **Expert users**: Clearer path (just `MFGProblem` + `MFGComponents`)
   - ✅ **New users**: Less confusion about which API to learn

---

## Implementation Plan

### Phase 1: Add Factory Problem Functions (Week 1)

1. Create `mfg_pde/factory/problems.py`
2. Implement problem factories:
   - `create_lq_problem()`
   - `create_crowd_motion_problem()`
   - `create_traffic_flow_problem()`
3. Add comprehensive docstrings and examples
4. Write unit tests

### Phase 2: Mark for Deprecation (v0.13.0)

1. Add deprecation warnings to:
   - `MFGProblemBuilder.__init__()`
   - `create_mfg_problem()`
2. Update documentation showing new patterns
3. Create migration guide

### Phase 3: Remove Deprecated APIs (v1.0.0)

1. Delete `MFGProblemBuilder` class
2. Delete `create_mfg_problem()` function
3. Update all internal code
4. Update all examples and tests
5. Update CHANGELOG with migration guide

---

## Comparison: Before vs After

### Before (4 Levels)

```
┌─────────────────────────────────────────┐
│ Level 1: Solver Factories               │
│ create_fast_solver()                    │ ← Used
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Level 2: Problem Factories              │
│ create_mfg_problem()                    │ ← REMOVE
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Level 3: Builder Pattern                │
│ MFGProblemBuilder                       │ ← REMOVE
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Level 4: Direct Construction            │
│ MFGProblem, MFGComponents               │ ← Used
└─────────────────────────────────────────┘
```

### After (2 Levels)

```
┌─────────────────────────────────────────┐
│ Level 1: Factory (Pre-configured)       │
│ create_lq_problem()                     │
│ create_crowd_motion_problem()           │
│ create_traffic_flow_problem()           │
└─────────────────────────────────────────┘
          ↓ returns MFGProblem
┌─────────────────────────────────────────┐
│ Level 2: Expert (Direct Control)        │
│ MFGProblem + MFGComponents              │
└─────────────────────────────────────────┘
```

---

## Recommendation

**Approve and implement this simplification**:

1. ✅ Aligns with stated 2-level architecture philosophy
2. ✅ Removes unnecessary abstraction layers
3. ✅ Reduces maintenance burden (~200 lines removed)
4. ✅ Improves clarity for new users
5. ✅ Low migration cost (rarely used APIs)

**Timeline**:
- v0.13.0: Add deprecation warnings
- v1.0.0: Remove deprecated APIs

---

**Next Action**: Approve proposal and create GitHub issue for implementation.
