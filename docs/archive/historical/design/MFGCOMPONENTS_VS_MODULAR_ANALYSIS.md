# MFGComponents vs Modular Problem Definition

**Date**: 2025-11-03
**Question**: Can MFGComponents be replaced by a modular MFGProblem approach?
**Answer**: Partially yes, but MFGComponents serves a different purpose

---

## Current Architecture

### **MFGComponents** (`mfg_pde/core/mfg_problem.py:28-118`)

**Purpose**: Container for mathematical components that define MFG problem physics

```python
@dataclass
class MFGComponents:
    # Physics specification
    hamiltonian_func: Callable | None = None
    potential_func: Callable | None = None
    initial_density_func: Callable | None = None
    final_value_func: Callable | None = None

    # Boundary and coupling
    boundary_conditions: BoundaryConditions | None = None
    coupling_func: Callable | None = None

    # ... many more optional components ...
```

**Used in**:
```python
components = MFGComponents(
    hamiltonian_func=my_hamiltonian,
    potential_func=my_potential
)
problem = MFGProblem(Nx=50, Nt=20, components=components)
```

### **Modular Approach** (Solver Composition)

**Purpose**: Compose numerical methods (HJB + FP + Coupling)

```python
hjb = HJBGFDMSolver(problem)
fp = FPParticleSolver(problem, num_particles=5000)
solver = FixedPointIterator(problem, hjb, fp)
```

---

## Key Distinction

| Aspect | MFGComponents | Modular Approach |
|:-------|:--------------|:-----------------|
| **Level** | Problem definition (physics) | Algorithm selection (numerics) |
| **What it defines** | Hamiltonian, potential, BCs | HJB solver, FP solver, coupling |
| **When specified** | Problem creation | Solver creation |
| **Abstraction** | Mathematical formulation | Numerical methods |
| **User control** | Physics of the problem | How to solve it |

**They operate at different levels**:
```
MFGComponents → Define WHAT (physics)
      ↓
   MFGProblem → Store problem data
      ↓
Modular Solvers → Define HOW (numerics)
```

---

## Analysis: Can MFGComponents be Replaced?

### **Option 1: Keep Current Design** ✅ RECOMMENDED

**Rationale**: MFGComponents and modular solvers serve different purposes

**Advantages**:
- Clear separation: physics (MFGComponents) vs numerics (modular solvers)
- MFGProblem remains algorithm-agnostic
- Users specify "what to solve" separately from "how to solve"
- Easier to swap numerical methods without changing problem

**Disadvantages**:
- Two configuration patterns to learn (but serving different purposes)

### **Option 2: Merge into Problem Subclasses**

**Idea**: Create specialized problem classes instead of components

```python
# Instead of:
components = MFGComponents(hamiltonian_func=crowd_H)
problem = MFGProblem(Nx=50, components=components)

# Use:
problem = CrowdMotionProblem(Nx=50, congestion_level=10.0)
```

**Advantages**:
- Domain-specific problem classes
- Fewer concepts (no MFGComponents)
- Type-safe problem definitions

**Disadvantages**:
- Explosion of problem classes for each domain
- Less flexible (can't mix and match components easily)
- Still need MFGComponents for truly custom problems

### **Option 3: Builder Pattern** (Already exists!)

**Current**: `MFGProblemBuilder` exists (`mfg_pde/core/mfg_problem.py:1140-1280`)

```python
problem = (MFGProblemBuilder()
    .spatial_domain(xmin=0, xmax=1, Nx=50)
    .time_domain(T=1.0, Nt=20)
    .hamiltonian(my_hamiltonian_func, my_dm_func)
    .initial_density(my_m0_func)
    .build())
```

**This already provides modular problem construction!**

---

## Proposed Refinement

### **Keep Three Patterns, Each with Clear Purpose**

#### **1. Default Problem** (Quick Start)

```python
# For exploration, learning, benchmarking
problem = MFGProblem(Nx=50, Nt=20, T=1.0)
```

**Use when**: Testing methods, learning framework, standard benchmarks

#### **2. Custom Problem via MFGComponents** (Research)

```python
# For custom physics, research problems
components = MFGComponents(
    hamiltonian_func=my_H,
    potential_func=my_V,
    initial_density_func=my_m0
)
problem = MFGProblem(Nx=50, Nt=20, components=components)
```

**Use when**: Novel Hamiltonians, custom physics, research exploration

#### **3. Builder Pattern** (Complex Configuration)

```python
# For step-by-step construction with validation
problem = (MFGProblemBuilder()
    .spatial_domain(xmin=0, xmax=1, Nx=50)
    .time_domain(T=1.0, Nt=20)
    .hamiltonian(H_func, dH_dm_func)
    .potential(V_func)
    .initial_density(m0_func)
    .final_value(g_func)
    .boundary_conditions(bc)
    .build())
```

**Use when**: Complex problems, step-by-step configuration, prefer fluent API

---

## Relationship to Modular Solver Composition

**These are orthogonal**:

```python
# Problem definition (physics) - Choose ONE:
# Option A: Default
problem = MFGProblem(Nx=50, Nt=20)

# Option B: Components
problem = MFGProblem(Nx=50, Nt=20, components=MFGComponents(hamiltonian_func=my_H))

# Option C: Builder
problem = MFGProblemBuilder().spatial_domain(...).hamiltonian(...).build()

# Solver composition (numerics) - ALWAYS modular:
hjb = HJBGFDMSolver(problem)          # Works with ANY problem
fp = FPParticleSolver(problem, ...)    # Works with ANY problem
solver = FixedPointIterator(problem, hjb, fp)  # Works with ANY problem
```

**Key insight**: Solver doesn't care how problem was created, only that it's an `MFGProblem`.

---

## Recommendation

### **Keep MFGComponents, Enhance Documentation**

**Why**:
1. **Serves different purpose** - Physics vs numerics separation is valuable
2. **Already well-designed** - Dataclass with optional fields is clean
3. **Flexible** - Can specify only what you need
4. **Composable** - Can build complex problems incrementally

**Improvements**:
1. **Document the distinction** clearly in user guide
2. **Add examples** showing when to use each pattern
3. **Consider Hamiltonian abstraction** (see HAMILTONIAN_ABSTRACTION_DESIGN.md)
4. **Keep builder pattern** as alternative interface

### **Future Enhancement: Domain-Specific Problem Classes**

When domain templates mature, can add:

```python
# mfg_pde/problems/crowd_motion.py (FUTURE)

class CrowdMotionProblem(MFGProblem):
    """
    Pre-configured problem for crowd motion dynamics.

    Internally uses MFGComponents with crowd-specific physics.
    """
    def __init__(
        self,
        *,
        domain: tuple[float, float],
        Nx: int,
        Nt: int,
        T: float,
        congestion_level: float = 10.0,
        exit_positions: list[float] | None = None
    ):
        # Create crowd-specific Hamiltonian
        hamiltonian = PowerLawHamiltonian(congestion_coeff=congestion_level)

        # Create crowd-specific components
        components = MFGComponents(
            hamiltonian=hamiltonian,
            # ... other crowd-specific setup ...
        )

        # Initialize parent
        super().__init__(
            xmin=domain[0],
            xmax=domain[1],
            Nx=Nx,
            Nt=Nt,
            T=T,
            components=components
        )
```

This provides **domain-specific convenience** while **keeping MFGComponents as the underlying mechanism**.

---

## Summary

**Question**: Can MFGComponents be replaced by modular approach?

**Answer**: **No** - they serve different purposes:

- **MFGComponents**: Modular **problem definition** (physics)
- **Modular solvers**: Modular **algorithm composition** (numerics)

**Both are valuable** and operate at different abstraction levels.

**Enhancement**: Consider adding domain-specific problem classes (like `CrowdMotionProblem`) that **use MFGComponents internally** to provide convenience without sacrificing flexibility.

---

**Last Updated**: 2025-11-03
**Decision**: Keep MFGComponents, enhance with Hamiltonian abstraction and domain classes
