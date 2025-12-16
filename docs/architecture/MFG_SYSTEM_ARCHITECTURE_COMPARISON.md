# MFGSystem vs Current Architecture Comparison

**Date**: 2025-12-17
**Related Issue**: #493

---

## 1. Current Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CURRENT ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      MFGComponents                               │   │
│  │  (Custom mathematical specification - optional)                  │   │
│  │                                                                  │   │
│  │  - hamiltonian_func: H(x, m, p, t)                              │   │
│  │  - hamiltonian_dm_func: dH/dm                                   │   │
│  │  - potential_func: V(x, t)                                      │   │
│  │  - initial_density_func: m_0(x)      ← Temporal BC (FP)         │   │
│  │  - final_value_func: u_T(x)          ← Temporal BC (HJB)        │   │
│  │  - boundary_conditions               ← Spatial BC (MIXED HERE!) │   │
│  │  - parameters: dict                                              │   │
│  └──────────────────────────────┬──────────────────────────────────┘   │
│                                 │                                       │
│                                 ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                       MFGProblem                                 │   │
│  │  (Unified problem definition)                                    │   │
│  │                                                                  │   │
│  │  - geometry: GeometryProtocol        ← Domain (no BC currently) │   │
│  │  - T, Nt: Time domain                                           │   │
│  │  - sigma: Diffusion                                             │   │
│  │  - coupling_coefficient: γ                                      │   │
│  │  - components: MFGComponents | None                              │   │
│  │  - m_init, u_fin: np.ndarray         ← Temporal BC              │   │
│  │                                                                  │   │
│  │  Methods (via mixins):                                           │   │
│  │  - H(), dH_dm()                      ← HamiltonianMixin         │   │
│  │  - get_boundary_conditions()         ← ConditionsMixin          │   │
│  │  - get_m_init(), get_u_fin()         ← Temporal BC access       │   │
│  └──────────────────────────────┬──────────────────────────────────┘   │
│                                 │                                       │
│                                 ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    FixedPointIterator                            │   │
│  │  (Coupling solver - orchestrates HJB and FP)                     │   │
│  │                                                                  │   │
│  │  - problem: MFGProblem                                           │   │
│  │  - hjb_solver: BaseHJBSolver                                     │   │
│  │  - fp_solver: BaseFPSolver                                       │   │
│  │  - damping_factor, anderson_accelerator, ...                     │   │
│  │                                                                  │   │
│  │  solve() → SolverResult(U, M, ...)                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Proposed MFGSystem Pattern (External Comment)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       PROPOSED MFGSystem PATTERN                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        Geometry                                  │   │
│  │  (Single source of truth for spatial domain + BC)               │   │
│  │                                                                  │   │
│  │  - bounds, Nx, dimension                                         │   │
│  │  - boundary_conditions: BoundaryConditions  ← SPATIAL BC HERE   │   │
│  │  - get_boundary_conditions()                                     │   │
│  └──────────────────────────────┬──────────────────────────────────┘   │
│                                 │                                       │
│                                 ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                       MFGSystem                                  │   │
│  │  (Orchestrator - injects geometry to solvers)                    │   │
│  │                                                                  │   │
│  │  def __init__(self, config):                                     │   │
│  │      # 1. Create single geometry (SSOT)                          │   │
│  │      self.geometry = GeometryFactory.create(config.geometry)     │   │
│  │                                                                  │   │
│  │      # 2. Inject same geometry to both solvers (DI)              │   │
│  │      self.hjb_solver = HJBSolver(geometry=self.geometry, ...)    │   │
│  │      self.fp_solver = FPSolver(geometry=self.geometry, ...)      │   │
│  │                                                                  │   │
│  │      # 3. Store temporal conditions                              │   │
│  │      self.m_init = config.payload.m_init   ← Temporal BC (FP)   │   │
│  │      self.u_fin = config.payload.u_fin     ← Temporal BC (HJB)  │   │
│  │                                                                  │   │
│  │  def solve(self):                                                │   │
│  │      u = self.hjb_solver.solve_backward(terminal=self.u_fin)     │   │
│  │      m = self.fp_solver.solve_forward(initial=self.m_init, u=u)  │   │
│  │      return u, m                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mapping: Current → Proposed

| Proposed (MFGSystem) | Current Equivalent | Notes |
|---------------------|-------------------|-------|
| `Geometry` | `MFGProblem.geometry` | Exists, but **lacks BC** |
| `Geometry.boundary_conditions` | `MFGComponents.boundary_conditions` | Wrong location |
| `MFGSystem` | `FixedPointIterator` | **Already exists!** |
| `MFGSystem.m_init` | `MFGProblem.m_init` | Correct location |
| `MFGSystem.u_fin` | `MFGProblem.u_fin` | Correct location |
| `config.geometry` | `MFGProblem` constructor | Partially exists |
| `config.payload` | `MFGComponents` | Temporal conditions mixed with spatial BC |

---

## 4. Key Insight: FixedPointIterator IS MFGSystem

The proposed `MFGSystem` pattern is **already implemented** as `FixedPointIterator`:

```python
# Current FixedPointIterator
class FixedPointIterator(BaseMFGSolver):
    def __init__(self, problem, hjb_solver, fp_solver, ...):
        self.problem = problem           # ≈ config
        self.hjb_solver = hjb_solver     # Injected
        self.fp_solver = fp_solver       # Injected

    def solve(self, ...):
        # Get temporal conditions
        M_initial = self.problem.get_m_init()
        U_terminal = self.problem.get_u_fin()

        # Iterate
        for iter in range(max_iterations):
            U = self.hjb_solver.solve_backward(U_terminal, M)
            M = self.fp_solver.solve_forward(M_initial, U)

        return SolverResult(U, M, ...)
```

**The only missing piece**: Geometry should own spatial BC.

---

## 5. What Needs to Change

### Current State (BC flow)

```
User → MFGComponents.boundary_conditions → MFGProblem.get_boundary_conditions()
                                                    ↓
                                           FPFDMSolver (uses it)
                                           HJBFDMSolver (doesn't use it)
```

### Target State (BC flow)

```
User → Geometry.boundary_conditions → Both solvers query geometry
                ↓
        ┌───────┴───────┐
        ↓               ↓
   HJBFDMSolver    FPFDMSolver
   (queries)       (queries)
```

---

## 6. Minimal Changes Required

### No New Classes Needed!

We don't need a new `MFGSystem` class. Instead:

### Step 1: Add BC to Geometry

```python
# geometry/grids/tensor_grid.py
class TensorProductGrid:
    def __init__(self, ..., boundary_conditions=None):
        self._bc = boundary_conditions or no_flux_bc(dimension)

    def get_boundary_conditions(self):
        return self._bc
```

### Step 2: MFGProblem delegates to geometry

```python
# core/mfg_problem.py
class MFGProblem:
    def get_boundary_conditions(self):
        # Priority: geometry (SSOT)
        if self.geometry is not None:
            return self.geometry.get_boundary_conditions()
        # Fallback: components (legacy)
        if self.components and self.components.boundary_conditions:
            return self.components.boundary_conditions
        return periodic_bc(dimension=self.dimension)
```

### Step 3: Solvers query geometry

```python
# Both HJB and FP solvers
class HJBFDMSolver:
    def __init__(self, problem=None, geometry=None, ...):
        self.geometry = problem.geometry if problem else geometry
        self.spatial_bc = self.geometry.get_boundary_conditions()
```

---

## 7. Summary: Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      UNIFIED ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                     ┌─────────────────────┐                             │
│                     │   User Config       │                             │
│                     │   (YAML/Python)     │                             │
│                     └──────────┬──────────┘                             │
│                                │                                        │
│              ┌─────────────────┼─────────────────┐                      │
│              │                 │                 │                      │
│              ▼                 ▼                 ▼                      │
│  ┌───────────────────┐ ┌─────────────┐ ┌────────────────┐              │
│  │    Geometry       │ │  Temporal   │ │   Physics      │              │
│  │                   │ │  Conditions │ │                │              │
│  │ - bounds, Nx      │ │             │ │ - sigma        │              │
│  │ - BC (spatial)    │ │ - m_init    │ │ - gamma        │              │
│  │                   │ │ - u_fin     │ │ - T, Nt        │              │
│  └─────────┬─────────┘ └──────┬──────┘ └───────┬────────┘              │
│            │                  │                │                        │
│            └──────────────────┼────────────────┘                        │
│                               │                                         │
│                               ▼                                         │
│                    ┌─────────────────────┐                              │
│                    │     MFGProblem      │                              │
│                    │                     │                              │
│                    │ - geometry ─────────┼──► Spatial BC (SSOT)        │
│                    │ - m_init, u_fin     │    Temporal BC              │
│                    │ - sigma, gamma, T   │    Physics                  │
│                    │ - components (opt)  │    Custom math              │
│                    └──────────┬──────────┘                              │
│                               │                                         │
│                               ▼                                         │
│                    ┌─────────────────────┐                              │
│                    │  FixedPointIterator │  ≈ MFGSystem                │
│                    │  (Coupling Solver)  │                              │
│                    │                     │                              │
│                    │ - problem ──────────┼──► Gets geometry, BC, etc.  │
│                    │ - hjb_solver ───────┼──► Injected, queries geom   │
│                    │ - fp_solver ────────┼──► Injected, queries geom   │
│                    │                     │                              │
│                    │ solve() → Result    │                              │
│                    └─────────────────────┘                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Conclusion

| Question | Answer |
|----------|--------|
| Is MFGSystem a new class? | **No** - it's `FixedPointIterator` |
| What role does MFGProblem play? | **Problem definition** - holds geometry + temporal BC + physics |
| What role does MFGComponents play? | **Custom math** - Hamiltonian, potential (optional) |
| What needs to change? | **Move spatial BC to geometry** |
| Is the proposed pattern valid? | **Yes** - we already have it, just need BC in geometry |

**Action**: Issue #493 covers the required changes. No architectural restructuring needed.
