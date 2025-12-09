# MFGProblem and MFGComponents: Structure and Implementation Audit Report

**Date**: 2025-12-09
**Last audited**: 2025-12-09
**Status**: [ANALYSIS]
**Related Issues**: #417, #419, #420, #421

---

## 1. Boundary Condition Mechanism (Detailed)

### Architecture Overview

```
User API                 Geometry Layer              Solver Layer
───────────────────────────────────────────────────────────────────

periodic_bc(dim=1)  ───►  BoundaryConditions  ───►  HJBGFDMSolver
                              │                          │
TensorProductGrid(..., bc) ◄──┘                          │
        │                                                 │
        └──────► geometry.boundary_conditions  ───────────┘
                              │
                              ▼
                     _apply_boundary_conditions_to_solution()
                     _apply_boundary_conditions_to_system()
```

### BC Flow Path

1. **User-Level**: Factory functions create BC specs
   - `periodic_bc(dimension=1)` → `BoundaryConditions(type="periodic", ...)`
   - `dirichlet_bc(value=0.0)` → `BoundaryConditions(type="dirichlet", value=0.0)`

2. **Geometry-Level**: BC stored in geometry object
   - `TensorProductGrid(dimension, bounds, num_points, boundary_conditions=bc)`
   - `geometry.boundary_conditions` holds the BC specification

3. **Solver-Level**: Solver accesses BC from geometry
   - `HJBGFDMSolver.__init__()` takes `boundary_conditions` and `boundary_indices` params
   - `_get_boundary_condition_property()` abstracts dict vs dataclass access
   - Two enforcement methods:
     - **Solution enforcement** (`hjb_gfdm.py:1877-1908`): Directly sets `u[boundary_indices] = bc_value`
     - **System enforcement** (`hjb_gfdm.py:1910-1939`): Sets Jacobian rows to identity, residual to zero

4. **Applicator Layer**: Method-specific BC application
   - `applicator_fdm.py` - FDM matrix modifications
   - `applicator_fem.py` - FEM weak form integration
   - `applicator_meshfree.py` - Meshfree constraint handling
   - `applicator_graph.py` - Network boundary handling

### BC Type Handling

| BC Type | Solution Enforcement | System Enforcement |
|:--------|:--------------------|:-------------------|
| Dirichlet | Set `u[i] = value` | Set row to identity, residual=0 |
| Neumann | No direct modification | Weak enforcement (residual) |
| Periodic | Wrap indices | Periodic stencil in matrix |

---

## 2. MFGComponents Audit

### Current Structure (`mfg_problem.py:30-65`)

```python
@dataclass
class MFGComponents:
    # Core Hamiltonian (both required together when provided)
    hamiltonian_func: Callable | None      # H(x, m, p, t) -> float
    hamiltonian_dm_func: Callable | None   # dH/dm(x, m, p, t) -> float  ← required when H given

    # Optional Jacobian for advanced solvers
    hamiltonian_jacobian_func: Callable | None

    # Potential function
    potential_func: Callable | None        # V(x, t) -> float

    # Initial/Terminal conditions
    initial_density_func: Callable | None  # m_0(x) -> float
    final_value_func: Callable | None      # u_T(x) -> float

    # Boundary conditions
    boundary_conditions: BoundaryConditions | None

    # Coupling terms (for advanced MFG formulations)
    coupling_func: Callable | None

    # Problem parameters and metadata
    parameters: dict[str, Any] = field(default_factory=dict)
    description: str = "MFG Problem"
    problem_type: str = "mfg"
```

### Observations

**Positive**:
1. Clear dataclass with type hints
2. Enforced pair constraint: `hamiltonian_dm_func` required with `hamiltonian_func` (validated at `mfg_problem.py:1312`)
3. Generic `parameters` dict for custom metadata
4. `final_value_func` is correct mathematically (more general than "terminal cost")

**Issues**:

| Issue | Severity | Location | Description |
|:------|:---------|:---------|:------------|
| ~~Signature inconsistency~~ | ~~Medium~~ | ~~Documentation~~ | [RESOLVED] Dataclass now correctly documents `H(x, m, p, t)` |
| Missing `running_cost_func` | Low | Dataclass | No explicit f(x, m) running cost field |
| No validation for other funcs | Low | `_validate_components` | Only H and H_dm validated, not potential/initial/final |

### Recommendation

[RESOLVED] The dataclass fields now have inline comments with clear signatures:

```python
hamiltonian_func: Callable | None = None  # H(x, m, p, t) -> float
```

---

## 3. MFGProblem Audit

### Current Structure

**Constructor overloading via parameters** (`mfg_problem.py:126-380`):
- Legacy 1D: `xmin, xmax, Nx, T, Nt, sigma, ...`
- Geometry-based: `geometry, T, Nt, sigma, ...`
- Custom: `components, ...`

### `is_custom` Flag Logic

```python
@property
def is_custom(self) -> bool:
    return self.components is not None and self.components.hamiltonian_func is not None
```

**Decision**: `is_custom=True` only if both `components` provided AND `hamiltonian_func` is set.

### Observations

**Positive**:
1. Single unified class handles all cases
2. Geometry abstraction allows 1D/2D/3D/meshfree
3. `solve()` method is the primary API (per CLAUDE.md)
4. Scalar-to-array conversion with deprecation warnings (lines 81-124)

**Issues**:

| Issue | Severity | Location | Description |
|:------|:---------|:---------|:------------|
| Massive file | Medium | `mfg_problem.py` | 1500+ lines, validation/helpers mixed with core |
| Conditional attributes | Low | Various | Some attrs only exist in certain modes (#417) |
| `num_points` vs `Nx+1` | Low | User confusion | `TensorProductGrid.num_points=[51]` means 50 intervals |
| Silent evaluation failures | High | Custom component eval | try/except returns NaN silently (#420) |
| Dual geometry source | Low | `geometry` param | Problem can store both `geometry` and `xmin/xmax` |

### State Mutation Concern (#419)

Solvers **do** temporarily mutate `problem.sigma`:

```python
# In solvers:
old_sigma = self.problem.sigma
self.problem.sigma = new_value  # MUTATION
# ... solve ...
self.problem.sigma = old_sigma  # RESTORE
```

**Risk**: Exception during solve leaves problem in inconsistent state.
**Mitigation**: Context manager pattern not currently used.

---

## 4. Overall Architecture Assessment

### Strengths

1. **Clear separation**: Problem definition vs Solver execution vs Geometry
2. **Unified entry point**: `problem.solve()` handles all complexity
3. **Extensible BC system**: Factory functions + applicators for different methods
4. **Type-safe**: Dataclass + type hints + validation

### Weaknesses

1. **Signature validation incomplete**: Only Hamiltonian validated, not other custom funcs
2. **Silent failure pattern**: Custom component exceptions caught and return NaN
3. ~~**Documentation gap**: `derivs` tuple not explained in code~~ [RESOLVED - now uses `H(x, m, p, t)`]
4. **Mixed geometry sources**: Both `geometry` object and legacy `xmin/xmax` stored
5. **Large monolithic file**: `mfg_problem.py` could be refactored (low priority)

### Recommended Actions (Prioritized)

| Priority | Action | Issue # | Status |
|:---------|:-------|:--------|:-------|
| P0 | Remove try/except that silences custom component errors | #420 | Open |
| P1 | ~~Add docstrings explaining `derivs` tuple format~~ | - | [RESOLVED] Signatures now use `H(x, m, p, t)` |
| P2 | Add context manager for sigma mutation (exception safety) | #419 | Open |
| P3 | Split `mfg_problem.py` into `problem.py` + `components.py` + `validation.py` | #417 | Low priority |
| P4 | Document geometry-with-geometry example | #421 | Open |

---

## 5. Summary

The MFGProblem/MFGComponents architecture is **fundamentally sound**:
- Clear dataclass for custom physics
- Unified problem class with geometry abstraction
- BC mechanism properly decoupled from grid representation

**Critical audit items**:
1. **Silent failures** (#420) - High risk, simple fix
2. ~~**Signature documentation**~~ - [RESOLVED] Now uses clear `H(x, m, p, t)` signature
3. **Exception safety** (#419) - Low risk, defensive improvement

The structure follows good separation of concerns. The main risks are in error handling, not architecture.

---

## Appendix: MFGComponents Signature Reference

| Component | Expected Signature | Notes |
|:----------|:-------------------|:------|
| `hamiltonian_func` | `H(x, m, p, t) -> float` | p is momentum/gradient |
| `hamiltonian_dm_func` | `dH_dm(x, m, p, t) -> float` | **Required** when `hamiltonian_func` provided |
| `potential_func` | `V(x, t) -> float` | Optional |
| `initial_density_func` | `m0(x) -> float` | Initial density m(0, x) |
| `final_value_func` | `uT(x) -> float` | Terminal condition u(T, x) |
| `coupling_func` | `coupling(x, m) -> float` | Optional coupling term |

---

## Appendix: Usage Modes

| Mode | Parameters | `is_custom` | Use Case |
|:-----|:-----------|:------------|:---------|
| **Mode 1: Default** | `xmin, xmax, Nx, T, Nt, sigma` | `False` | Standard LQ-type MFG |
| **Mode 2: Custom** | `components=MFGComponents(...)` | `True` | Custom Hamiltonians |
| **Mode 3: Geometry** | `geometry=TensorProductGrid(...)` | `False` | Multi-dimensional grids |
