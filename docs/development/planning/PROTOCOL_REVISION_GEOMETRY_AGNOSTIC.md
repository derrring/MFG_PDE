# Protocol Revision: Geometry-Agnostic Design

**Date**: 2025-11-09
**Issue**: Original Protocol too grid-specific
**Solution**: Split into universal + grid-specific protocols

---

## Problem: Not All Geometries Are Grid-Based

### Geometry Types in MFG_PDE

| Geometry Type | Structure | Has Regular Grid? |
|:--------------|:----------|:------------------|
| **CARTESIAN_GRID** | Tensor product grid | ✅ YES |
| **NETWORK** | Graph topology | ❌ NO |
| **MAZE** | Grid with obstacles | ⚠️ PARTIAL |
| **DOMAIN_2D/3D** | Complex boundary | ❌ NO (mesh-based) |
| **IMPLICIT** | Level set/SDF | ❌ NO |
| **AMR** | Adaptive mesh | ❌ NO (irregular) |

**Conclusion**: Properties like `grid_shape`, `grid_spacing` only apply to ~30% of geometries!

---

## User Preference: Keep xSpace/tSpace

**Rationale**:
- ✅ Established naming convention
- ✅ More concise than `spatial_grid`/`time_grid`
- ✅ Compatible with mesh-free geometries
- ✅ "Space" suffix is meaningful
- ❌ No need to alias for Protocol compliance

**Decision**: Protocol should use `xSpace`/`tSpace` directly, not require aliases.

---

## Revised Protocol Design: Two-Tier

### Tier 1: `MFGProblemProtocol` (Universal - ALL problems)

**Truly geometry-agnostic properties:**

```python
@runtime_checkable
class MFGProblemProtocol(Protocol):
    """
    Minimal protocol that ALL MFG problems must satisfy.

    Works for grids, networks, meshes, implicit domains, etc.
    """

    # ==================
    # Spatial (minimal)
    # ==================
    dimension: int | str  # int for grids/meshes, "network" for graphs

    # ==================
    # Temporal (universal)
    # ==================
    T: float              # Final time
    Nt: int               # Number of time steps
    tSpace: NDArray       # Time points array [t₀, t₁, ..., t_Nt]

    # ==================
    # Physical (universal)
    # ==================
    sigma: float | Callable  # Diffusion coefficient

    # ==================
    # MFG Components (universal)
    # ==================
    def hamiltonian(self, x, m, p, t) -> float: ...
    def terminal_cost(self, x) -> float: ...
    def initial_density(self, x) -> float: ...
    def running_cost(self, x, m, t) -> float: ...
```

**Key Changes**:
- ❌ Removed `grid_shape`, `grid_spacing` (grid-specific)
- ❌ Removed `spatial_bounds`, `spatial_discretization` (grid-specific)
- ✅ Kept `tSpace` (not `time_grid`)
- ❌ No `xSpace` requirement (format varies by geometry)
- ✅ `dimension` can be int OR string (for networks)

### Tier 2: `CartesianGridMFGProtocol` (Grid-based only)

**Additional properties for regular grids:**

```python
@runtime_checkable
class CartesianGridMFGProtocol(MFGProblemProtocol, Protocol):
    """
    Extended protocol for Cartesian grid-based MFG problems.

    Only applies to CARTESIAN_GRID geometry type.
    """

    # Spatial (grid-specific)
    dimension: int  # Must be int for grids (not "network")
    spatial_bounds: list[tuple[float, float]]  # [(x₀_min, x₀_max), ...]
    spatial_discretization: list[int]          # [N₀, N₁, ...]
    xSpace: NDArray | list[NDArray]            # Coordinate arrays

    # Grid properties
    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Shape of grid (N₀, N₁, ...)."""
        ...

    @property
    def grid_spacing(self) -> list[float]:
        """Spacing [Δx₀, Δx₁, ...] per dimension."""
        ...
```

---

## Usage Pattern

### For Dimension-Agnostic Solvers

**Use Tier 1** (works with ANY geometry):

```python
def solve_hjb_generic(problem: MFGProblemProtocol) -> NDArray:
    """Solver that works for grids, networks, meshes, etc."""
    T = problem.T
    Nt = problem.Nt
    tSpace = problem.tSpace
    sigma = problem.sigma

    # Geometry-specific logic handled internally
    # Don't assume grid structure!
    ...
```

### For Grid-Specific Solvers

**Use Tier 2** (requires regular grid):

```python
def solve_hjb_fdm(problem: CartesianGridMFGProtocol) -> NDArray:
    """FDM solver - requires regular grid."""
    dx = problem.grid_spacing
    grid_shape = problem.grid_shape

    # Can safely assume Cartesian grid structure
    ...
```

### Runtime Check

```python
if isinstance(problem, CartesianGridMFGProtocol):
    # Use FDM or other grid-based method
    return solve_hjb_fdm(problem)
elif isinstance(problem, MFGProblemProtocol):
    # Use particle method or neural network
    return solve_hjb_particle(problem)
```

---

## What MFGProblem Needs to Implement

### For Tier 1 Compliance (Universal)

**Already have:**
- ✅ `dimension`
- ✅ `T`, `Nt`, `tSpace` (currently named these!)
- ✅ `sigma`
- ✅ All MFG component methods

**Need to add:**
- ❌ Nothing! Already compliant for Tier 1

### For Tier 2 Compliance (Grid modes only)

**Already have (in nD mode):**
- ✅ `dimension` (int)
- ✅ `spatial_bounds`
- ✅ `spatial_discretization`
- ✅ `xSpace` (via `_grid`)

**Need to add:**
- [ ] `grid_shape` property (return `spatial_shape`)
- [ ] `grid_spacing` property (compute from bounds/discretization)

**Don't need (for non-grid modes):**
- 1D legacy mode: Tier 1 only
- Geometry mode: Tier 1 only (geometry handles spatial)
- Network mode: Tier 1 only

---

## Naming Decision Matrix

| Property | Protocol Name | Current Name | Decision |
|:---------|:--------------|:-------------|:---------|
| Time step | `tSpace` | `tSpace` | ✅ **KEEP** (no alias) |
| Spatial coords | _(geometry-specific)_ | `xSpace` (1D), `_grid` (nD) | ✅ **KEEP** (not in Tier 1) |
| Time step size | `Dt` | `Dt` | ✅ **KEEP** (no lowercase alias) |

**Rationale**:
- Current naming is fine
- No breaking changes needed
- Protocol adapts to codebase, not vice versa

---

## Implementation Plan (Revised)

### Phase 1a: Minimal Protocol (10 min)

1. **Simplify `MFGProblemProtocol`** to Tier 1 (remove grid properties)
2. **Test**: `isinstance(MFGProblem(...), MFGProblemProtocol)` → should pass NOW!
3. **No code changes to MFGProblem needed**

### Phase 1b: Grid Protocol (Optional, 30 min)

1. **Create `CartesianGridMFGProtocol`** extending Tier 1
2. **Add `grid_shape`/`grid_spacing` properties** to MFGProblem (nD mode only)
3. **Test**: Grid problems satisfy Tier 2

### Phase 1c: Solver Type Hints (15 min)

1. **Update FDM solvers**: `problem: CartesianGridMFGProtocol`
2. **Update particle solvers**: `problem: MFGProblemProtocol`
3. **Update PINN solvers**: `problem: MFGProblemProtocol`

**Total Time**: ~1 hour (Phase 1a is 10 min!)

---

## Benefits of Two-Tier Design

1. **Flexibility**: Universal Tier 1 works with ANY geometry
2. **Type Safety**: Grid solvers explicitly require grid properties (Tier 2)
3. **No Breaking Changes**: Respects existing naming (`xSpace`, `tSpace`, `Dt`)
4. **Future-Proof**: Easy to add `MeshMFGProtocol`, `NetworkMFGProtocol` etc.

---

## Conclusion

**Original Protocol**: Too grid-specific, forced grid assumptions on all geometries ❌

**Revised Protocol**: Two-tier design
- **Tier 1**: Geometry-agnostic, works everywhere ✅
- **Tier 2**: Grid-specific, type-safe for FDM/WENO solvers ✅

**Naming**: Keep `xSpace`, `tSpace`, `Dt` - no aliases needed ✅

**Next Step**: Implement Phase 1a (10 min) - simplify Protocol, test compliance
