# Phase 3.1: Unified MFGProblem Architecture Design

**Date**: 2025-11-03
**Status**: Design Phase
**Timeline**: 8-10 weeks
**Related**: Architecture Refactoring Plan (2025-11-02), Issue #200

---

## Executive Summary

Design unified MFGProblem class that supports all dimensions (1D, 2D, 3D, nD), all solver types (FDM, GFDM, SL, particle), and all domain types (grid, particle, network, manifold).

**Current Status**:
- ✅ MFGProblem already supports 1D and n-D initialization
- ✅ GridBasedMFGProblem provides dimension-agnostic grid interface
- ⚠️ Split between MFGProblem and GridBasedMFGProblem creates confusion
- ❌ Network and variational problems use separate classes
- ❌ Only 4/25 solver combinations work without adapters

**Goal**: Single `MFGProblem` class that eliminates custom problem code and adapter patterns.

---

## Current State Analysis

### Existing Problem Classes

1. **MFGProblem** (`mfg_pde/core/mfg_problem.py`)
   - **Purpose**: Legacy 1D + new n-D support
   - **Initialization**: Two modes
     * Legacy: `MFGProblem(Nx=100, xmin=0, xmax=1)`
     * N-D: `MFGProblem(spatial_bounds=[(0,1), (0,1)], spatial_discretization=[50, 50])`
   - **Attributes**: `dimension`, `spatial_shape`, `spatial_bounds`, `xSpace` (1D), etc.
   - **Hamilton**ian**: Built-in quadratic H(p) = |p|²/(2σ) + coupling_coefficient·m

2. **GridBasedMFGProblem** (`mfg_pde/core/highdim_mfg_problem.py`)
   - **Purpose**: Dimension-agnostic grid-based problems
   - **Initialization**: `GridBasedMFGProblem(domain_bounds=(0,1,0,1), grid_resolution=50)`
   - **Attributes**: `geometry` (TensorProductGrid wrapper), `mesh_data`, `collocation_points`
   - **Hamiltonian**: Abstract - must be provided by subclass

3. **HighDimMFGProblem** (Abstract)
   - **Purpose**: Base class for 2D/3D/nD problems with complex geometry
   - **Requires**: Geometry object (BaseGeometry), abstract `setup_components()`
   - **Use Cases**: Implicit domains, obstacles, complex boundaries

4. **NetworkMFGProblem** (`mfg_pde/core/network_mfg_problem.py`)
   - **Purpose**: MFG on graphs/networks
   - **Separate API**: Different initialization pattern

5. **VariationalMFGProblem** (`mfg_pde/core/variational_mfg_problem.py`)
   - **Purpose**: Variational formulation
   - **Separate API**: Different mathematical framework

### Current Confusion Points

| Confusion | Example | Impact |
|:----------|:--------|:-------|
| **Which class to use?** | User wants 2D problem - use MFGProblem or GridBasedMFGProblem? | 15min decision time |
| **Incompatible APIs** | `MFGProblem.xSpace` vs `GridBasedMFGProblem.geometry.grid` | Adapter code needed |
| **Solver compatibility** | HJBFD MSolver works with MFGProblem, not GridBasedMFGProblem (but actually does via dimension detection) | Trial and error |
| **Custom Hamiltonians** | MFGProblem has built-in H, GridBasedMFGProblem requires subclass | Code duplication |

---

## Phase 3.1 Design Goals

### Primary Goals

1. **Single Entry Point**: One `MFGProblem` class for all use cases
2. **Automatic Detection**: Detect dimension, domain type, solver compatibility
3. **Backward Compatible**: All existing code continues to work
4. **Zero Adapters**: Eliminate custom problem classes and adapter patterns
5. **Clear Errors**: Explicit messages for unsupported solver combinations

### Design Principles

1. **Progressive Disclosure**: Simple cases are simple, complex cases are possible
2. **Convention Over Configuration**: Sensible defaults, explicit overrides
3. **Fail Fast**: Clear error messages at problem creation time
4. **Type Safety**: Full type hints for IDE support

---

## Proposed Unified Architecture

### Core Concept

**Single `MFGProblem` class with multiple initialization patterns**:

```python
class MFGProblem:
    """
    Unified MFG problem class supporting all dimensions and solver types.

    Initialization Patterns:
    1. Simple 1D (legacy-compatible)
    2. Simple n-D grid
    3. Complex geometry (implicit domains, obstacles)
    4. Network/graph
    5. Custom components (advanced)
    """
```

### Initialization Modes

#### Mode 1: Simple 1D (Legacy - 100% Backward Compatible)

```python
# Exactly as before - no breaking changes
problem = MFGProblem(
    Nx=100,
    xmin=0.0,
    xmax=1.0,
    Nt=100,
    T=1.0,
    sigma=0.5,
    coupling_coefficient=1.0
)

# Auto-detected attributes:
problem.dimension          # 1
problem.domain_type        # "grid"
problem.solver_compatible  # ["fdm", "gfdm", "semi_lagrangian", "particle"]
```

#### Mode 2: Simple n-D Grid (Already Supported)

```python
# 2D grid
problem = MFGProblem(
    spatial_bounds=[(0, 1), (0, 1)],
    spatial_discretization=[50, 50],
    Nt=50,
    T=1.0,
    sigma=0.1
)

# 3D grid
problem = MFGProblem(
    spatial_bounds=[(0, 1), (0, 1), (0, 1)],
    spatial_discretization=[30, 30, 30],
    Nt=30
)

# Auto-detected attributes:
problem.dimension          # 2 or 3
problem.domain_type        # "grid"
problem.solver_compatible  # ["fdm", "gfdm", "particle"]
```

#### Mode 3: Complex Geometry (NEW)

```python
# Implicit domain with obstacles
from mfg_pde.geometry import Hyperrectangle, Hypersphere

domain = Hyperrectangle(bounds=[[0, 1], [0, 1]])
obstacles = [Hypersphere(center=[0.5, 0.5], radius=0.1)]

problem = MFGProblem(
    geometry=domain,
    obstacles=obstacles,
    time_domain=(1.0, 50),
    sigma=0.1
)

# Auto-detected attributes:
problem.dimension          # 2 (from geometry)
problem.domain_type        # "implicit"
problem.solver_compatible  # ["gfdm", "particle"]  # FDM not supported for complex geometry
```

#### Mode 4: Network/Graph (NEW)

```python
# Network MFG
import networkx as nx

graph = nx.grid_2d_graph(10, 10)

problem = MFGProblem(
    network=graph,
    time_domain=(1.0, 100),
    diffusion=0.1
)

# Auto-detected attributes:
problem.dimension          # "network"
problem.domain_type        # "network"
problem.solver_compatible  # ["network_solver"]
```

#### Mode 5: Custom Components (Advanced)

```python
# Full mathematical control
from mfg_pde.core import MFGComponents

components = MFGComponents(
    hamiltonian_func=lambda x, m, p, t: ...,
    hamiltonian_dm_func=lambda x, m, p, t: ...,
    initial_density_func=lambda x: ...,
    final_value_func=lambda x: ...,
)

problem = MFGProblem(
    spatial_bounds=[(0, 1), (0, 1)],
    spatial_discretization=[50, 50],
    Nt=50,
    components=components
)
```

---

## Implementation Strategy

### Step 1: Extend MFGProblem (Week 1-2)

**Add new initialization modes to existing `MFGProblem.__init__`**:

```python
class MFGProblem:
    def __init__(
        self,
        # Legacy 1D (existing)
        xmin: float | None = None,
        xmax: float | None = None,
        Nx: int | None = None,
        Lx: float | None = None,  # NEW: alternative to xmin/xmax

        # N-D grid (existing)
        spatial_bounds: list[tuple[float, float]] | None = None,
        spatial_discretization: list[int] | None = None,

        # Complex geometry (NEW)
        geometry: BaseGeometry | None = None,
        obstacles: list | None = None,

        # Network (NEW)
        network: NetworkGraph | None = None,

        # Common parameters
        T: float | None = None,
        Nt: int | None = None,
        time_domain: tuple[float, int] | None = None,  # NEW: alternative to T/Nt

        sigma: float = 1.0,
        diffusion: float | None = None,  # NEW: alias for sigma
        coupling_coefficient: float = 0.5,

        # Advanced
        components: MFGComponents | None = None,
        **kwargs
    ):
        # Detect initialization mode
        mode = self._detect_init_mode(
            Nx=Nx,
            spatial_bounds=spatial_bounds,
            geometry=geometry,
            network=network
        )

        # Dispatch to appropriate initializer
        if mode == "1d_legacy":
            self._init_1d_legacy(...)
        elif mode == "nd_grid":
            self._init_nd_grid(...)
        elif mode == "geometry":
            self._init_geometry(...)
        elif mode == "network":
            self._init_network(...)
        else:
            raise ValueError(f"Could not determine initialization mode...")

        # Set components
        self.components = components or self._create_default_components()

        # Auto-detect solver compatibility
        self._detect_solver_compatibility()
```

**Key Design Decision**: Extend existing `MFGProblem` rather than create new class.
**Rationale**: Perfect backward compatibility, single import, unified API.

### Step 2: Add Auto-Detection (Week 2-3)

```python
def _detect_solver_compatibility(self) -> None:
    """
    Determine which solver types are compatible with this problem.

    Sets self.solver_compatible: list[str]
    """
    compatible = []

    # FDM: Requires regular grid, no complex geometry
    if self.domain_type == "grid" and not self.has_obstacles:
        compatible.append("fdm")

    # GFDM: Works with grids and complex geometry
    if self.domain_type in ["grid", "implicit"]:
        compatible.append("gfdm")

    # Semi-Lagrangian: Works with grids
    if self.domain_type == "grid":
        compatible.append("semi_lagrangian")

    # Particle: Works with everything except networks
    if self.domain_type != "network":
        compatible.append("particle")

    # Network solver: Only for network problems
    if self.domain_type == "network":
        compatible.append("network_solver")

    self.solver_compatible = compatible

def validate_solver_type(self, solver_type: str) -> None:
    """
    Validate solver compatibility and raise clear error if incompatible.

    Called by solver constructors.
    """
    if solver_type not in self.solver_compatible:
        raise ValueError(
            f"Solver type '{solver_type}' incompatible with this problem.\n"
            f"Problem type: {self.domain_type}, dimension: {self.dimension}\n"
            f"Compatible solvers: {self.solver_compatible}\n"
            f"Suggestion: Use create_fast_solver() for automatic solver selection."
        )
```

### Step 3: Unified Attribute Interface (Week 3-4)

**Goal**: All problems provide consistent attributes regardless of initialization mode.

**Required Attributes** (all problem types):
```python
# Dimension and domain
self.dimension: int | str      # 1, 2, 3, ..., or "network"
self.domain_type: str          # "grid", "implicit", "network"
self.solver_compatible: list[str]

# Time domain
self.T: float
self.Nt: int
self.Dt: float
self.tSpace: np.ndarray        # (Nt+1,)

# Physical parameters
self.sigma: float
self.coupling_coefficient: float

# Components
self.components: MFGComponents

# Boundary conditions
self.boundary_conditions: BoundaryConditions
```

**Grid-Specific Attributes** (when domain_type == "grid"):
```python
# 1D grid (dimension == 1)
self.xmin, self.xmax, self.Lx, self.Nx, self.Dx: float/int
self.xSpace: np.ndarray        # (Nx+1,)

# N-D grid (dimension > 1)
self.spatial_bounds: list[tuple[float, float]]
self.spatial_discretization: list[int]
self.spatial_shape: tuple[int, ...]  # (Nx+1, Ny+1, ...)
self.grid: TensorProductGrid   # Unified grid object
```

**Geometry-Specific Attributes** (when domain_type == "implicit"):
```python
self.geometry: BaseGeometry
self.mesh_data: MeshData
self.collocation_points: np.ndarray  # (N, d)
self.has_obstacles: bool
self.obstacles: list | None
```

**Network-Specific Attributes** (when domain_type == "network"):
```python
self.network: NetworkGraph
self.num_nodes: int
self.adjacency_matrix: np.ndarray
```

### Step 4: Absorb GridBasedMFGProblem (Week 4-5)

**Strategy**: Make `GridBasedMFGProblem` an alias/factory for `MFGProblem`.

```python
# In highdim_mfg_problem.py
def GridBasedMFGProblem(
    domain_bounds: tuple,
    grid_resolution: int | tuple[int, ...],
    time_domain: tuple[float, int] = (1.0, 100),
    diffusion_coeff: float = 0.1,
) -> MFGProblem:
    """
    Factory function for grid-based MFG problems (backward compatible).

    DEPRECATED: Use MFGProblem() directly with spatial_bounds parameter.
    Will be removed in v2.0.0.
    """
    import warnings
    warnings.warn(
        "GridBasedMFGProblem is deprecated. Use MFGProblem() directly:\n"
        "  MFGProblem(spatial_bounds=..., spatial_discretization=...)",
        DeprecationWarning,
        stacklevel=2
    )

    # Convert domain_bounds to spatial_bounds format
    dimension = len(domain_bounds) // 2
    spatial_bounds = [
        (domain_bounds[2*i], domain_bounds[2*i+1])
        for i in range(dimension)
    ]

    # Normalize grid_resolution
    if isinstance(grid_resolution, int):
        spatial_discretization = [grid_resolution] * dimension
    else:
        spatial_discretization = list(grid_resolution)

    # Create MFGProblem
    return MFGProblem(
        spatial_bounds=spatial_bounds,
        spatial_discretization=spatial_discretization,
        time_domain=time_domain,
        diffusion=diffusion_coeff
    )
```

**Deprecation Timeline**:
- v0.9.0 (Phase 3.1): Add deprecation warning, keep functional
- v1.0.0: Make it loud warning
- v2.0.0: Remove entirely

### Step 5: Update Solvers (Week 5-7)

**Minimal Changes**: Solvers already handle dimension detection. Just need validation.

**Example** (HJBFDMSolver):
```python
class HJBFDMSolver:
    def __init__(self, problem: MFGProblem, **kwargs):
        # Validate compatibility
        problem.validate_solver_type("fdm")

        # Rest of initialization unchanged
        ...
```

### Step 6: Migration Guide & Testing (Week 7-8)

**Migration Guide** (`docs/migration/UNIFIED_MFGPROBLEM_MIGRATION.md`):
- Before/after examples for each use case
- Automated migration script
- Compatibility table

**Testing**:
- All existing tests pass (backward compatibility)
- New tests for each initialization mode
- Solver compatibility matrix tests
- Error message clarity tests

---

## Migration Examples

### Example 1: Simple 1D Problem

```python
# BEFORE (still works)
from mfg_pde.core.mfg_problem import MFGProblem
problem = MFGProblem(Nx=100, xmin=0, xmax=1, Nt=100)

# AFTER (same, no changes needed)
from mfg_pde import MFGProblem
problem = MFGProblem(Nx=100, xmin=0, xmax=1, Nt=100)
```

### Example 2: 2D Grid Problem

```python
# BEFORE (deprecated)
from mfg_pde.core.highdim_mfg_problem import GridBasedMFGProblem
problem = GridBasedMFGProblem(
    domain_bounds=(0, 1, 0, 1),
    grid_resolution=50,
    time_domain=(1.0, 100)
)

# AFTER (preferred)
from mfg_pde import MFGProblem
problem = MFGProblem(
    spatial_bounds=[(0, 1), (0, 1)],
    spatial_discretization=[50, 50],
    time_domain=(1.0, 100)
)
```

### Example 3: Complex Geometry

```python
# BEFORE (required custom subclass)
class MazeProblem(HighDimMFGProblem):
    def setup_components(self):
        self.components = MFGComponents(...)
    # ... 50+ lines of boilerplate

problem = MazeProblem(geometry=domain)

# AFTER (direct initialization)
from mfg_pde import MFGProblem
problem = MFGProblem(
    geometry=domain,
    obstacles=obstacles,
    time_domain=(1.0, 50),
    components=components  # Optional
)
```

---

## Success Criteria

### Phase 3.1 Complete When:

1. ✅ Single `MFGProblem` class handles all 5 initialization modes
2. ✅ All 25 solver×dimension combinations documented as supported or unsupported
3. ✅ Auto-detection correctly identifies solver compatibility
4. ✅ Clear error messages for incompatible solver types
5. ✅ All existing tests pass (100% backward compatibility)
6. ✅ Migration guide with automated script available
7. ✅ Research repo (mfg-research) migrated successfully
8. ✅ Zero custom problem classes needed for standard use cases

### Metrics

**Before**:
- 5 different problem classes
- Only 4/25 solver combinations work natively
- 1,080 lines of custom problem code per project

**After**:
- 1 unified problem class
- 25/25 combinations either work or have clear error
- 0 lines of custom problem code for standard cases
- Integration overhead: <2× (down from 7.6×)

---

## Risk Assessment

### Low Risk
- ✅ Backward compatibility (100% via legacy mode)
- ✅ Testing (existing tests ensure no regressions)

### Medium Risk
- ⚠️ API surface complexity (many parameters)
  * Mitigation: Clear documentation, validation, good defaults
- ⚠️ User confusion during transition
  * Mitigation: Deprecation warnings, migration guide, examples

### High Risk
- ❌ Solver compatibility edge cases
  * Mitigation: Comprehensive compatibility matrix tests
  * Conservative auto-detection (explicit > implicit)

---

## Timeline & Milestones

### Week 1-2: Core Extension
- Extend MFGProblem.__init__ with new modes
- Implement mode detection logic
- Add geometry and network initialization

### Week 3-4: Auto-Detection & Validation
- Implement solver compatibility detection
- Add validation methods
- Unified attribute interface

### Week 4-5: Deprecation Path
- GridBasedMFGProblem factory function
- Deprecation warnings
- Migration script

### Week 5-7: Solver Updates
- Add validation calls to all solvers
- Update solver documentation
- Error message improvements

### Week 7-8: Testing & Documentation
- Comprehensive test suite
- Migration guide
- Update all examples
- Research repo migration

---

## Next Steps

1. **Review & Approve**: Get feedback on this design
2. **Create Issue**: Track Phase 3.1 work
3. **Create Branch**: `feature/unified-mfg-problem`
4. **Start Implementation**: Begin with core extension

---

**Status**: Design Complete - Ready for Review
**Last Updated**: 2025-11-03
**Author**: Architecture Refactoring Team
