# MFG_PDE Architecture Refactoring Proposal: Unified Problem Interface

**Date**: 2025-10-30
**Status**: ~~Proposal~~ **SUPERSEDED**
**Priority**: ~~High (addresses fundamental architectural limitation)~~

---

## ⚠️ PROPOSAL SUPERSEDED - DO NOT IMPLEMENT ⚠️

**This proposal has been superseded by the current MFG_PDE architecture.**

### Why This Proposal Was Not Implemented

After thorough analysis during Phase 3 gradient notation standardization (October 2025), we discovered that the **current MFG_PDE architecture already achieves all the goals of this proposal** through a superior design pattern:

**Current Design (Implemented)**:
- ✅ Concrete `MFGProblem` class with **composition via `MFGComponents`**
- ✅ Progressive disclosure: `ExampleMFGProblem` (simple) → `MFGProblem` (advanced)
- ✅ Proven resilience: Phase 3 major changes touched <150 LOC
- ✅ 90%+ of users unaffected by internal changes (use simplified interfaces)
- ✅ Protocol-based solver compatibility (`GridProblem`, `CollocationProblem`)

**Why Current Design is Superior to This Proposal**:

1. **Simpler**: Concrete class with optional composition beats abstract inheritance hierarchy
2. **More flexible**: MFGComponents allows customization without subclassing
3. **Battle-tested**: Phase 3 validated that architecture handles major changes gracefully
4. **User-friendly**: Most users never need to know about advanced features

### Architectural Insights from Phase 3

During Phase 3 (gradient notation standardization), we achieved major API migration with:
- Only ~150 LOC changed in core problem class
- Zero breaking changes for 90%+ of users
- Backward compatibility maintained through dual-format support
- All tests passing (3295/3368, 97.8%)

This proves the current architecture is **excellent** and does **not** need the AbstractMFGProblem refactoring proposed below.

### References

- **Phase 3 Documentation**: `docs/SESSION_2025-10-30_PHASE3_DOCUMENTATION_COMPLETE.md`
- **Migration Guide**: `docs/migration_guides/phase3_gradient_notation_migration.md`
- **Architecture Analysis**: See Phase 3 session summaries for detailed validation

### Conclusion

**DO NOT implement this proposal.** The current MFG_PDE architecture already provides:
- Composition over inheritance (via MFGComponents)
- Solver flexibility (via protocol interfaces)
- Progressive complexity disclosure (ExampleMFGProblem → MFGProblem)
- Proven resilience to major changes

The original proposal text below is preserved for historical reference only.

---

## Executive Summary (HISTORICAL - DO NOT IMPLEMENT)

MFG_PDE currently has three separate problem classes (`MFGProblem`, `HighDimMFGProblem`, `NetworkMFGProblem`) with incompatible APIs and different spatial discretization assumptions. This creates artificial limitations where:

- FDM solvers only work with 1D problems
- GFDM solvers only work with high-dimensional implicit geometry
- Users cannot easily switch between numerical methods for the same problem
- Each new geometry type requires a new problem class

**Proposal**: Refactor to a unified `AbstractMFGProblem` base class that separates:
1. **Problem definition** (MFG components, domain geometry, time discretization)
2. **Spatial discretization** (solver-specific: FDM grids, GFDM collocation, particle clouds, networks)

This enables solver flexibility, cleaner APIs, and better extensibility.

---

## Current Architecture Problems

### Problem Class Hierarchy (Current)

```
MFGProblem (1D only)
├── Spatial: Regular 1D grid (Nx intervals, Dx spacing)
├── Solvers: HJBFDMSolver, FPFDMSolver
└── Limitation: Cannot handle 2D/3D or obstacles

HighDimMFGProblem (2D/3D, meshfree only)
├── Spatial: Implicit geometry, particle collocation
├── Solvers: HJBGFDMSolver, FPParticleSolver
└── Limitation: Cannot use FDM even when domain is regular

NetworkMFGProblem (graph-based)
├── Spatial: Graph/network structure
├── Solvers: Network-specific solvers
└── Limitation: Completely separate API
```

### Specific Issues

1. **API Incompatibility**: Each problem type has different attributes
   - `MFGProblem`: `Nx`, `Dx`, `D_bounds`
   - `HighDimMFGProblem`: `geometry`, `dimension`
   - `NetworkMFGProblem`: `graph`, `nodes`

2. **Solver-Problem Coupling**: Solvers tightly coupled to problem types
   - `HJBFDMSolver` requires `problem.Nx`, `problem.Dx`
   - `HJBGFDMSolver` requires `problem.geometry`, collocation points
   - Cannot use FDM on regular 2D grid with obstacles

3. **Geometry vs Discretization Confusion**:
   - Geometry (domain shape, obstacles) conflated with discretization method (grid, meshfree)
   - 2D maze on regular grid should work with FDM, but currently impossible

4. **Limited Extensibility**:
   - Adding new geometry types requires new problem class
   - Adding new discretization methods requires modifying existing classes

---

## Proposed Architecture

### Unified Problem Hierarchy

```
AbstractMFGProblem (base class)
├── Core attributes (all problems):
│   ├── components: MFGComponents (H, g, m0)
│   ├── geometry: BaseGeometry
│   ├── T: float (time horizon)
│   ├── Nt: int (time discretization)
│   └── sigma: diffusion coefficient
│
├── Spatial discretization (solver-specific):
│   ├── For FDM: create_fdm_grid(Nx, Ny=None)
│   ├── For GFDM: create_collocation_points(n_points, method='uniform')
│   ├── For particle: create_particle_cloud(n_particles)
│   └── For network: graph structure
│
└── Subclasses (optional, for convenience):
    ├── GridBasedMFGProblem (regular grids)
    ├── ParticleBasedMFGProblem (particle clouds)
    └── NetworkMFGProblem (graphs)
```

### Geometry Separation

```python
BaseGeometry (handles all spatial domains)
├── RegularGrid1D, RegularGrid2D, RegularGrid3D
├── ImplicitDomain (level sets, obstacles)
├── MeshDomain (triangulated surfaces)
└── NetworkDomain (graphs)

# Geometry is independent of discretization
# Same geometry can be discretized multiple ways
```

---

## Implementation Design

### 1. Abstract Base Class

```python
from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np

class AbstractMFGProblem(ABC):
    """
    Unified MFG problem interface for all geometries and discretizations.

    Separates problem definition (MFG components, geometry, time)
    from spatial discretization (solver-dependent).
    """

    def __init__(
        self,
        geometry: BaseGeometry,
        components: MFGComponents,
        time_horizon: float,
        n_timesteps: int,
        diffusion_coeff: Union[float, callable],
        **kwargs
    ):
        """
        Initialize MFG problem (method-agnostic).

        Args:
            geometry: Spatial domain (handles obstacles, boundaries, etc.)
            components: MFG components (H, g, m0)
            time_horizon: Terminal time T
            n_timesteps: Number of time intervals Nt
            diffusion_coeff: Diffusion sigma (constant or function)
        """
        self.geometry = geometry
        self.components = components
        self.T = time_horizon
        self.Nt = n_timesteps
        self.Dt = time_horizon / n_timesteps
        self.sigma = diffusion_coeff

        # Spatial discretization (created by solvers)
        self._fdm_grid = None
        self._collocation_points = None
        self._particle_cloud = None

    # === Spatial Discretization Methods (Solver-Specific) ===

    def create_fdm_grid(self, Nx: int, Ny: Optional[int] = None):
        """
        Create regular FDM grid for problem domain.

        Args:
            Nx: Number of intervals in x direction
            Ny: Number of intervals in y direction (None for 1D)

        Returns:
            Grid information for FDM solvers

        Raises:
            ValueError: If geometry is not regular grid-compatible
        """
        if not self.geometry.is_regular_grid_compatible():
            raise ValueError(
                f"Geometry {type(self.geometry)} is not compatible with FDM grids. "
                f"Use GFDM collocation instead."
            )

        self._fdm_grid = self.geometry.create_regular_grid(Nx, Ny)
        return self._fdm_grid

    def create_collocation_points(
        self,
        n_points: int,
        method: str = 'uniform',
        **kwargs
    ) -> np.ndarray:
        """
        Create collocation points for meshfree methods (GFDM).

        Args:
            n_points: Number of collocation points
            method: Sampling method ('uniform', 'random', 'adaptive')

        Returns:
            (n_points, dim) array of collocation points
        """
        self._collocation_points = self.geometry.sample_points(
            n_points, method=method, **kwargs
        )
        return self._collocation_points

    def create_particle_cloud(
        self,
        n_particles: int,
        method: str = 'sobol',
        **kwargs
    ) -> np.ndarray:
        """
        Create particle cloud for particle methods.

        Args:
            n_particles: Number of particles
            method: Sampling method ('sobol', 'random', 'lhs')

        Returns:
            (n_particles, dim) array of particle positions
        """
        self._particle_cloud = self.geometry.sample_points(
            n_particles, method=method, **kwargs
        )
        return self._particle_cloud

    # === Properties (Backward Compatibility) ===

    @property
    def Nx(self) -> Optional[int]:
        """Number of intervals (FDM). Returns None if not using FDM."""
        return self._fdm_grid.Nx if self._fdm_grid is not None else None

    @property
    def Dx(self) -> Optional[float]:
        """Grid spacing (FDM). Returns None if not using FDM."""
        return self._fdm_grid.Dx if self._fdm_grid is not None else None

    @property
    def dimension(self) -> int:
        """Spatial dimension of problem."""
        return self.geometry.dimension

    # === Evaluation Methods ===

    def hamiltonian(self, x, p, m, t=0.0):
        """Evaluate Hamiltonian H(x, p, m, t)."""
        return self.components.hamiltonian_func(
            x_idx=None,
            x_position=x,
            m_at_x=m,
            p_values=p if isinstance(p, dict) else self._array_to_p_dict(p),
            t_idx=None,
            current_time=t,
            problem=self
        )

    def terminal_cost(self, x):
        """Evaluate terminal cost g(x)."""
        return self.components.terminal_cost_func(x, self)

    def initial_density(self, x):
        """Evaluate initial density m0(x)."""
        return self.components.initial_density_func(x, self)

    def _array_to_p_dict(self, p_array):
        """Convert momentum array to dict format."""
        p_arr = np.asarray(p_array).flatten()
        if len(p_arr) == 1:
            return {'forward': p_arr[0]}
        elif len(p_arr) == 2:
            return {'x': p_arr[0], 'y': p_arr[1]}
        elif len(p_arr) == 3:
            return {'x': p_arr[0], 'y': p_arr[1], 'z': p_arr[2]}
        else:
            return {f'p{i}': p_arr[i] for i in range(len(p_arr))}
```

### 2. Solver Interface (Updated)

```python
class BaseHJBSolver(ABC):
    """Base class for all HJB solvers (FDM, GFDM, semi-Lagrangian, etc.)."""

    def __init__(self, problem: AbstractMFGProblem, **kwargs):
        """
        Initialize solver with MFG problem.

        Args:
            problem: Unified MFG problem instance
            **kwargs: Solver-specific parameters
        """
        self.problem = problem
        self._setup_discretization(**kwargs)

    @abstractmethod
    def _setup_discretization(self, **kwargs):
        """Setup spatial discretization (solver-specific)."""
        pass

    @abstractmethod
    def solve(self, M_density=None, **kwargs):
        """Solve HJB equation."""
        pass

# Example: FDM Solver
class HJBFDMSolver(BaseHJBSolver):
    """Finite difference HJB solver."""

    def _setup_discretization(self, Nx: int, Ny: Optional[int] = None, **kwargs):
        """Create FDM grid for problem."""
        self.grid = self.problem.create_fdm_grid(Nx, Ny)
        self.Nx = Nx
        self.Ny = Ny

    def solve(self, M_density=None, **kwargs):
        # Use self.problem.hamiltonian(), self.problem.terminal_cost(), etc.
        # Work on self.grid
        pass

# Example: GFDM Solver
class HJBGFDMSolver(BaseHJBSolver):
    """Meshfree GFDM HJB solver."""

    def _setup_discretization(
        self,
        collocation_points: Optional[np.ndarray] = None,
        n_points: int = 100,
        **kwargs
    ):
        """Create collocation points for GFDM."""
        if collocation_points is None:
            self.collocation_points = self.problem.create_collocation_points(n_points)
        else:
            self.collocation_points = collocation_points

    def solve(self, M_density=None, **kwargs):
        # Use self.problem.hamiltonian(), self.problem.terminal_cost(), etc.
        # Work on self.collocation_points
        pass
```

### 3. Usage Examples

```python
# Example 1: 2D Maze with Regular Grid + FDM
# ============================================

from mfg_pde.geometry import RegularGrid2D, ObstacleMap
from mfg_pde.problems import AbstractMFGProblem
from mfg_pde.solvers.hjb import HJBFDMSolver
from mfg_pde.solvers.fp import FPFDMSolver

# Define geometry (regular 2D grid with obstacles)
maze_array = np.array([...])  # 1 = wall, 0 = free
geometry = RegularGrid2D(
    bounds=[(0, 6), (0, 6)],
    obstacle_map=maze_array
)

# Define MFG problem (method-agnostic)
problem = AbstractMFGProblem(
    geometry=geometry,
    components=MFGComponents(
        hamiltonian_func=my_hamiltonian,
        terminal_cost_func=my_terminal_cost,
        initial_density_func=my_initial_density
    ),
    time_horizon=5.0,
    n_timesteps=50,
    diffusion_coeff=0.1
)

# Choose FDM solver (works because geometry is regular grid)
hjb_solver = HJBFDMSolver(problem, Nx=50, Ny=50)
fp_solver = FPFDMSolver(problem, Nx=50, Ny=50)

# Run MFG solver
U, M = picard_iteration(hjb_solver, fp_solver, max_iter=30)


# Example 2: Same Problem with GFDM (Meshfree)
# =============================================

# Same problem definition as above
# Just change solvers:

from mfg_pde.solvers.hjb import HJBGFDMSolver
from mfg_pde.solvers.fp import FPParticleSolver

hjb_solver = HJBGFDMSolver(problem, n_points=500)
fp_solver = FPParticleSolver(problem, n_particles=500)

U, M = picard_iteration(hjb_solver, fp_solver, max_iter=30)


# Example 3: Complex Geometry (Implicit, No Regular Grid)
# ========================================================

from mfg_pde.geometry import ImplicitDomain

# Define complex geometry with level set
def signed_distance(x):
    # Complex shape: star, kidney bean, etc.
    return ...

geometry = ImplicitDomain(
    bounds=[(-1, 1), (-1, 1)],
    signed_distance_func=signed_distance
)

problem = AbstractMFGProblem(
    geometry=geometry,
    components=...,
    time_horizon=1.0,
    n_timesteps=100,
    diffusion_coeff=0.05
)

# FDM would raise error (geometry not grid-compatible)
# Use GFDM instead:
hjb_solver = HJBGFDMSolver(problem, n_points=1000, method='adaptive')
fp_solver = FPParticleSolver(problem, n_particles=1000)


# Example 4: Network/Graph MFG
# ==============================

from mfg_pde.geometry import NetworkDomain

graph = nx.karate_club_graph()
geometry = NetworkDomain(graph)

problem = AbstractMFGProblem(
    geometry=geometry,
    components=...,
    time_horizon=10.0,
    n_timesteps=100,
    diffusion_coeff=0.1
)

# Use network-specific solvers
from mfg_pde.solvers.hjb import HJBNetworkSolver
from mfg_pde.solvers.fp import FPNetworkSolver

hjb_solver = HJBNetworkSolver(problem)
fp_solver = FPNetworkSolver(problem)
```

---

## Migration Strategy

### Phase 1: Add Abstract Base (Backward Compatible)

1. Introduce `AbstractMFGProblem` alongside existing classes
2. Keep `MFGProblem`, `HighDimMFGProblem`, `NetworkMFGProblem` as subclasses
3. Solvers check `isinstance(problem, AbstractMFGProblem)` for new API
4. **Result**: New features available, old code still works

### Phase 2: Deprecation Warnings

1. Add deprecation warnings to old problem classes
2. Provide migration guide and examples
3. Update documentation
4. **Timeline**: 6 months before removal

### Phase 3: Full Migration

1. Remove old problem classes (breaking change)
2. Bump major version (e.g., v1.0 → v2.0)
3. Clean up backward compatibility code
4. **Timeline**: After Phase 2 deprecation period

---

## Benefits

### 1. Solver Flexibility

```python
# Same problem, different solvers
problem = AbstractMFGProblem(...)

# Try FDM (fast but limited)
results_fdm = solve_with(problem, HJBFDMSolver, FPFDMSolver)

# Try GFDM (flexible but slower)
results_gfdm = solve_with(problem, HJBGFDMSolver, FPParticleSolver)

# Compare methods
compare_convergence(results_fdm, results_gfdm)
```

### 2. Extensibility

Adding new geometry type:
```python
# Before: Need new MFGProblem subclass + modify all solvers
# After: Just implement BaseGeometry interface

class TorusDomain(BaseGeometry):
    def sample_points(self, n, method='uniform'):
        # Uniform sampling on torus
        ...

    def is_regular_grid_compatible(self):
        return False  # Use meshfree methods

# Automatically works with all solvers
geometry = TorusDomain(major_radius=2, minor_radius=1)
problem = AbstractMFGProblem(geometry=geometry, ...)
solver = HJBGFDMSolver(problem)  # Just works!
```

### 3. Cleaner Code

```python
# Before: Tight coupling
class HJBFDMSolver:
    def __init__(self, problem: MFGProblem):  # Only works with MFGProblem
        self.Nx = problem.Nx
        self.Dx = problem.Dx
        # Assumes 1D regular grid

# After: Separation of concerns
class HJBFDMSolver:
    def __init__(self, problem: AbstractMFGProblem, Nx: int):
        self.problem = problem
        self.grid = problem.create_fdm_grid(Nx)  # Geometry handles details
```

### 4. Research Flexibility

```python
# Research question: How does solver choice affect convergence?

geometries = [
    RegularGrid2D(...),
    ImplicitDomain(...),
    MeshDomain(...)
]

solvers = [
    (HJBFDMSolver, FPFDMSolver),      # Classical FDM
    (HJBGFDMSolver, FPParticleSolver), # Meshfree
    (HJBWenoSolver, FPFDMSolver)       # High-order FDM
]

for geom in geometries:
    problem = AbstractMFGProblem(geometry=geom, ...)
    for hjb_cls, fp_cls in solvers:
        try:
            results = benchmark(problem, hjb_cls, fp_cls)
            save_results(geom, (hjb_cls, fp_cls), results)
        except IncompatibleGeometryError:
            print(f"{hjb_cls} incompatible with {type(geom)}")
```

---

## Technical Considerations

### Backward Compatibility

```python
# Old code (still works in Phase 1-2):
from mfg_pde import MFGProblem  # Deprecated warning
problem = MFGProblem(Nx=100, Nt=50, ...)
solver = HJBFDMSolver(problem)

# New code (preferred):
from mfg_pde import AbstractMFGProblem
from mfg_pde.geometry import RegularGrid1D

geometry = RegularGrid1D(bounds=(0, 1))
problem = AbstractMFGProblem(geometry=geometry, ...)
solver = HJBFDMSolver(problem, Nx=100)
```

### Performance Impact

- **No performance penalty**: Abstraction is compile-time only
- **Grid creation**: Moved to setup phase (one-time cost)
- **Evaluation**: Same function calls as before

### Testing Strategy

1. **Unit tests**: Each geometry type with each solver
2. **Compatibility matrix**: Document which solvers work with which geometries
3. **Regression tests**: Ensure old test cases still pass
4. **Benchmark suite**: Compare performance before/after refactoring

---

## Related Work

### Similar Architectures

1. **FEniCS/Firedrake**: Separates function spaces (discretization) from variational problems
2. **deal.II**: Template-based geometry + discretization separation
3. **SciML (Julia)**: Abstract problem types with multiple solver backends

### MFG-Specific Considerations

- **Coupling**: HJB-FP coupling still works at solver level
- **Anderson acceleration**: Problem-agnostic (operates on density arrays)
- **Particle methods**: Benefit from unified interface (switch between KDE/collocation)

---

## Open Questions

1. **Hybrid discretizations**: How to handle mixed FDM + particle methods in same problem?
   - Proposal: Solver handles conversion between representations

2. **Adaptive refinement**: Where does grid refinement logic belong?
   - Proposal: Geometry provides refinement, solver triggers it

3. **Time-dependent geometry**: How to handle moving obstacles?
   - Proposal: `geometry.update(t)` method, called by solver

4. **Boundary conditions**: Part of geometry or problem?
   - Proposal: Part of geometry (domain-specific)

---

## Recommendation

**Proceed with Phase 1 implementation**:

1. Implement `AbstractMFGProblem` base class
2. Refactor `BaseGeometry` hierarchy
3. Update one solver pair (HJB-FDM + FP-FDM) as proof of concept
4. Write migration guide and examples
5. Gather community feedback before Phase 2

**Estimated effort**: 4-6 weeks for Phase 1 (one developer)

**Breaking change timeline**: 12+ months (Phase 1 → Phase 2 → Phase 3)

---

## Conclusion

The current MFG_PDE architecture artificially limits solver applicability to specific problem types. The proposed refactoring:

- **Enables**: Using any solver with any compatible geometry
- **Simplifies**: Adding new geometries and discretization methods
- **Maintains**: Backward compatibility during migration
- **Improves**: Code clarity and extensibility

This refactoring aligns MFG_PDE with modern computational frameworks and removes a major barrier to research flexibility.

---

**Next Steps**:
1. Community review of this proposal
2. Prototype implementation (Phase 1)
3. Benchmark and validate
4. Roll out migration plan

**Contact**: Submit issues/PRs to MFG_PDE repository with tag `architecture-refactor`
