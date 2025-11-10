# Changelog

All notable changes to MFG_PDE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Deprecated

- **`Dt` attribute**: Use lowercase `dt` instead (Issue #245). Backward compatibility maintained via deprecated property that emits `DeprecationWarning`. Will be removed in v1.0.0.
- **`Dx` attribute**: Use lowercase `dx` instead (Issue #245). Backward compatibility maintained via deprecated property that emits `DeprecationWarning`. Will be removed in v1.0.0.

### Changed

- **Primary time step attribute**: Changed from `Dt` to `dt` throughout codebase (46 files, ~102 references) following official naming conventions (`docs/NAMING_CONVENTIONS.md` lines 24, 262)
  - Core: `mfg_pde/core/mfg_problem.py`, `mfg_pde/types/problem_protocols.py`
  - Solvers: All HJB, FP, and coupling solvers updated
  - Utilities: `experiment_manager.py`, `hjb_policy_iteration.py`
  - Tests: 15 test files (59 references)
  - Examples: 5 example files (8 references)
  - Benchmarks: 3 benchmark files (4 references)

- **Primary spatial spacing attribute**: Changed from `Dx` to `dx` for 1D problems (same scope as above)

### Migration Guide

**For users**: Update your code to use lowercase attributes:
```python
# OLD (deprecated but works with warnings in v0.12.0)
dt = problem.Dt
dx = problem.Dx

# NEW (recommended)
dt = problem.dt
dx = problem.dx
```

**For developers**: The deprecated uppercase properties will be completely removed in v1.0.0.

## [0.11.0] - 2025-11-10

**Major Release: Dual Geometry Architecture**

This release introduces complete dual geometry support, enabling HJB and FP solvers to use different discretizations. This enables multi-resolution methods (4-15× speedup), FEM meshes with obstacles, hybrid particle-grid methods, and network-based agent models.

### Added

**Dual Geometry Infrastructure (PR #258, Issues #257 & #245 Phase 4)**

- **GeometryProjector** class (`mfg_pde/geometry/projection.py`, 706 lines)
  - Automatic projection method selection based on geometry types
  - `project_hjb_to_fp()`: Maps HJB solution values to FP geometry
  - `project_fp_to_hjb()`: Maps FP density values to HJB geometry
  - Supports grid-to-grid, grid-to-particles, particles-to-grid (KDE)
  - Vectorized implementations for 1D/2D/3D

- **ProjectionRegistry** pattern
  - Decorator-based registration: `@ProjectionRegistry.register(SourceType, TargetType, direction)`
  - Hierarchical fallback: exact type → category match → generic
  - O(N) custom projectors (not O(N²))
  - User-extensible for custom geometry types

- **MFGProblem Dual Geometry Integration** (`mfg_pde/core/mfg_problem.py`)
  - New parameters: `hjb_geometry` and `fp_geometry`
  - Automatic `GeometryProjector` creation when geometries differ
  - Unified attribute access: `problem.hjb_geometry`, `problem.fp_geometry`
  - Full backward compatibility with single `geometry` parameter

- **FEM Mesh Support**
  - Automatic Delaunay interpolation for `UnstructuredMesh` ↔ `CartesianGrid` (requires scipy)
  - Nearest neighbor fallback when scipy unavailable
  - Works with Mesh2D, Mesh3D, TriangularAMRMesh
  - Graceful extrapolation handling (fills NaN with nearest neighbor)

- **Vectorized Grid Interpolators**
  - `SimpleGrid1D.get_interpolator()`: Binary search-based 1D interpolation
  - `SimpleGrid2D/3D.get_interpolator()`: RegularGridInterpolator wrapper
  - Accepts array of query points for batch interpolation
  - Used by projection system for efficient grid-to-mesh operations

### Documentation

**Comprehensive Dual Geometry Documentation** (5,000+ lines)

- **Theory**: `docs/theory/geometry_projection_mathematical_formulation.md` (556 lines)
  - Mathematical formulation of all projection methods
  - Error analysis (interpolation, KDE, nearest neighbor)
  - Performance complexity analysis (O(N log N), O(N), etc.)
  - Pseudocode for all algorithms

- **Developer Guide**: `docs/development/GEOMETRY_PROJECTION_IMPLEMENTATION_GUIDE.md` (797 lines)
  - Adding new geometry types and projections
  - Registry pattern usage and best practices
  - Debugging tips and performance optimization
  - Complete code examples for custom projections

- **User Guide**: `docs/user_guide/dual_geometry_usage.md` (679 lines)
  - Complete workflow examples
  - Use cases: multi-resolution, hybrid methods, network agents
  - Performance tips and FAQ
  - Best practices for choosing projection methods

- **FEM Mesh Guide**: `docs/user_guide/fem_mesh_projection_guide.md` (352 lines)
  - FEM mesh support levels (basic + optimized)
  - Comparison of nearest neighbor vs Delaunay
  - Use cases: complex domains, obstacles, CAD import
  - Complete examples with performance tips

- **Migration Guide Update**: `docs/migration/unified_problem_migration.md`
  - Updated with dual geometry integration
  - Examples showing unified API + dual geometry together
  - Updated deprecation timeline with v0.11.0 milestone

- **Completion Summary**: `docs/development/ISSUE_257_COMPLETION_SUMMARY.md` (379 lines)
  - Complete implementation details for all 5 phases
  - Performance impact and testing results
  - Known limitations and future enhancements

### Examples

- **Multi-Resolution MFG**: `examples/basic/dual_geometry_multiresolution.py` (323 lines)
  - Fine HJB grid (100×100) + coarse FP grid (25×25)
  - Demonstrates 4× speedup with minimal accuracy loss
  - Complete visualization of projections
  - Performance comparison with unified geometry

- **FEM Mesh with Obstacles**: `examples/advanced/dual_geometry_fem_mesh.py` (330 lines)
  - Complex domain with circular obstacle using Gmsh
  - Automatic vs manual Delaunay registration
  - Accuracy comparison of projection methods
  - Working example with 495 vertices, 884 elements

### Testing

- **Projection Tests**: `tests/unit/geometry/test_geometry_projection.py` (439 lines)
  - 20 unit tests covering all projection methods
  - Shape verification, accuracy tests, conservation tests
  - Tests for 1D, 2D, 3D projections
  - Registry pattern tests

- **Integration Tests**: `tests/unit/test_core/test_mfg_problem.py` (+131 lines)
  - 7 new tests for dual geometry MFGProblem integration
  - Backward compatibility verification
  - Error handling and validation tests

### Use Cases Enabled

| Use Case | HJB Geometry | FP Geometry | Benefit |
|----------|--------------|-------------|---------|
| Multi-resolution | Fine grid | Coarse grid | 4-15× speedup, 46% memory savings |
| Complex domains | Regular grid | FEM mesh | Fast HJB, handles obstacles naturally |
| Hybrid methods | Grid | Particles | Grid-based value, particle density |
| Network agents | Grid | Network graph | Spatial value, network-constrained agents |

### Performance

- Multi-resolution: 4-15× speedup (depending on resolution ratio)
- Projection overhead: <1% of solve time
- Memory savings: Up to 46% for 4× resolution ratio
- Grid→Points: O(N) with RegularGridInterpolator
- Particles→Grid KDE: GPU-accelerated available (1D)

### Changed

- README updated with v0.11.0 features and dual geometry examples
- Citation updated to v0.11.0

### Backward Compatibility

- ✅ Fully backward compatible
- Existing code using single `geometry` parameter continues to work
- `hjb_geometry` and `fp_geometry` are optional
- No breaking changes

### Closes

- Issue #257: Dual geometry architecture (5 phases complete)
- Issue #245 Phase 4: Documentation for unified MFG problem

---

## [0.10.0] - 2025-11-05

**Major Release: Geometry-First API**

This release introduces the geometry-first API, a new recommended pattern for constructing MFG problems using geometry objects. This provides better type safety, clearer separation of concerns, and unified support for diverse geometry types.

### Added

**PR #244: Phase 2 Array Notation - Backward Compatible Implementation**
- Added `_normalize_to_array()` helper method in `MFGProblem` (`mfg_problem.py:79-122`)
  - Automatically converts scalar inputs to arrays
  - Emits `DeprecationWarning` for scalar usage
  - Points users to `MATHEMATICAL_NOTATION_STANDARD.md`
- Updated `MFGProblem.__init__` signature to accept both scalar and array inputs:
  - `Nx`: `int | list[int]` (deprecated scalar, standard array)
  - `xmin`, `xmax`: `float | list[float]` (deprecated scalar, standard array)
- Both scalar and array inputs produce identical results with 100% backward compatibility
- Migration path for Phase 3 (v1.0.0): Remove deprecated scalar API

**PR #247: GeometryProtocol Foundation**
- Created `GeometryProtocol` runtime-checkable Protocol (`mfg_pde/geometry/geometry_protocol.py`)
  - Minimal interface for all geometry objects
  - Four required properties: `dimension`, `geometry_type`, `num_spatial_points`, `get_spatial_grid()`
- Created `GeometryType` enum with 7 types:
  - `CARTESIAN_GRID`: Regular tensor product grids
  - `NETWORK`: Graph/network geometries
  - `MAZE`: Maze environments
  - `DOMAIN_2D`, `DOMAIN_3D`, `DOMAIN_1D`: Cartesian/unstructured meshes
  - `IMPLICIT`: Level sets and signed distance functions
  - `CUSTOM`: User-defined geometries
- Added helper functions:
  - `detect_geometry_type()`: Self-aware type detection via attribute inspection
  - `is_geometry_compatible()`: Compatibility checking
  - `validate_geometry()`: Validation with informative error messages
- Implemented GeometryProtocol for 6 core geometry classes:
  - `Domain1D`: 1D Cartesian grids with grid caching
  - `BaseGeometry`: Abstract base for Domain2D/Domain3D meshes
  - `TensorProductGrid`: Arbitrary-dimension structured grids
  - `NetworkGeometry`: Graph-based geometries (Grid/Random/ScaleFree networks)
  - `ImplicitDomain`: Meshfree domains via signed distance functions (`Hyperrectangle`, `Hypersphere`)
  - `Grid` (mazes): Maze-based geometries from PerfectMazeGenerator
- Comprehensive design documentation (`docs/development/UNIFIED_GEOMETRY_PARAMETER_DESIGN.md`, 844 lines)

**Geometry-First API Implementation**
- Updated `MFGProblem._init_geometry()` to accept any GeometryProtocol-compliant object (`mfg_problem.py:647-768`)
  - Automatic geometry type detection via `geometry.geometry_type` enum
  - Specialized handling for CARTESIAN_GRID, IMPLICIT, DOMAIN_2D/3D, MAZE, NETWORK types
  - Generic fallback for CUSTOM geometries
- Added deprecation warnings for manual grid construction (`mfg_problem.py:350-363, 430-450`)
  - Warns users to migrate to geometry-first API
  - Points to migration guide with code examples
  - 100% backward compatibility maintained
- Created `docs/migration/GEOMETRY_FIRST_API_GUIDE.md` (400+ lines)
  - Quick start examples for all geometry types
  - Migration strategy from old to new API
  - Performance considerations and FAQ
- Created `examples/basic/geometry_first_api_demo.py` (350+ lines)
  - Demonstrates 8 geometry patterns (TensorProductGrid, Domain1D, Hyperrectangle, Hypersphere, Maze, 4D, reuse, refinement)
  - All examples tested and working
- Fixed normalization bug for implicit geometries (`mfg_problem.py:1158-1172`)
  - Handles `None` spatial_bounds for SDF-based geometries
  - Uses uniform approximation when structured grid info unavailable

### Changed

**API Improvements**
- `MFGProblem` now accepts both scalar and array notation for spatial parameters
- `MFGProblem` now accepts geometry objects via `geometry=` parameter (NEW recommended API)
- Array notation is the standard for manual construction (following `MATHEMATICAL_NOTATION_STANDARD.md`)
- Scalar inputs and manual grid construction trigger deprecation warnings

**Code Quality**
- Unified geometry interface across all geometry types via GeometryProtocol
- Protocol-based design enables duck typing without explicit inheritance
- Self-aware geometry types for automatic type detection in MFGProblem
- Enhanced type safety and consistency across geometry module
- Separation of concerns: geometry construction vs. problem temporal/diffusion parameters

### Deprecated

**API Patterns** (will be restricted in v1.0.0, removed in v2.0.0)
- Manual grid construction in `MFGProblem` (passing `spatial_bounds`, `spatial_discretization`, `xmin`, `xmax`, `Nx`)
  - Use geometry-first API instead: create geometry object, pass to `MFGProblem(geometry=...)`
  - Deprecation warnings provide migration examples
  - See `docs/migration/GEOMETRY_FIRST_API_GUIDE.md` for complete guide
- Scalar `Nx`, `xmin`, `xmax` parameters (if still using manual construction)
  - Use arrays instead: `Nx=[100]`, `xmin=[-2.0]`, `xmax=[2.0]`
  - Warnings guide users to `MATHEMATICAL_NOTATION_STANDARD.md`

**Deprecation Timeline**:
- v0.10.x: Warnings emitted, old API fully functional
- v0.11.x - v0.99.x: Continued warnings
- v1.0.0: Manual construction requires explicit `allow_manual_construction=True` flag
- v2.0.0: Complete removal of manual construction

### Documentation

- Array-Based Notation Migration plan (`docs/development/ARRAY_BASED_NOTATION_MIGRATION.md`)
- Mathematical Notation Standard (`docs/development/MATHEMATICAL_NOTATION_STANDARD.md`)
- Unified Geometry Parameter Design (`docs/development/UNIFIED_GEOMETRY_PARAMETER_DESIGN.md`)
- Geometry-First API Guide (`docs/migration/GEOMETRY_FIRST_API_GUIDE.md`)

### Future Work (Planned for 0.10.x series)

**v0.10.1** (Planned):
- Add GeometryProtocol compliance to AMR classes (OneDimensionalAMRMesh, AdaptiveMesh, TriangularAMRMesh, TetrahedralAMRMesh)
- Enable AMR meshes to be used directly in `MFGProblem(geometry=amr_mesh)`

**v0.10.2** (Planned):
- Design and implement dimension-agnostic boundary condition system (`BoundaryConditionND`)
- Support for nD boundary conditions (d > 3) with per-axis BC specification

**v0.10.3** (Planned):
- Rename `BaseGeometry` → `MeshGeometry` for clarity (breaking change with deprecation)
- Update all documentation and examples to reflect renamed class

### Testing

- All 3300+ tests passing
- Array notation backward compatibility validated
- GeometryProtocol compliance verified for all implemented geometries

## [0.9.1] - 2025-11-04

### Added

**PR #242: GFDM Operators with Unified Smoothing Kernels**
- `mfg_pde/utils/numerical/smoothing_kernels.py` (807 lines)
  - Unified kernel implementations: Gaussian, Wendland, Cubic Spline, Quintic Spline, Cubic, Quartic
  - Parameterized Wendland kernels: `WendlandKernel(k=0,1,2,3)` for C^0, C^2, C^4, C^6 smoothness
  - Arbitrary dimension support with proper normalization
  - Factory pattern: `create_kernel(kernel_type, dimension)`
  - Derivative support for gradient-based methods
- `mfg_pde/utils/numerical/gfdm_operators.py` (1050 lines)
  - Weighted least squares gradient/Hessian reconstruction
  - Support for structured and unstructured grids
  - Boundary condition handling (Dirichlet, Neumann)
  - Anisotropic/directional derivative support
- Theory documentation with differential operators (gradient, divergence, Laplacian)
- Comprehensive test suite (502 lines, 54 tests)
- Advanced example demos for nD geometry and implicit geometry

**PR #239: Maze Refactoring**
- Moved maze generation from `alg/reinforcement/environments` to `geometry/mazes`
- Makes maze utilities accessible to all solver types (PDE, particle, neural, RL)
- Backward compatibility through re-exports
- 6 core files relocated: `maze_generator`, `hybrid_maze`, `voronoi_maze`, `maze_config`, `maze_utils`, `maze_postprocessing`

### Changed
- Updated solver integrations to use unified kernel API
- Consolidated 4 separate Wendland classes into single parameterized implementation
- Updated test imports to reference new maze location

### Documentation
- Added `docs/theory/smoothing_kernels_mathematical_formulation.md` with complete mathematical foundations
- Dimension-specific formulas for differential operators (1D, 2D, 3D)
- SPH and GFDM application notes
- Implementation details with code references

### Testing
- All 3300+ tests passing
- New GFDM operator tests validated against analytical solutions
- Kernel tests cover edge cases, normalization, and derivatives

## [0.9.0] - 2025-11-03

### Phase 3 Complete: Unified Architecture

Major architecture refactoring completing Phase 3.1 (MFGProblem), Phase 3.2 (SolverConfig), and Phase 3.3 (Factory Integration).

### Added

**Issue #216: Missing Utilities (Complete - All 4 Parts)**
- **Part 1: Particle Interpolation** (commit 84e6e6d)
  - `interpolate_grid_to_particles()` - Grid → Particles (1D/2D/3D)
  - `interpolate_particles_to_grid()` - Particles → Grid (RBF, KDE, nearest)
  - `estimate_kde_bandwidth()` - Automatic bandwidth selection
  - Saves ~220 lines per research project
- **Part 2: Signed Distance Functions** (commit 83f59f4)
  - Primitives: `sdf_sphere()`, `sdf_box()` for 1D/2D/3D/nD
  - CSG operations: `sdf_union()`, `sdf_intersection()`, `sdf_complement()`, `sdf_difference()`
  - Smooth blending: `sdf_smooth_union()`, `sdf_smooth_intersection()`
  - Gradient: `sdf_gradient()` using finite differences
  - Saves ~150 lines per research project
- **Part 3: QP Solver Caching** (already existed)
  - `QPCache` - Hash-based caching with LRU eviction
  - `QPSolver` - Unified solver with warm-starting
  - Multiple backends: OSQP, scipy SLSQP, scipy L-BFGS-B
  - Saves ~180 lines per project + 2-5× GFDM speedup
- **Part 4: Convergence Monitoring** (already existed)
  - `AdvancedConvergenceMonitor` - Plotting, stagnation detection
  - `AdaptiveConvergenceWrapper` - Adaptive convergence criteria
  - Saves ~60 lines per project
- **Total Impact**: ~610 lines saved per research project + performance improvements

**Phase 3.1: Unified Problem Class (PR #218)**
- Single `MFGProblem` class replacing 5+ specialized problem classes
- Flexible `MFGComponents` system for custom problem definitions
- Auto-detection of problem types (standard, network, variational, stochastic, highdim)
- `MFGProblemBuilder` for programmatic problem construction
- Full backward compatibility with deprecated specialized classes

**Phase 3.2: Unified Configuration System (PR #222)**
- New `SolverConfig` class unifying 3 competing config systems
- Three usage patterns:
  - YAML files for experiments and reproducibility
  - Builder API for programmatic configuration
  - Presets for common use cases
- Modular config components: `PicardConfig`, `HJBConfig`, `FPConfig`, `BackendConfig`, `LoggingConfig`
- Preset configurations: fast, accurate, research, production, domain-specific
- YAML I/O with validation
- Legacy config compatibility layer

**Phase 3.3: Factory Integration (PR #224)**
- Unified problem factories supporting all MFG types:
  - `create_mfg_problem()` - Main factory for any problem type
  - `create_standard_problem()` - Standard HJB-FP MFG
  - `create_network_problem()` - Network/Graph MFG
  - `create_variational_problem()` - Variational/Lagrangian MFG
  - `create_stochastic_problem()` - Stochastic MFG with common noise
  - `create_highdim_problem()` - High-dimensional MFG (d > 3)
  - `create_lq_problem()` - Linear-Quadratic MFG
  - `create_crowd_problem()` - Crowd dynamics MFG
- Updated `solve_mfg()` interface:
  - New `config` parameter accepting `SolverConfig` instances or preset names
  - Deprecated `method` parameter (still works with warning)
  - Automatic config resolution from strings
- Extended `MFGComponents` for all problem types (network, variational, stochastic, highdim)
- Dual-output factory support: unified MFGProblem (default) or legacy classes (deprecated)
- New examples: `factory_demo.py`, updated `solve_mfg_demo.py`
- Comprehensive documentation:
  - Phase 3.3 design documents (2,000+ lines)
  - Problem type taxonomy
  - Migration guides
  - Completion summary

### Changed

**API Improvements**
- Simplified problem creation with unified factories
- Consistent configuration across all solver types
- Three flexible configuration patterns (YAML, Builder, Presets)
- Clearer separation: problem (math) vs solver (algorithm)

**Code Quality**
- Reduced code duplication through unification
- Better type safety with modern Python typing (`@overload`)
- Improved documentation with comprehensive examples
- Cleaner package structure

### Deprecated

**Problem Classes** (to be removed in v2.0.0)
- `LQMFGProblem` → Use `create_lq_problem()` or `MFGProblem`
- `NetworkMFGProblem` → Use `create_network_problem()` or `MFGProblem`
- `VariationalMFGProblem` → Use `create_variational_problem()` or `MFGProblem`
- `StochasticMFGProblem` → Use `create_stochastic_problem()` or `MFGProblem`

**Config Functions** (to be removed in v2.0.0)
- `create_fast_config()` → Use `presets.fast_solver()`
- `create_accurate_config()` → Use `presets.accurate_solver()`
- `create_research_config()` → Use `presets.research_solver()`
- Old `MFGSolverConfig` → Use new `SolverConfig`

**API Parameters** (to be removed in v2.0.0)
- `solve_mfg(method=...)` → Use `solve_mfg(config=...)`

### Migration Guide

**Old API**:
```python
from mfg_pde.problems import LQMFGProblem
from mfg_pde.config import create_accurate_config
from mfg_pde import solve_mfg

problem = LQMFGProblem(...)
result = solve_mfg(problem, method="accurate")
```

**New API** (Recommended):
```python
from mfg_pde.factory import create_lq_problem
from mfg_pde import solve_mfg

problem = create_lq_problem(...)
result = solve_mfg(problem, config="accurate")
```

### Documentation

- Added comprehensive Phase 3 design documents
- Created migration guides for Phase 3.2 and 3.3
- Updated examples with new unified API
- Added problem type taxonomy
- Created Phase 3 completion summary
- **New User Guides**:
  - `docs/user_guides/particle_interpolation.md` - Complete particle interpolation reference
  - `docs/user_guides/sdf_utilities.md` - Complete SDF utilities reference
  - `docs/migration/PHASE_3_MIGRATION_GUIDE.md` - Phase 3 migration guide
  - `docs/tutorials/01_getting_started.md` - Beginner tutorial
  - `docs/tutorials/02_configuration_patterns.md` - Configuration patterns tutorial

### Technical Details

**Total Changes**:
- ~8,000 lines added/modified
- 21 files changed
- 3 major PRs (#218, #222, #224)
- Full backward compatibility maintained

**Key Benefits**:
- Simpler, more consistent API
- Three flexible configuration patterns
- Better documentation and examples
- Easier to maintain and extend
- Better type safety
- Single source of truth

---

## [0.8.1] - 2025-10-08

### Fixed
- Full nD FP Solver implementation
- Semi-Lagrangian 2D solver
- Bug #8 resolution

---

## Historical Versions

Previous versions (< 0.8.1) were tracked in git history but not formally documented in CHANGELOG.

For detailed historical changes, see:
- Git commit history
- Closed issues and PRs
- Development documentation in `docs/development/`

---

**Note**: Starting with v0.9.0, all changes are documented in this CHANGELOG following semantic versioning and Keep a Changelog standards.
