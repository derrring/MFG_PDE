# Phase 2.3: Comprehensive Geometry Module Test Coverage

**Date**: 2025-10-13
**Status**: 📋 PLANNING
**Priority**: HIGH
**Estimated Duration**: 2-3 sessions

## Overview

Phase 2.3 focuses on comprehensive test coverage for the `mfg_pde/geometry/` module. Following the successful completion of Phase 2.2a (solver unit tests), this phase will ensure robust validation of all geometry components including domain definitions, boundary conditions, grid generation, and network structures.

## Current Status

### Existing Tests
Located in `tests/unit/test_geometry/`:
- `test_domain_1d.py` - 1D domain tests
- `test_boundary_conditions_1d.py` - 1D boundary condition tests
- `test_simple_grid.py` - Simple grid tests

**Coverage Gap**: 3 test files vs 19 geometry module files = **~16% coverage**

### Geometry Module Structure

```
mfg_pde/geometry/
├── __init__.py                    # Module exports
├── base_geometry.py               # Base geometry classes
├── domain_1d.py                   # 1D domain (tested)
├── domain_2d.py                   # 2D domain (no tests)
├── domain_3d.py                   # 3D domain (no tests)
├── boundary_conditions_1d.py      # 1D BCs (tested)
├── boundary_conditions_2d.py      # 2D BCs (no tests)
├── boundary_conditions_3d.py      # 3D BCs (no tests)
├── boundary_manager.py            # BC manager (no tests)
├── simple_grid.py                 # Simple grid (tested)
├── tensor_product_grid.py         # Tensor grids (no tests)
├── mesh_manager.py                # Mesh management (no tests)
├── network_geometry.py            # Network structures (no tests)
├── network_backend.py             # Network backend (no tests)
├── amr_1d.py                      # 1D AMR (no tests)
├── amr_triangular_2d.py           # 2D triangular AMR (no tests)
├── amr_quadtree_2d.py             # 2D quadtree AMR (no tests)
└── amr_tetrahedral_3d.py          # 3D tetrahedral AMR (no tests)
```

## Test Plan Structure

### Phase 2.3a: Core Geometry Components (Priority: CRITICAL)

Focus on fundamental geometry classes used by all solvers.

#### 1. Domain Tests
**Files to test**: `domain_1d.py` (existing), `domain_2d.py`, `domain_3d.py`

**Test file**: `tests/unit/test_geometry/test_domain_2d.py` (NEW)
- Initialization with rectangular/square domains
- Grid point generation
- Spatial discretization (dx, dy)
- Coordinate transformations
- Integration with sparse operators

**Test file**: `tests/unit/test_geometry/test_domain_3d.py` (NEW)
- Initialization with box domains
- 3D grid point generation
- Spatial discretization (dx, dy, dz)
- Memory efficiency validation
- Tensor product structure verification

**Estimated tests**: 25-30 per file (~50-60 total)

#### 2. Boundary Condition Tests
**Files to test**: `boundary_conditions_1d.py` (existing), `boundary_conditions_2d.py`, `boundary_conditions_3d.py`

**Test file**: `tests/unit/test_geometry/test_boundary_conditions_2d.py` (NEW)
- Dirichlet, Neumann, periodic, mixed BCs
- BC enforcement on domain boundaries
- Corner and edge handling
- BC matrix construction
- Mass conservation with no-flux conditions

**Test file**: `tests/unit/test_geometry/test_boundary_conditions_3d.py` (NEW)
- 3D BC types (Dirichlet, Neumann, periodic)
- Face, edge, and corner handling
- BC sparsity pattern validation
- Integration with 3D solvers

**Estimated tests**: 30-35 per file (~60-70 total)

#### 3. Boundary Manager Tests
**Files to test**: `boundary_manager.py`

**Test file**: `tests/unit/test_geometry/test_boundary_manager.py` (NEW)
- BC registration and lookup
- Multiple BC type management
- BC validation and constraints
- BC application to solution arrays

**Estimated tests**: 15-20 tests

#### 4. Grid Structure Tests
**Files to test**: `simple_grid.py` (existing), `tensor_product_grid.py`

**Test file**: `tests/unit/test_geometry/test_tensor_product_grid.py` (NEW)
- 2D tensor product construction
- 3D tensor product construction
- Memory efficiency (O(Nx+Ny+Nz) vs O(Nx*Ny*Nz))
- Sparse operator construction
- Gradient and Laplacian operators

**Estimated tests**: 25-30 tests

#### 5. Mesh Manager Tests
**Files to test**: `mesh_manager.py`

**Test file**: `tests/unit/test_geometry/test_mesh_manager.py` (NEW)
- Mesh creation and refinement
- Mesh quality metrics
- Neighbor queries
- Integration with solvers

**Estimated tests**: 20-25 tests

### Phase 2.3b: Network Geometry (Priority: HIGH)

#### 6. Network Geometry Tests
**Files to test**: `network_geometry.py`, `network_backend.py`

**Test file**: `tests/unit/test_geometry/test_network_geometry.py` (NEW)
- GridNetwork creation (tested in integration)
- Custom network graphs
- Network properties (adjacency, Laplacian)
- Node and edge operations
- Network traversal algorithms

**Test file**: `tests/unit/test_geometry/test_network_backend.py` (NEW)
- igraph backend integration
- networkx backend integration
- Backend switching and compatibility
- Graph property computation

**Estimated tests**: 30-35 per file (~60-70 total)

### Phase 2.3c: Adaptive Mesh Refinement (Priority: MEDIUM)

#### 7. AMR Tests
**Files to test**: `amr_1d.py`, `amr_triangular_2d.py`, `amr_quadtree_2d.py`, `amr_tetrahedral_3d.py`

**Test file**: `tests/unit/test_geometry/test_amr_1d.py` (NEW)
- Grid refinement and coarsening
- Error indicator computation
- Grid hierarchy management
- Solution interpolation/restriction

**Test file**: `tests/unit/test_geometry/test_amr_2d.py` (NEW)
- Triangular and quadtree refinement
- 2D error indicators
- Neighbor finding on refined grids
- Load balancing considerations

**Test file**: `tests/unit/test_geometry/test_amr_3d.py` (NEW)
- Tetrahedral mesh refinement
- 3D error estimation
- Memory efficiency validation

**Estimated tests**: 20-25 per AMR dimension (~60-75 total)

## Testing Methodology

### Test Categories

1. **Initialization Tests**
   - Constructor validation
   - Parameter constraints
   - Default values
   - Invalid input handling

2. **Geometric Property Tests**
   - Grid point generation
   - Spacing computations
   - Volume/area calculations
   - Coordinate transformations

3. **Boundary Condition Tests**
   - BC enforcement
   - Mass conservation
   - Compatibility with solvers
   - Edge/corner cases

4. **Operator Construction Tests**
   - Gradient operators
   - Laplacian operators
   - Divergence operators
   - Sparsity patterns

5. **Integration Tests**
   - Solver compatibility
   - Multi-dimensional consistency
   - Performance on realistic problems

### Key Testing Principles

1. **Mathematical Correctness**
   - Verify discretization accuracy
   - Check operator consistency
   - Validate conservation properties

2. **Memory Efficiency**
   - Validate tensor product memory usage
   - Check sparse matrix formats
   - Monitor memory scaling with grid size

3. **Numerical Properties**
   - Test stability of operators
   - Verify symmetry where applicable
   - Check positive definiteness of Laplacians

4. **Edge Cases**
   - Minimum grid sizes (single element)
   - Maximum practical grid sizes
   - Degenerate geometries
   - Mixed boundary conditions

## Implementation Schedule

### Week 1: Phase 2.3a - Core Geometry (5-6 test files)
- Day 1-2: Domain 2D/3D tests (~100-120 tests)
- Day 3: Boundary conditions 2D tests (~30-35 tests)
- Day 4: Boundary conditions 3D tests (~30-35 tests)
- Day 5: Boundary manager + tensor product grid tests (~40-55 tests)
- Day 6: Mesh manager tests (~20-25 tests)

**Total Week 1**: ~220-270 tests across 5-6 files

### Week 2: Phase 2.3b - Network Geometry (2 test files)
- Day 1-2: Network geometry tests (~30-35 tests)
- Day 3-4: Network backend tests (~30-35 tests)

**Total Week 2**: ~60-70 tests across 2 files

### Week 3: Phase 2.3c - AMR (3 test files, if time permits)
- Day 1-2: AMR 1D tests (~20-25 tests)
- Day 3-4: AMR 2D tests (~20-25 tests)
- Day 5-6: AMR 3D tests (~20-25 tests)

**Total Week 3**: ~60-75 tests across 3 files

## Success Criteria

### Minimum Requirements (Phase 2.3a)
- ✅ All core domain tests (1D, 2D, 3D) complete
- ✅ All boundary condition tests (1D, 2D, 3D) complete
- ✅ Boundary manager tests complete
- ✅ Tensor product grid tests complete
- ✅ All tests passing
- ✅ Documentation of key findings

### Stretch Goals (Phase 2.3b+c)
- ✅ Network geometry tests complete
- ✅ AMR tests (at least 1D and 2D) complete
- ✅ Performance benchmarks for grid operations

## Expected Challenges

1. **2D/3D Visualization**
   - Challenge: Validating 2D/3D geometric properties without visual inspection
   - Solution: Use analytical test cases with known solutions

2. **Sparse Matrix Operations**
   - Challenge: Testing sparse operator correctness
   - Solution: Compare against dense implementations for small grids

3. **AMR Complexity**
   - Challenge: AMR systems are complex with many edge cases
   - Solution: Start with simple refinement patterns, gradually increase complexity

4. **Network Backend Dependencies**
   - Challenge: igraph/networkx optional dependencies
   - Solution: Use pytest.importorskip and skip tests if not available

## Dependencies

### Required Packages
- numpy (always available)
- scipy (always available)
- pytest (testing)

### Optional Packages
- igraph (network tests)
- networkx (network tests)
- matplotlib (visualization validation - optional)

## Documentation

### Files to Create/Update
1. Test files (10-14 new files)
2. Phase 2.3 completion summary (end of phase)
3. Geometry module coverage report
4. Update STRATEGIC_DEVELOPMENT_ROADMAP_2026.md

## Integration with CI/CD

All new tests will:
- Run automatically on every commit
- Be included in coverage reports
- Follow existing test patterns from Phase 2.2a
- Use appropriate pytest markers (@pytest.mark.slow for expensive tests)

## Relation to Previous Phases

### Phase 2.1: Integration Tests
- Created high-level integration tests for full MFG solving
- Tested solver interactions

### Phase 2.2a: Solver Unit Tests ✅ COMPLETED
- Comprehensive unit tests for HJB and FP solvers
- 120+ tests across 5 files
- Established testing patterns

### Phase 2.3: Geometry Tests ⬅️ CURRENT PHASE
- Comprehensive unit tests for geometry module
- ~350-400 tests across 10-14 files
- Foundation for remaining module tests

### Future Phases
- **Phase 2.4**: Configuration system tests (`mfg_pde/config/`)
- **Phase 2.5**: Utility module tests (`mfg_pde/utils/`)
- **Phase 2.6**: Factory method tests (`mfg_pde/factory/`)

## Notes

- **Incremental approach**: Start with Phase 2.3a (critical components)
- **Validate as we go**: Run tests after each file creation
- **Document discoveries**: Note any geometry module issues or inconsistencies
- **Follow Phase 2.2a patterns**: Use established test structure and quality standards

---

**Next Step**: Begin Phase 2.3a with domain 2D tests
**Branch**: `test/phase2-3-geometry-coverage`
**Target**: ~220-270 tests in first week (core geometry components)
