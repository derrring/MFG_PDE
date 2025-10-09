# Phase 2.3: Geometry Module Test Plan

**Date**: 2025-10-09
**Phase**: 2.3 (Geometry Tests)
**Target**: 52% → 75% coverage (+483 lines)
**Status**: Planning

## Executive Summary

Phase 2.3 focuses on comprehensive test coverage for geometry modules, which are core to MFG solver correctness. Based on analysis of existing code, this phase will target boundary conditions, domain definitions, and grid management.

## Module Analysis

### Priority Modules (by criticality and size)

#### High Priority - Core Boundary Conditions
1. **boundary_conditions_1d.py** (185 lines)
   - Dataclass with validation
   - 5 BC types: periodic, dirichlet, neumann, no_flux, robin
   - Type checking methods
   - Matrix size computation
   - Factory functions
   - **Estimated**: 40-50 tests

2. **boundary_conditions_2d.py** (655 lines)
   - Similar structure to 1D
   - Additional complexity for 2D boundaries
   - **Estimated**: 50-60 tests

3. **boundary_conditions_3d.py** (656 lines)
   - 3D boundary management
   - **Estimated**: 50-60 tests

#### High Priority - Domain Definitions
4. **domain_1d.py** (69 lines)
   - Simple 1D domain setup
   - **Estimated**: 15-20 tests

5. **domain_2d.py** (436 lines)
   - 2D rectangular and custom domains
   - **Estimated**: 40-50 tests

6. **domain_3d.py** (561 lines)
   - 3D domain management
   - **Estimated**: 50-60 tests

#### Medium Priority - Grid Management
7. **base_geometry.py** (308 lines)
   - Abstract base class
   - **Estimated**: 30-40 tests

8. **simple_grid.py** (322 lines)
   - Basic grid implementation
   - **Estimated**: 35-45 tests

9. **tensor_product_grid.py** (329 lines)
   - Multi-dimensional grid
   - **Estimated**: 35-45 tests

#### Medium Priority - Boundary Management
10. **boundary_manager.py** (342 lines)
    - Boundary condition application
    - **Estimated**: 35-45 tests

11. **mesh_manager.py** (360 lines)
    - Mesh refinement and management
    - **Estimated**: 40-50 tests

#### Lower Priority - Specialized
12. **network_geometry.py** (716 lines)
    - Network/graph geometry
    - **Estimated**: 60-70 tests

13. **network_backend.py** (591 lines)
    - Network computation backend
    - **Estimated**: 50-60 tests

14. **AMR modules** (4 files, ~2,208 lines total)
    - amr_1d.py (530 lines)
    - amr_quadtree_2d.py (607 lines)
    - amr_triangular_2d.py (580 lines)
    - amr_tetrahedral_3d.py (491 lines)
    - **Estimated**: 150-200 tests total
    - **Note**: High complexity, may defer to separate phase

## Phase 2.3 Strategy

### Approach A: Core Boundary Conditions + Domains (Recommended)

**Phase 2.3a: 1D Boundary Conditions and Domain**
- boundary_conditions_1d.py (40-50 tests)
- domain_1d.py (15-20 tests)
- **Total**: ~55-70 tests, ~4-6 hours

**Phase 2.3b: 2D Boundary Conditions and Domain**
- boundary_conditions_2d.py (50-60 tests)
- domain_2d.py (40-50 tests)
- **Total**: ~90-110 tests, ~8-10 hours

**Phase 2.3c: 3D Boundary Conditions and Domain (Optional)**
- boundary_conditions_3d.py (50-60 tests)
- domain_3d.py (50-60 tests)
- **Total**: ~100-120 tests, ~10-12 hours

**Total Estimated**: 245-300 tests, 22-28 hours

### Approach B: Grid Management Focus

**Phase 2.3a: Base Geometry and Grids**
- base_geometry.py (30-40 tests)
- simple_grid.py (35-45 tests)
- tensor_product_grid.py (35-45 tests)
- **Total**: ~100-130 tests, ~10-12 hours

**Phase 2.3b: Boundary and Mesh Management**
- boundary_manager.py (35-45 tests)
- mesh_manager.py (40-50 tests)
- **Total**: ~75-95 tests, ~8-10 hours

**Total Estimated**: 175-225 tests, 18-22 hours

### Recommended Approach: **Approach A (Core BC + Domains)**

**Rationale**:
- Boundary conditions are fundamental to solver correctness
- Simpler dataclass structure (similar to solver_config in Phase 2.2a)
- Clear test patterns from Phase 2.2a can be reused
- Builds foundation for integration tests
- 1D → 2D → 3D progression is logical and manageable

## Test Patterns (From Phase 2.2a)

### Dataclass Configuration Testing
- ✅ Default values
- ✅ Validation in `__post_init__`
- ✅ Parameter range checks
- ✅ Type checking methods
- ✅ Factory functions
- ✅ String representation
- ✅ Edge cases

### Boundary Condition Specific Tests
- **Type Checking**: is_periodic(), is_dirichlet(), is_neumann(), is_no_flux(), is_robin()
- **Matrix Sizing**: get_matrix_size() for different BC types
- **Value Validation**: validate_values() for each BC type
- **Factory Functions**: periodic_bc(), dirichlet_bc(), neumann_bc(), no_flux_bc(), robin_bc()
- **Edge Cases**:
  - Missing required parameters
  - Invalid BC type strings
  - Boundary value consistency
  - Robin coefficient validation

## Detailed Test Plan: Phase 2.3a (1D BC + Domain)

### File 1: `tests/unit/test_geometry/test_boundary_conditions_1d.py`

**Estimated**: 680-750 lines, 45-50 tests

#### Test Categories:

**1. BoundaryConditions Class - Initialization (8 tests)**
- test_periodic_initialization
- test_dirichlet_initialization
- test_neumann_initialization
- test_no_flux_initialization
- test_robin_initialization
- test_missing_robin_coefficients_raises
- test_invalid_type_string
- test_default_values

**2. Type Checking Methods (5 tests)**
- test_is_periodic
- test_is_dirichlet
- test_is_neumann
- test_is_no_flux
- test_is_robin

**3. Matrix Size Computation (6 tests)**
- test_get_matrix_size_periodic
- test_get_matrix_size_dirichlet
- test_get_matrix_size_neumann
- test_get_matrix_size_no_flux
- test_get_matrix_size_robin
- test_get_matrix_size_invalid_type_raises

**4. Value Validation (8 tests)**
- test_validate_values_periodic_no_requirements
- test_validate_values_no_flux_no_requirements
- test_validate_values_dirichlet_missing_left_raises
- test_validate_values_dirichlet_missing_right_raises
- test_validate_values_neumann_missing_left_raises
- test_validate_values_neumann_missing_right_raises
- test_validate_values_robin_missing_params_raises
- test_validate_values_robin_complete

**5. String Representation (6 tests)**
- test_str_periodic
- test_str_dirichlet
- test_str_neumann
- test_str_no_flux
- test_str_robin
- test_str_unknown_type

**6. Factory Functions (10 tests)**
- test_factory_periodic_bc
- test_factory_dirichlet_bc
- test_factory_neumann_bc
- test_factory_no_flux_bc
- test_factory_robin_bc
- test_factory_dirichlet_values_stored
- test_factory_neumann_values_stored
- test_factory_robin_all_params_stored
- test_factory_functions_return_correct_type
- test_factory_validation_on_creation

**7. Edge Cases and Integration (5 tests)**
- test_multiple_bc_instances_independent
- test_bc_equality_comparison
- test_bc_serialization_roundtrip
- test_bc_immutability_after_creation
- test_bc_large_matrix_sizes

### File 2: `tests/unit/test_geometry/test_domain_1d.py`

**Estimated**: 350-400 lines, 18-22 tests

#### Test Categories:

**1. Domain Initialization (4 tests)**
- test_domain_default_initialization
- test_domain_custom_bounds
- test_domain_invalid_bounds_raises (xmax <= xmin)
- test_domain_boundary_condition_integration

**2. Domain Properties (4 tests)**
- test_domain_length_computation
- test_domain_grid_points
- test_domain_dx_spacing
- test_domain_boundary_access

**3. Domain Methods (6 tests)**
- test_domain_contains_point
- test_domain_boundary_distance
- test_domain_grid_generation
- test_domain_refinement
- test_domain_coarsening
- test_domain_subdivision

**4. Integration with BC (4 tests)**
- test_domain_with_periodic_bc
- test_domain_with_dirichlet_bc
- test_domain_with_neumann_bc
- test_domain_with_no_flux_bc

**5. Edge Cases (4 tests)**
- test_domain_very_small_length
- test_domain_very_large_length
- test_domain_many_grid_points
- test_domain_few_grid_points

## Implementation Timeline

### Session 1 (4-6 hours): Phase 2.3a
- Create test_boundary_conditions_1d.py (45-50 tests)
- Create test_domain_1d.py (18-22 tests)
- Documentation: PHASE2.3A summary
- **Deliverable**: ~65-72 tests, 1D geometry coverage

### Session 2 (8-10 hours): Phase 2.3b
- Create test_boundary_conditions_2d.py (50-60 tests)
- Create test_domain_2d.py (40-50 tests)
- Documentation: PHASE2.3B summary
- **Deliverable**: ~90-110 tests, 2D geometry coverage

### Session 3 (Optional, 10-12 hours): Phase 2.3c
- Create test_boundary_conditions_3d.py (50-60 tests)
- Create test_domain_3d.py (50-60 tests)
- Documentation: PHASE2.3C summary
- **Deliverable**: ~100-120 tests, 3D geometry coverage

## Success Metrics

### Coverage Goals
- boundary_conditions_1d.py: 20% → 90% (+70%)
- boundary_conditions_2d.py: 15% → 85% (+70%)
- boundary_conditions_3d.py: 10% → 80% (+70%)
- domain_1d.py: 25% → 95% (+70%)
- domain_2d.py: 20% → 85% (+65%)
- domain_3d.py: 15% → 80% (+65%)

### Test Quality
- All tests passing
- Zero flaky tests
- Comprehensive edge case coverage
- Clear docstrings
- Proper fixtures

### Documentation
- Phase 2.3a/b/c summaries
- Updated session summary
- Test patterns documented

## Risks and Mitigation

**Risk 1**: Complex geometry dependencies
- **Mitigation**: Start with 1D (simplest), build up incrementally

**Risk 2**: Integration with solvers
- **Mitigation**: Focus on unit tests first, defer integration to Phase 3

**Risk 3**: AMR complexity
- **Mitigation**: Defer AMR to separate phase if needed

**Risk 4**: Time estimates may be optimistic
- **Mitigation**: Complete Phase 2.3a first, reassess timeline

## Next Steps

1. Create `tests/unit/test_geometry/` directory
2. Start with `test_boundary_conditions_1d.py`
3. Follow established patterns from Phase 2.2a
4. Run tests incrementally to catch issues early
5. Document as we go

---

**Status**: Planning Complete
**Next Action**: Begin Phase 2.3a implementation
**Estimated Total**: 245-300 tests over 22-28 hours
**Target Coverage**: Geometry package 52% → 75%
