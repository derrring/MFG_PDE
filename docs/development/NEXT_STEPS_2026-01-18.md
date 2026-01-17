# Next Steps - 2026-01-18 (Updated)

**Current Status**: ✅ Issue #573 complete, starting #590 (Geometry Trait System)

## Recently Completed (2026-01-18)

### Issue #573: Non-Quadratic Hamiltonian Support ✅

**Commits**:
- `f5cb1039` - Documentation clarification + test suite (8/8 passing)
- `1c13a450` - L1 control demonstration example

**Key Achievement**: Clarified that `drift_field` parameter already supports ANY Hamiltonian - no API changes needed!

**Deliverables**:
- Updated FP FDM/GFDM docstrings with L1, quartic examples
- Created `test_fp_nonquadratic.py` (8 tests, all passing)
- Created `examples/advanced/mfg_l1_control.py` (comprehensive comparison)

---

## Current Work: Issue #590 (Geometry Trait System)

**Issue**: [#590](https://github.com/derrring/MFG_PDE/issues/590) - Phase 1: Geometry Trait System & Region Registry
**Part of**: #589 (Geometry & BC Architecture Master Tracking)
**Priority**: HIGH
**Size**: Medium (1-2 weeks)
**Status**: 🎯 **STARTING NOW**

### Objective

Formalize trait protocols for geometry capabilities, enabling:
1. **Solver-geometry compatibility validation**
2. **Geometry-agnostic algorithm design**
3. **Clear capability requirements** in solver APIs
4. **Better error messages** when geometries lack required features

### Implementation Plan

#### Phase 1.1: Protocol Definition (3-5 days)

**Files to Create**:
- `mfg_pde/geometry/protocols/__init__.py`
- `mfg_pde/geometry/protocols/operators.py` - Operator trait protocols
- `mfg_pde/geometry/protocols/topology.py` - Topological trait protocols
- `mfg_pde/geometry/protocols/regions.py` - Region marking protocols

**Key Protocols**:
```python
@runtime_checkable
class SupportsLaplacian(Protocol):
    """Geometry provides Laplacian operator."""
    def get_laplacian_operator(
        self,
        order: int = 2,
        boundary_conditions: BoundaryConditions | None = None
    ) -> LinearOperator: ...

@runtime_checkable
class SupportsGradient(Protocol):
    """Geometry provides gradient operator."""
    def get_gradient_operator(
        self,
        direction: int | None = None,
        order: int = 2
    ) -> LinearOperator: ...

@runtime_checkable
class SupportsDivergence(Protocol):
    """Geometry provides divergence operator."""
    def get_divergence_operator(
        self,
        order: int = 2
    ) -> LinearOperator: ...

@runtime_checkable
class SupportsAdvection(Protocol):
    """Geometry provides advection operator."""
    def get_advection_operator(
        self,
        velocity_field: np.ndarray | Callable,
        scheme: str = 'upwind'
    ) -> LinearOperator: ...
```

**Testing Strategy**:
- Protocol compliance checks for existing geometries
- Operator composition tests (Laplacian = div(grad))
- Integration with LinearOperator infrastructure (#595 ✅)

---

#### Phase 1.2: Retrofit TensorProductGrid (3-4 days)

**Goal**: Make TensorProductGrid advertise its capabilities via traits

**Current State**: TensorProductGrid already has operators (#595 complete)
- ✅ `LaplacianOperator` implemented
- ✅ `GradientOperator` implemented
- ✅ `DivergenceOperator` implemented
- ✅ `AdvectionOperator` implemented

**Needed**: Wrapper methods for protocol compliance

**Implementation**:
```python
class TensorProductGrid(BaseGeometry):
    """
    Tensor product grid with full operator support.

    Implements:
        - SupportsLaplacian
        - SupportsGradient
        - SupportsDivergence
        - SupportsAdvection
    """

    def get_laplacian_operator(
        self,
        order: int = 2,
        boundary_conditions: BoundaryConditions | None = None
    ) -> LinearOperator:
        """Get Laplacian operator (Protocol: SupportsLaplacian)."""
        from mfg_pde.geometry.operators import LaplacianOperator
        return LaplacianOperator(
            self,
            order=order,
            boundary_conditions=boundary_conditions or self.boundary_conditions
        )

    def get_gradient_operator(
        self,
        direction: int | None = None,
        order: int = 2
    ) -> LinearOperator:
        """Get gradient operator (Protocol: SupportsGradient)."""
        from mfg_pde.geometry.operators import GradientOperator
        return GradientOperator(self, direction=direction, order=order)

    # ... similar for divergence, advection
```

**Validation**:
```python
# Runtime protocol checks
assert isinstance(geometry, SupportsLaplacian)
assert isinstance(geometry, SupportsGradient)

# Operator functionality
L = geometry.get_laplacian_operator(order=2)
assert isinstance(L, scipy.sparse.linalg.LinearOperator)
```

---

#### Phase 1.3: Region Registry System (2-3 days)

**Goal**: Enable named boundary/subdomain marking

**Use Cases**:
- "inflow", "outflow", "wall" boundary regions
- "protected_zone", "free_region" subdomains
- Localized constraint enforcement

**Implementation**:
```python
class RegionRegistry:
    """
    Named region registry for boundaries and subdomains.

    Example:
        >>> registry = RegionRegistry()
        >>> registry.register_boundary("inflow", lambda x: x[0] < 0.1)
        >>> registry.register_subdomain("protected", lambda x: 0.3 < x[0] < 0.7)
        >>> inflow_nodes = registry.get_boundary_nodes("inflow")
    """

    def register_boundary(
        self,
        name: str,
        condition: Callable[[np.ndarray], bool]
    ): ...

    def register_subdomain(
        self,
        name: str,
        condition: Callable[[np.ndarray], bool]
    ): ...

    def get_boundary_nodes(self, name: str) -> np.ndarray: ...
    def get_subdomain_nodes(self, name: str) -> np.ndarray: ...
```

**Integration with Constraints** (#591 ✅):
```python
# Regional obstacle constraint
psi = ...  # Obstacle function
protected_region = registry.get_subdomain_nodes("protected")
constraint = ObstacleConstraint(psi, region=protected_region)
```

---

### Success Criteria

**Phase 1.1**:
- ✅ 4 trait protocols defined (Laplacian, Gradient, Divergence, Advection)
- ✅ Protocols use `@runtime_checkable` decorator
- ✅ Documentation with examples

**Phase 1.2**:
- ✅ TensorProductGrid implements all 4 protocols
- ✅ Protocol compliance verified with `isinstance()`
- ✅ All operators return `LinearOperator` instances
- ✅ Backward compatibility preserved (existing code works)

**Phase 1.3**:
- ✅ RegionRegistry class implemented
- ✅ Boundary and subdomain registration working
- ✅ Integration with constraint system validated

---

## After #590 Completion

### Next: Issue #596 (Solver Integration with Traits)

**Dependencies**: #590 complete ✅ (will be)
**Objective**: Refactor solvers to use trait-based geometry interfaces
**Impact**: Solvers become geometry-agnostic, clear capability requirements

**Refactoring Pattern**:
```python
# Before: hasattr duck typing
if hasattr(geometry, 'get_laplacian'):
    L = geometry.get_laplacian()

# After: Protocol-based validation
if not isinstance(geometry, SupportsLaplacian):
    raise TypeError(f"{type(geometry)} doesn't support Laplacian operator")

L = geometry.get_laplacian_operator(order=2)
```

**Solvers to Update**:
- HJB FDM, HJB GFDM, HJB Semi-Lagrangian
- FP FDM, FP GFDM
- Coupling solvers

---

## Progress Summary

**Completed Today (2026-01-18)**:
- ✅ Issue #573 - Non-quadratic Hamiltonian support
- ✅ Documentation updates (PRIORITY_LIST, NEXT_STEPS)

**Starting Now**:
- 🎯 Issue #590 - Geometry Trait System

**Completed Infrastructure** (Priorities 1-8):
- ✅ P1: FDM BC Bug Fix (#542)
- ✅ P2: Silent Fallbacks (#547)
- ✅ P3: hasattr Elimination (#543 all phases)
- ✅ P3.5: Adjoint Pairing (#580)
- ✅ P3.6: Ghost Nodes (#576)
- ✅ P4: Mixin Refactoring (#545)
- ✅ P5.5: Progress Bar Protocol (#587)
- ✅ P6.5: Adjoint BC (#574)
- ✅ P6.6: LinearOperator Architecture (#595)
- ✅ P6.7: Variational Inequalities (#591)
- ✅ P7: Solver Cleanup (#545)
- ✅ P8: Legacy Deprecation (#544 Phases 1-2)
- ✅ #573: Non-Quadratic H Support

---

**Last Updated**: 2026-01-18 (afternoon)
**Current Focus**: 🎯 Starting #590 Phase 1.1 (Protocol Definition)
**Next Milestone**: Complete #590 → Enable #596 (Solver Integration)
