# Phase 2.1: HJB Solver Trait Integration - Design Document

**Issue**: #596 Phase 2.1
**Date**: 2026-01-17
**Status**: Design Phase

## Objectives

Refactor HJB solvers to use trait-based geometry interfaces, eliminating manual operator implementations and enabling geometry-agnostic algorithm design.

## Current Architecture Analysis

### HJBFDMSolver (nD solver)

**File**: `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py` (~1,100 lines)

**Current Geometry Interaction**:
```python
# Direct attribute access (with try/except)
try:
    return problem.geometry.dimension
except AttributeError:
    pass

# Direct method calls
bounds = self.problem.geometry.get_bounds()
dx = self.problem.geometry.get_grid_spacing()[0]

# Instance checks
if not isinstance(problem.geometry, CartesianGrid):
    raise ValueError("nD FDM requires CartesianGrid")
```

**Current Operator Computation** (`_compute_gradients_nd()` - ~150 lines):
```python
def _compute_gradients_nd(self, U: NDArray, time: float = 0.0) -> dict[int, NDArray]:
    """Compute gradients using manual stencils with BC handling."""
    gradients: dict[int, NDArray] = {-1: U}
    ghost_values = self._get_ghost_values(U, time=time)

    for d in range(self.dimension):
        # Manual computation with ghost cells
        if self.use_upwind:
            # Godunov upwind stencil (50+ lines)
            ...
        else:
            # Central differences (30+ lines)
            ...
        gradients[d] = grad_d

    return gradients
```

**Issues**:
- Manual gradient computation duplicates `tensor_calculus.gradient()` logic
- Ghost value handling duplicated from BC applicator
- ~150 lines of error-prone stencil code
- Tightly coupled to CartesianGrid

### BaseHJBSolver (1D solver)

**File**: `mfg_pde/alg/numerical/hjb_solvers/base_hjb.py` (~900 lines)

**Current Operator Computation**:
```python
def _compute_gradient_array_1d(U_array, Dx, bc=None, upwind=False, time=0.0):
    """Compute gradient using tensor_calculus wrapper."""
    scheme = "upwind" if upwind else "central"
    grads = tensor_gradient(U_array, spacings=[Dx], scheme=scheme, bc=bc, time=time)
    return grads[0]

def _compute_laplacian_1d(U_array, Dx, bc=None, time=0.0):
    """Compute Laplacian using manual ghost cell method (~60 lines)."""
    ghost_left, ghost_right = _compute_laplacian_ghost_values_1d(U_array, bc, Dx, time)
    U_padded = np.concatenate([[ghost_left], U_array, [ghost_right]])
    laplacian = (U_padded[:-2] - 2 * U_padded[1:-1] + U_padded[2:]) / (Dx**2)
    return laplacian
```

**Issues**:
- Gradient already uses `tensor_calculus` (good!)
- Laplacian still manual (~60 lines)
- No trait validation

## Proposed Architecture

### Trait-Based Interface

```python
class HJBFDMSolver(BaseHJBSolver):
    """Finite Difference Method solver for HJB equation.

    Required Geometry Traits:
        - SupportsGradient: For computing ∇U in Hamiltonian H(x, ∇U, m)
        - (SupportsLaplacian): Optional, for viscosity term if not in Hamiltonian

    Compatible Geometries:
        - TensorProductGrid (continuous, structured)
        - ImplicitDomain (continuous, unstructured)
        - Any geometry implementing required traits
    """

    def __init__(self, problem: MFGProblem, ...):
        super().__init__(problem)

        # Validate geometry capabilities
        if not isinstance(problem.geometry, SupportsGradient):
            raise TypeError(
                f"{type(problem.geometry).__name__} doesn't implement SupportsGradient. "
                f"HJB solver requires gradient operator for Hamiltonian evaluation."
            )

        # Get operators via trait interface
        self._gradient_operators = problem.geometry.get_gradient_operator(
            scheme=self.advection_scheme.replace("gradient_", "")  # "upwind" or "centered"
        )

        # Operators automatically inherit BC from geometry (context inheritance)
        # No manual ghost value computation needed!
```

### Simplified Operator Usage

**Before** (~150 lines):
```python
def _compute_gradients_nd(self, U: NDArray, time: float = 0.0) -> dict[int, NDArray]:
    """Manual gradient computation with ghost values."""
    gradients: dict[int, NDArray] = {-1: U}
    ghost_values = self._get_ghost_values(U, time=time)

    for d in range(self.dimension):
        # 50+ lines of manual stencil computation
        if self.use_upwind:
            # Godunov upwind stencil
            ...
        else:
            # Central differences
            ...
        gradients[d] = grad_d

    return gradients
```

**After** (~20 lines):
```python
def _compute_gradients_nd(self, U: NDArray, time: float = 0.0) -> dict[int, NDArray]:
    """Compute gradients using trait-based operators."""
    gradients: dict[int, NDArray] = {-1: U}

    # Get gradient operators from geometry (with BC handling)
    grad_ops = self._gradient_operators

    # Apply operators to get gradients in each direction
    for d in range(self.dimension):
        # Operator call syntax: grad_ops[d](U) or grad_ops[d] @ U.ravel()
        gradients[d] = grad_ops[d](U)

    return gradients
```

**Benefits**:
- 130 lines eliminated
- BC handling automatic (operators inherit from geometry)
- Ghost value computation encapsulated in operator
- Works with any geometry implementing SupportsGradient

### 1D Solver Simplification

**Before**:
```python
def _compute_laplacian_1d(U_array, Dx, bc=None, time=0.0):
    """~60 lines of manual ghost cell computation."""
    ghost_left, ghost_right = _compute_laplacian_ghost_values_1d(U_array, bc, Dx, time)
    U_padded = np.concatenate([[ghost_left], U_array, [ghost_right]])
    laplacian = (U_padded[:-2] - 2 * U_padded[1:-1] + U_padded[2:]) / (Dx**2)
    return laplacian
```

**After**:
```python
def _compute_laplacian_1d(U_array, Dx, bc=None, time=0.0):
    """Use trait-based Laplacian operator."""
    # Get operator from problem geometry (passed as context)
    L = self.problem.geometry.get_laplacian_operator(order=2)
    return L(U_array)
```

## Implementation Plan

### Step 1: Add Trait Validation (hjb_fdm.py)

```python
def __init__(self, problem: MFGProblem, ...):
    super().__init__(problem)

    # Validate geometry traits
    from mfg_pde.geometry.protocols import SupportsGradient

    if not isinstance(problem.geometry, SupportsGradient):
        raise TypeError(
            f"HJB FDM solver requires geometry with SupportsGradient trait. "
            f"{type(problem.geometry).__name__} does not implement this trait."
        )

    # Get operators once during initialization
    scheme = "upwind" if self.use_upwind else "central"
    self._gradient_operators = problem.geometry.get_gradient_operator(scheme=scheme)
```

### Step 2: Refactor _compute_gradients_nd()

Replace manual gradient computation with operator calls:

```python
def _compute_gradients_nd(self, U: NDArray, time: float = 0.0) -> dict[int, NDArray]:
    """Compute gradients using trait-based operators.

    Uses geometry.get_gradient_operator() which automatically handles:
    - Boundary conditions via ghost cells
    - Scheme selection (upwind vs central)
    - Multi-dimensional stencils

    Args:
        U: Value function at current timestep
        time: Current time (for time-dependent BCs)

    Returns:
        Dict mapping dimension index to gradient array
    """
    gradients: dict[int, NDArray] = {-1: U}

    # Apply gradient operators in each direction
    for d in range(self.dimension):
        # Operator handles BC automatically via context inheritance
        gradients[d] = self._gradient_operators[d](U)

    return gradients
```

**Delete**: `_get_ghost_values()` method (~80 lines) - no longer needed!

### Step 3: Refactor 1D Functions (base_hjb.py)

**Option A**: Keep using `tensor_calculus` wrapper (already correct)
**Option B**: Use trait operators for consistency

Recommend **Option A** for 1D gradient (already uses tensor_calculus).
Recommend **Option B** for 1D Laplacian (currently manual).

```python
def _compute_laplacian_1d(U_array, Dx, bc=None, time=0.0):
    """Compute Laplacian using geometry operator.

    Note: This requires geometry context. If called standalone,
    create temporary operator:
        L = LaplacianOperator(spacings=[Dx], field_shape=U_array.shape, bc=bc)
        return L(U_array)
    """
    from mfg_pde.geometry.operators import LaplacianOperator
    L = LaplacianOperator(spacings=[Dx], field_shape=U_array.shape, bc=bc, time=time)
    return L(U_array)
```

### Step 4: Update Docstrings

Add required traits to solver class docstrings:

```python
class HJBFDMSolver(BaseHJBSolver):
    """Finite Difference Method solver for HJB equation.

    Solves backward HJB equation:
        ∂U/∂t + H(x, ∇U, m) = 0
        U(T, x) = φ(x, m(T))

    Required Geometry Traits:
        - SupportsGradient: Provides ∇U operator for Hamiltonian evaluation
        - (SupportsLaplacian): Optional, needed if viscosity not in Hamiltonian

    Compatible Geometries:
        - TensorProductGrid (structured grids)
        - ImplicitDomain (SDF-based domains)
        - Any geometry implementing SupportsGradient

    Example:
        >>> from mfg_pde import MFGProblem
        >>> from mfg_pde.geometry import TensorProductGrid
        >>>
        >>> grid = TensorProductGrid(...)  # Implements SupportsGradient ✓
        >>> problem = MFGProblem(geometry=grid, ...)
        >>> solver = HJBFDMSolver(problem, advection_scheme="gradient_upwind")
        >>> U_solution = solver.solve_hjb_backward(U_init, M_traj)
    """
```

### Step 5: Testing Strategy

**Existing Tests** (verify no regressions):
```bash
# Run all HJB tests
pytest tests/unit/test_hjb*.py -v
pytest tests/integration/test_mfg_solvers.py -v
```

**New Tests** (`tests/unit/test_hjb_trait_integration.py`):
```python
def test_hjb_fdm_validates_gradient_trait():
    """Verify HJBFDMSolver checks for SupportsGradient."""

    # Mock geometry without SupportsGradient
    class MockGeometry:
        dimension = 2

    problem = MFGProblem(geometry=MockGeometry(), ...)

    with pytest.raises(TypeError, match="doesn't implement SupportsGradient"):
        solver = HJBFDMSolver(problem)

def test_hjb_fdm_uses_gradient_operators():
    """Verify HJBFDMSolver uses geometry gradient operators."""

    grid = TensorProductGrid(...)
    problem = MFGProblem(geometry=grid, ...)
    solver = HJBFDMSolver(problem, advection_scheme="gradient_upwind")

    # Verify solver retrieved operators
    assert solver._gradient_operators is not None
    assert len(solver._gradient_operators) == grid.dimension

    # Verify operators are correct type
    from mfg_pde.geometry.operators import GradientComponentOperator
    assert all(isinstance(op, GradientComponentOperator) for op in solver._gradient_operators)
```

## Benefits

### Code Reduction
- **HJBFDMSolver**: ~150 lines → ~20 lines (130 lines eliminated)
- **base_hjb**: ~60 lines → ~5 lines (55 lines eliminated)
- **Total**: ~185 lines eliminated from solver code

### Maintainability
- Gradient/Laplacian logic centralized in operator classes
- BC handling automatic via context inheritance
- Easier to test (operators tested independently)

### Extensibility
- Solvers work with ANY geometry implementing traits
- Easy to add new geometry types (ImplicitDomain, graphs, meshfree)
- Operators can be optimized/GPU-accelerated without changing solver code

### Error Messages
```python
# Before (cryptic)
AttributeError: 'MockGeometry' object has no attribute 'get_grid_spacing'

# After (clear)
TypeError: MockGeometry doesn't implement SupportsGradient.
HJB solver requires gradient operator for Hamiltonian evaluation.
```

## Risks and Mitigations

### Risk 1: Performance Regression
**Concern**: Operator call overhead vs manual computation
**Mitigation**:
- Operators use same underlying `tensor_calculus` functions
- Benchmark before/after
- Profile to identify any hotspots

### Risk 2: Breaking Changes
**Concern**: Existing code relies on manual gradient computation
**Mitigation**:
- Keep existing tests passing
- Gradual rollout (HJB first, then FP, then coupling)
- Deprecation warnings if needed

### Risk 3: BC Handling Edge Cases
**Concern**: Operators might handle BCs differently
**Mitigation**:
- Operators already tested with BC handling (Phase 1.2)
- Verify results match for test problems
- Document any behavioral changes

## Timeline

- **Day 1**: Step 1 (Trait validation) + Step 4 (Docstrings)
- **Day 2**: Step 2 (Refactor nD gradients)
- **Day 3**: Step 3 (Refactor 1D Laplacian)
- **Day 4**: Step 5 (Testing and validation)
- **Day 5**: Code review, documentation, PR

## Success Criteria

✅ HJBFDMSolver validates SupportsGradient in __init__()
✅ _compute_gradients_nd() uses trait operators (~20 lines)
✅ _get_ghost_values() method deleted
✅ _compute_laplacian_1d() uses LaplacianOperator
✅ All existing HJB tests pass
✅ New trait integration tests added
✅ Docstrings document required traits
✅ Code review approved

## Next Steps (Phase 2.2)

After HJB integration complete:
- Refactor FP solvers (SupportsDivergence, SupportsAdvection)
- Refactor Picard/Newton coupling (uses both HJB and FP)
- Graph-based solvers (SupportsGraphLaplacian)
