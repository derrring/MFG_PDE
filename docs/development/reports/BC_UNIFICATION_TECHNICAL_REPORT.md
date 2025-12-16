# Technical Report: Boundary Condition Handling Unification

**Issue**: #486
**Date**: 2025-12-16
**Revision**: 1.2 (Performance & Scalability Review)
**Status**: Proposed
**Author**: MFG_PDE Development Team
**Reviewer**: External Technical Expert

---

## Executive Summary

This report documents the current state of boundary condition (BC) handling across the MFG_PDE solver ecosystem, identifies architectural inconsistencies, and proposes a unified approach. The goal is to enable all solvers to leverage the existing BC infrastructure in `geometry/boundary/` rather than using ad-hoc implementations.

---

## 1. Problem Statement

### 1.1 Background

Mean Field Game (MFG) problems require solving coupled Hamilton-Jacobi-Bellman (HJB) and Fokker-Planck (FP) equations on bounded domains. Proper boundary condition handling is essential for:

- **Physical correctness**: Reflecting domain constraints (walls, exits, periodic domains)
- **Numerical stability**: Preventing spurious oscillations at boundaries
- **Mass conservation**: Ensuring FP density integrates to 1 (for no-flux/periodic BCs)

### 1.2 Current Issues

The solver ecosystem has grown organically, resulting in:

1. **Inconsistent BC retrieval**: Some solvers use `problem.boundary_conditions`, others use `problem.get_boundary_conditions()`, and some ignore BCs entirely
2. **Hardcoded boundary stencils**: WENO and FDM solvers use one-sided finite differences regardless of BC type
3. **Limited BC type support**: FP solvers only support uniform BCs; mixed BCs cause runtime errors
4. **Duplicated infrastructure**: Each solver implements its own boundary handling instead of using centralized applicators

---

## 2. Current Architecture Analysis

### 2.1 BC Specification Layer

The BC specification infrastructure is well-designed and complete:

```
mfg_pde/geometry/boundary/
├── conditions.py      # BoundaryConditions class (uniform + mixed)
├── types.py           # BCType enum, BCSegment dataclass
├── applicator_fdm.py  # Ghost cell computation for FDM
├── applicator_meshfree.py  # BC handling for collocation methods
└── fdm_bc_1d.py       # Legacy 1D BC (deprecated)
```

**Key Classes**:

```python
@dataclass
class BoundaryConditions:
    dimension: int
    segments: list[BCSegment]      # Ordered by priority
    default_bc: BCType             # Fallback BC type
    default_value: float
    domain_bounds: np.ndarray | None
    domain_sdf: Callable | None    # For Lipschitz domains
    corner_strategy: Literal["priority", "average", "mollify"]
```

**Supported BC Types**:
- `BCType.DIRICHLET` - Prescribed value: $u = g$
- `BCType.NEUMANN` - Prescribed flux: $\partial u / \partial n = g$
- `BCType.ROBIN` - Mixed: $\alpha u + \beta \partial u / \partial n = g$
- `BCType.PERIODIC` - Wrap-around: $u(x_{min}) = u(x_{max})$
- `BCType.NO_FLUX` - Zero normal flux (alias for homogeneous Neumann)

**Mixed BC Support**:
```python
# Example: Exit at top, walls elsewhere
exit_seg = BCSegment(name="exit", bc_type=BCType.DIRICHLET,
                     value=0.0, boundary="y_max", priority=1)
wall_seg = BCSegment(name="walls", bc_type=BCType.NEUMANN, value=0.0)
bc = mixed_bc([exit_seg, wall_seg], dimension=2, domain_bounds=bounds)
```

### 2.2 BC Application Layer

Ghost cell applicators exist for structured grids:

```python
# From applicator_fdm.py
def apply_boundary_conditions_2d(
    field: NDArray,
    boundary_conditions: BoundaryConditions,
    domain_bounds: NDArray | None = None,
    time: float = 0.0,
    config: GhostCellConfig | None = None,
) -> NDArray:
    """Pad array with ghost cells computed from BC specification."""
```

**Ghost Cell Formulas** (cell-centered grid):

| BC Type | Formula |
|---------|---------|
| Dirichlet | $u_g = 2g - u_i$ |
| Neumann (left) | $u_g = u_i - 2 \Delta x \cdot g$ |
| Neumann (right) | $u_g = u_i + 2 \Delta x \cdot g$ |
| Robin | $u_g = \frac{g - u_i(\alpha/2 - \beta/2\Delta x)}{\alpha/2 + \beta/2\Delta x}$ |
| Periodic | $u_g^{left} = u_{N-1}$, $u_g^{right} = u_0$ |

### 2.3 Solver BC Usage Audit

| Solver | BC Retrieval | BC Application | Limitation |
|--------|--------------|----------------|------------|
| **HJB GFDM** | `get_boundary_conditions()` | Ghost particles | Fixed recently |
| **HJB Semi-Lagrangian** | `get_boundary_conditions()` | Characteristic clamping | Fixed recently |
| **HJB FDM** | None | One-sided stencils | No BC configuration |
| **HJB WENO** | None | One-sided stencils | No BC configuration |
| **FP FDM** | `problem.components.boundary_conditions` | `fp_fdm_bc.py` | Uniform BCs only |
| **FP Particle** | `geometry.get_boundary_handler()` | Reflection/periodic | Uniform BCs only |

---

## 3. Detailed Problem Analysis

### 3.1 HJB WENO: Hardcoded Boundary Stencils

**Location**: `hjb_solvers/hjb_weno.py:612-625`

```python
# Current implementation - hardcoded one-sided differences
u_x[0] = (-3 * u[0] + 4 * u[1] - u[2]) / (2 * dx)      # Forward
u_x[-1] = (u[-3] - 4 * u[-2] + 3 * u[-1]) / (2 * dx)   # Backward

# Second derivative: extrapolation
u_xx[0] = u_xx[1]
u_xx[-1] = u_xx[-2]
```

**Issue**: This implicitly assumes homogeneous Neumann BC ($\partial^2 u / \partial n^2 \approx 0$). Users cannot specify:
- Periodic domains (common in traffic flow models)
- Dirichlet boundaries (absorbing states)
- Non-zero Neumann flux

### 3.2 HJB FDM: Similar Hardcoding

**Location**: `hjb_solvers/hjb_fdm.py:501-511`

```python
# Boundaries: one-sided differences
grad_left = (U[1] - U[0]) / h      # Forward
grad_right = (U[-1] - U[-2]) / h   # Backward
```

**Issue**: Same limitation as WENO. The nD solver path (lines 150-185) has no BC integration at all.

### 3.3 FP Solvers: Mixed BC Failure

**Location**: `fp_solvers/fp_fdm.py:396`

```python
if self.boundary_conditions.type == "no_flux":  # Fails on mixed BC
```

**Issue**: The `.type` property raises `ValueError` for mixed BCs:

```python
# From conditions.py:143-144
if not self.is_uniform:
    raise ValueError("type property only valid for uniform BCs...")
```

This means FP solvers crash when users specify evacuation scenarios with exits (mixed Dirichlet/Neumann).

---

## 4. Proposed Solution

### 4.1 Design Principles

1. **Single source of truth**: All solvers retrieve BCs via `problem.get_boundary_conditions()`
2. **Centralized application**: Use `applicator_fdm.py` for ghost cell computation
3. **Graceful degradation**: Support mixed BCs where possible; fall back to default_bc for legacy code
4. **No silent failures**: Warn users when BC features are not fully supported

### 4.2 Implementation Strategy

#### Phase 1: Wire Existing Infrastructure (Low Risk)

**HJB WENO/FDM** - Replace hardcoded stencils:

```python
# Before
u_xx[0] = u_xx[1]  # Hardcoded

# After
from mfg_pde.geometry.boundary import apply_boundary_conditions_1d

def _compute_derivatives_with_bc(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute derivatives using BC-aware ghost cells."""
    bc = self._get_boundary_conditions()
    if bc is not None:
        u_padded = apply_boundary_conditions_1d(u, bc, self.domain_bounds)
        # Standard central differences on padded array
        u_x = (u_padded[2:] - u_padded[:-2]) / (2 * self.dx)
        u_xx = (u_padded[2:] - 2*u_padded[1:-1] + u_padded[:-2]) / self.dx**2
    else:
        # Fallback to current one-sided stencils
        u_x, u_xx = self._compute_derivatives_legacy(u)
    return u_x, u_xx
```

#### Phase 2: Mixed BC Support for FP Solvers (Medium Risk)

**FP FDM** - Handle mixed BCs gracefully:

```python
# Before
if self.boundary_conditions.type == "periodic":

# After
def _get_bc_type_safe(self) -> str:
    """Get BC type string, handling both uniform and mixed BCs."""
    bc = self.boundary_conditions
    if bc.is_uniform:
        return bc.type  # Returns string like "periodic", "no_flux"
    else:
        # For mixed BCs, use default_bc for global behavior
        # Individual boundaries handled by applicator
        return bc.default_bc.value
```

**Matrix Assembly** - Per-boundary handling:

```python
# In fp_fdm_operators.py
def add_interior_entries(..., boundary_conditions):
    if boundary_conditions.is_uniform:
        is_periodic = boundary_conditions.type == "periodic"
    else:
        # Check if ANY segment is periodic (requires special matrix structure)
        is_periodic = any(seg.bc_type == BCType.PERIODIC
                         for seg in boundary_conditions.segments)
```

#### Phase 3: Mandatory BC Base Class (Required - Per Expert Review)

> **Note**: Originally proposed as "optional enhancement". Expert review mandates this as a **required base class** to prevent future solver divergence.

```python
class BoundaryAwareSolver(ABC):
    """
    Abstract base class for all grid-based solvers requiring BC handling.

    This is MANDATORY for all structured grid solvers to ensure consistent
    BC handling across the ecosystem.

    Features:
    - Standardized BC retrieval from problem
    - Ghost cell padding with configurable depth
    - Strict mode for explicit BC requirement
    - Corner handling strategy delegation
    """

    # Configuration
    bc_strict_mode: bool = False  # If True, raises error when BC not provided
    ghost_cell_depth: int = 1     # Number of ghost cells (WENO may need 2-3)

    def __init__(self, problem, boundary_conditions=None, bc_strict_mode=False):
        self.problem = problem
        self._bc_strict_mode = bc_strict_mode
        self._boundary_conditions = boundary_conditions

        # Validate BC configuration in strict mode
        if self._bc_strict_mode:
            bc = self._get_boundary_conditions()
            if bc is None:
                raise ValueError(
                    "Strict mode enabled: boundary_conditions must be explicitly provided. "
                    "Either pass boundary_conditions parameter or set bc_strict_mode=False "
                    "to use implicit Neumann BCs."
                )

    def _get_boundary_conditions(self) -> BoundaryConditions | None:
        """Retrieve BCs using standard priority: explicit > problem > None."""
        # 1. Explicit parameter (highest priority)
        if self._boundary_conditions is not None:
            return self._boundary_conditions
        # 2. Problem accessor
        if hasattr(self.problem, 'get_boundary_conditions'):
            return self.problem.get_boundary_conditions()
        # 3. No BC available
        return None

    def _apply_bc_padding(
        self,
        field: np.ndarray,
        ghost_depth: int | None = None
    ) -> np.ndarray:
        """
        Apply ghost cell padding based on BCs.

        Args:
            field: Interior field values
            ghost_depth: Number of ghost cells (default: self.ghost_cell_depth)

        Returns:
            Padded array with ghost cells
        """
        bc = self._get_boundary_conditions()
        depth = ghost_depth or self.ghost_cell_depth

        if bc is None:
            # No BC: return field with zero-padded ghost cells (implicit Neumann)
            return np.pad(field, depth, mode='edge')

        return apply_boundary_conditions_nd(
            field, bc, self.domain_bounds,
            ghost_depth=depth,
            corner_strategy=bc.corner_strategy
        )

    def _get_bc_type_at_boundary(self, boundary: str) -> BCType:
        """Get BC type for specific boundary (handles mixed BCs)."""
        bc = self._get_boundary_conditions()
        if bc is None:
            return BCType.NEUMANN  # Default assumption
        if bc.is_uniform:
            return bc.default_bc
        # Mixed BC: find matching segment
        return bc.get_bc_type_at_boundary(boundary)

    @abstractmethod
    def _get_required_ghost_depth(self) -> int:
        """
        Return the number of ghost cells required by this solver's stencil.

        Override in subclasses:
        - FDM (2nd order): return 1
        - WENO (5th order): return 3
        - Compact schemes: return 2
        """
        pass
```

### 4.3 Validation Strategy

| Test Case | Solver | BC Type | Expected Behavior |
|-----------|--------|---------|-------------------|
| 1D periodic HJB | WENO, FDM | Periodic | Solution wraps at boundaries |
| 2D evacuation | FP FDM | Mixed (exit + walls) | Density drains through exit |
| 2D Dirichlet HJB | FDM | Dirichlet | Value fixed at boundary |
| 1D no-flux FP | FP FDM | No-flux | Mass conserved (integral = 1) |

---

## 5. Numerical Considerations (Expert Review)

This section addresses critical numerical issues identified during expert review that must be resolved before implementation.

### 5.1 The Corner Problem

**Issue**: In 2D/3D domains with mixed BCs, ghost cells at corners (e.g., $u_{-1,-1}$) require special handling when different BC types meet.

**Example**: Domain with Dirichlet wall on North ($u = g_N$) and Neumann wall on West ($\partial u/\partial n = g_W$):

```
       North (Dirichlet: u = g_N)
    ┌─────────────────────────┐
    │  u_{-1,-1}  │  u_{0,-1} │  ← Ghost row (North)
    │   (???)     │   (Dir)   │
    ├─────────────┼───────────┤
    │  u_{-1,0}   │  u_{0,0}  │
    │   (Neu)     │ (Interior)│
    └─────────────┴───────────┘
    West (Neumann)
```

The corner ghost cell $u_{-1,-1}$ has ambiguous definition. The `corner_strategy` parameter addresses this:

| Strategy | Formula | Use Case |
|----------|---------|----------|
| `"priority"` | Use higher-priority BC segment | Sharp corners, discontinuous solutions |
| `"average"` | $u_{corner} = \frac{1}{2}(u_{Dir} + u_{Neu})$ | Smooth solutions |
| `"mollify"` | Smooth blending within radius | Lipschitz domains, re-entrant corners |

**Action Required**: Audit `applicator_fdm.py` to verify correct corner handling for all BC type combinations before solver integration.

### 5.2 Stencil Order Consistency

**Issue**: Ghost cell extrapolation order must match interior scheme order to maintain global accuracy.

| Scheme | Interior Order | Required Ghost Depth | Ghost Cell Order |
|--------|---------------|---------------------|------------------|
| FDM (central) | $O(\Delta x^2)$ | 1 | 2nd order ($u_g = 2g - u_i$) |
| WENO-5 | $O(\Delta x^5)$ | 3 | 5th order extrapolation |
| Compact-4 | $O(\Delta x^4)$ | 2 | 4th order extrapolation |

**Current State**: Ghost cell formulas in `applicator_fdm.py` are 2nd order accurate. This is sufficient for standard FDM but **degrades WENO accuracy** near boundaries.

**Derivation** (Dirichlet, 2nd order):
$$u_g = 2g - u_i + O(\Delta x^2)$$

**High-order alternative** (Dirichlet, 4th order):
$$u_g = \frac{16g - 15u_0 + 5u_1 - u_2}{5} + O(\Delta x^4)$$

**Action Required**:
1. Verify ghost cell depth parameter is configurable per solver
2. Implement high-order ghost cell formulas for WENO (or document accuracy limitation)

### 5.3 FP Mass Conservation Verification

**Issue**: For no-flux and periodic BCs, the discrete FP operator must satisfy $\mathbf{1}^T A = \mathbf{0}^T$ (column sums = 0) to guarantee mass conservation.

**Mathematical Requirement**:
$$\frac{d}{dt} \int_\Omega m \, dx = \int_\Omega \partial_t m \, dx = \int_\Omega \nabla \cdot J \, dx = \int_{\partial\Omega} J \cdot n \, ds = 0$$

For discrete system $\dot{\mathbf{m}} = A\mathbf{m}$:
$$\frac{d}{dt} \sum_i m_i = \sum_i (A\mathbf{m})_i = \mathbf{1}^T A \mathbf{m} = 0 \iff \mathbf{1}^T A = \mathbf{0}^T$$

**Test Implementation**:
```python
def test_matrix_mass_conservation(A: sparse.csr_matrix, bc_type: str):
    """Verify discrete operator preserves mass for conservative BCs."""
    if bc_type in ["no_flux", "periodic"]:
        column_sums = np.abs(A.sum(axis=0)).max()
        assert column_sums < 1e-12, f"Mass conservation violated: max column sum = {column_sums}"
```

**Action Required**: Add unit test to `fp_fdm_operators.py` verifying column sums for no-flux and periodic matrix assembly.

### 5.4 Performance: Memory Allocation Overhead (Expert Review Round 2)

**Issue**: The proposed `apply_boundary_conditions_nd` creates new padded arrays at every timestep.

```python
# PROBLEMATIC: Allocate-copy-destroy per step
for t in range(n_steps):
    u_padded = apply_boundary_conditions_nd(u, bc)  # New allocation!
    u = compute_step(u_padded)[g:-g, g:-g]
```

**Impact**: For explicit schemes (RK4, Euler) with $10^4$+ steps, memory allocation overhead dominates runtime.

**Solution**: Pre-allocated view-based memory model:

```python
class BoundaryAwareSolver:
    def __init__(self, ...):
        g = self._get_required_ghost_depth()
        # Allocate padded state ONCE at initialization
        self._U_padded = np.zeros((Nx + 2*g, Ny + 2*g))
        # Create view into interior (zero-copy)
        self._U_interior = self._U_padded[g:-g, g:-g]

    def _apply_bc_in_place(self):
        """Update ghost cells without allocation."""
        g = self.ghost_cell_depth
        bc = self._get_boundary_conditions()

        # Left boundary (example for 1D Dirichlet)
        if bc.get_bc_type_at_boundary("x_min") == BCType.DIRICHLET:
            bc_value = bc.get_value_at_boundary("x_min")
            self._U_padded[:g] = 2 * bc_value - self._U_padded[g:2*g]

        # Right boundary
        if bc.get_bc_type_at_boundary("x_max") == BCType.DIRICHLET:
            bc_value = bc.get_value_at_boundary("x_max")
            self._U_padded[-g:] = 2 * bc_value - self._U_padded[-2*g:-g]

        # Similar for Neumann, Periodic, etc.

    def solve_timestep(self):
        self._apply_bc_in_place()  # Only touches boundary rim
        # Compute on full padded array, result written to interior view
        self._compute_step_in_place(self._U_padded)
```

**Performance Comparison**:

| Strategy | Allocation per Step | Memory Touched | Cache Efficiency |
|----------|-------------------|----------------|------------------|
| Allocate-copy | $O(N)$ | Full array | Poor |
| Pre-allocated view | $O(1)$ | Boundary rim only | Excellent |

**Action Required**:
1. Design `BoundaryAwareSolver` with pre-allocated padded state
2. Implement `_apply_bc_in_place()` method
3. Benchmark allocation overhead vs hardcoded indexing

### 5.5 N-Dimensional Corner Generalization (Expert Review Round 2)

**Issue**: Section 5.1 addresses 2D corners. In 3D:
- **Edges**: 2 boundaries intersect (12 edges per cube)
- **Corners**: 3 boundaries intersect (8 corners per cube)

Current `corner_strategy` is ambiguous for 3D corner where Dirichlet, Neumann, and Periodic all meet.

**Solution**: Hierarchical BC Precedence

```python
BC_PRECEDENCE = {
    BCType.PERIODIC: 1,    # Highest: wraps, effectively removes boundary
    BCType.DIRICHLET: 2,   # Hard constraint, overrides flux
    BCType.NEUMANN: 3,     # Flux constraint
    BCType.ROBIN: 4,       # Mixed constraint
    BCType.NO_FLUX: 5,     # Lowest priority
}

def resolve_nd_intersection(bc_types: list[BCType]) -> BCType:
    """Resolve N boundary types meeting at a point."""
    # Periodic is special: if ANY axis is periodic, it dominates
    if BCType.PERIODIC in bc_types:
        return BCType.PERIODIC

    # Otherwise, highest precedence wins
    return min(bc_types, key=lambda bc: BC_PRECEDENCE[bc])
```

**3D Example**: Corner at $(x_{min}, y_{max}, z_{min})$
- West face: Neumann
- North face: Dirichlet ($u = g$)
- Bottom face: No-flux

Resolution: Dirichlet wins → $u_{corner}$ extrapolated from Dirichlet formula.

**Action Required**: Extend `applicator_fdm.py` to handle 3D edges and corners with precedence logic.

### 5.6 High-Order Ghost Cell Extrapolation

**Issue**: 2nd-order ghost cells ($u_g = 2g - u_i$) degrade WENO-5 global accuracy from $O(\Delta x^5)$ to $O(\Delta x^2)$ near boundaries.

**Solution**: Lagrange polynomial extrapolation matching scheme order.

| Order | Dirichlet Formula | Points Used |
|-------|-------------------|-------------|
| 2nd | $u_g = 2g - u_0$ | 1 interior |
| 4th | $u_g = \frac{16g - 15u_0 + 5u_1 - u_2}{5}$ | 3 interior |
| 5th | Lagrange from $u_0, u_1, u_2, u_3$ | 4 interior |

**Neumann (4th order)**:
$$u_g = u_0 - 2\Delta x \cdot g + \frac{1}{3}(u_2 - 4u_1 + 3u_0)$$

**Implementation**:
```python
def high_order_dirichlet_ghost(u_interior: np.ndarray, bc_value: float, order: int) -> float:
    """Compute ghost cell value for Dirichlet BC at specified order."""
    if order == 2:
        return 2 * bc_value - u_interior[0]
    elif order == 4:
        return (16 * bc_value - 15 * u_interior[0] + 5 * u_interior[1] - u_interior[2]) / 5
    elif order == 5:
        # Lagrange extrapolation through boundary
        return lagrange_extrapolate(u_interior[:4], bc_value, target=-1)
```

**Action Required**: Add `ghost_order` parameter to `BoundaryAwareSolver`, implement high-order formulas.

---

## 6. Validation Strategy (Updated Per Expert Review)

### 6.1 Method of Manufactured Solutions (MMS)

> **Note**: Qualitative tests ("solution wraps at boundaries") are insufficient. MMS provides rigorous numerical verification.

**Procedure**:
1. Choose analytical solution $u^*(x,t)$ satisfying desired non-homogeneous BC
2. Compute forcing term $f(x,t) = \partial_t u^* + H(\nabla u^*) - \frac{\sigma^2}{2}\Delta u^*$
3. Solve modified PDE with forcing: $\partial_t u + H(\nabla u) - \frac{\sigma^2}{2}\Delta u = f$
4. Measure error: $\|u - u^*\|_\infty$, $\|u - u^*\|_2$
5. Verify convergence rate: error $\propto \Delta x^p$ where $p$ = expected order

**Example MMS Test** (1D Dirichlet):
```python
def test_hjb_dirichlet_mms():
    """MMS test for HJB with non-homogeneous Dirichlet BC."""
    # Manufactured solution: u*(x,t) = sin(pi*x) * exp(-t)
    # BC: u(0,t) = 0, u(1,t) = 0 (homogeneous Dirichlet)

    def u_exact(x, t):
        return np.sin(np.pi * x) * np.exp(-t)

    def forcing(x, t, sigma=0.1):
        # f = du*/dt + H(du*/dx) - sigma^2/2 * d^2u*/dx^2
        u = u_exact(x, t)
        u_t = -u  # d/dt[sin(pi*x)*exp(-t)] = -sin(pi*x)*exp(-t)
        u_x = np.pi * np.cos(np.pi * x) * np.exp(-t)
        u_xx = -np.pi**2 * u
        H = 0.5 * u_x**2  # Quadratic Hamiltonian
        return u_t + H - 0.5 * sigma**2 * u_xx

    # Run solver with forcing term
    errors = []
    for nx in [32, 64, 128, 256]:
        solver = HJBFDMSolver(problem, nx=nx, forcing=forcing)
        u_numerical = solver.solve()
        error = np.max(np.abs(u_numerical - u_exact(x_grid, t_final)))
        errors.append(error)

    # Verify 2nd order convergence
    rates = [np.log2(errors[i]/errors[i+1]) for i in range(len(errors)-1)]
    assert all(r > 1.9 for r in rates), f"Expected O(dx^2), got rates: {rates}"
```

### 6.2 Coupled HJB-FP Integration Tests

**Issue**: BC changes in HJB affect optimal control $\alpha^* = -\nabla_p H(\nabla u)$, which modifies FP drift.

**Test Case**: Evacuation problem with exit at boundary
1. Solve HJB with Dirichlet BC at exit ($u = 0$)
2. Compute optimal drift $\alpha^* = -\lambda \nabla u$
3. Solve FP with no-flux walls + absorbing exit
4. Verify: density drains through exit, mass decreases monotonically

```python
def test_coupled_evacuation():
    """Integration test: HJB BC affects FP flow correctly."""
    # Setup: 2D domain with exit at y=1
    bc_hjb = mixed_bc([
        BCSegment("exit", BCType.DIRICHLET, value=0.0, boundary="y_max"),
        BCSegment("walls", BCType.NEUMANN, value=0.0)
    ], dimension=2)

    bc_fp = mixed_bc([
        BCSegment("exit", BCType.DIRICHLET, value=0.0, boundary="y_max"),  # Absorbing
        BCSegment("walls", BCType.NO_FLUX)
    ], dimension=2)

    # Solve coupled system
    hjb_solver = HJBFDMSolver(problem, boundary_conditions=bc_hjb)
    u_solution = hjb_solver.solve()

    fp_solver = FPFDMSolver(problem, boundary_conditions=bc_fp, drift_from_hjb=u_solution)
    m_solution = fp_solver.solve()

    # Verify mass decreases (agents exit)
    mass_history = [np.sum(m_solution[t]) for t in range(len(m_solution))]
    assert all(mass_history[i] >= mass_history[i+1] for i in range(len(mass_history)-1))
```

---

## 7. Risk Assessment (Updated)

| Change | Original Risk | Updated Risk | Mitigation |
|--------|---------------|--------------|------------|
| Corner ghost cell handling | Not assessed | **High** | Phase 0 audit required |
| WENO ghost depth | Low | **Medium** | Verify 3-cell depth |
| HJB WENO BC wiring | Low | Low | Fallback to legacy if BC=None |
| HJB FDM BC wiring | Low | Low | Same fallback pattern |
| FP matrix conservation | Assumed | **Medium** | Column sum unit tests |
| FP FDM mixed BC | Medium | Medium | Extensive testing; gradual rollout |
| FP Particle mixed BC | Low | Low | Already has hasattr checks |
| Performance overhead | Not assessed | Low | Vectorized ghost padding |

**Backward Compatibility**: All changes are additive. Solvers without explicit BC specification continue to use current (implicit Neumann) behavior.

**New Risk**: Strict mode may break existing user code that relies on implicit BCs. Mitigation: strict mode is opt-in (default `False`).

---

## 8. Implementation Checklist (Revision 1.2 - Performance Focus)

### Phase 0: Core Architecture (Critical Path)
- [ ] **0a. Corner logic audit**: Verify `applicator_fdm.py` handles all BC type combinations at corners
- [ ] **0b. Document corner strategies**: Add examples for priority/average/mollify in different scenarios
- [ ] **0c. Pre-allocation design**: Prototype zero-copy view-based memory model
- [ ] **0d. N-D intersection logic**: Extend corner handling to 3D edges/corners with precedence
- [ ] **0e. Stateless verification**: Ensure ghost cell functions are purely functional (no side effects)

### Phase 1: HJB Solver Integration
- [ ] **1a. WENO ghost depth**: Verify WENO-5 requires 3 ghost cells, implement if needed
- [ ] **1b. `BoundaryAwareSolver` base class**: Implement with:
  - Strict mode configuration
  - Pre-allocated padded state (`_U_padded`, `_U_interior` view)
  - In-place BC application (`_apply_bc_in_place`)
  - Configurable ghost depth and order
- [ ] **1c. HJB WENO BC wiring**: Replace hardcoded stencils with ghost cell padding
- [ ] **1d. HJB FDM BC wiring**: Same for nD path
- [ ] **1e. High-order extrapolation**: Implement 4th/5th order ghost cell formulas for WENO

### Phase 2: FP Solver Integration
- [ ] **2a. Matrix conservation tests**: Unit tests verifying column sums = 0 for no-flux/periodic
- [ ] **2b. Safe BC type accessor**: Handle both uniform and mixed BCs in `fp_fdm.py`
- [ ] **2c. FP FDM mixed BC**: Update `fp_fdm_operators.py` for per-boundary handling
- [ ] **2d. FP Particle mixed BC**: Update reflection/periodic handling

### Phase 3: Validation & Testing
- [ ] **3a. MMS tests**: Implement Method of Manufactured Solutions for each solver
- [ ] **3b. Convergence rate verification**: Ensure $O(\Delta x^2)$ for FDM, $O(\Delta x^5)$ for WENO
- [ ] **3c. Coupled HJB-FP tests**: Integration tests for evacuation scenarios
- [ ] **3d. Mass conservation tests**: Verify $\int m \, dx = 1$ for closed domains
- [ ] **3e. Conservation stress test**: Run FP mass test for $10^5$ steps (floating-point drift check)

### Phase 4: Performance & Documentation
- [ ] **4a. Update user documentation**: BC usage examples, strict mode explanation
- [ ] **4b. Performance benchmark**: Compare pre-allocated vs allocate-per-step overhead
- [ ] **4c. Migration guide**: Instructions for users upgrading from implicit to explicit BCs
- [ ] **4d. Numba/Cython JIT**: Accelerate `applicator_fdm.py` ghost cell loops (optional)
- [ ] **4e. Memory benchmark**: Profile memory usage with pre-allocated vs dynamic allocation

---

## 9. References

1. **Ghost Cell Methods**: LeVeque, R.J. "Finite Difference Methods for Ordinary and Partial Differential Equations" (2007), Chapter 9
2. **MFG Boundary Conditions**: Achdou, Y. et al. "Mean Field Games: Numerical Methods" (2020), Section 4.3
3. **Method of Manufactured Solutions**: Roache, P.J. "Code Verification by the Method of Manufactured Solutions" (2002), Journal of Fluids Engineering
4. **WENO Schemes**: Shu, C.W. "High Order Weighted Essentially Non-oscillatory Schemes" (1998), ICASE Report
5. **Existing Infrastructure**: `mfg_pde/geometry/boundary/applicator_fdm.py` (internal documentation)

---

## Appendix A: File Locations (Updated)

```
mfg_pde/
├── geometry/boundary/
│   ├── conditions.py          # BC specification (complete)
│   ├── applicator_fdm.py      # Ghost cells (complete)
│   └── applicator_meshfree.py # Collocation BC (partial)
├── alg/numerical/
│   ├── hjb_solvers/
│   │   ├── hjb_weno.py        # Needs BC wiring
│   │   ├── hjb_fdm.py         # Needs BC wiring
│   │   ├── hjb_gfdm.py        # BC support added
│   │   └── hjb_semi_lagrangian.py  # BC support added
│   └── fp_solvers/
│       ├── fp_fdm.py          # Needs mixed BC support
│       ├── fp_fdm_bc.py       # No-flux enforcement (complete)
│       └── fp_particle.py     # Needs mixed BC support
```

---

## Appendix B: Mathematical Background

### B.1 HJB Equation with Boundary Conditions

$$\begin{cases}
\partial_t u + H(x, \nabla u) - \frac{\sigma^2}{2} \Delta u = 0 & \text{in } (0,T) \times \Omega \\
u(T, x) = g(x) & \text{terminal condition} \\
\mathcal{B}[u] = h(x) & \text{on } (0,T) \times \partial\Omega
\end{cases}$$

where $\mathcal{B}$ is the boundary operator:
- Dirichlet: $\mathcal{B}[u] = u$
- Neumann: $\mathcal{B}[u] = \partial u / \partial n$
- Robin: $\mathcal{B}[u] = \alpha u + \beta \partial u / \partial n$

### B.2 FP Equation with No-Flux Boundary

$$\begin{cases}
\partial_t m - \nabla \cdot (m \nabla_p H) - \frac{\sigma^2}{2} \Delta m = 0 & \text{in } (0,T) \times \Omega \\
m(0, x) = m_0(x) & \text{initial condition} \\
J \cdot n = 0 & \text{on } (0,T) \times \partial\Omega
\end{cases}$$

where $J = m \nabla_p H - \frac{\sigma^2}{2} \nabla m$ is the probability flux.

### B.3 Ghost Cell Derivation (Neumann)

For $\partial u / \partial n = g$ at left boundary with cell-centered grid:

$$\frac{u_0 - u_{-1}}{2\Delta x} = g \implies u_{-1} = u_0 - 2\Delta x \cdot g$$

where $u_{-1}$ is the ghost cell value and $u_0$ is the first interior cell.
