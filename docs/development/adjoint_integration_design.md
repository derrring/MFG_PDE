# Adjoint Enforcement Integration Design

## Status: ⚠️ REVISION REQUIRED (Issue #704)

## Executive Summary

**⚠️ CRITICAL UPDATE (2026-01):** The original `adjoint_mode="transpose"` implementation was **mathematically incorrect**. Simple matrix transpose of the HJB gradient operator does NOT give the correct FP divergence operator.

**Correct approach:** Use `gradient_upwind` (HJB) + `divergence_upwind` (FP) schemes. This pairing IS the correct discrete adjoint — both schemes use the same velocity field, and the divergence scheme is equivalent to the Jacobian transpose (not the gradient matrix transpose).

See `docs/theory/adjoint_discretization_mfg.md` for complete mathematical foundations.

---

## ⚠️ DEPRECATED: Matrix Transpose Approach

### What Was Implemented (INCORRECT)

```
FixedPointIterator (adjoint_mode="transpose")
    │
    ├── HJB Solver: build_advection_matrix(U) → A_gradient
    │
    └── FP Solver: solve_fp_step_adjoint_mode(m, A_gradient.T)  ❌ WRONG!
```

**Why it's wrong:**
- `build_advection_matrix()` returns the gradient operator matrix for $v \cdot \nabla u$
- Simple transpose gives $A_{\nabla}^T$, which is NOT the divergence operator
- Mass conservation is violated (experimentally observed 97% mass loss!)

**Key Files (to be deprecated):**
- `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py:1016` - `build_advection_matrix()` — ⚠️ Returns gradient matrix
- `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py:511` - `solve_fp_step_adjoint_mode()` — ⚠️ Uses wrong matrix
- `mfg_pde/alg/numerical/coupling/fixed_point_iterator.py` - `_solve_fp_strict_adjoint()` — ⚠️ Broken

---

## 1. Correct Approach: Scheme-Based Adjoint

### 1.1 The Correct Pairing

```python
# ✅ CORRECT: Use divergence_upwind for FP (this IS the discrete adjoint)
hjb_solver = HJBFDMSolver(problem, advection_scheme="gradient_upwind")
fp_solver = FPFDMSolver(problem, advection_scheme="divergence_upwind")

solver = FixedPointIterator(
    problem=problem,
    hjb_solver=hjb_solver,
    fp_solver=fp_solver,
    adjoint_mode="off",  # Standard mode - schemes handle adjoint structure
)
```

### 1.2 Why This Works

The `divergence_upwind` scheme constructs coefficients equivalent to the **Jacobian transpose** (not the gradient transpose):

| Scheme | Operator | Matrix Structure | Mass Conservation |
|--------|----------|------------------|-------------------|
| `gradient_upwind` | $v \cdot \nabla u$ | Row $i$: $[-v_i/h, +v_i/h, 0]$ | N/A (for HJB) |
| ❌ Naive `.T` | $(A_{\nabla})^T$ | Row $i$: $[0, +v_i/h, -v_{i+1}/h]$ | ❌ Broken |
| ✅ `divergence_upwind` | $\nabla \cdot (vm)$ | Row $i$: $[-v_{i-1}/h, +v_i/h, 0]$ | ✅ Conserved |

The key difference: velocity indices! `divergence_upwind` uses $v_{i-1}$ and $v_i$, which matches the Jacobian transpose structure from Achdou's method.

### 1.3 What `adjoint_mode` Should Mean (Future)

| Mode | Meaning | Status |
|------|---------|--------|
| `"off"` | Use independent scheme-based adjoint (recommended) | ✅ Works correctly |
| `"transpose"` | ~~Use A_gradient.T~~ | ❌ **DEPRECATED** - mathematically wrong |
| `"auto"` | ~~BC-aware transpose~~ | ❌ **DEPRECATED** - based on wrong foundation |
| `"verify"` | Compare schemes, log warnings | 🔲 Future: compare gradient_upwind vs divergence_upwind |

---

## 2. What Still Works

### 2.1 Verification (Still Useful)

Runtime verification that `gradient_upwind` and `divergence_upwind` produce consistent results:

```python
from mfg_pde.alg.numerical.adjoint import check_operator_adjoint

# Compare the schemes at a given U
A_hjb = hjb_solver._build_gradient_matrix(U)  # gradient_upwind
A_fp = fp_solver._build_divergence_matrix(U)   # divergence_upwind

# These should satisfy: A_fp ≈ A_jacobian.T (NOT A_gradient.T)
# The verification checks if FP's divergence matrix matches the expected structure
```

### 2.2 Diagnostics (Still Useful)

The `adjoint/` module's diagnostics can verify scheme consistency:

```python
from mfg_pde.alg.numerical.adjoint import diagnose_adjoint_error

report = diagnose_adjoint_error(A_gradient, A_divergence, geometry=geometry)
# Identifies if discrepancy is at boundaries or interior
# Useful for debugging scheme implementations
```

---

## 3. DEPRECATED: Matrix Transpose Integration

### 2.1 Verification Hook

Add verification at matrix exchange point:

```python
# In fixed_point_iterator.py

from mfg_pde.alg.numerical.adjoint import (
    verify_discrete_adjoint,
    diagnose_adjoint_error,
    check_operator_adjoint,
)

class FixedPointIterator:
    def __init__(
        self,
        ...,
        strict_adjoint: bool = False,
        adjoint_verification: bool = False,  # NEW
        adjoint_rtol: float = 1e-10,          # NEW
    ):
        self.strict_adjoint = strict_adjoint
        self.adjoint_verification = adjoint_verification
        self.adjoint_rtol = adjoint_rtol

    def _solve_fp_strict_adjoint(self, ...):
        for k in range(num_time_steps - 1):
            A_hjb = hjb_solver.build_advection_matrix(U_k)

            # NEW: Optional verification
            if self.adjoint_verification:
                A_fp_independent = fp_solver._build_advection_matrix(U_k)
                is_adj, err = check_operator_adjoint(A_hjb, A_fp_independent.T, self.adjoint_rtol)
                if not is_adj:
                    logger.warning(f"Adjoint mismatch at step {k}: error={err:.2e}")

            A_hjb_T = A_hjb.T.tocsr()
            M_next = fp_solver.solve_fp_step_adjoint_mode(M_current, A_hjb_T, sigma)
```

### 2.2 Diagnostics Integration

Add post-solve diagnostics:

```python
class FixedPointIterator:
    def solve(self) -> SolverResult:
        result = self._picard_iteration()

        # NEW: Adjoint consistency report
        if self.adjoint_verification:
            result.adjoint_report = self._generate_adjoint_report()

        return result

    def _generate_adjoint_report(self) -> AdjointDiagnosticReport:
        """Generate comprehensive adjoint consistency report."""
        from mfg_pde.alg.numerical.adjoint import diagnose_adjoint_error

        # Use final timestep operators
        A_hjb = self.hjb_solver.build_advection_matrix(self._U_final[-2])
        A_fp_ind = self.fp_solver._build_advection_matrix(self._U_final[-2])

        return diagnose_adjoint_error(
            A_hjb, A_fp_ind.T,
            geometry=self.problem.geometry,
        )
```

---

## 3. User-Facing API

### 3.1 Problem-Level Configuration

```python
from mfg_pde import MFGProblem

problem = MFGProblem(
    ...,
    # NEW: Adjoint consistency settings
    adjoint_mode="strict",      # "strict" | "verify" | "off"
    adjoint_diagnostics=True,   # Generate report after solve
)

result = problem.solve()

# Access adjoint report
if result.adjoint_report:
    print(result.adjoint_report)
```

### 3.2 Mode Definitions

| Mode | Description | Use Case |
|------|-------------|----------|
| `"off"` | No enforcement (default) | Backward compatibility, fast |
| `"verify"` | Build independently, verify | Development, debugging |
| `"strict"` | FP = HJB^T always | Production, guaranteed adjoint |

### 3.3 Solver-Level Override

```python
from mfg_pde.alg.numerical.coupling import FixedPointIterator

solver = FixedPointIterator(
    problem=problem,
    hjb_solver=hjb_solver,
    fp_solver=fp_solver,

    # Adjoint settings (override problem defaults)
    strict_adjoint=True,
    adjoint_verification=True,
    adjoint_rtol=1e-8,
)
```

---

## 4. Enforcement Architecture

### 4.1 Three-Level Enforcement

```
Level 1: COMPILE-TIME (type checking)
    - Protocol ensures solvers have required methods
    - build_advection_matrix() signature enforced

Level 2: INIT-TIME (validation)
    - Check BC compatibility between HJB and FP
    - Verify coupling coefficients match
    - Validate geometry supports required operators

Level 3: RUN-TIME (verification)
    - Optional matrix adjoint check at each timestep
    - Log warnings for adjoint mismatch
    - Generate diagnostic report post-solve
```

### 4.2 Protocol Definition

```python
# In mfg_pde/alg/numerical/adjoint/protocols.py (NEW)

from typing import Protocol, runtime_checkable
from scipy import sparse

@runtime_checkable
class AdjointCapableHJBSolver(Protocol):
    """Protocol for HJB solvers that support strict adjoint mode."""

    def build_advection_matrix(
        self,
        U: np.ndarray,
        time_index: int | None = None,
    ) -> sparse.csr_matrix:
        """Build advection matrix A such that FP can use A^T."""
        ...


@runtime_checkable
class AdjointCapableFPSolver(Protocol):
    """Protocol for FP solvers that support strict adjoint mode."""

    def solve_fp_step_adjoint_mode(
        self,
        m_current: np.ndarray,
        A_advection_T: sparse.csr_matrix,
        sigma: float,
    ) -> np.ndarray:
        """Solve FP step using externally provided advection matrix."""
        ...
```

### 4.3 Validation at Init

```python
class FixedPointIterator:
    def __init__(self, ..., strict_adjoint: bool = False):
        if strict_adjoint:
            self._validate_adjoint_capability()

    def _validate_adjoint_capability(self):
        """Ensure solvers support strict adjoint mode."""
        from mfg_pde.alg.numerical.adjoint.protocols import (
            AdjointCapableHJBSolver,
            AdjointCapableFPSolver,
        )

        if not isinstance(self.hjb_solver, AdjointCapableHJBSolver):
            raise TypeError(
                f"HJB solver {type(self.hjb_solver).__name__} does not support "
                "strict adjoint mode. Must implement build_advection_matrix()."
            )

        if not isinstance(self.fp_solver, AdjointCapableFPSolver):
            raise TypeError(
                f"FP solver {type(self.fp_solver).__name__} does not support "
                "strict adjoint mode. Must implement solve_fp_step_adjoint_mode()."
            )
```

---

## 5. Boundary Handling in Strict Adjoint Mode

### 5.1 BC Classification for Adjoint Mode

| BC Type | Mathematical Form | Transpose Valid? | `adjoint_mode="auto"` |
|---------|------------------|------------------|----------------------|
| **Reflecting** | ∂m/∂n = 0 | ✅ Yes | ✅ Supported |
| **Periodic** | m(x_min) = m(x_max) | ✅ Yes | ✅ Supported |
| **Zero-flux Neumann** | ∂m/∂n = 0 | ✅ Yes | ✅ Supported |
| **Absorbing/Outflow** | Mass exits (homogeneous) | ⚠️ Needs fix | ✅ Supported |
| **Dirichlet** | m = g(x,t) | ❌ No | ❌ NotImplementedError |
| **Neumann with flux** | ∂m/∂n = g ≠ 0 | ❌ No (source term) | ❌ NotImplementedError |
| **Robin** | αm + β∂m/∂n = g | ❌ No | ❌ NotImplementedError |
| **Inflow** | Specified flux in | ❌ No (source term) | ❌ NotImplementedError |

**Key Insight**: Transpose alone only works for **homogeneous** boundary conditions where no source term is needed. Non-zero flux or fixed-value BCs require source term handling in the FP equation.

### 5.2 Critical: Flow Direction Reversal

**Key insight**: Transpose reverses the flow direction!

```
HJB characteristic: dx/dt = -∂H/∂p (backward in time)
FP characteristic:  dx/dt = α = -∇U (forward in time, opposite direction)

HJB rightward (v > 0) → FP (transpose) flows LEFTWARD
HJB leftward (v < 0)  → FP (transpose) flows RIGHTWARD
```

**Consequence for Dirichlet BC:**
- If HJB has Dirichlet at x_max (right boundary)
- FP flow (via transpose) goes leftward
- Mass exits at x_min (LEFT boundary), not x_max
- Adjustment needed at the **upstream** boundary of FP flow

### 5.3 Verified Behavior

```python
# Reflecting/Periodic/Neumann BC: Transpose works correctly
m_final = A_hjb_reflect.T @ m0  # Mass stays in domain ✅

# Dirichlet BC: Transpose FAILS
m_final = A_hjb_dirichlet.T @ m0  # Mass stays at boundary ❌
# Expected: Mass should exit domain

# With BC-aware adjustment: Works correctly
A_fp = build_bc_aware_adjoint_matrix(A_hjb, bc_types, grid_shape, dx, dt)
m_final = A_fp @ m0  # Mass exits correctly ✅
```

### 5.4 BC-Aware Adjoint Strategy

```python
def build_adjoint_fp_matrix(A_hjb, bc_types, geometry):
    """
    Build FP matrix with BC-aware adjustment.

    Args:
        A_hjb: HJB advection matrix
        bc_types: Dict mapping boundary names to BC types
        geometry: Grid geometry

    Returns:
        A_fp: Properly adjoint FP matrix
    """
    A_fp = A_hjb.T.tocsr().tolil()  # Start with transpose

    boundary_indices = get_boundary_indices_from_geometry(geometry)

    for boundary_name, indices in boundary_indices.items():
        bc_type = bc_types.get(boundary_name, "reflecting")

        if bc_type in ("reflecting", "periodic"):
            # Transpose is correct, no adjustment needed
            pass

        elif bc_type == "absorbing":
            # Fix boundary rows: mass should flow out
            for i in indices:
                # Replace identity column effect with outflow
                # A_fp[i, :] should allow mass to exit
                _apply_outflow_bc(A_fp, i, geometry)

        elif bc_type == "inflow":
            # Inflow handled via source term, not matrix
            # Mark for separate treatment
            pass

    return A_fp.tocsr()
```

### 5.4 Implementation in FixedPointIterator

```python
class FixedPointIterator:
    def _solve_fp_strict_adjoint(self, ...):
        for k in range(num_time_steps - 1):
            A_hjb = hjb_solver.build_advection_matrix(U_k)

            # BC-aware transpose (NEW)
            if self.bc_aware_adjoint:
                bc_types = self._get_boundary_bc_types()
                A_fp = build_adjoint_fp_matrix(A_hjb, bc_types, self.geometry)
            else:
                # Simple transpose (only valid for reflecting/periodic)
                A_fp = A_hjb.T.tocsr()

            M_next = fp_solver.solve_fp_step_adjoint_mode(M_current, A_fp, sigma)
```

### 5.5 Recommended Usage

| BC Configuration | Recommended Mode |
|------------------|------------------|
| All reflecting/periodic/Neumann | `strict_adjoint=True` (simple transpose) |
| Has Dirichlet (absorbing/outflow) | `strict_adjoint=True, bc_aware_adjoint=True` |
| Has inflow BC | Handle source terms separately |

### 5.6 Summary

**Simple rule**:
- Transpose alone works for symmetric BC (reflecting, periodic, Neumann)
- Dirichlet BC requires `build_bc_aware_adjoint_matrix()` adjustment
- Remember flow direction reversal when determining which boundary needs adjustment

### 5.7 State-Dependent BC Integration

For cases where even BC-aware transpose is insufficient:

```python
from mfg_pde.alg.numerical.adjoint import create_adjoint_consistent_bc_1d

class FixedPointIterator:
    def _solve_fp_strict_adjoint(self, ...):
        for k in range(num_time_steps - 1):
            A_hjb = hjb_solver.build_advection_matrix(U_k)
            A_hjb_T = A_hjb.T.tocsr()

            # NEW: Apply boundary correction if needed
            if self.adjoint_bc_correction:
                bc = create_adjoint_consistent_bc_1d(
                    m_current=M_current,
                    dx=self.problem.geometry.get_grid_spacing()[0],
                    sigma=self.problem.sigma,
                )
                A_hjb_T = self._apply_boundary_correction(A_hjb_T, bc)

            M_next = fp_solver.solve_fp_step_adjoint_mode(M_current, A_hjb_T, sigma)
```

---

## 6. Implementation Plan

### Phase 1: Protocol & Validation ✅ COMPLETED
- [x] Add `AdjointCapableHJBSolver` protocol → `adjoint/protocols.py`
- [x] Add `AdjointCapableFPSolver` protocol → `adjoint/protocols.py`
- [x] Add validation in `FixedPointIterator.__init__` → Uses `validate_adjoint_capability()`

### Phase 2: Verification Integration ✅ COMPLETED
- [x] Add `adjoint_verification` parameter → `FixedPointIterator(adjoint_verification=True)`
- [x] Add `adjoint_rtol` parameter → Default `1e-10`
- [x] Integrate `check_operator_adjoint()` at matrix exchange → `_solve_fp_strict_adjoint()`
- [x] Add logging for adjoint mismatch → Logs warnings with mismatch count
- [x] Include verification results in `SolverResult.metadata`

### Phase 3: Diagnostics Integration ✅ COMPLETED
- [x] Add `_generate_adjoint_report()` method → `FixedPointIterator._generate_adjoint_report()`
- [x] Include `adjoint_report` in `SolverResult` → `result.metadata["adjoint_report"]`
- [ ] User-facing API in `MFGProblem` (Optional, deferred - users can access via solver directly)

### Phase 4: Full BC Support (Future - Issue #705)
- [ ] Dirichlet BC with source term handling
- [ ] Non-zero flux Neumann (∂m/∂n = g ≠ 0)
- [ ] Robin BC (αm + β∂m/∂n = g)
- [ ] Inflow BC with source term
- [ ] State-dependent BC in adjoint mode

**Current Limitation (v0.17.x)**:
`adjoint_mode="auto"` only supports homogeneous BCs:
- ✅ Reflecting (zero-flux Neumann): ∂m/∂n = 0
- ✅ Periodic
- ✅ Absorbing/Outflow (homogeneous only)
- ❌ Dirichlet: raises `NotImplementedError`
- ❌ Robin: raises `NotImplementedError`
- ❌ Inflow/Flux with source: raises `NotImplementedError`

For unsupported BC types, use `adjoint_mode="off"` and handle BCs manually.

---

## 7. Example Usage

### 7.1 Basic Strict Adjoint

```python
from mfg_pde import MFGProblem

problem = MFGProblem(
    geometry=grid,
    hamiltonian=H,
    terminal_cost=g,
    initial_density=m0,
    sigma=0.3,
    adjoint_mode="strict",
)

result = problem.solve()
# FP operator is always HJB^T - guaranteed adjoint consistency
```

### 7.2 With Verification and Diagnostics

```python
problem = MFGProblem(
    ...,
    adjoint_mode="verify",
    adjoint_diagnostics=True,
)

result = problem.solve()

# Check adjoint report
if result.adjoint_report.severity != ErrorSeverity.OK:
    print(result.adjoint_report)
    print(f"Boundary error: {result.adjoint_report.boundary_error:.2e}")
```

### 7.3 Low-Level Solver Access

```python
from mfg_pde.alg.numerical.coupling import FixedPointIterator
from mfg_pde.alg.numerical.adjoint import verify_discrete_adjoint

solver = FixedPointIterator(
    problem=problem,
    hjb_solver=HJBFDMSolver(problem),
    fp_solver=FPFDMSolver(problem),
    strict_adjoint=True,
)

# Manual verification
A_hjb = solver.hjb_solver.build_advection_matrix(U)
A_fp = solver.fp_solver._build_advection_matrix(U)

result = verify_discrete_adjoint(A_hjb, A_fp.T)
print(f"Adjoint: {result.is_adjoint}, error: {result.relative_error:.2e}")
```

---

## 8. Summary

### ⚠️ Critical Correction (2026-01)

| Component | Old Status | New Status | Notes |
|-----------|------------|------------|-------|
| `adjoint_mode="transpose"` | ✅ Implemented | ❌ **DEPRECATED** | Mathematically wrong |
| `adjoint_mode="auto"` | ✅ Implemented | ❌ **DEPRECATED** | Based on wrong foundation |
| `solve_fp_step_adjoint_mode()` | ✅ Exists | ⚠️ **DO NOT USE** | Uses wrong matrix |
| `build_advection_matrix()` | ✅ Exists | ⚠️ Returns gradient matrix | Not suitable for FP |
| **Correct approach** | N/A | ✅ Use scheme pairing | `gradient_upwind` + `divergence_upwind` |

### Correct Usage

```python
# ✅ CORRECT: Scheme-based adjoint (Achdou's method)
hjb_solver = HJBFDMSolver(problem, advection_scheme="gradient_upwind")
fp_solver = FPFDMSolver(problem, advection_scheme="divergence_upwind")

solver = FixedPointIterator(
    problem=problem,
    hjb_solver=hjb_solver,
    fp_solver=fp_solver,
    adjoint_mode="off",  # Use standard mode - schemes handle adjoint
)
result = solver.solve()

# Mass is conserved ✅
# Discrete duality is preserved ✅
# No need for matrix transpose tricks ✅
```

### What Still Works

| Component | Status | Use Case |
|-----------|--------|----------|
| `adjoint/` module diagnostics | ✅ Useful | Verify scheme consistency |
| `check_operator_adjoint()` | ✅ Useful | Debug scheme implementations |
| `diagnose_adjoint_error()` | ✅ Useful | Identify boundary vs interior issues |
| Protocol definitions | ⚠️ Needs update | Future: scheme compatibility protocols |

### Key References

- **Theory**: `docs/theory/adjoint_discretization_mfg.md` — Complete mathematical foundations
- **Achdou et al. (2010)**: "Mean Field Games: Numerical Methods" — Original structure-preserving method
- **Issue #704**: Adjoint module integration
- **Issue #705**: Full BC support (future work)

### Lessons Learned

1. **Simple transpose ≠ discrete adjoint** for gradient/divergence operators
2. **Jacobian transpose** (from linearized HJB) = discrete adjoint
3. **`divergence_upwind`** automatically implements the correct Jacobian transpose structure
4. **Always verify** mass conservation when implementing adjoint modes
