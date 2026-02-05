# [SPEC] Linear Solver System

**Document ID**: MFG-SPEC-LS-0.1
**Status**: DRAFT (Deferred — documented for future reference)
**Date**: 2026-02-05
**Dependencies**: MFG-SPEC-OP-0.1 (Operator System)

---

## 1. Motivation

Every implicit time step in MFG_PDE ends with a linear solve: `A x = b`.
The system matrix $A$ varies in structure depending on the PDE, discretization,
and dimension. Currently, solver selection is manual — mostly `spsolve()` everywhere,
with `splu()` in the implicit heat solver and iterative methods available but rarely used.

For current problem sizes (1D-2D, $N < 10^4$), `spsolve()` is adequate.
At larger scales ($N > 10^5$, 3D problems), the choice of linear solver,
preconditioner, and matrix format becomes critical. This spec designs a
**trait-based auto-selection system** that maps matrix properties to optimal
solver configurations.

---

## 2. Current State

### 2.1 SparseSolver Abstraction

**Location**: `mfg_pde/utils/sparse_operations.py:466-621`

A unified solver class already exists:

```python
class SparseSolver:
    def __init__(
        self,
        method: str = "direct",      # direct | cg | gmres | bicgstab
        backend: str = "scipy",       # scipy | cupy
        preconditioner: str | None = None,  # ilu | jacobi | None
    ): ...

    def solve(self, A, b, x0=None, callback=None) -> NDArray: ...
```

**Solver routing**:

| Method | SciPy Function | Applicable When |
|:-------|:---------------|:----------------|
| `direct` | `spsolve(A, b)` | Small/medium systems, any structure |
| `cg` | `cg(A, b, M=...)` | SPD matrices only |
| `gmres` | `gmres(A, b, M=...)` | General nonsymmetric |
| `bicgstab` | `bicgstab(A, b, M=...)` | General nonsymmetric (alternative) |

**GPU support**: CuPy backend for `cg` and `gmres` with automatic fallback.

### 2.2 Preconditioning

| Preconditioner | Status | Implementation |
|:---------------|:-------|:---------------|
| ILU (Incomplete LU) | Implemented | `spla.spilu(A.tocsc())` wrapped as `LinearOperator` |
| Jacobi (diagonal) | Declared | Parameter accepted but **not implemented** |
| AMG (algebraic multigrid) | Not available | Would require `pyamg` dependency |
| SSOR | Not available | Standard option, not implemented |

### 2.3 Where Linear Solves Happen

| Location | System | Matrix Properties | Current Solver |
|:---------|:-------|:------------------|:---------------|
| `base_hjb.py` Newton step | $J \delta U = -F$ | General sparse, nonsymmetric | `spsolve` |
| `fp_fdm_time_stepping.py` implicit step | $(I/\Delta t + A + D) m = b$ | Sparse, diag-dominant, often SPD | `spsolve` |
| `implicit_heat.py` | $(I - \theta\alpha\Delta t L) u = b$ | SPD (Laplacian-based) | `splu` with `gmres` fallback |
| `nonlinear_solvers.py` Newton | $J \delta x = -F$ | General sparse or dense | `spsolve` or `np.linalg.solve` |
| `nitsche_1d.py` FEM | Stiffness system | Symmetric, banded | `spsolve` |

### 2.4 Matrix Construction Patterns

| Format | Usage | Purpose |
|:-------|:------|:--------|
| LIL (List of Lists) | Construction phase | Efficient incremental assembly |
| COO (Coordinate) | Operator export | Efficient batch construction |
| CSR (Compressed Sparse Row) | Computation | Optimal for `matvec` and `spsolve` |
| CSC (Compressed Sparse Column) | Factorization | Required by `splu`, `spilu` |

**Standard workflow**: Build in LIL or COO $\to$ convert to CSR for solve.

### 2.5 Problem Scale Analysis

| Dimension | Typical $N$ | Matrix Size | `spsolve` Time | Bottleneck? |
|:----------|:------------|:------------|:---------------|:------------|
| 1D | 51-201 | $\leq 200 \times 200$ | $< 1$ ms | No |
| 2D | $50^2 = 2{,}500$ | $2{,}500 \times 2{,}500$ | $\sim 10$ ms | No |
| 2D fine | $200^2 = 40{,}000$ | $40{,}000 \times 40{,}000$ | $\sim 1$ s | Maybe |
| 3D | $50^3 = 125{,}000$ | $125{,}000 \times 125{,}000$ | $\sim 30$ s | **Yes** |
| 3D fine | $100^3 = 10^6$ | $10^6 \times 10^6$ | Minutes | **Yes** |

**Conclusion**: For 1D-2D problems at current scales, linear solve is
NOT the bottleneck (Picard iteration overhead dominates). For 3D, it
would become critical.

---

## 3. Proposed Trait System

### 3.1 Matrix Structure Traits

```python
class MatrixStructure(Enum):
    """Algebraic structure of the system matrix."""
    SPD = "spd"                          # Symmetric Positive Definite
    SYMMETRIC = "symmetric"              # Symmetric but indefinite
    NONSYMMETRIC = "nonsymmetric"        # General nonsymmetric
    SKEW_SYMMETRIC = "skew_symmetric"   # A^T = -A

class MatrixCoupling(Enum):
    """Block structure for coupled systems."""
    SCALAR = "scalar"                    # Single PDE, single unknown
    BLOCK_DIAGONAL = "block_diagonal"    # Decoupled system (HJB and FP separate)
    BLOCK_COUPLED = "block_coupled"      # Fully coupled (monolithic MFG)
    SADDLE_POINT = "saddle_point"        # Constrained optimization structure

class MatrixOrigin(Enum):
    """Physical origin of the dominant operator (guides preconditioner)."""
    LAPLACIAN = "laplacian"              # Diffusion-dominated -> AMG works well
    ADVECTION = "advection"              # Advection-dominated -> ILU or block methods
    MIXED = "mixed"                      # Both significant -> problem-specific
    IDENTITY_SHIFTED = "identity_shifted" # I + eps*L (implicit Euler with small dt)
```

### 3.2 SolverTraits Dataclass

```python
@dataclass(frozen=True)
class LinearSolverTraits:
    """Describes a linear system's properties for auto-selection."""

    structure: MatrixStructure
    coupling: MatrixCoupling = MatrixCoupling.SCALAR
    origin: MatrixOrigin = MatrixOrigin.MIXED
    estimated_size: int = 0          # N (number of unknowns)
    estimated_nnz_ratio: float = 0.0 # nnz / N (avg nonzeros per row)

    @property
    def is_small(self) -> bool:
        """Direct solve is cheap."""
        return self.estimated_size < 10_000

    @property
    def is_large(self) -> bool:
        """Iterative solve recommended."""
        return self.estimated_size > 50_000
```

### 3.3 Auto-Selection Logic

```python
def select_solver(traits: LinearSolverTraits) -> SparseSolver:
    """Select optimal solver + preconditioner from matrix traits.

    Decision tree:
    1. Small system (N < 10k) -> direct (spsolve)
    2. SPD + Laplacian origin -> CG + AMG
    3. SPD + other origin -> CG + ILU
    4. Nonsymmetric + small -> direct
    5. Nonsymmetric + large -> GMRES + ILU
    6. Saddle point -> specialized (Uzawa, Schur complement)
    """
    if traits.is_small:
        return SparseSolver(method="direct")

    if traits.structure == MatrixStructure.SPD:
        if traits.origin == MatrixOrigin.LAPLACIAN:
            return SparseSolver(method="cg", preconditioner="amg")
        return SparseSolver(method="cg", preconditioner="ilu")

    if traits.structure in (MatrixStructure.NONSYMMETRIC, MatrixStructure.SYMMETRIC):
        return SparseSolver(method="gmres", preconditioner="ilu")

    # Fallback
    return SparseSolver(method="direct")
```

### 3.4 Operator-to-Traits Inference

Operators (MFG-SPEC-OP-0.1) could advertise their matrix properties:

```python
class LaplacianOperator(PDEOperator):
    def linear_solver_traits(self, dt: float) -> LinearSolverTraits:
        """Traits for the implicit system (I - dt * L)."""
        return LinearSolverTraits(
            structure=MatrixStructure.SPD,
            origin=MatrixOrigin.LAPLACIAN,
            estimated_size=self.shape[0],
            estimated_nnz_ratio=2 * self._ndim + 1,  # stencil width
        )
```

This enables fully automatic solver selection:

```python
# In FP implicit solver:
L = DiffusionOperator(sigma, dx) + AdvectionOperator(alpha, dx)
system = IdentityOperator(N) / dt + L

# Auto-select solver from operator traits
traits = system.linear_solver_traits(dt)
solver = select_solver(traits)
m_next = solver.solve(system.to_sparse(), m_current / dt)
```

---

## 4. MFG-Specific Linear Solver Patterns

### 4.1 HJB Newton System

| Property | Value | Reason |
|:---------|:------|:-------|
| Structure | Nonsymmetric | Hamiltonian nonlinearity in Jacobian |
| Origin | Mixed | Diffusion + nonlinear advection |
| Size | $N_x$ per time step | Spatial DOFs only |
| Recommended | Direct (`spsolve`) for $N < 10^4$, GMRES+ILU otherwise |

The Jacobian $J = I/\Delta t + \partial H/\partial U - (\sigma^2/2) L$ is
nonsymmetric due to the Hamiltonian term. For 1D-2D problems, direct solve
is fast. For 3D, GMRES with ILU preconditioning is the standard choice.

### 4.2 FP Implicit System

| Property | Value | Reason |
|:---------|:------|:-------|
| Structure | Nonsymmetric (upwind advection), near-SPD (centered) | Advection breaks symmetry |
| Origin | Mixed (advection + diffusion) | Both terms significant |
| Size | $N_x$ per time step | Spatial DOFs only |
| Recommended | Direct for $N < 10^4$, GMRES+ILU otherwise |

When using centered advection (non-default), the system approaches SPD and
CG could be used. With the default upwind scheme, GMRES is required.

### 4.3 Monolithic MFG (Future)

If a `GlobalSpacetimeSolver` (MFG-SPEC-ST-0.8 §5.2) were implemented:

| Property | Value | Reason |
|:---------|:------|:-------|
| Structure | Saddle point | Coupled HJB-FP as KKT system |
| Coupling | Block-coupled | 2x2 block structure |
| Origin | Mixed | Laplacian + advection + coupling |
| Size | $2 N_t N_x$ | Full space-time DOFs |
| Recommended | Block preconditioned GMRES | Schur complement approximation |

This is research-grade and well beyond current scope.

---

## 5. Migration Path

### Phase A: Implement Missing Preconditioners (v0.18.x) — Low effort

1. Implement Jacobi preconditioner (diagonal scaling) — declared but missing
2. Add `pyamg` as optional dependency for AMG preconditioning
3. Document when to use each preconditioner

```python
# Jacobi (trivial implementation):
if self.preconditioner == "jacobi":
    diag = A.diagonal()
    diag[diag == 0] = 1.0  # avoid division by zero
    M = spla.LinearOperator(A.shape, lambda x: x / diag)
```

### Phase B: Add LinearSolverTraits (v0.19.x) — Low effort

1. Define `LinearSolverTraits` dataclass
2. Add `linear_solver_traits()` to operators that support `to_sparse()`
3. Implement `select_solver()` auto-selection function
4. Keep manual override: `SparseSolver(method="gmres")` still works

### Phase C: Wire into Solvers (post-v1.0) — Medium effort

1. FP implicit solver: Replace hardcoded `spsolve` with auto-selected solver
2. HJB Newton: Replace `spsolve(J, -F)` with trait-guided selection
3. Add condition number monitoring (optional, for diagnostics)

### Phase D: Advanced Solvers (post-v1.0) — Large effort

1. Block preconditioning for coupled systems
2. Matrix-free iterative solvers (no `to_sparse()` needed)
3. GPU-accelerated solvers via CuPy (extend existing infrastructure)

---

## 6. Explicitly Deferred

| Item | Reason | Revisit When |
|:-----|:-------|:-------------|
| AMG preconditioner | Requires `pyamg` dependency; not needed at current scale | When 3D problems with $N > 10^5$ become standard |
| Block preconditioning | Only relevant for monolithic MFG (not implemented) | When `GlobalSpacetimeSolver` is built |
| Matrix-free GMRES | All current operators can export sparse matrices | When operators become too large to materialize |
| Multigrid (geometric) | Requires mesh hierarchy; MFG uses single-level grids | When AMR is reintroduced |
| Direct solver comparison | UMFPACK vs CHOLMOD vs MUMPS vs SuperLU | When `spsolve` becomes the bottleneck |
| Condition monitoring | No condition number issues at current scale | When convergence failures are traced to conditioning |

---

## 7. Relationship to Other Specs

| Spec | Relationship |
|:-----|:-------------|
| MFG-SPEC-OP-0.1 | Operators provide `linear_solver_traits()` and `to_sparse()` |
| MFG-SPEC-TI-0.1 | Implicit `StepOperator.step()` requires a linear solve |
| MFG-SPEC-ST-0.8 | `GlobalSpacetimeSolver` would need block preconditioned solvers |
| Issue #658 | Operator cleanup provides the `to_sparse()` infrastructure |

---

## 8. Design Decisions

### Decision 1: Auto-selection is advisory, not mandatory

`select_solver(traits)` returns a recommendation. Solvers can always
override with explicit configuration. Auto-selection is a convenience
for default configurations, not a framework requirement.

### Decision 2: SparseSolver remains the entry point

No new solver abstraction. `SparseSolver` gains an optional `traits`
parameter for auto-configuration but keeps its current manual API.

### Decision 3: Matrix format conversion stays internal

Solvers should not worry about LIL vs CSR vs CSC. The `SparseSolver.solve()`
method handles format conversion internally (CSC for factorization, CSR
for iterative methods).

---

**Last Updated**: 2026-02-05
