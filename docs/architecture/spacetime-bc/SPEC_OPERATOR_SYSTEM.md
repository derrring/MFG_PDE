# [SPEC] Operator System Architecture

**Document ID**: MFG-SPEC-OP-0.1
**Status**: DRAFT (Phase 4 of migration plan — parallel track)
**Date**: 2026-02-05
**Dependencies**: MFG-SPEC-ST-0.8, Issue #658 (Operator Library Cleanup)

---

## 1. Motivation

MFG_PDE has a mature operator library (`mfg_pde/operators/differential/`)
with 8+ concrete operators inheriting from `scipy.sparse.linalg.LinearOperator`.
Geometries advertise operator support via 4 protocol traits. This system works
but has structural gaps:

1. **No operator algebra**: Cannot write `L = Laplacian + 0.5 * Advection` as
   a composed operator. Solvers manually assemble matrices.
2. **No operator metadata**: Cannot query stencil width, conservation property,
   or upwinding scheme from an operator instance.
3. **Protocol return type ambiguity**: `get_laplacian_operator()` returns
   `LinearOperator | Callable` — solvers must handle both.
4. **BC coupling unclear**: Some operators (LaplacianOperator) handle BC
   internally via ghost cells; others (AdvectionOperator) expect pre-padded input.

This spec proposes a trait system that formalizes these patterns.

---

## 2. Current State

### 2.1 Operator Protocols (Geometry Traits)

**Location**: `mfg_pde/geometry/protocols/operators.py`

| Protocol | Method | Returns |
|:---------|:-------|:--------|
| `SupportsLaplacian` | `get_laplacian_operator(order, bc)` | `LinearOperator \| Callable` |
| `SupportsGradient` | `get_gradient_operator(direction, order, scheme)` | `LinearOperator \| Callable \| tuple` |
| `SupportsDivergence` | `get_divergence_operator(order)` | `LinearOperator \| Callable` |
| `SupportsAdvection` | `get_advection_operator(velocity, scheme, conservative)` | `LinearOperator \| Callable` |

Usage pattern:
```python
if isinstance(geometry, SupportsLaplacian):
    L = geometry.get_laplacian_operator(bc=boundary_conditions)
    result = L @ u_flat  # or L(u_shaped)
```

### 2.2 Concrete Operators

**Location**: `mfg_pde/operators/differential/`

| Operator | File | Shape | scipy LO | Key Feature |
|:---------|:-----|:------|:---------|:------------|
| `PartialDerivOperator` | `gradient.py:61` | $(N, N)$ | Yes | Schemes: central, upwind, weno5 |
| `GradientOperator` | `gradient.py:247` | $N \to N \times d$ | **No** (changes shape) | Stacks PartialDerivOperators |
| `LaplacianOperator` | `laplacian.py:40` | $(N, N)$ | Yes | `as_scipy_sparse()` for implicit |
| `DiffusionOperator` | `diffusion.py:78` | $(N, N)$ | Yes | Scalar, constant tensor, varying tensor |
| `DirectDerivOperator` | `directional.py:52` | $(N, N)$ | Yes | $v \cdot \nabla u$ (constant or varying $v$) |
| `NormalDerivOperator` | `directional.py:240` | $(N, N)$ | Yes | $\partial u / \partial n$ via `from_sdf()`, `from_axis()` |
| `DivergenceOperator` | `divergence.py:43` | $(N, dN)$ | Yes | $\nabla \cdot F$ (dimension-reducing) |
| `AdvectionOperator` | `advection.py:53` | $(N, N)$ | Yes | Gradient or divergence form, upwind/centered |

### 2.3 Stencil Layer

**Location**: `mfg_pde/operators/stencils/finite_difference.py`

Low-level functions providing the computational kernels:

```python
gradient_central(u, axis, h)           # 2nd-order central difference
gradient_forward(u, axis, h)            # 1st-order forward
gradient_backward(u, axis, h)           # 1st-order backward
gradient_upwind(u, axis, h)             # Godunov upwind selection
laplacian_with_bc(u, spacings, bc)     # Laplacian with ghost cell BC
```

### 2.4 Issue #658 Status

**Phase 0-2**: Complete. All operators migrated from `tensor_calculus` to
standalone classes. `PartialDerivOperator`, `DirectDerivOperator`,
`NormalDerivOperator` added.

**Phase 3+**: Deferred. Composite operators (Jacobian, Hessian, Curl),
nonlinear operators, spectral operators.

### 2.5 What's Missing

| Gap | Impact | Severity |
|:----|:-------|:---------|
| No operator algebra (`L1 + L2`, `alpha * L`) | Manual matrix assembly in every solver | Medium |
| No conservation metadata | Cannot auto-verify mass conservation in FP | Low |
| No stencil width metadata | Cannot auto-compute ghost layer depth | Low |
| `LinearOperator \| Callable` return ambiguity | Solvers need `isinstance` checks | Medium |
| No BC-aware composition | Composed operators lose BC information | Medium |
| GradientOperator not a LinearOperator | Cannot use in scipy iterative solvers | Low |

---

## 3. Proposed Trait System

### 3.1 Atomic Traits

```python
class DifferentialType(Enum):
    """What differential operation the operator represents."""
    PARTIAL_DERIV = "partial_deriv"      # d/dx_i
    GRADIENT = "gradient"                # grad (full)
    DIVERGENCE = "divergence"            # div
    CURL = "curl"                        # curl (3D)
    LAPLACIAN = "laplacian"              # div(grad)
    DIFFUSION = "diffusion"              # div(Sigma * grad)
    ADVECTION = "advection"              # v . grad or div(v *)
    NORMAL_DERIV = "normal_deriv"        # d/dn

class StencilWidth(Enum):
    """Spatial extent of the discrete operator."""
    COMPACT = "compact"                  # 3-point (1D), 5-point (2D)
    WIDE = "wide"                        # 5-point (1D) for 4th-order
    WENO = "weno"                        # 5-point adaptive (WENO-5)

class ConservationProperty(Enum):
    """Whether the discrete operator preserves integral invariants."""
    CONSERVATIVE = "conservative"        # Telescoping: sum of fluxes = boundary terms
    NON_CONSERVATIVE = "non_conservative" # Gradient form: may not conserve mass

class UpwindScheme(Enum):
    """Numerical flux selection for advection-type operators."""
    CENTRAL = "central"                  # Symmetric, 2nd-order, oscillatory
    UPWIND = "upwind"                    # Godunov, 1st-order, diffusive
    WENO = "weno"                        # 5th-order ENO reconstruction
    LAX_FRIEDRICHS = "lax_friedrichs"   # Global maximum wave speed
```

### 3.2 OperatorTraits Dataclass

```python
@dataclass(frozen=True)
class OperatorTraits:
    """Metadata describing a discrete operator's properties."""

    differential_type: DifferentialType
    stencil_width: StencilWidth = StencilWidth.COMPACT
    conservation: ConservationProperty = ConservationProperty.NON_CONSERVATIVE
    upwind: UpwindScheme | None = None
    formal_order: int = 2
    is_linear: bool = True

    @property
    def ghost_depth(self) -> int:
        """Minimum ghost cell layers needed."""
        return {
            StencilWidth.COMPACT: 1,
            StencilWidth.WIDE: 2,
            StencilWidth.WENO: 3,
        }[self.stencil_width]
```

### 3.3 Enhanced Base Operator

```python
class PDEOperator(LinearOperator):
    """Base class for all MFG_PDE differential operators.

    Extends scipy.sparse.linalg.LinearOperator with:
    - Operator traits (metadata)
    - Operator algebra (+, *, @)
    - Optional sparse matrix export
    - BC awareness
    """

    traits: OperatorTraits

    # --- Operator Algebra ---

    def __add__(self, other: PDEOperator) -> CompositeOperator:
        """L1 + L2: sum of operators."""
        return CompositeOperator("+", self, other)

    def __mul__(self, scalar: float) -> ScaledOperator:
        """alpha * L: scalar scaling."""
        return ScaledOperator(scalar, self)

    def __matmul__(self, other: PDEOperator) -> CompositeOperator:
        """L1 @ L2: composition (matrix product)."""
        return CompositeOperator("@", self, other)

    def __neg__(self) -> ScaledOperator:
        """-L: negation."""
        return ScaledOperator(-1.0, self)

    # --- Sparse Export ---

    def to_sparse(self) -> sparse.spmatrix | None:
        """Export as explicit sparse matrix, if supported.

        Returns None for matrix-free operators that cannot be materialized.
        """
        return None  # Override in subclasses

    # --- BC Integration ---

    @property
    def boundary_conditions(self) -> BoundaryConditions | None:
        """BC baked into this operator, if any."""
        return None
```

### 3.4 CompositeOperator

```python
class CompositeOperator(PDEOperator):
    """Lazy composition of operators.

    Supports addition (+) and matrix multiplication (@).
    Evaluates lazily: only materializes when _matvec is called.
    """

    def __init__(self, op: str, left: PDEOperator, right: PDEOperator):
        self._op = op
        self._left = left
        self._right = right
        # Shape inference
        if op == "+":
            assert left.shape == right.shape
            shape = left.shape
        elif op == "@":
            assert left.shape[1] == right.shape[0]
            shape = (left.shape[0], right.shape[1])
        super().__init__(dtype=np.float64, shape=shape)

    def _matvec(self, x):
        if self._op == "+":
            return self._left @ x + self._right @ x
        elif self._op == "@":
            return self._left @ (self._right @ x)

    def to_sparse(self) -> sparse.spmatrix | None:
        """Materialize if both children can be materialized."""
        L = self._left.to_sparse()
        R = self._right.to_sparse()
        if L is None or R is None:
            return None
        if self._op == "+":
            return L + R
        elif self._op == "@":
            return L @ R
```

### 3.5 Operator Algebra Usage Example

```python
# Current (manual matrix assembly in FP solver):
A_adv = build_advection_matrix(velocity, dx, scheme="divergence_upwind")
D_diff = build_diffusion_matrix(sigma, dx)
lhs = eye(N) / dt + A_adv + D_diff
m_next = spsolve(lhs, m_current / dt)

# Proposed (operator algebra):
A = AdvectionOperator(velocity, dx, form="divergence", scheme="upwind")
D = DiffusionOperator(sigma, dx)
L = (1/dt) * IdentityOperator(N) + A + D

# Matrix-free iterative solve:
m_next, info = gmres(L, m_current / dt)

# Or materialize for direct solve:
m_next = spsolve(L.to_sparse(), m_current / dt)
```

---

## 4. Trait Mapping to Current Operators

| Current Operator | DifferentialType | Stencil | Conservation | Upwind |
|:-----------------|:-----------------|:--------|:-------------|:-------|
| `PartialDerivOperator` | PARTIAL_DERIV | COMPACT/WENO | N/A | central/upwind/weno5 |
| `GradientOperator` | GRADIENT | COMPACT | N/A | central |
| `LaplacianOperator` | LAPLACIAN | COMPACT | CONSERVATIVE | N/A |
| `DiffusionOperator` | DIFFUSION | COMPACT | CONSERVATIVE | N/A |
| `DirectDerivOperator` | ADVECTION | COMPACT | NON_CONSERVATIVE | central/upwind |
| `NormalDerivOperator` | NORMAL_DERIV | COMPACT | N/A | one_sided |
| `DivergenceOperator` | DIVERGENCE | COMPACT | CONSERVATIVE | N/A |
| `AdvectionOperator` | ADVECTION | COMPACT | Both | upwind/centered |

---

## 5. MFG-Specific Operator Patterns

### 5.1 HJB Hamiltonian Operator

The HJB Hamiltonian $H(x, m, \nabla u)$ is **nonlinear** in $\nabla u$.
It cannot be represented as a `LinearOperator`. Two approaches:

**Option A**: Keep nonlinear Hamiltonian outside the operator system.
Solvers compute $H$ directly from gradient values. Operator system only
handles the linear terms ($\Delta u$, time derivative).

**Option B**: Define a `NonlinearOperator` protocol:
```python
class NonlinearOperator(Protocol):
    def evaluate(self, u: NDArray, **params) -> NDArray: ...
    def jacobian(self, u: NDArray, **params) -> LinearOperator: ...
```

**Recommendation**: Option A for now. The Hamiltonian is problem-specific
and tightly coupled to the MFG formulation. Abstracting it prematurely
would create unnecessary indirection.

### 5.2 FP Advection-Diffusion

The FP equation $\partial_t m + \nabla \cdot (\alpha m) = (\sigma^2/2)\Delta m$
is fully linear (given $\alpha$ from HJB). The operator algebra would allow:

```python
L_fp = DiffusionOperator(sigma**2/2, dx) - AdvectionOperator(alpha, dx, form="divergence")
```

This is the primary beneficiary of operator algebra.

### 5.3 Conservation Verification

With `ConservationProperty` metadata, the coupling iterator could verify:

```python
if fp_operator.traits.conservation == ConservationProperty.NON_CONSERVATIVE:
    warnings.warn("FP operator is non-conservative; mass may not be preserved")
```

---

## 6. Migration Path

### Phase A: Add OperatorTraits (v0.18.x) — Low effort

Add `traits` property to existing operators without changing their behavior:

```python
class LaplacianOperator(LinearOperator):
    @property
    def traits(self) -> OperatorTraits:
        return OperatorTraits(
            differential_type=DifferentialType.LAPLACIAN,
            stencil_width=StencilWidth.COMPACT,
            conservation=ConservationProperty.CONSERVATIVE,
            formal_order=self._order,
        )
```

### Phase B: Operator Algebra (v0.19.x) — Medium effort

1. Create `PDEOperator` base class extending `LinearOperator`
2. Implement `CompositeOperator` and `ScaledOperator`
3. Migrate existing operators to inherit from `PDEOperator`
4. Add `to_sparse()` to operators that support it (Laplacian, Advection already have this)

### Phase C: Solver Integration (post-v1.0) — Medium effort

1. Refactor FP implicit solver to use operator algebra instead of manual matrix assembly
2. Unify return type: geometry protocols return `PDEOperator` (not `LinearOperator | Callable`)
3. Add BC-aware composition (composed operators inherit BC from components)

### Phase D: Extended Operators (post-v1.0) — Large effort

Per Issue #658 Phase 3+:
- `JacobianOperator`, `HessianOperator`, `CurlOperator`
- Nonlinear operators (p-Laplacian, Monge-Ampere)
- Spectral operators (Fourier, Chebyshev)

---

## 7. Explicitly Deferred

| Item | Reason | Revisit When |
|:-----|:-------|:-------------|
| Lazy computation graphs | MFG operators are simple chains, not DAGs | When operator fusion needed for GPU |
| JIT compilation of kernels | Numba used ad-hoc; no systematic JIT framework | When 3D problems need 10x speedup |
| Geometry-agnostic equation code | "Same code on cube and Mobius strip" is FEniCS scope | Never (wrong project) |
| Spectral operators | FFT-based Laplacian, Chebyshev differentiation | When periodic problems dominate |
| Tensor product structure | Kronecker-based operators for structured grids | When large 3D problems arise |

---

## 8. Relationship to Other Specs

| Spec | Relationship |
|:-----|:-------------|
| MFG-SPEC-ST-0.8 | `StepOperator` consumes operators; `TrajectorySolver` drives the time loop |
| MFG-SPEC-BC-0.2 | Operators must respect BC from `BCSegment`; enforcement is operator-internal |
| MFG-SPEC-TI-0.1 | `StepOperator.step()` applies operators within a single time step |
| MFG-SPEC-LS-0.1 | Implicit operators require linear solves; trait metadata guides solver selection |
| Issue #658 | Operator cleanup is the concrete implementation path for Phases A-B |

---

**Last Updated**: 2026-02-05
