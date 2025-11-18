# PDE Coefficient Implementation Roadmap

**Status**: Phase 1 Complete ✅
**Last Updated**: 2025-11-18
**Branch**: `feature/drift-strategy-pattern`

## Overview

This document tracks the implementation of flexible drift and diffusion coefficients in MFG_PDE, enabling:
1. State-dependent (nonlinear) PDEs: α(t,x,m), D(t,x,m)
2. Multi-dimensional (nD) support: 1D, 2D, 3D, arbitrary dimension
3. Advanced noise structures: Lévy processes, common noise, fractional diffusion

---

## Phase 1: Foundation (✅ COMPLETED)

### Objectives
Build type-safe, extensible base classes that can accommodate future features without breaking changes.

### Deliverables

#### 1. Unified Drift API (✅ Complete)
- **PR**: Feature drift-strategy-pattern
- **Files Modified**:
  - `mfg_pde/alg/numerical/fp_solvers/base_fp.py`
  - `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py`
  - `mfg_pde/alg/numerical/fp_solvers/fp_particle.py`
  - `mfg_pde/alg/numerical/fp_solvers/fp_network.py`

**Changes**:
```python
# BEFORE: MFG-specific parameter
def solve_fp_system(
    self,
    m_initial_condition: np.ndarray,
    U_solution_for_drift: np.ndarray,  # ❌ Too specific
)

# AFTER: General PDE parameter
def solve_fp_system(
    self,
    m_initial_condition: np.ndarray,
    drift_field: np.ndarray | Callable | None = None,  # ✅ General
)
```

**Benefits**:
- MFG case: `drift_field = U_hjb` (same as before)
- Pure diffusion: `drift_field = None`
- Custom drift: `drift_field = velocity_field`
- State-dependent: `drift_field = lambda t, x, m: -grad_potential(x)`

#### 2. Variable Diffusion API (✅ Complete)

**Files Modified**:
- `mfg_pde/alg/numerical/fp_solvers/base_fp.py` (all solvers)
- `mfg_pde/alg/numerical/hjb_solvers/base_hjb.py` (all solvers)

**Changes**:
```python
def solve_fp_system(
    ...,
    diffusion_field: float | np.ndarray | Callable | None = None,  # NEW
)

def solve_hjb_system(
    ...,
    diffusion_field: float | np.ndarray | None = None,  # NEW
)
```

**Implementation Status**:
- FP-FDM 1D: ✅ Full support for array diffusion
- FP-Particle: ✅ Constant diffusion only (Phase 2 for arrays)
- FP-Network: ✅ Variable diffusion_coefficient
- HJB-FDM 1D: ✅ Full support for array diffusion
- HJB-FDM nD: ⏳ Phase 2
- Other HJB: ⏳ Phase 2 (parameter added)

#### 3. Type Protocols (✅ Complete)

**New File**: `mfg_pde/types/pde_coefficients.py`

```python
@runtime_checkable
class DriftCallable(Protocol):
    """α(t, x, m) -> drift vector"""
    def __call__(
        self,
        t: float,
        x: NDArray[np.floating],
        m: NDArray[np.floating],
    ) -> NDArray[np.floating]: ...

@runtime_checkable
class DiffusionCallable(Protocol):
    """D(t, x, m) -> diffusion coefficient/tensor"""
    def __call__(
        self,
        t: float,
        x: NDArray[np.floating],
        m: NDArray[np.floating],
    ) -> NDArray[np.floating] | float: ...
```

**Features**:
- Runtime type checking
- Comprehensive documentation
- Dimension-agnostic design
- Grid-format agnostic

#### 4. Coupling Solver Compatibility (✅ Complete)

**Files Modified**:
- `mfg_pde/alg/numerical/coupling/fixed_point_iterator.py` (positional args, compatible)
- `mfg_pde/alg/numerical/coupling/hybrid_fp_particle_hjb_fdm.py` (updated to `drift_field`)

**Status**: MFG coupling works with new API ✅

### Summary: Phase 1 Achievements

| Feature | Status | Coverage |
|:--------|:-------|:---------|
| Drift field API | ✅ Complete | All FP/HJB solvers |
| Diffusion field API | ✅ Complete | All FP/HJB solvers |
| Type protocols | ✅ Complete | Runtime checkable |
| 1D FDM variable diffusion | ✅ Implemented | FP + HJB |
| Backward compatibility | ✅ Maintained | All existing code works |
| Documentation | ✅ Complete | Docstrings + examples |

**Commits**: 3 major commits
1. Unified drift+diffusion API in FP solvers
2. Added diffusion_field to HJB solvers
3. Type protocols for state-dependent coefficients

---

## Phase 2: State-Dependent & nD (🔄 NEXT)

### Objectives
1. Implement callable (state-dependent) coefficient evaluation
2. Complete nD support for all major solver types
3. Add anisotropic diffusion tensor support

### 2.1: Callable Evaluation in FP Solvers

**Priority**: High
**Estimated Effort**: Medium (2-3 days)

#### Implementation Plan

**File**: `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py`

Add callable evaluation in `_solve_fp_1d()`:

```python
def _solve_fp_1d(
    self,
    m_initial_condition: np.ndarray,
    drift_field: DriftField,
    show_progress: bool = True,
) -> np.ndarray:
    # Current: Only handles arrays
    if isinstance(drift_field, np.ndarray):
        effective_U = drift_field
    elif callable(drift_field):
        # NEW: Evaluate callable at each timestep
        effective_U = self._evaluate_drift_callable(drift_field, ...)

    # Rest of solver
    ...

def _evaluate_drift_callable(
    self,
    drift_func: DriftCallable,
    m_evolution: np.ndarray,  # (Nt+1, Nx)
) -> np.ndarray:
    """
    Evaluate state-dependent drift at all timesteps.

    Returns: Array of shape (Nt+1, Nx)
    """
    Nt, Nx = m_evolution.shape
    x_grid = self.problem.get_spatial_grid()  # (Nx,) for 1D

    drift_array = np.zeros((Nt, Nx))
    for t_idx in range(Nt):
        t = t_idx * self.problem.dt
        m_current = m_evolution[t_idx, :]
        drift_array[t_idx, :] = drift_func(t, x_grid, m_current)

    return drift_array
```

**Challenges**:
- Fixed-point coupling: Density `m` evolves during Picard iterations
- Solution: Re-evaluate callable at each Picard iteration
- Performance: Cache when possible, vectorize evaluation

#### Tasks

- [ ] Add `_evaluate_drift_callable()` helper to FPFDMSolver
- [ ] Add `_evaluate_diffusion_callable()` helper to FPFDMSolver
- [ ] Update `solve_fp_system()` to handle callables
- [ ] Add validation using protocols
- [ ] Write unit tests with state-dependent examples
- [ ] Update FPParticleSolver similarly
- [ ] Update FPNetworkSolver similarly

**Test Cases**:
```python
# Test 1: Porous medium equation (nonlinear diffusion)
def porous_medium_diffusion(t, x, m):
    return 0.1 * m  # D(m) = σ² m

# Test 2: Crowd avoidance (state-dependent drift)
def crowd_avoidance(t, x, m):
    grad_m = np.gradient(m, x)
    return -grad_m  # Drift away from crowds

# Test 3: Combined (nonlinear FP equation)
solver = FPFDMSolver(problem)
M = solver.solve_fp_system(
    m0,
    drift_field=crowd_avoidance,
    diffusion_field=porous_medium_diffusion
)
```

### 2.2: Callable Evaluation in HJB Solvers

**Priority**: High
**Estimated Effort**: Medium (2-3 days)

#### Implementation Plan

**File**: `mfg_pde/alg/numerical/hjb_solvers/base_hjb.py`

Update Newton solver to handle callable diffusion:

```python
def solve_hjb_system_backward(
    ...,
    diffusion_field: DiffusionField = None,
) -> np.ndarray:
    # NEW: Evaluate callable if provided
    if callable(diffusion_field):
        # Need M_density at each timestep for state-dependence
        sigma_array = self._evaluate_diffusion_callable(
            diffusion_field,
            M_density_from_prev_picard,
        )
    else:
        sigma_array = diffusion_field

    # Pass to timestep solver
    for n_idx in range(Nt - 2, -1, -1):
        sigma_at_n = sigma_array[n_idx, :] if sigma_array is not None else None
        U_new_n = solve_hjb_timestep_newton(..., sigma_at_n=sigma_at_n)
```

#### Tasks

- [ ] Add `_evaluate_diffusion_callable()` to base_hjb.py
- [ ] Update `solve_hjb_system_backward()` to handle callables
- [ ] Ensure consistency with FP callable evaluation
- [ ] Add validation for symmetric/positive-definite tensors
- [ ] Write unit tests
- [ ] Document state-dependent HJB examples

### 2.3: Complete nD Support

**Priority**: High
**Estimated Effort**: Large (1-2 weeks)

#### Current nD Status

**HJB-FDM**:
- 1D: ✅ Full support
- nD: ⏳ Basic structure exists (`_solve_hjb_nd()`), needs:
  - nD gradients/Laplacians
  - nD Hamiltonian evaluation
  - nD boundary conditions
  - Variable diffusion support

**FP-FDM**:
- 1D: ✅ Full support
- nD: ❌ Needs implementation

#### Implementation Tasks

**A. nD Differential Operators**

Create `mfg_pde/utils/numerical/operators_nd.py`:

```python
def gradient_nd(
    u: np.ndarray,  # Shape: (*spatial_shape,)
    grid_spacing: tuple[float, ...],
) -> np.ndarray:
    """
    Compute gradient ∇u in nD.

    Returns: Array of shape (*spatial_shape, d)
    """
    d = len(grid_spacing)
    grad = np.zeros((*u.shape, d))

    for dim in range(d):
        grad[..., dim] = np.gradient(u, grid_spacing[dim], axis=dim)

    return grad

def laplacian_nd(
    u: np.ndarray,  # Shape: (*spatial_shape,)
    grid_spacing: tuple[float, ...],
) -> np.ndarray:
    """
    Compute Laplacian Δu in nD.

    Returns: Array of shape (*spatial_shape,)
    """
    laplacian = np.zeros_like(u)

    for dim, dx in enumerate(grid_spacing):
        # Second derivative in dimension dim
        laplacian += np.gradient(np.gradient(u, dx, axis=dim), dx, axis=dim)

    return laplacian

def divergence_nd(
    v: np.ndarray,  # Shape: (*spatial_shape, d)
    grid_spacing: tuple[float, ...],
) -> np.ndarray:
    """
    Compute divergence ∇·v in nD.

    Returns: Array of shape (*spatial_shape,)
    """
    d = len(grid_spacing)
    div = np.zeros(v.shape[:-1])

    for dim in range(d):
        div += np.gradient(v[..., dim], grid_spacing[dim], axis=dim)

    return div
```

**B. nD HJB Implementation**

Update `HJBFDMSolver._solve_hjb_nd()`:

```python
def _solve_hjb_nd(
    self,
    M_density: NDArray,
    U_final: NDArray,
    U_prev: NDArray,
    diffusion_field: DiffusionField = None,
) -> NDArray:
    """Solve nD HJB using centralized nonlinear solvers."""

    # Import nD operators
    from mfg_pde.utils.numerical.operators_nd import gradient_nd, laplacian_nd

    # Get problem dimensions
    spatial_shape = self.shape
    d = len(spatial_shape)
    Nt = self.problem.Nt + 1

    # Allocate solution
    U_solution = np.zeros((Nt, *spatial_shape))
    U_solution[Nt - 1] = U_final.copy()

    # Backward iteration
    for n in range(Nt - 2, -1, -1):
        # Get diffusion at current time
        if diffusion_field is None:
            sigma = self.problem.sigma
        elif isinstance(diffusion_field, (int, float)):
            sigma = diffusion_field
        else:
            sigma = diffusion_field[n, ...]  # Spatially varying

        # Compute HJB residual in nD
        U_n = self._solve_hjb_timestep_nd(
            U_solution[n + 1],
            M_density[n],
            U_prev[n],
            sigma,
        )
        U_solution[n] = U_n

    return U_solution

def _solve_hjb_timestep_nd(
    self,
    U_next: NDArray,
    M_current: NDArray,
    U_prev: NDArray,
    sigma: float | NDArray,
) -> NDArray:
    """Solve single HJB timestep in nD using Newton iteration."""

    from mfg_pde.utils.numerical.operators_nd import gradient_nd, laplacian_nd

    # Newton iteration
    U_current = U_next.copy()

    for _ in range(self.max_newton_iterations):
        # Compute gradient and Laplacian
        grad_U = gradient_nd(U_current, self.grid_spacing)
        laplacian_U = laplacian_nd(U_current, self.grid_spacing)

        # HJB residual: -∂u/∂t + H(∇u, m) - σ²/2 Δu
        time_deriv = (U_current - U_next) / self.problem.dt

        # Hamiltonian evaluation (nD)
        H_values = self._evaluate_hamiltonian_nd(grad_U, M_current)

        # Diffusion term
        if isinstance(sigma, (int, float)):
            diffusion_term = -(sigma**2 / 2.0) * laplacian_U
        else:
            # Spatially varying
            diffusion_term = -(sigma**2 / 2.0) * laplacian_U

        residual = time_deriv + H_values + diffusion_term

        # Newton update (simplified - full version needs Jacobian)
        U_current -= 0.5 * residual  # Damped update

        # Check convergence
        if np.max(np.abs(residual)) < self.newton_tolerance:
            break

    return U_current
```

**C. nD FP Implementation**

Create `FPFDMSolver._solve_fp_nd()`:

```python
def _solve_fp_nd(
    self,
    m_initial_condition: NDArray,
    drift_field: DriftField,
    diffusion_field: DiffusionField,
) -> NDArray:
    """Solve nD FP equation forward in time."""

    from mfg_pde.utils.numerical.operators_nd import divergence_nd, laplacian_nd

    # Get dimensions
    spatial_shape = m_initial_condition.shape
    Nt = self.problem.Nt + 1

    # Allocate solution
    M = np.zeros((Nt, *spatial_shape))
    M[0] = m_initial_condition.copy()

    # Forward iteration
    for n in range(Nt - 1):
        t = n * self.problem.dt

        # Get drift at current time
        if drift_field is None:
            alpha = np.zeros((*spatial_shape, len(spatial_shape)))
        elif isinstance(drift_field, np.ndarray):
            alpha = drift_field[n, ...]
        elif callable(drift_field):
            x_grid = self._get_spatial_grid_nd()
            alpha = drift_field(t, x_grid, M[n])

        # Get diffusion at current time
        if diffusion_field is None:
            sigma = self.problem.sigma
        elif isinstance(diffusion_field, (int, float)):
            sigma = diffusion_field
        elif isinstance(diffusion_field, np.ndarray):
            sigma = diffusion_field[n, ...]
        elif callable(diffusion_field):
            x_grid = self._get_spatial_grid_nd()
            sigma = diffusion_field(t, x_grid, M[n])

        # FP time step: ∂m/∂t = -∇·(α m) + ∇·(D ∇m)
        # Advection term
        advection = -divergence_nd(alpha * M[n][..., None], self.grid_spacing)

        # Diffusion term
        if isinstance(sigma, (int, float)):
            diffusion = (sigma**2 / 2.0) * laplacian_nd(M[n], self.grid_spacing)
        else:
            # Spatially varying: ∇·(D ∇m)
            grad_m = gradient_nd(M[n], self.grid_spacing)
            diffusion = divergence_nd(sigma[..., None] * grad_m, self.grid_spacing)

        # Forward Euler (for simplicity - can improve with implicit schemes)
        M[n + 1] = M[n] + self.problem.dt * (advection + diffusion)

        # Enforce non-negativity and mass conservation
        M[n + 1] = np.maximum(M[n + 1], 0)
        M[n + 1] /= np.sum(M[n + 1])

    return M
```

#### Tasks

- [ ] Create `mfg_pde/utils/numerical/operators_nd.py`
- [ ] Implement `gradient_nd`, `laplacian_nd`, `divergence_nd`
- [ ] Update `HJBFDMSolver._solve_hjb_nd()` with variable diffusion
- [ ] Create `FPFDMSolver._solve_fp_nd()`
- [ ] Add nD Hamiltonian evaluation
- [ ] Add nD boundary condition handling
- [ ] Write nD test cases (2D, 3D)
- [ ] Performance benchmarks

### 2.4: Anisotropic Diffusion Tensors

**Priority**: Medium
**Estimated Effort**: Medium (3-5 days)

Support diffusion tensors D(x) instead of just scalars:

```python
# 2D anisotropic diffusion
D = np.zeros((Nx, Ny, 2, 2))
D[..., 0, 0] = 0.1  # σ_x²
D[..., 1, 1] = 0.5  # σ_y²
D[..., 0, 1] = D[..., 1, 0] = 0.02  # Cross-diffusion

M = solver.solve_fp_system(m0, diffusion_field=D)
```

**PDE formulation**:
```
∂m/∂t = ∇·(D ∇m) where D is a d×d tensor
```

#### Implementation

Update operators to handle tensors:

```python
def divergence_tensor_diffusion(
    m: NDArray,  # Density
    D: NDArray,  # Diffusion tensor (*spatial_shape, d, d)
    grid_spacing: tuple[float, ...],
) -> NDArray:
    """Compute ∇·(D ∇m) for tensor D."""
    grad_m = gradient_nd(m, grid_spacing)  # (*spatial_shape, d)

    # D @ ∇m at each point
    D_grad_m = np.einsum('...ij,...j->...i', D, grad_m)

    # ∇·(D ∇m)
    div = divergence_nd(D_grad_m, grid_spacing)

    return div
```

### Phase 2 Deliverables

- [ ] Callable drift/diffusion evaluation in all FP solvers
- [ ] Callable diffusion evaluation in all HJB solvers
- [ ] Complete nD FDM solvers (FP + HJB)
- [ ] Anisotropic diffusion tensor support
- [ ] Comprehensive test suite (unit + integration)
- [ ] Examples gallery (nonlinear PDEs, nD problems)
- [ ] Performance benchmarks
- [ ] Updated documentation

**Target Completion**: Q1 2026

---

## Phase 3: Advanced Features (🔮 FUTURE)

### 3.1: Lévy Processes (Jump-Diffusion)

Add support for non-local operators:

```python
from mfg_pde.types.noise_operators import PoissonJumpOperator

jump_op = PoissonJumpOperator(
    intensity=lambda z: 0.1 * np.exp(-z**2 / 2),
    jump_size=lambda x, z: z,
    z_grid=np.linspace(-5, 5, 100),
)

M = solver.solve_fp_system(
    m0,
    drift_field=drift,
    diffusion_field=0.1,
    jump_operator=jump_op,  # NEW
)
```

**PDE**:
```
∂m/∂t + ∇·(α m) = ∇·(D ∇m) + ∫[m(x-γ)J - mJ]dz
```

### 3.2: Common Noise (Extended MFG)

Support shared randomness across agents:

```python
from mfg_pde.types.noise_operators import CommonNoiseField

common = CommonNoiseField(
    common_volatility=0.5,
    y_dim=1,
    y_grid=np.linspace(-3, 3, 50),
)

from mfg_pde.alg.numerical.fp_solvers import FPExtendedMFGSolver

solver = FPExtendedMFGSolver(problem, common_noise=common)
M = solver.solve_fp_system(m0)  # Returns (Nt, Nx, Ny)
```

**PDE**: Master equation on extended state space (x,y)

### 3.3: Fractional Diffusion

Anomalous diffusion with fractional Laplacian:

```python
from mfg_pde.types.noise_operators import FractionalDiffusion

frac = FractionalDiffusion(
    fractional_order=0.7,  # α ∈ (0, 1)
    discretization="spectral",
)

M = solver.solve_fp_system(
    m0,
    fractional_diffusion=frac,  # NEW
)
```

**PDE**:
```
∂m/∂t = σ² (-Δ)^α m  for α ∈ (0, 1)
```

---

## Testing Strategy

### Unit Tests

**File**: `tests/unit/test_state_dependent_coefficients.py`

```python
class TestStateDependentDrift:
    """Test callable drift field evaluation."""

    def test_linear_drift_1d(self):
        """Test α(x) = -x in 1D."""
        def drift(t, x, m):
            return -x

        problem = ExampleMFGProblem(Nx=50)
        solver = FPFDMSolver(problem)
        M = solver.solve_fp_system(m0, drift_field=drift)

        assert M.shape == (51, 51)
        assert np.all(M >= 0)
        assert np.allclose(np.sum(M, axis=1), 1.0)

    def test_crowd_avoidance_drift(self):
        """Test state-dependent drift: α = -∇m."""
        def crowd_avoidance(t, x, m):
            return -np.gradient(m, x)

        # Run and verify
        ...

    def test_porous_medium_diffusion(self):
        """Test nonlinear diffusion: D(m) = m."""
        def diffusion(t, x, m):
            return 0.1 * m

        # Run and verify
        ...

class TestNDSupport:
    """Test multi-dimensional solvers."""

    def test_fp_2d_constant_coefficients(self):
        """Test 2D FP with constant drift and diffusion."""
        ...

    def test_hjb_2d_variable_diffusion(self):
        """Test 2D HJB with spatially varying σ(x,y)."""
        ...

    def test_fp_3d_anisotropic_diffusion(self):
        """Test 3D FP with diagonal diffusion tensor."""
        ...
```

### Integration Tests

**File**: `tests/integration/test_nonlinear_mfg.py`

```python
def test_mfg_with_state_dependent_coefficients():
    """Test full MFG coupling with state-dependent drift and diffusion."""

    # Define nonlinear problem
    def density_diffusion(t, x, m):
        return 0.1 * (1.0 + m)

    problem = ExampleMFGProblem()
    hjb_solver = HJBFDMSolver(problem)
    fp_solver = FPFDMSolver(problem)

    coupling_solver = FixedPointIterator(
        problem,
        hjb_solver,
        fp_solver,
    )

    result = coupling_solver.solve(max_iterations=50)

    assert result.converged
    assert result.iterations < 50
```

### Performance Benchmarks

**File**: `benchmarks/notebooks/state_dependent_performance.ipynb`

Compare:
- Array vs callable evaluation overhead
- Vectorized vs pointwise callable evaluation
- 1D vs 2D vs 3D scaling
- Sparse vs dense matrix formulations

---

## Documentation Updates

### User Guide

**File**: `docs/user_guide/advanced_pde_features.md`

Sections:
1. State-dependent coefficients
2. Multi-dimensional problems
3. Anisotropic diffusion
4. Nonlinear PDE examples
5. Performance tips

### API Reference

Update docstrings in:
- `BaseFPSolver.solve_fp_system()`
- `BaseHJBSolver.solve_hjb_system()`
- Add examples to type protocols

### Examples Gallery

Create:
- `examples/advanced/state_dependent_drift.py`
- `examples/advanced/porous_medium_equation.py`
- `examples/advanced/crowd_dynamics_2d.py`
- `examples/advanced/anisotropic_diffusion.py`

---

## Migration Guide

### For Users with Existing Code

**Breaking changes**: None! All existing code continues to work.

**New features** (opt-in):

```python
# OLD (still works)
M = solver.solve_fp_system(m0, U_hjb)

# NEW (more expressive)
M = solver.solve_fp_system(m0, drift_field=U_hjb)

# NEW (state-dependent)
M = solver.solve_fp_system(m0, drift_field=my_callable)
```

---

## Success Metrics

### Phase 2 Goals

- [ ] 90%+ test coverage for new features
- [ ] <10% performance overhead for callable evaluation
- [ ] nD solvers validated against analytical solutions
- [ ] Examples run successfully in CI
- [ ] Documentation complete and reviewed

### Performance Targets

- Callable evaluation: <2x slowdown vs arrays
- nD solvers: Scale as O(N^d) where d = dimension
- Memory: <3x overhead for nD vs 1D (per point)

---

## References

### Mathematical Background

1. **Nonlinear PDEs**: Vázquez, "The Porous Medium Equation" (2007)
2. **nD MFG**: Achdou & Capuzzo-Dolcetta, "Mean Field Games: Numerical Methods" (2010)
3. **Anisotropic Diffusion**: Weickert, "Anisotropic Diffusion in Image Processing" (1998)

### Implementation References

1. **nD Operators**: SciPy gradient/divergence implementations
2. **Sparse Solvers**: PyAMG, PETSc integration patterns
3. **Vectorization**: NumPy einsum for tensor operations

---

## Appendix: Design Decisions

### Why Type Protocols?

- **Runtime checking**: Validate user callables
- **IDE support**: Autocomplete and type hints
- **Documentation**: Clear contract for callable signatures
- **Extensibility**: Easy to add new protocol types

### Why Vectorized Evaluation?

- **Performance**: 10-100x faster than pointwise loops
- **NumPy-native**: Leverages BLAS/LAPACK
- **Maintainable**: Clean, readable code
- **Flexible**: Falls back to pointwise if needed

### Why Separate drift_field and diffusion_field?

- **Clarity**: Clear separation of advection vs diffusion
- **Flexibility**: Can vary independently
- **Physical meaning**: Matches mathematical formulation
- **Type safety**: Different return types (vector vs scalar)

### Why Not Merge with problem.sigma?

- **Separation of concerns**: Problem vs solver parameters
- **Flexibility**: Same problem, different diffusion scenarios
- **Backward compatible**: Existing code unchanged
- **Optional**: Can still use problem.sigma as default

---

**Status**: Ready for Phase 2 implementation ✅
**Next Steps**: Begin callable evaluation in FP solvers
**Questions**: Contact maintainers via GitHub issues
