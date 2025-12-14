# Phase 3.0: Tensor Diffusion Integration Design

**Status**: 🎯 IN PROGRESS
**Priority**: HIGH
**Estimated Effort**: 6-10 days
**Started**: 2025-11-19

---

## Objective

Integrate Phase 2.5 tensor diffusion operators into MFG solvers, enabling end-to-end MFG solutions with anisotropic diffusion.

**Phase 2.5 delivered**: Standalone tensor operators (`divergence_tensor_diffusion_2d()`)
**Phase 3.0 delivers**: Full MFG solver integration

---

## Tasks Breakdown

### Task 1: FP-FDM Integration (1-2 days) ⏳ IN PROGRESS

**Complexity**: Low-Medium
**Files**: `fp_fdm.py`

#### Current State

FP equation currently uses scalar Laplacian:
```python
∂m/∂t + ∇·(α m) = σ²/2 Δm
```

#### Target State

Replace with tensor operator:
```python
∂m/∂t + ∇·(α m) = ∇·(Σ ∇m)
```

where Σ can be:
- Scalar: σ² (isotropic)
- 2×2 tensor: [[σ₁₁, σ₁₂], [σ₁₂, σ₂₂]] (anisotropic)
- Spatially varying: Σ(x, y)
- State-dependent: Σ(t, x, m) (future)

#### Implementation Strategy

**Option A: Explicit Time Stepping** (RECOMMENDED)
- Simplest integration path
- Direct operator application per timestep
- No sparse matrix assembly changes
- Works for all tensor types

```python
# In _solve_timestep_full_nd() or equivalent
from mfg_pde.utils.numerical.tensor_operators import divergence_tensor_diffusion_nd

# Get diffusion tensor at this timestep
if tensor_diffusion_field is not None:
    # Evaluate Σ(t, x, m) if callable, or use array
    Sigma = evaluate_tensor_coefficient(tensor_diffusion_field, t, grid, m)

    # Apply tensor diffusion operator
    diffusion_term = divergence_tensor_diffusion_nd(
        m_current, Sigma, dx, dy, boundary_conditions
    )
else:
    # Fall back to scalar diffusion
    diffusion_term = sigma**2 * laplacian_2d(m_current, dx, dy, bc)

# Explicit update
m_next = m_current + dt * (diffusion_term - advection_term)
```

**Option B: Implicit Sparse Matrix** (DEFERRED)
- Requires assembling tensor-aware stencils
- Complex: Different stencil at each grid point for spatially-varying Σ
- Cross-terms σ₁₂ couple x and y derivatives
- Would need significant refactoring of matrix assembly code
- **Decision**: Defer to Phase 3.1 or later

#### Integration Points

1. **Add parameter**: `diffusion_tensor_field` to `solve_fp_system()`
2. **Type checking**: Detect scalar vs array vs tensor
3. **Operator dispatch**: Call `divergence_tensor_diffusion_nd()` for tensors
4. **Validation**: Use `CoefficientField.validate_tensor_psd()`
5. **Documentation**: Update docstrings with tensor examples

#### Testing Strategy

- [ ] Unit test: Isotropic tensor Σ = σ²I matches scalar diffusion
- [ ] Unit test: Diagonal tensor with different σ_x, σ_y
- [ ] Unit test: Full tensor with cross-diffusion σ₁₂
- [ ] Integration test: 2D FP solve with spatially-varying tensor
- [ ] Regression test: Ensure scalar diffusion still works

---

### Task 2: HJB-FDM Integration (1-2 days) ⏳ NOT STARTED

**Complexity**: Medium
**Files**: `hjb_fdm.py`, `base_hjb.py`

#### Current State

HJB equation with scalar diffusion:
```python
∂u/∂t + H(x, ∇u) + (σ²/2)Δu = 0
```

#### Target State

HJB with tensor diffusion:
```python
∂u/∂t + H(x, ∇u) + (1/2)∇·(Σ∇u) = 0
```

#### Key Insight

**Tensor diffusion affects the viscosity term, NOT the Hamiltonian control part**.

The Hamiltonian H(x, p) for MFG is:
```python
H = (1/2ν)|p|² - V(x) - m²
```

This does NOT change with tensor diffusion. The tensor Σ only appears in the Laplacian term.

**Exception**: If we later support optimal control under anisotropic noise, then:
```python
H_tensor = (1/2) pᵀ Σ⁻¹ p - V(x) - m²
```
But this requires matrix inversion and is a different problem (Phase 3.x advanced features).

#### Implementation Strategy

**Approach**: Replace Laplacian with tensor operator in timestep evolution

Current code structure (simplified):
```python
# In _solve_hjb_nd() or base_hjb
for n in range(Nt-1, -1, -1):  # Backward in time
    # Compute Hamiltonian
    H_values = _evaluate_hamiltonian_nd(U, M, gradients, sigma)

    # Implicit solve: U[n] from U[n+1], H, Laplacian
    # (∂u/∂t approximated by (U[n+1] - U[n])/dt)
    # Laplacian term: (σ²/2)Δu
```

**Modification**:
```python
from mfg_pde.utils.numerical.tensor_operators import divergence_tensor_diffusion_nd

for n in range(Nt-1, -1, -1):
    H_values = _evaluate_hamiltonian_nd(U, M, gradients, sigma)

    if tensor_diffusion_field is not None:
        # Evaluate Σ at this timestep
        Sigma = evaluate_tensor_coefficient(tensor_diffusion_field, t, grid, M[n])

        # Tensor viscosity term
        viscosity_term = divergence_tensor_diffusion_nd(
            U[n+1], Sigma, dx, dy, boundary_conditions
        )
    else:
        # Scalar viscosity
        viscosity_term = (sigma**2 / 2) * laplacian_2d(U[n+1], dx, dy, bc)

    # Implicit timestep (Newton or fixed-point)
    U[n] = solve_implicit_timestep(U[n+1], H_values, viscosity_term, dt)
```

#### Challenges

1. **Implicit solve complexity**: Newton solver expects specific Jacobian structure
2. **Gradient computation**: Need ∇u for Hamiltonian AND ∇·(Σ∇u) for viscosity
3. **Matrix assembly**: If using implicit matrix solve, need tensor-aware stencils

#### Recommended Approach

**Use operator splitting**:
1. Hamiltonian step (explicit): U* = U[n+1] - dt * H
2. Viscosity step (implicit): Solve (I - dt/2 * ∇·(Σ∇)) U[n] = U*

This decouples Hamiltonian from viscosity, making tensor integration cleaner.

**Alternative**: Keep existing Newton solver but modify residual function to use tensor operator.

#### Testing Strategy

- [ ] Unit test: Isotropic tensor matches scalar HJB
- [ ] Unit test: Diagonal tensor HJB convergence
- [ ] Unit test: Full tensor HJB stability
- [ ] Integration test: 2D HJB with anisotropic diffusion
- [ ] MFG coupling test: HJB + FP with tensor diffusion

---

### Task 3: MFG Coupling (1 day) ⏳ NOT STARTED

**Complexity**: Low
**Files**: `coupling/fixed_point_iterator.py`

#### Current State

`FixedPointIterator` already supports `diffusion_field` parameter (added in Phase 2.3).

Currently passes to solvers:
- Scalar: `sigma_value`
- Array: `sigma_array`
- Callable: `sigma_callable(t, x, m)`

#### Target State

Extend to support tensor diffusion:
- Scalar: `sigma` → `Σ = σ²I`
- Diagonal: `[σ_x², σ_y²]` → `Σ = diag([σ_x², σ_y²])`
- Full tensor: `Σ = [[σ₁₁, σ₁₂], [σ₁₂, σ₂₂]]`
- Callable tensor: `Σ(t, x, m)` → `2×2 array at each (t, x, m)`

#### Implementation Strategy

**Step 1**: Add `tensor_diffusion_field` parameter

```python
class FixedPointIterator:
    def __init__(
        self,
        problem,
        hjb_solver,
        fp_solver,
        diffusion_field=None,  # Existing: scalar/array/callable
        tensor_diffusion_field=None,  # NEW: tensor diffusion
        drift_field=None,
        ...
    ):
        if diffusion_field is not None and tensor_diffusion_field is not None:
            raise ValueError("Provide either diffusion_field OR tensor_diffusion_field, not both")

        self.diffusion_field = diffusion_field
        self.tensor_diffusion_field = tensor_diffusion_field
```

**Step 2**: Pass to solvers

```python
def solve(self):
    for iteration in range(self.max_iterations):
        # Solve HJB backward
        U = self.hjb_solver.solve_hjb_system(
            M_current,
            U_final,
            U_prev,
            diffusion_field=self.diffusion_field,
            tensor_diffusion_field=self.tensor_diffusion_field,  # NEW
        )

        # Solve FP forward
        M = self.fp_solver.solve_fp_system(
            m0,
            drift_field=alpha,
            diffusion_field=self.diffusion_field,
            tensor_diffusion_field=self.tensor_diffusion_field,  # NEW
        )

        # Check convergence
        ...
```

**Step 3**: Validation

Use `CoefficientField.validate_tensor_psd()` at initialization to catch invalid tensors early.

#### Testing Strategy

- [ ] Unit test: MFG with isotropic tensor (should match scalar)
- [ ] Unit test: MFG with diagonal tensor
- [ ] Integration test: Full MFG with spatially-varying tensor
- [ ] Integration test: MFG with callable tensor Σ(t, x, m)

---

### Task 4: 3D Tensor Operators (1 day) ⏳ NOT STARTED

**Complexity**: Low-Medium
**Files**: `tensor_operators.py`

#### Current State

Phase 2.5 delivered:
- ✅ 2D tensor operators fully implemented
- ✅ nD dispatcher exists
- ❌ 3D implementation raises `NotImplementedError`

```python
def _divergence_tensor_diffusion_3d(...):
    raise NotImplementedError("3D tensor diffusion not yet implemented")
```

#### Target State

Implement full 3D anisotropic diffusion:
```python
∇·(Σ ∇m) in 3D where Σ is 3×3 symmetric PSD matrix
```

#### Implementation Strategy

**Follow 2D pattern**:

1. Staggered grid with face-centered fluxes:
   - Nx+1 x-faces, Ny+1 y-faces, Nz+1 z-faces
2. Average tensor Σ to faces
3. Compute fluxes: F_x = Σ_x ∇m|_x-faces
4. Divergence: ∇·F = (F_x[i+1] - F_x[i])/dx + ...

**Code structure**:
```python
def _divergence_tensor_diffusion_3d(
    m: NDArray,  # (Nz, Ny, Nx)
    sigma_tensor: NDArray,  # (Nz, Ny, Nx, 3, 3) or (3, 3)
    dx: float,
    dy: float,
    dz: float,
    boundary_conditions: BoundaryConditions,
) -> NDArray:
    Nz, Ny, Nx = m.shape

    # Expand tensor if constant
    if sigma_tensor.ndim == 2:
        Sigma = np.tile(sigma_tensor, (Nz, Ny, Nx, 1, 1))
    else:
        Sigma = sigma_tensor

    # Apply BCs (ghost cells in all 3 directions)
    m_padded = _apply_bc_3d(m, boundary_conditions)

    # Compute gradients at faces
    dm_dx_x = (m_padded[1:-1, 1:-1, 1:] - m_padded[1:-1, 1:-1, :-1]) / dx
    dm_dy_y = (m_padded[1:-1, 1:, 1:-1] - m_padded[1:-1, :-1, 1:-1]) / dy
    dm_dz_z = (m_padded[1:, 1:-1, 1:-1] - m_padded[:-1, 1:-1, 1:-1]) / dz

    # Cross-derivatives (averaged to faces)
    # ... (6 cross-derivatives needed)

    # Average Σ to faces (3 face types)
    Sigma_x_faces = ... # (Nz, Ny, Nx+1, 3, 3)
    Sigma_y_faces = ... # (Nz, Ny+1, Nx, 3, 3)
    Sigma_z_faces = ... # (Nz+1, Ny, Nx, 3, 3)

    # Compute fluxes
    Fx = Sigma_x_faces[:, :, :, 0, 0] * dm_dx_x + \
         Sigma_x_faces[:, :, :, 0, 1] * dm_dy_x + \
         Sigma_x_faces[:, :, :, 0, 2] * dm_dz_x
    # ... similar for Fy, Fz

    # Divergence
    div_x = (Fx[:, :, 1:] - Fx[:, :, :-1]) / dx
    div_y = (Fy[:, 1:, :] - Fy[:, :-1, :]) / dy
    div_z = (Fz[1:, :, :] - Fz[:-1, :, :]) / dz

    return div_x + div_y + div_z
```

#### Testing Strategy

- [ ] Unit test: Isotropic 3D tensor matches 3D Laplacian
- [ ] Unit test: Diagonal 3D tensor
- [ ] Unit test: Full 3D tensor with cross-terms
- [ ] Unit test: 3D boundary conditions
- [ ] Unit test: 3D mass conservation

---

### Task 5: Callable Tensor Coefficients (2-3 days) ⏳ NOT STARTED

**Complexity**: Medium-High
**Files**: `pde_coefficients.py`, solvers

#### Current State

Phase 2 supports callable scalar diffusion: `D(t, x, m) → scalar`

Phase 2.5 supports constant/spatially-varying tensors: `Σ(x, y) → 2×2 array`

#### Target State

Support state-dependent tensor diffusion: `Σ(t, x, m) → d×d array`

**Example**: Anisotropy increases with crowding
```python
def crowd_tensor(t, x, m):
    """Perpendicular diffusion reduces as density increases."""
    sigma_parallel = 0.2
    sigma_perp = 0.05 * (1 - m / m_max)  # Reduces with crowding

    # Diagonal tensor in corridor frame
    return np.diag([sigma_parallel, sigma_perp])
```

#### Implementation Strategy

**Extend `CoefficientField.evaluate_at()`**:

```python
class CoefficientField:
    def evaluate_tensor_at(
        self,
        timestep_idx: int,
        grid: Grid,
        density: NDArray,
        dt: float,
    ) -> NDArray:
        """Evaluate tensor diffusion coefficient at specific state.

        Returns:
            Tensor Σ with shape:
            - (d, d) if constant
            - (Ny, Nx, d, d) if spatially varying
        """
        if self.is_callable():
            # Evaluate Σ(t, x, m) on grid
            t = timestep_idx * dt
            Sigma = np.zeros((*density.shape, 2, 2))

            for i in range(density.shape[0]):
                for j in range(density.shape[1]):
                    x = grid.coordinates[0][i]
                    y = grid.coordinates[1][j]
                    m = density[i, j]

                    Sigma[i, j] = self.field(t, np.array([x, y]), m)

            # Validate PSD at all points
            self.validate_tensor_psd(Sigma)
            return Sigma

        elif self.is_array():
            # Extract tensor from array
            if self.field.ndim == 2:
                # Constant tensor (d, d)
                return self.field
            elif self.field.ndim == 4:
                # Spatially varying (Ny, Nx, d, d)
                return self.field
            elif self.field.ndim == 5:
                # Spatiotemporal (Nt, Ny, Nx, d, d)
                return self.field[timestep_idx]

        else:
            # Scalar: Convert to isotropic tensor
            sigma_value = self.evaluate_at(timestep_idx, grid, density, dt)
            d = grid.dimension
            return sigma_value * np.eye(d)
```

#### Validation Strategy

- PSD check at every evaluation
- Shape validation: (d, d) or (Ny, Nx, d, d)
- NaN/Inf checking
- Symmetry enforcement

#### Testing Strategy

- [ ] Unit test: Callable returning constant tensor
- [ ] Unit test: Callable with density-dependent anisotropy
- [ ] Unit test: Spatiotemporal callable tensor
- [ ] Integration test: MFG with callable tensor Σ(t, x, m)
- [ ] Example: Crowd panic (anisotropy increases with density)

---

## Integration Testing Strategy

### Test Hierarchy

**Level 1: Unit Tests** (tensor operators already tested)
- ✅ Phase 2.5: 14 unit tests for tensor operators

**Level 2: Solver Unit Tests**
- FP-FDM with tensor diffusion (5 tests)
- HJB-FDM with tensor diffusion (5 tests)

**Level 3: Integration Tests**
- MFG coupling with tensor diffusion (3 tests)
- Isotropic tensor vs scalar (regression)
- Diagonal tensor convergence
- Full tensor with cross-diffusion

**Level 4: End-to-End Examples**
- Anisotropic corridor MFG (crowd dynamics)
- Traffic network with directional flow
- Biological migration with oriented movement

### Success Criteria

- [ ] All unit tests passing (solver + coupling)
- [ ] Integration tests passing
- [ ] At least 1 working end-to-end example
- [ ] Performance: <5x slowdown vs scalar diffusion
- [ ] Mass conservation maintained
- [ ] Documentation complete

---

## Risk Assessment

### Technical Risks

1. **Implicit solver complexity** (MEDIUM)
   - Current implicit timesteps use scalar Laplacian
   - Tensor diffusion requires modified Jacobian
   - **Mitigation**: Use explicit timestepping initially

2. **Performance overhead** (MEDIUM)
   - Tensor operators involve 7× more operations than scalar
   - **Mitigation**: Defer to Phase 3.4 (JIT compilation)

3. **3D implementation bugs** (LOW)
   - Following 2D pattern should be straightforward
   - **Mitigation**: Comprehensive unit tests

4. **Callable tensor validation** (MEDIUM)
   - PSD check at every grid point every timestep
   - **Mitigation**: Cache validation results when possible

### Schedule Risks

1. **Underestimated complexity** (LOW-MEDIUM)
   - Initial estimate: 6-10 days
   - Could extend if implicit solvers prove difficult
   - **Mitigation**: Start with explicit FP, defer implicit to Phase 3.x

2. **Testing bottleneck** (LOW)
   - Need comprehensive test suite
   - **Mitigation**: Test incrementally as features added

---

## Success Metrics

### Functional

- [ ] FP-FDM accepts tensor diffusion field
- [ ] HJB-FDM accepts tensor diffusion field
- [ ] MFG coupling passes tensor to both solvers
- [ ] 3D tensor operators implemented
- [ ] Callable tensor coefficients supported

### Performance

- [ ] <5x slowdown vs scalar diffusion (2D, 100×100 grid)
- [ ] <10x slowdown for 3D
- [ ] Mass conservation error < 1e-10

### Quality

- [ ] 90%+ test coverage for new code
- [ ] All examples run successfully
- [ ] Documentation complete
- [ ] No regressions in existing functionality

---

## Timeline

**Week 1** (Days 1-2):
- [x] Design document (this file)
- [ ] FP-FDM integration (explicit)
- [ ] Unit tests for FP-FDM

**Week 1** (Days 3-4):
- [ ] HJB-FDM integration
- [ ] Unit tests for HJB-FDM
- [ ] MFG coupling integration

**Week 2** (Days 5-6):
- [ ] 3D tensor operators
- [ ] 3D unit tests
- [ ] Integration tests

**Week 2** (Days 7-10):
- [ ] Callable tensor coefficients
- [ ] End-to-end examples
- [ ] Documentation and PR

---

## Next Steps

**Immediate** (Today):
1. Implement FP-FDM explicit integration
2. Write unit tests
3. Commit progress

**This Week**:
1. Complete FP-FDM and HJB-FDM integration
2. MFG coupling support
3. Basic examples working

**Next Week**:
1. 3D operators
2. Callable tensors
3. Comprehensive testing and documentation

---

**Status**: Design complete, starting implementation ✅
**Next**: FP-FDM explicit integration
