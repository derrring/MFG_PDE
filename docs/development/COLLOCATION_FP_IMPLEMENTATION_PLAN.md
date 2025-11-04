# Collocation FP Implementation Plan

**Date**: 2025-11-04
**Status**: PLANNING
**Target Release**: v1.0.0
**Priority**: MEDIUM
**Issue**: Collocation mode FP solver incomplete - density frozen at initial condition

---

## Problem Statement

### Current Status

The `FPParticleSolver` collocation mode (introduced in v0.8.0) exists but uses a simplified implementation that does NOT solve the continuity equation. Instead, density is frozen at the initial condition throughout time evolution.

**Location**: `mfg_pde/alg/numerical/fp_solvers/fp_particle.py:597-657`

**Critical Line**:
```python
# Line 652 - PROBLEM: Density never evolves!
M_solution[t_idx + 1, :] = M_solution[0, :]  # Just copies initial density
```

**TODO Comment** (line 642):
```python
# TODO: Implement proper advection on particles using drift from U_solution_for_drift
# (Full implementation would solve continuity equation on particle basis)
```

### Expected Behavior

Collocation mode should solve the **continuity equation** on fixed collocation points:

```
∂m/∂t + ∇·(m α) = σ²/2 Δm

where:
- m(t, xᵢ): density at collocation point xᵢ (Eulerian representation)
- α(t, xᵢ) = -coefCT ∇H(t, xᵢ): optimal control from Hamiltonian
- ∇·: divergence operator (computed via GFDM)
- Δ: Laplacian operator (computed via GFDM)
```

### Impact

**Current Users**: None directly affected (no production high-dimensional meshfree MFG users)

**Future Users**: Blocks adoption of:
- High-dimensional MFG (d ≥ 4) using meshfree methods
- True particle-collocation coupling with GFDM HJB solver
- Complex geometry problems requiring meshfree FP

**Tests**: Pass because they only validate API/shapes, not physical correctness

---

## Implementation Strategy

### Phase 1: Documentation & Warnings (Immediate - v0.9.1)

**Goal**: Make users aware of limitation

**Tasks**:
1. Update `_solve_fp_system_collocation()` docstring
2. Add `UserWarning` when collocation mode used
3. Update `particle_collocation_dual_mode_demo.py` with disclaimer
4. Create this implementation plan document
5. Create GitHub issue for tracking

**Estimated Time**: 2 hours
**PR**: `docs/collocation-fp-incomplete-warning`

---

### Phase 2: GFDM Spatial Operators (v1.0.0)

**Goal**: Implement divergence and Laplacian operators via GFDM

#### 2.1 Divergence Operator

**Purpose**: Compute ∇·F for vector field F on collocation points

**Mathematical Definition**:
```
∇·F = ∂F_x/∂x + ∂F_y/∂y + ∂F_z/∂z  (in 3D)
```

**Implementation**:
```python
def _compute_divergence_gfdm(
    self,
    vector_field: np.ndarray,  # (N_points, d) - vector at each point
    collocation_points: np.ndarray,  # (N_points, d)
) -> np.ndarray:  # (N_points,) - scalar divergence at each point
    """
    Compute divergence of vector field using GFDM.

    Uses same GFDM infrastructure as HJB solver:
    - Weighted least squares polynomial fitting
    - Taylor expansion coefficients
    - Derivative approximation on irregular points

    Args:
        vector_field: Vector values at collocation points
        collocation_points: Spatial positions

    Returns:
        Divergence at each collocation point
    """
    # Reuse GFDM infrastructure from hjb_gfdm.py
    # Compute ∂F_i/∂x_i for each dimension i, then sum
```

**Dependencies**: Reuse from `hjb_gfdm.py`:
- `_compute_taylor_coefficients()`
- `_construct_weight_matrix()`
- Neighbor search infrastructure

**Estimated Lines**: ~80 lines
**Estimated Time**: 4 hours

#### 2.2 Laplacian Operator

**Purpose**: Compute Δϕ for scalar field ϕ on collocation points

**Mathematical Definition**:
```
Δϕ = ∂²ϕ/∂x² + ∂²ϕ/∂y² + ∂²ϕ/∂z²  (in 3D)
```

**Implementation**:
```python
def _compute_laplacian_gfdm(
    self,
    scalar_field: np.ndarray,  # (N_points,) - scalar at each point
    collocation_points: np.ndarray,  # (N_points, d)
) -> np.ndarray:  # (N_points,) - scalar Laplacian at each point
    """
    Compute Laplacian of scalar field using GFDM.

    Uses second-order Taylor expansion coefficients
    to approximate second derivatives.

    Args:
        scalar_field: Scalar values at collocation points
        collocation_points: Spatial positions

    Returns:
        Laplacian at each collocation point
    """
    # Reuse GFDM infrastructure
    # Sum second derivatives: ∂²/∂x² + ∂²/∂y² + ...
```

**Dependencies**: Same as divergence

**Estimated Lines**: ~70 lines
**Estimated Time**: 4 hours

**Total for Phase 2**: ~150 lines, 8 hours

---

### Phase 3: Time Integration (v1.0.0)

**Goal**: Solve continuity equation forward in time

#### 3.1 Explicit Euler Integration

**Algorithm**:
```python
def _solve_fp_system_collocation(self, m_initial, U_drift):
    """
    Solve FP system in collocation mode via explicit Euler.

    Time discretization:
        m^{n+1} = m^n - Δt [∇·(m^n α^n) - σ²/2 Δm^n]

    Stability: CFL condition required (Δt ≤ Δx²/(2σ²))
    """
    M_solution = np.zeros((Nt, N_points))
    M_solution[0, :] = m_initial.copy()

    for t in range(Nt - 1):
        # 1. Compute optimal control: α = -coefCT ∇H
        grad_H = self._compute_gradient_gfdm(
            U_drift[t, :],
            self.collocation_points
        )
        alpha = -self.problem.coefCT * grad_H  # (N_points, d)

        # 2. Compute advection term: ∇·(m α)
        flux = M_solution[t, :, np.newaxis] * alpha  # (N_points, d)
        div_flux = self._compute_divergence_gfdm(
            flux,
            self.collocation_points
        )

        # 3. Compute diffusion term: σ²/2 Δm
        laplacian_m = self._compute_laplacian_gfdm(
            M_solution[t, :],
            self.collocation_points
        )
        diffusion = (self.problem.sigma**2 / 2) * laplacian_m

        # 4. Explicit Euler step
        M_solution[t+1, :] = M_solution[t, :] - Dt * (div_flux - diffusion)

        # 5. Enforce non-negativity (density constraint)
        M_solution[t+1, :] = np.maximum(M_solution[t+1, :], 0.0)

        # 6. Apply boundary conditions
        if boundary_indices is not None:
            M_solution[t+1, boundary_indices] = self._apply_boundary_conditions(
                M_solution[t+1, boundary_indices],
                boundary_type
            )

    return M_solution
```

**Estimated Lines**: ~60 lines
**Estimated Time**: 3 hours

#### 3.2 Stability Checking

**CFL Condition**:
```python
def _check_cfl_stability(self, Dt: float, collocation_points: np.ndarray) -> None:
    """
    Check CFL stability condition for explicit Euler.

    Condition: Δt ≤ C * (Δx²) / σ²
    where C ≈ 0.5 for stability, Δx = min neighbor distance
    """
    # Compute minimum neighbor distance
    min_spacing = self._compute_min_neighbor_distance(collocation_points)

    # CFL limit
    cfl_limit = 0.5 * (min_spacing**2) / (self.problem.sigma**2 + 1e-10)

    if Dt > cfl_limit:
        import warnings
        warnings.warn(
            f"CFL condition violated: Dt={Dt:.2e} > limit={cfl_limit:.2e}. "
            f"Solution may be unstable. Consider reducing timestep or using implicit method.",
            UserWarning
        )
```

**Estimated Lines**: ~30 lines
**Estimated Time**: 2 hours

**Total for Phase 3**: ~90 lines, 5 hours

---

### Phase 4: Validation & Testing (v1.0.0)

**Goal**: Ensure correctness via analytical solutions

#### 4.1 Analytical Test Cases

**Test 1: Pure Diffusion (1D)**
```python
def test_collocation_pure_diffusion_1d():
    """
    Test: ∂m/∂t = σ²/2 ∂²m/∂x² (no advection)

    Analytical solution: Gaussian spreading
    m(t,x) = (1/√(2π(σ_0² + σ²t))) exp(-(x-x_0)²/(2(σ_0² + σ²t)))
    """
    # Setup
    sigma = 0.1
    T = 1.0
    Nt = 100
    N_points = 100

    # Initial condition: Gaussian
    points = np.linspace(-3, 3, N_points).reshape(-1, 1)
    m_init = gaussian(points, mu=0, sigma=0.5)

    # Solve with collocation FP
    solver = FPParticleSolver(
        problem,
        mode='collocation',
        external_particles=points
    )
    M_numerical = solver.solve_fp_system(m_init, U_zero)

    # Analytical solution at t=T
    sigma_final = np.sqrt(0.5**2 + sigma**2 * T)
    m_analytical = gaussian(points, mu=0, sigma=sigma_final)

    # Compare
    error = np.linalg.norm(M_numerical[-1, :] - m_analytical) / np.linalg.norm(m_analytical)
    assert error < 0.05, f"Diffusion error too high: {error:.2%}"
```

**Test 2: Pure Advection (1D)**
```python
def test_collocation_pure_advection_1d():
    """
    Test: ∂m/∂t + ∂(m α)/∂x = 0 (no diffusion)

    With constant drift α, mass translates: m(t,x) = m_0(x - αt)
    """
    # Constant drift problem
    alpha_const = 0.5
    T = 1.0

    # Initial: Gaussian at x=0
    # Final: Gaussian at x=alpha*T

    # Validate translation
```

**Test 3: Advection-Diffusion (1D)**
```python
def test_collocation_advection_diffusion_1d():
    """
    Test: Full continuity equation with both terms

    Compare against FDM solver (reference)
    """
    # Use same problem for both solvers
    # Compare M_collocation vs M_fdm
```

**Test 4: Mass Conservation**
```python
def test_collocation_mass_conservation():
    """
    Verify: ∫ m(t,x) dx = ∫ m(0,x) dx for all t

    Mass must be conserved (up to numerical error)
    """
    # Compute total mass at each timestep
    # Assert: max deviation < 1%
```

**Estimated Lines**: ~300 lines (4 tests + utilities)
**Estimated Time**: 8 hours

---

### Phase 5: Advanced Features (v1.1+)

**Optional Enhancements** (post v1.0):

#### 5.1 Semi-Lagrangian Integration

**Advantages over Explicit Euler**:
- No CFL constraint (can use larger Δt)
- More stable for advection-dominated problems
- Natural for particle methods

**Algorithm Outline**:
```python
def _solve_fp_system_collocation_semilag(self, m_initial, U_drift):
    """
    Semi-Lagrangian: characteristic tracing + interpolation

    Steps:
    1. Backtrace characteristics: X(t-Δt) ← X(t) - α(X,t) Δt
    2. Interpolate density: m^n(X(t)) ← interp(m^{n-1}, X(t-Δt))
    3. Apply diffusion: m^{n+1} = m^n + Δt σ²/2 Δm^n
    """
```

**Estimated Lines**: ~150 lines
**Estimated Time**: 10 hours

#### 5.2 Implicit Time Integration

**Purpose**: Unconditional stability for diffusion

**Algorithm**: Crank-Nicolson or backward Euler

**Estimated Lines**: ~200 lines
**Estimated Time**: 12 hours

---

## Migration from Research Repository

### Research Assets to Review

**File**: `/Users/zvezda/OneDrive/code/mfg-research/experiments/maze_navigation/PURE_PARTICLE_FP_PROPOSAL.md`

**Status**: Design proposal (NOT implemented in research)

**Relevant Content**:
- Conceptual framework for pure particle FP
- GFDM integration approach
- Dimension-agnostic design philosophy

**Migration Decision**: Use as design reference, DO NOT migrate code (none exists)

### Research Code Status

**Finding**: Research repo uses production `FPParticleSolver` in collocation mode

**Evidence** (from PARTICLE_COLLOCATION_OVERLAP_ANALYSIS_2025-11-02.md):
```python
# Research solver delegates to production
from mfg_pde.alg.numerical.fp_solvers.fp_particle import FPParticleSolver

self.fp_solver = FPParticleSolver(
    problem=problem,
    mode="collocation",  # Uses production incomplete implementation!
    external_particles=collocation_points,
)
```

**Conclusion**: Research is blocked by same incomplete implementation. No separate research implementation exists to migrate.

---

## Development Workflow

### Step 1: Create Feature Branch

```bash
cd /Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE
git checkout main
git pull
git checkout -b feature/complete-collocation-fp
```

### Step 2: Immediate Documentation Updates (Phase 1)

**Files to Modify**:
1. `mfg_pde/alg/numerical/fp_solvers/fp_particle.py` (docstring + warning)
2. `examples/advanced/particle_collocation_dual_mode_demo.py` (disclaimer)
3. `docs/development/COLLOCATION_FP_IMPLEMENTATION_PLAN.md` (this file)

**Tests to Update**:
- `tests/integration/test_particle_collocation_mfg.py` (add note about incomplete)

**Commit Message**:
```
docs: Document collocation FP incomplete status

- Update _solve_fp_system_collocation docstring to reflect limitations
- Add UserWarning when collocation mode is used
- Update particle_collocation_dual_mode_demo.py with disclaimer
- Create COLLOCATION_FP_IMPLEMENTATION_PLAN.md

Addresses incomplete continuity equation implementation in collocation mode.
Density currently frozen at initial condition (line 652).
Full implementation planned for v1.0.0.
```

### Step 3: Create GitHub Issue

**Title**: Implement full continuity equation in collocation FP mode

**Labels**:
- `priority: medium`
- `area: algorithms`
- `size: medium`
- `type: enhancement`

**Body**: Link to this implementation plan

### Step 4: Implementation (Phases 2-4)

**New Feature Branch per Phase**:
```bash
# Phase 2
git checkout -b feature/collocation-fp-gfdm-operators
# Implement divergence + Laplacian
# PR → feature/complete-collocation-fp

# Phase 3
git checkout feature/complete-collocation-fp
git checkout -b feature/collocation-fp-time-integration
# Implement explicit Euler + CFL check
# PR → feature/complete-collocation-fp

# Phase 4
git checkout feature/complete-collocation-fp
git checkout -b feature/collocation-fp-validation
# Add analytical tests
# PR → feature/complete-collocation-fp
```

### Step 5: Final PR to Main

```bash
# When all phases complete
git checkout feature/complete-collocation-fp
gh pr create --title "Implement full continuity equation in collocation FP mode" \
             --body "..." \
             --base main
```

---

## Testing Strategy

### Unit Tests

**New File**: `tests/unit/test_fp_particle_collocation.py`

**Coverage**:
- GFDM divergence operator correctness
- GFDM Laplacian operator correctness
- Time integration accuracy
- CFL stability checking
- Boundary condition handling

### Integration Tests

**Existing File**: `tests/integration/test_particle_collocation_mfg.py`

**Enhancements**:
- Add analytical validation tests
- Add mass conservation checks
- Add convergence rate verification
- Compare with FDM reference solutions

### Regression Tests

**Ensure**:
- Hybrid mode unchanged (backward compatibility)
- Existing examples still work
- Performance not degraded

---

## Success Criteria

### Functional Requirements

1. ✅ Density evolves according to continuity equation
2. ✅ Mass conserved (deviation < 1%)
3. ✅ Matches analytical solutions (error < 5%)
4. ✅ Stable under CFL condition
5. ✅ Works for d=1,2,3 (extensible to higher d)

### Non-Functional Requirements

1. ✅ Performance: Similar to hybrid mode (~same complexity)
2. ✅ Backward compatibility: Existing code unaffected
3. ✅ Documentation: Comprehensive docstrings + examples
4. ✅ Tests: >90% coverage of new code

### Release Checklist

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Analytical validation tests pass
- [ ] Mass conservation verified
- [ ] CFL stability validated
- [ ] Documentation complete
- [ ] Examples updated
- [ ] CHANGELOG updated
- [ ] Version bumped to v1.0.0

---

## Timeline Estimate

### Minimal Implementation (Production-Ready)

| Phase | Tasks | Lines | Time | Cumulative |
|:------|:------|:------|:-----|:-----------|
| Phase 1 | Documentation + Warnings | 50 | 2 hrs | 2 hrs |
| Phase 2 | GFDM Operators | 150 | 8 hrs | 10 hrs |
| Phase 3 | Time Integration | 90 | 5 hrs | 15 hrs |
| Phase 4 | Validation | 300 | 8 hrs | 23 hrs |
| **Total** | **Minimal** | **~600** | **23 hrs** | **~3 days** |

### Complete Implementation (w/ Advanced Features)

| Phase | Tasks | Lines | Time | Cumulative |
|:------|:------|:------|:-----|:-----------|
| Phases 1-4 | As above | 600 | 23 hrs | 23 hrs |
| Phase 5.1 | Semi-Lagrangian | 150 | 10 hrs | 33 hrs |
| Phase 5.2 | Implicit Integration | 200 | 12 hrs | 45 hrs |
| **Total** | **Complete** | **~950** | **45 hrs** | **~6 days** |

### Recommended Approach

**v1.0.0**: Minimal implementation (Phases 1-4)
- 3 days development time
- Production-ready explicit Euler
- Full validation

**v1.1.0**: Advanced features (Phase 5)
- Additional 3 days
- Semi-Lagrangian + implicit options
- Extended validation

---

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|:-----|:------------|:-------|:-----------|
| GFDM divergence/Laplacian bugs | Medium | High | Extensive unit testing, compare with FDM |
| Stability issues (CFL violation) | Low | Medium | CFL checking, warn user, suggest implicit |
| Mass conservation errors | Low | High | Rigorous validation, analytical tests |
| Performance degradation | Low | Low | Profile, optimize if needed |

### Integration Risks

| Risk | Probability | Impact | Mitigation |
|:-----|:------------|:-------|:-----------|
| Breaking backward compatibility | Very Low | High | Regression tests, version checks |
| Research code dependency | Very Low | Low | Research uses production already |
| API changes required | Very Low | Medium | Extend, don't modify existing API |

---

## Open Questions

1. **Boundary Conditions**:
   - How to identify boundary particles automatically?
   - Should user provide boundary indices or auto-detect?
   - **Decision**: Start with user-provided, add auto-detection later

2. **Performance**:
   - Is explicit Euler fast enough for production?
   - Should we implement implicit from the start?
   - **Decision**: Start with explicit, add implicit in v1.1 if needed

3. **Dimension Support**:
   - Test d=1,2,3 initially or include d≥4?
   - **Decision**: Validate d=1,2,3; design for arbitrary d

4. **Integration with GFDM HJB**:
   - Should FP solver reuse HJB solver's GFDM infrastructure directly?
   - **Decision**: Yes, extract shared GFDM utilities to common module

---

## References

### Internal Documentation

- `docs/development/dual_mode_fp_particle_v0.8.0.md` - Original dual-mode design
- `docs/archived/2025-11/PARTICLE_COLLOCATION_OVERLAP_ANALYSIS_2025-11-02.md` - Research analysis
- `/tmp/collocation_fp_status_report.md` - Detailed status investigation

### Code References

- `mfg_pde/alg/numerical/fp_solvers/fp_particle.py:597-657` - Current incomplete implementation
- `mfg_pde/alg/numerical/hjb_solvers/hjb_gfdm.py` - GFDM infrastructure to reuse
- `examples/advanced/particle_collocation_dual_mode_demo.py` - Collocation demo

### Research References

- `mfg-research/experiments/maze_navigation/PURE_PARTICLE_FP_PROPOSAL.md` - Design proposal

---

**Document Status**: COMPLETE
**Next Actions**:
1. Create GitHub issue
2. Implement Phase 1 (documentation updates)
3. Create PR for Phase 1
4. Begin Phase 2 implementation

**Maintained By**: MFG_PDE Development Team
**Last Updated**: 2025-11-04
