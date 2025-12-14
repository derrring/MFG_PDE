# Phase 2 Design: State-Dependent PDE Coefficients

**Status**: Design Phase
**Created**: 2025-11-19
**Branch**: `feature/drift-strategy-pattern`
**Prerequisites**: Phase 1 Complete (unified API, type protocols, HJB array diffusion)

---

## Executive Summary

Phase 2 extends MFG_PDE to support **state-dependent (nonlinear) PDEs** by enabling:
1. Array diffusion in FP solvers (spatially varying)
2. Callable coefficients evaluated at runtime (state-dependent)
3. Seamless integration with MFG fixed-point coupling

This unlocks applications like porous medium equations, crowd dynamics with avoidance, and nonlinear diffusion processes.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Phase 2.1: Array Diffusion in FP Solvers](#2-phase-21-array-diffusion-in-fp-solvers)
3. [Phase 2.2: Callable Coefficient Evaluation](#3-phase-22-callable-coefficient-evaluation)
4. [Phase 2.3: MFG Coupling Integration](#4-phase-23-mfg-coupling-integration)
5. [Testing Strategy](#5-testing-strategy)
6. [Performance Considerations](#6-performance-considerations)
7. [API Design Principles](#7-api-design-principles)
8. [Implementation Roadmap](#8-implementation-roadmap)

---

## 1. Architecture Overview

### 1.1 Current State (Phase 1)

**Type Hierarchy**:
```python
# Type aliases (mfg_pde/types/pde_coefficients.py)
DriftField = NDArray[np.floating] | DriftCallable | None
DiffusionField = float | NDArray[np.floating] | DiffusionCallable | None
```

**Solver Signatures**:
```python
# FP solvers
def solve_fp_system(
    m_initial: np.ndarray,
    drift_field: DriftField = None,
    diffusion_field: DiffusionField = None,
) -> np.ndarray:
    ...

# HJB solvers
def solve_hjb_system(
    M_density: np.ndarray,
    U_terminal: np.ndarray,
    U_prev: np.ndarray,
    diffusion_field: float | np.ndarray | None = None,
) -> np.ndarray:
    ...
```

**Current Support Matrix**:

| Solver | Scalar | Array | Callable |
|:-------|:-------|:------|:---------|
| **HJB-FDM** | ✅ | ✅ | ❌ |
| **FP-FDM** (drift) | ✅ | ✅ | ❌ |
| **FP-FDM** (diffusion) | ✅ | ❌ | ❌ |

### 1.2 Phase 2 Target State

**Support Matrix After Phase 2**:

| Solver | Scalar | Array | Callable |
|:-------|:-------|:------|:---------|
| **HJB-FDM** | ✅ | ✅ | ✅ Phase 2.2 |
| **FP-FDM** (drift) | ✅ | ✅ | ✅ Phase 2.2 |
| **FP-FDM** (diffusion) | ✅ | ✅ Phase 2.1 | ✅ Phase 2.2 |

### 1.3 Key Design Challenges

#### Challenge 1: Circular Dependency in State-Dependent Coefficients

**Problem**: Callable coefficients depend on density `m(t,x)`, but we're solving for `m(t,x)`!

```python
# Porous medium equation: D(m) = σ² m
def porous_diffusion(t, x, m):
    return 0.1 * m  # ← m is what we're solving for!

# How do we evaluate this before solving?
M = solver.solve_fp_system(m0, diffusion_field=porous_diffusion)
```

**Solution Strategies**:
1. **Bootstrap**: Use initial density `m0` as approximation for all times
2. **Iterative**: Re-evaluate callable at each timestep using computed density
3. **Picard Coupling**: For MFG, re-evaluate at each Picard iteration

#### Challenge 2: Vectorized vs Pointwise Evaluation

**Trade-off**:
- **Vectorized**: Pass entire grid, return array (fast, memory-intensive)
- **Pointwise**: Call per grid point (slower, memory-efficient)

**Design Decision**: Support vectorized by default, allow pointwise fallback.

#### Challenge 3: Backward Compatibility

**Constraint**: Existing code must work without changes.

**Solution**: Optional parameters with sensible defaults.

---

## 2. Phase 2.1: Array Diffusion in FP Solvers

**Priority**: High
**Effort**: 1 day
**Files**: `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py`

### 2.1.1 Technical Specification

#### Current Limitation

```python
# fp_fdm.py:183-188
elif isinstance(diffusion_field, np.ndarray):
    raise NotImplementedError(
        "FPFDMSolver does not yet support spatially varying diffusion_field (np.ndarray). "
        "Pass constant diffusion as float or use problem.sigma. "
        "Support for spatially varying diffusion coming in Phase 2."
    )
```

#### Design: Per-Point Diffusion Indexing

**Mathematical Formulation**:

FP equation with spatially varying diffusion:
```
∂m/∂t + ∇·(m α) = ∇·(D(t,x) ∇m)
```

In 1D with isotropic diffusion `D(t,x) = σ²(t,x)/2`:
```
∂m/∂t + ∂/∂x(m α) = (σ²(t,x)/2) ∂²m/∂x²
```

**Finite Difference Discretization**:

At grid point `i`, timestep `k`:
```
(m[k+1,i] - m[k,i])/Δt + advection_term[i] =
    σ²[k,i]/(2Δx²) * (m[k+1,i+1] - 2m[k+1,i] + m[k+1,i-1])
```

**Matrix Assembly**:
```
Diagonal:    A[i,i]   = 1/Δt + σ²[k,i]/Δx² + advection_diagonal[i]
Lower:       A[i,i-1] = -σ²[k,i]/(2Δx²) + advection_lower[i]
Upper:       A[i,i+1] = -σ²[k,i]/(2Δx²) + advection_upper[i]
```

#### Proposed Implementation

```python
def _solve_fp_1d(
    self,
    m_initial_condition: np.ndarray,
    U_solution_for_drift: np.ndarray,
    show_progress: bool = True,
) -> np.ndarray:
    """Solve 1D FP system with spatially varying diffusion support."""

    # Existing setup code...
    Nx = self.problem.Nx + 1
    Dx = self.problem.dx
    Dt = self.problem.dt
    Nt = U_solution_for_drift.shape[0]
    sigma = self.problem.sigma  # Could be scalar or array

    # NEW: Determine diffusion type ONCE before loop
    diffusion_is_constant = isinstance(sigma, (int, float))
    diffusion_is_array = isinstance(sigma, np.ndarray)

    if diffusion_is_array:
        # Validate shape
        expected_shape = (Nt, Nx)
        if sigma.shape != expected_shape:
            raise ValueError(
                f"diffusion_field array shape {sigma.shape} doesn't match "
                f"expected shape {expected_shape} (Nt={Nt}, Nx={Nx})"
            )

    # Timestep loop
    for k_idx_fp in timestep_range:
        u_at_tk = U_solution_for_drift[k_idx_fp, :]

        # NEW: Extract diffusion for current timestep
        if diffusion_is_array:
            sigma_at_tk = sigma[k_idx_fp, :]  # Shape: (Nx,)
        else:
            sigma_at_tk = sigma  # Scalar (NumPy will broadcast)

        # Build sparse matrix (per boundary condition type)
        for i in range(Nx):
            # NEW: Extract diffusion at point i
            if diffusion_is_array:
                sigma_i = sigma_at_tk[i]
            else:
                sigma_i = sigma_at_tk  # Scalar

            # Diffusion terms (modified from constant sigma)
            val_A_ii += sigma_i**2 / Dx**2
            val_A_i_im1 = -(sigma_i**2) / (2 * Dx**2)
            val_A_i_ip1 = -(sigma_i**2) / (2 * Dx**2)

            # Advection terms (unchanged)
            val_A_ii += coupling_coefficient * (
                npart(u_at_tk[ip1] - u_at_tk[i]) +
                ppart(u_at_tk[i] - u_at_tk[im1])
            ) / Dx**2
            # ... etc
```

#### Key Changes

1. **Validation**: Check array shape matches `(Nt, Nx)`
2. **Extraction**: Index into diffusion array per timestep and point
3. **Assembly**: Use per-point `sigma_i` instead of global `sigma`
4. **Backward Compatibility**: Scalar diffusion still works (NumPy broadcasting)

#### Edge Cases

| Case | Handling |
|:-----|:---------|
| `sigma = 0.1` (scalar) | Works as before (broadcast) |
| `sigma = array` (wrong shape) | Raise `ValueError` with helpful message |
| `sigma = array` (NaN/Inf) | Inherit existing checks from `has_nan_or_inf` |
| `sigma[t,x] → 0` locally | Matrix becomes singular → document as user error |

### 2.1.2 Multi-Dimensional Extension

**nD FP Solver** (`_solve_fp_nd_full_system`):

Similar approach, but diffusion varies per dimension:

```python
# For dimension d at point (i, j, k, ...)
if diffusion_is_array:
    # Array shape: (Nt, N1, N2, ..., Nd) for scalar diffusion
    #          OR: (Nt, N1, N2, ..., Nd, d) for anisotropic
    sigma_at_point = diffusion_field[k_idx, multi_idx]
else:
    sigma_at_point = diffusion_field  # Scalar
```

**Defer to Phase 2.4** (nD support) - focus on 1D first.

### 2.1.3 Testing Strategy

**Test Cases**:

1. **Constant diffusion (regression)**:
   ```python
   # Should give same results as scalar
   sigma_array = np.full((Nt, Nx), 0.1)
   M1 = solver.solve_fp_system(m0, U, diffusion_field=0.1)
   M2 = solver.solve_fp_system(m0, U, diffusion_field=sigma_array)
   assert np.allclose(M1, M2)
   ```

2. **Linear spatial variation**:
   ```python
   # σ²(x) = 0.1 + 0.2 * x  (higher diffusion on right)
   x_grid = np.linspace(0, 1, Nx)
   sigma_field = np.zeros((Nt, Nx))
   for t in range(Nt):
       sigma_field[t, :] = 0.1 + 0.2 * x_grid

   M = solver.solve_fp_system(m0, U, diffusion_field=sigma_field)
   # Expect: density spreads faster on right side
   ```

3. **Temporal variation**:
   ```python
   # σ²(t) = 0.1 * (1 + sin(2πt))  (oscillating diffusion)
   t_grid = np.linspace(0, T, Nt)
   sigma_field = np.zeros((Nt, Nx))
   for t_idx, t in enumerate(t_grid):
       sigma_field[t_idx, :] = 0.1 * (1 + np.sin(2 * np.pi * t))
   ```

4. **Mass conservation**:
   ```python
   # Verify ∫m dx = 1 at all times (even with varying σ)
   for t in range(Nt):
       mass = np.trapz(M[t, :], dx=problem.dx)
       assert np.abs(mass - 1.0) < 1e-6
   ```

5. **Shape validation**:
   ```python
   # Wrong shape → clear error
   sigma_wrong = np.full((Nt, Nx + 1), 0.1)  # Wrong Nx
   with pytest.raises(ValueError, match="doesn't match expected shape"):
       solver.solve_fp_system(m0, U, diffusion_field=sigma_wrong)
   ```

---

## 3. Phase 2.2: Callable Coefficient Evaluation

**Priority**: High
**Effort**: 2-3 days
**Files**:
- `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py`
- `mfg_pde/alg/numerical/hjb_solvers/base_hjb.py`

### 3.1 Design Philosophy

**Goal**: Enable fully nonlinear PDEs where coefficients depend on solution state.

**Examples**:
- Porous medium equation: `D(m) = D₀ m^(n-1)`
- Crowd avoidance: `α(m) = -β ∇m`
- Temperature-dependent diffusion: `D(T) = D₀ exp(-E/(kT))`

### 3.2 Evaluation Strategy

#### Strategy 1: Bootstrap (FP Solvers)

**Approach**: Use density from previous timestep to evaluate callable.

```python
def _solve_fp_1d_with_callable_diffusion(
    self,
    m_initial: np.ndarray,
    U_drift: np.ndarray,
    diffusion_func: DiffusionCallable,
) -> np.ndarray:
    """Solve FP with state-dependent diffusion."""

    Nt, Nx = U_drift.shape
    x_grid = np.linspace(self.problem.xmin, self.problem.xmax, Nx)

    m_solution = np.zeros((Nt, Nx))
    m_solution[0, :] = m_initial

    for k in range(Nt - 1):
        # Evaluate diffusion at CURRENT density state
        t_current = k * self.problem.dt
        m_current = m_solution[k, :]

        # Call user function
        sigma_array = diffusion_func(t_current, x_grid, m_current)

        # Validate output
        if not isinstance(sigma_array, (float, np.ndarray)):
            raise TypeError(
                f"diffusion_field callable returned {type(sigma_array)}, "
                f"expected float or np.ndarray"
            )

        # Ensure correct shape
        if isinstance(sigma_array, np.ndarray):
            if sigma_array.shape != (Nx,):
                raise ValueError(
                    f"diffusion_field callable returned shape {sigma_array.shape}, "
                    f"expected ({Nx},) for 1D problem"
                )

        # Solve one timestep with this diffusion field
        m_next = self._solve_single_timestep_fp(
            m_current,
            U_drift[k, :],
            sigma_array,
            self.problem.dt,
            self.problem.dx,
        )

        m_solution[k + 1, :] = m_next

    return m_solution
```

**Pros**:
- Simple: evaluate callable once per timestep
- Causal: uses only past information (stable)

**Cons**:
- Bootstrap error: first timestep uses `m[0]` (may be inaccurate)
- No self-consistency: `m[k+1]` computed with `D(m[k])`, not `D(m[k+1])`

#### Strategy 2: Implicit Self-Consistency (Advanced)

**Approach**: Iterate to find self-consistent `(m, D)` at each timestep.

```python
def _solve_timestep_with_implicit_diffusion(
    m_current: np.ndarray,
    diffusion_func: DiffusionCallable,
    max_inner_iter: int = 10,
    tol: float = 1e-6,
) -> np.ndarray:
    """Solve timestep with implicit treatment of state-dependent D(m)."""

    # Initial guess: m_next ≈ m_current
    m_next_guess = m_current.copy()

    for inner_iter in range(max_inner_iter):
        # Evaluate D using current guess
        D_at_guess = diffusion_func(t, x_grid, m_next_guess)

        # Solve linear system with this D
        m_next_new = solve_linear_fp_step(m_current, D_at_guess, ...)

        # Check convergence
        error = np.linalg.norm(m_next_new - m_next_guess)
        if error < tol:
            return m_next_new

        m_next_guess = m_next_new

    # Failed to converge
    warnings.warn(f"Implicit diffusion iteration did not converge")
    return m_next_guess
```

**Pros**:
- Fully self-consistent
- Accurate for strongly nonlinear problems

**Cons**:
- Expensive: inner iteration loop per timestep
- Convergence not guaranteed (nonlinear solver issues)

**Phase 2 Decision**: Implement Strategy 1 (Bootstrap), document Strategy 2 for Phase 3.

### 3.3 Proposed Implementation: FP Solvers

#### Refactor: Extract Single-Timestep Solver

**Motivation**: Enable both array and callable diffusion to share core logic.

```python
# New method in FPFDMSolver
def _solve_single_timestep_fp(
    self,
    m_current: np.ndarray,
    u_current: np.ndarray,
    sigma_at_tk: float | np.ndarray,
    dt: float,
    dx: float,
) -> np.ndarray:
    """
    Solve single FP timestep: m[k] → m[k+1].

    Args:
        m_current: Density at time k, shape (Nx,)
        u_current: Value function at time k, shape (Nx,)
        sigma_at_tk: Diffusion at time k, scalar or shape (Nx,)
        dt: Time step
        dx: Spatial step

    Returns:
        m_next: Density at time k+1, shape (Nx,)
    """
    Nx = len(m_current)

    # Build sparse matrix (existing code, extracted)
    row_indices = []
    col_indices = []
    data_values = []

    for i in range(Nx):
        # Extract diffusion at point i
        if isinstance(sigma_at_tk, np.ndarray):
            sigma_i = sigma_at_tk[i]
        else:
            sigma_i = sigma_at_tk

        # Assemble matrix entries (existing logic)
        # ... (diffusion terms + advection terms) ...

    A_matrix = sparse.coo_matrix((data_values, (row_indices, col_indices)), shape=(Nx, Nx)).tocsr()
    b_rhs = m_current / dt

    m_next = sparse.linalg.spsolve(A_matrix, b_rhs)
    return m_next
```

#### Main Solver with Callable Support

```python
def solve_fp_system(
    self,
    m_initial_condition: np.ndarray,
    drift_field: DriftField = None,
    diffusion_field: DiffusionField = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Solve FP system with array or callable coefficients."""

    # 1. Handle drift_field (existing code)
    if drift_field is None:
        U_drift = np.zeros((Nt, Nx))
    elif isinstance(drift_field, np.ndarray):
        U_drift = drift_field
    elif callable(drift_field):
        # NEW: Evaluate callable drift
        U_drift = self._evaluate_drift_callable(drift_field, m_initial_condition)
    else:
        raise TypeError(f"drift_field must be None, array, or callable")

    # 2. Handle diffusion_field
    if diffusion_field is None:
        sigma_field = self.problem.sigma
        diffusion_mode = "constant"
    elif isinstance(diffusion_field, (int, float)):
        sigma_field = float(diffusion_field)
        diffusion_mode = "constant"
    elif isinstance(diffusion_field, np.ndarray):
        sigma_field = diffusion_field
        diffusion_mode = "array"
    elif callable(diffusion_field):
        # NEW: Defer evaluation to timestep loop
        sigma_field = diffusion_field
        diffusion_mode = "callable"
    else:
        raise TypeError(f"diffusion_field must be None, float, array, or callable")

    # 3. Route to appropriate solver
    if diffusion_mode == "callable":
        return self._solve_fp_1d_with_callable(
            m_initial_condition, U_drift, sigma_field, show_progress
        )
    else:
        # Existing path (handles constant and array)
        return self._solve_fp_1d(m_initial_condition, U_drift, show_progress)

def _solve_fp_1d_with_callable(
    self,
    m_initial: np.ndarray,
    U_drift: np.ndarray,
    diffusion_func: DiffusionCallable,
    show_progress: bool,
) -> np.ndarray:
    """Solve FP with callable diffusion (bootstrap strategy)."""

    Nt, Nx = U_drift.shape
    x_grid = np.linspace(self.problem.xmin, self.problem.xmax, Nx)

    m_solution = np.zeros((Nt, Nx))
    m_solution[0, :] = m_initial

    # Progress bar
    timestep_range = range(Nt - 1)
    if show_progress:
        from mfg_pde.utils.progress import tqdm
        timestep_range = tqdm(timestep_range, desc="FP (callable σ)", unit="step")

    for k in timestep_range:
        # Evaluate diffusion at current state
        t_current = k * self.problem.dt
        m_current = m_solution[k, :]

        sigma_array = diffusion_func(t_current, x_grid, m_current)

        # Validate callable output
        sigma_array = self._validate_callable_output(
            sigma_array, expected_shape=(Nx,), param_name="diffusion_field"
        )

        # Solve single timestep
        m_solution[k + 1, :] = self._solve_single_timestep_fp(
            m_current,
            U_drift[k, :],
            sigma_array,
            self.problem.dt,
            self.problem.dx,
        )

    return m_solution

def _validate_callable_output(
    self,
    output: Any,
    expected_shape: tuple,
    param_name: str,
) -> np.ndarray:
    """Validate output from callable coefficient function."""

    if isinstance(output, (int, float)):
        # Scalar → broadcast to array
        output = np.full(expected_shape, float(output))
    elif isinstance(output, np.ndarray):
        if output.shape != expected_shape:
            raise ValueError(
                f"{param_name} callable returned array with shape {output.shape}, "
                f"expected {expected_shape}"
            )
    else:
        raise TypeError(
            f"{param_name} callable returned {type(output)}, "
            f"expected float or np.ndarray"
        )

    # Check for NaN/Inf
    if np.any(np.isnan(output)) or np.any(np.isinf(output)):
        raise ValueError(f"{param_name} callable returned NaN or Inf values")

    return output
```

### 3.4 Proposed Implementation: HJB Solvers

**Challenge**: HJB solves backward in time, so callable diffusion cannot depend on `m` from same iteration.

**Solution**: Use `M_density_from_prev_picard` (from FP solve in previous Picard iteration).

```python
# In solve_hjb_system_backward():
def solve_hjb_system_backward(
    M_density_from_prev_picard: np.ndarray,
    U_final_condition_at_T: np.ndarray,
    U_from_prev_picard: np.ndarray,
    problem: MFGProblem,
    diffusion_field: float | np.ndarray | DiffusionCallable | None = None,
    ...
) -> np.ndarray:
    """Solve HJB backward with callable diffusion support."""

    Nt, Nx = M_density_from_prev_picard.shape

    # NEW: Handle callable diffusion
    if callable(diffusion_field):
        # Evaluate using M from previous Picard iteration
        x_grid = np.linspace(problem.xmin, problem.xmax, Nx)
        diffusion_array = np.zeros((Nt, Nx))

        for t_idx in range(Nt):
            t = t_idx * problem.dt
            m_at_t = M_density_from_prev_picard[t_idx, :]
            diffusion_array[t_idx, :] = diffusion_field(t, x_grid, m_at_t)

        effective_diffusion = diffusion_array
    else:
        effective_diffusion = diffusion_field

    # Rest of HJB solve (extract per-timestep as before)
    for n_idx in range(Nt - 2, -1, -1):
        if effective_diffusion is None:
            sigma_at_n = None
        elif isinstance(effective_diffusion, (int, float)):
            sigma_at_n = effective_diffusion
        else:
            sigma_at_n = effective_diffusion[n_idx, :]

        # Pass to Newton solver
        U_new_n = solve_hjb_timestep_newton(..., sigma_at_n=sigma_at_n)
```

### 3.5 Testing Strategy: Callable Coefficients

**Test Cases**:

1. **Callable returning constant (sanity check)**:
   ```python
   def constant_callable(t, x, m):
       return 0.1

   M1 = solver.solve_fp_system(m0, U, diffusion_field=0.1)
   M2 = solver.solve_fp_system(m0, U, diffusion_field=constant_callable)
   assert np.allclose(M1, M2)
   ```

2. **Position-dependent (compare to array)**:
   ```python
   def position_callable(t, x, m):
       return 0.1 + 0.2 * x

   # Precompute equivalent array
   x_grid = np.linspace(0, 1, Nx)
   sigma_array = np.zeros((Nt, Nx))
   for t in range(Nt):
       sigma_array[t, :] = 0.1 + 0.2 * x_grid

   M1 = solver.solve_fp_system(m0, U, diffusion_field=position_callable)
   M2 = solver.solve_fp_system(m0, U, diffusion_field=sigma_array)
   assert np.allclose(M1, M2, rtol=1e-5)
   ```

3. **State-dependent: Porous medium**:
   ```python
   def porous_medium(t, x, m):
       """D(m) = D₀ m  (linear in density)"""
       return 0.1 * m

   M = solver.solve_fp_system(m0, U, diffusion_field=porous_medium)

   # Verify: High-density regions should diffuse faster
   # Compare diffusion rate at peak vs tails
   ```

4. **Time-dependent**:
   ```python
   def oscillating(t, x, m):
       return 0.1 * (1 + 0.5 * np.sin(2 * np.pi * t))

   M = solver.solve_fp_system(m0, U, diffusion_field=oscillating)
   ```

5. **Invalid outputs (error handling)**:
   ```python
   def bad_shape(t, x, m):
       return np.ones(Nx + 1)  # Wrong shape!

   with pytest.raises(ValueError, match="expected.*shape"):
       solver.solve_fp_system(m0, U, diffusion_field=bad_shape)

   def returns_nan(t, x, m):
       return np.nan

   with pytest.raises(ValueError, match="NaN"):
       solver.solve_fp_system(m0, U, diffusion_field=returns_nan)
   ```

---

## 4. Phase 2.3: MFG Coupling Integration

**Priority**: High
**Effort**: 1 day
**Files**: `mfg_pde/alg/numerical/coupling/fixed_point_iterator.py`

### 4.1 Current Coupling Architecture

```python
# Simplified current flow:
for picard_iter in range(max_iterations):
    # 1. Solve HJB backward: M_old → U_new
    U_new = hjb_solver.solve_hjb_system(M_old, U_terminal, U_old)

    # 2. Solve FP forward: U_new → M_new
    M_new = fp_solver.solve_fp_system(m_initial, U_new)

    # 3. Damping and convergence check
    U = damping * U_new + (1 - damping) * U_old
    M = damping * M_new + (1 - damping) * M_old
```

**Issue**: User-provided callable coefficients need to be passed through and re-evaluated each Picard iteration.

### 4.2 Design: Coefficient Pass-Through

**Goal**: Allow users to specify callable coefficients at MFG solver level.

```python
class FixedPointIterator(BaseMFGSolver):
    def __init__(
        self,
        problem: MFGProblem,
        hjb_solver: BaseHJBSolver,
        fp_solver: BaseFPSolver,
        config: MFGSolverConfig | None = None,
        # NEW: Optional coefficient overrides
        drift_field: DriftField = None,
        diffusion_field: DiffusionField = None,
        **kwargs,
    ):
        super().__init__(problem)
        self.hjb_solver = hjb_solver
        self.fp_solver = fp_solver
        self.config = config

        # NEW: Store coefficient specifications
        self.drift_field_override = drift_field
        self.diffusion_field_override = diffusion_field

    def solve(self, ...) -> SolverResult:
        """Solve MFG system with callable coefficient support."""

        # Initialization (existing code)
        self.U = initialize_U(...)
        self.M = initialize_M(...)

        for picard_iter in range(max_iterations):
            U_old = self.U.copy()
            M_old = self.M.copy()

            # 1. Solve HJB backward
            # NEW: Pass diffusion_field (re-evaluated with M_old if callable)
            if callable(self.diffusion_field_override):
                # Re-evaluate callable with current density estimate
                diffusion_for_hjb = self._evaluate_diffusion_for_hjb(
                    self.diffusion_field_override, M_old
                )
            else:
                diffusion_for_hjb = self.diffusion_field_override

            U_new = self.hjb_solver.solve_hjb_system(
                M_old,
                final_u_cost,
                U_old,
                diffusion_field=diffusion_for_hjb,
            )

            # 2. Solve FP forward
            # NEW: Pass both drift and diffusion overrides
            # Drift: use U_new (standard MFG) OR user override
            if self.drift_field_override is not None:
                drift_for_fp = self.drift_field_override
            else:
                drift_for_fp = U_new  # Standard MFG coupling

            # Diffusion: use override if provided
            if callable(self.diffusion_field_override):
                # FP solver will re-evaluate callable at each timestep
                diffusion_for_fp = self.diffusion_field_override
            else:
                diffusion_for_fp = self.diffusion_field_override

            M_new = self.fp_solver.solve_fp_system(
                initial_m_dist,
                drift_field=drift_for_fp,
                diffusion_field=diffusion_for_fp,
            )

            # 3. Damping and convergence (existing code)
            ...
```

### 4.3 Helper: Evaluate Diffusion for HJB

```python
def _evaluate_diffusion_for_hjb(
    self,
    diffusion_func: DiffusionCallable,
    M_current: np.ndarray,
) -> np.ndarray:
    """
    Evaluate callable diffusion for HJB solver.

    Args:
        diffusion_func: User-provided callable D(t, x, m)
        M_current: Current density estimate from previous Picard iteration

    Returns:
        Diffusion array of shape (Nt, Nx) for HJB solver
    """
    Nt, Nx = M_current.shape
    x_grid = self._get_spatial_grid()

    diffusion_array = np.zeros((Nt, Nx))

    for t_idx in range(Nt):
        t = t_idx * self.problem.dt
        m_at_t = M_current[t_idx, :]
        diffusion_array[t_idx, :] = diffusion_func(t, x_grid, m_at_t)

    return diffusion_array

def _get_spatial_grid(self) -> np.ndarray:
    """Get spatial grid for coefficient evaluation."""
    if hasattr(self.problem, "Nx"):
        # 1D interface
        return np.linspace(self.problem.xmin, self.problem.xmax, self.problem.Nx + 1)
    elif hasattr(self.problem, "geometry"):
        # nD geometry interface
        return self.problem.geometry.get_spatial_grid()
    else:
        raise ValueError("Cannot infer spatial grid from problem")
```

### 4.4 Usage Example

```python
from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import create_mfg_solver

# Define state-dependent diffusion
def crowd_diffusion(t, x, m):
    """Diffusion increases in crowded areas."""
    return 0.1 * (1 + 2 * m)  # D = σ²(1 + 2m)

# Create MFG solver with callable diffusion
problem = ExampleMFGProblem(Nx=100, Nt=50)
solver = create_mfg_solver(
    problem,
    hjb_method="fdm",
    fp_method="fdm",
    diffusion_field=crowd_diffusion,  # ← Applied to both HJB and FP
)

result = solver.solve()
print(f"Converged in {result.iterations} iterations")
```

### 4.5 Testing: MFG with Callable Coefficients

**Test Cases**:

1. **Convergence with callable diffusion**:
   ```python
   def state_diffusion(t, x, m):
       return 0.1 * (1 + m)

   solver = FixedPointIterator(
       problem, hjb_solver, fp_solver,
       diffusion_field=state_diffusion
   )
   result = solver.solve()
   assert result.converged
   ```

2. **Compare to constant (special case)**:
   ```python
   def constant(t, x, m):
       return 0.1

   result1 = solver.solve(diffusion_field=0.1)
   result2 = solver.solve(diffusion_field=constant)

   # Should give very similar results
   assert np.allclose(result1.U, result2.U, rtol=1e-4)
   ```

3. **Custom drift (non-MFG)**:
   ```python
   def wind_field(t, x, m):
       return 0.5 * np.ones_like(x)  # Constant drift

   solver = FixedPointIterator(
       problem, hjb_solver, fp_solver,
       drift_field=wind_field,  # Override MFG drift
   )
   ```

---

## 5. Testing Strategy

### 5.1 Test Hierarchy

```
tests/unit/test_phase2_coefficients/
├── test_array_diffusion_fp.py          # Phase 2.1
├── test_callable_diffusion_fp.py       # Phase 2.2
├── test_callable_drift_fp.py           # Phase 2.2
├── test_callable_diffusion_hjb.py      # Phase 2.2
├── test_mfg_coupling_callable.py       # Phase 2.3
└── test_coefficient_validation.py      # Error handling

tests/integration/
├── test_porous_medium.py               # Nonlinear diffusion
├── test_crowd_avoidance.py             # State-dependent drift
└── test_phase2_convergence.py          # Numerical accuracy
```

### 5.2 Validation Test Matrix

| Test Type | Scalar | Array | Callable | Purpose |
|:----------|:-------|:------|:---------|:--------|
| **Type checking** | ✅ | ✅ | ✅ | Verify type protocols |
| **Shape validation** | N/A | ✅ | ✅ | Catch dimension errors |
| **NaN/Inf detection** | ✅ | ✅ | ✅ | Numerical stability |
| **Mass conservation** | ✅ | ✅ | ✅ | Physical validity |
| **Regression** | ✅ | ✅ | ✅ | Backward compatibility |

### 5.3 Benchmark Suite

**Analytical Solutions for Validation**:

1. **Pure diffusion (Gaussian)**:
   ```
   ∂m/∂t = σ²/2 ∂²m/∂x²
   Analytical: m(t,x) = (1/√(2πσ²t)) exp(-(x-x₀)²/(2σ²t))
   ```

2. **Linear advection-diffusion**:
   ```
   ∂m/∂t + v ∂m/∂x = D ∂²m/∂x²
   Analytical: Traveling Gaussian
   ```

3. **Porous medium (Barenblatt solution)**:
   ```
   ∂m/∂t = ∂²/∂x²(m²)
   Analytical: Self-similar solution with compact support
   ```

### 5.4 Performance Benchmarks

**Metrics**:
- Callable overhead vs array (expect 10-20% slowdown)
- Picard iteration count with state-dependent coefficients
- Memory usage with large grids

---

## 6. Performance Considerations

### 6.1 Computational Complexity

**Array Diffusion** (Phase 2.1):
- Time: Same as constant (`O(Nt * Nx * nnz)` for sparse solve)
- Memory: Additional `O(Nt * Nx)` for diffusion array
- **Impact**: Negligible (< 5% overhead)

**Callable Diffusion** (Phase 2.2):
- Time: `O(Nt)` callable evaluations per Picard iteration
  - Each evaluation: `O(Nx)` (vectorized) or `O(Nx²)` (pointwise)
- Memory: Same as array
- **Impact**: 10-20% overhead per iteration

### 6.2 Optimization Strategies

#### 1. Vectorization
```python
# ✅ GOOD: Vectorized (fast)
def diffusion_vec(t, x, m):
    return 0.1 * (1 + m)  # Element-wise operations

# ❌ BAD: Python loops (slow)
def diffusion_loop(t, x, m):
    result = np.zeros_like(m)
    for i in range(len(m)):
        result[i] = 0.1 * (1 + m[i])  # Loop in Python
    return result
```

#### 2. Caching (Future Enhancement)
```python
# Phase 3: Cache callable results if deterministic
@functools.lru_cache(maxsize=128)
def cached_diffusion(t_rounded, x_hash, m_hash):
    return expensive_calculation(t, x, m)
```

#### 3. JIT Compilation (Future Enhancement)
```python
# Phase 3: Numba/JAX acceleration
from numba import jit

@jit(nopython=True)
def fast_diffusion(t, x, m):
    return 0.1 * np.exp(-m)  # Compiled to machine code
```

### 6.3 Profiling Checkpoints

**Add timing instrumentation**:
```python
import time

def solve_fp_system(...):
    times = {
        "callable_eval": 0.0,
        "matrix_assembly": 0.0,
        "linear_solve": 0.0,
    }

    for k in range(Nt - 1):
        # Time callable evaluation
        t0 = time.time()
        sigma = diffusion_func(t, x, m)
        times["callable_eval"] += time.time() - t0

        # ... rest of solver ...

    if verbose:
        print(f"Callable eval: {times['callable_eval']:.3f}s")
```

---

## 7. API Design Principles

### 7.1 Principle 1: Progressive Disclosure

**Simple cases should be simple**:
```python
# Beginner: Constant diffusion
M = solver.solve_fp_system(m0, U, diffusion_field=0.1)

# Intermediate: Array diffusion
M = solver.solve_fp_system(m0, U, diffusion_field=sigma_array)

# Advanced: Callable diffusion
M = solver.solve_fp_system(m0, U, diffusion_field=my_func)
```

### 7.2 Principle 2: Type Safety

**Use protocols for runtime validation**:
```python
from mfg_pde.types import DiffusionCallable

def my_diffusion(t, x, m):
    return 0.1 * m

# Runtime check
assert isinstance(my_diffusion, DiffusionCallable)
```

### 7.3 Principle 3: Fail Fast

**Validate inputs immediately**:
```python
def solve_fp_system(..., diffusion_field):
    # Check type first
    if not isinstance(diffusion_field, (type(None), float, np.ndarray, DiffusionCallable)):
        raise TypeError(f"Invalid diffusion_field type: {type(diffusion_field)}")

    # Check shape (for arrays)
    if isinstance(diffusion_field, np.ndarray):
        expected_shape = (self.problem.Nt + 1, self.problem.Nx + 1)
        if diffusion_field.shape != expected_shape:
            raise ValueError(f"Shape mismatch: got {diffusion_field.shape}, expected {expected_shape}")
```

### 7.4 Principle 4: Helpful Errors

**Provide actionable error messages**:
```python
# ❌ BAD
raise ValueError("Invalid shape")

# ✅ GOOD
raise ValueError(
    f"diffusion_field array shape {diffusion_field.shape} doesn't match "
    f"expected shape {expected_shape} for problem with Nt={Nt}, Nx={Nx}. "
    f"Hint: diffusion_field should have shape (Nt+1, Nx+1) = ({Nt+1}, {Nx+1})"
)
```

---

## 8. Implementation Roadmap

### Timeline

| Phase | Tasks | Effort | Dependencies |
|:------|:------|:-------|:-------------|
| **2.1** | Array diffusion in FP-FDM | 1 day | Phase 1 complete |
| **2.2a** | Callable diffusion in FP-FDM | 1 day | Phase 2.1 |
| **2.2b** | Callable drift in FP-FDM | 0.5 day | Phase 2.2a |
| **2.2c** | Callable diffusion in HJB | 0.5 day | Phase 2.2a |
| **2.3** | MFG coupling integration | 1 day | Phase 2.2 |
| **Testing** | Comprehensive test suite | 1 day | All phases |
| **Documentation** | Examples and tutorials | 0.5 day | All phases |

**Total**: 5-6 days for full Phase 2 implementation.

### Implementation Order

1. ✅ **Phase 2.1: Array Diffusion in FP**
   - Lowest risk, enables next phases
   - Clear testing strategy
   - Files: `fp_fdm.py` only

2. **Phase 2.2a: Callable Diffusion in FP**
   - Refactor `_solve_single_timestep_fp` (enables reuse)
   - Implement bootstrap strategy
   - Comprehensive validation

3. **Phase 2.2b: Callable Drift in FP**
   - Similar to 2.2a, but simpler (drift less coupled)
   - Reuse validation infrastructure

4. **Phase 2.2c: Callable Diffusion in HJB**
   - Uses array from callable evaluation
   - Leverage existing array support

5. **Phase 2.3: MFG Coupling**
   - Integrate all pieces
   - Pass coefficients through coupling layer
   - Test full MFG with state-dependent coefficients

### Success Criteria

**Phase 2.1**:
- [ ] FP-FDM accepts array diffusion_field
- [ ] Mass conservation maintained
- [ ] Tests pass with spatially varying diffusion
- [ ] Performance within 5% of constant diffusion

**Phase 2.2**:
- [ ] FP-FDM accepts callable diffusion_field
- [ ] FP-FDM accepts callable drift_field
- [ ] HJB accepts callable diffusion_field
- [ ] Porous medium test case works
- [ ] Performance within 20% of array version

**Phase 2.3**:
- [ ] MFG solver passes callable coefficients to HJB/FP
- [ ] Fixed-point iteration converges with state-dependent coefficients
- [ ] Full integration test (MFG with porous medium) passes

---

## Appendix A: Mathematical Formulations

### A.1 FP Equation with State-Dependent Coefficients

**General Form**:
```
∂m/∂t + ∇·(α(t,x,m) m) = ∇·(D(t,x,m) ∇m)
```

**1D Discretization** (implicit Euler):
```
(m[k+1,i] - m[k,i])/Δt +
    (α[k,i] m[k+1,i])[i+1/2] - (α[k,i] m[k+1,i])[i-1/2] / Δx =
    (D[k,i] ∂m/∂x[k+1])[i+1/2] - (D[k,i] ∂m/∂x[k+1])[i-1/2] / Δx
```

**Upwind Discretization** (for advection):
```
α[i+1/2] ≈ (α[i+1] + α[i])/2
Use upwind: α⁺ = max(α, 0), α⁻ = min(α, 0)
```

### A.2 HJB Equation with State-Dependent Diffusion

**General Form**:
```
-∂u/∂t + H(∇u, x, m) = D(t,x,m)/2 Δu
```

**1D Discretization** (backward in time):
```
-(u[k,i] - u[k+1,i])/Δt + H[k,i] =
    D[k,i] (u[k,i+1] - 2u[k,i] + u[k,i-1]) / Δx²
```

---

## Appendix B: Example Use Cases

### B.1 Porous Medium Equation

```python
from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import create_mfg_solver

def porous_medium_diffusion(t, x, m, exponent=2):
    """
    Nonlinear diffusion: D(m) = D₀ m^(n-1)

    For n=2 (exponent=2): D(m) = D₀ m (porous medium)
    For n=1 (exponent=1): D(m) = D₀ (linear diffusion)
    For n<1: Fast diffusion
    For n>2: Slow diffusion
    """
    D0 = 0.1
    return D0 * m**(exponent - 1)

problem = ExampleMFGProblem(Nx=200, Nt=100, T=1.0)
solver = create_mfg_solver(
    problem,
    hjb_method="fdm",
    fp_method="fdm",
    diffusion_field=lambda t, x, m: porous_medium_diffusion(t, x, m, exponent=2),
)

result = solver.solve()

# Verify compact support (characteristic of porous medium)
import matplotlib.pyplot as plt
plt.plot(result.M[-1, :])
plt.title("Final density (compact support)")
plt.show()
```

### B.2 Crowd Avoidance Dynamics

```python
def crowd_avoidance_drift(t, x, m, sensitivity=0.5):
    """
    Agents move away from high-density regions.

    α(m) = -sensitivity * ∇m
    """
    grad_m = np.gradient(m, x[1] - x[0])  # Compute ∇m
    return -sensitivity * grad_m

def enhanced_diffusion_in_crowds(t, x, m, base_diffusion=0.1):
    """
    Diffusion increases in crowded regions.

    D(m) = D₀(1 + βm)
    """
    beta = 2.0
    return base_diffusion * (1 + beta * m)

solver = create_mfg_solver(
    problem,
    drift_field=crowd_avoidance_drift,
    diffusion_field=enhanced_diffusion_in_crowds,
)

result = solver.solve()
```

### B.3 Temperature-Dependent Transport

```python
def temperature_dependent_diffusion(t, x, m, activation_energy=1.0):
    """
    Arrhenius-type diffusion: D(T) = D₀ exp(-E/(kT))

    Model m as temperature field (rescaled).
    """
    D0 = 0.1
    k_boltzmann = 1.0

    # Ensure m > 0 (temperature must be positive)
    m_safe = np.maximum(m, 1e-6)

    return D0 * np.exp(-activation_energy / (k_boltzmann * m_safe))

solver = create_mfg_solver(
    problem,
    diffusion_field=temperature_dependent_diffusion,
)
```

---

## Appendix C: Known Limitations & Future Work

### C.1 Phase 2 Limitations

1. **No self-consistent implicit treatment**:
   - Callable diffusion evaluated at `m[k]`, not `m[k+1]`
   - May cause instability for strongly nonlinear problems
   - **Mitigation**: Small timesteps, damping

2. **1D only**:
   - Array and callable diffusion implemented for 1D FDM
   - nD support deferred to Phase 2.4

3. **FDM only**:
   - Particle, network, GFDM solvers still use constant diffusion
   - **Phase 3 extension**

4. **No anisotropic tensors**:
   - Diffusion assumed isotropic (scalar)
   - Full tensor `D[i,j](t,x,m)` deferred to Phase 3

### C.2 Phase 3 Enhancements

1. **Implicit callable evaluation**:
   - Iterate to find self-consistent `(m, D(m))`
   - Required for fast diffusion (`D(m) → 0`)

2. **Adaptive timestep control**:
   - Automatic `Δt` adjustment for stability
   - CFL condition for advection-dominated flows

3. **Anisotropic diffusion tensors**:
   - Support `D = [[D_xx, D_xy], [D_yx, D_yy]]`
   - Required for realistic transport models

4. **Performance optimization**:
   - JIT compilation (Numba/JAX)
   - Sparse matrix reuse (when structure unchanging)
   - GPU acceleration for callable evaluation

---

**End of Phase 2 Design Document**

**Next Steps**: Review design, then proceed to implementation starting with Phase 2.1.
