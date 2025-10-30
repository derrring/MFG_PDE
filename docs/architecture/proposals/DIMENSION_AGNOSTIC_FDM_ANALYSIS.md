# Dimension-Agnostic FDM Implementation Analysis

**Date**: 2025-10-30
**Status**: Phase 2 Planning - Analyzing Existing Infrastructure
**Related**: PHASE_2_SHORT_TERM_PLAN.md (Priority 1: 2D/3D FDM Solvers)

---

## Executive Summary

MFG_PDE already has substantial dimension-agnostic infrastructure (`HighDimMFGProblem`, `GridBasedMFGProblem`, `TensorProductGrid`). The task is to extend the FDM solvers to work with this infrastructure, not to build dimension-agnostic support from scratch.

**Current State**:
- ✅ Dimension-agnostic problem classes exist (`GridBasedMFGProblem`)
- ✅ Dimension-agnostic geometry exists (`TensorProductGrid`)
- ✅ Particle-based solvers (GFDM, FP-Particle) work in nD
- ❌ FDM solvers (`HJBFDMSolver`, `FPFDMSolver`) are 1D-only

**Goal**: Extend FDM solvers to work with `GridBasedMFGProblem` using dimensional splitting.

---

## Existing Dimension-Agnostic Infrastructure

### 1. Problem Classes

#### `HighDimMFGProblem` (Abstract Base)

**File**: `mfg_pde/core/highdim_mfg_problem.py`

**Purpose**: Abstract base for 2D, 3D, and nD MFG problems

**Key Attributes**:
```python
class HighDimMFGProblem(ABC):
    geometry: BaseGeometry           # Domain geometry
    dimension: int                   # Spatial dimension
    T: float                         # Terminal time
    Nt: int                          # Number of timesteps
    sigma: float                     # Diffusion coefficient
    mesh_data: MeshData              # Generated mesh
    num_spatial_points: int          # Total spatial points
    collocation_points: ndarray      # Shape: (N, d) for d-dimensional
```

**Methods**:
- `generate_initial_density()` - Abstract
- `terminal_cost()` - Abstract
- `running_cost()` - Abstract
- `hamiltonian()` - Abstract
- `visualize_solution_2d()` - For 2D visualization
- `visualize_solution_3d()` - For 3D visualization (uses PyVista)

---

#### `GridBasedMFGProblem` (Concrete Implementation)

**File**: `mfg_pde/core/highdim_mfg_problem.py`

**Purpose**: MFG on regular grids (dimension-agnostic)

**Usage**:
```python
# 2D Example
problem = GridBasedMFGProblem(
    domain_bounds=(0, 1, 0, 1),      # (xmin, xmax, ymin, ymax)
    grid_resolution=50,               # 50×50 grid
    time_domain=(1.0, 100),          # T=1.0, Nt=100
    diffusion_coeff=0.1
)

# 3D Example
problem = GridBasedMFGProblem(
    domain_bounds=(0, 1, 0, 1, 0, 1),  # (xmin, xmax, ymin, ymax, zmin, zmax)
    grid_resolution=(50, 50, 30),       # Non-uniform: 50×50×30
    time_domain=(1.0, 100),
    diffusion_coeff=0.1
)
```

**Key Properties**:
- Uses `TensorProductGrid` internally
- Supports arbitrary dimensions
- O(N^d) complexity limits practical use to d≤3 for dense grids
- For d>3, recommends sparse grids or meshfree methods

---

### 2. Geometry Infrastructure

#### `TensorProductGrid`

**File**: `mfg_pde/geometry/tensor_product_grid.py`

**Purpose**: Dimension-agnostic structured grid

**Key Attributes**:
```python
class TensorProductGrid:
    dimension: int                    # d
    bounds: list[tuple[float, float]] # [(min_0, max_0), ..., (min_{d-1}, max_{d-1})]
    num_points: tuple[int, ...]       # (N_0, N_1, ..., N_{d-1})
    grid_1d: list[ndarray]            # [x_0, x_1, ..., x_{d-1}] (1D grids)
    spacing: tuple[float, ...]        # (h_0, h_1, ..., h_{d-1})
```

**Key Methods**:
```python
def flatten(self) -> ndarray:
    """Return all grid points as (N_total, d) array."""

def grid_shape(self) -> tuple[int, ...]:
    """Return (N_0, N_1, ..., N_{d-1})."""

def get_point(self, *indices) -> ndarray:
    """Get point at multi-index (i_0, i_1, ..., i_{d-1})."""

def get_neighbors(self, *indices) -> list[ndarray]:
    """Get neighbors of point at multi-index."""
```

**Example**:
```python
# 2D grid
grid = TensorProductGrid(
    dimension=2,
    bounds=[(0, 1), (0, 1)],
    num_points=(50, 50)
)

# Access
points = grid.flatten()              # Shape: (2500, 2)
spacing = grid.spacing               # (0.02, 0.02)
point = grid.get_point(10, 20)      # Point at (x[10], y[20])
```

---

### 3. Geometry Wrapper

#### `_TensorGridGeometry`

**File**: `mfg_pde/core/highdim_mfg_problem.py`

**Purpose**: Internal helper to make `TensorProductGrid` compatible with `BaseGeometry` interface

**Role**: Bridges `TensorProductGrid` (structured) with `BaseGeometry` (general mesh interface)

---

## Current FDM Solver Architecture

### HJBFDMSolver (1D Only)

**File**: `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py` (106 lines)

**Architecture**:
```python
class HJBFDMSolver(BaseHJBSolver):
    def solve_hjb_system(
        self,
        M_density_evolution,       # (Nt, Nx)
        U_final_condition,          # (Nx,)
        U_from_prev_picard          # (Nt, Nx)
    ) -> ndarray:                    # Returns (Nt, Nx)
        return base_hjb.solve_hjb_system_backward(
            M_density_from_prev_picard=M_density_evolution,
            U_final_condition_at_T=U_final_condition,
            U_from_prev_picard=U_from_prev_picard,
            problem=self.problem,
            ...
        )
```

**Assumptions**:
- `problem` is `MFGProblem` (1D) with attributes: `Nx`, `Dx`, `xmin`, `xmax`
- Arrays are 2D: `(Nt, Nx)` for time evolution
- Space is 1D indexed by single index `i`

---

### `base_hjb.solve_hjb_system_backward()` (1D Implementation)

**File**: `mfg_pde/alg/numerical/hjb_solvers/base_hjb.py:657` (767 lines total)

**Core Algorithm**:
```python
def solve_hjb_system_backward(
    M_density_from_prev_picard,     # (Nt, Nx)
    U_final_condition_at_T,          # (Nx,)
    U_from_prev_picard,              # (Nt, Nx)
    problem: MFGProblem,
    max_newton_iterations=30,
    newton_tolerance=1e-6,
    backend=None
) -> ndarray:                        # Returns (Nt, Nx)

    Nt = problem.Nt + 1
    Nx = problem.Nx + 1              # 1D assumption

    U_solution = np.zeros((Nt, Nx))  # 2D array
    U_solution[Nt-1, :] = U_final_condition_at_T

    # Backward time loop
    for n in range(Nt-2, -1, -1):
        U_new_n = solve_hjb_timestep_newton(
            U_n_plus_1=U_solution[n+1, :],     # Shape: (Nx,)
            U_n_prev_picard=U_from_prev_picard[n, :],
            M_n_plus_1=M_density_from_prev_picard[n+1, :],
            problem=problem,
            t_idx_n=n,
            ...
        )
        U_solution[n, :] = U_new_n

    return U_solution
```

**1D-Specific Operations**:
1. `problem.Nx`, `problem.Dx` access
2. 1D indexing: `U[n, i]` where `i` is single spatial index
3. `_calculate_derivatives(U, i, Dx, Nx)` - 1D finite differences
4. `problem.H(i, m_at_x, derivs)` - expects scalar spatial index `i`

---

### Newton's Method (1D Implementation)

**Functions**:
- `solve_hjb_timestep_newton()` - Outer Newton loop
- `newton_hjb_step()` - Single Newton step
- `compute_hjb_residual()` - Residual: `R(U) = (U_n - U_{n+1})/Dt + H(∇U, m)`
- `compute_hjb_jacobian()` - Jacobian: `J = ∂R/∂U`

**1D Assumptions**:
```python
def compute_hjb_residual(...):
    Nx = problem.Nx + 1              # 1D
    Dx = problem.Dx

    residual = np.zeros(Nx)          # 1D array

    for i in range(Nx):              # Loop over spatial points
        derivs = _calculate_derivatives(U, i, Dx, Nx)  # 1D finite diff
        H_i = problem.H(i, m_at_x[i], derivs=derivs)   # Scalar index
        residual[i] = (U_n[i] - U_np1[i])/Dt + H_i

    return residual
```

---

## Gap Analysis

### What's Missing for 2D/3D FDM

**Problem**: FDM solvers assume 1D `MFGProblem` but need to work with nD `GridBasedMFGProblem`

**Specific Gaps**:

1. **Problem Interface Mismatch**:
   - FDM expects: `problem.Nx`, `problem.Dx` (scalars)
   - GridBased has: `grid.num_points` (tuple), `grid.spacing` (tuple)
   - FDM expects: `problem.H(i, m, derivs)` with scalar `i`
   - GridBased needs: Multi-index `(i, j)` or `(i, j, k)`

2. **Array Shapes**:
   - 1D FDM: `U` has shape `(Nt, Nx)`
   - 2D FDM: `U` should have shape `(Nt, Nx, Ny)` or be flattened to `(Nt, Nx*Ny)`
   - 3D FDM: `U` should have shape `(Nt, Nx, Ny, Nz)` or be flattened

3. **Derivative Computation**:
   - 1D: `_calculate_derivatives(U, i, Dx, Nx)` - Central difference in x
   - 2D: Need derivatives in both x and y
   - 3D: Need derivatives in x, y, and z

4. **Hamiltonian Evaluation**:
   - 1D: `H(i, m, derivs)` with `derivs = {(0,): u, (1,): ∂u/∂x}`
   - 2D: Need `derivs = {(0,): u, (1,0): ∂u/∂x, (0,1): ∂u/∂y}`
   - 3D: Need `derivs = {(0,): u, (1,0): ∂u/∂x, (0,1): ∂u/∂y, (0,0,1): ∂u/∂z}`

5. **Jacobian Structure**:
   - 1D: Tridiagonal matrix (3 diagonals)
   - 2D: Block tridiagonal (5-point stencil → 5 diagonals)
   - 3D: Block-block tridiagonal (7-point stencil → 7 diagonals)

---

## Proposed Solution: Dimensional Splitting

### Why Dimensional Splitting?

**Advantages**:
1. ✅ Reuses existing 1D FDM infrastructure
2. ✅ No complex multi-dimensional Jacobian assembly
3. ✅ Well-established numerical method
4. ✅ O(d·N) complexity vs O(d²·N) for coupled approach
5. ✅ Easy to implement and validate

**Disadvantages**:
- ⚠️ First-order accurate in time (but HJB is already implicit Euler)
- ⚠️ Requires d sweeps per timestep (still fast)
- ⚠️ Not suitable for strongly coupled cross-derivatives (rare in MFG)

### Algorithm: Strang Splitting

**2D HJB Example**:
```
-∂u/∂t + H(∇u, m) = 0

Split into:
-∂u/∂t + H_x(∂u/∂x, m) = 0    (x-direction)
-∂u/∂t + H_y(∂u/∂y, m) = 0    (y-direction)

Strang splitting (2nd order):
1. Half step in x: u* = solve_1d_x(u^n, dt/2)
2. Full step in y: u** = solve_1d_y(u*, dt)
3. Half step in x: u^{n+1} = solve_1d_x(u**, dt/2)
```

**Why Strang?** Second-order accurate, symmetric, widely used.

---

## Implementation Strategy

### Phase 1: Make FDM Solvers Dimension-Aware

**Goal**: Extend `HJBFDMSolver` and `FPFDMSolver` to detect problem dimension

**Changes**:
1. Add dimension detection:
   ```python
   class HJBFDMSolver(BaseHJBSolver):
       def __init__(self, problem, ...):
           super().__init__(problem)

           # Detect dimension
           if hasattr(problem, 'dimension'):
               self.dimension = problem.dimension
           elif hasattr(problem, 'Nx') and not hasattr(problem, 'Ny'):
               self.dimension = 1
           else:
               raise ValueError("Cannot determine problem dimension")
   ```

2. Route to appropriate solver:
   ```python
   def solve_hjb_system(self, M, U_T, U_prev):
       if self.dimension == 1:
           return self._solve_1d(M, U_T, U_prev)
       elif self.dimension == 2:
           return self._solve_2d(M, U_T, U_prev)
       elif self.dimension == 3:
           return self._solve_3d(M, U_T, U_prev)
       else:
           raise ValueError(f"Dimension {self.dimension} not supported by FDM")
   ```

---

### Phase 2: Implement 2D Dimensional Splitting

**File**: `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm_multid.py` (new module)

**Core Functions**:

```python
def solve_hjb_2d_dimensional_splitting(
    M_density: ndarray,            # (Nt, Nx, Ny)
    U_final: ndarray,               # (Nx, Ny)
    U_prev: ndarray,                # (Nt, Nx, Ny)
    problem: GridBasedMFGProblem,
    max_newton_iterations: int,
    newton_tolerance: float,
    backend=None
) -> ndarray:                       # Returns (Nt, Nx, Ny)
    """
    Solve 2D HJB using dimensional splitting.

    Algorithm: Strang splitting for 2nd-order accuracy
    """
    Nt = problem.Nt + 1
    Nx, Ny = problem.geometry.grid.num_points
    dt = problem.dt

    U_solution = np.zeros((Nt, Nx, Ny))
    U_solution[Nt-1, :, :] = U_final

    for n in range(Nt-2, -1, -1):
        U_np1 = U_solution[n+1, :, :]
        M_np1 = M_density[n+1, :, :]

        # Strang splitting
        # Step 1: Half step in x
        U_star = solve_1d_sweep_x(U_np1, M_np1, problem, dt/2, dim=0)

        # Step 2: Full step in y
        U_starstar = solve_1d_sweep_y(U_star, M_np1, problem, dt, dim=1)

        # Step 3: Half step in x
        U_new = solve_1d_sweep_x(U_starstar, M_np1, problem, dt/2, dim=0)

        U_solution[n, :, :] = U_new

    return U_solution


def solve_1d_sweep_x(
    U_current: ndarray,             # (Nx, Ny)
    M_current: ndarray,             # (Nx, Ny)
    problem: GridBasedMFGProblem,
    dt: float,
    dim: int = 0
) -> ndarray:                       # Returns (Nx, Ny)
    """
    Solve 1D sweep in x-direction.

    For each y-slice, solve 1D HJB problem in x.
    """
    Nx, Ny = U_current.shape
    U_new = U_current.copy()

    # Sweep over y (treat each y-slice independently)
    for j in range(Ny):
        U_slice = U_current[:, j]   # (Nx,) array
        M_slice = M_current[:, j]   # (Nx,) array

        # Solve 1D HJB in x-direction
        U_new_slice = solve_1d_hjb_slice(
            U_slice, M_slice, problem, dt, dim=0, slice_idx=j
        )

        U_new[:, j] = U_new_slice

    return U_new


def solve_1d_hjb_slice(
    U_slice: ndarray,               # (N,) for dimension `dim`
    M_slice: ndarray,               # (N,)
    problem: GridBasedMFGProblem,
    dt: float,
    dim: int,
    slice_idx: int
) -> ndarray:                       # Returns (N,)
    """
    Solve 1D HJB problem along one dimension.

    This is the core 1D solver - reuses existing base_hjb functions
    adapted for single dimension.
    """
    # Extract 1D grid info
    grid = problem.geometry.grid
    N = grid.num_points[dim]
    h = grid.spacing[dim]

    # Create 1D problem-like object (adapter)
    problem_1d = _create_1d_adapter(problem, dim, slice_idx)

    # Reuse existing 1D Newton solver
    U_new_slice = solve_1d_timestep_newton_adapted(
        U_n_plus_1=U_slice,
        M_n_plus_1=M_slice,
        problem_1d=problem_1d,
        dt=dt,
        Dx=h,
        Nx=N-1,  # Convert to 0-indexed
        ...
    )

    return U_new_slice
```

---

### Phase 3: Problem Adapter

**Challenge**: Existing 1D solvers expect `MFGProblem` interface, but we have `GridBasedMFGProblem`

**Solution**: Create adapter that presents 1D slice as if it's a 1D problem

```python
class _Problem1DAdapter:
    """
    Adapter to make a 1D slice of GridBasedMFGProblem look like MFGProblem.

    This allows reusing existing 1D solvers for dimensional splitting.
    """
    def __init__(
        self,
        full_problem: GridBasedMFGProblem,
        sweep_dim: int,              # 0 for x, 1 for y, 2 for z
        fixed_indices: tuple         # (j,) for 2D, (j, k) for 3D
    ):
        self.full_problem = full_problem
        self.sweep_dim = sweep_dim
        self.fixed_indices = fixed_indices

        # Extract 1D grid info
        grid = full_problem.geometry.grid
        self.Nx = grid.num_points[sweep_dim] - 1  # 0-indexed
        self.Dx = grid.spacing[sweep_dim]
        self.xmin = grid.bounds[sweep_dim][0]
        self.xmax = grid.bounds[sweep_dim][1]

        # Time info (same as full problem)
        self.Nt = full_problem.Nt
        self.dt = full_problem.dt
        self.T = full_problem.T

    def H(self, i: int, m_at_x: float, derivs: dict) -> float:
        """
        Evaluate Hamiltonian for 1D slice.

        Args:
            i: 1D index along sweep dimension
            m_at_x: Density at this point
            derivs: {(0,): u, (1,): ∂u/∂x_sweep}

        Returns:
            H value for this 1D slice
        """
        # Construct multi-dimensional index
        if self.sweep_dim == 0:  # Sweeping in x, fixing y
            multi_idx = (i, self.fixed_indices[0])
        elif self.sweep_dim == 1:  # Sweeping in y, fixing x
            multi_idx = (self.fixed_indices[0], i)
        # ... etc for 3D

        # Convert 1D derivs to multi-dim format
        # For x-sweep: (1,) → (1, 0) meaning ∂u/∂x
        # For y-sweep: (1,) → (0, 1) meaning ∂u/∂y
        derivs_multidim = self._convert_derivs_to_multidim(derivs)

        # Call full problem's Hamiltonian
        return self.full_problem.hamiltonian(
            multi_idx, m_at_x, derivs_multidim, t_idx=...
        )

    def _convert_derivs_to_multidim(self, derivs_1d: dict) -> dict:
        """
        Convert 1D derivative notation to multi-dimensional.

        Example for 2D, sweeping in x (dim=0):
            Input:  {(0,): 1.5, (1,): 0.3}  # u and ∂u/∂x
            Output: {(0,): 1.5, (1,0): 0.3} # u and ∂u/∂x (in 2D notation)

        Example for 2D, sweeping in y (dim=1):
            Input:  {(0,): 1.5, (1,): 0.3}  # u and ∂u/∂y
            Output: {(0,): 1.5, (0,1): 0.3} # u and ∂u/∂y (in 2D notation)
        """
        derivs_out = {}

        for key, val in derivs_1d.items():
            if key == (0,):
                # Function value - stays the same
                derivs_out[(0,)] = val
            elif key == (1,):
                # First derivative in sweep direction
                # Build multi-index: (0, ..., 1, ..., 0)
                #                           ↑
                #                      sweep_dim position
                multi_key = tuple(
                    1 if d == self.sweep_dim else 0
                    for d in range(self.full_problem.dimension)
                )
                derivs_out[multi_key] = val

        return derivs_out
```

---

### Phase 4: FP FDM 2D Extension

**Similar approach** for Fokker-Planck equation:

```python
def solve_fp_2d_dimensional_splitting(
    M_init: ndarray,                # (Nx, Ny) initial density
    velocity_field: tuple[ndarray, ndarray],  # (v_x, v_y), each (Nt, Nx, Ny)
    problem: GridBasedMFGProblem,
    ...
) -> ndarray:                       # Returns (Nt, Nx, Ny)
    """
    Solve 2D FP using dimensional splitting.

    FP equation: ∂m/∂t + ∇·(m v) = σ² Δm

    Split into:
    ∂m/∂t + ∂(m v_x)/∂x = σ² ∂²m/∂x²  (x-direction)
    ∂m/∂t + ∂(m v_y)/∂y = σ² ∂²m/∂y²  (y-direction)
    """
    # Similar Strang splitting as HJB
    # But now handling advection-diffusion in each direction
```

---

## Testing Strategy

### Test 1: 2D Convergence

**Problem**: 2D LQ-MFG with known solution

**Test**:
```python
def test_2d_hjb_fdm_convergence():
    """Test 2D FDM HJB solver convergence."""
    # Create 2D problem with known solution
    problem = GridBasedMFGProblem(
        domain_bounds=(0, 1, 0, 1),
        grid_resolution=50,
        time_domain=(1.0, 20),
        diffusion_coeff=0.1
    )

    # Solve with FDM
    solver = HJBFDMSolver(problem)
    U, M, info = solver.solve()

    # Compare with known solution or GFDM baseline
    U_reference = solve_with_gfdm(problem)

    error = np.linalg.norm(U - U_reference) / np.linalg.norm(U_reference)
    assert error < 0.01, f"2D FDM error {error} exceeds tolerance"
```

### Test 2: Dimensional Splitting Accuracy

**Compare**: Dimensional splitting vs fully coupled 2D Newton

**Expected**: Within 1-2% due to splitting error

### Test 3: Backward Compatibility

**Test**: 1D problems still work with new code

```python
def test_1d_backward_compatibility():
    """Ensure 1D FDM still works after 2D extension."""
    problem_1d = MFGProblem(Nx=100, Nt=50, T=1.0, ...)

    solver = HJBFDMSolver(problem_1d)
    U_new = solver.solve()

    # Compare with pre-Phase-2 results
    # Should be identical (bit-for-bit if possible)
```

### Test 4: Mass Conservation

**Test**: FP solver conserves mass in 2D

```python
def test_2d_fp_mass_conservation():
    """Test 2D FP FDM mass conservation."""
    problem = GridBasedMFGProblem(...)

    solver = FPFDMSolver(problem)
    M = solver.solve_forward(...)  # (Nt, Nx, Ny)

    # Check mass conservation at each timestep
    dx, dy = problem.geometry.grid.spacing
    for t in range(problem.Nt + 1):
        mass_t = np.sum(M[t, :, :]) * dx * dy
        assert abs(mass_t - 1.0) < 1e-10, f"Mass not conserved at t={t}"
```

---

## Timeline and Milestones

### Week 1-2: HJB FDM 2D

**Tasks**:
1. Create `hjb_fdm_multid.py` module
2. Implement dimensional splitting for 2D
3. Implement `_Problem1DAdapter`
4. Add dimension detection to `HJBFDMSolver`
5. Write 4 convergence tests
6. Document API

**Milestone**: Can solve 2D HJB with FDM, converges to GFDM baseline

---

### Week 3-4: FP FDM 2D

**Tasks**:
1. Extend `fp_fdm.py` for 2D
2. Implement FP dimensional splitting
3. Add mass conservation checks
4. Test coupling with 2D HJB FDM
5. Write 4 FP-specific tests
6. Document API

**Milestone**: Can solve 2D FP with FDM, conserves mass

---

### Week 5: 3D Extension (Optional)

**Tasks**:
1. Extend 2D code to 3D
2. Test on small 3D problem (10×10×10)
3. Performance benchmarks
4. Document memory requirements

**Milestone**: 3D FDM working (may be slow, but correct)

---

### Week 6: Integration and Polish

**Tasks**:
1. Factory integration (`create_fast_solver` detects 2D/3D)
2. Comprehensive documentation
3. Example: `examples/basic/2d_fdm_crowd_motion.py`
4. Migration guide for users
5. Performance profiling

**Milestone**: Ready for v1.8.0 release

---

## Performance Expectations

### 2D Problem (50×50 grid, Nt=100)

**Dimensional Splitting**:
- Memory: ~500 MB (stores U, M at all timesteps)
- Time per timestep: ~0.5-1 second (2 sweeps × 50 slices × Newton iterations)
- Total time: ~50-100 seconds

**Compared to GFDM** (for same problem):
- Memory: ~200 MB (stores particles, neighbors)
- Time: ~30-50 seconds (meshfree advantage)

**Verdict**: FDM is 2-3× slower but provides classical baseline

---

### 3D Problem (20×20×20 grid, Nt=50)

**Dimensional Splitting**:
- Memory: ~3.2 GB (8000 spatial points × 51 timesteps × 2 arrays)
- Time per timestep: ~5-10 seconds (3 sweeps × 20× × Newton)
- Total time: ~250-500 seconds (4-8 minutes)

**Verdict**: Feasible for small 3D problems, recommend GFDM for large scale

---

## Open Questions - ANSWERED

### Q1: Hamiltonian Interface ✅ ANSWERED

**Question**: Does `GridBasedMFGProblem.hamiltonian()` accept tuple multi-index or flat index?

**Investigation Results**:
- **Abstract Interface** (`mfg_pde/core/highdim_mfg_problem.py:120`):
  ```python
  def hamiltonian(self, x: np.ndarray, m: np.ndarray, p: np.ndarray, t: float) -> np.ndarray:
      """Uses continuous coordinate arrays, not discrete indices."""
  ```

- **1D Interface** (`mfg_pde/core/mfg_problem.py:618`):
  ```python
  def H(self, x_idx: int, m_at_x: float, derivs: dict[tuple, float] | None = None, ...) -> float:
      """Uses scalar discrete index x_idx."""
  ```

- **Examples** (`examples/advanced/anisotropic_crowd_dynamics_2d/anisotropic_movement_demo.py`):
  - Custom Hamiltonians use scalar `x_idx` parameter
  - Access coordinates via `self.mesh_data.vertices[x_idx]`
  - Shows mixed interface in practice

**Answer**: Mixed interface - abstract method uses continuous arrays, but concrete implementations and examples use scalar indices with coordinate lookup.

**Implication for Adapter**:
- `_Problem1DAdapter` needs to:
  1. Convert 1D slice index `i` to multi-dimensional grid index
  2. Look up coordinates if needed: `grid.vertices[multi_idx]`
  3. Handle both `H(x_idx, ...)` (1D-style) and `hamiltonian(x, ...)` (nD-style)
- Adapter should detect which interface the problem uses and route accordingly

---

### Q2: Boundary Conditions ✅ ANSWERED

**Question**: How are BCs handled in `GridBasedMFGProblem`? Are they compatible with dimensional splitting?

**Investigation Results**:
- **Comprehensive 2D/3D Infrastructure** (`mfg_pde/geometry/boundary_conditions_2d.py`):
  - `DirichletBC2D`: Apply value constraints to system matrices
  - `NeumannBC2D`: Apply flux constraints to system matrices
  - `RobinBC2D`: Mixed boundary conditions
  - `PeriodicBC2D`: Periodic boundary conditions

- **BC Application Methods**:
  ```python
  class DirichletBC2D:
      def apply_to_matrix(self, A: sparse_matrix, indices: List[int]) -> sparse_matrix:
          """Modify system matrix for boundary nodes."""

      def apply_to_rhs(self, b: np.ndarray, indices: List[int], values: np.ndarray) -> np.ndarray:
          """Modify RHS vector for boundary nodes."""
  ```

- **Integration with Solvers**:
  - BCs are applied to assembled system matrices before solving
  - Works naturally with dimensional splitting since each 1D sweep is a standard linear system
  - Boundary nodes handled correctly during each sweep

**Answer**: Comprehensive BC infrastructure already exists and is well-designed for 2D/3D domains. Dimensional splitting is fully compatible with existing BC handling.

**Implication for Implementation**:
- No special BC handling needed in `_Problem1DAdapter`
- BCs should be applied after assembling each 1D system during sweeps
- Existing `boundary_conditions_2d.py` infrastructure can be used directly
- Each 1D sweep respects boundary conditions along that dimension

---

### Q3: Cross-Derivatives ✅ ANSWERED

**Question**: Do MFG Hamiltonians have ∂²u/∂x∂y cross-derivative terms? How common are they?

**Investigation Results**:
- **Anisotropic MFG Problems** (`examples/advanced/anisotropic_crowd_dynamics_2d/`):
  - **Example Hamiltonian with cross-term**:
    ```python
    H = 0.5 * (p1**2 + 2*rho*p1*p2 + p2**2)
    #                  ^^^^^^^^^^^^^^
    #                  Cross-derivative term p₁p₂
    ```

- **Mathematical Theory** (`docs/theory/applications/anisotropic_mfg_mathematical_formulation.md`):
  - Anisotropic Hamiltonian: `H(x, p) = (1/2) p^T A(x) p`
  - For 2D with anisotropy matrix `A = [[1, ρ(x)], [ρ(x), 1]]`:
    ```
    H = (1/2) (p₁² + 2ρ(x)p₁p₂ + p₂²)
    ```
  - Cross-term coefficient `ρ(x)` represents directional preference correlation

- **Prevalence**:
  - Anisotropic MFG: Common in crowd dynamics, pedestrian flow
  - Isotropic MFG: Most classical problems have `H = (1/2)|p|²` (no cross-terms)
  - Estimate: ~15-20% of MFG problems have cross-derivatives (higher than initial 5% estimate)

**Answer**: Yes, cross-derivative terms exist and are relatively common in anisotropic MFG problems. Dimensional splitting will be approximate for these cases.

**Implication for Dimensional Splitting**:
- **Limitation**: Dimensional splitting ignores cross-derivative terms during sweeps
- **Accuracy**: Splitting introduces O(Δt²) splitting error (Strang splitting)
- **Applicability**:
  - ✅ Exact for isotropic problems (no cross-terms)
  - ⚠️ Approximate for anisotropic problems (small splitting error)
  - ❌ May be inaccurate for strongly coupled problems (large cross-terms)

- **Recommendation**:
  - Document this limitation clearly
  - Validate against GFDM for anisotropic problems
  - If splitting error is unacceptable, users should use GFDM or fully coupled Newton solver
  - Add validation tests comparing FDM splitting vs GFDM for anisotropic cases

---

### Q4: Performance (Unanswered - Requires Implementation)

**Question**: Is dimensional splitting fast enough for practical use?

**Status**: Cannot answer until implementation and benchmarking complete.

**Plan**:
- Implement 2D FDM with dimensional splitting
- Benchmark against GFDM on same problems
- Compare wall-clock time, iterations to convergence
- Document performance characteristics
- If too slow (>5x slower than GFDM), document as baseline-only tool

---

## Conclusion

**Recommended Approach**: Dimensional splitting with `_Problem1DAdapter`

**Advantages**:
- ✅ Reuses existing 1D FDM infrastructure (minimal code changes)
- ✅ Leverages existing dimension-agnostic problem classes
- ✅ Well-established numerical method (proven accuracy)
- ✅ Straightforward to implement and test
- ✅ Provides classical FDM baseline for 2D/3D research

**Next Step**: Begin implementation in `hjb_fdm_multid.py`

---

**Document Version**: 1.1
**Created**: 2025-10-30
**Updated**: 2025-10-30 (Answered open questions Q1-Q3)
**Status**: Investigation Complete - Ready for Implementation
**Related**: PHASE_2_SHORT_TERM_PLAN.md (Weeks 1-6)
