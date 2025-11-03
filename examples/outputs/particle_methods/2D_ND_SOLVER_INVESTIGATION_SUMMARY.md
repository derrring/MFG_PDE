# 2D nD Solver Investigation Summary

**Date**: 2025-11-03
**Framework**: MFG_PDE v0.9.0 (Phase 3 Unified Architecture)
**Issue**: 2D crowd evacuation showing minimal movement toward exit

## Problem Statement

Initial 2D crowd evacuation implementation showed two critical issues:
1. **Static errors**: Picard iteration errors weren't decreasing ("error keeps static")
2. **No evacuation**: Crowd barely moved toward exit (0% evacuation progress)

## Root Cause Analysis

### Incorrect Approach (Broken Implementation)

**File**: `examples/outputs/particle_methods/2d_crowd_density_evolution.py`

**Issues Identified**:

1. **Flattened 2D as 1D**:
   ```python
   # WRONG: Treating 30×30 grid as 900-point 1D problem
   total_points = (Nx + 1) ** 2  # 900 for 30×30
   problem = MFGProblem(
       xmin=0.0,
       xmax=float(total_points - 1),
       Nx=total_points - 1,  # Creates 1D problem with 900 points
   )
   ```

2. **MFGComponents with Complex Signature**:
   ```python
   # WRONG: Complex signature incompatible with nD HJB solver
   def hamiltonian_func(x_idx, x_position, m_at_x, derivs, t_idx, current_time, problem):
       du_dx = derivs.get((1, 0), 0.0)
       du_dy = derivs.get((0, 1), 0.0)
       # ...
   ```
   This signature doesn't provide the full gradient vector `p` that nD HJB solver expects.

3. **Missing Dimension Attributes**:
   - MFGProblem with custom components lacks `geometry.grid.dimension`
   - HJB solver's `_detect_dimension()` couldn't identify 2D nature
   - Defaulted to 1D solver mode despite grid adapter

4. **No Running Cost**:
   - Missing time penalty to encourage immediate evacuation
   - Only terminal cost (distance to exit at t=T)

**Observed Behavior**:
- Errors appeared to decrease but convergence was poor
- Completed in 647s (40 iterations)
- Perfect mass conservation (particle-based FP)
- **Minimal movement**: Center of mass (5.0, 7.0) → (5.0, 6.11) = 0.89m of 7m needed
- **0% evacuation progress**

### Correct Approach (Working Implementation)

**File**: `examples/outputs/particle_methods/2d_crowd_proper_nd.py`

**Key Patterns**:

1. **Use GridBasedMFGProblem Base Class**:
   ```python
   from mfg_pde.core.highdim_mfg_problem import GridBasedMFGProblem

   class CrowdEvacuation2D(GridBasedMFGProblem):
       def __init__(self, grid_resolution=30, ...):
           super().__init__(
               domain_bounds=(0.0, 10.0, 0.0, 10.0),  # 2D: (xmin, xmax, ymin, ymax)
               grid_resolution=grid_resolution,        # N×N grid
               time_domain=(time_horizon, num_timesteps),
               diffusion_coeff=diffusion_coeff,
           )
   ```
   This provides:
   - Proper `TensorProductGrid` with `dimension` attribute
   - Geometry structure that solvers can query
   - Automatic dimension detection

2. **Standard Hamiltonian Signature**:
   ```python
   def hamiltonian(self, x, m, p, t):
       """
       H(x, m, p, t) with p as full gradient vector.

       Args:
           x: (N, 2) spatial coordinates
           m: (N,) density values
           p: (N, 2) gradient vector ∇u
           t: scalar time

       Returns:
           (N,) Hamiltonian values
       """
       # Vectorized computation
       if p.ndim == 1:
           p_squared = np.sum(p**2)
       else:
           p_squared = np.sum(p**2, axis=1)

       # Kinetic + congestion
       return 0.5 * p_squared + self.congestion_weight * m * p_squared
   ```

3. **Vectorized Problem Methods**:
   ```python
   def initial_density(self, x):
       """x: (N, 2) → (N,)"""
       dist_sq = np.sum((x - self.start_location)**2, axis=1)
       return np.exp(-dist_sq / (2 * 1.5**2)) / total_mass

   def terminal_cost(self, x):
       """x: (N, 2) → (N,) - distance to exit at y=0"""
       return x[:, 1]

   def running_cost(self, x, t):
       """x: (N, 2) → (N,) - time penalty"""
       return np.ones(x.shape[0])
   ```

4. **Factory Pattern for Solver Creation**:
   ```python
   from mfg_pde.factory import create_basic_solver

   problem = CrowdEvacuation2D(grid_resolution=20, ...)
   solver = create_basic_solver(
       problem,
       damping=0.5,
       max_iterations=30,
   )
   result = solver.solve()  # Auto-detects 2D, uses HJB-FDM + FP-FDM
   ```

**Observed Behavior**:
- Proper exponential error convergence:
  - Iteration 1: U_err=2.31e-01, M_err=5.31e-01
  - Iteration 2: U_err=1.25e-01, M_err=2.69e-01 (÷2)
  - Iteration 3: U_err=6.47e-02, M_err=1.31e-01 (÷2)
  - Iteration 4: U_err=3.30e-02, M_err=6.35e-02 (÷2)
- True 2D solving confirmed by solver output: "HJB 2D-FDM (newton)"
- Consistent ~68s per iteration (vs erratic 10-17s in broken version)
- *(Simulation still running at time of writing)*

## Technical Insights

### Why MFGComponents Approach Failed

The `MFGComponents` interface with complex signatures like:
```python
hamiltonian_func(x_idx, x_position, m_at_x, derivs, t_idx, current_time, problem)
```

was designed for:
- 1D problems with explicit finite difference stencils
- Legacy compatibility with dimensional splitting approaches
- Cases where custom derivative computation is needed

It is **incompatible** with nD HJB FDM solver because:
- Solver computes full gradient vector `p = ∇u` automatically
- Expects `problem.hamiltonian(x, m, p, t)` method call
- Cannot map complex signature to standard interface

### GridBased2DAdapter is Not Enough

Simply using `GridBased2DAdapter` for index mapping is insufficient:
```python
# NOT ENOUGH - still creates 1D MFGProblem underneath
self.grid_adapter = GridBased2DAdapter(Nx1, Nx2, domain_bounds)
problem = MFGProblem(xmin=0.0, xmax=float(total_nodes-1), ...)
```

The adapter helps with coordinate transformations but doesn't give the problem structure that nD solvers need:
- No `geometry.grid.dimension` attribute
- No `TensorProductGrid` structure
- No automatic dimension detection

### Correct nD Pattern Hierarchy

```
GridBasedMFGProblem (base class)
├── TensorProductGrid (dimension-agnostic grid)
├── _TensorGridGeometry (geometry wrapper)
├── Standard method signatures:
│   ├── hamiltonian(x, m, p, t)
│   ├── initial_density(x)
│   ├── terminal_cost(x)
│   └── running_cost(x, t)
└── Auto-detected dimension

Factory: create_basic_solver(problem)
├── Detects dimension from problem.geometry.grid.dimension
├── Routes to HJB-FDM nD solver
└── Routes to FP-FDM nD solver
```

## Particle Methods with GridBasedMFGProblem

**Important clarification**: GridBasedMFGProblem does NOT force grid-based FP solvers. You can still use particle methods for the Fokker-Planck equation:

```python
problem = CrowdEvacuation2D(...)  # GridBasedMFGProblem

# Option 1: Both on grid (what we're using)
solver = create_basic_solver(problem, ...)  # HJB-FDM + FP-FDM

# Option 2: Hybrid (particles for FP)
from mfg_pde.config import ConfigBuilder
config = ConfigBuilder.for_problem(problem) \
    .with_fp_particle_solver(num_particles=4000) \
    .with_hjb_fdm_solver() \
    .build()
solver = PicardIterationSolver(problem, config)
```

The "broken-but-completed" implementation actually used **particle-based FP** (4000 particles with KDE reconstruction), which is why it had perfect mass conservation (0.0000% loss). The particle method worked fine - the issue was the problem structure not exposing 2D nature to HJB solver.

## Comparison: Broken vs Proper Implementation

| Aspect | Broken (Flattened) | Proper (nD) |
|:-------|:------------------|:------------|
| **Base Class** | Custom MFGProblem | GridBasedMFGProblem |
| **Problem Structure** | 900-point 1D | True 20×20 2D |
| **Hamiltonian Signature** | Complex (derivs dict) | Standard (gradient vector p) |
| **Dimension Detection** | Failed → 1D mode | Successful → 2D mode |
| **Error Convergence** | Poor/static | Exponential (÷2 per iteration) |
| **Iteration Time** | 10-17s (erratic) | 68s (consistent) |
| **FP Solver** | Particle-based (4000) | Grid-based FDM |
| **HJB Solver** | FDM 1D mode (wrong!) | FDM 2D mode (correct!) |
| **Evacuation Result** | 0% (minimal movement) | *(Still running)* |

## Phase 3 nD Support Status

From investigation of Phase 3 PR #217 (Nov 2, 2025):

**Production-Ready nD Capabilities**:
- ✅ HJB-FDM: Arbitrary nD (tested 1D-4D)
- ✅ FP-FDM: Full-dimensional system (no splitting)
- ✅ WENO schemes: Extended to arbitrary nD (tested 4D)
- ✅ Semi-Lagrangian: Extended to arbitrary nD (tested 3D)
- ✅ Particle interpolation: Extended to arbitrary nD (tested 5D)
- ✅ TensorProductGrid: Dimension-agnostic structure
- ✅ GridBasedMFGProblem: Standard interface for nD problems

**Known Issues** (from Phase 3):
- ⚠️ Config integration incomplete in HighDimMFGProblem methods
- ⚠️ `solve_with_damped_fixed_point()` uses old config interface
- ✅ **Workaround**: Use factory pattern `create_basic_solver()` instead

## Lessons Learned

### For Users

1. **Use GridBasedMFGProblem for nD problems** - Don't try to create custom MFGProblem with adapters
2. **Implement standard signatures** - `hamiltonian(x, m, p, t)` not complex signatures
3. **Use factory pattern** - `create_basic_solver()` handles dimension detection automatically
4. **Avoid MFGComponents for nD** - It's for 1D/legacy compatibility, not modern nD solvers
5. **Particle methods work fine** - Problem structure is what matters, not FP solver choice

### For Framework Developers

1. **GridBased2DAdapter is misleading** - It suggests you can adapt any problem to 2D, but you need proper base class
2. **Document standard vs complex signatures** - Make it clear when each should be used
3. **Deprecate HighDimMFGProblem.solve_*()** - Until config integration is complete
4. **Add dimension check warnings** - Warn if HJB solver detects different dimension than expected

## Next Steps

1. **Wait for proper nD simulation to complete** - Validate that true 2D solving produces better evacuation
2. **Compare evacuation dynamics** - Check if crowd properly moves toward exit with correct solver
3. **Parameter tuning** - Current setup may need longer time horizon or different costs
4. **Document pattern** - Create example showing correct GridBasedMFGProblem usage for 2D

## Code Locations

**Investigation Files**:
- `mfg_pde/core/highdim_mfg_problem.py:59-418` - GridBasedMFGProblem definition
- `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py:152-325` - nD HJB FDM solver
- `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py:383-600` - nD FP FDM solver
- `examples/basic/2d_crowd_motion_fdm.py` - Working 2D example

**Implementation Files**:
- `examples/outputs/particle_methods/2d_crowd_density_evolution.py` - Broken (flattened 1D)
- `examples/outputs/particle_methods/2d_crowd_proper_nd.py` - Correct (true nD)

**Documentation**:
- `docs/development/ND_SUPPORT_STATUS_2025-11-02.md` - Phase 3 nD capabilities
- `docs/development/HAMILTONIAN_SIGNATURE_ANALYSIS_2025-11-02.md` - Signature comparison

---

**Status**: Investigation complete, proper implementation running
**Key Finding**: GridBasedMFGProblem with standard signatures is the correct pattern for nD problems
**Remaining**: Validate evacuation dynamics with completed proper nD simulation
