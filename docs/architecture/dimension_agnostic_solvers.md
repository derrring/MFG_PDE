# Dimension-Agnostic MFG Solver Landscape

**Purpose**: Comprehensive overview of dimension-agnostic approaches for Mean Field Games in MFGarchon
**Date**: 2025-10-31
**Status**: Living document

---

## Executive Summary

MFGarchon supports multiple strategies for solving Mean Field Games in arbitrary dimensions (2D, 3D, 4D, ...). This document categorizes approaches by:
- **Grid structure**: Regular vs irregular vs dynamic
- **Dimension handling**: Direct vs dimensional splitting
- **Implementation status**: Production vs research

**Key Principle**: Dimension should be a parameter, not a constraint.

---

## Taxonomy of Approaches

### Classification by Grid Adaptation

```
┌──────────────────────────────────────────────────────────────┐
│               MFG Solver Dimension-Agnostic Methods           │
└──────────────────────────────────────────────────────────────┘

1. FIXED REGULAR GRID
   ├─ FDM 1D (production)
   │  └─ Files: mfgarchon/alg/numerical/{hjb,fp}_solvers/base_{hjb,fp}.py
   │  └─ Direct finite difference on uniform 1D grid
   │
   ├─ FDM nD via Full Coupled System (production - FP only)
   │  └─ Files: mfgarchon/alg/numerical/fp_solvers/fp_fdm.py (_solve_fp_nd_full_system)
   │  └─ Direct sparse linear system for nD FP equation
   │  └─ No splitting error, ~1-2% mass error
   │
   └─ FDM nD via Dimensional Splitting (DEPRECATED - archived)
      └─ Files: docs/archived_methods/dimensional_splitting/code/
      └─ FAILURE: -81% mass loss for MFG with advection
      └─ See: docs/archived_methods/dimensional_splitting/README.md

2. FIXED IRREGULAR GRID (Meshfree Eulerian)
   ├─ GFDM (production)
   │  └─ Files: mfgarchon/alg/numerical/hjb_solvers/gfdm_*.py
   │  └─ Generalized Finite Difference on scattered points
   │
   └─ Adaptive Eulerian (research, external)
      └─ Location: mfg-research/experiments/maze_navigation/
      └─ Lagrangian adaptation → fixed collocation grid

3. FULLY DYNAMIC GRID (Meshfree Lagrangian)
   └─ Fully Lagrangian Particle Methods (research, external)
      └─ Location: mfg-research/algorithms/particle_collocation/
      └─ Particles carry solution, move throughout solve
```

---

## Method 1: FDM nD with Full Coupled System (Production - FP Only)

### Overview

**Status**: ✅ Production-ready for FP equation, ❌ HJB not yet implemented
**Approach**: Direct sparse linear system for full nD discretization
**Key Idea**: Solve coupled advection-diffusion system without operator splitting
**Implementation**: `mfgarchon/alg/numerical/fp_solvers/fp_fdm.py:383` (`_solve_fp_nd_full_system`)

### Algorithm

For each timestep forward from 0 to T:
1. **Compute gradients**: ∇U(t,x) from value function (all dimensions)
2. **Compute velocity**: v = -coupling_coefficient · ∇U (advection field)
3. **Assemble system**: (I/Δt + A + D) where:
   - A: full nD advection operator with upwinding
   - D: full nD diffusion operator (Laplacian)
4. **Solve**: Sparse linear system m^{n+1} = solve(system_matrix, rhs)
5. **Enforce**: m ≥ 0 (non-negativity)

### Advantages

✅ **No splitting error**: Direct discretization preserves operator coupling
✅ **Mass conservative**: ~1-2% error (normal FDM accuracy)
✅ **Stable for high Pe**: Modern sparse solvers handle advection-diffusion
✅ **Dimension-agnostic**: Works for 2D, 3D, 4D+ automatically

### Limitations

⚠️ **Memory**: O(N^d) sparse matrix storage
⚠️ **Computational cost**: O(N^d) unknowns per timestep
⚠️ **HJB not implemented**: Currently FP only, HJB needs separate implementation

### When to Use

**Good for**:
- Multi-dimensional MFG problems (2D, 3D)
- Advection-dominated flows (high Péclet number)
- Production code requiring mass conservation

**Not recommended for**:
- Very high dimensions (d > 4) due to curse of dimensionality
- HJB equation (use GFDM or Semi-Lagrangian instead)

---

## Method 1b: FDM with Dimensional Splitting (DEPRECATED)

### Overview

**Status**: ⚠️ **DEPRECATED - ARCHIVED**
**Reason**: Catastrophic failure (-81% mass loss) for MFG with advection
**Archived**: `docs/archived_methods/dimensional_splitting/`

**Approach**: Operator splitting in space (Strang splitting)
**Key Idea**: Solve nD problem via alternating 1D sweeps
**Implementation**: Archived code (not recommended for use)
  - `hjb_fdm_multid.py`: HJB backward solve (archived)
  - `fp_fdm_multid.py`: FP forward solve (archived)
  - Failure documented in `docs/archived_methods/dimensional_splitting/README.md`

### Algorithm (2D Example)

For each timestep backward from T to 0:
1. **Half-step in x**: Solve 1D HJB along each x-line for Δt/2
2. **Full-step in y**: Solve 1D HJB along each y-line for Δt
3. **Half-step in x**: Solve 1D HJB along each x-line for Δt/2

**Generalization to d dimensions**:
- Forward sweeps: dimensions 0, 1, ..., d-1 (half timestep each)
- Backward sweeps: dimensions d-1, ..., 1, 0 (half timestep each)

### Advantages

✅ **Simple**: Reuses existing 1D FDM solver completely
✅ **Efficient**: O(d·N^(d-1)) 1D solves per timestep vs O(N^d) full nD solve
✅ **Automatic**: Works for any dimension with no code changes
✅ **Classical**: Well-understood numerical method (Strang 1968)

### Limitations

❌ **Curse of Dimensionality (CoD)**: Requires O(N^d) grid storage - **fundamentally impractical for d > 4**
   - D=2, N=100: 10,000 points (10 KB) ✅
   - D=3, N=50: 125,000 points (1 MB) ✅
   - D=4, N=30: 810,000 points (6 MB) ⚠️ Marginal
   - D=5, N=20: 3.2M points (25 MB) ❌ Impractical
   - D=10, N=10: 10 billion points (80 GB) ❌ Impossible

⚠️ **Splitting error**: O(Δt²) error for cross-derivative terms (Strang splitting)
⚠️ **Isotropic Hamiltonians**: Exact only for H = (1/2)|p|²
⚠️ **Anisotropic problems**: Approximate for H with cross-terms like p₁p₂
⚠️ **Mixed derivatives**: Cannot handle ∂²u/∂x∂y terms cleanly (common in finance)

### When to Use

⚠️ **NOT RECOMMENDED FOR ANY USE** ⚠️

**Historical context only**:
- Method was initially attractive for reusing 1D code
- Works perfectly for pure diffusion problems
- Fails catastrophically for MFG with advection

**Why it fails**:
- High Péclet number (advection >> diffusion) in MFG
- Operator non-commutativity: [∇·(mv), Δm] ≠ 0
- Error compounds over Picard iterations
- Result: -81% mass loss in realistic test cases

**Use instead**:
- FP: Full nD coupled system (`_solve_fp_nd_full_system`)
- HJB: GFDM or Semi-Lagrangian methods

### Implementation Details

**Files**:
- `mfgarchon/alg/numerical/hjb_solvers/hjb_fdm_multid.py`: Core splitting algorithm
- `mfgarchon/alg/numerical/hjb_solvers/hjb_fdm.py`: Dimension detection and routing

**Key Functions**:
```python
solve_hjb_nd_dimensional_splitting(M_density, U_final, U_prev, problem, ...)
    # Main entry point for nD FDM with splitting

_sweep_dimension(U, M, problem, dt, dim, ...)
    # Sweep along dimension `dim` using itertools.product()

_solve_1d_hjb_slice(U_slice, M_slice, problem, dt, dim, slice_indices, ...)
    # Solve single 1D HJB problem for one slice

_Problem1DAdapter(full_problem, sweep_dim, fixed_indices)
    # Adapter class: makes nD GridBasedMFGProblem look like 1D MFGProblem
```

**Complexity**:
- **Per timestep**: d sweeps (forward + backward)
- **Per sweep**: N^(d-1) independent 1D solves
- **Per 1D solve**: O(N × Newton_iterations)
- **Total**: O(d × N^d × Newton_iterations) per timestep

**Example (3D, 30×30×30 grid)**:
- 3 dimensions → 6 sweeps per timestep
- 30² = 900 independent 1D solves per sweep
- ~5,400 1D solves per timestep
- Each 1D solve is embarrassingly parallel (future optimization)

### Theory

**Strang Splitting Formula**:

For PDE: ∂u/∂t = L_x(u) + L_y(u) + L_z(u)

Approximate solution operator:
```
U(t + Δt) ≈ S_x(Δt/2) ∘ S_y(Δt/2) ∘ S_z(Δt) ∘ S_y(Δt/2) ∘ S_x(Δt/2) U(t)
```

where S_i(Δt) is the solution operator for dimension i.

**Splitting Error**:
- Symmetric splitting: O(Δt²) global error
- Exact for problems with L_x L_y = L_y L_x (commuting operators)
- For HJB: exact when Hamiltonian has no cross-derivatives

**References**:
- Strang, G. (1968). "On the construction and comparison of difference schemes"
- Marchuk, G. I. (1990). "Splitting and alternating direction methods"

---

## Method 2: GFDM (Generalized Finite Difference Method)

### Overview

**Approach**: Meshfree method using local polynomial approximation
**Key Idea**: Compute derivatives from scattered neighbors via least squares
**Implementation**: Production-ready in MFGarchon

### Algorithm

For each point x_i and each timestep:
1. **Find neighbors**: k nearest neighbors of x_i
2. **Local polynomial**: Fit polynomial p(x) ≈ u(x) near x_i
3. **Derivatives**: Evaluate ∇p(x_i), Δp(x_i) from polynomial
4. **PDE solve**: Use derivatives in HJB/FP discretization

### Advantages

✅ **Exact for cross-derivatives**: No splitting approximation
✅ **Flexible geometry**: Works on irregular domains, obstacles
✅ **Dimension-agnostic**: Same code for 2D, 3D, 4D, ...
✅ **Production-ready**: Tested and validated
✅ **Adaptive**: Can use non-uniform point distributions

### Limitations

⚠️ **Tuning required**: k_neighbors parameter affects stability
⚠️ **Slower**: More complex than FDM per point
⚠️ **Conditioning**: Ill-conditioned for very irregular point clouds

### When to Use

**Good for**:
- Anisotropic MFG problems (cross-derivatives important)
- Irregular domains (obstacles, curved boundaries)
- Production code requiring robustness
- Adaptive mesh refinement

**Not recommended for**:
- Very high dimensions (d > 4) due to curse of dimensionality
- Real-time applications (slower than FDM splitting)

### Implementation Details

**Files**:
- `mfgarchon/alg/numerical/hjb_solvers/gfdm_hjb.py`: HJB solver
- `mfgarchon/alg/numerical/fp_solvers/gfdm_fp.py`: FP solver
- `mfgarchon/geometry/meshfree/`: Geometry utilities

**Key Configuration**:
```python
from mfgarchon.alg.numerical.hjb_solvers import GFDMHJBSolver

solver = GFDMHJBSolver(
    problem,
    k_neighbors=20,  # More neighbors = more stable but slower
    polynomial_degree=2,  # Usually 2 is sufficient
)
```

**Complexity**:
- **Neighbor search**: O(N log N) with KD-tree
- **Per point**: O(k³) for polynomial fit (k = k_neighbors)
- **Total per timestep**: O(N × k³)

### Theory

**Local Polynomial Approximation**:

Near point x_i, approximate u(x) by polynomial:
```
p(x) = c₀ + c₁(x-x_i) + c₂(y-y_i) + c₃(x-x_i)² + c₄(x-x_i)(y-y_i) + ...
```

Determine coefficients {c_j} by minimizing:
```
∑_{x_j ∈ neighbors} w_j [u(x_j) - p(x_j)]²
```

where w_j are weights (e.g., inverse distance).

**Derivative Computation**:
- ∂u/∂x(x_i) = c₁
- ∂u/∂y(x_i) = c₂
- ∂²u/∂x²(x_i) = 2c₃
- ∂²u/∂x∂y(x_i) = c₄

**References**:
- Benito et al. (2001). "Influence of several factors in the generalized finite difference method"
- Gavete et al. (2003). "Generalized finite differences for solving 3D elliptic and parabolic equations"

---

## Method 3: Adaptive Eulerian (Research, External)

### Overview

**Approach**: Hybrid Lagrangian-Eulerian
**Key Idea**: Use particles to find good point distribution, then freeze for PDE solve
**Implementation**: Research code in mfg-research repository

**Location**: `mfg-research/experiments/maze_navigation/adaptive_eulerian_mfg_solver.py`

### Algorithm

**Phase 1: Lagrangian Adaptation**
1. Initialize particles from initial density m₀
2. Evolve particles following FP equation (Lagrangian)
3. Particles cluster where density is high
4. Result: Adapted point cloud reflecting solution structure

**Phase 2: Eulerian Solve**
1. Extract particle positions → collocation points
2. **Freeze positions** (now Eulerian)
3. Build meshfree discretization (GFDM or RBF)
4. Solve HJB/FP on fixed collocation grid

**Phase 3: Iterate**
1. Use solution to update particle distribution
2. Repeat until convergence

### Advantages

✅ **Automatic refinement**: Points concentrate where needed
✅ **Efficient**: Fewer points than uniform grid
✅ **Dimension-agnostic**: Particles work in any dimension
✅ **Stable**: Fixed grid during PDE solve (Eulerian stability)

### Limitations

⚠️ **Research-stage**: Not production-ready
⚠️ **Tuning**: Particle initialization and adaptation parameters
⚠️ **Complexity**: More moving parts than pure Eulerian

### When to Use

**Good for**:
- Research on adaptive methods
- Problems with localized features (shocks, fronts)
- Efficient high-dimensional solvers

**Not recommended for**:
- Production code (use GFDM instead)
- Simple problems (overhead not worth it)

### Implementation Notes

**Files in mfg-research**:
- `adaptive_eulerian_mfg_solver.py`: Main solver
- `adaptive_sampling.py`: Particle distribution utilities
- `collocation_grid_adapter.py`: Particle → collocation conversion

**Key Innovation**: Combines benefits of Lagrangian (adaptivity) and Eulerian (stability)

---

## Method 4: Fully Lagrangian Particle Methods (Research, External)

### Overview

**Approach**: Fully Lagrangian meshfree
**Key Idea**: Particles carry solution values, move throughout solve
**Implementation**: Research code in mfg-research repository

**Location**: `mfg-research/algorithms/particle_collocation/`

> **Note**: For basic meshfree density evolution in MFGarchon, use `FPGFDMSolver`.
> This section describes more advanced fully-Lagrangian research methods.

### Algorithm

**FP Solve (Lagrangian)**:
1. Particles represent density m(t,x)
2. Move particles according to optimal control
3. Update particle weights via FP dynamics

**HJB Solve (Meshfree Interpolation)**:
1. Store value function on particles
2. Interpolate u(x) from nearby particles (RBF or k-NN)
3. Compute ∇u via interpolation
4. Update u via HJB equation

### Advantages

✅ **Fully adaptive**: Grid moves with solution
✅ **Natural for transport**: Particles follow characteristics
✅ **Dimension-agnostic**: Just N particles in ℝ^d

### Limitations

⚠️ **Experimental**: Active research
⚠️ **Stability**: Interpolation errors accumulate
⚠️ **Complexity**: Neighbor search, weight management

### When to Use

**Good for**:
- Research on novel methods
- Problems with strong transport
- Exploring fully Lagrangian approaches

**Not recommended for**:
- Production code
- Benchmarking (too many parameters)

---

## Comparison Matrix

| Method | Grid Type | Cross-Deriv | Complexity | Status | Use Case |
|--------|-----------|-------------|------------|--------|----------|
| **FDM Full nD (FP)** | Regular | Exact | Medium | ✅ Prod | 2D/3D FP equation |
| **FDM Splitting** | Regular | Fails | Low | ⚠️ DEPRECATED | ~~NONE~~ (archived) |
| **GFDM** | Irregular fixed | Exact | Medium | ✅ Prod | Robust production |
| **Semi-Lagrangian** | Regular | Exact | Medium | ✅ Prod | HJB equation (1D/2D) |
| **Adaptive Eulerian** | Irregular fixed | Exact | Medium-High | 🔬 Research | Efficient adaptive |
| **Particle Dual** | Dynamic | Exact | High | 🔬 Research | Novel Lagrangian |

---

## Performance Characteristics

### Computational Cost (per timestep)

**2D Problem, 100×100 grid (N=10,000 points)**:

| Method | Operations | Relative Speed |
|--------|------------|----------------|
| FDM Splitting | 2 × 100 × 100 1D solves | 1.0× (baseline) |
| GFDM | 10,000 × k³ polynomial fits | 2-5× slower |
| Adaptive Eulerian | Particle moves + GFDM | Varies (3-10×) |
| Particle Dual | Interpolation + transport | 5-20× slower |

**3D Problem, 50×50×50 grid (N=125,000 points)**:

| Method | Memory | Time per step |
|--------|--------|---------------|
| FDM Splitting | O(N) = 1 MB | ~0.5s |
| GFDM | O(N×k) = 10 MB | ~2s |

### Accuracy Comparison

**Isotropic Test Problem** (H = (1/2)|p|²):

| Method | L² Error | Order |
|--------|----------|-------|
| FDM Splitting | ~1e-4 | O(Δx² + Δt²) |
| GFDM | ~1e-4 | O(Δx²) |

**Anisotropic Test Problem** (H = (1/2)(p₁² + 2ρp₁p₂ + p₂²)):

| Method | L² Error | Order |
|--------|----------|-------|
| FDM Splitting | ~1e-3 | O(Δt²) splitting error |
| GFDM | ~1e-4 | O(Δx²) |

---

## Design Philosophy: Why Multiple Methods?

### Principle 1: No One-Size-Fits-All

Different MFG problems have different needs:
- **Simple crowd dynamics** → FDM splitting (fast, simple)
- **Complex anisotropic** → GFDM (exact, robust)
- **Research exploration** → Adaptive/particle (novel, flexible)

### Principle 2: Classical + Modern

- **Classical methods** (FDM) provide trusted baselines
- **Modern methods** (GFDM, particles) push research frontier
- Having both enables validation and comparison

### Principle 3: Production + Research

- **Production code** (MFGarchon): Stable, tested, documented
- **Research code** (mfg-research): Experimental, flexible, evolving
- Clear separation prevents research instability from affecting production

---

## Dimension-Agnostic Infrastructure

### Common Components (Dimension-Independent)

**Problem Classes**:
```python
# 1D
from mfgarchon.core.mfg_problem import MFGProblem

# nD (arbitrary dimension)
from mfgarchon.core.highdim_mfg_problem import (
    HighDimMFGProblem,      # Abstract base
    GridBasedMFGProblem,    # Concrete: regular grids
)
```

**Geometry**:
```python
from mfgarchon.geometry.grids.tensor_grid import TensorProductGrid

# Works for any dimension
grid_2d = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx_points=[50, 50])
grid_3d = TensorProductGrid(dimension=3, bounds=[(0,1)]*3, Nx_points=[30]*3)
grid_4d = TensorProductGrid(dimension=4, bounds=[(0,1)]*4, Nx_points=[10]*4)
```

**Solver Interface**:
```python
# All solvers inherit from BaseHJBSolver
solver = HJBFDMSolver(problem)  # Automatically detects dimension
solver.solve_hjb_system(M_density, U_final, U_prev)  # Works for 1D/2D/3D/...
```

### Convention: Arrays Exclude Right Boundary

**GridBasedMFGProblem arrays**:
- Grid has `num_points[d]` points per dimension (e.g., 10 points)
- Arrays have shape `num_points[d] - 1` (e.g., 9 values)
- Interpretation: x₀, x₁, ..., x₈ (includes left boundary, excludes right)

**Rationale**: Matches FDM interior-point convention, simplifies Dirichlet BCs

---

## Selection Guide

### Decision Tree

```
START: Need to solve nD MFG problem
  |
  ├─ Which equation? FP or HJB?
  │   FP → Use FDM Full nD (production, 2D/3D)
  │   HJB ↓
  │
  ├─ What dimension?
  │   1D → Use FDM 1D (base_hjb.py)
  │   2D/3D ↓
  │
  ├─ Have irregular domain or obstacles?
  │   YES → Use GFDM (flexible geometry)
  │   NO ↓
  │
  ├─ Need pure FDM for regular grid?
  │   YES → Use Semi-Lagrangian (characteristic-based) or full nD HJB FDM (future)
  │   NO → Use GFDM (works on regular grids too)
  │
  ├─ Research on adaptive methods?
  │   YES → Explore Adaptive Eulerian or Particle Collocation
  │   NO → Use production methods (GFDM/Semi-Lagrangian)
  │
  └─ Need real-time performance?
      YES → Consider GFDM with coarse grid or Semi-Lagrangian
      NO → GFDM or Semi-Lagrangian acceptable
```

### Recommended Starting Point by Use Case

| Use Case | Recommended Method | Alternative |
|----------|-------------------|-------------|
| 2D/3D FP equation | FDM Full nD | GFDM |
| 2D/3D HJB equation | GFDM | Semi-Lagrangian |
| Production solver | GFDM | Semi-Lagrangian |
| Anisotropic problem | GFDM | Adaptive Eulerian (research) |
| Irregular domain | GFDM | - |
| Research on adaptation | Adaptive Eulerian | Particle Collocation |
| Novel Lagrangian methods | Particle Collocation | - |

---

## Future Directions

### Completed (2025-10-31 to 2025-11-01)

- ✅ **FP FDM nD**: Full coupled system solver (replaces dimensional splitting)
- ✅ **Bug Discovery**: Identified catastrophic failure of dimensional splitting (-81% mass loss)
- ✅ **Archival**: Moved dimensional splitting to `docs/archived_methods/`
- ✅ **Semi-Lagrangian 2D**: Fixed TensorProductGrid API issues
- ✅ **Documentation**: Updated to reflect deprecated methods

### In Progress (2025-11)

- 🔄 **Semi-Lagrangian nD**: Extending Semi-Lagrangian to arbitrary dimensions (current work)
- 🔄 **Testing**: Validating FDM Full nD vs GFDM on benchmark problems

### Next Steps (Phase 3)

- ⏳ **HJB FDM nD**: Full coupled system for HJB (like FP, without splitting)
- ⏳ **Performance**: Benchmark 2D/3D solvers
- ⏳ **User Guide**: Comprehensive multidimensional MFG tutorial
- ⏳ **Validation suite**: Automated comparison of FDM vs GFDM vs Semi-Lagrangian

### Long Term (6-12 months)

- 🔮 **Hybrid methods**: Combine FDM (regular regions) + GFDM (irregular boundaries)
- 🔮 **4D+ optimization**: Sparse grids, reduced-order models
- 🔮 **GPU acceleration**: Port GFDM to GPU for large-scale problems
- 🔮 **Adaptive FDM**: h-refinement for dimensional splitting

---

## Related Documentation

### MFGarchon

- `docs/architecture/README.md`: Overall architecture
- `docs/architecture/proposals/DIMENSION_AGNOSTIC_FDM_ANALYSIS.md`: FDM splitting design
- `docs/architecture/proposals/PHASE_2_SHORT_TERM_PLAN.md`: Implementation timeline

### mfg-research

- `experiments/maze_navigation/FP_SOLVER_ARCHITECTURE_ANALYSIS.md`: FP solver analysis
- `experiments/maze_navigation/docs/theory/meshfree_mfg_foundations.md`: Meshfree theory
- `experiments/maze_navigation/docs/theory/advanced_mfg_coupling.md`: MFG coupling

---

## References

### Dimensional Splitting

- Strang, G. (1968). "On the construction and comparison of difference schemes." *SIAM Journal on Numerical Analysis*, 5(3), 506-517.
- Marchuk, G. I. (1990). *Splitting and alternating direction methods*. Handbook of Numerical Analysis, 1, 197-462.

### Meshfree Methods

- Benito, J. J., et al. (2001). "Influence of several factors in the generalized finite difference method." *Applied Mathematical Modelling*, 25(12), 1039-1053.
- Gavete, L., et al. (2003). "Generalized finite differences for solving 3D elliptic and parabolic equations." *Applied Mathematical Modelling*, 27(4), 271-287.

### Mean Field Games

- Lasry, J. M., & Lions, P. L. (2007). "Mean field games." *Japanese Journal of Mathematics*, 2(1), 229-260.
- Cardaliaguet, P. (2013). "Notes on mean field games." Lecture notes, Université Paris-Dauphine.

---

**Document Version**: 1.0
**Last Updated**: 2025-10-31
**Maintainer**: MFGarchon Development Team
