# FDM Solver Limitation: Cannot Use with 2D Maze

**Date**: 2025-10-30
**Context**: Attempted pure FDM solver comparison for maze navigation problem
**Status**: BLOCKED by architectural limitation

---

## Problem Statement

User requested results from two MFG solver configurations:
1. **Pure FDM**: HJB-FDM + FP-FDM (baseline comparison)
2. **Hybrid**: HJB-FDM + FP-Particle

The maze navigation problem is 2D (6×6 maze on regular grid), but FDM solvers in MFG_PDE only work with 1D problems.

---

## Architecture Analysis

### What Exists in MFG_PDE

**1D Grid-Based**: `MFGProblem` (mfg_problem.py:70)
```python
class MFGProblem:
    def __init__(
        self,
        xmin: float = 0.0,    # 1D domain
        xmax: float = 1.0,
        Nx: int = 51,
        T: float = 1.0,
        Nt: int = 51,
        ...
    )
```

**2D/3D Grid-Based**: `GridBasedMFGProblem` (highdim_mfg_problem.py:351)
```python
class GridBasedMFGProblem(HighDimMFGProblem):
    def __init__(
        self,
        domain_bounds: tuple,          # e.g., (xmin, xmax, ymin, ymax) for 2D
        grid_resolution: int | tuple,  # e.g., 50 or (50, 50)
        time_domain: tuple = (1.0, 100),
        diffusion_coeff: float = 0.1,
    )
```

**FDM Solvers** (hjb_fdm.py:15, fp_fdm.py:15)
```python
class HJBFDMSolver(BaseHJBSolver):
    def __init__(self, problem: MFGProblem, ...)  # Only accepts 1D MFGProblem
```

### The Fragmentation

**Current Design**:
```
MFGProblem (1D)
    ├── HJBFDMSolver ✓ Works
    └── FPFDMSolver ✓ Works

GridBasedMFGProblem (2D/3D)
    ├── HJBFDMSolver ✗ Type mismatch
    ├── FPFDMSolver ✗ Type mismatch
    └── Workaround: create_1d_adapter_problem()  (highdim_mfg_problem.py:123)
```

**The "1D Adapter" Workaround** (highdim_mfg_problem.py:123-147):
```python
def create_1d_adapter_problem(self) -> MFGProblem:
    """
    Create a 1D adapter MFG problem for use with existing solvers.

    This method maps the high-dimensional problem to a 1D representation
    that can be used with the existing MFG_PDE solver infrastructure.
    """
    # Create 1D problem using linearized indexing of spatial points
    adapter_problem = MFGProblem(
        xmin=0.0,
        xmax=float(self.num_spatial_points - 1),  # Flatten 2D grid to 1D
        Nx=self.num_spatial_points - 1,
        T=self.T,
        Nt=self.Nt,
        sigma=self.sigma,
        components=self.components,
    )

    # Store reference to original high-dimensional problem
    adapter_problem._highdim_problem = self  # Backreference for reconstruction

    return adapter_problem
```

---

## Why This Approach is Problematic

### 1. **Semantic Mismatch**
The 1D adapter treats a 2D 50×50 grid (2500 points) as a 1D line with Nx=2499 intervals, destroying the geometric structure.

### 2. **FDM Assumptions Violated**
FDM solvers compute spatial derivatives using 1D stencils:
- `∂u/∂x` uses neighbors at `i-1, i, i+1` in 1D indexing
- But in 2D, neighbors should be `(i±1, j)` and `(i, j±1)` in grid coordinates
- The flattened indexing breaks the neighbor relationships

### 3. **Example**
```
2D Grid (3×3):        Flattened 1D (Nx=8):
(0,0) (1,0) (2,0)     0 1 2 3 4 5 6 7 8
(0,1) (1,1) (2,1)
(0,2) (1,2) (2,2)

Point (1,1) → index 4
True 2D neighbors: 1, 3, 5, 7 (cross pattern)
FDM 1D neighbors: 3, 5 (only left/right)
```

The FDM solver will compute `∂u/∂x` using indices `3, 4, 5`, which in the original 2D grid are `(0,1), (1,1), (2,1)` — a horizontal line, not a proper 2D derivative.

### 4. **Confirmed by Audit**
The architecture audit (MFG_PDE_ARCHITECTURE_AUDIT.md) identified this exact issue:
- **Finding #2**: "Five problem classes with incompatible APIs"
- **Finding #3**: "Solver-problem incompatibilities require workarounds"
- **Finding #5**: "1D adapter pattern breaks semantics"

---

## User Request: Why It Cannot Be Fulfilled

User asked for:
> "I still need these 2 mfg solvers to give a result"
> 1. pure fdm
> 2. hjb fdm + fp particle (hybrid)

**Fundamental Issue**:
- Pure FDM (HJB-FDM + FP-FDM) requires both solvers to work with a regular 2D grid
- MFG_PDE's FDM solvers are 1D-only by design
- The 1D adapter would produce mathematically incorrect results (wrong derivatives)

**User's Insight Was Correct**:
> "maze can still have grids under coordinate system ,right?"

Yes, mazes can have regular grids! The problem is not mathematical — it's architectural. MFG_PDE artificially separates:
- Problem definition (`GridBasedMFGProblem` for 2D grids)
- Solver implementation (`HJBFDMSolver` for 1D only)

---

## What Would Be Needed

### Option 1: Multi-Dimensional FDM Solvers
Create `HJB2DFDMSolver` and `FP2DFDMSolver` that:
- Accept `GridBasedMFGProblem`
- Implement proper 2D finite difference stencils
- Use 2D indexing: `u[i,j]` instead of `u[k]`

### Option 2: Unified Problem Class
Implement the architecture refactoring proposal:
- Abstract `MFGProblem` base class
- `GridMFGProblem(dimension, bounds, resolution)` for any dimension
- FDM solvers detect dimension and apply appropriate stencils

### Option 3: Workaround (Current Session)
Skip pure FDM comparison and note the architectural limitation. Focus on:
- HJB-GFDM + FP-Particle (working)
- Particle collocation methods (working)
- Document FDM limitation for future work

---

## What Currently Works

**Meshfree Methods**: HJB-GFDM + FP-Particle ✓
- GFDM (Generalized Finite Difference Method) is meshfree
- Works with arbitrary particle distributions
- No grid structure required
- Already tested in `demo_mfg_picard.py`

**Hybrid Approach**: HJB-FDM (1D) + FP-Particle ✗
- Would require 1D adapter → incorrect
- FP-Particle expects 2D particles
- Incompatible mixing

---

## Recommendations

### Immediate (This Session)
1. Document this architectural limitation
2. Skip pure FDM comparison (cannot produce correct results)
3. Focus on working solver combinations:
   - HJB-GFDM + FP-Particle (current baseline)
   - HJB-GFDM + FP-Particle with Anderson acceleration (tested)

### Near-Term (Post-Session)
1. Add to architecture refactoring proposal:
   - Priority: Dimension-agnostic FDM solvers
   - Estimated effort: 2-3 weeks
   - Blocker for: 2D/3D grid-based MFG with FDM

### Long-Term (Refactoring)
1. Implement unified `MFGProblem` architecture
2. Dimension-aware solver dispatch
3. Proper 2D/3D FDM stencils

---

## Session Log

**File**: test_solver_comparison.py
**Error**: `ModuleNotFoundError: No module named 'mfg_pde.alg.numerical.hjb_solvers.hjb_fd'`
**Corrected Import**: `hjb_fdm` (not `hjb_fd`)
**Result**: Discovered that even with correct import, solver architecture blocks 2D usage

**Verification**:
```bash
$ ls -la /path/to/MFG_PDE/mfg_pde/alg/numerical/hjb_solvers/ | grep hjb_fd
-rw-r--r-- hjb_fdm.py  # Correct module name
-rw-r--r-- hjb_gfdm.py  # Meshfree alternative (works for 2D)
```

**User Convention**:
> "not that Nx stands for number of intervals, not knots, in our convention"

This is honored in MFGProblem (Nx intervals → Nx+1 grid points), but the 1D-only design still blocks 2D grids.

---

## Conclusion

**Pure FDM comparison cannot be completed** due to MFG_PDE's architectural limitation: FDM solvers are 1D-only, while the maze problem is inherently 2D.

**User's request confirmed the architecture audit findings**: The fragmented design prevents natural use of grid-based solvers on 2D regular grids.

**Next steps**: Document this limitation and proceed with working solver comparisons (meshfree methods).
