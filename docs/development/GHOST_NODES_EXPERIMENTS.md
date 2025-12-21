# Ghost Nodes Experiments - Crowd Evacuation Validation

**Date**: 2025-12-20
**Status**: ⏳ **IN PROGRESS**
**Experiment**: Crowd Evacuation (Exp1 Baseline)
**Goal**: Validate Ghost Nodes method performance vs FDM baseline

---

## Experiment Configuration

### Solvers Being Tested

1. **FDM-FDM** (baseline) - Already complete
   - HJB: FDM with row replacement BC
   - FP: FDM with upwind advection
   - **Metrics**: 11.17 units displacement (100% baseline)

2. **GFDM-LCR-FDM** (rotation fix only) - Already complete
   - HJB: GFDM with Local Coordinate Rotation
   - FP: FDM with upwind advection
   - **Metrics**: 5.40 units displacement (48% of baseline)
   - **Issues**: 12% wrong-sign gradients, 7x gradient variance

3. **GFDM-Ghost-FDM** ⏳ **RUNNING NOW**
   - HJB: GFDM with Ghost Nodes (NEW!)
   - FP: FDM with upwind advection
   - **Expected**: ~10-11 units displacement (90-100% of baseline)

4. **GFDM-Ghost-Particle** ⏳ **QUEUED**
   - HJB: GFDM with Ghost Nodes
   - FP: Particle method
   - **Expected**: Similar to GFDM-Ghost-FDM with particle noise

---

## Implementation Details

### Ghost Nodes Configuration

```python
hjb_solver = HJBGFDMSolver(
    problem,
    collocation_points=coll.points,  # 625 interior + 100 boundary = 725 total
    boundary_indices=coll.boundary_indices,
    delta=3.0 * avg_spacing,  # δ ≈ 2.13 for this problem
    qp_optimization_level="none",
    adaptive_neighborhoods=True,
    use_ghost_nodes=True,  # Enable Ghost Nodes for Neumann BC
)
```

### Expected Ghost Node Statistics

- **Boundary points**: 100
- **Ghost nodes per boundary point**: ~10-15
- **Total ghost neighbors**: ~1000-1500 (negligible memory overhead)
- **BC enforcement**: Structural (symmetric stencils), no row replacement

---

## Files Created/Modified

### MFG_PDE Repository

**Core Implementation**:
- `mfg_pde/alg/numerical/hjb_solvers/hjb_gfdm.py`:
  - Added `use_ghost_nodes` parameter
  - Implemented `_create_ghost_neighbors()` (lines 887-948)
  - Implemented `_apply_ghost_nodes_to_neighborhoods()` (lines 950-1020)
  - Implemented `_get_values_with_ghosts()` (lines 1022-1060)
  - Modified `approximate_derivatives()` (line 1683)
  - Modified BC enforcement to skip row replacement (line 2140)

**Tests & Validation**:
- `scripts/verify_rotation_matrix_fix.py` - Rotation matrix validation
- `scripts/test_ghost_nodes_smoke.py` - Ghost nodes smoke test
- `scripts/visualize_ghost_nodes.py` - Visual proof of concept
- `tests/unit/test_hjb_gfdm_rotation.py` - Unit tests for rotation

**Documentation**:
- `docs/development/LCR_BOUNDARY_BUG_ANALYSIS.md` - Rotation bug analysis
- `docs/development/GHOST_NODES_IMPLEMENTATION.md` - Implementation guide
- `docs/development/GHOST_NODES_EXPERIMENTS.md` - This file

### MFG-Research Repository

**Experiments**:
- `experiments/crowd_evacuation_2d/experiments/exp1_baseline_fdm.py`:
  - Added `run_gfdm_ghost_fdm()` function (lines 653-725)
  - Added `run_gfdm_ghost_particle()` function (lines 728-800)
  - Updated solver dispatch (lines 1050-1053)
  - Updated argparse choices (lines 1151-1156)

**Expected Outputs**:
- `results/exp1_baseline/solution_gfdm_ghost_fdm.h5` - Solution data
- `results/exp1_baseline/metrics_gfdm_ghost_fdm.json` - Performance metrics
- `results/exp1_baseline/solution_gfdm_ghost_particle.h5` - Solution data
- `results/exp1_baseline/metrics_gfdm_ghost_particle.json` - Performance metrics

---

## Running the Experiments

### Already Running

```bash
cd mfg-research/experiments/crowd_evacuation_2d
python experiments/exp1_baseline_fdm.py --solver gfdm_ghost_fdm
```

**Status**: ⏳ Running in background (task ID: b8b3f28)

### Next to Run

```bash
python experiments/exp1_baseline_fdm.py --solver gfdm_ghost_particle
```

---

## Expected Results

### Performance Metrics

| Metric | FDM (Baseline) | GFDM-LCR | GFDM-Ghost (Expected) |
|--------|----------------|----------|----------------------|
| **Displacement** | 11.17 units | 5.40 units (48%) | ~10-11 units (90-100%) |
| **Wrong-sign gradients** | 0% | 12% | ~0-2% |
| **Gradient std** | 1.98 | 13.57 (7x) | ~3-5 (1.5-2.5x) |
| **Newton convergence** | 50 iters | 30 iters (oscillatory) | ~20-30 iters (stable) |
| **Time** | 226 s | 930 s | ~900-1000 s (similar) |

### Success Criteria

✅ **Minimum Success**:
- Displacement ≥ 80% of FDM baseline
- Wrong-sign gradients < 5%
- Gradient std < 4x FDM baseline
- Stable Newton convergence (no oscillations)

🎯 **Target Success**:
- Displacement ≥ 90% of FDM baseline
- Wrong-sign gradients < 2%
- Gradient std < 2.5x FDM baseline
- Solution quality comparable to FDM

---

## Analysis Plan

Once experiments complete:

1. **Load and compare solutions**:
   ```python
   import h5py
   with h5py.File('results/exp1_baseline/solution_gfdm_ghost_fdm.h5', 'r') as f:
       U_ghost = f['U'][:]
       M_ghost = f['M'][:]
   ```

2. **Compute displacement metrics**:
   - Center of mass trajectory
   - Final position vs exit
   - Total displacement

3. **Gradient quality analysis**:
   - Gradient statistics at t=0 (terminal time)
   - Wrong-sign gradient count
   - Boundary gradient accuracy

4. **Convergence analysis**:
   - Newton iteration count
   - Error history
   - Temporal stability

5. **Visual comparison**:
   - Density evolution comparison
   - Velocity field comparison
   - Gradient field comparison

---

## Next Steps

1. ⏳ **Monitor GFDM-Ghost-FDM** (currently running)
2. **Run GFDM-Ghost-Particle** when FDM complete
3. **Analyze results** and create comparison plots
4. **Document findings** in experiment report
5. **Update STRATEGIC_DEVELOPMENT_ROADMAP** if successful
6. **Publish results** if performance gains confirmed

---

## Technical Notes

### Ghost Node Statistics from Initialization

From smoke test output:
```
Ghost nodes created for 40 boundary points
  Example (boundary point 128):
    Number of ghosts: 12
    Ghost indices: [-1280000 -1280001 -1280002]...
    Mirror indices: [23 39 71]...
```

For the actual crowd evacuation problem (100 boundary points, δ ≈ 2.13):
- Expected ghosts per boundary point: ~10-15
- Total augmented neighbors: ~1000-1500
- Memory overhead: Negligible (<1% of total storage)

### Computational Cost

Ghost nodes add:
- **Neighborhood augmentation**: One-time cost during initialization (~0.1s)
- **Value mapping overhead**: ~10% per derivative evaluation
- **No additional Newton iterations** (if BC enforcement works)
- **Overall**: Expected 5-15% slowdown vs non-ghost GFDM

---

**Status**: Experiments in progress. Check back for results.
**Estimated completion**: ~15-30 minutes per solver
