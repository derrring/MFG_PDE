# Southwest Corner Issue - Ghost Nodes Analysis

**Date**: 2025-12-20
**Status**: 🔍 **INVESTIGATION IN PROGRESS**
**Context**: Ghost nodes solve northwest corner but not southwest corner

---

## Problem Statement

Ghost Nodes method shows **asymmetric performance** between corners:
- **Northwest corner (x=0, y=10)**: ✅ 59% → 0% wrong-sign gradients (COMPLETELY SOLVED)
- **Southwest corner (x=0, y=0)**: ⚠️ 41% → 31% wrong-sign gradients (PARTIALLY IMPROVED)

This creates a **25% residual error** concentrated in the southwest corner region.

---

## Regional Analysis

### Wrong-Sign Gradient Distribution

| Region | FDM | GFDM-LCR | GFDM-Ghost | Improvement |
|:-------|:----|:---------|:-----------|:------------|
| **Southwest (x<5, y<5)** | 0% | 41% | **31%** | 10 pp reduction (76% of LCR remains) |
| **Northwest (x<5, y>5)** | 0% | 59% | **0%** | 59 pp reduction (100% solved) ✅ |
| **Southeast (x>15, y<5)** | 0% | 0% | 0% | N/A |
| **Northeast (x>15, y>5)** | 0% | 1% | 0% | 1 pp reduction |
| **West boundary (x<2)** | 0% | 67.9% | **16.7%** | 51.2 pp reduction (75% improvement) |
| **South boundary (y<2)** | 0% | 10.4% | **7.9%** | 2.5 pp reduction (24% improvement) |

### Key Observations

1. **Perfect NW solution**: Ghost nodes eliminate 100% of wrong-sign gradients at northwest corner
2. **Problematic SW corner**: 31% wrong-sign gradients remain (vs 0% in NW)
3. **Directional bias**: South boundary (7.9%) worse than East/North (0%)
4. **Corner interaction**: SW region combines issues from both West and South boundaries

---

## Grid Coverage Analysis

**Verified**: Both FDM and GFDM have identical grid coverage:
- Southwest corner (x<3, y<3): 36 grid points (6×6)
- Corner point (0,0): ✅ Explicitly included
- Grid spacing: 0.5 units (uniform)

**Conclusion**: Issue is NOT from missing collocation points or inadequate coverage.

---

## Hypotheses for SW Corner Failure

### 1. Corner Normal Direction
- **Northwest corner**: Normal points outward at angle ≈225° (into third quadrant)
- **Southwest corner**: Normal points outward at angle ≈225° (same direction)
- **Impact**: If corner normal computation differs, ghost node placement would differ

### 2. Terminal Cost Gradient Direction
Terminal cost: `g(x,y) = ||x - (20,5)||²`

At corners:
- **NW (0,10)**: ∇g points toward (20,5), angle ≈ -11° (southeast)
- **SW (0,0)**: ∇g points toward (20,5), angle ≈ +14° (northeast)

**Critical difference**: SW gradient points **away from both boundaries**, while NW gradient points **parallel to north boundary**.

### 3. Ghost Node Mirror Selection
- Ghosts are created by reflecting **interior neighbors** across boundary
- At SW corner, interior neighbors are at angles ≈45-135° (northeast quadrant)
- At NW corner, interior neighbors are at angles ≈270-360° (southeast quadrant)
- **Different neighbor geometries** → **different ghost quality**

### 4. Neumann BC vs Terminal Cost Conflict
Neumann BC requires: ∂u/∂n = 0 at boundary

But terminal cost creates:
- ∂g/∂n at NW corner: ≈ 0 (gradient nearly tangent to boundary)
- ∂g/∂n at SW corner: ≈ 14.1 (gradient strongly normal to boundary)

**SW corner has stronger BC incompatibility** → harder for ghost nodes to enforce

### 5. Stencil Quality Near Corners
- Corners have **fewest interior neighbors** (≈90° sector available)
- Ghost nodes double the neighborhood, but mirror quality depends on:
  - Distance of interior neighbors from corner
  - Angle distribution of interior neighbors
  - Taylor expansion accuracy in reflected direction

---

## Diagnostic Evidence Needed

To identify root cause, we need:

1. **Corner normal vectors**: Compare ∇n at (0,0) vs (0,10)
2. **Ghost node count**: Compare # ghosts created at SW vs NW corner
3. **Mirror point distribution**: Visualize ghost-mirror pairs at both corners
4. **Symmetry errors**: Check ||ghost_offset + mirror_offset|| at each corner
5. **Terminal cost gradients**: Compute ∇g · n at both corners

---

## Visualization Results

Created visualizations in `results/exp1_baseline/`:
- `southwest_boundary_analysis.png`: Spatial distribution of wrong-sign gradients
- `sw_corner_value_comparison.png`: Value function differences at SW corner
- `ghost_nodes_comprehensive_comparison.png`: Overall performance comparison

**Next**: Create corner-specific ghost node analysis to compare SW vs NW.

---

## Proposed Fixes

### Short-term (Diagnostic)
1. ✅ Run `analyze_corner_ghost_nodes.py` to visualize ghost creation at SW vs NW
2. Compare corner normal computation at (0,0) vs (0,10)
3. Check if Sobol point distribution creates asymmetry near corners

### Medium-term (Algorithmic)
1. **Adaptive ghost positioning**: Optimize ghost locations based on terminal cost gradient
2. **Corner-specific delta**: Use smaller δ near corners for denser ghost coverage
3. **Weighted ghost values**: Weight mirror values by distance/angle instead of simple reflection
4. **Hybrid approach**: Combine ghosts with terminal cost smoothing specifically at SW corner

### Long-term (Theoretical)
1. Derive optimal ghost positioning for corners with BC-incompatible terminal costs
2. Prove convergence rates depend on ∇g · n magnitude
3. Generalize to arbitrary corner angles and terminal cost geometries

---

## Success Criteria for SW Corner Fix

Minimum acceptable:
- Wrong-sign gradients < 5% in SW region (currently 31%)
- Match NW corner performance (0% wrong-sign)

Target:
- Wrong-sign gradients < 2% in SW region
- Overall displacement ≥ 90% of FDM baseline (currently 63%)

---

**Next Action**: Run corner ghost node visualization to compare SW vs NW creation process.

**Files**:
- `hjb_gfdm.py:887-1060` - Ghost node implementation
- `analyze_southwest_boundary.py` - Spatial analysis script
- `analyze_corner_ghost_nodes.py` - Corner comparison script (to be run)
