# BC Compatibility Hypothesis - REJECTED

**Date**: 2025-12-20
**Status**: ❌ **HYPOTHESIS REJECTED**
**Experiment**: GFDM-Ghost with BC-Compatible Terminal Cost

---

## Executive Summary

**Hypothesis**: The SW corner asymmetry (ghost nodes work at NW but not SW) is caused by BC incompatibility where the terminal cost gradient violates Neumann BC (∂g/∂n ≠ 0).

**Test**: Created BC-compatible terminal cost with ∂g/∂n ≈ 0.33 at corners (vs 49.0 original) using boundary blending.

**Result**: ❌ **HYPOTHESIS DEFINITIVELY REJECTED**

Enforcing BC compatibility made performance **dramatically worse**, not better:
- SW corner wrong-sign gradients: 31% → **69%** (123% increase)
- NW corner wrong-sign gradients: 0% → **47%** (introduced new errors)
- Overall displacement: 7.06 units → **2.49 units** (65% degradation)

---

## Experimental Results

### Performance Comparison

| Metric | FDM Baseline | Ghost (Original) | Ghost (Compatible) | Change |
|:-------|:-------------|:-----------------|:-------------------|:-------|
| **SW wrong-sign %** | 0.0% | 31.0% | **69.0%** | +38 pp ❌ |
| **NW wrong-sign %** | 0.0% | 0.0% | **47.0%** | +47 pp ❌ |
| **Displacement** | 11.17 units | 7.06 units (63%) | **2.49 units (22%)** | -65% ❌ |
| **Runtime** | 226s | 832s | 826s | Similar |

### Key Observations

1. **Uniform degradation**: Both corners (SW and NW) now perform poorly
2. **Massive displacement loss**: Crowd barely moves toward exit (2.49 vs 11.17 units)
3. **Wrong-sign explosion**: Gradients pointing wrong direction across entire west boundary

---

## Terminal Cost Analysis

### Original (Incompatible) Terminal Cost
```
g(x,y) = 0.5 * ||x - (20,5)||²
```

**BC Violation**:
- ∂g/∂n at SW corner (0,0): 49.0 (strong violation)
- ∂g/∂n at NW corner (0,10): 49.0 (strong violation)

**Performance**:
- Ghost nodes: 63% of FDM
- SW corner: 31% wrong-sign

### Compatible (Blended) Terminal Cost
```
g_blend(x,y) = (1-β(d)) * g_original(x,y) + β(d) * c0
where β(d) transitions from 0 (interior) to 1 (boundary)
```

**BC Compliance**:
- ∂g/∂n at SW corner: 0.33 (149x reduction)
- ∂g/∂n at NW corner: 0.33 (149x reduction)

**Performance**:
- Ghost nodes: **22% of FDM** (65% worse than original!)
- SW corner: **69% wrong-sign** (123% worse!)
- NW corner: **47% wrong-sign** (introduced errors where there were none!)

---

## Analysis: Why Compatible Cost Failed

### Hypothesis 1: Terminal Cost Encodes Problem Physics

The original terminal cost `g(x,y) = ||x - exit||²` encodes the **correct evacuation direction** even though it violates Neumann BC. The gradient ∇g points toward the exit, which is the physically correct optimal control.

By blending to a constant value near boundaries, we **destroyed the directional information** that guides the crowd toward the exit. The compatible cost creates a "flat plateau" where agents have no preference for movement direction.

### Hypothesis 2: BC Violation is Physically Necessary

For evacuation problems, **Neumann BC may be mathematically incorrect** for the actual physics:

- No-flux BC: ∂m/∂n = 0 (appropriate for density - no mass crosses boundary)
- Value function BC: ∂u/∂n = 0 (may be inappropriate - agents at boundary DO prefer to move toward exit)

The original incompatible cost correctly represents that **boundary agents are not indifferent** - they want to move parallel to the boundary toward the nearest exit.

### Hypothesis 3: Ghost Nodes Amplify Wrong BC

Ghost nodes are **high-fidelity BC enforcers**. When we enforce the wrong BC (∂u/∂n = 0), they do so perfectly, which:
1. Removes exit-seeking gradients near boundaries
2. Creates stagnation zones where agents don't know which way to go
3. Leads to wrong-sign gradients as the solver tries to reconcile incompatible constraints

---

## Revised Understanding

### The True Problem

The SW corner asymmetry is **NOT caused by BC incompatibility**. Instead:

1. **Ghost nodes work correctly** - they enforce whatever BC we give them
2. **The original terminal cost is physically correct** - it encodes evacuation physics
3. **Neumann BC may be wrong** for value functions in evacuation problems
4. **The asymmetry (NW works, SW fails) has a different cause** - possibly:
   - Local geometry effects
   - Interaction between terminal cost gradient direction and corner position
   - Stencil quality differences at corners with different geometric configurations

### Why FDM Appears to "Work"

FDM's success may be due to:
- Numerical viscosity smoothing BC violations
- Less aggressive BC enforcement (row replacement is "softer" than symmetric stencils)
- Structured grid benefiting from uniform spacing

---

## Implications

### For Ghost Nodes Method

1. **Ghost nodes are working as designed** - they are not the problem
2. **BC enforcement quality is not the issue** - more accurate BC enforcement made things worse
3. **Need to reconsider what BC is appropriate** for evacuation MFG problems

### For Boundary Conditions

1. **Neumann BC (∂u/∂n = 0) may be physically incorrect** for value functions in bounded evacuation domains
2. **The terminal cost should guide agents toward exits**, even at boundaries
3. **Alternative BC formulations needed** (e.g., directional BC, Robin BC, or relaxed Neumann)

### For SW Corner Asymmetry

The different performance at SW vs NW corners is **NOT explained by BC incompatibility**. Possible actual causes:
1. Terminal cost gradient direction relative to corner geometry
2. Collocation point distribution artifacts
3. Stencil reconstruction quality for different neighbor configurations
4. Numerical artifacts in ghost node placement

---

## Conclusions

**Main Finding**: Enforcing BC compatibility via terminal cost blending is **counterproductive** for evacuation problems. The original "incompatible" terminal cost is physically correct and necessary for proper evacuation dynamics.

**Methodological Lesson**: Diagnostic experiments with unexpected negative results are valuable - they **definitively rule out hypotheses** and redirect research.

**Path Forward**:
1. Accept that ∂g/∂n ≠ 0 is **physically necessary**, not a bug
2. Investigate SW corner asymmetry through **geometric** rather than **analytical** lens
3. Consider alternative BC formulations that preserve exit-seeking behavior

---

## Files

**Experiment**:
- `solution_gfdm_ghost_blend_fdm.h5` - Compatible cost results (poor performance)
- `metrics_gfdm_ghost_blend_fdm.json` - Quantitative metrics

**Analysis**:
- `ghost_compatible_cost_comparison.png` - Visual comparison
- `terminal_cost_compatible_blend.npz` - BC-compatible terminal cost

**Code**:
- `create_compatible_terminal_cost_v2.py` - Blending implementation
- `analyze_compatible_cost_results.py` - Results analysis
- `exp1_baseline_fdm.py:176-188` - Blend mode in experiment runner

---

**Status**: Hypothesis rejected, new direction needed
**Next**: Investigate geometric causes of SW corner asymmetry
