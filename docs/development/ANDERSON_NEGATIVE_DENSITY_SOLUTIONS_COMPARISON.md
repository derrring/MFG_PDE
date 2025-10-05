# Anderson Acceleration Negative Density - Solutions Comparison

**Date**: 2025-10-05
**Status**: 🔄 **Analysis & Testing**
**Issue**: Anderson acceleration produces negative density values via overshoot

## Problem Summary

Anderson acceleration uses linear extrapolation of past iterates with potentially negative coefficients (from least-squares optimization), causing density M to overshoot below zero. This creates a dilemma:

- **Cannot clamp to zero**: Violates mass conservation `∫M dx = constant`
- **Cannot keep negative**: Violates probability interpretation `M(t,x) ≥ 0`

## Three Proposed Solutions

### Option 1: Anderson for U Only, Standard Damping for M

**Concept**: Apply Anderson acceleration only to value function U, use standard Picard damping for density M.

#### Implementation
```python
# In FixedPointIterator.solve(), replace lines 288-307:

if self.use_anderson and self.anderson_accelerator is not None:
    # Apply Picard damping first
    U_damped = self.thetaUM * U_new_tmp_hjb + (1 - self.thetaUM) * U_old_current_picard_iter
    M_damped = self.thetaUM * M_new_tmp_fp + (1 - self.thetaUM) * M_old_current_picard_iter

    # Apply Anderson to U ONLY
    x_current_U = U_old_current_picard_iter.flatten()
    f_current_U = U_damped.flatten()
    x_next_U = self.anderson_accelerator.update(x_current_U, f_current_U, method="type1")

    self.U = x_next_U.reshape(U_old_current_picard_iter.shape)
    self.M = M_damped  # Standard damping only - preserves non-negativity
else:
    # Standard damping
    self.U = self.thetaUM * U_new_tmp_hjb + (1 - self.thetaUM) * U_old_current_picard_iter
    self.M = self.thetaUM * M_new_tmp_fp + (1 - self.thetaUM) * M_old_current_picard_iter
```

#### Pros
- ✅ **Guaranteed non-negativity**: `M ≥ 0` always (convex combination of non-negative values)
- ✅ **Preserves mass conservation**: No clamping/projection needed
- ✅ **Simple implementation**: Minimal code changes
- ✅ **Still accelerates HJB**: U convergence is often the bottleneck
- ✅ **Mathematically clean**: No ad-hoc fixes

#### Cons
- ⚠️ **Slower M convergence**: FP iteration not accelerated
- ⚠️ **Asymmetric treatment**: U and M handled differently
- ⚠️ **May need more iterations**: Overall convergence could slow down

#### Performance Prediction
- **Best case**: HJB is bottleneck → Anderson on U gives 2-3× speedup
- **Worst case**: FP is bottleneck → No acceleration, possible slowdown
- **Expected**: Moderate improvement (1.5-2× faster than no Anderson)

---

### Option 2: Non-Negativity Projection After Anderson

**Concept**: Apply Anderson to both U and M, then project M onto non-negative cone while preserving mass.

#### Implementation
```python
# In FixedPointIterator.solve(), after line 307:

if self.use_anderson and self.anderson_accelerator is not None:
    # [Existing Anderson code lines 288-307]
    ...
    self.U = x_next[:n_u].reshape(U_old_current_picard_iter.shape)
    self.M = x_next[n_u:].reshape(M_old_current_picard_iter.shape)

    # PROJECT M to non-negative cone while preserving mass
    for t_idx in range(self.M.shape[0]):
        M_t = self.M[t_idx, :]

        if (M_t < 0).any():
            # Projection: M_proj = argmin ||M - M_anderson||^2
            #             subject to: M ≥ 0, ∫M dx = ∫M_anderson dx

            total_mass = np.trapezoid(M_t, dx=Dx)

            # Simple projection: clamp negatives, redistribute to maintain mass
            M_pos = np.maximum(M_t, 0)
            mass_pos = np.trapezoid(M_pos, dx=Dx)

            if mass_pos > 1e-10:
                # Rescale to preserve total mass
                self.M[t_idx, :] = M_pos * (total_mass / mass_pos)
            else:
                # Fallback: uniform distribution
                self.M[t_idx, :] = total_mass / Lx  # Uniform
```

#### Pros
- ✅ **Full Anderson acceleration**: Both U and M accelerated
- ✅ **Enforces non-negativity**: M ≥ 0 guaranteed
- ✅ **Preserves mass**: `∫M dx` maintained (approximately)
- ✅ **Fastest convergence**: Maximum acceleration potential

#### Cons
- ❌ **Complex implementation**: Non-trivial projection algorithm
- ❌ **Approximation errors**: Projection introduces additional error
- ❌ **Not exact mass conservation**: Numerical integration errors accumulate
- ❌ **Potential instability**: Projection + extrapolation can oscillate
- ⚠️ **Computationally expensive**: Projection at every iteration

#### Performance Prediction
- **Best case**: Both HJB & FP accelerated → 3-5× speedup
- **Worst case**: Projection causes instability → Divergence or slowdown
- **Expected**: 2-3× faster but with occasional instability

#### Mathematical Concerns

**Problem**: This is not truly conservative!

```
∫M_projected dx ≈ ∫M_anderson dx  (numerical integration error)
```

Over many iterations, errors accumulate. The rescaling is a **heuristic**, not a rigorous projection.

**Rigorous projection** would require:
```
min ||M - M_anderson||^2
s.t. M ≥ 0
     ∫M dx = const  (continuous constraint!)
```

This is a constrained QP problem that needs discretization-aware formulation.

---

### Option 3: Reduce Anderson Aggressiveness

**Concept**: Keep Anderson for both U and M, but reduce aggressiveness to avoid negative overshoot.

#### Implementation A: Reduce Anderson Depth
```python
# In test/example configuration:
mfg_solver = FixedPointIterator(
    problem,
    hjb_solver=hjb_solver,
    fp_solver=fp_solver,
    use_anderson=True,
    anderson_depth=2,  # Reduced from 5 (fewer past iterates)
    anderson_beta=0.5, # Reduced mixing (if parameter exists)
    thetaUM=0.5,
)
```

#### Implementation B: Safeguard Anderson Update
```python
# In FixedPointIterator.solve(), after Anderson update:

if self.use_anderson and self.anderson_accelerator is not None:
    # [Existing Anderson code]
    ...
    M_anderson = x_next[n_u:].reshape(M_old_current_picard_iter.shape)

    # SAFEGUARD: If Anderson produces negatives, fall back to damping
    if (M_anderson < 0).any():
        print(f"  Anderson overshoot detected, using standard damping for M")
        self.M = self.thetaUM * M_new_tmp_fp + (1 - self.thetaUM) * M_old_current_picard_iter
    else:
        self.M = M_anderson

    self.U = x_next[:n_u].reshape(U_old_current_picard_iter.shape)
```

#### Pros
- ✅ **Simple safeguard**: Fallback to damping when needed
- ✅ **Preserves mass**: No projection, uses standard damping for negative cases
- ✅ **Adaptive**: Anderson when safe, damping when overshoot
- ✅ **Easy to tune**: Single parameter (depth or beta)

#### Cons
- ⚠️ **Unpredictable**: Sometimes accelerates, sometimes doesn't
- ⚠️ **Parameter tuning**: Need to find sweet spot (problem-dependent)
- ⚠️ **Suboptimal acceleration**: Reduced depth means less speedup
- ❌ **Doesn't solve root cause**: Just avoids symptoms

#### Performance Prediction
- **Best case**: Reduced depth sufficient → 1.5-2× speedup, no negatives
- **Worst case**: Still produces negatives → Frequent fallbacks, 1.2× speedup
- **Expected**: Moderate acceleration with occasional fallbacks

---

## Comparative Analysis

### Mass Conservation Quality

| Option | Mass Conservation | Notes |
|:-------|:-----------------|:------|
| **Option 1** | ✅ **Exact** | Convex combination preserves `∫M dx` exactly |
| **Option 2** | ⚠️ **Approximate** | Projection error accumulates over iterations |
| **Option 3** | ✅ **Exact** (fallback) | Safeguard uses standard damping |

**Winner**: Option 1 (or Option 3 with fallback)

### Non-Negativity Guarantee

| Option | Guarantee | Notes |
|:-------|:----------|:------|
| **Option 1** | ✅ **Always** | `M = θ*M_new + (1-θ)*M_old`, both ≥ 0 |
| **Option 2** | ✅ **After projection** | Projection enforces M ≥ 0 |
| **Option 3** | ✅ **With fallback** | Falls back to damping if negative |

**Winner**: All options guarantee non-negativity

### Convergence Speed

| Option | Expected Speedup | Stability |
|:-------|:----------------|:----------|
| **Option 1** | 1.5-2× | ✅ High |
| **Option 2** | 2-3× | ⚠️ Medium (projection can cause oscillation) |
| **Option 3** | 1.2-2× | ✅ High (adaptive fallback) |

**Winner**: Option 2 (if stable), Option 1 (for reliability)

### Implementation Complexity

| Option | Code Changes | Maintenance |
|:-------|:------------|:------------|
| **Option 1** | ✅ Minimal (~10 lines) | ✅ Simple |
| **Option 2** | ❌ Significant (~30 lines + projection) | ❌ Complex |
| **Option 3** | ✅ Small (~15 lines) | ✅ Simple |

**Winner**: Option 1 or Option 3

### Theoretical Soundness

| Option | Mathematical Rigor | Notes |
|:-------|:------------------|:------|
| **Option 1** | ✅ **Clean** | Standard convex combination |
| **Option 2** | ⚠️ **Heuristic** | Projection is approximate |
| **Option 3** | ✅ **Principled** | Adaptive with fallback |

**Winner**: Option 1

---

## Recommendation Matrix

### By Priority

| Priority | Best Option | Reason |
|:---------|:-----------|:-------|
| **Mass conservation** | Option 1 | Exact preservation |
| **Convergence speed** | Option 2 | Full acceleration (if stable) |
| **Robustness** | Option 1 | No approximations |
| **Simplicity** | Option 1 | Minimal code changes |
| **Adaptability** | Option 3 | Handles various problems |

### By Problem Type

| Problem Characteristics | Recommended | Reason |
|:----------------------|:-----------|:-------|
| **HJB is bottleneck** | **Option 1** | Anderson on U sufficient |
| **FP is bottleneck** | Option 2 or 3 | Need M acceleration |
| **High stochasticity (large σ)** | **Option 1** | More stable |
| **Tight coupling** | Option 2 | Need both accelerated |
| **Research/exploration** | Option 3 | Easy to tune |

---

## Testing Plan

### Test 1: Convergence Rate
Compare iterations to convergence for each option:
- Problem: Standard 1D MFG (Nx=50, Nt=20, σ=0.1)
- Tolerance: 1e-4
- Metric: Iterations to convergence

### Test 2: Mass Conservation Error
Track `|∫M(t,x)dx - ∫M(0,x)dx|` over iterations:
- Metric: Maximum mass error across all timesteps
- Option 1 should have machine precision (~1e-15)
- Option 2 may accumulate error (~1e-8 to 1e-12)

### Test 3: Negative Density Frequency
Count how often M < 0 occurs:
- Option 1: 0 occurrences (guaranteed)
- Option 2: 0 after projection
- Option 3: Depends on depth parameter

### Test 4: Computational Cost
Wall-clock time per iteration:
- Option 1: Baseline (Anderson on U only)
- Option 2: +overhead from projection
- Option 3: Baseline (no extra computation)

---

## Final Recommendation: **Option 1** ✅

**Rationale**:
1. **Mathematically clean**: No approximations, exact conservation
2. **Simple implementation**: Minimal code changes
3. **Robust**: Guaranteed non-negativity without projection
4. **Sufficient acceleration**: HJB is typically the bottleneck
5. **Easy to understand**: Clear physical/mathematical interpretation

**When to consider alternatives**:
- **Option 2**: If benchmarking shows FP is bottleneck AND projection remains stable
- **Option 3**: If Option 1 convergence is too slow for specific problem class

---

## Implementation Order

1. ✅ **Implement Option 1** (primary solution)
2. ⚠️ **Benchmark against no Anderson** (establish baseline)
3. 🔬 **Optionally implement Option 3** (for comparison)
4. 📊 **Compare convergence rates and robustness**
5. 📝 **Document findings and update defaults**

---

**Next**: Proceed with Option 1 implementation
