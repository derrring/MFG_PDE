# Revised Analysis: Mass Conservation Test Failures

**Status**: Investigation Complete  
**Date**: 2025-10-05  
**Original Analysis**: Incorrect - focused on discretization details  
**Revised Finding**: Solver convergence failure, not discretization bug

## Key Insight from User

> "FDM doesn't have mass conservative property. That's the advantage of particle-based FP over FDM, right?"

**Correct!** Standard FDM for Fokker-Planck does NOT guarantee exact mass conservation. This is a fundamental limitation of the method.

## What the Tests Are Actually Checking

The mass conservation tests (with tolerance `1e-2` = 1%) are checking whether:
1. Solver produces **physically reasonable** solutions
2. Discretization error is **acceptably small** (not perfect conservation)
3. Solution is **numerically stable**

## Real Problem: Solver Divergence

### Evidence

```python
problem = ExampleMFGProblem(Nx=20, Nt=8, T=0.5)
solver = create_fast_solver(problem, "fixed_point")
result = solver.solve()

# Mass evolution:
t=0: mass=1.000000 (initial)
t=1: mass=0.334419 (66% loss - diverging)
t=2: mass=0.290241 (continuing to degrade)
...
t=8: mass=189.018496 (EXPLODED!)

# Solver output:
"WARNING: Max iterations (10) reached"
"Errors: U=1.42e+00, M=2.33e+00"  # Large errors, no convergence
```

### Root Cause

The **Picard fixed-point iteration** is not converging for this problem with current settings:
- Only 10 Picard iterations (too few)
- Damping factor 0.7 (may be inappropriate)
- FDM solvers may be unstable for this parameter regime

## Why FDM Appears to Fail

1. **Not a discretization bug**: The boundary stencils may be fine
2. **Convergence issue**: Picard iteration diverges → produces unphysical densities
3. **Amplification**: Small FDM errors get amplified by divergent iteration
4. **Cascading failure**: Bad density → bad HJB solution → worse density → ...

## Correct Interpretation of Test Failures

**Tests are not wrong** to expect ~1% mass conservation from FDM. A well-implemented, **converged** FDM solution should have:
- Mass error: O(Δt²) + O(Δx²) ≈ 0.1-1% for reasonable discretizations
- Stable time integration
- Convergent fixed-point iteration

**Current failure mode**: Solver doesn't converge → solution is garbage → mass conservation violated as side effect

## Solutions

### Option 1: Fix Convergence (Short-term)

Make FDM solver actually converge:
```python
config = create_accurate_config()  # More iterations, tighter tolerance
solver = create_accurate_solver(problem, "fixed_point")
```

### Option 2: Use Particle Methods (Recommended)

Particle methods are:
- ✅ Naturally mass-conserving (by construction)
- ✅ More robust for MFG problems
- ✅ Better for long-time integration

```python
solver = create_fast_solver(problem, solver_type="particle_collocation")
```

### Option 3: Update Tests

Change tests to either:
1. Use particle methods (test what we recommend for production)
2. Use `create_accurate_solver` (more iterations for FDM convergence)
3. Mark FDM tests as `xfail` with note about convergence requirements

## Recommended Action

**Update `create_fast_solver` default** to use particle methods instead of FDM:

```python
# Current (uses FDM - unstable):
solver = create_fast_solver(problem, "fixed_point")

# Proposed (uses particles - stable):
solver = create_fast_solver(problem, "particle_collocation")
```

**Rationale**:
- "Fast" should mean "fast to get working results", not "fast execution but unreliable"
- Particle methods are more robust
- FDM can be accessed via `create_accurate_solver` for users who need it

## Conclusion

The mass conservation test failures are **correct** - they're catching that the FDM-based fixed-point solver doesn't converge with default "fast" settings. 

**Not a bug in FDM discretization** - it's a **feature request** to change defaults to more robust particle methods.

---

**Action Item**: Change `create_fast_solver` to use particle methods by default, making FDM opt-in for advanced users.
