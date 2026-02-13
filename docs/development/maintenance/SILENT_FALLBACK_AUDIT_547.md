# Silent Fallback Audit - Issue #547

**Date**: 2026-01-11
**Branch**: `chore/eliminate-silent-fallbacks-547`
**Issue**: #547 - Eliminate silent fallbacks in error handling

## Executive Summary

**Total Violations Found**: 13 instances of `except Exception:`
- **0** bare `except:` (good!)
- **4** High-risk silent fallbacks (mask bugs)
- **5** Medium-risk backend/feature detection
- **4** Low-risk (already have warnings/logging)

**Custom Exception Utilities Available**: ✅ `mfg_pde/utils/exceptions.py`
- Custom exception hierarchy: `MFGSolverError`, `LibraryError`, etc.
- Helper functions: `check_numerical_stability()`, `validate_*()`
- Will use these instead of standard exceptions where appropriate

## Detailed Analysis

### Category 1: HIGH RISK - Silent Failures That Mask Bugs

#### 1. `mfg_pde/alg/numerical/hjb_solvers/base_hjb.py:990`

**Context**: Newton iteration error handling
```python
try:
    # ... Newton iteration logic ...
    l2_error_of_step = np.linalg.norm(delta_U) * np.sqrt(dx_norm)
except Exception:
    pass  # SILENT!
```

**Risk**: **CRITICAL**
- Silently catches ALL exceptions during Newton iteration
- No logging, no warning, no error message
- Masks numerical errors, convergence failures, shape mismatches
- Code continues with potentially corrupted state

**Fix**: Use `check_numerical_stability()` from `mfg_pde/utils/exceptions`
```python
from mfg_pde.utils.exceptions import check_numerical_stability

# Instead of catching, check proactively
try:
    if np.any(np.isnan(delta_U)) or np.any(np.isinf(delta_U)):
        delta_U = np.zeros_like(U_n_current_newton_iterate)
    else:
        l2_error_of_step = np.linalg.norm(delta_U) * np.sqrt(dx_norm)
except (ValueError, LinAlgError) as e:
    logger.warning(
        f"Newton iteration L2 error computation failed: {e}. "
        f"Using zero update for this step."
    )
    l2_error_of_step = np.inf
```

**Alternative**: Remove try/except entirely if the NaN/inf checks are sufficient

**Priority**: **FIX IMMEDIATELY**

---

#### 2. `mfg_pde/geometry/graph/network_geometry.py:1257`

**Context**: Spectral analysis of graph Laplacian
```python
try:
    stats.update({
        "algebraic_connectivity": eigenvals[1] if len(eigenvals) > 1 else 0,
        # ... more spectral properties ...
    })
except Exception:
    # Skip spectral analysis if it fails
    pass  # SILENT!
```

**Risk**: **HIGH**
- Silently skips spectral analysis without warning
- User expects graph statistics but gets incomplete data
- Masks scipy/numpy errors in eigenvalue computation
- No indication in returned `stats` that analysis failed

**Fix**: Add logging + mark failure in stats
```python
except (LinAlgError, ValueError) as e:
    logger.warning(
        f"Spectral analysis failed for graph with {self.network_data.num_nodes} nodes: {e}"
    )
    stats.update({
        "spectral_analysis_failed": True,
        "algebraic_connectivity": None,
    })
```

**Priority**: **FIX IMMEDIATELY**

---

### Category 2: MEDIUM RISK - Backend/Feature Detection

#### 3. `mfg_pde/backends/torch_backend.py:41`

**Context**: MPS (Apple Silicon) device detection
```python
try:
    test_tensor = torch.tensor([1.0], device="mps")
    MPS_FUNCTIONAL = True
except Exception:
    MPS_FUNCTIONAL = False
    warnings.warn("MPS detected but not functional, falling back to CPU")
```

**Risk**: **MEDIUM**
- Catches all exceptions (RuntimeError, ValueError, ImportError)
- Has warning (good!) but too broad
- Could mask unexpected errors

**Fix**: Be specific about expected failures
```python
except (RuntimeError, TypeError) as e:
    MPS_FUNCTIONAL = False
    warnings.warn(
        f"MPS device detected but not functional ({type(e).__name__}): {e}. "
        f"Falling back to CPU.",
        stacklevel=2
    )
```

**Priority**: Medium

---

#### 4. `mfg_pde/backends/torch_backend.py:481`

**Context**: torch.vmap availability check
```python
if hasattr(torch, "vmap"):
    try:
        return torch.vmap(func)
    except Exception:
        return func  # SILENT fallback
return func
```

**Risk**: **MEDIUM**
- Silent fallback from vmap to original function
- User expects vectorized performance, gets slower version
- No indication that optimization failed

**Fix**: Add logging
```python
if hasattr(torch, "vmap"):
    try:
        return torch.vmap(func)
    except (AttributeError, TypeError) as e:
        logger.debug(
            f"torch.vmap failed for function {func.__name__}: {e}. "
            f"Using non-vectorized version."
        )
        return func
return func
```

**Priority**: Medium

---

#### 5-6. `mfg_pde/backends/__init__.py:194, 208`

**Context**: Backend info retrieval for PyTorch and JAX
```python
try:
    info["torch_info"] = { ... }
except Exception:
    info["torch_info"] = {"error": "PyTorch available but info retrieval failed"}
```

**Risk**: **LOW**
- Catches exceptions but sets error message in returned dict
- User sees "error" key in info
- Non-critical (diagnostic info only)

**Fix**: Be specific + add context
```python
except (ImportError, AttributeError, RuntimeError) as e:
    info["torch_info"] = {
        "error": f"PyTorch available but info retrieval failed: {type(e).__name__}"
    }
    logger.debug(f"Failed to retrieve PyTorch info: {e}")
```

**Priority**: Low

---

#### 7. `mfg_pde/backends/jax_backend.py:338`

**Context**: GPU memory monitoring
```python
try:
    # ... get GPU memory stats ...
    return {...}
except Exception:
    return None
```

**Risk**: **MEDIUM**
- Silent return of None
- Caller must check for None
- Could mask errors in memory query

**Fix**: Be specific
```python
except (RuntimeError, AttributeError) as e:
    logger.debug(f"Failed to get JAX GPU memory stats: {e}")
    return None
```

**Priority**: Medium

---

### Category 3: LOW RISK - Already Have Warnings

#### 8. `mfg_pde/visualization/mathematical_plots.py:85`

**Context**: LaTeX setup for matplotlib
```python
try:
    rcParams["text.usetex"] = True
    # ...
except Exception:
    warnings.warn("LaTeX setup failed, falling back to mathtext", stacklevel=2)
    self.use_latex = False
```

**Risk**: **LOW**
- Has warning (good!)
- Reasonable fallback for optional LaTeX rendering
- User is notified

**Fix**: Be specific about LaTeX errors
```python
except (OSError, RuntimeError) as e:
    warnings.warn(
        f"LaTeX setup failed ({type(e).__name__}), falling back to mathtext. "
        f"Install LaTeX for publication-quality rendering.",
        stacklevel=2
    )
    self.use_latex = False
```

**Priority**: Low

---

#### 9. `mfg_pde/alg/neural/dgm/sampling.py:200`

**Context**: Quasi-MC sampling fallback
```python
try:
    sampler = QuasiMCSampler(...)
    return sampler.sample(num_points)
except Exception:
    logger.warning("Quasi-MC sampling failed, using uniform fallback")
    sampler = UniformMCSampler(...)
    return sampler.sample(num_points)
```

**Risk**: **LOW**
- Has logger.warning (good!)
- Reasonable fallback strategy
- User is notified of degraded performance

**Fix**: Be specific
```python
except (ImportError, ValueError, NotImplementedError) as e:
    logger.warning(
        f"Quasi-MC sampling failed ({type(e).__name__}): {e}. "
        f"Using uniform fallback. Performance may be degraded."
    )
    # ...
```

**Priority**: Low

---

### Category 4: RE-RAISES (Not True Fallbacks)

#### 10. `mfg_pde/utils/cli.py:470`

**Context**: CLI exception handling
```python
except Exception:
    # Let exceptions propagate to caller
    # Entry-point scripts should catch and convert to sys.exit()
    raise
```

**Risk**: **NONE**
- Catches then immediately re-raises
- Comment explains intent
- Not a silent fallback

**Fix**: Remove unnecessary try/except or make it specific
```python
# If no specific handling needed, remove the try/except entirely
# OR if needed for cleanup:
except Exception:
    # Cleanup code here
    raise  # Always re-raise
```

**Priority**: Low (cosmetic)

---

#### 11. `mfg_pde/utils/performance/monitoring.py:250`

**Context**: Performance tracking
```python
except Exception:
    # Still track failed execution time
    execution_time = time.time() - start_time
    print(f"WARNING: Performance tracking: {name} failed after {execution_time:.2f}s")
    raise
```

**Risk**: **NONE**
- Tracks execution time then re-raises
- Has warning message
- Not a silent fallback

**Fix**: Use logger instead of print, be specific
```python
except Exception as e:
    execution_time = time.time() - start_time
    logger.warning(
        f"Performance tracking: {name} failed after {execution_time:.2f}s: "
        f"{type(e).__name__}: {e}"
    )
    raise
```

**Priority**: Low

---

### Category 5: Serialization Fallbacks

#### 12. `mfg_pde/workflow/workflow_manager.py:124`

**Context**: JSON serialization fallback
```python
try:
    json.dumps(value)  # Test if JSON serializable
    serialized[key] = value
except Exception:
    serialized[key] = str(value)
```

**Risk**: **LOW**
- Silent conversion to string
- Reasonable for serialization
- Loss of type information but functional

**Fix**: Be specific + add debug logging
```python
try:
    json.dumps(value)
    serialized[key] = value
except (TypeError, ValueError):
    # Non-JSON-serializable object, convert to string
    logger.debug(
        f"Workflow parameter '{key}' is not JSON-serializable "
        f"({type(value).__name__}), converting to string"
    )
    serialized[key] = str(value)
```

**Priority**: Low

---

#### 13. `mfg_pde/geometry/implicit/implicit_domain.py:86`

**Context**: Volume computation fallback
```python
try:
    volume = self.compute_volume(n_monte_carlo=10000)
    # ... use volume ...
except Exception:
    # Fallback: use bounding box
    bounds = self.get_bounding_box()
    # ... estimate from bbox ...
```

**Risk**: **MEDIUM**
- Silent fallback from accurate to approximate volume
- No warning to user about degraded accuracy
- Comment explains logic but user doesn't see it

**Fix**: Add warning
```python
try:
    volume = self.compute_volume(n_monte_carlo=10000)
    h = 0.1
    n_points = int(volume / (h**self.dimension))
    return max(n_points, 100)
except (ValueError, RuntimeError) as e:
    logger.warning(
        f"Volume computation failed for implicit domain: {e}. "
        f"Using bounding box approximation (may overestimate point count)."
    )
    bounds = self.get_bounding_box()
    bbox_volume = np.prod(bounds[:, 1] - bounds[:, 0])
    h = 0.1
    return max(int(bbox_volume / (h**self.dimension)), 100)
```

**Priority**: Medium

---

## Summary by Priority

### 🔴 HIGH PRIORITY (Fix Immediately)
1. `base_hjb.py:990` - Newton iteration silent failure
2. `network_geometry.py:1257` - Spectral analysis silent skip

### 🟡 MEDIUM PRIORITY (Fix Next)
3. `torch_backend.py:41` - MPS detection
4. `torch_backend.py:481` - vmap fallback
5. `jax_backend.py:338` - GPU memory stats
6. `implicit_domain.py:86` - Volume computation fallback

### 🟢 LOW PRIORITY (Cleanup)
7-8. `backends/__init__.py:194, 208` - Info retrieval (already has error dict)
9. `mathematical_plots.py:85` - LaTeX setup (already has warning)
10. `dgm/sampling.py:200` - Quasi-MC fallback (already has warning)
11. `cli.py:470` - Re-raises (cosmetic cleanup)
12. `monitoring.py:250` - Re-raises (use logger)
13. `workflow_manager.py:124` - Serialization (add debug log)

## Implementation Plan

### Day 1: Audit ✅
- [x] Search for broad exception patterns
- [x] Analyze all 13 instances in context
- [x] Categorize by risk level
- [x] Create this audit document

### Day 2: High + Medium Priority
- [ ] Fix #1: `base_hjb.py:990` - Add specific exceptions + logging
- [ ] Fix #2: `network_geometry.py:1257` - Add warning + failure marker
- [ ] Fix #3-4: `torch_backend.py` - Specific exceptions
- [ ] Fix #5: `jax_backend.py` - Add logging
- [ ] Fix #6: `implicit_domain.py` - Add warning
- [ ] Run tests after each fix

### Day 3: Low Priority + Finalize
- [ ] Fix #7-13: Backend info, visualization, serialization
- [ ] Add unit tests for error paths
- [ ] Document error handling patterns in `ERROR_HANDLING_GUIDE.md`
- [ ] Create PR and merge

## Testing Strategy

For each fix:
1. Add unit test that triggers the exception path
2. Verify logging/warning appears in output
3. Verify fallback behavior is correct
4. Verify no silent failures

Example test structure:
```python
def test_newton_iteration_error_handling(caplog):
    """Test that Newton iteration errors are logged, not silent."""
    # Trigger error condition
    with pytest.raises(LinAlgError):
        solver._newton_step_with_invalid_input()

    # Verify warning was logged
    assert "Newton iteration step error" in caplog.text
```

## Success Criteria

- [ ] All 13 broad `except Exception:` replaced with specific types
- [ ] All silent fallbacks have logging (warning for degradation, debug for minor)
- [ ] High-risk issues (#1-2) fixed and tested
- [ ] Documentation created: `ERROR_HANDLING_GUIDE.md`
- [ ] PR created and CI passes
- [ ] Issue #547 closed

---

**Next Steps**: Begin Day 2 implementation (High + Medium priority fixes)
