# Hamiltonian Signature Analysis and Unification Proposal

**Date**: 2025-11-02
**Investigation**: Complete audit of Hamiltonian signatures across MFG_PDE
**Key Finding**: Multiple inconsistent signatures exist - unification needed

---

## Executive Summary

**Problem**: MFG_PDE has **at least 5 different Hamiltonian signatures** across the codebase, causing:
- Confusion for users about which signature to use
- Incompatibility between problem classes and solvers
- Brittle code requiring multiple interface checks

**Root Cause**: Historical evolution from 1D → nD without systematic signature design

**Impact**:
- Medium priority: Not blocking but reduces code quality
- Affects API consistency and user experience
- Makes problem definitions harder to reuse across solvers

**Proposed Solution**: Unified signature with backward compatibility layer

---

## Current Hamiltonian Signatures (Ordered by Frequency)

### Signature 1: `hamiltonian(x, m, p, t)` - **Most Common**
**Frequency**: ~25 occurrences
**Used by**: GridBasedMFGProblem (official), most examples, most tests

```python
def hamiltonian(self, x: np.ndarray, m: np.ndarray, p: np.ndarray, t: float) -> np.ndarray:
    """
    Args:
        x: Spatial coordinates (dimension,)
        m: Density value(s)
        p: Momentum/gradient vector (dimension,)
        t: Time
    """
    return 0.5 * np.sum(p**2)
```

**Files using this**:
- `mfg_pde/core/highdim_mfg_problem.py:120` (official interface)
- `examples/advanced/weno_4d_test.py:48`
- `examples/advanced/semi_lagrangian_3d_test.py:49`
- `benchmarks/validation/*.py` (multiple)
- `tests/integration/*.py` (multiple)

**Assessment**: ✅ **RECOMMENDED AS STANDARD**
- Clean mathematical ordering: state (x, m) → control (p) → time (t)
- Dimension-agnostic
- Works for both scalar and vector x, p

---

### Signature 2: `hamiltonian(x, p, m, t)` - **Legacy 1D**
**Frequency**: ~15 occurrences
**Used by**: Some examples, some demos

```python
def hamiltonian(x, p, m, t=0):
    """
    Args:
        x: Spatial coordinate (scalar or array)
        p: Momentum (scalar or vector)
        m: Density
        t: Time (optional)
    """
    if isinstance(p, (tuple, list)):
        return 0.5 * sum(p_i**2 for p_i in p)
    else:
        return 0.5 * p**2
```

**Files using this**:
- `examples/basic/custom_hamiltonian_derivs_demo.py:193`
- `examples/basic/acceleration_comparison.py:42`
- `examples/basic/towel_beach_demo.py:85`

**Assessment**: ⚠️ Legacy compatibility only
- Inconsistent parameter order (p before m)
- Still useful for simple 1D examples
- Confuses users about standard ordering

---

### Signature 3: `hamiltonian(t, x, p, m)` - **DGM/PINN Style**
**Frequency**: ~5 occurrences
**Used by**: Neural network solvers (DGM, PINN)

```python
def hamiltonian(self, t: float, x: np.ndarray, p: np.ndarray, m: float) -> float:
    """
    Args:
        t: Time (first for neural network convention)
        x: Spatial coordinates
        p: Momentum
        m: Density
    """
    return 0.5 * np.sum(p**2) + self.coefCT * m
```

**Files using this**:
- `examples/basic/dgm_simple_validation.py:64`
- `examples/advanced/pinn_bayesian_mfg_demo.py:92`

**Assessment**: ⚠️ Neural network convention
- Time-first ordering common in ML/PDE literature
- Incompatible with FDM/FEM solvers
- Isolated to neural solvers

---

### Signature 4: Custom Orderings - **Problem-Specific**
**Frequency**: ~8 occurrences
**Used by**: Specialized applications

```python
# Portfolio optimization: momentum first, state second
def hamiltonian(pW, palpha, W, alpha, m, params):
    return 0.5 * (pW**2 + palpha**2) + cost(W, alpha, m)

# Traffic flow: gradient components separated
def hamiltonian(px, py, m, lambda_param, gamma):
    return 0.5 * (px**2 + py**2) + lambda_param * m

# Network MFG: node-based
def hamiltonian(self, node: int, neighbors: list[int], m: np.ndarray, p: np.ndarray, t: float):
    return sum(0.5 * p[i]**2 for i in neighbors)
```

**Files using this**:
- `examples/advanced/portfolio_optimization_2d_demo.py:144`
- `examples/advanced/epidemic_modeling_2d_demo.py:189`
- `examples/advanced/traffic_flow_2d_demo.py:123`
- `mfg_pde/core/network_mfg_problem.py:168`

**Assessment**: ℹ️ Intentional domain-specific APIs
- Specialized for particular application domains
- Not compatible with general solvers
- Require custom solver implementations

---

### Signature 5: Legacy Index-Based - **1D MFGProblem**
**Frequency**: ~5 occurrences
**Used by**: Old 1D solver interface

```python
def H(self, x_idx: int, m: float, derivs: dict, t_idx: int = 0):
    """
    Args:
        x_idx: Grid point index
        m: Density at grid point
        derivs: Dictionary of derivatives {(order,): value}
        t_idx: Time index
    """
    p = derivs.get((1,), 0.0)  # First derivative
    return 0.5 * p**2
```

**Files using this**:
- Legacy MFGProblem (backward compatibility)
- Some factory tests

**Assessment**: 🗑️ Deprecated
- Index-based instead of value-based
- Hard to use, error-prone
- Maintained only for backward compatibility

---

## Signature Comparison Matrix

| Signature | Order | Type Safety | Dimension Support | Solver Compatibility | Recommendation |
|:----------|:------|:------------|:------------------|:---------------------|:---------------|
| `(x, m, p, t)` | State → Control | ✅ Strong | ✅ nD | ✅ FDM, FEM, WENO, Semi-Lagrangian | ✅ **STANDARD** |
| `(x, p, m, t)` | Mixed | ⚠️ Moderate | ⚠️ 1D/nD | ⚠️ Limited | ⚠️ Legacy only |
| `(t, x, p, m)` | Time-first | ✅ Strong | ✅ nD | ⚠️ Neural only | ⚠️ Neural network convention |
| Custom | Application | ⚠️ Weak | ℹ️ Varies | ❌ None | ℹ️ Specialized |
| `H(idx, m, derivs, t)` | Index-based | ❌ Weak | ❌ 1D only | ❌ Legacy 1D | 🗑️ Deprecated |

---

## Solver Expectations

### FDM-Based Solvers (HJB-FDM, FP-FDM)
**Current**: Dual interface support (line 312-323 in hjb_fdm.py)
```python
if hasattr(self.problem, "hamiltonian"):
    # NEW: Use modern signature (x, m, p, t)
    H_values[idx] = self.problem.hamiltonian(x_coords, m_at_point, p, t=0.0)
elif hasattr(self.problem, "H"):
    # LEGACY: Use old signature (idx, m, derivs)
    H_values[idx] = self.problem.H(multi_idx, m_at_point, derivs=derivs_at_point)
else:
    raise AttributeError("Problem must have 'hamiltonian' or 'H' method")
```

**Assessment**: Good backward compatibility, but encourages inconsistency

### WENO Solver
**Current**: Relies on problem.hamiltonian(x, p, m, t)
**Issue**: Parameter order differs from GridBasedMFGProblem standard

### Semi-Lagrangian Solver
**Current**: Implements custom Hamiltonian evaluation (lines 853-901 in hjb_semi_lagrangian.py)
```python
# Hardcoded quadratic Hamiltonian
p_norm_sq = np.sum(p_vec**2)
return 0.5 * p_norm_sq + coef_CT * m
```

**Issue**: Doesn't call problem.hamiltonian() at all!

### Neural Solvers (DGM, PINN)
**Current**: Require `(t, x, p, m)` ordering
**Reason**: PDE-NET and similar architectures use time-first convention

---

## Mathematical Consistency

### Standard MFG Hamiltonian
$$H(x, m, p, t) = \frac{1}{2}|p|^2 + V(x) + C(m)$$

**Natural argument grouping**:
1. **State variables**: $(x, m)$ - describe current configuration
2. **Control variable**: $p$ - control to optimize over
3. **Parameter**: $t$ - independent variable (time)

**Recommended order**: `H(x, m, p, t)`
- Groups state before control
- Time last (often fixed or optional)
- Matches partial differential equation convention: $H(x, m, \nabla u)$

---

## Proposal: Unified Signature with Compatibility Layer

### Recommended Standard (Phase 1)

**Official signature** for all new code:
```python
def hamiltonian(self, x: np.ndarray, m: np.ndarray | float, p: np.ndarray, t: float = 0.0) -> float | np.ndarray:
    """
    Hamiltonian function H(x, m, p, t).

    Args:
        x: Spatial coordinates
            - 1D: shape (1,) or scalar
            - nD: shape (dimension,)
        m: Density value(s)
            - Scalar for single point
            - Array for multiple points
        p: Momentum/gradient vector
            - 1D: shape (1,) or scalar
            - nD: shape (dimension,)
        t: Time (default 0.0)

    Returns:
        Hamiltonian value(s)
            - Scalar for single evaluation
            - Array for vectorized evaluation
    """
    return 0.5 * np.sum(p**2, axis=-1) + self.potential(x, t) + self.coupling(m)
```

**Advantages**:
- Consistent with GridBasedMFGProblem (official interface)
- Dimension-agnostic
- Clear mathematical semantics
- Works for both scalar and vectorized evaluation

---

### Backward Compatibility Layer (Phase 2)

Create adapter to support legacy signatures:

```python
class HamiltonianAdapter:
    """
    Adapter to support multiple Hamiltonian signatures.

    Automatically detects signature and converts to standard (x, m, p, t).
    """

    def __init__(self, hamiltonian_func: Callable):
        self.func = hamiltonian_func
        self.signature = self._detect_signature(hamiltonian_func)

    def _detect_signature(self, func: Callable) -> str:
        """Detect Hamiltonian signature from function parameters."""
        import inspect
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Remove 'self' if present
        if params and params[0] == 'self':
            params = params[1:]

        # Match against known patterns
        if params == ['x', 'm', 'p', 't'] or params[:3] == ['x', 'm', 'p']:
            return "standard"  # (x, m, p, t)
        elif params == ['x', 'p', 'm', 't'] or params[:3] == ['x', 'p', 'm']:
            return "legacy"  # (x, p, m, t)
        elif params == ['t', 'x', 'p', 'm'] or params[:4] == ['t', 'x', 'p', 'm']:
            return "neural"  # (t, x, p, m)
        else:
            return "unknown"

    def __call__(self, x, m, p, t=0.0):
        """Call with standard signature, adapting if necessary."""
        if self.signature == "standard":
            return self.func(x, m, p, t)
        elif self.signature == "legacy":
            return self.func(x, p, m, t)
        elif self.signature == "neural":
            return self.func(t, x, p, m)
        else:
            # Try standard signature
            try:
                return self.func(x, m, p, t)
            except TypeError:
                raise ValueError(
                    f"Hamiltonian signature not recognized. "
                    f"Use standard signature: hamiltonian(x, m, p, t)"
                )
```

**Usage in solvers**:
```python
# In solver __init__
if hasattr(problem, 'hamiltonian'):
    self._hamiltonian = HamiltonianAdapter(problem.hamiltonian)
else:
    # Fallback to legacy H() method
    self._hamiltonian = self._create_legacy_adapter(problem.H)

# In solver computation
H_value = self._hamiltonian(x, m, p, t)  # Always use standard signature
```

---

## Migration Strategy

### Phase 1: Documentation (Immediate)
1. ✅ Document standard signature in development guide
2. ✅ Update CLAUDE.md with Hamiltonian signature standard
3. ✅ Create this analysis document

### Phase 2: Core Classes (1-2 weeks)
1. ⚠️ Verify GridBasedMFGProblem uses standard signature
2. ⚠️ Update HighDimMFGProblem abstract method signature
3. ⚠️ Add signature validation to MFGComponents

### Phase 3: Solver Updates (2-3 weeks)
1. ⚠️ Implement HamiltonianAdapter in base solver classes
2. ⚠️ Update FDM solvers to use adapter
3. ⚠️ Update WENO solver to use adapter
4. ⚠️ Update Semi-Lagrangian to call problem.hamiltonian() instead of hardcoding

### Phase 4: Examples Migration (1-2 weeks)
1. ⚠️ Update all examples to use standard signature
2. ⚠️ Add deprecation warnings for legacy signatures
3. ⚠️ Create migration examples

### Phase 5: Deprecation (6 months later)
1. ⚠️ Mark legacy signatures as deprecated
2. ⚠️ Remove adapter code in next major version

---

## Special Cases

### Network MFG
**Current signature**: `hamiltonian(node, neighbors, m, p, t)`
**Recommendation**: Keep as-is - intentionally different API
**Reason**: Graph-based problems require different interface

### Variational MFG
**Current**: Defines Hamiltonian through builder pattern
**Recommendation**: Internal conversion to standard signature

### Neural Solvers (DGM, PINN)
**Current signature**: `hamiltonian(t, x, p, m)`
**Recommendation**: Keep as-is, use adapter when interfacing with FDM
**Reason**: ML/PDE convention uses time-first ordering

---

## Implementation Priority

### High Priority (Phase 1-2)
- ✅ Documentation of standard signature
- ⚠️ HamiltonianAdapter implementation
- ⚠️ Core problem class updates

### Medium Priority (Phase 3)
- ⚠️ Solver adapter integration
- ⚠️ Semi-Lagrangian fix to call problem.hamiltonian()

### Low Priority (Phase 4-5)
- ⚠️ Example migration
- ⚠️ Deprecation warnings

---

## Recommendations

### For New Code (Immediate)
✅ **Always use**: `hamiltonian(x, m, p, t)`
❌ **Never use**: Other signatures without strong justification

### For Existing Code (Short-term)
⚠️ **Keep working** but plan migration
⚠️ **Document** which signature is used
⚠️ **Test** compatibility with target solvers

### For Solvers (Medium-term)
⚠️ **Implement** HamiltonianAdapter for robustness
⚠️ **Validate** signature at problem creation time
⚠️ **Warn** users about non-standard signatures

### For Package Architecture (Long-term)
⚠️ **Enforce** standard signature in abstract base classes
⚠️ **Deprecate** legacy signatures gracefully
⚠️ **Remove** deprecated signatures in next major version

---

## Testing Strategy

### Signature Detection Tests
```python
def test_hamiltonian_adapter_standard():
    def ham(x, m, p, t=0.0):
        return 0.5 * p**2
    adapter = HamiltonianAdapter(ham)
    assert adapter.signature == "standard"
    assert adapter(1.0, 0.5, 2.0, 0.0) == 2.0

def test_hamiltonian_adapter_legacy():
    def ham(x, p, m, t=0.0):
        return 0.5 * p**2
    adapter = HamiltonianAdapter(ham)
    assert adapter.signature == "legacy"
    assert adapter(1.0, 0.5, 2.0, 0.0) == 2.0  # Adapts to (x, p, m, t)
```

### Integration Tests
- Test all solver types with standard signature
- Test backward compatibility with legacy signatures
- Test adapter performance overhead (should be negligible)

---

## Summary

**Current State**: Inconsistent Hamiltonian signatures across 5+ variants

**Recommended Standard**: `hamiltonian(x, m, p, t)`
- Mathematical clarity: state → control → time
- Dimension-agnostic
- Consistent with GridBasedMFGProblem

**Migration Path**:
1. Document standard (immediate)
2. Implement adapter (1-2 weeks)
3. Update solvers (2-3 weeks)
4. Migrate examples (1-2 weeks)
5. Deprecate legacy (6 months later)

**Effort Estimate**: 4-6 weeks for complete migration

**Priority**: Medium (improves consistency but not blocking)

---

**Document Version**: 1.0
**Author**: Claude Code
**Date**: 2025-11-02
**Status**: Analysis Complete, Proposal for Review
