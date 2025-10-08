# Apple Silicon (MPS) Acceleration Tracking

**Status**: ⏳ Monitoring upstream development
**Priority**: Medium (Performance enhancement)
**Target**: Phase 3.x (Backend optimization)

## Overview

Track Apple Silicon (M1/M2/M3) GPU acceleration support across backend frameworks to enable native MPS acceleration for MFG solvers.

## Current Status (October 2025)

### PyTorch MPS
- **Status**: ✅ Available and stable
- **Detection**: Working (`torch.backends.mps.is_available()`)
- **MFG_PDE Integration**: ⚠️ Backend initializes, but solver has tensor type incompatibilities
- **Issue**: `"can't assign numpy.ndarray to torch.FloatTensor"`
- **Root Cause**: Solver internals mix NumPy arrays with PyTorch tensors
- **Required Work**: Consistent tensor type handling in solver pipeline

### JAX MPS (Metal)
- **Status**: 🧪 Experimental (`jax-metal` package)
- **Official Support**: Not yet production-ready
- **Documentation**: https://github.com/google/jax/tree/main/jax_plugins/metal_plugin
- **MFG_PDE Integration**: ❌ Not attempted (waiting for stable release)
- **Tracking**: Monitor JAX releases for production-ready Metal backend

### Current Workaround
All backends run on CPU on Apple Silicon:
- NumPy: Native CPU
- JAX: CPU-only (no Metal)
- PyTorch: CPU fallback (MPS incompatible with current solver)
- Numba: JIT-compiled CPU

## Implementation Roadmap

### Phase 1: PyTorch MPS Integration (Near-term)
**Goal**: Enable PyTorch MPS for existing solvers

**Required Changes**:
1. **Solver Tensor Type Consistency**
   - Ensure all solver operations use backend-native arrays
   - Add type conversion at solver boundaries
   - Test with `HJBFDMSolver`, `FPParticleSolver`

2. **Backend Interface Refinement**
   - Guarantee backend array types propagate correctly
   - Add runtime type checking/conversion utilities
   - Handle edge cases (boundary conditions, initialization)

3. **Validation**
   - Run all examples with `device="mps"`
   - Compare accuracy vs NumPy baseline
   - Benchmark performance improvements

**Estimated Effort**: 2-3 weeks
**Dependencies**: None (can start immediately)

### Phase 2: JAX Metal Support (Future)
**Goal**: Enable JAX GPU acceleration on Apple Silicon when production-ready

**Tracking Criteria**:
- JAX official announcement of stable Metal backend
- `jax-metal` package reaches v1.0 or recommended for production
- Community adoption and stability reports

**Required Changes**:
1. Update `mfg_pde/backends/__init__.py` to detect JAX Metal
2. Add `jax_metal` to available backends check
3. Test solver compatibility
4. Update documentation

**Estimated Effort**: 1 week (assuming JAX handles device abstraction well)
**Dependencies**: JAX Metal stable release

## Monitoring Resources

### JAX Metal Plugin
- **GitHub**: https://github.com/google/jax/tree/main/jax_plugins/metal_plugin
- **Installation**: `pip install jax-metal`
- **Check Status**: Monitor JAX release notes

### PyTorch MPS
- **Documentation**: https://pytorch.org/docs/stable/notes/mps.html
- **Status**: Stable since PyTorch 1.12 (2022)
- **Known Limitations**:
  - Limited float64 support (uses float32)
  - Some operations fallback to CPU

### Community Tracking
- JAX GitHub Discussions: https://github.com/google/jax/discussions
- PyTorch Forums: https://discuss.pytorch.org/c/mps/
- MFG_PDE should check quarterly for updates

## Expected Performance Gains

Based on typical MPS acceleration benchmarks:

| Backend | Current (CPU) | Expected (MPS) | Speedup |
|:--------|:--------------|:---------------|:--------|
| NumPy | Baseline | N/A | 1.0x |
| PyTorch | ~1.0x | 3-5x | 3-5x |
| JAX | ~1.0x | 4-8x | 4-8x |
| Numba | ~1.0x | N/A | 1.0x |

**Notes**:
- Speedup varies by problem size (larger = better)
- Memory transfer overhead affects small problems
- Best for iterative solvers (HJB, FP)

## Decision Points

### When to Integrate PyTorch MPS
- ✅ **Now**: Start refactoring for tensor type consistency
- 📅 **Target**: Phase 3.2 (Q4 2025)
- 🎯 **Success Metric**: 3x+ speedup on M1/M2/M3 for large problems (Nx > 200)

### When to Integrate JAX Metal
- ⏸️ **Wait**: Until official stable release announcement
- 📊 **Monitor**: Quarterly checks (Jan/Apr/Jul/Oct)
- ✅ **Integrate**: When declared production-ready by JAX team

## Implementation Notes

### Tensor Type Consistency Pattern
```python
# Current (problematic for MPS)
def solve(self):
    u = np.zeros(self.shape)  # NumPy array
    result = self.backend.compute(u)  # Backend expects tensor
    return result  # Type mismatch!

# Fixed (backend-aware)
def solve(self):
    u = self.backend.zeros(self.shape)  # Backend-native array
    result = self.backend.compute(u)  # Consistent types
    return result  # Works with MPS!
```

### Detection Pattern for JAX Metal
```python
def get_available_backends():
    backends = {}

    # ... existing JAX detection ...

    # JAX Metal (when available)
    try:
        import jax
        metal_devices = jax.devices("metal")
        backends["jax_metal"] = len(metal_devices) > 0
    except:
        backends["jax_metal"] = False

    return backends
```

## Related Issues

- Issue #110: Network MFG basic examples (includes acceleration_comparison.py)
- Future: Create issue for "PyTorch MPS solver integration"
- Future: Create issue for "JAX Metal backend support"

## References

1. **JAX Metal Plugin**: https://github.com/google/jax/tree/main/jax_plugins/metal_plugin
2. **PyTorch MPS Backend**: https://pytorch.org/docs/stable/notes/mps.html
3. **Apple Metal Performance Shaders**: https://developer.apple.com/metal/
4. **MFG_PDE Backend System**: `mfg_pde/backends/`

---

**Last Updated**: October 8, 2025
**Next Review**: January 2026
**Owner**: @derrring
