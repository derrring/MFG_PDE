# Particle Collocation Solver Consolidation Guide

## Overview

The MFG_PDE package has consolidated multiple redundant particle collocation solver variants into a streamlined architecture. This guide explains the changes and provides migration instructions.

## Changes Made

### **1. Removed Deprecated Classes**

**Removed:**
- `EnhancedParticleCollocationSolver` (completely removed)

**Deprecated but Kept for Compatibility:**
- `SilentAdaptiveParticleCollocationSolver` → Use `AdaptiveParticleCollocationSolver(verbose=False)`
- `HighPrecisionAdaptiveParticleCollocationSolver` → Use `AdaptiveParticleCollocationSolver(precision="high")`
- `QuietAdaptiveParticleCollocationSolver` → Use `AdaptiveParticleCollocationSolver(verbose=False)`

### **2. Unified Adaptive Solver**

**Before (4 separate classes):**
```python
# Different classes for different configurations
solver1 = AdaptiveParticleCollocationSolver(...)  # Standard
solver2 = SilentAdaptiveParticleCollocationSolver(...)  # Quiet
solver3 = HighPrecisionAdaptiveParticleCollocationSolver(...)  # High precision
solver4 = QuietAdaptiveParticleCollocationSolver(...)  # Deprecated alias
```

**After (1 configurable class):**
```python
# Single class with precision and verbosity parameters
solver1 = AdaptiveParticleCollocationSolver(precision="standard", verbose=True, ...)
solver2 = AdaptiveParticleCollocationSolver(precision="standard", verbose=False, ...)
solver3 = AdaptiveParticleCollocationSolver(precision="high", verbose=True, ...)
solver4 = AdaptiveParticleCollocationSolver(precision="fast", verbose=False, ...)
```

### **3. Simplified Architecture**

**Current Clean Architecture:**
- **`ParticleCollocationSolver`** - Base implementation
- **`AdaptiveParticleCollocationSolver`** - Configurable adaptive convergence with decorator pattern
- **`MonitoredParticleCollocationSolver`** - Direct integration of advanced convergence monitoring

## Migration Guide

### **1. Replace Deprecated Classes**

```python
# OLD: Multiple specific classes
from mfg_pde.alg.mfg_solvers import (
    SilentAdaptiveParticleCollocationSolver,
    HighPrecisionAdaptiveParticleCollocationSolver,
    EnhancedParticleCollocationSolver  # Will cause ImportError
)

# NEW: Single configurable class
from mfg_pde.alg.mfg_solvers import AdaptiveParticleCollocationSolver
```

### **2. Update Solver Creation**

```python
# OLD: Specific classes
silent_solver = SilentAdaptiveParticleCollocationSolver(problem, points)
precise_solver = HighPrecisionAdaptiveParticleCollocationSolver(problem, points)

# NEW: Configuration parameters
silent_solver = AdaptiveParticleCollocationSolver(problem, points, verbose=False)
precise_solver = AdaptiveParticleCollocationSolver(problem, points, precision="high")
```

### **3. Factory Usage**

The factory has been updated to use the consolidated architecture:

```python
from mfg_pde.factory import create_solver

# Factory automatically uses AdaptiveParticleCollocationSolver with verbose=False
adaptive_solver = create_solver(problem, solver_type="adaptive_particle", preset="fast")
```

### **4. Convenience Functions**

Use the new convenience functions for common configurations:

```python
from mfg_pde.alg.mfg_solvers.adaptive_particle_collocation_solver import (
    create_fast_adaptive_solver,
    create_accurate_adaptive_solver,
    create_silent_adaptive_solver
)

# Quick creation for common use cases
fast_solver = create_fast_adaptive_solver(problem, points)
accurate_solver = create_accurate_adaptive_solver(problem, points)
silent_solver = create_silent_adaptive_solver(problem, points)
```

## Precision Levels

The unified `AdaptiveParticleCollocationSolver` supports three precision levels:

### **"fast"** - Optimized for Speed
- `classical_tol ≥ 5e-3`
- `wasserstein_tol ≥ 5e-4`
- `u_magnitude_tol ≥ 5e-3`
- `history_length ≤ 5`

### **"standard"** - Balanced (Default)
- Uses constructor parameter values as-is
- Good balance of speed and accuracy

### **"high"** - Optimized for Accuracy
- `classical_tol ≤ 5e-4`
- `wasserstein_tol ≤ 5e-5`
- `u_magnitude_tol ≤ 5e-4`
- `history_length ≥ 15`

## Architecture Benefits

### **Before Consolidation Problems:**
- 4 nearly identical classes for different configurations
- Decorator vs direct integration redundancy
- Confusing factory methods that called each other
- Deprecated classes still exported

### **After Consolidation Benefits:**
- Single configurable class with clear parameters
- Eliminated architectural redundancy
- Streamlined factory integration
- Clear deprecation path with warnings
- Better maintainability and user experience

## Compatibility

### **Breaking Changes:**
- `EnhancedParticleCollocationSolver` - **REMOVED** (use `MonitoredParticleCollocationSolver`)

### **Deprecation Warnings:**
- `SilentAdaptiveParticleCollocationSolver` - Will show deprecation warning
- `HighPrecisionAdaptiveParticleCollocationSolver` - Will show deprecation warning
- `QuietAdaptiveParticleCollocationSolver` - Will show deprecation warning

### **Backward Compatibility:**
- All deprecated classes still work but show warnings
- Factory methods continue to work
- Existing code will run but should be updated

## Recommended Action

1. **Update imports** to use `AdaptiveParticleCollocationSolver` directly
2. **Replace deprecated classes** with configured instances
3. **Use precision parameter** instead of specific precision classes
4. **Test thoroughly** after migration to ensure behavior is preserved

---

**Implementation Date**: 2025-09-19
**Status**: ✅ **COMPLETED**
**Breaking Changes**: Minimal (only `EnhancedParticleCollocationSolver` removed)
**Migration Required**: Update imports and class usage