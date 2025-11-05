# MFG_PDE Solver Hierarchy Strategy

**Date**: 2025-10-05
**Status**: Strategic Framework
**Purpose**: Establish clear solver hierarchy for development and applications

## Solver Hierarchy

### 📊 Three-Tier System

```
┌─────────────────────────────────────────────────────────┐
│  TIER 1: BASIC SOLVER (Benchmark, Poor Quality)        │
│  • HJB-FDM + FP-FDM (Upwind + Damped Fixed Point)      │
│  • Fast, simple, but approximate                        │
│  • Benchmark for validating advanced methods            │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  TIER 2: STANDARD SOLVER (Production, Good Quality)    │
│  • HJB-FDM + FP-Particle (Hybrid)                       │
│  • Reliable mass conservation                           │
│  • Default for most applications                        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  TIER 3: ADVANCED SOLVERS (Research, High Quality)     │
│  • Semi-Lagrangian, WENO, DGM, etc.                     │
│  • Specialized for specific applications                │
│  • Performance/accuracy trade-offs                      │
└─────────────────────────────────────────────────────────┘
```

## Tier 1: Basic Solver (Benchmark)

### HJB-FDM + FP-FDM

**Characteristics:**
- **Quality**: Poor (1-10% mass error)
- **Speed**: Fast (simple discretization)
- **Purpose**: Benchmark and validation reference
- **Method**: Upwind + Damped Fixed Point

**Configuration:**
```python
from mfg_pde.factory import create_basic_solver

# Tier 1: Basic FDM benchmark
solver = create_basic_solver(problem)
```

**Use Cases:**
1. **Benchmarking**: Reference for validating advanced methods
2. **Quick testing**: Rapid prototyping and debugging
3. **Teaching**: Simple numerical scheme for education
4. **Comparison**: Baseline for performance comparisons

**Strengths:**
- Simple implementation
- Fast execution
- Easy to understand
- Stable with damping

**Limitations:**
- Poor mass conservation (1-10% error)
- Requires careful damping tuning (θ ∈ [0.5, 0.7])
- Slow Picard convergence (50-100 iterations)
- First-order accuracy O(Δx, Δt)

**Factory Implementation:**
```python
def create_basic_solver(problem, **kwargs):
    """
    Create basic FDM benchmark solver (Tier 1).

    Uses HJB-FDM + FP-FDM with upwind + damped fixed point.
    Fast but approximate - primarily for benchmarking.
    """
    from mfg_pde.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
    from mfg_pde.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
    from mfg_pde.alg.numerical.mfg_solvers import FixedPointIterator

    hjb_solver = HJBFDMSolver(problem=problem)
    fp_solver = FPFDMSolver(problem=problem)

    # Default damping optimized for FDM stability
    damping = kwargs.pop('damping', 0.6)
    max_iterations = kwargs.pop('max_iterations', 100)

    return FixedPointIterator(
        problem=problem,
        hjb_solver=hjb_solver,
        fp_solver=fp_solver,
        damping_factor=damping,
        **kwargs
    )
```

## Tier 2: Standard Solver (Production)

### HJB-FDM + FP-Particle (Hybrid)

**Characteristics:**
- **Quality**: Good (10⁻¹⁵ mass error)
- **Speed**: Moderate (5000 particles)
- **Purpose**: Default production solver
- **Method**: FDM for HJB, Lagrangian particles for FP

**Configuration:**
```python
from mfg_pde.factory import create_fast_solver

# Tier 2: Standard hybrid solver (DEFAULT)
solver = create_fast_solver(problem, "fixed_point")
```

**Use Cases:**
1. **Production**: Default for applications
2. **Research**: Reliable baseline for experiments
3. **Validation**: Ground truth for mass conservation
4. **Publications**: Standard method for papers

**Strengths:**
- Perfect mass conservation (10⁻¹⁵ error)
- Fast convergence (10-20 iterations)
- No damping tuning needed
- Robust and reliable

**Limitations:**
- More memory (5000 particles)
- Slightly slower than pure FDM
- Stochastic noise in density

**Factory Implementation (Current):**
```python
def create_fast_solver(problem, solver_type="fixed_point", **kwargs):
    """
    Create standard production solver (Tier 2).

    Default: HJB-FDM + FP-Particle hybrid.
    Reliable mass conservation, fast convergence.
    """
    if solver_type == "fixed_point" and "fp_solver" not in kwargs:
        from mfg_pde.alg.numerical.fp_solvers.fp_particle import FPParticleSolver
        from mfg_pde.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver

        hjb_solver = HJBFDMSolver(problem=problem)
        fp_solver = FPParticleSolver(problem=problem, num_particles=5000)

        kwargs['hjb_solver'] = hjb_solver
        kwargs['fp_solver'] = fp_solver

    # ... rest of factory logic
```

## Tier 3: Advanced Solvers (Research)

### Specialized High-Performance Methods

**Available Advanced Solvers:**

1. **Semi-Lagrangian**
   - High accuracy for smooth solutions
   - Good CFL independence
   - Complex interpolation

2. **WENO Family**
   - High-order accuracy (up to 5th order)
   - Shock capturing capability
   - Computationally expensive

3. **Deep Galerkin (DGM)**
   - High-dimensional capability
   - Neural network based
   - GPU acceleration

4. **Particle Collocation**
   - Adaptive particle distribution
   - Quadratic programming
   - Advanced mass conservation

**Configuration:**
```python
from mfg_pde.factory import create_accurate_solver

# Tier 3: Advanced methods
solver = create_accurate_solver(problem, solver_type="weno")
solver = create_accurate_solver(problem, solver_type="semi_lagrangian")
solver = create_accurate_solver(problem, solver_type="dgm")
```

**Use Cases:**
1. **Research**: Novel algorithm development
2. **High-accuracy needs**: Precision applications
3. **Specialized problems**: Domain-specific requirements
4. **Performance optimization**: GPU/parallel computing

## Benchmarking Strategy

### Primary Benchmark: Tier 1 vs Tier 2

**Always compare advanced methods against both:**

1. **Basic FDM (Tier 1)**: Simplest baseline
   - Measures improvement in accuracy
   - Shows convergence benefits
   - Validates mass conservation gains

2. **Hybrid (Tier 2)**: Standard reference
   - Production-quality comparison
   - Practical performance benchmark
   - Real-world applicability

**Example Benchmark Suite:**
```python
from mfg_pde.factory import create_basic_solver, create_fast_solver

# Tier 1: Basic benchmark
solver_basic = create_basic_solver(problem)
result_basic = solver_basic.solve()

# Tier 2: Standard solver
solver_standard = create_fast_solver(problem, "fixed_point")
result_standard = solver_standard.solve()

# Tier 3: Your advanced method
solver_advanced = YourAdvancedSolver(problem)
result_advanced = solver_advanced.solve()

# Compare all three
compare_solvers(result_basic, result_standard, result_advanced)
```

### Benchmark Metrics

**For all comparisons:**

| Metric | Basic (T1) | Standard (T2) | Advanced (T3) |
|--------|------------|---------------|---------------|
| **Mass conservation** | ~1-10% | ~10⁻¹⁵ | ??? |
| **Convergence rate** | 50-100 iter | 10-20 iter | ??? |
| **Execution time** | Fast | Moderate | ??? |
| **Memory usage** | Low | Moderate | ??? |
| **Accuracy (L²)** | O(Δx, Δt) | O(Δx², Δt) | ??? |

**Success criteria for Tier 3 methods:**
- Better than Tier 1 in at least 2 metrics
- Competitive with Tier 2 in mass conservation
- Clear advantage in specific use case

## Implementation Roadmap

### Phase 1: Factory Updates ✅ (Current PR #80)
- [x] Tier 2 (Hybrid) as default
- [x] Tier 1 (FDM) available via explicit choice
- [ ] Add `create_basic_solver()` convenience function

### Phase 2: Documentation
- [ ] Update all examples to use standard solver
- [ ] Create benchmarking template
- [ ] Document when to use each tier

### Phase 3: Testing Infrastructure
- [ ] Benchmark suite comparing all 3 tiers
- [ ] Performance regression tests
- [ ] Mass conservation validation tests

## Usage Guidelines

### For Users (Applications)

**Default choice:**
```python
# Use Tier 2 (Standard) for most applications
solver = create_fast_solver(problem, "fixed_point")
```

**When to use Tier 1 (Basic FDM):**
- Quick testing/debugging
- Educational purposes
- Simple benchmarking
- Very large grids where speed matters more than accuracy

**When to use Tier 3 (Advanced):**
- High accuracy requirements
- Specialized problem types
- Research and development
- Performance-critical applications

### For Developers (New Methods)

**Validation protocol:**
1. **Implement** your Tier 3 method
2. **Benchmark** against Tier 1 (Basic FDM)
   - Measure improvement in key metrics
   - Document where it's better/worse
3. **Compare** with Tier 2 (Standard Hybrid)
   - Production-quality baseline
   - Real-world performance
4. **Document** specific use cases where Tier 3 excels

**Example workflow:**
```python
# Step 1: Implement
class MyAdvancedSolver(BaseMFGSolver):
    # ... implementation

# Step 2: Benchmark against Tier 1
solver_basic = create_basic_solver(problem)
solver_mine = MyAdvancedSolver(problem)
benchmark_comparison(solver_basic, solver_mine)

# Step 3: Compare with Tier 2
solver_standard = create_fast_solver(problem, "fixed_point")
production_comparison(solver_standard, solver_mine)

# Step 4: Document advantages
"""
MyAdvancedSolver outperforms:
- Tier 1 (Basic): 10x better mass conservation
- Tier 2 (Standard): 2x faster for high-dimensional problems
Recommended for: problems with d > 3
"""
```

## Summary Table

| Tier | Name | Quality | Speed | Use Case | Default? |
|------|------|---------|-------|----------|----------|
| **1** | Basic FDM | Poor | Fast | Benchmark | No |
| **2** | Hybrid | Good | Moderate | Production | **Yes** ✅ |
| **3** | Advanced | Varies | Varies | Research | No |

## Factory API

### Proposed Complete API

```python
# Tier 1: Basic benchmark solver
from mfg_pde.factory import create_basic_solver
solver = create_basic_solver(problem, damping=0.6, max_iterations=100)

# Tier 2: Standard production solver (default)
from mfg_pde.factory import create_fast_solver
solver = create_fast_solver(problem, "fixed_point")  # Uses hybrid

# Tier 3: Advanced solvers
from mfg_pde.factory import create_accurate_solver
solver = create_accurate_solver(problem, solver_type="weno")
solver = create_accurate_solver(problem, solver_type="semi_lagrangian")
solver = create_accurate_solver(problem, solver_type="dgm")

# Or explicitly specify components
solver = create_fast_solver(
    problem,
    "fixed_point",
    hjb_solver=MyCustomHJBSolver(problem),
    fp_solver=MyCustomFPSolver(problem)
)
```

## Conclusion

**Strategic Framework:**
1. **Tier 1 (Basic FDM)**: Simple benchmark, poor quality but fast
2. **Tier 2 (Hybrid)**: Standard solver, good quality and reliable ✅ **DEFAULT**
3. **Tier 3 (Advanced)**: Research methods, specialized use cases

**Benchmarking:**
- Always compare new methods against **both** Tier 1 and Tier 2
- Tier 1: Shows improvement in fundamentals
- Tier 2: Shows practical advantage

**Development:**
- Build advanced solvers on top of established benchmarks
- Maintain Tier 1 and Tier 2 as stable references
- Document where Tier 3 methods excel

This hierarchy provides a clear path for both users (choose the right tool) and developers (validate new methods).

---

**Status**: Strategic framework established
**Next Steps**: Implement `create_basic_solver()` and update documentation
**Maintainer**: Keep Tier 1 and Tier 2 stable as benchmarks
