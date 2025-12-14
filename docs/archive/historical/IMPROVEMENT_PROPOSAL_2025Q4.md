# MFG_PDE Improvement Proposal - Q4 2025

**Date**: October 8, 2025
**Author**: Claude (AI Assistant)
**Status**: 📋 Proposal for Review
**Target**: Phase 3.x enhancements

---

## Executive Summary

This document proposes **three strategic improvement tracks** for MFG_PDE, prioritized by impact on research capability and user experience. All proposals leverage existing infrastructure and require minimal disruption to current codebase.

### Proposed Tracks

| Track | Impact | Effort | Priority | Timeline |
|:------|:-------|:-------|:---------|:---------|
| **A. PyTorch/MPS Boundary Fixes** | 🔴 High | Low (1 month) | **P0** | Nov 2025 |
| **B. Solver Robustness & Diagnostics** | 🟡 Medium | Medium (6 weeks) | **P1** | Dec 2025 - Jan 2026 |
| **C. Example Quality & Coverage** | 🟢 Medium | Low (ongoing) | **P2** | Continuous |

---

## Track A: PyTorch/MPS Boundary Fixes ⚡

**Goal**: Enable full PyTorch/MPS acceleration for RL and Neural paradigms.

### Current State

- ✅ Neural solvers (PINN, DGM): Already PyTorch-native
- ✅ RL infrastructure: Already PyTorch-native
- ⚠️ **Blocker**: Numerical solvers create NumPy arrays → type error when backend is PyTorch
- 📊 **Evidence**: `acceleration_comparison.py` shows MPS initializes but fails with `"can't assign numpy.ndarray to torch.FloatTensor"`

### Root Cause Analysis

**Problem**: Solvers hardcode `np.zeros()` instead of using `backend.zeros()`.

```python
# Current pattern (breaks with PyTorch backend)
class FixedPointIterator:
    def solve(self):
        U = np.zeros((Nt, Nx))  # Hardcoded NumPy
        M = np.zeros((Nt, Nx))
        # ... if backend is PyTorch, type mismatch occurs
```

**Impact**:
- RL environments can't use PyTorch backend for acceleration
- Neural-numerical hybrid methods fail
- No GPU acceleration on Apple Silicon (M1/M2/M3)

### Proposed Solution

**Phase 1: Core Boundary Fixes (Week 1-2)**

**File**: `mfg_pde/alg/numerical/mfg_solvers/fixed_point_iterator.py`

```python
class FixedPointIterator:
    def __init__(self, problem, hjb_solver, fp_solver, backend=None, **kwargs):
        self.problem = problem
        self.hjb_solver = hjb_solver
        self.fp_solver = fp_solver

        # Ensure backend consistency
        if backend is None:
            backend = getattr(hjb_solver, 'backend', create_backend("numpy"))
        self.backend = backend

        # Propagate backend to sub-solvers
        self.hjb_solver.backend = self.backend
        self.fp_solver.backend = self.backend

    def solve(self, max_iterations=100, tolerance=1e-6):
        # Use backend-native array creation
        Nt, Nx = self.problem.Nt, self.problem.Nx

        # Create arrays using backend (not np.zeros!)
        U = self.backend.zeros((Nt + 1, Nx), dtype=self.backend.float64)
        M = self.backend.zeros((Nt + 1, Nx), dtype=self.backend.float64)

        # Initialize with backend-native arrays
        U[-1, :] = self.backend.array(self.problem.terminal_value(...))
        M[0, :] = self.backend.array(self.problem.initial_density(...))

        # Rest of algorithm works transparently
        for iteration in range(max_iterations):
            U_new = self.hjb_solver.solve(M)  # Backend-consistent
            M_new = self.fp_solver.solve(U_new)
            # ...
```

**Files to Modify** (~10 files):
1. ✅ `mfg_pde/alg/numerical/mfg_solvers/fixed_point_iterator.py` - Core coupling
2. ✅ `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py` - HJB base
3. ✅ `mfg_pde/alg/numerical/fp_solvers/fp_particle.py` - FP base
4. ✅ `mfg_pde/factory.py` - Factory functions
5. ✅ Backend initialization in solver constructors

**Phase 2: RL Environment Conversions (Week 3)**

**Pattern**: Explicit conversions at RL ↔ Numerical boundary

```python
class MFGEnvironment(gym.Env):
    """RL environment wrapper for MFG problems."""

    def __init__(self, problem, use_gpu=False):
        # RL side: Always use PyTorch (for GPU training)
        self.device = "mps" if use_gpu and torch.backends.mps.is_available() else "cpu"

        # Numerical solver: Can use any backend (NumPy is fastest for CPU)
        self.numerical_backend = create_backend("numpy")  # Or "jax"
        self.solver = create_standard_solver(problem, backend=self.numerical_backend)

    def step(self, action):
        # Convert PyTorch action → NumPy for solver
        action_np = action.detach().cpu().numpy()

        # Solve numerically (fast NumPy/JAX backend)
        reward_np = self.solver.compute_reward(action_np)

        # Convert NumPy result → PyTorch for RL algorithm
        reward = torch.from_numpy(reward_np).to(self.device)

        return obs, reward, done, info
```

**Locations**:
- `mfg_pde/alg/rl/environments/` - RL environment wrappers
- `examples/basic/rl_*.py` - RL examples

**Phase 3: Testing & Validation (Week 4)**

```python
# New test file: tests/integration/test_backend_compatibility.py

@pytest.mark.parametrize("backend_name", ["numpy", "torch", "jax"])
def test_solver_backend_consistency(backend_name):
    """Verify solvers work with all backends."""
    problem = create_lq_problem()
    backend = create_backend(backend_name)

    solver = create_standard_solver(problem, backend=backend)
    result = solver.solve(max_iterations=10)

    # Should complete without type errors
    assert result.convergence_achieved or result.iterations == 10

    # Verify array types match backend
    if backend_name == "torch":
        assert torch.is_tensor(result.U)
    elif backend_name == "jax":
        assert isinstance(result.U, jax.Array)
    else:
        assert isinstance(result.U, np.ndarray)

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_end_to_end():
    """Test full PyTorch MPS acceleration pipeline."""
    problem = create_lq_problem()
    backend = create_backend("torch", device="mps")

    solver = create_standard_solver(problem, backend=backend)
    result = solver.solve(max_iterations=20, tolerance=1e-4)

    # Verify computation on MPS device
    assert result.U.device.type == "mps"
    assert result.M.device.type == "mps"
```

### Success Metrics

**Quantitative**:
- ✅ All solvers pass with `backend="torch"`
- ✅ MPS test completes without type errors
- ✅ 3-5x speedup on M1/M2 for large problems (Nx > 200)

**Qualitative**:
- ✅ RL training can leverage GPU acceleration
- ✅ Neural-numerical hybrid methods work seamlessly
- ✅ Examples run on Apple Silicon with `device="mps"`

### Estimated Effort

- **Phase 1**: 1-2 weeks (core fixes)
- **Phase 2**: 1 week (RL boundary conversions)
- **Phase 3**: 1 week (testing)
- **Total**: ~1 month

### Risks & Mitigation

| Risk | Impact | Mitigation |
|:-----|:-------|:-----------|
| Backend API inconsistencies | Medium | Comprehensive backend interface tests |
| Performance regression on CPU | Low | Benchmark NumPy vs backend.zeros() |
| Breaking changes to user code | Low | Maintain backward compatibility |

---

## Track B: Solver Robustness & Diagnostics 🔧

**Goal**: Improve solver convergence and provide actionable diagnostics when solvers struggle.

### Current Pain Points

**Evidence from Examples**:
1. `acceleration_comparison.py`: Max iterations reached, no convergence
   ```
   WARNING: Max iterations (20) reached
   Consider: reducing time step, better initialization, or more iterations
   ```

2. `el_farol_bar_demo.py`: Converges, but takes 16 iterations (~50s)

3. User experience: "Why isn't my solver converging?"

### Root Cause Analysis

**Problem 1: Poor Initial Guesses**
```python
# Current: Cold start with zeros
U = np.zeros(shape)  # Often far from solution
```

**Problem 2: Fixed Damping Parameters**
```python
# Current: Fixed damping
damping_factor = 0.7  # May be too aggressive or too conservative
```

**Problem 3: Limited Diagnostics**
```python
# Current: Only final convergence status
Converged: False  # But why? What's going wrong?
```

### Proposed Solutions

#### Enhancement 1: Smart Initialization Strategies

**Add warm start capabilities**:

```python
class FixedPointIterator:
    def solve(self, max_iterations=100, tolerance=1e-6, warm_start=None):
        """
        Solve MFG system with optional warm start.

        Args:
            warm_start: dict with keys 'U', 'M' for initial guess
                       Or 'auto' for automatic initialization
        """
        if warm_start == 'auto':
            # Automatic smart initialization
            U, M = self._smart_initialization()
        elif warm_start is not None:
            U = warm_start['U']
            M = warm_start['M']
        else:
            # Cold start (current behavior)
            U = self.backend.zeros(...)
            M = self.backend.zeros(...)

        # Continue solving...

    def _smart_initialization(self):
        """Generate smart initial guess based on problem structure."""
        # Strategy 1: Use terminal/initial conditions to extrapolate
        U = self._extrapolate_from_terminal()

        # Strategy 2: Solve simplified problem first
        M = self._solve_simplified_fp()

        return U, M
```

**Benefits**:
- Fewer iterations to convergence
- Better success rate on difficult problems
- Reusable solutions (save converged state, reuse for similar problem)

#### Enhancement 2: Adaptive Damping

**Current**: Fixed `damping_factor = 0.7`

**Proposed**: Adaptive damping based on convergence behavior

```python
class AdaptiveDampingStrategy:
    """Adjust damping based on convergence progress."""

    def __init__(self, initial_damping=0.7, patience=5):
        self.damping = initial_damping
        self.patience = patience
        self.stagnation_count = 0
        self.last_error = float('inf')

    def update(self, current_error):
        """Adjust damping based on error reduction."""
        if current_error < self.last_error * 0.95:
            # Good progress: increase damping (more aggressive)
            self.damping = min(0.9, self.damping * 1.05)
            self.stagnation_count = 0
        else:
            # Stagnation: decrease damping (more conservative)
            self.stagnation_count += 1
            if self.stagnation_count >= self.patience:
                self.damping = max(0.3, self.damping * 0.8)
                self.stagnation_count = 0

        self.last_error = current_error
        return self.damping

# Usage in FixedPointIterator
def solve(self, max_iterations=100, adaptive_damping=False):
    if adaptive_damping:
        damping_strategy = AdaptiveDampingStrategy()

    for iteration in range(max_iterations):
        # Solve sub-problems...

        if adaptive_damping:
            self.damping_factor = damping_strategy.update(current_error)

        # Update with adaptive damping...
```

**Benefits**:
- Automatic parameter tuning
- Better convergence on stiff problems
- Reduced manual parameter tweaking

#### Enhancement 3: Rich Convergence Diagnostics

**Add structured diagnostics to `SolverResult`**:

```python
@dataclass
class ConvergenceDiagnostics:
    """Detailed convergence analysis."""

    # Basic metrics
    converged: bool
    iterations: int
    final_error_U: float
    final_error_M: float

    # Convergence behavior
    convergence_rate: float  # Estimated from error history
    stagnation_detected: bool
    oscillation_detected: bool

    # Recommendations
    suggestions: list[str]  # Human-readable improvement suggestions

    def __str__(self):
        """User-friendly diagnostic report."""
        report = f"Convergence: {'✓' if self.converged else '✗'}\n"
        report += f"Iterations: {self.iterations}\n"
        report += f"Final errors: U={self.final_error_U:.2e}, M={self.final_error_M:.2e}\n"

        if not self.converged:
            report += "\nSuggestions:\n"
            for suggestion in self.suggestions:
                report += f"  • {suggestion}\n"

        return report

def analyze_convergence(error_history_U, error_history_M, iterations, tolerance):
    """Generate diagnostic insights from convergence history."""
    diagnostics = ConvergenceDiagnostics(...)

    # Analyze convergence rate
    if iterations > 5:
        rate = np.mean(error_history_U[1:] / error_history_U[:-1])
        diagnostics.convergence_rate = rate

        if rate > 0.95:
            diagnostics.stagnation_detected = True
            diagnostics.suggestions.append(
                "Slow convergence detected. Try: "
                "(1) Reduce time step dt, (2) Increase damping, (3) Better initialization"
            )

    # Detect oscillations
    if len(error_history_U) > 10:
        recent_errors = error_history_U[-10:]
        if np.std(recent_errors) > 0.1 * np.mean(recent_errors):
            diagnostics.oscillation_detected = True
            diagnostics.suggestions.append(
                "Error oscillations detected. Try: "
                "(1) Reduce damping factor, (2) Add Anderson acceleration"
            )

    return diagnostics
```

**Usage**:
```python
result = solver.solve(max_iterations=50)
print(result.diagnostics)

# Output:
# Convergence: ✗
# Iterations: 50
# Final errors: U=2.5e-03, M=1.8e-02
#
# Suggestions:
#   • Slow convergence detected. Try: (1) Reduce time step dt, (2) Increase damping, (3) Better initialization
#   • Error oscillations detected. Try: (1) Reduce damping factor, (2) Add Anderson acceleration
```

#### Enhancement 4: Convergence Visualization Tools

**Add utility for convergence analysis**:

```python
# New file: mfg_pde/utils/convergence_plots.py

def plot_convergence_analysis(result, save_path=None):
    """Generate comprehensive convergence diagnostic plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Error history (log scale)
    ax = axes[0, 0]
    iterations = np.arange(1, len(result.error_history_U) + 1)
    ax.semilogy(iterations, result.error_history_U, 'b-', label='U error')
    ax.semilogy(iterations, result.error_history_M, 'r-', label='M error')
    ax.axhline(result.tolerance, color='k', linestyle='--', label='Tolerance')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Error')
    ax.set_title('Convergence History')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Convergence rate
    ax = axes[0, 1]
    if len(result.error_history_U) > 1:
        rates = result.error_history_U[1:] / result.error_history_U[:-1]
        ax.plot(iterations[1:], rates, 'g-')
        ax.axhline(1.0, color='k', linestyle='--', label='No improvement')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Error reduction ratio')
        ax.set_title('Convergence Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Panel 3: Error components breakdown
    # Panel 4: Diagnostic summary

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
```

### Success Metrics

**Quantitative**:
- ✅ 30% fewer iterations with smart initialization
- ✅ 50% more problems converge within iteration limit
- ✅ Adaptive damping improves convergence rate by 20%

**Qualitative**:
- ✅ Users get actionable suggestions when convergence fails
- ✅ Easier to diagnose difficult problems
- ✅ Reduced trial-and-error parameter tuning

### Estimated Effort

- **Enhancement 1** (Smart init): 1 week
- **Enhancement 2** (Adaptive damping): 1 week
- **Enhancement 3** (Diagnostics): 1 week
- **Enhancement 4** (Visualization): 1 week
- **Testing & docs**: 2 weeks
- **Total**: ~6 weeks

---

## Track C: Example Quality & Coverage 🎓

**Goal**: Ensure all examples run reliably and demonstrate best practices.

### Current State Assessment

**Recent Issues**:
- ✅ v1.7.0 shipped with 3 broken examples (el_farol, santa_fe, towel_beach)
- ✅ Fixed in PR #110 with proper NetworkMFG infrastructure
- ⚠️ `acceleration_comparison.py` shows non-convergence (may confuse users)

### Proposed Quality Standards

#### Standard 1: Example Verification Checklist

**Pre-release checklist** (add to CLAUDE.md):

```markdown
## Example Quality Checklist

Before committing new examples to `examples/basic/` or `examples/advanced/`:

- [ ] **Runs successfully**: Example completes without errors
- [ ] **Converges** (if applicable): Solvers reach tolerance or show expected behavior
- [ ] **Documented**: Comprehensive docstring with theory references
- [ ] **Output saved**: Results saved to `examples/outputs/[category]/`
- [ ] **README updated**: Added to relevant README with difficulty rating
- [ ] **Dependencies noted**: Optional dependencies clearly marked
- [ ] **Timing reasonable**: Completes in < 5 minutes (basic) or < 30 minutes (advanced)
```

#### Standard 2: Example Categories & Difficulty

**Refine difficulty ratings**:

| Rating | Time | Iterations | Complexity | Target Audience |
|:-------|:-----|:-----------|:-----------|:----------------|
| ⭐ | < 30s | < 10 | Single concept | Beginners |
| ⭐⭐ | 30s-5min | 10-50 | 2-3 concepts | Intermediate |
| ⭐⭐⭐ | > 5min | > 50 | Multi-concept | Advanced researchers |

**Current examples needing adjustment**:

1. `acceleration_comparison.py`: Mark as ⭐⭐⭐ (shows non-convergence, requires interpretation)
2. `el_farol_bar_demo.py`: ⭐⭐ (16 iterations, ~50s)
3. `santa_fe_bar_demo.py`: ⭐⭐⭐ (not fully tested, long runtime expected)

#### Standard 3: Automated Example Testing

**Add to CI/CD pipeline**:

```yaml
# .github/workflows/test_examples.yml

name: Example Tests

on: [push, pull_request]

jobs:
  test-basic-examples:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        example:
          - lq_mfg_demo.py
          - el_farol_bar_demo.py
          - towel_beach_demo.py
          # ... all basic examples

    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -e .[all]

      - name: Run example
        run: |
          cd examples/basic
          timeout 300 python ${{ matrix.example }}  # 5 min timeout
        env:
          MPLBACKEND: Agg  # Non-interactive plotting

      - name: Check output exists
        run: |
          ls examples/outputs/basic/
```

**Benefits**:
- Catch broken examples before release
- Verify examples work across Python versions
- Ensure dependencies are correct

#### Standard 4: Example Template

**Create template for new examples**:

```python
# File: examples/templates/example_template.py

"""
[Example Title]: [One-line description]

[Paragraph describing the problem and what this example demonstrates]

Mathematical Formulation:
    [Key equations using LaTeX syntax]

Key Features:
    - Feature 1
    - Feature 2
    - Feature 3

References:
    - Paper citation 1
    - docs/theory/[relevant_theory_file].md

Example Usage:
    python examples/[category]/[example_name].py

Expected Output:
    - Convergence in ~X iterations
    - Saved to examples/outputs/[category]/[output_name].png
    - Runtime: ~X seconds
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde import ExampleMFGProblem  # Or relevant imports
from mfg_pde.factory import create_standard_solver
from mfg_pde.utils.mfg_logging import configure_research_logging, get_logger

# Configure logging
configure_research_logging("example_name", level="INFO")
logger = get_logger(__name__)

# Output directory
EXAMPLE_DIR = Path(__file__).parent
OUTPUT_DIR = EXAMPLE_DIR.parent / "outputs" / "category"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_problem():
    """Create MFG problem instance."""
    # Problem setup
    pass


def visualize_results(result):
    """Generate visualization of results."""
    # Plotting code
    pass


def main():
    """Run example and demonstrate key concepts."""
    logger.info("=" * 70)
    logger.info("[Example Title]")
    logger.info("=" * 70)

    # Step 1: Create problem
    logger.info("\n[1/3] Creating problem...")
    problem = create_problem()

    # Step 2: Solve
    logger.info("\n[2/3] Solving...")
    solver = create_standard_solver(problem)
    result = solver.solve(max_iterations=50, tolerance=1e-4)

    logger.info(f"  Converged: {result.convergence_achieved}")
    logger.info(f"  Iterations: {result.iterations}")

    # Step 3: Visualize
    logger.info("\n[3/3] Generating visualization...")
    fig = visualize_results(result)

    output_path = OUTPUT_DIR / "output_name.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"  Saved: {output_path}")

    plt.show()

    logger.info("\n" + "=" * 70)
    logger.info("[Example Title] Complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
```

### Specific Improvements

#### Fix 1: `acceleration_comparison.py` Convergence

**Current**: Shows non-convergence (confusing for users)

**Fix**: Add note in docstring and output

```python
"""
...

Note: This example uses a challenging problem to stress-test backends.
Non-convergence is expected and demonstrates solver behavior under difficult conditions.
For reliable convergence examples, see lq_mfg_demo.py or el_farol_bar_demo.py.
"""

def main():
    # ... existing code ...

    logger.info("\nNote: This benchmark uses a challenging problem configuration.")
    logger.info("Non-convergence is expected and demonstrates solver stress testing.")
    logger.info("Backends are compared on equal footing (same problem, same iterations).")
```

#### Fix 2: Create "Quick Start" Example

**New**: `examples/basic/quickstart.py`

```python
"""
Quickstart: 5-Minute Introduction to MFG_PDE.

Solves the simplest possible MFG problem in ~10 seconds:
- Linear-Quadratic Hamiltonian
- 1D spatial domain
- Small grid (Nx=50)
- Guaranteed convergence

Perfect first example for new users.
"""

def create_simple_lq_problem():
    """Simplest LQ-MFG problem."""
    def hamiltonian(x, p, m):
        return 0.5 * p**2 + 0.5 * x**2

    return ExampleMFGProblem(
        xmin=-3.0, xmax=3.0, Nx=50,  # Small grid
        T=1.0, Nt=20,  # Few time steps
        sigma=1.0,  # Large diffusion (easier convergence)
        hamiltonian=hamiltonian
    )

def main():
    problem = create_simple_lq_problem()
    solver = create_standard_solver(problem)

    result = solver.solve(max_iterations=20, tolerance=1e-3)

    print(f"✓ Converged in {result.iterations} iterations!")
    print(f"✓ Final error: {result.error_history_U[-1]:.2e}")
    print("\nNext steps:")
    print("  - Try lq_mfg_demo.py for analytical validation")
    print("  - Try el_farol_bar_demo.py for coordination games")
    print("  - See examples/basic/README.md for full catalog")
```

### Success Metrics

**Quantitative**:
- ✅ 100% of basic examples run successfully in CI
- ✅ All examples complete in documented time
- ✅ 0 broken examples shipped in releases

**Qualitative**:
- ✅ New users can run first example in < 5 minutes
- ✅ Examples demonstrate best practices
- ✅ Clear difficulty progression

### Estimated Effort

- **Standards documentation**: 2 days
- **CI setup**: 3 days
- **Fix existing examples**: 1 week
- **Create quickstart**: 2 days
- **Total**: ~2 weeks (ongoing maintenance)

---

## Implementation Priority & Timeline

### Phase 1: Foundation (November 2025)
**Focus**: Track A (PyTorch/MPS fixes)

**Week 1-2**: Core boundary fixes
- Fix `FixedPointIterator` array initialization
- Update solver constructors
- Backend propagation in factories

**Week 3**: RL environment conversions
- Add explicit PyTorch ↔ NumPy conversions
- Update RL examples

**Week 4**: Testing & validation
- Comprehensive backend compatibility tests
- MPS end-to-end validation
- Performance benchmarks

**Deliverable**: ✅ Full PyTorch/MPS support for RL/Neural paradigms

### Phase 2: Robustness (December 2025 - January 2026)
**Focus**: Track B (Solver improvements)

**Weeks 1-4**: Core enhancements
- Smart initialization strategies
- Adaptive damping
- Rich diagnostics
- Convergence visualization

**Weeks 5-6**: Testing & documentation
- Validation on existing examples
- User guide updates
- API documentation

**Deliverable**: ✅ Robust solvers with actionable diagnostics

### Phase 3: Quality (Ongoing)
**Focus**: Track C (Example quality)

**Immediate** (October 2025):
- Fix `acceleration_comparison.py` documentation
- Create quickstart example

**Continuous**:
- CI for example testing
- Example quality checklist enforcement
- Regular example reviews

**Deliverable**: ✅ Reliable, pedagogical examples

---

## Success Criteria & KPIs

### Technical Metrics

| Metric | Baseline | Target | Measurement |
|:-------|:---------|:-------|:------------|
| MPS compatibility | 0% | 100% (RL/Neural) | Test suite pass rate |
| Convergence rate | ~50-60% | 80%+ | % examples converging |
| Iteration reduction | Baseline | -30% | Smart init impact |
| Example reliability | 75% (3/12 broken) | 100% | CI pass rate |

### User Experience Metrics

| Metric | Current | Target | How to Measure |
|:-------|:--------|:-------|:---------------|
| Time to first success | ~30 min | < 5 min | Quickstart runtime |
| GPU utilization | 0% | > 80% (on MPS) | RL training monitoring |
| Diagnostic clarity | Low | High | User feedback |
| Example trust | Medium | High | GitHub issues/questions |

---

## Risk Assessment

### Track A Risks

| Risk | Probability | Impact | Mitigation |
|:-----|:-----------|:-------|:-----------|
| Backend API breaks existing code | Low | High | Backward compatibility layer |
| PyTorch overhead on CPU | Medium | Low | Benchmark and document |
| Type conversion bugs | Medium | Medium | Comprehensive test coverage |

### Track B Risks

| Risk | Probability | Impact | Mitigation |
|:-----|:-----------|:-------|:-----------|
| Smart init worse than cold start | Low | Medium | Make warm start optional |
| Adaptive damping instability | Medium | Medium | Conservative defaults, opt-in |
| Over-engineering diagnostics | Low | Low | Start simple, iterate |

### Track C Risks

| Risk | Probability | Impact | Mitigation |
|:-----|:-----------|:-------|:-----------|
| CI overhead slows development | Medium | Low | Run on PR only, not every commit |
| Example maintenance burden | Medium | Low | Template + checklist |
| False sense of quality | Low | Medium | Manual review still required |

---

## Resource Requirements

### Development Time

| Track | Weeks | FTE | Priority |
|:------|:------|:----|:---------|
| A (PyTorch/MPS) | 4 | 1.0 | P0 |
| B (Robustness) | 6 | 0.8 | P1 |
| C (Examples) | 2 + ongoing | 0.2 | P2 |

**Total**: ~3 months for all tracks (can parallelize B and C)

### Infrastructure

- ✅ GitHub Actions (already available)
- ✅ PyTorch, JAX (already installed)
- ✅ Test infrastructure (already exists)
- ⚠️ Apple Silicon runner (optional, for MPS CI testing)

---

## Alternatives Considered

### Alternative 1: Full Tensor Rewrite (Rejected)
**Approach**: Rewrite all numerical solvers to be tensor-native

**Pros**: Complete flexibility, maximum performance
**Cons**: 6+ months effort, high risk, breaks existing code

**Decision**: ❌ Rejected - Backend abstraction is sufficient

### Alternative 2: Separate PyTorch Solvers (Rejected)
**Approach**: Duplicate solvers for PyTorch vs NumPy

**Pros**: No risk to existing code
**Cons**: Code duplication, maintenance nightmare

**Decision**: ❌ Rejected - Violates DRY principle

### Alternative 3: External Diagnostics Package (Considered)
**Approach**: Separate package for convergence analysis

**Pros**: Clean separation, optional dependency
**Cons**: Integration overhead, fragmentation

**Decision**: ⏸️ Deferred - Start integrated, extract later if needed

---

## Recommendation

**Approve Track A (PyTorch/MPS) for immediate implementation** (November 2025):
- ✅ High impact (enables GPU acceleration for RL/Neural)
- ✅ Low risk (boundary fixes only, no algorithm changes)
- ✅ Short timeline (1 month)
- ✅ Clear success criteria

**Approve Track B (Robustness) for Q4 2025/Q1 2026**:
- ✅ Medium-high impact (better user experience)
- ✅ Medium complexity (new features, well-scoped)
- ✅ Reasonable timeline (6 weeks)

**Approve Track C (Examples) as continuous process**:
- ✅ Important for maintainability
- ✅ Low overhead with automation
- ✅ Prevents future v1.7.0-type issues

---

## Next Steps

1. **Review this proposal** with maintainer (@derrring)
2. **Create GitHub issues** for each track:
   - Issue: "Enable PyTorch/MPS acceleration for RL/Neural paradigms"
   - Issue: "Improve solver robustness and diagnostics"
   - Issue: "Enhance example quality and reliability"
3. **Begin Track A implementation** (if approved)
4. **Monitor JAX Metal status** (quarterly reviews)

---

**Proposal prepared by**: Claude (AI Assistant)
**Date**: October 8, 2025
**Review due**: October 15, 2025
**Implementation start**: November 1, 2025 (if approved)
