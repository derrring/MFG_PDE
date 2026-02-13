# Claude Code Preferences for MFG_PDE

This file contains preferences and conventions for Claude Code when working with the MFG_PDE repository.

## ⚠️ **Communication Principles**

**Objectivity and Honesty**:
- ❌ **NO flattery, NO superlatives, NO praise**
- ✅ **Factual, technical, honest assessment** of code and results
- ✅ **Point out issues, limitations, and areas needing improvement**
- ✅ **Acknowledge uncertainty**: "This may have issues", "Needs validation"

**Example**: Say "Implementation complete. Validation needed" not "Excellent work! This implementation is amazing!"

**Rationale**: Research and production code require objective critique, not encouragement.

---

## 🎯 **Repository Mission & Scope** ⚠️ **CRITICAL**

### **MFG_PDE: Public Infrastructure Package**
Production-ready infrastructure for Mean Field Games research and applications.

**Scope**:
- ✅ Core infrastructure (solvers, backends, config, geometry, workflow, visualization)
- ✅ Classical algorithms (FDM, FEM, GFDM, DGM, PINN, Actor-Critic, PPO)
- ✅ Standard examples (LQ, crowd motion, traffic flow, tutorials)

### **MFG-Research: Private Research Repository**
Novel research, experimental algorithms, unpublished methods.

**Scope**:
- 🔬 Research algorithms (novel schemes, experimental architectures)
- 🔬 Research applications (case studies, parameter studies, publications)

**Key Principle**: MFG-Research **imports** MFG_PDE but **never modifies** it.

### **Decision Criteria**

| Criterion | MFG_PDE (Public) | MFG-Research (Private) |
|:----------|:-----------------|:-----------------------|
| **Maturity** | Production-ready, tested | Experimental |
| **Publication** | Published methods | Unpublished |
| **Stability** | Stable API, versioned | Breaking changes OK |
| **Documentation** | Comprehensive | Minimal |
| **Testing** | Full coverage | Exploratory |

### **Migration Path: Research → Infrastructure**
When research matures: Add tests, write docs, ensure API consistency, open PR in MFG_PDE.

### **Bug Fixes from Research** ⚠️ **CRITICAL**
When bugs are discovered and validated in mfg-research:

**Requirements before modifying MFG_PDE**:
1. GitHub issue created with quantified validation evidence
2. Standalone validation experiment in mfg-research demonstrating the fix
3. Discussion and approval of approach
4. Reference to validation experiment in code comments

**Code Comment Format**:
```python
# Issue #542 fix. Validated in:
# mfg-research/experiments/crowd_evacuation_2d/runners/exp14b_fdm_bc_fix_validation.py
# Achieves 23x error reduction (47.98% -> 2.06%) for 1D corridor evacuation problem.
```

**Principles**:
- ❌ Don't add legacy fallbacks - use mature utilities directly
- ❌ Don't make broad changes without discussion
- ✅ Keep changes minimal and focused
- ✅ Run tests before and after changes
- ✅ Verify by re-running research experiments

---

## 🎨 **API Design Principles** ⚠️ **CRITICAL**

### **No Premature Convenience**
Do NOT add convenience wrappers, factory shortcuts, or "easy modes" until the core API is stable and mature.

**Why**: Premature convenience creates technical debt, confuses users, and obscures the actual API.

**Anti-patterns to avoid**:
- ❌ Multiple ways to do the same thing (`solve_mfg()` vs `create_solver().solve()`)
- ❌ Magic parameters (`method="fast"` vs explicit `tolerance=1e-4`)
- ❌ Wrapper functions that hide the real API
- ❌ Factory explosion (8 solver factories when 1 suffices)

**Correct approach**:
- ✅ One clear path: `problem.solve()`
- ✅ Explicit parameters: `tolerance=1e-6, max_iterations=100`
- ✅ Domain model IS the API: `MFGProblem` knows how to solve itself
- ✅ Advanced options via composition, not factory proliferation

### **Primary API Pattern**
```python
# The primary API is simple and direct
problem = MFGProblem(...)
result = problem.solve()

# Customization via explicit parameters
result = problem.solve(max_iterations=200, tolerance=1e-8)

# Advanced: Factory API only when truly needed
solver = create_standard_solver(problem, custom_config=config)
```

### **When to Add Convenience**
Only after v1.0.0 when:
- Core API is stable and well-tested
- Clear user demand exists
- The convenience doesn't obscure the primary API

---

## 🏗️ **Repository Structure**

### **Top-Level Directories**
- **`mfg_pde/`** - Core package code
- **`tests/`** - Unit and integration tests only
- **`benchmarks/`** - Performance benchmark scripts (peer to `tests/`)
- **`examples/`** - Demos organized by complexity: `basic/`, `advanced/`, `notebooks/`, `tutorials/`
- **`docs/`** - Documentation by category
- **`archive/`** - Historical code (do not modify)

### **Package Structure** (`mfg_pde/`)
`alg/`, `backends/`, `config/`, `core/`, `factory/`, `geometry/`, `hooks/`, `solvers/`, `types/`, `utils/`, `visualization/`, `workflow/`, `compat/`, `meta/`

---

## 📝 **Coding Standards**

### **Type Checking Philosophy** ⚠️ **IMPORTANT**
Pragmatic approach optimized for research:
- ✅ **Strict typing**: Public APIs, factory functions, core algorithms, config systems
- ⚠️ **Relaxed typing**: Utilities, examples, optional dependencies

### **Modern Python Typing** ⚠️ **CRITICAL**

**Division of Responsibility**:

| Tool | Primary User | Purpose | When to Use |
|:-----|:-------------|:--------|:------------|
| **`@overload`** | Library Author | Precise type signatures | Multiple return types |
| **`isinstance()`** | Both | Runtime type checking | Safe branching, type guards |
| **`cast()`** | Library User | Override imprecise hints | Library hints too generic |

**Library Authors**: Use `@overload` for factory functions with multiple return types.

**Library Users**: Use `isinstance()` for type branching, `cast()` sparingly for external library issues.

### **Import Style**
```python
from mfg_pde import MFGProblem, BoundaryConditions
from mfg_pde.factory import create_fast_solver
from mfg_pde.utils.mfg_logging import get_logger, configure_research_logging
```

### **Documentation Style**
- Use comprehensive docstrings with LaTeX math: `$u(t,x)$`, `$m(t,x)$`
- Reference code locations: `file_path:line_number`
- Follow `docs/development/CONSISTENCY_GUIDE.md`

### **Mathematical Typesetting**
- **Theory docs (.md)**: Full LaTeX using `$...$` delimiters
- **Docstrings**: UTF-8 math symbols allowed (e.g., `∂u/∂t + H(∇u) = 0`)
- **Code output/logging**: ASCII only for terminal compatibility

### **Text and Symbol Standards** ⚠️ **CRITICAL**
**NO emojis in Python files** (code comments, docstrings, output):
```python
# ✅ GOOD - Plain ASCII
# Solve the HJB equation using Newton iteration
print("Convergence: ||u - u_prev|| < 1e-6")

# ❌ BAD - No emojis
# 🚀 Solve the HJB equation
print("Convergence: ‖u - u_prev‖ < 1e-6")
```

**Emoji usage**: ✅ Markdown docs | ❌ Python files, output, docstrings

### **Fail Fast & Surface Problems** ⚠️ **CRITICAL**
Prioritize surfacing problems early during development:
- ✅ **Let problems emerge**: Allow exceptions to propagate rather than catching and silencing them.
- ❌ **NO silent fallbacks**: Do not provide default values or fallbacks for failed operations without explicit permission.
- ❌ **NO over-defensive programming**: Avoid excessive null checks or safety guards that mask logic errors.
- ❌ **NO `hasattr()` for duck typing**: Use explicit interfaces (Protocol/ABC) or `getattr()` for optional attributes.
- ❌ **NO ambiguous returns**: Do not return `None`, `0`, or `1` to indicate failure without a reasonable following action or error propagation.

#### **hasattr() Usage Rules** ⚠️ **CRITICAL**

**Prohibited**: Duck typing with hasattr()
**Goal**: Move from "guessing" object capabilities to explicit contracts (ABC/Protocol).

**🔴 Bad: Loose Duck Typing**
```python
# Ambiguous: Is 'solver' a valid object or just something with a 'solve' method?
if hasattr(solver, "solve"):
    solver.solve()
```

**🟢 Good: Structural Typing (Protocol/ABC)**
```python
# Explicit: We expect a specific interface
if isinstance(solver, BaseSolver):
    solver.solve()
```

**🟢 Good: Optional Configuration (getattr pattern)**
```python
# Clean: Optional config with default fallback
# Replaces: if hasattr(config, 'tol'): tol = config.tol else: tol = 1e-6
tolerance = getattr(config, "tolerance", 1e-6)
```

**🟢 Good: Optional Attributes with Explicit Initialization**
```python
# In __init__:
self._cached_matrix: np.ndarray | None = None

# In usage:
if self._cached_matrix is None:  # ✅ Fast, type-safe, JIT-friendly
    self._cached_matrix = expensive_computation()
```

**🟡 Acceptable: External Library Feature Detection**
```python
# Backend compatibility - PyTorch MPS support (Issue #543 acceptable)
# hasattr required: torch.backends.mps added in PyTorch 1.12+
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
```

**❌ Bad: Runtime Attribute Addition**
```python
# Breaks object shape stability (confuses JIT compilers, static analyzers)
if not hasattr(self, "_grid"):
    self._grid = make_grid()
```

**Key Benefits of Explicit Initialization**:
1. **Object Shape Stability**: All attributes exist from construction (helps Numba/JAX)
2. **Type Safety**: Static analyzers can track `Optional[T]` types
3. **Performance**: Single `is None` check vs `hasattr()` + `getattr()`
4. **Clear Intent**: `None` means "not yet computed", not "maybe doesn't exist"

**For Optional Methods**: Use `callable()` check or Protocol:
```python
# For optional callable attributes
cleaner = getattr(solver, "cleanup", None)
if callable(cleaner):  # ✅ Ensures it's actually a method
    cleaner()

# Better: Use Protocol for type safety
from typing import Protocol

class HasCleanup(Protocol):
    def cleanup(self) -> None: ...

if isinstance(solver, HasCleanup):  # ✅ Type-safe
    solver.cleanup()
```

### **Boundary Condition Coupling Patterns** ⚠️ **IMPORTANT**

When implementing coupled PDE systems (like HJB-FP in MFG), boundary conditions may need to couple between equations.

**Pattern: Adjoint-Consistent Boundary Conditions (Issue #574, #625)**

For reflecting boundaries in MFG systems, the HJB boundary condition couples to the FP density gradient to maintain equilibrium consistency. This is implemented using the **BCValueProvider pattern** with `AdjointConsistentProvider` stored in `BCSegment.value`.

**Architecture Principles**:
1. **Use existing framework**: Leverage Robin BC infrastructure (`BCType.ROBIN` with α=0, β=1)
2. **Provider pattern**: Store intent in BC object, resolve at iteration time
3. **Clean separation**: Solver stays generic; iterator handles coupling via `using_resolved_bc()`
4. **Dimension-agnostic**: Architecture supports 1D, 2D, nD (1D implemented)

**Example** (Provider pattern, v0.18.0+):
```python
from mfg_pde.geometry.boundary import (
    AdjointConsistentProvider, BCSegment, BCType, BoundaryConditions, neumann_bc
)

# Standard Neumann BC (default)
problem = MFGProblem(..., boundary_conditions=neumann_bc(dimension=1))

# Adjoint-consistent BC via provider pattern (Issue #625)
bc = BoundaryConditions(segments=[
    BCSegment(name="left_ac", bc_type=BCType.ROBIN,
              alpha=0.0, beta=1.0,
              value=AdjointConsistentProvider(side="left", diffusion=sigma),
              boundary="x_min"),
    BCSegment(name="right_ac", bc_type=BCType.ROBIN,
              alpha=0.0, beta=1.0,
              value=AdjointConsistentProvider(side="right", diffusion=sigma),
              boundary="x_max"),
], dimension=1)
problem = MFGProblem(..., boundary_conditions=bc)
```

**How it works internally**:
1. `AdjointConsistentProvider` stored as intent in `BCSegment.value`
2. `FixedPointIterator` calls `problem.using_resolved_bc(state)` each iteration
3. Provider computes concrete Robin BC value from current density: `g = -σ²/2 · ∂ln(m)/∂n`
4. Solver receives resolved BC (no MFG coupling knowledge needed)

**When to use adjoint-consistent BC**:
- ✅ Reflecting boundaries with stall point at domain boundary
- ✅ Near equilibrium or high-accuracy requirements
- ✅ Boundary stall configurations (dramatic improvement: >1000x in some cases)
- ❌ Not needed for interior stall points or periodic BC

**Implementation**:
- Provider: `mfg_pde/geometry/boundary/providers.py`
  - `BCValueProvider` protocol, `AdjointConsistentProvider` implementation
- BC coupling utilities: `mfg_pde/geometry/boundary/bc_coupling.py`
  - `create_adjoint_consistent_bc_1d()`: Creates Robin BC from density
- Iterator integration: `mfg_pde/alg/numerical/coupling/fixed_point_iterator.py`
  - Resolves providers via `problem.using_resolved_bc(state)`

**Reference**: See `docs/development/TOWEL_ON_BEACH_1D_PROTOCOL.md` § Boundary Condition Consistency Issue

### **File Path Anchoring** ⚠️ **CRITICAL**
Always anchor output paths to **project root**, never to CWD:
- ✅ `Path(__file__).resolve().parent.parent / "results"` — fixed to project structure
- ✅ `${hydra:runtime.cwd}/results` — fixed to launch location (Hydra/OmegaConf)
- ❌ `Path("results")` or `os.getcwd()` — changes with `cd`, causes recursive nesting

### **Development Documentation** ⚠️ **CRITICAL**
Always document significant changes:

1. **Roadmap**: Update the strategic development roadmap for major features
2. **Changes**: Document architecture changes in `ARCHITECTURAL_CHANGES.md`
3. **Theory**: Create notes for mathematical features with code references

**Naming**: `docs/theory/[feature]_mathematical_formulation.md`, `docs/development/[feature]_implementation_summary.md`

**Status Marking**:
- `✅ COMPLETED` - Finished features
- `[RESOLVED]` / `[CLOSED]` - Fixed issues (prefix filename)
- `[WIP]` - Work in progress
- `[ARCHIVED]` - Obsolete content

### **Documentation Hygiene Checkpoints** ⚠️ **MANDATORY**

Claude Code must proactively check at these triggers:

1. **After Phase Completion**: Run `check_docs_structure.py`, consolidate if >60 active docs, create phase summary, archive detailed docs
2. **Before Creating Directory**: Check if similar exists, verify ≥3 files planned, ask user
3. **On [COMPLETED] Tag**: Immediately suggest moving to archive
4. **Weekly Audit**: Report active docs count, suggest consolidation if needed
5. **Before Commits**: Run `check_docs_structure.py` on docs/ changes

**Key Principle**: Be proactive, not reactive.

### **Logging and Progress Bars**
```python
from mfg_pde.utils.mfg_logging import get_logger, configure_research_logging
from mfg_pde.utils.progress import create_progress_bar, solver_progress

configure_research_logging("session_name", level="INFO")
logger = get_logger(__name__)

# Type-safe progress bar pattern (Issue #587 Protocol)
progress = create_progress_bar(range(max_iterations), verbose=True, desc="Picard")
for i in progress:
    progress.update_metrics(error=error, norm=norm)  # Always works, no hasattr needed
    if converged:
        progress.log("Converged!")  # Always works
        break

# Alternative: High-level solver progress utility
with solver_progress(max_iterations, "Solving MFG") as progress:
    for iteration in range(max_iterations):
        error = step()
        progress.update(1, error=error)
```

**Progress Bar Backend**: Rich only (v0.16.15+). External tqdm eliminated.

**Key Principles**:
- ✅ Use `create_progress_bar()` for type-safe polymorphism (Issue #587)
- ✅ Call `update_metrics()` and `log()` directly - no hasattr checks needed
- ✅ Protocol pattern ensures both verbose=True and verbose=False work identically
- ❌ DO NOT use hasattr checks on progress bars
- 📚 Legacy `tqdm` alias kept for backward compatibility, use `create_progress_bar()` in new code

---

## 🎨 **Visualization & Output**

### **Plotting Preferences**
- **Primary**: Matplotlib with immediate rendering (`plt.show()`)
- **On-demand**: Plotly/Bokeh for interactive features
- **Style**: Publication-ready, consistent notation u(t,x) and m(t,x)

### **Notebook-Based Reporting** ⚠️ **CRITICAL**
Primary method for research outputs, benchmarks, analyses. Use notebooks for:
- Algorithm comparisons, convergence analysis, validation
- Tutorials, research experiments, demonstrations

**Benefits**: Reproducibility, shareability, self-documenting.

### **Output Directory Structure**
```
examples/outputs/         # Gitignored (regenerable)
examples/outputs/reference/  # Tracked (for comparison)
benchmarks/results/       # Gitignored (raw data)
benchmarks/reports/*.html # Tracked (exported reports)
```

### **File Organization Rules**
1. **Python scripts**: Save to `examples/outputs/[category]/`, never root
2. **Notebooks**: Track .ipynb with cleared outputs + exported HTML
3. **Documentation**: Markdown preferred, notebooks for tutorials
4. **Benchmarks**: Use notebooks in `benchmarks/notebooks/`, export HTML to `benchmarks/reports/`

### **Incremental Data Saving** ⚠️ **CRITICAL**
For long-running computations (GFDM solvers, Picard iterations, parameter sweeps):
- ✅ **Save after each iteration**: Don't wait until experiment completes
- ✅ **Use HDF5 with incremental writes**: Append results after each step
- ✅ **Include metadata**: Config, timestamp, partial progress in file attrs
- **Rationale**: Long experiments may crash. Incremental saves preserve progress.

### **Heavy Computational Tasks** ⚠️ **CRITICAL**
- ❌ **Never limit timeout** for long-running solvers (GFDM with high collocation, multi-parameter sweeps)
- ✅ **Run in background**: Use `&` or `run_in_background` for tasks > 5 minutes
- ✅ **Monitor via logs**: Check progress with `tail -f` rather than waiting inline
- ✅ **Trust incremental saves**: If process crashes, partial results are preserved
- **Rationale**: MFG computations can take hours. Arbitrary timeouts lose all progress.

---

## 🧪 **Testing Philosophy**

MFG_PDE uses a **hybrid testing approach** optimized for research code that evolves rapidly.

### **1. Unit Tests (`tests/`) - For Stable APIs**

**When to write unit tests**:
- ✅ Public APIs (`solve_mfg()`, `create_*_solver()`, factory functions)
- ✅ Core infrastructure (config, problem, result, backend)
- ✅ Numerical correctness validation (convergence criteria, boundary conditions)
- ✅ Data structures and utilities used across multiple modules

**Characteristics**:
- Comprehensive test coverage with fixtures and mocks
- Run in CI/CD on every commit
- Should be maintained as API stabilizes
- Located in `tests/unit/` and `tests/integration/`

**Current state**: Comprehensive unit test suite covering stable framework APIs ✅

### **2. Smoke Tests (`if __name__ == "__main__"`) - For Development**

**When to add inline smoke tests**:
- ✅ Algorithm implementations (HJB solvers, FP solvers, coupling methods)
- ✅ Rapidly changing experimental code
- ✅ Modules that benefit from visual verification
- ✅ Utilities and helper functions

**Example pattern**:
```python
# mfg_pde/alg/numerical/hjb_solvers/my_solver.py

class MySolver:
    """Implementation of my solver."""
    def solve(self):
        # Solver logic...
        return result

if __name__ == "__main__":
    """Quick smoke test for development."""
    from mfg_pde import MFGProblem
    import matplotlib.pyplot as plt

    print("Testing MySolver...")

    # Basic convergence test
    problem = MFGProblem()
    solver = MySolver(problem)
    result = solver.solve()

    assert result.converged, "Convergence test failed"
    print(f"✓ Converged in {result.iterations} iterations")

    # Quick visualization (optional)
    plt.semilogy(result.error_history_U)
    plt.title("Convergence History")
    plt.show()

    print("All smoke tests passed! ✓")
```

**Usage**:
```bash
# Quick test during development
python mfg_pde/alg/numerical/hjb_solvers/my_solver.py
```

**Benefits**:
- Fast iteration - no need to update separate test files
- Visual debugging - see algorithm output immediately
- Self-documenting - shows usage example
- Low maintenance - simple asserts, no mock overhead
- Colocated - tests travel with code

### **3. Examples (`examples/`) - For User Documentation**

**Purpose**:
- Demonstrate complete workflows
- Show best practices
- Validate across versions
- Tutorial content

**Not for**: Quick algorithm testing (use smoke tests instead)

### **Decision Matrix**

| Code Type | Changes Frequently? | Public API? | Test Type |
|:----------|:-------------------|:------------|:----------|
| `solve_mfg()` | No | Yes | Unit tests ✅ |
| Config system | No | Yes | Unit tests ✅ |
| New HJB solver | Yes | Maybe | Smoke tests ✅ |
| Experimental RL | Yes | No | Smoke tests ✅ |
| Visualization | Sometimes | Yes | Smoke tests ✅ |
| Utility function | No | Internal | Unit tests OR smoke tests |

### Rationale

Research code evolves quickly. Inline smoke tests:
- ✅ Reduce maintenance burden on rapidly changing algorithms
- ✅ Enable visual verification during development
- ✅ Provide immediate feedback without test suite overhead
- ✅ Delete naturally when code is refactored/removed

Unit tests remain essential for:
- ✅ Public APIs that users depend on
- ✅ Numerical correctness that must not regress
- ✅ Core infrastructure that rarely changes

---

## 🔧 **Development Workflow**

### **Deprecation Policy** ⚠️ **CRITICAL**

**Core Principle**: Deprecated code must immediately redirect to new standard.

When deprecating any API element (parameter, function, class, module):
1. **Immediate redirection**: Old API MUST call new API internally (zero behavior difference)
2. **Equivalence test**: MUST verify old and new give identical results
3. **Complete migration**: Update ALL call sites (direct, factory, defaults, examples, tests)
4. **Timeline**: 3 minor versions OR 6 months before removal

**Reference**: `docs/development/DEPRECATION_LIFECYCLE_POLICY.md`

**Lesson from Issue #616** (`conservative=` parameter):
- Deprecated with warning but wrong default → 1 month of catastrophic bugs (99.4% mass error)
- Factory not updated → production code got broken behavior
- No equivalence test → bug not caught until user validation

**Enforcement**:
- ✅ Equivalence test is mandatory
- ✅ Pre-commit hook blocks deprecated usage in production code
- ✅ CI verifies all call sites updated

### **Branch Naming** ⚠️ **MANDATORY**
Always use `<type>/<short-description>` format:

**Types**: `feature/`, `fix/`, `chore/`, `docs/`, `refactor/`, `test/`

**Examples**: `feature/semi-lagrangian-solver`, `fix/convergence-criteria-bug`

### **Hierarchical Branch Structure** ⚠️ **CRITICAL**

**Principle**: Organize work through hierarchical structures reflecting logical dependencies.

**Core Rules**:
1. ❌ Never commit directly to `main`
2. ✅ Create feature branches for all work
3. ✅ Use parent branches for related work
4. **Merge order**: Child → Parent → Main (always respect this order)

**When to use hierarchy**:
- Multi-step refactoring (each step = child)
- Feature development (sub-features = children)
- Systematic cleanup (categories = children)
- Related changes (logical grouping)

**Branch lifecycle**:
- Create child when starting sub-feature
- Merge to parent when logical unit complete
- Keep child alive if more related work remains
- Delete child only when entire sub-feature done

**Example**:
```bash
# Create parent and child
git checkout -b chore/code-quality-cleanup
git checkout -b chore/fix-unused-variables
# Work, commit, push
git checkout chore/code-quality-cleanup
git merge chore/fix-unused-variables --no-ff
# Keep child alive for more work OR delete if complete
```

### Branch Management ⚠️ **ADAPTIVE**

**Branch Count Guidelines**:
- **< 5**: Healthy
- **5-10**: Warning - cleanup immediately
- **> 10**: Critical - stop creating until cleanup

**Decision Framework**: "Should I create new branch or reuse existing?"
- **Reuse** if: Related to existing PR, part of feature in progress, fixing bugs in same area
- **Create new** if: Independent work, >1 week effort, experimental, parallel development needed

**Daily Habits**:
- Rebase long-lived branches onto main daily
- Create PR immediately when pushing
- Delete merged branches immediately
- Merge early, merge often

**Weekly Audit**:
```bash
# Check branch health
git branch -r | wc -l  # Count active branches
gh pr list --state open  # Review open PRs
# Clean up merged branches
gh pr list --state merged --limit 20 --json headRefName --jq '.[].headRefName' | \
  xargs -I {} git push origin --delete {}
```

### **GitHub Issue and PR Management** ⚠️ **MANDATORY**

**Required Labels (all 4)**:
1. `priority:` high/medium/low
2. `area:` algorithms/config/core/documentation/geometry/performance/testing/visualization
3. `size:` small/medium/large
4. `type:` bug/enhancement/chore/refactor/infrastructure/research/type-checking/question

**Workflow**:
```bash
# Check issue labels
gh issue view [issue_number]
# Apply labels before work
gh issue edit [issue_number] --add-label "priority: medium,area: algorithms,size: small,type: enhancement"
# Create properly named branch
git checkout -b feature/descriptive-name
# Create PR with inherited labels
gh pr create --title "Title" --body "Fixes #[issue_number]" \
  --label "priority: medium,area: algorithms,size: small,type: enhancement,status: in-review"
```

### **Feature Development Process**
1. Create/verify issue with proper labels
2. Create branch: `git checkout -b <type>/descriptive-name`
3. Add core code to `mfg_pde/` subdirectory
4. Create examples in `examples/basic/` or `examples/advanced/`
5. Add tests to `tests/unit/` or `tests/integration/`
6. Update docs in `docs/` category
7. Add benchmarks if applicable
8. Label PR matching issue

### **Code Quality Expectations**
- Follow `docs/development/CONSISTENCY_GUIDE.md`
- Use type hints and comprehensive docstrings
- Include error handling and validation
- Support interactive and non-interactive usage

### **Ruff Version Management** ⚠️ **IMPORTANT**
**Strategy**: Pin with periodic automated updates

**Why pin**: Consistent formatting, reproducible, no surprise CI failures, controlled review

**Update**: Monthly automated check via GitHub Action, manual via `python scripts/update_ruff_version.py`

### **Defensive Programming** ⚠️ **CRITICAL**
Test before extend:
- ✅ Verify existing functionality works before adding features
- ✅ Run tests after changes
- ✅ Check examples execute successfully
- ❌ Don't build on untested foundations

**Pre-commit checklist**:
```bash
pytest tests/unit/test_affected_module.py
mypy mfg_pde/affected_module.py
ruff check mfg_pde/affected_module.py
python examples/basic/relevant_example.py
git add ... && git commit -m "..."
```

### **Smart .gitignore** ⚠️ **CRITICAL**
Use targeted patterns preserving valuable code:
```bash
# ✅ GOOD: Root-level only
/*.png
/*_analysis.py

# ❌ BAD: Too broad
*.png
*_analysis.py

# Always preserve
!examples/**/*.py
!tests/**/*.py
!docs/**/*.md
```

---

## 📊 **Package Management**

**Dependencies**:
- **Core**: numpy, scipy, matplotlib
- **Interactive**: plotly, jupyter, nbformat (with fallbacks)
- **Progress**: rich
- **Optional**: psutil

**Installation**: Support both development (`pip install -e .`) and user installation.

---

## 🎯 **Working Preferences**

### **GitHub Label System** ⚠️ **IMPORTANT**

**Hierarchical Taxonomy** (formalized 2026-02-05):

| Prefix | Purpose | Labels |
|:-------|:--------|:-------|
| **`area:`** | Functional domain | `algorithms`, `config`, `core`, `documentation`, `geometry`, `performance`, `testing`, `visualization` |
| **`type:`** | Work nature | `bug`, `enhancement`, `chore`, `refactor`, `infrastructure`, `research`, `type-checking`, `question` |
| **`priority:`** | Urgency | `high`, `medium`, `low` |
| **`size:`** | Effort estimate | `small` (hours-1d), `medium` (1-3d), `large` (1+wk) |
| **`status:`** | Workflow state | `blocked`, `in-review`, `needs-testing` |
| **`resolution:`** | Completion type | `merged`, `superseded`, `wontfix`, `duplicate`, `invalid` |

**Required on every issue** (4 dimensions): `area:`, `type:`, `priority:`, `size:`

**Non-taxonomic labels** (kept for GitHub conventions): `good first issue`, `help wanted`, `automated`

**Color Coding**:
- Red/Orange: `priority:` labels
- Blue: `area:` labels (technical domains)
- Purple: `type:` labels
- Green: `resolution:` labels, `status: in-review`
- Yellow: `status: needs-testing`, `priority: medium`
- Gray: `type: chore`, `automated`

**Rules**:
- No bare labels — all issue-classification labels must use a prefix
- One label per `priority:` and `size:` dimension (no duplicates)
- Multiple `area:` labels allowed for cross-cutting issues
- **Scaling**: Subdivide areas before creating new label types (e.g., `area: algorithms-hjb`)

### **Task Management**
- Use TodoWrite tool for multi-step tasks
- Track progress and mark completion immediately
- Apply consistent GitHub labels

### **Progressive Logging** ⚠️ **CRITICAL**

**Principle**: Log incrementally, summarize at milestones only.

**During Work**:
- ✅ Log steps in simple files, use TodoWrite, write technical notes
- ❌ Don't create frequent summaries

**Bug Reporting**:
- ✅ Bugs belong to GitHub issues (use `gh issue create`)
- ❌ Don't create markdown files in docs/bugs/ directory

**Summary Creation**:
- ✅ Only create summaries after important phases
- ✅ Always ask user first if summary is needed
- ❌ Don't create markdown summaries after each operation

**Create Summaries at**:
- Phase completion, critical milestones, before/after breaks, investigation conclusions

### **Directory Management** ⚠️ **CRITICAL**

**Before creating new directories**:
1. Analyze existing content first
2. Create detailed migration plan
3. Identify and merge duplicates
4. Update cross-references
5. Move systematically

**Guidelines**:
- Minimum 5+ files per directory
- Clear, distinct functional purpose
- User-centric organization
- Max 3 levels depth
- Test before finalizing

### **Repository Management**
- **Preserve**: Examples, tests, research-grade code
- **Target cleanup**: Use root-level patterns (`/*.py`) not global (`*.py`)
- **Archive vs Delete**: Remove clutter, archive valuable history only
- **Clean root**: Keep root clean, preserve organized subdirectories

---

## 🤖 **AI Interaction Design**

For advanced MFG research, follow `docs/development/AI_INTERACTION_DESIGN.md`:

**Research Context**:
- Graduate/research level mathematical rigor
- High-performance implementation with complexity analysis
- Journal-quality exposition
- Integration of mathematical traditions

**Prompt Templates**: Theoretical investigation, computational implementation, cross-field analysis

**Quality Standards**: Mathematical rigor, computational excellence, academic communication

---

## 📜 **Solo Maintainer's Protocol**

Self-governance for disciplined changes:

1. **Propose in Issue**: Document reasoning in GitHub issue
2. **Implement in PR**: Work in feature branch, submit PR
3. **AI-Assisted Review**: Self-review with AI against standards
4. **Verify Issue Completion**: Before closing an issue or creating PR, verify EVERY point mentioned in the issue has been solved/discussed/treated
5. **Merge on Pass**: Merge only after all checks pass

Enforce with GitHub branch protection rules on `main`.

### **Issue Completion Verification** ⚠️ **CRITICAL**

Before marking an issue as complete or creating a PR:

1. **Read the original issue** - Don't rely on memory or commit messages
2. **Check every acceptance criterion** - All checkboxes must be addressed
3. **Verify all discussion points** - Each question answered or documented
4. **Confirm scope completeness** - All subtasks, not just main objective
5. **Document any deviations** - If scope changed, update issue before closing

**Anti-pattern**: Closing issues based on commit messages without re-reading the original issue requirements.

---

## 🔍 **Quality Assurance**

### **Before Completing Tasks**
- Verify imports work
- Check file placement
- Update documentation
- Test examples execute
- Maintain consistency

### **Consistency Checks**
- Mathematical notation: u(t,x), m(t,x)
- Import patterns: Use factory methods
- Documentation: Professional, research-grade
- Organization: Follow established structure

### **Documentation Status Tracking** ⚠️ **MANDATORY**

**Status Categories**: `[COMPLETED]`, `[RESOLVED]`, `[ANALYSIS]`, `[EVALUATION]`, `[SUPERSEDED]`

**Cleanup Principles**:
- Mark completed work
- Use descriptive status indicators
- Remove obsolete content
- Consolidate overlaps
- Ensure docs reflect current state

---

**Last Updated**: 2026-02-06
**Repository Version**: v0.17.10 (Pre-1.0.0)
**Claude Code**: Always reference this file for MFG_PDE conventions
