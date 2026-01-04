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
- ❌ **NO `hasattr()`**: Use explicit interfaces or handle `AttributeError` instead of checking for attribute existence.
- ❌ **NO ambiguous returns**: Do not return `None`, `0`, or `1` to indicate failure without a reasonable following action or error propagation.

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
from mfg_pde.utils.progress import tqdm  # Uses rich (preferred) with tqdm fallback

configure_research_logging("session_name", level="INFO")
logger = get_logger(__name__)

for iteration in tqdm(range(max_iterations), desc="Solving MFG"):
    # solver logic
```

**Progress Bar Backend**: Rich only. No fallback - if rich is not installed, ImportError is raised.

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
1. `priority: high/medium/low`
2. `area: algorithms/geometry/performance/config/visualization`
3. `size: small/medium/large`
4. `type: bug/enhancement/documentation/type-checking`

**Workflow**:
```bash
# Check issue labels
gh issue view [issue_number]
# Apply labels before work
gh issue edit [issue_number] --add-label "priority: medium,area: algorithms,size: small,enhancement"
# Create properly named branch
git checkout -b feature/descriptive-name
# Create PR with inherited labels
gh pr create --title "Title" --body "Fixes #[issue_number]" \
  --label "priority: medium,area: algorithms,size: small,enhancement,status: in-review"
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

**Hierarchical Taxonomy**:
- **`area:`** - Functional domains (algorithms, config, documentation, geometry, performance)
- **`type:`** - Work nature (infrastructure, enhancement, bug, research)
- **`priority:`** - Urgency (high, medium, low)
- **`size:`** - Effort (small, medium, large)
- **`status:`** - Workflow state (blocked, in-review, needs-testing)
- **`resolution:`** - Completion type (merged, superseded, wontfix)

**Color Coding**: Red (priority), Blue (technical), Green (completed), Purple (special), Yellow (attention)

**Scaling**: Subdivide areas before creating new label types (e.g., `area: algorithms-hjb`)

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
4. **Merge on Pass**: Merge only after all checks pass

Enforce with GitHub branch protection rules on `main`.

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

**Last Updated**: 2026-01-05
**Repository Version**: v0.16.14 (Pre-1.0.0)
**Claude Code**: Always reference this file for MFG_PDE conventions
