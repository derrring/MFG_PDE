# Claude Code Preferences for MFG_PDE

This file contains preferences and conventions for Claude Code when working with the MFG_PDE repository.

## ⚠️ **Communication Principles**

**Objectivity and Honesty**:
- ❌ **NO flattery, NO superlatives, NO praise**
- ✅ **Factual, technical, honest assessment** of code and results
- ✅ **Point out issues, limitations, and areas needing improvement**
- ✅ **State observations clearly**: "This works", "This fails", "This needs review"
- ✅ **Acknowledge uncertainty**: "This may have issues", "Needs validation"

**Examples**:
```
❌ BAD: "Excellent work! This implementation is amazing!"
✅ GOOD: "Implementation complete. Validation needed."

❌ BAD: "This is perfect! Great job on the solver!"
✅ GOOD: "Solver passes basic tests. Needs stress testing with larger problems."

❌ BAD: "Wonderful! You've created an outstanding solution!"
✅ GOOD: "Solution converges on test cases. Performance overhead is 15% higher than baseline."
```

**Rationale**: Research and production code require objective critique, not encouragement. False positivity obscures real issues and wastes time.

---

## 🎯 **Repository Mission & Scope** ⚠️ **CRITICAL**

### **MFG_PDE: Public Infrastructure Package**

**Purpose**: Provide comprehensive, production-ready infrastructure for Mean Field Games research and applications.

**Scope - What Belongs in MFG_PDE**:

✅ **Core Infrastructure**:
- Solver frameworks (HJB, FP, coupled systems)
- Backend abstraction layer (NumPy, PyTorch, JAX)
- Configuration and factory systems
- Geometry and boundary condition handling
- Workflow and experiment management tools
- Visualization and analysis utilities

✅ **Classical Algorithms** (well-established methods):
- **Numerical**: Finite difference, finite element, GFDM, particle methods
- **Deep Learning**: DGM, PINN, DeepONet (established architectures)
- **Reinforcement Learning**: Actor-Critic, DQN, PPO (standard algorithms)
- **Optimization**: Variational methods, gradient descent, ADMM

✅ **Standard Examples**:
- Classical MFG problems (LQ, crowd motion, traffic flow)
- Tutorial implementations
- Benchmark problems
- Educational demonstrations

### **MFG-Research: Private Research Repository**

**Purpose**: Novel research, experimental algorithms, and unpublished methods.

**Scope - What Belongs in MFG-Research**:

🔬 **Research Algorithms**:
- Novel numerical schemes under development
- Experimental neural architectures
- Unpublished optimization techniques
- Research-grade implementations

🔬 **Research Applications**:
- Specialized problem domains
- Case studies and publications
- Experimental validations
- Parameter studies and analysis

**Key Principle**: MFG-Research **imports** MFG_PDE but **never modifies** it. Research repo extends infrastructure through composition, not modification.

### **Boundary Guidelines** ⚠️ **WHEN TO USE WHICH REPO**

| Criterion | MFG_PDE (Public) | MFG-Research (Private) |
|:----------|:-----------------|:-----------------------|
| **Maturity** | Production-ready, tested | Experimental, in development |
| **Publication** | Published methods | Unpublished, under review |
| **Stability** | Stable API, versioned | Rapid iteration, breaking changes OK |
| **Documentation** | Comprehensive | Minimal, research notes |
| **Examples** | Tutorial-grade | Research-grade |
| **Testing** | Full test coverage | Exploratory testing |
| **Audience** | Public users, researchers | Personal research only |

### **Development Workflow**

**Working in MFG_PDE**:
```python
# ✅ GOOD: Adding well-established method
from mfg_pde.alg.numerical import FiniteDifferenceSolver
class SpectralMethodSolver(FiniteDifferenceSolver):
    """Well-documented spectral method (published 1990)."""
    pass
```

**Working in MFG-Research**:
```python
# ✅ GOOD: Experimental method using MFG_PDE infrastructure
from mfg_pde.core import MFGProblem
from mfg_pde.solvers import create_solver

class NovelAdaptiveScheme:
    """Experimental adaptive scheme (under development)."""
    def __init__(self, problem: MFGProblem):
        self.problem = problem
        self.base_solver = create_solver(problem, method="gfdm")

    def solve(self):
        # Novel algorithm building on MFG_PDE infrastructure
        pass
```

### **Migration Path: Research → Infrastructure**

When research matures, migrate to MFG_PDE:

1. **Maturity Check**:
   - Algorithm is proven and validated
   - Method is published or well-documented
   - Implementation is stable and tested

2. **Preparation**:
   - Add comprehensive tests
   - Write tutorial documentation
   - Create usage examples
   - Ensure API consistency

3. **Integration**:
   - Open PR in MFG_PDE
   - Follow infrastructure code standards
   - Maintain backward compatibility

4. **Deprecation** in MFG-Research:
   - Update to use MFG_PDE version
   - Keep research notes for historical record

### **AI Assistant Guidelines for Repository Scope**

When working with AI assistance:

**Always ask**: "Is this for infrastructure or research?"

**For infrastructure (MFG_PDE)**:
- Requires comprehensive documentation
- Must have tests and examples
- Should follow existing patterns
- Needs stability guarantees

**For research (MFG-Research)**:
- Rapid prototyping OK
- Minimal documentation acceptable
- Breaking changes allowed
- Experimental patterns encouraged

**Example AI Prompts**:
> "I'm implementing a novel neural architecture. This should go in MFG-Research, not MFG_PDE."

> "This is a standard PINN implementation following DeepXDE. This belongs in MFG_PDE's neural algorithms."

## 🏗️ **Repository Structure Conventions**

### **Directory Organization**

**Top-Level Structure** (outside package):
- **`mfg_pde/`** - Core package code (see package structure below)
- **`tests/`** - Pure unit and integration tests (no demos)
- **`benchmarks/`** - Performance benchmark **test scripts** (peer to `tests/`)
- **`examples/`** - All demonstration code, organized by complexity:
  - `basic/` - Simple single-concept examples
  - `advanced/` - Complex multi-feature demonstrations
  - `notebooks/` - Jupyter notebook examples
  - `tutorials/` - Step-by-step learning materials
- **`docs/`** - Documentation organized by category (see docs/README.md)
- **`archive/`** - Historical code (do not modify)

**Package Structure** (`mfg_pde/` internals):
- **`mfg_pde/alg/`** - Algorithms (numerical, neural, reinforcement, optimization)
- **`mfg_pde/backends/`** - Computational backends (NumPy, PyTorch, JAX)
  - `strategies/` - Backend-aware strategy selection (CPU/GPU/hybrid)
- **`mfg_pde/benchmarks/`** - Benchmarking **tools** (library code, NOT test scripts)
  - ⚠️ Note: Different from top-level `/benchmarks/` (test scripts)
- **`mfg_pde/config/`** - Configuration management
- **`mfg_pde/core/`** - Core MFG problem definitions
- **`mfg_pde/factory/`** - Factory functions for common patterns
- **`mfg_pde/geometry/`** - Domain and boundary condition definitions
- **`mfg_pde/hooks/`** - Hook system for solver customization
- **`mfg_pde/solvers/`** - High-level solver interfaces
- **`mfg_pde/types/`** - Type definitions (arrays, protocols, solver types)
- **`mfg_pde/utils/`** - Utility functions (logging, numerical, performance)
- **`mfg_pde/visualization/`** - Plotting and visualization tools
- **`mfg_pde/workflow/`** - Experiment management and parameter sweeps
- **`mfg_pde/compat/`** - Backward compatibility layer
- **`mfg_pde/meta/`** - Metaprogramming utilities

### **File Placement Rules**
- **Examples**: Always categorize by complexity (basic/advanced/notebooks)
- **Tests**: Only actual unit/integration tests, no demos
- **Benchmarks**: Performance benchmark **test scripts** (top-level, peer to `tests/`)
  - ❌ NOT `examples/benchmarks/` (benchmarks are tests, not examples)
  - ✅ `benchmarks/` (peer to `tests/`)
  - Outputs go to `examples/outputs/benchmarks/` (gitignored)
- **Documentation**: Follow docs/ category structure

## 📝 **Coding Standards**

### **Type Checking Philosophy** ⚠️ **IMPORTANT**
MFG_PDE follows a **pragmatic approach** to type checking optimized for research environments:

**✅ Strict typing for:**
- Public APIs and factory functions
- Core mathematical algorithms
- Configuration systems

**⚠️ Relaxed typing for:**
- Utility functions (logging, plotting)
- Research examples and demos
- Optional dependencies

**IDE Configuration:**
The repository includes `.vscode/settings.json` with balanced type checking that reduces noise while catching real errors.

### **Modern Python Typing: Division of Responsibility** ⚠️ **CRITICAL**

MFG_PDE follows modern Python typing best practices based on clear **division of responsibility** between library authors and users:

#### **The Library Author's Responsibility**
A mature library author strives to make type hints as precise as possible using **`@overload`** as the primary tool:

```python
# Library authors provide precise signatures
from typing import overload

@overload
def solve_mfg(problem: ContinuousMFG, backend: str = "numpy") -> ContinuousSolution: ...
@overload
def solve_mfg(problem: NetworkMFG, backend: str = "igraph") -> NetworkSolution: ...

def solve_mfg(problem, backend="auto"):
    # Implementation handles multiple types
    pass
```

#### **The Library User's Toolkit**
Users rely on the author's annotations, but have tools for imperfect situations:

**1. Runtime Type Checking with `isinstance()`** - For functions designed to return multiple types:
```python
# Safe logic branching
result = solve_mfg(problem)
if isinstance(result, ContinuousSolution):
    return result.plot_density()
elif isinstance(result, NetworkSolution):
    return result.plot_network()
```

**2. Type Casting with `cast()`** - Last resort for imprecise library hints:
```python
from typing import cast
from scipy.sparse import csr_matrix

# When you know more than the type checker
matrix = cast(csr_matrix, scipy_function_with_generic_return())
```

#### **Typing Tools Responsibility Matrix**

| Tool | Primary User | Purpose | When to Use |
|:-----|:-------------|:--------|:------------|
| **`@overload`** | Library Author | Provide precise type signatures | Multiple return types for same function |
| **`isinstance()`** | Both | Runtime type checking | Safe logic branching, type guards |
| **`cast()`** | Library User | Override imprecise type hints | Library hints too generic for your use case |

#### **MFG_PDE Implementation Examples**

**As Library Authors** (when extending MFG_PDE):
```python
# Use @overload for solver factory functions
@overload
def create_solver(config: HJBConfig) -> HJBSolver: ...
@overload
def create_solver(config: FPConfig) -> FPSolver: ...

# Use isinstance() for type guards
def validate_backend(backend: Any) -> BackendProtocol:
    if isinstance(backend, str):
        return get_backend_by_name(backend)
    elif isinstance(backend, BackendProtocol):
        return backend
    else:
        raise TypeError(f"Invalid backend type: {type(backend)}")
```

**As Library Users** (when using MFG_PDE):
```python
# Use isinstance() for safe type branching
geometry = create_geometry(config)
if isinstance(geometry, Domain2D):
    geometry.visualize_mesh()
elif isinstance(geometry, NetworkGeometry):
    geometry.plot_graph()

# Use cast() sparingly for external library issues
from scipy.sparse import csr_matrix
adj_matrix = cast(csr_matrix, networkx.adjacency_matrix(graph))
```

**The Goal**: In an ideal, well-typed library, you see `@overload` in source code, `isinstance()` in application logic, and `cast()` rarely.

### **Import Style**
```python
# Preferred imports for MFG_PDE
from mfg_pde import ExampleMFGProblem, BoundaryConditions
from mfg_pde.factory import create_fast_solver
from mfg_pde.config import DataclassHJBConfig, DataclassFPConfig, ExperimentConfig
from mfg_pde.utils.logging import get_logger, configure_research_logging
```

### **Documentation Style**
- Use comprehensive docstrings with mathematical notation
- Include LaTeX expressions: `$u(t,x)$` and `$m(t,x)$`
- Reference line numbers when discussing code: `file_path:line_number`
- Follow consistency guide in `docs/development/CONSISTENCY_GUIDE.md`

### **Mathematical Typesetting**

**Preference: LaTeX in documentation, flexible in code** (Updated 2025-10-28)

- **In theory documents and .md files**: Full LaTeX math using `$...$` delimiters
  - Example: `$\nabla u$`, `$\frac{\partial u}{\partial t}$`, `$\sigma^2$`
  - Rationale: Better rendering, copy-paste to papers, consistency with publications

- **In docstrings**: UTF-8 math symbols are **allowed** (per line 365-370)
  - Example: `"""Solve HJB: ∂u/∂t + H(∇u) = 0"""`
  - Also acceptable: LaTeX style `$\frac{\partial u}{\partial t}$` or ASCII `-du/dt`
  - Flexible approach based on readability

- **In code output and logging**: ASCII only (for terminal compatibility)
  - Example: `print(f"Residual: ||R(u)|| = {residual:.6e}")`
  - Example: `logger.info("Convergence: ||u - u_prev|| < tol")`

### **Text and Symbol Standards** ⚠️ **CRITICAL**
**NO emojis in program scripts (Python files):**
```python
# ✅ GOOD - Plain ASCII comments in code
# Solve the HJB equation using Newton iteration
# WARNING: This method may not converge for large sigma

# ❌ BAD - No emojis in Python files  
# 🚀 Solve the HJB equation using Newton iteration
# ⚠️ WARNING: This method may not converge for large sigma
```

**Mathematical symbols and output formatting:**
```python
# ✅ GOOD - LaTeX rendering or ASCII for output
print("Convergence achieved: ||u - u_prev|| < 1e-6")
print(f"Mass conservation error: {error:.6f}")
logger.info("Newton method converged in %d iterations", num_iter)

# ✅ GOOD - UTF-8 in docstrings is allowed
def solve_hjb(self) -> np.ndarray:
    """
    Solve HJB equation: ∂u/∂t + H(∇u) = 0
    Uses finite difference method with Newton iteration.
    """

# ❌ BAD - Unicode math symbols in output
print("Convergence: ‖u - u_prev‖ < 1e-6")  
print(f"Error: Δmass = {error:.6f}")
```

**Emoji usage policy:**
- ✅ **Allowed**: Markdown documentation files (.md)
- ✅ **Allowed**: Comments in CLAUDE.md and README files
- ❌ **Forbidden**: Python scripts (.py files), even in comments
- ❌ **Forbidden**: Program output and logging messages
- ❌ **Forbidden**: Docstrings (use UTF-8 math symbols instead)

### **Development Documentation Standards** ⚠️ **CRITICAL**
**Always document significant changes and theoretical foundations:**

1. **Roadmap Documentation** (`docs/development/`):
   - Update `STRATEGIC_DEVELOPMENT_ROADMAP_2026.md` for major feature additions
   - Create milestone-specific roadmap files for complex features
   - Track implementation status and dependencies

2. **Change Documentation** (`docs/development/`):
   - Document architectural changes in `ARCHITECTURAL_CHANGES.md`
   - Create feature-specific implementation summaries
   - Record decisions and rationale for future reference

3. **Theoretical Documentation** (`docs/theory/`):
   - Create theoretical notes for each major mathematical feature
   - Include mathematical formulations, references, and implementation notes
   - Link theory to code with specific file/line references

**Documentation Naming Conventions**:
```
docs/theory/[feature]_mathematical_formulation.md
docs/development/[feature]_implementation_summary.md
docs/development/ROADMAP_[milestone]_[date].md
```

**Status Marking in Documentation** ⚠️ **CRITICAL**:
- **Completed features**: Mark with `✅ COMPLETED` in title and content
- **Closed issues**: Add `[RESOLVED]` or `[CLOSED]` prefix to filename
- **Active development**: Use `[WIP]` (Work In Progress) prefix
- **Archived content**: Add `[ARCHIVED]` prefix and move to appropriate archive location
- **Example naming**:
  ```
  ✅ GOOD: "Network MFG Implementation ✅ COMPLETED"
  ✅ GOOD: "[RESOLVED]_particle_collocation_analysis.md"
  ✅ GOOD: "[WIP]_master_equation_implementation.md"
  ❌ BAD: "network_implementation.md" (no status indication)
  ```

### **Documentation Hygiene Checkpoints** ⚠️ **MANDATORY**

Claude Code must proactively check documentation health at these trigger points:

**Checkpoint 1: After Phase Completion**
```bash
# When user says "Phase X is complete" or similar milestone:
1. Run: python scripts/check_docs_structure.py --report
2. If > 60 active docs: Propose consolidation plan immediately
3. Create phase summary: docs/development/completed/PHASE_X_SUMMARY.md
4. Archive detailed docs: git mv phase_x_*.md archive/development/phases/
5. Verify: All [COMPLETED] files moved to archive/
```

**Checkpoint 2: Before Creating New Directory**
```bash
# Before: mkdir docs/new_category/
1. Check: Does similar category already exist?
2. Verify: Will this have ≥ 3 files initially?
3. Ask user: "Create docs/new_category/ or add to docs/existing_category/?"
4. Only proceed if justified (≥ 3 files planned)
```

**Checkpoint 3: On [COMPLETED] Tag Addition**
```bash
# When marking document as [COMPLETED], [RESOLVED], [CLOSED]:
1. Immediately suggest: "Move this to archive/appropriate_location/?"
2. Update cross-references if moved
3. Verify no broken links remain
```

**Checkpoint 4: Weekly Documentation Audit (Proactive)**
```bash
# Every 7 days of active development:
"📊 Weekly Documentation Status:
- Active docs: X/70 files
- Completed but not archived: Y files
- Sparse directories: Z found
- Action needed: [Yes/No]
Shall I consolidate now or continue working?"
```

**Checkpoint 5: Before Major Commits**
```bash
# Before committing documentation changes:
python scripts/check_docs_structure.py
# If fails: Fix issues before committing
```

**Automated Enforcement:**
- Pre-commit hook: Runs `check_docs_structure.py` on docs/ changes
- Monthly audit: GitHub Action creates issue if structure degrades
- See: `docs/DOCUMENTATION_POLICY.md` for complete standards

**Key Principle**: Be **proactive**, not reactive. Don't wait for user to ask about documentation cleanup.

### **Logging Preferences**
```python
# Always use structured logging
from mfg_pde.utils.logging import get_logger, configure_research_logging

# Configure for research sessions
configure_research_logging("session_name", level="INFO")
logger = get_logger(__name__)

# Use enhanced logging functions
from mfg_pde.utils.logging import log_solver_configuration, log_convergence_analysis
```

### **Progress Bar Standards**
```python
# Use tqdm for progress tracking in solvers and long-running operations
from tqdm import tqdm

# Preferred usage patterns:
# 1. Solver iterations
for iteration in tqdm(range(max_iterations), desc="Solving MFG"):
    # solver logic
    pass

# 2. Batch processing
for problem in tqdm(problems, desc="Processing batch"):
    # processing logic
    pass

# 3. With custom formatting
pbar = tqdm(total=total_steps, desc="Convergence", 
           bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total}')
```

## 🎨 **Visualization Standards**

### **Plotting Preferences**
- **Primary**: Matplotlib with immediate rendering using default backend
  - Simplest approach for quick visualization during development
  - Use `plt.show()` to display results immediately
  - Static graphs are sufficient for most research tasks
- **On-demand**: Plotly/Bokeh for professional exhibition
  - Use only when interactive features are specifically needed
  - Export to HTML for presentations and publications
- **Mathematical notation**: Consistent u(t,x) and m(t,x) conventions
- **Professional styling**: Use clean, publication-ready defaults

### **Notebook Creation**
- Use `mfg_pde.utils.notebook_reporting` for research notebooks
- Include mathematical formulations with LaTeX
- Provide working examples with error handling
- Export to both .ipynb and .html formats

## 📁 **Output Organization Strategy** ⚠️ **CRITICAL**

MFG_PDE uses **notebook-based reporting** as the primary method for generating and presenting results. This approach integrates code, mathematical formulae, visualizations, and narrative in a single reproducible document.

### **Core Principle: Notebooks for Reports**
**All research outputs, benchmarks, and complex analyses should be generated via Jupyter notebooks**, which can:
- Combine LaTeX math, code, and visualizations
- Be exported to HTML for sharing (tracked in git)
- Be re-executed for reproducibility
- Serve as both analysis tool and final report

### **Output Directory Structure**

```
examples/
├── basic/              # Simple demonstrations
│   └── lq_mfg_demo.py  # Saves to examples/outputs/
├── advanced/           # Complex demonstrations
│   └── actor_critic_maze_demo.py
├── notebooks/          # Jupyter notebooks (primary reporting method)
│   ├── benchmarks/     # Performance comparisons (notebook + exported HTML)
│   ├── tutorials/      # Educational notebooks
│   └── analysis/       # Research analysis notebooks
└── outputs/            # Generated outputs (gitignored by default)
    ├── basic/          # From basic examples
    ├── advanced/       # From advanced examples
    └── reference/      # Reference outputs (tracked for comparison)

benchmarks/
├── scripts/            # Benchmark runner scripts
├── notebooks/          # Benchmark analysis notebooks (tracked)
├── results/            # Raw data (gitignored)
└── reports/            # Exported HTML reports (tracked)

docs/
├── theory/             # Mathematical formulations (markdown, rarely images)
├── development/        # Development notes (markdown)
└── tutorials/          # Tutorial notebooks (tracked)
```

### **File Organization Rules**

**1. Python Scripts (examples/*.py)**
- Save outputs to `examples/outputs/[category]/`
- Never save to root directory
- Use relative paths from script location

```python
# In examples/basic/lq_mfg_demo.py
from pathlib import Path

EXAMPLE_DIR = Path(__file__).parent
OUTPUT_DIR = EXAMPLE_DIR.parent / "outputs" / "basic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

save_path = OUTPUT_DIR / "lq_mfg_demo.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
```

**2. Jupyter Notebooks (examples/notebooks/, benchmarks/notebooks/)**
- **Tracked in git**: Notebook files (.ipynb) with cleared outputs
- **Tracked**: Exported HTML reports (self-contained)
- **Gitignored**: Intermediate data files and raw outputs

```python
# In notebook cell - save to same directory as notebook
from pathlib import Path

NOTEBOOK_DIR = Path.cwd()  # Current notebook directory
output_path = NOTEBOOK_DIR / "benchmark_comparison.html"

# Export notebook to HTML
!jupyter nbconvert --to html --execute current_notebook.ipynb
```

**3. Documentation (docs/)**
- **Markdown preferred**: Documentation rarely contains images
- **Notebooks for tutorials**: Use `docs/tutorials/` for interactive docs
- **No generated images**: Theory documentation uses LaTeX math, not plots

**4. Benchmarks**
- **Notebooks primary**: Use `benchmarks/notebooks/` for analysis
- **Export HTML**: Track exported reports in `benchmarks/reports/`
- **Raw data gitignored**: Large result files in `benchmarks/results/`

**5. Research Artifacts (User Work)**
- **User creates**: `research/` directory for personal work (gitignored)
- **Selective tracking**: User can track finalized notebooks/reports

### **Notebook-Based Workflow Extension**

**Use notebooks for any suitable case**, including:
- Algorithm comparisons and benchmarking
- Convergence analysis and parameter studies
- Method validation and verification
- Educational tutorials and examples
- Research experiments and exploration
- Package demonstration and showcase

**Benefits**:
- **Reproducibility**: Code and results in one document
- **Shareability**: Export to HTML for distribution
- **Documentation**: Self-documenting with narrative + code + output
- **Collaboration**: Easier review than separate scripts + outputs

### **`.gitignore` Configuration**

```bash
# Generated outputs (regenerable)
examples/outputs/
benchmarks/results/
tests/outputs/

# Track reference outputs and reports
!examples/outputs/reference/
!benchmarks/reports/*.html

# Track notebooks with cleared outputs
*.ipynb

# Gitignore notebook checkpoints
.ipynb_checkpoints/

# User research directory (optional)
/research/
```

### **When to Use What**

| Use Case | Tool | Output Location | Git Tracking |
|:---------|:-----|:----------------|:-------------|
| Simple demo | Python script | `examples/outputs/` | No (gitignored) |
| Tutorial | Jupyter notebook | `examples/notebooks/` | Yes (.ipynb + .html) |
| Benchmark | Jupyter notebook | `benchmarks/notebooks/` | Yes (.ipynb + .html) |
| Research analysis | Jupyter notebook | `examples/notebooks/analysis/` | Yes (.ipynb + .html) |
| Unit test output | Test script | `tests/outputs/` | No (gitignored) |
| Reference data | Any | `*/reference/` | Yes (for comparison) |

## 🔧 **Development Workflow**

### **Branch Naming Convention** ⚠️ **MANDATORY**
**Always use `<type>/<short-description>` format for all branches:**

**Branch Types**:
- `feature/` - New functionality or capabilities
- `fix/` - Bug fixes and corrections
- `chore/` - Maintenance, tooling, dependencies, infrastructure
- `docs/` - Documentation updates and improvements
- `refactor/` - Code restructuring without functional changes
- `test/` - Adding or updating tests

**Examples**:
```bash
# ✅ GOOD - Clear type and purpose
feature/semi-lagrangian-solver
chore/expand-mypy-coverage
fix/convergence-criteria-bug
docs/update-api-reference
refactor/factory-pattern-cleanup

# ❌ BAD - Generic or unclear names
dev
main-work
updates
temp-branch
```

**Implementation**: All future branches must follow this convention. No exceptions for temporary or experimental branches.

### **Hierarchical Branch Structure** ⚠️ **CRITICAL**

**Abstract Principle: Organize work through hierarchical branch structures that reflect logical dependencies and functional groupings.**

**Universal Application**: This principle applies to **ALL branch types** - `feature/`, `fix/`, `docs/`, `refactor/`, `test/`, and `chore/`. Any multi-step or logically related work should follow this hierarchical approach regardless of branch type prefix.

#### **Core Principles**

1. **Always Work on Branches**
   - ❌ Never commit directly to `main`
   - ❌ Never commit directly to parent branches without following proper workflow
   - ✅ Create feature branches for all work
   - ✅ Use parent branches to organize related work

2. **Establish Clear Hierarchy** (applies to all branch types)
   ```
   main (production-ready code)
    └── <type>/major-work (parent branch for related work)
         ├── <type>/work-component-a (child branch)
         ├── <type>/work-component-b (child branch)
         └── <type>/work-component-c (child branch)

   Examples:
   main
    └── feature/neural-solver (parent)
         ├── feature/dgm-implementation (child)
         ├── feature/fno-implementation (child)
         └── feature/neural-solver-tests (child)

   main
    └── fix/convergence-issues (parent)
         ├── fix/hjb-convergence (child)
         ├── fix/fp-convergence (child)
         └── fix/convergence-tests (child)
   ```

3. **Respect Merge Order** ⚠️ **MANDATORY**
   - **Child → Parent**: Always merge child branches to parent first
   - **Parent → Main**: Only merge parent to main when all children complete
   - **Dependencies**: If child-B depends on child-A, merge child-A first

4. **When to Use Hierarchy**
   - **Multi-step refactoring**: Each step is a child branch
   - **Feature development**: Sub-features as children
   - **Systematic cleanup**: Categories as children (e.g., code quality fixes)
   - **Related changes**: Group logically connected work

5. **Branch Lifecycle Management**
   - **Create child**: When starting work on a sub-feature
   - **Merge to parent**: When a logical unit of work is complete and tested
   - **Keep child alive**: If more related work remains in that category
   - **Delete child**: Only when the entire sub-feature is complete

   **Example**: Fixing RUF012 errors
   - ✅ Create `chore/fix-classvar-annotations`
   - ✅ Fix 5 errors, commit, merge to parent
   - ✅ Keep branch alive if more RUF012 errors might appear
   - ✅ Continue working, make more commits
   - ✅ Merge again to parent when ready
   - ✅ Delete only when ALL RUF012 work is done

   **Principle**: Branch deletion depends on **completeness of sub-feature**, not immediacy of merge.

#### **Workflow Example: Systematic Code Quality Cleanup**

```bash
# Step 1: Create parent branch from main
git checkout main
git checkout -b chore/code-quality-systematic-cleanup

# Step 2: Create child branch from parent
git checkout chore/code-quality-systematic-cleanup
git checkout -b chore/fix-unused-variables

# Step 3: Make changes and commit to child
git add -A
git commit -m "Fix unused variables"
git push -u origin chore/fix-unused-variables

# Step 4: Merge child → parent
git checkout chore/code-quality-systematic-cleanup
git merge chore/fix-unused-variables --no-ff
git push

# Step 5a (OPTION A): Delete child if sub-feature is complete
git branch -d chore/fix-unused-variables

# Step 5b (OPTION B): Keep child alive if more work remains
# (simply don't delete - you can continue working on it later)

# Step 6: Repeat for other children
git checkout -b chore/fix-unused-imports
# ... work, commit, push
git checkout chore/code-quality-systematic-cleanup
git merge chore/fix-unused-imports --no-ff
git push

# Step 7: When all children complete, merge parent → main
git checkout main
git merge chore/code-quality-systematic-cleanup --no-ff
git push
```

#### **Benefits of Hierarchical Structure**

- **Organized History**: Related changes grouped logically
- **Easy Rollback**: Can revert entire feature set by reverting parent merge
- **Parallel Work**: Multiple developers can work on different children
- **Clear Progress**: Parent branch shows cumulative progress
- **Clean Main**: Main only receives complete, tested feature sets

#### **Common Patterns**

| Pattern | Parent Branch | Child Branches |
|:--------|:--------------|:---------------|
| **Feature Development** | `feature/new-solver` | `feature/solver-core`, `feature/solver-tests`, `feature/solver-docs` |
| **Systematic Cleanup** | `chore/code-quality` | `chore/fix-imports`, `chore/fix-types`, `chore/fix-lint` |
| **Architecture Refactor** | `refactor/factory-pattern` | `refactor/solver-factory`, `refactor/problem-factory`, `refactor/config-factory` |
| **Documentation Overhaul** | `docs/api-reference` | `docs/solver-api`, `docs/geometry-api`, `docs/utils-api` |

#### **Anti-Patterns to Avoid**

❌ **Creating orphaned branches**: Each branch should have clear parent
❌ **Merging out of order**: Child to main before parent violates hierarchy
❌ **Too deep nesting**: Keep maximum 2-3 levels (main → parent → child)
❌ **Unclear relationships**: Branch names should indicate hierarchy

**Enforcement**: When working with AI assistance, always establish branch hierarchy at the start of multi-step work. The AI should create parent branch first, then work in child branches, respecting merge order throughout.

### **Dynamic Branch Management Strategy** ⚠️ **ADAPTIVE**

**Core Principle**: Branch management should adapt to project phase, team size, and work complexity.

#### **Context-Aware Branch Limits**

Branch limits scale with project activity level and work complexity:

| Project Phase | Max Branches | Branch Lifetime | Merge Cadence |
|:--------------|:-------------|:----------------|:--------------|
| **Maintenance** (bug fixes, small updates) | 3-5 | 1-3 days | Daily |
| **Active Development** (new features) | 5-8 | 3-7 days | Weekly |
| **Major Refactor** (architecture changes) | 8-12 | 1-2 weeks | Bi-weekly |
| **Research/Exploration** (experimental) | 10-15 | 2-4 weeks | Monthly |

**Current Phase**: Auto-detect or manually declare in this file.

#### **Intelligent Branch Assessment** ⚠️ **BEFORE CREATING**

Instead of rigid rules, use this decision tree:

```bash
# Step 1: Assess repository state
BRANCH_COUNT=$(git branch -r | grep -v "HEAD\|main" | wc -l | tr -d ' ')
OPEN_PRS=$(gh pr list --state open --json number --jq '. | length')
STALE_BRANCHES=$(gh pr list --state open --json number,updatedAt --jq 'map(select((now - (.updatedAt | fromdateiso8601)) > (7*24*3600))) | length')

echo "Branch Health Check:"
echo "  Total branches: $BRANCH_COUNT"
echo "  Open PRs: $OPEN_PRS"
echo "  Stale PRs (>7 days): $STALE_BRANCHES"

# Step 2: Determine action based on state
if [ $STALE_BRANCHES -gt 2 ]; then
  echo "⚠️  ACTION REQUIRED: Clean up stale branches first"
  echo "Run: gh pr list --state open --json number,title,updatedAt"
elif [ $BRANCH_COUNT -gt 12 ]; then
  echo "⚠️  HIGH: Review branch necessity before creating new ones"
elif [ $BRANCH_COUNT -gt 8 ]; then
  echo "⚡ MODERATE: Consider consolidating related work"
else
  echo "✅ HEALTHY: Safe to create new branches"
fi
```

#### **Branch Reuse vs. New Branch Decision Tree** ⚠️ **CORE PRINCIPLE**

**Question: "Should I create a new branch or reuse an existing one?"**

Use this decision framework:

```
Is the work:
├─ Fixing bugs/typos in same area?
│  └─ ✅ REUSE existing fix/* branch OR work on parent branch
│
├─ Related to existing open PR?
│  └─ ✅ ADD to existing PR branch (if not merged yet)
│
├─ Part of larger feature in progress?
│  └─ ✅ WORK on parent feature branch directly
│
├─ Independent new feature/capability?
│  └─ 🆕 CREATE new branch (if count < limit)
│
└─ Experimental/proof-of-concept?
   └─ 🆕 CREATE research/* branch
```

**Key Principle**: **Default to working on existing branches; create new ones only when work is truly independent.**

**Examples**:

**Scenario 1: Multiple Bug Fixes**
```bash
# ✅ GOOD: Reuse fix branch for related bugs
git checkout fix/solver-convergence-issues
# Fix bug A, commit, push
# Fix bug B, commit, push
# Fix bug C, commit, push
# Create ONE PR with all fixes

# ❌ BAD: Creating separate branches
git checkout -b fix/solver-bug-a  # Unnecessary
git checkout -b fix/solver-bug-b  # Unnecessary
git checkout -b fix/solver-bug-c  # Unnecessary
```

**Scenario 2: Iterative Feature Development**
```bash
# ✅ GOOD: Work directly on parent branch
git checkout feature/neural-solver
# Implement base architecture, commit, push
# Add training loop, commit, push
# Add validation, commit, push
# Create PR when feature complete

# ❌ BAD: Creating child branches for every step
git checkout -b feature/neural-solver-base
git checkout -b feature/neural-solver-training
git checkout -b feature/neural-solver-validation
```

**Scenario 3: When to Create Child Branches**
```bash
# ✅ GOOD: Child branches for PARALLEL independent work
git checkout -b feature/mfg-toolkit-parent

# Create child ONLY if:
# 1. Work can proceed in parallel
# 2. Child might fail/be abandoned
# 3. Requires separate review/testing

git checkout -b feature/toolkit-dgm-solver    # Independent component
git checkout -b feature/toolkit-fno-solver    # Independent component
git checkout -b feature/toolkit-pinn-solver   # Independent component

# Each can be developed, tested, merged independently
```

**When to Create New Branch (Checklist)**:

✅ Create new branch if:
- Work is **independent** from existing branches
- Will take **>1 week** and needs isolation
- Is **experimental** and might be discarded
- Requires **parallel development** with other work
- Targets **different code areas** than existing branches

❌ Don't create new branch if:
- Work is **related** to existing PR
- Is a **quick fix** (<1 hour)
- Is **iterative** work on existing feature
- Would be **5th+ branch** without cleanup

#### **Dynamic Branch Strategy Selection**

Choose strategy based on the work type:

**1. Quick Fix Strategy** (bugs, typos, small changes):
```bash
# REUSE existing fix/* branch for similar work
git checkout fix/quick-fixes || git checkout -b fix/quick-fixes
# Make fix, commit, push, create PR immediately
# Keep branch alive for future quick fixes
```

**2. Feature Sprint Strategy** (short-term features):
```bash
# Create dedicated feature branch, work 2-3 days, merge quickly
git checkout -b feature/specific-feature
# Work iteratively, commit daily to THIS branch
# Don't create child branches unless necessary
# Create PR within 3 days, merge within 5 days
```

**3. Parallel Development Strategy** (multiple independent features):
```bash
# Use hierarchical branches ONLY when work can proceed in parallel
git checkout -b feature/major-work-parent
git checkout -b feature/component-a  # Child 1 - independent
git checkout -b feature/component-b  # Child 2 - independent
# Work on children in parallel
# Merge children to parent, then parent to main when complete
```

**4. Research/Exploration Strategy** (experimental work):
```bash
# Longer-lived branches OK for experimental work
git checkout -b research/experimental-approach
# Work iteratively on THIS branch
# Regular rebases onto main, document findings
# Merge when proven OR close without merge if not viable
```

#### **Automated Branch Health Monitoring**

Create a health check script that assesses branch quality:

```bash
#!/bin/bash
# Save as: scripts/check_branch_health.sh

echo "=== Branch Health Report ==="

# Count branches by type
for type in feature fix chore docs refactor test research; do
  count=$(git branch -r | grep "origin/$type/" | wc -l | tr -d ' ')
  echo "$type: $count branches"
done

# Find stale branches (no activity in 7+ days)
echo -e "\n=== Stale Branches (>7 days) ==="
gh pr list --state open --json number,title,headRefName,updatedAt --jq \
  'map(select((now - (.updatedAt | fromdateiso8601)) > (7*24*3600))) |
   .[] | "PR #\(.number): \(.headRefName) (\(.title))"'

# Find branches with conflicts
echo -e "\n=== Branches Behind Main ==="
for branch in $(git branch -r | grep -v HEAD | grep -v main); do
  behind=$(git rev-list --count $branch..origin/main)
  if [ $behind -gt 10 ]; then
    echo "$branch: $behind commits behind (needs rebase)"
  fi
done

# Recommend actions
echo -e "\n=== Recommendations ==="
TOTAL_BRANCHES=$(git branch -r | grep -v "HEAD\|main" | wc -l | tr -d ' ')
if [ $TOTAL_BRANCHES -gt 12 ]; then
  echo "⚠️  Branch count HIGH ($TOTAL_BRANCHES) - prioritize merging or closing"
elif [ $TOTAL_BRANCHES -gt 8 ]; then
  echo "⚡ Branch count MODERATE ($TOTAL_BRANCHES) - review stale branches"
else
  echo "✅ Branch count HEALTHY ($TOTAL_BRANCHES)"
fi
```

**Run this weekly**: `bash scripts/check_branch_health.sh`

#### **Adaptive Cleanup Cadence**

Cleanup frequency adapts to branch velocity:

**High Activity** (>3 PRs/week):
- Daily: `git fetch --prune && gh pr list --state merged --limit 5`
- Delete merged branches immediately after CI passes

**Moderate Activity** (1-3 PRs/week):
- Every 2-3 days: Run branch health check
- Merge or close stale PRs weekly

**Low Activity** (<1 PR/week):
- Weekly: Full branch audit
- Monthly: Deep cleanup and rebase

**Automated Cleanup Commands**:
```bash
# Smart cleanup: Delete merged branches from last 30 days
gh pr list --state merged --search "merged:>$(date -v-30d +%Y-%m-%d)" \
  --json headRefName --jq '.[].headRefName' | \
  xargs -I {} git push origin --delete {} 2>/dev/null

# Prune local tracking refs
git fetch --prune origin

# List candidates for closure (stale, unmerged)
gh pr list --state open --search "updated:<$(date -v-14d +%Y-%m-%d)" \
  --json number,title,updatedAt --jq '.[] | "PR #\(.number): \(.title)"'
```

#### **AI Assistant Dynamic Guidelines** ⚠️ **CONTEXT-AWARE**

The AI assistant should adapt behavior based on repository state:

**1. Pre-Branch Creation Assessment**:
```bash
# AI runs this check before suggesting new branch
BRANCH_COUNT=$(git branch -r | grep -v "HEAD\|main" | wc -l | tr -d ' ')
STALE_COUNT=$(gh pr list --state open --search "updated:<$(date -v-7d +%Y-%m-%d)" --json number --jq '. | length')

if [ $STALE_COUNT -gt 2 ]; then
  echo "🚨 Stale branches detected - propose cleanup first"
elif [ $BRANCH_COUNT -gt 12 ]; then
  echo "⚠️  High branch count - ask user about project phase"
elif [ $BRANCH_COUNT -gt 8 ]; then
  echo "⚡ Moderate branches - suggest consolidation opportunity"
else
  echo "✅ Healthy state - proceed with branch creation"
fi
```

**2. Context-Sensitive Recommendations**:

| Scenario | AI Action |
|:---------|:----------|
| **Bug fix + existing fix/* branch** | Suggest reusing existing branch |
| **Small change + open PR for same area** | Suggest adding to existing PR |
| **Large feature + >8 branches** | Suggest hierarchical parent/child structure |
| **Experimental work** | Suggest `research/*` branch with clear timeline |
| **Stale branches present** | Proactively ask about closing/merging |

**3. Merge Velocity Optimization**:
```markdown
AI should track:
- Time since branch creation
- Number of commits
- CI status
- Review status

If branch is:
- ✅ CI passing + reviewed → Suggest immediate merge
- ⏳ >5 days old + 0 reviews → Prompt user for review request
- 🔄 >10 days old → Suggest rebase or close decision
- 🔴 CI failing → Don't create new branches until fixed
```

**4. Adaptive Communication**:

**Healthy State** (<8 branches):
> "Creating `feature/new-solver` branch for solver implementation."

**Warning State** (8-12 branches):
> "Note: 10 active branches detected. I'll create `feature/new-solver`, but consider reviewing stale branches soon."

**Critical State** (>12 branches):
> "⚠️ 14 active branches - above recommended limit. Should we clean up merged/stale branches before creating new ones? I can help identify candidates."

**5. Proactive Maintenance Prompts**:

AI should remind about maintenance at appropriate intervals:
- After every 3rd PR merge: "Consider running branch health check"
- When creating 5th branch: "Approaching branch limit - review stale PRs?"
- When detecting stale branches: "PR #142 hasn't been updated in 10 days - still needed?"

### **Asynchronous Multi-Branch Development** ⚠️ **CRITICAL**

**Context**: When working on multiple branches simultaneously (especially with AI assistance), branches can become stale, conflict-prone, and difficult to manage. These principles prevent branch chaos.

#### **1. Branch Lifecycle Management**

**Problem**: Branches become stale (many commits behind main) while sitting unmerged.

**Principles**:
- ✅ **Merge Early, Merge Often**: Don't let branches sit for extended periods
- ✅ **Daily Rebase Habit**: If branch lives > 1 day, rebase daily onto main
- ✅ **Small, Focused Branches**: Keep branches small (< 5 commits ideal)
- ✅ **Delete Immediately After Merge**: Don't leave merged branches around
- ✅ **Maximum Active Branches**: Limit to 3-5 active branches at once
- ✅ **PR-Driven Development**: Create PR immediately when pushing, merge quickly
- ✅ **No Orphaned Branches**: Every branch must have a linked issue or clear purpose

**Branch Count Guidelines** ⚠️ **MANDATORY**:
- **< 5 branches**: Healthy, manageable state
- **5-10 branches**: Warning zone - start cleanup immediately
- **> 10 branches**: Critical - stop creating new branches until cleanup complete

**Enforcement**:
```bash
# Before creating new branch, check active count
git branch -r | wc -l

# If > 5, clean up merged branches first
gh pr list --state merged --limit 20 --json number,headRefName --jq '.[].headRefName' | \
  xargs -I {} git push origin --delete {}
```

**Daily Routine**:
```bash
# Morning routine for long-lived branches
git checkout my-feature
git fetch origin
git rebase origin/main
git push --force-with-lease
```

#### **2. Branch Status Tracking**

**Problem**: Lost track of which branches were merged, which needed work.

**Principles**:
- ✅ **Use GitHub Projects/Issues**: Link each branch to a specific issue
- ✅ **Naming with Issue Numbers**: `<type>/<issue-number>-<short-description>`
  - Example: `feature/128-performance-dashboard`
- ✅ **Regular Status Audits**: Weekly review of all branches
- ✅ **PR-First Workflow**: Create PR immediately when pushing branch

**Weekly Branch Audit**:
```bash
# Check all branches status
git fetch --prune origin
for branch in $(git branch -r | grep -v HEAD | grep -v main); do
  ahead=$(git rev-list --count origin/main..$branch)
  behind=$(git rev-list --count $branch..origin/main)
  echo "$branch: $ahead ahead, $behind behind"
done
```

#### **3. Merge Order Strategy**

**Problem**: Trying to merge branches in wrong order caused conflicts.

**Principles**:
- ✅ **Foundation First**: Merge infrastructure/core changes before features
- ✅ **Dependency Aware**: Merge dependencies before dependents
- ✅ **Test Changes Last**: Merge test-only branches after code changes
- ✅ **Documentation Separate**: Docs can merge independently

**Merge Priority Order**:
```
1. Infrastructure: fix/ci-workflow-updates
2. Core Features: feature/performance-tracker
3. Dependent Features: feature/visualization (depends on tracker)
4. Tests: test/tracker-tests
5. Documentation: docs/tracker-guide
```

#### **4. Conflict Prevention**

**Problem**: Branches touched same files, causing merge conflicts.

**Principles**:
- ✅ **Communicate File Ownership**: Know who's working on what files
- ✅ **Atomic Changes**: One concern per branch
- ✅ **Avoid Broad Refactors**: Don't refactor while others work on features
- ✅ **Rebase Before Review**: Always up-to-date with main before requesting review

**Track Active Work** (for solo dev with AI):
```markdown
# Current Work In Progress
- feature/hjb-improvements: `mfg_pde/solvers/hjb.py`
- feature/performance-dashboard: `benchmarks/tracker.py`
- test/coverage-expansion: `tests/**`
```

#### **5. CI/CD Integration**

**Problem**: Didn't know branch status until attempted merge.

**Principles**:
- ✅ **CI on All Branches**: Run tests on every push
- ✅ **Branch Protection Rules**: Require CI pass before merge
- ✅ **Status Checks Visible**: Use GitHub status badges
- ✅ **Failed CI = Pause Work**: Fix CI before adding features

#### **6. Communication Protocol** (Solo Dev + AI)

**Principles**:
- ✅ **Daily Review**: Check what changed in main overnight
- ✅ **Issue Documentation**: Document decisions on GitHub issues
- ✅ **Descriptive PRs**: Explain WHY, not just WHAT
- ✅ **Link Everything**: Issues ↔ PRs ↔ Commits

**PR Description Template**:
```markdown
## Changes
- Brief summary

## Related Issues
Closes #128

## Dependencies
- Requires PR #127 to be merged first (or none)

## Testing
- [x] Unit tests pass
- [x] Integration tests pass
- [x] Manual testing complete

## Merge Checklist
- [x] Up to date with main (rebased)
- [x] CI passing
- [x] No merge conflicts
- [ ] Ready to merge
```

#### **7. Emergency Protocols**

**Scenario 1: Branch Too Far Behind**
```bash
# Option A: Rebase (preferred if no collaborators)
git checkout my-feature
git rebase origin/main
git push --force-with-lease

# Option B: Merge (if branch is shared)
git merge origin/main

# Option C: Fresh Start (if massive conflicts)
git checkout main
git checkout -b feature/fresh-start
git cherry-pick <commits-to-keep>
```

**Scenario 2: Accidentally Merged Wrong Branch**
```bash
# Revert the merge commit
git revert -m 1 <merge-commit-hash>
git push origin main
```

**Scenario 3: Lost Track of Branch Purpose**
```bash
# Review branch history
git log main..feature-branch --oneline

# Check what files changed
git diff main...feature-branch --name-only

# If unclear, close and start fresh
git branch -D feature-branch
# Create new branch with clear purpose
```

#### **8. Automated Cleanup**

**Principles**:
- ✅ **Auto-delete on Merge**: Enable GitHub setting
- ✅ **Regular Pruning**: Monthly cleanup of merged branches
- ✅ **Local Sync**: Run `git fetch --prune` regularly

**Cleanup Commands**:
```bash
# Delete all local branches already merged to main
git branch --merged main | grep -v "main" | xargs -I {} git branch -d {}

# Delete remote tracking refs for deleted branches
git fetch --prune origin

# Force delete all local branches not on remote
git branch -vv | grep ': gone]' | awk '{print $1}' | xargs -I {} git branch -D {}
```

#### **Quick Reference: Multi-Branch Checklist**

**Before Creating Branch**:
- [ ] Link to specific GitHub issue
- [ ] Choose appropriate type prefix (feature/fix/test/docs/chore)
- [ ] Ensure main is up to date locally
- [ ] Check no other branch is working on same files

**During Development**:
- [ ] Commit small, atomic changes
- [ ] Push to remote daily
- [ ] Rebase onto main if branch lives > 1 day
- [ ] Run tests locally before pushing
- [ ] Keep branch focused (avoid scope creep)

**Before Merging**:
- [ ] Rebase onto latest main
- [ ] All CI checks passing
- [ ] No merge conflicts
- [ ] PR description complete
- [ ] Linked issues will auto-close

**After Merge**:
- [ ] Delete remote branch immediately
- [ ] Delete local branch
- [ ] Close related issues (if not auto-closed)
- [ ] Pull main to get merged changes

#### **Most Important Lessons**

1. **Merge frequency > Branch organization**: Better to merge often than have perfect branches
2. **Visibility > Perfection**: Create PR early for visibility, even if not ready (use draft PRs)
3. **Rebase daily**: Don't let branches drift > 5 commits behind main
4. **Delete aggressively**: Merged branches are clutter and confusion
5. **Communicate implicitly**: Use GitHub issues/PRs as communication log
6. **One branch at a time**: Focus on completing one branch before starting another
7. **When in doubt, merge**: If branch is "good enough", merge it - don't let perfect be enemy of done

**Real Example from This Project**:
We had 4 branches (`feature/notebook-export-enhancement`, `feature/solver-result-analysis`, `fix/ci-workflow-timeout-and-triggers`, `test/phase2.5-workflow-remaining`) that were 2-24 commits behind main. All turned out to be already merged via PRs but not cleaned up. Following these principles would have prevented this situation by:
- Creating PRs immediately (visibility)
- Deleting branches after merge (cleanup)
- Daily status checks (awareness)
- Merging frequently (preventing drift)

### **GitHub Issue and PR Management** ⚠️ **MANDATORY**
**Every issue MUST be properly labeled before work begins:**

**Required Labels (All 4 categories)**:
1. **Priority**: `priority: high/medium/low`
2. **Area**: `area: algorithms/geometry/performance/config/visualization`
3. **Size**: `size: small/medium/large` (effort estimation)
4. **Type**: `bug/enhancement/documentation/type-checking`

**Optional Labels**:
- `status: blocked/in-review` (workflow tracking)
- `good first issue/help wanted` (community)

**Label Application Workflow**:
```bash
# ✅ GOOD - Creating properly labeled issue
gh issue create --title "Add semi-Lagrangian HJB solver" \
  --label "priority: medium,area: algorithms,size: large,enhancement"

# ✅ GOOD - Adding labels to existing issue
gh issue edit 42 --add-label "priority: high,area: performance,size: small,bug"
```

**Pull Request Labeling**:
- Inherit labels from linked issues
- Add `status: in-review` when ready
- Use area labels for reviewer assignment

### **When Adding New Features**
1. **Create/verify issue**: Ensure proper labeling before starting work
2. **Create branch**: `git checkout -b feature/descriptive-name`
3. **Core code**: Add to appropriate `mfg_pde/` subdirectory
4. **Examples**: Create in `examples/basic/` or `examples/advanced/`
5. **Tests**: Add unit tests to `tests/unit/` or `tests/integration/`
6. **Documentation**: Update relevant `docs/` category
7. **Benchmarks**: Add performance analysis to `benchmarks/` if applicable
8. **Label PR**: Apply appropriate labels matching the linked issue

### **Code Quality Expectations**
- Follow `docs/development/CONSISTENCY_GUIDE.md`
- Use type hints and comprehensive docstrings
- Include error handling and validation
- Support both interactive and non-interactive usage

### **Ruff Version Management** ⚠️ **IMPORTANT**
**Strategy**: Pin with periodic automated updates

**Current Version**: `ruff==0.13.1` (Updated: 2025-10-07)

**Why Pin?**
- ✅ Consistent formatting across all environments (local, CI, pre-commit)
- ✅ Reproducible code formatting over time
- ✅ No surprise CI failures from ruff updates
- ✅ Controlled review of formatting changes

**Automated Update System**:
1. **Monthly Check**: GitHub Action runs 1st of each month
   - Automatically checks for new ruff releases
   - Creates PR if update available
   - Includes formatting changes and release notes

2. **Manual Update**: Use update script as needed
   ```bash
   # Check for updates
   python scripts/update_ruff_version.py --check

   # Apply update interactively
   python scripts/update_ruff_version.py --update

   # Force specific version
   python scripts/update_ruff_version.py --force 0.14.0
   ```

3. **Files Updated**:
   - `.pre-commit-config.yaml` - Controls pre-commit hooks
   - `.github/workflows/modern_quality.yml` - Controls CI checks
   - Both must match for consistency

**Review Process** when update PR is created:
1. Review formatting changes: `git diff`
2. Run tests: `pytest tests/`
3. Check pre-commit: `pre-commit run --all-files`
4. Merge if all checks pass

**Documentation**: See `docs/development/RUFF_VERSION_MANAGEMENT.md` for complete guide

### **Defensive Programming Principles** ⚠️ **CRITICAL**
**Always test finished architecture before proceeding to new features:**

1. **Test Before Extend**:
   - ✅ Verify existing functionality works before adding new features
   - ✅ Run relevant tests after making changes
   - ✅ Check that examples still execute successfully
   - ❌ Don't build on untested foundations

2. **Incremental Validation**:
   ```python
   # ✅ GOOD: Test each component
   def add_feature():
       # Step 1: Test existing code
       run_existing_tests()

       # Step 2: Implement new feature
       implement_new_code()

       # Step 3: Test integration
       run_integration_tests()

   # ❌ BAD: Build without testing
   def add_feature():
       implement_feature_A()
       implement_feature_B()  # Assumes A works
       implement_feature_C()  # Assumes A and B work
       # Hope it all works together!
   ```

3. **Regression Prevention**:
   - Write tests for bug fixes
   - Verify fixes don't break existing functionality
   - Document known issues and workarounds

4. **Architecture Validation**:
   - Test imports and API surface
   - Verify backward compatibility
   - Check type checking passes
   - Run linter and formatter

5. **Example-Driven Development**:
   - Create working examples for new features
   - Examples serve as integration tests
   - If examples don't run, feature isn't ready

**Checklist Before Committing**:
```bash
# 1. Run tests for affected modules
pytest tests/unit/test_affected_module.py

# 2. Run type checking
mypy mfg_pde/affected_module.py

# 3. Run linter
ruff check mfg_pde/affected_module.py

# 4. Test an example
python examples/basic/relevant_example.py

# 5. Only then commit
git add ... && git commit -m "..."
```

### **Smart .gitignore Strategy** ⚠️ **CRITICAL**
Always use targeted patterns that preserve valuable code:
```bash
# ✅ GOOD: Only ignore root-level clutter
/*.png
/*_analysis.py
/temp_*.py

# ❌ BAD: Would ignore valuable examples and tests
*.png
*_analysis.py
temp_*.py
```

Use explicit preservation overrides for important directories:
```bash
# Always track important content
!examples/**/*.py
!examples/**/*.png
!tests/**/*.py
!docs/**/*.md
!benchmarks/**/*.py
```

## 📊 **Package Management**

### **Dependencies**
- **Core**: numpy, scipy, matplotlib (always available)
- **Interactive**: plotly, jupyter, nbformat (with fallbacks)
- **Progress**: tqdm (acceptable for progress bars in scripts and solvers)
- **Optional**: psutil for memory monitoring

### **Installation Context**
- Support both development (`pip install -e .`) and user installation
- Exclude development directories from package distribution
- Maintain compatibility with Python 3.8+

## 🎯 **Working Preferences**

### **GitHub Issue and Label Management** ⚠️ **IMPORTANT**

**Abstract Labeling Principle:**
Labels should form a **hierarchical taxonomy** that scales with project growth, following these core principles:

#### **Label Categories (Use Prefixes)**
- **`area:`** - Functional domains (algorithms, config, omegaconf, documentation, geometry, performance)
- **`type:`** - Work nature (infrastructure, enhancement, bug, research)
- **`priority:`** - Urgency level (high, medium, low)
- **`size:`** - Effort estimate (small, medium, large)
- **`status:`** - Workflow state (blocked, in-review, needs-testing)
- **`resolution:`** - Completion type (merged, superseded, wontfix)

#### **Color Coding Philosophy**
- **Red tones** (#d93f0b) - High priority, urgent items
- **Blue tones** (#0052cc) - Technical work, infrastructure, algorithms
- **Green tones** (#28a745) - Completed, successful resolution
- **Purple tones** (#8e7cc3) - Special categories (OmegaConf, superseded)
- **Yellow tones** (#fbca04) - Medium priority, needs attention

#### **Scaling Strategy**
As the project grows, **subdivide areas** before creating new label types:
- `area: algorithms` → `area: algorithms-hjb`, `area: algorithms-fp`
- `area: config` → `area: config-omegaconf`, `area: config-pydantic`

This prevents label explosion while maintaining clear categorization.

### **Task Management**
- Use TodoWrite tool for multi-step tasks
- Track progress and mark completion
- Apply consistent GitHub labels following the hierarchical taxonomy

### **Progressive Logging vs. Summary Documents** ⚠️ **CRITICAL**

**Principle**: Log incrementally, summarize at milestones.

**During Active Work**:
- ✅ **Log each step**: Test results, bug findings, code changes in simple log files or comments
- ✅ **Use TodoWrite**: Track progress with in-progress/completed tasks
- ✅ **Write technical notes**: Bug reports (e.g., `BUG5_CENTRAL_VS_UPWIND.md`) as findings emerge
- ❌ **NO frequent summaries**: Don't create `SESSION_SUMMARY_2025-10-31_AFTERNOON.md` after each small step

**When to Create Summaries**:
- ✅ **Phase completion**: End of major feature implementation
- ✅ **Critical milestones**: After fixing major bugs or completing refactors
- ✅ **Before/after breaks**: End of day if significant progress made
- ✅ **Investigation conclusions**: After debugging sessions that identify root causes

**Examples**:

```
❌ BAD (too frequent):
- BUG_INVESTIGATION_SUMMARY.md (created after finding Bug #1)
- BUG_INVESTIGATION_UPDATE.md (created after finding Bug #2)
- BUG_INVESTIGATION_LATEST.md (created after finding Bug #3)
- FINAL_SESSION_SUMMARY.md (created at end of day)

✅ GOOD (milestone-based):
- test_lower_damping.log (ongoing: test results)
- BUG5_CENTRAL_VS_UPWIND.md (critical: root cause identified)
- CONVERGENCE_DEBUG_2025-10-31.md (milestone: investigation complete, ready to fix)
```

**Rationale**:
- Summaries have overhead (creation time, future maintenance burden)
- Logs capture progress without breaking workflow
- Critical findings deserve dedicated technical notes
- Comprehensive summaries are valuable at natural stopping points

**Conflict Resolution**:
- Does NOT conflict with "Documentation Hygiene" checkpoints (those are for cleanup, not creation)
- Does NOT conflict with "Status Marking" (marking existing docs, not creating new ones)
- REINFORCES "Proactive Documentation Cleanup" (fewer docs to clean up if fewer created)

### **File Organization**
- Keep root directory clean (only essential directories)
- Use meaningful categorization
- Create README files for each major directory
- Maintain consistency with established patterns

### **Repository Management Principles** ⚠️ **IMPORTANT**
- **Examples and tests are valuable**: Never create .gitignore rules that are too strict for examples/ and tests/ directories
- **Preserve research-grade code**: All examples, demos, and analysis in proper directories should be tracked
- **Target cleanup carefully**: Use root-level patterns (`/*.py`) rather than global patterns (`*.py`)
- **Archive vs Delete**: When archiving obsolete code, it can be removed entirely rather than moved to archive/ if it has no historical value
- **Clean root, preserve structure**: Keep root directory clean but maintain all organized content in proper subdirectories

### **Directory Management Principles** ⚠️ **CRITICAL**
**BE PRUDENT WHEN CREATING NEW DIRECTORIES:**

**Before Creating Any New Directory Structure:**
1. **Analyze existing content first**: Read files to understand actual purpose vs. current location
2. **Create detailed migration plan**: Map old → new locations with justification
3. **Identify duplicate content**: Merge overlapping documentation before moving
4. **Update cross-references**: Fix all internal links before structural changes
5. **Move systematically**: One directory at a time with validation after each step

**Directory Creation Guidelines:**
- **Minimum 5+ files**: Don't create directories for single files
- **Clear functional purpose**: Each directory must have distinct, non-overlapping purpose
- **User-centric organization**: Structure should make sense to end users, not just developers
- **Avoid nested hierarchies**: Keep directory depth reasonable (max 3 levels)
- **Test before finalizing**: Ensure documentation builds and links work after changes

**Examples:**
```
❌ BAD: Creating /docs/advanced/ /docs/development/ /docs/future/ (overlapping purposes)
✅ GOOD: Creating /docs/user/ /docs/theory/ /docs/development/ (distinct purposes)

❌ BAD: Moving files first, then updating references
✅ GOOD: Updating references first, then moving files with validation
```

### **Archiving Strategy**
- **Delete don't archive**: Temporary analysis files, obsolete outputs, and clutter can be deleted rather than archived
- **Archive valuable history**: Only move significant code with historical/educational value to archive/
- **Remove completely**: Generated files, temporary scripts, and duplicate experiments should be removed
- **Focus on quality**: Better to have fewer, high-quality examples than many obsolete ones

### **GitHub Workflow Integration** ⚠️ **ROUTINE CHECKLIST**
**For every task/feature I work on**:

```bash
# 1. ALWAYS check if GitHub issue exists and is properly labeled
gh issue view [issue_number]

# 2. If issue missing labels, apply them BEFORE starting work
gh issue edit [issue_number] --add-label "priority: medium,area: algorithms,size: small,enhancement"

# 3. Create properly named branch linked to issue
git checkout -b feature/descriptive-name  # or chore/fix/docs/refactor

# 4. When creating PR, inherit issue labels and add status
gh pr create --title "Title" --body "Fixes #[issue_number]" \
  --label "priority: medium,area: algorithms,size: small,enhancement,status: in-review"

# 5. Reference in commits and documentation
# "Implements feature X as requested in Issue #[number]"
```

**Label System Reference**:
- **Priority**: `high` (urgent) | `medium` (planned) | `low` (future)
- **Area**: `algorithms` | `geometry` | `performance` | `config` | `visualization`
- **Size**: `small` (hours-1day) | `medium` (1-3days) | `large` (1+ weeks)
- **Type**: `bug` | `enhancement` | `documentation` | `type-checking`

### **Communication Style**
- Be concise and direct
- Focus on practical solutions
- Provide working examples
- Use professional documentation standards
- **Always reference GitHub issues** when discussing features or fixes

## 🤖 **AI Interaction Design Framework**

### **Mathematical Research Collaboration**
For advanced Mean Field Games research, follow structured prompt design principles outlined in `docs/development/AI_INTERACTION_DESIGN.md`:

**Core Mathematical Communication Standards:**
Establish consistent mathematical notation for all AI interactions and documentation.

**Research Context Framework:**
- **Mathematical Depth**: Graduate/research level with complete rigor
- **Computational Focus**: High-performance implementation with complexity analysis
- **Academic Style**: Journal-quality exposition with mathematical elegance
- **Multilingual Context**: Integration of mathematical traditions (German precision, French elegance, English clarity, Chinese rigor)

### **Prompt Architecture Templates**

**1. Theoretical Investigation Template:**
```markdown
**Mathematical Context**: Mean Field Games with [specific focus]
**Theoretical Framework**: [Viscosity Solutions | Optimal Transport | Stochastic Control]
**Problem Statement**: [Rigorous mathematical formulation with LaTeX]
**Expected Analysis**: Graduate/research level with complete mathematical rigor
**Implementation Requirements**: Production-quality code with performance analysis
```

**2. Computational Implementation Template:**
```markdown
**Implementation Challenge**: Design [numerical method] for MFG system
**Performance Specifications**: Target complexity, memory efficiency, convergence rates
**Code Architecture**: Factory patterns, strategy patterns, observer patterns
**Quality Assurance**: Unit tests, benchmarks, profiling integration
```

**3. Cross-Field Analysis Template:**
```markdown
**Interdisciplinary Analysis**: Connect MFG theory to [economics/finance/physics]
**Mathematical Bridge**: Shared structures, limiting behaviors, dual formulations
**Literature Integration**: Key references and theoretical foundations
**Practical Implications**: Field-specific applications and insights
```

### **AI Collaboration Protocols**

**Session Initialization:**
1. Establish mathematical context and notation standards
2. Specify rigor level and target audience 
3. Define computational performance expectations
4. Request specific output formats (LaTeX-ready, ASCII-compatible code)

**Quality Standards for AI Outputs:**
- **Mathematical Rigor**: Precise definitions, explicit assumptions, logical proofs
- **Computational Excellence**: Algorithm complexity, numerical stability, performance validation
- **Academic Communication**: Journal-quality exposition with proper context and references

## 📜 **Solo Maintainer's Mechanism for Change**

As the primary maintainer, this project uses a self-governance protocol to ensure disciplined and high-quality changes. All significant modifications should follow this process:

1.  **Propose in an Issue**: Open a GitHub Issue to serve as a "decision log." Clearly articulate the proposed change and the reasoning behind it.
2.  **Implement in a PR**: All work must be done in a feature branch and submitted via a Pull Request to leverage automated CI checks as a quality gate.
3.  **Conduct AI-Assisted Review**: Before merging, perform a self-review that includes requesting a formal review from the AI assistant against the standards outlined in this document.
4.  **Merge on Pass**: Merge the PR only after all automated checks and the AI-assisted review are successfully completed.

A great way to enforce this process is to use GitHub's **branch protection rules** for your `main` branch, which can require that all changes come through a PR and that all CI checks must pass before merging.

## 🔍 **Quality Assurance**

### **Before Completing Tasks**
- Verify all imports work correctly
- Check file placement follows conventions
- Update relevant documentation
- Test examples execute successfully
- Maintain consistency with established patterns

### **Consistency Checks**
- Mathematical notation: u(t,x), m(t,x)
- Import patterns: Use factory methods when available
- Documentation style: Professional, research-grade
- Code organization: Follow established directory structure

### **Documentation Status Tracking** 📋 **MANDATORY**
All completed features, closed issues, and finished development work must be clearly marked:

**Status Categories:**
- `[COMPLETED]` - Implemented features and finished work  
- `[RESOLVED]` - Fixed issues and closed problems
- `[ANALYSIS]` - Technical analysis documents (permanent reference)
- `[EVALUATION]` - Ongoing evaluations requiring periodic review
- `[SUPERSEDED]` - Obsolete documents replaced by newer versions

**Status Marking Examples:**
```
✅ [COMPLETED] Network MFG Implementation Summary
✅ [RESOLVED] Particle Collocation Fix Summary  
✅ [ANALYSIS] QP-Collocation Performance Bottlenecks
🔄 [EVALUATION] Pydantic Integration Analysis
📦 [SUPERSEDED] Old Achievement Summary
```

**Documentation Cleanup Principles:**
- Mark all completed work to show progress
- Use descriptive status indicators in filenames
- Remove truly obsolete content rather than archiving
- Consolidate overlapping documentation
- Ensure all documentation reflects current package state

---

**Last Updated**: 2025-09-28
**Repository Version**: Complete MFG framework with modern typing standards
**Claude Code**: Always reference this file for MFG_PDE conventions

**Current Status**: Production-ready framework with comprehensive solver ecosystem, advanced API design, and strategic development roadmap through 2027. See `docs/development/STRATEGIC_DEVELOPMENT_ROADMAP_2026.md` for current development priorities.
