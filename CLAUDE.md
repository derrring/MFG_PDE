# Claude Code Preferences for MFG_PDE

This file contains preferences and conventions for Claude Code when working with the MFG_PDE repository.

## 🏗️ **Repository Structure Conventions**

### **Directory Organization**
- **`mfg_pde/`** - Core package code only
- **`tests/`** - Pure unit and integration tests (no demos)
- **`examples/`** - All demonstration code, organized by complexity:
  - `basic/` - Simple single-concept examples
  - `advanced/` - Complex multi-feature demonstrations
  - `notebooks/` - Jupyter notebook examples
  - `tutorials/` - Step-by-step learning materials
- **`benchmarks/`** - Performance comparisons and method analysis
- **`docs/`** - Documentation organized by category (see docs/README.md)
- **`archive/`** - Historical code (do not modify)

### **File Placement Rules**
- **Examples**: Always categorize by complexity (basic/advanced/notebooks)
- **Tests**: Only actual unit/integration tests, no demos
- **Benchmarks**: Performance analysis and method comparisons
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
from mfg_pde.config import create_fast_config
from mfg_pde.utils.logging import get_logger, configure_research_logging
```

### **Documentation Style**
- Use comprehensive docstrings with mathematical notation
- Include LaTeX expressions: `$u(t,x)$` and `$m(t,x)$`
- Reference line numbers when discussing code: `file_path:line_number`
- Follow consistency guide in `docs/development/CONSISTENCY_GUIDE.md`

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

# Step 5: Repeat for other children
git checkout -b chore/fix-unused-imports
# ... work, commit, push
git checkout chore/code-quality-systematic-cleanup
git merge chore/fix-unused-imports --no-ff
git push

# Step 6: When all children complete, merge parent → main
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
- Create summary documents for major changes
- Apply consistent GitHub labels following the hierarchical taxonomy

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
