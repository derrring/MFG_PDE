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
   - Update `CONSOLIDATED_ROADMAP_2025.md` for major feature additions
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
- **Primary**: Use Plotly for interactive plots when possible
- **Fallback**: Matplotlib with professional styling
- **Mathematical notation**: Consistent u(t,x) and m(t,x) conventions
- **Export**: Support both HTML (interactive) and PNG (static)

### **Notebook Creation**
- Use `mfg_pde.utils.notebook_reporting` for research notebooks
- Include mathematical formulations with LaTeX
- Provide working examples with error handling
- Export to both .ipynb and .html formats

## 🔧 **Development Workflow**

### **When Adding New Features**
1. **Core code**: Add to appropriate `mfg_pde/` subdirectory
2. **Examples**: Create in `examples/basic/` or `examples/advanced/`
3. **Tests**: Add unit tests to `tests/unit/` or `tests/integration/`
4. **Documentation**: Update relevant `docs/` category
5. **Benchmarks**: Add performance analysis to `benchmarks/` if applicable

### **Code Quality Expectations**
- Follow `docs/development/CONSISTENCY_GUIDE.md`
- Use type hints and comprehensive docstrings
- Include error handling and validation
- Support both interactive and non-interactive usage

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

### **Task Management**
- Use TodoWrite tool for multi-step tasks
- Track progress and mark completion
- Create summary documents for major changes

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

### **Communication Style**
- Be concise and direct
- Focus on practical solutions
- Provide working examples
- Use professional documentation standards

## 🔍 **Quality Assurance**

### **Before Completing Tasks**
- Verify all imports work correctly
- Check file placement follows conventions
- Update relevant documentation
- Test examples execute successfully
- Maintain consistency with existing patterns

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

**Last Updated**: 2025-07-31  
**Repository Version**: Network MFG implementation with enhanced visualization and smart cleanup  
**Claude Code**: Always reference this file for MFG_PDE conventions  

**Key Recent Updates**:
- ✅ Network MFG implementation with Lagrangian formulations
- ✅ Enhanced visualization system with trajectory tracking
- ✅ Smart .gitignore strategy preserving valuable examples/tests
- ✅ Repository cleanup principles for sustainable development