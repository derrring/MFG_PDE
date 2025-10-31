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
from mfg_pde import ExampleMFGProblem, BoundaryConditions
from mfg_pde.factory import create_fast_solver
from mfg_pde.utils.logging import get_logger, configure_research_logging
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

### **Development Documentation** ⚠️ **CRITICAL**
Always document significant changes:

1. **Roadmap**: Update `STRATEGIC_DEVELOPMENT_ROADMAP_2026.md` for major features
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
from mfg_pde.utils.logging import get_logger, configure_research_logging
from tqdm import tqdm

configure_research_logging("session_name", level="INFO")
logger = get_logger(__name__)

for iteration in tqdm(range(max_iterations), desc="Solving MFG"):
    # solver logic
```

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

### **Branch Management** ⚠️ **ADAPTIVE**

**Context-Aware Limits**:

| Project Phase | Max Branches | Lifetime | Merge Cadence |
|:--------------|:-------------|:---------|:--------------|
| Maintenance | 3-5 | 1-3 days | Daily |
| Active Development | 5-8 | 3-7 days | Weekly |
| Major Refactor | 8-12 | 1-2 weeks | Bi-weekly |

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

**Current**: `ruff==0.13.1` (Updated: 2025-10-07)

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
- **Progress**: tqdm
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

**Last Updated**: 2025-11-01
**Repository Version**: Complete MFG framework with modern typing standards
**Claude Code**: Always reference this file for MFG_PDE conventions

**Current Status**: Production-ready framework with comprehensive solver ecosystem, advanced API design, and strategic development roadmap through 2027. See `docs/development/STRATEGIC_DEVELOPMENT_ROADMAP_2026.md` for current priorities.
