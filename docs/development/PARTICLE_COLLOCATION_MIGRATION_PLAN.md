# Particle-Collocation Migration Plan: MFG_PDE → MFG-Research

**Status**: PLANNING
**Date**: 2025-10-12
**Rationale**: Particle-collocation methods are novel research contributions, not established infrastructure

---

## 🎯 **Strategic Decision**

**Move particle-collocation family to mfg-research** to maintain clean separation:
- **MFG_PDE**: Classical, well-established algorithms only
- **MFG-Research**: Novel methods including particle-collocation

### **Why This Migration Makes Sense**

1. **Novel Research**: Particle-collocation + QP optimization is your research contribution
2. **Under Development**: Still being refined and validated
3. **Not Widely Published**: Not yet a standard method in MFG literature
4. **Experimental Features**: Advanced convergence monitoring, QP constraints
5. **Research Freedom**: Needs ability to iterate without infrastructure constraints

---

## 📊 **Current State Analysis**

### **In MFG_PDE** (infrastructure repo):

**Core Implementation**:
```
mfg_pde/alg/numerical/mfg_solvers/particle_collocation_solver.py (670 lines)
```

**Factory Integration**:
```
mfg_pde/factory/solver_factory.py         - creates particle_collocation solvers
mfg_pde/factory/pydantic_solver_factory.py - validated particle_collocation config
```

**Tests** (extensive):
```
tests/integration/test_particle_collocation.py
tests/integration/test_mass_conservation_1d.py
tests/integration/test_mass_conservation_1d_simple.py
tests/boundary_conditions/test_*.py (4 files)
tests/svd_implementation/test_*.py (2 files)
tests/property_based/test_mfg_properties.py
tests/unit/test_factory_patterns.py
tests/unit/test_factory/test_pydantic_solver_factory.py
```

**Examples**:
```
examples/basic/multi_paradigm_comparison.py
examples/advanced/factory_patterns_example.py
examples/advanced/test_m_matrix_verification.py
benchmarks/solver_comparisons/*.py
```

**Dependencies** (files that import particle-collocation):
```
mfg_pde/core/highdim_mfg_problem.py
mfg_pde/core/plugin_system.py
mfg_pde/utils/cli.py
mfg_pde/utils/notebooks/reporting.py
mfg_pde/benchmarks/highdim_benchmark_suite.py
```

### **In MFG-Research** (research repo):

**Experimental Work**:
```
experiments/qp_particle_collocation/
├── implementation.py              - QP-enhanced version
├── experiments.py                 - Experimental runs
└── benchmarks/
    └── qp_optimization_success_summary.py
```

---

## 🚀 **Migration Strategy**

### **Option A: Complete Removal** (RECOMMENDED)

**Keep MFG_PDE clean** - remove all particle-collocation code:

**Advantages**:
- ✅ Clean separation of infrastructure vs research
- ✅ MFG_PDE remains purely classical methods
- ✅ No confusion about method maturity
- ✅ Forces proper research workflow

**Disadvantages**:
- ⚠️ Breaks examples/tests that use particle-collocation
- ⚠️ Requires updating factory system
- ⚠️ Need to document the removal

### **Option B: Deprecation Path**

**Gradual transition** with clear warnings:

**Phase 1**: Mark as experimental, add deprecation warnings
**Phase 2**: Move to separate experimental module
**Phase 3**: Remove from MFG_PDE entirely

### **Option C: Keep Minimal Stub**

**Maintain compatibility** but defer to mfg-research:

```python
# mfg_pde/alg/numerical/mfg_solvers/particle_collocation_solver.py
class ParticleCollocationSolver:
    """
    DEPRECATED: Particle-collocation methods have moved to mfg-research.

    This solver combines novel particle methods with GFDM collocation.
    As an experimental research method, it now lives in the mfg-research repo.

    To use particle-collocation:
    1. Install mfg-research package
    2. Import from mfg_research.algorithms.particle_collocation

    See: https://github.com/yourusername/mfg-research
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ParticleCollocationSolver has moved to mfg-research. "
            "Install mfg-research to use this method."
        )
```

---

## 📋 **Recommended Migration Plan** (Option A)

### **Step 1: Backup Current State** ✅
```bash
# Already done - everything is in git history
git log --all --oneline -- '*particle*collocation*'
```

### **Step 2: Move to MFG-Research**

**Create proper structure in mfg-research**:
```
mfg-research/
└── algorithms/
    └── particle_collocation/
        ├── __init__.py
        ├── solver.py               # Base particle-collocation (no QP)
        ├── qp_constraints.py       # Optional QP constraint layer
        ├── convergence.py          # Advanced monitoring
        ├── utils.py                # Helper functions
        └── tests/
            ├── test_basic.py
            ├── test_qp_constraints.py
            ├── test_mass_conservation.py
            └── test_boundary_conditions.py
```

**Unified Design Philosophy**:
QP particle-collocation is realized by adding an **optional QP constraint** to the base particle-collocation solver:

```python
# Base particle-collocation (no constraints)
solver = ParticleCollocationSolver(
    problem=problem,
    collocation_points=points,
    use_qp_constraints=False  # Default: pure particle-collocation
)

# QP-enhanced particle-collocation (with constraints)
solver = ParticleCollocationSolver(
    problem=problem,
    collocation_points=points,
    use_qp_constraints=True,   # Enable QP optimization
    qp_config={
        "constraint_type": "no_flux",
        "tolerance": 1e-6
    }
)
```

This design:
- ✅ Keeps particle-collocation as one unified method
- ✅ QP constraints are an optional enhancement layer
- ✅ No separate "QP solver" class needed
- ✅ Clean architecture: base method + optional feature

**Copy files from MFG_PDE**:
```bash
# In mfg-research repo
mkdir -p algorithms/particle_collocation
cp ~/code/MFG_PDE/mfg_pde/alg/numerical/mfg_solvers/particle_collocation_solver.py \
   algorithms/particle_collocation/solver.py

# Copy relevant tests
mkdir -p algorithms/particle_collocation/tests
cp ~/code/MFG_PDE/tests/integration/test_particle_collocation.py \
   algorithms/particle_collocation/tests/
```

### **Step 3: Remove from MFG_PDE**

**Delete implementation**:
```bash
# In MFG_PDE repo
git rm mfg_pde/alg/numerical/mfg_solvers/particle_collocation_solver.py
```

**Update factory systems**:
```python
# mfg_pde/factory/solver_factory.py
# Remove "particle_collocation" from SOLVER_TYPES
# Remove particle_collocation creation logic

# mfg_pde/factory/pydantic_solver_factory.py
# Remove particle_collocation config handling
```

**Update/remove tests**:
```bash
# Delete or skip particle-collocation tests
git rm tests/integration/test_particle_collocation.py
# OR add skip decorators to all tests
```

**Update examples**:
```bash
# Remove or update examples that use particle-collocation
# examples/basic/multi_paradigm_comparison.py - remove particle-collocation comparison
# benchmarks/solver_comparisons/*.py - remove particle-collocation
```

### **Step 4: Update Documentation**

**Add to MFG_PDE README**:
```markdown
## Advanced Methods

Some advanced research methods have moved to separate repositories:

- **Particle-Collocation Methods**: Novel QP-enhanced particle-collocation schemes
  are available in the [mfg-research](https://github.com/yourusername/mfg-research) repository.
```

**Update CLAUDE.md**:
```markdown
### Moved to MFG-Research

The following methods have graduated to research status:
- Particle-collocation solvers (experimental, under active research)
```

### **Step 5: Create Migration Notice**

**Add MIGRATION_NOTICE.md**:
```markdown
# Particle-Collocation Migration Notice

**Effective Date**: 2025-10-12

## What Changed

Particle-collocation methods have moved from MFG_PDE to mfg-research.

## Why

These methods are novel research contributions, not established infrastructure.
Keeping them in mfg-research allows:
- Rapid iteration without infrastructure constraints
- Clear distinction between established and experimental methods
- Proper research workflow and validation

## How to Use Particle-Collocation Now

Install mfg-research:
```bash
pip install -e /path/to/mfg-research
```

Update imports:
```python
# Old (MFG_PDE)
from mfg_pde.alg.numerical.mfg_solvers import ParticleCollocationSolver

# New (MFG-Research)
from mfg_research.algorithms.particle_collocation import ParticleCollocationSolver
```

## For Existing Code

Code using particle-collocation will need to:
1. Install mfg-research package
2. Update imports
3. Adjust to new API (if changed)
```

### **Step 6: Test Everything**

**Run full test suite**:
```bash
# In MFG_PDE
pytest tests/ -v

# Should pass with particle-collocation tests removed/skipped
```

**Verify examples still work**:
```bash
# Test classical examples
python examples/basic/lq_mfg_demo.py
python examples/basic/finite_difference_demo.py
```

### **Step 7: Commit and Document**

**Create PR in MFG_PDE**:
```bash
git checkout -b refactor/move-particle-collocation-to-research
git add -A
git commit -m "refactor: Move particle-collocation to mfg-research

Particle-collocation methods are novel research contributions that belong
in mfg-research, not the infrastructure repo.

Changes:
- Remove particle_collocation_solver.py
- Update factory systems to remove particle-collocation
- Remove/skip particle-collocation tests
- Update examples to use classical methods only
- Add migration notice and documentation

Rationale:
- Maintains clean infrastructure/research separation
- Allows research methods to evolve freely
- Keeps MFG_PDE focused on established algorithms

See: docs/development/PARTICLE_COLLOCATION_MIGRATION_PLAN.md"

git push origin refactor/move-particle-collocation-to-research
```

---

## 🎯 **Expected Outcome**

### **MFG_PDE After Migration**:
```
Classical Methods Only:
✅ Finite Difference (established)
✅ Finite Element (established)
✅ Standard Particle Methods (established)
✅ Basic GFDM (established)
❌ Particle-Collocation (MOVED → research)
❌ QP-Enhanced Methods (MOVED → research)
```

### **MFG-Research After Migration**:
```
Novel Research Methods:
✅ Particle-Collocation Solver
✅ QP-Enhanced Particle-Collocation
✅ Advanced Convergence Monitoring
✅ Experimental Boundary Handling
```

---

## ⚠️ **Migration Checklist**

Before starting migration:

- [ ] Backup current state (git tags/branches)
- [ ] Document all dependencies
- [ ] Identify all code using particle-collocation
- [ ] Plan test migration strategy
- [ ] Create mfg-research package structure
- [ ] Update mfg-research CLAUDE.md with new methods

During migration:

- [ ] Copy implementation to mfg-research
- [ ] Remove from MFG_PDE
- [ ] Update factory systems
- [ ] Update/remove tests
- [ ] Update examples
- [ ] Update documentation
- [ ] Add migration notices

After migration:

- [ ] Run full test suites (both repos)
- [ ] Verify examples work
- [ ] Update README files
- [ ] Create migration guide
- [ ] Test imports in both repos

---

## 📚 **Questions to Consider**

1. **Should we keep particle-collocation tests in MFG_PDE?**
   - Pro: Tests the infrastructure's plugin system
   - Con: Tests experimental code in infrastructure repo
   - **Decision**: Remove or mark as integration tests for mfg-research

2. **What about factory support?**
   - Pro: Nice to have unified factory
   - Con: Infrastructure shouldn't know about research methods
   - **Decision**: Remove from standard factory, use mfg-research's own factory

3. **Examples that compare methods?**
   - Pro: Useful to see particle-collocation vs classical
   - Con: Requires mfg-research as dependency
   - **Decision**: Move comparison examples to mfg-research

---

## 🚀 **Timeline Estimate**

- **Step 1-2** (Setup & Copy): 1-2 hours
- **Step 3** (Remove from MFG_PDE): 2-3 hours
- **Step 4** (Update docs): 1 hour
- **Step 5** (Create notices): 30 min
- **Step 6** (Testing): 1-2 hours
- **Step 7** (PR & Review): 1 hour

**Total**: ~8-10 hours of focused work

---

**Status**: Ready to proceed pending approval
**Next Step**: Choose migration option and create implementation branch
