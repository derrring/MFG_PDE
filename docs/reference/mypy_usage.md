# MyPy Working Strategies - Production-Tested Techniques ✅

**Date**: 2024-09-24
**Status**: Phase 1 Proven - **55 errors reduced (13.3% improvement)**
**Production Compatibility**: ✅ Full toolchain compatibility (ruff, pre-commit, mypy)

---

## 🎯 **Core Philosophy: Strategic Value Over Error Elimination**

**Key Principle**: Focus on the 20% of code that provides 80% of the safety benefits. Success is measured by production health, not error count.

**Strategic Success Metrics** (Production Health First):

### **Primary Metrics (Production Health)**
1. **✅ CI/CD Pipeline Stability**: Do changes pass all quality gates?
2. **✅ Developer Velocity**: Are developers blocked by typing issues?
3. **✅ Runtime Error Prevention**: Fewer TypeError/AttributeError in production?

### **Secondary Metrics (Code Quality)**
4. **✅ IDE Support Quality**: Better autocomplete and refactoring?
5. **✅ Code Documentation**: Types as living documentation?
6. **✅ Team Onboarding**: Easier for new developers to understand code?

### **Tertiary Metrics (Hygiene)**
7. **✅ Error count reduction**: Measurable mypy error count decrease
8. **✅ Tool compatibility**: Passes ruff, pre-commit, mypy without issues
9. **✅ Team adoption**: Enhanced rather than impeded development flow

### **Warning Metrics (Over-Engineering Detection)**
- Time spent on typing > time spent on features
- Frequent `# type: ignore` additions
- Team resistance to typing practices
- Complex type gymnastics for simple functions

**The 80/20 Rule Applied**:
- **High-value targets**: Public APIs, data structures, critical logic, factory functions
- **Lower-value targets**: Internal utilities, test helpers, one-off scripts

---

## 📚 **Proven Working Strategies**

### **Strategy 1: Manual Stub Generation for Complex Libraries**

**When to Use**: External libraries with poor/missing type stubs that cause many import-not-found errors

**✅ Success Pattern**:
```python
# stubs/polars.pyi - Minimal, production-compatible
from typing import Any

class DataFrame:
    def __init__(self, data: Any = None, *, schema: Any = None) -> None: ...
    def select(self, *exprs: str | Expr | list[Expr]) -> DataFrame: ...
    def filter(self, predicate: Expr) -> DataFrame: ...
    def group_by(self, *columns: str) -> GroupBy: ...
    # Only include methods actually used in codebase

def col(name: str) -> Expr: ...
def __getattr__(name: str) -> Any: ...  # Catch-all for missing methods
```

**Configuration**:
```toml
# pyproject.toml
[tool.mypy]
mypy_path = "stubs"
```

**Impact**: **-35 errors** from Polars typing alone

**Why It Works**:
- Covers only actual usage patterns (no unused complexity)
- Production-compatible syntax (tested with Python 3.12)
- Tool-friendly (passes ruff without violations)
- Self-contained (no external dependencies)

---

### **Strategy 2: "Define Before Use" Pattern for Loop Variables**

**When to Use**: `var-annotated` errors for variables assigned inside loops or conditionals

**✅ Success Pattern**:
```python
# ❌ WRONG - mypy can't guarantee variable existence
for item in items:
    processed_value: ProcessedType = process(item)  # Doesn't work

# ✅ CORRECT - define before use
processed_value: ProcessedType | None = None  # or just ProcessedType
for item in items:
    processed_value = process(item)  # Now mypy knows it exists

# ✅ CORRECT - for loop variables specifically
neighbor_point: np.ndarray  # Declare before loop
for j, neighbor_point in enumerate(neighbor_points):
    # Variable guaranteed to exist
```

**Real Examples**:
```python
# Fixed in hjb_gfdm.py
neighbor_point: np.ndarray
for j, neighbor_point in enumerate(neighbor_points):
    delta_x = neighbor_point - center_point

# Fixed in convergence.py
self.error_history: deque[float] = deque(maxlen=history_length)

# Fixed in plugin_system.py
discovered: list[str] = []
```

**Impact**: **-10 errors** from correct loop variable typing

**Why It Works**:
- Satisfies mypy's "define before use" requirement
- Prevents NameError at runtime if loop doesn't execute
- Clear intention about variable's lifetime and scope

---

### **Strategy 3: Strategic Variable Annotations for Containers**

**When to Use**: Empty containers (`[]`, `{}`) that mypy can't infer types for

**✅ Success Pattern**:
```python
# ❌ Causes var-annotated errors
self.convergence_history = []
self.dual_vars = {}

# ✅ Clear, specific typing
self.convergence_history: list[dict[str, float]] = []
self.dual_vars: dict[str, np.ndarray] = {}
self.flow_analysis: dict[str, Any] = {
    "total_flow": [],
    "patterns": {},
}
```

**Impact**: **-6 errors** from container typing

**Why It Works**:
- Provides clear documentation of data structure expectations
- Enables better IDE support and error checking
- Minimal syntax overhead for significant mypy benefit

---

### **Strategy 4: Unused Type Ignore Cleanup**

**When to Use**: After successfully implementing stubs or type improvements

**✅ Success Pattern**:
```python
# Before (with stub implementation)
pl.DataFrame(data)  # type: ignore  # Now unnecessary

# After (stub makes it work)
pl.DataFrame(data)  # Clean, typed code
```

**Process**:
1. Implement type improvement (stub, annotation, etc.)
2. Test that mypy errors are resolved
3. Remove now-unnecessary `# type: ignore` comments
4. Verify full toolchain still passes

**Impact**: **-4 errors** from cleanup, improved code maintainability

**Why It Works**:
- Reduces technical debt and comment noise
- Ensures type improvements provide real value
- Maintains clean, professional codebase

---

## 🚫 **Anti-Patterns: What NOT to Do**

### **❌ Generated Stubs from `stubgen`**
```bash
stubgen -p polars -o stubs  # Creates syntax errors
```
**Problems**: Invalid syntax, tool incompatibility, maintenance burden
**Strategic Impact**: **High negative** - creates false positives that slow development
**Solution**: Use manual stubs covering only actual usage patterns

### **❌ Complex Loop Variable Annotation**
```python
# This doesn't work
for i, item in enumerate(items):
    item: ComplexType = item  # Redundant and ineffective
```
**Strategic Impact**: **Low negative** - wastes time without benefit
**Solution**: Use "define before use" pattern

### **❌ Over-Engineering Simple Cases**
```python
# Unnecessarily complex
def simple_function(x):  # type: ignore  # Complex type signature would be overkill
    return x + 1

# Better approach - strategic typing
def simple_function(x: float) -> float:  # Public API gets typing
    return x + 1
```
**Strategic Impact**: **Medium negative** - reduces developer velocity
**Solution**: Apply 80/20 rule - type public APIs, keep utilities simple

### **❌ Pursuing Perfect Type Coverage**
**Problem**: Trying to eliminate every single mypy error
**Strategic Impact**: **Very high negative** - diminishing returns, team resistance
**Solution**: Focus on high-value targets, accept strategic `Any` usage

### **❌ Ignoring False Positives**
**Problem**: Accepting noisy false positives from dynamic libraries
**Strategic Impact**: **High negative** - reduces trust in type system
**Solution**: Strategic stub creation to eliminate false positive noise

---

## 🎯 **Strategic Decision Framework** (80/20 Rule Applied)

### **Understanding False Positives vs Genuine Issues**
**Key Insight**: Not all mypy errors are created equal. Strategic typing focuses on eliminating false positives (mypy complains, code is fine) while addressing genuine bugs.

**False Positive Identification**:
- Dynamic library attributes (`attr-defined` from polars, JAX)
- Metaprogramming patterns mypy can't follow
- Scientific computing type transitions (list → numpy array)
- Complex logic where mypy loses type guards

**Genuine Issue Identification**:
- Missing None checks causing AttributeError
- Type mismatches in function signatures
- Incorrect return type annotations

### **Priority 1: Eliminate False Positives (80/20 High-Value)**
**Impact**: Very High - Improves developer experience and trust in type system
**Strategic Value**: Removes friction, builds team confidence in typing
- [x] **Manual stubs** for dynamic libraries causing 15+ attr-defined errors (polars.pyi proven)
- [x] **TYPE_CHECKING isolation** for complex libraries (JAX, OmegaConf pattern proven)
- [x] **Any escape hatch** for legitimate type lifecycle transitions (Week 1 success)

### **Priority 2: Type Public APIs (80/20 High-Value)**
**Impact**: High - Enhances team collaboration and IDE support
**Strategic Value**: Maximum benefit for team productivity and code documentation
- [ ] **Factory function signatures** - object creation patterns (create_solver, create_config)
- [ ] **Solver interface typing** - core algorithmic APIs (solve, update, converge methods)
- [ ] **Configuration system typing** - TypedDict/Pydantic patterns (SolverConfig, ProblemConfig)
- [ ] **Core data structure typing** - business domain objects (MFGProblem, MFGSolution)

### **Priority 3: Genuine Bug Prevention (Medium-Value)**
**Impact**: Medium - Prevents runtime AttributeErrors and TypeError
**Strategic Value**: Catches real bugs mypy can legitimately detect
- [ ] **None guard additions** where mypy correctly identifies missing checks
- [ ] **Assignment compatibility fixes** for NumPy/scientific computing (beyond Any escape hatch)
- [ ] **Return type consistency** in critical logic paths (Optional returns)

### **Priority 4: Cleanup & Maintenance (Low-Value)**
**Impact**: Low - Code hygiene, no functional benefit
**Strategic Value**: Maintains clean codebase after higher-priority work
- [x] **Type ignore cleanup** after successful improvements (4 errors removed)
- [ ] **Unused import removal** flagged by tools
- [ ] **Documentation typing** for non-critical utilities

### **Strategic "Never Do" List** (Anti-Patterns with Impact Assessment)
- ❌ **Complex generic typing** for research/experimental code → **High negative impact** (reduces velocity)
- ❌ **Perfect coverage pursuit** in low-value utility functions → **Very high negative impact** (diminishing returns)
- ❌ **Generated stubs** from stubgen → **High negative impact** (syntax errors, tool incompatibility)
- ❌ **Type annotation** of private/internal helpers unless causing issues → **Medium negative impact** (time waste)
- ❌ **Pursuing mypy silence over production health** → **Critical negative impact** (misaligned priorities)

---

## 🔧 **Implementation Checklist**

### **Before Starting**
- [ ] Establish baseline error count: `mypy mfg_pde/ | grep "Found.*errors"`
- [ ] Identify error categories: `mypy mfg_pde/ | grep "error:" | cut -d: -f4 | sort | uniq -c`
- [ ] Plan 2-3 specific targets based on highest error counts

### **During Implementation**
- [ ] Test each change immediately: `mypy target_file.py`
- [ ] Verify tool compatibility: `ruff check target_file.py`
- [ ] Measure incremental progress: Count errors before/after each change
- [ ] Document working patterns for team knowledge

### **After Each Improvement**
- [ ] Run full mypy check: `mypy mfg_pde/`
- [ ] Test pre-commit hooks: `pre-commit run --files target_file.py`
- [ ] Update progress documentation
- [ ] Clean up any newly-unused type ignores

### **Phase Completion**
- [ ] Comprehensive error count measurement
- [ ] Document successful techniques and ROI
- [ ] Plan next phase targets
- [ ] Create team knowledge sharing materials

---

## 📊 **Phase 1 Results Validation**

```bash
# Measurement commands
echo "Baseline: 414 errors"
mypy mfg_pde/ --show-error-codes | grep "Found.*errors"
echo "Reduction: $((414 - current_count)) errors"
echo "Improvement: $(echo "scale=1; (414-current_count)*100/414" | bc)%"
```

**Proven Results**:
- **Starting baseline**: 414 errors
- **Phase 1 completion**: 359 errors
- **Total reduction**: 55 errors (13.3% improvement)
- **Techniques used**: 4 proven strategies
- **Production compatibility**: 100% (all tools pass)

---

## 🚀 **Phase 2 Strategy Planning**

### **Next High-Impact Targets**
Based on remaining error distribution:

1. **Assignment type errors** (~80 remaining)
   - Target: Strategic `Any` → specific types
   - Estimated impact: -20 to -30 errors

2. **Attr-defined errors** (~40 remaining)
   - Target: Missing method stubs for additional libraries
   - Estimated impact: -15 to -20 errors

3. **Import-not-found errors** (~60 remaining)
   - Target: Additional selective library stubs
   - Estimated impact: -10 to -15 errors

**Phase 2 Target**: Additional 45-65 error reduction (12-18% improvement)

---

## 📚 **Team Knowledge & Best Practices**

### **Sharing with Team**
- [ ] Document in team wiki/confluence
- [ ] Create coding standards update
- [ ] Share in team meeting with examples
- [ ] Add to onboarding materials

### **Continuous Improvement**
- [ ] Monitor error count trends over time
- [ ] Share new patterns discovered during development
- [ ] Update strategies based on tool evolution (mypy, ruff updates)
- [ ] Measure impact on development velocity

---

**Summary**: These strategies provide a proven, production-ready approach to systematic typing improvements that deliver measurable benefits while maintaining full toolchain compatibility.

---

*Document created from Phase 1 implementation results*
*All patterns tested in production environment with MFG_PDE codebase*
