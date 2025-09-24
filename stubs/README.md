# MFG_PDE Strategic Type Stubs

This directory contains minimal, production-compatible type stubs for complex libraries that cause false positive mypy errors.

## 🎯 **Strategic Philosophy**

**Goal**: Eliminate false positives (mypy wrong) while maintaining production compatibility
**Approach**: Minimal coverage focused on actual usage patterns in MFG_PDE codebase
**Priority**: Production health over perfect typing

## 📁 **Current Stubs**

### **polars/** - Data Analysis Library
- **Purpose**: Eliminate DataFrame/Expr attribute errors
- **Coverage**: ~90% of MFG_PDE polars usage patterns
- **Impact**: High - Used in experiment analysis and data processing

### **bokeh/** - Visualization Library
- **Purpose**: Handle dynamic plotting attribute access
- **Coverage**: ColorBar and common plotting components
- **Impact**: Medium - Used in advanced visualizations

### **networkx/** - Graph Analysis Library
- **Purpose**: Network MFG problem graph operations
- **Coverage**: Core Graph/DiGraph classes and algorithms
- **Impact**: Medium - Used in network MFG solvers

### **memory_profiler/** - Performance Monitoring
- **Purpose**: Memory usage analysis decorators
- **Coverage**: profile() and memory_usage() functions
- **Impact**: Low - Used in benchmarking and profiling

### **jax/** - Accelerated Computing Library
- **Purpose**: JAX transformation functions and Array type compatibility
- **Coverage**: jit, grad, vmap, Array.at, while_loop, jax.numpy
- **Impact**: High - Used in JAXMFGSolver and accelerated computations

### **omegaconf/** - Configuration Management
- **Purpose**: Complex configuration objects and merging
- **Coverage**: DictConfig, ListConfig, create, load, save, merge
- **Impact**: Medium - Used in advanced configuration management

### **attrs/attr/** - Dataclass Decorators
- **Purpose**: Advanced dataclass functionality and validation
- **Coverage**: define, frozen, field, converters, validators, asdict
- **Impact**: Medium - Used in structured data classes

## 🔧 **Adding New Stubs**

### **Strategic Criteria**
Only add stubs for libraries that:
1. **Cause 5+ false positive errors** in MFG_PDE codebase
2. **Have complex/dynamic APIs** that mypy can't analyze
3. **Are used in multiple modules** (not one-off imports)
4. **Break production toolchain** (ruff, pre-commit issues)

### **Implementation Pattern**
```python
# stubs/library_name/__init__.pyi
from typing import Any

# Core classes with methods actually used
class MainClass:
    def used_method(self, arg: str) -> Any: ...

# Functions actually used
def used_function(**kwargs: Any) -> Any: ...

# Strategic catch-all
def __getattr__(name: str) -> Any: ...
```

### **Testing Process**
1. **Identify errors**: `mypy module_using_library.py | grep attr-defined`
2. **Add minimal methods**: Only cover actual usage patterns
3. **Test compatibility**: Ensure ruff, pre-commit, mypy pass
4. **Measure impact**: Count error reduction before/after
5. **Document**: Update this README with results

## 📈 **Stub Effectiveness**

| Library | Errors Before | Errors After | Impact |
|---------|---------------|--------------|--------|
| polars  | ~80          | ~15         | 81% reduction |
| bokeh   | ~10          | ~2          | 80% reduction |
| networkx| ~5           | ~1          | 80% reduction |
| memory_profiler | ~3   | ~0          | 100% reduction |

## 🚀 **Future Expansion**

### **Planned Stubs** (when needed)
- **JAX**: If TYPE_CHECKING isolation isn't sufficient
- **OmegaConf**: If configuration typing becomes problematic
- **Pydantic**: If validation typing causes issues
- **Matplotlib**: If pyplot dynamic attributes cause problems

### **Strategic Decision Framework**
- **High Priority**: Libraries with 10+ attr-defined errors
- **Medium Priority**: Libraries breaking toolchain compatibility
- **Low Priority**: Libraries with occasional import-not-found errors
- **Never**: Libraries with good official stubs (numpy, scipy, etc.)

---

**Key Principle**: These stubs transform typing from friction into value by eliminating false positives while maintaining the benefits of static analysis where it actually helps.
