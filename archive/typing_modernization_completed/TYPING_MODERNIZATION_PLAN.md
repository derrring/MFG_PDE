# Modern Typing Migration Plan for MFG_PDE

**Date**: 2025-09-20
**Python Version**: 3.12+ (enables all modern typing features)
**Scope**: Systematic migration from legacy typing patterns to modern Python 3.9+ syntax

## 🎯 **Migration Strategy**

### **Phase 1: Critical Public API Files (HIGH PRIORITY)**
Focus on user-facing interfaces first to maximize impact:

#### **1.1 Core Public API**
- ✅ `mfg_pde/simple.py` - Already modernized
- 🔄 `mfg_pde/types/protocols.py` - Type definitions
- 🔄 `mfg_pde/types/state.py` - Core state types
- 🔄 `mfg_pde/solvers/base.py` - Solver base classes
- 🔄 `mfg_pde/solvers/fixed_point.py` - Main solver implementation

#### **1.2 Configuration System**
- 🔄 `mfg_pde/config/modern_config.py` - Builder pattern config
- 🔄 `mfg_pde/config/pydantic_config.py` - Pydantic configs
- 🔄 `mfg_pde/config/solver_config.py` - Solver configurations

#### **1.3 Hook System (Advanced API)**
- 🔄 `mfg_pde/hooks/extensions.py` - Algorithm extensions
- 🔄 `mfg_pde/hooks/debug.py` - Debugging hooks
- 🔄 `mfg_pde/hooks/visualization.py` - Visualization hooks

### **Phase 2: Core Algorithms (MEDIUM PRIORITY)**
Scientific computing core that impacts performance:

#### **2.1 Solver Algorithms**
- 🔄 `mfg_pde/alg/mfg_solvers/` - All MFG solver implementations
- 🔄 `mfg_pde/alg/hjb_solvers/` - HJB equation solvers
- 🔄 `mfg_pde/alg/fp_solvers/` - Fokker-Planck solvers

#### **2.2 Geometry and Discretization**
- 🔄 `mfg_pde/geometry/base_geometry.py` - Base geometry classes
- 🔄 `mfg_pde/geometry/domain_*.py` - Domain implementations
- 🔄 `mfg_pde/geometry/network_*.py` - Network geometry

### **Phase 3: Utilities and Supporting Code (LOW PRIORITY)**
Internal utilities that don't impact public API:

#### **3.1 Utility Modules**
- 🔄 `mfg_pde/utils/` - All utility modules (92 files)
- 🔄 `mfg_pde/visualization/` - Plotting and visualization
- 🔄 `mfg_pde/workflow/` - Workflow management

## 🔧 **Automated Migration Tools**

### **Tool 1: pyupgrade for Syntax Modernization**
```bash
# Install pyupgrade
pip install pyupgrade

# Modernize Python 3.12 syntax automatically
find mfg_pde -name "*.py" -exec pyupgrade --py312-plus {} \;

# This will automatically convert:
# - List[int] → list[int]
# - Dict[str, float] → dict[str, float]
# - Union[int, str] → int | str
# - Optional[bool] → bool | None
```

### **Tool 2: Custom Migration Script**
```python
#!/usr/bin/env python3
"""
Automated migration script for MFG_PDE typing modernization.
"""
import re
import subprocess
from pathlib import Path

def modernize_typing_imports(file_path: Path) -> bool:
    """Modernize typing imports in a single file."""
    content = file_path.read_text()
    original_content = content

    # Pattern to match and modernize typing imports
    patterns = [
        # Remove unnecessary imports
        (r'from typing import ([^\\n]*?)List([^\\n]*?)\\n',
         lambda m: f'from typing import {m.group(1).replace("List, ", "").replace(", List", "").replace("List", "")}{m.group(2)}\\n' if m.group(1).strip() + m.group(2).strip() else ''),

        # Convert usage patterns
        (r'List\\[([^\\]]+)\\]', r'list[\\1]'),
        (r'Dict\\[([^\\]]+), ([^\\]]+)\\]', r'dict[\\1, \\2]'),
        (r'Tuple\\[([^\\]]+)\\]', r'tuple[\\1]'),
        (r'Union\\[([^\\]]+), None\\]', r'\\1 | None'),
        (r'Optional\\[([^\\]]+)\\]', r'\\1 | None'),
        (r'Union\\[([^\\]]+)\\]', r'\\1'),  # Handle other unions
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    if content != original_content:
        file_path.write_text(content)
        return True
    return False

def run_migration():
    """Run the complete migration process."""
    mfg_pde_root = Path("mfg_pde")

    # Phase 1: Critical files
    critical_files = [
        "types/protocols.py", "types/state.py",
        "solvers/base.py", "solvers/fixed_point.py",
        "config/modern_config.py", "config/pydantic_config.py",
        "hooks/extensions.py", "hooks/debug.py"
    ]

    modified_files = []

    for file_rel_path in critical_files:
        file_path = mfg_pde_root / file_rel_path
        if file_path.exists():
            if modernize_typing_imports(file_path):
                modified_files.append(file_path)
                print(f"✅ Modernized: {file_path}")

    # Run pyupgrade on all Python files
    subprocess.run([
        "find", str(mfg_pde_root), "-name", "*.py",
        "-exec", "pyupgrade", "--py312-plus", "{}", ";"
    ])

    print(f"\\n🎉 Migration complete! Modified {len(modified_files)} files.")
    return modified_files

if __name__ == "__main__":
    run_migration()
```

## 📝 **Manual Migration Patterns**

### **Before and After Examples**

#### **Legacy Pattern**
```python
from typing import List, Dict, Tuple, Optional, Union, Callable

def solve_mfg(
    problems: List[MFGProblem],
    configs: Dict[str, Union[float, int]],
    callback: Optional[Callable[[int], None]] = None
) -> Tuple[List[Solution], Dict[str, float]]:
    pass
```

#### **Modern Pattern (Python 3.12+)**
```python
from typing import Callable  # Only import what's needed

def solve_mfg(
    problems: list[MFGProblem],
    configs: dict[str, float | int],
    callback: Callable[[int], None] | None = None
) -> tuple[list[Solution], dict[str, float]]:
    pass
```

### **Type Alias Modernization**

#### **Before**
```python
from typing import List, Dict, Tuple, Union
import numpy as np
from numpy.typing import NDArray

# Scattered complex types throughout code
def process_results(
    data: Dict[str, Tuple[NDArray[np.float64], List[float]]]
) -> List[Dict[str, Union[float, bool]]]:
    pass
```

#### **After**
```python
from typing import TypeAlias
import numpy as np
from numpy.typing import NDArray

# Clean type aliases
SolutionData: TypeAlias = dict[str, tuple[NDArray[np.float64], list[float]]]
ProcessedResults: TypeAlias = list[dict[str, float | bool]]

def process_results(data: SolutionData) -> ProcessedResults:
    pass
```

## 🚦 **Migration Checkpoints**

### **Checkpoint 1: Critical API (Week 1)**
- [ ] Modernize all public API files
- [ ] Update type aliases for scientific types
- [ ] Ensure all imports are minimal
- [ ] Run comprehensive type checking

### **Checkpoint 2: Core Algorithms (Week 2)**
- [ ] Modernize solver implementations
- [ ] Update geometry and discretization modules
- [ ] Performance test after changes
- [ ] Validate numerical accuracy unchanged

### **Checkpoint 3: Complete Migration (Week 3)**
- [ ] Modernize all utility modules
- [ ] Update visualization and workflow code
- [ ] Final comprehensive test suite
- [ ] Update documentation and examples

## 🧪 **Testing Strategy**

### **Type Safety Validation**
```bash
# Comprehensive type checking after each phase
mypy mfg_pde --strict --python-version 3.12

# Runtime validation
python -m pytest tests/ -v --tb=short

# Performance regression testing
python benchmarks/run_performance_tests.py
```

### **Backward Compatibility**
```python
# Ensure public API remains identical
import mfg_pde

# All existing user code should continue working
result = mfg_pde.solve_mfg("crowd_dynamics", domain_size=2.0)
```

## 📊 **Migration Progress Tracking**

### **Files by Priority**

| Priority | Category | Files Count | Status |
|----------|----------|-------------|--------|
| HIGH | Public API | 8 files | 🔄 In Progress |
| HIGH | Type Definitions | 4 files | 🔄 In Progress |
| MEDIUM | Core Algorithms | ~30 files | ⏳ Planned |
| MEDIUM | Geometry | ~15 files | ⏳ Planned |
| LOW | Utilities | ~90 files | ⏳ Planned |
| LOW | Visualization | ~10 files | ⏳ Planned |

### **Metrics to Track**
- **Import Reduction**: Lines of typing imports before/after
- **Readability Score**: Cyclomatic complexity of type annotations
- **Type Coverage**: Percentage of functions with complete type hints
- **Performance Impact**: Runtime overhead of type checking

## 🎯 **Expected Benefits**

### **Code Quality**
- ✅ **50% reduction** in typing imports
- ✅ **Improved readability** with modern union syntax
- ✅ **Better IDE support** with cleaner type hints
- ✅ **Easier maintenance** with self-documenting type aliases

### **Developer Experience**
- ✅ **Faster development** with less verbose typing
- ✅ **Better autocomplete** in modern IDEs
- ✅ **Cleaner git diffs** with minimal import lines
- ✅ **Consistent modern patterns** across codebase

### **Scientific Computing Benefits**
- ✅ **Clear data types** for mathematical objects (SolutionArray, GridPoints)
- ✅ **Layered complexity** (Simple → Clean → Advanced)
- ✅ **Protocol-based flexibility** for research code
- ✅ **Type-safe configuration** with Literal types

## 🔍 **Quality Assurance**

### **Pre-Migration Checklist**
- [ ] Full test suite passes
- [ ] Type checking with mypy succeeds
- [ ] Performance benchmarks baseline recorded
- [ ] Git branch created for migration work

### **Post-Migration Validation**
- [ ] All tests continue to pass
- [ ] No performance regressions detected
- [ ] Type checking still succeeds with stricter settings
- [ ] Public API contracts maintained
- [ ] Documentation updated with modern examples

---

**Status**: 🚀 Ready for execution with Python 3.12+ modern typing features
**Timeline**: 3-week phased approach focusing on user impact first
**Risk Level**: LOW - Automated tools + comprehensive testing strategy