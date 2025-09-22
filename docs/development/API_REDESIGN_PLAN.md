# MFG_PDE API Redesign Plan: Progressive Disclosure with Hooks

**Status**: 🎉 **PHASE 4 COMPLETED** - Ready for Phase 5 (Documentation)
**Date**: 2025-09-20
**Last Updated**: 2025-09-20
**Goal**: Balance type safety for maintainers with simplicity for research users

**🏆 PROGRESS SUMMARY:**
- ✅ **Phase 1**: Foundation infrastructure complete
- ✅ **Phase 2**: Core object redesign complete
- ✅ **Phase 3**: Simple facade API complete
- ✅ **Phase 4**: Advanced features complete
- 🎯 **Phase 5**: Documentation and migration (next)

**🚀 MAJOR MILESTONE ACHIEVED:**
The complete three-layer API architecture is now fully implemented with:
- **Layer 1 (90% users)**: Dead-simple `solve_mfg()` interface
- **Layer 2 (9% users)**: Clean object-oriented `FixedPointSolver` with hooks
- **Layer 3 (1% users)**: Powerful extension system with 20+ specialized hooks

---

## 📋 **Executive Summary**

This document outlines a comprehensive API redesign for MFG_PDE that addresses the core tension between **type safety** (needed for maintainable code) and **usability** (needed for research productivity).

**Key Innovation**: Instead of hiding complexity or exposing it all, we **structure access to complexity** through progressive disclosure and a sophisticated hooks system.

### **Target User Profiles**
- **90% Basic Users**: Want simple, working solutions with minimal cognitive load
- **8% Advanced Users**: Need customization power but clean interfaces
- **2% Expert Users**: Require full access to internal types and algorithms

---

## 🏗️ **Architectural Strategy**

### **Core Principle: "Clean Public Interface with Structured Escape Hatches"**

Rather than the traditional choice between simple APIs (limiting) or complex APIs (overwhelming), we provide:

1. **Clean public methods** with simple type signatures
2. **Documented internal APIs** for advanced customization
3. **Structured hooks system** for deep algorithm control
4. **Progressive type disclosure** as users need more power

### **Physical Module Structure**

```
mfg_pde/
├── __init__.py              # 90% users - dead simple facade
├── core.py                  # 8% users - clean object interfaces
├── hooks.py                 # 2% users - advanced customization
├── types.py                 # Advanced users - full type system
├── presets.py               # Common configurations
├── _internal/               # Maintainers only
│   ├── solvers/            # Complex solver implementations
│   ├── algorithms/         # Low-level mathematical routines
│   └── type_definitions/   # All the Union[A,B,C] complexity
└── contrib/                # Experimental features
```

---

## 🎯 **Three-Layer API Design**

### **Layer 1: Simple Facade (90% of users)**
```python
# mfg_pde/__init__.py
from .simple import solve_mfg, create_problem, load_example

# One-line solutions for common problems
result = solve_mfg("crowd_dynamics", domain_size=5.0, time_horizon=2.0)
result = solve_mfg("portfolio_optimization", fast=True)
result = solve_mfg("traffic_flow", accurate=True, tolerance=1e-10)
```

**Design Principles:**
- **String-based configuration** (no complex types visible)
- **Smart defaults** based on problem analysis
- **Preset configurations** for common scenarios
- **Zero import complexity** - everything from main module

### **Layer 2: Clean Object Interface (8% of users)**
```python
# mfg_pde/core.py
from mfg_pde.core import MFGProblem, FixedPointSolver, MFGResult

# Clean, memorable interfaces
problem = MFGProblem.from_hamiltonian(my_hamiltonian, domain=(0,1))
solver = FixedPointSolver(tolerance=1e-6, max_iterations=200)
result = solver.solve(problem)

# Rich result objects
result.plot_solution()
result.convergence_history
result.export_matlab("results.mat")
```

**Design Principles:**
- **3-4 core types maximum** (MFGProblem, MFGSolver, MFGResult, SolverConfig)
- **Method chaining support** for fluent APIs
- **Rich result objects** with built-in analysis
- **No Union types** in public signatures

### **Layer 3: Hooks and Advanced Types (2% of users)**
```python
# mfg_pde/hooks.py + mfg_pde/types.py
from mfg_pde.hooks import SolverHooks
from mfg_pde.types import SpatialTemporalState, HamiltonianFunction

class DebugHooks(SolverHooks):
    def on_iteration_end(self, state: SpatialTemporalState):
        print(f"Iteration {state.iteration}: residual={state.residual}")
        self.plot_intermediate_solution(state.u)

solver.solve(problem, hooks=DebugHooks())
```

**Design Principles:**
- **Full type information available** for customization
- **Documented internal APIs** with stability guarantees
- **Powerful hooks system** for algorithm control
- **Explicit advanced import paths**

---

## 🔧 **The Hooks Pattern: Core Innovation**

### **Problem Statement**
Research users need to:
- **Inspect intermediate algorithm states** for understanding
- **Customize specific algorithmic steps** for experimentation
- **Add logging, plotting, and analysis** during solving
- **Control solver behavior** based on runtime conditions

Traditional approaches fail:
- **Multiple callbacks**: `solve(problem, on_start=f1, on_step=f2, on_end=f3, ...)` → parameter explosion
- **Inheritance**: Forces users to understand entire solver class
- **Configuration files**: Not flexible enough for research needs

### **Hooks Solution**
```python
class SolverHooks:
    """Base class for solver customization."""

    def on_solve_start(self, initial_state: SpatialTemporalState) -> None:
        """Called before solving begins."""
        pass

    def on_iteration_start(self, state: SpatialTemporalState) -> None:
        """Called at the start of each iteration."""
        pass

    def on_iteration_end(self, state: SpatialTemporalState) -> Optional[str]:
        """Called after each iteration. Return value controls flow."""
        pass

    def on_convergence_check(self, state: SpatialTemporalState) -> Optional[bool]:
        """Called during convergence checking. Return True to force convergence."""
        pass

    def on_solve_end(self, result: MFGResult) -> None:
        """Called just before returning final result."""
        pass
```

### **Benefits of Hooks Pattern**
1. **Clean signatures**: `solve(problem, hooks=None)` never changes
2. **Logical grouping**: Related functionality in one class
3. **Composable**: Multiple hooks can be combined
4. **Type-safe**: Full access to internal types when needed
5. **Proven pattern**: Used successfully by PyTorch Lightning, Keras, etc.

---

## 📚 **Documentation Strategy**

### **Tiered Documentation Structure**

**🟢 Tier 1: Quick Start (90% of docs) - Simple API Only**
```markdown
# Getting Started with MFG_PDE

## Solve Your First Problem
```python
from mfg_pde import solve_mfg
result = solve_mfg("crowd_dynamics")
result.plot()
```

## Common Problems
- Crowd dynamics
- Portfolio optimization
- Traffic flow
- Custom problems
```

**🟡 Tier 2: User Guide (8% of docs) - Core Objects**
```markdown
# Advanced Usage

## Working with Core Objects
```python
from mfg_pde.core import MFGProblem, FixedPointSolver

problem = MFGProblem.from_lagrangian(my_lagrangian)
solver = FixedPointSolver(config)
result = solver.solve(problem)
```

## Customization and Configuration
## Result Analysis and Visualization
```

**🔴 Tier 3: Developer Reference (2% of docs) - Full Type System**
```markdown
# Internal APIs and Types

⚠️ **Warning**: These APIs are for advanced users and may change between versions.

## Hooks System
## Internal Type Reference
## Extension Points
## Contributing to Core Algorithms
```

### **Documentation Rules**
1. **Never show complex types in Tier 1** - users should succeed without seeing them
2. **Introduce types gradually in Tier 2** - only the 3-4 core types
3. **Full type reference only in Tier 3** - complete but clearly marked as advanced
4. **Examples first, reference last** - working code before type signatures

---

## 🎨 **User Experience Examples**

### **Example 1: Beginner Research User**
```python
# Goal: "I just want to solve a crowd dynamics problem"
from mfg_pde import solve_mfg

result = solve_mfg("crowd_dynamics",
                   domain_size=10.0,
                   time_horizon=2.0,
                   crowd_size=1000)

result.plot_density_evolution()
result.save("my_results.h5")
```

**UX Notes:**
- Zero imports of complex types
- String-based problem specification
- Immediate visualization
- One-line save functionality

### **Example 2: Intermediate Research User**
```python
# Goal: "I want to customize the algorithm parameters"
from mfg_pde.core import MFGProblem, FixedPointSolver

problem = MFGProblem.crowd_dynamics(domain=(0, 10), initial_density="gaussian")
solver = FixedPointSolver(method="semi_lagrangian", tolerance=1e-8, max_iterations=500)

result = solver.solve(problem)
print(f"Converged in {result.iterations} iterations")
result.convergence_history.plot()
```

**UX Notes:**
- Clean object interfaces
- Method chaining possible
- Rich result analysis
- No complex types in user code

### **Example 3: Advanced Research User**
```python
# Goal: "I want to implement a custom algorithm variant"
from mfg_pde.core import FixedPointSolver
from mfg_pde.hooks import SolverHooks
from mfg_pde.types import SpatialTemporalState
import matplotlib.pyplot as plt

class CustomAlgorithmHooks(SolverHooks):
    def __init__(self):
        self.residual_history = []

    def on_iteration_end(self, state: SpatialTemporalState) -> Optional[str]:
        # Custom convergence criterion
        self.residual_history.append(state.residual)

        if len(self.residual_history) > 5:
            recent_improvement = self.residual_history[-5] / self.residual_history[-1]
            if recent_improvement < 1.1:  # Less than 10% improvement
                print("Stopping due to slow convergence")
                return "stop"

        # Custom visualization every 10 iterations
        if state.iteration % 10 == 0:
            plt.figure(figsize=(12, 4))
            plt.subplot(131)
            plt.imshow(state.u, aspect='auto')
            plt.title(f"Value Function (iter {state.iteration})")
            plt.subplot(132)
            plt.imshow(state.m, aspect='auto')
            plt.title(f"Density (iter {state.iteration})")
            plt.subplot(133)
            plt.plot(self.residual_history)
            plt.title("Convergence History")
            plt.yscale('log')
            plt.show()

        return None

# Use the custom hooks
solver = FixedPointSolver()
result = solver.solve(problem, hooks=CustomAlgorithmHooks())
```

**UX Notes:**
- Full access to internal state
- Complete control over algorithm flow
- Type-safe customization
- Professional research capabilities

---

## 🚀 **Implementation Roadmap**

### **Phase 1: Foundation (Weeks 1-2)**
**Goal**: Establish core infrastructure

**🔧 Tasks:**
1. **Create new module structure**
   ```bash
   mkdir mfg_pde/{hooks,types,_internal}
   touch mfg_pde/{hooks/__init__.py,types/__init__.py}
   ```

2. **Implement base hooks system**
   ```python
   # mfg_pde/hooks/base.py
   class SolverHooks:
       def on_solve_start(self, state): pass
       def on_iteration_end(self, state): pass
       # ... other hook points
   ```

3. **Define core type protocols**
   ```python
   # mfg_pde/types/protocols.py
   @runtime_checkable
   class MFGProblem(Protocol):
       def get_domain_bounds(self) -> Tuple[float, float]: ...
   ```

4. **Move existing complex types to _internal**
   ```python
   # mfg_pde/_internal/type_definitions.py
   ComplexInternalType: TypeAlias = Union[A, B, C, ...]
   ```

**✅ Success Criteria:** ✅ **COMPLETED 2025-09-20**
- [x] New module structure in place (`mfg_pde/hooks/`, `mfg_pde/types/`)
- [x] Base hooks system functional (`SolverHooks` with all hook points)
- [x] Core protocols defined (`MFGProblem`, `MFGResult` protocols)
- [x] No breaking changes to existing API

### **Phase 2: Core Object Redesign (Weeks 3-4)**
**Goal**: Implement clean public interfaces

**🔧 Tasks:**
1. **Redesign FixedPointSolver with hooks**
   ```python
   class FixedPointSolver:
       def solve(self, problem: MFGProblem, hooks: Optional[SolverHooks] = None):
           # Clean interface with hooks support
   ```

2. **Create MFGResult with rich analysis**
   ```python
   class MFGResult:
       def plot_solution(self): ...
       def convergence_history(self): ...
       def export_matlab(self, filename): ...
   ```

3. **Implement SolverConfig builder pattern**
   ```python
   class SolverConfig:
       @classmethod
       def for_high_accuracy(cls): ...
       @classmethod
       def for_fast_prototyping(cls): ...
   ```

**✅ Success Criteria:** ✅ **COMPLETED 2025-09-20**
- [x] FixedPointSolver uses hooks pattern (full integration with all hook points)
- [x] MFGResult has rich analysis methods (`plot_solution()`, `export_data()`, convergence info)
- [x] Configuration builders work (`SolverConfig` with fluent API `.with_tolerance()`, etc.)
- [x] All existing functionality preserved

### **Phase 3: Simple Facade (Weeks 5-6)**
**Goal**: Create dead-simple public API

**🔧 Tasks:**
1. **Implement mfg_pde/__init__.py facade**
   ```python
   def solve_mfg(problem_type: str, **kwargs) -> MFGResult:
       # Smart problem detection and solver selection
   ```

2. **Create preset configurations**
   ```python
   PRESETS = {
       "research_prototype": {...},
       "production_quality": {...},
       "high_performance": {...}
   }
   ```

3. **Add smart defaults system**
   ```python
   def auto_configure_solver(problem, performance_target):
       # Analyze problem and choose optimal settings
   ```

**✅ Success Criteria:** ✅ **COMPLETED 2025-09-20**
- [x] One-line solve functions work (`solve_mfg()`, `solve_mfg_auto()`, `solve_mfg_smart()`)
- [x] Preset configurations cover common cases (6 generic + 4 problem-specific presets)
- [x] Smart defaults produce good results (intelligent `auto_config()` with parameter analysis)
- [x] Documentation shows simple examples first (comprehensive examples in docstrings)

**🎯 ADDITIONAL FEATURES IMPLEMENTED:**
- [x] Parameter validation system (`validate_problem_parameters()`)
- [x] Configuration recommendations (`get_config_recommendation()`)
- [x] Problem discovery API (`get_available_problems()`, `suggest_problem_setup()`)
- [x] Smart problem-specific auto-tuning based on parameter analysis

### **Phase 4: Advanced Features (Weeks 7-8)**
**Goal**: Add power-user capabilities

**🔧 Tasks:**
1. **Implement hook composition**
   ```python
   class MultiHook(SolverHooks):
       def __init__(self, *hooks): ...
   ```

2. **Add control flow in hooks**
   ```python
   def on_iteration_end(self, state) -> Optional[str]:
       return "stop" | "restart" | None
   ```

3. **Create debugging and visualization hooks**
   ```python
   class DebuggingHooks(SolverHooks): ...
   class PlottingHooks(SolverHooks): ...
   ```

4. **Implement extension points for algorithms**
   ```python
   class CustomizableSolver(FixedPointSolver):
       def customize_iteration_step(self, custom_func): ...
   ```

**✅ Success Criteria:** ✅ **COMPLETED 2025-09-20**
- [x] Hook composition works (MultiHook, ConditionalHook, PriorityHook, FilterHook, TransformHook, ChainHook)
- [x] Control flow mechanisms functional (AdaptiveControlHook, PerformanceControlHook, WatchdogHook, ConditionalStopHook)
- [x] Rich debugging capabilities (DebugHook, PerformanceHook, ConvergenceAnalysisHook, StateInspectionHook)
- [x] Algorithm customization possible (Extension hooks for HJB, FP, convergence, pre/post-processing)

**🎯 ADDITIONAL FEATURES IMPLEMENTED:**
- [x] Advanced hook composition (FilterHook, TransformHook, ChainHook, PriorityHook)
- [x] Sophisticated control flow (adaptive control, performance monitoring, watchdog protection)
- [x] Comprehensive debugging suite (debug levels, performance tracking, convergence analysis, state inspection)
- [x] Rich visualization capabilities (real-time plotting, animations, progress bars, structured logging)
- [x] Algorithm extension points (custom HJB/FP solvers, preprocessing/postprocessing, adaptive parameters)

### **Phase 5: Documentation and Migration (Weeks 9-10)**
**Goal**: Complete transition to new API

**🔧 Tasks:**
1. **Write tiered documentation**
   - Tier 1: Simple API examples
   - Tier 2: Core object usage
   - Tier 3: Advanced hooks and types

2. **Create migration guide**
   ```markdown
   # Migrating from Old API to New API
   ## Simple cases
   ## Advanced cases
   ## Breaking changes
   ```

3. **Add deprecation warnings**
   ```python
   @deprecated("Use mfg_pde.solve_mfg() instead")
   def old_function(): ...
   ```

4. **Update all examples and tutorials**

**✅ Success Criteria:**
- [ ] Complete documentation set
- [ ] Migration guide helps users transition
- [ ] All examples use new API
- [ ] Deprecation path is clear

---

## ⚡ **Quick Wins and Validation**

### **Immediate Validation Steps**
1. **Write the simple facade first** - can it handle 90% of current examples?
2. **Create one hooks example** - does it feel natural for research users?
3. **Test the type import experience** - is the progression from simple to complex smooth?

### **Early User Testing**
- **Beginner test**: Give someone the simple API with no explanation
- **Intermediate test**: Can they customize without reading docs?
- **Expert test**: Can they implement a custom algorithm variant?

### **Success Metrics**
- **Time to first result**: < 5 minutes for simple problems
- **Lines of code**: 90% of use cases in < 10 lines
- **Learning curve**: Natural progression from simple to advanced
- **No regression**: All existing functionality preserved

---

## 🎯 **Design Principles Summary**

1. **Progressive Disclosure**: Complexity available but not forced
2. **Structured Access**: Hooks and protocols, not free-form APIs
3. **Type Safety**: Full typing for maintainers, hidden from casual users
4. **Physical Separation**: Module structure enforces design
5. **Proven Patterns**: Hooks pattern used by major libraries
6. **Research-Friendly**: Power users get full algorithm control
7. **Forward Compatible**: Easy to add features without breaking changes

This design balances the needs of all user types while maintaining a clean, maintainable codebase for the development team.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Create comprehensive API design plan documentation", "status": "completed", "activeForm": "Creating API design documentation"}, {"content": "Document layered type system strategy", "status": "completed", "activeForm": "Documenting type strategy"}, {"content": "Document hooks pattern implementation", "status": "in_progress", "activeForm": "Documenting hooks pattern"}, {"content": "Create implementation roadmap", "status": "pending", "activeForm": "Creating implementation roadmap"}]