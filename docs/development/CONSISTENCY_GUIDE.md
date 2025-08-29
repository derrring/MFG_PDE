# MFG_PDE Comprehensive Consistency Guide

**Version:** 1.0  
**Date:** July 26, 2025  
**Purpose:** Master guide for maintaining consistency across all aspects of the MFG_PDE repository  

This document consolidates all consistency standards into a single authoritative reference, covering code, documentation, mathematical notation, naming conventions, testing patterns, and architectural decisions.

## Table of Contents

1. [Code Consistency Standards](#code-consistency-standards)
2. [Documentation Consistency](#documentation-consistency)
3. [Mathematical Notation Standards](#mathematical-notation-standards)
4. [API & Interface Consistency](#api--interface-consistency)
5. [Testing & Validation Consistency](#testing--validation-consistency)
6. [File Organization & Structure](#file-organization--structure)
7. [Configuration & Factory Patterns](#configuration--factory-patterns)
8. [Error Handling & Exceptions](#error-handling--exceptions)
9. [Import & Module Organization](#import--module-organization)
10. [Version Control & Git Practices](#version-control--git-practices)
11. [Consistency Maintenance Schedule](#consistency-maintenance-schedule)
12. [Automated Consistency Checks](#automated-consistency-checks)

---

## Code Consistency Standards

### 1. Class Naming Conventions

**✅ CURRENT STANDARD NAMES:**
```python
# HJB Solvers
HJBGFDMQPSolver              # NOT: HJBGFDMSmartQPSolver
HJBGFDMTunedQPSolver         # NOT: HJBGFDMTunedSmartQPSolver
HJBFDMSolver                 # Standard FDM solver

# Particle Solvers
SilentAdaptiveParticleCollocationSolver    # NOT: QuietAdaptiveParticleCollocationSolver
MonitoredParticleCollocationSolver         # NOT: EnhancedParticleCollocationSolver
ParticleCollocationSolver                  # Base solver

# Composite Solvers
FixedPointIterator                         # NOT: DampedFixedPointIterator in docs
ConfigAwareFixedPointIterator              # Modern configuration-aware version
```

**✅ BACKWARD COMPATIBILITY:**
- All old names must remain available with deprecation warnings
- New documentation should use only standard names
- Examples should demonstrate modern names with old names mentioned as "deprecated"

### 2. Parameter Naming Standards

**✅ NEWTON METHOD PARAMETERS:**
```python
# Modern standard names
max_newton_iterations    # NOT: NiterNewton, newton_max_iter
newton_tolerance        # NOT: l2errBoundNewton, newton_tol
newton_damping         # NOT: newton_damping_factor (if used)
```

**✅ PICARD METHOD PARAMETERS:**
```python
# Modern standard names
max_picard_iterations   # NOT: Niter_max, picard_max_iter
picard_tolerance       # NOT: l2errBoundPicard, picard_tol  
picard_damping_factor  # NOT: damping_factor, damping
convergence_tolerance  # NOT: l2errBound, conv_tol
```

**✅ SOLVER-SPECIFIC PARAMETERS:**
```python
# Particle methods
num_particles          # NOT: N_particles, particle_count
kde_bandwidth         # NOT: bandwidth, kde_bw

# GFDM methods  
delta                 # Neighborhood size - keep as is
taylor_order          # NOT: order, taylor_degree
weight_function       # NOT: kernel, weight_type
weight_scale          # NOT: scale, weight_factor

# QP methods
use_monotone_constraints    # NOT: enable_qp, qp_constraints
qp_solver                  # NOT: quadprog_solver
```

### 3. Method Signature Consistency

**✅ SOLVE METHOD SIGNATURE:**
```python
def solve(self, 
          max_picard_iterations: int = 20,
          picard_tolerance: float = 1e-3,
          convergence_tolerance: float = 1e-5,
          verbose: bool = False,
          **kwargs) -> Union[Tuple, SolverResult]:
    """Consistent solve method signature across all solvers"""
```

**✅ INITIALIZATION SIGNATURE:**
```python
def __init__(self,
             problem: MFGProblem,
             # Solver-specific required parameters first
             collocation_points: np.ndarray = None,  # If applicable
             # Then optional parameters in logical order
             max_newton_iterations: int = 30,
             newton_tolerance: float = 1e-6,
             **kwargs):
```

### 4. Return Value Consistency

**✅ STRUCTURED RESULTS (MODERN):**
```python
# New solvers should return structured results
from mfg_pde.utils.solver_result import SolverResult

result = SolverResult(
    solution=U,
    density=M,
    convergence_info=convergence_info,
    timing_info=timing_info,
    metadata=metadata
)
return result
```

**✅ TUPLE RETURNS (LEGACY COMPATIBILITY):**
```python
# Legacy solvers maintain tuple returns
return U, M, info  # Where info is a dict with convergence data
```

---

## Documentation Consistency

### 1. Code Examples in Documentation

**✅ MANDATORY PATTERNS:**
```python
# 1. Always show modern factory pattern first
from mfg_pde.factory import create_fast_solver
from mfg_pde.config import create_fast_config

config = create_fast_config()
solver = create_fast_solver(problem, "solver_type", config=config)
result = solver.solve()

# 2. Then show direct class usage as alternative
from mfg_pde.alg import SolverClassName

solver = SolverClassName(
    problem=problem,
    max_newton_iterations=30,     # Use modern parameter names
    newton_tolerance=1e-6
)
U, M, info = solver.solve()
```

**❌ PROHIBITED PATTERNS:**
```python
# DON'T use old class names in examples
from mfg_pde.alg import HJBGFDMSmartQPSolver  # OLD

# DON'T use old parameter names
solver.solve(NiterNewton=30)  # OLD

# DON'T show only direct usage without factory pattern
```

### 2. Mathematical Notation Standards

**✅ COORDINATE CONVENTIONS:**
- Functions: `u(t,x)`, `m(t,x)` - time first, space second
- Array indexing: `U[t_idx, x_idx]` - matches mathematical convention
- Documentation: Always specify `u(t,x)` not `u(x,t)`

**✅ VARIABLE NAMING:**
```python
# Mathematical functions (lowercase in math, arrays can be uppercase)
u(t,x)    # Value function (continuous)
m(t,x)    # Density function (continuous) 
U         # Discretized value function array
M         # Discretized density array

# Derivatives
∂u/∂t     # Temporal derivative
∂u/∂x     # Spatial derivative  
∇u        # Gradient (spatial)
```

**✅ EQUATION FORMATTING:**
```latex
# HJB Equation (use lowercase u)
-\frac{\partial u}{\partial t} + H(t,x,\nabla u, m) = 0

# Fokker-Planck Equation (use lowercase m) 
\frac{\partial m}{\partial t} - \nabla \cdot (m \nabla H_p) - \frac{\sigma^2}{2}\Delta m = 0
```

### 3. Documentation Structure Standards

**✅ README.md STRUCTURE:**
```markdown
# Title
Brief description

## Quick Start
### Modern Factory Pattern (Recommended)
### Direct Class Usage (Alternative)

## Features
## Installation  
## Documentation Links
## Core Solvers
## Performance
## Testing
## Examples
## Contributing
## License
```

**✅ MARKDOWN FORMATTING:**
- Use `### ` for subsections, not `####` in main README
- Code blocks always specify language: ````python`, `bash`, `json`
- Consistent emoji usage: ✅❌⚠ only for status indicators
- No decorative emojis (🚀🎉📊)

---

## Mathematical Notation Standards

### 1. LaTeX Configuration Consistency

**✅ STANDARD MATPLOTLIB LATEX CONFIG:**
```python
from matplotlib import rcParams

# Apply this configuration in ALL visualization modules
rcParams['text.usetex'] = False  # Set to True if LaTeX is installed
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['mathtext.fontset'] = 'cm'
rcParams['axes.formatter.use_mathtext'] = True
```

**✅ DOCSTRING LATEX SYNTAX:**
```python
def plot_hjb_analysis(self, U, x_grid, t_grid):
    """
    Plot HJB analysis with proper LaTeX escaping in docstrings.
    
    Mathematical expressions in docstrings require double backslashes:
    - Value function: $u(t,x)$
    - Spatial gradient: $\\frac{\\partial u}{\\partial x}$
    - HJB equation: $-\\frac{\\partial u}{\\partial t} + H = 0$
    """
```

**✅ PLOT TITLES AND LABELS:**
```python
# Use raw strings for plot titles to avoid escape issues
ax.set_title(r'Value Function $u(t,x)$')
ax.set_xlabel(r'Time $t$')
ax.set_ylabel(r'Space $x$')

# Complex expressions
title = r"HJB Analysis: $-\frac{\partial u}{\partial t} + H(t,x,\nabla u, m) = 0$"
```

### 2. Variable Naming Mathematical Consistency

**✅ MATHEMATICAL VARIABLES:**
```python
# Problem definition
T          # Final time (scalar)
sigma      # Diffusion coefficient  
x_grid     # Spatial discretization
t_grid     # Temporal discretization
X, T       # Meshgrid arrays (note order: spatial first in meshgrid)

# Solutions
U          # Value function array U[t_idx, x_idx]
M          # Density array M[t_idx, x_idx]
u_0        # Initial condition for value function
m_0        # Initial condition for density

# Derivatives and gradients
u_x        # Spatial derivative ∂u/∂x
u_t        # Temporal derivative ∂u/∂t
grad_u     # Spatial gradient ∇u
laplacian_u # Laplacian Δu
```

---

## API & Interface Consistency

### 1. Factory Pattern Standards

**✅ FACTORY FUNCTION SIGNATURES:**
```python
def create_fast_solver(problem: MFGProblem,
                      solver_type: str,
                      config: Optional[MFGSolverConfig] = None,
                      **kwargs) -> MFGSolver:
    """Standard factory function signature"""

def create_accurate_solver(problem: MFGProblem,
                          solver_type: str, 
                          config: Optional[MFGSolverConfig] = None,
                          **kwargs) -> MFGSolver:
    """High-accuracy configuration factory"""
```

**✅ SOLVER TYPE STRINGS:**
```python
# Standardized solver type identifiers
"particle_collocation"           # ParticleCollocationSolver
"adaptive_particle"             # SilentAdaptiveParticleCollocationSolver  
"monitored_particle"            # MonitoredParticleCollocationSolver
"hjb_gfdm_qp"                  # HJBGFDMQPSolver
"hjb_gfdm_tuned_qp"            # HJBGFDMTunedQPSolver
"fixed_point_iterator"         # FixedPointIterator
"config_aware_iterator"        # ConfigAwareFixedPointIterator
```

### 2. Configuration System Standards

**✅ CONFIGURATION CLASS HIERARCHY:**
```python
@dataclass
class NewtonConfig:
    max_iterations: int = 30
    tolerance: float = 1e-6
    damping: float = 1.0

@dataclass  
class PicardConfig:
    max_iterations: int = 20
    tolerance: float = 1e-3
    damping_factor: float = 0.5

@dataclass
class MFGSolverConfig:
    newton: NewtonConfig = field(default_factory=NewtonConfig)
    picard: PicardConfig = field(default_factory=PicardConfig)
    return_structured: bool = False
    verbose: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## Testing & Validation Consistency

### 1. Test File Organization

**✅ TEST DIRECTORY STRUCTURE:**
```
tests/
├── integration/           # End-to-end solver tests
├── unit/                 # Individual component tests  
├── mass_conservation/    # Mass conservation validation
├── method_comparisons/   # Comparative analysis
├── fixtures/            # Shared test data and utilities
└── regression/          # Regression test suite
```

**✅ TEST NAMING CONVENTIONS:**
```python
# Test class names
class TestHJBGFDMQPSolver:           # Test{ClassName}
class TestParticleCollocationSolver:
class TestConfigurationSystem:

# Test method names  
def test_solve_convergence(self):           # test_{functionality}
def test_mass_conservation_accuracy(self):
def test_factory_pattern_creation(self):
def test_backward_compatibility_warnings(self):
```

### 2. Validation Standards

**✅ CONVERGENCE VALIDATION:**
```python
def validate_convergence(result, expected_tolerance=1e-5):
    """Standard convergence validation pattern"""
    assert result.converged, "Solver failed to converge"
    assert result.final_error < expected_tolerance
    assert result.iterations > 0

def validate_mass_conservation(M, tolerance=1e-3):
    """Standard mass conservation check"""
    total_mass = np.trapz(M[-1, :], x_grid)
    mass_error = abs(total_mass - 1.0)
    assert mass_error < tolerance, f"Mass error {mass_error} exceeds {tolerance}"
```

**✅ PERFORMANCE VALIDATION:**
```python
def validate_performance_regression(timing_data, baseline_time, tolerance=1.5):
    """Ensure performance doesn't regress significantly"""
    assert timing_data['solve_time'] < baseline_time * tolerance
```

---

## File Organization & Structure

### 1. Module Organization Standards

**✅ PACKAGE STRUCTURE:**
```
mfg_pde/
├── __init__.py              # Main exports
├── core/                    # Core problem definitions
├── alg/                     # Algorithm implementations
│   ├── hjb_solvers/        # HJB-specific solvers
│   ├── fp_solvers/         # FP-specific solvers  
│   └── composite/          # Multi-component solvers
├── config/                 # Configuration system
├── factory/               # Factory patterns
├── utils/                 # Utilities and tools
│   ├── validation.py      # Validation utilities
│   ├── exceptions.py      # Exception definitions
│   ├── solver_result.py   # Result structures
│   └── mathematical_visualization.py
└── visualization/         # Future: separate visualization package
```

**✅ FILE NAMING CONVENTIONS:**
```python
# Solver files
hjb_gfdm_qp.py                    # NOT: hjb_gfdm_smart_qp.py
particle_collocation_solver.py   # NOT: enhanced_particle_collocation.py

# Utility files
mathematical_visualization.py    # NOT: math_viz.py
solver_decorators.py            # NOT: decorators.py (too generic)
```

### 2. Import Organization in Files

**✅ STANDARD IMPORT ORDER:**
```python
#!/usr/bin/env python3
"""Module docstring"""

# 1. Standard library imports
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import warnings

# 2. Third-party imports (with try/except for optional)
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# 3. Internal package imports (absolute preferred)
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.utils.validation import validate_solution_array
from .logging import get_logger  # Relative only within same package
```

---

## Configuration & Factory Patterns

### 1. Configuration Consistency

**✅ CONFIGURATION CREATION PATTERNS:**
```python
# Factory functions for common configurations
def create_fast_config() -> MFGSolverConfig:
    """Fast, reasonable accuracy configuration"""
    return MFGSolverConfig(
        newton=NewtonConfig(max_iterations=15, tolerance=1e-4),
        picard=PicardConfig(max_iterations=10, tolerance=1e-3)
    )

def create_accurate_config() -> MFGSolverConfig:
    """High accuracy configuration"""
    return MFGSolverConfig(
        newton=NewtonConfig(max_iterations=50, tolerance=1e-8),
        picard=PicardConfig(max_iterations=30, tolerance=1e-6)
    )

def create_research_config() -> MFGSolverConfig:
    """Research-grade configuration with full output"""
    return MFGSolverConfig(
        newton=NewtonConfig(max_iterations=100, tolerance=1e-10),
        picard=PicardConfig(max_iterations=50, tolerance=1e-8),
        return_structured=True,
        verbose=True
    )
```

### 2. Factory Implementation Consistency

**✅ FACTORY VALIDATION PATTERNS:**
```python
def create_solver(problem, solver_type, config=None, **kwargs):
    """Standard factory pattern with validation"""
    # 1. Validate inputs
    if not isinstance(problem, MFGProblem):
        raise TypeError("problem must be MFGProblem instance")
    
    # 2. Set default configuration
    if config is None:
        config = create_fast_config()
    
    # 3. Solver type dispatch with clear error messages
    solver_map = {
        "particle_collocation": ParticleCollocationSolver,
        "adaptive_particle": SilentAdaptiveParticleCollocationSolver,
        # ... etc
    }
    
    if solver_type not in solver_map:
        available = ", ".join(solver_map.keys())
        raise ValueError(f"Unknown solver_type '{solver_type}'. Available: {available}")
    
    # 4. Create and return solver
    solver_class = solver_map[solver_type]
    return solver_class(problem, **config.to_solver_kwargs(), **kwargs)
```

---

## Error Handling & Exceptions

### 1. Exception Class Hierarchy

**✅ STANDARD EXCEPTION CLASSES:**
```python
# Base exception for MFG_PDE
class MFGPDEError(Exception):
    """Base exception for MFG_PDE package"""
    pass

# Specific exception categories
class MathematicalVisualizationError(MFGPDEError):
    """Visualization-related errors"""
    pass

class SolverError(MFGPDEError):
    """Solver-related errors"""  
    pass

class ConvergenceError(SolverError):
    """Convergence failure errors"""
    pass

class ConfigurationError(MFGPDEError):
    """Configuration-related errors"""
    pass

class ValidationError(MFGPDEError):
    """Input validation errors"""
    pass
```

**✅ EXCEPTION USAGE CONSISTENCY:**
```python
# Use specific exception types with actionable messages
raise ConfigurationError(
    f"Invalid newton_tolerance {tolerance}. Must be positive float."
    f" Try: config.newton.tolerance = 1e-6"
)

raise ConvergenceError(
    f"Solver failed to converge after {max_iterations} iterations."
    f" Final error: {final_error:.2e}. "
    f" Try: Increase max_iterations or tolerance."
)
```

### 2. Warning Standards

**✅ DEPRECATION WARNINGS:**
```python
import warnings

def legacy_parameter_warning(old_name, new_name):
    warnings.warn(
        f"Parameter '{old_name}' is deprecated. Use '{new_name}' instead. "
        f"'{old_name}' will be removed in v2.0.",
        DeprecationWarning,
        stacklevel=3
    )

# Usage in solver constructors
def __init__(self, problem, NiterNewton=None, max_newton_iterations=None, **kwargs):
    if NiterNewton is not None:
        legacy_parameter_warning("NiterNewton", "max_newton_iterations")
        if max_newton_iterations is None:
            max_newton_iterations = NiterNewton
    
    self.max_newton_iterations = max_newton_iterations or 30
```

---

## Import & Module Organization

### 1. Package-Level Imports

**✅ __init__.py EXPORT STANDARDS:**
```python
# mfg_pde/__init__.py
"""MFG_PDE: Numerical Solvers for Mean Field Games"""

# Core exports (most commonly used)
from .core.mfg_problem import MFGProblem, ExampleMFGProblem
from .core.boundaries import BoundaryConditions

# Commonly used solvers (not all solvers)
from .alg.particle_collocation_solver import ParticleCollocationSolver
from .alg.adaptive_particle_collocation_solver import SilentAdaptiveParticleCollocationSolver

# Modern patterns (factory and config)
from .factory import create_fast_solver, create_accurate_solver, create_research_solver
from .config import MFGSolverConfig, create_fast_config, create_accurate_config

# Version info
__version__ = "1.4.0"
__author__ = "MFG_PDE Development Team"

# All exports for * imports (discouraged but needed)
__all__ = [
    "MFGProblem", "ExampleMFGProblem", "BoundaryConditions",
    "ParticleCollocationSolver", "SilentAdaptiveParticleCollocationSolver", 
    "create_fast_solver", "create_accurate_solver", "create_research_solver",
    "MFGSolverConfig", "create_fast_config", "create_accurate_config"
]
```

### 2. Conditional Import Patterns

**✅ OPTIONAL DEPENDENCY HANDLING:**
```python
# Standard pattern for optional dependencies
try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None  # Explicit None assignment

# Usage with clear error messages
def create_interactive_plot(data):
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for interactive plots. "
            "Install with: pip install plotly"
        )
    # ... plotting code
```

---

## Version Control & Git Practices

### 1. Commit Message Standards

**✅ COMMIT MESSAGE FORMAT:**
```
type(scope): short description

Longer description explaining what changed and why.

- Specific change 1  
- Specific change 2
- Breaking change info (if applicable)

Closes #123
```

**✅ COMMIT TYPES:**
- `feat`: New feature
- `fix`: Bug fix  
- `refactor`: Code refactoring without functional changes
- `docs`: Documentation changes
- `test`: Test additions or modifications
- `style`: Code style/formatting changes
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**✅ EXAMPLES:**
```
feat(factory): add create_research_solver with enhanced configuration

Added high-accuracy solver factory function for research applications
with verbose output and structured results.

- New create_research_solver() function
- Enhanced MFGSolverConfig with research preset
- Comprehensive validation and error handling

Closes #145

fix(visualization): correct double-named exception class

Fixed MathematicalMathematicalVisualizationError -> MathematicalVisualizationError
in advanced_visualization.py line 50.

Breaking change: Exception class name changed (unlikely to affect users)
```

### 2. Branch Naming Standards

**✅ BRANCH NAMING:**
```
feature/factory-pattern-implementation
fix/visualization-exception-naming  
refactor/parameter-naming-standardization
docs/consistency-guide-creation
test/mass-conservation-validation
hotfix/critical-import-error
```

---

## Consistency Maintenance Schedule

### 1. Regular Maintenance Tasks

**✅ DAILY (During Active Development):**
- [ ] New code follows naming conventions
- [ ] Imports are properly organized  
- [ ] Documentation uses modern API examples
- [ ] Tests use current parameter names

**✅ WEEKLY:**
- [ ] Run consistency validation script (when available)
- [ ] Check for deprecated parameter usage in new code
- [ ] Verify factory pattern examples work
- [ ] Review exception class usage

**✅ MONTHLY:**
- [ ] Full repository consistency audit
- [ ] Update version numbers and dates in documentation
- [ ] Archive obsolete examples
- [ ] Cross-reference validation
- [ ] Terminology consistency review

**✅ QUARTERLY:**
- [ ] Comprehensive documentation review
- [ ] API consistency evaluation
- [ ] Performance regression testing
- [ ] Update consistency guidelines based on new patterns

### 2. Before Major Releases

**✅ PRE-RELEASE CHECKLIST:**
- [ ] All examples use modern API
- [ ] No deprecated patterns in documentation
- [ ] Cross-references validated
- [ ] Exception handling consistent
- [ ] Import organization standardized
- [ ] Mathematical notation unified
- [ ] Test coverage for consistency rules
- [ ] Factory patterns demonstrate correctly

---

## Automated Consistency Checks

### 1. Future Automation Ideas

**✅ POTENTIAL AUTOMATED CHECKS:**
```python
# File: scripts/consistency_check.py

def check_import_organization(file_path):
    """Validate import order and organization"""
    pass

def check_parameter_naming(file_path):
    """Find deprecated parameter names in code"""
    pass

def check_class_naming(file_path):
    """Validate class names follow standards"""
    pass

def check_documentation_examples(file_path):
    """Validate code examples in markdown work"""
    pass

def check_cross_references(docs_dir):
    """Validate all file paths and links"""
    pass

def check_mathematical_notation(file_path):
    """Validate LaTeX notation consistency"""
    pass
```

### 2. Git Hooks Integration

**✅ PRE-COMMIT HOOK CHECKS:**
```bash
#!/bin/bash
# .git/hooks/pre-commit

# Check for deprecated parameter names
echo "Checking for deprecated parameter names..."
if grep -r "NiterNewton\|l2errBoundNewton\|Niter_max" --include="*.py" .; then
    echo "ERROR: Found deprecated parameter names"
    exit 1
fi

# Check for old class names in new files
echo "Checking for deprecated class names..."
if grep -r "HJBGFDMSmartQPSolver\|QuietAdaptive" --include="*.py" .; then
    echo "ERROR: Found deprecated class names"
    exit 1
fi

# Validate import organization
python scripts/check_import_organization.py

echo "Consistency checks passed!"
```

---

## Consistency Violation Examples

### ❌ COMMON VIOLATIONS TO AVOID:

**Code Inconsistencies:**
```python
# Wrong: Mixed parameter naming  
solver = HJBGFDMQPSolver(problem, NiterNewton=30, newton_tolerance=1e-6)

# Wrong: Deprecated class names
from mfg_pde.alg import HJBGFDMSmartQPSolver

# Wrong: Inconsistent import organization
from mfg_pde.core.boundaries import BoundaryConditions
import numpy as np  # Should be before mfg_pde imports
from mfg_pde.core.mfg_problem import MFGProblem
```

**Documentation Inconsistencies:**
```python
# Wrong: Using deprecated names in examples
solver = QuietAdaptiveParticleCollocationSolver(...)

# Wrong: Not showing factory pattern
# Only showing direct class usage without modern alternatives

# Wrong: Inconsistent mathematical notation
"Value function U(x,t)"  # Should be u(t,x)
```

**API Inconsistencies:**
```python
# Wrong: Inconsistent solve method signatures
def solve(self, Niter=20, tol=1e-3):  # Old style

# Wrong: Inconsistent return patterns within same module
return U, M  # Some methods
return {"solution": U, "density": M}  # Other methods  
```

---

## Quick Reference Checklist

### ✅ BEFORE COMMITTING CODE:

**Code:**
- [ ] Class names use standardized modern names
- [ ] Parameter names follow newton/picard conventions  
- [ ] Import organization follows standard order
- [ ] Exception classes use MathematicalVisualizationError family
- [ ] Mathematical notation uses u(t,x), m(t,x) conventions

**Documentation:**
- [ ] Examples show factory pattern first, direct usage second
- [ ] All code examples use modern parameter names
- [ ] Mathematical expressions use proper LaTeX syntax
- [ ] Cross-references point to existing files
- [ ] Terminology uses standardized terms

**Testing:**
- [ ] Tests use modern parameter names
- [ ] Test names follow test_{functionality} pattern
- [ ] Validation uses standard tolerance values
- [ ] Mass conservation checks included where applicable

**Files:**
- [ ] File names follow module naming conventions
- [ ] Directory structure follows established hierarchy
- [ ] Version numbers and dates are current
- [ ] Backward compatibility maintained with deprecation warnings

---

**This consistency guide should be consulted before making any changes to the codebase and updated whenever new standards are established. When in doubt, follow the patterns established by the most recently written, thoroughly reviewed code.**
