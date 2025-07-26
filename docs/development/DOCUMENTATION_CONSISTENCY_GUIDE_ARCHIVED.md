# Documentation Consistency Guide

**Version:** 1.0  
**Date:** July 26, 2025  
**Purpose:** Maintain consistent, professional documentation across the MFG_PDE repository  

This guide establishes standards for documentation consistency similar to `FORMATTING_PREFERENCES.md` but focused on documentation patterns, code examples, and cross-reference accuracy.

## Table of Contents

1. [Code Examples Standards](#code-examples-standards)
2. [API Usage Patterns](#api-usage-patterns)
3. [Cross-Reference Guidelines](#cross-reference-guidelines)
4. [Terminology Standards](#terminology-standards)
5. [Formatting Conventions](#formatting-conventions)
6. [Consistency Checklist](#consistency-checklist)
7. [Maintenance Schedule](#maintenance-schedule)

---

## Code Examples Standards

### 1. Modern API Usage

**✅ DO: Use modern class names and parameter names**
```python
# Correct - Modern standardized names
from mfg_pde.alg.hjb_solvers import HJBGFDMQPSolver, HJBGFDMTunedQPSolver
from mfg_pde.alg import SilentAdaptiveParticleCollocationSolver

solver = HJBGFDMQPSolver(
    problem=problem,
    max_newton_iterations=30,  # Modern parameter name
    newton_tolerance=1e-6      # Modern parameter name
)
```

**❌ DON'T: Use deprecated class names or parameters**
```python
# Incorrect - Deprecated names
from mfg_pde.alg.hjb_solvers import HJBGFDMSmartQPSolver  # OLD
from mfg_pde.alg import QuietAdaptiveParticleCollocationSolver  # OLD

solver = HJBGFDMQPSolver(
    problem=problem,
    NiterNewton=30,        # OLD parameter name
    l2errBoundNewton=1e-6  # OLD parameter name
)
```

### 2. Factory Pattern Examples

**✅ DO: Show modern factory pattern as primary approach**
```python
# Recommended approach
from mfg_pde.factory import create_fast_solver
from mfg_pde.config import create_fast_config

config = create_fast_config()
solver = create_fast_solver(
    problem=problem,
    solver_type="adaptive_particle", 
    config=config
)

result = solver.solve()
U, M = result.solution, result.density
```

**✅ ALSO SHOW: Direct class usage as alternative**
```python
# Alternative direct approach
from mfg_pde.alg import SilentAdaptiveParticleCollocationSolver

solver = SilentAdaptiveParticleCollocationSolver(
    problem=problem,
    collocation_points=collocation_points,
    max_picard_iterations=20,
    picard_tolerance=1e-5
)

U, M, info = solver.solve()
```

### 3. Import Statements

**✅ DO: Use clean, organized imports**
```python
# Standard library first
import numpy as np
import matplotlib.pyplot as plt

# Third-party libraries
from scipy import optimize

# MFG_PDE imports - organized by module
from mfg_pde.core import ExampleMFGProblem, BoundaryConditions
from mfg_pde.alg import SilentAdaptiveParticleCollocationSolver
from mfg_pde.config import create_fast_config
from mfg_pde.factory import create_fast_solver
```

**❌ DON'T: Mix import styles or use deprecated paths**
```python
# Incorrect - Mixed styles and old paths
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.alg.enhanced_particle_collocation_solver import EnhancedParticleCollocationSolver  # OLD
import numpy as np
from mfg_pde.config import *  # Avoid star imports
```

---

## API Usage Patterns

### 1. Exception Handling

**✅ DO: Use standardized exception class name**
```python
try:
    result = solver.solve()
except MathematicalVisualizationError as e:
    print(f"Visualization failed: {e}")
```

**❌ DON'T: Use inconsistent exception names**
```python
# Incorrect - Old inconsistent naming
except VisualizationError as e:  # OLD NAME
    pass
```

### 2. Configuration Usage

**✅ DO: Demonstrate configuration system**
```python
from mfg_pde.config import MFGSolverConfig, NewtonConfig, PicardConfig

# Custom configuration
config = MFGSolverConfig(
    newton=NewtonConfig(max_iterations=50, tolerance=1e-8),
    picard=PicardConfig(max_iterations=25, tolerance=1e-6),
    return_structured=True
)

# Save/load configurations
config_dict = config.to_dict()
config_restored = MFGSolverConfig.from_dict(config_dict)
```

### 3. Mathematical Visualization

**✅ DO: Use consistent mathematical notation**
```python
from mfg_pde.utils.mathematical_visualization import MFGMathematicalVisualizer

visualizer = MFGMathematicalVisualizer(backend="matplotlib", enable_latex=True)

# Use lowercase u for continuous functions, (t,x) coordinate convention
fig = visualizer.plot_hjb_analysis(
    U, x_grid, t_grid,
    title=r"HJB Analysis: $-\frac{\partial u}{\partial t} + H(t,x,\nabla u, m) = 0$"
)
```

---

## Cross-Reference Guidelines

### 1. File Path References

**✅ DO: Use correct relative paths and verify they exist**
```markdown
<!-- Correct relative path -->
See [examples directory](../examples/) for working implementations.

<!-- Verify file exists before referencing -->
Run the example: [particle_collocation_mfg_example.py](../examples/particle_collocation_mfg_example.py)
```

**❌ DON'T: Use broken or unverified paths**
```markdown
<!-- Incorrect - path may not exist -->
See [old examples](../../archive/old_examples/particle_collocation_no_flux_bc.py)
```

### 2. Documentation Links

**✅ DO: Maintain consistent cross-references**
```markdown
For detailed implementation, see:
- [Mathematical Background](../theory/mathematical_background.md)
- [Configuration System Guide](../development/v1.4_visualization_consistency_fixes.md)
- [Factory Pattern Examples](../../examples/factory_patterns_example.py)
```

### 3. Version References

**✅ DO: Use current dates and version numbers**
```markdown
**Last Updated:** July 26, 2025
**Version:** v1.4 - Visualization Consistency Fixes
```

**❌ DON'T: Leave outdated version information**
```markdown
**Last Updated:** July 25, 2024  <!-- OLD DATE -->
```

---

## Terminology Standards

### 1. Solver Method Names

**✅ STANDARD TERMS:**
- "Particle-Collocation" (with hyphen)
- "QP-Collocation" (not "QP Particle-Collocation")
- "Hybrid Particle-FDM" (consistent capitalization)
- "GFDM" (Generalized Finite Difference Method)
- "HJB" (Hamilton-Jacobi-Bellman)
- "FP" (Fokker-Planck)

### 2. Technical Terminology

**✅ CONSISTENT USAGE:**
- "Factory pattern" (not "factory patterns" in headings)
- "Configuration system" (not "config system")
- "Mass conservation" (not "mass preservation")
- "Convergence criteria" (not "convergence criterion" for multiple)
- "Boundary conditions" (not "boundary condition" for the concept)

### 3. Parameter Names

**✅ CURRENT STANDARD NAMES:**
```python
# Newton parameters
max_newton_iterations  # NOT: NiterNewton
newton_tolerance      # NOT: l2errBoundNewton

# Picard parameters  
max_picard_iterations # NOT: Niter_max
picard_tolerance     # NOT: l2errBoundPicard
convergence_tolerance # NOT: l2errBound
```

---

## Formatting Conventions

### 1. Code Block Languages

**✅ DO: Always specify language for syntax highlighting**
````markdown
```python
# Python code example
from mfg_pde import ExampleMFGProblem
```

```bash
# Shell commands
pip install -e .
```

```json
{
  "configuration": "example"
}
```
````

### 2. Heading Styles

**✅ DO: Use consistent heading hierarchy**
```markdown
# Main Title (H1)
## Major Section (H2)
### Subsection (H3)
#### Minor Section (H4)
```

**❌ DON'T: Skip heading levels or use inconsistent styles**
```markdown
# Main Title
### Skipped H2 Level  <!-- WRONG -->
## Back to H2 Level
```

### 3. Emoji Usage

**✅ DO: Follow FORMATTING_PREFERENCES - minimal emoji usage**
```markdown
✅ Success indicator (acceptable)
❌ Error indicator (acceptable)
⚠ Warning indicator (acceptable)

<!-- Avoid decorative emojis -->
```

**❌ DON'T: Use excessive decorative emojis**
```markdown
🚀 Amazing new features! 🎉 <!-- AVOID -->
📊 Data analysis 📈 <!-- AVOID -->
```

---

## Consistency Checklist

### Before Committing Documentation Changes:

#### 1. Code Examples Review
- [ ] All class names use modern standardized names
- [ ] Parameter names use current conventions (max_newton_iterations, etc.)
- [ ] Import statements are clean and organized
- [ ] Both factory pattern and direct usage shown where appropriate
- [ ] Exception handling uses MathematicalVisualizationError consistently

#### 2. Cross-References Verification
- [ ] All file paths verified to exist
- [ ] Relative paths are correct
- [ ] Version numbers and dates are current
- [ ] Links between documents work correctly

#### 3. Terminology Consistency
- [ ] Method names follow standard terminology
- [ ] Technical terms used consistently throughout
- [ ] Parameter names match current API

#### 4. Formatting Standards
- [ ] Code blocks specify language
- [ ] Heading hierarchy is correct
- [ ] Emoji usage follows FORMATTING_PREFERENCES
- [ ] Markdown formatting is consistent

#### 5. API Compatibility
- [ ] Examples work with current codebase
- [ ] Modern features are demonstrated
- [ ] Deprecated usage is avoided
- [ ] Factory patterns shown as primary approach

---

## Maintenance Schedule

### Regular Consistency Checks

**✅ WEEKLY (During Active Development):**
- Verify new examples use modern API
- Check that README examples work with current code
- Update any deprecated class/parameter references

**✅ MONTHLY:**
- Full cross-reference link validation
- Terminology consistency review across all .md files
- Version number and date updates

**✅ QUARTERLY:**
- Comprehensive documentation review
- Archive outdated examples that are no longer relevant
- Update consistency guidelines based on new patterns

### After Major API Changes:

1. **Immediate (Same Day):**
   - Update main README.md examples
   - Fix any breaking changes in active examples
   - Update parameter names in all documentation

2. **Within 1 Week:**
   - Review all .md files for consistency
   - Update cross-references to new classes/methods
   - Verify all example code works

3. **Within 1 Month:**
   - Archive superseded examples
   - Update comprehensive documentation
   - Create migration guides if needed

---

## Implementation Notes

### Automated Checks (Future Enhancement)

Consider implementing automated checks for:
- Link validation in .md files
- Code example syntax verification
- Terminology consistency scanning
- Import statement validation

### Documentation Testing

Include documentation examples in test suite:
```python
# Test that README examples actually work
def test_readme_examples():
    # Extract and test code from README.md
    pass
```

### Review Process

For documentation changes:
1. Run through consistency checklist
2. Test all code examples
3. Verify cross-references
4. Check terminology usage
5. Validate formatting standards

---

**This guide should be consulted before making any documentation changes and updated whenever new API patterns are established.**