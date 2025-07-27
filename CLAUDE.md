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

## 📊 **Package Management**

### **Dependencies**
- **Core**: numpy, scipy, matplotlib (always available)
- **Interactive**: plotly, jupyter, nbformat (with fallbacks)
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

---

**Last Updated**: 2025-07-26  
**Repository Version**: Post-reorganization clean structure  
**Claude Code**: Always reference this file for MFG_PDE conventions