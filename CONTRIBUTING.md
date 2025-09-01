# Contributing to MFG_PDE

Thank you for your interest in contributing to the Mean Field Games Partial Differential Equations (MFG_PDE) library! This guide will help you understand our development workflow and coding standards.

## Repository Structure

Our codebase follows a strict organizational pattern:

- **`mfg_pde/`** - Core package code only
- **`tests/`** - Unit and integration tests (no demos)
- **`examples/`** - Demonstration code organized by complexity:
  - `basic/` - Simple single-concept examples
  - `advanced/` - Complex multi-feature demonstrations
  - `notebooks/` - Jupyter notebook examples
  - `tutorials/` - Step-by-step learning materials
- **`benchmarks/`** - Performance comparisons and method analysis
- **`docs/`** - Documentation organized by category
- **`archive/`** - Historical code (do not modify)

## Code Style and Standards

### Import Conventions
```python
# Preferred imports for MFG_PDE
from mfg_pde import ExampleMFGProblem, BoundaryConditions
from mfg_pde.factory import create_fast_solver
from mfg_pde.config import create_fast_config
from mfg_pde.utils.logging import get_logger, configure_research_logging
```

### Text and Symbol Standards
- **NO emojis in Python files** - Keep code comments in plain ASCII
- **UTF-8 math symbols allowed in docstrings**: Use `u/t`, `u` for mathematical notation
- **ASCII for output**: Use `||u - u_prev||` instead of `u - u_prev` in print statements

### Documentation Standards
- Use comprehensive docstrings with LaTeX mathematical notation
- Include examples: `$u(t,x)$` and `$m(t,x)$`
- Reference code locations: `file_path:line_number`
- Follow consistency guide in `docs/development/CONSISTENCY_GUIDE.md`

### Logging Standards
```python
from mfg_pde.utils.logging import get_logger, configure_research_logging

configure_research_logging("session_name", level="INFO")
logger = get_logger(__name__)
```

## Development Workflow

### Adding New Features
1. **Core functionality**: Add to appropriate `mfg_pde/` subdirectory
2. **Examples**: Create in `examples/basic/` or `examples/advanced/`
3. **Tests**: Add to `tests/unit/` or `tests/integration/`
4. **Documentation**: Update relevant `docs/` category
5. **Benchmarks**: Add performance analysis if applicable

### Code Quality Requirements
- Use type hints and comprehensive docstrings
- Include error handling and validation
- Support both interactive and non-interactive usage
- Follow mathematical notation conventions: `u(t,x)`, `m(t,x)`

### Testing
- Write unit tests for all new functionality
- Include integration tests for complex features
- Use meaningful test names and documentation
- Test both success and error cases

## Mathematical Standards

### Notation Consistency
- **State variables**: `u(t,x)` for value function, `m(t,x)` for mass distribution
- **Equations**: Use LaTeX in docstrings, ASCII in code comments
- **Variables**: Consistent naming across modules

### Numerical Methods
- Document convergence properties and stability
- Include complexity analysis for new algorithms
- Provide benchmarks for performance-critical code

## Visualization Guidelines

### Plotting Standards
- **Primary**: Use Plotly for interactive plots when possible
- **Fallback**: Matplotlib with professional styling
- **Export**: Support both HTML (interactive) and PNG (static)
- **Mathematical notation**: Consistent u(t,x) and m(t,x) conventions

## Pull Request Process

1. **Fork** the repository and create a feature branch
2. **Implement** your changes following the coding standards
3. **Test** your changes thoroughly
4. **Document** new features and update relevant docs
5. **Submit** a pull request with clear description

### PR Requirements
- All tests must pass
- Code follows style guidelines
- New features include tests and documentation
- Mathematical notation is consistent
- No emojis in Python code files

## Quality Assurance

Before submitting:
- Verify all imports work correctly
- Check file placement follows conventions
- Update relevant documentation
- Test examples execute successfully
- Run linting and type checking

## Getting Help

- Check existing documentation in `docs/`
- Review examples in `examples/`
- Look at existing code patterns in `mfg_pde/`
- Open an issue for questions or bug reports

## Mathematical Background

This library implements sophisticated Mean Field Games methods. Contributors should be familiar with:
- Partial differential equations
- Optimal control theory
- Numerical analysis
- Hamilton-Jacobi-Bellman equations

For mathematical foundations, see `docs/theory/` directory.

---

By contributing to MFG_PDE, you agree to maintain these standards and help build a high-quality, research-grade mathematical software library.
