# Docstring Standards for MFG_PDE Package

**Version:** 1.4.0  
**Last Updated:** 2025-07-27  
**Applies to:** All public classes, methods, and functions  

## Overview

This document establishes comprehensive docstring standards for the MFG_PDE package to ensure consistency, clarity, and maintainability across all modules. All public code must follow these standards.

## Docstring Format

We use **Google-style docstrings** with additional mathematical notation support for scientific computing contexts.

## Class Docstrings

### Template

```python
class SolverExample:
    """Brief one-line description of the class purpose.
    
    Longer description explaining the purpose, mathematical background,
    and usage context. Include references to relevant papers when applicable.
    
    Mathematical Formulation:
        Brief description of the mathematical problem being solved.
        Use LaTeX notation: $u(t,x)$ represents the value function.
        
    Attributes:
        parameter_name (type): Description with valid ranges and units.
        convergence_tolerance (float): Tolerance for convergence, typically 1e-6.
        
    Example:
        Basic usage example showing typical parameters::
        
            problem = ExampleMFGProblem(Nx=50, T=1.0)
            solver = SolverExample(problem)
            result = solver.solve(max_iterations=100)
            
    References:
        [1] Lasry & Lions, "Mean Field Games", Japanese Journal of Mathematics, 2007.
        [2] Achdou & Capuzzo-Dolcetta, "Mean Field Games: Numerical Methods", SIAM, 2010.
    
    Note:
        Special considerations, performance notes, or important warnings.
    """
```

### Required Sections for Classes

1. **Brief Description**: One-line summary
2. **Extended Description**: Detailed explanation of purpose and context
3. **Mathematical Formulation**: For mathematical classes (optional for utilities)
4. **Attributes**: All public attributes with types and descriptions
5. **Example**: Working code example
6. **References**: Academic papers or documentation (when applicable)
7. **Note**: Important warnings or considerations (when applicable)

## Method Docstrings

### Template

```python
def solve_method(self, parameter: float, option: str = "default") -> SolutionResult:
    """Solve the mathematical problem with given parameters.
    
    Detailed description of what the method does, any important
    algorithmic details, and convergence behavior.
    
    Mathematical Background:
        Solves the system: $\\partial_t u + H(x, \\nabla u, m) = 0$
        where $H$ is the Hamiltonian and $m$ is the density.
    
    Args:
        parameter (float): Description with valid range [0.1, 10.0].
        option (str, optional): Solver option. Must be one of:
            - "default": Standard algorithm (recommended)
            - "fast": Faster but less accurate
            - "accurate": Slower but higher precision
            Defaults to "default".
            
    Returns:
        SolutionResult: Object containing:
            - solution (np.ndarray): Main solution array of shape (Nt+1, Nx+1)
            - convergence_info (dict): Convergence statistics including:
                - iterations (int): Number of iterations used
                - final_error (float): Final convergence error
                - converged (bool): Whether convergence was achieved
            - metadata (dict): Additional solver information
            
    Raises:
        ConvergenceError: When solver fails to converge within max_iterations.
        ConfigurationError: When parameters are invalid or incompatible.
        ValueError: When input arrays have incorrect shapes or invalid values.
        
    Example:
        Solve a standard MFG problem::
        
            >>> solver = SolverExample()
            >>> result = solver.solve_method(1.0, "accurate")
            >>> print(f"Converged in {result.convergence_info['iterations']} iterations")
            >>> plt.imshow(result.solution)
            
    Note:
        For large problems (Nx > 200), consider using option="fast" to reduce
        computation time. Memory usage scales as O(Nx * Nt).
        
    See Also:
        :meth:`solve_fast`: Faster variant with reduced accuracy
        :meth:`get_convergence_history`: Detailed convergence analysis
    """
```

### Required Sections for Methods

1. **Brief Description**: One-line summary of what method does
2. **Extended Description**: Detailed explanation including algorithms
3. **Mathematical Background**: For mathematical methods (optional for utilities)
4. **Args**: All parameters with types, descriptions, and valid ranges
5. **Returns**: Complete description of return values and their structure
6. **Raises**: All exceptions that can be raised
7. **Example**: Working code example with expected output
8. **Note**: Performance considerations, memory usage, special cases
9. **See Also**: Related methods or functions (when applicable)

## Function Docstrings

### Template

```python
def utility_function(data: np.ndarray, threshold: float = 1e-6) -> Tuple[bool, float]:
    """Check convergence of iterative algorithm.
    
    Analyzes the convergence behavior of an iterative sequence by computing
    the relative change between consecutive iterations.
    
    Args:
        data (np.ndarray): Sequence of iteration values, shape (n_iterations,).
        threshold (float, optional): Convergence threshold. Defaults to 1e-6.
        
    Returns:
        tuple: A tuple containing:
            - converged (bool): True if sequence has converged
            - final_error (float): Final relative error
            
    Raises:
        ValueError: If data array is empty or has less than 2 elements.
        
    Example:
        Check convergence of a solution sequence::
        
            >>> errors = np.array([1.0, 0.1, 0.01, 0.001])
            >>> converged, error = utility_function(errors, 1e-2)
            >>> print(f"Converged: {converged}, Final error: {error}")
            Converged: True, Final error: 0.001
    """
```

## Mathematical Notation Standards

### LaTeX in Docstrings

Use LaTeX notation for mathematical expressions:

```python
"""
The Hamilton-Jacobi-Bellman equation: $\\partial_t u + H(x, \\nabla u, m) = 0$
where:
- $u(t,x)$: Value function
- $m(t,x)$: Density function  
- $H$: Hamiltonian
- $\\nabla u$: Gradient of value function
"""
```

### Standard Variable Conventions

**Spatial and Temporal:**
- `x`: Spatial coordinate ∈ [xmin, xmax]
- `t`: Time coordinate ∈ [0, T]
- `Nx`: Number of spatial grid points
- `Nt`: Number of temporal grid points

**MFG Variables:**
- `U`: Value function $u(t,x)$ solution array
- `M`: Density function $m(t,x)$ solution array
- `sigma`: Diffusion coefficient $\\sigma$
- `coupling_coefficient`: Coupling strength (modern name for coefCT)

## Type Annotation Standards

### Required Type Annotations

All public functions and methods must include complete type annotations:

```python
from typing import Dict, List, Optional, Tuple, Union, Any
from numpy.typing import NDArray
import numpy as np

def process_solution(
    U: NDArray[np.float64],
    M: NDArray[np.float64], 
    config: Dict[str, Any],
    return_metadata: bool = True
) -> Union[Tuple[np.ndarray, np.ndarray], SolutionResult]:
    """Process MFG solution arrays with optional metadata."""
```

### Standard Type Aliases

```python
# Use these standard type aliases for consistency
SpatialArray = NDArray[np.float64]      # Shape: (Nx+1,)
TemporalArray = NDArray[np.float64]     # Shape: (Nt+1,)
SolutionArray = NDArray[np.float64]     # Shape: (Nt+1, Nx+1)
ConfigDict = Dict[str, Any]             # Configuration parameters
```

## Cross-Reference Standards

### See Also Sections

Link to related functionality using Sphinx-style references:

```python
"""
See Also:
    :class:`~mfg_pde.alg.base_mfg_solver.MFGSolver`: Base solver interface
    :meth:`~mfg_pde.core.mfg_problem.ExampleMFGProblem.H`: Hamiltonian function
    :func:`~mfg_pde.factory.create_solver`: Solver factory function
    :doc:`../theory/mathematical_background`: Mathematical foundations
"""
```

## Example Quality Standards

### Good Examples

```python
def solve(self, max_iterations: int = 100) -> SolutionResult:
    """Solve the MFG system using fixed-point iteration.
    
    Example:
        Solve with custom parameters::
        
            >>> problem = ExampleMFGProblem(Nx=50, T=1.0, sigma=0.5)
            >>> solver = FixedPointSolver(problem)
            >>> result = solver.solve(max_iterations=200)
            >>> print(f"Converged: {result.converged}")
            >>> 
            >>> # Visualize results
            >>> import matplotlib.pyplot as plt
            >>> plt.figure(figsize=(12, 5))
            >>> plt.subplot(1, 2, 1)
            >>> plt.imshow(result.U, aspect='auto', origin='lower')
            >>> plt.title('Value Function U(t,x)')
            >>> plt.subplot(1, 2, 2)
            >>> plt.imshow(result.M, aspect='auto', origin='lower')
            >>> plt.title('Density Function M(t,x)')
            >>> plt.show()
            
        Expected output shows convergence in typically 50-100 iterations
        for standard problems with default parameters.
    """
```

### Example Requirements

1. **Runnable**: Examples must execute without errors
2. **Realistic**: Use realistic parameter values
3. **Complete**: Show imports, setup, and expected output
4. **Educational**: Demonstrate best practices and common usage patterns

## Error Documentation Standards

### Exception Documentation

Document all possible exceptions with specific conditions:

```python
def solve(self, max_iterations: int) -> SolutionResult:
    """
    Raises:
        ConvergenceError: When algorithm fails to converge within max_iterations.
            This typically occurs with:
            - Very small tolerance values (< 1e-12)
            - Ill-conditioned problems
            - Insufficient iterations for problem complexity
            
        ConfigurationError: When solver configuration is invalid:
            - max_iterations <= 0
            - tolerance <= 0 or >= 1
            - Incompatible boundary conditions
            
        ValueError: When input data is invalid:
            - Arrays with NaN or infinite values
            - Mismatched array dimensions
            - Negative time or spatial steps
            
        MemoryError: When problem size exceeds available memory.
            Typically occurs when Nx * Nt > 10^6 elements.
    """
```

## Performance Documentation

### Performance Notes

Include performance characteristics for computational methods:

```python
def solve_large_problem(self, Nx: int, Nt: int) -> SolutionResult:
    """
    Note:
        Performance characteristics:
        - Time complexity: O(Nx * Nt * max_iterations)
        - Memory usage: O(Nx * Nt) for solution arrays
        - Recommended limits: Nx * Nt < 10^6 for interactive use
        
        For large problems (Nx > 500 or Nt > 1000):
        - Consider using memory monitoring (@memory_monitored decorator)
        - Enable progress tracking for long computations
        - Use warm start from coarser grid solutions
        
        Typical performance on modern hardware:
        - Small problems (Nx=50, Nt=50): < 1 second
        - Medium problems (Nx=200, Nt=100): 10-30 seconds  
        - Large problems (Nx=500, Nt=200): 5-15 minutes
    """
```

## Validation Checklist

### Before Committing Code

Ensure all docstrings meet these standards:

- [ ] Brief description is clear and informative
- [ ] All parameters documented with types and valid ranges
- [ ] Return values completely described with structure
- [ ] All exceptions documented with triggering conditions
- [ ] Working example provided that executes correctly
- [ ] Mathematical notation uses LaTeX format
- [ ] Cross-references use proper Sphinx syntax
- [ ] Performance characteristics documented for computational methods
- [ ] Type annotations are complete and accurate

### Automated Checks

Use these tools to validate docstring quality:

```bash
# Check docstring coverage
pydocstyle mfg_pde/

# Validate type annotations  
mypy mfg_pde/ --strict

# Test examples in docstrings
python -m doctest mfg_pde/module_name.py
```

## Migration from Legacy Documentation

### Updating Existing Docstrings

When updating legacy docstrings:

1. **Preserve Mathematical Content**: Keep accurate mathematical descriptions
2. **Modernize Parameter Names**: Use current parameter names (see parameter_migration.py)
3. **Add Missing Sections**: Include all required sections from templates
4. **Improve Examples**: Ensure examples use modern API and best practices
5. **Update References**: Check that academic references are current and accessible

### Consistency with Existing Code

- Follow patterns established in recently updated modules
- Use consistent terminology throughout the package
- Maintain backward compatibility in documented API
- Reference the CONSISTENCY_GUIDE.md for style decisions

## Tools and Automation

### Recommended Tools

- **pydocstyle**: Docstring style checking
- **sphinx**: Documentation generation
- **mypy**: Type annotation validation
- **doctest**: Example testing
- **flake8-docstrings**: Integration with linting pipeline

### IDE Integration

Configure your IDE for docstring support:

- Use docstring templates matching these standards
- Enable type checking and linting
- Set up auto-formatting for docstring text
- Configure spell-checking for documentation

This comprehensive standard ensures that all MFG_PDE documentation is consistent, informative, and maintainable while supporting the mathematical and computational nature of the package.
