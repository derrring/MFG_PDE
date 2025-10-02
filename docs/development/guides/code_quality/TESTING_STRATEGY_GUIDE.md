# Unit Testing Strategy Guide for MFG_PDE

**Date:** July 27, 2025  
**Purpose:** Comprehensive guide for implementing effective unit testing in scientific computing packages  
**Target Audience:** Developers and researchers working on MFG_PDE  

## Executive Summary

Unit testing in scientific computing requires a balanced approach that validates both computational correctness and software engineering principles. This guide provides a comprehensive strategy for implementing effective testing throughout the MFG_PDE package.

**Key Principle:** *Test strategically, not everywhere* - Focus on critical functionality, edge cases, and public APIs while maintaining computational efficiency.

## 🧪 **Testing Philosophy for Scientific Computing**

### What to Test (High Priority):
1. **Public APIs** - All user-facing functions and classes
2. **Mathematical correctness** - Numerical algorithms and formulations
3. **Configuration validation** - Parameter ranges and constraints
4. **Error handling** - Exception conditions and edge cases
5. **Integration points** - Component interactions
6. **Performance regressions** - Critical computational paths

### What NOT to Test (Low Priority):
1. **Private implementation details** - Internal helper functions
2. **Third-party libraries** - numpy, scipy functionality
3. **Simple getters/setters** - Trivial property access
4. **Generated code** - Auto-generated configuration classes
5. **Plotting/visualization** - Visual output (test data generation instead)

## 📁 **Test Organization Structure**

### Recommended Directory Layout:
```
MFG_PDE/
├── mfg_pde/                    # Source code
├── tests/                      # Test suite (separate from source)
│   ├── unit/                   # Fast, isolated tests
│   │   ├── test_config/        # Configuration testing
│   │   ├── test_solvers/       # Solver unit tests
│   │   ├── test_utils/         # Utility function tests
│   │   └── test_core/          # Core functionality tests
│   ├── integration/            # Cross-component tests
│   │   ├── test_solver_integration.py
│   │   ├── test_factory_patterns.py
│   │   └── test_end_to_end.py
│   ├── performance/            # Performance benchmarks
│   │   ├── test_memory_usage.py
│   │   ├── test_execution_time.py
│   │   └── benchmarks/
│   ├── mathematical/           # Mathematical validation
│   │   ├── test_mass_conservation.py
│   │   ├── test_convergence_properties.py
│   │   └── test_numerical_stability.py
│   ├── fixtures/               # Test data and fixtures
│   │   ├── problems/           # Sample problems
│   │   ├── configurations/     # Test configurations
│   │   └── reference_solutions/ # Known good solutions
│   └── conftest.py             # Pytest configuration
├── pytest.ini                 # Test configuration
└── pyproject.toml             # Tool configuration
```

### Why This Structure Works:
- **Separation of concerns** - Different test types in different directories
- **Parallel execution** - Tests can run independently
- **Selective running** - Run only specific test categories
- **Clear organization** - Easy to find and maintain tests

## 🎯 **Testing Strategy by Component**

### 1. **Configuration Testing** (High Coverage ~95%)

```python
# tests/unit/test_config/test_pydantic_config.py
import pytest
from pydantic import ValidationError
from mfg_pde.config.pydantic_config import NewtonConfig, MFGSolverConfig

class TestNewtonConfig:
    """Test Newton solver configuration validation."""
    
    def test_valid_configuration(self):
        """Test creation with valid parameters."""
        config = NewtonConfig(
            max_iterations=20,
            tolerance=1e-6,
            damping_factor=0.8
        )
        assert config.max_iterations == 20
        assert config.tolerance == 1e-6
        assert config.damping_factor == 0.8
    
    def test_parameter_validation_ranges(self):
        """Test parameter range validation."""
        # Test invalid max_iterations
        with pytest.raises(ValidationError, match="ge=1"):
            NewtonConfig(max_iterations=0)
        
        # Test invalid tolerance
        with pytest.raises(ValidationError, match="gt=1e-15"):
            NewtonConfig(tolerance=0)
        
        # Test invalid damping_factor
        with pytest.raises(ValidationError, match="le=1.0"):
            NewtonConfig(damping_factor=1.5)
    
    def test_default_values(self):
        """Test default parameter values."""
        config = NewtonConfig()
        assert config.max_iterations == 30
        assert config.tolerance == 1e-6
        assert config.damping_factor == 1.0
    
    @pytest.mark.parametrize("max_iter,tol,damping", [
        (10, 1e-4, 0.5),
        (50, 1e-8, 1.0),
        (100, 1e-10, 0.1),
    ])
    def test_parameter_combinations(self, max_iter, tol, damping):
        """Test various parameter combinations."""
        config = NewtonConfig(
            max_iterations=max_iter,
            tolerance=tol,
            damping_factor=damping
        )
        assert config.max_iterations == max_iter
        assert config.tolerance == tol
        assert config.damping_factor == damping
```

### 2. **Solver Testing** (Medium Coverage ~80%)

```python
# tests/unit/test_solvers/test_hjb_fdm.py
import pytest
import numpy as np
from mfg_pde.alg.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem

class TestHJBFDMSolver:
    """Test HJB FDM solver functionality."""
    
    @pytest.fixture
    def simple_problem(self):
        """Create simple test problem."""
        return ExampleMFGProblem(Nx=10, Nt=5, T=0.5)
    
    @pytest.fixture
    def solver(self, simple_problem):
        """Create solver instance."""
        return HJBFDMSolver(
            problem=simple_problem,
            max_newton_iterations=10,
            newton_tolerance=1e-4
        )
    
    def test_solver_initialization(self, solver, simple_problem):
        """Test solver is properly initialized."""
        assert solver.problem == simple_problem
        assert solver.max_newton_iterations == 10
        assert solver.newton_tolerance == 1e-4
        assert solver.hjb_method_name == "FDM"
    
    def test_parameter_validation(self, simple_problem):
        """Test parameter validation in constructor."""
        # Test invalid max_iterations
        with pytest.raises(ValueError, match="max_newton_iterations must be >= 1"):
            HJBFDMSolver(simple_problem, max_newton_iterations=0)
        
        # Test invalid tolerance
        with pytest.raises(ValueError, match="newton_tolerance must be > 0"):
            HJBFDMSolver(simple_problem, newton_tolerance=-1e-6)
    
    def test_solve_hjb_system_dimensions(self, solver):
        """Test that solve_hjb_system returns correct dimensions."""
        Nt, Nx = solver.problem.Nt, solver.problem.Nx
        
        # Create test inputs
        M_density = np.random.rand(Nt + 1, Nx + 1)
        U_final = np.random.rand(Nx + 1)
        U_prev = np.random.rand(Nt + 1, Nx + 1)
        
        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)
        
        # Check dimensions
        assert U_solution.shape == (Nt + 1, Nx + 1)
        assert not np.any(np.isnan(U_solution))
        assert not np.any(np.isinf(U_solution))
    
    def test_legacy_parameter_migration(self, simple_problem):
        """Test legacy parameter names are properly migrated."""
        # This should work without errors and issue deprecation warnings
        with pytest.warns(DeprecationWarning, match="NiterNewton"):
            solver = HJBFDMSolver(
                simple_problem,
                NiterNewton=15,  # Legacy parameter
                l2errBoundNewton=1e-5  # Legacy parameter
            )
        
        # Check parameters were migrated
        assert solver.max_newton_iterations == 15
        assert solver.newton_tolerance == 1e-5
```

### 3. **Mathematical Validation Testing** (Critical ~100%)

```python
# tests/mathematical/test_mass_conservation.py
import pytest
import numpy as np
from mfg_pde import ExampleMFGProblem, create_fast_solver

class TestMassConservation:
    """Test mass conservation properties of MFG solvers."""
    
    @pytest.fixture(params=["fixed_point", "particle_collocation"])
    def solver_type(self, request):
        """Test multiple solver types."""
        return request.param
    
    @pytest.fixture(params=[
        {"Nx": 20, "Nt": 10, "T": 0.5},
        {"Nx": 30, "Nt": 15, "T": 1.0},
    ])
    def problem_config(self, request):
        """Test multiple problem configurations."""
        return request.param
    
    def test_mass_conservation_property(self, solver_type, problem_config):
        """Test that mass is conserved throughout evolution."""
        problem = ExampleMFGProblem(**problem_config)
        solver = create_fast_solver(problem, solver_type)
        
        result = solver.solve()
        
        # Calculate initial mass
        initial_mass = np.sum(problem.m_init) * problem.Dx
        
        # Check mass conservation at each time step
        for t_idx in range(problem.Nt + 1):
            current_mass = np.sum(result.M_solution[t_idx, :]) * problem.Dx
            mass_error = abs(current_mass - initial_mass)
            
            # Mass should be conserved within numerical tolerance
            assert mass_error < 1e-2, (
                f"Mass not conserved at t_idx={t_idx}: "
                f"initial={initial_mass:.6f}, current={current_mass:.6f}, "
                f"error={mass_error:.6e}"
            )
    
    def test_non_negativity_property(self, solver_type, problem_config):
        """Test that density remains non-negative."""
        problem = ExampleMFGProblem(**problem_config)
        solver = create_fast_solver(problem, solver_type)
        
        result = solver.solve()
        
        # Density should be non-negative everywhere
        assert np.all(result.M_solution >= -1e-10), (
            f"Negative density found: min={np.min(result.M_solution):.6e}"
        )
    
    @pytest.mark.parametrize("sigma", [0.1, 0.5, 1.0, 2.0])
    def test_mass_conservation_across_diffusion_coefficients(self, sigma):
        """Test mass conservation for different diffusion coefficients."""
        problem = ExampleMFGProblem(Nx=25, Nt=10, sigma=sigma)
        solver = create_fast_solver(problem, "fixed_point")
        
        result = solver.solve()
        
        initial_mass = np.sum(problem.m_init) * problem.Dx
        final_mass = np.sum(result.M_solution[-1, :]) * problem.Dx
        
        mass_error = abs(final_mass - initial_mass)
        assert mass_error < 1e-2, (
            f"Mass not conserved for sigma={sigma}: "
            f"error={mass_error:.6e}"
        )
```

### 4. **Performance Testing** (Selective)

```python
# tests/performance/test_execution_time.py
import pytest
import time
from mfg_pde import ExampleMFGProblem, create_fast_solver

class TestPerformance:
    """Test performance characteristics and regressions."""
    
    @pytest.mark.performance
    def test_small_problem_execution_time(self):
        """Test that small problems complete within reasonable time."""
        problem = ExampleMFGProblem(Nx=20, Nt=10)
        solver = create_fast_solver(problem, "fixed_point")
        
        start_time = time.time()
        result = solver.solve()
        execution_time = time.time() - start_time
        
        # Small problem should complete quickly
        assert execution_time < 10.0, f"Execution too slow: {execution_time:.2f}s"
        assert result is not None
    
    @pytest.mark.performance
    @pytest.mark.parametrize("size_factor", [1, 2, 4])
    def test_scaling_behavior(self, size_factor):
        """Test performance scaling with problem size."""
        base_nx, base_nt = 15, 8
        problem = ExampleMFGProblem(
            Nx=base_nx * size_factor, 
            Nt=base_nt * size_factor
        )
        solver = create_fast_solver(problem, "fixed_point")
        
        start_time = time.time()
        result = solver.solve()
        execution_time = time.time() - start_time
        
        # Execution time should scale reasonably
        expected_max_time = 5.0 * (size_factor ** 2)  # Quadratic scaling expectation
        assert execution_time < expected_max_time, (
            f"Scaling performance issue: {execution_time:.2f}s > {expected_max_time:.2f}s "
            f"for size_factor={size_factor}"
        )
```

### 5. **Integration Testing** (Medium Coverage ~70%)

```python
# tests/integration/test_end_to_end.py
import pytest
from mfg_pde import ExampleMFGProblem, create_research_solver
from mfg_pde.config.pydantic_config import create_research_config

class TestEndToEndWorkflow:
    """Test complete workflows from problem setup to solution."""
    
    def test_complete_research_workflow(self):
        """Test full research workflow with modern API."""
        # Create problem
        problem = ExampleMFGProblem(Nx=25, Nt=12, T=1.0)
        
        # Create configuration
        config = create_research_config()
        
        # Create solver
        solver = create_research_solver(problem, "fixed_point", config=config)
        
        # Solve
        result = solver.solve()
        
        # Validate result
        assert result is not None
        assert hasattr(result, 'U_solution')
        assert hasattr(result, 'M_solution')
        assert result.U_solution.shape == (problem.Nt + 1, problem.Nx + 1)
        assert result.M_solution.shape == (problem.Nt + 1, problem.Nx + 1)
    
    def test_factory_pattern_consistency(self):
        """Test that different factory methods produce consistent results."""
        problem = ExampleMFGProblem(Nx=20, Nt=8)
        
        # Create solvers using different factory methods
        solver1 = create_fast_solver(problem, "fixed_point")
        solver2 = create_accurate_solver(problem, "fixed_point")
        
        # Both should solve successfully
        result1 = solver1.solve()
        result2 = solver2.solve()
        
        assert result1 is not None
        assert result2 is not None
        
        # Solutions should have same dimensions
        assert result1.U_solution.shape == result2.U_solution.shape
        assert result1.M_solution.shape == result2.M_solution.shape
```

## 🛠️ **Testing Infrastructure Setup**

### 1. **Pytest Configuration** (`pytest.ini`):

```ini
[tool:pytest]
minversion = 6.0
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --tb=short
    --cov=mfg_pde
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-fail-under=80
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (slower, cross-component)
    performance: Performance tests (may be slow)
    mathematical: Mathematical property validation
    slow: Slow tests (may take >10 seconds)
filterwarnings =
    ignore::DeprecationWarning:mfg_pde.*
    ignore::PendingDeprecationWarning
```

### 2. **Test Fixtures** (`tests/conftest.py`):

```python
import pytest
import numpy as np
from mfg_pde import ExampleMFGProblem
from mfg_pde.config.pydantic_config import create_fast_config

@pytest.fixture
def small_problem():
    """Small problem for fast tests."""
    return ExampleMFGProblem(Nx=10, Nt=5, T=0.5)

@pytest.fixture
def medium_problem():
    """Medium problem for integration tests."""
    return ExampleMFGProblem(Nx=25, Nt=12, T=1.0)

@pytest.fixture
def fast_config():
    """Fast configuration for testing."""
    return create_fast_config()

@pytest.fixture
def deterministic_arrays():
    """Deterministic arrays for reproducible tests."""
    np.random.seed(42)
    return {
        'U': np.random.rand(11, 21),
        'M': np.random.rand(11, 21),
    }

@pytest.fixture(scope="session")
def reference_solutions():
    """Load reference solutions for validation."""
    # Load pre-computed reference solutions
    return load_reference_data()

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "mathematical: Mathematical property validation tests"
    )
```

## 🚀 **Running Tests Effectively**

### **Command Examples:**

```bash
# Run all tests
pytest

# Run only unit tests (fast)
pytest tests/unit/

# Run tests with coverage
pytest --cov=mfg_pde --cov-report=html

# Run specific test categories
pytest -m "unit"                    # Only unit tests
pytest -m "not slow"               # Skip slow tests
pytest -m "mathematical"           # Only mathematical validation

# Run tests in parallel
pytest -n auto                     # Auto-detect CPU cores

# Run specific test file
pytest tests/unit/test_config/test_pydantic_config.py

# Run specific test method
pytest tests/unit/test_solvers/test_hjb_fdm.py::TestHJBFDMSolver::test_solver_initialization

# Run with verbose output
pytest -v -s

# Run performance tests
pytest -m performance --tb=short
```

### **Continuous Integration Integration:**

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e .[dev]
    
    - name: Run unit tests
      run: pytest tests/unit/ -v
    
    - name: Run integration tests
      run: pytest tests/integration/ -v
    
    - name: Run mathematical validation
      run: pytest tests/mathematical/ -v
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## 📊 **Testing Metrics and Goals**

### **Coverage Targets:**

| Component | Target Coverage | Justification |
|-----------|----------------|---------------|
| **Public APIs** | 95%+ | User-facing functionality |
| **Configuration** | 95%+ | Critical for reliability |
| **Core Solvers** | 85%+ | Mathematical correctness |
| **Utilities** | 80%+ | Supporting functionality |
| **Error Handling** | 90%+ | Robustness critical |
| **Factory Patterns** | 90%+ | Integration points |

### **Test Performance Goals:**

- **Unit tests**: <1 second each
- **Integration tests**: <10 seconds each
- **Full test suite**: <5 minutes
- **Mathematical validation**: <30 seconds total

## ⚠️ **Common Testing Pitfalls to Avoid**

### 1. **Over-Testing Implementation Details**
```python
# ❌ BAD: Testing private methods
def test_private_helper_function(self):
    solver = HJBFDMSolver(problem)
    result = solver._internal_helper(data)  # Don't test private methods

# ✅ GOOD: Testing public behavior
def test_solver_convergence(self):
    solver = HJBFDMSolver(problem)
    result = solver.solve()  # Test public interface
    assert result.convergence_info.converged
```

### 2. **Testing Random Numbers Without Seeds**
```python
# ❌ BAD: Non-deterministic tests
def test_random_initialization(self):
    problem = ExampleMFGProblem()  # Random initial condition
    solver = create_solver(problem)
    result = solver.solve()
    assert result.final_error < 1e-6  # May fail randomly

# ✅ GOOD: Deterministic tests
def test_convergence_behavior(self):
    np.random.seed(42)  # Fixed seed
    problem = ExampleMFGProblem()
    solver = create_solver(problem)
    result = solver.solve()
    assert result.final_error < 1e-6
```

### 3. **Slow Tests in Unit Test Suite**
```python
# ❌ BAD: Slow test in unit suite
def test_large_problem_solving(self):
    problem = ExampleMFGProblem(Nx=1000, Nt=500)  # Very large
    solver = create_solver(problem)
    result = solver.solve()  # Takes minutes

# ✅ GOOD: Use performance test marker
@pytest.mark.performance
def test_large_problem_performance(self):
    problem = ExampleMFGProblem(Nx=1000, Nt=500)
    # ... test logic
```

## 🎯 **Recommended Testing Schedule**

### **During Development:**
- Run unit tests frequently (`pytest tests/unit/`)
- Run affected integration tests when changing interfaces
- Use `pytest -x` to stop on first failure

### **Before Commits:**
- Run full test suite (`pytest`)
- Check coverage (`pytest --cov=mfg_pde`)
- Run mathematical validation tests

### **In CI/CD:**
- Run all tests on multiple Python versions
- Generate coverage reports
- Run performance regression tests

## 📋 **Testing Checklist**

### **For New Features:**
- [ ] Unit tests for all public methods
- [ ] Parameter validation tests
- [ ] Error condition tests
- [ ] Integration tests if applicable
- [ ] Mathematical property tests if applicable
- [ ] Documentation examples are tested

### **For Bug Fixes:**
- [ ] Regression test reproducing the bug
- [ ] Fix implementation
- [ ] Verify test passes
- [ ] Check no other tests break

### **For Performance Changes:**
- [ ] Benchmark before changes
- [ ] Implement changes
- [ ] Benchmark after changes
- [ ] Add performance regression test

## 🏆 **Best Practices Summary**

1. **Test Pyramid**: Many unit tests, fewer integration tests, minimal E2E tests
2. **Fast Feedback**: Unit tests should run in <1 second each
3. **Deterministic**: Use fixed seeds for reproducible tests
4. **Clear Names**: Test names should describe what is being tested
5. **Single Assertion**: Each test should validate one specific behavior
6. **Proper Fixtures**: Use fixtures for test data setup
7. **Mark Categories**: Use pytest markers to categorize tests
8. **Coverage Goals**: Aim for 80%+ overall, 95%+ for critical paths
9. **Documentation**: Include docstrings explaining complex test logic
10. **Maintenance**: Regularly review and update tests as code evolves

**Remember: Good tests are an investment in code quality, maintainability, and developer confidence. Test strategically based on risk and importance, not just for coverage numbers.**
