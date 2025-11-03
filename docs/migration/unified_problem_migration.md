# Migration Guide: Unified MFGProblem Interface

**Version**: MFG_PDE v0.9.0+
**Status**: Active Migration Path
**Deprecation Timeline**: v0.9.0 (warnings) → v2.0.0 (removal)

---

## Overview

MFG_PDE v0.9.0 introduces a unified `MFGProblem` class that supports all problem types through a single interface. The old `GridBasedMFGProblem` class has been converted to a factory function with deprecation warnings.

**Benefits of Migration**:
- Single unified API for all problem types
- Automatic solver compatibility detection
- Better error messages with specific recommendations
- Support for complex geometries and networks
- Simplified parameter interface with aliases

---

## Deprecation Timeline

| Version | Date | Status |
|:--------|:-----|:-------|
| v0.9.0 | 2025-Q1 | DeprecationWarning added |
| v1.0.0 | 2025-Q2 | Warning becomes prominent |
| v2.0.0 | 2026-Q1 | GridBasedMFGProblem removed |

---

## Quick Migration Examples

### 2D Grid-Based Problems

**Old API (deprecated)**:
```python
from mfg_pde import GridBasedMFGProblem

problem = GridBasedMFGProblem(
    domain_bounds=(0, 1, 0, 1),  # (xmin, xmax, ymin, ymax)
    grid_resolution=50,           # Uniform 50×50 grid
    time_domain=(1.0, 100),       # (T, Nt)
    diffusion_coeff=0.1
)
```

**New API (recommended)**:
```python
from mfg_pde import MFGProblem

problem = MFGProblem(
    spatial_bounds=[(0, 1), (0, 1)],  # List of (min, max) per dimension
    spatial_discretization=[50, 50],   # List of grid points per dimension
    T=1.0,
    Nt=100,
    sigma=0.1
)
```

### 3D Grid-Based Problems

**Old API (deprecated)**:
```python
problem = GridBasedMFGProblem(
    domain_bounds=(0, 1, 0, 1, 0, 1),  # (xmin, xmax, ymin, ymax, zmin, zmax)
    grid_resolution=30,                 # Uniform 30×30×30 grid
    time_domain=(1.0, 50),
    diffusion_coeff=0.1
)
```

**New API (recommended)**:
```python
problem = MFGProblem(
    spatial_bounds=[(0, 1), (0, 1), (0, 1)],
    spatial_discretization=[30, 30, 30],
    T=1.0,
    Nt=50,
    sigma=0.1
)
```

### Non-Uniform Grids

**Old API (deprecated)**:
```python
problem = GridBasedMFGProblem(
    domain_bounds=(0, 2, 0, 1),
    grid_resolution=(100, 50),  # 100×50 grid
    time_domain=(1.0, 100),
    diffusion_coeff=0.05
)
```

**New API (recommended)**:
```python
problem = MFGProblem(
    spatial_bounds=[(0, 2), (0, 1)],
    spatial_discretization=[100, 50],
    T=1.0,
    Nt=100,
    sigma=0.05
)
```

---

## Parameter Mapping

| Old Parameter | New Parameter | Notes |
|:--------------|:--------------|:------|
| `domain_bounds` | `spatial_bounds` | Changed from flat tuple to list of tuples |
| `grid_resolution` | `spatial_discretization` | Changed from int/tuple to list |
| `time_domain` | `T`, `Nt` | Unpacked into separate parameters |
| `diffusion_coeff` | `sigma` | Renamed to match mathematical notation |

### Parameter Format Differences

**domain_bounds → spatial_bounds**:
```python
# Old: Flat tuple (2*d values)
domain_bounds = (xmin, xmax, ymin, ymax, zmin, zmax)

# New: List of tuples (d pairs)
spatial_bounds = [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
```

**grid_resolution → spatial_discretization**:
```python
# Old: int (uniform) or tuple
grid_resolution = 50              # Uniform
grid_resolution = (50, 30, 40)    # Non-uniform

# New: list (always)
spatial_discretization = [50, 50, 50]    # Uniform
spatial_discretization = [50, 30, 40]    # Non-uniform
```

---

## New Features in Unified Interface

### 1. Parameter Aliases (Convenience)

```python
# Alternative time specification
problem = MFGProblem(
    spatial_bounds=[(0, 1), (0, 1)],
    spatial_discretization=[50, 50],
    time_domain=(1.0, 100),  # Alternative to T/Nt
    sigma=0.1
)

# Alternative diffusion parameter
problem = MFGProblem(
    spatial_bounds=[(0, 1), (0, 1)],
    spatial_discretization=[50, 50],
    T=1.0, Nt=100,
    diffusion=0.1  # Alternative to sigma
)

# Alternative 1D domain (centered at 0)
problem = MFGProblem(
    Lx=2.0,  # Domain [-1, 1] with length 2
    Nx=100,
    T=1.0, Nt=50,
    sigma=0.1
)
```

### 2. Solver Compatibility Detection

```python
problem = MFGProblem(
    spatial_bounds=[(0, 1), (0, 1)],
    spatial_discretization=[50, 50],
    T=1.0, Nt=100, sigma=0.1
)

# Automatic compatibility detection
print(problem.solver_compatible)
# ['fdm', 'semi_lagrangian', 'gfdm', 'particle']

print(problem.solver_recommendations)
# {'fast': 'fdm', 'accurate': 'fdm'}

# Get comprehensive info
info = problem.get_solver_info()
print(info['default_solver'])  # 'fdm'
```

### 3. Helpful Error Messages

```python
problem = MFGProblem(
    spatial_bounds=[(0, 1), (0, 1), (0, 1), (0, 1)],  # 4D
    spatial_discretization=[10, 10, 10, 10],
    T=1.0, Nt=50, sigma=0.1
)

# This will fail with helpful message:
try:
    problem.validate_solver_type('fdm')
except ValueError as e:
    print(e)
    # Solver type 'fdm' is incompatible with this problem.
    #
    # Problem Configuration:
    #   Domain type: grid
    #   Dimension: 4
    #   Has obstacles: False
    #
    # Reason: FDM only supports dimensions 1-3 (got dimension 4)
    #
    # Compatible solvers: ['semi_lagrangian', 'gfdm', 'particle']
    #
    # Suggestion: Use 'particle' solver for high-dimensional problems (d≥4)
```

---

## Migration Strategies

### Strategy 1: Gradual Migration (Recommended)

Migrate file-by-file, suppressing warnings temporarily:

```python
import warnings

# Suppress deprecation warnings during migration
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    problem = GridBasedMFGProblem(...)  # Old API still works
```

### Strategy 2: Automated Migration Script

Use this script to automatically update your code:

```python
# scripts/migrate_problem_api.py
import re
import sys
from pathlib import Path

def migrate_file(file_path: Path) -> None:
    """Migrate GridBasedMFGProblem calls to MFGProblem."""
    content = file_path.read_text()

    # Pattern: GridBasedMFGProblem(domain_bounds=..., grid_resolution=..., ...)
    pattern = r'GridBasedMFGProblem\(\s*domain_bounds\s*=\s*\(([^)]+)\)\s*,\s*grid_resolution\s*=\s*([^,\)]+)\s*(?:,\s*time_domain\s*=\s*\(([^)]+)\))?\s*(?:,\s*diffusion_coeff\s*=\s*([^,\)]+))?\s*\)'

    def replace_call(match):
        bounds_str = match.group(1)
        resolution_str = match.group(2).strip()
        time_str = match.group(3) if match.group(3) else "1.0, 100"
        diffusion_str = match.group(4).strip() if match.group(4) else "0.1"

        # Parse bounds
        bounds = [x.strip() for x in bounds_str.split(',')]
        dimension = len(bounds) // 2

        # Convert bounds to list of tuples
        spatial_bounds = ', '.join(
            f'({bounds[2*i]}, {bounds[2*i+1]})'
            for i in range(dimension)
        )

        # Convert resolution
        if resolution_str.startswith('('):
            # Tuple format
            spatial_disc = resolution_str.replace('(', '[').replace(')', ']')
        else:
            # Uniform int
            spatial_disc = f'[{resolution_str}] * {dimension}'

        # Parse time domain
        T, Nt = time_str.split(',')

        return (
            f'MFGProblem(\n'
            f'    spatial_bounds=[{spatial_bounds}],\n'
            f'    spatial_discretization={spatial_disc},\n'
            f'    T={T.strip()}, Nt={Nt.strip()},\n'
            f'    sigma={diffusion_str}\n'
            f')'
        )

    migrated = re.sub(pattern, replace_call, content)

    if migrated != content:
        file_path.write_text(migrated)
        print(f"Migrated: {file_path}")

if __name__ == "__main__":
    # Usage: python scripts/migrate_problem_api.py path/to/file.py
    for path_str in sys.argv[1:]:
        migrate_file(Path(path_str))
```

### Strategy 3: Wrapper Function (Temporary)

Create a temporary wrapper during migration:

```python
# utils/migration_helpers.py
import warnings
from mfg_pde import MFGProblem

def create_grid_problem_legacy(
    domain_bounds: tuple,
    grid_resolution: int | tuple,
    time_domain: tuple = (1.0, 100),
    diffusion_coeff: float = 0.1,
) -> MFGProblem:
    """
    Legacy wrapper for GridBasedMFGProblem.
    Use during migration period, then remove.
    """
    warnings.warn("Using legacy wrapper - migrate to MFGProblem directly", FutureWarning)

    dimension = len(domain_bounds) // 2
    spatial_bounds = [(domain_bounds[2*i], domain_bounds[2*i+1]) for i in range(dimension)]

    if isinstance(grid_resolution, int):
        spatial_discretization = [grid_resolution] * dimension
    else:
        spatial_discretization = list(grid_resolution)

    T, Nt = time_domain

    return MFGProblem(
        spatial_bounds=spatial_bounds,
        spatial_discretization=spatial_discretization,
        T=T, Nt=Nt, sigma=diffusion_coeff
    )
```

---

## Testing After Migration

### 1. Verify Problem Attributes

```python
# After migration, verify key attributes match
problem_old = GridBasedMFGProblem(
    domain_bounds=(0, 1, 0, 1),
    grid_resolution=50,
    time_domain=(1.0, 100),
    diffusion_coeff=0.1
)

problem_new = MFGProblem(
    spatial_bounds=[(0, 1), (0, 1)],
    spatial_discretization=[50, 50],
    T=1.0, Nt=100, sigma=0.1
)

# These should match
assert problem_old.dimension == problem_new.dimension
assert problem_old.T == problem_new.T
assert problem_old.Nt == problem_new.Nt
assert problem_old.sigma == problem_new.sigma
```

### 2. Check Solver Compatibility

```python
# Test solver creation
from mfg_pde.factory import create_fast_solver

solver = create_fast_solver(problem_new, solver_type='fdm')
result = solver.solve()

assert result is not None
```

### 3. Numerical Results Comparison

```python
# Compare solutions (should be identical)
solver_old = create_fast_solver(problem_old, solver_type='fdm')
solver_new = create_fast_solver(problem_new, solver_type='fdm')

result_old = solver_old.solve()
result_new = solver_new.solve()

import numpy as np
assert np.allclose(result_old.U, result_new.U, atol=1e-12)
```

---

## Troubleshooting

### Issue 1: Import Errors

**Problem**: `AttributeError: module 'mfg_pde' has no attribute 'GridBasedMFGProblem'`

**Solution**: Update MFG_PDE to v0.9.0+:
```bash
pip install --upgrade mfg_pde>=0.9.0
```

### Issue 2: Solver Not Found

**Problem**: "Solver type 'X' is incompatible with this problem"

**Solution**: Check compatibility and use recommended solver:
```python
info = problem.get_solver_info()
print(f"Compatible solvers: {info['compatible']}")
print(f"Recommended: {info['recommendations']}")
```

### Issue 3: Attribute Access

**Problem**: Code accesses `problem.domain_bounds` which doesn't exist in new API

**Solution**: For backward compatibility, GridBasedMFGProblem factory adds these attributes:
```python
problem = GridBasedMFGProblem(...)  # Factory function
# problem.domain_bounds still exists for compatibility
```

For new code, compute from `spatial_bounds`:
```python
problem = MFGProblem(spatial_bounds=[(0, 1), (0, 1)], ...)
# Convert to old format if needed:
domain_bounds = tuple(x for bounds in problem.spatial_bounds for x in bounds)
```

---

## Support

For migration assistance:
- **Documentation**: `docs/architecture/unified_problem_design.md`
- **Issues**: https://github.com/derrring/MFG_PDE/issues
- **Examples**: See `examples/basic/` for updated examples

---

**Last Updated**: 2025-11-03
**Document Version**: 1.0
**Related**: PHASE_3_1_UNIFIED_PROBLEM_DESIGN.md, ARCHITECTURE_REFACTORING_PLAN_2025-11-02.md
