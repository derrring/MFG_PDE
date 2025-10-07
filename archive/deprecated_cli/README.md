# Deprecated CLI Interface

**Date Removed**: October 7, 2025
**Reason**: CLI interface removed to focus on Python API
**Issue**: #108

## Why Removed

The command-line interface (`mfg-pde`) was removed because:

1. **Low utility**: MFG problems are complex and require Python code, not simple CLI commands
2. **Dependency overhead**: Adds `click` dependency without proportional value
3. **Ecosystem alignment**: Scientific packages (numpy, scipy, mfglib) don't provide CLIs
4. **Maintenance burden**: Additional code to maintain with minimal usage
5. **Focus**: Package strength is Python API, not CLI

## Migration

**Before** (deprecated):
```bash
mfg-pde solve --problem lq_mfg --nx 50 --nt 20
```

**After** (recommended):
```python
from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import create_fast_solver

problem = ExampleMFGProblem(T=1.0, xmin=0, xmax=1, Nx=50, Nt=40)
solver = create_fast_solver(problem)
result = solver.solve()
```

## Archive Contents

- `cli.py`: Original CLI implementation (preserved for reference)

This directory is kept for historical reference only. The code is no longer maintained or supported.
