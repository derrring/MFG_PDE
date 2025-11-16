# Dependency Management Guide

## Overview

MFG_PDE provides clear error messages and diagnostics for optional dependencies through `mfg_pde/utils/dependencies.py`.

## Core Utilities

### `check_dependency()`

Check if an optional dependency is available with helpful error messages.

```python
from mfg_pde.utils.dependencies import check_dependency

# Raise ImportError with install instructions if missing
check_dependency('torch', purpose='neural network solvers')

# Return bool instead of raising
has_torch = check_dependency('torch', raise_on_missing=False)
if has_torch:
    import torch
    # Use PyTorch features

# Custom install command
check_dependency('package', install_command='conda install package')
```

### `show_optional_features()`

Display status of all optional features with version information.

```python
from mfg_pde import show_optional_features

show_optional_features()
```

**Output**:
```
MFG_PDE Optional Features Status
================================================================================

Core (always available):
  ✓ numpy 2.3.3 - Array operations and numerical computing
  ✓ scipy 1.16.2 - Scientific computing and optimization
  ✓ matplotlib 3.10.6 - Plotting and visualization

Neural Methods:
  ✓ torch 2.8.0 - Deep learning backends, RL algorithms, GPU acceleration
  ✗ jax - JAX backend, autodiff, GPU kernels (pip install mfg-pde[performance])

Visualization:
  ✓ plotly 6.3.0 - Interactive visualizations, 3D plots
  ✗ bokeh - Interactive plots, dashboards (pip install mfg-pde[visualization])

GPU Acceleration:
  ✗ cupy - GPU arrays and operations (pip install mfg-pde[gpu])
...
```

### `is_available()`

Simple check without error messages.

```python
from mfg_pde.utils.dependencies import is_available

if is_available('plotly'):
    import plotly.graph_objects as go
    # Use Plotly features
else:
    # Fallback to matplotlib
    import matplotlib.pyplot as plt
```

### `@require_dependencies` Decorator

Decorator to require dependencies for a function or class.

```python
from mfg_pde.utils.dependencies import require_dependencies

@require_dependencies('torch', purpose='Mean Field DDPG')
def create_ddpg_agent(env):
    import torch
    # Implementation uses PyTorch
    ...

@require_dependencies('jax', 'jaxlib')
class JAXBackend:
    # Implementation uses JAX
    ...
```

## Import Patterns

### Module-Level Availability Flags

Use pre-checked flags for performance:

```python
from mfg_pde.utils.dependencies import TORCH_AVAILABLE, PLOTLY_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    # Use PyTorch

if PLOTLY_AVAILABLE:
    import plotly.graph_objects as go
    # Use Plotly
```

Available flags:
- `TORCH_AVAILABLE`
- `JAX_AVAILABLE`
- `GYMNASIUM_AVAILABLE`
- `PLOTLY_AVAILABLE`
- `BOKEH_AVAILABLE`
- `POLARS_AVAILABLE`
- `NUMBA_AVAILABLE`
- `CUPY_AVAILABLE`
- `IGRAPH_AVAILABLE`
- `NETWORKX_AVAILABLE`

### Try-Except Pattern

For optional dependencies, always use try-except:

```python
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Later in code
if PLOTLY_AVAILABLE:
    # Use Plotly
else:
    # Fallback or raise helpful error
    from mfg_pde.utils.dependencies import check_dependency
    check_dependency('plotly', purpose='interactive visualizations')
```

### Runtime Checks

Check dependencies when features are actually used:

```python
def create_interactive_plot(data):
    """Create interactive plot (requires Plotly)."""
    from mfg_pde.utils.dependencies import check_dependency

    check_dependency('plotly', purpose='interactive visualizations')

    import plotly.graph_objects as go
    # Create plot
    ...
```

## Dependency Groups

### Core (always required)
- numpy
- scipy
- matplotlib
- tqdm
- omegaconf

### Neural (`pip install mfg-pde[neural]`)
- torch

### Reinforcement (`pip install mfg-pde[reinforcement]`)
- gymnasium

### Numerical (`pip install mfg-pde[numerical]`)
- igraph
- networkx

### Visualization (`pip install mfg-pde[visualization]`)
- plotly
- bokeh

### Performance (`pip install mfg-pde[performance]`)
- jax
- polars
- numba

### GPU (`pip install mfg-pde[gpu]`)
- cupy

### All (`pip install mfg-pde[all]`)
- All optional groups

## Error Message Format

When a dependency is missing, users see:

```
ImportError: torch required for neural network solvers.

Install options:
  1. pip install mfg-pde[neural]
  2. pip install torch
```

## Adding New Optional Dependencies

1. **Add to DEPENDENCY_MAP** in `mfg_pde/utils/dependencies.py`:

```python
DEPENDENCY_MAP = {
    # ...
    "new_package": {
        "install_group": "performance",
        "install_cmd": "pip install mfg-pde[performance]",
        "alternative": "pip install new_package",
        "description": "What this package does",
        "required": False,
    },
}
```

2. **Add availability flag**:

```python
# At module level in dependencies.py
NEW_PACKAGE_AVAILABLE = is_available("new_package")
```

3. **Update `get_available_features()`**:

```python
def get_available_features() -> dict[str, bool]:
    return {
        # ...
        "new_package": is_available("new_package"),
    }
```

4. **Use in code**:

```python
from mfg_pde.utils.dependencies import check_dependency

def feature_using_new_package():
    check_dependency('new_package', purpose='specific feature')
    import new_package
    # Use package
```

## Best Practices

1. **Always use clear purpose descriptions**:
   ```python
   check_dependency('torch', purpose='neural network solvers')  # ✅ Good
   check_dependency('torch')  # ⚠️ Less helpful
   ```

2. **Check at usage time, not import time**:
   ```python
   # ✅ Good - check when feature is used
   def train_neural_solver():
       check_dependency('torch', purpose='neural network training')
       import torch
       # ...

   # ❌ Bad - fails at import for users who don't need this
   import torch  # Unchecked at module level
   ```

3. **Provide fallbacks when possible**:
   ```python
   if is_available('plotly'):
       # Use interactive Plotly visualization
   else:
       # Fallback to static matplotlib
   ```

4. **Use `raise_on_missing=False` for optional features**:
   ```python
   if check_dependency('cupy', raise_on_missing=False):
       # Use GPU acceleration
   else:
       # Use CPU fallback
   ```

## Troubleshooting

**Q: How do I know what features are available?**

```python
from mfg_pde import show_optional_features
show_optional_features()
```

**Q: How do I install all optional dependencies?**

```bash
pip install mfg-pde[all]
```

**Q: How do I check if a specific feature is available?**

```python
from mfg_pde.utils.dependencies import is_available
print(f"PyTorch available: {is_available('torch')}")
```

**Q: I get "ImportError" even though the package is installed**

Check the package name matches what you `import`:
```python
# Package name: python-igraph, import name: igraph
check_dependency('igraph')  # Correct
check_dependency('python-igraph')  # Wrong
```

## See Also

- Issue #278 (Dependency Management)
- `mfg_pde/utils/dependencies.py` (implementation)
- `tests/unit/test_utils/test_dependencies.py` (tests)
- CLAUDE.md (Development Guidelines)

---

**Last Updated**: 2025-11-11
**Status**: Active dependency management system
