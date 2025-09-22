# Using UV with MFG_PDE

This document explains how to use the modern `uv` package manager with MFG_PDE instead of conda.

## Why UV?

- **Fast**: 10-100x faster than pip/conda for package resolution and installation
- **Modern**: Uses the latest Python packaging standards (PEP 621, 517, 518)
- **Deterministic**: Lock files ensure reproducible environments
- **Simple**: Single tool for package management, virtual environments, and project management

## Installation

Install uv:
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

## Quick Start

### 1. Basic Development Environment
```bash
# Create virtual environment and install all dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

### 2. Install with Specific Feature Sets
```bash
# Development tools
uv sync --extra dev

# Performance optimization
uv sync --extra performance

# Documentation tools
uv sync --extra docs

# All features
uv sync --extra all

# Multiple features
uv sync --extra "dev,performance,docs"
```

### 3. GPU Acceleration
```bash
# Install with GPU support
uv sync --extra gpu
```

## Available Feature Sets

### Core Dependencies (Always Installed)
- **Scientific**: NumPy 2.0+, SciPy, Matplotlib, Pandas
- **Interactive**: Jupyter, JupyterLab, IPython
- **Visualization**: Plotly, Seaborn
- **Data**: H5py, Pydantic
- **Utilities**: tqdm, psutil, igraph

### Optional Feature Sets

#### `dev` - Development Tools
- pytest (testing)
- black (code formatting)
- isort (import sorting)
- mypy (type checking)
- pre-commit (git hooks)

#### `performance` - Performance Libraries
- numba (JIT compilation)
- jax/jaxlib (GPU acceleration)
- polars (fast data processing)
- memory-profiler, line-profiler

#### `performance-optimized` - Maximum Performance
- All performance libraries
- cython (C extensions)
- joblib (parallel processing)
- mpi4py (MPI support)
- zarr (cloud storage)

#### `docs` - Documentation
- sphinx (documentation generator)
- sphinx-rtd-theme
- nbsphinx (Jupyter notebook docs)

#### `viz` - Advanced Visualization
- bokeh (interactive plots)

#### `networks` - Network Analysis
- networkit (large-scale networks)
- networkx (comprehensive algorithms)

#### `gpu` - GPU Acceleration
- jax[cuda12_pip] (CUDA support)
- cupy (CUDA arrays)

#### `typing` - Type Support
- types-tqdm, types-setuptools, types-psutil

#### `all` - Everything
- Installs all optional dependencies

## Migration from Conda

### Old Conda Workflow
```bash
# Old way with conda
conda env create -f environment.yml
conda activate mfg_env
```

### New UV Workflow
```bash
# New way with uv (much faster)
uv sync --extra all
source .venv/bin/activate
```

### Performance Comparison
- **Conda**: ~2-5 minutes for full environment
- **UV**: ~10-30 seconds for full environment

## Common Commands

### Project Management
```bash
# Install the project in development mode
uv pip install -e .

# Add a new dependency
uv add numpy>=2.1

# Add a development dependency
uv add --dev pytest-mock

# Remove a dependency
uv remove outdated-package

# Update all dependencies
uv lock --upgrade
```

### Virtual Environment Management
```bash
# Create new environment with specific Python
uv venv --python 3.12

# Show environment info
uv pip list

# Install from requirements
uv pip install -r requirements.txt

# Generate requirements.txt
uv pip freeze > requirements.txt
```

### Running Scripts
```bash
# Run Python with uv environment
uv run python examples/basic/simple_mfg_demo.py

# Run with specific extras
uv run --extra performance python benchmarks/solver_comparison.py

# Run tests
uv run pytest

# Run with environment variables
uv run --env CUDA_VISIBLE_DEVICES=0 python examples/gpu_demo.py
```

## Lock File Management

UV creates `uv.lock` which ensures reproducible installs:

```bash
# Update lock file
uv lock

# Install exact versions from lock
uv sync --frozen

# Check for updates
uv lock --upgrade
```

## Integration with IDEs

### VS Code
Add to `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.terminal.activateEnvironment": true
}
```

### PyCharm
1. File → Settings → Project → Python Interpreter
2. Add Interpreter → Existing Environment
3. Select `.venv/bin/python`

## Troubleshooting

### Common Issues

1. **Missing `uv.lock`**: Run `uv lock` to generate
2. **Old virtual environment**: Delete `.venv/` and run `uv sync`
3. **Permission errors**: Check file permissions on `.venv/`
4. **Package conflicts**: Run `uv lock --upgrade` to resolve

### Performance Tips

1. **Use `--frozen`** for CI/CD: `uv sync --frozen`
2. **Cache location**: UV automatically caches packages in `~/.cache/uv/`
3. **Parallel installs**: UV installs packages in parallel by default

## Comparison with Previous Setup

| Feature | Conda (Old) | UV (New) |
|---------|-------------|----------|
| Install speed | 2-5 minutes | 10-30 seconds |
| Lock files | environment.yml | uv.lock |
| Reproducibility | Partial | Full |
| Dependency resolution | Slow | Fast |
| Virtual environments | Manual | Automatic |
| Cross-platform | Good | Excellent |
| Package ecosystem | Conda-forge | PyPI |

## CI/CD Integration

### GitHub Actions
```yaml
- name: Set up Python with UV
  uses: actions/setup-python@v4
  with:
    python-version: '3.12'

- name: Install UV
  run: curl -LsSf https://astral.sh/uv/install.sh | sh

- name: Install dependencies
  run: uv sync --frozen --extra dev

- name: Run tests
  run: uv run pytest
```

### Docker
```dockerfile
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project
COPY . /app
WORKDIR /app

# Install dependencies
RUN uv sync --frozen

# Run application
CMD ["uv", "run", "python", "main.py"]
```

## Legacy Support

The conda files are kept in `environments/` for reference:
- `environment.yml` → Main development environment
- `environments/performance.yml` → Performance-optimized build

To continue using conda:
```bash
conda env create -f environment.yml
```

However, **uv is recommended** for new development due to its speed and modern approach.