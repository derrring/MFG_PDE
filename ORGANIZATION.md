# MFG_PDE Repository Organization

## Directory Structure

```
MFG_PDE/
├── README.md                    # Project overview and quick start
├── CONTRIBUTING.md             # Development guidelines
├── CLAUDE.md                   # AI interaction preferences
├── ORGANIZATION.md             # This file - repository structure guide
├── pyproject.toml              # Modern Python packaging
├── setup.py                    # Legacy packaging support
├── environment.yml             # General development environment
├── mfg_env_pde.yml            # Primary NumPy 2.0+ environment  
├── conda_performance.yml      # Performance-optimized environment
├── mypy.ini                   # Type checking configuration
├── pytest.ini                # Testing configuration
│
├── scripts/                   # Utility scripts
│   ├── README.md             # Script documentation
│   ├── manage_environments.sh # Multi-environment management
│   └── setup_env_vars.sh     # Environment variable setup
│
├── mfg_pde/                  # Core package
│   ├── __init__.py           # Package initialization
│   ├── core/                 # Core problem definitions
│   ├── alg/                  # Numerical algorithms
│   ├── geometry/             # Computational geometry
│   ├── factory/              # Solver factories
│   ├── config/               # Configuration management
│   ├── utils/                # Utilities and helpers
│   ├── visualization/        # Plotting and visualization
│   ├── backends/             # Computational backends (NumPy/JAX/Numba)
│   ├── workflow/             # Experiment management
│   ├── accelerated/          # High-performance implementations
│   └── meta/                 # Meta-programming framework
│
├── examples/                 # Demonstration code
│   ├── README.md            # Examples overview
│   ├── basic/               # Simple single-concept examples
│   ├── advanced/            # Complex multi-feature demonstrations
│   ├── notebooks/           # Jupyter notebook examples
│   └── plugins/             # Plugin development examples
│
├── tests/                   # Test suite
│   ├── README.md           # Testing guide
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   ├── property_based/     # Property-based testing
│   ├── boundary_conditions/ # Boundary condition tests
│   └── mathematical/       # Mathematical property verification
│
├── benchmarks/             # Performance evaluation
│   ├── README.md          # Benchmarking guide
│   ├── solver_comparisons/ # Solver performance analysis
│   └── amr_evaluation/    # Adaptive mesh refinement benchmarks
│
├── docs/                  # Documentation
│   ├── README.md         # Documentation overview
│   ├── user/             # User guides and tutorials
│   ├── development/      # Development documentation
│   ├── theory/           # Mathematical background
│   ├── api/              # API reference
│   ├── tutorials/        # Step-by-step tutorials
│   └── reference/        # Quick reference materials
│
└── archive/              # Historical code and analysis
    ├── ARCHIVE_SUMMARY.md # Archive contents overview
    ├── old_examples/     # Superseded examples
    ├── superseded_tests/ # Replaced test implementations
    ├── obsolete_solvers/ # Deprecated solver implementations
    └── old_docs/         # Historical documentation
```

## Key Principles

### 1. Separation of Concerns
- **Core (`mfg_pde/`)**: Production code only
- **Examples (`examples/`)**: Demonstration and educational code
- **Tests (`tests/`)**: Verification and validation
- **Benchmarks (`benchmarks/`)**: Performance analysis
- **Documentation (`docs/`)**: User and developer guides
- **Archive (`archive/`)**: Historical preservation

### 2. Complexity Stratification
- **Basic Examples**: Single-concept demonstrations
- **Advanced Examples**: Multi-feature integrations
- **Tutorials**: Progressive learning materials
- **Reference**: Quick lookup information

### 3. Clean Root Directory
- Essential configuration files only
- No temporary outputs or analysis files
- Scripts organized in dedicated directory
- Clear separation of development vs. distribution files

### 4. Type Safety and Modern Python
- Type hints throughout codebase (`mypy.ini`)
- Modern packaging (`pyproject.toml`)
- Comprehensive testing (`pytest.ini`)
- Environment management (multiple `.yml` files)

## Navigation Guide

### For Users
1. **Start**: `README.md` → `docs/user/` → `examples/basic/`
2. **Learn**: `docs/tutorials/` → `examples/advanced/`
3. **Reference**: `docs/api/` → `docs/reference/`

### For Developers
1. **Setup**: `CONTRIBUTING.md` → environment files → `scripts/`
2. **Architecture**: `docs/development/` → `mfg_pde/` source
3. **Testing**: `tests/README.md` → test categories
4. **Performance**: `benchmarks/` → optimization guides

### For Researchers
1. **Theory**: `docs/theory/` → mathematical background
2. **Examples**: `examples/notebooks/` → research demonstrations
3. **Extensions**: `examples/plugins/` → custom development
4. **Meta-Programming**: `mfg_pde/meta/` → automatic code generation

## File Naming Conventions

### Code Files
- **Classes**: `PascalCase` (e.g., `MFGProblem`)
- **Functions**: `snake_case` (e.g., `create_solver`)
- **Modules**: `snake_case` (e.g., `mfg_problem.py`)
- **Constants**: `UPPER_CASE` (e.g., `DEFAULT_CONFIG`)

### Documentation
- **Guides**: `TITLE_CASE.md` (e.g., `INSTALLATION_GUIDE.md`)
- **References**: `lowercase.md` (e.g., `api_reference.md`)
- **Status Indicators**: `[STATUS]_filename.md` (e.g., `[COMPLETED]_feature.md`)

### Examples and Tests
- **Examples**: Descriptive names (e.g., `lagrangian_vs_hamiltonian_example.py`)
- **Tests**: `test_` prefix (e.g., `test_mfg_problem.py`)
- **Benchmarks**: `benchmark_` or descriptive (e.g., `solver_performance_analysis.py`)

## Maintenance Guidelines

### Regular Cleanup
- Archive obsolete examples and tests rather than deleting
- Update status indicators in documentation files
- Maintain clean root directory (no temporary files)
- Organize scripts and utilities in dedicated directories

### Version Management
- Use semantic versioning for releases
- Tag important development milestones
- Maintain backward compatibility in public APIs
- Document breaking changes in upgrade guides

### Quality Assurance
- All code passes type checking (`mypy`)
- Comprehensive test coverage (unit + integration)
- Performance benchmarks for critical paths
- Documentation updated with code changes

This organizational structure supports both rapid development and long-term maintainability while providing clear pathways for users, developers, and researchers to engage with the MFG_PDE framework.