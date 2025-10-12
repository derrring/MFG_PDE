# Migration Notice: Particle-Collocation Methods

**Date**: 2025-10-12
**Affected Version**: Post v0.3.x
**Migration Type**: Code Relocation

## Summary

Particle-collocation methods have been **moved from MFG_PDE to the mfg-research repository** as part of the architectural separation between stable infrastructure and novel research algorithms.

## What Changed

### Removed from MFG_PDE

The following have been removed from the core `mfg_pde` package:

1. **Solver Implementation**
   - `mfg_pde/alg/numerical/mfg_solvers/particle_collocation_solver.py` (removed)

2. **Factory Support**
   - `solver_type="particle_collocation"` removed from factory functions
   - `create_monitored_solver()` removed
   - Pydantic factory support for particle-collocation removed

3. **High-Dimensional Problem Methods**
   - `HighDimMFGProblem.solve_with_particle_collocation()` method removed
   - Hybrid strategies simplified to use only fixed-point methods

4. **CLI and Configuration**
   - `--solver-type particle_collocation` CLI option removed
   - Plugin system no longer lists particle_collocation as core solver

### New Location: mfg-research Repository

Particle-collocation methods are now available in the **mfg-research** repository at:

```
mfg-research/
└── algorithms/
    └── particle_collocation/
        ├── __init__.py
        ├── solver.py                  # ParticleCollocationSolver
        ├── tests/
        │   └── test_basic.py
        └── README.md
```

## Migration Guide

### For Users

If you were using particle-collocation methods, you have two options:

#### Option 1: Use mfg-research Repository (Recommended)

```bash
# Clone mfg-research alongside MFG_PDE
cd /path/to/your/projects
git clone https://github.com/yourusername/mfg-research.git

# Install MFG_PDE as usual
pip install -e MFG_PDE

# Use particle-collocation from mfg-research
```

**Updated Code:**

```python
# OLD CODE (no longer works):
from mfg_pde.factory import create_solver
solver = create_solver(problem, solver_type="particle_collocation")

# NEW CODE:
# Add mfg-research to Python path
import sys
from pathlib import Path
sys.path.insert(0, str(Path("/path/to/mfg-research")))

# Import from mfg-research
from algorithms.particle_collocation import ParticleCollocationSolver
import numpy as np

collocation_points = np.linspace(problem.xmin, problem.xmax, problem.Nx).reshape(-1, 1)
solver = ParticleCollocationSolver(
    problem=problem,
    collocation_points=collocation_points,
    num_particles=5000,
    use_qp_constraints=False  # or True for QP-enhanced version
)
```

#### Option 2: Use Fixed-Point Methods (Stable Alternative)

For production use, switch to the stable fixed-point iterator:

```python
from mfg_pde.factory import create_solver

# Use fixed_point solver (stable, well-tested)
solver = create_solver(
    problem=problem,
    solver_type="fixed_point",
    preset="accurate"
)
```

### For Developers

If you have code that depends on particle-collocation:

1. **Factory Code**: Update `solver_type` to `"fixed_point"`
2. **Direct Imports**: Import from mfg-research instead
3. **Tests**: Update or skip tests that use particle-collocation
4. **Configuration Files**: Remove `particle_collocation` from solver type lists

### High-Dimensional Problems

The `HighDimMFGProblem` class has been simplified:

```python
# OLD CODE (no longer works):
result = problem.solve_with_particle_collocation(num_particles=5000)

# NEW CODE - Use damped fixed point:
result = problem.solve_with_damped_fixed_point(
    damping_factor=0.6,
    max_iterations=30,
    tolerance=1e-4
)

# Or use adaptive strategy (two-phase fixed-point):
result = problem.solve(
    strategy="adaptive",  # Uses two-phase fixed-point
    max_iterations=50,
    tolerance=1e-4
)
```

## Rationale

### Why the Change?

1. **Architectural Clarity**: Separation between stable infrastructure (MFG_PDE) and novel research algorithms (mfg-research)

2. **Development Velocity**: Research algorithms can evolve rapidly without impacting stable package

3. **Dependency Management**: Experimental methods can have different dependencies without bloating core package

4. **Academic Focus**: mfg-research better reflects the experimental nature of novel algorithms

### What This Means

- **MFG_PDE**: Stable, production-ready infrastructure with classical methods
- **mfg-research**: Novel algorithms under active research and development

## Timeline

- **2025-10-08**: Particle-collocation successfully integrated into mfg-research
- **2025-10-12**: Particle-collocation removed from MFG_PDE (this migration)

## Support

### Questions?

- **Issues with migration**: Open an issue in MFG_PDE repository
- **Particle-collocation bugs**: Open an issue in mfg-research repository
- **General questions**: Discussion forum or email maintainer

### Documentation

- **MFG_PDE Documentation**: Core package functionality
- **mfg-research/algorithms/particle_collocation/README.md**: Particle-collocation usage guide
- **mfg-research/docs/methods/particle_collocation.md**: Method documentation

## Backward Compatibility

⚠️ **Breaking Change**: This is a breaking change for code that uses particle-collocation methods.

**Affected Code**: Any code using `solver_type="particle_collocation"` or calling `solve_with_particle_collocation()`.

**Migration Required**: Yes, follow the migration guide above.

## Future Plans

- Particle-collocation methods will continue development in mfg-research
- QP-enhanced versions remain in experimental phase
- No plans to reintegrate into MFG_PDE unless methods become widely adopted and fully validated

## Related Changes

See also:
- [Architectural Changes Documentation](docs/development/ARCHITECTURAL_CHANGES.md)
- [Strategic Development Roadmap](docs/development/STRATEGIC_DEVELOPMENT_ROADMAP_2026.md)
- [mfg-research Repository](https://github.com/yourusername/mfg-research)
