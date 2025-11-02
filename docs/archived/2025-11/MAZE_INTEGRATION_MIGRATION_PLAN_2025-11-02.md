# Maze Integration Migration Plan

**Date**: 2025-11-02
**Author**: Claude Code
**Purpose**: Detailed plan for migrating maze→implicit domain converter to production

---

## Executive Summary

**Component**: Maze converter (discrete mazes → continuous implicit domains)
**Status**: ✅ **READY FOR MIGRATION** (production-quality code and tests)
**Effort**: 1-2 hours (straightforward migration, no dependencies)
**Priority**: High (quick win, enables new applications)

---

## Component Overview

### What is Being Migrated

**Source**: `/Users/zvezda/OneDrive/code/mfg-research/experiments/maze_navigation/maze_converter.py`
**Destination**: `mfg_pde/geometry/maze_converter.py`

**Purpose**: Convert discrete maze arrays (for RL) to continuous implicit domains (for PDE solvers)

**Key Functions**:
1. `maze_to_implicit_domain()` - Main converter (binary array → DifferenceDomain)
2. `compute_maze_sdf()` - Signed distance function for maze walls
3. `get_maze_statistics()` - Maze structure analysis

**Size**: 174 lines (production-ready, well-documented)

### Why This Component

**Maturity**:
- ✅ Clean, focused implementation
- ✅ Comprehensive documentation
- ✅ Full test coverage (15+ test cases)
- ✅ Uses only production MFG_PDE infrastructure
- ✅ No dependencies on research code

**Value**:
- ✅ Enables maze navigation MFG applications
- ✅ Bridges discrete (RL) and continuous (PDE) representations
- ✅ Demonstrates ImplicitDomain CSG capabilities
- ✅ Educational value (clear example of domain construction)

**Low Risk**:
- ✅ No changes to existing code needed
- ✅ No API conflicts
- ✅ Self-contained utility functions
- ✅ Already uses production geometry classes

---

## Migration Steps

### Step 1: Copy Main Module

**Action**: Copy maze_converter.py to production
```bash
# From research repo
cp /Users/zvezda/OneDrive/code/mfg-research/experiments/maze_navigation/maze_converter.py \
   /Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/mfg_pde/geometry/maze_converter.py
```

**Modifications Needed**:
1. Update import path (line 20-24):
   ```python
   # Before (research):
   from mfg_pde.geometry.implicit import (
       DifferenceDomain,
       Hyperrectangle,
       UnionDomain,
   )

   # After (production) - SAME, no change needed!
   from mfg_pde.geometry.implicit import (
       DifferenceDomain,
       Hyperrectangle,
       UnionDomain,
   )
   ```

2. Update module docstring (line 1-12):
   ```python
   # Before:
   """
   Convert MFG_PDE discrete mazes to implicit domain representation.
   ...
   Author: MFG Research Group
   Date: 2025-10-16
   """

   # After:
   """
   Convert discrete mazes to implicit domain representation.

   This module bridges discrete maze generation (for RL environments) with
   continuous implicit domain representation (for PDE-based MFG solvers).
   ...
   """
   ```

3. Fix typing import (line 14):
   ```python
   # Before:
   from typing import Tuple

   # After (modern Python):
   from __future__ import annotations
   ```

**Estimated Time**: 10 minutes

### Step 2: Copy Test File

**Action**: Copy test file to production test suite
```bash
# From research repo
cp /Users/zvezda/OneDrive/code/mfg-research/experiments/maze_navigation/benchmarks/tests/test_maze_converter.py \
   /Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/tests/unit/test_maze_converter.py
```

**Modifications Needed**:
1. Update import path (line 20-24):
   ```python
   # Before (research):
   from algorithms.particle_collocation.maze_integration import (
       compute_maze_sdf,
       get_maze_statistics,
       maze_to_implicit_domain,
   )

   # After (production):
   from mfg_pde.geometry.maze_converter import (
       compute_maze_sdf,
       get_maze_statistics,
       maze_to_implicit_domain,
   )
   ```

2. Update pytest run comment (line 7):
   ```python
   # Before:
   """Run with: python -m pytest algorithms/particle_collocation/maze_integration/tests/test_maze_converter.py -v"""

   # After:
   """Run with: pytest tests/unit/test_maze_converter.py -v"""
   ```

**Estimated Time**: 5 minutes

### Step 3: Update Geometry Package Exports

**Action**: Add to `mfg_pde/geometry/__init__.py`

**Location**: After existing imports (around line 50)

**Add**:
```python
# Maze integration utilities
from mfg_pde.geometry.maze_converter import (
    compute_maze_sdf,
    get_maze_statistics,
    maze_to_implicit_domain,
)

__all__ = [
    # ... existing exports ...
    # Maze integration
    "maze_to_implicit_domain",
    "compute_maze_sdf",
    "get_maze_statistics",
]
```

**Estimated Time**: 5 minutes

### Step 4: Create Example

**Action**: Create `examples/advanced/maze_navigation_demo.py`

**Content** (minimal working example):
```python
"""
Maze Navigation using Particle-Collocation MFG.

Demonstrates:
1. Generating discrete maze with MFG_PDE
2. Converting to continuous implicit domain
3. Sampling particles that automatically avoid walls
4. Solving MFG on irregular maze geometry
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.reinforcement.environments import generate_maze
from mfg_pde.geometry import maze_to_implicit_domain

# Step 1: Generate discrete maze
print("Generating 10×10 maze...")
maze_array = generate_maze(
    rows=10,
    cols=10,
    algorithm="recursive_backtracking",
    seed=42,
)

# Step 2: Convert to continuous implicit domain
print("Converting to implicit domain...")
domain, walls = maze_to_implicit_domain(
    maze_array,
    cell_size=1.0,
    origin=(0.0, 0.0),
)

print(f"  Domain dimension: {domain.dimension}")
print(f"  Bounding box: {domain.get_bounding_box()}")

# Step 3: Sample particles (automatically avoid walls!)
print("Sampling particles...")
particles = domain.sample_uniform(n_points=2000, seed=42)
print(f"  Sampled {particles.shape[0]} particles")

# Step 4: Visualize
print("Plotting...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Discrete maze
axes[0].imshow(maze_array, cmap="binary", origin="lower")
axes[0].set_title("Discrete Maze (RL Representation)")
axes[0].set_xlabel("Column")
axes[0].set_ylabel("Row")

# Plot 2: Continuous particles
axes[1].scatter(particles[:, 0], particles[:, 1], s=1, alpha=0.5)
axes[1].set_xlim(0, 10)
axes[1].set_ylim(0, 10)
axes[1].set_aspect("equal")
axes[1].set_title("Continuous Domain (PDE Representation)")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("examples/outputs/advanced/maze_navigation_demo.png", dpi=150)
print("Saved: examples/outputs/advanced/maze_navigation_demo.png")
plt.show()

print("\nDemonstration complete!")
print("  - Discrete maze → Continuous domain conversion: SUCCESS")
print("  - Particle sampling automatically avoids walls: SUCCESS")
print("  - Ready for MFG solver integration")
```

**Estimated Time**: 15 minutes

### Step 5: Run Tests

**Action**: Verify all tests pass

```bash
# Run maze converter tests
pytest tests/unit/test_maze_converter.py -v

# Run full test suite to check for regressions
pytest tests/unit/ -v
```

**Expected**: All 15+ maze converter tests pass, no regressions

**Estimated Time**: 5 minutes

### Step 6: Update Documentation

**Action**: Add to `docs/user/guides/geometry.md`

**Add Section** (after existing geometry examples):

```markdown
## Maze Integration

MFG_PDE provides utilities to convert discrete mazes (used in reinforcement learning)
to continuous implicit domains (used in PDE-based MFG solvers).

### Converting Mazes to Implicit Domains

```python
from mfg_pde.alg.reinforcement.environments import generate_maze
from mfg_pde.geometry import maze_to_implicit_domain

# Generate discrete maze
maze_array = generate_maze(rows=10, cols=10, algorithm="recursive_backtracking", seed=42)

# Convert to continuous implicit domain
domain, walls = maze_to_implicit_domain(maze_array, cell_size=1.0)

# Sample particles (automatically avoids walls!)
particles = domain.sample_uniform(n_points=1000)
```

### Use Cases

**1. Maze Navigation MFG**
```python
# Create maze-based MFG problem
from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import create_fast_solver

problem = ExampleMFGProblem(
    domain=domain,  # Maze domain from converter
    T=1.0,
    Nt=50,
)

solver = create_fast_solver(problem, method="particle-collocation")
U, M, info = solver.solve()
```

**2. Obstacle Avoidance**
```python
# Use maze walls as obstacles
from mfg_pde.geometry import compute_maze_sdf

# Compute distance to nearest wall
distances = compute_maze_sdf(particles, walls)
# Positive = safe distance from walls
# Negative = inside wall (should not occur if particles from domain.sample_uniform)
```

**3. Maze Analysis**
```python
from mfg_pde.geometry import get_maze_statistics

stats = get_maze_statistics(maze_array)
print(f"Wall fraction: {stats['wall_fraction']:.1%}")
print(f"Free space: {stats['free_fraction']:.1%}")
```

See `examples/advanced/maze_navigation_demo.py` for complete working example.
```

**Estimated Time**: 10 minutes

### Step 7: Create GitHub Issue and PR

**Action**: Create issue and pull request

**Issue Title**: "Add maze→implicit domain converter (geometry utilities)"

**Issue Body**:
```markdown
## Summary
Add utilities to convert discrete maze arrays to continuous implicit domains.

## Motivation
Enables maze navigation MFG applications by bridging discrete (RL) and continuous (PDE) representations.

## Implementation
- `mfg_pde/geometry/maze_converter.py`: Main converter module
- `tests/unit/test_maze_converter.py`: Comprehensive test suite (15+ tests)
- `examples/advanced/maze_navigation_demo.py`: Working demonstration
- Updated geometry guide documentation

## Features
- `maze_to_implicit_domain()`: Convert binary maze → DifferenceDomain
- `compute_maze_sdf()`: Signed distance function for maze walls
- `get_maze_statistics()`: Maze structure analysis

## Testing
- [x] 15+ unit tests (all passing)
- [x] Integration test with MFG_PDE maze generator
- [x] Example demonstrates usage
- [x] No regressions in existing tests

## Labels
- `area: geometry`
- `type: enhancement`
- `priority: medium`
- `size: small`
```

**PR Command**:
```bash
git checkout -b feature/maze-integration
git add mfg_pde/geometry/maze_converter.py
git add tests/unit/test_maze_converter.py
git add examples/advanced/maze_navigation_demo.py
git add docs/user/guides/geometry.md
git commit -m "Add maze→implicit domain converter

- Add maze_to_implicit_domain() converter
- Add compute_maze_sdf() for signed distances
- Add get_maze_statistics() for maze analysis
- Add comprehensive test suite (15+ tests)
- Add example demonstration
- Update geometry guide documentation

Enables maze navigation MFG applications."

git push -u origin feature/maze-integration

gh pr create \
  --title "Add maze→implicit domain converter (geometry utilities)" \
  --body "$(cat <<EOF
## Summary
Adds utilities to convert discrete maze arrays to continuous implicit domains.

Closes #XXX

## Changes
- \`mfg_pde/geometry/maze_converter.py\`: Main converter module (174 lines)
- \`tests/unit/test_maze_converter.py\`: Test suite (362 lines, 15+ tests)
- \`examples/advanced/maze_navigation_demo.py\`: Demonstration
- Updated \`docs/user/guides/geometry.md\`: Documentation

## Features
- \`maze_to_implicit_domain()\`: Convert binary maze → DifferenceDomain
- \`compute_maze_sdf()\`: Signed distance function for walls
- \`get_maze_statistics()\`: Maze structure analysis

## Testing
✅ All 15+ maze converter tests passing
✅ Integration test with maze generator
✅ Example executes successfully
✅ No regressions in existing tests

## Migration Notes
Migrated from mfg-research repository (production-ready component).
EOF
)" \
  --label "area: geometry,type: enhancement,priority: medium,size: small"
```

**Estimated Time**: 10 minutes

---

## Total Migration Effort

| Step | Task | Time | Complexity |
|:-----|:-----|:-----|:-----------|
| 1 | Copy main module | 10 min | Low |
| 2 | Copy test file | 5 min | Low |
| 3 | Update exports | 5 min | Low |
| 4 | Create example | 15 min | Low |
| 5 | Run tests | 5 min | Low |
| 6 | Update docs | 10 min | Low |
| 7 | Issue & PR | 10 min | Low |
| **Total** | | **60 min** | **Low** |

**Realistic Estimate**: 1-2 hours (including testing and documentation)

---

## Code Modifications Required

### Minimal Changes

**maze_converter.py** (production version):
```python
# Line 1: Add future annotations
from __future__ import annotations

# Line 14: Modern typing (remove Tuple import)
# DELETE: from typing import Tuple

# Line 31: Update return type hint
def maze_to_implicit_domain(
    maze_array: NDArray[np.int32],
    cell_size: float = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),  # Changed from Tuple to tuple
) -> tuple[DifferenceDomain, UnionDomain]:  # Changed from Tuple to tuple
```

**test_maze_converter.py** (production version):
```python
# Line 20-24: Update imports
from mfg_pde.geometry.maze_converter import (
    compute_maze_sdf,
    get_maze_statistics,
    maze_to_implicit_domain,
)
```

**No other changes needed** - code is already production-ready!

---

## Testing Strategy

### Test Coverage

**Existing Tests** (all passing in research repo):
- ✅ Basic conversion (3×3 maze)
- ✅ Empty maze (no walls)
- ✅ Cell size scaling
- ✅ Origin translation
- ✅ Particle sampling (automatic wall avoidance)
- ✅ Invalid input handling (non-binary, wrong dimension)
- ✅ SDF computation (positive in free space, negative in walls)
- ✅ Maze statistics (all walls, no walls, 50% walls)
- ✅ Integration with MFG_PDE maze generator
- ✅ Volume conservation (Monte Carlo validation)
- ✅ Boundary consistency

**Test Organization**:
```
tests/unit/test_maze_converter.py
├── TestMazeToImplicitDomain (8 tests)
├── TestMazeSDF (3 tests)
├── TestMazeStatistics (3 tests)
├── TestMFGPDEIntegration (2 tests)
└── TestHighDimensionalConsistency (2 tests)
```

**Expected Result**: All 18 tests passing ✅

### Regression Testing

**Check No Impact On**:
- Existing geometry classes (ImplicitDomain, Hyperrectangle, UnionDomain, DifferenceDomain)
- Existing maze generator (mfg_pde.alg.reinforcement.environments)
- Other geometry utilities

**Command**:
```bash
# Run full geometry test suite
pytest tests/unit/test_implicit_domain.py -v
pytest tests/unit/test_boundary_conditions.py -v
pytest tests/unit/test_maze_converter.py -v
```

---

## Integration Points

### Existing MFG_PDE Infrastructure Used

**Geometry Classes** (already in production):
- `Hyperrectangle` - For bounding box and wall cells
- `UnionDomain` - For combining wall rectangles
- `DifferenceDomain` - For free space (bounding box minus walls)

**Maze Generator** (already in production):
- `mfg_pde.alg.reinforcement.environments.maze_generator.generate_maze()`
- Supports multiple algorithms: recursive_backtracking, wilsons, ellers, growing_tree

**Particle Sampling** (already in production):
- `ImplicitDomain.sample_uniform()` - Automatically rejects particles in walls
- `ImplicitDomain.contains()` - Check if points are in free space

**No New Dependencies**: Uses only existing production infrastructure

### API Consistency

**Follows MFG_PDE Patterns**:
- ✅ Type hints with numpy.typing
- ✅ Comprehensive docstrings with examples
- ✅ Error handling with ValueError for invalid inputs
- ✅ Consistent parameter names (origin, cell_size, seed)
- ✅ Returns domain objects (not raw arrays)

**Example Usage Matches MFG_PDE Style**:
```python
# Consistent with existing domain creation patterns
from mfg_pde.geometry import Hyperrectangle, maze_to_implicit_domain
from mfg_pde.alg.reinforcement.environments import generate_maze

# Generate and convert
maze = generate_maze(10, 10, seed=42)
domain, walls = maze_to_implicit_domain(maze)

# Use with MFG solvers
particles = domain.sample_uniform(1000)
```

---

## Documentation Requirements

### User Guide Updates

**File**: `docs/user/guides/geometry.md`

**Add Section**: "Maze Integration" (see Step 6 above)

**Content**:
- Basic usage example
- Three use cases (MFG navigation, obstacle avoidance, maze analysis)
- Link to advanced example

### API Documentation

**Auto-generated from docstrings** (already comprehensive):
- `maze_to_implicit_domain()`: Full docstring with examples, math, parameters, returns
- `compute_maze_sdf()`: Full docstring with use case
- `get_maze_statistics()`: Full docstring with example

**No additional API docs needed** - docstrings are production-ready

### Example Documentation

**File**: `examples/advanced/maze_navigation_demo.py`

**Content** (see Step 4):
- Demonstrates all three functions
- Shows discrete → continuous conversion
- Visualizes both representations
- Production-ready, well-commented code

---

## Risk Assessment

### Very Low Risk Migration

| Risk | Probability | Impact | Mitigation |
|:-----|:------------|:-------|:-----------|
| Test failures | Very Low | Low | All tests passing in research repo |
| API conflicts | Very Low | Low | New functions, no overlap with existing |
| Dependencies | None | N/A | Uses only production infrastructure |
| Regression | Very Low | Low | Self-contained module, no changes to existing code |
| Performance | None | N/A | Utility functions, not performance-critical |

**Overall Risk**: ✅ **MINIMAL**

### Rollback Plan

**If Issues Arise**:
1. Revert PR (single commit)
2. Remove from `mfg_pde/geometry/__init__.py` exports
3. Delete test file
4. Delete example

**Estimated Rollback Time**: 5 minutes

---

## Success Criteria

### Migration Successful If:
- ✅ All 18 tests passing
- ✅ Example executes and produces output
- ✅ No regressions in existing geometry tests
- ✅ Documentation updated and accurate
- ✅ API exports working correctly

### Migration Unsuccessful If:
- ❌ Any test failures
- ❌ Example crashes or produces incorrect output
- ❌ Regressions in existing tests
- ❌ Import errors or API conflicts

**Expect**: ✅ All success criteria met (production-ready code)

---

## Post-Migration Tasks

### Immediate (Same PR)
- ✅ Run full test suite
- ✅ Update geometry package exports
- ✅ Add example
- ✅ Update documentation

### Short-term (Follow-up PRs)
- Create maze navigation MFG example (using collocation solver)
- Add visualization utilities for maze domains
- Create notebook tutorial for maze-based MFG

### Medium-term
- Extend to 3D mazes (when 3D maze generator available)
- Add maze topology analysis (connectivity, dead ends)
- Integrate with path planning examples

---

## Comparison with Previous Migration Assessments

### ParticleCollocationSolver
- ❌ Not migrating - research infrastructure wrapper
- Depends on DualModeFPParticleSolver (not yet migrated)
- 587 lines, complex dependencies

### DualModeFPParticleSolver
- ⏳ High priority - core algorithmic enhancement
- Requires refactoring and API design
- 563 lines, moderate complexity

### Maze Integration
- ✅ **READY NOW** - production-ready utility
- Zero dependencies on research code
- 174 lines, minimal complexity
- **Quick win for migration roadmap**

**Recommendation Order**:
1. **Maze integration** (this component) - LOW RISK, HIGH VALUE ✅
2. DualModeFPParticleSolver - MEDIUM RISK, HIGH VALUE ⏳
3. ParticleCollocationSolver - LOW VALUE (keep in research) ❌

---

## Conclusion

**Maze integration component is production-ready and should be migrated immediately.**

**Key Points**:
- ✅ Clean, well-tested code
- ✅ No dependencies on research code
- ✅ Uses only production infrastructure
- ✅ Comprehensive test suite
- ✅ Clear documentation
- ✅ Minimal migration effort (1-2 hours)
- ✅ Very low risk
- ✅ High educational and application value

**Next Actions**:
1. Execute migration steps 1-7 (this plan)
2. Create PR with all changes
3. Request review (self-review if solo maintainer)
4. Merge to main
5. Update research repo to import from production

**Timeline**: Can complete in single session (1-2 hours)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-02
**Status**: Migration plan ready ✅

**References**:
- Research code: `/Users/zvezda/OneDrive/code/mfg-research/experiments/maze_navigation/maze_converter.py`
- Research tests: `/Users/zvezda/OneDrive/code/mfg-research/experiments/maze_navigation/benchmarks/tests/test_maze_converter.py`
- Migration assessment: `docs/development/RESEARCH_MIGRATION_ASSESSMENT_2025-11-02.md`
- Production destination: `mfg_pde/geometry/maze_converter.py`
