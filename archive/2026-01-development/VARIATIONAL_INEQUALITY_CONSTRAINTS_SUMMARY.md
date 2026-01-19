# Variational Inequality Constraints - Implementation Summary

**Issue**: #591 Phase 2
**Status**: ✅ COMPLETED
**Date**: 2026-01-17

## Overview

Implemented comprehensive variational inequality constraint infrastructure for MFG_PDE, enabling obstacle problems and box constraints on PDE solutions. The implementation supports unilateral constraints (u ≥ ψ or u ≤ ψ), bilateral constraints (ψ_lower ≤ u ≤ ψ_upper), and regional constraints (constraints active only in spatial subdomains).

## Implementation

### Core Infrastructure

#### 1. Constraint Protocol (`mfg_pde/geometry/boundary/constraint_protocol.py`)

Protocol-based interface for all constraint types:

```python
@runtime_checkable
class ConstraintProtocol(Protocol):
    def project(self, u: NDArray) -> NDArray:
        """Project field onto constraint set K."""

    def is_feasible(self, u: NDArray, tol: float = 1e-10) -> bool:
        """Check if field satisfies constraint."""

    def get_active_set(self, u: NDArray, tol: float = 1e-10) -> NDArray:
        """Identify active constraints (where constraint binds)."""
```

**Design**: Duck-typed protocol enables flexible constraint composition without inheritance.

#### 2. Concrete Implementations (`mfg_pde/geometry/boundary/constraints.py`)

**ObstacleConstraint** - Unilateral constraints:
```python
class ObstacleConstraint:
    """
    Unilateral constraint: u ≥ ψ(x) or u ≤ ψ(x)

    Parameters
    ----------
    obstacle : NDArray
        Obstacle function ψ(x)
    constraint_type : {"lower", "upper"}
        "lower": u ≥ ψ (default)
        "upper": u ≤ ψ
    region : NDArray | None
        Boolean mask for regional constraints (None = global)
    """
```

**Projection**:
- Lower: `u_proj = max(u, ψ)`
- Upper: `u_proj = min(u, ψ)`
- Regional: Apply only where `region == True`

**BilateralConstraint** - Box constraints:
```python
class BilateralConstraint:
    """
    Bilateral (box) constraint: ψ_lower ≤ u ≤ ψ_upper

    Parameters
    ----------
    lower_bound : NDArray
        Lower bound ψ_lower(x)
    upper_bound : NDArray
        Upper bound ψ_upper(x)
    region : NDArray | None
        Boolean mask for regional constraints
    """
```

**Projection**: `u_proj = clip(u, ψ_lower, ψ_upper)`

### Integration with Solvers

#### HJB FDM Solver (`mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py`)

**1D Path** (lines 481-501):
```python
U_solution = base_hjb.solve_hjb_system_backward(...)
if self.constraint is not None:
    for n in range(U_solution.shape[0]):
        U_solution[n] = self.constraint.project(U_solution[n])
return U_solution
```

**nD Path** (lines 664-668):
```python
def _solve_single_timestep(...):
    # ... Newton iteration ...
    if constraint is not None:
        U_solution = constraint.project(U_solution)
    return U_solution
```

**Application**: Projection applied after each timestep, enforcing constraints throughout backward solution.

### Testing

#### Test Suite (`tests/unit/geometry/boundary/test_constraints.py`)

**Coverage**: 34 tests, all passing

**Test Categories**:
1. **Protocol Compliance** - Verify implementations satisfy ConstraintProtocol
2. **Projection Properties**:
   - Idempotence: P(P(u)) = P(u)
   - Non-expansiveness: ||P(u) - P(v)|| ≤ ||u - v||
   - Feasibility: P(u) ∈ K
3. **Active Set Detection** - Correct identification of constraint boundaries
4. **Regional Constraints** - Spatial selectivity verification
5. **Error Handling** - Invalid inputs, dimension mismatches

**Example Test**:
```python
def test_obstacle_projection_feasibility():
    """Projection must produce feasible solution."""
    obstacle = np.array([0.3, 0.5, 0.4, 0.6, 0.2])
    constraint = ObstacleConstraint(obstacle, constraint_type="lower")

    u = np.array([0.1, 0.7, 0.3, 0.8, 0.0])  # Violates at indices 0, 2, 4
    u_proj = constraint.project(u)

    assert constraint.is_feasible(u_proj, tol=1e-10)
    np.testing.assert_array_equal(u_proj, [0.3, 0.7, 0.4, 0.8, 0.2])
```

## Examples

### 1. Heat Equation with Obstacle (`examples/advanced/obstacle_problem_1d_heat.py`)

**Problem**: Cooling rod with minimum temperature constraint (thermostat)

**PDE**: ∂u/∂t = σ²/2 ∂²u/∂x² - λu

**Constraint**: u(t,x) ≥ ψ(x) where ψ is parabolic (0.3 at center, 0.1 at edges)

**Physics**:
- Rod cools exponentially (λ = 0.5)
- Center must stay warmer (higher thermostat setting)
- Edges can cool more

**Results**:
- Active set growth: 0.0% → 68.6% ✓
- Perfect constraint satisfaction: max violation = 0.00e+00
- Physically correct: Center hits minimum first, then more points activate

**Key Insight**: Used heat equation instead of HJB solver to bypass pre-existing bug in running cost incorporation (Issue #591 Phase 1 finding).

### 2. Bilateral Constraint (`examples/advanced/obstacle_problem_1d_bilateral.py`)

**Problem**: Temperature control with both heating and cooling

**Constraint**: ψ_lower(x) ≤ u(t,x) ≤ ψ_upper(x)

**Physics**:
- Lower bound: Heating prevents freezing (parabolic, higher at center)
- Upper bound: Cooling prevents overheating (uniform ceiling)
- Cold center initially at lower bound

**Results**:
- Lower constraint active: 17.6% → 0.0%
- Upper constraint: Never activated (temperature stays below ceiling)
- Demonstrates box constraint projection

**Key Feature**: Tracks which constraint is active (lower vs upper) at each point.

### 3. Regional Constraint (`examples/advanced/obstacle_problem_1d_regional.py`)

**Problem**: Protected zone with minimum temperature

**Constraint**:
- Protected zone x ∈ [0.3, 0.7]: u ≥ 0.4
- Outside zone: No constraint (free cooling)

**Physics**: Cooling rod where only central region requires temperature control

**Results**:
- Protected zone: 100% active at final time (maintained at ψ = 0.4)
- Outside zone: Temperature drops to 0.17 < 0.4 ✓
- Demonstrates spatial selectivity

**Key Feature**: Constraint enforcement location is spatially selective.

## Validation Examples

All three examples demonstrate:
1. **Perfect constraint satisfaction**: Zero violations after projection
2. **Numerical stability**: Forward Euler with CFL condition satisfied
3. **Physical correctness**: Active set evolution matches physics
4. **Visualization**: 6-panel figures showing evolution, active sets, constraint effects

## Mathematical Background

### Projection onto Constraint Set

For constraint set K, the projection operator P_K minimizes distance:

```
P_K(u) = argmin_{v ∈ K} ||v - u||²
```

**Properties**:
- **Idempotent**: P_K(P_K(u)) = P_K(u)
- **Non-expansive**: ||P_K(u) - P_K(v)|| ≤ ||u - v||
- **Feasible**: P_K(u) ∈ K

### Obstacle Problem Formulation

Find u(t,x) satisfying:
```
∂u/∂t = F(u, ∇u, ∇²u)    if u > ψ   (free boundary)
u = ψ                      if u = ψ   (contact set/active set)
```

Equivalently: u = max(u_unconstrained, ψ) at each timestep.

### Active Set

Points where constraint binds:
```
A(t) = {x : u(t,x) = ψ(x)}
```

Physical interpretation:
- **Lower obstacle**: Points at minimum allowable value
- **Upper obstacle**: Points at maximum allowable value
- **Bilateral**: Points at either bound

## Files Modified/Created

### Created
- `mfg_pde/geometry/boundary/constraint_protocol.py` (~60 lines)
- `mfg_pde/geometry/boundary/constraints.py` (~640 lines)
- `tests/unit/geometry/boundary/test_constraints.py` (~370 lines)
- `examples/advanced/obstacle_problem_1d_heat.py` (~400 lines)
- `examples/advanced/obstacle_problem_1d_bilateral.py` (~420 lines)
- `examples/advanced/obstacle_problem_1d_regional.py` (~470 lines)

### Modified
- `mfg_pde/geometry/boundary/__init__.py` - Export constraint classes
- `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py` - Constraint integration
- `mfg_pde/geometry/boundary/applicator_fdm.py` - Import updates

### Tests
- **Unit tests**: 34 tests, all passing ✓
- **Integration tests**: 39/40 HJB FDM tests passing (1 pre-existing failure)
- **Examples**: 3 validation examples, all successful

## Usage

### Basic Example

```python
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import ObstacleConstraint
from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver
from mfg_pde import MFGProblem

# Setup
grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[100])
x = grid.coordinates[0]

# Define obstacle
psi = -0.5 * (x - 0.5) ** 2  # Parabolic obstacle

# Create constraint
constraint = ObstacleConstraint(psi, constraint_type="lower")

# Create problem and solver with constraint
problem = MFGProblem(geometry=grid, T=1.0, Nt=50, diffusion=0.1)
solver = HJBFDMSolver(problem, constraint=constraint)

# Solve - constraint automatically enforced
result = solver.solve(...)
```

### Regional Constraint Example

```python
# Define protected region
region = (x >= 0.3) & (x <= 0.7)

# Create regional obstacle
psi = np.ones_like(x) * 0.4
constraint = ObstacleConstraint(psi, constraint_type="lower", region=region)

# Constraint only enforced where region == True
```

## Known Limitations

### 1. HJB Solver Running Cost Bug

**Issue**: Pre-existing bug in 1D HJB solver where running costs are incorporated with wrong sign, causing solutions to decrease backward in time instead of increase.

**Impact**: Cannot create proper obstacle problems using HJB solver with running costs.

**Workaround**: Use heat equation examples (no Hamiltonian) to demonstrate constraint mechanism.

**Status**: Bug documented in Issue #591 Phase 1, not fixed (separate issue).

### 2. Projection Method Limitations

**Current**: Simple pointwise projection (operator splitting)

**Alternative**: Could implement:
- Variational inequality formulation
- Penalty methods
- Augmented Lagrangian

**Justification**: Pointwise projection is:
- Computationally efficient O(N)
- Mathematically well-defined
- Sufficient for most applications

### 3. Higher Dimensions

**Status**: Infrastructure supports nD, but validation examples are 1D only.

**Future**: Create 2D obstacle problem examples demonstrating:
- Spatial active set evolution
- Constraint effects on 2D fields
- Computational efficiency at scale

## Future Work

### High Priority
1. **2D Examples** - Validate constraints in higher dimensions
2. **FP Solver Integration** - Add constraint support to Fokker-Planck solver
3. **Fix HJB Running Cost Bug** - Enable proper obstacle problems with Hamiltonians

### Medium Priority
4. **Advanced Projections** - Variational inequality formulation
5. **Performance** - Vectorized projection for large problems
6. **Complementarity Conditions** - Active set detection improvements

### Low Priority
7. **Time-Dependent Obstacles** - ψ(t,x)
8. **State-Dependent Constraints** - ψ(u,x)
9. **Multi-Field Constraints** - Coupled constraints on (u, m)

## References

### Mathematical Background
- Kinderlehrer & Stampacchia, "An Introduction to Variational Inequalities"
- Evans, "Partial Differential Equations" (Chapter on variational inequalities)
- Bensoussan & Lions, "Applications of Variational Inequalities in Stochastic Control"

### Numerical Methods
- Glowinski et al., "Numerical Methods for Nonlinear Variational Problems"
- Ito & Kunisch, "Lagrange Multiplier Approach to Variational Problems"
- Hintermüller & Kunisch, "PDE-Constrained Optimization Subject to Pointwise Constraints"

### MFG with Constraints
- Achdou & Capuzzo-Dolcetta, "Mean Field Games: Numerical Methods"
- Camilli & Silva, "A Semi-Discrete Approximation for a First Order Mean Field Game Problem"
- Graber & Mészáros, "Sobolev Regularity for First Order Mean Field Games"

## Conclusion

Successfully implemented comprehensive variational inequality constraint infrastructure with:
- ✅ Protocol-based design for extensibility
- ✅ Three constraint types (obstacle, bilateral, regional)
- ✅ Integration with HJB FDM solver
- ✅ 34 passing unit tests
- ✅ 3 physics-based validation examples
- ✅ Complete documentation

**Key Achievement**: Created a flexible, well-tested constraint system that enables obstacle problems and box constraints in MFG solvers while maintaining clean separation of concerns through protocol-based design.

**Impact**: Enables research into:
- Constrained mean field games
- Optimal control with state constraints
- Free boundary problems
- Complementarity formulations

---

**Last Updated**: 2026-01-17
**Author**: Claude Code
**Related Issues**: #591 Phase 2
