# BC Solver Integration Design

**Issue**: #527 (BC Infrastructure Integration)
**Created**: 2026-01-11
**Status**: Design Proposal

## Problem Statement

BC handling is currently **paradigm-specific** with duplicate implementations:
- Numerical solvers: Apply BCs to fields (ghost cells, matrix modifications)
- Neural solvers: Need boundary loss terms
- RL solvers: Environment boundary behavior
- Optimization solvers: Domain constraints

But all paradigms need **access to the same BC information**.

## Current Architecture

```
BaseMFGSolver (all solvers)
    └── problem: MFGProblem
            └── geometry: GeometryProtocol
                    └── boundary_conditions: BoundaryConditions
```

All solvers already have access to BC via `self.problem.geometry.boundary_conditions`.

## Proposed Design: Paradigm-Specific BC Helpers

### Principle: One Source, Multiple Applications

```
┌─────────────────────────────────────────────────────────────┐
│                    BoundaryConditions                       │
│  (Single source of truth: geometry.boundary_conditions)     │
└───────────────┬─────────────────┬───────────────┬──────────┘
                │                 │               │
    ┌───────────▼───────┐   ┌────▼────┐   ┌─────▼─────┐
    │   Numerical       │   │ Neural  │   │    RL     │
    │ dispatch.apply_bc │   │ bc_loss │   │ env_bc    │
    │ (field ops)       │   │ (train) │   │ (agents)  │
    └───────────────────┘   └─────────┘   └───────────┘
```

### 1. Numerical Paradigm: Field Operations

**Existing infrastructure**: `dispatch.apply_bc()`

```python
from mfg_pde.geometry.boundary.dispatch import apply_bc

class BaseNumericalSolver(BaseMFGSolver):
    def apply_boundary_conditions(self, field: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Apply BCs to field using unified dispatch."""
        bc = self.problem.geometry.boundary_conditions
        return apply_bc(
            geometry=self.problem.geometry,
            field=field,
            boundary_conditions=bc,
            time=time,
            discretization=self.discretization_type,
        )
```

**Solver-specific discretization types**:
- HJB FDM: `DiscretizationType.FDM`
- HJB GFDM: `DiscretizationType.GFDM`
- FP Particle: `DiscretizationType.MESHFREE`

### 2. Neural Paradigm: Boundary Loss Terms

**New helper**: `compute_boundary_loss()`

```python
from mfg_pde.geometry.boundary import BCType

class BaseNeuralSolver(BaseMFGSolver):
    def sample_boundary_points(self, n_points: int) -> np.ndarray:
        """Sample points on domain boundary for BC loss."""
        return self.problem.geometry.sample_boundary_points(n_points)

    def compute_boundary_loss(
        self,
        network_output: torch.Tensor,
        boundary_points: np.ndarray,
        time: float = 0.0,
    ) -> torch.Tensor:
        """Compute BC loss based on boundary_conditions type."""
        bc = self.problem.geometry.boundary_conditions

        if bc.default_bc == BCType.DIRICHLET:
            # L_bc = ||u(x_b) - g(x_b)||²
            target = bc.get_value_at_points(boundary_points, time)
            return torch.mean((network_output - target) ** 2)

        elif bc.default_bc == BCType.NEUMANN:
            # L_bc = ||∂u/∂n - g||²
            normal_grad = self._compute_normal_gradient(network_output, boundary_points)
            target = bc.get_value_at_points(boundary_points, time)
            return torch.mean((normal_grad - target) ** 2)

        elif bc.default_bc == BCType.PERIODIC:
            # L_bc = ||u(x_min) - u(x_max)||²
            return self._compute_periodic_loss(network_output, boundary_points)

        # ... other BC types
```

### 3. RL Paradigm: Environment Boundaries

**Environment BC configuration**:

```python
from mfg_pde.geometry.boundary import BCType

class BaseRLSolver(BaseMFGSolver):
    def _configure_environment_boundaries(self, env_config: dict) -> dict:
        """Configure environment boundaries based on problem BCs."""
        bc = self.problem.geometry.boundary_conditions
        bounds = self.problem.geometry.get_bounds()

        env_config["bounds"] = {
            "low": bounds[0],
            "high": bounds[1],
        }

        # Map BCType to environment behavior
        if bc.default_bc == BCType.PERIODIC:
            env_config["boundary_mode"] = "wrap"  # Wrap around
        elif bc.default_bc == BCType.REFLECTING:
            env_config["boundary_mode"] = "reflect"  # Elastic bounce
        elif bc.default_bc == BCType.DIRICHLET:
            env_config["boundary_mode"] = "absorb"  # Episode ends
        else:
            env_config["boundary_mode"] = "clip"  # Default: clip to bounds

        return env_config
```

### 4. Optimization Paradigm: Domain Constraints

**Constraint generation**:

```python
class BaseOptimizationSolver(BaseMFGSolver):
    def get_domain_constraints(self) -> list[dict]:
        """Generate optimization constraints from BCs."""
        bc = self.problem.geometry.boundary_conditions
        constraints = []

        for segment in bc.segments:
            if segment.bc_type == BCType.DIRICHLET:
                # Equality constraint at boundary
                constraints.append({
                    "type": "eq",
                    "region": segment.region,
                    "value": segment.value,
                })
            elif segment.bc_type == BCType.NEUMANN:
                # Gradient constraint at boundary
                constraints.append({
                    "type": "grad",
                    "region": segment.region,
                    "normal_grad": segment.value,
                })

        return constraints
```

## Implementation Plan

### Phase 1: Add Helper Methods to Base Classes (Low Risk)

1. Add `apply_boundary_conditions()` to `BaseNumericalSolver`
2. Add `sample_boundary_points()` and `compute_boundary_loss()` to `BaseNeuralSolver`
3. Add `_configure_environment_boundaries()` to `BaseRLSolver`
4. Add `get_domain_constraints()` to `BaseOptimizationSolver`

**These are opt-in** - solvers can use them or keep custom implementations.

### Phase 2: Migrate Numerical Solvers

Wire existing solvers to use `dispatch.apply_bc()`:
1. HJB FDM (template)
2. HJB WENO
3. FP FDM
4. FP Particle

### Phase 3: Enable Neural BC Loss

Add boundary loss support to:
1. PINN solvers
2. DGM solvers

### Phase 4: RL Environment Integration

Update RL environments to read BC config from problem.

## Geometry Protocol Extension

The `GeometryProtocol` should provide:

```python
class GeometryProtocol(Protocol):
    # Existing
    @property
    def dimension(self) -> int: ...
    def get_bounds(self) -> tuple[np.ndarray, np.ndarray] | None: ...

    # BC-related (new or clarified)
    @property
    def boundary_conditions(self) -> BoundaryConditions: ...

    def sample_boundary_points(self, n_points: int) -> np.ndarray:
        """Sample points on domain boundary."""
        ...

    def get_boundary_indices(self, points: np.ndarray) -> np.ndarray:
        """Get indices of points that are on boundary."""
        ...

    def get_outward_normal(self, point: np.ndarray) -> np.ndarray:
        """Get outward normal vector at boundary point."""
        ...
```

## BC Type to Paradigm Mapping

| BCType | Numerical | Neural | RL | Optimization |
|--------|-----------|--------|-----|--------------|
| DIRICHLET | Ghost cells | Value loss | Absorbing | Equality constraint |
| NEUMANN | Ghost cells | Gradient loss | Reflecting | Gradient constraint |
| PERIODIC | Wrap-around | Periodicity loss | Wrap | Periodic variables |
| REFLECTING | N/A | N/A | Elastic bounce | N/A |
| ROBIN | Ghost cells | Mixed loss | Partial absorb | Mixed constraint |
| NO_FLUX | Zero gradient | Zero gradient loss | Reflecting | Zero gradient |

## Benefits

1. **Single source of truth**: All paradigms read from `geometry.boundary_conditions`
2. **Paradigm-appropriate handling**: Each paradigm handles BCs its own way
3. **Opt-in migration**: Solvers can adopt helpers incrementally
4. **Testable**: Each helper can be tested independently
5. **Documented**: Clear mapping from BC types to paradigm behavior

## Related

- `geometry/boundary/dispatch.py` - Existing numerical BC dispatch
- `BC_CAPABILITY_MATRIX.md` - Current solver BC support
- Issue #535 - BC framework enhancement
- Issue #549 - Non-tensor-product geometry BC
