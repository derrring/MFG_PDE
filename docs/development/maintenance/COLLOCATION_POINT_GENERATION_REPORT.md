# Collocation Point Generation for Meshfree MFG Solvers

## Technical Report

**Project**: MFG_PDE (Mean Field Games PDE Solver Framework)
**Date**: 2025-12-16
**Status**: ✅ AUDITED - Ready for Implementation

---

## 1. Problem Statement

The GFDM solver for Hamilton-Jacobi-Bellman (HJB) equations requires:
- **Interior collocation points**: Where PDE residuals are enforced
- **Boundary collocation points**: Where boundary conditions are enforced
- **Boundary indices**: To distinguish interior from boundary points

Currently, the solver accepts these as input parameters but lacks unified infrastructure to **generate** them for complex geometries.

### Observed Issue
Without proper boundary point identification, GFDM solutions exhibit:
- Boundary layer blow-up
- Incorrect BC enforcement
- Poor accuracy near domain boundaries

---

## 2. External Audit Summary

### Critical Issues Identified

| Issue | Original Plan | Required Fix |
|-------|---------------|--------------|
| **High-dim sampling** | Rejection sampling | Newton/level-set projection (rejection fails for d > 3) |
| **Normal computation** | Finite difference | Analytical/AD gradients (mandatory for Neumann BC accuracy) |
| **Point clustering** | Random/Sobol | Poisson Disk Sampling (prevents ill-conditioned GFDM stencils) |
| **Corner handling** | Not addressed | Explicit corner detection & inclusion |

### Boundary Density Formula

For consistent GFDM convergence, boundary density should scale with surface area:

$$N_{bdy} \approx C \cdot \text{Area}(\partial \Omega) \cdot \left(\frac{N_{int}}{\text{Vol}(\Omega)}\right)^{(d-1)/d}$$

### Answers to Open Questions

| Question | Answer |
|----------|--------|
| Boundary point density | Scale with surface area (formula above) |
| Normal computation | Analytical/AD - mandatory, not FD |
| Quasi-random for boundaries | Use Poisson Disk instead for spacing guarantees |
| Corner handling | Explicitly sample corners, always include vertices |

---

## 3. Approved Architecture

### 3.1 Two-Layer Design (Finalized)

```
┌────────────────────────────────────────────────────────────┐
│ Layer 2: geometry/collocation.py                           │
│ ├── CollocationSampler (strategy dispatch)                 │
│ │   ├── sample_interior(n, method) → points                │
│ │   ├── sample_boundary(n, method) → points, normals       │
│ │   └── generate_collocation() → CollocationPointSet       │
│ ├── CartesianGridCollocation (extraction)                  │
│ ├── MeshCollocation (extraction)                           │
│ ├── ImplicitDomainCollocation (Newton projection)          │
│ └── CSGDomainCollocation (multi-component)                 │
└───────────────────┬────────────────────────────────────────┘
                    │ uses
                    ▼
┌────────────────────────────────────────────────────────────┐
│ Layer 1: utils/numerical/particle/sampling.py              │
│ ├── QuasiMCSampler (Sobol, Halton)                         │
│ ├── PoissonDiskSampler ← NEW (Bridson's algorithm)         │
│ ├── UniformMCSampler                                       │
│ └── StratifiedMCSampler                                    │
└────────────────────────────────────────────────────────────┘
```

### 3.2 Core Data Structure (Revised)

```python
@dataclass
class CollocationPointSet:
    """Container for collocation point data."""
    points: np.ndarray        # (N, d) all collocation points
    is_boundary: np.ndarray   # (N,) bool - faster than index array
    normals: np.ndarray       # (N, d) - zeros for interior points
    region_ids: np.ndarray    # (N,) int - maps to physical BC regions

    @property
    def boundary_indices(self) -> np.ndarray:
        """For backward compatibility with GFDM solver."""
        return np.where(self.is_boundary)[0]

    @property
    def interior_indices(self) -> np.ndarray:
        return np.where(~self.is_boundary)[0]
```

### 3.3 Algorithm by Geometry Type

| Geometry Type | Interior Sampling | Boundary Sampling | Complexity |
|---------------|-------------------|-------------------|------------|
| **Cartesian Grid** | Grid node extraction | Face node extraction + corners | O(N) |
| **Unstructured Mesh** | Vertex extraction | Edge/face vertex extraction | O(N) |
| **Implicit (SDF)** | Poisson Disk + rejection | Newton projection to φ=0 | O(N log N) |
| **CSG Composite** | Poisson Disk + combined SDF | Per-component Newton projection | O(N log N) |

---

## 4. Implementation Plan

### Phase 1: Layer 1 - PoissonDiskSampler

Add to `utils/numerical/particle/sampling.py`:

```python
class PoissonDiskSampler(MCSampler):
    """
    Poisson Disk Sampling using Bridson's algorithm.

    Guarantees minimum distance between points, preventing
    ill-conditioned GFDM stencils from point clustering.

    Reference: Bridson, R. "Fast Poisson disk sampling in arbitrary
    dimensions." SIGGRAPH 2007.
    """

    def __init__(
        self,
        domain: MCDomain,
        config: MCConfig,
        min_distance: float | None = None,
        k_candidates: int = 30,
    ):
        super().__init__(domain, config)
        self.min_distance = min_distance
        self.k_candidates = k_candidates

    def sample(self, num_samples: int) -> NDArray:
        """Generate well-spaced points via dart throwing."""
        ...
```

### Phase 2: Layer 2 - CollocationSampler

Create `geometry/collocation.py`:

```python
class CollocationSampler:
    """Unified collocation sampler for any geometry type."""

    def __init__(self, geometry: GeometryProtocol):
        self.geometry = geometry
        self._strategy = self._select_strategy()

    def _select_strategy(self) -> BaseCollocationStrategy:
        """Dispatch to geometry-specific implementation."""
        if hasattr(self.geometry, 'get_grid_points'):
            return CartesianGridCollocation(self.geometry)
        elif hasattr(self.geometry, 'vertices'):
            return MeshCollocation(self.geometry)
        elif hasattr(self.geometry, 'signed_distance'):
            if hasattr(self.geometry, 'domains'):
                return CSGDomainCollocation(self.geometry)
            return ImplicitDomainCollocation(self.geometry)
        raise ValueError(f"Unsupported geometry: {type(self.geometry)}")

    def generate_collocation(
        self,
        n_interior: int,
        n_boundary: int,
        interior_method: str = "poisson_disk",
        boundary_method: str = "poisson_disk",
    ) -> CollocationPointSet:
        ...
```

### Phase 3: GFDM Integration

Update GFDM solver to optionally accept `CollocationPointSet`:

```python
# New convenience API
solver = HJBGFDMSolver.from_geometry(
    problem,
    n_interior=400,
    n_boundary=100,
)

# Or explicit
coll = CollocationSampler(geometry).generate_collocation(400, 100)
solver = HJBGFDMSolver(problem, collocation=coll)
```

---

## 5. Implementation Status

| Component | Status |
|-----------|--------|
| Issue #482 created | ✅ |
| Architecture designed | ✅ |
| External audit completed | ✅ |
| `PoissonDiskSampler` (Layer 1) | ⏳ In Progress |
| `geometry/collocation.py` (Layer 2) | ⏳ Pending |
| GFDM integration | ⏳ Pending |
| Tests | ⏳ Pending |

---

## 6. Related Issues

- **#482**: Boundary Point Generation Module for Meshfree Methods
- **#483**: GFDM Performance Optimization

---

## References

1. Bridson, R. "Fast Poisson disk sampling in arbitrary dimensions." SIGGRAPH 2007.
2. Benito, J.J., et al. "Influence of several factors in the generalized finite difference method." Applied Mathematical Modelling, 2001.
3. Gavete, L., et al. "Generalized finite differences for solving 3D elliptic and parabolic equations." Applied Mathematical Modelling, 2016.

---

*Report updated after external audit - 2025-12-16*
