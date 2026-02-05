# [SPEC] Periodic Boundary Conditions on Implicit Geometries

**Document ID**: MFG-SPEC-ADD-01
**Status**: DRAFT
**Date**: 2026-01-30
**Parent**: MFG-SPEC-GEO-1.0

---

## 1. Overview: Topology vs. Function

While periodic BCs on arbitrary implicit manifolds require complex tessellation,
MFG_PDE identifies canonical cases where $O(1)$ complexity is achievable.

Two types of periodicity:
- **Topological**: Domain connectivity wraps around (e.g., Torus).
- **Functional**: The SDF $\phi(x)$ repeats in space (e.g., TPMS).

**MFG_PDE relevance**: Periodic domains arise in:
- Homogenization-based MFG (periodic cost landscapes)
- Crowd models on periodic structures (corridors, transit systems)
- Mean field limit with wrap-around state space (angular variables)

---

## 2. Canonical Cases

### 2.1 The Rectangular Torus (Standard Periodic Box)

Most common case. Used in DNS turbulence and homogenization.

- **Geometry**: $\Omega = [0, L_x] \times [0, L_y]$.
- **Model**: $u(0, y) = u(L_x, y)$ and $u(x, 0) = u(x, L_y)$.
- **Architecture**:
  - Connectivity: Implicit (modulo indexing)
  - Boundary: `NoBoundary` (topological closure)
- **Kernel**: `idx_neighbor = (idx + stride) % total_size`
- **Performance**: Zero memory overhead; ALU-only.

**MFG_PDE status**: Partially supported via `BCType.PERIODIC` and
`SupportsPeriodic` protocol. Modulo-based stencils implemented in
`boundary/periodic.py`.

### 2.2 The Sphere Surface ($S^2$)

> **Status**: DEFERRED. Not a current MFG use case.

**Strategy A: Level-Set Embedding**

- SDF: $\phi(x) = \|x\|_2 - R$ in 3D Cartesian grid.
- Periodicity: Sphere is a closed manifold — no boundary.
- Solver: Narrow band $\Gamma = \{x : |\phi(x)| < \epsilon\}$.
- Topology: Automatically closed via narrow band connectedness.

**Strategy B: Spherical Coordinates**

- $(\theta, \varphi) \in [0, 2\pi) \times [0, \pi]$.
- Longitude $\theta$: Periodic (wrap-around).
- Latitude $\varphi$: Singular (pole handling required).
- Metric: Manifold metric with $\sin(\varphi)$ scale factors.

### 2.3 The Solid Ball ($B^3$) / Polar Coordinates

> **Status**: DEFERRED.

- Geometry: $(r, \theta, z)$ or $(r, \theta, \varphi)$.
- Periodicity: Angular direction $\theta$ only.
- Architecture:
  - Axis $r$: Bounded (wall at $R$, singularity at $0$)
  - Axis $\theta$: Periodic (modulo stride)
- Implementation: Standard 1D periodic stencils on $\theta$-axis tensor slices.

### 2.4 Triply Periodic Minimal Surfaces (TPMS)

> **Status**: OUT OF SCOPE (materials science, not MFG).

- Definition: Implicit $\phi(x)$ is periodic.
  Example: Gyroid $\sin x \cos y + \sin y \cos z + \sin z \cos x = 0$.
- Architecture:
  - Domain: `ImplicitDomain` with `CartesianGrid` background.
  - Boundary: `NoBoundary` (periodic background).
  - Mechanism: Periodicity enforced by background grid. Implicit surface
    naturally connects across periodic boundaries.

---

## 3. Configuration Examples

### Example 1: Periodic RVE

```python
# Standard periodic simulation box for homogenization MFG
rve_spec = DomainSpec(
    connectivity=Connectivity.Implicit,
    structure=Structure.Structured,
    boundary=Boundary.NoBoundary,  # Flags periodic topology
    embedding=Embedding.Grid,
    metric=Metric.Euclidean,
)
# Solver generates modulo-based indexers
```

### Example 2: MFG on Angular State Space

```python
# Crowd model where agents have periodic heading angle
from mfg_pde import MFGProblem
from mfg_pde.geometry.boundary import periodic_bc

problem = MFGProblem(
    spatial_bounds=[(0, 2 * np.pi)],
    spatial_discretization=[128],
    Nt=100,
    T=1.0,
    boundary_conditions=periodic_bc(dimension=1),
)
```

---

## 4. Implementation Notes

### NoBoundary Semantics

The `NoBoundary` label means the domain has no boundary in the topological
sense (it is a closed manifold). In the code:

- `BCType.PERIODIC` communicates this to applicators
- `FDMApplicator` uses `np.roll` or modulo indexing
- `get_boundary_indices()` returns empty array
- `is_on_boundary()` returns all False

### Interaction with BC Framework

Periodic BC fits the compositional framework as:
- **Region**: Global (applies everywhere, including where boundary "would be")
- **MathType**: Periodic (topological, not a differential constraint)
- **ValueSource**: N/A (no value, just connectivity)
- **Enforcement**: Modulo indexing (specialized, not Strong/Weak/GhostFluid)

This is a degenerate case where the 4-axis model reduces to a single trait.

---

## 5. References

- Sethian, J. A. (1999). *Level Set Methods and Fast Marching Methods*.
  Cambridge University Press.
- Torquato, S. (2002). *Random Heterogeneous Materials*. Springer-Verlag.

---

**Last Updated**: 2026-02-05
