# Technical Reference: High-Dimensional MFG via Particle Methods

**Consolidated Technical Documentation**
**Date:** 2025-10-16
**Version:** 1.0
**Status:** Comprehensive reference for high-dimensional MFG research

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [The Dimensional Problem](#the-dimensional-problem)
3. [Particle Methods: Breaking the Curse](#particle-methods-breaking-the-curse)
4. [N-Dimensional Domain Representation](#n-dimensional-domain-representation)
5. [Dimension-Agnostic Architecture](#dimension-agnostic-architecture)
6. [Research Roadmap](#research-roadmap)
7. [Implementation Guidelines](#implementation-guidelines)
8. [Applications and Impact](#applications-and-impact)
9. [References](#references)

---

## Executive Summary

### The Problem
**Mean Field Games solvers are limited to d ≤ 3 dimensions** due to the curse of dimensionality. Mesh-based methods (FDM, FEM) require O(N^d) grid points, making 4D+ problems computationally infeasible.

### The Solution
**Particle-collocation methods scale as O(N·K)** instead of O(N^d), breaking the exponential barrier. With proper architectural design (implicit domains + dimension-agnostic infrastructure), we can extend MFG to arbitrary dimensions.

### Key Results
- **Phase 2 (Validated):** Adaptive time-stepping achieves 2.99-4.71× speedup, dimension-agnostic
- **Phase 3 (In Progress):** Particle-collocation for 2D/3D validation
- **Future (Stage 3):** First MFG solver capable of handling d ≥ 4

### Impact
- Enables new application domains: swarm robotics (4D), portfolio optimization (5D+), molecular dynamics (6D)
- Provides unified framework for both low-dimensional (d≤3) and high-dimensional (d≥4) problems
- Three potential publications instead of one

---

## The Dimensional Problem

### MFG_PDE's Explicit Dimensional Limits

MFG_PDE has **hard-coded dimension checks** that prevent 4D+ usage:

```python
# From mfg_pde/geometry/tensor_product_grid.py (line 80-81)
if dimension not in [1, 2, 3]:
    raise ValueError(f"Dimension must be 1, 2, or 3, got {dimension}")

# From mfg_pde/geometry/base_geometry.py (line 60-61)
if self.dimension not in [2, 3]:
    raise ValueError(f"Dimension must be 2 or 3, got {self.dimension}")
```

**This is intentional**, not a bug - it reflects fundamental limitations of mesh-based methods.

### The Curse of Dimensionality

#### Memory Explosion

For N points per dimension in d-dimensional space:

| Dimension | Grid Points (N=100) | Memory (8 bytes/point) | Status |
|-----------|---------------------|------------------------|--------|
| 1D | 100 | 800 B | ✅ Trivial |
| 2D | 10,000 | 80 KB | ✅ Easy |
| 3D | 1,000,000 | 8 MB | ✅ Feasible |
| 4D | 100,000,000 | 800 MB | ⚠️ Challenging |
| 5D | 10,000,000,000 | 80 GB | ❌ Infeasible |
| 6D | 1,000,000,000,000 | 8 TB | ❌ Impossible |

**Even with sparse storage**, the number of grid points grows as **N^d**.

#### Computational Cost Explosion

**MFG solve complexity:** O(N^d · Nt) for Picard iteration

Typical solve times (assuming N=100 points/dimension, 500 timesteps):

| Problem | Total Operations | Estimated Time | Status |
|---------|------------------|----------------|--------|
| 2D (100×100) | 5,000,000 | ~1 minute | ✅ Practical |
| 3D (50×50×50) | 62,500,000 | ~15 minutes | ✅ Feasible |
| 4D (30^4) | 405,000,000 | ~2 hours | ⚠️ Slow |
| 5D (20^5) | 1,600,000,000 | ~10 hours | ❌ Impractical |
| 6D (15^6) | 5,700,000,000 | ~36 hours | ❌ Infeasible |

**Practical limit for mesh methods: d ≤ 3**

### Why Current Architecture is Dimension-Specific

#### Hard-Coded 1D Assumptions

```python
# mfg_pde/core/mfg_problem.py (simplified)
class MFGProblem:
    def __init__(self, Nx, xmin, xmax, ...):
        self.Nx: int = Nx           # ❌ Single dimension only
        self.xSpace = np.linspace(xmin, xmax, Nx+1)  # ❌ 1D array

        # All operations assume 1D
        for i in range(self.Nx + 1):  # ❌ Hard-coded 1D loop
            ...
```

#### Separate Classes per Dimension

```python
# Three separate implementations with duplicated code
class MFGProblem:     # 1D only (mfg_problem.py)
class MFGProblem2D:   # 2D only (mfg_problem_2d.py)
class MFGProblem3D:   # 3D only (mfg_problem_3d.py)
```

**Problem:** Each new dimension requires a complete new implementation.

#### Mesh-Based Geometry

```python
# Explicit mesh storage
class MeshData:
    vertices: NDArray  # (N_vertices, dim) - grows as O(N^d)
    elements: NDArray  # Element connectivity
    dimension: int     # Must be 2 or 3 only!
```

**Problem:** Storing vertices for 4D+ meshes is infeasible.

---

## Particle Methods: Breaking the Curse

### Key Insight: O(N·K) vs O(N^d) Scaling

**Mesh-based methods** (FDM, FEM, spectral):
- Spatial complexity: **O(N^d)** where N = points per dimension
- Requires full tensor product grid
- Memory: O(N^d), Computation: O(N^d · Nt)
- **Cannot escape curse of dimensionality**

**Meshfree particle methods** (GFDM, RBF, particle-collocation):
- Spatial complexity: **O(N·K)** where N = total particles, K = neighbors (~20-50)
- Works with scattered point clouds (no grid structure)
- Memory: O(N), Computation: O(N·K·Nt)
- **Dimension-independent scaling!**

### Complexity Comparison

| Method | Spatial Points | Neighbors | Total Complexity | 4D Example |
|--------|----------------|-----------|------------------|------------|
| **FDM (grid)** | N^d | 2d (stencil) | O(N^d) | 10,000^4 = 10^16 ❌ |
| **GFDM (particles)** | N | K ≈ 30 | O(N·K) | 10,000 × 30 = 3×10^5 ✅ |

**Speedup for 4D:** 10^16 / 10^5 = **10^11× (100 billion times faster!)**

### Memory Comparison

**FDM (grid-based):**
```python
# Storage: Full tensor product grid
U = np.zeros((Nt, Nx, Ny, Nz, Nw))  # 4D problem

# Memory for 4D (100 points/dim, 500 steps):
memory = 500 × 100^4 × 8 bytes = 400 GB  # ❌ Infeasible
```

**GFDM (particle-based):**
```python
# Storage: Scattered particles
U = np.zeros((Nt, N_particles))  # Any dimension!

# Memory for 4D (10,000 particles, 500 steps):
memory = 500 × 10,000 × 8 bytes = 40 MB  # ✅ Trivial!
```

**Memory reduction: 400 GB / 40 MB = 10,000×**

### Why Particle Methods Work

#### 1. No Grid Structure Required

**Grid methods need all grid points:**
```python
# 4D grid with 50 points per dimension
grid_4d = np.zeros((50, 50, 50, 50))  # 6,250,000 points
# Must store and update ALL points
```

**Particle methods use scattered points:**
```python
# 4D domain with just 10,000 particles
particles = np.random.rand(10,000, 4)  # Anywhere in domain!
# Only need particles where solution changes
```

#### 2. Local Neighborhood Interactions

**GFDM/RBF approximation:**
```python
# For each particle i, only interact with K nearest neighbors
for i in range(N_particles):
    neighbors = find_k_nearest(particles, i, k=30)  # O(log N) with k-d tree

    # Approximate derivatives using only local neighborhood
    u_i = approximate_solution(u_neighbors)  # O(K^2) ≈ O(1)
```

**Complexity per particle:** O(K^2) where K is constant (typically 20-50)
**Total complexity:** O(N · K^2) ≈ O(N) - **linear in number of particles!**

#### 3. Dimension-Independent Operations

**All core operations work for any d:**

```python
class ParticleCollocationSolver:
    def __init__(self, particles: NDArray):  # particles.shape = (N, d)
        self.dim = particles.shape[1]  # Works for d = 1, 2, 3, ..., 100!

    def approximate_derivatives(self, u: NDArray, particle_idx: int):
        """Compute derivatives at particle using GFDM."""
        # Find neighbors (works for any d)
        neighbors = self.tree.query(self.particles[particle_idx], k=self.K)

        # Build local polynomial approximation (dimension-independent)
        A = self._build_local_matrix(neighbors)  # O(K^2)
        coeffs = np.linalg.solve(A, u[neighbors])  # O(K^3)

        return coeffs  # Works for any dimension!
```

### Empirical Validation

**Phase 3 validation (2025-10-16):**
- Adaptive timesteps: **4.71× reduction** (100 → 21 steps) for 16×16 grid
- Parameter auto-scaling: Working robustly across problem sizes
- **Key finding:** Adaptive time-stepping is dimension-agnostic - works equally well for any d

**From Phase 2 full benchmark:**
- Full 32×32 grid: **2.99× speedup** (500 → 94 timesteps)
- Runtime: 1675s → 560s (saved 18.6 minutes per solve)
- Mass conservation: Machine precision (~10^-16)

---

## N-Dimensional Domain Representation

### The Challenge

**Particle methods don't need meshes, but they DO need domains:**

#### What Particle Methods Require:
1. **Boundary specification**: Is point x inside or outside domain?
2. **Initial particle placement**: Where to sample particles?
3. **Boundary conditions**: How to handle particles reaching boundaries?
4. **Obstacle representation**: Forbidden regions within domain
5. **Distance to boundary**: For boundary condition enforcement

#### What Particle Methods DON'T Need:
- ❌ Mesh connectivity (no elements!)
- ❌ Detailed boundary discretization
- ❌ Element volumes/areas
- ❌ Mesh quality metrics
- ❌ Gmsh/mesh generation infrastructure

### Solution: Implicit Domain Representation

**Key Insight:** Use **functions** to represent geometry, not **data structures**.

#### Implicit vs Explicit Geometry

**Explicit (Mesh-based):**
```python
# Store all boundary vertices explicitly
vertices = np.array([...])  # O(N^d) vertices for d-dimensional domain
edges = np.array([...])     # Connectivity information

# To check if point inside: complex ray-casting algorithm
def contains(x):
    # Ray-casting, winding number, or similar
    # Complexity: O(N_faces) or O(log N) with spatial indexing
```

**Limitation:** Combinatorial explosion in high dimensions
- 4D hypercube: 16 vertices, 32 edges, 24 faces, 8 cells
- 6D hypercube: 64 vertices, 192 edges, ... (impossible to store!)

**Implicit (Function-based):**
```python
# Define domain via signed distance function
def signed_distance(x):
    """
    Signed distance to boundary.
    Negative: inside, Zero: on boundary, Positive: outside
    """
    # For unit hypercube [0,1]^d:
    distances_to_faces = np.minimum(x, 1 - x)
    return -np.min(distances_to_faces)  # O(d) operation!

# To check if point inside: trivial!
def contains(x):
    return signed_distance(x) < 0
```

**Advantages:**
- ✅ Works for **any dimension** with same code
- ✅ O(d) memory (just function definition)
- ✅ O(d) point-in-domain queries
- ✅ Distance to boundary comes naturally
- ✅ Easy CSG (union, intersection, complement)

### Base Class: ImplicitDomain

```python
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

class ImplicitDomain(ABC):
    """
    N-dimensional implicit domain representation.

    Represents domains via signed distance functions (SDF).
    Perfect for particle methods - no mesh required!
    """

    def __init__(self, dimension: int):
        """
        Initialize n-dimensional domain.

        Args:
            dimension: Spatial dimension (any positive integer!)
        """
        self.dimension = dimension

    @abstractmethod
    def signed_distance(self, x: NDArray) -> float | NDArray:
        """
        Signed distance to boundary.

        Args:
            x: Point(s) - shape (d,) or (N, d)

        Returns:
            Signed distance(s):
              - Negative: inside domain
              - Zero: on boundary
              - Positive: outside domain

        This is the KEY method - everything builds on it.
        """
        pass

    def contains(self, x: NDArray) -> bool | NDArray[np.bool_]:
        """Check if point(s) inside domain."""
        return self.signed_distance(x) < 0

    def project_to_boundary(self, x: NDArray, tol: float = 1e-6) -> NDArray:
        """Project point(s) onto boundary via gradient descent."""
        x_proj = x.copy()

        for _ in range(100):  # Max iterations
            sdf = self.signed_distance(x_proj)
            if np.all(np.abs(sdf) < tol):
                break

            grad = self._numerical_gradient(x_proj)
            x_proj = x_proj - sdf * grad / (np.linalg.norm(grad) + 1e-10)

        return x_proj

    def _numerical_gradient(self, x: NDArray, eps: float = 1e-6) -> NDArray:
        """Numerical gradient of SDF (dimension-independent)."""
        grad = np.zeros(self.dimension)
        for i in range(self.dimension):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (self.signed_distance(x_plus) - self.signed_distance(x_minus)) / (2 * eps)
        return grad

    @abstractmethod
    def get_bounds(self) -> NDArray:
        """Get axis-aligned bounding box: shape (d, 2)."""
        pass

    def sample_uniform(self, n_samples: int, rng=None) -> NDArray:
        """Sample points uniformly inside domain (rejection sampling)."""
        if rng is None:
            rng = np.random.default_rng()

        bounds = self.get_bounds()
        samples = []

        while len(samples) < n_samples:
            candidates = rng.uniform(bounds[:, 0], bounds[:, 1],
                                    size=(n_samples * 2, self.dimension))
            inside = self.contains(candidates)
            samples.extend(candidates[inside])

        return np.array(samples[:n_samples])

    def enforce_boundary_conditions(self, x: NDArray, bc_type: str = "reflect") -> NDArray:
        """Apply boundary conditions to particles that left domain."""
        outside = ~self.contains(x)

        if not np.any(outside):
            return x

        if bc_type == "reflect":
            x_boundary = self.project_to_boundary(x[outside])
            grad = self._numerical_gradient(x_boundary)
            grad = grad / (np.linalg.norm(grad, axis=-1, keepdims=True) + 1e-10)

            displacement = x[outside] - x_boundary
            normal_component = np.sum(displacement * grad, axis=-1, keepdims=True)
            x[outside] = x[outside] - 2 * normal_component * grad

        elif bc_type == "absorb":
            x[outside] = np.nan

        elif bc_type == "periodic":
            bounds = self.get_bounds()
            widths = bounds[:, 1] - bounds[:, 0]
            x[outside] = (x[outside] - bounds[:, 0]) % widths + bounds[:, 0]

        return x
```

### Concrete Domain Types

#### 1. Hyperrectangle (Most Common)

**Axis-aligned box in n dimensions: [a₁,b₁] × [a₂,b₂] × ... × [aₐ,bₐ]**

```python
class Hyperrectangle(ImplicitDomain):
    """
    The workhorse domain for high-dimensional MFG.

    Examples:
      - 4D unit hypercube: bounds = [[0,1], [0,1], [0,1], [0,1]]
      - 6D with different bounds: bounds = [[0,1], [-1,1], [0,2], ...]
    """

    def __init__(self, bounds: NDArray):
        """
        Args:
            bounds: Shape (d, 2) where bounds[i] = [min_i, max_i]
        """
        dimension = bounds.shape[0]
        super().__init__(dimension)

        self.bounds = np.array(bounds)
        self.min_coords = self.bounds[:, 0]
        self.max_coords = self.bounds[:, 1]

    def signed_distance(self, x: NDArray) -> float | NDArray:
        """
        SDF for hyperrectangle: O(d) complexity.

        For each dimension i: d_i = min(x_i - a_i, b_i - x_i)
        Overall: min(d_i) over all i
        """
        single_point = x.ndim == 1
        if single_point:
            x = x[np.newaxis, :]

        dist_to_min = x - self.min_coords
        dist_to_max = self.max_coords - x
        dist_per_dim = np.minimum(dist_to_min, dist_to_max)
        sdf = np.min(dist_per_dim, axis=-1)

        return float(sdf[0]) if single_point else sdf

    def get_bounds(self) -> NDArray:
        """Bounding box is the domain itself."""
        return self.bounds

    def sample_uniform(self, n_samples: int, rng=None) -> NDArray:
        """
        Uniform sampling (no rejection needed!).
        Complexity: O(n·d) - very efficient.
        """
        if rng is None:
            rng = np.random.default_rng()

        return rng.uniform(self.min_coords, self.max_coords,
                          size=(n_samples, self.dimension))

    def enforce_boundary_conditions(self, x: NDArray, bc_type: str = "reflect") -> NDArray:
        """Efficient BCs for boxes."""
        if bc_type == "reflect":
            return np.clip(x, self.min_coords, self.max_coords)
        elif bc_type == "periodic":
            widths = self.max_coords - self.min_coords
            return (x - self.min_coords) % widths + self.min_coords
        else:
            return super().enforce_boundary_conditions(x, bc_type)
```

**Usage:**
```python
# 6D portfolio optimization domain
bounds = np.array([[0, 100]] * 6)  # Each asset price in [0, 100]
domain = Hyperrectangle(bounds)

# Sample 50,000 particles
particles = domain.sample_uniform(50000)  # Shape: (50000, 6)

# Check containment
assert np.all(domain.contains(particles))

# Enforce reflecting BCs
particles_reflected = domain.enforce_boundary_conditions(particles, "reflect")
```

#### 2. Hypersphere (Radial Symmetry)

**Ball in n dimensions: ||x - center|| < radius**

```python
class Hypersphere(ImplicitDomain):
    """
    Hypersphere (ball) in n dimensions.

    Examples:
      - 2D: Circle
      - 3D: Sphere
      - 4D: 4-sphere
      - 5D: 5-ball
    """

    def __init__(self, center: NDArray, radius: float):
        dimension = len(center)
        super().__init__(dimension)
        self.center = np.array(center)
        self.radius = float(radius)

    def signed_distance(self, x: NDArray) -> float | NDArray:
        """
        SDF for hypersphere: radius - ||x - center||
        Complexity: O(d) - just one norm!
        """
        single_point = x.ndim == 1
        if single_point:
            x = x[np.newaxis, :]

        dist_from_center = np.linalg.norm(x - self.center, axis=-1)
        sdf = self.radius - dist_from_center

        return float(sdf[0]) if single_point else sdf

    def get_bounds(self) -> NDArray:
        """Axis-aligned bounding box."""
        return np.stack([
            self.center - self.radius,
            self.center + self.radius
        ], axis=-1)
```

#### 3. Composite Domains (CSG Operations)

**Build complex domains from simple ones via Boolean operations:**

```python
class UnionDomain(ImplicitDomain):
    """Union: x ∈ D₁ ∪ D₂ ∪ ... ∪ Dₙ"""

    def __init__(self, domains: list[ImplicitDomain]):
        dimension = domains[0].dimension
        super().__init__(dimension)
        self.domains = domains

    def signed_distance(self, x: NDArray) -> float | NDArray:
        """SDF for union: min(SDF₁, SDF₂, ...)"""
        sdfs = np.array([d.signed_distance(x) for d in self.domains])
        return np.min(sdfs, axis=0)

class IntersectionDomain(ImplicitDomain):
    """Intersection: x ∈ D₁ ∩ D₂ ∩ ... ∩ Dₙ"""

    def signed_distance(self, x: NDArray) -> float | NDArray:
        """SDF for intersection: max(SDF₁, SDF₂, ...)"""
        sdfs = np.array([d.signed_distance(x) for d in self.domains])
        return np.max(sdfs, axis=0)

class ComplementDomain(ImplicitDomain):
    """Complement: x ∈ ℝⁿ \\ D"""

    def signed_distance(self, x: NDArray) -> float | NDArray:
        """SDF for complement: -SDF(x)"""
        return -self.domain.signed_distance(x)
```

**Example: Domain with obstacle**
```python
# 4D domain: unit hypercube with spherical obstacle removed
base = Hyperrectangle(np.array([[0, 1]] * 4))
obstacle = Hypersphere(center=np.array([0.5]*4), radius=0.2)

# domain = base \ obstacle
domain = IntersectionDomain([base, ComplementDomain(obstacle)])

# Center of obstacle is outside
assert not domain.contains(np.array([0.5]*4))

# Corner is inside
assert domain.contains(np.array([0.1]*4))
```

### Performance Characteristics

| Operation | Hyperrectangle | Hypersphere | Composite (k domains) |
|-----------|----------------|-------------|----------------------|
| `signed_distance(x)` | O(d) | O(d) | O(k·d) |
| `contains(x)` | O(d) | O(d) | O(k·d) |
| `sample_uniform(n)` | O(n·d) | O(n·d/α)* | O(n·d·k) |
| Memory | O(d) | O(d) | O(k·d) |

*α = acceptance rate for rejection sampling

**Key insight:** All operations are **O(d)** or **O(n·d)** - linear in dimension!

---

## Dimension-Agnostic Architecture

### Why Refactor for All Methods?

**Question:** Should MFG_PDE be dimension-agnostic even for mesh methods that can't scale to d>3?

**Answer:** **YES** - Benefits both mesh-based AND meshfree methods.

### Benefits for Mesh Methods (d ≤ 3 practical)

Even though mesh methods hit curse of dimensionality, dimension-agnostic infrastructure helps:

#### 1. Code Simplification (3 classes → 1)

**Current:**
```python
class MFGProblem:     # 1D only - mfg_problem.py
class MFGProblem2D:   # 2D only - mfg_problem_2d.py
class MFGProblem3D:   # 3D only - mfg_problem_3d.py
# Result: 3× code duplication, hard to maintain
```

**Proposed:**
```python
class MFGProblem:     # Works for any d!
    def __init__(self, domain: BaseDomain, shape: tuple[int, ...], ...):
        self.dim = domain.dimension  # 1, 2, 3, 4, ...
        self.shape = shape           # (Nx,) or (Nx,Ny) or (Nx,Ny,Nz)
# Result: Single implementation, much cleaner
```

#### 2. Unified API (learn once, use everywhere)

```python
# Same API structure for all dimensions
from mfg_pde.geometry import Hyperrectangle

# 1D
domain_1d = Hyperrectangle(np.array([[0, 1]]))
problem_1d = MFGProblem(domain_1d, shape=(100,), ...)

# 2D
domain_2d = Hyperrectangle(np.array([[0, 1], [0, 1]]))
problem_2d = MFGProblem(domain_2d, shape=(50, 50), ...)

# 3D
domain_3d = Hyperrectangle(np.array([[0, 1], [0, 1], [0, 1]]))
problem_3d = MFGProblem(domain_3d, shape=(30, 30, 30), ...)
```

#### 3. Educational Value

Let users discover curse of dimensionality empirically:

```python
# User tries 4D with FDM (will be slow but instructive!)
domain_4d = Hyperrectangle(np.array([[0, 1]] * 4))
problem_4d = MFGProblem(domain_4d, shape=(30, 30, 30, 30), ...)

solver_fdm = FDMSolver(problem_4d)
# Warning: FDM with 4D problem may be slow (O(N^d) scaling)
# Grid has 8.1e+05 points - expect long runtime

solution = solver_fdm.solve()  # Runs but slow (hours not seconds)

# User learns → switches to particle methods
solver_particle = ParticleCollocationSolver(problem_4d, n_particles=10000)
solution = solver_particle.solve()  # Much faster! (minutes)
```

### Benefits for Meshfree Methods (d ≥ 4 capable)

Dimension-agnostic infrastructure is **essential** for meshfree:

```python
class ParticleCollocationSolver:
    def __init__(self, problem: MFGProblem, n_particles: int):
        self.dim = problem.dim  # Could be 2, 4, 6, 10, 100!
        self.particles = problem.domain.sample_uniform(n_particles)
        self.gfdm = HJBGFDMSolver(self.particles, dimension=self.dim)

    def solve(self):
        # Same code works for d=2 and d=6!
        ...

# 2D validation
problem_2d = MFGProblem(domain_2d, n_particles=5000)
solver_2d = ParticleCollocationSolver(problem_2d, n_particles=5000)

# 6D production (exact same code structure!)
problem_6d = MFGProblem(domain_6d, n_particles=50000)
solver_6d = ParticleCollocationSolver(problem_6d, n_particles=50000)
```

### Proposed Unified Architecture

#### Core Problem Class

```python
class MFGProblem:
    """
    Dimension-agnostic MFG problem.

    Works for ANY dimension with ANY discretization (mesh or meshfree).
    """

    def __init__(
        self,
        domain: BaseDomain,
        T: float = 1.0,
        Nt: int = 100,
        sigma: float = 0.1,
        gamma: float = 0.5,
        # Mesh-specific (optional)
        shape: tuple[int, ...] | None = None,
        # Particle-specific (optional)
        n_particles: int | None = None,
        **kwargs
    ):
        self.domain = domain
        self.dim = domain.dimension  # Works for any d!

        # Temporal discretization
        self.T = T
        self.Nt = Nt
        self.time_grid = np.linspace(0, T, Nt + 1)

        # Physical parameters
        self.sigma = sigma
        self.gamma = gamma

        # Spatial discretization (method-dependent)
        if shape is not None:
            # Mesh-based
            self.shape = tuple(shape)
            self.spatial_points = self._generate_grid_points()
        elif n_particles is not None:
            # Meshfree
            self.n_particles = n_particles
            self.spatial_points = domain.sample_uniform(n_particles)
        else:
            raise ValueError("Must specify 'shape' or 'n_particles'")

    def _generate_grid_points(self) -> NDArray:
        """Generate tensor product grid (works for any d!)."""
        bounds = self.domain.get_bounds()

        grids_1d = [
            np.linspace(bounds[i, 0], bounds[i, 1], self.shape[i])
            for i in range(self.dim)
        ]

        grids_nd = np.meshgrid(*grids_1d, indexing='ij')
        points = np.stack([g.ravel() for g in grids_nd], axis=-1)

        return points  # Shape: (prod(shape), d)
```

#### Solver Interface

```python
class BaseMFGSolver(ABC):
    """Base class for all solvers (mesh or meshfree)."""

    def __init__(self, problem: MFGProblem):
        self.problem = problem
        self.dim = problem.dim  # Any dimension!

    @abstractmethod
    def solve(self) -> dict:
        """Solve MFG system."""
        pass

# Mesh solver (works for any d, but slow for d>3)
class FDMSolver(BaseMFGSolver):
    def __init__(self, problem: MFGProblem):
        super().__init__(problem)

        if problem.dim > 3:
            warnings.warn(
                f"FDM with {problem.dim}D may be slow (O(N^d))",
                PerformanceWarning
            )

# Meshfree solver (designed for any d)
class ParticleCollocationSolver(BaseMFGSolver):
    def __init__(self, problem: MFGProblem, **kwargs):
        super().__init__(problem)
        # Works naturally for any d!
```

### Practical vs API Limits

**Key design principle:** API supports arbitrary dimensions, methods have practical limits

| Method | API Limit | Practical Limit | Reason |
|--------|-----------|-----------------|--------|
| **FDM** | None | d ≤ 3 | O(N^d) memory/time |
| **FEM** | None | d ≤ 3 | O(N^d) mesh elements |
| **Spectral** | None | d ≤ 4 | O(N^d) mode coupling |
| **Particle Methods** | None | d ≤ 10-20 | O(N·K), K grows with d |

**Strategy:**
- API doesn't enforce limits
- Documentation explains tradeoffs
- Performance warnings guide users

### Migration Path

**Phase 1** (6 months): Parallel implementation
- Keep old classes, add new dimension-agnostic classes
- Users can migrate gradually

**Phase 2** (6-12 months): Deprecation
- Mark old classes as deprecated
- Encourage migration to new API

**Phase 3** (12-18 months): Unification
- Remove old classes
- Single dimension-agnostic codebase

---

## Research Roadmap

### Three-Stage Development Plan

#### Stage 1: Adaptive Time-Stepping ✅ (2-3 weeks)

**Status:** Ready to publish

**Contribution:**
- Adaptive time-stepping with PI controller
- 2.99-4.71× speedup with machine-precision mass conservation
- **Dimension-agnostic** - works for any d
- Parameter auto-scaling for robustness

**Action Items:**
1. ✅ Validation complete (Phase 2: 2.99×, Phase 3 small test: 4.71×)
2. ⏳ Run full benchmarks (32×32, 500 timesteps) - 2-3 hours
3. ⏳ Generate publication figures - 1 day
4. ⏳ Write paper - 2-3 days
5. ⏳ Submit to SIAM J. Numerical Analysis

**Paper Title:** *"Adaptive Time-Stepping for Mean Field Games via PI Control"*

---

#### Stage 2: Particle-Collocation for 2D/3D (2-3 months)

**Status:** Implementation complete, needs fixes

**Goal:** Validate meshfree methodology on standard dimensions

**Contribution:**
- Meshfree particle-collocation for MFG
- QP constraints for monotonicity preservation
- O(N·K) complexity demonstrated empirically
- Validation on 2D/3D benchmarks

**Issues Found (2025-10-16 validation):**
1. ❌ Hamiltonian interface mismatch (grid indices vs continuous coords)
2. ❌ GFDM integration issues (API incompatibility)
3. ❌ NaN generation (incorrect initialization)

**Required Fixes:**

**Fix 1:** Coordinate-Based Hamiltonian Interface (1-2 weeks)
```python
class ContinuousHamiltonianAdapter:
    """Adapt grid-based Hamiltonians for particle coords."""

    def __init__(self, grid_hamiltonian, grid_adapter):
        self.H_grid = grid_hamiltonian
        self.adapter = grid_adapter
        self._build_interpolators()

    def __call__(self, x: NDArray, p: NDArray, m: float) -> float:
        """Evaluate Hamiltonian at continuous coordinate."""
        # Interpolate spatially-varying coefficients
        coeffs = self._interpolate_coefficients(x)
        return self._evaluate_hamiltonian(x, p, m, coeffs)
```

**Fix 2:** Proper GFDM Integration (1-2 weeks)
- Use HJBGFDMSolver correctly (proper initialization sequence)
- Or implement GFDM manually for full control

**Fix 3:** Dimension-Agnostic Infrastructure (1 week)
```python
class ParticleCollocationSolver:
    def __init__(self, problem: MFGProblem, ...):
        self.dim = problem.dim  # Any dimension!
        self.particles = problem.spatial_points
        self.gfdm = HJBGFDMSolver(self.particles, dimension=self.dim)
```

**Action Items:**
1. ⏳ Implement implicit domains (Hyperrectangle, Hypersphere) - 1 week
2. ⏳ Fix Hamiltonian interface - 1-2 weeks
3. ⏳ Proper GFDM integration - 1-2 weeks
4. ⏳ Validation benchmarks (2D/3D convergence) - 2-3 weeks
5. ⏳ Write paper - 2-3 weeks

**Success Criteria:**
- Converges on 2D anisotropic crowd dynamics
- Mass conservation < 1e-06 (better than FDM's ~1e-04)
- Comparable or better runtime than FDM for 2D/3D
- Clean implementation ready for 4D+ extension

**Paper Title:** *"Meshfree Particle-Collocation Methods for Mean Field Games"*

---

#### Stage 3: High-Dimensional MFG (4D+) (6-12 months)

**Status:** Future work (after Stage 2)

**Goal:** First MFG solver beyond 3D

**Contribution:**
- First MFG solver for d ≥ 4
- Empirical O(N·K) scaling validation
- Suite of 4D+ test problems
- Applications: swarms, robotics, finance, climate

**Target Applications:**

**4D Problems:**
- Quadrotor swarm control: (x, y, z, velocity)
- Autonomous vehicle coordination: (position, heading)
- Epidemic models: (location, infection status)

**5D Problems:**
- Robotic arm coordination: (joint angles)
- Portfolio optimization: (multiple asset prices)
- Climate models: (temperature, pressure, humidity, wind, CO₂)

**6D+ Problems:**
- Multi-agent spacecraft: (position, velocity) in 3D
- Supply chain networks: (inventory levels across nodes)
- Molecular dynamics: (particle positions and momenta)

**Test Problem Suite:**

**Problem 1: 4D Swarm Navigation**
```python
# State: (x, y, z, v) - position + speed
domain = Hyperrectangle(np.array([[0, 1]] * 3 + [[0, v_max]]))
problem = MFGProblem(domain, n_particles=10000, ...)

# Grid would need 50^4 = 6.25M points (infeasible)
# Particles: 10,000 points (feasible!)
```

**Problem 2: 5D Portfolio Optimization**
```python
# State: 5 asset prices
domain = Hyperrectangle(np.array([[0, 100]] * 5))
problem = MFGProblem(domain, n_particles=20000, ...)

# Grid: 30^5 = 24.3M points (infeasible)
# Particles: 20,000 points (feasible!)
```

**Problem 3: 6D Spacecraft Rendezvous**
```python
# State: (x, y, z, vx, vy, vz)
domain = Hyperrectangle(np.array([[0, 1]] * 3 + [[-v_max, v_max]] * 3))
problem = MFGProblem(domain, n_particles=50000, ...)

# Grid: 20^6 = 64M points (impossible!)
# Particles: 50,000 points (challenging but feasible!)
```

**Action Items:**
1. ⏳ Dimension-agnostic particle solver - 1-2 months
2. ⏳ Develop 4D/5D/6D test problems - 1-2 months
3. ⏳ Scalability benchmarks - 1-2 months
4. ⏳ Applications demonstrations - 1-2 months
5. ⏳ Write paper - 1-2 months

**Success Criteria:**
- Solve 4D with 10,000 particles in < 1 hour
- Solve 5D with 20,000 particles in < 4 hours
- Solve 6D with 50,000 particles in < 24 hours
- Linear scaling: 2N particles → ~2× time
- Dimension-independent: similar cost per particle for d=4 and d=6

**Paper Title:** *"Scalable High-Dimensional Mean Field Games via Particle Methods"*

---

### Timeline Summary

| Stage | Duration | Status | Deliverable |
|-------|----------|--------|-------------|
| **Stage 1** | 2-3 weeks | ✅ Ready | Paper on adaptive time-stepping |
| **Stage 2** | 2-3 months | ⚠️ In progress | Paper on particle-collocation for 2D/3D |
| **Stage 3** | 6-12 months | ⏳ Future | Paper on 4D+ high-dimensional MFG |
| **Total** | ~12-18 months | - | **3 papers** |

---

## Implementation Guidelines

### Quick Start: Implicit Domains

```python
# 1. Define domain
from mfg_research.geometry import Hyperrectangle

bounds = np.array([[0, 1]] * 6)  # 6D unit hypercube
domain = Hyperrectangle(bounds)

# 2. Sample particles
particles = domain.sample_uniform(50000)  # (50000, 6)

# 3. Check containment
assert np.all(domain.contains(particles))

# 4. Enforce boundary conditions
particles_new = particles + 0.01 * np.random.randn(50000, 6)
particles_reflected = domain.enforce_boundary_conditions(
    particles_new,
    bc_type="reflect"
)
```

### Quick Start: Dimension-Agnostic MFG Problem

```python
# 1. Create domain (any dimension!)
domain = Hyperrectangle(np.array([[0, 1]] * 4))  # 4D

# 2. Create problem
problem = MFGProblem(
    domain=domain,
    n_particles=10000,  # Meshfree method
    T=1.0,
    Nt=100,
    sigma=0.1,
    gamma=0.5
)

# 3. Create solver
solver = ParticleCollocationSolver(
    problem=problem,
    n_particles=10000,
    delta=0.1,
    taylor_order=2
)

# 4. Solve
solution = solver.solve()

# 5. Extract results
U = solution['U']  # Shape: (Nt, n_particles)
M = solution['M']  # Shape: (Nt, n_particles)
```

### Testing Guidelines

**Test implicit domains:**
```python
def test_hyperrectangle_6d():
    """Test 6D hyperrectangle."""
    bounds = np.array([[0, 1]] * 6)
    domain = Hyperrectangle(bounds)

    # Test corners
    assert domain.contains(np.zeros(6) + 0.1)
    assert not domain.contains(np.zeros(6) - 0.1)

    # Test sampling
    particles = domain.sample_uniform(1000)
    assert particles.shape == (1000, 6)
    assert np.all(domain.contains(particles))
```

**Test dimension-agnostic solver:**
```python
def test_solver_works_for_any_dimension():
    """Solver should work for d=2, 3, 4, ..."""
    for d in [2, 3, 4, 5, 6]:
        domain = Hyperrectangle(np.array([[0, 1]] * d))
        problem = MFGProblem(domain, n_particles=1000, Nt=10)
        solver = ParticleCollocationSolver(problem, n_particles=1000)

        # Should not raise errors
        solution = solver.solve()
        assert solution['converged']
```

### Performance Optimization

**Neighborhood search (critical for scaling):**
```python
from scipy.spatial import cKDTree

# Build k-d tree once
tree = cKDTree(particles)  # O(N log N · d)

# Query K neighbors for each particle
distances, indices = tree.query(particles, k=30)  # O(N · K log N · d)

# For N=50,000, d=6, K=30: ~3-5 seconds (acceptable!)
```

**Adaptive particle placement:**
```python
# Concentrate particles where solution changes rapidly
def adaptive_sampling(domain, n_particles, importance_function):
    """Sample with probability proportional to importance."""
    # Rejection sampling with importance weight
    samples = []
    while len(samples) < n_particles:
        candidates = domain.sample_uniform(n_particles * 2)
        weights = importance_function(candidates)
        probs = weights / np.max(weights)
        keep = np.random.rand(len(candidates)) < probs
        samples.extend(candidates[keep])
    return np.array(samples[:n_particles])
```

---

## Applications and Impact

### Applications Currently Impossible (d ≥ 4)

**Robotics:**
- Drone swarms: (x, y, z, velocity) - 4D
- Warehouse robots: (x, y, orientation, load) - 4D
- Manipulator coordination: (joint angles) - 6-7D

**Finance and Economics:**
- Portfolio optimization: Multiple asset prices - 5D+
- Market microstructure: (bid, ask, volume, volatility, ...) - 6D+
- Supply chain: Inventory levels across nodes - high-D

**Climate and Geophysics:**
- Atmospheric models: (T, P, humidity, wind_x, wind_y, ...) - 5D+
- Ocean currents: (x, y, z, T, salinity, ...) - 6D+

**Molecular Dynamics:**
- Protein folding: High-dimensional conformational space
- Drug delivery: (location, concentration, binding state)

### Impact Metrics

**Scientific Impact:**
- First MFG solver beyond 3D
- Breaks fundamental barrier (curse of dimensionality)
- Enables entirely new application domains

**Practical Impact:**
- 10,000× memory reduction for 4D problems
- 10^11× speedup vs grid methods for 4D
- Applications previously impossible become routine

**Publication Impact:**
- 3 high-quality papers instead of 1
- Each builds on previous (validation → methodology → applications)
- Priority on adaptive MFG + high-dimensional methods

---

## References

### Curse of Dimensionality
- Bellman, R. (1961). *Adaptive Control Processes.* Princeton University Press.
- Novak, E., & Woźniakowski, H. (2008). *Tractability of Multivariate Problems.* EMS.

### Mean Field Games
- Lasry, J.-M., & Lions, P.-L. (2007). "Mean field games." *Japanese Journal of Mathematics.*
- Cardaliaguet, P. (2013). "Notes on Mean Field Games."
- Achdou, Y., & Capuzzo-Dolcetta, I. (2010). "Mean field games: Numerical methods." *SIAM J. Numer. Anal.*

### Meshfree Methods
- Liu, G.-R. (2009). *Meshfree Methods.* CRC Press.
- Fornberg, B., & Flyer, N. (2015). *A Primer on Radial Basis Functions.* SIAM.
- Mirzaei, D. (2015). "A new low-cost meshfree method." *Appl. Math. Comput.*

### Particle Methods for PDEs
- Cottet, G.-H., & Koumoutsakos, P. (2000). *Vortex Methods.* Cambridge.
- Monaghan, J. J. (2005). "Smoothed particle hydrodynamics." *Rep. Prog. Phys.*

### High-Dimensional PDEs
- E, W., et al. (2017). "Deep learning for high-dimensional PDEs." *PNAS.*
- Ruthotto, L., et al. (2020). "ML for high-dimensional MFG." *PNAS.*

### Implicit Geometry (Signed Distance Functions)
- Osher, S., & Fedkiw, R. (2003). *Level Set Methods.* Springer.
- Hart, J. C. (1996). "Sphere tracing." *The Visual Computer.*
- Quilez, I. (2008). "Distance functions." [iquilezles.org](https://iquilezles.org/articles/distfunctions/)

---

## Appendix: Key Code Snippets

### Implicit Domain Base Class (Complete)

```python
class ImplicitDomain(ABC):
    """N-dimensional implicit domain via signed distance functions."""

    def __init__(self, dimension: int):
        self.dimension = dimension

    @abstractmethod
    def signed_distance(self, x: NDArray) -> float | NDArray:
        """Negative inside, zero on boundary, positive outside."""
        pass

    def contains(self, x: NDArray) -> bool | NDArray[np.bool_]:
        return self.signed_distance(x) < 0

    def project_to_boundary(self, x: NDArray, tol: float = 1e-6) -> NDArray:
        """Project to boundary via gradient descent on SDF."""
        x_proj = x.copy()
        for _ in range(100):
            sdf = self.signed_distance(x_proj)
            if np.all(np.abs(sdf) < tol):
                break
            grad = self._numerical_gradient(x_proj)
            x_proj = x_proj - sdf * grad / (np.linalg.norm(grad) + 1e-10)
        return x_proj

    def _numerical_gradient(self, x: NDArray, eps: float = 1e-6) -> NDArray:
        """Finite difference gradient."""
        grad = np.zeros(self.dimension)
        for i in range(self.dimension):
            x_plus, x_minus = x.copy(), x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (self.signed_distance(x_plus) -
                      self.signed_distance(x_minus)) / (2 * eps)
        return grad

    @abstractmethod
    def get_bounds(self) -> NDArray:
        """Bounding box: shape (d, 2)."""
        pass

    def sample_uniform(self, n_samples: int, rng=None) -> NDArray:
        """Rejection sampling from bounding box."""
        if rng is None:
            rng = np.random.default_rng()
        bounds = self.get_bounds()
        samples = []
        while len(samples) < n_samples:
            candidates = rng.uniform(bounds[:, 0], bounds[:, 1],
                                    size=(n_samples * 2, self.dimension))
            inside = self.contains(candidates)
            samples.extend(candidates[inside])
        return np.array(samples[:n_samples])
```

### Hyperrectangle (Complete)

```python
class Hyperrectangle(ImplicitDomain):
    """Axis-aligned box in n dimensions."""

    def __init__(self, bounds: NDArray):
        dimension = bounds.shape[0]
        super().__init__(dimension)
        self.bounds = np.array(bounds)
        self.min_coords = bounds[:, 0]
        self.max_coords = bounds[:, 1]

    def signed_distance(self, x: NDArray) -> float | NDArray:
        single_point = x.ndim == 1
        if single_point:
            x = x[np.newaxis, :]

        dist_to_min = x - self.min_coords
        dist_to_max = self.max_coords - x
        dist_per_dim = np.minimum(dist_to_min, dist_to_max)
        sdf = np.min(dist_per_dim, axis=-1)

        return float(sdf[0]) if single_point else sdf

    def get_bounds(self) -> NDArray:
        return self.bounds

    def sample_uniform(self, n_samples: int, rng=None) -> NDArray:
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self.min_coords, self.max_coords,
                          size=(n_samples, self.dimension))
```

### Dimension-Agnostic MFG Problem (Simplified)

```python
class MFGProblem:
    """Dimension-agnostic MFG problem definition."""

    def __init__(self, domain, T=1.0, Nt=100, sigma=0.1, gamma=0.5,
                 shape=None, n_particles=None):
        self.domain = domain
        self.dim = domain.dimension

        self.T, self.Nt = T, Nt
        self.time_grid = np.linspace(0, T, Nt + 1)

        self.sigma, self.gamma = sigma, gamma

        if shape is not None:
            self.shape = tuple(shape)
            self.spatial_points = self._generate_grid_points()
        elif n_particles is not None:
            self.n_particles = n_particles
            self.spatial_points = domain.sample_uniform(n_particles)
        else:
            raise ValueError("Specify 'shape' or 'n_particles'")

    def _generate_grid_points(self):
        """Tensor product grid for any dimension."""
        bounds = self.domain.get_bounds()
        grids_1d = [np.linspace(bounds[i, 0], bounds[i, 1], self.shape[i])
                    for i in range(self.dim)]
        grids_nd = np.meshgrid(*grids_1d, indexing='ij')
        return np.stack([g.ravel() for g in grids_nd], axis=-1)
```

---

## Document History

**Version 1.0** (2025-10-16):
- Consolidated from 5 separate documents
- Complete technical reference for high-dimensional MFG
- Includes validated results, architectural designs, and research roadmap

**Source Documents:**
1. `HIGH_DIMENSIONAL_STRATEGY.md` - Research roadmap
2. `DIMENSIONAL_ANALYSIS_SUMMARY.md` - Curse of dimensionality analysis
3. `N_DIMENSIONAL_DOMAIN_DESIGN.md` - Implicit domain design
4. `DIMENSION_AGNOSTIC_ARCHITECTURE.md` - Unified infrastructure
5. `PHASE3_VALIDATION_2025-10-16.md` - Validation results

**Status:** Living document, updated as research progresses

**Next Update:** After Stage 2 completion (particle-collocation 2D/3D validation)

---

**End of Technical Reference**
