# GFDM Computational Optimization

**Purpose**: Analysis of computational bottlenecks and optimization strategies for GFDM solvers, focusing on neighbor search and sparse matrix operations.

**Level**: Implementation/Performance

**Status**: Current implementation analysis with proposed improvements

---

## Overview

GFDM solvers face computational challenges distinct from grid-based methods due to their meshfree nature. This document analyzes:

1. **Current implementation** in `hjb_gfdm.py`
2. **Neighbor search strategies** (KDTree vs Cell Lists vs Brute Force)
3. **Sparse matrix opportunities** for large-scale problems
4. **Recommendations** by problem type

---

## 1. Current Implementation Analysis

### 1.1 Initialization Phase

The GFDM solver performs expensive operations at initialization (`__init__`):

```python
# hjb_gfdm.py:281-428
def _build_neighborhood_structure(self):
    distances = cdist(self.collocation_points, self.collocation_points)  # O(N^2)
    for i in range(self.n_points):
        neighbor_mask = distances[i, :] < delta_current
        ...
```

**Complexity**: $O(N^2)$ distance computation, $O(N \cdot k)$ neighborhood storage

### 1.2 Taylor Matrix Precomputation

```python
# hjb_gfdm.py:449-549
def _build_taylor_matrices(self):
    for i in range(self.n_points):
        # Per-point SVD/QR decomposition
        U, S, Vt = np.linalg.svd(WA, full_matrices=False)  # O(k^3)
```

**Complexity**: $O(N \cdot k^3)$ where $k$ is neighborhood size

### 1.3 Per-Timestep Derivative Computation

```python
# hjb_gfdm.py:587-700
def approximate_derivatives(self, u_values, point_idx):
    # SVD-based solve using precomputed decomposition
    derivative_coeffs = Vt.T @ (U.T @ Wb / S)  # O(k^2)
```

**Complexity**: $O(N \cdot k^2)$ per time step

### 1.4 Summary: Current Bottlenecks

| Phase | Current | Complexity | Bottleneck For |
|:------|:--------|:-----------|:---------------|
| Distance computation | `cdist` | $O(N^2)$ | Large $N$ (> 10K) |
| Taylor matrices | Dense SVD | $O(N \cdot k^3)$ | Large $k$ |
| Derivative solve | Dense matvec | $O(N \cdot k^2 \cdot N_t)$ | Many time steps |
| Memory | Dense `distances` | $O(N^2)$ | Large $N$ |

---

## 2. Neighbor Search Strategies

### 2.1 Decision Matrix

| Method | Best For | Search Complexity | Build Complexity |
|:-------|:---------|:------------------|:-----------------|
| **Brute Force** (`cdist`) | $d > 20$ or small $N$ | $O(N)$ per query | $O(1)$ |
| **KDTree** | $d \le 15$, large $N$, static points | $O(\log N)$ avg | $O(N \log N)$ |
| **Cell Lists** | Fixed radius $\delta$, moving particles | $O(1)$ avg | $O(N)$ |

### 2.2 Dimensional Analysis

**Curse of Dimensionality**: KDTree degrades when $N < 2^d$ because branch pruning becomes ineffective.

| Dimension $d$ | Threshold $N$ | Recommendation |
|:--------------|:--------------|:---------------|
| 1-3 | Any | KDTree |
| 4-10 | $N > 10^4$ | KDTree |
| 10-20 | $N > 10^6$ | KDTree or Brute |
| > 20 | Any | Brute Force |

### 2.3 Static vs Moving Points

**GFDM (HJB solver)**: Collocation points are **static** throughout the solve. Tree/grid construction is paid once at initialization.

**FP Particle solver**: Particles **move** every time step. Rebuild cost matters:
- KDTree: $O(N \log N)$ per step
- Cell Lists: $O(N)$ per step

### 2.4 Recommended Implementation

For GFDM with typical MFG problems ($d \le 3$, $N = 10^2 - 10^5$):

```python
# Proposed: Replace cdist with cKDTree
from scipy.spatial import cKDTree

def _build_neighborhood_structure(self):
    tree = cKDTree(self.collocation_points)

    for i in range(self.n_points):
        # O(log N) per query instead of O(N)
        neighbor_indices = tree.query_ball_point(
            self.collocation_points[i], r=self.delta
        )
        neighbor_distances = tree.query(
            self.collocation_points[neighbor_indices], k=1
        )[0]
        ...
```

**Expected Speedup**: 10-100x for neighbor search phase when $N > 10^4$

---

## 3. Sparse Matrix Opportunities

### 3.1 Sparsity Structure

GFDM has inherent sparsity: each point only depends on $k \ll N$ neighbors.

| Matrix | Size | Non-zeros per row | Density |
|:-------|:-----|:------------------|:--------|
| Neighborhood | $N \times N$ | $k$ | $k/N$ |
| Global gradient $D_x$ | $N \times N$ | $k$ | $k/N$ |
| Global Laplacian $\Delta_h$ | $N \times N$ | $k$ | $k/N$ |

For $N = 10^4$, $k = 20$: density = 0.2% (99.8% zeros)

### 3.2 Current: Point-by-Point Computation

```python
# Current approach: O(N * k^2) per timestep
for i in range(N):
    derivs = self.approximate_derivatives(u_values, i)
    du_dx[i] = derivs[(1, 0)]
    du_dy[i] = derivs[(0, 1)]
```

### 3.3 Proposed: Sparse Global Operators

Precompute sparse differentiation matrices at initialization:

```python
from scipy.sparse import lil_matrix, csr_matrix

def _build_sparse_operators(self):
    """Build sparse global differentiation matrices."""
    Dx = lil_matrix((self.n_points, self.n_points))
    Dy = lil_matrix((self.n_points, self.n_points))
    Laplacian = lil_matrix((self.n_points, self.n_points))

    for i in range(self.n_points):
        # Extract precomputed MLS coefficients
        coeffs = self._get_derivative_coefficients(i)
        neighbors = self.neighborhoods[i]['indices']

        for j, neighbor_idx in enumerate(neighbors):
            if neighbor_idx >= 0:  # Not ghost
                Dx[i, neighbor_idx] = coeffs['dx'][j]
                Dy[i, neighbor_idx] = coeffs['dy'][j]
                Laplacian[i, neighbor_idx] = coeffs['laplacian'][j]

    # Convert to CSR for fast matvec
    self._Dx = Dx.tocsr()
    self._Dy = Dy.tocsr()
    self._Laplacian = Laplacian.tocsr()

def compute_derivatives_fast(self, u_values):
    """O(nnz) sparse matvec instead of O(N * k^2) point-by-point."""
    du_dx = self._Dx @ u_values
    du_dy = self._Dy @ u_values
    return du_dx, du_dy
```

### 3.4 Expected Benefits

| Operation | Current | With Sparse |
|:----------|:--------|:------------|
| Init (one-time) | $O(N^2)$ | $O(N \cdot k)$ + sparse build |
| Derivatives (per step) | $O(N \cdot k^2)$ | $O(N \cdot k)$ |
| Memory | $O(N^2)$ | $O(N \cdot k)$ |

**Speedup**:
- Memory: $N/k \approx 500$x for $N=10^4$, $k=20$
- Per-step: $k$x $\approx 20$x

---

## 4. Implementation Priority

### 4.1 For GFDM HJB Solver (Static Points)

| Priority | Optimization | Effort | Benefit |
|:---------|:-------------|:-------|:--------|
| **1** | Sparse global operators | Medium | 10-20x per step |
| **2** | KDTree neighbor search | Low | 10-100x init |
| **3** | Parallel point loops | Low | 2-8x (cores) |

**Rationale**: Sparse operators provide the biggest benefit because they accelerate every time step. KDTree only helps initialization.

### 4.2 For FP Particle Solver (Moving Points)

| Priority | Optimization | Effort | Benefit |
|:---------|:-------------|:-------|:--------|
| **1** | Cell Lists | Medium | $O(N)$ rebuild |
| **2** | GPU kernels | High | 10-100x |
| **3** | Adaptive neighborhoods | Medium | Varies |

**Rationale**: Particles move, so neighbor search is repeated every step. Cell Lists have $O(N)$ rebuild vs KDTree's $O(N \log N)$.

---

## 5. Weight Function Considerations

### 5.1 Current Implementation

```python
# hjb_gfdm.py:551-585
def _compute_weights(self, distances):
    if self.weight_function == "wendland":
        kernel = WendlandKernel(k=2)  # C^4 continuity
        return kernel(distances, h=self.delta)
    elif self.weight_function == "gaussian":
        kernel = GaussianKernel()
        return kernel(distances, h=self.weight_scale)
    ...
```

### 5.2 Compact vs Global Support

| Kernel | Support | Sparse Compatible | Notes |
|:-------|:--------|:------------------|:------|
| **Wendland** | Compact ($r < h$) | Yes | Exact zeros outside support |
| **Gaussian** | Global (infinite) | Truncate | Must set cutoff |
| **Inverse Distance** | Global | Truncate | Must set cutoff |

**Recommendation**: Use Wendland for sparse implementations (natural cutoff at $\delta$).

---

## 6. Benchmarking Guidance

### 6.1 Test Parameters

```python
# Scaling test configurations
test_configs = [
    {"N": 100, "k": 10, "d": 2},    # Small
    {"N": 1000, "k": 20, "d": 2},   # Medium
    {"N": 10000, "k": 30, "d": 2},  # Large
    {"N": 100000, "k": 50, "d": 2}, # Very large
]
```

### 6.2 Metrics to Track

1. **Init time**: Neighborhood + Taylor matrix construction
2. **Per-step time**: Derivative computation
3. **Memory**: Peak and steady-state
4. **Accuracy**: Compare against dense implementation

---

## 7. References

### Spatial Data Structures

1. **Bentley (1975)**: "Multidimensional binary search trees used for associative searching"
   - KDTree original paper

2. **Allen & Tildesley (1987)**: *Computer Simulation of Liquids*
   - Cell Lists for particle simulations

3. **Muja & Lowe (2009)**: "Fast Approximate Nearest Neighbors with Automatic Algorithm Configuration"
   - FLANN library, automatic algorithm selection

### Sparse Linear Algebra

4. **Davis (2006)**: *Direct Methods for Sparse Linear Systems*
   - Sparse matrix algorithms

5. **Saad (2003)**: *Iterative Methods for Sparse Linear Systems*
   - Sparse iterative solvers

### GFDM Specific

6. **Benito et al. (2001)**: "Influence of several factors in the generalized finite difference method"
   - Original GFDM formulation

---

## 8. Code References

- **Base GFDM**: `mfg_pde/utils/numerical/gfdm_operators.py`
  - `GFDMOperator` class: neighborhoods, Taylor matrices, derivatives
  - Reusable for FP solvers, general meshfree computation

- **HJB Solver**: `mfg_pde/alg/numerical/hjb_solvers/hjb_gfdm.py`
  - `HJBGFDMSolver` class: composes GFDMOperator
  - Adds: ghost particles, adaptive delta, QP constraints

- **Weight kernels**: `mfg_pde/utils/numerical/kernels.py`
  - Wendland, Gaussian implementations (general numerical kernels)

---

## Maintenance

**Last Updated**: 2025-12-03

**Status**:
- Current implementation documented
- Sparse optimization proposed (not implemented)
- KDTree optimization proposed (not implemented)

**Related Issues**: Performance optimization for large-scale MFG problems
