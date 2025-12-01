# Adaptive Mesh Refinement for Mean Field Games

**Date**: August 1, 2025
**Status**: Production Ready
**Architecture**: Enhancement-based (AMR wraps any base solver)
**Modules**: `mfg_pde.geometry.amr_mesh`, `mfg_pde.alg.amr_enhancement`, `mfg_pde.geometry.one_dimensional_amr`

## Table of Contents

1. [Introduction and Motivation](#introduction-and-motivation)
2. [Mathematical Foundation](#mathematical-foundation)
3. [AMR Framework](#amr-framework)
4. [Error Estimation](#error-estimation)
5. [Conservative Interpolation](#conservative-interpolation)
6. [System Architecture](#system-architecture)
7. [Integration with MFG Solvers](#integration-with-mfg-solvers)
8. [Performance Optimization](#performance-optimization)
9. [Usage Guide](#usage-guide)
10. [API Reference](#api-reference)

---

## Introduction and Motivation

### Executive Summary

This document presents the comprehensive framework for **Adaptive Mesh Refinement (AMR)** in Mean Field Games, addressing the computational challenges of capturing sharp fronts, boundary layers, and multi-scale phenomena inherent in MFG solutions. The AMR system provides automatic spatial resolution adaptation while maintaining mass conservation and Nash equilibrium properties.

### AMR Motivation for Mean Field Games

Mean Field Games present unique computational challenges that make adaptive mesh refinement particularly valuable:

#### Challenge 1: Multi-scale Dynamics

- **Agent Concentration**: Agents concentrate in regions of low cost, creating sharp density gradients that require high spatial resolution
- **Boundary Layers**: Value function exhibits boundary layers near obstacles or domain boundaries requiring localized refinement
- **Focusing Effects**: Optimal control policies create focusing effects where agents converge to narrow corridors or regions

#### Challenge 2: Computational Efficiency

- **Prohibitive Uniform Grids**: Uniform fine grids across entire domain are computationally prohibitive for large-scale problems
- **Sparse Feature Distribution**: Most domain regions may require only coarse resolution, with fine resolution needed only locally
- **Dynamic Evolution**: Solution features evolve in time, requiring temporal mesh adaptation

#### Challenge 3: Conservation Properties

- **Mass Conservation**: Must maintain `∫_Ω m(x,t) dx = ∫_Ω m₀(x) dx` throughout mesh adaptation
- **Nash Equilibrium**: Solutions must remain optimal under mesh refinement/coarsening
- **Boundary Conditions**: Proper preservation of boundary conditions across mesh hierarchy levels

#### Expected Benefits

**Computational Efficiency:**
- 3-10× speedup for problems with localized features
- Memory reduction of 30-70% compared to uniform fine grids
- Enables large-domain problems previously intractable

**Physical Fidelity:**
- Proper resolution of sharp fronts and shocks
- Accurate capture of boundary layer phenomena
- Multi-scale feature representation

**Research Capabilities:**
- Enable new classes of complex MFG problems
- Support high-accuracy analysis and validation
- Facilitate convergence studies and error analysis

---

## Mathematical Foundation

### Mean Field Game System on Adaptive Meshes

The Mean Field Game system consists of the Hamilton-Jacobi-Bellman (HJB) equation and the Fokker-Planck-Kolmogorov (FPK) equation:

**HJB Equation (backward in time):**
```
∂u/∂t + H(x, ∇u, m) = 0,  (x,t) ∈ Ω × [0,T]
u(x,T) = g(x,m(x,T))
```

**FPK Equation (forward in time):**
```
∂m/∂t - σ²/2 Δm - div(m ∇ₚH(x, ∇u, m)) = 0,  (x,t) ∈ Ω × [0,T]
m(x,0) = m₀(x)
```

where the Hamiltonian `H(x,p,m)` may exhibit strong spatial variation requiring adaptive resolution.

### Hierarchical Mesh Structure

The AMR system uses a **hierarchical mesh structure** (quadtree in 2D, octree in 3D, binary tree in 1D) where the computational domain Ω is recursively subdivided:

- **Root Level (Level 0)**: Ω = Ω₀
- **Level k**: Each cell Ωᵢᵏ can be subdivided into children cells Ωⱼᵏ⁺¹
  - 2 children in 1D
  - 4 children in 2D (quadtree)
  - 8 children in 3D (octree)

**Cell Hierarchy Property:**
```
Ωᵢᵏ = ⋃ⱼ∈C(i) Ωⱼᵏ⁺¹
```
where C(i) denotes the children of cell i.

**Refinement Ratio:**
```
Level 0: Base mesh                    Δx₀
Level 1: Refined regions              Δx₁ = Δx₀/2
Level 2: Highly refined regions       Δx₂ = Δx₀/4
...
Level L: Maximum refinement           Δxₗ = Δx₀/2^L
```

### Solution Representation

On the adaptive mesh, the discrete solutions are represented as:
- **Value Function**: u(x,t) ≈ ∑ᵢ uᵢᵏ(t) φᵢᵏ(x)
- **Density Function**: m(x,t) ≈ ∑ᵢ mᵢᵏ(t) ψᵢᵏ(x)

where φᵢᵏ and ψᵢᵏ are basis functions on cell Ωᵢᵏ.

### Error Estimation Theory

**A Posteriori Error Indicators:**

The AMR system uses gradient-based error estimation:

```
ηᵢᵏ = max(ηᵢᵏ⁽ᵘ⁾, ηᵢᵏ⁽ᵐ⁾)
```

where:
- **Value Function Error**: ηᵢᵏ⁽ᵘ⁾ = hᵢᵏ ||∇u||_{L²(Ωᵢᵏ)} + (hᵢᵏ)² ||∇²u||_{L²(Ωᵢᵏ)}
- **Density Function Error**: ηᵢᵏ⁽ᵐ⁾ = hᵢᵏ ||∇m||_{L²(Ωᵢᵏ)} + (hᵢᵏ)² ||∇²m||_{L²(Ωᵢᵏ)}

Here hᵢᵏ = diam(Ωᵢᵏ) is the cell diameter.

**Refinement Criterion:**
```
Refine Ωᵢᵏ if: ηᵢᵏ > τ_ref
Coarsen Ωᵢᵏ if: ηᵢᵏ < τ_coarse < τ_ref
```

### Alternative Error Estimation Strategies

#### Strategy 1: Gradient-Based Refinement
Refine where solution gradients are large:
```
E_grad(K) = ||∇u||_{L²(K)} + ||∇m||_{L²(K)}
```

#### Strategy 2: Multi-scale Feature Detection
Identify features requiring resolution:
```
E_feature(K) = |u_max - u_min|_K + |m_max - m_min|_K + λ · |∇·(m∇u)|_K
```

**Physical interpretation:**
- Value function variation captures cost landscape complexity
- Density variation captures population concentration
- Coupling term captures MFG interaction strength

#### Strategy 3: Nash Equilibrium Error
Measure deviation from equilibrium conditions:
```
E_nash(K) = ||∂u/∂t + H(x,∇u,m)||_{L²(K)} + ||∂m/∂t - div(m·D_p H)||_{L²(K)}
```

**Advantage**: Directly measures MFG system satisfaction

#### Strategy 4: Richardson Extrapolation
A posteriori error estimation using multiple resolution levels:
```
E_richardson(K) = |u_h(K) - u_{h/2}(K)| + |m_h(K) - m_{h/2}(K)|
```

---

## AMR Framework

### Core Data Structures

#### 1D Interval Structure

```python
@dataclass
class Interval1D:
    interval_id: int
    x_min: float
    x_max: float
    level: int = 0
    parent_id: Optional[int] = None
    children_ids: Optional[List[int]] = None

    @property
    def center(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    def subdivide(self) -> Tuple['Interval1D', 'Interval1D']:
        """Subdivide interval into two children."""
```

#### 2D QuadTree Node Structure

```python
@dataclass
class QuadTreeNode:
    level: int                    # Refinement level (0 = root)
    x_min, x_max: float          # Cell boundaries
    y_min, y_max: float
    parent: Optional['QuadTreeNode'] = None
    children: Optional[List['QuadTreeNode']] = None
    is_leaf: bool = True
    cell_id: int = None
    solution_data: Dict[str, NDArray] = None
    error_estimate: float = 0.0

    @property
    def center_x(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2.0

    @property
    def dx(self) -> float:
        return self.x_max - self.x_min

    @property
    def dy(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.dx * self.dy

    def contains_point(self, x: float, y: float) -> bool:
        """Test if point lies within cell."""
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max)

    def subdivide(self) -> List['QuadTreeNode']:
        """4-way cell subdivision."""
```

#### Refinement Criteria Configuration

```python
@dataclass
class AMRRefinementCriteria:
    error_threshold: float = 1e-4           # Main refinement threshold
    gradient_threshold: float = 0.1         # Gradient-based threshold
    max_refinement_levels: int = 5          # Maximum levels
    min_cell_size: float = 1e-6            # Minimum cell size
    coarsening_threshold: float = 0.1       # Coarsening factor
    solution_variance_threshold: float = 1e-5
    density_gradient_threshold: float = 0.05
    adaptive_error_scaling: bool = True
```

### AMR Algorithm Implementation

#### Mesh Adaptation Cycle

```python
def adapt_mesh(self, solution_data: Dict[str, NDArray]) -> Dict[str, int]:
    """Complete mesh adaptation cycle."""
    stats = {'total_refined': 0, 'total_coarsened': 0}

    for iteration in range(max_iterations):
        # Phase 1: Refinement
        refined = self.refine_mesh(solution_data)

        # Phase 2: Coarsening
        coarsened = self.coarsen_mesh(solution_data)

        stats['total_refined'] += refined
        stats['total_coarsened'] += coarsened

        # Mark solution transfer needed
        if refined > 0 or coarsened > 0:
            self.solution_transfer_needed = True

        # Convergence check
        if refined == 0 and coarsened == 0:
            break

    return stats
```

#### Error-Driven Refinement

```python
def refine_mesh(self, solution_data: Dict[str, NDArray]) -> int:
    """Refine cells based on error estimates."""
    cells_to_refine = []

    for node in self.leaf_nodes:
        # Check refinement constraints
        if (node.level >= self.criteria.max_refinement_levels or
            min(node.dx, node.dy) <= self.criteria.min_cell_size):
            continue

        # Compute error estimate
        error = self.error_estimator.estimate_error(node, solution_data)
        node.error_estimate = error

        # Refinement decision
        if error > self.criteria.error_threshold:
            cells_to_refine.append(node)

    # Execute refinement
    refined_count = 0
    for node in cells_to_refine:
        children = node.subdivide()
        self._update_leaf_tracking(node, children)
        refined_count += 1

    return refined_count
```

#### Conservative Coarsening

```python
def coarsen_mesh(self, solution_data: Dict[str, NDArray]) -> int:
    """Coarsen cells with low error estimates."""
    coarsening_threshold = (self.criteria.error_threshold *
                          self.criteria.coarsening_threshold)

    # Group siblings for coarsening
    parent_groups = self._group_siblings()

    coarsened_count = 0
    for parent, children in parent_groups.items():
        if len(children) != self.children_per_parent:  # 2 for 1D, 4 for 2D
            continue

        # Check coarsening criterion for all children
        all_low_error = all(
            self.error_estimator.estimate_error(child, solution_data)
            < coarsening_threshold
            for child in children
        )

        if all_low_error:
            self._coarsen_parent(parent, children)
            coarsened_count += 1

    return coarsened_count
```

---

## Error Estimation

### Gradient-Based Error Estimator

The primary error estimator uses finite difference approximations:

```python
class GradientErrorEstimator(BaseErrorEstimator):
    def estimate_error(self, node: QuadTreeNode,
                      solution_data: Dict[str, NDArray]) -> float:
        """Estimate cell error using solution gradients."""

        U, M = solution_data['U'], solution_data['M']
        i, j = self._get_cell_indices(node)

        # Finite difference gradients
        dU_dx = (U[i+1, j] - U[i-1, j]) / (2.0 * node.dx)
        dU_dy = (U[i, j+1] - U[i, j-1]) / (2.0 * node.dy)

        dM_dx = (M[i+1, j] - M[i-1, j]) / (2.0 * node.dx)
        dM_dy = (M[i, j+1] - M[i, j-1]) / (2.0 * node.dy)

        # Combined error indicator
        grad_U = np.sqrt(dU_dx**2 + dU_dy**2)
        grad_M = np.sqrt(dM_dx**2 + dM_dy**2)

        return max(grad_U, grad_M)
```

### JAX-Accelerated Error Computation

For high-performance applications:

```python
@jax.jit
def compute_error_indicators(U: jnp.ndarray, M: jnp.ndarray,
                           dx: float, dy: float) -> jnp.ndarray:
    """GPU-accelerated error indicator computation."""

    # Compute gradients using JAX
    dU_dx = jnp.gradient(U, dx, axis=0)
    dU_dy = jnp.gradient(U, dy, axis=1)
    dM_dx = jnp.gradient(M, dx, axis=0)
    dM_dy = jnp.gradient(M, dy, axis=1)

    # Gradient magnitudes
    grad_U = jnp.sqrt(dU_dx**2 + dU_dy**2)
    grad_M = jnp.sqrt(dM_dx**2 + dM_dy**2)

    # Combined error with curvature terms
    d2U_dx2 = jnp.gradient(dU_dx, dx, axis=0)
    d2U_dy2 = jnp.gradient(dU_dy, dy, axis=1)
    curvature_U = jnp.abs(d2U_dx2) + jnp.abs(d2U_dy2)

    d2M_dx2 = jnp.gradient(dM_dx, dx, axis=0)
    d2M_dy2 = jnp.gradient(dM_dy, dy, axis=1)
    curvature_M = jnp.abs(d2M_dx2) + jnp.abs(d2M_dy2)

    error_indicator = jnp.maximum(grad_U, grad_M)
    curvature_indicator = jnp.maximum(curvature_U, curvature_M)

    return error_indicator + 0.1 * curvature_indicator
```

---

## Conservative Interpolation

### Mathematical Foundation

#### Mass Conservation for Density

When transferring density m between mesh levels, we enforce:

**Global Mass Conservation:**
```
∫_Ω m^{new}(x) dx = ∫_Ω m^{old}(x) dx
```

**Local Conservation (for each parent cell):**
```
∫_{Ωᵢᵏ} m^{old}(x) dx = ∑_{j∈C(i)} ∫_{Ωⱼᵏ⁺¹} m^{new}(x) dx
```

#### Gradient Preservation for Value Function

For the value function u, we use **bilinear interpolation** that preserves gradient information:

```
u^{new}(x) = ∑ᵢ u^{old}(xᵢ) Lᵢ(x)
```

where Lᵢ(x) are Lagrange basis functions ensuring C¹ continuity.

#### Restriction and Prolongation Operators

**Restriction (Fine → Coarse):**
```
R: V^{k+1} → V^k
(Ru)ᵢᵏ = (1/|Ωᵢᵏ|) ∫_{Ωᵢᵏ} u(x) dx
```

**Prolongation (Coarse → Fine):**
```
P: V^k → V^{k+1}
(Pu)(x) = ∑ᵢ uᵢᵏ φᵢᵏ(x)  for x ∈ Ωⱼᵏ⁺¹
```

### Implementation

#### Mass-Conserving Density Transfer

```python
def _conservative_interpolation(self, U: NDArray, M: NDArray) -> Tuple[NDArray, NDArray]:
    """Conservative solution transfer between mesh levels."""

    # Create coordinate mapping
    nx, ny = U.shape
    x_coords = np.linspace(self.adaptive_mesh.x_min, self.adaptive_mesh.x_max, nx)
    y_coords = np.linspace(self.adaptive_mesh.y_min, self.adaptive_mesh.y_max, ny)

    U_new = np.zeros_like(U)
    M_new = np.zeros_like(M)

    # Interpolate each grid point
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            containing_node = self.adaptive_mesh._find_containing_node(x, y)

            if containing_node is not None:
                # Get region averages
                old_U_val, old_M_val = self._get_region_values(
                    U, M, containing_node, x_coords, y_coords
                )

                # Conservative interpolation
                M_new[i, j] = self._conservative_density_interpolation(
                    old_M_val, containing_node, x, y
                )

                # Gradient-preserving interpolation
                U_new[i, j] = self._gradient_preserving_interpolation(
                    old_U_val, containing_node, x, y
                )

    # Enforce global mass conservation
    M_new = self._enforce_mass_conservation(M, M_new)

    return U_new, M_new
```

#### Mass Conservation Enforcement

```python
def _enforce_mass_conservation(self, old_M: NDArray, new_M: NDArray) -> NDArray:
    """Enforce global mass conservation after interpolation."""

    old_mass = np.sum(old_M)
    new_mass = np.sum(new_M)

    if new_mass > 0:
        # Scale to conserve total mass
        conservation_factor = old_mass / new_mass
        new_M = new_M * conservation_factor
    else:
        # Fallback to original distribution
        new_M = old_M.copy()

    return new_M
```

#### JAX-Accelerated Conservative Interpolation

```python
@jax.jit
def conservative_mass_interpolation_2d(density: jnp.ndarray,
                                     old_dx: float, old_dy: float,
                                     new_dx: float, new_dy: float) -> jnp.ndarray:
    """Mass-conserving 2D interpolation with JAX."""

    # Compute total mass
    old_mass = jnp.sum(density) * old_dx * old_dy

    # Create coordinate mapping
    old_shape = density.shape
    new_shape = (int(old_shape[0] * old_dx / new_dx),
                int(old_shape[1] * old_dy / new_dy))

    # High-order interpolation using map_coordinates
    from jax.scipy.ndimage import map_coordinates

    x_new = jnp.linspace(0, old_shape[0] - 1, new_shape[0])
    y_new = jnp.linspace(0, old_shape[1] - 1, new_shape[1])
    X_new, Y_new = jnp.meshgrid(x_new, y_new, indexing='ij')
    coords = jnp.stack([X_new.ravel(), Y_new.ravel()])

    # Conservative interpolation
    density_interp = map_coordinates(density, coords, order=1, mode='nearest')
    density_interp = density_interp.reshape(new_shape)

    # Enforce mass conservation
    new_mass = jnp.sum(density_interp) * new_dx * new_dy
    conservation_factor = old_mass / (new_mass + 1e-12)

    return density_interp * conservation_factor
```

---

## System Architecture

### Package Structure

```python
# Adaptive Mesh Refinement Architecture
mfg_pde/
├── geometry/
│   ├── amr_mesh.py              # 2D quadtree AMR mesh
│   ├── one_dimensional_amr.py   # 1D interval-based AMR
│   └── triangular_amr.py        # Triangular mesh AMR (future)
├── alg/
│   └── amr_enhancement.py       # AMR enhancement wrapper
└── factory.py                    # AMR factory functions
```

### Core Components

#### AMR Enhancement Wrapper

The AMR system is implemented as an **enhancement wrapper** that can be applied to any base MFG solver:

```python
class AMREnhancedSolver:
    """AMR enhancement wrapper for existing MFG solvers."""

    def __init__(self, base_solver: MFGSolver, amr_mesh: Any,
                 error_estimator: BaseErrorEstimator,
                 amr_config: Optional[Dict[str, Any]] = None):
        self.base_solver = base_solver
        self.amr_mesh = amr_mesh
        self.error_estimator = error_estimator
        self.amr_config = amr_config or {}

        # AMR parameters
        self.adaptation_frequency = self.amr_config.get('adaptation_frequency', 5)
        self.max_adaptations = self.amr_config.get('max_adaptations', 3)

    def solve(self, max_iterations: int = 100, tolerance: float = 1e-6,
              verbose: Optional[bool] = None, **kwargs) -> Dict[str, Any]:
        """Solve MFG problem with adaptive mesh refinement enhancement."""
```

#### Dimensional AMR Classes

**1D AMR**: `OneDimensionalAMRMesh` - Interval-based hierarchical refinement
**2D Structured AMR**: `AdaptiveMesh` - Quadtree-based refinement
**2D Triangular AMR**: `TriangularAMRMesh` - Uses MeshData infrastructure (future)

### Integration Points

1. **Solver Integration**: AMR wraps any base solver (fixed_point, particle_collocation, etc.)
2. **Geometry Pipeline**: Extends existing mesh infrastructure
3. **Visualization**: Real-time mesh adaptation display
4. **Problem Definition**: AMR parameters in solver configuration

---

## Integration with MFG Solvers

### AMR Enhancement Architecture

**Design Philosophy**: AMR is a mesh adaptation technique, not a solution method. Therefore, it is implemented as an **enhancement wrapper** that can be applied to any base MFG solver.

### AMR-Solver Integration Loop

```python
def solve(self, max_iterations: int = 100, tolerance: float = 1e-6,
          verbose: bool = True) -> SolverResult:
    """Solve MFG with adaptive mesh refinement."""

    U, M = self._initialize_solution()
    convergence_history = []

    # Main AMR-solver loop
    for amr_cycle in range(self.max_amr_cycles):
        if verbose:
            pbar = tqdm(range(max_iterations),
                       desc=f"AMR Cycle {amr_cycle+1}/{self.max_amr_cycles}")

        # Solve on current mesh
        cycle_converged = False
        for iteration in pbar:
            # MFG solver step
            U_new, M_new, residual = self._solver_step(U, M)

            # Check convergence
            change = self._compute_solution_change(U, M, U_new, M_new)
            convergence_history.append(change)

            U, M = U_new, M_new

            # Periodic mesh adaptation
            if (iteration + 1) % self.amr_frequency == 0:
                self._adapt_mesh({'U': U, 'M': M}, verbose=verbose)

            # Convergence check
            if change < tolerance:
                cycle_converged = True
                break

        # Final mesh adaptation for this cycle
        adaptation_stats = self._adapt_mesh({'U': U, 'M': M}, verbose=verbose)

        # Check mesh convergence
        if (adaptation_stats['total_refined'] == 0 and
            adaptation_stats['total_coarsened'] == 0):
            if verbose:
                print("Mesh adaptation converged")
            break

        # Project solution to new mesh
        if adaptation_stats['total_refined'] > 0 or adaptation_stats['total_coarsened'] > 0:
            U, M = self._project_solution_to_new_mesh(U, M)

    return self._create_result(U, M, convergence_history, tolerance)
```

### Temporal Adaptation Strategy

**Challenge**: MFG solutions evolve dynamically, requiring mesh adaptation in time.

**Algorithm: Adaptive Time-Space Refinement**
```
1. For each time step t^n → t^{n+1}:
   a. Predict solution using current mesh
   b. Estimate errors using chosen strategy
   c. Refine/coarsen mesh based on error thresholds
   d. Project solution to new mesh conservatively
   e. Solve MFG system on adapted mesh
   f. Update solution and advance time
```

**Refinement Threshold Evolution:**
```
Refinement threshold: α(t) = α₀ · (1 + β·||m(·,t) - m₀||_{L¹})
```

**Rationale**: As solution evolves from initial condition, allow higher resolution where needed.

---

## Performance Optimization

### JAX-Accelerated Operations

The AMR system includes comprehensive JAX acceleration:

```python
class JAXAcceleratedAMR:
    """JAX-accelerated AMR operations for high performance."""

    @staticmethod
    @jax.jit
    def compute_error_indicators(U: jnp.ndarray, M: jnp.ndarray,
                               dx: float, dy: float) -> jnp.ndarray:
        """Compute error indicators for all cells using JAX."""
        # [Implementation shown in Error Estimation section]

    @staticmethod
    @jax.jit
    def gradient_preserving_interpolation_2d(values: jnp.ndarray,
                                           old_dx: float, old_dy: float,
                                           new_dx: float, new_dy: float) -> jnp.ndarray:
        """Gradient-preserving interpolation with cubic splines."""
        # [Implementation shown in Conservative Interpolation section]
```

### Memory Optimization Strategies

1. **Lazy Evaluation**: Error indicators computed only when needed
2. **Efficient Storage**: Quadtree nodes store only essential data
3. **Batch Processing**: Multiple cells processed together for efficiency
4. **Memory Pools**: Reuse allocated arrays to reduce garbage collection

**Hierarchical Storage:**
- Coarse cells: Store full solution data
- Fine cells: Store refinement corrections
- Ghost cells: Communication between levels

### Computational Complexity

**Uniform Grid Complexity:**
- Memory: `O(N^d)` where `N` is grid points per dimension
- Time per step: `O(N^d)` for explicit schemes, `O(N^d log N)` for implicit

**AMR Grid Complexity:**
- Memory: `O(N_active)` where `N_active << N^d` is active cells
- Time per step: `O(N_active + adaptation_cost)`
- Adaptation cost: `O(N_active log N_active)` for tree operations

**Expected Speedup:**
For problems with localized features covering fraction `f` of domain:
```
Speedup ≈ 1/(f + (1-f)/r^d)
```
where `r` is refinement ratio. For `f = 0.1` and `r = 2` in 2D:
```
Speedup ≈ 1/(0.1 + 0.9/4) = 1/0.325 ≈ 3.1×
```

### Parallel Processing Considerations

- **Thread Safety**: All AMR operations are thread-safe
- **GPU Compatibility**: JAX operations automatically utilize available GPUs
- **Vectorization**: Batch operations on multiple cells simultaneously

---

## Usage Guide

### Basic AMR Usage

```python
from mfg_pde import MFGProblem
from mfg_pde.factory import create_amr_solver
from mfg_pde.geometry import Domain1D, periodic_bc

### 1D AMR Example ###
# Create 1D problem
domain_1d = Domain1D(0.0, 2.0, periodic_bc())
problem_1d = MFGProblem(T=1.0, xmin=0.0, xmax=2.0, Nx=50, Nt=20)
problem_1d.domain = domain_1d
problem_1d.dimension = 1

# Create AMR-enhanced solver (any base solver can be enhanced)
amr_solver = create_amr_solver(
    problem_1d,
    base_solver_type="fixed_point",  # Base solver to enhance
    error_threshold=1e-4,            # Refine when error > threshold
    max_levels=5,                    # Maximum refinement levels
    initial_intervals=20,            # Initial 1D intervals
    adaptation_frequency=5,          # Adapt every 5 iterations
    max_adaptations=3                # Maximum AMR cycles
)

# Solve with AMR enhancement
result = amr_solver.solve(max_iterations=50, tolerance=1e-6, verbose=True)

# Access AMR statistics
print(f"AMR enabled: {result['amr_enabled']}")
print(f"Base solver: {result['base_solver_type']}")
print(f"Total adaptations: {result['total_adaptations']}")
print(f"Final mesh size: {len(result.get('grid_points', []))}")

### 2D AMR Example ###
# Create 2D problem
problem_2d = MFGProblem(T=1.0, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
                               Nx=32, Ny=32, Nt=20)
problem_2d.dimension = 2

# Create AMR-enhanced 2D solver
amr_solver_2d = create_amr_solver(
    problem_2d,
    base_solver_type="particle_collocation",  # Different base solver
    error_threshold=1e-4,
    max_levels=4
)

result_2d = amr_solver_2d.solve(max_iterations=50, verbose=True)
```

### Using Factory with AMR Enhancement Flag

```python
from mfg_pde.factory import create_solver

# Any solver can be enhanced with AMR
amr_enhanced_solver = create_solver(
    problem,
    solver_type="fixed_point",      # Base solver type
    enable_amr=True,                # Enable AMR enhancement
    amr_config={
        'error_threshold': 1e-4,
        'max_levels': 5,
        'adaptation_frequency': 5,
        'max_adaptations': 3
    }
)

result = amr_enhanced_solver.solve()
```

### Advanced AMR Enhancement Configuration

```python
from mfg_pde.alg.amr_enhancement import create_amr_enhanced_solver
from mfg_pde.factory import create_solver

# Step 1: Create any base solver
base_solver = create_solver(
    problem,
    solver_type="monitored_particle",  # Any solver type
    preset="accurate"
)

# Step 2: Enhance with AMR
amr_config = {
    'error_threshold': 1e-5,
    'max_levels': 6,
    'adaptation_frequency': 3,
    'max_adaptations': 4,
    'verbose': True
}

amr_enhanced_solver = create_amr_enhanced_solver(
    base_solver=base_solver,
    dimension=2,  # or auto-detected
    amr_config=amr_config
)

result = amr_enhanced_solver.solve()
```

### Dimensional Consistency

The AMR enhancement works consistently across all problem dimensions:

```python
# 1D AMR - interval-based refinement
from mfg_pde.geometry import Domain1D, periodic_bc

domain_1d = Domain1D(0.0, 1.0, periodic_bc())
problem_1d = MFGProblem(T=1.0, xmin=0.0, xmax=1.0, Nx=50, Nt=20)
problem_1d.domain = domain_1d
problem_1d.dimension = 1

amr_1d = create_amr_solver(problem_1d, base_solver_type="fixed_point")

# 2D Structured AMR - quadtree-based refinement
problem_2d = MFGProblem(T=1.0, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0,
                               Nx=32, Ny=32, Nt=20)
problem_2d.dimension = 2

amr_2d = create_amr_solver(problem_2d, base_solver_type="particle_collocation")
```

### Performance Tuning

**For Speed:**
```python
amr_solver = create_amr_solver(
    problem,
    base_solver_type="fixed_point",
    error_threshold=1e-3,    # Less aggressive refinement
    max_levels=3,            # Fewer levels
    adaptation_frequency=10, # Less frequent adaptation
)
```

**For Accuracy:**
```python
amr_solver = create_amr_solver(
    problem,
    base_solver_type="monitored_particle",
    error_threshold=1e-6,    # More aggressive refinement
    max_levels=7,            # More levels allowed
    adaptation_frequency=3,  # More frequent adaptation
)
```

### Monitoring AMR Enhancement Performance

```python
result = amr_solver.solve(verbose=True)

# AMR Enhancement provides comprehensive statistics
print("AMR Enhancement Results:")
print(f"  AMR enabled: {result['amr_enabled']}")
print(f"  Base solver type: {result['base_solver_type']}")
print(f"  Total adaptations: {result['total_adaptations']}")
print(f"  Mesh generations: {result['mesh_generations']}")

# Mesh statistics (varies by dimension)
mesh_stats = result['mesh_statistics']
print("Mesh Statistics:")
print(f"  Elements/Intervals: {mesh_stats.get('total_intervals', mesh_stats.get('total_cells', 'N/A'))}")
print(f"  Max refinement level: {mesh_stats.get('max_level', 0)}")
print(f"  Level distribution: {mesh_stats.get('level_distribution', {})}")

# Adaptation history
print("Adaptation History:")
for i, adaptation in enumerate(result['adaptation_history']):
    print(f"  Cycle {i+1}: {adaptation.get('total_refined', 0)} elements refined")

# Convergence information
print("Convergence:")
print(f"  Solution converged: {result.get('converged', False)}")
print(f"  AMR converged: {result.get('amr_converged', False)}")
print(f"  Total iterations: {result.get('iterations', 'N/A')}")
```

---

## API Reference

### AMR Enhancement Architecture

#### `AMREnhancedSolver`
```python
class AMREnhancedSolver:
    """AMR enhancement wrapper for existing MFG solvers."""

    def __init__(self, base_solver: MFGSolver, amr_mesh: Any,
                 error_estimator: BaseErrorEstimator,
                 amr_config: Optional[Dict[str, Any]] = None): ...

    def solve(self, max_iterations: int = 100, tolerance: float = 1e-6,
              verbose: Optional[bool] = None, **kwargs) -> Dict[str, Any]:
        """Solve MFG problem with adaptive mesh refinement enhancement."""
```

#### `create_amr_enhanced_solver`
```python
def create_amr_enhanced_solver(
    base_solver: MFGSolver,
    dimension: int = None,
    amr_config: Optional[Dict[str, Any]] = None
) -> AMREnhancedSolver:
    """Create AMR-enhanced version of any MFG solver."""
```

### Dimensional AMR Classes

#### 1D AMR: `OneDimensionalAMRMesh`
```python
class OneDimensionalAMRMesh:
    """1D adaptive mesh using interval-based hierarchical refinement."""

    def __init__(self, domain_1d: Domain1D, initial_num_intervals: int = 10,
                 refinement_criteria: Optional[AMRRefinementCriteria] = None): ...

    def adapt_mesh_1d(self, solution_data: Dict[str, np.ndarray],
                      error_estimator: BaseErrorEstimator) -> Dict[str, int]: ...

    def get_grid_points(self) -> Tuple[np.ndarray, np.ndarray]: ...
    def export_to_mesh_data(self) -> MeshData: ...
```

#### 1D Interval: `Interval1D`
```python
@dataclass
class Interval1D:
    interval_id: int
    x_min: float
    x_max: float
    level: int = 0
    parent_id: Optional[int] = None
    children_ids: Optional[List[int]] = None

    @property
    def center(self) -> float: ...
    @property
    def width(self) -> float: ...
    def subdivide(self) -> Tuple['Interval1D', 'Interval1D']: ...
```

#### 2D Structured AMR: `AdaptiveMesh`
```python
class AdaptiveMesh:
    """2D adaptive mesh using quadtree hierarchical refinement."""

    def __init__(self, domain_bounds: Tuple[float, float, float, float],
                 initial_resolution: Tuple[int, int] = (32, 32),
                 refinement_criteria: Optional[AMRRefinementCriteria] = None): ...

    def adapt_mesh(self, solution_data: Dict[str, NDArray],
                   error_estimator: BaseErrorEstimator,
                   max_iterations: int = 5) -> Dict[str, int]: ...
    def get_mesh_statistics(self) -> Dict[str, Any]: ...
    def interpolate_solution(self, coarse_solution: Dict[str, NDArray],
                           target_grid: Tuple[int, int]) -> Dict[str, NDArray]: ...
```

#### 2D QuadTree Node: `QuadTreeNode`
```python
@dataclass
class QuadTreeNode:
    level: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    parent: Optional['QuadTreeNode'] = None
    children: Optional[List['QuadTreeNode']] = None
    is_leaf: bool = True

    @property
    def center_x(self) -> float: ...
    @property
    def center_y(self) -> float: ...
    @property
    def dx(self) -> float: ...
    @property
    def dy(self) -> float: ...
    @property
    def area(self) -> float: ...

    def contains_point(self, x: float, y: float) -> bool: ...
    def subdivide(self) -> List['QuadTreeNode']: ...
```

### Refinement Criteria

#### `AMRRefinementCriteria`
```python
@dataclass
class AMRRefinementCriteria:
    error_threshold: float = 1e-4
    gradient_threshold: float = 0.1
    max_refinement_levels: int = 5
    min_cell_size: float = 1e-6
    coarsening_threshold: float = 0.1
    solution_variance_threshold: float = 1e-5
    density_gradient_threshold: float = 0.05
    adaptive_error_scaling: bool = True
```

### Error Estimators

#### `BaseErrorEstimator`
```python
class BaseErrorEstimator(ABC):
    """Abstract base for AMR error estimation."""

    @abstractmethod
    def estimate_error(self, node: Union[QuadTreeNode, Interval1D],
                      solution_data: Dict[str, NDArray]) -> float:
        """Compute error indicator for cell."""
```

#### `GradientErrorEstimator`
```python
class GradientErrorEstimator(BaseErrorEstimator):
    """Gradient-based error estimation."""

    def estimate_error(self, node, solution_data) -> float:
        """Estimate error using solution gradients."""
```

### Factory Functions

#### `create_amr_solver`
```python
def create_amr_solver(
    problem: MFGProblem,
    base_solver_type: str = "fixed_point",
    error_threshold: float = 1e-4,
    max_levels: int = 5,
    **kwargs
) -> AMREnhancedSolver:
    """Create an AMR-enhanced MFG solver."""
```

#### `create_amr_mesh`
```python
def create_amr_mesh(
    domain_bounds: Tuple[float, ...],
    dimension: int,
    error_threshold: float = 1e-4,
    max_levels: int = 5,
    **kwargs
) -> Union[OneDimensionalAMRMesh, AdaptiveMesh]:
    """Create an adaptive mesh with specified parameters."""
```

### JAX Acceleration Functions

#### `JAXAcceleratedAMR`
```python
class JAXAcceleratedAMR:
    @staticmethod
    @jax.jit
    def compute_error_indicators(U: jnp.ndarray, M: jnp.ndarray,
                               dx: float, dy: float) -> jnp.ndarray: ...

    @staticmethod
    @jax.jit
    def conservative_mass_interpolation_2d(density: jnp.ndarray,
                                         old_dx: float, old_dy: float,
                                         new_dx: float, new_dy: float) -> jnp.ndarray: ...

    @staticmethod
    @jax.jit
    def gradient_preserving_interpolation_2d(values: jnp.ndarray,
                                           old_dx: float, old_dy: float,
                                           new_dx: float, new_dy: float) -> jnp.ndarray: ...
```

### Configuration Examples

#### Basic Configuration
```python
amr_solver = create_amr_solver(
    problem,
    error_threshold=1e-4,
    max_levels=5,
    base_solver_type="fixed_point"
)
```

#### High-Accuracy Configuration
```python
amr_solver = create_amr_solver(
    problem,
    error_threshold=1e-6,
    max_levels=7,
    base_solver_type="particle_collocation",
    adaptation_frequency=3,
    max_adaptations=5
)
```

#### Custom Domain Configuration
```python
amr_solver = create_amr_solver(
    problem,
    error_threshold=1e-5,
    max_levels=6,
    base_solver_type="fixed_point",
    initial_resolution=(64, 64)  # For 2D
)
```

---

## Mathematical References

1. **Adaptive Mesh Refinement Theory**: Berger, M. J., & Oliger, J. (1984). Adaptive mesh refinement for hyperbolic partial differential equations.
2. **Conservative Interpolation**: LeVeque, R. J. (2002). Finite Volume Methods for Hyperbolic Problems.
3. **Error Estimation for PDEs**: Verfürth, R. (2013). A posteriori error estimation techniques for finite element methods.
4. **Mean Field Games**: Carmona, R., & Delarue, F. (2018). Probabilistic Theory of Mean Field Games with Applications.
5. **AMR in Practice**: MacNeice, P. et al. (2000). PARAMESH: A parallel adaptive mesh refinement community toolkit.
6. **HJB Numerics**: Achdou, Y. & Capuzzo-Dolcetta, I. (2010). Mean field games: numerical methods.
7. **AMR for PDEs**: Almgren, A.S. et al. (2010). CASTRO: A new compressible astrophysical solver.

## Performance Benchmarks

### Expected Performance Gains

- **Accuracy Improvement**: 2-5× better solution accuracy for same computational cost
- **Memory Efficiency**: 30-70% reduction in memory usage for problems with localized features
- **Computational Speed**: 1.5-3× faster convergence due to focused refinement
- **GPU Acceleration**: 5-15× speedup with JAX backend on appropriate hardware

### Recommended Use Cases

- **High-Gradient Solutions**: Problems with sharp boundary layers or discontinuities
- **Multi-Scale Problems**: Solutions with features at different scales
- **Complex Geometries**: Irregular domains requiring adaptive resolution
- **Research Applications**: High-accuracy solutions for theoretical analysis
- **Large Domains**: Problems where uniform fine grids are computationally prohibitive

---

**Last Updated**: August 1, 2025
**Module Version**: MFG_PDE 2.0+
**Maintainer**: MFG_PDE Development Team
