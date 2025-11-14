# Capacity-Constrained Mean Field Games: Example Pattern

**Author**: MFG_PDE Development Team
**Date**: 2025-11-12
**Status**: ✅ IMPLEMENTED AS EXAMPLE
**Location**: `examples/advanced/capacity_constrained_mfg/`
**Related Issues**: Capacity-constrained MFG, maze navigation, congestion modeling

**Architecture Note**: This implementation demonstrates how to extend the MFG_PDE framework
with application-specific features. It resides in `examples/` rather than `mfg_pde/core/`
to keep the core framework minimal. Users can adapt this pattern for their own
capacity-constrained or congestion-aware MFG applications.

## Table of Contents

1. [Mathematical Framework](#mathematical-framework)
2. [Architectural Design](#architectural-design)
3. [Implementation Details](#implementation-details)
4. [Usage Guide](#usage-guide)
5. [Computational Considerations](#computational-considerations)
6. [References](#references)

---

## Mathematical Framework

### Standard MFG System

The classical Mean Field Game system consists of coupled PDEs:

**HJB Equation** (backward in time):
```
-∂u/∂t + H(x, m, ∇u, t) = 0    in Ω × [0,T]
u(T, x) = g(x)                   terminal condition
```

**Fokker-Planck Equation** (forward in time):
```
∂m/∂t - σΔm + div(m·∇_p H(x, m, ∇u, t)) = 0    in Ω × [0,T]
m(0, x) = m₀(x)                                  initial condition
```

**Standard Hamiltonian**:
```
H(x, m, p, t) = (1/2)|p|² + α·m
```

where:
- `u(t,x)`: Value function (cost-to-go)
- `m(t,x)`: Agent density
- `p = ∇u`: Momentum (co-state)
- `σ`: Diffusion coefficient
- `α`: Coupling coefficient (density interaction)

### Capacity-Constrained Extension

For maze navigation and crowd dynamics, we extend the Hamiltonian with a **congestion term**:

```
H(x, m, ∇u, t) = (1/2)|∇u|² + α·m + γ·g(m(x)/C(x))
```

**New Components**:

1. **Capacity Field** `C(x)`: Spatially-varying corridor capacity
   - `C(x) > 0` everywhere in domain Ω
   - Large in wide corridors, small near walls
   - Computed from maze geometry via distance transform

2. **Congestion Cost** `g(ρ)`: Convex penalty function
   - Input: congestion ratio `ρ = m/C` (dimensionless)
   - Properties: `g(0) = 0`, `g'(ρ) > 0`, `g''(ρ) ≥ 0` (convex)
   - Models crowding effects

3. **Congestion Weight** `γ ≥ 0`: Penalty strength
   - `γ = 0`: No congestion (free flow)
   - `γ → ∞`: Hard capacity constraint

**Physical Interpretation**:

- When `m(x) < C(x)`: Free flow, low congestion cost
- When `m(x) ≈ C(x)`: Near capacity, cost increases rapidly
- When `m(x) > C(x)`: Overcapacity, high penalty (soft barrier)

The term `γ·g(m/C)` creates a **"soft wall" effect** that naturally encourages agents to:
- Avoid overcrowded regions
- Seek alternative routes with available capacity
- Distribute optimally across corridors

### Modified MFG System

**HJB Equation** (with congestion):
```
-∂u/∂t + (1/2)|∇u|² + α·m + γ·g(m/C) = 0
```

**FP Equation** (with congestion coupling):
```
∂m/∂t - σΔm + div(m·∇u) + div(m·∇_m H) = 0
```

where the density gradient of the Hamiltonian is:
```
∂H/∂m = α + γ·g'(m/C) / C
```

This couples the density back into the FP equation, creating a **nonlinear drift term** that depends on local congestion.

---

## Architectural Design

### Framework Philosophy: Minimal Core

**Why Examples, Not Core?**

The capacity-constrained MFG infrastructure resides in `examples/advanced/` rather than
`mfg_pde/core/` to maintain a **minimal framework design**:

**Rationale**:
1. **Framework minimalism**: Core provides base classes (`MFGProblem`, protocols, solvers)
2. **Application-specific**: Capacity constraints are problem context, not universal infrastructure
3. **User extensibility**: Examples serve as templates users can adapt and customize
4. **Reduced coupling**: Core remains focused on essential MFG machinery

**User Decision**: "Core is rather minimal framework. Even congestion is a very important
application, but apart from our example, users can also build their own based on core."

### Design Pattern: Composition

The implementation follows **composition over inheritance**:

```
CapacityConstrainedMFGProblem (Example)
    ├─ Extends: MFGProblem (Core framework)
    ├─ Composes: CapacityField (Application-specific)
    └─ Composes: CongestionModel (Application-specific)
```

**Why Not Extend Geometry?**

Alternative approach (rejected):
```
CapacityAwareGeometry extends BaseGeometry
    └─ Adds: capacity_field attribute
```

**Reasons for separate classes**:

1. **Separation of concerns**: Capacity is **problem context**, not geometry structure
2. **Reusability**: CapacityField can work with any geometry (SimpleGrid, Mesh, PointCloud)
3. **Orthogonality**: Not all geometries need capacity constraints
4. **Protocol compliance**: Doesn't pollute GeometryProtocol with maze-specific features
5. **Flexibility**: Users can swap congestion models without changing core framework

### Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  CapacityConstrainedMFGProblem              │
│                                                             │
│  hamiltonian(x, m, p, t):                                   │
│    H_base = 0.5*|p|² + α·m                                  │
│    C_x = capacity_field.interpolate_at_positions(x)        │
│    congestion = congestion_model.cost(m, C_x)              │
│    return H_base + γ·congestion                            │
│                                                             │
│  hamiltonian_dm(x, m, p, t):                               │
│    ∂H/∂m = α + γ·congestion_model.derivative(m, C_x)       │
└─────────────────────────────────────────────────────────────┘
          │                                   │
          │                                   │
          ▼                                   ▼
┌──────────────────────┐          ┌────────────────────────┐
│   CapacityField      │          │   CongestionModel      │
│                      │          │                        │
│  capacity: NDArray   │          │  cost(m, C) → g(m/C)   │
│  epsilon: float      │          │  derivative(m, C)      │
│  bounds: tuple       │          │  → ∂g/∂m               │
│                      │          └────────────────────────┘
│  from_maze_geometry()│                    │
│  interpolate_at_pos()│                    │
│  particle_in_cell()  │          ┌─────────┴─────────────┐
└──────────────────────┘          │                       │
                                  ▼                       ▼
                        QuadraticCongestion    LogBarrierCongestion
                        ExponentialCongestion  PiecewiseCongestion
```

---

## Implementation Details

### 1. CapacityField

**File**: `examples/advanced/capacity_constrained_mfg/capacity_field.py` (479 lines)

**Core Functionality**:

```python
class CapacityField:
    """Spatially-varying capacity field for maze navigation."""

    def __init__(self, capacity, epsilon=1e-3, cell_size=1.0, bounds=None):
        # Regularization: C_eff(x) = max(C(x), epsilon)
        self.capacity = np.clip(capacity, epsilon, None)
        self.epsilon = epsilon

    @classmethod
    def from_maze_geometry(cls, maze_array, wall_thickness=1.0,
                           epsilon=1e-3, normalization="max"):
        """Compute capacity from distance transform."""
        # 1. Detect passages (0=wall or 1=passage)
        passages = (maze_array > 0) if np.mean(maze_array) > 0.5 else (maze_array == 0)

        # 2. Euclidean Distance Transform
        distance = distance_transform_edt(passages)

        # 3. Normalize capacity
        if normalization == "max":
            capacity = distance / distance.max()

        # 4. Regularize: prevent singularities
        capacity_regularized = np.clip(capacity, epsilon, None)
        capacity_regularized[~passages] = epsilon  # Safety

        return cls(capacity=capacity_regularized, epsilon=epsilon)
```

**Key Methods**:

1. **`interpolate_at_positions(positions, method="linear")`**
   - For particle solvers: get C(x_i) at arbitrary positions
   - Uses `scipy.ndimage.map_coordinates`
   - Supports nearest, linear, cubic interpolation

2. **`particle_in_cell_projection(particle_positions, particle_masses, grid_shape)`**
   - O(N) alternative to O(N²) KDE for particle-grid coupling
   - Nearest-grid-point (NGP) binning
   - Preserves mass conservation

**Regularization Strategy**:

Problem: Near walls, `C(x) → 0`, causing division by zero in `g(m/C)`.

Solution: Enforce `C_eff(x) = max(C(x), ε)` with `ε ~ 10⁻³`.

This creates a "minimum capacity" everywhere, ensuring numerical stability while preserving the soft barrier effect.

### 2. CongestionModel Hierarchy

**File**: `examples/advanced/capacity_constrained_mfg/congestion.py` (557 lines)

**Abstract Base Class**:

```python
class CongestionModel(ABC):
    """Abstract base for congestion cost functions g(ρ)."""

    @abstractmethod
    def cost(self, density: NDArray, capacity: NDArray) -> NDArray:
        """Compute g(m/C) for each grid point."""
        pass

    @abstractmethod
    def derivative(self, density: NDArray, capacity: NDArray) -> NDArray:
        """Compute ∂g/∂m = g'(m/C) / C for HJB-FP coupling."""
        pass
```

**Concrete Implementations**:

1. **QuadraticCongestion**: `g(ρ) = ρ²`
   - Standard Hughes pedestrian flow model
   - Smooth, moderate penalty growth
   - Derivative: `∂g/∂m = 2m/C²`

2. **ExponentialCongestion**: `g(ρ) = (exp(βρ) - 1) / β`
   - Sharp penalty near capacity
   - Parameter β controls steepness
   - Derivative: `∂g/∂m = exp(βm/C) / C`

3. **LogBarrierCongestion**: `g(ρ) = -log(1 - ρ)` for `ρ < threshold`
   - Barrier function approach (interior point method)
   - Prevents overcapacity mathematically
   - **Critical feature**: Piecewise linear extension for `ρ ≥ threshold`

4. **PiecewiseCongestion**: Different regimes
   - Free flow: `g(ρ) = k₁·ρ` for `ρ < ρ₀`
   - Congested: `g(ρ) = k₂·(ρ - ρ₀)²` for `ρ ≥ ρ₀`
   - Models traffic-like phase transitions

**LogBarrier Stability** (Critical Design):

```python
class LogBarrierCongestion(CongestionModel):
    def __init__(self, threshold=0.95, penalty_slope=10.0):
        self.threshold = threshold
        self.penalty_slope = penalty_slope
        self._barrier_at_threshold = -np.log(1.0 - threshold)

    def cost(self, density, capacity):
        ratio = density / capacity
        cost = np.zeros_like(ratio)

        # Safe region: ρ < threshold
        safe_mask = ratio < self.threshold
        cost[safe_mask] = -np.log(1.0 - ratio[safe_mask])

        # Extension region: ρ ≥ threshold (prevents NaN!)
        unsafe_mask = ~safe_mask
        excess = ratio[unsafe_mask] - self.threshold
        cost[unsafe_mask] = self._barrier_at_threshold + self.penalty_slope * excess

        return cost
```

**Why Extension is Needed**:

- During numerical iterations, transient states may have `m > C`
- Naive `-log(1 - m/C)` → NaN when `m/C ≥ 1`
- Piecewise extension provides bounded penalty in overcapacity regime
- Gradient remains continuous at threshold

### 3. CapacityConstrainedMFGProblem

**File**: `examples/advanced/capacity_constrained_mfg/problem.py` (426 lines)

**Class Definition**:

```python
class CapacityConstrainedMFGProblem(MFGProblem):
    """MFG problem with capacity constraints."""

    def __init__(self, capacity_field, congestion_model,
                 congestion_weight=1.0, **kwargs):
        # Initialize base problem
        super().__init__(**kwargs)

        # Store capacity components
        self.capacity_field = capacity_field
        self.congestion_model = congestion_model
        self.congestion_weight = congestion_weight

    def hamiltonian(self, x, m, p, t):
        """Compute H = (1/2)|p|² + α·m + γ·g(m/C)."""
        # Base Hamiltonian
        p_array = np.atleast_1d(p)
        H_base = 0.5 * np.sum(p_array**2)
        if hasattr(self, "coupling_coefficient"):
            H_base += self.coupling_coefficient * m

        # Get capacity at position x
        x_array = np.atleast_1d(x).reshape(1, -1)
        C_x = self.capacity_field.interpolate_at_positions(x_array)[0]

        # Compute congestion cost
        congestion = self.congestion_model.cost(
            density=np.array([m]),
            capacity=np.array([C_x])
        )[0]

        return H_base + self.congestion_weight * congestion

    def hamiltonian_dm(self, x, m, p, t):
        """Compute ∂H/∂m = α + γ·(∂g/∂m)."""
        # Base derivative
        H_dm_base = self.coupling_coefficient if hasattr(self, "coupling_coefficient") else 0.0

        # Get capacity
        x_array = np.atleast_1d(x).reshape(1, -1)
        C_x = self.capacity_field.interpolate_at_positions(x_array)[0]

        # Congestion derivative
        congestion_deriv = self.congestion_model.derivative(
            density=np.array([m]),
            capacity=np.array([C_x])
        )[0]

        return H_dm_base + self.congestion_weight * congestion_deriv
```

**Helper Methods**:

```python
def get_capacity_at_grid(self, grid_positions):
    """Get C(x) at grid points for visualization."""
    return self.capacity_field.interpolate_at_positions(grid_positions)

def get_congestion_ratio(self, density, positions):
    """Compute ρ(x) = m(x)/C(x) for analysis."""
    capacity = self.get_capacity_at_grid(positions)
    return density / capacity
```

---

## Usage Guide

### Basic Usage

```python
from mfg_pde.geometry.graph import PerfectMazeGenerator, MazeConfig
from examples.advanced.capacity_constrained_mfg import (
    CapacityField,
    CapacityConstrainedMFGProblem,
    QuadraticCongestion,
)

# 1. Generate maze
maze = create_perfect_maze(rows=20, cols=20, wall_thickness=3)
maze_array = maze.to_numpy_array(wall_thickness=3)

# 2. Compute capacity field
capacity = CapacityField.from_maze_geometry(
    maze_array,
    wall_thickness=3,
    epsilon=1e-3,
    normalization="max"
)

# 3. Create capacity-constrained problem
problem = CapacityConstrainedMFGProblem(
    capacity_field=capacity,
    congestion_model=QuadraticCongestion(),
    congestion_weight=1.0,
    spatial_bounds=[(0, 1), (0, 1)],
    spatial_discretization=[63, 63],
    T=1.0,
    Nt=50,
    sigma=0.01,
)

# 4. Solve with any MFG solver
from mfg_pde.solvers import solve_mfg
result = solve_mfg(problem, method="semi_lagrangian")
```

### Advanced: Custom Congestion Model

```python
from examples.advanced.capacity_constrained_mfg import LogBarrierCongestion

# Strong barrier near capacity
congestion_model = LogBarrierCongestion(
    threshold=0.95,      # Switch to linear extension at 95% capacity
    penalty_slope=20.0   # Strong penalty in overcapacity regime
)

problem = CapacityConstrainedMFGProblem(
    capacity_field=capacity,
    congestion_model=congestion_model,
    congestion_weight=5.0,  # High weight → strong avoidance
    # ... other parameters
)
```

### Visualization

```python
from examples.advanced.capacity_constrained_mfg import visualize_capacity_field

# Visualize capacity with maze overlay
visualize_capacity_field(capacity, maze_array, figsize=(12, 5))

# Visualize congestion ratio after solving
import numpy as np
positions = np.array([[x, y] for x in range(63) for y in range(63)])
congestion_ratio = problem.get_congestion_ratio(
    density=result.m_final.ravel(),
    positions=positions
)
# Plot with plt.imshow(congestion_ratio.reshape(63, 63))
```

---

## Computational Considerations

### Efficiency: Particle-in-Cell (PIC) vs KDE

For particle-based FP solvers, we need to project particle density onto grid for capacity evaluation.

**Kernel Density Estimation (KDE)**:
- Complexity: O(N_particles × N_grid) ≈ O(N²)
- Smooth, accurate density estimate
- **Too expensive** for large-scale problems

**Particle-in-Cell (PIC)**:
- Complexity: O(N_particles) ≈ O(N)
- Nearest-grid-point binning
- Mass-conserving
- **Chosen approach** for efficiency

Implementation in `CapacityField.particle_in_cell_projection()`.

### Regularization Parameter Selection

The parameter `ε` (minimum capacity) balances:

1. **Numerical stability**: `ε` prevents division by zero
2. **Physical accuracy**: Small `ε` preserves barrier effect

**Recommended values**:
- Standard grids: `ε = 10⁻³` to `10⁻²`
- Fine grids (high resolution): `ε = 10⁻⁴`
- Coarse grids: `ε = 10⁻²`

**Rule of thumb**: `ε ≈ 0.01 × mean(C)`

### Solver Compatibility

The capacity-constrained framework is compatible with:

✅ **Grid-based solvers**:
- Semi-Lagrangian schemes
- Finite difference methods
- Direct capacity lookup at grid points

✅ **Particle-based solvers**:
- Lagrangian methods
- GFDM (particle collocation + FDM)
- Uses `interpolate_at_positions()` and `particle_in_cell_projection()`

✅ **Hybrid solvers** (Issue #257):
- Different geometries for HJB and FP
- Uses `GeometryProjector` for coupling
- Capacity field interpolated to both grids

⚠️ **Neural network solvers**:
- PINN, DGM: Require custom implementation
- Capacity field can be embedded as input feature
- Congestion term added to loss function

---

## References

### Theoretical Foundations

1. **Hughes, R. L. (2002)**. "A continuum theory for the flow of pedestrians."
   *Transportation Research Part B*, 36(6), 507-535.
   - Original pedestrian flow model with capacity constraints

2. **Achdou, Y., Camilli, F., & Capuzzo-Dolcetta, I. (2020)**. "Mean field games with congestion."
   *Annales de l'Institut Henri Poincaré C, Analyse non linéaire*, 37(3), 637-663.
   - Mathematical analysis of capacity-constrained MFG

3. **Di Francesco, M., & Fagioli, S. (2013)**. "Measure solutions for non-local interaction PDEs with two distinct species."
   *Nonlinearity*, 26(10), 2777-2808.
   - Two-species MFG with cross-interaction (theoretical foundation)

4. **Degond, P., Appert-Rolland, C., Moussaïd, M., Pettré, J., & Theraulaz, G. (2013)**. "A hierarchy of heuristic-based models of crowd dynamics."
   *Journal of Statistical Physics*, 152(6), 1033-1068.
   - Multi-scale crowd modeling

### Numerical Methods

5. **Achdou, Y., & Capuzzo-Dolcetta, I. (2010)**. "Mean field games: numerical methods."
   *SIAM Journal on Numerical Analysis*, 48(3), 1136-1162.
   - Finite difference methods for MFG

6. **Hockney, R. W., & Eastwood, J. W. (1988)**. *Computer Simulation Using Particles*.
   Cambridge University Press.
   - Particle-in-Cell methods (PIC, CIC, NGP)

### Related Work

7. **Maury, B., Roudneff-Chupin, A., & Santambrogio, F. (2010)**. "A macroscopic crowd motion model of gradient flow type."
   *Mathematical Models and Methods in Applied Sciences*, 20(10), 1787-1821.
   - Gradient flow formulation for crowd dynamics

8. **Carlini, E., & Silva, F. J. (2014)**. "A semi-Lagrangian scheme for a degenerate second order mean field game system."
   *Discrete and Continuous Dynamical Systems*, 35(9), 4269-4292.
   - Semi-Lagrangian schemes for degenerate MFG

---

## Implementation Files

| Component | File | Lines | Status |
|:----------|:-----|------:|:-------|
| Package Init | `examples/advanced/capacity_constrained_mfg/__init__.py` | 84 | ✅ Complete |
| CapacityField | `examples/advanced/capacity_constrained_mfg/capacity_field.py` | 479 | ✅ Complete |
| CongestionModel | `examples/advanced/capacity_constrained_mfg/congestion.py` | 557 | ✅ Complete |
| CapacityConstrainedMFGProblem | `examples/advanced/capacity_constrained_mfg/problem.py` | 426 | ✅ Complete |
| Example Script | `examples/advanced/capacity_constrained_mfg/example_maze_mfg.py` | 333 | ✅ Complete |

**Total Implementation**: ~1,879 lines (example pattern complete)

**Note**: This is an example implementation residing in `examples/` to demonstrate
how users can extend the MFG_PDE framework with application-specific features.
Users can adapt this pattern for their own capacity-constrained problems.

---

## Future Extensions

### Planned Features

1. **MazeNavigationMFG** (next task)
   - Auto-detection of entry/exit points from maze
   - Distance-based terminal cost
   - Entry-localized initial density

2. **Anisotropic Capacity**
   - Direction-dependent capacity: `C(x, direction)`
   - Models one-way corridors, turnstiles

3. **Time-Varying Capacity**
   - Dynamic capacity: `C(x, t)`
   - Models closing doors, changing traffic patterns

4. **Multi-Population Extension**
   - Different agent types with different capacities
   - Extends to Di Francesco & Fagioli framework

### Research Directions

- **Optimal capacity design**: Inverse problem to design optimal corridor widths
- **Learning-based congestion models**: Data-driven `g(ρ)` from observations
- **Multi-scale coupling**: Microscopic (particle) ↔ Macroscopic (PDE)

---

**Last Updated**: 2025-11-12
**Implementation Status**: ✅ Complete as example pattern
**Architecture**: Minimal core framework - capacity constraints as extensible example
**Next**: Users can adapt this pattern for custom capacity-constrained MFG applications
