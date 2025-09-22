# 2D Anisotropic Crowd Dynamics: Mean Field Games Experiment

## Overview

This document describes a comprehensive numerical experiment for testing 2D Mean Field Games with non-separable Hamiltonians using the **anisotropic crowd dynamics model**. This experiment demonstrates how spatial coupling between movement directions creates complex evacuation patterns in structured environments.

## Mathematical Framework

### Problem Formulation

We consider a crowd evacuation scenario on a 2D domain $\Omega = [0,1]^2$ where architectural features create direction-dependent movement preferences.

#### MFG System

**Hamilton-Jacobi-Bellman (HJB) equation** (backward in time):
$$-\frac{\partial u}{\partial t} + H(x, \nabla u, m) = f(x)$$

**Fokker-Planck (FP) equation** (forward in time):
$$\frac{\partial m}{\partial t} + \text{div}(m \cdot D_p H(x, \nabla u, m)) - \sigma \Delta m = 0$$

#### Non-Separable Hamiltonian

The key feature is the **anisotropic Hamiltonian**:
$$H(x, p, m) = \frac{1}{2} p^T A(x) p + \gamma m(x,t) |p|^2$$

where the **anisotropy matrix** is:
$$A(x) = \begin{pmatrix} 1 & \rho(x) \\ \rho(x) & 1 \end{pmatrix}$$

with $|\rho(x)| < 1$ ensuring positive definiteness.

#### Expanded PDE System

**HJB equation**:
$$-\frac{\partial u}{\partial t} + \frac{1}{2}\left[\left(\frac{\partial u}{\partial x_1}\right)^2 + 2\rho(x)\frac{\partial u}{\partial x_1}\frac{\partial u}{\partial x_2} + \left(\frac{\partial u}{\partial x_2}\right)^2\right] + \gamma m(x,t)|\nabla u|^2 = f(x)$$

**FP equation**:
$$\frac{\partial m}{\partial t} + \text{div}\left(m \begin{pmatrix} \frac{\partial u}{\partial x_1} + \rho(x)\frac{\partial u}{\partial x_2} + 2\gamma m\frac{\partial u}{\partial x_1} \\ \rho(x)\frac{\partial u}{\partial x_1} + \frac{\partial u}{\partial x_2} + 2\gamma m\frac{\partial u}{\partial x_2} \end{pmatrix}\right) - \sigma \Delta m = 0$$

### Physical Interpretation

- **Cross-coupling term** $\rho(x) \frac{\partial u}{\partial x_1}\frac{\partial u}{\partial x_2}$: Represents architectural influence on movement preferences
- **Density-dependent friction** $\gamma m |\nabla u|^2$: Models congestion effects where movement slows in crowded areas
- **Spatial variation** $\rho(x)$: Creates checkerboard pattern of preferred movement directions

## Experimental Configuration

### Domain and Geometry

**Spatial domain**: $\Omega = [0,1]^2$ (unit square)
**Physical interpretation**: Public space with structured architectural features

### Barrier Configurations and Analysis

The experiment can be significantly enhanced with internal barriers to model realistic architectural constraints and test numerical methods on complex geometries.

#### Benefits of Adding Barriers

**Enhanced Physical Realism**:
- Models actual evacuation scenarios with architectural features (columns, furniture, walls)
- Creates realistic crowd channeling and bottleneck effects
- Enables validation against experimental crowd dynamics data with known obstacles

**Mathematical Interest**:
- Tests non-convex domain handling in MFG systems
- Creates flow separation and interaction through anisotropic coupling
- Generates complex boundary layer effects between geometry and $\rho(x)$ patterns

**Computational Challenges**:
- Validates numerical methods on irregular geometries
- Tests stability of cross-coupling terms with complex boundaries
- May require adaptive meshing and immersed boundary techniques

#### Barrier Configuration Options

**Configuration A: Central Obstacle**
- **Geometry**: Circular barrier at center $(0.5, 0.4)$ with radius $0.15$
- **Purpose**: Tests flow splitting around central obstacle
- **Expected behavior**: Symmetric flow patterns with anisotropy-induced asymmetries
- **Validation focus**: Mass conservation around curved boundaries

**Configuration B: Anisotropy-Aligned Barriers** ⭐ **RECOMMENDED**
- **Geometry**:
  - Diagonal barrier 1: from $(0.1, 0.1)$ to $(0.4, 0.4)$ (aligns with Region A $(1,1)$ preference)
  - Diagonal barrier 2: from $(0.9, 0.1)$ to $(0.6, 0.4)$ (aligns with Region B $(1,-1)$ preference)
- **Strategic design**: Barriers **enhance** rather than conflict with natural anisotropic flow
- **Expected behavior**: Amplified channeling effects, improved evacuation efficiency
- **Research value**: Tests synergy between geometric design and mathematical preferences

**Configuration C: Corridor System**
- **Geometry**: Two rectangular columns
  - Column 1: $[0.3, 0.4] \times [0.3, 0.7]$
  - Column 2: $[0.6, 0.7] \times [0.3, 0.7]$
- **Purpose**: Models bottleneck scenarios with constrained passages
- **Expected behavior**: Congestion effects amplified by geometric constraints
- **Validation focus**: Density buildup and pressure dynamics

#### Barrier-Anisotropy Interaction Analysis

**Flow Enhancement Mechanisms**:
- **Geometric channeling**: Barriers create physical flow corridors
- **Anisotropic amplification**: $\rho(x)$ preferences align with barrier-induced channels
- **Emergent optimization**: Combined effect may improve evacuation times

**Mathematical Coupling**:
The Hamiltonian near barriers becomes:
$$H_{\text{barrier}}(x, p, m) = \begin{cases}
\frac{1}{2}p^T A(x) p + \gamma m |p|^2 & \text{if } x \notin \mathcal{B} \\
+\infty & \text{if } x \in \mathcal{B}
\end{cases}$$

where $\mathcal{B}$ represents barrier regions.

### Anisotropy Pattern

**Spatial variation**:
$$\rho(x_1, x_2) = 0.5 \sin(\pi x_1) \cos(\pi x_2)$$

This creates a **checkerboard pattern** with four distinct regions:

| Region | Location | $\rho$ sign | Preferred Direction |
|--------|----------|-------------|-------------------|
| A | $(0.25, 0.25)$ | $\rho > 0$ | $(1,1)$ diagonal |
| B | $(0.75, 0.25)$ | $\rho < 0$ | $(1,-1)$ diagonal |
| C | $(0.25, 0.75)$ | $\rho < 0$ | $(1,-1)$ diagonal |
| D | $(0.75, 0.75)$ | $\rho > 0$ | $(1,1)$ diagonal |

### Boundary Conditions

**Mixed boundary conditions** representing evacuation scenario:

- **Walls** (no-flux): $x_1 = 0$, $x_1 = 1$, $x_2 = 0$
  $$m \cdot v^* \cdot \nu = 0$$

- **Exit** (Dirichlet): $x_2 = 1$
  $$u(x_1, 1, t) = 0$$

### Physical Parameters

```python
# Anisotropy strength
rho_amplitude = 0.5

# Congestion coupling
gamma = 0.1

# Diffusion coefficient
sigma = 0.01

# Final time
T_final = 1.0
```

### Initial and Terminal Conditions

**Initial density** (Gaussian blob in lower-left):
$$m_0(x_1, x_2) = \frac{2}{\sigma_{\text{init}}^2} \exp\left(-\frac{(x_1-0.2)^2 + (x_2-0.2)^2}{\sigma_{\text{init}}^2}\right)$$
with $\sigma_{\text{init}} = 0.15$

**Terminal condition** for HJB (distance to exit):
$$u(x_1, x_2, T) = (x_1 - 0.5)^2 + (x_2 - 1.0)^2$$

**Running cost**:
$$f(x) = 1.0 \quad \text{(uniform cost)}$$

## Implementation in MFG_PDE Package

### Problem Class Structure

```python
from mfg_pde import ExampleMFGProblem, BoundaryConditions
from mfg_pde.factory import create_fast_solver
from mfg_pde.config import create_fast_config
import numpy as np

class AnisotropicCrowdDynamics(ExampleMFGProblem):
    """
    2D MFG with anisotropic Hamiltonian for crowd evacuation.

    Mathematical formulation:
    - HJB: -∂u/∂t + ½[u_x² + 2ρ(x)u_x u_y + u_y²] + γm|∇u|² = 1
    - FP: ∂m/∂t + div(m[∇u + 2γm∇u]) - σΔm = 0
    """

    def __init__(self, gamma=0.1, sigma=0.01, rho_amplitude=0.5):
        self.gamma = gamma
        self.sigma = sigma
        self.rho_amplitude = rho_amplitude

        # Domain configuration
        super().__init__(
            domain_bounds=[(0.0, 1.0), (0.0, 1.0)],
            time_horizon=1.0,
            grid_size=(64, 64)
        )
```

### Hamiltonian Implementation

```python
def compute_hamiltonian(self, x, p, m):
    """
    Non-separable anisotropic Hamiltonian.

    Args:
        x: spatial coordinates (N, 2)
        p: momentum/gradient (N, 2)
        m: density (N,)

    Returns:
        H: Hamiltonian values (N,)
    """
    x1, x2 = x[:, 0], x[:, 1]
    p1, p2 = p[:, 0], p[:, 1]

    # Anisotropy function
    rho = self.rho_amplitude * np.sin(np.pi * x1) * np.cos(np.pi * x2)

    # Quadratic form: ½[p₁² + 2ρp₁p₂ + p₂²]
    kinetic_energy = 0.5 * (p1**2 + 2*rho*p1*p2 + p2**2)

    # Density-dependent friction: γm|p|²
    friction_term = self.gamma * m * (p1**2 + p2**2)

    return kinetic_energy + friction_term

def compute_hamiltonian_gradient(self, x, p, m):
    """
    Gradient ∇_p H for velocity computation.

    Returns:
        grad_p_H: velocity field (N, 2)
    """
    x1, x2 = x[:, 0], x[:, 1]
    p1, p2 = p[:, 0], p[:, 1]

    rho = self.rho_amplitude * np.sin(np.pi * x1) * np.cos(np.pi * x2)

    # ∂H/∂p₁ = p₁ + ρp₂ + 2γmp₁
    grad_p1 = p1 + rho*p2 + 2*self.gamma*m*p1

    # ∂H/∂p₂ = ρp₁ + p₂ + 2γmp₂
    grad_p2 = rho*p1 + p2 + 2*self.gamma*m*p2

    return np.column_stack([grad_p1, grad_p2])
```

### Boundary Conditions Setup

```python
def setup_boundary_conditions(self):
    """Configure mixed boundary conditions for evacuation scenario."""

    # No-flux boundary conditions on walls
    no_flux_boundaries = [
        ('left', {'type': 'no_flux'}),    # x₁ = 0
        ('right', {'type': 'no_flux'}),   # x₁ = 1
        ('bottom', {'type': 'no_flux'})   # x₂ = 0
    ]

    # Dirichlet condition at exit
    exit_boundary = ('top', {
        'type': 'dirichlet',
        'value': lambda x, t: 0.0  # u = 0 at exit
    })

    return BoundaryConditions([
        *no_flux_boundaries,
        exit_boundary
    ])

def setup_barriers(self, configuration='anisotropy_aligned'):
    """Configure internal barriers within domain."""

    if configuration == 'central_obstacle':
        barriers = [CircularBarrier(center=(0.5, 0.4), radius=0.15)]

    elif configuration == 'anisotropy_aligned':
        barriers = [
            LinearBarrier(start=(0.1, 0.1), end=(0.4, 0.4)),  # Region A alignment
            LinearBarrier(start=(0.9, 0.1), end=(0.6, 0.4))   # Region B alignment
        ]

    elif configuration == 'corridor_system':
        barriers = [
            RectangularBarrier(bounds=[(0.3, 0.4), (0.3, 0.7)]),
            RectangularBarrier(bounds=[(0.6, 0.7), (0.3, 0.7)])
        ]

    else:
        barriers = []  # No barriers

    return barriers
```

### Barrier Implementation Strategies

#### Method 1: Penalty Method (Recommended for Initial Implementation)

```python
def add_barrier_penalty(self, x, u, m):
    """Add large penalty for barrier regions using smooth approximation."""

    # Compute distance to barriers
    barrier_distance = self.compute_barrier_distance(x)

    # Smooth penalty function (avoids numerical issues)
    epsilon = 0.02  # Barrier thickness parameter
    penalty_strength = 1e6

    # Exponential penalty for value function
    u_penalty = penalty_strength * np.exp(-barrier_distance / epsilon)

    # Density suppression in barriers (smooth cutoff)
    density_cutoff = 0.5 * (1 + np.tanh(barrier_distance / epsilon))
    m_corrected = m * density_cutoff

    return u + u_penalty, m_corrected

def compute_barrier_distance(self, x):
    """Compute signed distance to barrier boundaries."""
    distances = []

    for barrier in self.barriers:
        if isinstance(barrier, CircularBarrier):
            center_dist = np.linalg.norm(x - barrier.center, axis=1)
            distances.append(center_dist - barrier.radius)

        elif isinstance(barrier, LinearBarrier):
            # Distance to line segment
            line_dist = self.point_to_line_distance(x, barrier.start, barrier.end)
            distances.append(line_dist)

        elif isinstance(barrier, RectangularBarrier):
            # Distance to rectangle boundary
            rect_dist = self.point_to_rectangle_distance(x, barrier.bounds)
            distances.append(rect_dist)

    # Return minimum distance to any barrier
    return np.minimum.reduce(distances) if distances else np.full(len(x), np.inf)
```

#### Method 2: Modified Grid Approach

```python
class BarrierAwareGrid:
    """Grid that handles internal barriers through masking and modified operators."""

    def __init__(self, domain_bounds, grid_size, barriers):
        self.barriers = barriers
        self.setup_masked_grid()

    def setup_masked_grid(self):
        """Create grid with barrier cells marked and special boundary conditions."""

        # Standard uniform grid
        self.x_grid = np.linspace(self.domain_bounds[0][0], self.domain_bounds[0][1], self.grid_size[0])
        self.y_grid = np.linspace(self.domain_bounds[1][0], self.domain_bounds[1][1], self.grid_size[1])
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)

        # Mark barrier cells
        self.active_mask = ~self.compute_barrier_mask()

        # Identify barrier-adjacent cells for special treatment
        self.barrier_neighbors = self.find_barrier_neighbors()

        # Setup modified finite difference operators
        self.setup_barrier_aware_operators()

    def apply_no_flux_at_barriers(self, u, m):
        """Enforce no-flux conditions at barrier boundaries."""

        u_modified = u.copy()
        m_modified = m.copy()

        for i, j in self.barrier_neighbors:
            # Compute barrier normal direction
            normal = self.compute_barrier_normal(i, j)

            # Project velocity tangentially to barrier
            velocity = -self.compute_gradient_at_point(u, i, j)
            velocity_normal = np.dot(velocity, normal)
            velocity_tangential = velocity - velocity_normal * normal

            # Update value function to enforce tangential flow
            # This is an approximation - exact enforcement requires iterative methods
            u_modified[i, j] = self.project_gradient_tangentially(u, i, j, normal)

        # Zero density inside barriers
        m_modified[~self.active_mask] = 0.0

        return u_modified, m_modified
```

#### Method 3: Level Set Approach (Advanced)

```python
def setup_level_set_barriers(self):
    """Use level set functions to represent complex barrier geometries."""

    def barrier_level_set(x):
        """
        Level set function: negative inside barriers, positive outside.
        Zero level set represents barrier boundary.
        """
        level_values = []

        for barrier in self.barriers:
            if isinstance(barrier, CircularBarrier):
                center_dist = np.linalg.norm(x - barrier.center, axis=1)
                level_values.append(center_dist - barrier.radius)

            elif isinstance(barrier, LinearBarrier):
                # Signed distance to line (with thickness)
                line_dist = self.signed_distance_to_line(x, barrier.start, barrier.end)
                thickness = getattr(barrier, 'thickness', 0.05)
                level_values.append(line_dist - thickness/2)

        # Union of all barriers (minimum distance)
        return np.minimum.reduce(level_values) if level_values else np.full(len(x), np.inf)

    def modified_hamiltonian_with_barriers(self, x, p, m):
        """Hamiltonian modified to handle barrier constraints via level sets."""

        # Compute level set values
        phi = barrier_level_set(x)

        # Base Hamiltonian computation
        base_H = self.compute_base_hamiltonian(x, p, m)

        # Near barriers (|φ| < ε), modify momentum to enforce tangential flow
        epsilon = 0.05
        near_barrier_mask = np.abs(phi) < epsilon

        if np.any(near_barrier_mask):
            # Compute gradient of level set (barrier normal)
            grad_phi = self.compute_level_set_gradient(phi, x)
            normal = grad_phi / (np.linalg.norm(grad_phi, axis=1, keepdims=True) + 1e-12)

            # Project momentum tangentially for cells near barriers
            p_normal_component = np.sum(p * normal, axis=1, keepdims=True)
            p_tangential = p - p_normal_component * normal

            # Use projected momentum for barrier-adjacent regions
            p_modified = np.where(near_barrier_mask[:, None], p_tangential, p)

            return self.compute_base_hamiltonian(x, p_modified, m)

        return base_H

    return barrier_level_set, modified_hamiltonian_with_barriers
```

### Initial and Terminal Conditions Setup

```python
def setup_initial_conditions(self):
    """Configure initial density and terminal value function."""

    def initial_density(x):
        """Gaussian blob in lower-left corner."""
        x1, x2 = x[:, 0], x[:, 1]
        sigma_init = 0.15
        return (2/sigma_init**2) * np.exp(
            -((x1 - 0.2)**2 + (x2 - 0.2)**2) / sigma_init**2
        )

    def terminal_condition(x):
        """Distance-based terminal cost."""
        x1, x2 = x[:, 0], x[:, 1]
        return (x1 - 0.5)**2 + (x2 - 1.0)**2

    return {
        'initial_density': initial_density,
        'terminal_value': terminal_condition,
        'running_cost': lambda x, m: np.ones_like(m)  # f(x) = 1
    }
```

### Solver Configuration

```python
def create_experiment_solver():
    """Create optimized solver for anisotropic experiment."""

    # Configure for non-separable 2D problem
    config = create_fast_config(
        solver_type='semi_lagrangian',
        spatial_scheme='central_difference',
        temporal_scheme='semi_implicit',
        grid_refinement='adaptive'
    )

    # Adjust for cross-coupling stability
    config.update({
        'cfl_safety_factor': 0.3,  # Conservative for cross-derivatives
        'nonlinear_tolerance': 1e-6,
        'max_iterations': 1000,
        'diffusion_implicit': True  # Implicit diffusion for stability
    })

    return create_fast_solver(config)
```

## Computational Implementation

### Discretization Strategy

**Spatial discretization**: $N_x = N_y = 64$ (uniform grid)

**Finite difference schemes**:
- **Standard derivatives**: Central differences for $\partial u/\partial x_1$, $\partial u/\partial x_2$
- **Cross-derivative**: Product of central differences for $\frac{\partial u}{\partial x_1}\frac{\partial u}{\partial x_2}$
- **Laplacian**: 5-point stencil for diffusion term

**Temporal scheme**: Semi-implicit approach
- HJB equation: Explicit treatment of nonlinear Hamiltonian
- FP equation: Implicit diffusion, explicit transport

#### Special Considerations for Barriers

**Modified CFL Condition**:
```python
# Enhanced time step constraints with barriers
dt_barrier = min(
    0.5 * min(dx, dy) / max(abs(v)),                    # standard transport CFL
    0.1 * min_barrier_distance / max(abs(v)),           # barrier resolution CFL
    0.5 * min(dx**2, dy**2) / sigma,                    # diffusion CFL
    0.05 / (gamma * max(m) + barrier_penalty_strength)  # nonlinear + penalty CFL
)
```

**Grid Refinement Near Barriers**:
For accurate representation of barrier boundaries:
- **Adaptive mesh**: Higher resolution near $\partial\mathcal{B}$
- **Barrier thickness**: Ensure $\epsilon_{\text{barrier}} \geq 2\Delta x$
- **Smooth transitions**: Use hyperbolic tangent profiles for penalty functions

### Algorithm Outline

```python
def solve_anisotropic_mfg(barrier_config='none'):
    """Main solution algorithm with optional barrier configurations."""

    # Initialize problem with barrier configuration
    problem = AnisotropicCrowdDynamics(
        gamma=0.1,
        sigma=0.01,
        rho_amplitude=0.5,
        barrier_configuration=barrier_config
    )
    solver = create_experiment_solver()

    # Setup logging and monitoring
    from mfg_pde.utils.logging import configure_research_logging, get_logger
    configure_research_logging(f"anisotropic_crowd_{barrier_config}", level="INFO")
    logger = get_logger(__name__)

    # Log barrier configuration
    if barrier_config != 'none':
        logger.info(f"Running experiment with barrier configuration: {barrier_config}")
        logger.info(f"Number of barriers: {len(problem.barriers)}")

    # Solve coupled system
    solution = solver.solve(problem)

    # Verify and analyze results
    verify_solution(solution)
    analyze_evacuation_patterns(solution)

    # Additional barrier-specific analysis
    if barrier_config != 'none':
        analyze_barrier_effects(solution, problem.barriers)

    return solution

def verify_solution(solution):
    """Comprehensive solution verification."""

    # Mass conservation check
    total_mass_history = [
        np.trapz(np.trapz(m, axis=1), axis=0)
        for m in solution.density_history
    ]
    mass_variation = np.max(total_mass_history) - np.min(total_mass_history)
    assert mass_variation < 1e-6, f"Mass not conserved: variation = {mass_variation}"

    # Boundary condition verification
    for t_idx, (u, m) in enumerate(zip(solution.value_history, solution.density_history)):
        # Check exit condition: u ≈ 0 on top boundary
        exit_values = u[-1, :]  # x₂ = 1 boundary
        assert np.max(np.abs(exit_values)) < 1e-3, "Exit condition violated"

        # Check no-flux on walls
        verify_no_flux_conditions(u, m, t_idx)
```

### Performance Optimization

**Key optimizations for MFG_PDE package**:

1. **Vectorized Hamiltonian computation**: Use NumPy broadcasting for efficient evaluation
2. **Sparse linear algebra**: Leverage scipy.sparse for large grid systems
3. **Adaptive time stepping**: Adjust $\Delta t$ based on CFL conditions
4. **Memory management**: Efficient storage of solution history
5. **Parallel processing**: OpenMP-style parallelization for grid operations

```python
# Example: Vectorized anisotropy computation
@numba.jit(nopython=True)
def compute_rho_vectorized(x1_grid, x2_grid, amplitude):
    """Fast computation of ρ(x) on full grid."""
    return amplitude * np.sin(np.pi * x1_grid) * np.cos(np.pi * x2_grid)
```

## Expected Results and Analysis

### Predicted Evolution Phases

#### Without Barriers (Baseline)

**Phase 1** (t ∈ [0, 0.3]): **Initial redistribution**
- Crowd spreads from lower-left corner
- Density gradients align with local anisotropy patterns
- Formation of preferred flow directions in each region

**Phase 2** (t ∈ [0.3, 0.7]): **Channel formation**
- Emergence of distinct diagonal flow channels:
  - Regions A,D: $(1,1)$ diagonal flow toward upper-right
  - Regions B,C: $(1,-1)$ diagonal flow toward upper-left
- Rotational circulation patterns between regions

**Phase 3** (t ∈ [0.7, 1.0]): **Convergence and evacuation**
- All channels redirect toward exit at $x_2 = 1$
- Congestion effects become significant near exit
- Final evacuation with density-dependent slowdown

#### With Barriers (Enhanced Scenarios)

**Configuration A: Central Obstacle**
- **Phase 1**: Flow splits around circular barrier
- **Phase 2**: Asymmetric splitting due to anisotropy influence
  - Left flow: enhanced by $\rho < 0$ regions
  - Right flow: enhanced by $\rho > 0$ regions
- **Phase 3**: Reunification with complex vortex patterns

**Configuration B: Anisotropy-Aligned Barriers**
- **Phase 1**: Immediate channeling along barrier-defined corridors
- **Phase 2**: **Amplified anisotropic effects**:
  - Super-diagonal channels with enhanced velocities
  - Reduced cross-flow between regions
  - Higher evacuation efficiency
- **Phase 3**: Coordinated multi-channel evacuation

**Configuration C: Corridor System**
- **Phase 1**: Bottleneck formation at corridor entrance
- **Phase 2**: **Pressure buildup** and density waves
  - Congestion amplified by geometric constraints
  - Anisotropy helps organize queuing patterns
- **Phase 3**: Sequential evacuation through narrow passages

### Enhanced Flow Patterns with Barriers

**Vortex Generation**:
Around circular barriers, the combination of geometric deflection and anisotropic preferences creates **complex circulation patterns**:
$$\omega_{\text{effective}} = \omega_{\text{geometric}} + \omega_{\text{anisotropic}}$$

**Channel Amplification**:
Anisotropy-aligned barriers create **superposition effects**:
- Geometric channeling: $v_{\text{geom}} \propto \nabla \phi_{\text{barrier}}$ (normal to barriers)
- Anisotropic preference: $v_{\text{aniso}} \propto A(x) \nabla u$
- Combined effect: Enhanced flow efficiency when aligned

**Bottleneck Dynamics**:
In corridor configurations, anisotropy influences **queue organization**:
- Regions with compatible $\rho(x)$ form more orderly queues
- Cross-cutting anisotropy creates lateral pressure
- Exit efficiency depends on $\rho(x)$ values near bottlenecks

### Key Performance Metrics

```python
def compute_evacuation_metrics(solution, barriers=None):
    """Analyze evacuation efficiency and flow patterns with barrier considerations."""

    metrics = {}

    # Basic evacuation time analysis
    total_mass = [np.sum(m) for m in solution.density_history]
    initial_mass = total_mass[0]

    # Time for 90% evacuation
    evacuation_90_idx = np.where(np.array(total_mass) <= 0.1 * initial_mass)[0]
    metrics['T_90_percent'] = solution.time_grid[evacuation_90_idx[0]] if len(evacuation_90_idx) > 0 else None

    # Peak density tracking
    metrics['peak_density'] = np.max([np.max(m) for m in solution.density_history])

    # Velocity magnitude analysis
    velocity_magnitudes = []
    for u in solution.value_history:
        grad_u = np.gradient(u)
        vel_mag = np.sqrt(grad_u[0]**2 + grad_u[1]**2)
        velocity_magnitudes.append(np.max(vel_mag))
    metrics['max_velocity'] = np.max(velocity_magnitudes)

    # Anisotropy utilization (measure of directional preference usage)
    metrics['anisotropy_utilization'] = compute_anisotropy_utilization(solution)

    # Barrier-specific metrics
    if barriers:
        metrics.update(compute_barrier_specific_metrics(solution, barriers))

    return metrics

def compute_barrier_specific_metrics(solution, barriers):
    """Compute metrics specific to barrier configurations."""

    barrier_metrics = {}

    # Flow splitting efficiency (for central obstacle)
    if any(isinstance(b, CircularBarrier) for b in barriers):
        barrier_metrics['flow_splitting_asymmetry'] = analyze_flow_splitting(solution, barriers)

    # Channel amplification factor (for aligned barriers)
    if any(isinstance(b, LinearBarrier) for b in barriers):
        barrier_metrics['channel_amplification'] = analyze_channel_amplification(solution, barriers)

    # Bottleneck pressure analysis (for corridor systems)
    if any(isinstance(b, RectangularBarrier) for b in barriers):
        barrier_metrics['bottleneck_pressure'] = analyze_bottleneck_dynamics(solution, barriers)

    # General barrier effects
    barrier_metrics['circulation_strength'] = compute_circulation_around_barriers(solution, barriers)
    barrier_metrics['barrier_efficiency_gain'] = compare_evacuation_efficiency(solution)

    return barrier_metrics

def analyze_flow_splitting(solution, barriers):
    """Analyze asymmetry in flow splitting around circular barriers."""
    # Implementation for analyzing left vs right flow around obstacles
    pass

def analyze_channel_amplification(solution, barriers):
    """Measure how barriers amplify anisotropic channeling effects."""
    # Implementation for measuring velocity enhancement in barrier-aligned regions
    pass

def analyze_bottleneck_dynamics(solution, barriers):
    """Analyze pressure buildup and queue organization in corridor systems."""
    # Implementation for measuring density gradients and flow organization
    pass
```

### Visualization Outputs

**Required visualizations**:

1. **Density evolution**: Animated heatmaps showing $m(x,t)$
2. **Velocity fields**: Quiver plots of $v^*(x,t) = -\nabla_p H$
3. **Anisotropy overlay**: $\rho(x)$ field with flow streamlines
4. **Evacuation metrics**: Time series of mass, peak density, velocity
5. **Cross-sectional analysis**: Density profiles along key diagonal lines

**Additional barrier-specific visualizations**:

6. **Barrier influence maps**: Showing how barriers modify flow patterns
7. **Circulation analysis**: Vorticity fields around obstacles
8. **Channel efficiency plots**: Comparing flow rates in different configurations
9. **Pressure buildup visualization**: Density gradients near bottlenecks
10. **Comparative animations**: Side-by-side with/without barriers

```python
def create_visualization_suite(solution):
    """Generate comprehensive visualization package."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Density evolution animation
    create_density_animation(solution)

    # Velocity field snapshots
    create_velocity_field_plots(solution, t_snapshots=[0.2, 0.5, 0.8])

    # Anisotropy influence analysis
    create_anisotropy_analysis_plots(solution)

    # Performance metrics dashboard
    create_metrics_dashboard(solution)
```

## Validation Protocol

### Mathematical Verification

1. **Mass conservation**: $\frac{d}{dt}\int_\Omega m dx = 0$
2. **Energy consistency**: HJB and FP equations satisfy coupled equilibrium
3. **Boundary compliance**: No-flux conditions maintained throughout evolution
4. **Convergence analysis**: Grid refinement study ($32^2$, $64^2$, $128^2$)

**Additional verification for barriers**:
5. **Barrier impermeability**: Zero flux through barrier boundaries
6. **Penalty function smoothness**: Continuity of modified Hamiltonian
7. **Geometric accuracy**: Correct barrier shape representation on grid

### Physical Validation

1. **Directional preferences**: Verify flow aligns with $\rho(x)$ patterns
2. **Congestion effects**: Confirm density-dependent velocity reduction
3. **Boundary behavior**: Realistic sliding along walls, proper exit flow
4. **Emergency evacuation realism**: Compare with crowd dynamics literature

**Barrier-specific physical validation**:
5. **Flow splitting realism**: Symmetric splitting for isotropic cases
6. **Channel formation**: Enhanced flow in barrier-aligned corridors
7. **Bottleneck behavior**: Realistic queue formation and pressure buildup
8. **Circulation patterns**: Physical vortex formation around obstacles

### Computational Validation

```python
def run_validation_suite():
    """Comprehensive validation testing."""

    # Grid convergence study
    grid_sizes = [32, 64, 128]
    convergence_rates = []

    reference_solution = solve_reference_problem(grid_size=128)

    for grid_size in grid_sizes[:-1]:
        coarse_solution = solve_anisotropic_mfg(grid_size=grid_size)
        error = compute_solution_error(coarse_solution, reference_solution)
        convergence_rates.append(error)

    # Verify second-order spatial convergence
    spatial_order = estimate_convergence_order(convergence_rates, grid_sizes[:-1])
    assert 1.8 <= spatial_order <= 2.2, f"Spatial convergence order: {spatial_order}"

    # Time step sensitivity analysis
    dt_values = [0.001, 0.0005, 0.0001]
    temporal_errors = []

    for dt in dt_values:
        solution_dt = solve_anisotropic_mfg(time_step=dt)
        temporal_errors.append(compute_temporal_error(solution_dt))

    # Performance benchmarking
    benchmark_solver_performance()
```

## Implementation Notes

### Package Integration

This experiment integrates with existing MFG_PDE package components:

- **Core solver**: Uses `mfg_pde.solvers.SemiLagrangianSolver`
- **Factory patterns**: `mfg_pde.factory.create_fast_solver()`
- **Configuration**: `mfg_pde.config.create_fast_config()`
- **Utilities**: `mfg_pde.utils.logging`, `mfg_pde.utils.visualization`

### Dependencies

```python
# Core numerical libraries
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

# MFG_PDE package components
from mfg_pde import ExampleMFGProblem, BoundaryConditions
from mfg_pde.factory import create_fast_solver
from mfg_pde.config import create_fast_config
from mfg_pde.utils.logging import get_logger, configure_research_logging

# Visualization and analysis
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm import tqdm

# Optional performance enhancements
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
```

### File Organization

```
examples/advanced/2d_anisotropic_crowd_dynamics/
├── README.md                                    # This document (comprehensive experiment guide)
├── anisotropic_2d_problem.py                   # Core 2D MFG problem implementation with barriers
├── solver_config.py                            # Solver configuration and optimization
├── experiment_runner.py                        # Main experiment orchestration script
├── numerical_demo.py                           # Numerically stable demonstration
├── validation/
│   └── convergence_study.py                    # Grid refinement and convergence analysis
├── analysis/
│   └── visualization_tools.py                  # Comprehensive visualization suite
└── results/                                     # Generated output directory
    ├── experiment_summary.json                 # Experiment results summary
    ├── density_evolution.html                  # Interactive density animation
    ├── velocity_field.png                      # Velocity field visualization
    ├── anisotropy_analysis.png                 # Anisotropy pattern analysis
    ├── barrier_influence.png                   # Barrier effects analysis
    ├── metrics_dashboard.html                  # Interactive metrics dashboard
    ├── convergence/                            # Convergence study results
    │   ├── convergence_results.json           # Numerical convergence data
    │   └── convergence_report.md              # Convergence analysis report
    └── comparative_analysis.json              # Multi-configuration comparison
```

## Research Applications

### Immediate Applications

**Without Barriers (Baseline)**:
1. **Anisotropic flow modeling**: Understanding directional preferences in open spaces
2. **Architectural influence**: How building design affects movement patterns
3. **Mathematical method development**: Computational techniques for non-separable systems

**With Barriers (Enhanced Applications)**:
1. **Evacuation planning**: Optimize barrier placement and corridor design for emergency scenarios
2. **Crowd management**: Predict and prevent density hotspots using strategic obstacle placement
3. **Urban design**: Inform pedestrian area layout with integrated architectural features
4. **Event planning**: Stadium and venue circulation optimization with barriers and channeling
5. **Emergency response**: Design barrier configurations that enhance rather than impede evacuation

### Methodological Contributions

**Core Mathematical Advances**:
1. **Non-separable MFG numerics**: Computational techniques for cross-coupled systems
2. **Anisotropic modeling**: Framework for direction-dependent preferences in MFG
3. **Boundary condition handling**: Mixed Dirichlet-Neumann treatment for complex domains
4. **Validation protocols**: Systematic verification for complex MFG systems

**Barrier-Specific Innovations**:
5. **Immersed boundary MFG**: Techniques for internal obstacles in mean field systems
6. **Geometric-anisotropic coupling**: Mathematical framework for barrier-preference interaction
7. **Multi-scale flow analysis**: From local barrier effects to global evacuation patterns
8. **Optimization methodology**: Barrier placement strategies based on MFG solutions

### Research Questions Enabled by Barriers

**Fundamental Questions**:
- How do architectural features interact with crowd psychology (modeled via anisotropy)?
- What barrier configurations optimize evacuation efficiency for given anisotropy patterns?
- Can barriers be designed to **enhance** natural flow preferences rather than obstruct them?

**Computational Questions**:
- How do penalty methods compare to level set approaches for barrier implementation?
- What grid resolution is required for accurate barrier boundary representation?
- How do cross-coupling terms affect numerical stability near barrier boundaries?

**Applied Questions**:
- What is the optimal barrier placement for real architectural scenarios?
- How do barriers affect evacuation time variance across different crowd densities?
- Can anisotropic preferences be deliberately induced through barrier design?

### Future Extensions

**3D and Multi-Scale Extensions**:
1. **Multi-level buildings**: 3D evacuation with stairwells and barriers
2. **Network-integrated barriers**: Barriers on graph-based domains
3. **Multi-scale modeling**: Local barrier effects in global transportation networks

**Advanced Barrier Modeling**:
4. **Adaptive barriers**: Time-dependent barrier configurations (opening/closing doors)
5. **Permeable barriers**: Partial barriers with controlled flow rates
6. **Dynamic barriers**: Moving obstacles like vehicles or temporary structures

**Realistic Enhancements**:
7. **Multi-population models**: Different agent types with varying barrier interactions
8. **Stochastic barriers**: Random obstacle placement for robustness analysis
9. **Real-data calibration**: Parameter fitting to experimental crowd-barrier data
10. **Machine learning integration**: AI-optimized barrier placement using MFG simulations

### Barrier Design Optimization Framework

The enhanced experiment opens possibilities for **systematic barrier optimization**:

```python
def optimize_barrier_placement(anisotropy_field, domain, exit_locations):
    """
    Find optimal barrier configuration for given anisotropy and exits.

    Returns:
        optimal_barriers: List of barrier objects
        evacuation_time: Minimized evacuation time
        efficiency_gain: Improvement over no-barrier case
    """

    # Search space: barrier types, positions, orientations
    # Objective: minimize evacuation time, maximize flow efficiency
    # Constraints: architectural feasibility, safety regulations
```

This framework enables **evidence-based architectural design** where barrier placement is optimized mathematically rather than based solely on intuition or simple heuristics.

## Conclusion

This experiment provides a comprehensive framework for testing 2D Mean Field Games with non-separable Hamiltonians, significantly enhanced by the inclusion of barrier configurations. The anisotropic crowd dynamics model demonstrates how spatial coupling creates emergent evacuation patterns that cannot be captured by separable formulations, while barriers add realistic architectural constraints that amplify these effects.

### Core Mathematical Contributions

The implementation leverages the MFG_PDE package's modular architecture while introducing new computational techniques for handling:
- **Cross-dimensional coupling terms** in non-separable Hamiltonians
- **Internal barriers** through penalty methods, grid masking, and level set approaches
- **Complex boundary conditions** mixing no-flux walls, exit boundaries, and barrier interfaces

### Enhanced Experimental Value

The barrier configurations transform this from a purely mathematical exercise into a practical tool with immediate applications:

**Without barriers**: Validates computational methods for non-separable systems and provides baseline understanding of anisotropic flow patterns.

**With barriers**: Creates realistic evacuation scenarios that test the interaction between geometric constraints and mathematical preferences, opening new research directions in architectural optimization.

### Key Technical Innovations

1. **Robust cross-derivative handling**: Stable finite difference schemes for $\rho(x)\frac{\partial u}{\partial x_1}\frac{\partial u}{\partial x_2}$ terms
2. **Multi-method barrier implementation**: Penalty, grid masking, and level set approaches for different barrier types
3. **Enhanced validation protocols**: Mathematical, physical, and barrier-specific verification procedures
4. **Barrier-anisotropy coupling analysis**: Framework for understanding geometric-preference interactions

### Research Impact

This enhanced experiment enables investigation of fundamental questions:
- How do architectural features interact with crowd psychology?
- Can barriers be designed to **enhance** rather than obstruct natural flow patterns?
- What mathematical frameworks best capture geometric-anisotropic coupling?

### Future Directions

The barrier framework opens pathways to:
- **Optimization-based design**: Mathematical approaches to architectural planning
- **Multi-scale modeling**: From local barrier effects to city-wide transportation
- **AI-integrated planning**: Machine learning approaches to barrier placement
- **Real-world validation**: Calibration against experimental crowd dynamics data

### Conclusion

This experiment establishes a new standard for testing non-separable MFG systems, combining mathematical rigor with practical relevance. The barrier enhancements demonstrate that sophisticated mathematical models can directly inform real-world design decisions, bridging the gap between theoretical MFG research and practical applications in crowd management and architectural design.

The implementation serves as both a validation tool for numerical methods and a foundation for evidence-based approaches to evacuation planning, positioning MFG theory as a practical tool for improving public safety through optimized architectural design.

---

**Implementation Status**: 🔄 [WIP] - Ready for numerical implementation
**Last Updated**: 2025-09-19
**MFG_PDE Package Version**: 0.3.0+
**Computational Complexity**: O(N²T) for N×N grid over T timesteps