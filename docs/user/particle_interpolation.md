# Particle Interpolation Utilities

**Feature**: Convert between grid and particle representations
**Added**: v0.9.0 (Issue #216, Part 1/4)
**Module**: `mfg_pde.utils.numerical.particle_interpolation`

---

## Overview

Particle interpolation utilities enable seamless conversion between:
- **Grid representations** (regular grids for PDE solvers)
- **Particle representations** (scattered points for particle methods)

This is essential for hybrid MFG solvers and visualization of particle simulations.

---

## Quick Start

```python
from mfg_pde.utils.numerical import (
    interpolate_grid_to_particles,
    interpolate_particles_to_grid,
    estimate_kde_bandwidth,
)
import numpy as np

# Grid → Particles
x_grid = np.linspace(0, 1, 51)
u_grid = np.exp(-10 * (x_grid - 0.5)**2)
particles = np.random.uniform(0, 1, 100)
u_particles = interpolate_grid_to_particles(u_grid, grid_bounds=(0, 1), particle_positions=particles)

# Particles → Grid
u_grid_reconstructed = interpolate_particles_to_grid(
    u_particles, particles, grid_shape=(51,), grid_bounds=(0, 1), method="rbf"
)
```

---

## Grid → Particles: `interpolate_grid_to_particles()`

### 1D Example

```python
import numpy as np
from mfg_pde.utils.numerical import interpolate_grid_to_particles

# Define grid values
x = np.linspace(0, 1, 51)
u = np.sin(2 * np.pi * x)

# Interpolate to particles
particles = np.array([0.25, 0.5, 0.75])
u_particles = interpolate_grid_to_particles(
    grid_values=u,
    grid_bounds=(0, 1),
    particle_positions=particles,
    method="linear"  # or "cubic", "nearest"
)

print(u_particles)  # [1.0, 0.0, -1.0] (approximately)
```

### 2D Example

```python
# Create 2D grid
x = y = np.linspace(0, 1, 51)
X, Y = np.meshgrid(x, y, indexing='ij')
u_grid = np.exp(-10 * ((X - 0.5)**2 + (Y - 0.5)**2))

# Random particles
particles = np.random.uniform(0, 1, (100, 2))

# Interpolate
u_particles = interpolate_grid_to_particles(
    grid_values=u_grid,
    grid_bounds=((0, 1), (0, 1)),
    particle_positions=particles,
    method="linear"
)
```

### Methods

- **`linear`**: Fast, C⁰ continuous (default)
- **`cubic`**: Smooth, C¹ continuous (1D/2D only)
- **`nearest`**: Fastest, discontinuous

---

## Particles → Grid: `interpolate_particles_to_grid()`

### Methods

#### RBF (Radial Basis Functions)

Best for: Smooth reconstruction, accuracy
```python
u_grid = interpolate_particles_to_grid(
    particle_values=values,
    particle_positions=particles,
    grid_shape=(51,),
    grid_bounds=(0, 1),
    method="rbf",
    kernel="thin_plate_spline"  # or "gaussian", "multiquadric"
)
```

#### KDE (Kernel Density Estimation)

Best for: Density fields, probability distributions
```python
u_grid = interpolate_particles_to_grid(
    particle_values=values,
    particle_positions=particles,
    grid_shape=(51,),
    grid_bounds=(0, 1),
    method="kde",
    bandwidth=0.1  # or None for auto-selection
)
```

#### Nearest Neighbor

Best for: Speed, categorical data
```python
u_grid = interpolate_particles_to_grid(
    particle_values=values,
    particle_positions=particles,
    grid_shape=(51,),
    grid_bounds=(0, 1),
    method="nearest"
)
```

---

## Bandwidth Estimation: `estimate_kde_bandwidth()`

### Scott's Rule (Default)

```python
particles = np.random.randn(100, 2)
bw = estimate_kde_bandwidth(particles, method="scott")
```

### Silverman's Rule

```python
bw = estimate_kde_bandwidth(particles, method="silverman")
```

Both methods automatically adapt to dimensionality.

---

## Use Cases

### 1. Hybrid Solver Integration

```python
# FP particle solver → HJB grid solver
m_particles = fp_solver.get_distribution()  # Particle positions
m_grid = interpolate_particles_to_grid(
    particle_values=weights,
    particle_positions=m_particles,
    grid_shape=(nx, ny),
    grid_bounds=domain.bounds,
    method="kde"
)

# Use m_grid in HJB solver
u_grid = hjb_solver.solve(m_grid)

# HJB grid → particles for control
u_particles = interpolate_grid_to_particles(
    grid_values=u_grid,
    grid_bounds=domain.bounds,
    particle_positions=m_particles
)
```

### 2. Visualization

```python
# Visualize particle simulation on regular grid
import matplotlib.pyplot as plt

particles, values = run_particle_simulation()
u_grid = interpolate_particles_to_grid(
    values, particles,
    grid_shape=(100, 100),
    grid_bounds=((0, 1), (0, 1)),
    method="rbf"
)

plt.imshow(u_grid.T, origin='lower', extent=[0, 1, 0, 1])
plt.colorbar()
plt.scatter(particles[:, 0], particles[:, 1], c='red', s=1, alpha=0.5)
plt.show()
```

### 3. Initial Condition Generation

```python
# Start with grid-based m0, convert to particles
x = np.linspace(0, 1, 101)
m0_grid = np.exp(-50 * (x - 0.5)**2)

# Sample particles from m0
particles = sample_from_distribution(m0_grid, num_particles=1000)

# Get values at particle positions
m0_particles = interpolate_grid_to_particles(
    m0_grid,
    grid_bounds=(0, 1),
    particle_positions=particles
)
```

---

## Performance

### Grid → Particles

| Dimension | Method | 1000 particles | 10K particles |
|:----------|:-------|---------------:|--------------:|
| 1D | linear | <1 ms | ~5 ms |
| 2D | linear | ~2 ms | ~20 ms |
| 3D | linear | ~10 ms | ~100 ms |

### Particles → Grid

| Method | 1000 particles | 10K particles | 100K particles |
|:-------|---------------:|--------------:|---------------:|
| RBF | ~50 ms | ~500 ms | ~5 s |
| KDE | ~10 ms | ~100 ms | ~1 s |
| Nearest | ~5 ms | ~50 ms | ~500 ms |

*Benchmarks on M1 Mac, numpy backend*

---

## API Reference

### `interpolate_grid_to_particles()`

```python
def interpolate_grid_to_particles(
    grid_values: NDArray[np.floating],
    grid_bounds: tuple[float, float] | tuple[tuple[float, float], ...],
    particle_positions: NDArray[np.floating],
    method: Literal["linear", "cubic", "nearest"] = "linear",
) -> NDArray[np.floating]
```

**Parameters**:
- `grid_values`: Values on regular grid (1D, 2D, or 3D array)
- `grid_bounds`: Grid bounds (1D: `(xmin, xmax)`, 2D: `((xmin, xmax), (ymin, ymax))`)
- `particle_positions`: Particle positions (shape: `(N,)` for 1D, `(N, d)` for d-D)
- `method`: Interpolation method

**Returns**: Values at particle positions (shape: `(N,)`)

---

### `interpolate_particles_to_grid()`

```python
def interpolate_particles_to_grid(
    particle_values: NDArray[np.floating],
    particle_positions: NDArray[np.floating],
    grid_shape: tuple[int, ...],
    grid_bounds: tuple[float, float] | tuple[tuple[float, float], ...],
    method: Literal["rbf", "kde", "nearest"] = "rbf",
    **kwargs,
) -> NDArray[np.floating]
```

**Parameters**:
- `particle_values`: Values at particles (shape: `(N,)`)
- `particle_positions`: Particle positions (shape: `(N,)` or `(N, d)`)
- `grid_shape`: Output grid shape (e.g., `(51,)` for 1D, `(51, 51)` for 2D)
- `grid_bounds`: Grid bounds
- `method`: Interpolation method
- `**kwargs`: Method-specific parameters
  - RBF: `kernel`, `epsilon`
  - KDE: `bandwidth`

**Returns**: Grid values (shape: `grid_shape`)

---

## Examples

See `tests/unit/utils/test_particle_interpolation.py` for comprehensive examples.

---

**Documentation Version**: 1.0
**Last Updated**: 2025-11-03
**MFG_PDE Version**: v0.9.0+
