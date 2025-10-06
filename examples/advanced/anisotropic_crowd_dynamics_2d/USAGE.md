# Usage Guide: 2D Anisotropic Crowd Dynamics Experiment

This guide provides practical instructions for running the 2D anisotropic crowd dynamics experiment.

## Quick Start

### Running a Single Experiment

```bash
# Basic experiment with default parameters
python experiment_runner.py

# Experiment with specific barrier configuration
python experiment_runner.py --barrier-config anisotropy_aligned

# High-resolution experiment
python experiment_runner.py --grid-size 128 --barrier-config corridor_system
```

### Running Comparative Analysis

```bash
# Compare all barrier configurations
python experiment_runner.py --mode comparative

# Compare with custom output directory
python experiment_runner.py --mode comparative --output-dir my_results/
```

### Running Convergence Study

```bash
# Grid convergence analysis
python experiment_runner.py --mode convergence

# Quick convergence check (smaller grids)
python experiment_runner.py --mode convergence --grid-size 32
```

### Complete Experimental Suite

```bash
# Run all experiments (single + comparative + convergence)
python experiment_runner.py --mode all --output-dir complete_study/
```

## Command Line Options

### Basic Parameters

- `--mode`: Experiment type
  - `single`: Single configuration experiment
  - `comparative`: Compare multiple barrier configurations
  - `convergence`: Grid and temporal convergence study
  - `all`: Complete experimental suite

- `--barrier-config`: Barrier configuration type
  - `none`: No barriers (baseline)
  - `central_obstacle`: Circular barrier in center
  - `anisotropy_aligned`: Diagonal barriers aligned with anisotropy
  - `corridor_system`: Rectangular columns creating bottlenecks

### Physical Parameters

- `--gamma`: Density-velocity coupling strength (default: 0.1)
- `--sigma`: Diffusion coefficient (default: 0.01)
- `--rho-amplitude`: Anisotropy strength (default: 0.5, must be < 1.0)

### Computational Parameters

- `--grid-size`: Spatial grid resolution NxN (default: 64)
- `--output-dir`: Output directory for results (default: "results/")
- `--no-validation`: Skip validation checks for faster execution

## Example Workflows

### 1. Parameter Sensitivity Study

```bash
# Test different anisotropy strengths
python experiment_runner.py --rho-amplitude 0.3 --output-dir rho_03/
python experiment_runner.py --rho-amplitude 0.5 --output-dir rho_05/
python experiment_runner.py --rho-amplitude 0.7 --output-dir rho_07/

# Test different congestion coupling
python experiment_runner.py --gamma 0.05 --output-dir gamma_005/
python experiment_runner.py --gamma 0.1 --output-dir gamma_01/
python experiment_runner.py --gamma 0.2 --output-dir gamma_02/
```

### 2. Resolution Study

```bash
# Compare different grid resolutions
python experiment_runner.py --grid-size 32 --output-dir grid_32/
python experiment_runner.py --grid-size 64 --output-dir grid_64/
python experiment_runner.py --grid-size 128 --output-dir grid_128/
```

### 3. Barrier Effectiveness Analysis

```bash
# Compare evacuation efficiency across barrier types
python experiment_runner.py --mode comparative --output-dir barrier_study/

# Focus on anisotropy-aligned barriers with different parameters
python experiment_runner.py --barrier-config anisotropy_aligned --rho-amplitude 0.3 --output-dir aligned_weak/
python experiment_runner.py --barrier-config anisotropy_aligned --rho-amplitude 0.7 --output-dir aligned_strong/
```

## Programmatic Usage

### Basic Python Usage

```python
from anisotropic_crowd_dynamics import create_anisotropic_problem
from solver_config import create_experiment_solver
from analysis.visualization_tools import create_visualization_suite

# Create and solve problem
problem = create_anisotropic_problem(
    barrier_config='anisotropy_aligned',
    gamma=0.1,
    sigma=0.01,
    rho_amplitude=0.5
)

solver = create_experiment_solver(problem)
solution = solver.solve(problem)

# Create visualizations
visualizer = create_visualization_suite(problem, solution, "my_results/")
```

### Advanced Configuration

```python
from anisotropic_crowd_dynamics import AnisotropicCrowdDynamics, LinearBarrier
from solver_config import create_anisotropic_solver_config

# Custom problem with specific barriers
class CustomProblem(AnisotropicCrowdDynamics):
    def _setup_barriers(self, configuration):
        # Custom barrier placement
        return [
            LinearBarrier(start=(0.2, 0.2), end=(0.8, 0.8)),
            LinearBarrier(start=(0.2, 0.8), end=(0.8, 0.2))
        ]

# Custom solver configuration
solver_config = create_anisotropic_solver_config(
    has_barriers=True,
    grid_size=(128, 128),
    cfl_safety_factor=0.1  # More conservative
)

problem = CustomProblem(barrier_configuration='custom')
solver = create_experiment_solver(problem, solver_config)
```

### Batch Processing

```python
import itertools
from experiment_runner import AnisotropicExperiment

# Parameter sweep
barriers = ['none', 'central_obstacle', 'anisotropy_aligned']
gammas = [0.05, 0.1, 0.2]
rho_amplitudes = [0.3, 0.5, 0.7]

results = {}

for barrier, gamma, rho in itertools.product(barriers, gammas, rho_amplitudes):
    experiment = AnisotropicExperiment(
        barrier_config=barrier,
        gamma=gamma,
        rho_amplitude=rho,
        output_dir=f"batch_results/{barrier}_g{gamma}_r{rho}/"
    )

    result = experiment.run_single_configuration(validate=False)
    results[(barrier, gamma, rho)] = result

# Analyze batch results
for key, result in results.items():
    barrier, gamma, rho = key
    metrics = result['metrics']
    print(f"{barrier}, γ={gamma}, ρ={rho}: T_90={metrics.get('t_90_percent', 'N/A')}")
```

## Output Structure

After running experiments, the output directory contains:

### Single Experiment Output

```
results/
├── experiment_summary.json          # Numerical results and metrics
├── density_evolution.html           # Interactive density animation
├── velocity_field.png              # Velocity field visualization
├── anisotropy_analysis.png          # Anisotropy effects analysis
├── barrier_influence.png           # Barrier effects (if applicable)
└── metrics_dashboard.html          # Interactive metrics dashboard
```

### Comparative Study Output

```
results/
├── comparative_analysis.json       # Quantitative comparison
├── comparative_report.md           # Summary report
├── none/                           # Results for each configuration
├── central_obstacle/
├── anisotropy_aligned/
└── corridor_system/
```

### Convergence Study Output

```
results/convergence/
├── convergence_results.json        # Numerical convergence data
└── convergence_report.md          # Analysis summary
```

## Performance Optimization

### For Large Grids (≥128×128)

```bash
# Use performance-optimized settings
python experiment_runner.py --grid-size 128 --no-validation
```

```python
# In code: use high-performance configuration
from solver_config import create_performance_optimized_config

config = create_performance_optimized_config(
    grid_size=(128, 128),
    target_accuracy='fast'  # Prioritize speed over accuracy
)
```

### For High Accuracy

```bash
# Slower but more accurate
python experiment_runner.py --grid-size 64 --sigma 0.005
```

```python
# In code: high-accuracy configuration
config = create_performance_optimized_config(
    grid_size=(64, 64),
    target_accuracy='high_accuracy'
)
```

## Troubleshooting

### Common Issues

1. **Memory errors with large grids**
   - Reduce grid size: `--grid-size 32`
   - Skip validation: `--no-validation`

2. **Slow convergence**
   - Increase diffusion: `--sigma 0.02`
   - Reduce anisotropy: `--rho-amplitude 0.3`

3. **Numerical instabilities**
   - Check that `|rho_amplitude| < 1.0`
   - Reduce time step via solver configuration

4. **Visualization errors**
   - Install plotly: `pip install plotly`
   - Check output directory permissions

### Performance Tuning

```python
# For debugging: reduce problem size
problem = create_anisotropic_problem(
    gamma=0.1,
    sigma=0.02,  # Higher diffusion for stability
    rho_amplitude=0.3  # Weaker anisotropy
)
problem.grid_size = (32, 32)  # Smaller grid
problem.time_horizon = 0.5   # Shorter simulation
```

## Advanced Features

### Custom Barrier Configurations

```python
from anisotropic_crowd_dynamics import CircularBarrier, LinearBarrier

# Create custom barriers
custom_barriers = [
    CircularBarrier(center=(0.3, 0.7), radius=0.1),
    LinearBarrier(start=(0.5, 0.0), end=(0.5, 1.0), thickness=0.05),
    LinearBarrier(start=(0.0, 0.5), end=(1.0, 0.5), thickness=0.05)
]

# Use in problem
problem = AnisotropicCrowdDynamics(barrier_configuration='custom')
problem.barriers = custom_barriers
```

### Custom Anisotropy Functions

```python
class CustomAnisotropicProblem(AnisotropicCrowdDynamics):
    def compute_anisotropy(self, x):
        """Custom anisotropy pattern."""
        x1, x2 = x[:, 0], x[:, 1]
        # Radial anisotropy pattern
        r = np.sqrt((x1 - 0.5)**2 + (x2 - 0.5)**2)
        return self.rho_amplitude * np.sin(4 * np.pi * r)
```

This usage guide provides comprehensive instructions for running and customizing the 2D anisotropic crowd dynamics experiment. For detailed mathematical and implementation information, see the main README.md.
