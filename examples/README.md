# MFG_PDE Examples

Comprehensive examples and demonstrations organized by complexity and learning path.

## Quick Start

**New to MFG_PDE?** Start here:

```bash
# Complete the 5-tutorial learning path (90 minutes total)
python examples/tutorials/01_hello_mfg.py
python examples/tutorials/02_custom_hamiltonian.py
python examples/tutorials/03_2d_geometry.py
python examples/tutorials/04_particle_methods.py
python examples/tutorials/05_config_system.py
```

**Experienced user?** Jump to:
- `basic/` - Single-concept demos by category
- `advanced/` - Multi-feature applications and advanced methods

---

## Directory Structure

### 📚 [tutorials/](tutorials/) - **START HERE**
5-step learning path for newcomers (10-25 min each):

| Tutorial | Topic | Time | Difficulty |
|----------|-------|------|------------|
| [01_hello_mfg.py](tutorials/01_hello_mfg.py) | Simplest MFG setup | 10 min | ⭐ Beginner |
| [02_custom_hamiltonian.py](tutorials/02_custom_hamiltonian.py) | Custom problem definition | 15 min | ⭐ Beginner |
| [03_2d_geometry.py](tutorials/03_2d_geometry.py) | 2D spatial domains | 20 min | ⭐⭐ Intermediate |
| [04_particle_methods.py](tutorials/04_particle_methods.py) | Particle vs grid solvers | 25 min | ⭐⭐ Intermediate |
| [05_config_system.py](tutorials/05_config_system.py) | ConfigBuilder API | 20 min | ⭐⭐ Intermediate |

[Full Tutorial Guide →](tutorials/README.md)

---

### 🔰 [basic/](basic/) - Single-Concept Demos
Focused examples demonstrating one concept at a time.

#### [core_infrastructure/](basic/core_infrastructure/)
Core MFG_PDE API usage:
- `solve_mfg_demo.py` - Basic `solve_mfg()` usage
- `lq_mfg_demo.py` - Linear-Quadratic MFG problems
- `custom_hamiltonian_derivs_demo.py` - Custom Hamiltonian derivatives

#### [geometry/](basic/geometry/)
Domain and geometry demos:
- `2d_crowd_motion_fdm.py` - 2D crowd motion basics
- `geometry_first_api_demo.py` - Geometry-first API
- `amr_geometry_protocol_demo.py` - Adaptive mesh refinement geometry
- `dual_geometry_simple_example.py` - Dual geometry basics
- `dual_geometry_multiresolution.py` - Multi-resolution grids

#### [solvers/](basic/solvers/)
Solver comparisons:
- `policy_iteration_lq_demo.py` - Policy iteration for LQ problems
- `acceleration_comparison.py` - JAX vs NumPy acceleration

#### [utilities/](basic/utilities/)
Tools and utilities:
- `hdf5_save_load_demo.py` - HDF5 I/O for results
- `common_noise_lq_demo.py` - Stochastic MFG with common noise
- `solver_result_analysis_demo.py` - Result analysis tools
- `utility_demo.py` - General utilities

[Full Basic Guide →](basic/README.md)

---

### 🚀 [advanced/](advanced/) - Multi-Feature Demos
Complex examples combining multiple features and advanced techniques.

#### [comprehensive/](advanced/comprehensive/)
Full-featured demonstrations:
- `mfg_rl_comprehensive_demo.py` - Complete RL workflow
- `all_maze_algorithms_visualization.py` - Maze generation algorithms
- `pydantic_validation_example.py` - Configuration validation

#### [geometry_advanced/](advanced/geometry_advanced/)
Advanced geometric features:
- `arbitrary_nd_geometry_demo.py` - nD geometry (3D, 4D, ...)
- `maze_implicit_geometry_demo.py` - Maze + implicit geometry hybrid
- `triangular_amr_integration.py` - Triangular mesh AMR
- `amr_1d_geometry_demo.py` - 1D adaptive mesh refinement
- `dual_geometry_fem_mesh.py` - FEM mesh integration
- `mfg_2d_geometry_example.py` - 2D geometry examples

#### [solvers_advanced/](advanced/solvers_advanced/)
Advanced numerical methods:
- `semi_lagrangian_validation.py` - Semi-Lagrangian HJB solver
- `semi_lagrangian_2d_enhancements.py` - 2D Semi-Lagrangian
- `weno_family_comparison_demo.py` - WENO schemes comparison
- `hybrid_fp_particle_hjb_fdm_demo.py` - Hybrid particle-FDM methods
- `particle_collocation_dual_mode_demo.py` - Particle collocation
- `jax_acceleration_demo.py` - JAX GPU acceleration

#### [optimization/](advanced/optimization/)
Constrained optimization:
- `primal_dual_constrained_example.py` - Primal-dual methods
- `lagrangian_constrained_optimization.py` - Lagrangian optimization

#### [machine_learning/](advanced/machine_learning/)
Neural network and RL solvers:
- `pinn_mfg_example.py` - Physics-Informed Neural Networks
- `pinn_bayesian_mfg_demo.py` - Bayesian PINN with uncertainty
- `adaptive_pinn_demo.py` - Adaptive training strategies
- `dgm_simple_validation.py` - Deep Galerkin Method
- `neural_operator_mfg_demo.py` - Neural operators for MFG

##### [rl_algorithms/](advanced/machine_learning/rl_algorithms/)
Reinforcement learning methods:
- `rl_intro_comparison.py` - RL algorithms overview
- `nash_q_learning_demo.py` - Nash Q-Learning
- `continuous_action_ddpg_demo.py` - Deep Deterministic Policy Gradient
- `continuous_control_comparison.py` - Continuous control methods

#### [applications/](advanced/applications/)
Real-world applications:
- `traffic_flow_2d_demo.py` - Traffic flow modeling
- `portfolio_optimization_2d_demo.py` - Financial portfolio optimization
- `epidemic_modeling_2d_demo.py` - Epidemic spread dynamics
- `predator_prey_mfg.py` - Predator-prey systems
- `heterogeneous_traffic_multi_pop.py` - Multi-population traffic
- `network_mfg_comparison_example.py` - Network MFG
- `el_farol_bar_demo.py` - El Farol Bar problem
- `santa_fe_bar_demo.py` - Santa Fe Bar problem
- `towel_beach_demo.py` - Beach towel placement game

#### [visualization/](advanced/visualization/)
Advanced visualization:
- `visualize_2d_density_evolution.py` - 2D density visualization

[Full Advanced Guide →](advanced/README.md)

---

### 📓 [notebooks/](notebooks/)
Jupyter notebook demonstrations and interactive tutorials (legacy).

---

### 📂 [outputs/](outputs/)
Generated outputs and visualizations (gitignored, auto-regenerated).

---

## Finding the Right Example

| I want to... | Go to... |
|--------------|----------|
| **Learn MFG_PDE from scratch** | [tutorials/](tutorials/) (start with 01) |
| **See a basic LQ problem** | [basic/core_infrastructure/lq_mfg_demo.py](basic/core_infrastructure/lq_mfg_demo.py) |
| **Work with 2D problems** | [tutorials/03_2d_geometry.py](tutorials/03_2d_geometry.py) or [basic/geometry/2d_crowd_motion_fdm.py](basic/geometry/2d_crowd_motion_fdm.py) |
| **Use particle methods** | [tutorials/04_particle_methods.py](tutorials/04_particle_methods.py) |
| **Configure solvers** | [tutorials/05_config_system.py](tutorials/05_config_system.py) |
| **Try advanced solvers** | [advanced/solvers_advanced/](advanced/solvers_advanced/) |
| **Apply MFG to real problems** | [advanced/applications/](advanced/applications/) |
| **Use neural networks** | [advanced/machine_learning/](advanced/machine_learning/) |
| **Work with complex geometry** | [advanced/geometry_advanced/](advanced/geometry_advanced/) |

---

## Requirements

**Minimal** (core examples):
```bash
pip install numpy scipy matplotlib
```

**Full** (all features):
```bash
pip install numpy scipy matplotlib plotly jupyter
pip install jax jaxlib  # Optional: GPU acceleration
pip install torch       # Optional: Neural network solvers
```

**From source**:
```bash
git clone https://github.com/derrring/MFG_PDE.git
cd MFG_PDE
pip install -e .
```

---

## Running Examples

```bash
# Run any example
python examples/tutorials/01_hello_mfg.py
python examples/basic/core_infrastructure/solve_mfg_demo.py
python examples/advanced/applications/traffic_flow_2d_demo.py

# Outputs saved to examples/outputs/ (gitignored)
ls examples/outputs/tutorials/
```

---

## Organization Philosophy

**Progressive Learning**:
1. **tutorials/** - Structured learning path (beginner → intermediate)
2. **basic/** - Single-concept demos (learn individual features)
3. **advanced/** - Multi-feature demos (combine techniques)

**Hierarchical Structure**:
- Categories group related examples
- Consistent naming: `{topic}_{variant}_demo.py`
- Clear README in each category

**Research vs Infrastructure**:
- Examples in this repo: **Infrastructure demos** (stable, documented)
- Research experiments: Moved to `mfg-research` repository

---

## Example Statistics

| Category | Subdirectories | Files | Total Lines |
|----------|----------------|-------|-------------|
| **tutorials/** | - | 5 | ~1,200 |
| **basic/** | 4 | 14 | ~2,000 |
| **advanced/** | 7 | 35 | ~10,000 |
| **Total** | 11 | 54 | ~13,200 |

---

## Contributing

Found an issue or want to add an example?
1. Check if it belongs in MFG_PDE (infrastructure) or mfg-research (experiments)
2. Place it in the appropriate category
3. Follow naming conventions: `{topic}_{variant}_demo.py`
4. Include docstrings with mathematical background
5. Add entry to category README
6. Open a PR with clear description

---

## Recent Updates (2025-11)

**November 2025** - Examples Redesign:
- ✅ Created 5-tutorial learning path (tutorials 01-05)
- ✅ Reorganized basic/ into 4 categories
- ✅ Reorganized advanced/ into 7 categories
- ✅ Moved research code to mfg-research repository
- ✅ Updated all documentation

**Previous Achievements**:
- 100% strategic typing coverage (366 → 0 MyPy errors)
- WENO5 solver, GPU acceleration, hybrid methods
- High-dimensional MFG capabilities
- Benchmarking integration

---

## Support

- **Documentation**: [../docs/](../docs/)
- **GitHub Issues**: https://github.com/derrring/MFG_PDE/issues
- **Discussions**: https://github.com/derrring/MFG_PDE/discussions

---

*Examples last updated: 2025-11-11 - Examples Redesign Edition*
