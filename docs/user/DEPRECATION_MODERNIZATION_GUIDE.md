# Deprecation Modernization Guide

**Auto-generated** by `scripts/generate_deprecation_guide.py`
**Total deprecated items**: 158
**Versions covered**: v0.19.0, v0.18.7, v0.18.6, v0.18.0, v0.17.6, v0.17.12, v0.17.1, v0.17.0, v0.16.11, v0.16.0, v0.14.0, v0.12.0

---

## Overview

This guide documents deprecated usage patterns in MFGArchon and provides
migration paths to modern APIs. All deprecated patterns emit warnings at
runtime and will be removed at the version specified.

To find deprecated usage in your code:
```bash
python -W error::DeprecationWarning -c 'import mfgarchon; ...'
```

---

## Deprecated since v0.19.0

*1 items*

### Functions / Classes

- **`optimal_control_drift()`** — use `use H.optimal_control(x, m, grad_U, t) directly, or let FixedPointIterator handle it automatically` instead (remove by v0.25.0)

---

## Deprecated since v0.18.7

*1 items*

### Parameters

- **`tensor_volatility_field`** in `HJBFDMSolver.solve_hjb_system()` — use `volatility_field (pass (d,d) array or callable returning (d,d))` instead (remove by v0.25.0)

---

## Deprecated since v0.18.6

*5 items*

### Parameters

- **`potential_field`** in `FPFDMSolver.solve_fp_system()` — use `drift_field` instead (remove by v0.25.0)
- **`velocity_field`** in `FPFDMSolver.solve_fp_system()` — use `drift_field` instead (remove by v0.25.0)
- **`drift_field`** in `FPSLAdjointSolver.solve_fp_system()` — use `potential_field` instead (remove by v0.25.0)
- **`drift_field`** in `FPSLJacobianSolver.solve_fp_system()` — use `potential_field` instead (remove by v0.25.0)
- **`drift_field`** in `FPSLSolver.solve_fp_system()` — use `potential_field` instead (remove by v0.25.0)

---

## Deprecated since v0.18.0

*14 items*

### Functions / Classes

- **`GradientComponentOperator()`** — use `PartialDerivOperator` instead (remove by v1.0.0)
- **`_compute_sdf_gradient()`** — use `use mfgarchon.operators.differential.function_gradient() instead` instead (remove by v0.25.0)
- **`_compute_upwind_advection()`** — use `AdvectionOperator` instead (remove by v0.25.0)
- **`mfgarchon.geometry.operators.AdvectionOperator()`** — use `AdvectionOperator` instead (remove by v1.0.0)
- **`mfgarchon.geometry.operators.DivergenceOperator()`** — use `DivergenceOperator` instead (remove by v1.0.0)
- **`mfgarchon.geometry.operators.GeometryProjector()`** — use `GeometryProjector` instead (remove by v1.0.0)
- **`mfgarchon.geometry.operators.GradientComponentOperator()`** — use `PartialDerivOperator` instead (remove by v1.0.0)
- **`mfgarchon.geometry.operators.InterfaceJumpOperator()`** — use `InterfaceJumpOperator` instead (remove by v1.0.0)
- **`mfgarchon.geometry.operators.InterpolationOperator()`** — use `InterpolationOperator` instead (remove by v1.0.0)
- **`mfgarchon.geometry.operators.LaplacianOperator()`** — use `LaplacianOperator` instead (remove by v1.0.0)
- **`mfgarchon.geometry.operators.PartialDerivOperator()`** — use `PartialDerivOperator` instead (remove by v1.0.0)
- **`mfgarchon.geometry.operators.ProjectionRegistry()`** — use `ProjectionRegistry` instead (remove by v1.0.0)
- **`mfgarchon.geometry.operators.schemes.compute_weno5_derivative_1d()`** — use `compute_weno5_derivative_1d` instead (remove by v1.0.0)
- **`mixed_bc()`** — use `Use BoundaryConditions(segments=[...]) directly` instead (remove by v0.25.0)

---

## Deprecated since v0.17.6

*2 items*

### Functions / Classes

- **`FPSLAdjointSolver()`** — use `FPSLSolver` instead (remove by v1.0.0)
- **`__init__()`** — use `FPSLSolver` instead (remove by v0.25.0)

---

## Deprecated since v0.17.12

*2 items*

### Parameters

- **`format_type`** in `Mesh1D.export_mesh()` — use `file_format` instead (remove by v0.25.0)
- **`format_type`** in `Mesh3D.export_mesh()` — use `file_format` instead (remove by v0.25.0)

---

## Deprecated since v0.17.1

*7 items*

### Parameters

- **`Lx`** in `MFGProblem.__init__()` — use `geometry=TensorProductGrid(...)` instead (remove by v0.25.0)
- **`Nx`** in `MFGProblem.__init__()` — use `geometry=TensorProductGrid(...)` instead (remove by v0.25.0)
- **`xmax`** in `MFGProblem.__init__()` — use `geometry=TensorProductGrid(...)` instead (remove by v0.25.0)
- **`xmin`** in `MFGProblem.__init__()` — use `geometry=TensorProductGrid(...)` instead (remove by v0.25.0)

### Functions / Classes

- **`MFGDriftField()`** — use `DriftField` instead (remove by v1.0.0)
- **`_solve_fp_1d()`** — use `solve_fp_system` instead (remove by v0.25.0)
- **`_solve_fp_1d_with_callable()`** — use `solve_fp_system` instead (remove by v0.25.0)

---

## Deprecated since v0.17.0

*117 items*

### Parameters

- **`enable_curriculum`** in `AdaptiveTrainingConfig.__init__()` — use `training_mode` instead (remove by v0.25.0)
- **`enable_multiscale`** in `AdaptiveTrainingConfig.__init__()` — use `training_mode` instead (remove by v0.25.0)
- **`enable_refinement`** in `AdaptiveTrainingConfig.__init__()` — use `training_mode` instead (remove by v0.25.0)
- **`sigma`** in `AdjointConsistentProvider.__init__()` — use `diffusion` instead (remove by v0.25.0)
- **`use_control_variates`** in `DGMConfig.__init__()` — use `variance_reduction` instead (remove by v0.25.0)
- **`use_importance_sampling`** in `DGMConfig.__init__()` — use `variance_reduction` instead (remove by v0.25.0)
- **`use_batch_norm`** in `DeepONetConfig.__init__()` — use `normalization` instead (remove by v0.25.0)
- **`use_layer_norm`** in `DeepONetConfig.__init__()` — use `normalization` instead (remove by v0.25.0)
- **`diffusion_field`** in `FPFDMSolver.solve_fp_system()` — use `volatility_field` instead (remove by v0.25.0)
- **`m_initial_condition`** in `FPFDMSolver.solve_fp_system()` — use `M_initial` instead (remove by v0.25.0)
- **`tensor_diffusion_field`** in `FPFDMSolver.solve_fp_system()` — use `volatility_field` instead (remove by v0.25.0)
- **`volatility_matrix`** in `FPFDMSolver.solve_fp_system()` — use `volatility_field` instead (remove by v0.25.0)
- **`diffusion_field`** in `FPGFDMSolver.solve_fp_system()` — use `volatility_field` instead (remove by v0.25.0)
- **`m_initial_condition`** in `FPNetworkSolver.solve_fp_system()` — use `M_initial` instead (remove by v0.25.0)
- **`external_particles`** in `FPParticleSolver.__init__()` — use `num_particles` instead (remove by v0.25.0)
- **`mode`** in `FPParticleSolver.__init__()` — use `density_mode` instead (remove by v0.25.0)
- **`normalize_kde_output`** in `FPParticleSolver.__init__()` — use `kde_normalization` instead (remove by v0.25.0)
- **`normalize_only_initial`** in `FPParticleSolver.__init__()` — use `kde_normalization` instead (remove by v0.25.0)
- **`diffusion_field`** in `FPParticleSolver.solve_fp_system()` — use `volatility_field` instead (remove by v0.25.0)
- **`m_initial_condition`** in `FPParticleSolver.solve_fp_system()` — use `M_initial` instead (remove by v0.25.0)
- **`diffusion_field`** in `FPSLAdjointSolver.solve_fp_system()` — use `volatility_field` instead (remove by v0.25.0)
- **`m_initial_condition`** in `FPSLAdjointSolver.solve_fp_system()` — use `M_initial` instead (remove by v0.25.0)
- **`diffusion_field`** in `FPSLJacobianSolver.solve_fp_system()` — use `volatility_field` instead (remove by v0.25.0)
- **`m_initial_condition`** in `FPSLJacobianSolver.solve_fp_system()` — use `M_initial` instead (remove by v0.25.0)
- **`diffusion_field`** in `FPSLSolver.solve_fp_system()` — use `volatility_field` instead (remove by v0.25.0)
- **`m_initial_condition`** in `FPSLSolver.solve_fp_system()` — use `M_initial` instead (remove by v0.25.0)
- **`M_density_evolution`** in `HJBFDMSolver.solve_hjb_system()` — use `M_density` instead (remove by v0.25.0)
- **`M_density_evolution_from_FP`** in `HJBFDMSolver.solve_hjb_system()` — use `M_density` instead (remove by v0.25.0)
- **`U_final_condition`** in `HJBFDMSolver.solve_hjb_system()` — use `U_terminal` instead (remove by v0.25.0)
- **`U_final_condition_at_T`** in `HJBFDMSolver.solve_hjb_system()` — use `U_terminal` instead (remove by v0.25.0)
- **`U_from_prev_picard`** in `HJBFDMSolver.solve_hjb_system()` — use `U_coupling_prev` instead (remove by v0.25.0)
- **`bc_values`** in `HJBFDMSolver.solve_hjb_system()` — use `BCValueProvider in BoundaryConditions` instead (remove by v0.25.0)
- **`NiterNewton`** in `HJBGFDMSolver.__init__()` — use `max_newton_iterations` instead (remove by v0.25.0)
- **`l2errBoundNewton`** in `HJBGFDMSolver.__init__()` — use `newton_tolerance` instead (remove by v0.25.0)
- **`M_density_evolution_from_FP`** in `HJBGFDMSolver.solve_hjb_system()` — use `M_density` instead (remove by v0.25.0)
- **`U_final_condition_at_T`** in `HJBGFDMSolver.solve_hjb_system()` — use `U_terminal` instead (remove by v0.25.0)
- **`U_from_prev_picard`** in `HJBGFDMSolver.solve_hjb_system()` — use `U_coupling_prev` instead (remove by v0.25.0)
- **`M_density_evolution`** in `HJBNetworkSolver.solve_hjb_system()` — use `M_density` instead (remove by v0.25.0)
- **`M_density_evolution_from_FP`** in `HJBNetworkSolver.solve_hjb_system()` — use `M_density` instead (remove by v0.25.0)
- **`U_final_condition_at_T`** in `HJBNetworkSolver.solve_hjb_system()` — use `U_terminal` instead (remove by v0.25.0)
- **`U_from_prev_picard`** in `HJBNetworkSolver.solve_hjb_system()` — use `U_coupling_prev` instead (remove by v0.25.0)
- **`M_density_evolution_from_FP`** in `HJBSemiLagrangianSolver.solve_hjb_system()` — use `M_density` instead (remove by v0.25.0)
- **`U_final_condition_at_T`** in `HJBSemiLagrangianSolver.solve_hjb_system()` — use `U_terminal` instead (remove by v0.25.0)
- **`U_from_prev_picard`** in `HJBSemiLagrangianSolver.solve_hjb_system()` — use `U_coupling_prev` instead (remove by v0.25.0)
- **`M_density_evolution_from_FP`** in `HJBWenoSolver.solve_hjb_system()` — use `M_density` instead (remove by v0.25.0)
- **`U_final_condition_at_T`** in `HJBWenoSolver.solve_hjb_system()` — use `U_terminal` instead (remove by v0.25.0)
- **`U_from_prev_picard`** in `HJBWenoSolver.solve_hjb_system()` — use `U_coupling_prev` instead (remove by v0.25.0)
- **`nt`** in `HamiltonianBuilder.domain()` — use `Nt` instead (remove by v0.25.0)
- **`nx`** in `HamiltonianBuilder.domain()` — use `Nx` instead (remove by v0.25.0)
- **`nt`** in `LagrangianBuilder.domain()` — use `Nt` instead (remove by v0.25.0)
- **`nx`** in `LagrangianBuilder.domain()` — use `Nx` instead (remove by v0.25.0)
- **`nt`** in `MFGSystemBuilder.domain()` — use `Nt` instead (remove by v0.25.0)
- **`nx`** in `MFGSystemBuilder.domain()` — use `Nx` instead (remove by v0.25.0)
- **`show_edges`** in `Mesh1D.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`show_quality`** in `Mesh1D.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`show_edges`** in `Mesh2D.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`show_quality`** in `Mesh2D.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`show_edges`** in `Mesh3D.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`show_quality`** in `Mesh3D.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`m_initial_condition`** in `NetworkFPSolver.solve_fp_system()` — use `M_initial` instead (remove by v0.25.0)
- **`M_density_evolution`** in `NetworkHJBSolver.solve_hjb_system()` — use `M_density` instead (remove by v0.25.0)
- **`M_density_evolution_from_FP`** in `NetworkHJBSolver.solve_hjb_system()` — use `M_density` instead (remove by v0.25.0)
- **`U_final_condition_at_T`** in `NetworkHJBSolver.solve_hjb_system()` — use `U_terminal` instead (remove by v0.25.0)
- **`U_from_prev_picard`** in `NetworkHJBSolver.solve_hjb_system()` — use `U_coupling_prev` instead (remove by v0.25.0)
- **`M_density_evolution`** in `NetworkPolicyIterationHJBSolver.solve_hjb_system()` — use `M_density` instead (remove by v0.25.0)
- **`M_density_evolution_from_FP`** in `NetworkPolicyIterationHJBSolver.solve_hjb_system()` — use `M_density` instead (remove by v0.25.0)
- **`U_final_condition_at_T`** in `NetworkPolicyIterationHJBSolver.solve_hjb_system()` — use `U_terminal` instead (remove by v0.25.0)
- **`U_from_prev_picard`** in `NetworkPolicyIterationHJBSolver.solve_hjb_system()` — use `U_coupling_prev` instead (remove by v0.25.0)
- **`use_batch_norm`** in `PINNConfig.__init__()` — use `normalization` instead (remove by v0.25.0)
- **`use_layer_norm`** in `PINNConfig.__init__()` — use `normalization` instead (remove by v0.25.0)
- **`nx`** in `SparseMatrixOptimizer.create_laplacian_3d()` — use `Nx` instead (remove by v0.25.0)
- **`ny`** in `SparseMatrixOptimizer.create_laplacian_3d()` — use `Ny` instead (remove by v0.25.0)
- **`nz`** in `SparseMatrixOptimizer.create_laplacian_3d()` — use `Nz` instead (remove by v0.25.0)
- **`dimension`** in `TensorProductGrid.__init__()` — use `len(bounds) (dimension is inferred from bounds)` instead (remove by v0.25.0)
- **`num_points`** in `TensorProductGrid.__init__()` — use `Nx_points` instead (remove by v0.25.0)
- **`show_edges`** in `UnstructuredMesh.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`show_quality`** in `UnstructuredMesh.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`show_edges`** in `_MeshGeneratorBase.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`show_quality`** in `_MeshGeneratorBase.visualize_mesh()` — use `mode` instead (remove by v0.25.0)
- **`bc_values`** in `_compute_laplacian_1d()` — use `Robin BC segments via AdjointConsistentProvider` instead (remove by v0.25.0)
- **`auto_progress`** in `enhanced_solver_method()` — use `options=SolverMonitoringOptions.PROGRESS` instead (remove by v0.25.0)
- **`monitor_convergence`** in `enhanced_solver_method()` — use `options=SolverMonitoringOptions.CONVERGENCE` instead (remove by v0.25.0)
- **`timing`** in `enhanced_solver_method()` — use `options=SolverMonitoringOptions.TIMING` instead (remove by v0.25.0)
- **`NiterNewton`** in `solve_hjb_system_backward()` — use `max_newton_iterations` instead (remove by v0.25.0)
- **`l2errBoundNewton`** in `solve_hjb_system_backward()` — use `newton_tolerance` instead (remove by v0.25.0)
- **`NiterNewton`** in `solve_hjb_timestep_newton()` — use `max_newton_iterations` instead (remove by v0.25.0)
- **`l2errBoundNewton`** in `solve_hjb_timestep_newton()` — use `newton_tolerance` instead (remove by v0.25.0)

### Functions / Classes

- **`AdaptiveConvergenceWrapper()`** — use `ConvergenceWrapper` instead (remove by v1.0.0)
- **`AdvancedConvergenceMonitor()`** — use `DistributionConvergenceMonitor` instead (remove by v1.0.0)
- **`OscillationDetector()`** — use `_ErrorHistoryTracker` instead (remove by v1.0.0)
- **`StochasticConvergenceMonitor()`** — use `RollingConvergenceMonitor` instead (remove by v1.0.0)
- **`__init__()`** — use `Use TaylorOperator from gfdm_strategies instead: from mfgarchon.alg.numerical.gfdm_components.gfdm_strategies import TaylorOperator` instead (remove by v0.25.0)
- **`_deprecated_xp_zeros()`** — use `Use backend.zeros() instead for device consistency.` instead (remove by v0.25.0)
- **`_init_1d_legacy()`** — use `geometry-first API with TensorProductGrid` instead (remove by v0.25.0)
- **`_init_nd()`** — use `geometry-first API with TensorProductGrid` instead (remove by v0.25.0)
- **`apply_boundary_conditions_1d()`** — use `Use pad_array_with_ghosts() or PreallocatedGhostBuffer instead. See issue #577.` instead (remove by v0.25.0)
- **`apply_boundary_conditions_2d()`** — use `Use pad_array_with_ghosts() or PreallocatedGhostBuffer instead. See issue #577.` instead (remove by v0.25.0)
- **`apply_boundary_conditions_3d()`** — use `Use pad_array_with_ghosts() or PreallocatedGhostBuffer instead. See issue #577.` instead (remove by v0.25.0)
- **`apply_boundary_conditions_nd()`** — use `Use pad_array_with_ghosts() or PreallocatedGhostBuffer instead. See issue #577.` instead (remove by v0.25.0)
- **`compute_adjoint_consistent_bc_values()`** — use `Use mfgarchon.alg.numerical.adjoint.compute_adjoint_consistent_bc_values instead.` instead (remove by v0.25.0)
- **`compute_boundary_log_density_gradient_1d()`** — use `Use mfgarchon.alg.numerical.adjoint.compute_boundary_log_density_gradient_1d instead.` instead (remove by v0.25.0)
- **`create_adjoint_consistent_bc_1d()`** — use `Use mfgarchon.alg.numerical.adjoint.create_adjoint_consistent_bc_1d instead.` instead (remove by v0.25.0)
- **`create_default_monitor()`** — use `Use create_distribution_monitor() instead.` instead (remove by v0.25.0)
- **`create_solver()`** — use `Use the new three-mode solving API instead (Issue #580):
  - Safe Mode: problem.solve(scheme=NumericalScheme.FDM_UPWIND)
  - Expert Mode: problem.solve(hjb_solver=hjb, fp_solver=fp)
  - Auto Mode: problem.solve()
See examples/basic/three_mode_api_demo.py for details.` instead (remove by v0.25.0)
- **`create_stochastic_monitor()`** — use `Use create_rolling_monitor() instead.` instead (remove by v0.25.0)
- **`get_ghost_values_nd()`** — use `Use pad_array_with_ghosts() or PreallocatedGhostBuffer instead. See issue #577.` instead (remove by v0.25.0)
- **`validate_adjoint_capability()`** — use `validate_scheme_pairing` instead (remove by v0.25.0)
- **`wrap_positions()`** — use `Use mfgarchon.geometry.boundary.periodic.wrap_positions instead.` instead (remove by v0.25.0)

### Properties

- **`Lx`** (property) — use `compute from geometry bounds` instead (remove by v0.25.0)
- **`Nx`** (property) — use `problem.geometry.num_spatial_points - 1` instead (remove by v0.25.0)
- **`_grid`** (property) — use `problem.geometry` instead (remove by v0.25.0)
- **`dx`** (property) — use `compute from geometry bounds and num_points` instead (remove by v0.25.0)
- **`num_points`** (property) — use `Use Nx_points instead.` instead (remove by v0.25.0)
- **`xSpace`** (property) — use `problem.geometry.get_spatial_grid()` instead (remove by v0.25.0)
- **`xmax`** (property) — use `problem.geometry.get_bounds()[1][0]` instead (remove by v0.25.0)
- **`xmin`** (property) — use `problem.geometry.get_bounds()[0][0]` instead (remove by v0.25.0)

### Value

- **`HJBGFDMSolver.__init__.qp_optimization_level`** (value) — use `values [smart, tuned, basic] -> [auto, auto, auto]` instead (remove by v0.25.0)

---

## Deprecated since v0.16.11

*1 items*

### Functions / Classes

- **`__init__()`** — use `Use ZeroFluxCalculator instead for J*n = 0 (mass conservation).` instead (remove by v0.25.0)

---

## Deprecated since v0.16.0

*2 items*

### Parameters

- **`NiterNewton`** in `HJBFDMSolver.__init__()` — use `max_newton_iterations` instead (remove by v0.25.0)
- **`l2errBoundNewton`** in `HJBFDMSolver.__init__()` — use `newton_tolerance` instead (remove by v0.25.0)

---

## Deprecated since v0.14.0

*5 items*

### Functions / Classes

- **`dirichlet_bc()`** — use `Use from mfgarchon.geometry import dirichlet_bc; bc = dirichlet_bc(value=..., dimension=1).` instead (remove by v0.25.0)
- **`neumann_bc()`** — use `Use from mfgarchon.geometry import neumann_bc; bc = neumann_bc(value=..., dimension=1).` instead (remove by v0.25.0)
- **`no_flux_bc()`** — use `Use from mfgarchon.geometry import no_flux_bc; bc = no_flux_bc(dimension=1).` instead (remove by v0.25.0)
- **`periodic_bc()`** — use `Use from mfgarchon.geometry import periodic_bc; bc = periodic_bc(dimension=1).` instead (remove by v0.25.0)
- **`robin_bc()`** — use `Use from mfgarchon.geometry import robin_bc; bc = robin_bc(alpha=..., beta=..., dimension=1).` instead (remove by v0.25.0)

---

## Deprecated since v0.12.0

*1 items*

### Functions / Classes

- **`apply_boundary_conditions()`** — use `Use MeshfreeApplicator from mfgarchon.geometry.boundary instead.` instead (remove by v0.25.0)

---

## Migration Help

If you encounter a deprecation warning not listed here,
please file an issue at https://github.com/derrring/MFGArchon/issues
