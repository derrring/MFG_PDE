# Examples Redesign Proposal

**Date**: 2025-11-11
**Status**: PROPOSAL
**Purpose**: Comprehensive cleanup and reorganization of MFG_PDE examples

---

## Executive Summary

This proposal outlines a systematic reorganization of the `examples/` directory to:
1. **Separate infrastructure demos from research experiments**
2. **Create clear learning paths** for new users
3. **Move experimental code** to mfg-research repository
4. **Reduce repository size** and improve maintainability
5. **Align with CLAUDE.md principles** (infrastructure vs research separation)

**Impact**: ~10 files moved to research, ~20 files reorganized, clearer structure for users

---

## Current State Analysis

### Repository Statistics

```
examples/
├── basic/          10 files, 740KB  - Recently updated (Nov 3-10)
├── advanced/       39 files, 4.5MB  - Mix of infrastructure + research
├── archive/         9 files, 116KB  - Old API demos
├── outputs/        13 files, 3.6MB  - ISSUE: Contains Python scripts (should only have outputs)
├── notebooks/       2 files, 1.2MB  - Working demos
└── plugins/         1 file,  24KB   - Custom solver example

TOTAL: 77 Python files, ~10MB
```

### Key Issues Identified

#### 1. **Examples/Outputs Violation**
**Problem**: `examples/outputs/` contains experimental Python scripts instead of just regenerable outputs

```bash
examples/outputs/particle_methods/
├── 2d_crowd_proper_nd.py                      # 533 lines - EXPERIMENTAL SCRIPT
├── particle_methods_comparison_2d.py          # 490 lines - RESEARCH EXPERIMENT
├── 2d_crowd_particle_comparison.py            # 482 lines - RESEARCH EXPERIMENT
├── 2d_crowd_density_evolution.py              # 585 lines - RESEARCH EXPERIMENT
└── 1d_particle_comparison.py                  # Research script
```

**CLAUDE.md states**: `examples/outputs/` should be gitignored regenerable outputs only.

**These are research experiments** with comparative analysis, not infrastructure demos.

#### 2. **Research Code in Advanced Examples**
**Problem**: `examples/advanced/anisotropic_crowd_dynamics_2d/` is a research package (7 files, complex)

```bash
examples/advanced/anisotropic_crowd_dynamics_2d/
├── anisotropic_2d_problem.py       (686 lines)
├── experiment_runner.py            (572 lines)
├── analysis/visualization_tools.py (603 lines)
└── validation/convergence_study.py
```

**Analysis**:
- Multi-file research package with experiment runners
- Should be in mfg-research based on complexity and purpose
- Similar to mfg-research/experiments/anisotropic_crowd_qp

#### 3. **Archive Directory Ambiguity**
**Problem**: `examples/archive/` exists but purpose unclear

```bash
examples/archive/
├── api_demos/          # Old API patterns
├── backend_demos/      # Backend comparisons
└── crowd_dynamics/     # 1D hybrid demos
```

**Question**: Should these be deleted or moved to mfg-research/archives/?

#### 4. **No Clear Learning Path**
**Problem**: No structured onboarding for new users
- Which example to start with?
- What's the progression from beginner to advanced?
- Where are tutorials vs comprehensive demos?

---

## Proposed New Structure

### Philosophy

**MFG_PDE Examples** (Public Infrastructure):
- ✅ Demonstrate core infrastructure features
- ✅ Educational progression (beginner → advanced)
- ✅ Single-concept focused examples
- ✅ Well-documented, production-ready code
- ❌ NO experimental algorithms
- ❌ NO comparative research studies

**mfg-research/experiments** (Private Research):
- ✅ Experimental algorithms and methods
- ✅ Comparative studies and benchmarks
- ✅ Parameter sweeps and ablation studies
- ✅ Research-grade multi-file packages
- ✅ Unpublished novel approaches

### Proposed Directory Structure

```
examples/
├── tutorials/                    # NEW: Structured learning path
│   ├── 01_hello_mfg.py          # 5 min: Solve first MFG
│   ├── 02_custom_hamiltonian.py # 10 min: Define custom problem
│   ├── 03_2d_geometry.py        # 15 min: Work with 2D domains
│   ├── 04_particle_methods.py   # 15 min: Particle-based solvers
│   └── 05_config_system.py      # 10 min: ConfigBuilder API
│
├── basic/                        # Single-concept infrastructure demos
│   ├── core_infrastructure/
│   │   ├── solve_mfg_demo.py              # solve_mfg() function
│   │   ├── lq_mfg_demo.py                 # LQ problem (analytical solution)
│   │   └── custom_hamiltonian_derivs_demo.py
│   │
│   ├── geometry/
│   │   ├── 2d_crowd_motion_fdm.py         # 2D rectangular domain
│   │   ├── geometry_first_api_demo.py     # Geometry-first pattern
│   │   ├── dual_geometry_simple_example.py
│   │   └── dual_geometry_multiresolution.py
│   │
│   ├── solvers/
│   │   ├── policy_iteration_lq_demo.py    # HJB policy iteration
│   │   └── acceleration_comparison.py     # Backend acceleration
│   │
│   └── utilities/
│       ├── utility_demo.py                # Utility functions
│       ├── hdf5_save_load_demo.py         # I/O utilities
│       ├── common_noise_lq_demo.py        # Stochastic MFG
│       └── solver_result_analysis_demo.py
│
├── advanced/                      # Complex multi-feature demos
│   ├── comprehensive/
│   │   ├── mfg_rl_comprehensive_demo.py
│   │   └── all_maze_algorithms_visualization.py
│   │
│   ├── geometry_advanced/
│   │   ├── arbitrary_nd_geometry_demo.py
│   │   ├── maze_implicit_geometry_demo.py
│   │   ├── triangular_amr_integration.py
│   │   ├── amr_1d_geometry_demo.py
│   │   ├── dual_geometry_fem_mesh.py
│   │   └── mfg_2d_geometry_example.py
│   │
│   ├── solvers_advanced/
│   │   ├── semi_lagrangian_validation.py
│   │   ├── semi_lagrangian_2d_enhancements.py
│   │   ├── weno_family_comparison_demo.py
│   │   ├── hybrid_fp_particle_hjb_fdm_demo.py
│   │   └── gfdm_meshfree_demo.py
│   │
│   ├── optimization/
│   │   ├── primal_dual_constrained_example.py
│   │   ├── lagrangian_constrained_optimization.py
│   │   └── variational_solvers/   # If we add more examples
│   │
│   ├── machine_learning/
│   │   ├── pinn_mfg_example.py
│   │   ├── pinn_bayesian_mfg_demo.py
│   │   ├── adaptive_pinn_demo.py
│   │   ├── dgm_simple_validation.py
│   │   ├── neural_operator_mfg_demo.py
│   │   └── rl_algorithms/
│   │       ├── rl_intro_comparison.py
│   │       ├── nash_q_learning_demo.py
│   │       ├── continuous_action_ddpg_demo.py
│   │       └── continuous_control_comparison.py
│   │
│   ├── applications/
│   │   ├── traffic_flow_2d_demo.py
│   │   ├── portfolio_optimization_2d_demo.py
│   │   ├── epidemic_modeling_2d_demo.py
│   │   ├── predator_prey_mfg.py
│   │   ├── heterogeneous_traffic_multi_pop.py
│   │   ├── network_mfg_comparison_example.py
│   │   ├── el_farol_bar_demo.py
│   │   ├── santa_fe_bar_demo.py
│   │   └── towel_beach_demo.py
│   │
│   └── visualization/
│       ├── advanced_visualization_example.py
│       └── visualize_2d_density_evolution.py
│
├── notebooks/                     # Jupyter tutorials and analysis
│   ├── tutorials/
│   │   └── MFG_Getting_Started.ipynb  # Interactive tutorial
│   │
│   └── analysis/
│       └── MFG_Working_Demo.ipynb     # Comprehensive analysis
│
├── plugins/                       # Extension examples
│   └── example_custom_solver.py
│
└── outputs/                       # CLEANED: Only regenerable outputs
    ├── tutorials/                 # Gitignored
    ├── basic/                     # Gitignored
    ├── advanced/                  # Gitignored
    └── reference/                 # Tracked (regression testing baselines)
```

**Total After Cleanup**: ~60 examples (down from 77)

---

## Files To Move/Delete

### Move to mfg-research/experiments/

#### 1. Particle Methods Research (from examples/outputs/)
```bash
# Move entire comparative study
examples/outputs/particle_methods/*.py  → mfg-research/experiments/particle_methods_comparison_2d/
```

**Rationale**: These are research experiments comparing different methods, not infrastructure demos.

**Files**:
- `2d_crowd_proper_nd.py` (533 lines)
- `particle_methods_comparison_2d.py` (490 lines)
- `2d_crowd_particle_comparison.py` (482 lines)
- `2d_crowd_density_evolution.py` (585 lines)
- `1d_particle_comparison.py`
- Associated markdown summaries

#### 2. Anisotropic Crowd Dynamics Research (from examples/advanced/)
```bash
examples/advanced/anisotropic_crowd_dynamics_2d/  → mfg-research/experiments/anisotropic_crowd_dynamics_2d/
```

**Rationale**: Multi-file research package with experiment runners and validation studies.

**Files**: 7 files, ~3000 lines total

#### 3. Decision: Archive Directory
**Option A (Recommended)**: Move to mfg-research/archives/legacy_examples/
```bash
examples/archive/  → mfg-research/archives/legacy_examples/
```

**Option B**: Delete entirely (if truly obsolete)

**Decision needed from user.**

---

## Migration Plan

### Phase 1: Analysis and Backup (CURRENT)
✅ Analyze current structure
✅ Identify research vs infrastructure examples
⬜ Create this proposal document
⬜ Get user approval for plan

### Phase 2: mfg-research Preparation
1. Create target directories in mfg-research:
   ```bash
   mkdir -p mfg-research/experiments/particle_methods_comparison_2d
   mkdir -p mfg-research/experiments/anisotropic_crowd_dynamics_2d
   mkdir -p mfg-research/archives/legacy_examples  # If keeping archive
   ```

2. Ensure mfg-research git is clean

### Phase 3: Move Research Code
1. Move particle methods experiments:
   ```bash
   git mv examples/outputs/particle_methods/*.py \
          ~/OneDrive/code/mfg-research/experiments/particle_methods_comparison_2d/
   ```

2. Move anisotropic research:
   ```bash
   git mv examples/advanced/anisotropic_crowd_dynamics_2d \
          ~/OneDrive/code/mfg-research/experiments/
   ```

3. Handle archive (user decision):
   ```bash
   # Option A: Move to research
   git mv examples/archive ~/OneDrive/code/mfg-research/archives/legacy_examples

   # Option B: Delete
   git rm -r examples/archive
   ```

### Phase 4: Reorganize MFG_PDE Examples
1. Create new structure:
   ```bash
   mkdir -p examples/tutorials
   mkdir -p examples/basic/{core_infrastructure,geometry,solvers,utilities}
   mkdir -p examples/advanced/{comprehensive,geometry_advanced,solvers_advanced,optimization,machine_learning,applications,visualization}
   ```

2. Move files to new locations:
   ```bash
   # Use git mv to preserve history
   git mv examples/basic/solve_mfg_demo.py examples/basic/core_infrastructure/
   git mv examples/basic/2d_crowd_motion_fdm.py examples/basic/geometry/
   # ... (continue for all files)
   ```

3. Update import paths in moved files (if necessary)

### Phase 5: Create Tutorials
1. Create `examples/tutorials/01_hello_mfg.py` - Simplest possible MFG
2. Create `examples/tutorials/02_custom_hamiltonian.py` - Custom problem definition
3. Create `examples/tutorials/03_2d_geometry.py` - 2D domain basics
4. Create `examples/tutorials/04_particle_methods.py` - Particle solver intro
5. Create `examples/tutorials/05_config_system.py` - ConfigBuilder usage

### Phase 6: Update Documentation
1. Create `examples/README.md` with new structure explanation
2. Update `examples/tutorials/README.md` with learning path
3. Update `examples/basic/README.md` for each subdirectory
4. Update `examples/advanced/README.md` for each subdirectory
5. Update `CLAUDE.md` examples section

### Phase 7: Testing and Validation
1. Test all examples execute successfully:
   ```bash
   # Run smoke test on all examples
   find examples/tutorials examples/basic examples/advanced -name "*.py" -exec python {} \;
   ```

2. Update CI if needed

3. Commit changes in both repositories:
   ```bash
   # In MFG_PDE
   git add examples/
   git commit -m "refactor: Reorganize examples into clear infrastructure demos"

   # In mfg-research
   git add experiments/particle_methods_comparison_2d/ experiments/anisotropic_crowd_dynamics_2d/
   git commit -m "feat: Add particle methods and anisotropic research from MFG_PDE"
   ```

---

## Decision Points for User

### 1. Archive Directory Fate
**Question**: What should we do with `examples/archive/`?

**Option A**: Move to mfg-research/archives/legacy_examples
**Option B**: Delete entirely
**Option C**: Keep in MFG_PDE but document as "Historical API demos"

**Recommendation**: Option A (move to research)

### 2. Tutorials Scope
**Question**: How comprehensive should the tutorials/ directory be?

**Option A**: Minimal (5 tutorials as proposed)
**Option B**: Comprehensive (10-15 tutorials covering all major features)

**Recommendation**: Option A initially, expand based on user feedback

### 3. Examples/Outputs Cleanup
**Question**: Should we delete the output files (PNGs, logs) in examples/outputs/?

**Option A**: Keep PNGs as reference outputs (examples/outputs/reference/)
**Option B**: Delete all, regenerate on demand

**Recommendation**: Option A for key visualizations

### 4. Advanced/Particle_methods Directory
**Question**: There's also `examples/advanced/particle_methods/` - keep or merge?

**Need to check**: Does this directory exist and what does it contain?

---

## Benefits of Proposed Redesign

### For New Users
✅ Clear learning path (tutorials 01-05)
✅ Single-concept examples in basic/
✅ Obvious progression to advanced/
✅ Less overwhelming (60 vs 77 examples)

### For Developers
✅ Clear separation: infrastructure vs research
✅ Easier to maintain (less duplication)
✅ Aligned with CLAUDE.md philosophy
✅ Git history preserved with git mv

### For Repository Health
✅ Reduced size (~4MB moved to research)
✅ Faster CI (fewer examples to test)
✅ Clearer purpose for each directory
✅ Better adherence to outputs/ gitignore policy

---

## Risks and Mitigation

### Risk 1: Breaking User Workflows
**Risk**: Users may have bookmarked old example locations
**Mitigation**: Add redirect README files at old locations pointing to new paths

### Risk 2: Import Path Issues
**Risk**: Examples might import from each other, breaking after moves
**Mitigation**: Test all examples after reorganization, update imports

### Risk 3: Loss of Context
**Risk**: Moving files loses surrounding context from original location
**Mitigation**: Use `git mv` to preserve history, add migration notes in commit messages

### Risk 4: Research Repository Sync
**Risk**: mfg-research might have conflicting experiments
**Mitigation**: Check mfg-research first, rename if conflicts exist

---

## Success Criteria

1. ✅ All examples execute successfully after reorganization
2. ✅ Clear README in each subdirectory explains purpose
3. ✅ Tutorials provide learning path for new users
4. ✅ No research experiments remain in MFG_PDE examples/
5. ✅ examples/outputs/ contains only regenerable outputs
6. ✅ Git history preserved for all moved files
7. ✅ Both repositories pass CI after changes

---

## Timeline Estimate

**Phase 1-2** (Analysis + Prep): 30 min - CURRENT
**Phase 3** (Move research code): 30 min
**Phase 4** (Reorganize examples): 1 hour
**Phase 5** (Create tutorials): 2 hours
**Phase 6** (Update docs): 1 hour
**Phase 7** (Testing + validation): 1 hour

**Total Estimated Time**: 6 hours

---

## Next Steps

**Immediate**:
1. User reviews this proposal
2. User decides on decision points (archive fate, tutorial scope, etc.)
3. User approves proceeding with reorganization

**Upon Approval**:
1. Create feature branch: `chore/examples-redesign`
2. Execute Phase 2-7 systematically
3. Create PR with comprehensive testing
4. Update both repositories in coordinated commits

---

## Appendix: File Inventory

### Files Moving to mfg-research
```
examples/outputs/particle_methods/
  - 2d_crowd_proper_nd.py (533 lines)
  - particle_methods_comparison_2d.py (490 lines)
  - 2d_crowd_particle_comparison.py (482 lines)
  - 2d_crowd_density_evolution.py (585 lines)
  - 1d_particle_comparison.py
  - COMPARISON_SUMMARY.md
  - 2D_DENSITY_EVOLUTION_SUMMARY.md
  - PARTICLE_METHODS_EXPERIMENT_PLAN.md
  - 2D_ND_SOLVER_INVESTIGATION_SUMMARY.md
  - RICH_INTEGRATION_STATUS.md

examples/advanced/anisotropic_crowd_dynamics_2d/ (7 files)
  - anisotropic_2d_problem.py (686 lines)
  - experiment_runner.py (572 lines)
  - analysis/visualization_tools.py (603 lines)
  - numerical_demo.py
  - solver_config.py
  - anisotropic_movement_demo.py
  - validation/convergence_study.py

examples/archive/ (9 files, optional)
  - All files in api_demos/, backend_demos/, crowd_dynamics/
```

**Total**: ~15-24 files (depending on archive decision)

### Files Staying in MFG_PDE (Reorganized)
```
examples/basic/ (10 files) - All retained, reorganized into subdirectories
examples/advanced/ (32 files) - Retained after removing anisotropic package
examples/notebooks/ (2 files) - Retained
examples/plugins/ (1 file) - Retained
```

**Total**: ~45 files + 5 new tutorials = ~50 examples

---

**Document Status**: PROPOSAL
**Author**: Claude Code
**Date**: 2025-11-11
**Requires**: User approval before execution
