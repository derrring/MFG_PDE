# MFG_PDE Theory Documentation Index

**Document Type**: Navigation and Reference Guide
**Created**: October 8, 2025
**Status**: Comprehensive index of all theory documentation
**Last Updated**: October 8, 2025

---

## Overview

This index provides a comprehensive guide to all theoretical documentation in the MFG_PDE package. Documents are organized by topic with status indicators, cross-references, and implementation connections.

**Enhancement Status Legend**:
- ✅ **Enhanced**: Comprehensive mathematical rigor with footnoted references
- 📝 **Adequate**: Good coverage, could benefit from additional rigor
- 🔄 **In Progress**: Currently being enhanced
- ⭐ **New**: Recently created comprehensive document

---

## 1. Core Mathematical Foundations

### 1.1 General Foundations

**`mathematical_background.md`** ✅ **Enhanced**
- **Content**: Complete mathematical foundations for MFG theory
- **Sections**: Function spaces, stochastic processes, PDEs, optimal control, measure theory, viscosity solutions, numerical analysis
- **Size**: 391 lines, 17 footnoted references + categorized bibliography
- **Use For**: First reference for mathematical definitions and standard results
- **Related**: All other theory documents reference this

**`NOTATION_STANDARDS.md`** ⭐ **New**
- **Content**: Cross-document notation consistency guide
- **Sections**: Core notation, cross-document mapping, theorem numbering, citation standards, rigor checklist
- **Size**: 370 lines
- **Use For**: Ensuring consistency when writing/reviewing theory documents
- **Related**: Referenced by all enhanced documents

### 1.2 MFG-Specific Theory

**`stochastic_differential_games_theory.md`** ✅ **Enhanced**
- **Content**: N-player games to MFG limit, propagation of chaos, convergence theory
- **Key Results**: Sznitman propagation of chaos with O(1/√N) rate, master equation formulation
- **Size**: 30K, 7 footnoted references
- **Use For**: Rigorous foundations for MFG limit and convergence rates
- **Related**: `stochastic_mfg_common_noise.md`, `mathematical_background.md`
- **Implementation**: All solvers rely on this convergence theory

**`convergence_criteria.md`** ✅ **Enhanced**
- **Content**: Robust convergence criteria for coupled HJB-FPK systems
- **Key Results**: Wasserstein distance criteria, statistical stabilization, fixed-point contraction theorems
- **Size**: 391 lines, 19 footnoted references + additional bibliography
- **Use For**: Designing stopping criteria for MFG solvers
- **Related**: `mathematical_background.md`, `stochastic_differential_games_theory.md`
- **Implementation**: `mfg_pde/utils/convergence.py`

---

## 2. Specialized MFG Formulations

### 2.1 Stochastic MFG

**`stochastic_mfg_common_noise.md`** ✅ **Enhanced**
- **Content**: Common noise MFG with Monte Carlo methods
- **Key Results**: Conditional HJB-FPK system, Monte Carlo error O(1/√K), QMC variance reduction
- **Size**: 21K, 8 footnoted references
- **Use For**: Production implementation of stochastic MFG with external uncertainty
- **Related**: `stochastic_processes_and_functional_calculus.md`, `stochastic_differential_games_theory.md`
- **Implementation**: `mfg_pde/alg/numerical/stochastic/`, `examples/basic/common_noise_lq_demo.py`

**`stochastic_processes_and_functional_calculus.md`** 📝 **Adequate**
- **Content**: Noise processes (OU, CIR, GBM, Jump Diffusion), functional derivative computation
- **Size**: 459 lines, 3 references
- **Use For**: Implementation guide for stochastic MFG components
- **Related**: `stochastic_mfg_common_noise.md`, `mathematical_background.md`
- **Implementation**: `mfg_pde/core/stochastic/`, `mfg_pde/utils/functional_calculus.py`

### 2.2 Network MFG

**`network_mfg_mathematical_formulation.md`** ✅ **Enhanced**
- **Content**: MFG on discrete graph structures
- **Key Results**: Graph operators, Lagrangian formulation, high-order discretization (MUSCL, Godunov), convergence theory
- **Size**: 528 lines, 30 footnoted references + additional bibliography
- **Use For**: Complete reference for network MFG theory and numerical methods
- **Related**: `mathematical_background.md`, `convergence_criteria.md`
- **Implementation**: `mfg_pde/core/network_mfg_problem.py`, `mfg_pde/alg/mfg_solvers/network_mfg_solver.py`

### 2.3 Variational MFG

**`variational_mfg_theory.md`** ⭐ **New** ✅ **Enhanced**
- **Content**: Complete variational formulation of MFG as optimization problem
- **Key Results**: JKO scheme, displacement convexity, Benamou-Brenier formulation, Sinkhorn algorithm, Schrödinger bridge
- **Size**: 25+ footnoted references + additional bibliography
- **Use For**: Gradient-based numerical methods, optimal transport connections, convexity analysis
- **Related**: `information_geometry_mfg.md`, `mathematical_background.md`, `convergence_criteria.md`
- **Implementation**: `mfg_pde/alg/optimization/wasserstein_gradient_flow.py`, `mfg_pde/alg/optimization/sinkhorn_solver.py`

### 2.4 Anisotropic MFG

**`anisotropic_mfg_mathematical_formulation.md`** ✅ **Enhanced**
- **Content**: MFG with direction-dependent kinetic energy (anisotropic Hamiltonians), crowd dynamics, barriers
- **Key Results**: Uniform ellipticity, 2D anisotropy matrix, cross-derivative discretization, Nash equilibrium characterization
- **Size**: 20 footnoted references + additional bibliography
- **Use For**: Crowd evacuation with channeling, traffic flow with lane preferences, pedestrian dynamics
- **Related**: `mathematical_background.md`, `evacuation_mfg_mathematical_formulation.md`, `convergence_criteria.md`
- **Implementation**: `examples/advanced/anisotropic_crowd_dynamics_2d/`

---

## 3. Information Geometry and Geometric Perspectives

**`information_geometry_mfg.md`** ✅ **Enhanced**
- **Content**: Geometric framework for MFG on probability manifolds
- **Key Results**: Fisher-Rao metric, Wasserstein geometry, WFR metric (unbalanced transport), natural gradient, JKO scheme
- **Size**: 46K, 13 footnoted references + categorized bibliography
- **Use For**: Geometric perspective on MFG, gradient flows, optimal transport connections
- **Related**: `variational_mfg_theory.md`, `mathematical_background.md`
- **Implementation**: `mfg_pde/alg/optimization/` (planned Phase 4.6)

**`IG_MFG_SYNTHESIS.md`** ⭐ **New**
- **Content**: Conceptual synthesis of information geometry + MFG paradigm shift
- **Key Insights**: Geometry as modeling axiom, PDEs to geometric trajectories, three canonical geometries
- **Size**: 10K
- **Use For**: Understanding the "why" behind information-geometric MFG formulations
- **Related**: `information_geometry_mfg.md`, `variational_mfg_theory.md`

---

## 4. Application-Specific Formulations

### 4.1 Spatial Competition Problems

**`spatial_competition_mfg.md`** ⭐ **New** ✅ **Enhanced**
- **Content**: Comprehensive spatial competition theory (Towel on Beach, Hotelling, Wardrop)
- **Key Results**: Three equilibrium regimes (single-peak, intermediate, crater), parameter sensitivity, Hotelling stability analysis
- **Size**: 13 footnoted references + additional bibliography
- **Use For**: Spatial competition models, urban planning, location-based services
- **Related**: `mathematical_background.md`, `convergence_criteria.md`
- **Supersedes**: Previous towel_beach documents (removed)

### 4.2 Coordination Problems

**`coordination_games_mfg.md`** ⭐ **New** ✅ **Enhanced**
- **Content**: Comprehensive coordination game theory (El Farol Bar, Arthur's paradox, discrete/continuous formulations)
- **Key Results**: Discrete state MFG formulation, continuous preference evolution, formulation comparison, minority game connections
- **Size**: 22 footnoted references + additional bibliography
- **Use For**: Discrete choice models, coordination games, online platforms, urban planning
- **Related**: `mathematical_background.md`, `convergence_criteria.md`
- **Supersedes**: Previous el_farol/santa_fe documents (removed)

### 4.3 Evacuation and Crowd Dynamics

**`evacuation_mfg_mathematical_formulation.md`** ✅ **Enhanced**
- **Content**: Three mathematical approaches to evacuation (absorption, target payoff, free boundary), mass-decreasing MFG
- **Key Results**: Absorbing boundary analysis, IMEX stability, target payoff convergence, success rate metrics
- **Size**: 15 footnoted references + additional bibliography
- **Use For**: Emergency evacuation planning, crowd management, exit capacity analysis
- **Related**: `anisotropic_mfg_mathematical_formulation.md`, `mathematical_background.md`, `convergence_criteria.md`
- **Implementation**: `examples/advanced/anisotropic_crowd_dynamics_2d/`

---

## 5. Advanced Topics

### 5.1 Sensitivity Analysis

**`MFG_Initial_Distribution_Sensitivity_Analysis.md`** 📝 **Adequate**
- **Content**: Sensitivity of equilibria to initial conditions
- **Size**: 27K
- **Use For**: Understanding robustness of MFG solutions
- **Related**: `convergence_criteria.md`, `mathematical_background.md`

### 5.2 Adaptive Methods

**`adaptive_mesh_refinement_mfg.md`** 📝 **Adequate**
- **Content**: AMR for MFG systems
- **Size**: 14K
- **Use For**: Efficient numerical methods for complex geometries
- **Related**: `mathematical_background.md`

---

## 6. Numerical Methods (Subdirectory)

**Location**: `docs/theory/numerical_methods/`

**`semi_lagrangian_methods.md`** 📝 **Adequate**
- **Content**: Semi-Lagrangian discretization for HJB equations
- **Use For**: High-order accuracy without CFL restrictions

**`lagrangian_formulation.md`** 📝 **Adequate**
- **Content**: Particle-based Lagrangian MFG solvers
- **Use For**: Mesh-free methods, high-dimensional problems

**`adaptive_mesh_refinement.md`** 📝 **Adequate**
- **Content**: AMR strategies for MFG
- **Use For**: Efficient resolution of sharp gradients

**`README.md`**
- **Content**: Overview of numerical methods theory

---

## 7. Reinforcement Learning Formulations (Subdirectory)

**Location**: `docs/theory/reinforcement_learning/`

**Documents**:
- `maddpg_for_mfg_formulation.md` - Multi-agent deep deterministic policy gradient
- `nash_q_learning_formulation.md` - Nash Q-learning for MFG
- `sac_mfg_formulation.md` - Soft actor-critic for MFG
- `td3_mfg_formulation.md` - Twin delayed DDPG
- `continuous_action_mfg_theory.md` - Continuous action spaces
- `heterogeneous_agents_formulation.md` - Non-identical agents
- `action_space_scalability.md` - Scaling to high-dimensional actions
- `maddpg_architecture_design.md` - Network architecture design
- `README.md` - Overview

**Status**: 📝 **Adequate** (application-focused, less mathematical rigor needed)

---

## 8. Cross-Document Connections

### Key Dependency Graph

```
mathematical_background.md (foundation)
    ├── stochastic_differential_games_theory.md
    │   ├── stochastic_mfg_common_noise.md
    │   └── network_mfg_mathematical_formulation.md
    │
    ├── information_geometry_mfg.md
    │   ├── variational_mfg_theory.md
    │   └── IG_MFG_SYNTHESIS.md
    │
    └── convergence_criteria.md
        └── All solver implementations
```

### Thematic Groupings

**Core PDE Theory**:
- `mathematical_background.md`
- `stochastic_differential_games_theory.md`
- `convergence_criteria.md`

**Geometric Perspective**:
- `information_geometry_mfg.md`
- `variational_mfg_theory.md`
- `IG_MFG_SYNTHESIS.md`

**Stochastic Extensions**:
- `stochastic_mfg_common_noise.md`
- `stochastic_processes_and_functional_calculus.md`

**Discrete Formulations**:
- `network_mfg_mathematical_formulation.md`

**Applications**:
- Spatial: `spatial_competition_mfg.md`, `evacuation_mfg_*.md`
- Coordination: `coordination_games_mfg.md`
- Crowds: `anisotropic_mfg_*.md`

---

## 9. Enhancement Summary Statistics

### Documents Enhanced (October 8, 2025)

**Major Enhancements** (9 documents):
1. `mathematical_background.md`: 59 → 391 lines, 17 references
2. `convergence_criteria.md`: 82 → 391 lines, 19 references
3. `network_mfg_mathematical_formulation.md`: 211 → 528 lines, 30 references
4. `stochastic_differential_games_theory.md`: Enhanced, 7 references
5. `information_geometry_mfg.md`: WFR section added, 13 references
6. `stochastic_mfg_common_noise.md`: Monte Carlo theory, 8 references
7. `variational_mfg_theory.md`: **NEW**, 25 references ⭐
8. `evacuation_mfg_mathematical_formulation.md`: 3 approaches, 15 references ✅
9. `anisotropic_mfg_mathematical_formulation.md`: Rigorous formulation, 20 references ✅

**New Documents Created** (5):
1. `NOTATION_STANDARDS.md`: 370 lines ⭐
2. `variational_mfg_theory.md`: Comprehensive variational formulation ⭐
3. `IG_MFG_SYNTHESIS.md`: Conceptual synthesis (10K) ⭐
4. `spatial_competition_mfg.md`: Consolidated spatial competition (13 references) ⭐
5. `coordination_games_mfg.md`: Consolidated coordination games (22 references) ⭐

**Documents Consolidated and Removed** (4 → 2):
- Previous towel_beach documents (2) → `spatial_competition_mfg.md` (removed originals)
- Previous el_farol/santa_fe documents (2) → `coordination_games_mfg.md` (removed originals)

**Total References Added**: 189 footnoted citations + extensive additional bibliographies

**Enhancement Features**:
- ✅ Precise definitions with explicit domains
- ✅ Theorem statements with labeled hypotheses
- ✅ Proof sketches or complete references
- ✅ Footnoted citations with full bibliographic details
- ✅ Implementation connections (file:line references)
- ✅ Cross-document links
- ✅ Consistent notation following NOTATION_STANDARDS.md

---

## 10. Usage Guide

### For New Users

**Start Here**:
1. `mathematical_background.md` - Get foundations
2. `stochastic_differential_games_theory.md` - Understand MFG limit
3. Choose specialization:
   - **PDE Methods**: `convergence_criteria.md`, `variational_mfg_theory.md`
   - **Geometric**: `information_geometry_mfg.md`, `IG_MFG_SYNTHESIS.md`
   - **Stochastic**: `stochastic_mfg_common_noise.md`
   - **Networks**: `network_mfg_mathematical_formulation.md`

### For Developers

**Implementing New Solvers**:
1. Check `mathematical_background.md` for standard definitions
2. Reference `convergence_criteria.md` for stopping criteria
3. Follow `NOTATION_STANDARDS.md` for consistency
4. Link to appropriate specialized formulation

**Writing New Theory Docs**:
1. Use `NOTATION_STANDARDS.md` as template
2. Include footnoted references
3. Add implementation connections
4. Update this index

### For Researchers

**By Application**:
- **Optimal Transport**: `variational_mfg_theory.md`, `information_geometry_mfg.md`
- **Numerical Analysis**: `convergence_criteria.md`, `numerical_methods/`
- **Stochastic Analysis**: `stochastic_mfg_common_noise.md`, `stochastic_differential_games_theory.md`
- **Graph Theory**: `network_mfg_mathematical_formulation.md`

**By Mathematical Tool**:
- **Viscosity Solutions**: `mathematical_background.md` §6
- **Wasserstein Distance**: `variational_mfg_theory.md` §3, `information_geometry_mfg.md` §3
- **Functional Derivatives**: `stochastic_processes_and_functional_calculus.md`
- **Monte Carlo**: `stochastic_mfg_common_noise.md` §3

---

## 11. Future Enhancements

### High Priority

**Enhancement**:
- Add references to application-specific docs (evacuation, anisotropic, etc.)
- Enhance numerical methods subdirectory with rigorous error analysis
- Add master equation theory document

### Medium Priority

**New Documents**:
- Master equation theory (extending `stochastic_processes_and_functional_calculus.md`)
- Multi-population MFG
- Finite state space MFG (Markov chains)

**Enhancements**:
- Add more examples to each theory document
- Cross-validation of all numerical examples against theory

---

## 12. Maintenance Guidelines

### Document Status Review

**Quarterly**:
- Check all implementation links are valid
- Update references if new literature emerges
- Verify cross-document consistency

**When Adding New Features**:
- Update relevant theory document
- Add entry to this index
- Check NOTATION_STANDARDS.md compliance

### Reference Management

**Standards**:
- Use footnote format for all citations
- Include full bibliographic details
- Add categorized additional references at end
- See `NOTATION_STANDARDS.md` §3.3 for format

---

**Document Status**: Comprehensive index of all MFG_PDE theory documentation
**Last Updated**: October 8, 2025
**Maintained By**: Primary maintainer with AI assistance
**Next Review**: January 2026
