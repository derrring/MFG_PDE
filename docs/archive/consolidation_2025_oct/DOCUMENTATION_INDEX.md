# MFG_PDE Complete Documentation Index

**Last Updated**: October 8, 2025 (Development section updated after cleanup)
**Status**: Comprehensive navigation guide for all documentation
**Purpose**: Central hub for finding any documentation in the repository

---

## Quick Navigation

- 📖 [Theory Documentation](#1-theory-documentation) - Mathematical foundations and formulations
- 💻 [User Guides](#2-user-guides) - Getting started and usage
- 🔧 [Development Documentation](#3-development-documentation) - Architecture and design
- 📊 [Examples](#4-examples-and-tutorials) - Working code demonstrations
- 🚀 [API Reference](#5-api-reference) - Package API documentation
- 🎯 [Advanced Topics](#6-advanced-topics) - Specialized features

---

## 1. Theory Documentation

**Location**: `docs/theory/`
**Index**: See [`docs/theory/THEORY_DOCUMENTATION_INDEX.md`](theory/THEORY_DOCUMENTATION_INDEX.md)

### Quick Access to Enhanced Documents ✅

**Core Foundations**:
- [`mathematical_background.md`](theory/mathematical_background.md) - Complete mathematical foundations (391 lines, 17 refs)
- [`NOTATION_STANDARDS.md`](theory/NOTATION_STANDARDS.md) - Cross-document notation guide (370 lines)
- [`convergence_criteria.md`](theory/convergence_criteria.md) - Solver convergence theory (391 lines, 19 refs)

**MFG Formulations**:
- [`stochastic_differential_games_theory.md`](theory/stochastic_differential_games_theory.md) - N-player to MFG limit (30K, 7 refs)
- [`network_mfg_mathematical_formulation.md`](theory/network_mfg_mathematical_formulation.md) - Graph-based MFG (528 lines, 30 refs)
- [`variational_mfg_theory.md`](theory/variational_mfg_theory.md) ⭐ NEW - Optimization formulation (25 refs)

**Geometric Perspectives**:
- [`information_geometry_mfg.md`](theory/information_geometry_mfg.md) - Geometric MFG framework (46K, 13 refs)
- [`IG_MFG_SYNTHESIS.md`](theory/IG_MFG_SYNTHESIS.md) ⭐ NEW - Conceptual synthesis (10K)

**Stochastic MFG**:
- [`stochastic_mfg_common_noise.md`](theory/stochastic_mfg_common_noise.md) - Common noise formulation (21K, 8 refs)
- [`stochastic_processes_and_functional_calculus.md`](theory/stochastic_processes_and_functional_calculus.md) - Implementation guide

**Applications**:
- [`spatial_competition_mfg.md`](theory/spatial_competition_mfg.md) ⭐ NEW - Spatial competition (Hotelling, Wardrop)
- [`coordination_games_mfg.md`](theory/coordination_games_mfg.md) ⭐ NEW - Coordination games (El Farol)
- [`evacuation_mfg_mathematical_formulation.md`](theory/evacuation_mfg_mathematical_formulation.md) ✅ - Crowd evacuation
- [`anisotropic_mfg_mathematical_formulation.md`](theory/anisotropic_mfg_mathematical_formulation.md) ✅ - Direction-dependent dynamics

**Total**: 18 main theory documents + numerical_methods/ + reinforcement_learning/ subdirectories

---

## 2. User Guides

### Getting Started

**Installation and Setup**:
- [`../README.md`](../README.md) - Main repository README with quick start
- Installation: `pip install -e .` from repository root

**First Steps**:
1. **Basic Example**: `examples/basic/particle_collocation_mfg_example.py`
2. **Theory Foundation**: `docs/theory/mathematical_background.md`
3. **User Documentation**: (planned) `docs/user/getting_started.md`

### Usage Patterns

**Problem Definition**:
- See `examples/basic/` for simple problem setups
- See theory documents for mathematical formulations
- API reference: `mfg_pde.core.mfg_problem.MFGProblem`

**Solver Selection**:
- Quick reference: `examples/advanced/factory_patterns_example.py`
- Theory: `docs/theory/convergence_criteria.md`
- Development: `docs/development/SOLVER_HIERARCHY_STRATEGY.md`

**Configuration Management**:
- Examples: `examples/advanced/configuration_demos/`
- Design: `docs/advanced/design/` (OmegaConf integration)
- API: Progressive disclosure patterns

---

## 3. Development Documentation

**Location**: `docs/development/`
**Last Updated**: October 8, 2025

### Organization Structure

**Active Development** (`docs/development/` - 53 files):
- Strategic planning and current work
- [`STRATEGIC_DEVELOPMENT_ROADMAP_2026.md`](development/STRATEGIC_DEVELOPMENT_ROADMAP_2026.md) - Main roadmap (Phase 2 ✅, Phase 4 planned)
- Architecture and design documents

**Completed Work** (`docs/development/completed/` - 35 files):
- Session summaries and historical records
- [`[COMPLETED]_SESSION_2025-10-08_SUMMARY.md`](development/completed/[COMPLETED]_SESSION_2025-10-08_SUMMARY.md) - Latest session (HDF5, health checks, fixes)
- Phase completion summaries
- Issue resolutions

**Analysis** (`docs/development/analysis/` - 15 files):
- Technical analyses and investigations
- Mass conservation analyses
- Solver API reviews
- Performance bottleneck investigations

**Sessions** (`docs/development/sessions/` - 3 files):
- Historical session notes
- Development progression tracking

### Key Active Documents

**Strategic Planning**:
- [`STRATEGIC_DEVELOPMENT_ROADMAP_2026.md`](development/STRATEGIC_DEVELOPMENT_ROADMAP_2026.md) (v1.7, Oct 8 2025)
- [`CONSISTENCY_GUIDE.md`](development/CONSISTENCY_GUIDE.md) - Coding standards
- [`RUFF_VERSION_MANAGEMENT.md`](development/RUFF_VERSION_MANAGEMENT.md) - Tooling standards

**Architecture**:
- [`PROGRESSIVE_DISCLOSURE_API_DESIGN.md`](development/PROGRESSIVE_DISCLOSURE_API_DESIGN.md) - API design philosophy
- [`TWO_LEVEL_API_INTEGRITY_CHECK.md`](development/TWO_LEVEL_API_INTEGRITY_CHECK.md) - API consistency
- [`MULTI_POPULATION_ARCHITECTURE_DESIGN.md`](development/MULTI_POPULATION_ARCHITECTURE_DESIGN.md) - Multi-population systems

**Current Work** (as of Oct 8, 2025):
- ✅ HDF5 support (Issue #122) - COMPLETED
- ⏳ Config system unification (Issue #113) - PLANNED
- ⏳ Performance benchmarking (Issue #115) - PLANNED

### Recent Completed Work (October 2025)

**Session 2025-10-08** - Comprehensive session including:
- File format evaluation (HDF5 vs Parquet vs Zarr) → HDF5 selected as primary
- Issue scope management (closed 4 over-development issues)
- Package health validation (96.3% test pass rate)
- Critical bug fixes (FixedPointIterator, test imports)
- **HDF5 implementation** - Full support for solver data persistence
  - Core utilities (save_solution, load_solution, save_checkpoint, load_checkpoint)
  - SolverResult integration (save_hdf5(), load_hdf5())
  - 14 comprehensive tests (all passing ✅)
  - Working example with visualization

**Key Achievements**:
- Solver unification completed
- Dead code analysis performed
- Test suite improved from 95.4% → 96.3%

### Planning Documents

**Location**: `docs/planning/roadmaps/` (9 files)

**Active Plans**:
- [`PHASE_2.2_STOCHASTIC_MFG_PLAN.md`](planning/roadmaps/PHASE_2.2_STOCHASTIC_MFG_PLAN.md) - Stochastic MFG extensions (Q1 2026)
- [`PHASE_3_5_PLANNING.md`](planning/roadmaps/PHASE_3_5_PLANNING.md) - Phase 3.5 planning
- [`MASTER_EQUATION_IMPLEMENTATION_PLAN.md`](planning/roadmaps/MASTER_EQUATION_IMPLEMENTATION_PLAN.md) - Master equation work
- [`POST_V1_4_0_ROADMAP.md`](planning/roadmaps/POST_V1_4_0_ROADMAP.md) - Post v1.4.0 development

### How to Navigate Development Docs

1. **For current work**: Check [`STRATEGIC_DEVELOPMENT_ROADMAP_2026.md`](development/STRATEGIC_DEVELOPMENT_ROADMAP_2026.md)
2. **For recent changes**: See [`completed/[COMPLETED]_SESSION_*.md`](development/completed/)
3. **For technical analysis**: Browse [`analysis/`](development/analysis/)
4. **For planning**: See [`../planning/roadmaps/`](planning/roadmaps/)

### Documentation Stats (After Oct 8 Cleanup)

- **Total documentation**: 238 markdown files
- **Active development**: 53 files (down from 62)
- **Completed work**: 35 files
- **Analysis**: 15 files
- **Planning**: 9 files

**Recent cleanup** (Oct 8, 2025):
- ✅ Consolidated 9 session docs → 1 comprehensive summary
- ✅ Moved 9 completed analyses to appropriate directories
- ✅ Moved 4 planning docs to docs/planning/roadmaps/
- ✅ Removed 1 obsolete planning document
- **Result**: Better organized, easier to navigate

---

## 4. Examples and Tutorials

**Location**: `examples/`
**Index**: See [`examples/README.md`](../examples/README.md)

### Quick Reference by Use Case

**Learning the Basics**:
- `examples/basic/particle_collocation_mfg_example.py` - First example to try
- `examples/basic/simple_logging_demo.py` - Logging system introduction
- `examples/notebooks/working_demo/` - Interactive Jupyter notebook

**Advanced Features**:
- `examples/advanced/hybrid_fp_particle_hjb_fdm_demo.py` - Hybrid methods
- `examples/advanced/highdim_mfg_capabilities/` - High-dimensional MFG suite
- `examples/advanced/anisotropic_crowd_dynamics_2d/` - Complex 2D dynamics

**Research Workflows**:
- `examples/advanced/interactive_research_notebook_example.py` - Automation
- `examples/advanced/advanced_visualization_example.py` - Analysis plots
- `examples/notebooks/` - Notebook generation tools

**Configuration Examples**:
- `examples/advanced/configuration_demos/` - Professional config patterns
- `examples/advanced/factory_patterns_example.py` - Solver factories

### Example Documentation

Each subdirectory contains:
- **README.md**: Overview and usage instructions
- **USAGE.md** (where applicable): Detailed usage guide
- Inline code documentation with mathematical formulas

**Example Categories**:
- Basic (⭐): 5 examples - single concepts
- Advanced (⭐⭐⭐): 12+ examples - complex workflows
- Notebooks (⭐⭐): 4+ examples - interactive analysis

---

## 5. API Reference

### Core Modules

**Problem Definition**:
- `mfg_pde.core.mfg_problem.MFGProblem` - Base problem class
- `mfg_pde.core.network_mfg_problem.NetworkMFGProblem` - Graph-based
- `mfg_pde.core.highdim_mfg_problem` - High-dimensional problems

**Geometry**:
- `mfg_pde.geometry.Domain1D/2D/3D` - Spatial domains
- `mfg_pde.geometry.BoundaryConditions` - Boundary handling
- `mfg_pde.geometry.MeshRefinement` - AMR capabilities

**Solvers**:
- `mfg_pde.alg.numerical.hjb_solvers` - HJB equation solvers
- `mfg_pde.alg.numerical.fp_solvers` - Fokker-Planck solvers
- `mfg_pde.alg.mfg_solvers` - Coupled MFG solvers
- `mfg_pde.alg.optimization` - Variational methods (JKO, Sinkhorn)

**Utilities**:
- `mfg_pde.utils.convergence` - Convergence monitoring
- `mfg_pde.utils.logging` - Structured logging
- `mfg_pde.utils.visualization` - Plotting utilities

**Stochastic**:
- `mfg_pde.core.stochastic.noise_processes` - OU, CIR, GBM, Jump processes
- `mfg_pde.utils.functional_calculus` - Functional derivatives

### API Documentation Status

**Current**: Comprehensive docstrings in code (Google/NumPy style)
**Planned**: Sphinx-generated HTML documentation (Phase 4.x)
**Alternative**: Use IDE autocomplete and docstring tooltips

---

## 6. Advanced Topics

**Location**: `docs/advanced/`

### Adaptive Mesh Refinement

**Theory**:
- `docs/theory/adaptive_mesh_refinement_mfg.md` - AMR for MFG systems
- `docs/advanced/adaptive_mesh_refinement.md` - Implementation guide

**Design**:
- `docs/advanced/design/geometry_amr_integration.md` - Architecture
- `docs/advanced/design/amr_performance.md` - Performance analysis
- `docs/advanced/amr_mesh_types_analysis.md` - Mesh type comparison

**Examples**:
- Examples integrated in high-dimensional capabilities

### Benchmarking

**Framework**:
- `docs/advanced/design/benchmarking.md` - Benchmarking system design
- `benchmarks/` directory - Performance tests

**Examples**:
- `examples/advanced/highdim_mfg_capabilities/benchmark_performance.py`

### Geometry System

**Design**:
- `docs/advanced/design/geometry_system.md` - System architecture
- Integration with Gmsh, Meshio, PyVista

**Usage**:
- `examples/advanced/highdim_mfg_capabilities/demo_3d_*.py`

### Hybrid Methods

**Design**:
- `docs/advanced/design/hybrid_maze_generation.md` - Maze generation
- Particle-FDM hybrid approaches

**Examples**:
- `examples/advanced/hybrid_fp_particle_hjb_fdm_demo.py`
- `examples/advanced/quick_hybrid_demo.py`

---

## 7. Numerical Methods (Specialized)

**Location**: `docs/theory/numerical_methods/`

**Methods Documented**:
- `semi_lagrangian_methods.md` - Semi-Lagrangian discretization
- `lagrangian_formulation.md` - Particle-based methods
- `adaptive_mesh_refinement.md` - AMR strategies
- `README.md` - Overview of numerical methods

**Implementation**:
- Core methods: `mfg_pde/alg/numerical/`
- Examples: `examples/basic/semi_lagrangian_example.py`

---

## 8. Reinforcement Learning for MFG

**Location**: `docs/theory/reinforcement_learning/`

**Formulations**:
- `maddpg_for_mfg_formulation.md` - Multi-agent DDPG
- `nash_q_learning_formulation.md` - Nash Q-learning
- `sac_mfg_formulation.md` - Soft actor-critic
- `td3_mfg_formulation.md` - Twin delayed DDPG
- `continuous_action_mfg_theory.md` - Continuous actions
- `heterogeneous_agents_formulation.md` - Non-identical agents
- `action_space_scalability.md` - High-dimensional actions
- `maddpg_architecture_design.md` - Network architecture

**Implementation**:
- `mfg_pde/alg/reinforcement/` - RL solvers
- Examples: (planned) `examples/reinforcement_learning/`

---

## 9. Documentation by User Type

### For Students and Researchers

**Start Here**:
1. `docs/theory/mathematical_background.md` - Foundations
2. `docs/theory/stochastic_differential_games_theory.md` - MFG limit
3. `examples/basic/` - Simple examples
4. Choose specialization from `docs/theory/THEORY_DOCUMENTATION_INDEX.md`

**Research Tools**:
- Theory documents with full references
- Examples with mathematical formulas
- Notebook generation for analysis

### For Software Developers

**Start Here**:
1. `README.md` - Package overview
2. `examples/basic/` - API introduction
3. `docs/development/ARCHITECTURAL_CHANGES.md` - System design
4. `docs/development/PROGRESSIVE_DISCLOSURE_API_DESIGN.md` - API philosophy

**Development Guides**:
- Solver hierarchy and extension
- Configuration management
- Testing and validation

### For Application Engineers

**Start Here**:
1. `examples/README.md` - Example overview
2. Find relevant application example
3. Adapt to your problem
4. Reference theory docs as needed

**Application Areas**:
- Traffic flow: Network MFG examples
- Crowd dynamics: Anisotropic MFG, evacuation examples
- Economics: Spatial competition examples
- Custom: Factory patterns and configuration examples

---

## 10. Documentation Contribution Guidelines

### Adding New Documentation

**Theory Documents**:
1. Follow `docs/theory/NOTATION_STANDARDS.md`
2. Use footnoted references with full citations
3. Add entry to `docs/theory/THEORY_DOCUMENTATION_INDEX.md`
4. Link to implementation with file:line references

**Example Documentation**:
1. Add README.md in example subdirectory
2. Include mathematical formulation
3. Provide usage instructions
4. Update `examples/README.md`

**Development Documents**:
1. Place in appropriate `docs/development/` subdirectory
2. Mark status: `[WIP]`, `[COMPLETED]`, `[RESOLVED]`
3. Link to related issues/PRs
4. Update this index

### Documentation Standards

**Formatting**:
- Markdown with GitHub flavor
- LaTeX math with `$...$` (inline) or `$$...$$` (display)
- Code blocks with language specification
- Consistent heading hierarchy

**Cross-References**:
- Use relative paths for links
- Reference line numbers for code: `file.py:123`
- Link to theory docs for mathematical background
- Update indexes when adding new documents

**Status Indicators**:
- ✅ Enhanced/Complete
- ⭐ New
- 📝 Adequate
- 🔄 In Progress
- 📋 Planned

---

## 11. Search Guide

### By Topic

**Looking for mathematical foundations?**
→ `docs/theory/mathematical_background.md`

**Need solver convergence criteria?**
→ `docs/theory/convergence_criteria.md`

**Want to understand geometric perspective?**
→ `docs/theory/information_geometry_mfg.md`, `docs/theory/IG_MFG_SYNTHESIS.md`

**Implementing stochastic MFG?**
→ `docs/theory/stochastic_mfg_common_noise.md`, `examples/basic/common_noise_lq_demo.py`

**Working with networks/graphs?**
→ `docs/theory/network_mfg_mathematical_formulation.md`

**Need variational formulation?**
→ `docs/theory/variational_mfg_theory.md`

**High-dimensional problems?**
→ `examples/advanced/highdim_mfg_capabilities/`

**Configuration management?**
→ `examples/advanced/configuration_demos/`

### By File Type

**Theory (`.md` in `docs/theory/`)**: 30+ documents covering all mathematical aspects
**Examples (`.py` in `examples/`)**: 20+ working demonstrations
**Development (`.md` in `docs/development/`)**: Architecture and design decisions
**Advanced (`.md` in `docs/advanced/`)**: Specialized topics (AMR, benchmarking, geometry)

---

## 12. Quick Statistics

### Documentation Metrics

**Theory Documentation**:
- Main documents: 18
- Subdirectories: 2 (numerical_methods, reinforcement_learning)
- Total theory docs: 30+
- References added (Oct 2025): 119 footnoted citations
- New documents created: 3 (NOTATION_STANDARDS, variational_mfg_theory, THEORY_DOCUMENTATION_INDEX)

**Examples**:
- Basic: 5 examples
- Advanced: 12+ examples
- Notebooks: 4+ examples
- Total: 21+ working demonstrations

**Development Documentation**:
- Active documents: ~15
- Completed/archived: 20+
- Design documents: 5+

**Total Documentation**:
- Theory: ~2,500 lines enhanced (Oct 2025)
- Total: Comprehensive coverage of all package features

### Enhancement Status (October 2025)

**Fully Enhanced** ✅:
- 7 core theory documents with full mathematical rigor
- NOTATION_STANDARDS.md created
- THEORY_DOCUMENTATION_INDEX.md created
- variational_mfg_theory.md created (NEW)

**Adequate** 📝:
- Application-specific theory docs
- Example documentation
- Most development docs

**Planned** 📋:
- Sphinx HTML API documentation
- Additional tutorials
- Video walkthroughs

---

## 13. Update History

**October 8, 2025**:
- Created DOCUMENTATION_INDEX.md
- Enhanced 7 core theory documents with mathematical rigor
- Added 119 footnoted references to theory docs
- Created NOTATION_STANDARDS.md
- Created variational_mfg_theory.md (comprehensive variational formulation)
- Created THEORY_DOCUMENTATION_INDEX.md

**Previous**:
- Examples reorganized (2025-09)
- Strategic typing excellence (2025-09)
- Phase 2.2 stochastic MFG completion (2025-10-05)
- Strategic roadmap through 2027 created

---

**Maintained By**: Primary maintainer with AI assistance
**Next Review**: January 2026
**Feedback**: Open GitHub issue for documentation improvements
