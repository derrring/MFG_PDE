# Documentation Recategorization Plan

**Date**: October 2, 2025
**Status**: Top-Level Recategorization Plan
**Context**: Move docs/development/ files to appropriate top-level docs/ categories

---

## Current Top-Level Structure

```
docs/
├── advanced/          # Advanced topics and case studies
├── development/       # 63+ development files (TO RECATEGORIZE)
├── examples/          # Example documentation
├── reference/         # Quick reference materials
├── theory/            # Mathematical foundations
└── user/              # User-facing documentation
```

---

## Recategorization Strategy

### **Files to Keep in docs/development/**
True development-only content that doesn't fit elsewhere:

**Core Development Process** (6 files):
```
CONSISTENCY_GUIDE.md                      # Code standards
STRATEGIC_DEVELOPMENT_ROADMAP_2026.md     # Overall roadmap
ARCHITECTURAL_CHANGES.md                  # Change log
CODE_REVIEW_GUIDELINES.md                 # Review process
SELF_GOVERNANCE_PROTOCOL.md               # Governance
ORGANIZATION.md                            # Project structure
```

**CI/CD & Tooling** (8 files):
```
typing/CI_CD_STRATEGIC_TYPING_EXPERIENCE_GUIDE.md
typing/STRATEGIC_TYPING_PATTERNS_REFERENCE.md
typing/CI_CD_TROUBLESHOOTING_QUICK_REFERENCE.md
guides/tooling/UV_INTEGRATION_GUIDE.md
guides/tooling/UV_SCIENTIFIC_COMPUTING_GUIDE.md
guides/tooling/UV_SETUP.md
guides/tooling/logging_guide.md
DOCUMENTATION_REORGANIZATION_PLAN.md
```

**Development Infrastructure** (existing subdirectories):
```
analysis/              # Technical analyses
architecture/          # Architecture docs
strategy/              # Strategic planning
maintenance/           # Maintenance procedures
performance_data/      # Performance data
```

---

## Files to Move to Other Categories

### **1. Move to docs/theory/** - Mathematical/Theoretical Content

**New subdirectory: docs/theory/reinforcement_learning/**
```
roadmaps/reinforcement_learning/CONTINUOUS_ACTION_MFG_RESEARCH_ROADMAP.md
  → theory/reinforcement_learning/continuous_action_mfg_theory.md

roadmaps/reinforcement_learning/RL_ACTION_SPACE_SCALABILITY_ANALYSIS.md
  → theory/reinforcement_learning/action_space_scalability.md

design/continuous_action_architecture_sketch.py
  → theory/reinforcement_learning/continuous_action_architecture_sketch.py
```

**New subdirectory: docs/theory/numerical_methods/**
```
design/ADAPTIVE_MESH_REFINEMENT_DESIGN.md
  → theory/numerical_methods/adaptive_mesh_refinement.md

design/SEMI_LAGRANGIAN_IMPLEMENTATION.md
  → theory/numerical_methods/semi_lagrangian_methods.md

design/LAGRANGIAN_MFG_SYSTEM.md
  → theory/numerical_methods/lagrangian_formulation.md
```

**Existing theory/ files** (keep as-is):
- mathematical_background.md
- network_mfg_mathematical_formulation.md
- adaptive_mesh_refinement_mfg.md
- convergence_criteria.md
- el_farol_bar_mathematical_formulation.md
- etc.

---

### **2. Move to docs/user/** - User-Facing Guides

**New subdirectory: docs/user/guides/**
```
guides/features/BACKEND_USAGE_GUIDE.md
  → user/guides/backend_usage.md

guides/features/MAZE_GENERATION_GUIDE.md
  → user/guides/maze_generation.md

guides/features/HOOKS_IMPLEMENTATION_GUIDE.md
  → user/guides/hooks.md (merge with advanced_hooks.md?)

guides/features/PLUGIN_DEVELOPMENT_GUIDE.md
  → user/guides/plugin_development.md
```

**New subdirectory: docs/user/collaboration/**
```
guides/collaboration/AI_INTERACTION_DESIGN.md
  → user/collaboration/ai_assisted_development.md

guides/collaboration/GITHUB_LABEL_SYSTEM.md
  → user/collaboration/github_workflow.md

guides/collaboration/GITHUB_ISSUE_EXAMPLE.md
  → user/collaboration/issue_templates.md
```

---

### **3. Move to docs/reference/** - Quick References

```
typing/MODERN_PYTHON_TYPING_GUIDE.md
  → reference/python_typing.md

typing/MYPY_WORKING_STRATEGIES.md
  → reference/mypy_usage.md

typing/SYSTEMATIC_TYPING_METHODOLOGY.md
  → reference/typing_methodology.md
```

---

### **4. Move to docs/advanced/** - Advanced Topics

**System Design** (new subdirectory: docs/advanced/design/):
```
design/GEOMETRY_SYSTEM_DESIGN.md
  → advanced/design/geometry_system.md

design/HYBRID_MAZE_GENERATION_DESIGN.md
  → advanced/design/hybrid_maze_generation.md

design/API_REDESIGN_PLAN.md
  → advanced/design/api_architecture.md

design/BENCHMARK_DESIGN.md
  → advanced/design/benchmarking.md

design/amr_performance_architecture.md
  → advanced/design/amr_performance.md

design/geometry_amr_consistency_analysis.md
  → advanced/design/geometry_amr_integration.md
```

---

### **5. Create docs/planning/** - Planning & Roadmaps

**New top-level category for strategic planning:**

```
docs/planning/
├── README.md                                    # Planning index
├── roadmaps/
│   ├── REINFORCEMENT_LEARNING_ROADMAP.md
│   ├── ALGORITHM_REORGANIZATION_PLAN.md
│   ├── PRAGMATIC_TYPING_PHASE_2_ROADMAP.md
│   └── BRANCH_STRATEGY_PARADIGMS.md
├── completed/
│   ├── [COMPLETED]_MAZE_ENVIRONMENT_IMPLEMENTATION_SUMMARY.md
│   ├── [COMPLETED]_RL_MAZE_ROADMAP_PROGRESS.md
│   ├── ADAPTIVE_PINN_INTEGRATION_SUMMARY.md
│   ├── maze_development_final_summary.md
│   └── development_status_summary.md
├── reports/
│   ├── CODEBASE_QUALITY_ASSESSMENT.md
│   ├── CONSISTENCY_CHECK_REPORT.md
│   ├── ENVIRONMENT_STATUS_REPORT.md
│   ├── DOCUMENTATION_CLEANUP_ACTION_PLAN.md
│   ├── META_PROGRAMMING_FRAMEWORK_REPORT.md
│   ├── [EVALUATION]_monitoring_tools_comparison.md
│   └── DEVELOPMENT_PRIORITY_HIERARCHY.md
└── governance/
    └── next_development_priorities.md
```

**Rationale**: Planning documents are valuable to users/researchers, not just developers

---

## Proposed Final Structure

```
docs/
├── README.md                          # Updated with new structure
│
├── user/                              # User-facing documentation
│   ├── guides/                        # NEW: Feature usage guides
│   │   ├── backend_usage.md
│   │   ├── maze_generation.md
│   │   ├── hooks.md
│   │   └── plugin_development.md
│   ├── collaboration/                 # NEW: Collaboration workflows
│   │   ├── ai_assisted_development.md
│   │   ├── github_workflow.md
│   │   └── issue_templates.md
│   ├── tutorials/                     # Existing
│   ├── quickstart.md                  # Existing
│   └── ...existing files...
│
├── theory/                            # Mathematical foundations
│   ├── reinforcement_learning/        # NEW: RL theory
│   │   ├── continuous_action_mfg_theory.md
│   │   ├── action_space_scalability.md
│   │   └── continuous_action_architecture_sketch.py
│   ├── numerical_methods/             # NEW: Numerical methods
│   │   ├── adaptive_mesh_refinement.md
│   │   ├── semi_lagrangian_methods.md
│   │   └── lagrangian_formulation.md
│   ├── mathematical_background.md     # Existing
│   ├── network_mfg_mathematical_formulation.md
│   └── ...existing files...
│
├── advanced/                          # Advanced topics
│   ├── design/                        # NEW: System design
│   │   ├── geometry_system.md
│   │   ├── hybrid_maze_generation.md
│   │   ├── api_architecture.md
│   │   ├── benchmarking.md
│   │   ├── amr_performance.md
│   │   └── geometry_amr_integration.md
│   └── ...existing files...
│
├── reference/                         # Quick references
│   ├── python_typing.md               # NEW
│   ├── mypy_usage.md                  # NEW
│   ├── typing_methodology.md          # NEW
│   └── ...existing files...
│
├── planning/                          # NEW: Planning & roadmaps
│   ├── README.md
│   ├── roadmaps/
│   ├── completed/
│   ├── reports/
│   └── governance/
│
├── development/                       # TRUE development content only
│   ├── README.md                      # Slimmed down
│   ├── CONSISTENCY_GUIDE.md
│   ├── STRATEGIC_DEVELOPMENT_ROADMAP_2026.md
│   ├── ARCHITECTURAL_CHANGES.md
│   ├── CODE_REVIEW_GUIDELINES.md
│   ├── SELF_GOVERNANCE_PROTOCOL.md
│   ├── ORGANIZATION.md
│   ├── typing/                        # CI/CD type guides only
│   │   ├── CI_CD_STRATEGIC_TYPING_EXPERIENCE_GUIDE.md
│   │   ├── STRATEGIC_TYPING_PATTERNS_REFERENCE.md
│   │   └── CI_CD_TROUBLESHOOTING_QUICK_REFERENCE.md
│   ├── tooling/                       # Dev tooling only
│   │   ├── UV_INTEGRATION_GUIDE.md
│   │   ├── UV_SCIENTIFIC_COMPUTING_GUIDE.md
│   │   ├── UV_SETUP.md
│   │   └── logging_guide.md
│   ├── analysis/                      # Existing
│   ├── architecture/                  # Existing
│   ├── strategy/                      # Existing
│   └── maintenance/                   # Existing
│
└── examples/                          # Existing
```

---

## Implementation Strategy

### Phase 1: Create New Top-Level Categories (10 min)
```bash
cd docs/
mkdir -p planning/{roadmaps,completed,reports,governance}
mkdir -p theory/{reinforcement_learning,numerical_methods}
mkdir -p user/{guides,collaboration}
mkdir -p advanced/design
mkdir -p reference
```

### Phase 2: Move Files to New Locations (30 min)
Use `git mv` to preserve history. Examples:

```bash
# To theory/
git mv development/roadmaps/reinforcement_learning/CONTINUOUS_ACTION_MFG_RESEARCH_ROADMAP.md \
        theory/reinforcement_learning/continuous_action_mfg_theory.md

# To user/
git mv development/guides/features/BACKEND_USAGE_GUIDE.md \
        user/guides/backend_usage.md

# To planning/
git mv development/roadmaps/ planning/roadmaps/
git mv development/completed/ planning/completed/
git mv development/reports/ planning/reports/

# To advanced/
git mv development/design/ advanced/design/

# To reference/
git mv development/typing/MODERN_PYTHON_TYPING_GUIDE.md \
        reference/python_typing.md
```

### Phase 3: Update Cross-References (20 min)
- Update main docs/README.md
- Update category README files
- Fix internal links
- Update CLAUDE.md if needed

### Phase 4: Slim Down development/ (10 min)
- Keep only true development content
- Update development/README.md
- Remove empty subdirectories

### Phase 5: Test & Validate (10 min)
- Verify all links work
- Check GitHub rendering
- Ensure no files lost

**Total Time**: ~80 minutes

---

## Benefits of Top-Level Recategorization

1. **User-Centric**: Users find guides in docs/user/, not buried in development/
2. **Theory-Focused**: Mathematical content in docs/theory/ where researchers expect it
3. **Clear Separation**: Development vs usage vs theory vs planning
4. **Better Discovery**: Top-level categories match user mental models
5. **Scalability**: Each category can grow independently

---

## Risks & Mitigations

**Risk 1: Many Link Updates**
- **Mitigation**: Systematic grep + replace, test thoroughly

**Risk 2: User Confusion** (short-term)
- **Mitigation**: Update main README prominently, add "Moved" notices

**Risk 3: Breaking External Links**
- **Mitigation**: Check for external references first

---

## Decision: Proceed?

**Question for User**:
Should we proceed with this top-level recategorization? This is a more significant change than the previous within-development reorganization.

**Alternative**: Keep the current structure (development/ organized internally) and only move clearly user-facing content to docs/user/

**Recommendation**: Proceed with full recategorization for better long-term organization, but do it carefully with proper link updates.

---

**Status**: Awaiting approval to proceed
**Estimated Impact**: High (improves discoverability significantly)
**Estimated Risk**: Medium (many link updates needed)
