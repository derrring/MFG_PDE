# Research Repository Migration Assessment

**Date**: 2025-11-02
**Author**: Claude Code
**Purpose**: Assess research implementations for potential migration to MFG_PDE production

---

## Executive Summary

**Status**: Production MFG_PDE repo now has comprehensive nD support (PR #217 merged). Research repo contains experimental algorithms that may be ready for graduation.

**Key Finding**: The research repo contains **mature particle collocation methods** that could be graduated to production, but **full Lagrangian MFG** should remain in research stage.

---

## Production Repo Status (as of PR #217)

### Recently Completed ✅
1. **WENO Solver** - Extended to arbitrary nD (4D tested)
2. **Semi-Lagrangian Solver** - Extended to arbitrary nD (3D tested)
3. **Particle Interpolation** - Extended to arbitrary nD (5D tested)
4. **HamiltonianAdapter** - Signature unification implemented
5. **Semi-Lagrangian Refactoring** - Now calls `problem.hamiltonian()`

### Production Capabilities
- **nD Support**: 1D through 5D+ (tested up to 5D)
- **Solvers**: HJB-FDM, HJB-WENO, HJB-Semi-Lagrangian, HJB-GFDM, FP-FDM, FP-Particle
- **Backends**: NumPy (ready), JAX (partial), PyTorch (partial)
- **Geometry**: ImplicitDomain, CSG operations, SDF, particle sampling

---

## Research Repo Analysis

### 1. Particle Collocation Methods

**Location**: `/Users/zvezda/OneDrive/code/mfg-research/algorithms/particle_collocation/`

#### Components

##### A. DualModeFPParticleSolver (`fp_particle_dual_mode.py`)
**Status**: Mature - 22,005 lines
**Purpose**: Dual-mode FP solver (hybrid/collocation)

**Features**:
- Switch between grid-based and particle-based FP
- Interpolation between modes
- Compatible with existing MFG_PDE infrastructure

**Assessment**:
- ⚠️ **Needs Review** - Very large file (22k lines) suggests it may need refactoring
- ✅ **Well-documented** - Has test file and proof-of-concept
- ⏳ **Graduation Candidate** - IF refactored and fully tested

##### B. ParticleCollocationSolver (`solver.py`)
**Status**: Research-grade - 24,828 lines
**Purpose**: Base particle-collocation solver class

**Assessment**:
- ⚠️ **Very large** - 24k lines suggests monolithic design
- 🔍 **Needs investigation** - Check if overlaps with existing `HJBGFDMSolver`
- ⚠️ **May need refactoring** before production

##### C. Maze Integration
**Location**: `particle_collocation/proof_of_concept/`
**Purpose**: Convert discrete mazes to continuous implicit domains

**Key Architecture** (from docs):
```
Layer 1: Discrete Maze Generation (MFG_PDE)
         ↓
Layer 2: Continuous Geometry Conversion (mfg-research)
         ↓
Layer 3: Particle-Collocation HJB Solver (MFG_PDE)
```

**Assessment**:
- ✅ **Good architecture** - Reuses MFG_PDE infrastructure
- ✅ **Well-documented** - Comprehensive architecture document
- ⏳ **Graduation Candidate** - Layer 2 (maze→implicit domain converter)

#### Graduation Readiness: Particle Collocation

| Component | Lines | Maturity | Tests | Docs | Production Ready? |
|:----------|:------|:---------|:------|:-----|:------------------|
| DualModeFPParticleSolver | 22k | Medium | Yes | Yes | ⏳ After refactor |
| ParticleCollocationSolver | 24k | Low | ? | Yes | ❌ Needs work |
| Maze Integration | ? | Medium | ? | ✅ | ⏳ Maybe |

**Recommendation**:
- 🔍 **Investigate overlap** with existing `HJBGFDMSolver` in MFG_PDE
- ⚠️ **Refactor if unique** - Break down 22k-24k line files into modules
- ✅ **Merge maze integration** - Small, well-defined component

---

### 2. Full Lagrangian MFG

**Location**: Research analysis document only (no implementation)
**Status**: Theoretical exploration
**Assessment**: From `FULLY_LAGRANGIAN_MFG_RESEARCH_ASSESSMENT.md`

#### Key Findings from Research Analysis

**Pros**:
- ✅ High novelty (almost unexplored territory)
- ✅ Intellectually interesting
- ✅ Potential for high-dimensional problems

**Cons**:
- ❌ Very high risk (may not work)
- ❌ No clear killer application
- ❌ 18-24 month timeline (PhD-level)
- ❌ Requires deformation gradient tracking
- ❌ Complex Hamiltonian transformation

**Recommendation from Research**:
```
Short-term (6 months): Focus on Eulerian-Lagrangian ✅
Medium-term (12 months): Adaptive collocation ⭐
Long-term (2+ years): Full Lagrangian ⚠️ (PhD thesis level)
```

**Production Decision**:
- ❌ **Do NOT migrate** - Too experimental
- ✅ **Keep in research** - Requires 2+ years of development
- ℹ️ **Re-assess in 2026** - After Eulerian-Lagrangian proven

---

### 3. Experiments Status

#### Maze Navigation
**Location**: `experiments/maze_navigation/`
**Status**: Active research
**Key Results**:
- Anderson acceleration integrated
- Bug #15 fixed in MFG_PDE v1.7.3+
- QP sigma API issue resolved (PR #201)

**Migration Status**:
- ✅ **Bug fixes already migrated** to MFG_PDE
- ⏳ **Algorithm graduation pending** publication

#### Anisotropic Crowd QP
**Location**: `experiments/anisotropic_crowd_qp/`
**Status**: Application-specific research
**Assessment**:
- 🔬 **Research-stage** - Unpublished
- ⏳ **Wait for publication** before migration

---

## Current MFG_PDE vs Research Repo Capabilities

### Already in Production (MFG_PDE)

| Capability | Status | Version |
|:-----------|:-------|:--------|
| HJB-GFDM Solver | ✅ Full nD | v0.5.0+ |
| Particle-based FP | ✅ Full nD | v0.6.0+ |
| ImplicitDomain geometry | ✅ Full nD | v0.7.0+ |
| Particle sampling | ✅ `sample_uniform()` | v0.7.0+ |
| CSG operations | ✅ Union, Difference, etc. | v0.7.0+ |
| QP monotonicity | ✅ Via HJBGFDMSolver | v0.7.3+ |

### Research Repo Additions

| Capability | Location | Maturity | Overlap with Production? |
|:-----------|:---------|:---------|:------------------------|
| Dual-mode FP | `fp_particle_dual_mode.py` | Medium | ⚠️ Needs investigation |
| Collocation solver | `solver.py` | Low | ⚠️ May overlap with GFDM |
| Maze→implicit | `maze_integration/` | Medium | ✅ No overlap |
| Full Lagrangian | Theory only | Very Low | ❌ N/A |

---

## Migration Recommendations

### Immediate Actions (This Week)

1. **Investigate Overlap** 🔍
   ```bash
   # Check if ParticleCollocationSolver duplicates HJBGFDMSolver
   diff -r mfg_pde/alg/numerical/hjb_solvers/hjb_gfdm.py \
           mfg-research/algorithms/particle_collocation/solver.py
   ```

2. **Review DualModeFPParticleSolver** 📝
   - Check if it adds value over existing `FPParticleSolver`
   - Identify unique features
   - Assess refactoring effort

3. **Extract Maze Integration** ✂️
   - Small, well-defined component
   - Good architecture documentation
   - No overlap with production

### Short-term (1-2 Weeks)

4. **Refactor Large Files** 🏗️
   - Break down 22k-24k line files into modules
   - Follow MFG_PDE architecture patterns
   - Add comprehensive tests

5. **Create Migration Plan** 📋
   - Prioritize components by maturity
   - Identify testing requirements
   - Plan API integration

### Medium-term (1 Month)

6. **Gradual Migration** 🚀
   - Migrate maze integration first (lowest risk)
   - Then dual-mode FP (if unique features identified)
   - Integrate into existing solver ecosystem

7. **Documentation** 📚
   - Update MFG_PDE docs with new capabilities
   - Create migration guide for research users
   - Document differences from research version

---

## Specific Migration Candidates

### High Priority: Maze Integration 🌟

**Component**: Discrete maze → Implicit domain converter
**Why Migrate**:
- ✅ Small, well-defined
- ✅ Excellent documentation
- ✅ No overlap with production
- ✅ Enables new applications

**Migration Plan**:
```
1. Create mfg_pde/geometry/maze_converter.py
2. Add tests (maze sampling, boundary detection)
3. Create example: examples/advanced/maze_navigation_demo.py
4. Document in docs/user/guides/geometry.md
```

**Estimated Effort**: 1-2 days

### Medium Priority: DualModeFPParticleSolver ⏳

**Component**: Dual-mode FP solver
**Why Investigate**:
- ⚠️ 22k lines - may contain valuable features
- ✅ Has tests and proof-of-concept
- ⚠️ Needs refactoring

**Investigation Plan**:
```
1. Read fp_particle_dual_mode.py to identify unique features
2. Check overlap with existing FPParticleSolver
3. Extract unique features if any
4. Propose refactored architecture
```

**Estimated Effort**: 2-3 days investigation + 1 week refactoring

### Low Priority: ParticleCollocationSolver ❓

**Component**: Base particle-collocation solver
**Why Low Priority**:
- ⚠️ 24k lines - likely duplicates GFDM
- ⚠️ Unclear maturity level
- ⚠️ May be experiment-specific

**Decision Criteria**:
```
IF (unique features AND well-tested):
    Refactor and migrate
ELSE:
    Keep in research repo
```

**Estimated Effort**: Unknown - depends on overlap analysis

### Not Migrating: Full Lagrangian MFG ❌

**Component**: Full Lagrangian formulation
**Why Not**:
- ❌ Theory only (no implementation)
- ❌ 18-24 month development timeline
- ❌ High risk, uncertain payoff
- ❌ No compelling application identified

**Future Re-assessment**: 2026-Q4 (after 1+ year of research progress)

---

## Integration Strategy

### Architecture Principles

**Reuse Over Reimplementation**:
- Research repo already follows this principle
- Migration should preserve this philosophy
- Leverage existing MFG_PDE infrastructure

**Staged Migration**:
```
Stage 1: experiments/ (research repo)
         ↓
Stage 2: algorithms/ (research repo) ← Current state
         ↓
Stage 3: MFG_PDE (production) ← Graduation target
```

### API Compatibility

**Research → Production Migration Checklist**:
- [ ] Type hints (strict typing)
- [ ] Comprehensive docstrings
- [ ] Unit tests (>80% coverage)
- [ ] Integration tests
- [ ] Examples and tutorials
- [ ] API documentation
- [ ] Backward compatibility

---

## Timeline Proposal

### Week 1 (Current)
- [x] Complete production nD support (PR #217) ✅
- [ ] Investigate particle collocation overlap
- [ ] Review maze integration code

### Week 2-3
- [ ] Extract maze integration component
- [ ] Refactor if needed
- [ ] Add tests and examples
- [ ] Create PR for maze integration

### Month 2
- [ ] Analyze DualModeFPParticleSolver
- [ ] Decide on migration (yes/no)
- [ ] Refactor if migrating
- [ ] Create PR if ready

### Month 3+
- [ ] Evaluate ParticleCollocationSolver
- [ ] Make graduation decision
- [ ] Plan long-term research directions

---

## Risk Assessment

### Migration Risks

| Risk | Probability | Impact | Mitigation |
|:-----|:------------|:-------|:-----------|
| Code duplication | High | Medium | Thorough overlap analysis |
| API incompatibility | Medium | High | Careful refactoring |
| Test coverage gaps | Medium | High | Comprehensive testing |
| Performance regression | Low | High | Benchmarking |

### Mitigation Strategies

1. **Overlap Analysis** 🔍
   - Use diff tools
   - Check functionality overlap
   - Identify unique features

2. **Incremental Migration** 📦
   - Migrate one component at a time
   - Maintain research repo compatibility
   - Keep migration reversible

3. **Comprehensive Testing** ✅
   - Unit tests for all functions
   - Integration tests with existing solvers
   - Regression tests for performance

---

## Success Criteria

### Migration Successful If:
- ✅ No functionality duplicated
- ✅ API consistent with MFG_PDE patterns
- ✅ Tests passing (>80% coverage)
- ✅ Examples working
- ✅ Documentation complete
- ✅ Performance maintained or improved

### Migration Unsuccessful If:
- ❌ Significant code duplication
- ❌ API breaks existing code
- ❌ Tests failing or insufficient
- ❌ Performance degraded
- ❌ Complexity increased without clear benefit

---

## Recommendations Summary

### Immediate (This Week)
1. ✅ **Investigate overlap** between research particle collocation and production GFDM
2. ✅ **Review maze integration** for quick migration win
3. ✅ **Read DualModeFPParticleSolver** to understand unique features

### Short-term (2-4 Weeks)
4. ⏳ **Migrate maze integration** (high priority, low risk)
5. ⏳ **Decide on DualModeFPParticleSolver** (after investigation)

### Medium-term (1-3 Months)
6. ⏳ **Evaluate ParticleCollocationSolver** (low priority)
7. ⏳ **Plan adaptive collocation** (research repo continuation)

### Long-term (6+ Months)
8. ❌ **Keep full Lagrangian MFG in research** (not ready for production)
9. ⏳ **Re-assess in 2026** (after more research progress)

---

## Conclusion

**Production MFG_PDE Status**: Excellent nD support, comprehensive solver ecosystem, ready for advanced applications.

**Research Repo Status**: Contains mature particle collocation work that may be ready for graduation, but needs overlap analysis and refactoring.

**Next Steps**:
1. Investigate overlap with existing GFDM solver
2. Extract and migrate maze integration (quick win)
3. Decide on DualModeFPParticleSolver after analysis

**Key Principle**: Migrate only what adds clear value without duplication.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-02
**Status**: Assessment complete, awaiting investigation phase
