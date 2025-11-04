# Test Suite CI Reduction Analysis - Document Index

**Generated**: 2025-11-04  
**Status**: Complete analysis ready for implementation

## Documents

### 1. Executive Summary (START HERE)
**File**: `TEST_SUITE_CI_REDUCTION_SUMMARY.txt`  
**Purpose**: Quick reference, key findings, recommendations  
**Audience**: Decision makers, team leads  
**Read time**: 10 minutes

Contains:
- Headline findings (283-383 tests can be excluded)
- 40-60 second CI time savings (40% reduction)
- Tier 1 & Tier 2 recommendation structure
- Core functionality that must be kept
- Quick reference for specific files to mark
- Risk assessment
- Validation commands

**Best for**: Quick decision making, understanding scope

### 2. Main Analysis Report
**File**: `TEST_SUITE_CI_REDUCTION_ANALYSIS.md`  
**Purpose**: Comprehensive analysis with detailed breakdowns  
**Audience**: Developers implementing the changes  
**Read time**: 30-45 minutes

Contains:
- Complete test count breakdown by category
- Optional dependency analysis (PyTorch, scipy, plotly)
- Slow tests inventory (already handled)
- Experimental algorithm tests
- Core functionality tests (MUST KEEP)
- Detailed exclusion candidates with rationale
- Pytest marker definitions
- Three CI configuration options (A, B, C)
- Implementation checklist (4 phases)
- Risk assessment with mitigations
- Time savings estimates

**Best for**: Understanding the analysis, making implementation decisions

### 3. Implementation Guide
**File**: `TEST_EXCLUSION_IMPLEMENTATION_GUIDE.md`  
**Purpose**: File-by-file implementation instructions  
**Audience**: Developers marking tests  
**Read time**: 20-30 minutes (reference while working)

Contains:
- Tier 1 exclusion breakdown (PyTorch, Benchmark, Experimental)
- Tier 2 exclusion breakdown (Visualization, RL environment, GPU)
- File-by-file list with test counts and locations
- Code examples for adding markers (module-level, class-level, individual)
- Validation checklist with commands
- Phased implementation schedule
- Quick reference: files to modify
- Expected results comparison (before/after)

**Best for**: During implementation, file reference

---

## Implementation Path

### Step 1: Review & Approval
1. Read `TEST_SUITE_CI_REDUCTION_SUMMARY.txt` (10 min)
2. Review key recommendations in main analysis (20 min)
3. Approve Tier 1 (recommended) or Tier 1+2 (aggressive)

### Step 2: Execute Implementation
1. Open `TEST_EXCLUSION_IMPLEMENTATION_GUIDE.md`
2. Follow Phase 1-4 schedule
3. Use file-by-file list for marking tests
4. Run validation commands from summary

### Step 3: CI Configuration
1. Update `.github/workflows/tests.yml`
2. Choose Option A (recommended), B (balanced), or C (current)
3. Update CONTRIBUTING.md with new CI strategy

### Step 4: Validation
1. Run test collection commands
2. Verify counts match expectations
3. Run test suites locally
4. Monitor CI metrics

---

## Key Data Points

### Reduction Tiers

**Tier 1 (Recommended)**: 283 tests, 40s time savings
- PyTorch tests (120): RL algorithms, neural operators, GPU
- Benchmark tests (48): Performance analysis only
- Experimental (55): Research algorithms
- Environment (60): Maze/problem environments

**Tier 2 (Optional)**: Additional 100 tests, 25s more savings
- Visualization (35): Interactive plots (plotly)
- RL environment (50): Gymnasium-specific
- GPU (15): CUDA-specific

### Core to Keep

1,132 tests (39% of suite) are ESSENTIAL:
- Solvers (105): FDM, GFDM, semi-Lagrangian, particle, network
- Problems (55): Problem definitions and validation
- Factory & Config (338): Infrastructure backbone
- Type System (147): API stability
- Geometry (487): Domain representation

### CI Configuration Options

| Option | Tests | Time | Use Case |
|:-------|:-----:|:----:|:---------|
| A (Aggressive) | 3,217 | 70-90s | Pull request feedback |
| B (Balanced) | 3,272 | 100-115s | More thorough validation |
| C (Current) | 3,443 | 120-150s | Nightly full suite |

---

## Files to Modify

### Configuration
- `pytest.ini` - Add marker definitions (1 file)

### Test Files by Category

**PyTorch** (21 files): unit (15), integration (4), root (2)
**Benchmark** (8 files): unit (8)
**Experimental** (5 files): unit (4), root (1)
**Environment** (12 files): unit (12)
**Visualization** (2 files): unit (2)

---

## Success Criteria

After implementation:

```bash
# Should exclude 120 torch tests
pytest --collect-only -m optional_torch -q | tail -1
# Result: ~120 tests

# Should exclude 48 benchmark tests
pytest --collect-only -m benchmark -q | tail -1
# Result: ~48 tests

# Should reduce to core suite
pytest --collect-only -m "not slow and not optional_torch and not benchmark and not experimental" -q | tail -1
# Result: ~3,217 tests (92% of original)

# Full suite should still work
pytest -m "not slow" --co | tail -1
# Result: ~3,443 tests (current baseline)
```

---

## Effort Estimate

| Phase | Duration | Tasks |
|:------|:--------:|:------|
| 1 - Add Markers | 2-3h | Update pytest.ini, mark PyTorch & benchmark (29 files) |
| 2 - Extend | 2-3h | Mark experimental, environment, visualization (19 files) |
| 3 - CI Config | 1h | Update workflow, docs |
| 4 - Testing | 2-4h | Run validations, verify no regressions |
| 5 - Monitor | ongoing | Track metrics, adjust as needed |

**Total**: 7-11 hours of work

---

## Questions?

- **Why exclude PyTorch tests?** They're optional dependency, heavy (30s overhead), not core infrastructure
- **Why not Tier 2?** Visualization/gym tests less critical, but can be added later if more speed needed
- **Can we revert?** Yes, removing markers takes 5 minutes
- **What if markers cause issues?** Run with `-m "not optional_torch"` to verify other tests still pass
- **How do we know this works?** Validation commands provided to verify each tier

---

## Document Relationships

```
TEST_SUITE_CI_REDUCTION_SUMMARY.txt (Executive)
    ↓
    ├── For decision: Review recommendations
    ├── For implementation: Reference Test counts & file list
    └── For validation: Run validation commands
    
TEST_SUITE_CI_REDUCTION_ANALYSIS.md (Detailed)
    ↓
    ├── Contains: Full breakdown, analysis, rationale
    ├── Use: Understand the "why" behind recommendations
    └── Reference: During implementation for specific details
    
TEST_EXCLUSION_IMPLEMENTATION_GUIDE.md (Tactical)
    ↓
    ├── Contains: File-by-file list, code examples
    ├── Use: During implementation phase
    └── Reference: While marking tests, validation checklist
```

---

## Next Actions

1. [ ] Read summary (10 min)
2. [ ] Review main analysis (20 min)
3. [ ] Approve strategy (5 min)
4. [ ] Execute Phase 1-2 per implementation guide (4-6h)
5. [ ] Update CI config per Phase 3 (1h)
6. [ ] Run validations per Phase 4 (1-2h)
7. [ ] Monitor metrics in Phase 5 (ongoing)

**Estimated total effort**: 7-11 hours  
**Expected benefit**: 40-60s CI reduction per run (40% faster)

---

**Last Updated**: 2025-11-04  
**Analysis by**: Claude Code (Haiku 4.5)  
**For questions**: See main analysis documents
