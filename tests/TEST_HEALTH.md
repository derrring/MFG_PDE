# Test Suite Health Report

Last updated: 2025-12-05

## Summary

| Metric | Value |
|:-------|:------|
| Total test files | 170 |
| Files with xfail/skip | 49 |
| Tests requiring PyTorch | ~24 |
| Tests requiring h5py | ~13 |
| Pre-existing numerical issues | ~17 |

## Skip/XFail Categories

### 1. Optional Dependencies (Expected)

These tests are skipped when optional dependencies aren't installed:

| Dependency | Tests Affected | Status |
|:-----------|:---------------|:-------|
| PyTorch | ~24 | Expected - RL algorithms |
| h5py | ~13 | Expected - HDF5 I/O |
| scipy | ~9 | Expected - sparse operations |
| Gymnasium | ~6 | Expected - RL environments |
| Plotly/Bokeh | ~4 | Expected - interactive visualization |

### 2. Pre-existing Numerical Issues (Tracked)

| Issue | Tests | Root Cause |
|:------|:------|:-----------|
| Semi-Lagrangian overflow | 17 | NaN/Inf in `_solve_crank_nicolson_diffusion` |
| Shape mismatch (solve_mfg) | 16 | FP solver returns (Nt+1,) vs (Nt,) expected |
| GFDM slow tests | 2 | Tests take 5+ minutes each |

### 3. API Migration Pending

| Issue | Tests | Target |
|:------|:------|:-------|
| Array validation | 11 | Phase 3.5 |
| Factory signatures | 7 | Issue #277 |
| Voronoi maze | 1 | Module not implemented |

## Test Tiers

Tests are organized into tiers for CI efficiency:

| Tier | Description | CI Trigger |
|:-----|:------------|:-----------|
| tier1 | Fast unit tests (<1s) | Every commit |
| tier2 | Medium tests (1-30s) | PRs |
| tier3 | Slow integration (>30s) | Merge to main |
| tier4 | Performance tests | Weekly/manual |

## Running Tests

```bash
# Quick validation (PRs)
pytest tests/ -m "not slow" --maxfail=10

# Full suite (main branch)
pytest tests/ -m "not tier4"

# Only fast tests
pytest tests/ -m "tier1"

# Skip optional dependencies
pytest tests/ --ignore=tests/unit/test_alg/test_neural
```

## Known Issues

### Critical (Blocking)
- None currently

### High Priority
- Semi-Lagrangian solver numerical stability (#365)
- solve_mfg shape mismatch (#365)

### Medium Priority
- Array validation tests need fixing
- Factory signature validation

### Low Priority
- Voronoi maze module implementation
