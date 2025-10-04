# Obsolete Files Audit - October 2025

**Date**: 2025-10-04
**Context**: Post-Phase 3 cleanup after tiered backend integration
**Purpose**: Identify and recommend removal of obsolete test files and examples

---

## 📋 Summary

After Phase 3 (tiered backend integration) completion, multiple test files from Phase 2 experimental work are now obsolete. Additionally, mass conservation tests have accumulated many redundant versions.

**Recommendation**: Delete 11 obsolete/redundant test files to improve maintainability.

---

## 🗑️ Recommended Deletions

### Category 1: Phase 2 Backend Tests (OBSOLETE - Phase 3 supersedes)

**Context**: Phase 2 used manual backend parameter passing. Phase 3 implements tiered auto-selection (`torch > jax > numpy`), making Phase 2 tests obsolete.

| File | Reason | Replacement |
|:-----|:-------|:------------|
| `test_phase2_backend_hjb.py` | Phase 2 HJB backend integration test | `test_tiered_backend_factory.py` |
| `test_phase2_complete.py` | Phase 2 complete integration test | `test_cross_backend_consistency.py` |
| `test_backend_integration.py` | Generic backend parameter test | `test_tiered_backend_factory.py` |
| `test_wrapper_approach.py` | Abandoned wrapper strategy (unused) | Direct backend integration |

**Action**: Delete all 4 Phase 2 test files.

---

### Category 2: Mass Conservation Tests (REDUNDANT - Too many versions)

**Context**: Multiple experimental iterations of mass conservation tests created during stochastic particle method development.

| File | Purpose | Keep? |
|:-----|:--------|:------|
| `test_kde_mass_conservation.py` | KDE-specific mass conservation | ✅ **KEEP** (specialized) |
| `test_mass_conservation_1d.py` | 1D mass conservation test | ⚠️ **MERGE with simple version** |
| `test_mass_conservation_1d_simple.py` | Simplified 1D version | ⚠️ **KEEP only if different** |
| `test_mass_conservation_attempts.py` | Experimental attempts | ❌ **DELETE** (experimental) |
| `test_mass_conservation_fast.py` | Fast version (reduced params) | ⚠️ **KEEP for CI speed** |
| `test_stochastic_mass_conservation.py` | Full stochastic analysis | ✅ **KEEP** (comprehensive) |
| `test_stochastic_mass_conservation_simple.py` | Simplified stochastic | ❌ **DELETE** (redundant) |

**Recommended Actions**:
1. **DELETE**: `test_mass_conservation_attempts.py` (experimental, no longer needed)
2. **DELETE**: `test_stochastic_mass_conservation_simple.py` (full version is sufficient)
3. **EVALUATE**: Compare `test_mass_conservation_1d.py` vs `_simple.py` - keep only one
4. **KEEP**: `test_kde_mass_conservation.py`, `test_mass_conservation_fast.py`, `test_stochastic_mass_conservation.py`

---

### Category 3: Acceleration Tests (EXPERIMENTAL - May be obsolete)

**Context**: Anderson acceleration and two-level damping tests from experimental solver optimization work.

| File | Purpose | Keep? |
|:-----|:--------|:------|
| `test_anderson_acceleration.py` | Anderson acceleration comparison | ⚠️ **EVALUATE** - Is Anderson still used? |
| `test_two_level_damping.py` | Two-level damping (Picard + Anderson) | ⚠️ **EVALUATE** - Active feature? |

**Action Required**: Check if Anderson acceleration is actively used in `FixedPointIterator`:
- If YES → Keep tests as performance benchmarks
- If NO → Delete as obsolete experiments

---

### Category 4: Phase 3 Tests (CURRENT - KEEP ALL)

| File | Purpose | Status |
|:-----|:--------|:-------|
| `test_tiered_backend_factory.py` | Tiered auto-selection (`torch > jax > numpy`) | ✅ Active |
| `test_cross_backend_consistency.py` | Numerical consistency across backends | ✅ Active |
| `test_torch_kde.py` | PyTorch KDE validation | ✅ Active |
| `test_mps_scaling.py` | MPS performance benchmarks | ✅ Active |

**Action**: Keep all Phase 3 tests - these validate current backend system.

---

### Category 5: Specialized Numerical Method Tests (KEEP IF USED)

| File | Purpose | Keep? |
|:-----|:--------|:------|
| `test_collocation_gfdm_hjb.py` | Collocation GFDM for HJB | ⚠️ **Check if method still implemented** |
| `test_particle_collocation.py` | Particle collocation methods | ⚠️ **Check if method still implemented** |
| `test_weight_functions.py` | Weight function tests | ⚠️ **Check if used in current solvers** |

**Action Required**: Verify these numerical methods are still actively used in the package.

---

## 📊 Cleanup Summary

### Immediate Deletions (11 files)

**Phase 2 Backend Tests** (4 files - 100% obsolete):
```bash
rm tests/integration/test_phase2_backend_hjb.py
rm tests/integration/test_phase2_complete.py
rm tests/integration/test_backend_integration.py
rm tests/integration/test_wrapper_approach.py
```

**Redundant Mass Conservation Tests** (2 files):
```bash
rm tests/integration/test_mass_conservation_attempts.py
rm tests/integration/test_stochastic_mass_conservation_simple.py
```

**To Evaluate** (5 files):
1. Compare `test_mass_conservation_1d.py` vs `test_mass_conservation_1d_simple.py` → Delete one
2. Check if Anderson acceleration is used → Delete `test_anderson_acceleration.py` and `test_two_level_damping.py` if not
3. Verify specialized methods are used → Delete `test_collocation_gfdm_hjb.py`, `test_particle_collocation.py`, `test_weight_functions.py` if obsolete

---

## 🔍 Examples Directory

**Status**: Not yet audited
**Count**: 84 example files (82 in basic/advanced)

**Next Steps**:
1. Audit `examples/basic/` for obsolete demos
2. Audit `examples/advanced/` for redundant examples
3. Check for Phase 2 backend examples (should be updated to Phase 3)
4. Identify superseded examples from old API versions

---

## ✅ Recommended Actions

### Step 1: Immediate Cleanup (No risk)
```bash
# Delete Phase 2 tests (100% obsolete)
git rm tests/integration/test_phase2_backend_hjb.py
git rm tests/integration/test_phase2_complete.py
git rm tests/integration/test_backend_integration.py
git rm tests/integration/test_wrapper_approach.py

# Delete redundant mass conservation tests
git rm tests/integration/test_mass_conservation_attempts.py
git rm tests/integration/test_stochastic_mass_conservation_simple.py

git commit -m "🧹 Remove obsolete Phase 2 backend tests and redundant mass conservation tests"
```

### Step 2: Evaluate and Cleanup (Requires verification)
1. **Check Anderson Acceleration Usage**:
   ```bash
   grep -r "use_anderson" mfg_pde/
   ```
   - If used → Keep tests as benchmarks
   - If not used → Delete tests

2. **Compare Mass Conservation 1D Tests**:
   ```bash
   diff tests/integration/test_mass_conservation_1d.py \
        tests/integration/test_mass_conservation_1d_simple.py
   ```
   - If substantially different → Keep both
   - If redundant → Delete one

3. **Verify Specialized Methods**:
   ```bash
   grep -r "collocation_gfdm" mfg_pde/
   grep -r "particle_collocation" mfg_pde/
   grep -r "weight_function" mfg_pde/
   ```
   - If used → Keep tests
   - If not used → Delete tests

### Step 3: Examples Audit
- **TODO**: Audit `examples/` directory for obsolete demos
- **TODO**: Update any Phase 2 backend examples to Phase 3 API
- **TODO**: Remove redundant RL examples if any

---

## 📈 Expected Impact

**Before Cleanup**:
- Integration tests: 20 files
- Obsolete tests: ~6 files (30%)
- Redundant tests: ~5 files (25%)

**After Cleanup**:
- Integration tests: ~9-14 files (depends on evaluation)
- All tests validate current Phase 3 system
- Clear separation: core tests vs specialized method tests
- Improved maintainability and CI speed

---

## 🔗 Related Issues

- Phase 3 completion: Merged to main (commit `7a0c77d`)
- Backend integration: See `docs/development/PHASE3_TIERED_BACKEND_STRATEGY.md`
- MPS performance: See `docs/user/guides/backend_usage.md`

---

**Next Review**: After examples directory audit
