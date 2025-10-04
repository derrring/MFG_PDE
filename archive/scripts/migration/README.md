# Archived Migration Scripts

**Date Archived**: 2025-10-04
**Reason**: Migration tasks completed, scripts no longer needed

---

## Scripts in This Directory

### `update_package_imports.py`
**Original Purpose**: Update imports from old `alg` structure to paradigm-based structure

**Migration Task**: Convert old import paths to new paradigm organization
- Old: `from mfg_pde.alg.hjb_solvers import`
- New: `from mfg_pde.alg.numerical.hjb_solvers import`

**Status**: ✅ **COMPLETE** - All imports successfully migrated
- Verified: 0 old import paths found in codebase (2025-10-04)
- All code uses new paradigm-based structure

---

### `fix_polars_stubs.py`
**Original Purpose**: Fix Polars stub generation for Python 3.12 compatibility

**Issues Fixed**:
- Invalid parameter syntax in generated stubs
- PEP 695 type alias compatibility
- Ruff compliance for generated code

**Status**: ✅ **NO LONGER NEEDED**
- Polars stubs now work correctly with current version (≥0.20)
- No Polars-related mypy errors in package
- Python 3.12 compatibility achieved upstream

---

## Historical Context

These scripts were essential during the package modernization phase:
- Algorithm paradigm reorganization (2024-2025)
- Python 3.12 migration
- Type system modernization

They successfully completed their mission and are preserved here for historical reference.

---

**Do Not Delete**: These scripts document important migration steps and may be useful references for future restructuring efforts.
