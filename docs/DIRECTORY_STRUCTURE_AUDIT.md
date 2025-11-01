# Directory Structure Audit

**Date**: 2025-10-31
**Status**: Issues Found - Requires Cleanup
**Branch**: feature/phase3-validation-performance

---

## Issue Summary

Found duplicate directory structures between root and package levels that create confusion:

1. **benchmarks**: 3 locations (ROOT, PACKAGE, outputs)
2. **configs**: 3 locations (ROOT, PACKAGE, PACKAGE/configs)

---

## Detailed Analysis

### 1. Benchmarks Duplication

| Location | Purpose | Contents | Status |
|:---------|:--------|:---------|:-------|
| `./benchmarks/` | **User-facing benchmarks** | Executable scripts (run_benchmarks.py, etc.) | ✅ Valid |
| `./mfg_pde/benchmarks/` | **Benchmark infrastructure** | Importable utilities (highdim_benchmark_suite.py) | ✅ Valid |
| `./examples/outputs/benchmarks/` | **Output directory** | Generated results (gitignored) | ✅ Valid |

**Assessment**: This is **ACCEPTABLE** - distinct purposes.
- Root benchmarks = standalone scripts users run
- Package benchmarks = library code for building benchmarks
- Output benchmarks = results storage

**Recommendation**: Document this distinction clearly in README.

---

### 2. Configs Duplication ⚠️ CONFUSING

| Location | Purpose | Contents | Status |
|:---------|:--------|:---------|:-------|
| `./configs/` | **User config files** | YAML/JSON configs (paradigm/, problems/) | ❓ Unclear |
| `./mfg_pde/config/` | **Config module** | Python code (pydantic_config.py, etc.) | ✅ Valid |
| `./mfg_pde/config/configs/` | **Default YAML configs** | YAML files (base_mfg.yaml, solver.yaml) | ⚠️ **NESTED REDUNDANCY** |

**Problems**:
1. **Triple-nesting confusion**: configs/ vs config/ vs config/configs/
2. **Unclear ownership**: Which configs/ is canonical?
3. **Import confusion**: `from mfg_pde.config import ...` vs `configs/...`

**Current State**:
```
ROOT configs/          # User-level configs
├── paradigm/
│   └── ... (YAML files)
└── problems/
    └── ... (YAML files)

PACKAGE mfg_pde/config/    # Config module (Python)
├── __init__.py
├── pydantic_config.py
├── omegaconf_manager.py
└── configs/               # ⚠️ DEFAULT CONFIGS (YAML)
    ├── base_mfg.yaml
    ├── solver.yaml
    ├── experiment.yaml
    └── beach_problem.yaml
```

---

## Recommended Cleanup

### Option 1: Consolidate YAML Configs (Recommended)

**Action**: Merge all YAML configs into single location

**Before**:
```
./configs/                       # User configs
./mfg_pde/config/                # Python module
./mfg_pde/config/configs/        # Default YAML configs
```

**After**:
```
./configs/                       # ALL YAML configs (user + defaults)
├── defaults/                    # Default configs (formerly mfg_pde/config/configs/)
│   ├── base_mfg.yaml
│   ├── solver.yaml
│   └── experiment.yaml
├── paradigm/                    # User configs
└── problems/                    # User configs

./mfg_pde/config/                # ONLY Python code
├── __init__.py
├── pydantic_config.py
├── omegaconf_manager.py
└── solver_config.py
```

**Benefits**:
- ✅ Clear separation: Python code vs YAML configs
- ✅ Single source of truth for all configs
- ✅ No triple-nesting confusion
- ✅ Easier to find configs

**Migration**:
```bash
# Move default YAML configs to root
mv mfg_pde/config/configs/*.yaml configs/defaults/

# Update Python imports
# Change: mfg_pde/config/configs/base_mfg.yaml
# To:     configs/defaults/base_mfg.yaml
```

---

### Option 2: Rename for Clarity (Alternative)

Keep separate but rename to avoid confusion:

**Before**:
```
./configs/                       # Plural
./mfg_pde/config/                # Singular
./mfg_pde/config/configs/        # Nested plural
```

**After**:
```
./user_configs/                  # User-facing configs (YAML)
./mfg_pde/config/                # Python config module
./mfg_pde/config/defaults/       # Default YAML configs (renamed from configs/)
```

**Benefits**:
- ✅ Clear naming distinction
- ✅ Less invasive change
- ❌ Still maintains separation (may be desired)

---

### Option 3: No Change - Document Only (Minimal)

Keep structure as-is but document clearly:

**Actions**:
1. Add `configs/README.md` explaining purpose
2. Add `mfg_pde/config/README.md` explaining purpose
3. Update main `README.md` with directory structure explanation

**Benefits**:
- ✅ Zero code changes
- ❌ Confusion remains for new developers

---

## Comparison with CLAUDE.md Principles

CLAUDE.md doesn't explicitly address this scenario, but general principles apply:

**From CLAUDE.md**:
> "Repository organization should be clear and unambiguous"

**Current state violates**:
- Clarity: Three nested "config" directories is confusing
- Discoverability: Hard to know which config to modify

**Recommended**: **Option 1** (consolidate YAML configs) aligns best with clean architecture.

---

## Other Directory Analysis

### ✅ Clean Directories (No Issues)

| Directory | Purpose | Status |
|:----------|:--------|:-------|
| `./docs/` | Documentation | ✅ Root only |
| `./examples/` | User examples | ✅ Root only |
| `./tests/` | Test suite | ✅ Root only |
| `./scripts/` | Utility scripts | ✅ Root only |
| `./mfg_pde/alg/` | Algorithms | ✅ Package only |
| `./mfg_pde/core/` | Core types | ✅ Package only |
| `./mfg_pde/geometry/` | Geometry module | ✅ Package only |

No duplication issues found in these directories.

---

## Decision Matrix

| Criterion | Option 1 (Consolidate) | Option 2 (Rename) | Option 3 (Document) |
|:----------|:----------------------|:------------------|:--------------------|
| **Clarity** | ⭐⭐⭐⭐⭐ Best | ⭐⭐⭐⭐ Good | ⭐⭐ Poor |
| **Effort** | ⚠️ Medium (migration) | ⚠️ Medium (refactor) | ✅ Low |
| **Breaking Changes** | ⚠️ Config paths change | ⚠️ Directory names change | ✅ None |
| **Maintainability** | ⭐⭐⭐⭐⭐ Best | ⭐⭐⭐⭐ Good | ⭐⭐ Poor |
| **Discoverability** | ⭐⭐⭐⭐⭐ Best | ⭐⭐⭐⭐ Good | ⭐⭐ Poor |

**Recommendation**: **Option 1** (Consolidate YAML configs)
- Best long-term solution
- Migration script can automate most changes
- Breaking change acceptable in pre-v1.0 (currently v0.8.0-phase2)

---

## Implementation Plan (If Option 1 Chosen)

### Step 1: Create Migration Branch
```bash
git checkout -b chore/consolidate-configs
```

### Step 2: Create New Structure
```bash
mkdir -p configs/defaults
mv mfg_pde/config/configs/*.yaml configs/defaults/
```

### Step 3: Update Python Imports
Search and replace:
- Old: `mfg_pde/config/configs/base_mfg.yaml`
- New: `configs/defaults/base_mfg.yaml`

Files likely affected:
- `mfg_pde/config/__init__.py`
- `mfg_pde/config/omegaconf_manager.py`
- `mfg_pde/config/pydantic_config.py`

### Step 4: Update Documentation
- `README.md` - Document directory structure
- `configs/README.md` - Explain all config types
- `docs/user/guides/configuration.md` - Update config paths

### Step 5: Add Deprecation Warning
```python
# In mfg_pde/config/__init__.py
import warnings

def load_config(path):
    if "mfg_pde/config/configs" in path:
        warnings.warn(
            "Loading configs from mfg_pde/config/configs/ is deprecated. "
            "Use configs/defaults/ instead. "
            "Support will be removed in v1.0.0.",
            DeprecationWarning,
            stacklevel=2
        )
        # Automatically redirect to new location
        path = path.replace("mfg_pde/config/configs/", "configs/defaults/")
    # ... proceed with loading
```

### Step 6: Testing
- Run full test suite
- Verify all config loading works
- Check examples still run

### Step 7: Documentation
- Add migration guide for users
- Update CHANGELOG
- Create PR with comprehensive description

---

## Immediate Action

**For Phase 3 Work**: Proceed without cleanup for now.

**Reason**:
- Phase 3 focuses on validation/performance
- Config consolidation is orthogonal to Phase 3 goals
- Can address in separate chore branch (chore/consolidate-configs)

**Recommendation**: Create issue to track config consolidation, defer to post-Phase 3.

---

## Related Files

### Benchmarks
- Root: `benchmarks/README.md` - Explains root benchmarks
- Package: `mfg_pde/benchmarks/__init__.py` - Benchmark utilities

### Configs
- Root: `configs/` - No README (should add one)
- Package: `mfg_pde/config/__init__.py` - Config module entry point

---

## Questions for User

1. **Priority**: Should config consolidation happen before or after Phase 3?
2. **Breaking Changes**: Acceptable to change config paths in v0.x.x?
3. **Preferred Option**: Option 1 (consolidate), Option 2 (rename), or Option 3 (document)?

---

**Status**: Audit complete, awaiting decision on cleanup approach

**Next Steps**:
1. User decides on cleanup option
2. If Option 1/2: Create chore/consolidate-configs branch
3. If Option 3: Add README files to explain structure
4. Continue with Phase 3 work regardless

---

**Document Status**: Initial audit
**Author**: Claude Code + Development Team
**Last Updated**: 2025-10-31
