# Private Development Documents

**Directory Purpose**: Contains unpublished research, work-in-progress implementations, and private development notes that should **NOT** be committed to the public repository.

---

## Automatic Gitignore Protection

All files with `[PRIVATE]` in the filename are **automatically excluded** from git tracking via `.gitignore`.

### Usage: `[PRIVATE]` Naming Convention

**Basic Format**: `[PRIVATE]_descriptive_name.ext`

**With Status Tags** (recommended): `[PRIVATE][STATUS]_descriptive_name.ext`

**Status Tags**:
- `[WIP]` - Work in progress (actively being developed)
- `[DRAFT]` - Draft version (needs review)
- `[COMPLETED]` - Finished work (for reference)
- `[ARCHIVED]` - Old version (kept for history)

**Examples**:
- `[PRIVATE][WIP]_qp_particle_collocation_implementation_reflections.md`
- `[PRIVATE][DRAFT]_experiment_notes.txt`
- `[PRIVATE][COMPLETED]_draft_algorithm.py`
- `[PRIVATE]_performance_data.pdf` (no status tag needed)

**Why This Works**:
- `.gitignore` pattern: `**/*[PRIVATE]*.md` (and `.py`, `.txt`, `.pdf`)
- Simple and consistent across all file types
- Easy to identify private files at a glance
- Works anywhere in the repository

### Alternative: Private Directories
Files in these directories are also ignored (regardless of naming):
- `docs/theory/private/`
- `docs/development/private/`

---

## Usage Guidelines

### When to Use Private Documents

1. **Unpublished Research**: Papers, methods, or algorithms not yet published
2. **Work-in-Progress**: Incomplete implementations or experimental features
3. **Proprietary Methods**: Collaborator-provided algorithms or techniques
4. **Sensitive Data**: Performance benchmarks, hardware-specific notes
5. **Draft Documentation**: Pre-review implementation plans

### Naming Convention

**Format**: `[PRIVATE]_<descriptive_name>.ext`

**Examples**:
- `[PRIVATE]_qp_particle_collocation_implementation_status.md` (Current file in this directory)
- `[PRIVATE]_master_equation_formulation.md`
- `[PRIVATE]_gpu_acceleration_benchmarks.md`
- `[PRIVATE]_collaboration_algorithm_notes.md`
- `[PRIVATE]_draft_experiment.py`

---

## Current Private Documents

### Active Research (Unpublished)

1. **`[PRIVATE]_qp_particle_collocation_implementation_status.md`**
   - **Location**: `docs/development/private/`
   - **Topic**: QP-constrained particle-collocation method implementation status
   - **Status**: Implementation ~40% theoretically correct, infrastructure 85% complete
   - **Purpose**: Track existing implementation vs. paper requirements
   - **Public Release**: After paper submission/acceptance

2. **`[PRIVATE]_particle_collocation_qp_monotone.md`**
   - **Location**: `docs/theory/numerical_methods/`
   - **Topic**: Mathematical theory for QP-constrained monotone particle-collocation
   - **Status**: Complete (15,000+ words)
   - **Purpose**: Theoretical foundation for unpublished paper
   - **Public Release**: After paper submission/acceptance

3. **`[PRIVATE][WIP]_qp_particle_collocation_implementation_reflections.md`**
   - **Location**: `docs/development/`
   - **Topic**: Deep reflections on fundamental implementation challenges
   - **Status**: Draft complete (18,000 words), pending implementation testing
   - **Purpose**: Analysis of "optimization variable mismatch" problem and recommended hybrid approach
   - **Key Insight**: GFDM optimizes Taylor coefficients D, but monotonicity constrains FD weights w - no tractable direct relationship
   - **Next Steps**: Implement hybrid approach with M-matrix verification
   - **Public Release**: After paper submission/acceptance (may inform discussion section)

---

## Verification

Check that private files are properly ignored:
```bash
# Should show "nothing to commit" for private files
git status docs/development/private/

# List all gitignored files in this directory
git status --ignored docs/development/private/
```

---

## Safety Reminder

⚠️ **Before any git commit or push**:
1. Run `git status` to verify private files aren't tracked
2. Check that filenames follow the `PRIVATE_*` pattern
3. Ensure all sensitive content is in private directories

✅ **If a private file accidentally gets staged**:
```bash
git reset HEAD docs/development/private/PRIVATE_file.md
```

---

**Last Updated**: 2025-10-11
**Protection Method**: Automatic via `.gitignore` patterns
**Status**: ✅ Active - All files in this directory are gitignored
