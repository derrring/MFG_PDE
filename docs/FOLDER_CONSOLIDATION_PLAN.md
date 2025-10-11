# docs/development/ Folder Consolidation Plan

## Problem
Currently 19 subdirectories under `docs/development/` with overlapping purposes, making navigation confusing.

## Goal
Reduce to 12 clear, distinct subdirectories with no functional overlap.

## Consolidation Mapping

### 1. Merge architecture/ → design/
**Reason**: Both contain design and architecture documents
```
architecture/ (2 files) → design/
```

### 2. Merge future_enhancements/ → planning/
**Reason**: Future plans belong with other forward-looking documents
```
future_enhancements/ (2 files) → planning/
plans/ (3 files) → planning/
```

### 3. Merge roadmaps/ + strategy/ + tracks/ → planning/
**Reason**: All are forward-looking strategic documents
```
roadmaps/ (4 files) → planning/
strategy/ (4 files) → planning/
tracks/ (1 file) → planning/
```

### 4. Move api_audit_2025-10-10/ → analysis/
**Reason**: Temporal folder names are anti-pattern, audits belong in analysis
```
api_audit_2025-10-10/ (1 file) → analysis/
```

### 5. Merge technical/ → analysis/
**Reason**: Technical notes are a form of analysis
```
technical/ (3 files) → analysis/
```

## Final Structure (12 directories)

```
docs/development/
├── analysis/           (28+1+3 = 32 files) - All technical analysis
├── design/             (7+2 = 9 files) - Architecture and design
├── planning/           (3+2+4+4+1 = 14 files) - All forward-looking docs
├── decisions/          (2 files) - Decision records
├── governance/         (2 files) - Process and policies
├── guides/             (3 files) - How-to guides
├── maintenance/        (4 files) - Infrastructure and tooling
├── paradigms/          (4 files) - Framework paradigms
├── status/             (3 files) - Current state
├── typing/             (5 files) - Type system development
├── sessions/           (1 file) - Session summaries
└── archive/            (0 files) - Historical content
```

**Result**: 19 → 12 subdirectories (37% reduction)

## Implementation Order

1. Create planning/ directory
2. Move files:
   - future_enhancements/* → planning/
   - plans/* → planning/
   - roadmaps/* → planning/
   - strategy/* → planning/
   - tracks/* → planning/
3. Move architecture/* → design/
4. Move api_audit_2025-10-10/* → analysis/
5. Move technical/* → analysis/
6. Remove empty directories
7. Update READMEs
8. Commit and push
