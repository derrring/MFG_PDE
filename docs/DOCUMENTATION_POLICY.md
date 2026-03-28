# Documentation Policy

**Purpose**: Maintain clean, navigable, and valuable documentation through **content quality** over arbitrary limits.

---

## Core Principles

### Quality Over Quantity
Focus on **consolidating duplicate content** and **organizing related docs**, not counting files:

**Key Quality Metrics:**
1. 🎯 **No Duplicate Topics**: Same subject shouldn't have multiple files
   - Example: "architecture_design.md" and "design_architecture.md" → consolidate
   - Exception: Different perspectives (user guide vs developer deep-dive)

2. 🔗 **Related Docs Together**: Similar topics should be consolidated
   - Example: 3 separate AMR docs → 1 comprehensive AMR guide
   - Benefit: Easier to maintain, better for readers

3. 📦 **Archive Completed Work**: Finished projects belong in `archive/`
   - Rule: Files marked `[COMPLETED]`, `[CLOSED]`, `[RESOLVED]` → archive
   - Benefit: Active docs remain current and actionable

4. 📂 **Meaningful Structure**: Directories should reflect actual organization
   - No single-file directories (except special cases)
   - No many READMEs proliferation (use index docs instead)

**Check quality**: `python scripts/check_docs_structure.py --report`

### The Rule of 3
- **No directories with < 3 files** (except `archive/`, `private/`)
- If < 3 files: consolidate into parent or remove directory
- Single-file directories create confusion, not organization

### The Archive Principle
- **All completed work goes to archive/**
- Active docs = current, actionable, frequently referenced
- Archive = historical, completed, reference-only

### Why No Hard Limits?
**Problem with hard limits** (e.g., "max 70 files"):
- ❌ Arbitrary numbers don't reflect actual needs
- ❌ Some categories naturally need more docs (API reference)
- ❌ Penalizes growth in appropriate areas
- ❌ Focuses on counting instead of quality

**Content-based approach** (current system):
- ✅ Identifies duplicate content that should be consolidated
- ✅ Finds related docs that belong together
- ✅ Allows natural growth where needed (reference docs)
- ✅ Focuses on actual problems (duplication, disorganization)

**Example from real audit:**
- Found: "README.md" appears 19 times across directories
- Issue: Not "too many files", but **unnecessary duplication**
- Solution: Consolidate READMEs, use index pages instead
- Result: Better organization AND fewer files

---

## Consolidation Triggers

### 1. Phase Completion ✅
**When:** User declares "Phase X is complete" or similar

**Action:**
```bash
# Create single phase summary
docs/development/completed/PHASE_X_SUMMARY.md

# Archive detailed implementation docs
mv docs/development/phase_x_*.md archive/development/phases/

# Keep only: overview, current status, next steps
```

**Example:** Phase 2 (backend integration) had 8 separate docs → consolidated into `PHASE_2_COMPLETE_SUMMARY.md`

### 2. Related Topic Growth 📚
**When:** 5+ files on same topic created

**Action:**
```bash
# Instead of:
docs/user/mfg_basics_1.md
docs/user/mfg_basics_2.md
docs/user/mfg_basics_3.md
docs/user/mfg_basics_4.md
docs/user/mfg_basics_5.md

# Create:
docs/user/MEAN_FIELD_GAMES_COMPLETE.md  # Consolidated guide

# Archive originals:
mv docs/user/mfg_basics_*.md archive/docs/
```

### 3. Monthly Audit 📅
**When:** 1st of each month (automated via GitHub Action)

**Action:**
- Run: `python scripts/check_docs_structure.py --report`
- Review flagged issues
- Plan consolidation if needed

---

## File Naming Conventions

### Status Prefixes
```bash
# Active work
[WIP]_feature_name.md         # Work in progress
FEATURE_NAME.md                # Standard active doc

# Completed work (MUST be archived)
[COMPLETED]_feature_name.md   # Move to archive/
[RESOLVED]_issue_name.md      # Move to archive/
[CLOSED]_analysis.md          # Move to archive/

# Special categories
[PRIVATE]_internal_notes.md   # Use docs/*/private/ instead
[ANALYSIS]_topic.md           # Permanent analysis (OK to keep active)
```

### Directory Structure
```
docs/
├── user/              # User-facing documentation
├── development/       # Developer guides and status
│   ├── completed/    # Consolidated phase summaries
│   └── private/      # Internal notes (gitignored)
├── theory/           # Mathematical theory
│   └── private/      # Internal derivations (gitignored)
├── reference/        # API reference (auto-generated)
└── archive/          # Historical content
    ├── development/
    │   ├── sessions/ # Old session summaries
    │   ├── phases/   # Individual phase docs
    │   └── analyses/ # Historical analyses
    └── theory/       # Superseded theory docs
```

---

## Archive Strategy

### What to Archive
✅ **Archive (move to `archive/`):**
- Completed phase implementation details
- Old session summaries (keep latest 3 active)
- Superseded documentation
- Detailed analyses after consolidation

❌ **Do NOT archive:**
- Current roadmaps and status docs
- Active feature documentation
- Ongoing work-in-progress
- Permanent reference materials

### How to Archive
```bash
# Create archive structure if needed
mkdir -p archive/development/{sessions,phases,analyses}

# Move completed work
git mv docs/development/PHASE_2_DETAILED.md archive/development/phases/
git mv docs/development/SESSION_SUMMARY_2025-10-08.md archive/development/sessions/

# Update cross-references in active docs
# Commit with descriptive message
git commit -m "docs: Archive Phase 2 detailed implementation docs"
```

---

## Zero Tolerance Rules

### ❌ Forbidden Practices

1. **No [COMPLETED] files outside archive/**
   - Completed work must be archived or status removed
   - Check: `python scripts/check_docs_structure.py`

2. **No duplicate/overlapping documentation**
   - One topic = one comprehensive doc
   - Consolidate instead of creating similar docs

3. **No orphaned directories**
   - If 0 files: remove directory
   - If < 3 files: consolidate into parent

4. **No untracked clutter in docs/**
   - All docs/ files should be tracked in git
   - Use private/ subdirectories for gitignored content

---

## Proactive Documentation Hygiene

### For AI Assistants (Claude Code)

**Auto-trigger checks when:**
- User says "phase X complete" → Run consolidation check
- Creating 3rd file on same topic → Suggest consolidation
- 5+ session summaries exist → Suggest archiving older ones
- [COMPLETED] tag added → Prompt to archive
- New directory created → Verify necessity (≥ 3 files planned?)

**Before major commits:**
```bash
python scripts/check_docs_structure.py --report
```

**Monthly reminder:**
```bash
# 1st of month: Check documentation health
python scripts/check_docs_structure.py --report

# If issues found: create consolidation plan
```

---

## Consolidation Workflow

### Step 1: Identify Consolidation Candidates
```bash
# Find completed files not archived
find docs -name "*COMPLETED*" ! -path "*/archive/*"

# Find sparse directories (< 3 files)
python scripts/check_docs_structure.py --report
```

### Step 2: Create Consolidated Document
```markdown
# CONSOLIDATED_TOPIC.md

**Consolidated Documentation** - Last Updated: YYYY-MM-DD

This document consolidates information from:
- source_doc_1.md
- source_doc_2.md
- source_doc_3.md

## Table of Contents
1. [Section from Doc 1](#section-1)
2. [Section from Doc 2](#section-2)
...

## Section 1
(Content from source_doc_1.md)

## Section 2
(Content from source_doc_2.md)

---

**Original Documents** (archived):
- `archive/path/source_doc_1.md`
- `archive/path/source_doc_2.md`
```

### Step 3: Archive Sources
```bash
# Move to archive
git mv docs/topic/source_*.md archive/development/

# Commit consolidation
git add docs/topic/CONSOLIDATED_TOPIC.md archive/
git commit -m "docs: Consolidate topic documentation (3 docs → 1)"
```

### Step 4: Update Cross-References
```bash
# Find references to archived docs
rg "source_doc_1.md" docs/

# Update to point to consolidated doc
# Commit reference updates
```

---

## Validation Commands

### Check Documentation Health
```bash
# Basic check
python scripts/check_docs_structure.py

# Detailed report
python scripts/check_docs_structure.py --report

# Auto-fix (where possible)
python scripts/check_docs_structure.py --fix
```

### Pre-commit Hook (Automated)
```bash
# Install pre-commit hooks
pre-commit install

# Hook will run check_docs_structure.py on docs/ changes
```

### Monthly Audit (Automated)
- GitHub Action runs 1st of each month
- Creates issue if problems detected
- Check: `.github/workflows/docs_audit.yml`

---

## Examples

### Good Documentation Structure ✅
```
docs/
├── development/
│   ├── ARCHITECTURE.md           # Consolidated (5 sources)
│   ├── API_DESIGN.md             # Consolidated (3 sources)
│   ├── ROADMAP_2026.md           # Active roadmap
│   └── completed/
│       ├── PHASE_2_SUMMARY.md    # Consolidated phase
│       └── PHASE_3_SUMMARY.md    # Consolidated phase
└── theory/
    ├── MEAN_FIELD_GAMES.md       # Comprehensive guide
    ├── NUMERICAL_METHODS.md      # Consolidated methods
    └── APPLICATIONS.md            # Consolidated applications

Total: ~15 active docs, well-organized
```

### Bad Documentation Structure ❌
```
docs/
├── development/
│   ├── backend_design_v1.md
│   ├── backend_design_v2.md
│   ├── backend_design_final.md   # Duplication!
│   ├── [COMPLETED]_phase2.md     # Not archived!
│   ├── session_summary_2025-09-01.md
│   ├── session_summary_2025-09-08.md
│   ├── session_summary_2025-09-15.md
│   ├── ... (12 more session summaries)
│   └── temp/                     # Empty directory!
└── theory/
    ├── mfg_1.md
    ├── mfg_2.md
    ├── mfg_3.md
    ├── ... (20 fragmented theory docs)

Total: 80+ docs, poor organization
```

---

## Quick Reference

| Situation | Action |
|:----------|:-------|
| Phase completed | Create summary, archive details |
| 5+ files on topic | Consolidate into single guide |
| [COMPLETED] tag | Archive to appropriate location |
| New directory | Ensure ≥ 3 files, else consolidate |
| Approaching 70 docs | Run consolidation pass |
| Monthly (1st) | Run audit, address issues |

**Automated Check:** `python scripts/check_docs_structure.py`

**Policy Version:** 1.0 (2025-10-11)

---

**See Also:**
- `CONSOLIDATION_ROADMAP.md` - Active consolidation plan
- `CLAUDE.md` - Full project conventions
- `scripts/check_docs_structure.py` - Validation tool
