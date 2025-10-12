#!/usr/bin/env python3
"""
Documentation structure validation script.

Focuses on content quality rather than arbitrary limits:
1. Detect duplicate/overlapping content (same topic in multiple files)
2. Find related docs that should be consolidated
3. Identify [COMPLETED] files outside archive/
4. Detect sparse directories (organizational issues)

ABSTRACT CONSOLIDATION CRITERIA:
================================

1. **Completed Work Principle**
   - Any file with [COMPLETED], [CLOSED], [RESOLVED] → archive/
   - Completed phases (all sub-phases done) → archive/phase_N_details/
   - Session summaries > 7 days old → archive/sessions/

2. **Topic Consolidation Trigger**
   - Same topic in 3+ files → Consolidate into one comprehensive document
   - Phase sub-documents (2.1, 2.2, 2.3...) when phase complete → Archive details, keep summary

3. **README Minimalism**
   - Only keep READMEs at major category levels (root, development/, theory/, user/)
   - Remove subdirectory READMEs (they fragment navigation)

4. **Directory Health**
   - Directories with < 3 files → Merge content into parent or remove
   - Empty directories → Remove immediately
   - Redundant nesting (dir with only 1 subdir) → Flatten

5. **Duplicate Detection**
   - Normalize filenames (strip dates, versions, status tags)
   - Group by topic: "PHASE_2_SUMMARY_2025-10-08" → "phase_2"
   - If multiple files map to same topic → Review for consolidation

6. **Directory Purpose Clarity** ⚠️ **CRITICAL**
   - Maximum 12 subdirectories per major category (development/, theory/, user/)
   - Each subdirectory must have DISTINCT, non-overlapping purpose
   - Directory names must be self-explanatory (no ambiguous names)
   - Overlapping purposes → Consolidate directories

   **Common Overlaps to Avoid**:
   - design/ + architecture/ → Merge to design/
   - plans/ + future_enhancements/ + roadmaps/ → Merge to planning/
   - strategy/ + roadmaps/ + tracks/ → Merge to planning/
   - technical/ + analysis/ → Keep separate only if clearly distinct
   - status/ + progress/ + state/ → Merge to status/

   **Directory Naming Standards**:
   - Use clear functional names: analysis/, guides/, planning/
   - Avoid temporal names: api_audit_2025-10-10/ (use analysis/ instead)
   - Avoid vague names: misc/, other/, temp/
   - Use plural for collections: guides/, sessions/, decisions/

Usage:
    python scripts/check_docs_structure.py          # Check only
    python scripts/check_docs_structure.py --fix    # Auto-fix issues
    python scripts/check_docs_structure.py --report # Detailed report
"""

import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

# Universal thresholds
MIN_FILES_PER_DIR = 3
ALLOWED_SPARSE_DIRS = {"archive", "private", ".git", ".github"}

# Content similarity threshold (0.0-1.0)
SIMILARITY_THRESHOLD = 0.6  # 60% similar content = potential duplicate


def normalize_topic(filename: str) -> str:
    """Extract normalized topic from filename."""
    # Remove status tags
    topic = re.sub(r"\[(COMPLETED|WIP|CLOSED|RESOLVED|ARCHIVED|PRIVATE)\]_?", "", filename, flags=re.IGNORECASE)
    # Remove dates
    topic = re.sub(r"_?\d{4}-\d{2}-\d{2}", "", topic)
    # Remove versions
    topic = re.sub(r"_?v\d+", "", topic)
    # Remove common prefixes/suffixes
    topic = re.sub(r"^(SESSION|PHASE)_?", "", topic, flags=re.IGNORECASE)
    topic = re.sub(r"_(SUMMARY|STATUS|GUIDE|OVERVIEW|ANALYSIS)$", "", topic, flags=re.IGNORECASE)
    # Normalize to lowercase, replace separators
    topic = topic.lower().replace("-", "_").replace(" ", "_")
    # Remove multiple underscores
    topic = re.sub(r"_+", "_", topic).strip("_")
    return topic


def calculate_content_similarity(file1: Path, file2: Path) -> float:
    """Calculate content similarity between two markdown files."""
    try:
        content1 = file1.read_text(encoding="utf-8", errors="ignore")
        content2 = file2.read_text(encoding="utf-8", errors="ignore")

        # Remove markdown formatting for better comparison
        content1 = re.sub(r"[#*`\[\]()]", "", content1)
        content2 = re.sub(r"[#*`\[\]()]", "", content2)

        # Use SequenceMatcher for similarity
        return SequenceMatcher(None, content1, content2).ratio()
    except Exception:
        return 0.0


def check_docs_structure(docs_dir: Path, fix: bool = False, report: bool = False) -> int:
    """Check documentation structure focusing on content quality."""
    issues = []
    warnings = []

    # Collect all active docs
    md_files = list(docs_dir.glob("**/*.md"))
    active_docs = [f for f in md_files if "archive" not in f.parts]

    # Check 1: Find duplicate/similar content
    duplicates = defaultdict(list)

    # Group by normalized topic
    for doc in active_docs:
        topic = normalize_topic(doc.stem)
        if topic:  # Ignore empty topics
            duplicates[topic].append(doc)

    # Find potential duplicates (same topic, different files)
    duplicate_groups = {topic: docs for topic, docs in duplicates.items() if len(docs) > 1}

    if duplicate_groups:
        issues.append(
            f"Found {len(duplicate_groups)} topics with duplicate/similar files:\n"
            + "\n".join(
                f"   '{topic}' ({len(docs)} files):\n"
                + "\n".join(f"      - {d.relative_to(docs_dir)}" for d in docs[:3])
                + (f"\n      ... and {len(docs) - 3} more" if len(docs) > 3 else "")
                for topic, docs in sorted(duplicate_groups.items(), key=lambda x: len(x[1]), reverse=True)[:5]
            )
            + (f"\n   ... and {len(duplicate_groups) - 5} more duplicate groups" if len(duplicate_groups) > 5 else "")
            + "\n   → Consolidate these into single comprehensive documents."
        )

    # Check 2: Find related docs that should be grouped
    by_category = defaultdict(list)
    for doc in active_docs:
        relative = doc.relative_to(docs_dir)
        category = relative.parts[0] if len(relative.parts) > 1 else "root"
        by_category[category].append(doc)

    # Analyze each category for consolidation opportunities
    for category, docs in by_category.items():
        if category in ["root", "planning"] and len(docs) > 5:
            warnings.append(
                f"Category '{category}/' has {len(docs)} files\n"
                f"   Many root-level files suggest need for better organization.\n"
                f"   Consider creating subdirectories or moving to appropriate categories."
            )

    # Check 2: Sparse directories (< 3 files)
    sparse_dirs = []
    for subdir in docs_dir.rglob("*"):
        if not subdir.is_dir() or any(skip in subdir.parts for skip in ALLOWED_SPARSE_DIRS):
            continue

        files = list(subdir.glob("*.md"))
        if 0 < len(files) < MIN_FILES_PER_DIR:
            sparse_dirs.append((subdir, len(files)))

    if sparse_dirs:
        issues.append(
            f"Sparse directories detected ({len(sparse_dirs)} total):\n"
            + "\n".join(f"   - {d.relative_to(docs_dir)} ({count} files)" for d, count in sparse_dirs[:5])
            + ("\n   ..." if len(sparse_dirs) > 5 else "")
            + "\n   Consider consolidating into parent or removing directory."
        )

    # Check 3: [COMPLETED] files outside archive/
    completed_files = [
        f for f in active_docs if any(tag in f.name for tag in ["[COMPLETED]", "[CLOSED]", "[RESOLVED]", "[ARCHIVED]"])
    ]

    if completed_files:
        issues.append(
            f"{len(completed_files)} completed files not archived:\n"
            + "\n".join(f"   - {f.relative_to(docs_dir)}" for f in completed_files[:5])
            + ("\n   ..." if len(completed_files) > 5 else "")
            + "\n   Move to archive/ or remove status prefix."
        )

    # Check 4: Duplicate concepts (heuristic: similar filenames)
    stems = [f.stem.lower().replace("_", " ").replace("-", " ") for f in active_docs]
    duplicates = [name for name, count in Counter(stems).items() if count > 1]

    if duplicates:
        warnings.append(
            "Potential duplicate concepts detected:\n"
            + "\n".join(f"   - {name} (appears {Counter(stems)[name]} times)" for name in duplicates[:3])
            + ("\n   ..." if len(duplicates) > 3 else "")
        )

    # Check 5: Empty directories
    empty_dirs = []
    for subdir in docs_dir.rglob("*"):
        if not subdir.is_dir() or any(skip in subdir.parts for skip in ALLOWED_SPARSE_DIRS):
            continue

        if not list(subdir.glob("*.md")):
            empty_dirs.append(subdir)

    if empty_dirs:
        issues.append(
            f"Empty directories detected ({len(empty_dirs)} total):\n"
            + "\n".join(f"   - {d.relative_to(docs_dir)}" for d in empty_dirs[:5])
            + ("\n   ..." if len(empty_dirs) > 5 else "")
            + "\n   Consider removing unused directories."
        )

    # Check 6: Redundant folder hierarchy (directories with only one subdirectory)
    redundant_dirs = []
    for subdir in docs_dir.rglob("*"):
        if not subdir.is_dir() or any(skip in subdir.parts for skip in ALLOWED_SPARSE_DIRS):
            continue

        # Get immediate children
        children = list(subdir.iterdir())
        subdirs = [c for c in children if c.is_dir() and not c.name.startswith(".")]
        files = [c for c in children if c.is_file() and c.suffix == ".md"]

        # If directory has exactly 1 subdirectory and no files, it's redundant
        if len(subdirs) == 1 and len(files) == 0:
            redundant_dirs.append((subdir, subdirs[0]))

    if redundant_dirs:
        warnings.append(
            f"Redundant folder hierarchy detected ({len(redundant_dirs)} cases):\n"
            + "\n".join(
                f"   {parent.relative_to(docs_dir)}/ → {child.relative_to(docs_dir)}/"
                for parent, child in redundant_dirs[:5]
            )
            + ("\n   ..." if len(redundant_dirs) > 5 else "")
            + "\n   Consider merging: move child contents to parent and remove child directory."
        )

    # Generate report
    if report:
        print("=" * 70)
        print("DOCUMENTATION QUALITY REPORT")
        print("=" * 70)
        print(f"\nTotal markdown files: {len(md_files)}")
        print(f"Active docs: {len(active_docs)}")
        print(f"Archived docs: {len(md_files) - len(active_docs)}")
        print(f"Directories: {len([d for d in docs_dir.rglob('*') if d.is_dir()])}")

        # Category breakdown
        print("\n📂 Documentation by Category:")
        for category, docs in sorted(by_category.items(), key=lambda x: len(x[1]), reverse=True):
            total_size = sum(d.stat().st_size for d in docs) / 1024
            print(f"   {category:20s}: {len(docs):3d} files ({total_size:6.1f} KB)")

        # Consolidation opportunities
        print("\n🔄 Consolidation Opportunities:")
        print(f"   Duplicate topic groups  : {len(duplicate_groups)}")
        print(
            f"   [COMPLETED] to archive  : {len([d for d in active_docs if '[COMPLETED]' in d.name or '[CLOSED]' in d.name or '[RESOLVED]' in d.name])}"
        )
        print(f"   Sparse directories      : {len(sparse_dirs)}")
        print(f"   Empty directories       : {len(empty_dirs)}")

        print(f"\nQuality: {'✅ GOOD' if not issues else '⚠️  NEEDS CONSOLIDATION'}")
        print()

    # Report issues
    if issues or warnings:
        if issues:
            print("❌ Documentation Structure Issues:")
            print("=" * 70)
            for i, issue in enumerate(issues, 1):
                print(f"\n{i}. {issue}")

        if warnings:
            print("\n⚠️  Warnings:")
            print("=" * 70)
            for i, warning in enumerate(warnings, 1):
                print(f"\n{i}. {warning}")

        print("\n" + "=" * 70)

        if not fix:
            print("\nTo auto-fix some issues: python scripts/check_docs_structure.py --fix")
            print("For detailed report: python scripts/check_docs_structure.py --report")

        return 1 if issues else 0
    else:
        print("✅ Documentation quality is good!")
        if report:
            print(f"\n   Active docs: {len(active_docs)}")
            print("   No duplicate topic groups")
            print(f"   Well-organized directories: All have ≥ {MIN_FILES_PER_DIR} files")
            print("   No completed files outside archive")
        return 0


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Check documentation structure against project standards")
    parser.add_argument("--fix", action="store_true", help="Auto-fix issues where possible")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    args = parser.parse_args()

    docs_dir = Path(__file__).parent.parent / "docs"

    if not docs_dir.exists():
        print(f"Error: docs/ directory not found at {docs_dir}")
        return 1

    return check_docs_structure(docs_dir, fix=args.fix, report=args.report)


if __name__ == "__main__":
    sys.exit(main())
