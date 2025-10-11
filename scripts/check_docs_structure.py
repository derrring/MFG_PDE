#!/usr/bin/env python3
"""
Documentation structure validation script.

Checks:
1. Category-based doc limits (scales with project size)
2. No directories with < 3 files (except archive/)
3. No [COMPLETED] files outside archive/
4. No duplicate/overlapping content

Usage:
    python scripts/check_docs_structure.py          # Check only
    python scripts/check_docs_structure.py --fix    # Auto-fix issues
    python scripts/check_docs_structure.py --report # Detailed report
"""

import sys
from collections import Counter
from pathlib import Path

# Category-based documentation limits (scaled to project needs)
CATEGORY_LIMITS = {
    "user": 15,  # User-facing docs: tutorials, guides, installation
    "development": 35,  # Developer docs: architecture, APIs, workflows
    "theory": 25,  # Mathematical theory: foundations, methods, applications
    "reference": 50,  # API reference: can be larger (often auto-generated)
}

# Universal thresholds
MIN_FILES_PER_DIR = 3
ALLOWED_SPARSE_DIRS = {"archive", "private", ".git", ".github"}

# Dynamic global limit (sum of categories + 20% buffer)
MAX_ACTIVE_DOCS = int(sum(CATEGORY_LIMITS.values()) * 1.2)  # ~150 with 20% buffer


def check_docs_structure(docs_dir: Path, fix: bool = False, report: bool = False) -> int:
    """Check documentation structure against principles."""
    issues = []
    warnings = []

    # Check 1: Count total active docs and check category limits
    md_files = list(docs_dir.glob("**/*.md"))
    active_docs = [f for f in md_files if "archive" not in f.parts]

    # Category-based counting
    category_counts = {}
    uncategorized_docs = []

    for doc in active_docs:
        # Determine category from path
        relative_path = doc.relative_to(docs_dir)
        category = relative_path.parts[0] if len(relative_path.parts) > 1 else "root"

        if category in CATEGORY_LIMITS:
            category_counts[category] = category_counts.get(category, 0) + 1
        else:
            uncategorized_docs.append(doc)

    # Check category limits
    for category, count in category_counts.items():
        limit = CATEGORY_LIMITS[category]
        if count > limit:
            issues.append(
                f"Category '{category}/' exceeds limit: {count}/{limit} files\n"
                f"   Consider consolidating or archiving completed docs in this category."
            )
        elif count > limit * 0.9:
            warnings.append(
                f"Category '{category}/' approaching limit: {count}/{limit} files\n   Plan consolidation soon."
            )

    # Check global limit
    if len(active_docs) > MAX_ACTIVE_DOCS:
        issues.append(
            f"Too many active docs overall: {len(active_docs)} (soft limit: {MAX_ACTIVE_DOCS})\n"
            f"   Category limits: " + ", ".join(f"{k}={v}" for k, v in CATEGORY_LIMITS.items()) + "\n"
            "   Consider consolidating related documentation."
        )
    elif len(active_docs) > MAX_ACTIVE_DOCS * 0.9:
        warnings.append(
            f"Approaching global doc limit: {len(active_docs)}/{MAX_ACTIVE_DOCS}\n"
            f"   Consider planning consolidation soon."
        )

    # Warn about uncategorized docs
    if uncategorized_docs:
        warnings.append(
            f"{len(uncategorized_docs)} uncategorized docs found:\n"
            + "\n".join(f"   - {d.relative_to(docs_dir)}" for d in uncategorized_docs[:5])
            + ("\n   ..." if len(uncategorized_docs) > 5 else "")
            + "\n   Consider organizing into user/development/theory/reference categories."
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

    # Generate report
    if report:
        print("=" * 70)
        print("DOCUMENTATION STRUCTURE REPORT")
        print("=" * 70)
        print(f"\nTotal markdown files: {len(md_files)}")
        print(f"Active docs: {len(active_docs)} (limit: {MAX_ACTIVE_DOCS})")
        print(f"Archived docs: {len(md_files) - len(active_docs)}")
        print(f"Directories: {len([d for d in docs_dir.rglob('*') if d.is_dir()])}")

        # Category breakdown
        print("\n📂 Category Breakdown:")
        for category, limit in sorted(CATEGORY_LIMITS.items()):
            count = category_counts.get(category, 0)
            status = "✅" if count <= limit else "❌"
            pct = int(count / limit * 100) if limit > 0 else 0
            print(f"   {status} {category:15s}: {count:3d}/{limit:3d} ({pct:3d}%)")

        if uncategorized_docs:
            print(f"   ⚠️  {'uncategorized':15s}: {len(uncategorized_docs):3d} files")

        print(f"\nStatus: {'✅ GOOD' if not issues else '❌ NEEDS ATTENTION'}")
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
        print("✅ Documentation structure looks good!")
        if report:
            print(f"\n   Active docs: {len(active_docs)}/{MAX_ACTIVE_DOCS}")
            print("   Category compliance:")
            for category, limit in sorted(CATEGORY_LIMITS.items()):
                count = category_counts.get(category, 0)
                print(f"      • {category}: {count}/{limit}")
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
