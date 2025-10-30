#!/usr/bin/env python3
"""
Script to update use_monotone_constraints to qp_optimization_level API.

This script:
1. Removes use_monotone_constraints parameter usage
2. Updates old QP level names ("basic" -> "always", "smart" -> "auto")
"""

import re
from pathlib import Path

# Files to update (remaining Python files in MFG_PDE)
files_to_update = [
    "benchmarks/solver_comparisons/fixed_method_comparison.py",
    "benchmarks/solver_comparisons/comprehensive_final_evaluation.py",
]


def update_file(file_path):
    """Update a single file."""
    path = Path(file_path)
    if not path.exists():
        print(f"SKIP: {file_path} (not found)")
        return False

    content = path.read_text()
    original_content = content

    # Pattern 1: Remove use_monotone_constraints=True, (with comma)
    content = re.sub(r"use_monotone_constraints=True,\s*\n\s*", "", content)

    # Pattern 2: Remove use_monotone_constraints=True (no comma)
    content = re.sub(r",\s*use_monotone_constraints=True", "", content)

    # Pattern 3: Remove use_monotone_constraints=False, (with comma)
    content = re.sub(r"use_monotone_constraints=False,\s*\n\s*", "", content)

    # Pattern 4: Remove use_monotone_constraints=False (no comma)
    content = re.sub(r",\s*use_monotone_constraints=False", "", content)

    # Pattern 5: Update qp_optimization_level="basic" -> "always"
    content = re.sub(r'qp_optimization_level="basic"', 'qp_optimization_level="always"', content)

    # Pattern 6: Update qp_optimization_level="smart" -> "auto"
    content = re.sub(r'qp_optimization_level="smart"', 'qp_optimization_level="auto"', content)

    if content != original_content:
        path.write_text(content)
        print(f"UPDATED: {file_path}")
        return True
    else:
        print(f"NO CHANGE: {file_path}")
        return False


if __name__ == "__main__":
    print("Updating QP API usage in MFG_PDE repository...")
    print("=" * 60)

    updated_count = 0
    for file_path in files_to_update:
        if update_file(file_path):
            updated_count += 1

    print("=" * 60)
    print(f"Updated {updated_count}/{len(files_to_update)} files")
    print("\nDone!")
