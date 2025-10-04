#!/usr/bin/env python3
"""
Script to update package imports from old alg structure to new paradigm-based structure.

This script systematically updates import statements in examples and tests to use
the new paradigm-based algorithm structure.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Import mapping from old paths to new paradigm-based paths
IMPORT_MAPPINGS = {
    # HJB Solvers - numerical paradigm
    "from mfg_pde.alg.hjb_solvers import": "from mfg_pde.alg.numerical.hjb_solvers import",
    "from mfg_pde.alg.hjb_solvers.": "from mfg_pde.alg.numerical.hjb_solvers.",

    # FP Solvers - numerical paradigm
    "from mfg_pde.alg.fp_solvers import": "from mfg_pde.alg.numerical.fp_solvers import",
    "from mfg_pde.alg.fp_solvers.": "from mfg_pde.alg.numerical.fp_solvers.",

    # MFG Solvers - numerical paradigm
    "from mfg_pde.alg.mfg_solvers import": "from mfg_pde.alg.numerical.mfg_solvers import",
    "from mfg_pde.alg.mfg_solvers.": "from mfg_pde.alg.numerical.mfg_solvers.",

    # Variational Solvers - optimization paradigm
    "from mfg_pde.alg.variational_solvers import": "from mfg_pde.alg.optimization.variational_solvers import",
    "from mfg_pde.alg.variational_solvers.": "from mfg_pde.alg.optimization.variational_solvers.",

    # Neural Solvers - neural paradigm
    "from mfg_pde.alg.neural_solvers import": "from mfg_pde.alg.neural.pinn_solvers import",
    "from mfg_pde.alg.neural_solvers.": "from mfg_pde.alg.neural.pinn_solvers.",

    # Direct solver imports that might be used
    "from mfg_pde.alg.particle_collocation_solver": "from mfg_pde.alg.numerical.mfg_solvers.particle_collocation_solver",
}

# Preferred paradigm-level imports
PARADIGM_IMPORTS = {
    # Suggest using paradigm-level imports for common cases
    "HJBFDMSolver": "from mfg_pde.alg.numerical import HJBFDMSolver",
    "HJBWenoSolver": "from mfg_pde.alg.numerical import HJBWenoSolver",
    "VariationalMFGSolver": "from mfg_pde.alg.optimization import VariationalMFGSolver",
    "MFGPINNSolver": "from mfg_pde.alg.neural import MFGPINNSolver",
}

def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in directory and subdirectories."""
    return list(directory.rglob("*.py"))

def update_imports_in_file(file_path: Path, dry_run: bool = True) -> Tuple[int, List[str]]:
    """Update imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes = []

        # Apply import mappings
        for old_import, new_import in IMPORT_MAPPINGS.items():
            if old_import in content:
                content = content.replace(old_import, new_import)
                changes.append(f"  {old_import} → {new_import}")

        # Count total changes
        num_changes = len(changes)

        if num_changes > 0 and not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

        return num_changes, changes

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, []

def update_directory(directory: Path, dry_run: bool = True) -> None:
    """Update all Python files in directory."""
    print(f"\n{'🔍 ANALYZING' if dry_run else '✏️ UPDATING'} {directory}")
    print("=" * 60)

    python_files = find_python_files(directory)
    total_files = len(python_files)
    files_changed = 0
    total_changes = 0

    for file_path in python_files:
        num_changes, changes = update_imports_in_file(file_path, dry_run)

        if num_changes > 0:
            files_changed += 1
            total_changes += num_changes
            relative_path = file_path.relative_to(directory.parent)
            print(f"\n📝 {relative_path}")
            for change in changes:
                print(change)

    print(f"\n📊 Summary:")
    print(f"  Files checked: {total_files}")
    print(f"  Files needing updates: {files_changed}")
    print(f"  Total import changes: {total_changes}")

    if dry_run and files_changed > 0:
        print(f"\n💡 Run with dry_run=False to apply changes")

def main():
    """Main function to update package imports."""
    print("🏗️ Package Import Update Script")
    print("Converting from old alg structure to new paradigm-based structure")

    root_dir = Path(__file__).parent.parent

    # Update examples
    examples_dir = root_dir / "examples"
    if examples_dir.exists():
        update_directory(examples_dir, dry_run=False)  # Apply changes

    # Update tests
    tests_dir = root_dir / "tests"
    if tests_dir.exists():
        update_directory(tests_dir, dry_run=False)  # Apply changes

    print("\n" + "="*60)
    print("✅ Import updates applied successfully!")
    print("🎯 Next steps:")
    print("1. Test updated examples and tests")
    print("2. Create multi-paradigm demonstration examples")
    print("3. Consider paradigm-level imports for cleaner code")

if __name__ == "__main__":
    main()
