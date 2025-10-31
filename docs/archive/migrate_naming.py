"""
Systematic naming convention migration for MFG_PDE solvers.

Replaces mathematical notation with full English names:
- self.Nx/Ny/Nz → self.num_grid_points_x/y/z
- self.Dx/Dy/Dz → self.grid_spacing_x/y/z
"""

import re
from pathlib import Path


def migrate_file(filepath: Path) -> tuple[int, list[str]]:
    """Migrate naming in a single file. Returns (num_replacements, changes_made)."""

    with open(filepath, 'r') as f:
        content = f.read()

    original = content
    changes = []

    # Replacement mapping (order matters!)
    replacements = [
        # Grid points
        (r'\bself\.Nx\b', 'self.num_grid_points_x'),
        (r'\bself\.Ny\b', 'self.num_grid_points_y'),
        (r'\bself\.Nz\b', 'self.num_grid_points_z'),

        # Grid spacing
        (r'\bself\.Dx\b', 'self.grid_spacing_x'),
        (r'\bself\.Dy\b', 'self.grid_spacing_y'),
        (r'\bself\.Dz\b', 'self.grid_spacing_z'),
    ]

    total_count = 0
    for pattern, replacement in replacements:
        count = len(re.findall(pattern, content))
        if count > 0:
            content = re.sub(pattern, replacement, content)
            changes.append(f"  {pattern} → {replacement}: {count} occurrences")
            total_count += count

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)

    return total_count, changes


if __name__ == "__main__":
    # Files to migrate
    files = [
        "mfg_pde/alg/numerical/hjb_solvers/hjb_weno.py",
        "mfg_pde/alg/numerical/hjb_solvers/base_hjb.py",
        "mfg_pde/alg/numerical/hjb_solvers/hjb_fdm_multid.py",
        "mfg_pde/alg/numerical/fp_solvers/fp_fdm.py",
        "mfg_pde/alg/numerical/fp_solvers/fp_fdm_multid.py",
    ]

    print("=" * 70)
    print("  Naming Convention Migration")
    print("=" * 70)
    print()

    total_changes = 0
    for filepath_str in files:
        filepath = Path(filepath_str)
        if not filepath.exists():
            print(f"⚠️  {filepath}: NOT FOUND, skipping")
            continue

        count, changes = migrate_file(filepath)
        if count > 0:
            print(f"✓ {filepath.name}: {count} replacements")
            for change in changes:
                print(change)
            print()
            total_changes += count
        else:
            print(f"  {filepath.name}: No changes needed")

    print("=" * 70)
    print(f"Total replacements: {total_changes}")
    print("=" * 70)
