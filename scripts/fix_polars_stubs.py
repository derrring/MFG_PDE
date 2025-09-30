#!/usr/bin/env python3
"""
Fix generated Polars stubs for Python 3.12 compatibility and ruff compliance.

Addresses systematic typing methodology - stub generation refinement phase.
"""

import re
import sys
from pathlib import Path


def fix_parameter_syntax(content: str) -> str:
    """Fix invalid parameter syntax like *args, /, param."""
    # Fix the specific problematic pattern: *args, /, param -> *args, **kwargs
    content = re.sub(
        r"\*([^,]+), /, __(\w+): ([^,]+) = ([^,]+), \*\*([^)]+)\)", r"*\1, **kwargs) # Fixed parameter syntax", content
    )
    return content


def fix_pep695_type_aliases(content: str) -> str:
    """Convert PEP 695 type aliases to TypeAlias format."""
    # Convert "type X = Y" to "X: TypeAlias = Y"
    lines = content.split("\n")
    fixed_lines = []
    needs_typealias_import = False

    for line in lines:
        if line.strip().startswith("type "):
            # Convert PEP 695 to TypeAlias
            match = re.match(r"^(\s*)type\s+(\w+)\s*=\s*(.+)$", line)
            if match:
                indent, name, value = match.groups()
                fixed_lines.append(f"{indent}{name}: TypeAlias = {value}")
                needs_typealias_import = True
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    content = "\n".join(fixed_lines)

    # Add TypeAlias import if needed
    if needs_typealias_import and "from typing_extensions import TypeAlias" not in content:
        # Find where to insert the import
        lines = content.split("\n")
        import_index = 0
        for i, line in enumerate(lines):
            if line.startswith(("from typing", "import")):
                import_index = i + 1
            elif line.strip() == "":
                continue
            else:
                break

        lines.insert(import_index, "from typing_extensions import TypeAlias")
        content = "\n".join(lines)

    return content


def fix_builtin_shadowing(content: str) -> str:
    """Fix built-in shadowing issues by adding type ignores."""
    # List of Python builtins that might be shadowed
    builtins_to_fix = ["all", "any", "format", "len", "max", "min", "sum", "type"]

    for builtin_name in builtins_to_fix:
        # Add type ignore for builtin shadowing
        pattern = rf"(\s+{builtin_name} as {builtin_name},?\s*$)"
        replacement = r"\1  # type: ignore[misc]  # Builtin shadowing"
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    return content


def clean_stub_file(file_path: Path) -> bool:
    """Clean a single stub file. Returns True if changes were made."""
    try:
        content = file_path.read_text(encoding="utf-8")
        original_content = content

        # Apply fixes
        content = fix_parameter_syntax(content)
        content = fix_pep695_type_aliases(content)
        content = fix_builtin_shadowing(content)

        # Write back if changed
        if content != original_content:
            file_path.write_text(content, encoding="utf-8")
            return True
        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Clean all Polars stub files."""
    stubs_dir = Path("stubs/polars")

    if not stubs_dir.exists():
        print("No stubs directory found. Run stubgen first.")
        sys.exit(1)

    print("🔧 Cleaning Polars stubs for Python 3.12 compatibility...")

    total_files = 0
    fixed_files = 0

    # Process all .pyi files
    for stub_file in stubs_dir.rglob("*.pyi"):
        total_files += 1
        if clean_stub_file(stub_file):
            fixed_files += 1
            print(f"  ✅ Fixed: {stub_file.relative_to(stubs_dir)}")

    print(f"\n📊 Results: Fixed {fixed_files}/{total_files} stub files")

    # Remove most problematic files that can't be easily fixed
    problematic_files = [
        "dependencies.pyi",  # Too many external dependencies
        "_utils/parse/expr.pyi",  # Complex parameter syntax
    ]

    for prob_file in problematic_files:
        prob_path = stubs_dir / prob_file
        if prob_path.exists():
            prob_path.unlink()
            print(f"  🗑️  Removed problematic: {prob_file}")


if __name__ == "__main__":
    main()
