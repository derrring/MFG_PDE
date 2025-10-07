#!/usr/bin/env python3
"""
Manual script to check and update ruff version across the repository.

Usage:
    python scripts/update_ruff_version.py --check    # Check for updates only
    python scripts/update_ruff_version.py --update   # Check and apply updates
    python scripts/update_ruff_version.py --force VERSION  # Force specific version

Examples:
    python scripts/update_ruff_version.py --check
    python scripts/update_ruff_version.py --update
    python scripts/update_ruff_version.py --force 0.14.0
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import requests


def get_current_version() -> str:
    """Get current ruff version from .pre-commit-config.yaml."""
    config_path = Path(".pre-commit-config.yaml")

    if not config_path.exists():
        print("❌ Error: .pre-commit-config.yaml not found")
        sys.exit(1)

    content = config_path.read_text()

    # Find ruff version
    match = re.search(r"astral-sh/ruff-pre-commit\s+rev:\s*v([0-9.]+)", content)

    if not match:
        print("❌ Error: Could not find ruff version in .pre-commit-config.yaml")
        sys.exit(1)

    return match.group(1)


def get_latest_version() -> str:
    """Get latest ruff version from GitHub API."""
    try:
        response = requests.get("https://api.github.com/repos/astral-sh/ruff/releases/latest", timeout=10)
        response.raise_for_status()
        version = response.json()["tag_name"].lstrip("v")
        return version
    except Exception as e:
        print(f"❌ Error fetching latest version: {e}")
        sys.exit(1)


def compare_versions(current: str, latest: str) -> str:
    """Compare version strings and return status."""
    current_parts = [int(x) for x in current.split(".")]
    latest_parts = [int(x) for x in latest.split(".")]

    if current_parts < latest_parts:
        return "outdated"
    elif current_parts == latest_parts:
        return "current"
    else:
        return "ahead"


def update_files(new_version: str) -> None:
    """Update ruff version in configuration files."""
    files_updated = []

    # Update .pre-commit-config.yaml
    config_path = Path(".pre-commit-config.yaml")
    content = config_path.read_text()
    updated = re.sub(r"(astral-sh/ruff-pre-commit\s+rev:\s*)v[0-9.]+", rf"\1v{new_version}", content)

    if updated != content:
        config_path.write_text(updated)
        files_updated.append(".pre-commit-config.yaml")

    # Update .github/workflows/modern_quality.yml
    workflow_path = Path(".github/workflows/modern_quality.yml")

    if workflow_path.exists():
        content = workflow_path.read_text()
        updated = re.sub(r"ruff==[0-9.]+", f"ruff=={new_version}", content)

        if updated != content:
            workflow_path.write_text(updated)
            files_updated.append(".github/workflows/modern_quality.yml")

    return files_updated


def run_formatting() -> bool:
    """Run ruff format on the codebase."""
    try:
        print("\n📝 Running ruff format...")
        result = subprocess.run(["ruff", "format", "mfg_pde/"], capture_output=True, text=True, check=False)

        if result.returncode == 0:
            print("✅ Formatting complete")
            return True
        else:
            print(f"⚠️  Formatting had issues:\n{result.stderr}")
            return False
    except FileNotFoundError:
        print("⚠️  ruff not found in PATH, skipping formatting")
        return False


def main():
    parser = argparse.ArgumentParser(description="Update ruff version across repository")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Check for updates only")
    group.add_argument("--update", action="store_true", help="Check and apply updates")
    group.add_argument("--force", metavar="VERSION", help="Force update to specific version")

    args = parser.parse_args()

    print("🔍 Ruff Version Manager\n")

    # Get current version
    current = get_current_version()
    print(f"📌 Current version: v{current}")

    if args.force:
        # Force specific version
        target_version = args.force.lstrip("v")
        print(f"🎯 Forcing update to: v{target_version}")

        files_updated = update_files(target_version)

        if files_updated:
            print(f"\n✅ Updated {len(files_updated)} file(s):")
            for f in files_updated:
                print(f"   - {f}")

            run_formatting()

            print("\n✨ Done! Next steps:")
            print("   1. Review changes: git diff")
            print("   2. Test locally: pytest tests/")
            print("   3. Run pre-commit: pre-commit run --all-files")
            print("   4. Commit changes: git commit -am 'chore: Update ruff to vX.Y.Z'")
        else:
            print("\n⚠️  No files needed updating")

    else:
        # Check for latest version
        latest = get_latest_version()
        print(f"🆕 Latest version:  v{latest}")

        status = compare_versions(current, latest)

        if status == "current":
            print("\n✅ Ruff is up to date!")
            sys.exit(0)
        elif status == "ahead":
            print("\n⚠️  You're ahead of the latest release")
            print("   (Possibly using a pre-release or beta version)")
            sys.exit(0)
        else:
            # Outdated
            print(f"\n🆕 Update available: v{current} → v{latest}")

            if args.check:
                print("\n📋 To update, run:")
                print("   python scripts/update_ruff_version.py --update")
                sys.exit(0)

            if args.update:
                # Fetch release notes
                try:
                    response = requests.get(
                        f"https://api.github.com/repos/astral-sh/ruff/releases/tags/v{latest}",
                        timeout=10,
                    )
                    if response.ok:
                        print(f"\n📰 Release notes: {response.json()['html_url']}")
                except Exception:
                    pass

                confirm = input("\n❓ Proceed with update? [y/N] ").lower()

                if confirm != "y":
                    print("❌ Update cancelled")
                    sys.exit(0)

                files_updated = update_files(latest)

                if files_updated:
                    print(f"\n✅ Updated {len(files_updated)} file(s):")
                    for f in files_updated:
                        print(f"   - {f}")

                    run_formatting()

                    print("\n✨ Done! Next steps:")
                    print("   1. Review changes: git diff")
                    print("   2. Test locally: pytest tests/")
                    print("   3. Run pre-commit: pre-commit run --all-files")
                    print(f"   4. Commit: git commit -am 'chore: Update ruff to v{latest}'")
                else:
                    print("\n⚠️  No files needed updating")


if __name__ == "__main__":
    main()
