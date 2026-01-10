import importlib
import pkgutil

import mfg_pde


def check_imports(package):
    """Recursively import all submodules to check for circular imports."""
    prefix = package.__name__ + "."

    # Track successful and failed imports
    successful = []
    failed = []

    for _, name, _ispkg in pkgutil.walk_packages(package.__path__, prefix):
        try:
            print(f"Importing {name}...", end="", flush=True)
            importlib.import_module(name)
            print(" ✅")
            successful.append(name)
        except ImportError as e:
            print(f" ⚠️  Skipped (Optional Dependency): {e}")
            failed.append((name, str(e)))
        except Exception as e:
            print(f" ❌ Error importing {name}: {e}")
            failed.append((name, str(e)))

    print("\n" + "=" * 50)
    print("IMPORT CHECK SUMMARY")
    print("=" * 50)
    print(f"Total modules checked: {len(successful) + len(failed)}")
    print(f"Successful imports: {len(successful)}")
    print(f"Failed imports: {len(failed)}")

    if failed:
        print("\nFailures:")
        for name, error in failed:
            print(f"  - {name}: {error}")


if __name__ == "__main__":
    print("Checking for circular imports in mfg_pde...")
    try:
        check_imports(mfg_pde)
    except Exception as e:
        print(f"Critical error during check: {e}")
    print("Import check complete.")
