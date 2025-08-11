#!/usr/bin/env python3
"""
MFG_PDE Environment Verification Script

Run this script to verify your Anaconda environment is properly configured
for MFG_PDE with NumPy 2.0+ compatibility.
"""


def verify_environment():
    """Comprehensive environment verification."""
    print("🔍 MFG_PDE Environment Verification")
    print("=" * 50)

    # Check Python version
    import sys

    print(f"✅ Python: {sys.version.split()[0]} ({sys.executable})")

    # Check core packages
    try:
        import numpy as np

        print(f"✅ NumPy: {np.__version__}")

        import scipy

        print(f"✅ SciPy: {scipy.__version__}")

        import matplotlib

        print(f"✅ Matplotlib: {matplotlib.__version__}")

    except ImportError as e:
        print(f"❌ Missing core package: {e}")
        return False

    # Check NumPy compatibility
    print("\n🔧 NumPy Compatibility Check")
    print("-" * 30)

    try:
        from mfg_pde.utils.numpy_compat import get_numpy_info, trapezoid

        info = get_numpy_info()
        print(f"✅ NumPy version: {info['numpy_version']}")
        print(f"✅ Has trapezoid: {info['has_trapezoid']}")
        print(f"✅ Has trapz: {info['has_trapz']}")
        print(f"✅ Has SciPy trapezoid: {info['has_scipy_trapezoid']}")
        print(f"✅ Recommended method: {info['recommended_method']}")
        print(f"✅ NumPy 2.0+ ready: {info['is_numpy_2_plus']}")

        # Test trapezoid function
        x = np.linspace(0, 1, 11)
        y = x**2
        result = trapezoid(y, x=x)
        expected = 1 / 3
        error = abs(result - expected)
        print(f"✅ Integration test: {result:.6f} (error: {error:.2e})")

    except Exception as e:
        print(f"❌ NumPy compatibility issue: {e}")
        return False

    # Check MFG_PDE installation
    print("\n🎯 MFG_PDE Installation Check")
    print("-" * 35)

    try:
        import mfg_pde

        print(f"✅ MFG_PDE version: {mfg_pde.__version__}")

        # Test core functionality
        from mfg_pde.alg.variational_solvers import VariationalMFGSolver
        from mfg_pde.core.lagrangian_mfg_problem import create_quadratic_lagrangian_mfg

        problem = create_quadratic_lagrangian_mfg(xmin=0, xmax=1, Nx=10, T=0.1, Nt=5)
        solver = VariationalMFGSolver(problem)

        print("✅ Lagrangian MFG problem creation: OK")
        print("✅ Variational solver creation: OK")

        # Test trapezoid usage in solver
        initial_guess = solver.create_initial_guess('gaussian')
        mass_error = solver.compute_mass_conservation_error(initial_guess)
        print(f"✅ Mass conservation test: {mass_error:.2e}")

    except Exception as e:
        print(f"❌ MFG_PDE functionality issue: {e}")
        return False

    # Check optional packages
    print("\n🔧 Optional Packages")
    print("-" * 20)

    optional_packages = [
        ('plotly', 'Interactive plotting'),
        ('bokeh', 'Advanced visualizations'),
        ('jax', 'Automatic differentiation'),
        ('numba', 'JIT compilation'),
        ('pandas', 'Data manipulation'),
        ('polars', 'Fast data processing'),
        ('jupyter', 'Notebook environment'),
    ]

    for package, description in optional_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package}: {version} ({description})")
        except ImportError:
            print(f"⚪ {package}: Not installed ({description})")

    print("\n🎉 Environment Verification Complete!")
    print("✅ Your environment is ready for MFG_PDE development")
    return True


if __name__ == "__main__":
    success = verify_environment()
    exit(0 if success else 1)
