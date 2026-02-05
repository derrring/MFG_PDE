#!/usr/bin/env python3
"""
Test Visualization Modules

Quick test to verify both advanced_visualization and mathematical_visualization
modules are working correctly with proper LaTeX support and backend fallbacks.
"""

import os
import sys
from pathlib import Path

import numpy as np

# Add the parent directory to the path so we can import mfg_pde
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mfg_pde.utils import configure_logging, get_logger
from mfg_pde.visualization.interactive_plots import quick_2d_plot as create_mfg_solution_plot
from mfg_pde.visualization.mathematical_plots import create_mathematical_visualizer


def create_test_data():
    """Create simple test data for visualization."""
    x_grid = np.linspace(0, 1, 30)
    t_grid = np.linspace(0, 1, 20)
    X, T = np.meshgrid(x_grid, t_grid)

    # Simple value function and density
    U = np.sin(np.pi * X) * np.exp(-2 * T)
    M = np.exp(-10 * (X - 0.5) ** 2) / np.sqrt(np.pi / 10)

    return U.T, M.T, x_grid, t_grid


def test_advanced_visualization():
    """Test advanced visualization module."""
    logger = get_logger(__name__)
    logger.info("Testing advanced visualization module")

    _U, M, x_grid, t_grid = create_test_data()
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    try:
        # Test MFG solution plot
        create_mfg_solution_plot(x_grid, t_grid, M, title="Test MFG Solution")

        logger.info("Advanced visualization test successful")
        return True

    except Exception as e:
        logger.error(f"Advanced visualization test failed: {e}")
        return False


def test_mathematical_visualization():
    """Test mathematical visualization module."""
    logger = get_logger(__name__)
    logger.info("Testing mathematical visualization module")

    U, _M, x_grid, _t_grid = create_test_data()
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    try:
        # Test HJB analysis
        visualizer = create_mathematical_visualizer()
        du_dx = np.gradient(U, x_grid, axis=0)
        visualizer.visualize_hjb_equation(x_grid, U[:, 0], du_dx[:, 0], title="Test HJB Analysis")

        logger.info("Mathematical visualization test successful")
        return True

    except Exception as e:
        logger.error(f"Mathematical visualization test failed: {e}")
        return False


def test_quick_functions():
    """Test quick visualization functions."""
    logger = get_logger(__name__)
    logger.info("Testing quick visualization functions")

    U, M, x_grid, t_grid = create_test_data()
    Path("test_output")

    try:
        # Test quick plot solution
        create_mfg_solution_plot(x_grid, t_grid, M, title="Quick Solution")

        # Test quick HJB analysis
        visualizer = create_mathematical_visualizer()
        du_dx = np.gradient(U, x_grid, axis=0)
        visualizer.visualize_hjb_equation(x_grid, U[:, 0], du_dx[:, 0], title="Quick HJB")

        logger.info("Quick functions test successful")
        return True

    except Exception as e:
        logger.error(f"Quick functions test failed: {e}")
        return False


def main():
    """Run all visualization tests."""
    print("Testing MFG_PDE Visualization Modules")
    print("=" * 50)

    # Configure logging
    configure_logging(level="INFO", use_colors=True)
    logger = get_logger(__name__)

    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    # Run tests
    tests = [
        ("Advanced Visualization", test_advanced_visualization),
        ("Mathematical Visualization", test_mathematical_visualization),
        ("Quick Functions", test_quick_functions),
    ]

    results = {}
    for test_name, test_func in tests:
        logger.info(f"Running {test_name} test...")
        results[test_name] = test_func()

    # Summary
    print("\nTest Results:")
    print("-" * 30)
    for test_name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {test_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n[SUCCESS] All visualization tests passed!")
        print(f"Output files created in: {output_dir}")

        # List output files
        output_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.html"))
        if output_files:
            print("Generated files:")
            for file_path in sorted(output_files):
                print(f"  {file_path}")
    else:
        print("\n[WARNING] Some tests failed - check logs above")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
