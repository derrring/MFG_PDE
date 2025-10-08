#!/usr/bin/env python3
"""
HDF5 Save/Load Demonstration

This example demonstrates how to save and load MFG solver results using HDF5 format.
HDF5 is the recommended format for solver outputs due to:
- Native support for multidimensional NumPy arrays
- Efficient compression
- Rich metadata capabilities
- Scientific computing standard

Usage:
    python examples/basic/hdf5_save_load_demo.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.utils.solver_result import SolverResult

# Check HDF5 availability
try:
    from mfg_pde.utils.io import HDF5_AVAILABLE
except ImportError:
    HDF5_AVAILABLE = False

if not HDF5_AVAILABLE:
    print("ERROR: h5py not installed. Install with: pip install h5py")
    exit(1)


# ============================================================================
# Main Demonstration
# ============================================================================


def create_mock_solver_result():
    """Create mock solver result for demonstration purposes."""
    Nt, Nx = 51, 101
    T = 1.0
    x_min, x_max = 0.0, 1.0

    # Create mock solution arrays (simulated diffusion + advection)
    x = np.linspace(x_min, x_max, Nx)
    t = np.linspace(0, T, Nt)
    X, T_grid = np.meshgrid(x, t)

    # Value function U(t,x): parabolic terminal condition propagated backward
    U = (1 - T_grid) * 0.5 * (X - 0.5) ** 2 + T_grid * 0.1

    # Density M(t,x): Gaussian that diffuses over time
    sigma_init = 0.1
    sigma_t = sigma_init + 0.2 * T_grid
    M = np.exp(-0.5 * ((X - 0.5) / sigma_t) ** 2) / (sigma_t * np.sqrt(2 * np.pi))
    M = M / M.sum(axis=1, keepdims=True)  # Normalize each time slice

    # Mock convergence history
    error_history_U = np.array([1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4])
    error_history_M = np.array([2e-1, 8e-2, 2e-2, 8e-3, 2e-3, 8e-4, 2e-4])

    # Create SolverResult
    result = SolverResult(
        U=U,
        M=M,
        iterations=7,
        error_history_U=error_history_U,
        error_history_M=error_history_M,
        solver_name="MockFixedPointIterator",
        convergence_achieved=True,
        execution_time=3.142,
        metadata={"problem_type": "LinearQuadratic", "T": T, "Nx": Nx - 1, "Nt": Nt - 1},
    )

    return result, x, t


def main():
    """Demonstrate HDF5 save/load workflow."""
    print("=" * 70)
    print("HDF5 Save/Load Demonstration")
    print("=" * 70)

    # Setup output directory
    EXAMPLE_DIR = Path(__file__).parent
    OUTPUT_DIR = EXAMPLE_DIR.parent / "outputs" / "basic"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create mock solver result
    print("\n1. Creating mock solver result...")
    result, x_grid, t_grid = create_mock_solver_result()

    print(f"   Solution shape: {result.solution_shape}")
    print(f"   Converged: {result.convergence_achieved}")
    print(f"   Iterations: {result.iterations}")
    print(f"   Final error U: {result.final_error_U:.2e}")
    print(f"   Final error M: {result.final_error_M:.2e}")
    print(f"   Execution time: {result.execution_time:.3f}s")

    # Save to HDF5
    hdf5_file = OUTPUT_DIR / "mock_mfg_solution.h5"
    print(f"\n2. Saving result to HDF5: {hdf5_file}")

    # Option 1: Using SolverResult.save_hdf5() method (recommended)
    result.save_hdf5(hdf5_file, x_grid=x_grid, t_grid=t_grid)
    print(f"   Saved with compression (file size: {hdf5_file.stat().st_size / 1024:.2f} KB)")

    # Load from HDF5
    print(f"\n3. Loading result from HDF5: {hdf5_file}")
    loaded_result = SolverResult.load_hdf5(hdf5_file)

    print(f"   Loaded solver: {loaded_result.solver_name}")
    print(f"   Loaded shape: {loaded_result.solution_shape}")
    print(f"   Converged: {loaded_result.convergence_achieved}")
    print(f"   Iterations: {loaded_result.iterations}")

    # Verify data integrity
    print("\n4. Verifying data integrity...")
    u_diff = np.max(np.abs(result.U - loaded_result.U))
    m_diff = np.max(np.abs(result.M - loaded_result.M))

    print(f"   Max U difference: {u_diff:.2e} (should be ~0)")
    print(f"   Max M difference: {m_diff:.2e} (should be ~0)")

    if u_diff < 1e-10 and m_diff < 1e-10:
        print("   Data integrity verified!")
    else:
        print("   Warning: Data mismatch detected")

    # Get HDF5 file info
    print("\n5. HDF5 file information:")
    from mfg_pde.utils.io.hdf5_utils import get_hdf5_info

    info = get_hdf5_info(hdf5_file)
    print(f"   Format version: {info.get('format_version', 'unknown')}")
    print(f"   U shape: {info.get('U_shape', 'unknown')}")
    print(f"   M shape: {info.get('M_shape', 'unknown')}")
    if "x_grid_size" in info:
        print(f"   X grid size: {info['x_grid_size']}")
    if "t_grid_size" in info:
        print(f"   T grid size: {info['t_grid_size']}")

    # Visualize loaded result
    print("\n6. Creating visualization...")
    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    x_min, x_max = x_grid[0], x_grid[-1]
    T = t_grid[-1]

    # Plot U(t,x)
    im1 = axes[0].imshow(loaded_result.U, aspect="auto", origin="lower", extent=[x_min, x_max, 0, T], cmap="RdBu_r")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    axes[0].set_title("Value Function U(t,x)")
    plt.colorbar(im1, ax=axes[0])

    # Plot M(t,x)
    im2 = axes[1].imshow(loaded_result.M, aspect="auto", origin="lower", extent=[x_min, x_max, 0, T], cmap="viridis")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    axes[1].set_title("Density M(t,x)")
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()

    output_png = OUTPUT_DIR / "hdf5_demo_solution.png"
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    print(f"   Saved visualization: {output_png}")

    # Display if in interactive mode
    try:
        plt.show(block=False)
        plt.pause(0.1)
    except Exception:
        pass

    # Additional HDF5 features demonstration
    print("\n7. Additional HDF5 features:")

    # Option 2: Using low-level functions directly
    from mfg_pde.utils.io.hdf5_utils import load_solution, save_solution

    alt_file = OUTPUT_DIR / "mock_solution_alt.h5"

    metadata = result.metadata.copy()
    metadata["converged"] = result.convergence_achieved

    save_solution(result.U, result.M, metadata, alt_file)
    print(f"   Saved using low-level API: {alt_file}")

    _, _, meta_loaded = load_solution(alt_file)
    print(f"   Loaded metadata keys: {list(meta_loaded.keys())}")

    # Checkpoint/resume demonstration
    checkpoint_file = OUTPUT_DIR / "solver_checkpoint.h5"
    print("\n8. Checkpoint save/load demonstration:")

    from mfg_pde.utils.io.hdf5_utils import load_checkpoint, save_checkpoint

    checkpoint_state = {
        "U": result.U,
        "M": result.M,
        "iteration": result.iterations,
        "residuals_u": result.error_history_U,
        "residuals_m": result.error_history_M,
    }

    save_checkpoint(checkpoint_state, checkpoint_file)
    print(f"   Checkpoint saved: {checkpoint_file}")

    loaded_checkpoint = load_checkpoint(checkpoint_file)
    print(f"   Checkpoint loaded with keys: {list(loaded_checkpoint.keys())}")
    print(f"   Iteration at checkpoint: {loaded_checkpoint['iteration']}")

    print("\n" + "=" * 70)
    print("HDF5 demonstration complete!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  1. Use result.save_hdf5() for simple save/load")
    print("  2. HDF5 preserves full solution fidelity")
    print("  3. Automatic compression reduces file size")
    print("  4. Metadata and grids included automatically")
    print("  5. Checkpoint/resume for long computations")
    print("\nFiles created:")
    print(f"  - {hdf5_file}")
    print(f"  - {output_png}")
    print(f"  - {alt_file}")
    print(f"  - {checkpoint_file}")


if __name__ == "__main__":
    main()
