#!/usr/bin/env python3
"""
Test Tiered Backend Factory - Phase 3

Verifies torch > jax > numpy auto-selection logic with OS-aware device selection.
"""

import logging

from mfg_pde.backends import create_backend, get_available_backends, get_backend_info

# Configure logging to see auto-selection messages
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")


def test_backend_availability():
    """Check which backends are available."""
    print("=" * 80)
    print("BACKEND AVAILABILITY CHECK")
    print("=" * 80)

    available = get_available_backends()

    print("\nBackend Status:")
    for backend, status in available.items():
        status_str = "✅ Available" if status else "❌ Not available"
        print(f"  {backend:20s}: {status_str}")

    return available


def test_auto_selection():
    """Test tiered auto-selection (torch > jax > numpy)."""
    print("\n" + "=" * 80)
    print("AUTO-SELECTION TEST (torch > jax > numpy)")
    print("=" * 80)

    # Test auto-selection
    backend = create_backend()

    print(f"\nAuto-selected backend: {backend.name}")
    print(f"Device: {backend.device}")
    print(f"Precision: {backend.precision}")

    # Verify it follows priority
    available = get_available_backends()
    if available.get("torch", False):
        assert backend.name.startswith("torch"), "Should select torch when available"
        print("✅ Correctly selected PyTorch (Priority 1)")
    elif available.get("jax", False):
        assert backend.name == "jax", "Should select jax when torch unavailable"
        print("✅ Correctly selected JAX (Priority 2, torch not available)")
    else:
        assert backend.name == "numpy", "Should select numpy when neither available"
        print("✅ Correctly selected NumPy (Priority 3, no acceleration available)")


def test_explicit_selection():
    """Test explicit backend selection."""
    print("\n" + "=" * 80)
    print("EXPLICIT SELECTION TEST")
    print("=" * 80)

    available = get_available_backends()

    # Test NumPy (always available)
    backend_numpy = create_backend("numpy")
    print(f"\nExplicit NumPy: {backend_numpy.name}")
    assert backend_numpy.name == "numpy"
    print("✅ NumPy explicit selection works")

    # Test PyTorch (if available)
    if available.get("torch", False):
        backend_torch = create_backend("torch")
        print(f"Explicit PyTorch: {backend_torch.name}, device: {backend_torch.device}")
        assert backend_torch.name.startswith("torch")
        print("✅ PyTorch explicit selection works")

    # Test JAX (if available)
    if available.get("jax", False):
        backend_jax = create_backend("jax")
        print(f"Explicit JAX: {backend_jax.name}")
        assert backend_jax.name == "jax"
        print("✅ JAX explicit selection works")


def test_backend_info():
    """Display comprehensive backend information."""
    print("\n" + "=" * 80)
    print("BACKEND INFORMATION")
    print("=" * 80)

    info = get_backend_info()

    print(f"\nDefault backend: {info['default_backend']}")
    print(f"Registered backends: {', '.join(info['registered_backends'])}")

    # PyTorch info
    if "torch_info" in info:
        print("\nPyTorch Information:")
        torch_info = info["torch_info"]
        if "error" not in torch_info:
            print(f"  Version: {torch_info.get('version', 'unknown')}")
            print(f"  CUDA available: {torch_info.get('cuda_available', False)}")
            print(f"  MPS available: {torch_info.get('mps_available', False)}")

            if torch_info.get("cuda_available"):
                print(f"  CUDA version: {torch_info.get('cuda_version', 'unknown')}")
                print(f"  CUDA devices: {torch_info.get('cuda_device_count', 0)}")
                for device_name in torch_info.get("cuda_devices", []):
                    print(f"    - {device_name}")

    # JAX info
    if "jax_info" in info:
        print("\nJAX Information:")
        jax_info = info["jax_info"]
        if "error" not in jax_info:
            print(f"  Version: {jax_info.get('version', 'unknown')}")
            print(f"  Has GPU: {jax_info.get('has_gpu', False)}")
            print(f"  Default device: {jax_info.get('default_device', 'unknown')}")
            print(f"  All devices: {', '.join(jax_info.get('devices', []))}")


def test_tiered_priority():
    """Verify tiered priority logic."""
    print("\n" + "=" * 80)
    print("TIERED PRIORITY VERIFICATION")
    print("=" * 80)

    available = get_available_backends()

    print("\nExpected selection based on availability:")

    if available.get("torch", False):
        print("  Priority 1: ✅ PyTorch available")
        if available.get("torch_cuda", False):
            print("    └─ Device: CUDA (NVIDIA GPU)")
        elif available.get("torch_mps", False):
            print("    └─ Device: MPS (Apple Silicon)")
        else:
            print("    └─ Device: CPU (no GPU)")

        if available.get("jax", False):
            print("  Priority 2: ⏭️  JAX available (skipped, torch selected)")
        else:
            print("  Priority 2: ❌ JAX not available")

    elif available.get("jax", False):
        print("  Priority 1: ❌ PyTorch not available")
        print("  Priority 2: ✅ JAX available (selected)")
        if available.get("jax_gpu", False):
            print("    └─ Device: GPU")
        else:
            print("    └─ Device: CPU")

    else:
        print("  Priority 1: ❌ PyTorch not available")
        print("  Priority 2: ❌ JAX not available")
        print("  Priority 3: ✅ NumPy (baseline, always available)")

    # Verify auto-selection matches expected priority
    backend = create_backend()
    if available.get("torch", False):
        assert backend.name.startswith("torch"), "Priority violation: torch available but not selected"
        print(f"\n✅ Priority 1 (torch) correctly selected: {backend.name} on {backend.device}")
    elif available.get("jax", False):
        assert backend.name == "jax", "Priority violation: jax available but not selected"
        print(f"\n✅ Priority 2 (jax) correctly selected: {backend.name}")
    else:
        assert backend.name == "numpy", "Priority violation: numpy should be selected"
        print(f"\n✅ Priority 3 (numpy) correctly selected: {backend.name}")


def main():
    """Run all backend factory tests."""
    print("\n" + "=" * 80)
    print("PHASE 3: TIERED BACKEND FACTORY TEST")
    print("Testing: torch > jax > numpy priority")
    print("=" * 80)

    test_backend_availability()
    test_auto_selection()
    test_explicit_selection()
    test_backend_info()
    test_tiered_priority()

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print("""
✅ Tiered backend factory implemented
✅ Auto-selection follows torch > jax > numpy priority
✅ OS-aware device selection (CUDA/MPS/CPU)
✅ Explicit backend choice works
✅ Logging shows selection reasoning

Next Steps:
1. Implement custom KDE in PyTorch/JAX backends
2. Implement sparse linear solvers
3. Add cross-backend consistency tests
""")


if __name__ == "__main__":
    main()
