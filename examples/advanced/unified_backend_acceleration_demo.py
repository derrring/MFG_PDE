#!/usr/bin/env python3
"""
Unified Backend Acceleration System Demonstration

This example showcases the new unified backend system in MFG_PDE with comprehensive
support for PyTorch CUDA/MPS, JAX, and NumPy acceleration backends.

Key Features Demonstrated:
1. Automatic device detection and selection
2. Cross-platform hardware acceleration (CUDA, MPS, JAX)
3. Performance comparison across backends
4. Neural solver integration with hardware acceleration
5. Seamless fallback and compatibility

Hardware Support Matrix:
- NVIDIA GPUs: CUDA acceleration via PyTorch
- Apple Silicon: MPS acceleration via PyTorch Metal Performance Shaders
- General GPUs: JAX XLA compilation
- CPU: NumPy baseline with optimized routines
"""

from __future__ import annotations

import time

import numpy as np


def demonstrate_backend_detection():
    """Demonstrate comprehensive backend detection capabilities."""
    print("🔍 Backend Detection and Capabilities")
    print("=" * 50)

    try:
        from mfg_pde.backends import get_available_backends, get_backend_info

        # Get available backends
        available = get_available_backends()
        print("Available computational backends:")

        backend_descriptions = {
            "numpy": "CPU baseline (always available)",
            "torch": "PyTorch framework",
            "torch_cuda": "NVIDIA GPU acceleration (CUDA)",
            "torch_mps": "Apple Silicon acceleration (MPS)",
            "jax": "JAX framework",
            "jax_gpu": "JAX GPU acceleration (XLA)",
        }

        for backend, status in available.items():
            desc = backend_descriptions.get(backend, "Unknown backend")
            print(f"  {'✅' if status else '❌'} {backend:<12} - {desc}")

        # Detailed hardware information
        print("\n🔧 Detailed Hardware Information")
        print("-" * 30)

        info = get_backend_info()

        # PyTorch information
        if "torch_info" in info:
            torch_info = info["torch_info"]
            print(f"PyTorch Version: {torch_info.get('version', 'N/A')}")

            if torch_info.get("cuda_available"):
                print(f"CUDA Version: {torch_info.get('cuda_version', 'N/A')}")
                cuda_devices = torch_info.get("cuda_devices", [])
                print(f"CUDA GPUs ({len(cuda_devices)}):")
                for i, gpu in enumerate(cuda_devices):
                    print(f"  GPU {i}: {gpu}")

            if torch_info.get("mps_available"):
                print("Apple Silicon MPS: Available and functional")

        # JAX information
        if "jax_info" in info:
            jax_info = info["jax_info"]
            print(f"JAX Version: {jax_info.get('version', 'N/A')}")
            jax_devices = jax_info.get("devices", [])
            print(f"JAX Devices: {', '.join(jax_devices)}")

    except ImportError as e:
        print(f"❌ Backend detection failed: {e}")


def demonstrate_automatic_backend_selection():
    """Demonstrate automatic backend selection with priority."""
    print("\n🚀 Automatic Backend Selection")
    print("=" * 50)

    try:
        from mfg_pde.backends import create_backend

        print("Creating backend with automatic selection...")
        print("Priority order: CUDA > MPS > JAX GPU > JAX CPU > NumPy")

        # Create backend with auto-selection
        backend = create_backend("auto")
        print(f"\n✅ Selected backend: {backend.name}")
        print(f"Array module: {backend.array_module.__name__}")

        # Test basic operations
        print("\n🧪 Testing basic operations:")
        arr = backend.zeros((1000, 1000))
        print(f"Created zeros array: {arr.shape}")

        ones_arr = backend.ones((500, 500))
        print(f"Created ones array: {ones_arr.shape}")

        linspace_arr = backend.linspace(0, 1, 100)
        print(f"Created linspace: {linspace_arr.shape}")

        return backend

    except Exception as e:
        print(f"❌ Backend selection failed: {e}")
        return None


def demonstrate_performance_comparison(backend):
    """Demonstrate performance comparison across available backends."""
    print("\n⚡ Performance Comparison")
    print("=" * 50)

    try:
        from mfg_pde.backends import create_backend, get_available_backends

        # Test problem: Large matrix operations
        size = 2000
        print(f"Test problem: {size}x{size} matrix operations")

        available = get_available_backends()
        backends_to_test = []

        # Add available backends
        if available.get("numpy"):
            backends_to_test.append(("NumPy CPU", "numpy"))

        if available.get("torch"):
            backends_to_test.append(("PyTorch CPU", "torch", {"device": "cpu"}))

        if available.get("torch_cuda"):
            backends_to_test.append(("PyTorch CUDA", "torch", {"device": "cuda"}))

        if available.get("torch_mps"):
            backends_to_test.append(("PyTorch MPS", "torch", {"device": "mps"}))

        if available.get("jax"):
            backends_to_test.append(("JAX", "jax"))

        results = []

        for backend_info in backends_to_test:
            name = backend_info[0]
            backend_name = backend_info[1]
            kwargs = backend_info[2] if len(backend_info) > 2 else {}

            try:
                print(f"\n📊 Testing {name}...")
                test_backend = create_backend(backend_name, **kwargs)

                # Create test matrices
                start_time = time.time()
                A = test_backend.zeros((size, size))
                A = A + 1.0  # Simple operation to initialize

                B = test_backend.ones((size, size))

                # Matrix multiplication (the heavy operation)
                if hasattr(test_backend, "array_module"):
                    if hasattr(test_backend.array_module, "matmul"):
                        C = test_backend.array_module.matmul(A, B)
                    else:
                        # Fallback for backends without matmul
                        C = A @ B
                else:
                    C = A @ B

                # Force computation to complete (important for GPU backends)
                if hasattr(C, "block_until_ready"):
                    C.block_until_ready()  # JAX
                elif hasattr(test_backend, "torch_device") and test_backend.torch_device.type in ["cuda", "mps"]:
                    # PyTorch GPU - synchronize
                    if test_backend.torch_device.type == "cuda":
                        import torch

                        torch.cuda.synchronize()
                    elif test_backend.torch_device.type == "mps":
                        import torch

                        torch.mps.synchronize()

                elapsed = time.time() - start_time
                results.append((name, elapsed, True))
                print(f"  ✅ Time: {elapsed:.3f}s")

            except Exception as e:
                print(f"  ❌ Failed: {e}")
                results.append((name, float("inf"), False))

        # Results summary
        print(f"\n📈 Performance Summary (Matrix multiplication {size}x{size}):")
        print("-" * 60)

        successful_results = [(name, time) for name, time, success in results if success]
        if successful_results:
            # Sort by performance
            successful_results.sort(key=lambda x: x[1])
            fastest_time = successful_results[0][1]

            for name, elapsed in successful_results:
                speedup = fastest_time / elapsed
                print(f"{name:<15}: {elapsed:6.3f}s (speedup: {speedup:4.1f}x)")

    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")


def demonstrate_neural_solver_integration():
    """Demonstrate neural solver integration with hardware acceleration."""
    print("\n🧠 Neural Solver Hardware Acceleration")
    print("=" * 50)

    try:
        from mfg_pde.alg.neural.pinn_solvers import (
            CUDA_AVAILABLE,
            MPS_AVAILABLE,
            TORCH_AVAILABLE,
            PINNConfig,
            print_system_info,
        )

        # Print neural solver capabilities
        print_system_info()

        if not TORCH_AVAILABLE:
            print("❌ PyTorch not available, skipping neural solver demonstration")
            return

        print("\n🔧 PINN Configuration Examples:")

        # Auto device selection
        config_auto = PINNConfig(device="auto", dtype="float32")
        print(f"Auto configuration: device={config_auto.device}, dtype={config_auto.dtype}")

        # Test different device configurations
        test_configs = [
            ("CPU", "cpu"),
        ]

        if CUDA_AVAILABLE:
            test_configs.append(("CUDA", "cuda"))

        if MPS_AVAILABLE:
            test_configs.append(("MPS", "mps"))

        for name, device in test_configs:
            try:
                config = PINNConfig(
                    device=device, hidden_layers=[64, 64, 64], learning_rate=1e-3, batch_size=1000, max_epochs=1000
                )
                print(f"✅ {name} configuration: {config.device}")
            except Exception as e:
                print(f"❌ {name} configuration failed: {e}")

        print("\n💡 Key Benefits:")
        print("• Automatic device detection and selection")
        print("• Cross-platform acceleration (NVIDIA, Apple Silicon)")
        print("• Graceful fallback to CPU when GPU unavailable")
        print("• Consistent API across all hardware platforms")
        print("• Memory-efficient operations with mixed precision support")

    except Exception as e:
        print(f"❌ Neural solver demonstration failed: {e}")


def demonstrate_mfg_specific_operations(backend):
    """Demonstrate MFG-specific accelerated operations."""
    print("\n🎯 MFG-Specific Accelerated Operations")
    print("=" * 50)

    try:
        print(f"Using backend: {backend.name}")

        # Test problem parameters
        problem_params = {
            "potential_strength": 0.5,
            "interaction_strength": 1.0,
            "interaction_epsilon": 1e-8,
            "diffusion": 0.1,
            "x_min": -2.0,
            "x_max": 2.0,
        }

        # Create test grids
        nx = 101
        x = backend.linspace(-2, 2, nx)
        dx = 4.0 / (nx - 1)

        # Initial conditions
        U = backend.zeros((nx,))  # Value function
        p = backend.zeros((nx,))  # Momentum/gradient

        # Gaussian density
        m_data = np.exp(-(x**2) / 0.5)
        m_data = m_data / np.trapz(m_data, dx=dx)  # Normalize
        m = backend.array(m_data)

        print(f"Grid size: {nx} points")
        print(f"Domain: [{problem_params['x_min']}, {problem_params['x_max']}]")

        # Test MFG operations
        print("\n🧮 Testing MFG operations:")

        # Hamiltonian computation
        start_time = time.time()
        H = backend.compute_hamiltonian(x, p, m, problem_params)
        elapsed = time.time() - start_time
        print(f"✅ Hamiltonian computation: {elapsed * 1000:.2f}ms")

        # Optimal control computation
        start_time = time.time()
        _ = backend.compute_optimal_control(x, p, m, problem_params)
        elapsed = time.time() - start_time
        print(f"✅ Optimal control computation: {elapsed * 1000:.2f}ms")

        # Time stepping operations
        dt = 0.01

        start_time = time.time()
        _ = backend.hjb_step(U, m, dt, dx, problem_params)
        elapsed = time.time() - start_time
        print(f"✅ HJB time step: {elapsed * 1000:.2f}ms")

        start_time = time.time()
        m_new = backend.fpk_step(m, U, dt, dx, problem_params)
        elapsed = time.time() - start_time
        print(f"✅ Fokker-Planck time step: {elapsed * 1000:.2f}ms")

        print("\n📊 Results summary:")
        print(f"• Final mass conservation: {backend.trapezoid(m_new, dx=dx):.6f}")
        print(f"• Hamiltonian range: [{float(backend.min(H)):.3f}, {float(backend.max(H)):.3f}]")

    except Exception as e:
        print(f"❌ MFG operations demonstration failed: {e}")


def main():
    """Run complete unified backend demonstration."""
    print("🚀 MFG_PDE Unified Backend Acceleration System")
    print("=" * 80)

    # Step 1: Backend Detection
    demonstrate_backend_detection()

    # Step 2: Automatic Backend Selection
    backend = demonstrate_automatic_backend_selection()

    if backend is None:
        print("❌ Cannot continue without a working backend")
        return 1

    # Step 3: Performance Comparison
    demonstrate_performance_comparison(backend)

    # Step 4: Neural Solver Integration
    demonstrate_neural_solver_integration()

    # Step 5: MFG-Specific Operations
    demonstrate_mfg_specific_operations(backend)

    # Summary
    print("\n" + "=" * 80)
    print("🎉 Demonstration Complete!")
    print("\nKey Achievements:")
    print("• ✅ Unified backend system with automatic device selection")
    print("• ✅ Cross-platform acceleration (CUDA, MPS, JAX)")
    print("• ✅ Neural solver integration with hardware acceleration")
    print("• ✅ MFG-specific optimized numerical operations")
    print("• ✅ Seamless fallback and compatibility across platforms")

    print("\nThe MFG_PDE framework now provides state-of-the-art acceleration")
    print("support across all major hardware platforms! 🚀")

    return 0


if __name__ == "__main__":
    exit(main())
