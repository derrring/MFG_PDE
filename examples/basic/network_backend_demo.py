#!/usr/bin/env python3
"""
Network Backend Demonstration.

This example shows how the unified backend system automatically selects
the optimal network library (igraph, networkit, networkx) based on:
- Network size
- Operation type
- Available libraries
- Performance requirements

The system provides automatic fallbacks and performance optimization.
"""

import time
from typing import Any, Dict

import numpy as np

# MFG_PDE imports
from mfg_pde.geometry.network_backend import NetworkBackendType, OperationType, get_backend_manager
from mfg_pde.geometry.network_geometry import GridNetwork, NetworkType, RandomNetwork, ScaleFreeNetwork, create_network


def demonstrate_backend_selection():
    """Demonstrate automatic backend selection based on network size."""
    print("=" * 80)
    print("NETWORK BACKEND SELECTION DEMONSTRATION")
    print("=" * 80)

    manager = get_backend_manager()

    # Show available backends
    info = manager.get_backend_info()
    print("Available Network Backends:")
    for backend_name in info['available_backends']:
        caps = info['backend_capabilities'][backend_name]
        print(f"  • {backend_name}:")
        print(f"    - Max recommended nodes: {caps['max_recommended_nodes']:,}")
        print(f"    - Speed rating: {caps['speed_rating']}/5")
        print(f"    - Algorithm coverage: {caps['algorithm_coverage']}/5")
        print(f"    - Parallel support: {caps['parallel_support']}")
    print()

    # Test different network sizes
    test_sizes = [100, 1000, 10000, 100000]

    print("Backend Selection by Network Size:")
    print("-" * 50)
    for size in test_sizes:
        general_backend = manager.choose_backend(size, OperationType.GENERAL)
        large_scale_backend = manager.choose_backend(size, OperationType.LARGE_SCALE)
        viz_backend = manager.choose_backend(size, OperationType.VISUALIZATION)

        print(f"Network size {size:6,}:")
        print(f"  General:      {general_backend.value}")
        print(f"  Large-scale:  {large_scale_backend.value}")
        print(f"  Visualization: {viz_backend.value}")
    print()


def benchmark_backends():
    """Benchmark different network creation methods."""
    print("BACKEND PERFORMANCE COMPARISON")
    print("=" * 50)

    sizes = [100, 500, 1000]
    network_types = ['grid', 'random', 'scale_free']

    for network_type in network_types:
        print(f"\n{network_type.upper()} Networks:")
        print("-" * 30)

        for size in sizes:
            print(f"Size {size:4d}: ", end="")

            # Time network creation
            start_time = time.time()

            if network_type == 'grid':
                width = int(np.sqrt(size))
                height = size // width
                network = create_network('grid', size, width=width, height=height)
            elif network_type == 'random':
                network = create_network('random', size, connection_prob=0.1)
            else:  # scale_free
                network = create_network('scale_free', size, num_edges_per_node=3)

            # Create the actual network
            network_data = network.create_network(seed=42)
            creation_time = time.time() - start_time

            backend_used = network_data.backend_type.value if network_data.backend_type else "unknown"
            print(f"Backend: {backend_used:8s} | Time: {creation_time:.4f}s | " f"Edges: {network_data.num_edges:4d}")


def demonstrate_network_types():
    """Show different network topologies with backend optimization."""
    print("\nNETWORK TOPOLOGY DEMONSTRATION")
    print("=" * 50)

    # Grid network (regular structure)
    print("1. Grid Network (5x5):")
    grid = GridNetwork(5, 5, backend_preference=NetworkBackendType.IGRAPH)
    grid_data = grid.create_network()
    print(f"   Backend: {grid_data.backend_type.value if grid_data.backend_type else 'unknown'}")
    print(f"   Nodes: {grid_data.num_nodes}, Edges: {grid_data.num_edges}")
    print(f"   Connected: {grid_data.is_connected}")
    print()

    # Random network (Erdős–Rényi model)
    print("2. Random Network (50 nodes, p=0.15):")
    random_net = RandomNetwork(50, 0.15, backend_preference=NetworkBackendType.IGRAPH)
    random_data = random_net.create_network(seed=42)
    print(f"   Backend: {random_data.backend_type.value if random_data.backend_type else 'unknown'}")
    print(f"   Nodes: {random_data.num_nodes}, Edges: {random_data.num_edges}")
    print(f"   Density: {random_data.metadata.get('density', 0):.3f}")
    print()

    # Scale-free network (Barabási-Albert model)
    print("3. Scale-Free Network (30 nodes, m=2):")
    sf_net = ScaleFreeNetwork(30, 2, backend_preference=NetworkBackendType.IGRAPH)
    sf_data = sf_net.create_network(seed=42)
    print(f"   Backend: {sf_data.backend_type.value if sf_data.backend_type else 'unknown'}")
    print(f"   Nodes: {sf_data.num_nodes}, Edges: {sf_data.num_edges}")
    print(f"   Avg degree: {sf_data.metadata.get('average_degree', 0):.2f}")
    print()


def show_backend_capabilities():
    """Show detailed backend capabilities."""
    print("BACKEND CAPABILITIES COMPARISON")
    print("=" * 50)

    manager = get_backend_manager()

    # Show capability matrix
    print(f"{'Backend':12} | {'Max Nodes':>10} | {'Speed':>5} | {'Memory':>6} | {'Algorithms':>10} | {'Parallel':>8}")
    print("-" * 70)

    for backend_type in [NetworkBackendType.NETWORKX, NetworkBackendType.IGRAPH, NetworkBackendType.NETWORKIT]:
        caps = manager.get_capabilities(backend_type)
        available = "✓" if backend_type in manager.available_backends else "✗"

        print(
            f"{backend_type.value:12} | {caps.max_recommended_nodes:>10,} | "
            f"{caps.speed_rating:>5}/5 | {caps.memory_efficiency:>6}/5 | "
            f"{caps.algorithm_coverage:>10}/5 | {caps.parallel_support!s:>8} | {available}"
        )

    print()
    print("Recommendations:")
    print("• Small networks (<1K nodes):     Any backend, NetworkX for algorithms")
    print("• Medium networks (1K-100K):      igraph for best balance")
    print("• Large networks (>100K nodes):   networkit for performance")
    print("• Visualization:                  igraph or NetworkX")
    print("• Algorithm-heavy workloads:      NetworkX")


def main():
    """Main demonstration function."""
    try:
        demonstrate_backend_selection()
        benchmark_backends()
        demonstrate_network_types()
        show_backend_capabilities()

        print("\n" + "=" * 80)
        print("BACKEND DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("Key Benefits of the Unified Backend System:")
        print("• Automatic performance optimization")
        print("• Graceful fallbacks when libraries are missing")
        print("• Consistent API across all backends")
        print("• Network size-based backend selection")
        print("• Easy extensibility for new backends")

    except Exception as e:
        print(f"Error in backend demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
