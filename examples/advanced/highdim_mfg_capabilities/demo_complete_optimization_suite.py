#!/usr/bin/env python3
"""
Complete High-Dimensional MFG Optimization Suite Demonstration

This script demonstrates the full capabilities of the extended MFG_PDE package
including 3D geometry, advanced boundary conditions, performance optimization,
adaptive mesh refinement, and comprehensive benchmarking.
"""

import sys
import time
from pathlib import Path

# Add the package to the path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np

from mfg_pde import MFGComponents
from mfg_pde.benchmarks import HighDimMFGBenchmark
from mfg_pde.core.highdim_mfg_problem import GridBasedMFGProblem
from mfg_pde.geometry import Domain3D
from mfg_pde.geometry.boundary.bc_3d import (
    BoundaryConditionManager3D,
    DirichletBC3D,
    NeumannBC3D,
)
from mfg_pde.geometry.tetrahedral_amr import TetrahedralAMRMesh
from mfg_pde.utils.mfg_logging import configure_research_logging, get_logger
from mfg_pde.utils.performance_optimization import AdvancedSparseOperations, PerformanceMonitor, SparseMatrixOptimizer

# Configure logging
configure_research_logging("complete_optimization_demo", level="INFO")
logger = get_logger(__name__)


def demo_3d_geometry_with_boundary_conditions():
    """Demonstrate 3D geometry creation with advanced boundary conditions."""
    logger.info("=== 3D Geometry with Advanced Boundary Conditions ===")

    try:
        # Create 3D domain (fallback to simple grid if Gmsh not available)
        try:
            domain = Domain3D(domain_type="box", bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), mesh_size=0.2)
            mesh_data = domain.generate_mesh()
            logger.info(f"✅ Created 3D domain with {len(mesh_data.vertices)} vertices")
        except Exception as e:
            logger.warning(f"Gmsh not available, using simple grid: {e}")
            from mfg_pde.geometry.grids.grid_2d import SimpleGrid3D

            domain = SimpleGrid3D(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(16, 16, 16))
            mesh_data = domain.generate_mesh()
            logger.info(f"✅ Created simple 3D grid with {len(mesh_data.vertices)} vertices")

        # Create boundary condition manager
        bc_manager = BoundaryConditionManager3D()

        # Add mixed boundary conditions
        bc_manager.add_condition(DirichletBC3D(0.0, "Zero BC"), 0)  # x_min face
        bc_manager.add_condition(DirichletBC3D(1.0, "Unit BC"), 1)  # x_max face
        bc_manager.add_condition(NeumannBC3D(0.0, "No Flux"), 2)  # y_min face

        logger.info("✅ Created boundary condition manager with mixed conditions")

        # Validate boundary conditions
        is_valid = bc_manager.validate_all_conditions(mesh_data)
        logger.info(f"✅ Boundary conditions validation: {'PASSED' if is_valid else 'FAILED'}")

        return mesh_data, bc_manager

    except Exception as e:
        logger.error(f"❌ 3D geometry demo failed: {e}")
        return None, None


def demo_adaptive_mesh_refinement():
    """Demonstrate adaptive mesh refinement capabilities."""
    logger.info("=== Adaptive Mesh Refinement Demo ===")

    try:
        # Create initial coarse mesh
        from mfg_pde.geometry.grids.grid_2d import SimpleGrid3D

        domain = SimpleGrid3D(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), resolution=(8, 8, 8))
        mesh_data = domain.generate_mesh()

        logger.info(f"Initial mesh: {len(mesh_data.vertices)} vertices")

        # Create AMR mesh
        amr_mesh = TetrahedralAMRMesh(mesh_data)

        # Create mock solution with high gradients near center
        vertices = mesh_data.vertices
        center = np.array([0.5, 0.5, 0.5])
        distances = np.linalg.norm(vertices - center, axis=1)
        solution = np.exp(-10 * distances**2)  # Gaussian with sharp gradient

        # Perform refinement
        refined = amr_mesh.refine_mesh(solution, refinement_fraction=0.3)

        if refined:
            logger.info("✅ AMR refinement successful")
            logger.info(f"Quality metrics: {amr_mesh.compute_quality_metrics()}")
        else:
            logger.info("✅ AMR completed (no refinement needed)")

        return amr_mesh

    except Exception as e:
        logger.error(f"❌ AMR demo failed: {e}")
        return None


def demo_performance_optimization():
    """Demonstrate performance optimization tools."""
    logger.info("=== Performance Optimization Demo ===")

    try:
        monitor = PerformanceMonitor()

        # Demonstrate sparse matrix operations
        with monitor.monitor_operation("sparse_matrix_creation"):
            laplacian = SparseMatrixOptimizer.create_laplacian_3d(
                nx=16, ny=16, nz=16, dx=1.0 / 15, dy=1.0 / 15, dz=1.0 / 15
            )

        creation_metrics = monitor.metrics_history[-1]
        logger.info(f"✅ Created 3D Laplacian: {laplacian.shape} with {laplacian.nnz} nonzeros")
        logger.info(f"   Creation time: {creation_metrics.duration:.3f}s, Memory: {creation_metrics.memory_used:.1f}MB")

        # Test matrix optimization
        with monitor.monitor_operation("matrix_optimization"):
            SparseMatrixOptimizer.optimize_matrix_structure(laplacian)

        opt_metrics = monitor.metrics_history[-1]
        logger.info(f"✅ Matrix optimization completed in {opt_metrics.duration:.3f}s")

        # Test advanced sparse operations
        try:
            # Create smaller operators for Kronecker product demo
            op_1d = SparseMatrixOptimizer.create_laplacian_3d(8, 1, 1, 1.0 / 7, 1.0, 1.0)
            operators = [op_1d, op_1d, op_1d]

            with monitor.monitor_operation("tensor_product"):
                tensor_op = AdvancedSparseOperations.tensor_product_operator(operators)

            tensor_metrics = monitor.metrics_history[-1]
            logger.info(f"✅ Tensor product operator: {tensor_op.shape}")
            logger.info(f"   Time: {tensor_metrics.duration:.3f}s")

        except Exception as e:
            logger.warning(f"Tensor product demo failed: {e}")

        return monitor

    except Exception as e:
        logger.error(f"❌ Performance optimization demo failed: {e}")
        return None


class SimpleMFGProblem(GridBasedMFGProblem):
    """Simple MFG problem implementation for demonstration."""

    def __init__(self, domain_bounds, grid_resolution, time_domain=(1.0, 11)):
        super().__init__(
            domain_bounds=domain_bounds, grid_resolution=grid_resolution, time_domain=time_domain, diffusion_coeff=0.1
        )
        self.dimension = len(grid_resolution)

    def setup_components(self) -> MFGComponents:
        """Setup simple MFG components."""

        def simple_hamiltonian(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem, **kwargs):
            try:
                p_forward = p_values.get("forward", 0.0)
                p_backward = p_values.get("backward", 0.0)
                p_magnitude = abs(p_forward - p_backward)
                kinetic = 0.5 * p_magnitude**2
                congestion = 0.1 * m_at_x * p_magnitude**2
                result = kinetic + congestion
                return min(result, 1e3) if not (np.isnan(result) or np.isinf(result)) else 0.0
            except Exception:
                return 0.0

        def hamiltonian_dm(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem, **kwargs):
            try:
                p_forward = p_values.get("forward", 0.0)
                p_backward = p_values.get("backward", 0.0)
                p_magnitude = abs(p_forward - p_backward)
                result = 0.1 * p_magnitude**2
                return min(result, 1e3) if not (np.isnan(result) or np.isinf(result)) else 0.0
            except Exception:
                return 0.0

        def initial_density_grid(x_position):
            try:
                x_idx = int(x_position)
                if x_idx >= self.num_spatial_points:
                    return 1e-10
                coords = self.mesh_data.vertices[x_idx]
                center = np.array([0.5] * self.dimension)
                distance = np.linalg.norm(coords - center)
                density = np.exp(-(distance**2) / (2 * 0.2**2))
                return max(density, 1e-10)
            except Exception:
                return 1e-10

        def terminal_cost_grid(x_position):
            try:
                x_idx = int(x_position)
                if x_idx >= self.num_spatial_points:
                    return 0.0
                coords = self.mesh_data.vertices[x_idx]
                target = np.array([0.8] * self.dimension)
                distance = np.linalg.norm(coords - target)
                return 0.5 * distance**2
            except Exception:
                return 0.0

        def running_cost_grid(x_idx, x_position, m_at_x, t_idx, current_time, problem, **kwargs):
            return 0.01

        return MFGComponents(
            hamiltonian_func=simple_hamiltonian,
            hamiltonian_dm_func=hamiltonian_dm,
            initial_density_func=initial_density_grid,
            final_value_func=terminal_cost_grid,
        )

    def hamiltonian(self, x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem, **kwargs):
        """Direct implementation for abstract method compatibility."""
        try:
            p_forward = p_values.get("forward", 0.0)
            p_backward = p_values.get("backward", 0.0)
            p_magnitude = abs(p_forward - p_backward)
            kinetic = 0.5 * p_magnitude**2
            congestion = 0.1 * m_at_x * p_magnitude**2
            result = kinetic + congestion
            return min(result, 1e3) if not (np.isnan(result) or np.isinf(result)) else 0.0
        except Exception:
            return 0.0

    def initial_density(self, x_position):
        """Direct implementation for abstract method compatibility."""
        try:
            x_idx = int(x_position)
            if x_idx >= self.num_spatial_points:
                return 1e-10
            coords = self.mesh_data.vertices[x_idx]
            center = np.array([0.5] * self.dimension)
            distance = np.linalg.norm(coords - center)
            density = np.exp(-(distance**2) / (2 * 0.2**2))
            return max(density, 1e-10)
        except Exception:
            return 1e-10

    def terminal_cost(self, x_position):
        """Direct implementation for abstract method compatibility."""
        try:
            x_idx = int(x_position)
            if x_idx >= self.num_spatial_points:
                return 0.0
            coords = self.mesh_data.vertices[x_idx]
            target = np.array([0.8] * self.dimension)
            distance = np.linalg.norm(coords - target)
            return 0.5 * distance**2
        except Exception:
            return 0.0

    def running_cost(self, x_idx, x_position, m_at_x, t_idx, current_time, problem, **kwargs):
        """Direct implementation for abstract method compatibility."""
        return 0.01


def demo_high_dimensional_solving():
    """Demonstrate high-dimensional MFG solving with optimization."""
    logger.info("=== High-Dimensional MFG Solving Demo ===")

    try:
        # Create 3D problem
        problem = SimpleMFGProblem(
            domain_bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
            grid_resolution=(12, 12, 12),
            time_domain=(1.0, 11),  # Smaller problem for demo
        )

        logger.info(f"Created 3D MFG problem: {12**3} vertices")

        # Solve with performance monitoring
        monitor = PerformanceMonitor()

        with monitor.monitor_operation("mfg_solve_3d"):
            result = problem.solve_with_damped_fixed_point(damping_factor=0.5, max_iterations=20, tolerance=1e-3)

        solve_metrics = monitor.metrics_history[-1]

        if result.get("converged", False):
            logger.info(f"✅ 3D MFG solution converged in {result['iterations']} iterations")
            logger.info(f"   Solve time: {solve_metrics.duration:.2f}s")
            logger.info(f"   Memory used: {solve_metrics.memory_used:.1f}MB")
            logger.info(f"   Final residual: {result.get('final_residual', 'N/A'):.2e}")
        else:
            logger.warning("⚠️ 3D MFG solution did not converge")

        return result, solve_metrics

    except Exception as e:
        logger.error(f"❌ High-dimensional solving demo failed: {e}")
        return None, None


def demo_comprehensive_benchmarking():
    """Demonstrate comprehensive benchmarking capabilities."""
    logger.info("=== Comprehensive Benchmarking Demo ===")

    try:
        # Create benchmark instance
        benchmark = HighDimMFGBenchmark("demo_benchmark_results")

        # Run quick convergence benchmark
        quick_grids = [(8, 8), (12, 12), (6, 6, 6)]

        logger.info("Running quick benchmark suite...")
        results = benchmark.run_convergence_benchmark(grid_sizes=quick_grids, solver_methods=["damped_fixed_point"])

        logger.info(f"✅ Completed {len(results)} benchmark tests")

        # Generate analysis report
        analysis = benchmark.generate_benchmark_report(save_plots=True)

        if analysis:
            summary = analysis.get("summary", {})
            logger.info("✅ Benchmark analysis completed:")
            logger.info(f"   Success rate: {summary.get('success_rate', 0):.1%}")
            logger.info(f"   Average solve time: {summary.get('average_solve_time', 0):.3f}s")
            logger.info(f"   Dimensions tested: {summary.get('dimensions_tested', [])}")

        return analysis

    except Exception as e:
        logger.error(f"❌ Benchmarking demo failed: {e}")
        return None


def main():
    """Run complete optimization suite demonstration."""
    logger.info("🚀 Starting Complete High-Dimensional MFG Optimization Suite Demo")
    logger.info("=" * 80)

    start_time = time.time()

    # Demo 1: 3D Geometry with Boundary Conditions
    mesh_data, _bc_manager = demo_3d_geometry_with_boundary_conditions()

    # Demo 2: Adaptive Mesh Refinement
    amr_mesh = demo_adaptive_mesh_refinement()

    # Demo 3: Performance Optimization Tools
    perf_monitor = demo_performance_optimization()

    # Demo 4: High-Dimensional MFG Solving
    mfg_result, _solve_metrics = demo_high_dimensional_solving()

    # Demo 5: Comprehensive Benchmarking
    benchmark_analysis = demo_comprehensive_benchmarking()

    # Summary
    total_time = time.time() - start_time

    logger.info("=" * 80)
    logger.info("🎯 COMPLETE OPTIMIZATION SUITE DEMONSTRATION SUMMARY")
    logger.info("=" * 80)

    # Count successful demonstrations
    demos = [
        ("3D Geometry & Boundary Conditions", mesh_data is not None),
        ("Adaptive Mesh Refinement", amr_mesh is not None),
        ("Performance Optimization", perf_monitor is not None),
        ("High-Dimensional MFG Solving", mfg_result is not None),
        ("Comprehensive Benchmarking", benchmark_analysis is not None),
    ]

    successful_demos = sum(1 for _, success in demos if success)
    total_demos = len(demos)

    logger.info(f"Demonstrations completed: {successful_demos}/{total_demos}")
    logger.info(f"Total execution time: {total_time:.2f} seconds")

    for demo_name, success in demos:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"   {demo_name}: {status}")

    if successful_demos == total_demos:
        logger.info("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        logger.info("The high-dimensional MFG optimization suite is fully operational.")
    else:
        logger.warning(f"⚠️ {total_demos - successful_demos} demonstrations had issues")
        logger.info("Check logs above for specific error details.")

    logger.info("=" * 80)

    return successful_demos == total_demos


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
