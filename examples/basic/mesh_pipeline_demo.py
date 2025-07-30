"""
Demonstration of the Gmsh → Meshio → PyVista mesh generation pipeline.

This example shows how to use the new geometry system for 2D domain meshing,
quality analysis, and visualization with the MFG_PDE framework.
"""

import numpy as np
import sys
from pathlib import Path

# Add MFG_PDE to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mfg_pde.geometry import Domain2D, MeshPipeline, MeshManager, BoundaryManager
from mfg_pde.utils.logging import get_logger, configure_basic_logging

# Configure logging
configure_basic_logging(level="INFO")
logger = get_logger(__name__)


def demo_simple_rectangle():
    """Demonstrate simple rectangular domain meshing."""
    logger.info("🔸 Demo 1: Simple rectangular domain")
    
    # Create rectangular domain
    domain = Domain2D(
        domain_type="rectangle",
        bounds=(0.0, 2.0, 0.0, 1.0),  # (xmin, xmax, ymin, ymax)
        mesh_size=0.1
    )
    
    # Create and execute pipeline
    pipeline = MeshPipeline(domain, output_dir="./output/rectangle")
    mesh_data = pipeline.execute_pipeline(
        stages=["generate", "analyze", "export"],
        export_formats=["msh", "vtk", "xdmf"]
    )
    
    # Print summary
    logger.info(f"Generated mesh: {mesh_data.num_vertices} vertices, "
               f"{mesh_data.num_elements} triangles")
    
    # Visualize (uncomment to show interactive plot)
    # pipeline.visualize_interactive(show_quality=True)
    
    return mesh_data


def demo_domain_with_holes():
    """Demonstrate domain with circular holes."""
    logger.info("🔸 Demo 2: Domain with circular holes")
    
    # Define holes
    holes = [
        {"type": "circle", "center": (0.5, 0.5), "radius": 0.15},
        {"type": "circle", "center": (1.5, 0.3), "radius": 0.1},
        {"type": "rectangle", "bounds": (0.2, 0.4, 0.7, 0.9)}
    ]
    
    # Create domain with holes
    domain = Domain2D(
        domain_type="rectangle",
        bounds=(0.0, 2.0, 0.0, 1.0),
        holes=holes,
        mesh_size=0.05
    )
    
    # Execute pipeline
    pipeline = MeshPipeline(domain, output_dir="./output/with_holes")
    mesh_data = pipeline.execute_pipeline()
    
    logger.info(f"Mesh with holes: {mesh_data.num_vertices} vertices, "
               f"{mesh_data.num_elements} triangles")
    
    return mesh_data


def demo_circular_domain():
    """Demonstrate circular domain meshing."""
    logger.info("🔸 Demo 3: Circular domain")
    
    # Create circular domain
    domain = Domain2D(
        domain_type="circle",
        center=(0.0, 0.0),
        radius=1.0,
        mesh_size=0.1
    )
    
    # Execute pipeline
    pipeline = MeshPipeline(domain, output_dir="./output/circle")
    mesh_data = pipeline.execute_pipeline()
    
    logger.info(f"Circular mesh: {mesh_data.num_vertices} vertices, "
               f"{mesh_data.num_elements} triangles")
    
    return mesh_data


def demo_polygon_domain():
    """Demonstrate custom polygonal domain."""
    logger.info("🔸 Demo 4: Custom polygonal domain")
    
    # Define L-shaped domain vertices
    vertices = [
        (0.0, 0.0),
        (1.0, 0.0), 
        (1.0, 0.5),
        (0.5, 0.5),
        (0.5, 1.0),
        (0.0, 1.0)
    ]
    
    # Create polygonal domain
    domain = Domain2D(
        domain_type="polygon",
        vertices=vertices,
        mesh_size=0.08
    )
    
    # Execute pipeline
    pipeline = MeshPipeline(domain, output_dir="./output/polygon")
    mesh_data = pipeline.execute_pipeline()
    
    logger.info(f"L-shaped mesh: {mesh_data.num_vertices} vertices, "
               f"{mesh_data.num_elements} triangles")
    
    return mesh_data


def demo_boundary_conditions():
    """Demonstrate boundary condition management."""
    logger.info("🔸 Demo 5: Boundary condition management")
    
    # Create simple domain
    domain = Domain2D(
        domain_type="rectangle", 
        bounds=(0.0, 1.0, 0.0, 1.0),
        mesh_size=0.1
    )
    
    # Generate mesh
    mesh_data = domain.generate_mesh()
    
    # Create boundary manager
    boundary_manager = BoundaryManager(mesh_data)
    
    # Add boundary conditions for different regions
    # Note: This assumes Gmsh assigns region IDs to rectangle boundaries
    
    # Bottom boundary (region 1): Dirichlet u = 0
    boundary_manager.add_boundary_condition(
        region_id=1,
        bc_type="dirichlet", 
        value=0.0,
        description="Bottom boundary: u = 0"
    )
    
    # Right boundary (region 2): Neumann du/dn = 1
    boundary_manager.add_boundary_condition(
        region_id=2,
        bc_type="neumann",
        gradient_value=1.0,
        description="Right boundary: du/dn = 1"
    )
    
    # Top boundary (region 3): Dirichlet u = sin(πx)
    boundary_manager.add_boundary_condition(
        region_id=3,
        bc_type="dirichlet",
        value=lambda coords: np.sin(np.pi * coords[:, 0]),
        description="Top boundary: u = sin(πx)"
    )
    
    # Left boundary (region 4): No-flux (homogeneous Neumann)
    boundary_manager.add_boundary_condition(
        region_id=4,
        bc_type="neumann",
        gradient_value=0.0,
        description="Left boundary: no flux"
    )
    
    # Print boundary condition summary
    summary = boundary_manager.get_summary()
    logger.info("Boundary conditions summary:")
    for region_id, info in summary["regions"].items():
        logger.info(f"  Region {region_id}: {info['type']} ({info['num_nodes']} nodes)")
    
    # Visualize boundary conditions (uncomment to show)
    # boundary_manager.visualize_boundary_conditions()
    
    return boundary_manager


def demo_mesh_quality_analysis():
    """Demonstrate mesh quality analysis and comparison."""
    logger.info("🔸 Demo 6: Mesh quality analysis")
    
    # Create domains with different mesh sizes
    coarse_domain = Domain2D(
        domain_type="rectangle",
        bounds=(0.0, 1.0, 0.0, 1.0),
        mesh_size=0.2
    )
    
    fine_domain = Domain2D(
        domain_type="rectangle", 
        bounds=(0.0, 1.0, 0.0, 1.0),
        mesh_size=0.05
    )
    
    # Generate meshes
    coarse_mesh = coarse_domain.generate_mesh()
    fine_mesh = fine_domain.generate_mesh()
    
    # Analyze quality
    coarse_quality = coarse_domain.compute_mesh_quality()
    fine_quality = fine_domain.compute_mesh_quality()
    
    # Compare results
    logger.info("Mesh quality comparison:")
    logger.info(f"Coarse mesh ({coarse_mesh.num_elements} elements):")
    for metric, value in coarse_quality.items():
        logger.info(f"  {metric}: {value:.6f}")
        
    logger.info(f"Fine mesh ({fine_mesh.num_elements} elements):")
    for metric, value in fine_quality.items():
        logger.info(f"  {metric}: {value:.6f}")
    
    return coarse_quality, fine_quality


def demo_batch_processing():
    """Demonstrate batch mesh generation with MeshManager."""
    logger.info("🔸 Demo 7: Batch mesh processing")
    
    # Create mesh manager
    manager = MeshManager()
    
    # Define multiple geometries
    geometries = {
        "rect_coarse": {
            "type": "rectangle",
            "bounds": (0.0, 1.0, 0.0, 1.0),
            "mesh_size": 0.15
        },
        "rect_fine": {
            "type": "rectangle", 
            "bounds": (0.0, 1.0, 0.0, 1.0),
            "mesh_size": 0.05
        },
        "circle": {
            "type": "circle",
            "kwargs": {"center": (0.5, 0.5), "radius": 0.4},
            "mesh_size": 0.08
        }
    }
    
    # Create geometries and pipelines
    for name, config in geometries.items():
        manager.create_geometry(name, config)
        manager.create_pipeline(name, name, f"./output/batch/{name}")
    
    # Batch generate meshes
    results = manager.batch_generate_meshes(
        list(geometries.keys()),
        stages=["generate", "analyze"]
    )
    
    # Compare quality
    quality_comparison = manager.compare_mesh_quality(list(geometries.keys()))
    
    logger.info("Batch processing results:")
    for name, mesh_data in results.items():
        logger.info(f"  {name}: {mesh_data.num_vertices} vertices, "
                   f"{mesh_data.num_elements} elements")
    
    return results, quality_comparison


def main():
    """Run all mesh pipeline demonstrations."""
    logger.info("🚀 Starting mesh pipeline demonstrations")
    
    try:
        # Run demonstrations
        demo_simple_rectangle()
        demo_domain_with_holes()
        demo_circular_domain()
        demo_polygon_domain()
        demo_boundary_conditions()
        demo_mesh_quality_analysis()
        demo_batch_processing()
        
        logger.info("✅ All demonstrations completed successfully!")
        
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("To run this demo, install: pip install gmsh meshio pyvista")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()