"""
Mesh management and pipeline orchestration for MFG_PDE.

This module implements the core MeshManager and MeshPipeline classes that coordinate
the Gmsh → Meshio → PyVista workflow for professional mesh generation and analysis.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from pathlib import Path
import logging

from .base_geometry import BaseGeometry, MeshData
from .domain_2d import Domain2D

logger = logging.getLogger(__name__)


class MeshPipeline:
    """
    Professional mesh generation pipeline: Gmsh → Meshio → PyVista.
    
    This class orchestrates the complete mesh workflow from geometry definition
    to visualization and analysis, ensuring seamless integration between tools.
    """
    
    def __init__(self, 
                 geometry: Union[BaseGeometry, Dict],
                 output_dir: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialize mesh pipeline.
        
        Args:
            geometry: Geometry object or configuration dictionary
            output_dir: Directory for mesh files and outputs
            verbose: Enable detailed logging
        """
        
        if isinstance(geometry, dict):
            self.geometry = self._create_geometry_from_config(geometry)
        else:
            self.geometry = geometry
            
        self.output_dir = Path(output_dir) if output_dir else Path("./mesh_output")
        self.output_dir.mkdir(exist_ok=True)
        
        self.verbose = verbose
        self.mesh_data: Optional[MeshData] = None
        
        # Pipeline stages tracking
        self.stages_completed = {
            "geometry_created": False,
            "mesh_generated": False,
            "quality_analyzed": False,
            "visualization_ready": False
        }
        
    def _create_geometry_from_config(self, config: Dict) -> BaseGeometry:
        """Create geometry object from configuration dictionary."""
        geometry_type = config.get("type", "rectangle")
        dimension = config.get("dimension", 2)
        
        if dimension == 2:
            return Domain2D(
                domain_type=geometry_type,
                bounds=config.get("bounds", (0.0, 1.0, 0.0, 1.0)),
                holes=config.get("holes", []),
                mesh_size=config.get("mesh_size", 0.1),
                **config.get("kwargs", {})
            )
        else:
            raise NotImplementedError(f"Dimension {dimension} not yet supported")
    
    def execute_pipeline(self, 
                        stages: List[str] = None,
                        export_formats: List[str] = None) -> MeshData:
        """
        Execute complete mesh pipeline.
        
        Args:
            stages: Pipeline stages to execute ["generate", "analyze", "visualize", "export"]
            export_formats: Mesh formats to export ["msh", "vtk", "xdmf", "stl"]
            
        Returns:
            Generated mesh data
        """
        
        if stages is None:
            stages = ["generate", "analyze", "visualize"]
            
        if export_formats is None:
            export_formats = ["msh", "vtk"]
        
        logger.info("🚀 Starting Gmsh → Meshio → PyVista pipeline")
        
        # Stage 1: Generate mesh
        if "generate" in stages:
            self.mesh_data = self._stage_generate_mesh()
            self.stages_completed["mesh_generated"] = True
            
        # Stage 2: Analyze quality
        if "analyze" in stages:
            self._stage_analyze_quality()
            self.stages_completed["quality_analyzed"] = True
            
        # Stage 3: Prepare visualization
        if "visualize" in stages:
            self._stage_prepare_visualization()
            self.stages_completed["visualization_ready"] = True
            
        # Stage 4: Export mesh
        if "export" in stages:
            self._stage_export_mesh(export_formats)
            
        logger.info("✅ Pipeline completed successfully")
        return self.mesh_data
    
    def _stage_generate_mesh(self) -> MeshData:
        """Pipeline Stage 1: Generate mesh using Gmsh."""
        logger.info("📐 Stage 1: Generating mesh with Gmsh")
        
        try:
            # Create geometry and generate mesh
            mesh_data = self.geometry.generate_mesh()
            
            logger.info(f"✅ Mesh generated: {mesh_data.num_vertices} vertices, "
                       f"{mesh_data.num_elements} elements")
            
            return mesh_data
            
        except Exception as e:
            logger.error(f"❌ Mesh generation failed: {e}")
            raise
    
    def _stage_analyze_quality(self):
        """Pipeline Stage 2: Analyze mesh quality."""
        logger.info("🔍 Stage 2: Analyzing mesh quality")
        
        try:
            quality_metrics = self.geometry.compute_mesh_quality()
            
            # Log quality summary
            if self.verbose:
                logger.info("📊 Mesh Quality Metrics:")
                for metric, value in quality_metrics.items():
                    logger.info(f"   {metric}: {value:.6f}")
                    
            # Save quality report
            self._save_quality_report(quality_metrics)
            
        except Exception as e:
            logger.error(f"❌ Quality analysis failed: {e}")
            raise
    
    def _stage_prepare_visualization(self):
        """Pipeline Stage 3: Prepare PyVista visualization."""
        logger.info("🎨 Stage 3: Preparing PyVista visualization")
        
        try:
            # Convert to PyVista format
            pyvista_mesh = self.mesh_data.to_pyvista()
            
            # Add quality data if available
            if "quality" in self.mesh_data.quality_metrics:
                quality_data = self.mesh_data.quality_metrics["quality"]
                if isinstance(quality_data, (list, np.ndarray)):
                    pyvista_mesh.cell_data["quality"] = quality_data
                    
            logger.info("✅ PyVista mesh prepared for visualization")
            
        except Exception as e:
            logger.error(f"❌ Visualization preparation failed: {e}")
            raise
    
    def _stage_export_mesh(self, formats: List[str]):
        """Pipeline Stage 4: Export mesh in multiple formats."""
        logger.info(f"💾 Stage 4: Exporting mesh in formats: {formats}")
        
        try:
            for fmt in formats:
                filename = self.output_dir / f"mesh.{fmt}"
                self.geometry.export_mesh(fmt, str(filename))
                logger.info(f"   Exported: {filename}")
                
        except Exception as e:
            logger.error(f"❌ Mesh export failed: {e}")
            raise
    
    def _save_quality_report(self, quality_metrics: Dict[str, float]):
        """Save mesh quality report to file."""
        report_file = self.output_dir / "quality_report.txt"
        
        with open(report_file, "w") as f:
            f.write("Mesh Quality Analysis Report\n")
            f.write("=" * 30 + "\n\n")
            
            f.write(f"Mesh Information:\n")
            f.write(f"  Vertices: {self.mesh_data.num_vertices}\n")
            f.write(f"  Elements: {self.mesh_data.num_elements}\n")
            f.write(f"  Element Type: {self.mesh_data.element_type}\n")
            f.write(f"  Dimension: {self.mesh_data.dimension}\n\n")
            
            f.write("Quality Metrics:\n")
            for metric, value in quality_metrics.items():
                f.write(f"  {metric}: {value:.6f}\n")
                
        logger.info(f"📄 Quality report saved: {report_file}")
    
    def visualize_interactive(self, 
                            show_quality: bool = True,
                            show_edges: bool = True,
                            scalars: Optional[str] = None):
        """Create interactive PyVista visualization."""
        if not self.stages_completed["visualization_ready"]:
            self._stage_prepare_visualization()
            
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("pyvista is required for interactive visualization")
            
        mesh = self.mesh_data.to_pyvista()
        plotter = pv.Plotter(title="MFG_PDE Mesh Visualization")
        
        # Choose scalars for coloring
        if scalars is None and show_quality:
            if "quality" in mesh.cell_data:
                scalars = "quality"
                
        # Add mesh to plotter
        plotter.add_mesh(
            mesh, 
            scalars=scalars,
            show_edges=show_edges,
            cmap="viridis" if scalars else None
        )
        
        # Add information text
        info_text = (f"Vertices: {self.mesh_data.num_vertices}\n"
                    f"Elements: {self.mesh_data.num_elements}\n"
                    f"Type: {self.mesh_data.element_type}")
        plotter.add_text(info_text, position="upper_left")
        
        plotter.show()
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline execution."""
        return {
            "geometry_type": type(self.geometry).__name__,
            "stages_completed": self.stages_completed,
            "mesh_info": {
                "vertices": self.mesh_data.num_vertices if self.mesh_data else 0,
                "elements": self.mesh_data.num_elements if self.mesh_data else 0,
                "element_type": self.mesh_data.element_type if self.mesh_data else None,
                "dimension": self.mesh_data.dimension if self.mesh_data else None
            },
            "output_directory": str(self.output_dir)
        }


class MeshManager:
    """
    High-level mesh management for MFG problems.
    
    This class provides simplified interfaces for common mesh operations,
    managing multiple geometries and coordinating with MFG solvers.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize mesh manager.
        
        Args:
            config: Global mesh configuration
        """
        self.config = config or {}
        self.geometries: Dict[str, BaseGeometry] = {}
        self.pipelines: Dict[str, MeshPipeline] = {}
        
    def create_geometry(self, 
                       name: str,
                       geometry_config: Dict) -> BaseGeometry:
        """Create and register a geometry."""
        if geometry_config.get("dimension", 2) == 2:
            geometry = Domain2D(
                domain_type=geometry_config.get("type", "rectangle"),
                bounds=geometry_config.get("bounds", (0.0, 1.0, 0.0, 1.0)),
                holes=geometry_config.get("holes", []),
                mesh_size=geometry_config.get("mesh_size", 0.1),
                **geometry_config.get("kwargs", {})
            )
        else:
            raise NotImplementedError("3D geometries not yet implemented")
            
        self.geometries[name] = geometry
        return geometry
    
    def create_pipeline(self, 
                       name: str,
                       geometry_name: str,
                       output_dir: Optional[str] = None) -> MeshPipeline:
        """Create and register a mesh pipeline."""
        if geometry_name not in self.geometries:
            raise ValueError(f"Geometry '{geometry_name}' not found")
            
        geometry = self.geometries[geometry_name]
        pipeline = MeshPipeline(geometry, output_dir)
        self.pipelines[name] = pipeline
        
        return pipeline
    
    def batch_generate_meshes(self, 
                             pipeline_names: List[str],
                             stages: List[str] = None) -> Dict[str, MeshData]:
        """Generate multiple meshes in batch."""
        results = {}
        
        for name in pipeline_names:
            if name not in self.pipelines:
                logger.warning(f"Pipeline '{name}' not found, skipping")
                continue
                
            logger.info(f"Processing pipeline: {name}")
            try:
                results[name] = self.pipelines[name].execute_pipeline(stages)
            except Exception as e:
                logger.error(f"Pipeline '{name}' failed: {e}")
                
        return results
    
    def compare_mesh_quality(self, pipeline_names: List[str]) -> Dict[str, Dict]:
        """Compare quality metrics across multiple meshes."""
        comparison = {}
        
        for name in pipeline_names:
            if name not in self.pipelines:
                continue
                
            pipeline = self.pipelines[name]
            if pipeline.mesh_data:
                geometry = pipeline.geometry
                quality = geometry.compute_mesh_quality()
                comparison[name] = quality
                
        return comparison