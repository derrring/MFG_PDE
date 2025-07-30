"""
2D domain implementation for MFG_PDE geometry system.

This module implements 2D geometric domains using the Gmsh → Meshio → PyVista pipeline,
supporting rectangular domains, complex shapes, holes, and CAD import.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from .base_geometry import BaseGeometry, MeshData


class Domain2D(BaseGeometry):
    """2D domain with complex geometry support using Gmsh pipeline."""
    
    def __init__(self, 
                 domain_type: str = "rectangle",
                 bounds: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
                 holes: Optional[List[Dict]] = None,
                 mesh_size: float = 0.1,
                 **kwargs):
        """
        Initialize 2D domain.
        
        Args:
            domain_type: "rectangle", "circle", "polygon", "custom", "cad_import"
            bounds: (xmin, xmax, ymin, ymax) for rectangular domains
            holes: List of hole specifications [{"type": "circle", "center": (x,y), "radius": r}, ...]
            mesh_size: Target mesh element size
            **kwargs: Additional domain-specific parameters
        """
        super().__init__(dimension=2)
        
        self.domain_type = domain_type
        self.bounds_rect = bounds  # (xmin, xmax, ymin, ymax)
        self.holes = holes or []
        self.mesh_size = mesh_size
        self.kwargs = kwargs
        
        # Domain-specific parameters
        self._setup_domain_parameters()
        
    def _setup_domain_parameters(self):
        """Setup parameters based on domain type."""
        if self.domain_type == "rectangle":
            self.xmin, self.xmax, self.ymin, self.ymax = self.bounds_rect
            
        elif self.domain_type == "circle":
            self.center = self.kwargs.get("center", (0.5, 0.5))
            self.radius = self.kwargs.get("radius", 0.4)
            
        elif self.domain_type == "polygon":
            self.vertices = self.kwargs.get("vertices", [])
            if len(self.vertices) < 3:
                raise ValueError("Polygon must have at least 3 vertices")
                
        elif self.domain_type == "custom":
            self.geometry_func = self.kwargs.get("geometry_func")
            if self.geometry_func is None:
                raise ValueError("Custom domain requires 'geometry_func' parameter")
                
        elif self.domain_type == "cad_import":
            self.cad_file = self.kwargs.get("cad_file")
            if self.cad_file is None:
                raise ValueError("CAD import requires 'cad_file' parameter")
    
    def create_gmsh_geometry(self) -> Any:
        """Create 2D geometry using Gmsh API."""
        try:
            import gmsh
        except ImportError:
            raise ImportError("gmsh is required for mesh generation")
            
        gmsh.initialize()
        gmsh.clear()
        gmsh.model.add("domain_2d")
        
        if self.domain_type == "rectangle":
            self._create_rectangle_gmsh()
        elif self.domain_type == "circle":
            self._create_circle_gmsh()
        elif self.domain_type == "polygon":
            self._create_polygon_gmsh()
        elif self.domain_type == "custom":
            self._create_custom_gmsh()
        elif self.domain_type == "cad_import":
            self._import_cad_gmsh()
        else:
            raise ValueError(f"Unknown domain type: {self.domain_type}")
            
        # Add holes if specified
        self._add_holes_gmsh()
        
        # Set mesh parameters
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.mesh_size / 10)
        
        self._gmsh_model = gmsh.model
        return gmsh.model
    
    def _create_rectangle_gmsh(self):
        """Create rectangular domain in Gmsh."""
        import gmsh
        
        # Create corner points
        p1 = gmsh.model.geo.addPoint(self.xmin, self.ymin, 0.0)
        p2 = gmsh.model.geo.addPoint(self.xmax, self.ymin, 0.0)
        p3 = gmsh.model.geo.addPoint(self.xmax, self.ymax, 0.0)
        p4 = gmsh.model.geo.addPoint(self.xmin, self.ymax, 0.0)
        
        # Create edges
        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)
        
        # Create curve loop and surface
        curve_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        self.main_surface = gmsh.model.geo.addPlaneSurface([curve_loop])
        
        # Add physical groups for boundary conditions
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, [l1], 1)  # Bottom boundary
        gmsh.model.addPhysicalGroup(1, [l2], 2)  # Right boundary  
        gmsh.model.addPhysicalGroup(1, [l3], 3)  # Top boundary
        gmsh.model.addPhysicalGroup(1, [l4], 4)  # Left boundary
        gmsh.model.addPhysicalGroup(2, [self.main_surface], 1)  # Domain
        
    def _create_circle_gmsh(self):
        """Create circular domain in Gmsh."""
        import gmsh
        
        center_x, center_y = self.center
        
        # Create center point
        center_point = gmsh.model.geo.addPoint(center_x, center_y, 0.0)
        
        # Create points on circumference
        p1 = gmsh.model.geo.addPoint(center_x + self.radius, center_y, 0.0)
        p2 = gmsh.model.geo.addPoint(center_x, center_y + self.radius, 0.0)
        p3 = gmsh.model.geo.addPoint(center_x - self.radius, center_y, 0.0)
        p4 = gmsh.model.geo.addPoint(center_x, center_y - self.radius, 0.0)
        
        # Create circular arcs
        c1 = gmsh.model.geo.addCircleArc(p1, center_point, p2)
        c2 = gmsh.model.geo.addCircleArc(p2, center_point, p3)
        c3 = gmsh.model.geo.addCircleArc(p3, center_point, p4)
        c4 = gmsh.model.geo.addCircleArc(p4, center_point, p1)
        
        # Create curve loop and surface
        curve_loop = gmsh.model.geo.addCurveLoop([c1, c2, c3, c4])
        self.main_surface = gmsh.model.geo.addPlaneSurface([curve_loop])
        
        # Add physical groups
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, [c1, c2, c3, c4], 1)  # Circular boundary
        gmsh.model.addPhysicalGroup(2, [self.main_surface], 1)  # Domain
        
    def _create_polygon_gmsh(self):
        """Create polygonal domain in Gmsh."""
        import gmsh
        
        # Create points
        points = []
        for x, y in self.vertices:
            points.append(gmsh.model.geo.addPoint(x, y, 0.0))
            
        # Create lines
        lines = []
        for i in range(len(points)):
            next_i = (i + 1) % len(points)
            lines.append(gmsh.model.geo.addLine(points[i], points[next_i]))
            
        # Create curve loop and surface
        curve_loop = gmsh.model.geo.addCurveLoop(lines)
        self.main_surface = gmsh.model.geo.addPlaneSurface([curve_loop])
        
        # Add physical groups
        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, lines, 1)  # Boundary
        gmsh.model.addPhysicalGroup(2, [self.main_surface], 1)  # Domain
        
    def _create_custom_gmsh(self):
        """Create custom domain using user-provided function."""
        # User-provided geometry function should create Gmsh geometry
        # and set self.main_surface
        self.main_surface = self.geometry_func(gmsh)
        
    def _import_cad_gmsh(self):
        """Import CAD geometry into Gmsh."""
        import gmsh
        
        # Import CAD file
        gmsh.model.occ.importShapes(self.cad_file)
        gmsh.model.occ.synchronize()
        
        # Get surfaces (assuming single surface for now)
        surfaces = gmsh.model.getEntities(2)
        if len(surfaces) == 0:
            raise ValueError("No surfaces found in CAD file")
        self.main_surface = surfaces[0][1]
        
    def _add_holes_gmsh(self):
        """Add holes to the domain."""
        if not self.holes:
            return
            
        import gmsh
        
        hole_surfaces = []
        
        for hole in self.holes:
            if hole["type"] == "circle":
                center = hole["center"]
                radius = hole["radius"]
                
                # Create hole geometry
                center_point = gmsh.model.geo.addPoint(center[0], center[1], 0.0)
                
                # Create points on circumference
                p1 = gmsh.model.geo.addPoint(center[0] + radius, center[1], 0.0)
                p2 = gmsh.model.geo.addPoint(center[0], center[1] + radius, 0.0)
                p3 = gmsh.model.geo.addPoint(center[0] - radius, center[1], 0.0)
                p4 = gmsh.model.geo.addPoint(center[0], center[1] - radius, 0.0)
                
                # Create circular arcs
                c1 = gmsh.model.geo.addCircleArc(p1, center_point, p2)
                c2 = gmsh.model.geo.addCircleArc(p2, center_point, p3)
                c3 = gmsh.model.geo.addCircleArc(p3, center_point, p4)
                c4 = gmsh.model.geo.addCircleArc(p4, center_point, p1)
                
                # Create hole curve loop and surface
                hole_loop = gmsh.model.geo.addCurveLoop([c1, c2, c3, c4])
                hole_surface = gmsh.model.geo.addPlaneSurface([hole_loop])
                hole_surfaces.append(hole_surface)
                
            elif hole["type"] == "rectangle":
                bounds = hole["bounds"]  # (xmin, xmax, ymin, ymax)
                xmin, xmax, ymin, ymax = bounds
                
                # Create rectangle hole
                p1 = gmsh.model.geo.addPoint(xmin, ymin, 0.0)
                p2 = gmsh.model.geo.addPoint(xmax, ymin, 0.0)
                p3 = gmsh.model.geo.addPoint(xmax, ymax, 0.0)
                p4 = gmsh.model.geo.addPoint(xmin, ymax, 0.0)
                
                l1 = gmsh.model.geo.addLine(p1, p2)
                l2 = gmsh.model.geo.addLine(p2, p3)
                l3 = gmsh.model.geo.addLine(p3, p4)
                l4 = gmsh.model.geo.addLine(p4, p1)
                
                hole_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
                hole_surface = gmsh.model.geo.addPlaneSurface([hole_loop])
                hole_surfaces.append(hole_surface)
                
        # Subtract holes from main surface
        if hole_surfaces:
            gmsh.model.geo.synchronize()
            
            # Boolean difference: main_surface - holes
            result = gmsh.model.occ.cut(
                [(2, self.main_surface)],
                [(2, surf) for surf in hole_surfaces]
            )
            gmsh.model.occ.synchronize()
            
            if result[0]:
                self.main_surface = result[0][0][1]
                
    def set_mesh_parameters(self, 
                          mesh_size: Optional[float] = None,
                          algorithm: str = "delaunay",
                          **kwargs) -> None:
        """Set mesh generation parameters."""
        if mesh_size is not None:
            self.mesh_size = mesh_size
            
        self.mesh_algorithm = algorithm
        self.mesh_kwargs = kwargs
        
    def generate_mesh(self) -> MeshData:
        """Generate 2D triangular mesh using Gmsh → Meshio pipeline."""
        try:
            import gmsh
            import meshio
        except ImportError:
            raise ImportError("gmsh and meshio are required for mesh generation")
            
        # Create geometry
        self.create_gmsh_geometry()
        
        # Generate mesh
        gmsh.model.mesh.generate(2)  # 2D mesh
        
        # Extract mesh data
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        element_types, element_tags, element_connectivity = gmsh.model.mesh.getElements()
        
        # Process vertices
        vertices = node_coords.reshape(-1, 3)[:, :2]  # Remove z-coordinate for 2D
        
        # Process elements (triangles)
        triangles = None
        for i, elem_type in enumerate(element_types):
            if elem_type == 2:  # Triangle element type in Gmsh
                connectivity = element_connectivity[i]
                triangles = connectivity.reshape(-1, 3) - 1  # Convert to 0-based indexing
                break
                
        if triangles is None:
            raise RuntimeError("No triangular elements found in generated mesh")
            
        # Get boundary information
        boundary_tags = np.zeros(len(vertices), dtype=int)  # Placeholder
        element_tags_array = np.ones(len(triangles), dtype=int)  # All elements in same region
        
        # Create boundary faces (edges for 2D)
        boundary_faces = self._extract_boundary_edges(triangles)
        
        gmsh.finalize()
        
        # Create MeshData object
        self.mesh_data = MeshData(
            vertices=vertices,
            elements=triangles,
            element_type="triangle",
            boundary_tags=boundary_tags,
            element_tags=element_tags_array,
            boundary_faces=boundary_faces,
            dimension=2,
            metadata={
                "domain_type": self.domain_type,
                "mesh_size": self.mesh_size,
                "bounds": self.bounds_rect
            }
        )
        
        return self.mesh_data
    
    def _extract_boundary_edges(self, triangles: np.ndarray) -> np.ndarray:
        """Extract boundary edges from triangular mesh."""
        # Create edge dictionary to find boundary edges
        edge_count = {}
        
        # Count each edge
        for triangle in triangles:
            edges = [
                tuple(sorted([triangle[0], triangle[1]])),
                tuple(sorted([triangle[1], triangle[2]])),
                tuple(sorted([triangle[2], triangle[0]]))
            ]
            
            for edge in edges:
                edge_count[edge] = edge_count.get(edge, 0) + 1
                
        # Boundary edges appear only once
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        
        return np.array(boundary_edges)
    
    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get 2D domain bounding box."""
        if self.domain_type == "rectangle":
            min_coords = np.array([self.xmin, self.ymin])
            max_coords = np.array([self.xmax, self.ymax])
            return min_coords, max_coords
            
        elif self.domain_type == "circle":
            center_x, center_y = self.center
            min_coords = np.array([center_x - self.radius, center_y - self.radius])
            max_coords = np.array([center_x + self.radius, center_y + self.radius])
            return min_coords, max_coords
            
        elif self.domain_type == "polygon":
            vertices_array = np.array(self.vertices)
            min_coords = np.min(vertices_array, axis=0)
            max_coords = np.max(vertices_array, axis=0)
            return min_coords, max_coords
            
        else:
            # For custom domains, generate mesh first to get bounds
            if self.mesh_data is None:
                self.generate_mesh()
            return self.mesh_data.bounds
    
    def export_mesh(self, format: str, filename: str) -> None:
        """Export mesh in specified format using meshio."""
        if self.mesh_data is None:
            self.generate_mesh()
            
        meshio_mesh = self.mesh_data.to_meshio()
        meshio_mesh.write(filename, file_format=format)
        
    def refine_mesh(self, 
                   refinement_factor: float = 0.5,
                   adaptive: bool = False,
                   indicator_function: Optional[Callable] = None) -> MeshData:
        """Refine mesh uniformly or adaptively."""
        if adaptive and indicator_function is None:
            raise ValueError("Adaptive refinement requires indicator_function")
            
        if adaptive:
            # Adaptive refinement based on indicator function
            return self._refine_adaptive(indicator_function)
        else:
            # Uniform refinement
            self.mesh_size *= refinement_factor
            return self.generate_mesh()
            
    def _refine_adaptive(self, indicator_function: Callable) -> MeshData:
        """Adaptive mesh refinement using indicator function."""
        # This is a placeholder for adaptive refinement
        # Full implementation would involve error estimation and local refinement
        raise NotImplementedError("Adaptive refinement not yet implemented")