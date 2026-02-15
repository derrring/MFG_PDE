"""Shared Gmsh mesh base class for 2D and 3D unstructured meshes.

Extracts common constructor, ``set_mesh_parameters``, and ``create_gmsh_geometry``
from Mesh2D and Mesh3D into a single intermediate base class using the
Template Method pattern.

Part of: Issue #802 Phase 3 - Mesh class consolidation
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from mfg_pde.geometry.base import UnstructuredMesh

if TYPE_CHECKING:
    from collections.abc import Callable


class _GmshMeshBase(UnstructuredMesh):
    """Shared Gmsh-based mesh generation logic for 2D and 3D meshes.

    Uses the Template Method pattern: ``create_gmsh_geometry`` defines the
    Gmsh pipeline (initialise -> create shape -> cut holes -> set options),
    while subclasses supply dimension-specific hooks.

    Subclasses **must** implement::

        _setup_domain_parameters  - parse domain kwargs into attributes
        _get_domain_type_dispatch - map domain_type -> creator method
        _get_model_name           - Gmsh model name string
        _add_holes_gmsh           - boolean-cut holes (primitives differ)
        generate_mesh             - element extraction (triangles vs tetrahedra)
        export_mesh               - format-specific export

    Subclasses **may** override::

        _post_gmsh_options        - extra Gmsh options (default: no-op)
    """

    def __init__(
        self,
        dimension: int,
        domain_type: str,
        bounds_tuple: tuple,
        holes: list[dict] | None = None,
        mesh_size: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Initialize Gmsh mesh domain.

        Args:
            dimension: Spatial dimension (2 or 3)
            domain_type: Domain shape identifier (e.g. "rectangle", "box")
            bounds_tuple: Bounding coordinates in dimension-specific format
            holes: List of hole specifications
            mesh_size: Target mesh element size
            **kwargs: Additional domain-specific parameters
        """
        super().__init__(dimension=dimension)
        self.domain_type = domain_type
        self._bounds_tuple = bounds_tuple
        self.holes = holes or []
        self.mesh_size = mesh_size
        self.kwargs = kwargs

        self._setup_domain_parameters()

    # ------------------------------------------------------------------
    # Abstract / hook methods
    # ------------------------------------------------------------------

    @abstractmethod
    def _setup_domain_parameters(self) -> None:
        """Parse domain-specific kwargs into instance attributes."""
        ...

    @abstractmethod
    def _get_domain_type_dispatch(self) -> dict[str, Callable[[], None]]:
        """Return mapping from domain_type string to Gmsh geometry creator."""
        ...

    @abstractmethod
    def _get_model_name(self) -> str:
        """Return Gmsh model name (e.g. ``'domain_2d'``)."""
        ...

    @abstractmethod
    def _add_holes_gmsh(self) -> None:
        """Add holes to domain via Gmsh boolean operations."""
        ...

    def _post_gmsh_options(self) -> None:
        """Hook for additional Gmsh options after standard mesh size setup."""

    # ------------------------------------------------------------------
    # Concrete shared methods
    # ------------------------------------------------------------------

    def set_mesh_parameters(
        self,
        mesh_size: float | None = None,
        algorithm: str = "delaunay",
        **kwargs: Any,
    ) -> None:
        """Set mesh generation parameters."""
        if mesh_size is not None:
            self.mesh_size = mesh_size
        self.mesh_algorithm = algorithm
        self.mesh_kwargs = kwargs

    def create_gmsh_geometry(self) -> Any:
        """Create geometry using Gmsh API (template method).

        Pipeline: initialise Gmsh -> dispatch to shape creator ->
        boolean-cut holes -> set mesh size options -> subclass hook.
        """
        try:
            import gmsh
        except ImportError:
            raise ImportError("gmsh is required for mesh generation") from None

        gmsh.initialize()
        gmsh.clear()
        gmsh.model.add(self._get_model_name())

        dispatch = self._get_domain_type_dispatch()
        creator = dispatch.get(self.domain_type)
        if creator is None:
            raise ValueError(f"Unknown domain type: {self.domain_type}")
        creator()

        self._add_holes_gmsh()

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.mesh_size / 10)
        self._post_gmsh_options()

        return gmsh.model
