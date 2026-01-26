"""
Unified boundary condition dispatch for solvers.

This module provides a single entry point for BC application that:
1. Auto-detects geometry type and dimension
2. Selects the appropriate applicator
3. Handles BC validation and caching

This is the recommended interface for solvers that implement BoundaryCapable.

Usage:
    >>> from mfg_pde.geometry.boundary.dispatch import apply_bc, get_applicator_for_geometry
    >>>
    >>> # Simple usage: auto-detect everything
    >>> padded_field = apply_bc(geometry, field, boundary_conditions)
    >>>
    >>> # For repeated application (performance): get reusable applicator
    >>> applicator = get_applicator_for_geometry(geometry, bc)
    >>> padded_field_1 = applicator.apply(field_1)
    >>> padded_field_2 = applicator.apply(field_2)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .applicator_base import DiscretizationType

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from mfg_pde.geometry.protocol import GeometryProtocol

    from .conditions import BoundaryConditions


def _has_implicit_boundary(geometry: object) -> bool:
    """
    Check if geometry uses implicit (SDF-based) boundary definition.

    ImplicitApplicator is used when geometry has SDF-based methods for
    boundary detection but is NOT a structured grid (like TensorProductGrid).

    Args:
        geometry: Geometry object to check

    Returns:
        True if geometry has SDF-based boundary detection
    """
    # Check for SDF-based boundary detection methods
    has_sdf = hasattr(geometry, "sdf") or hasattr(geometry, "is_on_boundary")

    # Check if it's a structured grid (which uses FDMApplicator)
    is_structured = hasattr(geometry, "grid_points") and hasattr(geometry, "spacing")

    # Use ImplicitApplicator for SDF-based non-structured geometries
    return has_sdf and not is_structured


def get_applicator_for_geometry(
    geometry: GeometryProtocol,
    discretization: DiscretizationType | str = DiscretizationType.FDM,
) -> object:
    """
    Get the appropriate BC applicator for a geometry.

    This factory function selects the right applicator class based on:
    1. Geometry dimension
    2. Discretization type (FDM, FEM, GFDM, etc.)

    Args:
        geometry: Geometry object implementing GeometryProtocol
        discretization: Discretization method. Options:
            - DiscretizationType.FDM (default): Ghost cell method
            - DiscretizationType.GFDM: Meshfree collocation
            - DiscretizationType.FEM: Matrix modification
            - DiscretizationType.GRAPH: Network boundaries

    Returns:
        Appropriate applicator instance

    Example:
        >>> from mfg_pde.geometry import TensorProductGrid
        >>> from mfg_pde.geometry.boundary.dispatch import get_applicator_for_geometry
        >>>
        >>> grid = TensorProductGrid(bounds=[(0, 1), (0, 1)], Nx_points=[11, 11])
        >>> applicator = get_applicator_for_geometry(grid)
        >>> # applicator is an FDMApplicator
    """
    # Normalize discretization type
    if isinstance(discretization, str):
        discretization = DiscretizationType[discretization.upper()]

    dim = geometry.dimension

    if discretization == DiscretizationType.FDM:
        from .applicator_fdm import FDMApplicator

        return FDMApplicator(dimension=dim)

    elif discretization == DiscretizationType.GFDM:
        # GFDM uses meshfree applicator
        from .applicator_meshfree import MeshfreeApplicator

        return MeshfreeApplicator(geometry=geometry)

    elif discretization == DiscretizationType.FEM:
        from .applicator_fem import FEMApplicator

        return FEMApplicator(dimension=dim)

    elif discretization == DiscretizationType.GRAPH:
        from .applicator_graph import GraphApplicator

        # Get node count from geometry
        # Issue #543: Use getattr() to normalize attribute naming
        num_nodes = getattr(geometry, "num_nodes", None) or getattr(geometry, "num_spatial_points", None)
        if num_nodes is None:
            raise ValueError("Graph geometry must have 'num_nodes' or 'num_spatial_points' attribute")
        return GraphApplicator(num_nodes=num_nodes)

    elif discretization == DiscretizationType.MESHFREE:
        # Issue #637: Use ImplicitApplicator if geometry has SDF-based boundaries
        # Otherwise use MeshfreeApplicator
        if _has_implicit_boundary(geometry):
            from .applicator_implicit import ImplicitApplicator

            return ImplicitApplicator(geometry=geometry)
        else:
            from .applicator_meshfree import MeshfreeApplicator

            return MeshfreeApplicator(geometry=geometry)

    else:
        raise ValueError(f"Unsupported discretization type: {discretization}")


def apply_bc(
    geometry: GeometryProtocol,
    field: NDArray[np.floating],
    boundary_conditions: BoundaryConditions,
    *,
    time: float = 0.0,
    discretization: DiscretizationType | str = DiscretizationType.FDM,
    ghost_depth: int = 1,
    points: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """
    Apply boundary conditions to a field using geometry-appropriate method.

    This is the unified entry point for BC application. It:
    1. Detects geometry dimension and type
    2. Selects the appropriate applicator
    3. Applies BCs and returns the result

    Args:
        geometry: Geometry object implementing GeometryProtocol
        field: Field array to apply BCs to (interior values)
        boundary_conditions: BC specification
        time: Time for time-dependent BCs (default: 0.0)
        discretization: Discretization method (default: FDM)
        ghost_depth: Ghost cell depth for FDM (default: 1)
        points: Collocation points for meshfree methods (optional).
                If not provided, uses geometry.get_collocation_points().

    Returns:
        Field with BCs applied (padded for FDM, modified for FEM/GFDM)

    Example:
        >>> from mfg_pde.geometry import TensorProductGrid
        >>> from mfg_pde.geometry.boundary import neumann_bc
        >>> from mfg_pde.geometry.boundary.dispatch import apply_bc
        >>>
        >>> grid = TensorProductGrid(bounds=[(0, 1)], Nx_points=[11])
        >>> field = np.ones(11)
        >>> bc = neumann_bc(dimension=1)
        >>> padded = apply_bc(grid, field, bc)
        >>> # padded.shape == (13,) for ghost_depth=1

    Performance Note:
        For repeated application (e.g., time-stepping), use
        get_applicator_for_geometry() to get a reusable applicator
        instead of calling apply_bc() repeatedly.
    """
    # Normalize discretization type
    if isinstance(discretization, str):
        discretization = DiscretizationType[discretization.upper()]

    if discretization == DiscretizationType.FDM:
        # Issue #577 Phase 3: Use pad_array_with_ghosts() for all BCs
        # Geometry parameter enables region_name resolution for mixed BCs
        from .applicator_fdm import pad_array_with_ghosts

        return pad_array_with_ghosts(
            field,
            boundary_conditions,
            ghost_depth=ghost_depth,
            time=time,
            geometry=geometry,
        )

    elif discretization in (DiscretizationType.GFDM, DiscretizationType.MESHFREE):
        # Meshfree: use MeshfreeApplicator with unified API
        from .applicator_meshfree import MeshfreeApplicator

        applicator = MeshfreeApplicator(geometry=geometry)

        # Get collocation points if not provided
        if points is None:
            points = geometry.get_collocation_points()

        # Use unified apply() method (Issue #636 Phase 2)
        return applicator.apply(field, boundary_conditions, points, time=time)

    elif discretization == DiscretizationType.FEM:
        # FEM: would modify matrix/rhs, not field directly
        raise NotImplementedError(
            "FEM BC application requires matrix/rhs modification. Use FEMApplicator.apply(matrix, rhs, mesh) instead."
        )

    else:
        raise ValueError(f"Unsupported discretization type: {discretization}")


def validate_bc_compatibility(
    boundary_conditions: BoundaryConditions,
    geometry: GeometryProtocol,
    discretization: DiscretizationType | str = DiscretizationType.FDM,
) -> list[str]:
    """
    Validate that boundary conditions are compatible with geometry and discretization.

    Args:
        boundary_conditions: BC specification to validate
        geometry: Target geometry
        discretization: Discretization method

    Returns:
        List of warning/error messages (empty if valid)

    Example:
        >>> from mfg_pde.geometry import TensorProductGrid
        >>> from mfg_pde.geometry.boundary import robin_bc
        >>> from mfg_pde.geometry.boundary.dispatch import validate_bc_compatibility
        >>>
        >>> grid = TensorProductGrid(bounds=[(0, 1)], Nx_points=[11])
        >>> bc = robin_bc(dimension=1, alpha=1.0, beta=1.0)
        >>> issues = validate_bc_compatibility(bc, grid, discretization="FDM")
        >>> # Robin BC is supported for FDM, so issues == []
    """
    issues = []

    # Normalize discretization
    if isinstance(discretization, str):
        discretization = DiscretizationType[discretization.upper()]

    # Check dimension match
    if boundary_conditions.dimension != geometry.dimension:
        issues.append(
            f"Dimension mismatch: BC dimension ({boundary_conditions.dimension}) "
            f"!= geometry dimension ({geometry.dimension})"
        )

    # Check BC type support by discretization
    from .types import BCType

    # Issue #543: Use getattr() for optional attribute instead of hasattr
    bc_type = getattr(boundary_conditions, "default_bc", None)

    if discretization == DiscretizationType.FDM:
        # FDM supports: Dirichlet, Neumann, Robin, Periodic, No-flux
        unsupported_fdm = {BCType.REFLECTING}  # Reflecting is for particles
        if bc_type in unsupported_fdm:
            issues.append(f"BC type {bc_type} is not supported for FDM discretization")

    elif discretization in (DiscretizationType.GFDM, DiscretizationType.MESHFREE):
        # GFDM: primarily Dirichlet and Neumann
        limited_gfdm = {BCType.ROBIN, BCType.REFLECTING}
        if bc_type in limited_gfdm:
            issues.append(
                f"BC type {bc_type} has limited support for GFDM/Meshfree. Consider Dirichlet or Neumann instead."
            )

    return issues


__all__ = [
    "apply_bc",
    "get_applicator_for_geometry",
    "validate_bc_compatibility",
]
