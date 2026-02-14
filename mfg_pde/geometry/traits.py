"""
Geometry trait enums and protocols for dispatch.

Realizes 3 of the 8 atomic traits from SPEC-GEO-1.0 (GEOMETRY_AND_TOPOLOGY.md)
as lightweight enums + opt-in protocols. These traits drive actual dispatch
decisions in solvers and factories; the remaining 5 traits are handled by
other mechanisms (see GEOMETRY_BC_INFRASTRUCTURE.md Section 5.3).

Trait overview:
    ConnectivityType — how neighbor relationships are determined
    StructureType    — whether the geometry has logical (i,j,k) indexing
    BoundaryDef      — how the domain boundary is defined

Usage:
    # Fine-grained: check only the trait you need
    if isinstance(geometry, StructureAware):
        if geometry.structure_type == StructureType.STRUCTURED:
            field = field.reshape(grid_shape)

    # Composite: check all traits at once
    if isinstance(geometry, TraitAwareGeometry):
        spec = DomainSpec.from_geometry(geometry)

    # Fallback for non-trait-aware geometries
    elif geometry.geometry_type == GeometryType.CARTESIAN_GRID:
        field = field.reshape(grid_shape)  # legacy path

References:
    - GEOMETRY_BC_INFRASTRUCTURE.md Section 5.5 Stage 1
    - GEOMETRY_AND_TOPOLOGY.md (SPEC-GEO-1.0) Sections 2.1, 2.2, 2.5
    - Issue #732 Tier 1b
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

# =============================================================================
# Trait Enums
# =============================================================================


class ConnectivityType(Enum):
    """How neighbor relationships are determined.

    Directly dictates memory access patterns and kernel selection.

    Values:
        IMPLICIT: Neighbors via stride arithmetic (TensorProductGrid).
            Zero memory overhead, favorable for SIMD/vectorization.
        EXPLICIT: Neighbors stored in adjacency structure (Mesh, NetworkGeometry).
            Memory bandwidth bound, supports arbitrary topology.
        DYNAMIC: Neighbors via runtime spatial search (ImplicitDomain, meshfree).
            High compute cost, suitable for meshfree/SPH methods.
    """

    IMPLICIT = "implicit"
    EXPLICIT = "explicit"
    DYNAMIC = "dynamic"


class StructureType(Enum):
    """Whether the geometry has regular logical indexing.

    Determines whether fields can be reshaped to (Nx, Ny, ...) grids
    and whether stencil-based operators (FDM, WENO) are applicable.

    Values:
        STRUCTURED: Nodes form a regular lattice with logical (i,j,k) coords.
        UNSTRUCTURED: Arbitrary point cloud, no concept of logical coords.
    """

    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"


class BoundaryDef(Enum):
    """How the domain boundary is defined.

    Influences BC enforcement strategy and boundary detection algorithms.

    Values:
        BOX: Axis-aligned hyper-rectangular bounds (AABB).
            Cheapest: boundary detection via coordinate comparison.
        MESH: Explicit boundary elements (facets from Gmsh).
            Supports curved and complex boundaries.
        IMPLICIT: Signed distance function phi(x) = 0.
            Dimension-agnostic, natural for CSG.
        NONE: No boundary (periodic domains, graphs, open domains).
    """

    BOX = "box"
    MESH = "mesh"
    IMPLICIT = "implicit"
    NONE = "none"


# =============================================================================
# Individual Trait Protocols (one per trait, composable)
# =============================================================================


@runtime_checkable
class ConnectivityAware(Protocol):
    """Geometry that declares its connectivity type.

    Solvers use this to select kernel implementations:
    - IMPLICIT -> vectorized stencil operations
    - EXPLICIT -> sparse matrix operations
    - DYNAMIC  -> spatial search + local approximation
    """

    @property
    def connectivity_type(self) -> ConnectivityType: ...


@runtime_checkable
class StructureAware(Protocol):
    """Geometry that declares its structure type.

    Solvers use this to determine whether fields can be reshaped
    to grid arrays and whether FDM stencils are applicable.
    """

    @property
    def structure_type(self) -> StructureType: ...


@runtime_checkable
class BoundaryAware(Protocol):
    """Geometry that declares its boundary definition type.

    BC applicators use this to select enforcement strategy:
    - BOX -> coordinate comparison + ghost cells
    - MESH -> element-based enforcement
    - IMPLICIT -> SDF projection
    - NONE -> no enforcement needed
    """

    @property
    def boundary_def(self) -> BoundaryDef: ...


# =============================================================================
# Composite Protocol
# =============================================================================


@runtime_checkable
class TraitAwareGeometry(ConnectivityAware, StructureAware, BoundaryAware, Protocol):
    """Geometry implementing all current trait properties.

    Use for full trait dispatch (e.g., DomainSpec inference).
    For partial dispatch, use the individual protocols instead.

    Adding future traits (MetricAware, EmbeddingAware, ...) requires:
    1. New enum + new individual protocol (additive)
    2. New composite that extends this one (additive)
    3. Geometry classes add 1 property each (no existing properties change)
    """

    ...
