"""
Geometry Infrastructure for Meshfree MFG Methods

DEPRECATION NOTICE:
    This module has been graduated to MFG_PDE (PR #189). After PR #189 merges,
    import from mfg_pde.geometry.implicit instead:

        from mfg_pde.geometry.implicit import Hyperrectangle, Hypersphere, DifferenceDomain

    This local copy will be removed once PR #189 is merged to MFG_PDE main.

This module provides dimension-agnostic implicit domain representation for
particle-collocation methods. No mesh generation required!

Key Components:
- ImplicitDomain: Base class using signed distance functions (SDFs)
- Hyperrectangle: Axis-aligned boxes (most common)
- Hypersphere: Balls/circles (for obstacles)
- CSG Operations: Union, Intersection, Complement (for mazes)

Example: Domain with Circular Obstacle
---------------------------------------
>>> from geometry import Hyperrectangle, Hypersphere, DifferenceDomain
>>>
>>> # Base domain: unit square [0,1]²
>>> square = Hyperrectangle(np.array([[0, 1], [0, 1]]))
>>>
>>> # Circular obstacle at center
>>> obstacle = Hypersphere(center=[0.5, 0.5], radius=0.2)
>>>
>>> # Navigable domain = square \\ obstacle
>>> domain = DifferenceDomain(square, obstacle)
>>>
>>> # Sample particles (automatically avoids obstacle!)
>>> particles = domain.sample_uniform(5000)
>>> assert np.all(domain.contains(particles))  # All valid


Example: Maze Environment
-------------------------
>>> from geometry import MazeEnvironment
>>>
>>> # ASCII maze representation
>>> maze_map = '''
... ###########
... #.........#
... #.###.###.#
... #.........#
... ###########
... '''
>>>
>>> # Create domain
>>> maze = MazeEnvironment.from_ascii_map(maze_map)
>>>
>>> # Sample particles only in free space
>>> particles = maze.sample_uniform(10000)


Advantages Over Mesh Methods:
- Memory: O(d) vs O(N^d) for mesh
- Obstacles: Free with CSG operations
- Dimension-agnostic: Same code for 2D, 3D, 4D, ..., 100D
- Particle-friendly: Natural boundary handling

References:
- STAGE2_IMPLEMENTATION_PLAN.md (Week 1-2)
- TECHNICAL_REFERENCE_HIGH_DIMENSIONAL_MFG.md Section 4
"""

from .csg_operations import (
    ComplementDomain,
    DifferenceDomain,
    IntersectionDomain,
    UnionDomain,
)
from .hyperrectangle import Hyperrectangle
from .hypersphere import Hypersphere
from .implicit_domain import ImplicitDomain

__all__ = [
    "ComplementDomain",
    "DifferenceDomain",
    "Hyperrectangle",
    "Hypersphere",
    "ImplicitDomain",
    "IntersectionDomain",
    "UnionDomain",
]

__version__ = "0.1.0"
