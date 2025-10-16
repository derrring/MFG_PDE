"""
Implicit Domain Geometry for Meshfree Methods

This module provides dimension-agnostic implicit domain representation using
signed distance functions (SDFs). Complements MFG_PDE's existing mesh-based
geometry for particle-collocation and meshfree methods.

Key Advantages:
- No mesh generation required (O(d) storage vs O(N^d))
- Natural obstacle representation via CSG operations
- Dimension-agnostic (works for 2D, 3D, 4D, ..., any d)
- Efficient for particle methods

Components:
- ImplicitDomain: Abstract base class using signed distance functions
- Hyperrectangle: Axis-aligned boxes in n dimensions
- Hypersphere: Balls/spheres in n dimensions
- CSG Operations: Union, Intersection, Complement, Difference

Example - Domain with Obstacle:
    >>> from mfg_pde.geometry.implicit import Hyperrectangle, Hypersphere, DifferenceDomain
    >>> import numpy as np
    >>>
    >>> # Create base domain (unit square)
    >>> square = Hyperrectangle(np.array([[0, 1], [0, 1]]))
    >>>
    >>> # Add circular obstacle
    >>> obstacle = Hypersphere(center=[0.5, 0.5], radius=0.2)
    >>>
    >>> # Navigable domain = base - obstacle
    >>> domain = DifferenceDomain(square, obstacle)
    >>>
    >>> # Sample particles (automatically avoids obstacle)
    >>> particles = domain.sample_uniform(5000, seed=42)
    >>> assert np.all(domain.contains(particles))

Example - 4D Hypercube:
    >>> # Works for arbitrary dimensions
    >>> hypercube_4d = Hyperrectangle(np.array([[0, 1]] * 4))
    >>> particles_4d = hypercube_4d.sample_uniform(10000)
    >>> assert particles_4d.shape == (10000, 4)

Relation to Existing Geometry:
- Complements Domain2D/Domain3D (mesh-based)
- Use when:
  - d > 3 (mesh generation impractical)
  - Particle methods (no mesh needed)
  - Complex obstacles (CSG operations easier than remeshing)

For maze environments, see mfg_pde.alg.reinforcement.environments.

Mathematical Foundation:
    An implicit domain D ⊂ ℝ^d is defined by signed distance function φ:
        x ∈ D  ⟺  φ(x) < 0   (interior)
        x ∈ ∂D ⟺  φ(x) = 0   (boundary)
        x ∉ D  ⟺  φ(x) > 0   (exterior)

References:
    - Osher & Fedkiw (2003): Level Set Methods and Dynamic Implicit Surfaces
    - MFG_PDE geometry system: mfg_pde/geometry/README.md

Author: MFG_PDE Team
Date: October 2025
"""

from .csg_operations import ComplementDomain, DifferenceDomain, IntersectionDomain, UnionDomain
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
