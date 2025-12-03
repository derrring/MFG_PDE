"""
1D FDM Boundary Conditions for MFG Problems.

.. deprecated:: 0.14.0
    This module is deprecated. Use the unified boundary condition API instead:

    **Old (deprecated):**
        from mfg_pde.geometry.boundary.fdm_bc_1d import BoundaryConditions, periodic_bc
        bc = BoundaryConditions(type="periodic")
        bc = periodic_bc()

    **New (recommended):**
        from mfg_pde.geometry import periodic_bc
        bc = periodic_bc(dimension=1)

    The unified API from conditions.py supports all dimensions and mixed BCs.
    This module will be removed in v1.0.0.

This module provides simple boundary condition specification for 1D finite
difference methods. Uses left/right value pattern for 1D domain endpoints.

For multi-dimensional or segment-based BC specification, use conditions.py.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass


def _emit_deprecation_warning(func_name: str = "BoundaryConditions") -> None:
    """Emit deprecation warning for fdm_bc_1d module usage."""
    warnings.warn(
        f"mfg_pde.geometry.boundary.fdm_bc_1d.{func_name} is deprecated. "
        "Use the unified API instead:\n"
        "  from mfg_pde.geometry import periodic_bc, dirichlet_bc, neumann_bc\n"
        "  bc = periodic_bc(dimension=1)\n"
        "This module will be removed in v1.0.0.",
        DeprecationWarning,
        stacklevel=3,
    )


@dataclass
class BoundaryConditions:
    """
    Boundary condition configuration for 1D MFG problems.

    .. deprecated:: 0.14.0
        Use :class:`mfg_pde.geometry.boundary.conditions.BoundaryConditions` instead.

    This class defines boundary conditions for the spatial domain boundaries,
    specifying how the solution should behave at the domain endpoints.

    Note on matrix dimensions for different boundary condition types:
    - periodic: M × M (domain is treated as circular)
    - dirichlet: (M-1) × (M-1) (solution fixed at boundaries)
    - neumann: (M+1) × (M+1) (gradient specified at boundaries)
    - no_flux: M × M (special case of Neumann for FP equations: F(boundary) = 0)
    - robin: (M+1) × (M+1) (mixed boundary condition: αu + βdu/dn = g)

    where M is the number of interior grid points.
    """

    type: str  # 'periodic', 'dirichlet', 'neumann', 'no_flux', or 'robin'

    # Boundary values
    # For Dirichlet: value of u at boundary
    # For Neumann: value of du/dn at boundary
    # For no_flux: F(boundary) = 0 where F = v*m - D*dm/dx
    left_value: float | None = None
    right_value: float | None = None

    # Robin boundary condition parameters: αu + βdu/dn = g
    # α coefficients (multiplier of solution value)
    left_alpha: float | None = None  # coefficient of u at left boundary
    left_beta: float | None = None  # coefficient of du/dn at left boundary
    right_alpha: float | None = None  # coefficient of u at right boundary
    right_beta: float | None = None  # coefficient of du/dn at right boundary

    def __post_init__(self):
        """Validate boundary condition parameters."""
        if self.type == "robin":
            if any(
                v is None
                for v in [
                    self.left_alpha,
                    self.left_beta,
                    self.right_alpha,
                    self.right_beta,
                ]
            ):
                raise ValueError("Robin boundary conditions require alpha and beta coefficients")

    def is_periodic(self) -> bool:
        """Check if boundary conditions are periodic."""
        return self.type == "periodic"

    def is_dirichlet(self) -> bool:
        """Check if boundary conditions are Dirichlet."""
        return self.type == "dirichlet"

    def is_neumann(self) -> bool:
        """Check if boundary conditions are Neumann."""
        return self.type == "neumann"

    def is_no_flux(self) -> bool:
        """Check if boundary conditions are no-flux."""
        return self.type == "no_flux"

    def is_robin(self) -> bool:
        """Check if boundary conditions are Robin."""
        return self.type == "robin"

    def get_matrix_size(self, num_interior_points: int) -> int:
        """
        Get the size of the system matrix for these boundary conditions.

        Args:
            num_interior_points: Number of interior grid points (M)

        Returns:
            Size of the system matrix
        """
        if self.type == "periodic":
            return num_interior_points
        elif self.type == "dirichlet":
            return num_interior_points - 1
        elif self.type in ["neumann", "robin"]:
            return num_interior_points + 1
        elif self.type == "no_flux":
            return num_interior_points
        else:
            raise ValueError(f"Unknown boundary condition type: {self.type}")

    def validate_values(self):
        """Validate that required values are provided for the boundary condition type."""
        if self.type == "dirichlet":
            if self.left_value is None or self.right_value is None:
                raise ValueError("Dirichlet boundary conditions require left_value and right_value")

        elif self.type == "neumann":
            if self.left_value is None or self.right_value is None:
                raise ValueError("Neumann boundary conditions require left_value and right_value")

        elif self.type == "robin":
            required_params = [
                self.left_alpha,
                self.left_beta,
                self.left_value,
                self.right_alpha,
                self.right_beta,
                self.right_value,
            ]
            if any(param is None for param in required_params):
                raise ValueError(
                    "Robin boundary conditions require left_alpha, left_beta, left_value, "
                    "right_alpha, right_beta, and right_value"
                )

    def __str__(self) -> str:
        """String representation of boundary conditions."""
        if self.type == "periodic":
            return "Periodic"
        elif self.type == "dirichlet":
            return f"Dirichlet(left={self.left_value}, right={self.right_value})"
        elif self.type == "neumann":
            return f"Neumann(left={self.left_value}, right={self.right_value})"
        elif self.type == "no_flux":
            return "No-flux"
        elif self.type == "robin":
            return (
                f"Robin(left: {self.left_alpha}u + {self.left_beta}u' = {self.left_value}, "
                f"right: {self.right_alpha}u + {self.right_beta}u' = {self.right_value})"
            )
        else:
            return f"Unknown({self.type})"


# Convenience functions for common boundary condition types
def periodic_bc() -> BoundaryConditions:
    """Create periodic boundary conditions.

    .. deprecated:: 0.14.0
        Use ``from mfg_pde.geometry import periodic_bc; bc = periodic_bc(dimension=1)``
    """
    _emit_deprecation_warning("periodic_bc")
    return BoundaryConditions(type="periodic")


def dirichlet_bc(left_value: float, right_value: float) -> BoundaryConditions:
    """Create Dirichlet boundary conditions.

    .. deprecated:: 0.14.0
        Use ``from mfg_pde.geometry import dirichlet_bc; bc = dirichlet_bc(value=..., dimension=1)``
    """
    _emit_deprecation_warning("dirichlet_bc")
    return BoundaryConditions(type="dirichlet", left_value=left_value, right_value=right_value)


def neumann_bc(left_gradient: float, right_gradient: float) -> BoundaryConditions:
    """Create Neumann boundary conditions.

    .. deprecated:: 0.14.0
        Use ``from mfg_pde.geometry import neumann_bc; bc = neumann_bc(value=..., dimension=1)``
    """
    _emit_deprecation_warning("neumann_bc")
    return BoundaryConditions(type="neumann", left_value=left_gradient, right_value=right_gradient)


def no_flux_bc() -> BoundaryConditions:
    """Create no-flux boundary conditions.

    .. deprecated:: 0.14.0
        Use ``from mfg_pde.geometry import no_flux_bc; bc = no_flux_bc(dimension=1)``
    """
    _emit_deprecation_warning("no_flux_bc")
    return BoundaryConditions(type="no_flux")


def robin_bc(
    left_alpha: float,
    left_beta: float,
    left_value: float,
    right_alpha: float,
    right_beta: float,
    right_value: float,
) -> BoundaryConditions:
    """Create Robin boundary conditions.

    .. deprecated:: 0.14.0
        Use ``from mfg_pde.geometry import robin_bc; bc = robin_bc(alpha=..., beta=..., dimension=1)``
    """
    _emit_deprecation_warning("robin_bc")
    return BoundaryConditions(
        type="robin",
        left_alpha=left_alpha,
        left_beta=left_beta,
        left_value=left_value,
        right_alpha=right_alpha,
        right_beta=right_beta,
        right_value=right_value,
    )
