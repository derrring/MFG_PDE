"""
1D Domain and Boundary Conditions for MFG Problems.

This module provides boundary condition management for 1D MFG problems,
serving as the foundation for the geometry system and maintaining compatibility
with existing solvers.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass
class BoundaryConditions:
    """
    Boundary condition configuration for 1D MFG problems.

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
    left_value: Optional[float] = None
    right_value: Optional[float] = None

    # Robin boundary condition parameters: αu + βdu/dn = g
    # α coefficients (multiplier of solution value)
    left_alpha: Optional[float] = None  # coefficient of u at left boundary
    left_beta: Optional[float] = None  # coefficient of du/dn at left boundary
    right_alpha: Optional[float] = None  # coefficient of u at right boundary
    right_beta: Optional[float] = None  # coefficient of du/dn at right boundary

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


class Domain1D:
    """
    1D domain specification for MFG problems.

    This class encapsulates the spatial domain definition and boundary conditions
    for 1D MFG problems, providing a unified interface for domain management.
    """

    def __init__(self, xmin: float, xmax: float, boundary_conditions: BoundaryConditions):
        """
        Initialize 1D domain.

        Args:
            xmin: Left boundary of domain
            xmax: Right boundary of domain
            boundary_conditions: Boundary condition specification
        """
        if xmax <= xmin:
            raise ValueError("xmax must be greater than xmin")

        self.xmin = xmin
        self.xmax = xmax
        self.length = xmax - xmin
        self.boundary_conditions = boundary_conditions

        # Validate boundary conditions
        self.boundary_conditions.validate_values()

    def create_grid(self, num_points: int) -> tuple[float, list[float]]:
        """
        Create spatial grid for the domain.

        Args:
            num_points: Number of grid points (including boundaries)

        Returns:
            Tuple of (grid_spacing, grid_points)
        """
        if num_points < 2:
            raise ValueError("num_points must be at least 2")

        dx = self.length / (num_points - 1)
        x_points = [self.xmin + i * dx for i in range(num_points)]

        return dx, x_points

    def get_matrix_size(self, num_interior_points: int) -> int:
        """Get system matrix size for this domain's boundary conditions."""
        return self.boundary_conditions.get_matrix_size(num_interior_points)

    def __str__(self) -> str:
        """String representation of domain."""
        return f"Domain1D([{self.xmin}, {self.xmax}], {self.boundary_conditions})"

    def __repr__(self) -> str:
        """Detailed representation of domain."""
        return f"Domain1D(xmin={self.xmin}, xmax={self.xmax}, " f"length={self.length}, bc={self.boundary_conditions})"


# Convenience functions for common boundary condition types
def periodic_bc() -> BoundaryConditions:
    """Create periodic boundary conditions."""
    return BoundaryConditions(type="periodic")


def dirichlet_bc(left_value: float, right_value: float) -> BoundaryConditions:
    """Create Dirichlet boundary conditions."""
    return BoundaryConditions(type="dirichlet", left_value=left_value, right_value=right_value)


def neumann_bc(left_gradient: float, right_gradient: float) -> BoundaryConditions:
    """Create Neumann boundary conditions."""
    return BoundaryConditions(type="neumann", left_value=left_gradient, right_value=right_gradient)


def no_flux_bc() -> BoundaryConditions:
    """Create no-flux boundary conditions."""
    return BoundaryConditions(type="no_flux")


def robin_bc(
    left_alpha: float,
    left_beta: float,
    left_value: float,
    right_alpha: float,
    right_beta: float,
    right_value: float,
) -> BoundaryConditions:
    """Create Robin boundary conditions."""
    return BoundaryConditions(
        type="robin",
        left_alpha=left_alpha,
        left_beta=left_beta,
        left_value=left_value,
        right_alpha=right_alpha,
        right_beta=right_beta,
        right_value=right_value,
    )
