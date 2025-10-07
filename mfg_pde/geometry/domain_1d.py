"""
1D Domain for MFG Problems.

This module provides the 1D domain specification for MFG problems.
Boundary conditions are now managed in boundary_conditions_1d.py.
"""

from __future__ import annotations

from .boundary_conditions_1d import BoundaryConditions  # noqa: TC001


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
        return f"Domain1D(xmin={self.xmin}, xmax={self.xmax}, length={self.length}, bc={self.boundary_conditions})"
