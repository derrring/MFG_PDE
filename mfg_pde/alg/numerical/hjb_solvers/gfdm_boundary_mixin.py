"""
GFDM Boundary Handling Mixin for HJB Solvers.

This module provides boundary-related functionality for GFDM-based HJB solvers:
- Outward normal computation for boundary points
- Local Coordinate Rotation (LCR) for boundary stencil conditioning
- Ghost nodes method for structural Neumann BC enforcement
- Wind-dependent BC for viscosity solutions

These methods are extracted from HJBGFDMSolver to improve maintainability.

Issue #531: GFDM boundary stencil degeneracy fix.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class GFDMBoundaryMixin:
    """
    Mixin providing boundary handling methods for GFDM solvers.

    This mixin expects the following attributes from the host class:
    - collocation_points: np.ndarray of shape (N, dim)
    - dimension: int
    - domain_bounds: list of (low, high) tuples
    - boundary_indices: np.ndarray of boundary point indices
    - boundary_conditions: dict or BoundaryConditions object
    - neighborhoods: dict mapping point index to neighborhood info
    - _use_ghost_nodes: bool
    - _use_wind_dependent_bc: bool
    - problem: MFGProblem instance
    """

    # Type hints for attributes expected from host class
    collocation_points: np.ndarray
    dimension: int
    domain_bounds: list[tuple[float, float]]
    boundary_indices: np.ndarray
    boundary_conditions: Any
    neighborhoods: dict
    _use_ghost_nodes: bool
    _use_wind_dependent_bc: bool
    _boundary_normals: np.ndarray | None
    _gfdm_operator: Any

    def _compute_outward_normal(self, point_idx: int) -> np.ndarray:
        """
        Compute outward normal vector at a boundary point.

        For rectangular domains, the normal is determined by which boundary
        the point lies on. For corner points (on multiple boundaries), returns
        the average of all boundary normals.

        Args:
            point_idx: Index of boundary point

        Returns:
            Unit outward normal vector, shape (dimension,)
        """
        point = self.collocation_points[point_idx]
        normal = np.zeros(self.dimension)
        tol = 1e-10

        for d in range(self.dimension):
            low, high = self.domain_bounds[d]
            if abs(point[d] - low) < tol:
                normal[d] -= 1.0  # At lower boundary, normal points inward (negative)
            if abs(point[d] - high) < tol:
                normal[d] += 1.0  # At upper boundary, normal points outward (positive)

        # Normalize (handles corners where multiple boundaries meet)
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal /= norm
        else:
            # Fallback: no clear boundary detected, use zero vector
            normal = np.zeros(self.dimension)

        return normal

    def _compute_boundary_normals(self) -> np.ndarray | None:
        """
        Compute outward normal vectors for all boundary points.

        Tries to use geometry infrastructure (BoundaryConditions.get_outward_normal)
        first, falls back to axis-aligned computation for rectangular domains.

        Returns:
            Array of shape (n_boundary, dimension) with unit normal vectors,
            or None if no boundary points.
        """
        if len(self.boundary_indices) == 0:
            return None

        normals = np.zeros((len(self.boundary_indices), self.dimension))

        # Try to use geometry infrastructure first (supports SDF domains)
        bc_obj = None
        if hasattr(self, "problem") and hasattr(self.problem, "geometry") and self.problem.geometry is not None:
            geom = self.problem.geometry
            if hasattr(geom, "boundary_conditions"):
                bc_obj = geom.boundary_conditions

        for local_idx, global_idx in enumerate(self.boundary_indices):
            point = self.collocation_points[global_idx]

            # Try BoundaryConditions.get_outward_normal (works for SDF domains)
            if bc_obj is not None and hasattr(bc_obj, "get_outward_normal"):
                normal = bc_obj.get_outward_normal(point)
                if normal is not None:
                    normals[local_idx] = normal
                    continue

            # Fallback: axis-aligned computation for rectangular domains
            normals[local_idx] = self._compute_outward_normal(global_idx)

        return normals

    def _create_bc_config(self) -> dict:
        """
        Create unified BC configuration dict (single source of truth).

        This ensures consistent BC handling between:
        - _apply_boundary_conditions_to_sparse_system (Jacobian)
        - _apply_boundary_conditions_to_solution (solution)
        - DirectCollocationHandler.apply_to_residual (residual)

        Returns:
            BC configuration dict with keys: type, values, normals
        """
        # Resolve BC type from boundary_conditions (single source of truth)
        # BC validation deferred to solve time - allow None during initialization
        bc_type = self._get_boundary_condition_property("type")

        if bc_type is None:
            # Try to infer from BC object
            try:
                bc_type = self.boundary_conditions.default_bc.value.lower()
            except AttributeError:
                # No BC specified - return None (validation deferred to solve time)
                return None

        if isinstance(bc_type, str):
            bc_type = bc_type.lower()

        # Get BC values (for Dirichlet or non-zero Neumann)
        bc_value = self._get_boundary_condition_property("value")

        # Build values dict for per-point BC values
        if callable(bc_value):
            # Time-dependent or space-dependent BC - will be evaluated at runtime
            bc_values = bc_value
        else:
            # Constant value for all boundary points
            bc_values = bc_value

        return {
            "type": bc_type,
            "values": bc_values,
            "normals": self._boundary_normals,
        }

    def _build_neumann_bc_weights(self) -> dict[int, tuple[np.ndarray, np.ndarray, float]]:
        """
        Build GFDM weights for normal derivative at boundary points.

        For Neumann BC (du/dn = 0), we need weights such that:
            du/dn approx sum_j w_j u_j = 0

        Returns:
            Dictionary mapping boundary point index to (neighbor_indices, weights, center_weight)
        """
        bc_weights: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}

        for i in self.boundary_indices:
            weights_data = self._gfdm_operator.get_derivative_weights(i)
            if weights_data is None:
                continue

            neighbor_indices = weights_data["neighbor_indices"]
            grad_weights = weights_data["grad_weights"]  # shape: (d, n_neighbors)

            # Get outward normal at this point
            normal = self._compute_outward_normal(i)

            # Normal derivative weights: du/dn = n . grad_u = sum_d n_d (du/dx_d)
            # = sum_d n_d (sum_j w^d_j u_j) = sum_j (sum_d n_d w^d_j) u_j
            normal_weights = np.zeros(len(neighbor_indices))
            for k in range(len(neighbor_indices)):
                for d in range(self.dimension):
                    normal_weights[k] += normal[d] * grad_weights[d, k]

            # Center contribution: sum rule requires center weight = -sum_j neighbor_weights
            # But this is already handled in grad_weights via _build_differentiation_matrices
            # We need to extract the center weight separately
            center_weight = -np.sum(normal_weights[neighbor_indices >= 0])

            bc_weights[i] = (neighbor_indices, normal_weights, center_weight)

        return bc_weights

    def _count_deep_neighbors(self, point_idx: int, neighbor_points: np.ndarray, min_depth: float) -> int:
        """
        Count neighbors with sufficient depth in the inward normal direction.

        For boundary points, "deep" neighbors are those that lie sufficiently
        into the interior (in the direction opposite to the outward normal).
        This ensures the stencil can capture normal derivatives accurately.

        Args:
            point_idx: Index of the center point
            neighbor_points: Array of neighbor coordinates, shape (n_neighbors, dim)
            min_depth: Minimum required depth (projection onto inward normal)

        Returns:
            Number of neighbors with depth >= min_depth
        """
        center = self.collocation_points[point_idx]
        outward_normal = self._compute_outward_normal(point_idx)

        # Inward normal (direction into the domain)
        inward_normal = -outward_normal

        # Compute depth: projection of (neighbor - center) onto inward normal
        offsets = neighbor_points - center  # shape: (n_neighbors, dim)
        depths = offsets @ inward_normal  # shape: (n_neighbors,)

        return int(np.sum(depths >= min_depth))

    # =========================================================================
    # Local Coordinate Rotation (LCR) Methods
    # =========================================================================

    def _build_rotation_matrix(self, normal: np.ndarray) -> np.ndarray:
        """
        Build rotation matrix that aligns first axis with the given normal vector.

        For Local Coordinate Rotation (LCR) at boundary points, this rotation
        transforms neighbor offsets so that:
        - First axis (x') aligns with outward normal
        - Remaining axes (y', z', ...) are tangential to boundary

        This improves numerical conditioning for normal derivative computation.

        Args:
            normal: Unit outward normal vector, shape (dimension,)

        Returns:
            Rotation matrix R, shape (dimension, dimension).
            To transform: x_rotated = R @ x_original
        """
        dim = len(normal)

        if dim == 1:
            # 1D: trivial - just sign
            return np.array([[np.sign(normal[0]) if abs(normal[0]) > 1e-10 else 1.0]])

        elif dim == 2:
            # 2D: Rotation matrix that maps e_x to normal
            # R = [n_x, -n_y]
            #     [n_y,  n_x]
            # where n = (n_x, n_y) is the normal
            # Verification: R @ e_x = R @ [1,0]^T = [n_x, n_y]^T = n
            n_x, n_y = normal
            return np.array([[n_x, -n_y], [n_y, n_x]])

        elif dim == 3:
            # 3D: Use Rodrigues' rotation formula
            # Rotate from e_x = (1,0,0) to normal
            e_x = np.array([1.0, 0.0, 0.0])

            # Check if normal is already aligned with e_x
            dot = np.dot(e_x, normal)
            if abs(dot - 1.0) < 1e-10:
                return np.eye(3)
            if abs(dot + 1.0) < 1e-10:
                # Opposite direction: rotate 180 deg around y-axis
                return np.diag([1.0, -1.0, -1.0])

            # General case: Rodrigues' formula
            v = np.cross(e_x, normal)
            s = np.linalg.norm(v)
            c = dot

            v_skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + v_skew + v_skew @ v_skew * (1 - c) / (s * s)
            return R

        else:
            # Higher dimensions: Use Householder reflection
            # Maps e_1 to normal
            e_1 = np.zeros(dim)
            e_1[0] = 1.0

            if np.allclose(normal, e_1):
                return np.eye(dim)
            if np.allclose(normal, -e_1):
                R = np.eye(dim)
                R[0, 0] = -1.0
                return R

            # Householder: R = I - 2*v*v^T / (v^T*v) where v = e_1 - normal
            v = e_1 - normal
            v = v / np.linalg.norm(v)
            return np.eye(dim) - 2.0 * np.outer(v, v)

    def _apply_local_coordinate_rotation(self) -> None:
        """
        Apply Local Coordinate Rotation (LCR) to boundary point stencils.

        For each boundary point, rotates the neighbor offsets to align with
        the boundary normal before computing Taylor expansion weights.
        This improves accuracy for normal derivatives at boundaries.

        Must be called after _build_neighborhood_structure() and before
        derivative computations.

        Issue #531: GFDM boundary stencil degeneracy fix (part 2).
        """
        if len(self.boundary_indices) == 0:
            return

        # Store rotation matrices for boundary points
        self._boundary_rotations: dict[int, np.ndarray] = {}

        boundary_set = set(self.boundary_indices)

        for i in boundary_set:
            normal = self._compute_outward_normal(i)

            # Skip if normal is zero (shouldn't happen for proper boundary points)
            if np.linalg.norm(normal) < 1e-10:
                continue

            # Build rotation matrix
            R = self._build_rotation_matrix(normal)
            self._boundary_rotations[i] = R

            # Get neighborhood
            neighborhood = self.neighborhoods[i]
            neighbor_points = neighborhood["points"]
            center = self.collocation_points[i]

            # Rotate neighbor offsets
            offsets = neighbor_points - center
            rotated_offsets = (R @ offsets.T).T  # shape: (n_neighbors, dim)

            # Store rotated neighbor points (for Taylor matrix computation)
            # The rotated points are: center + rotated_offset
            # But since Taylor expansion only uses offsets, we can store
            # the rotated offsets directly or update the neighborhood
            neighborhood["rotated_offsets"] = rotated_offsets
            neighborhood["rotation_matrix"] = R

    def _rotate_derivatives_back(
        self, derivatives: dict[tuple[int, ...], float], R: np.ndarray
    ) -> dict[tuple[int, ...], float]:
        """
        Rotate derivatives from rotated frame back to original coordinates.

        For LCR boundary points, derivatives are computed in a rotated frame
        where the first axis aligns with the boundary normal. This method
        transforms them back to the original coordinate frame.

        Args:
            derivatives: Dictionary mapping multi-indices to derivative values
            R: Rotation matrix used for the forward transformation

        Returns:
            Rotated derivatives dictionary

        Mathematical transformation:
        - First derivatives (gradient): grad_orig = R^T @ grad_rotated
        - Second derivatives (Hessian): H_orig = R^T @ H_rotated @ R
        """
        if self.dimension != 2:
            # Only 2D transformation is implemented for now
            # Higher dimensions would need more complex Hessian transformation
            return derivatives

        rotated = dict(derivatives)
        R_T = R.T

        # Extract first derivatives (gradient)
        u_xp = derivatives.get((1, 0), 0.0)  # du/dx' (rotated frame)
        u_yp = derivatives.get((0, 1), 0.0)  # du/dy' (rotated frame)

        # Transform gradient: [u_x, u_y] = R^T @ [u_x', u_y']
        grad_rotated = np.array([u_xp, u_yp])
        grad_orig = R_T @ grad_rotated
        rotated[(1, 0)] = float(grad_orig[0])
        rotated[(0, 1)] = float(grad_orig[1])

        # Extract second derivatives (Hessian in rotated frame)
        u_xxp = derivatives.get((2, 0), 0.0)
        u_xyp = derivatives.get((1, 1), 0.0)
        u_yyp = derivatives.get((0, 2), 0.0)

        # Hessian in rotated frame
        H_rotated = np.array([[u_xxp, u_xyp], [u_xyp, u_yyp]])

        # Transform Hessian: H_orig = R^T @ H_rotated @ R
        H_orig = R_T @ H_rotated @ R
        rotated[(2, 0)] = float(H_orig[0, 0])
        rotated[(1, 1)] = float(H_orig[0, 1])  # = H_orig[1, 0] by symmetry
        rotated[(0, 2)] = float(H_orig[1, 1])

        return rotated

    # =========================================================================
    # Ghost Nodes Methods
    # =========================================================================

    def _create_ghost_neighbors(
        self, point_idx: int, neighbor_points: np.ndarray, neighbor_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create ghost neighbors for a boundary point to enforce Neumann BC structurally.

        For Neumann boundary conditions (du/dn = 0), ghost nodes provide a structural
        enforcement by creating mirror-image neighbors outside the domain. This avoids
        terminal cost/BC incompatibility issues.

        Method (Method of Images):
        1. Identify the boundary point's outward normal n
        2. For each interior neighbor, reflect it across the boundary plane
        3. Ghost point position: p_ghost = p_interior - 2*(p_interior - p_boundary).n * n
        4. During solution, enforce u(p_ghost) = u(p_interior_mirror)

        Args:
            point_idx: Index of boundary point
            neighbor_points: Array of neighbor coordinates (n_neighbors, dim)
            neighbor_indices: Array of neighbor point indices (n_neighbors,)

        Returns:
            Tuple of (ghost_points, ghost_indices, mirror_indices):
            - ghost_points: Coordinates of ghost neighbors (n_ghosts, dim)
            - ghost_indices: Indices for ghost points (negative to distinguish from real points)
            - mirror_indices: Indices of real points that mirror to each ghost

        Issue #531: Terminal BC compatibility via structural Neumann enforcement
        """
        center = self.collocation_points[point_idx]
        normal = self._compute_outward_normal(point_idx)

        # For each neighbor, check if it's inside the domain (positive normal component)
        # and create a corresponding ghost outside
        offsets = neighbor_points - center
        normal_components = offsets @ normal  # Projection onto normal

        # Select neighbors with positive normal component (interior side)
        # NOTE: For boundary points, normal points OUTWARD, so interior neighbors
        # have NEGATIVE projection (pointing inward, opposite to normal)
        interior_mask = normal_components < -1e-10  # Interior = opposite to outward normal
        interior_offsets = offsets[interior_mask]
        interior_indices = neighbor_indices[interior_mask]

        if len(interior_offsets) == 0:
            # No interior neighbors to mirror (shouldn't happen for valid boundary points)
            return np.zeros((0, self.dimension)), np.array([]), np.array([])

        # Create ghost points by reflecting across boundary plane
        # Reflection formula: offset_ghost = offset - 2*(offset.n)*n
        projections = interior_offsets @ normal  # Shape: (n_interior,)
        ghost_offsets = interior_offsets - 2 * projections[:, np.newaxis] * normal[np.newaxis, :]
        ghost_points = center + ghost_offsets

        # Assign negative indices to ghost points (distinguishes from real points)
        # Use -(point_idx * 10000 + i) to ensure uniqueness
        n_ghosts = len(ghost_points)
        ghost_indices = -(point_idx * 10000 + np.arange(n_ghosts))

        return ghost_points, ghost_indices, interior_indices

    def _apply_ghost_nodes_to_neighborhoods(self) -> None:
        """
        Apply ghost nodes method to boundary point neighborhoods.

        For each boundary point with Neumann BC, augments the neighborhood with
        ghost neighbors that enforce du/dn = 0 structurally through symmetry.

        Must be called after _build_neighborhood_structure() and before
        Taylor matrix construction.

        Issue #531: Terminal BC compatibility fix.
        """
        if len(self.boundary_indices) == 0:
            return

        # Storage for ghost node mappings
        self._ghost_node_map: dict[int, dict] = {}

        # Get global BC type - skip ghost nodes if BC not specified (validation deferred)
        bc_type_val = self._get_boundary_condition_property("type")

        # Determine BC type
        if bc_type_val is not None:
            bc_type = bc_type_val.lower() if isinstance(bc_type_val, str) else str(bc_type_val).lower()
        else:
            # Try to infer from BoundaryConditions object
            try:
                from mfg_pde.geometry.boundary import BCType

                default_bc = self.boundary_conditions.default_bc
                bc_type = default_bc.value.lower() if isinstance(default_bc, BCType) else str(default_bc).lower()
            except AttributeError:
                # No BC specified - skip ghost node creation (validation deferred to solve time)
                return

        # Ghost nodes only apply to Neumann BC
        if bc_type not in ("neumann", "no_flux"):
            return

        boundary_set = set(self.boundary_indices)

        for i in boundary_set:
            # Get existing neighborhood
            neighborhood = self.neighborhoods[i]
            neighbor_points = neighborhood["points"]
            neighbor_indices = neighborhood["indices"]

            # Create ghost neighbors
            ghost_points, ghost_indices, mirror_indices = self._create_ghost_neighbors(
                i, neighbor_points, neighbor_indices
            )

            if len(ghost_points) == 0:
                continue

            # Augment neighborhood with ghost points
            augmented_points = np.vstack([neighbor_points, ghost_points])
            augmented_indices = np.concatenate([neighbor_indices, ghost_indices])

            # Recompute distances for augmented neighborhood
            center = self.collocation_points[i]
            augmented_distances = np.linalg.norm(augmented_points - center, axis=1)

            # Update neighborhood
            neighborhood["points"] = augmented_points
            neighborhood["indices"] = augmented_indices
            neighborhood["distances"] = augmented_distances
            neighborhood["size"] = len(augmented_points)
            neighborhood["has_ghost"] = True  # Flag to trigger slow path with ghost handling
            neighborhood["ghost_count"] = len(ghost_points)

            # Store ghost node mapping for later symmetry enforcement
            # Build index map: ghost_idx -> mirror_idx
            ghost_to_mirror = {}
            for ghost_idx, mirror_idx in zip(ghost_indices, mirror_indices, strict=True):
                ghost_to_mirror[int(ghost_idx)] = int(mirror_idx)

            self._ghost_node_map[i] = {
                "ghost_indices": ghost_indices,
                "mirror_indices": mirror_indices,
                "ghost_to_mirror": ghost_to_mirror,
                "n_ghosts": len(ghost_points),
            }

    def _get_values_with_ghosts(
        self, u_values: np.ndarray, neighbor_indices: np.ndarray, point_idx: int | None = None
    ) -> np.ndarray:
        """
        Get u values for neighbors, mapping ghost indices to mirror values.

        Ghost points (negative indices) get their values from corresponding mirror points
        (positive indices) to enforce du/dn = 0 symmetrically.

        If wind-dependent BC is enabled, ghost mirroring only occurs when flow is INTO
        the boundary (grad_u.n > 0). When flow is OUT (grad_u.n < 0), extrapolates from interior.

        Args:
            u_values: Solution vector at all collocation points
            neighbor_indices: Array of neighbor indices (may include negative ghost indices)
            point_idx: Index of the point whose neighbors are being queried (for wind-dependent BC)

        Returns:
            Array of u values at neighbor locations (ghosts mapped to mirrors or extrapolated)
        """
        if not self._use_ghost_nodes or not hasattr(self, "_ghost_node_map"):
            # No ghost nodes, just return direct indexing
            return u_values[neighbor_indices]

        # Build combined mapping from all boundary points
        ghost_to_mirror_global: dict[int, int] = {}
        for ghost_info in self._ghost_node_map.values():
            ghost_to_mirror_global.update(ghost_info["ghost_to_mirror"])

        # Check if wind-dependent BC and point is boundary
        use_wind_check = self._use_wind_dependent_bc and point_idx is not None and point_idx in self._ghost_node_map

        # Compute flow direction if needed
        if use_wind_check:
            # Compute gradient at boundary point
            grad_u = self._compute_gradient_at_point(u_values, point_idx)
            # Get outward normal
            normal = self._compute_outward_normal(point_idx)
            # Characteristic direction: grad_u . n
            # For Hamilton-Jacobi, characteristics move as v = -grad_u
            # Characteristics cross boundary outward when v.n < 0, i.e., -grad_u.n < 0, i.e., grad_u.n > 0
            # We enforce BC only when characteristics come FROM outside (grad_u.n < 0)
            grad_dot_normal = np.dot(grad_u, normal)
            # Decide whether to enforce BC
            enforce_bc = grad_dot_normal < 0  # Only enforce if characteristics entering FROM outside
        else:
            enforce_bc = True  # Default: always enforce

        # Get values, mapping ghosts to mirrors (conditionally if wind-dependent)
        u_neighbors = np.zeros(len(neighbor_indices))
        for i, idx in enumerate(neighbor_indices):
            if idx < 0:
                # Ghost index
                if enforce_bc:
                    # Flow INTO boundary or standard ghost nodes: use mirror
                    mirror_idx = ghost_to_mirror_global.get(int(idx))
                    if mirror_idx is not None:
                        u_neighbors[i] = u_values[mirror_idx]
                    else:
                        # Shouldn't happen, but fallback to zero
                        u_neighbors[i] = 0.0
                else:
                    # Flow OUT OF boundary: LINEAR EXTRAPOLATION (transparent BC)
                    # u_ghost = 2*u_boundary - u_mirror - epsilon*(u_boundary - u_mirror)
                    # where epsilon is hyperviscosity parameter (0 = no damping, 1 = full reflection)
                    # This maintains the interior gradient through the boundary,
                    # effectively creating a one-sided stencil without restructuring
                    mirror_idx = ghost_to_mirror_global.get(int(idx))
                    if mirror_idx is not None and point_idx is not None:
                        u_boundary = u_values[point_idx]
                        u_mirror = u_values[mirror_idx]

                        # Apply hyperviscosity if enabled
                        epsilon = getattr(self, "_wind_bc_hyperviscosity", 0.0)
                        if epsilon > 0:
                            # Damped extrapolation: blend toward boundary value
                            u_neighbors[i] = (2.0 - epsilon) * u_boundary - (1.0 - epsilon) * u_mirror
                        else:
                            # Standard linear extrapolation
                            u_neighbors[i] = 2.0 * u_boundary - u_mirror
                    elif point_idx is not None:
                        # Fallback
                        u_neighbors[i] = u_values[point_idx]
                    else:
                        u_neighbors[i] = 0.0
            else:
                # Regular index
                u_neighbors[i] = u_values[int(idx)]

        return u_neighbors

    # =========================================================================
    # Helper Methods (referenced by boundary methods)
    # =========================================================================

    def _get_boundary_condition_property(self, property_name: str, default: Any = None) -> Any:
        """
        Get a property from boundary_conditions (dict or BoundaryConditions object).

        This is a helper method that should be defined in the host class.
        If not defined there, this fallback implementation handles basic cases.
        """
        # If host class has this method, it will be used instead
        if isinstance(self.boundary_conditions, dict):
            return self.boundary_conditions.get(property_name, default)
        elif hasattr(self.boundary_conditions, property_name):
            return getattr(self.boundary_conditions, property_name, default)
        return default

    def _compute_gradient_at_point(self, u_values: np.ndarray, point_idx: int) -> np.ndarray:
        """
        Compute gradient at a single point using GFDM weights.

        This method should be defined in the host class with full GFDM implementation.
        This is a placeholder that raises NotImplementedError if not overridden.
        """
        # This should be overridden by the host class
        raise NotImplementedError("_compute_gradient_at_point must be implemented in the host class")
