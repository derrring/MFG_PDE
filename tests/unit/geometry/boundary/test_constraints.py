"""
Unit tests for variational inequality constraint implementations.

Tests:
- ConstraintProtocol compliance
- ObstacleConstraint (lower and upper)
- BilateralConstraint
- Projection properties (idempotence, non-expansiveness, feasibility)
- Active set detection
- Regional constraints

Created: 2026-01-17 (Issue #591 - Phase 2.1: Constraint Tests)
Part of: Issue #589 Phase 2 (Tier 2 BCs - Variational Constraints)
"""

import pytest

import numpy as np

from mfg_pde.geometry.boundary import (
    BilateralConstraint,
    ConstraintProtocol,
    ObstacleConstraint,
)


class TestConstraintProtocol:
    """Test ConstraintProtocol compliance."""

    def test_obstacle_constraint_is_protocol(self):
        """ObstacleConstraint should implement ConstraintProtocol."""
        psi = np.zeros(10)
        constraint = ObstacleConstraint(psi, constraint_type="lower")
        assert isinstance(constraint, ConstraintProtocol)

    def test_bilateral_constraint_is_protocol(self):
        """BilateralConstraint should implement ConstraintProtocol."""
        lower = -np.ones(10)
        upper = np.ones(10)
        constraint = BilateralConstraint(lower, upper)
        assert isinstance(constraint, ConstraintProtocol)


class TestObstacleConstraintLower:
    """Test ObstacleConstraint with lower bound (u ≥ ψ)."""

    def test_projection_1d(self):
        """Test projection onto lower obstacle in 1D."""
        psi = np.array([0.5, 1.0, 0.0])
        constraint = ObstacleConstraint(psi, constraint_type="lower")

        u = np.array([1.0, 0.5, -0.5])
        u_proj = constraint.project(u)

        expected = np.array([1.0, 1.0, 0.0])  # max(u, psi)
        assert np.allclose(u_proj, expected)

    def test_projection_2d(self):
        """Test projection onto lower obstacle in 2D."""
        psi = np.zeros((5, 5))
        constraint = ObstacleConstraint(psi, constraint_type="lower")

        u = -0.5 * np.ones((5, 5))
        u_proj = constraint.project(u)

        expected = np.zeros((5, 5))  # max(u, 0) = 0
        assert np.allclose(u_proj, expected)

    def test_projection_idempotence(self):
        """Projection should be idempotent: P(P(u)) = P(u)."""
        psi = np.array([0.5, 1.0, 0.0])
        constraint = ObstacleConstraint(psi, constraint_type="lower")

        u = np.array([1.0, 0.5, -0.5])
        u_proj = constraint.project(u)
        u_proj_twice = constraint.project(u_proj)

        assert np.allclose(u_proj_twice, u_proj)

    def test_projection_non_expansive(self):
        """Projection should be non-expansive: ||P(u) - P(v)|| ≤ ||u - v||."""
        psi = np.array([0.5, 1.0, 0.0])
        constraint = ObstacleConstraint(psi, constraint_type="lower")

        u = np.array([1.0, 0.5, -0.5])
        v = np.array([0.8, 0.7, -0.3])

        u_proj = constraint.project(u)
        v_proj = constraint.project(v)

        dist_original = np.linalg.norm(u - v)
        dist_projected = np.linalg.norm(u_proj - v_proj)

        assert dist_projected <= dist_original + 1e-10  # Allow numerical error

    def test_projection_ensures_feasibility(self):
        """Projected field should always be feasible."""
        psi = np.array([0.5, 1.0, 0.0])
        constraint = ObstacleConstraint(psi, constraint_type="lower")

        u = np.array([1.0, 0.5, -0.5])
        u_proj = constraint.project(u)

        assert constraint.is_feasible(u_proj)

    def test_is_feasible(self):
        """Test feasibility checking for lower obstacle."""
        psi = np.array([0.5, 1.0, 0.0])
        constraint = ObstacleConstraint(psi, constraint_type="lower")

        u_feasible = np.array([1.0, 1.5, 0.5])
        u_infeasible = np.array([0.3, 0.5, -0.1])

        assert constraint.is_feasible(u_feasible)
        assert not constraint.is_feasible(u_infeasible)

    def test_get_active_set(self):
        """Test active set detection for lower obstacle."""
        psi = np.array([0.5, 1.0, 0.0])
        constraint = ObstacleConstraint(psi, constraint_type="lower")

        u = np.array([1.0, 1.0, 0.0])  # Last two at obstacle
        active = constraint.get_active_set(u, tol=1e-8)

        expected_active = np.array([False, True, True])
        assert np.array_equal(active, expected_active)

    def test_regional_constraint(self):
        """Test constraint with regional mask."""
        psi = np.zeros((5, 5))
        region = np.zeros((5, 5), dtype=bool)
        region[2, 2] = True  # Only constrain center point

        constraint = ObstacleConstraint(psi, constraint_type="lower", region=region)

        u = -np.ones((5, 5))  # All negative
        u_proj = constraint.project(u)

        # Only center should be projected to 0, rest unchanged
        assert u_proj[2, 2] == 0.0
        assert u_proj[0, 0] == -1.0
        assert u_proj[4, 4] == -1.0

    def test_invalid_constraint_type(self):
        """Test that invalid constraint_type raises ValueError."""
        psi = np.zeros(10)
        with pytest.raises(ValueError, match="constraint_type must be"):
            ObstacleConstraint(psi, constraint_type="invalid")

    def test_region_shape_mismatch(self):
        """Test that region shape mismatch raises ValueError."""
        psi = np.zeros((5, 5))
        region = np.zeros((3, 3), dtype=bool)  # Wrong shape
        with pytest.raises(ValueError, match=r"region shape.*doesn't match"):
            ObstacleConstraint(psi, constraint_type="lower", region=region)

    def test_field_shape_mismatch_project(self):
        """Test that field shape mismatch raises ValueError in project()."""
        psi = np.zeros((5, 5))
        constraint = ObstacleConstraint(psi, constraint_type="lower")

        u = np.zeros((3, 3))  # Wrong shape
        with pytest.raises(ValueError, match=r"Field shape.*doesn't match"):
            constraint.project(u)


class TestObstacleConstraintUpper:
    """Test ObstacleConstraint with upper bound (u ≤ ψ)."""

    def test_projection_capacity_limit(self):
        """Test projection onto upper obstacle (capacity constraint)."""
        m_max = np.array([0.5, 0.5, 0.5])
        constraint = ObstacleConstraint(m_max, constraint_type="upper")

        m = np.array([0.3, 0.7, 0.4])
        m_proj = constraint.project(m)

        expected = np.array([0.3, 0.5, 0.4])  # min(m, m_max)
        assert np.allclose(m_proj, expected)

    def test_projection_idempotence(self):
        """Projection should be idempotent for upper constraint."""
        m_max = np.array([0.5, 0.5, 0.5])
        constraint = ObstacleConstraint(m_max, constraint_type="upper")

        m = np.array([0.3, 0.7, 0.4])
        m_proj = constraint.project(m)
        m_proj_twice = constraint.project(m_proj)

        assert np.allclose(m_proj_twice, m_proj)

    def test_is_feasible(self):
        """Test feasibility checking for upper obstacle."""
        m_max = np.array([0.5, 0.5, 0.5])
        constraint = ObstacleConstraint(m_max, constraint_type="upper")

        m_feasible = np.array([0.3, 0.4, 0.5])
        m_infeasible = np.array([0.6, 0.7, 0.4])

        assert constraint.is_feasible(m_feasible)
        assert not constraint.is_feasible(m_infeasible)

    def test_get_active_set(self):
        """Test active set detection for upper obstacle."""
        m_max = np.array([0.5, 0.5, 0.5])
        constraint = ObstacleConstraint(m_max, constraint_type="upper")

        m = np.array([0.3, 0.5, 0.5])  # Last two at capacity
        active = constraint.get_active_set(m, tol=1e-8)

        expected_active = np.array([False, True, True])
        assert np.array_equal(active, expected_active)


class TestBilateralConstraint:
    """Test BilateralConstraint (box constraint: ψ_lower ≤ u ≤ ψ_upper)."""

    def test_projection_1d(self):
        """Test projection onto box constraint in 1D."""
        lower = np.array([-1.0, -1.0, -1.0])
        upper = np.array([1.0, 1.0, 1.0])
        constraint = BilateralConstraint(lower, upper)

        u = np.array([0.5, -1.5, 2.0])
        u_proj = constraint.project(u)

        expected = np.array([0.5, -1.0, 1.0])  # clip to [-1, 1]
        assert np.allclose(u_proj, expected)

    def test_projection_2d(self):
        """Test projection onto box constraint in 2D."""
        lower = -np.ones((5, 5))
        upper = np.ones((5, 5))
        constraint = BilateralConstraint(lower, upper)

        u = 2.0 * np.ones((5, 5))
        u_proj = constraint.project(u)

        expected = np.ones((5, 5))  # clip to upper bound
        assert np.allclose(u_proj, expected)

    def test_projection_idempotence(self):
        """Projection should be idempotent: P(P(u)) = P(u)."""
        lower = np.array([-1.0, -1.0, -1.0])
        upper = np.array([1.0, 1.0, 1.0])
        constraint = BilateralConstraint(lower, upper)

        u = np.array([0.5, -1.5, 2.0])
        u_proj = constraint.project(u)
        u_proj_twice = constraint.project(u_proj)

        assert np.allclose(u_proj_twice, u_proj)

    def test_projection_non_expansive(self):
        """Projection should be non-expansive: ||P(u) - P(v)|| ≤ ||u - v||."""
        lower = np.array([-1.0, -1.0, -1.0])
        upper = np.array([1.0, 1.0, 1.0])
        constraint = BilateralConstraint(lower, upper)

        u = np.array([0.5, -1.5, 2.0])
        v = np.array([0.3, -1.2, 1.8])

        u_proj = constraint.project(u)
        v_proj = constraint.project(v)

        dist_original = np.linalg.norm(u - v)
        dist_projected = np.linalg.norm(u_proj - v_proj)

        assert dist_projected <= dist_original + 1e-10

    def test_projection_ensures_feasibility(self):
        """Projected field should always be feasible."""
        lower = np.array([-1.0, -1.0, -1.0])
        upper = np.array([1.0, 1.0, 1.0])
        constraint = BilateralConstraint(lower, upper)

        u = np.array([0.5, -1.5, 2.0])
        u_proj = constraint.project(u)

        assert constraint.is_feasible(u_proj)

    def test_is_feasible(self):
        """Test feasibility checking for bilateral constraint."""
        lower = np.array([-1.0, -1.0, -1.0])
        upper = np.array([1.0, 1.0, 1.0])
        constraint = BilateralConstraint(lower, upper)

        u_feasible = np.array([0.5, -0.5, 1.0])
        u_infeasible = np.array([0.5, -1.5, 2.0])

        assert constraint.is_feasible(u_feasible)
        assert not constraint.is_feasible(u_infeasible)

    def test_get_active_set(self):
        """Test active set detection for bilateral constraint."""
        lower = np.array([-1.0, -1.0, -1.0])
        upper = np.array([1.0, 1.0, 1.0])
        constraint = BilateralConstraint(lower, upper)

        u = np.array([0.5, -1.0, 1.0])  # Last two at bounds
        active = constraint.get_active_set(u, tol=1e-8)

        expected_active = np.array([False, True, True])
        assert np.array_equal(active, expected_active)

    def test_state_dependent_bounds(self):
        """Test bilateral constraint with spatially-varying bounds."""
        x = np.linspace(0, 1, 10)
        lower = -0.5 * np.ones(10)
        upper = 0.5 + 0.3 * x  # Upper bound varies with position

        constraint = BilateralConstraint(lower, upper)

        u = np.ones(10)  # Constant field
        u_proj = constraint.project(u)

        # At x=0: upper=0.5, so u_proj=0.5
        # At x=1: upper=0.8, so u_proj=0.8
        assert u_proj[0] <= 0.55  # Allow tolerance
        assert u_proj[-1] <= 0.85

    def test_invalid_bounds(self):
        """Test that lower > upper raises ValueError."""
        lower = np.array([1.0, 0.0, 0.0])
        upper = np.array([0.0, 1.0, 1.0])  # lower[0] > upper[0]

        with pytest.raises(ValueError, match="lower_bound must be ≤ upper_bound"):
            BilateralConstraint(lower, upper)

    def test_bounds_shape_mismatch(self):
        """Test that bounds shape mismatch raises ValueError."""
        lower = np.zeros((5, 5))
        upper = np.ones((3, 3))  # Wrong shape

        with pytest.raises(ValueError, match=r"lower_bound shape.*doesn't match"):
            BilateralConstraint(lower, upper)

    def test_regional_bilateral(self):
        """Test bilateral constraint with regional mask."""
        lower = -np.ones((5, 5))
        upper = np.ones((5, 5))
        region = np.zeros((5, 5), dtype=bool)
        region[2, 2] = True  # Only constrain center

        constraint = BilateralConstraint(lower, upper, region=region)

        u = 2.0 * np.ones((5, 5))  # All exceed upper bound
        u_proj = constraint.project(u)

        # Only center should be clipped to 1.0, rest unchanged
        assert u_proj[2, 2] == 1.0
        assert u_proj[0, 0] == 2.0
        assert u_proj[4, 4] == 2.0


class TestConstraintMathematicalProperties:
    """Test mathematical properties of constraint projections."""

    def test_fixed_point_property(self):
        """If u ∈ K, then P_K(u) = u (fixed point property)."""
        # Lower obstacle
        psi = np.array([0.5, 1.0, 0.0])
        constraint = ObstacleConstraint(psi, constraint_type="lower")

        u_in_K = np.array([1.0, 1.5, 0.5])  # All u >= psi
        u_proj = constraint.project(u_in_K)

        assert np.allclose(u_proj, u_in_K)

    def test_distance_to_constraint_set(self):
        """Projection minimizes distance to constraint set."""
        psi = np.zeros(10)
        constraint = ObstacleConstraint(psi, constraint_type="lower")

        u = -np.ones(10)  # Outside K
        u_proj = constraint.project(u)

        # Any other feasible point should be further from u
        v_feasible = 0.5 * np.ones(10)  # v ∈ K

        dist_proj = np.linalg.norm(u - u_proj)
        dist_other = np.linalg.norm(u - v_feasible)

        assert dist_proj <= dist_other

    def test_active_set_complementarity(self):
        """Active set should satisfy complementarity condition."""
        psi = np.array([0.5, 1.0, 0.0])
        constraint = ObstacleConstraint(psi, constraint_type="lower")

        u = np.array([1.0, 1.0, 0.0])
        active = constraint.get_active_set(u, tol=1e-8)

        # Where active, u should equal psi
        assert np.allclose(u[active], psi[active])

        # Where inactive, u should be strictly > psi
        assert np.all(u[~active] > psi[~active] - 1e-8)


class TestConstraintRepr:
    """Test string representations."""

    def test_obstacle_repr(self):
        """Test ObstacleConstraint __repr__."""
        psi = np.zeros((5, 5))
        constraint = ObstacleConstraint(psi, constraint_type="lower")
        repr_str = repr(constraint)

        assert "ObstacleConstraint" in repr_str
        assert "lower" in repr_str
        assert "(5, 5)" in repr_str
        assert "everywhere" in repr_str

    def test_obstacle_repr_regional(self):
        """Test ObstacleConstraint __repr__ with regional mask."""
        psi = np.zeros((5, 5))
        region = np.zeros((5, 5), dtype=bool)
        region[:3, :3] = True  # 9 points constrained

        constraint = ObstacleConstraint(psi, constraint_type="lower", region=region)
        repr_str = repr(constraint)

        assert "regional" in repr_str
        assert "9 points" in repr_str

    def test_bilateral_repr(self):
        """Test BilateralConstraint __repr__."""
        lower = -np.ones((5, 5))
        upper = np.ones((5, 5))
        constraint = BilateralConstraint(lower, upper)
        repr_str = repr(constraint)

        assert "BilateralConstraint" in repr_str
        assert "(5, 5)" in repr_str
        assert "everywhere" in repr_str
