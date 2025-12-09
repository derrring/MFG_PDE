"""
Monotonicity statistics tracking for numerical schemes.

This module provides utilities for tracking M-matrix property satisfaction
in finite difference and meshfree discretizations.

For a monotone scheme, the discretization matrix must be an M-matrix:
- Diagonal elements: a_ii > 0 (positive diagonal)
- Off-diagonal elements: a_ij <= 0 for i != j (non-positive off-diagonal)
- Row sums: sum_j a_ij >= 0 (weak diagonal dominance)

These conditions ensure that the maximum principle holds for the discretized PDE.
"""

from __future__ import annotations


class MonotonicityStats:
    """
    Track M-matrix property satisfaction statistics across a solve.

    This class records point-by-point verification of monotonicity conditions
    and computes aggregate statistics.

    Attributes:
        total_points: Number of points verified
        monotone_points: Number of points satisfying M-matrix property
        violations_by_point: Mapping from point index to list of diagnostics
        worst_violations: List of worst violations ordered by severity

    Example:
        >>> stats = MonotonicityStats()
        >>> stats.record_point(0, True, {"is_monotone": True, "violation_severity": 0.0})
        >>> stats.record_point(1, False, {"is_monotone": False, "violation_severity": 0.1})
        >>> stats.get_success_rate()
        50.0
        >>> summary = stats.get_summary()
        >>> summary["num_violating_points"]
        1
    """

    def __init__(self):
        """Initialize empty statistics tracker."""
        self.total_points = 0
        self.monotone_points = 0
        self.violations_by_point: dict[int, list[dict]] = {}
        self.worst_violations: list[dict] = []

    def record_point(self, point_idx: int, is_monotone: bool, diagnostics: dict):
        """
        Record M-matrix verification result for a single point.

        Args:
            point_idx: Index of the collocation/grid point
            is_monotone: Whether M-matrix property is satisfied
            diagnostics: Dictionary with detailed diagnostic info including:
                - is_monotone: bool
                - center_ok: bool (diagonal condition)
                - neighbors_ok: bool (off-diagonal condition)
                - w_center: float (center weight)
                - min_neighbor_weight: float
                - max_neighbor_weight: float
                - num_violations: int
                - num_neighbors: int
                - violation_severity: float (magnitude of worst violation)
        """
        self.total_points += 1
        if is_monotone:
            self.monotone_points += 1
        else:
            if point_idx not in self.violations_by_point:
                self.violations_by_point[point_idx] = []
            self.violations_by_point[point_idx].append(diagnostics)
            severity = diagnostics.get("violation_severity", 0.0)
            self.worst_violations.append({"point_idx": point_idx, "severity": severity})

    def get_success_rate(self) -> float:
        """
        Compute percentage of points satisfying M-matrix property.

        Returns:
            Success rate as percentage (0-100). Returns 0.0 if no points recorded.
        """
        if self.total_points == 0:
            return 0.0
        return 100.0 * self.monotone_points / self.total_points

    def get_summary(self) -> dict:
        """
        Get summary statistics for M-matrix verification.

        Returns:
            Dictionary with keys:
                - success_rate: Percentage of monotone points (0-100)
                - monotone_points: Count of points satisfying M-matrix property
                - total_points: Total number of points checked
                - num_violating_points: Number of distinct points with violations
                - max_violation_severity: Maximum violation magnitude
        """
        success_rate = self.get_success_rate()
        num_violating_points = len(self.violations_by_point)
        max_violation = 0.0
        if self.worst_violations:
            max_violation = max(v["severity"] for v in self.worst_violations)

        return {
            "success_rate": success_rate,
            "monotone_points": self.monotone_points,
            "total_points": self.total_points,
            "num_violating_points": num_violating_points,
            "max_violation_severity": max_violation,
        }

    def reset(self):
        """Reset all statistics to initial state."""
        self.total_points = 0
        self.monotone_points = 0
        self.violations_by_point.clear()
        self.worst_violations.clear()


if __name__ == "__main__":
    """Smoke test for MonotonicityStats."""
    print("Testing MonotonicityStats...")

    stats = MonotonicityStats()

    # Record some test points
    stats.record_point(0, True, {"is_monotone": True, "violation_severity": 0.0})
    stats.record_point(1, True, {"is_monotone": True, "violation_severity": 0.0})
    stats.record_point(2, False, {"is_monotone": False, "violation_severity": 0.05})
    stats.record_point(3, True, {"is_monotone": True, "violation_severity": 0.0})
    stats.record_point(4, False, {"is_monotone": False, "violation_severity": 0.1})

    # Test success rate
    rate = stats.get_success_rate()
    assert rate == 60.0, f"Expected 60.0, got {rate}"
    print(f"  Success rate: {rate}%")

    # Test summary
    summary = stats.get_summary()
    assert summary["total_points"] == 5
    assert summary["monotone_points"] == 3
    assert summary["num_violating_points"] == 2
    assert summary["max_violation_severity"] == 0.1
    print(f"  Summary: {summary}")

    # Test reset
    stats.reset()
    assert stats.total_points == 0
    assert stats.get_success_rate() == 0.0
    print("  Reset works correctly")

    print("MonotonicityStats smoke tests passed!")
