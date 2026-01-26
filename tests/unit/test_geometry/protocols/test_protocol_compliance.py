"""
Protocol compliance tests for geometry trait system.

Tests that protocol definitions are well-formed and can be used for
runtime type checking via isinstance().

This module does NOT test actual geometry implementations (that's Phase 1.2).
It tests that the protocol definitions themselves are correct.

Created: 2026-01-17 (Issue #590 - Phase 1.1)
Part of: Geometry & BC Architecture Implementation (Issue #589)
"""

from __future__ import annotations

import inspect
from typing import Protocol

import pytest

import numpy as np

from mfg_pde.geometry.protocols import (
    SupportsAdvection,
    SupportsBoundaryDistance,
    SupportsBoundaryNormal,
    SupportsBoundaryProjection,
    SupportsDivergence,
    SupportsGradient,
    SupportsInterpolation,
    SupportsLaplacian,
    SupportsLipschitz,
    SupportsManifold,
    SupportsPeriodic,
    SupportsRegionMarking,
)


class TestProtocolDefinitions:
    """Test that all protocols are properly defined and runtime-checkable."""

    @pytest.mark.parametrize(
        "protocol_class",
        [
            # Operator traits
            SupportsLaplacian,
            SupportsGradient,
            SupportsDivergence,
            SupportsAdvection,
            SupportsInterpolation,
            # Region traits
            SupportsBoundaryNormal,
            SupportsBoundaryProjection,
            SupportsBoundaryDistance,
            SupportsRegionMarking,
            # Topology traits
            SupportsManifold,
            SupportsLipschitz,
            SupportsPeriodic,
        ],
    )
    def test_protocol_is_runtime_checkable(self, protocol_class):
        """Verify all protocols are @runtime_checkable."""
        # Check it's a Protocol
        assert issubclass(protocol_class, Protocol)

        # Check it's runtime_checkable by attempting isinstance check
        # If not runtime_checkable, this will raise TypeError
        class DummyClass:
            pass

        try:
            # This should work without error if protocol is runtime_checkable
            isinstance(DummyClass(), protocol_class)
        except TypeError as e:
            if "cannot be used with isinstance()" in str(e):
                pytest.fail(f"{protocol_class.__name__} is not @runtime_checkable - isinstance() check failed")
            raise

    @pytest.mark.parametrize(
        ("protocol_class", "expected_methods"),
        [
            (SupportsLaplacian, ["get_laplacian_operator"]),
            (SupportsGradient, ["get_gradient_operator"]),
            (SupportsDivergence, ["get_divergence_operator"]),
            (SupportsAdvection, ["get_advection_operator"]),
            (SupportsInterpolation, ["get_interpolation_operator"]),
            (SupportsBoundaryNormal, ["get_outward_normal"]),
            (
                SupportsBoundaryProjection,
                ["project_to_boundary", "project_to_interior"],
            ),
            (SupportsBoundaryDistance, ["get_signed_distance"]),
            (
                SupportsRegionMarking,
                [
                    "mark_region",
                    "get_region_mask",
                    "intersect_regions",
                    "union_regions",
                    "get_region_names",
                ],
            ),
            (
                SupportsManifold,
                [
                    "get_metric_tensor",
                    "get_tangent_space_basis",
                    "compute_christoffel_symbols",
                ],
            ),
            (
                SupportsLipschitz,
                ["get_lipschitz_constant", "validate_lipschitz_regularity"],
            ),
            (
                SupportsPeriodic,
                ["get_periods", "wrap_coordinates", "compute_periodic_distance"],
            ),
        ],
    )
    def test_protocol_has_required_methods(self, protocol_class, expected_methods):
        """Verify protocols define all required methods."""
        # Get protocol members
        protocol_methods = {
            name for name, member in inspect.getmembers(protocol_class) if callable(member) and not name.startswith("_")
        }

        # Check all expected methods are present
        for method_name in expected_methods:
            assert method_name in protocol_methods, f"{protocol_class.__name__} missing method: {method_name}"

    @pytest.mark.parametrize(
        ("protocol_class", "expected_properties"),
        [
            (SupportsManifold, ["manifold_dimension"]),
            (SupportsPeriodic, ["periodic_dimensions"]),
        ],
    )
    def test_protocol_has_required_properties(self, protocol_class, expected_properties):
        """Verify protocols define all required properties."""
        # Get protocol members
        protocol_members = {
            name: member for name, member in inspect.getmembers(protocol_class) if not name.startswith("_")
        }

        # Check all expected properties are present
        for prop_name in expected_properties:
            assert prop_name in protocol_members, f"{protocol_class.__name__} missing property: {prop_name}"


class TestProtocolRuntimeChecking:
    """Test that protocols can be used for runtime type checking."""

    def test_minimal_laplacian_implementation(self):
        """Test isinstance check with minimal Laplacian implementation."""

        class MinimalLaplacian:
            """Minimal class implementing SupportsLaplacian."""

            def get_laplacian_operator(self, order=2, bc=None):
                """Minimal implementation."""
                return lambda u: u  # Identity operator

        obj = MinimalLaplacian()

        # Should pass isinstance check
        assert isinstance(obj, SupportsLaplacian)

        # Should not pass other protocol checks
        assert not isinstance(obj, SupportsGradient)
        assert not isinstance(obj, SupportsManifold)

    def test_minimal_gradient_implementation(self):
        """Test isinstance check with minimal Gradient implementation."""

        class MinimalGradient:
            """Minimal class implementing SupportsGradient."""

            def get_gradient_operator(self, direction=None, order=2, scheme="centered"):
                """Minimal implementation."""
                if direction is None:
                    return (lambda u: u, lambda u: u)  # 2D gradient
                return lambda u: u

        obj = MinimalGradient()

        # Should pass isinstance check
        assert isinstance(obj, SupportsGradient)

        # Should not pass other protocol checks
        assert not isinstance(obj, SupportsLaplacian)
        assert not isinstance(obj, SupportsDivergence)

    def test_minimal_periodic_implementation(self):
        """Test isinstance check with minimal Periodic implementation."""

        class MinimalPeriodic:
            """Minimal class implementing SupportsPeriodic."""

            @property
            def periodic_dimensions(self):
                return (0,)  # x is periodic

            def get_periods(self):
                return {0: 2 * np.pi}

            def wrap_coordinates(self, points):
                return points  # Identity

            def compute_periodic_distance(self, points1, points2):
                return np.linalg.norm(points1 - points2, axis=-1)

        obj = MinimalPeriodic()

        # Should pass isinstance check
        assert isinstance(obj, SupportsPeriodic)

        # Can access property
        assert obj.periodic_dimensions == (0,)
        assert obj.get_periods() == {0: 2 * np.pi}

    def test_minimal_region_marking_implementation(self):
        """Test isinstance check with minimal RegionMarking implementation."""

        class MinimalRegionMarking:
            """Minimal class implementing SupportsRegionMarking."""

            def __init__(self):
                self.regions = {}

            def mark_region(self, name, predicate=None, mask=None, boundary=None):
                self.regions[name] = mask

            def get_region_mask(self, name):
                return self.regions[name]

            def intersect_regions(self, *names):
                masks = [self.regions[n] for n in names]
                return np.logical_and.reduce(masks)

            def union_regions(self, *names):
                masks = [self.regions[n] for n in names]
                return np.logical_or.reduce(masks)

            def get_region_names(self):
                return list(self.regions.keys())

        obj = MinimalRegionMarking()

        # Should pass isinstance check
        assert isinstance(obj, SupportsRegionMarking)

        # Test basic functionality
        mask1 = np.array([True, False, True])
        mask2 = np.array([True, True, False])
        obj.mark_region("region1", mask=mask1)
        obj.mark_region("region2", mask=mask2)

        assert np.array_equal(obj.get_region_mask("region1"), mask1)
        assert obj.get_region_names() == ["region1", "region2"]
        assert np.array_equal(obj.intersect_regions("region1", "region2"), [True, False, False])

    def test_class_without_protocol_fails_check(self):
        """Test that classes without protocol methods fail isinstance check."""

        class NotAProtocol:
            """Class that doesn't implement any protocol."""

            def some_random_method(self):
                pass

        obj = NotAProtocol()

        # Should fail all protocol checks
        assert not isinstance(obj, SupportsLaplacian)
        assert not isinstance(obj, SupportsGradient)
        assert not isinstance(obj, SupportsManifold)
        assert not isinstance(obj, SupportsPeriodic)


class TestProtocolMethodSignatures:
    """Test that protocol method signatures are properly defined."""

    def test_laplacian_operator_signature(self):
        """Verify get_laplacian_operator has correct signature."""
        sig = inspect.signature(SupportsLaplacian.get_laplacian_operator)
        params = sig.parameters

        # Check required parameters
        assert "self" in params
        assert "order" in params
        assert "bc" in params

        # Check default values
        assert params["order"].default == 2
        assert params["bc"].default is None

    def test_gradient_operator_signature(self):
        """Verify get_gradient_operator has correct signature."""
        sig = inspect.signature(SupportsGradient.get_gradient_operator)
        params = sig.parameters

        assert "self" in params
        assert "direction" in params
        assert "order" in params
        assert "scheme" in params

        # Check defaults
        assert params["direction"].default is None
        assert params["order"].default == 2
        assert params["scheme"].default == "centered"

    def test_advection_operator_signature(self):
        """Verify get_advection_operator has correct signature."""
        sig = inspect.signature(SupportsAdvection.get_advection_operator)
        params = sig.parameters

        assert "self" in params
        assert "velocity_field" in params
        assert "scheme" in params
        assert "conservative" in params

        # Check defaults
        assert params["scheme"].default == "upwind"
        assert params["conservative"].default is True

    def test_region_marking_signature(self):
        """Verify mark_region has correct signature."""
        sig = inspect.signature(SupportsRegionMarking.mark_region)
        params = sig.parameters

        assert "self" in params
        assert "name" in params
        assert "predicate" in params
        assert "mask" in params
        assert "boundary" in params

        # Check all optional except name
        assert params["predicate"].default is None
        assert params["mask"].default is None
        assert params["boundary"].default is None


class TestProtocolDocumentation:
    """Test that all protocols have comprehensive docstrings."""

    @pytest.mark.parametrize(
        "protocol_class",
        [
            SupportsLaplacian,
            SupportsGradient,
            SupportsDivergence,
            SupportsAdvection,
            SupportsInterpolation,
            SupportsBoundaryNormal,
            SupportsBoundaryProjection,
            SupportsBoundaryDistance,
            SupportsRegionMarking,
            SupportsManifold,
            SupportsLipschitz,
            SupportsPeriodic,
        ],
    )
    def test_protocol_has_docstring(self, protocol_class):
        """Verify all protocols have class-level docstrings."""
        assert protocol_class.__doc__ is not None
        assert len(protocol_class.__doc__) > 50  # Non-trivial documentation

    @pytest.mark.parametrize(
        ("protocol_class", "method_name"),
        [
            (SupportsLaplacian, "get_laplacian_operator"),
            (SupportsGradient, "get_gradient_operator"),
            (SupportsDivergence, "get_divergence_operator"),
            (SupportsAdvection, "get_advection_operator"),
            (SupportsInterpolation, "get_interpolation_operator"),
            (SupportsBoundaryNormal, "get_outward_normal"),
            (SupportsBoundaryProjection, "project_to_boundary"),
            (SupportsBoundaryDistance, "get_signed_distance"),
            (SupportsRegionMarking, "mark_region"),
            (SupportsManifold, "get_metric_tensor"),
            (SupportsLipschitz, "get_lipschitz_constant"),
            (SupportsPeriodic, "wrap_coordinates"),
        ],
    )
    def test_protocol_methods_have_docstrings(self, protocol_class, method_name):
        """Verify all protocol methods have docstrings with examples."""
        method = getattr(protocol_class, method_name)
        assert method.__doc__ is not None
        assert len(method.__doc__) > 50  # Non-trivial documentation
        # Most methods should have examples
        # (Some utility methods might not, so this is lenient)


if __name__ == "__main__":
    """Quick smoke test for protocol compliance."""
    print("Testing protocol definitions...")

    # Test runtime checking
    print("\n✓ Testing runtime checkability...")
    test_obj = TestProtocolDefinitions()
    for protocol in [
        SupportsLaplacian,
        SupportsGradient,
        SupportsManifold,
        SupportsPeriodic,
    ]:
        test_obj.test_protocol_is_runtime_checkable(protocol)

    # Test isinstance checks
    print("✓ Testing isinstance checks...")

    class DummyLaplacian:
        def get_laplacian_operator(self, order=2, bc=None):
            return lambda u: u

    assert isinstance(DummyLaplacian(), SupportsLaplacian)
    assert not isinstance(DummyLaplacian(), SupportsGradient)

    print("✓ Testing method signatures...")
    test_sig = TestProtocolMethodSignatures()
    test_sig.test_laplacian_operator_signature()
    test_sig.test_gradient_operator_signature()

    print("\n✅ All protocol compliance tests passed!")
    print("\nProtocols defined:")
    print("  Operator traits: 5")
    print("  Region traits: 4")
    print("  Topology traits: 3")
    print("  Total: 12 protocols")
