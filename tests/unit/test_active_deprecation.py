"""
Test that @deprecated_parameter is ACTIVE (issues warnings automatically).

This is a critical test to verify that Issue #616 cannot recur.
The decorator must issue warnings without manual warnings.warn() in function body.

Created: 2026-01-20
"""

from __future__ import annotations

import pytest

from mfg_pde.utils.deprecation import deprecated_parameter


def test_deprecated_parameter_issues_warning_automatically():
    """
    CRITICAL TEST: Verify decorator issues warnings WITHOUT manual warnings.warn().

    This prevents Issue #616 recurrence where developers forget to add warnings.
    """

    @deprecated_parameter(
        param_name="old_param",
        since="v0.17.0",
        replacement="new_param",
    )
    def my_function(new_param: str = "default", old_param: str | None = None):
        # ⚠️ CRITICAL: NO manual warnings.warn() here!
        # Decorator must handle warnings automatically.
        if old_param is not None:
            new_param = old_param
        return new_param

    # Test 1: Using deprecated parameter SHOULD warn (automatic)
    with pytest.warns(DeprecationWarning, match="old_param.*deprecated"):
        result = my_function(old_param="value")

    assert result == "value"

    # Test 2: Using new parameter SHOULD NOT warn
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = my_function(new_param="value")

    # Verify no warnings issued
    assert len(w) == 0
    assert result == "value"


def test_deprecated_parameter_catches_positional_usage():
    """Verify decorator catches deprecated parameter passed positionally."""

    @deprecated_parameter(
        param_name="old_param",
        since="v0.17.0",
        replacement="new_param",
    )
    def my_function(new_param: str = "default", old_param: str | None = None):
        if old_param is not None:
            new_param = old_param
        return new_param

    # Pass deprecated parameter as keyword (should warn)
    with pytest.warns(DeprecationWarning, match="old_param.*deprecated"):
        my_function(old_param="value")

    # Pass as positional (should also warn if not None)
    # Note: Positional would be my_function("new_value", "old_value")
    with pytest.warns(DeprecationWarning, match="old_param.*deprecated"):
        my_function("new_value", "old_value")


def test_deprecated_parameter_none_default_no_warning():
    """Verify decorator doesn't warn when deprecated param uses None default."""
    import warnings

    @deprecated_parameter(
        param_name="old_param",
        since="v0.17.0",
        replacement="new_param",
    )
    def my_function(new_param: str = "default", old_param: str | None = None):
        if old_param is not None:
            new_param = old_param
        return new_param

    # Calling without specifying old_param (uses None default) - no warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = my_function()

    assert len(w) == 0
    assert result == "default"


def test_deprecated_parameter_with_conversion_logic():
    """Test real-world scenario like conservative → advection_scheme."""

    @deprecated_parameter(
        param_name="conservative",
        since="v0.17.0",
        replacement="advection_scheme",
    )
    def create_solver(advection_scheme: str = "divergence_upwind", conservative: bool | None = None):
        # Real-world redirection logic (Issue #616 scenario)
        if conservative is not None:
            advection_scheme = "divergence_upwind" if conservative else "gradient_upwind"
        return advection_scheme

    # Old API: conservative=True (should warn and redirect)
    with pytest.warns(DeprecationWarning, match="conservative.*deprecated"):
        result = create_solver(conservative=True)

    assert result == "divergence_upwind"

    # Old API: conservative=False (should warn and redirect)
    with pytest.warns(DeprecationWarning, match="conservative.*deprecated"):
        result = create_solver(conservative=False)

    assert result == "gradient_upwind"

    # New API: advection_scheme directly (no warning)
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = create_solver(advection_scheme="divergence_upwind")

    assert len(w) == 0
    assert result == "divergence_upwind"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
