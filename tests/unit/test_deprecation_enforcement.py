"""
Test deprecation enforcement mechanisms.

Tests the @deprecated decorator and AST-based checker to ensure
the deprecation lifecycle policy is enforced correctly.

Created: 2026-01-20 (Issue #616)
Reference: docs/development/DEPRECATION_LIFECYCLE_POLICY.md
"""

from __future__ import annotations

import warnings

import pytest

from mfg_pde.utils.deprecation import (
    check_removal_readiness,
    deprecated,
    deprecated_parameter,
    get_deprecated_parameters,
    get_deprecation_metadata,
)


def test_deprecated_decorator_issues_warning():
    """Verify @deprecated decorator issues DeprecationWarning."""

    @deprecated(
        since="v0.17.0",
        replacement="use new_function()",
    )
    def old_function():
        return "result"

    with pytest.warns(DeprecationWarning, match="old_function.*deprecated.*v0.17.0"):
        result = old_function()

    assert result == "result"


def test_deprecated_decorator_stores_metadata():
    """Verify @deprecated stores discoverable metadata."""

    @deprecated(
        since="v0.17.0",
        replacement="use new_function()",
        reason="Renamed for clarity",
        removal_blockers=["internal_usage", "equivalence_test"],
    )
    def old_function():
        return "result"

    meta = get_deprecation_metadata(old_function)

    assert meta is not None
    assert meta["since"] == "v0.17.0"
    assert meta["replacement"] == "use new_function()"
    assert meta["reason"] == "Renamed for clarity"
    assert meta["symbol"] == "old_function"
    assert "removal_blockers" in meta
    assert "internal_usage" in meta["removal_blockers"]
    assert "equivalence_test" in meta["removal_blockers"]


def test_deprecated_parameter_decorator():
    """Verify @deprecated_parameter marks parameters."""

    @deprecated_parameter(
        param_name="old_param",
        since="v0.17.0",
        replacement="new_param",
    )
    def my_function(new_param: str = "default", old_param: str | None = None):
        if old_param is not None:
            warnings.warn(
                "old_param is deprecated. Use new_param instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            new_param = old_param
        return new_param

    # Check metadata stored
    params = get_deprecated_parameters(my_function)
    assert len(params) == 1
    assert params[0]["param"] == "old_param"
    assert params[0]["since"] == "v0.17.0"
    assert params[0]["replacement"] == "new_param"

    # Check warning issued when using old parameter
    with pytest.warns(DeprecationWarning, match="old_param.*deprecated"):
        result = my_function(old_param="value")

    assert result == "value"

    # No warning when using new parameter
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        result = my_function(new_param="value")

    assert result == "value"


def test_deprecated_function_redirects_correctly():
    """Verify deprecated function redirects to new implementation."""

    def new_function(x: int) -> int:
        return x * 2

    @deprecated(
        since="v0.17.0",
        replacement="use new_function()",
    )
    def old_function(x: int) -> int:
        # Deprecated function MUST redirect to new function
        return new_function(x)

    # Verify both give same result
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress warning for this test
        result_old = old_function(5)

    result_new = new_function(5)

    assert result_old == result_new == 10


def test_get_deprecation_metadata_returns_none_for_non_deprecated():
    """Verify get_deprecation_metadata returns None for regular functions."""

    def regular_function():
        return "result"

    meta = get_deprecation_metadata(regular_function)
    assert meta is None


def test_get_deprecated_parameters_returns_empty_for_non_deprecated():
    """Verify get_deprecated_parameters returns empty list for regular functions."""

    def regular_function(param: str = "default"):
        return param

    params = get_deprecated_parameters(regular_function)
    assert params == []


def test_multiple_deprecated_parameters():
    """Verify function can have multiple deprecated parameters."""

    @deprecated_parameter(
        param_name="old_param1",
        since="v0.16.0",
        replacement="new_param1",
    )
    @deprecated_parameter(
        param_name="old_param2",
        since="v0.17.0",
        replacement="new_param2",
    )
    def my_function(
        new_param1: str = "default1",
        new_param2: str = "default2",
        old_param1: str | None = None,
        old_param2: str | None = None,
    ):
        # Redirection logic
        if old_param1 is not None:
            new_param1 = old_param1
        if old_param2 is not None:
            new_param2 = old_param2
        return new_param1, new_param2

    params = get_deprecated_parameters(my_function)
    assert len(params) == 2
    # Note: Decorators apply bottom-up, so old_param2 is added first
    param_names = {p["param"] for p in params}
    assert "old_param1" in param_names
    assert "old_param2" in param_names


def test_removal_readiness_with_blockers():
    """Verify check_removal_readiness correctly evaluates blockers."""

    @deprecated(
        since="v0.17.0",
        replacement="use new_function()",
        removal_blockers=["internal_usage", "equivalence_test", "migration_docs"],
    )
    def old_function():
        return "result"

    # No blockers cleared - not ready
    status = check_removal_readiness(old_function, "v0.20.0", completed_blockers=[])
    assert not status["ready"]
    assert not status["blockers_cleared"]
    assert len(status["remaining_blockers"]) == 3

    # Some blockers cleared - still not ready
    status = check_removal_readiness(old_function, "v0.20.0", completed_blockers=["internal_usage"])
    assert not status["ready"]
    assert not status["blockers_cleared"]
    assert len(status["remaining_blockers"]) == 2

    # All blockers cleared - ready
    status = check_removal_readiness(
        old_function,
        "v0.20.0",
        completed_blockers=["internal_usage", "equivalence_test", "migration_docs"],
    )
    assert status["ready"]
    assert status["blockers_cleared"]
    assert len(status["remaining_blockers"]) == 0


def test_removal_readiness_non_deprecated():
    """Verify check_removal_readiness handles non-deprecated objects."""

    def regular_function():
        return "result"

    status = check_removal_readiness(regular_function, "v0.20.0")
    assert not status["ready"]
    assert "Not a deprecated object" in status["blocking_reasons"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
