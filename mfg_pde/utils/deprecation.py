"""
Deprecation utilities for MFG_PDE.

Provides decorators and helpers for managing deprecated APIs with strict
enforcement of the deprecation lifecycle policy.

Created: 2026-01-20 (Issue #616)
Reference: docs/development/DEPRECATION_LIFECYCLE_POLICY.md
"""

from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def deprecated(
    since: str,
    replacement: str,
    reason: str = "",
    removal_blockers: list[str] | None = None,
) -> Callable[[F], F]:
    """
    Mark a function, class, or method as deprecated.

    This decorator:
    1. Stores deprecation metadata on the object
    2. Issues DeprecationWarning when called
    3. Can be discovered by AST-based enforcement tools

    Removal happens when ALL conditions are met:
    - Deprecated for ≥3 versions OR ≥6 months (whichever is longer)
    - All removal blockers resolved
    - No internal production usage
    - Equivalence test exists

    Args:
        since: Version when deprecation started (e.g., "v0.17.0")
        replacement: New API to use instead (e.g., "use advection_scheme parameter")
        reason: Optional explanation of why deprecated
        removal_blockers: Conditions that prevent removal (default: standard checklist)
            - "internal_usage": Production code still calls this
            - "equivalence_test": No test verifying old = new
            - "migration_docs": No migration guide in DEPRECATION_MODERNIZATION_GUIDE.md

    Example:
        >>> @deprecated(
        ...     since="v0.17.0",
        ...     replacement="use create_distribution_monitor()",
        ...     reason="Renamed for clarity",
        ...     removal_blockers=["internal_usage", "equivalence_test"],
        ... )
        ... def create_default_monitor():
        ...     return create_distribution_monitor()

    Note:
        The decorated function MUST redirect to the new API internally.
        See DEPRECATION_LIFECYCLE_POLICY.md for full requirements.
    """
    # Default removal blockers if not specified
    if removal_blockers is None:
        removal_blockers = ["internal_usage", "equivalence_test", "migration_docs"]

    def decorator(func: F) -> F:
        # Store metadata on the function object (discoverable by AST tools)
        func._deprecation_meta = {  # type: ignore[attr-defined]
            "since": since,
            "replacement": replacement,
            "reason": reason,
            "removal_blockers": removal_blockers,
            "symbol": func.__name__,
        }

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Issue deprecation warning
            msg = f"{func.__name__} is deprecated since {since}. {replacement}"
            if reason:
                msg += f" Reason: {reason}"

            warnings.warn(msg, DeprecationWarning, stacklevel=2)

            # Call the actual function (which MUST redirect to new API)
            return func(*args, **kwargs)

        # Preserve metadata on wrapper
        wrapper._deprecation_meta = func._deprecation_meta  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    return decorator


def deprecated_parameter(
    param_name: str,
    since: str,
    replacement: str,
    removal_blockers: list[str] | None = None,
) -> Callable[[F], F]:
    """
    Mark a function parameter as deprecated.

    Use this when deprecating a parameter while keeping the function.
    The decorator ACTIVELY checks for parameter usage and issues warnings.

    IMPORTANT: Unlike the passive approach, this decorator wraps the function
    and uses inspect.signature to catch deprecated parameter usage automatically.
    No need to manually add warnings.warn() in function body.

    Args:
        param_name: Name of deprecated parameter
        since: Version when deprecation started
        replacement: New parameter to use
        removal_blockers: Conditions preventing removal (default: standard checklist)

    Example:
        >>> @deprecated_parameter(
        ...     param_name="conservative",
        ...     since="v0.17.0",
        ...     replacement="advection_scheme",
        ...     removal_blockers=["internal_usage", "equivalence_test"],
        ... )
        ... def FPFDMSolver(advection_scheme="divergence_upwind", conservative=None):
        ...     # Decorator automatically warns if conservative is passed
        ...     if conservative is not None:
        ...         advection_scheme = "divergence_upwind" if conservative else "gradient_upwind"
        ...     # ... continue with advection_scheme

    Note:
        The function body MUST still handle the deprecated parameter and redirect
        to the new parameter. The decorator only issues the warning automatically.
    """
    if removal_blockers is None:
        removal_blockers = ["internal_usage", "equivalence_test", "migration_docs"]

    def decorator(func: F) -> F:
        # Store parameter deprecation metadata
        if not hasattr(func, "_deprecated_parameters"):
            func._deprecated_parameters = []  # type: ignore[attr-defined]

        func._deprecated_parameters.append(  # type: ignore[attr-defined]
            {
                "param": param_name,
                "since": since,
                "replacement": replacement,
                "removal_blockers": removal_blockers,
            }
        )

        # Get function signature for parameter inspection
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Bind arguments to see which parameters are actually set
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                # Check if the deprecated parameter was explicitly passed
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]

                    # Only warn if user explicitly set it (not None default)
                    # This handles both positional and keyword argument cases
                    if value is not None:
                        warnings.warn(
                            f"Parameter '{param_name}' in '{func.__name__}' is deprecated "
                            f"since {since}. Use '{replacement}' instead.",
                            DeprecationWarning,
                            stacklevel=2,
                        )
            except TypeError:
                # If binding fails, fall through to original function
                # (Let the original function handle signature errors)
                pass

            # Call original function
            return func(*args, **kwargs)

        # Preserve metadata on wrapper
        wrapper._deprecated_parameters = func._deprecated_parameters  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    return decorator


def get_deprecation_metadata(obj: Any) -> dict[str, Any] | None:
    """
    Get deprecation metadata from a decorated object.

    Args:
        obj: Function, class, or method to inspect

    Returns:
        Deprecation metadata dict, or None if not deprecated

    Example:
        >>> meta = get_deprecation_metadata(create_default_monitor)
        >>> print(meta["since"])
        v0.17.0
    """
    return getattr(obj, "_deprecation_meta", None)


def get_deprecated_parameters(func: Callable[..., Any]) -> list[dict[str, str]]:
    """
    Get list of deprecated parameters for a function.

    Args:
        func: Function to inspect

    Returns:
        List of parameter metadata dicts

    Example:
        >>> params = get_deprecated_parameters(FPFDMSolver.__init__)
        >>> print(params[0]["param"])
        conservative
    """
    return getattr(func, "_deprecated_parameters", [])


def check_removal_readiness(
    obj: Any,
    current_version: str,
    completed_blockers: list[str] | None = None,
) -> dict[str, Any]:
    """
    Check if a deprecated object is ready for removal.

    Removal requires:
    1. Deprecated for ≥3 versions OR ≥6 months (whichever is longer)
    2. All removal blockers cleared
    3. No internal production usage

    Args:
        obj: Deprecated function, class, or method
        current_version: Current version string (e.g., "v0.20.0")
        completed_blockers: List of blockers that have been resolved

    Returns:
        Dict with keys:
            - "ready": bool - Whether removal is allowed
            - "blocking_reasons": list[str] - Why removal is blocked
            - "minimum_age_met": bool - Whether deprecation is old enough
            - "blockers_cleared": bool - Whether all conditions met

    Example:
        >>> meta = check_removal_readiness(
        ...     old_function,
        ...     current_version="v0.20.0",
        ...     completed_blockers=["internal_usage", "equivalence_test"],
        ... )
        >>> if meta["ready"]:
        ...     print("Can safely remove old_function")
    """
    meta = get_deprecation_metadata(obj)
    if meta is None:
        return {
            "ready": False,
            "blocking_reasons": ["Not a deprecated object"],
            "minimum_age_met": False,
            "blockers_cleared": False,
        }

    completed_blockers = completed_blockers or []
    blocking_reasons = []

    # Check removal blockers
    required_blockers = set(meta.get("removal_blockers", []))
    completed_set = set(completed_blockers)
    remaining_blockers = required_blockers - completed_set

    blockers_cleared = len(remaining_blockers) == 0
    if not blockers_cleared:
        blocking_reasons.extend([f"Blocker not cleared: {b}" for b in remaining_blockers])

    # Check minimum age (3 minor versions)
    # Parse version strings: "v0.17.0" → (0, 17, 0)
    since_version_str = meta["since"]
    try:
        # Import packaging for version parsing
        from packaging import version

        current_ver = version.parse(current_version.lstrip("v"))
        since_ver = version.parse(since_version_str.lstrip("v"))

        # Calculate minor version difference
        # e.g., v0.17.0 → v0.20.0 is 3 minor versions
        minor_diff = (current_ver.major - since_ver.major) * 100 + (current_ver.minor - since_ver.minor)

        minimum_age_met = minor_diff >= 3

        if not minimum_age_met:
            blocking_reasons.append(
                f"Not deprecated long enough (since {since_version_str}, "
                f"current {current_version}, need 3 minor versions)"
            )
    except (ImportError, ValueError):
        # Fallback if packaging not available or version parse fails
        # Be conservative - assume age not met
        minimum_age_met = False
        blocking_reasons.append(
            f"Cannot verify age (since {since_version_str}, requires 'packaging' library for version parsing)"
        )

    ready = blockers_cleared and minimum_age_met

    return {
        "ready": ready,
        "blocking_reasons": blocking_reasons,
        "minimum_age_met": minimum_age_met,
        "blockers_cleared": blockers_cleared,
        "remaining_blockers": list(remaining_blockers),
    }


def deprecated_alias(
    old_name: str,
    new_class: type,
    since: str,
) -> Callable[..., Any]:
    """
    Create a deprecated alias for a renamed class or function.

    Use this when renaming a class/function but keeping the old name for
    backward compatibility. The alias will emit a deprecation warning
    when used.

    Args:
        old_name: The deprecated name (for warning message)
        new_class: The new class/function to redirect to
        since: Version when deprecation started (e.g., "v0.17.1")

    Returns:
        A wrapper function that creates instances of new_class with warning

    Example:
        >>> # In module where class was renamed:
        >>> class DriftField:  # New name
        ...     def __init__(self, U, cost, geometry):
        ...         ...
        ...
        >>> # Create deprecated alias
        >>> MFGDriftField = deprecated_alias("MFGDriftField", DriftField, "v0.17.1")
        >>>
        >>> # Usage (will warn):
        >>> drift = MFGDriftField(U, cost, grid)  # DeprecationWarning emitted

    Note:
        For simple renames, this creates a factory function that:
        1. Emits DeprecationWarning
        2. Returns new_class(*args, **kwargs)

        For classes, isinstance checks will work with the new class name only.
    """

    def alias_wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            f"{old_name} is deprecated since {since}. Use {new_class.__name__} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return new_class(*args, **kwargs)

    # Store metadata for discovery
    alias_wrapper._deprecation_meta = {  # type: ignore[attr-defined]
        "since": since,
        "replacement": new_class.__name__,
        "reason": f"Renamed to {new_class.__name__}",
        "removal_blockers": ["internal_usage", "equivalence_test"],
        "symbol": old_name,
        "alias_for": new_class.__name__,
    }

    # Preserve docstring
    alias_wrapper.__doc__ = f"""
    Deprecated alias for {new_class.__name__}.

    .. deprecated:: {since.lstrip("v")}
        Use `{new_class.__name__}` instead. Will be removed in v1.0.0.
    """
    alias_wrapper.__name__ = old_name

    return alias_wrapper


def validate_kwargs(
    kwargs: dict[str, Any],
    deprecated_kwargs: dict[str, str],
    recognized_kwargs: set[str],
    context: str = "function",
    error_on_deprecated: bool = True,
    warn_on_unrecognized: bool = True,
) -> None:
    """
    Validate kwargs against deprecated and recognized sets.

    Use this for functions/methods that accept **kwargs to prevent silent fail
    when users pass deprecated or unrecognized parameters.

    Args:
        kwargs: The kwargs dict to validate
        deprecated_kwargs: Mapping of deprecated kwarg names to their replacements
            Example: {"hamiltonian": "MFGComponents.hamiltonian_func"}
        recognized_kwargs: Set of valid kwargs that are actually consumed
            Example: {"m_initial", "u_final", "boundary_conditions"}
        context: Description for error messages (e.g., "MFGProblem", "create_solver")
        error_on_deprecated: If True, raise ValueError for deprecated kwargs.
            If False, emit DeprecationWarning.
        warn_on_unrecognized: If True, emit UserWarning for unrecognized kwargs.

    Raises:
        ValueError: If deprecated kwargs are passed and error_on_deprecated=True

    Example:
        >>> DEPRECATED = {"hamiltonian": "Use MFGComponents.hamiltonian_func"}
        >>> RECOGNIZED = {"m_initial", "u_final"}
        >>> validate_kwargs(
        ...     kwargs={"hamiltonian": my_func, "typo_param": 123},
        ...     deprecated_kwargs=DEPRECATED,
        ...     recognized_kwargs=RECOGNIZED,
        ...     context="MFGProblem",
        ... )
        ValueError: Deprecated kwargs in MFGProblem: 'hamiltonian' -> Use MFGComponents...

    Issue #666: Prevents silent fail where user-provided kwargs are ignored.
    """
    # Check for deprecated kwargs
    deprecated_found = []
    for kwarg_name, replacement in deprecated_kwargs.items():
        if kwarg_name in kwargs:
            deprecated_found.append((kwarg_name, replacement))

    if deprecated_found:
        msg_lines = [f"Deprecated kwargs detected in {context}:", ""]
        for name, replacement in deprecated_found:
            msg_lines.append(f"  - '{name}' -> {replacement}")

        msg = "\n".join(msg_lines)

        if error_on_deprecated:
            raise ValueError(msg)
        else:
            warnings.warn(msg, DeprecationWarning, stacklevel=3)

    # Check for unrecognized kwargs (possibly typos)
    if warn_on_unrecognized:
        all_known = set(deprecated_kwargs.keys()) | recognized_kwargs
        unrecognized = set(kwargs.keys()) - all_known
        if unrecognized:
            warnings.warn(
                f"Unrecognized kwargs in {context}: {sorted(unrecognized)}. "
                f"These may be silently ignored. Known kwargs: {sorted(recognized_kwargs)}",
                UserWarning,
                stacklevel=3,
            )


__all__ = [
    "deprecated",
    "deprecated_parameter",
    "deprecated_alias",
    "get_deprecation_metadata",
    "get_deprecated_parameters",
    "check_removal_readiness",
    "validate_kwargs",
]
