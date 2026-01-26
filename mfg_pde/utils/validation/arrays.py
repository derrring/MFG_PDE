"""
Validation for array inputs (dtype, shape, dimension).

This module validates numpy arrays for correct dtype, shape, and
dimension compatibility with geometry.

Issue #687: Array/Tensor validation (dtype, shape, dimension)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfg_pde.utils.validation.protocol import ValidationResult

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray


def validate_array_dtype(
    arr: NDArray,
    name: str,
    required_dtype: DTypeLike = np.float64,
    *,
    auto_convert: bool = False,
) -> ValidationResult:
    """
    Validate array has correct dtype.

    Args:
        arr: Array to validate
        name: Name for error messages
        required_dtype: Required dtype (default: float64)
        auto_convert: If True, add warning instead of error for convertible types

    Returns:
        ValidationResult

    Issue #687: Array dtype validation
    """
    result = ValidationResult()

    if arr.dtype == required_dtype:
        return result

    # Check if it's a compatible floating type
    if np.issubdtype(arr.dtype, np.floating):
        if auto_convert:
            result.add_warning(
                f"{name} has dtype {arr.dtype}, expected {required_dtype}. Auto-converting.",
                location=name,
            )
            result.context["needs_conversion"] = True
            result.context["original_dtype"] = arr.dtype
        else:
            result.add_warning(
                f"{name} has dtype {arr.dtype}, expected {required_dtype}. This may cause precision loss.",
                location=name,
                suggestion=f"Use {name}.astype({required_dtype})",
            )
    elif np.issubdtype(arr.dtype, np.integer):
        result.add_warning(
            f"{name} has integer dtype {arr.dtype}, expected {required_dtype}. Implicit conversion may occur.",
            location=name,
            suggestion=f"Use {name}.astype({required_dtype})",
        )
    else:
        result.add_error(
            f"{name} has incompatible dtype {arr.dtype}, expected {required_dtype}",
            location=name,
        )

    return result


def validate_array_shape(
    arr: NDArray,
    expected_shape: tuple[int, ...],
    name: str,
) -> ValidationResult:
    """
    Validate array has expected shape.

    Args:
        arr: Array to validate
        expected_shape: Expected shape tuple
        name: Name for error messages

    Returns:
        ValidationResult

    Issue #687: Array shape validation
    """
    result = ValidationResult()

    if arr.shape != expected_shape:
        result.add_error(
            f"{name} has shape {arr.shape}, expected {expected_shape}",
            location=name,
            suggestion=f"Reshape {name} to {expected_shape}",
        )

    return result


def validate_field_shape(
    field: NDArray,
    spatial_shape: tuple[int, ...],
    name: str,
    *,
    allow_temporal: bool = True,
    n_timesteps: int | None = None,
) -> ValidationResult:
    """
    Validate a field array (spatial or spatiotemporal).

    A field can be:
    - Spatial only: shape matches spatial_shape
    - Spatiotemporal: shape is (Nt, *spatial_shape)

    Args:
        field: Field array to validate
        spatial_shape: Expected spatial shape
        name: Name for error messages
        allow_temporal: Whether to allow spatiotemporal shape
        n_timesteps: Expected number of timesteps (if known)

    Returns:
        ValidationResult

    Issue #687: Field shape validation
    """
    result = ValidationResult()

    # Check spatial shape
    if field.shape == spatial_shape:
        result.context["is_temporal"] = False
        return result

    # Check spatiotemporal shape
    if allow_temporal and len(field.shape) == len(spatial_shape) + 1:
        if field.shape[1:] == spatial_shape:
            result.context["is_temporal"] = True
            result.context["n_timesteps"] = field.shape[0]

            # Verify timestep count if known
            if n_timesteps is not None and field.shape[0] != n_timesteps:
                result.add_error(
                    f"{name} has {field.shape[0]} timesteps, expected {n_timesteps}",
                    location=name,
                )
            return result

    # Shape doesn't match
    expected = f"{spatial_shape}"
    if allow_temporal:
        expected += f" or (Nt, {', '.join(map(str, spatial_shape))})"

    result.add_error(
        f"{name} has shape {field.shape}, expected {expected}",
        location=name,
    )

    return result


def validate_field_dimension(
    field: NDArray,
    expected_dimension: int,
    name: str,
) -> ValidationResult:
    """
    Validate field has correct spatial dimension.

    Args:
        field: Field array
        expected_dimension: Expected spatial dimension (1, 2, or 3)
        name: Name for error messages

    Returns:
        ValidationResult

    Issue #687: Dimension validation
    """
    result = ValidationResult()

    # Infer spatial dimension from array
    ndim = field.ndim

    # Could be spatial only or spatiotemporal
    # Assume first dimension is time if ndim > expected_dimension
    if ndim == expected_dimension:
        inferred_dim = ndim
    elif ndim == expected_dimension + 1:
        inferred_dim = ndim - 1  # Subtract time dimension
    else:
        result.add_error(
            f"{name} has {ndim} dimensions, expected {expected_dimension} "
            f"(spatial) or {expected_dimension + 1} (spatiotemporal)",
            location=name,
        )
        return result

    if inferred_dim != expected_dimension:
        result.add_error(
            f"{name} has spatial dimension {inferred_dim}, expected {expected_dimension}",
            location=name,
        )

    return result


def validate_finite(
    arr: NDArray,
    name: str,
) -> ValidationResult:
    """
    Validate array contains only finite values (no NaN or Inf).

    Args:
        arr: Array to validate
        name: Name for error messages

    Returns:
        ValidationResult with location of first non-finite value

    Issue #687: NaN/Inf detection
    """
    result = ValidationResult()

    if np.all(np.isfinite(arr)):
        return result

    # Find non-finite values
    nan_mask = np.isnan(arr)
    inf_mask = np.isinf(arr)

    n_nan = np.sum(nan_mask)
    n_inf = np.sum(inf_mask)

    # Get location of first issue
    if n_nan > 0:
        first_idx = np.unravel_index(np.argmax(nan_mask), arr.shape)
        result.add_error(
            f"{name} contains {n_nan} NaN values. First at index {first_idx}",
            location=name,
        )
    if n_inf > 0:
        first_idx = np.unravel_index(np.argmax(inf_mask), arr.shape)
        result.add_error(
            f"{name} contains {n_inf} Inf values. First at index {first_idx}",
            location=name,
        )

    result.context["n_nan"] = n_nan
    result.context["n_inf"] = n_inf

    return result


def validate_non_negative(
    arr: NDArray,
    name: str,
    *,
    tolerance: float = 0.0,
) -> ValidationResult:
    """
    Validate array values are non-negative.

    Args:
        arr: Array to validate
        name: Name for error messages
        tolerance: Allow small negative values up to this magnitude

    Returns:
        ValidationResult

    Useful for density validation (m >= 0).
    """
    result = ValidationResult()

    min_val = np.min(arr)
    if min_val < -tolerance:
        n_negative = np.sum(arr < -tolerance)
        result.add_error(
            f"{name} has {n_negative} negative values (min={min_val:.6e})",
            location=name,
            suggestion=f"Ensure {name} is non-negative",
        )
        result.context["min_value"] = min_val
        result.context["n_negative"] = n_negative

    return result
