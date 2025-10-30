"""
Tests for Gradient Notation Utilities.

Tests backward compatibility functions for converting between legacy string-key
gradient notation and standard tuple multi-index notation.
"""

import warnings

import numpy as np
import pytest

from mfg_pde.compat.gradient_notation import (
    check_derivs_format,
    derivs_to_gradient_array,
    derivs_to_p_values_1d,
    ensure_tuple_notation,
    gradient_array_to_derivs,
    p_values_to_derivs_1d,
)


def test_derivs_to_p_values_1d():
    """Test conversion from tuple notation to legacy 1D string keys."""
    derivs = {(1,): 0.5}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        p_values = derivs_to_p_values_1d(derivs)

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "legacy string-key gradient notation" in str(w[0].message)

    assert p_values == {"forward": 0.5, "backward": 0.5}


def test_p_values_to_derivs_1d():
    """Test conversion from legacy 1D string keys to tuple notation."""
    p_values = {"forward": 0.6, "backward": 0.4}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        derivs = p_values_to_derivs_1d(p_values, u_value=1.0)

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)

    # Should use central difference approximation
    assert derivs[(0,)] == 1.0  # Function value
    assert derivs[(1,)] == 0.5  # (0.6 + 0.4) / 2


def test_derivs_to_gradient_array_1d():
    """Test extracting 1D gradient as array."""
    derivs = {(1,): 0.5}
    p = derivs_to_gradient_array(derivs, dimension=1)

    assert isinstance(p, np.ndarray)
    assert p.shape == (1,)
    assert p[0] == 0.5


def test_derivs_to_gradient_array_2d():
    """Test extracting 2D gradient as array."""
    derivs = {(1, 0): 0.5, (0, 1): 0.3}
    p = derivs_to_gradient_array(derivs, dimension=2)

    assert isinstance(p, np.ndarray)
    assert p.shape == (2,)
    assert np.allclose(p, [0.5, 0.3])


def test_derivs_to_gradient_array_3d():
    """Test extracting 3D gradient as array."""
    derivs = {(1, 0, 0): 0.5, (0, 1, 0): 0.3, (0, 0, 1): 0.2}
    p = derivs_to_gradient_array(derivs, dimension=3)

    assert isinstance(p, np.ndarray)
    assert p.shape == (3,)
    assert np.allclose(p, [0.5, 0.3, 0.2])


def test_gradient_array_to_derivs_1d():
    """Test converting 1D gradient array to tuple notation."""
    p = np.array([0.5])
    derivs = gradient_array_to_derivs(p, u_value=1.0)

    assert derivs[(0,)] == 1.0
    assert derivs[(1,)] == 0.5


def test_gradient_array_to_derivs_2d():
    """Test converting 2D gradient array to tuple notation."""
    p = np.array([0.5, 0.3])
    derivs = gradient_array_to_derivs(p, u_value=1.0)

    assert derivs[(0, 0)] == 1.0
    assert derivs[(1, 0)] == 0.5
    assert derivs[(0, 1)] == 0.3


def test_gradient_array_to_derivs_3d():
    """Test converting 3D gradient array to tuple notation."""
    p = np.array([0.5, 0.3, 0.2])
    derivs = gradient_array_to_derivs(p, u_value=1.0)

    assert derivs[(0, 0, 0)] == 1.0
    assert derivs[(1, 0, 0)] == 0.5
    assert derivs[(0, 1, 0)] == 0.3
    assert derivs[(0, 0, 1)] == 0.2


def test_check_derivs_format_tuple():
    """Test format detection for tuple notation."""
    derivs = {(1, 0): 0.5, (0, 1): 0.3}
    fmt = check_derivs_format(derivs)
    assert fmt == "tuple"


def test_check_derivs_format_string_1d():
    """Test format detection for 1D string keys."""
    derivs = {"forward": 0.5, "backward": 0.4}
    fmt = check_derivs_format(derivs)
    assert fmt == "string_1d"


def test_check_derivs_format_string_2d():
    """Test format detection for 2D string keys."""
    derivs = {"x": 0.5, "y": 0.3}
    fmt = check_derivs_format(derivs)
    assert fmt == "string_2d"


def test_check_derivs_format_empty():
    """Test format detection for empty dictionary."""
    derivs = {}
    fmt = check_derivs_format(derivs)
    assert fmt == "unknown"


def test_ensure_tuple_notation_already_tuple():
    """Test ensure_tuple_notation with already compliant input."""
    derivs = {(1, 0): 0.5, (0, 1): 0.3}
    result = ensure_tuple_notation(derivs)

    assert result is derivs  # Should return same object
    assert result == {(1, 0): 0.5, (0, 1): 0.3}


def test_ensure_tuple_notation_from_1d_strings():
    """Test ensure_tuple_notation converting from 1D string keys."""
    derivs = {"forward": 0.6, "backward": 0.4}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = ensure_tuple_notation(derivs, u_value=1.0)

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)

    assert result[(0,)] == 1.0
    assert result[(1,)] == 0.5  # Central difference


def test_ensure_tuple_notation_from_2d_strings_x_y():
    """Test ensure_tuple_notation converting from 2D string keys (x, y)."""
    derivs = {"x": 0.5, "y": 0.3}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = ensure_tuple_notation(derivs, dimension=2, u_value=1.0)

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)

    assert result[(0, 0)] == 1.0
    assert result[(1, 0)] == 0.5
    assert result[(0, 1)] == 0.3


def test_ensure_tuple_notation_from_2d_strings_dx_dy():
    """Test ensure_tuple_notation converting from 2D string keys (dx, dy)."""
    derivs = {"dx": 0.5, "dy": 0.3}

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = ensure_tuple_notation(derivs, dimension=2, u_value=1.0)

    assert result[(0, 0)] == 1.0
    assert result[(1, 0)] == 0.5
    assert result[(0, 1)] == 0.3


def test_ensure_tuple_notation_missing_dimension():
    """Test ensure_tuple_notation raises error when dimension missing."""
    derivs = {"x": 0.5, "y": 0.3}

    with pytest.raises(ValueError, match="dimension parameter required"):
        ensure_tuple_notation(derivs)  # Missing dimension parameter


def test_ensure_tuple_notation_unknown_format():
    """Test ensure_tuple_notation raises error for unknown format."""
    derivs = {"unknown_key": 0.5}

    with pytest.raises(ValueError, match="Unknown derivative format"):
        ensure_tuple_notation(derivs)


def test_roundtrip_1d():
    """Test roundtrip conversion for 1D."""
    # Start with tuple notation
    original = {(0,): 1.0, (1,): 0.5}

    # Convert to legacy
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_values = derivs_to_p_values_1d(original)

        # Convert back to tuple
        result = p_values_to_derivs_1d(p_values, u_value=1.0)

    assert result[(0,)] == 1.0
    assert result[(1,)] == 0.5


def test_roundtrip_2d():
    """Test roundtrip conversion for 2D."""
    # Start with gradient array
    p_original = np.array([0.5, 0.3])
    derivs = gradient_array_to_derivs(p_original, u_value=1.0)

    # Extract back to array
    p_result = derivs_to_gradient_array(derivs, dimension=2)

    assert np.allclose(p_result, p_original)


def test_missing_keys_use_defaults():
    """Test that missing keys return default 0.0."""
    derivs = {(1, 0): 0.5}  # Missing (0, 1)
    p = derivs_to_gradient_array(derivs, dimension=2)

    assert p[0] == 0.5
    assert p[1] == 0.0  # Default for missing key


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
