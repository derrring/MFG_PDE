#!/usr/bin/env python3
"""
Tests for HamiltonianAdapter

Validates automatic signature detection and conversion.
"""

import pytest

import numpy as np

from mfg_pde.utils import HamiltonianAdapter, adapt_hamiltonian, create_hamiltonian_adapter


class TestSignatureDetection:
    """Test automatic signature detection."""

    def test_detect_standard_signature(self):
        """Test detection of standard (x, m, p, t) signature."""

        def hamiltonian_standard(x, m, p, t=0):
            return 0.5 * p**2 + m

        adapter = HamiltonianAdapter(hamiltonian_standard)
        assert adapter.signature_type == "standard"
        assert adapter.uses_standard is True

    def test_detect_legacy_signature(self):
        """Test detection of legacy (x, p, m, t) signature."""

        def hamiltonian_legacy(x, p, m, t=0):
            return 0.5 * p**2 + m

        adapter = HamiltonianAdapter(hamiltonian_legacy)
        assert adapter.signature_type == "legacy"
        assert adapter.uses_standard is False

    def test_detect_neural_signature(self):
        """Test detection of neural (t, x, p, m) signature."""

        def hamiltonian_neural(t, x, p, m):
            return 0.5 * p**2 + m

        adapter = HamiltonianAdapter(hamiltonian_neural)
        assert adapter.signature_type == "neural"
        assert adapter.uses_standard is False

    def test_manual_signature_hint(self):
        """Test manual signature hint override."""

        def some_hamiltonian(a, b, c, d):
            return 0.5 * c**2 + b

        # Force standard interpretation
        adapter = HamiltonianAdapter(some_hamiltonian, signature_hint="standard")
        assert adapter.signature_type == "standard"

        # Force legacy interpretation
        adapter = HamiltonianAdapter(some_hamiltonian, signature_hint="legacy")
        assert adapter.signature_type == "legacy"


class TestStandardSignature:
    """Test adapter with standard signature functions."""

    def test_standard_scalar(self):
        """Test standard signature with scalar inputs."""

        def hamiltonian(x, m, p, t=0):
            return 0.5 * p**2 + m * x + t

        adapter = HamiltonianAdapter(hamiltonian)
        result = adapter(x=1.0, m=0.5, p=2.0, t=0.1)

        expected = 0.5 * 2.0**2 + 0.5 * 1.0 + 0.1
        assert result == pytest.approx(expected)

    def test_standard_array(self):
        """Test standard signature with array inputs."""

        def hamiltonian(x, m, p, t=0):
            return 0.5 * np.sum(p**2) + m

        adapter = HamiltonianAdapter(hamiltonian)
        p_array = np.array([1.0, 2.0, 3.0])
        result = adapter(x=np.zeros(3), m=0.5, p=p_array, t=0.0)

        expected = 0.5 * (1.0 + 4.0 + 9.0) + 0.5
        assert result == pytest.approx(expected)


class TestLegacySignature:
    """Test adapter with legacy signature functions."""

    def test_legacy_scalar(self):
        """Test legacy (x, p, m) signature conversion."""

        def hamiltonian_legacy(x, p, m, t=0):
            return 0.5 * p**2 + m * x + t

        adapter = HamiltonianAdapter(hamiltonian_legacy)
        result = adapter(x=1.0, m=0.5, p=2.0, t=0.1)

        expected = 0.5 * 2.0**2 + 0.5 * 1.0 + 0.1
        assert result == pytest.approx(expected)
        assert adapter.signature_type == "legacy"

    def test_legacy_array(self):
        """Test legacy signature with arrays."""

        def hamiltonian_legacy(x, p, m, t=0):
            return 0.5 * np.sum(p**2) + m

        adapter = HamiltonianAdapter(hamiltonian_legacy)
        p_array = np.array([1.0, 2.0])
        result = adapter(x=np.zeros(2), m=0.5, p=p_array, t=0.0)

        expected = 0.5 * (1.0 + 4.0) + 0.5
        assert result == pytest.approx(expected)


class TestNeuralSignature:
    """Test adapter with neural network signature."""

    def test_neural_scalar(self):
        """Test neural (t, x, p, m) signature conversion."""

        def hamiltonian_neural(t, x, p, m):
            return 0.5 * p**2 + m * x + t

        adapter = HamiltonianAdapter(hamiltonian_neural)
        result = adapter(x=1.0, m=0.5, p=2.0, t=0.1)

        expected = 0.5 * 2.0**2 + 0.5 * 1.0 + 0.1
        assert result == pytest.approx(expected)
        assert adapter.signature_type == "neural"


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_hamiltonian_adapter(self):
        """Test factory function."""

        def hamiltonian(x, p, m, t=0):
            return 0.5 * p**2

        adapter = create_hamiltonian_adapter(hamiltonian)
        assert isinstance(adapter, HamiltonianAdapter)
        assert adapter.signature_type == "legacy"

    def test_create_hamiltonian_adapter_none(self):
        """Test factory with None input."""
        adapter = create_hamiltonian_adapter(None)
        assert adapter is None

    def test_adapt_hamiltonian_direct(self):
        """Test one-shot adaptation."""

        def hamiltonian_legacy(x, p, m, t=0):
            return 0.5 * p**2 + m

        result = adapt_hamiltonian(hamiltonian_legacy, x=1.0, m=0.5, p=2.0, t=0.0)

        expected = 0.5 * 2.0**2 + 0.5
        assert result == pytest.approx(expected)


class TestMethodSignatures:
    """Test with class methods."""

    def test_class_method_standard(self):
        """Test adapter with class method (standard signature)."""

        class Problem:
            def hamiltonian(self, x, m, p, t=0):
                return 0.5 * p**2 + m

        problem = Problem()
        adapter = HamiltonianAdapter(problem.hamiltonian)
        assert adapter.signature_type == "standard"

        result = adapter(x=1.0, m=0.5, p=2.0, t=0.0)
        assert result == pytest.approx(2.5)

    def test_class_method_legacy(self):
        """Test adapter with class method (legacy signature)."""

        class Problem:
            def hamiltonian(self, x, p, m, t=0):
                return 0.5 * p**2 + m

        problem = Problem()
        adapter = HamiltonianAdapter(problem.hamiltonian)
        assert adapter.signature_type == "legacy"

        result = adapter(x=1.0, m=0.5, p=2.0, t=0.0)
        assert result == pytest.approx(2.5)


class TestAdapterInfo:
    """Test adapter information methods."""

    def test_get_info(self):
        """Test get_info() method."""

        def hamiltonian_legacy(x, p, m, t=0):
            return 0.5 * p**2

        adapter = HamiltonianAdapter(hamiltonian_legacy)
        info = adapter.get_info()

        assert info["signature_type"] == "legacy"
        assert info["uses_standard"] is False
        assert info["function_name"] == "hamiltonian_legacy"
        assert "x" in info["parameter_names"]

    def test_repr(self):
        """Test string representation."""

        def my_hamiltonian(x, m, p, t=0):
            return 0.5 * p**2

        adapter = HamiltonianAdapter(my_hamiltonian)
        repr_str = repr(adapter)

        assert "HamiltonianAdapter" in repr_str
        assert "standard" in repr_str
        assert "my_hamiltonian" in repr_str


class TestWarnings:
    """Test that warnings are issued for non-standard signatures."""

    def test_legacy_signature_warning(self):
        """Test that legacy signature triggers warning."""

        def hamiltonian_legacy(x, p, m, t=0):
            return 0.5 * p**2

        with pytest.warns(FutureWarning, match="non-standard signature"):
            HamiltonianAdapter(hamiltonian_legacy)

    def test_standard_no_warning(self):
        """Test that standard signature doesn't trigger warning."""

        def hamiltonian_standard(x, m, p, t=0):
            return 0.5 * p**2

        # Should not warn - use warnings.catch_warnings instead
        import warnings

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            HamiltonianAdapter(hamiltonian_standard)

        # Filter for FutureWarnings
        future_warnings = [w for w in warning_list if issubclass(w.category, FutureWarning)]
        assert len(future_warnings) == 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_lambda_function(self):
        """Test with lambda function."""
        hamiltonian = lambda x, m, p, t=0: 0.5 * p**2 + m  # noqa: E731

        adapter = HamiltonianAdapter(hamiltonian)
        result = adapter(x=1.0, m=0.5, p=2.0, t=0.0)
        assert result == pytest.approx(2.5)

    def test_unknown_signature_with_hint(self):
        """Test unknown signature with explicit hint."""

        def weird_hamiltonian(foo, bar, baz, qux=0):
            # Will be interpreted as (x, m, p, t) with hint
            return 0.5 * baz**2 + bar

        # Use hint to specify standard signature
        adapter = HamiltonianAdapter(weird_hamiltonian, signature_hint="standard")
        result = adapter(x=1.0, m=0.5, p=2.0, t=0.0)
        assert result == pytest.approx(2.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
