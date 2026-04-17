"""Equivalence tests for PicardConfig damping_* -> relaxation_* rename (v0.19.1).

Per CLAUDE.md deprecation policy: every deprecated API must have an equivalence
test proving the old and new paths produce identical behavior. These tests
guard against silent divergence during the deprecation window (Issue #616
lesson).

The legacy names are accepted via `@model_validator(mode="before")` in
`PicardConfig._translate_legacy_damping_names`. Removal is scheduled for
v0.25.0 per the standard 3-version window.
"""

from __future__ import annotations

import warnings

import pytest

from mfgarchon.config import PicardConfig


class TestPicardRelaxationAliasEquivalence:
    """Prove legacy `damping_*` kwargs produce identical instances to canonical `relaxation_*`."""

    def test_damping_factor_equivalent_to_relaxation(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy = PicardConfig(damping_factor=0.7)
        canonical = PicardConfig(relaxation=0.7)
        assert legacy.relaxation == canonical.relaxation == 0.7
        assert legacy == canonical

    def test_damping_factor_M_equivalent_to_relaxation_M(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy = PicardConfig(damping_factor=0.5, damping_factor_M=0.3)
        canonical = PicardConfig(relaxation=0.5, relaxation_M=0.3)
        assert legacy.relaxation_M == canonical.relaxation_M == 0.3
        assert legacy == canonical

    def test_damping_schedule_equivalent_to_relaxation_schedule(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy = PicardConfig(damping_schedule="harmonic", damping_schedule_M="sqrt")
        canonical = PicardConfig(relaxation_schedule="harmonic", relaxation_schedule_M="sqrt")
        assert legacy.relaxation_schedule == canonical.relaxation_schedule == "harmonic"
        assert legacy.relaxation_schedule_M == canonical.relaxation_schedule_M == "sqrt"
        assert legacy == canonical

    def test_adaptive_damping_equivalent_to_adaptive_relaxation(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy = PicardConfig(adaptive_damping=True)
        canonical = PicardConfig(adaptive_relaxation=True)
        assert legacy.adaptive_relaxation is canonical.adaptive_relaxation is True
        assert legacy == canonical

    def test_all_five_legacy_names_together(self):
        """Full equivalence: every legacy kwarg translated correctly in one call."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy = PicardConfig(
                damping_factor=0.6,
                damping_factor_M=0.2,
                damping_schedule="exponential",
                damping_schedule_M="harmonic",
                adaptive_damping=True,
            )
        canonical = PicardConfig(
            relaxation=0.6,
            relaxation_M=0.2,
            relaxation_schedule="exponential",
            relaxation_schedule_M="harmonic",
            adaptive_relaxation=True,
        )
        assert legacy == canonical

    def test_model_dump_produces_canonical_keys_only(self):
        """Serialization must use canonical field names only, regardless of input names."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy = PicardConfig(damping_factor=0.9, damping_factor_M=0.1)
        dumped = legacy.model_dump()
        assert "relaxation" in dumped
        assert "relaxation_M" in dumped
        # Legacy names must not leak into the serialized form
        assert "damping_factor" not in dumped
        assert "damping_factor_M" not in dumped


class TestPicardRelaxationAliasWarnings:
    """Verify DeprecationWarning semantics."""

    @pytest.mark.parametrize(
        "legacy_name,canonical_name,value",
        [
            ("damping_factor", "relaxation", 0.7),
            ("damping_factor_M", "relaxation_M", 0.3),
            ("damping_schedule", "relaxation_schedule", "harmonic"),
            ("damping_schedule_M", "relaxation_schedule_M", "sqrt"),
            ("adaptive_damping", "adaptive_relaxation", True),
        ],
    )
    def test_each_legacy_name_emits_deprecation_warning(self, legacy_name, canonical_name, value):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            PicardConfig(**{legacy_name: value})
            dep_warnings = [
                x for x in w
                if issubclass(x.category, DeprecationWarning)
                and legacy_name in str(x.message)
            ]
        assert len(dep_warnings) == 1, f"Expected 1 DeprecationWarning for '{legacy_name}', got {len(dep_warnings)}"
        assert canonical_name in str(dep_warnings[0].message)

    def test_canonical_name_emits_no_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            PicardConfig(relaxation=0.7, relaxation_M=0.3)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) == 0


class TestPicardRelaxationAliasCollision:
    """Passing both the legacy name and canonical name must raise."""

    def test_both_damping_factor_and_relaxation_raises(self):
        with pytest.raises(ValueError, match="both legacy .* and canonical"):
            PicardConfig(damping_factor=0.5, relaxation=0.8)

    def test_both_damping_factor_M_and_relaxation_M_raises(self):
        with pytest.raises(ValueError, match="both legacy .* and canonical"):
            PicardConfig(damping_factor_M=0.5, relaxation_M=0.8)

    def test_both_damping_schedule_and_relaxation_schedule_raises(self):
        with pytest.raises(ValueError, match="both legacy .* and canonical"):
            PicardConfig(damping_schedule="harmonic", relaxation_schedule="sqrt")
