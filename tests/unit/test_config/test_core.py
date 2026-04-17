"""Unit tests for `mfgarchon.config.core` (Issue #1010 B2).

Covers the canonical composite/top-level Pydantic configs that form the core of
MFGArchon's solver configuration:
- LoggingConfig
- BackendConfig
- PicardConfig (defaults + range validators only — alias tests live in
  test_picard_relaxation_alias.py)
- MFGSolverConfig
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mfgarchon.config import (
    BackendConfig,
    LoggingConfig,
    MFGSolverConfig,
    PicardConfig,
    SolverConfig,
)


class TestLoggingConfig:
    def test_defaults(self):
        c = LoggingConfig()
        assert c.level == "INFO"
        assert c.progress_bar is True
        assert c.save_intermediate is False
        assert c.output_dir is None

    def test_level_enum(self):
        for lvl in ("DEBUG", "INFO", "WARNING", "ERROR"):
            assert LoggingConfig(level=lvl).level == lvl

    def test_invalid_level_rejected(self):
        with pytest.raises(ValidationError):
            LoggingConfig(level="TRACE")

    def test_save_intermediate_requires_output_dir(self):
        with pytest.raises(ValidationError, match="output_dir must be provided"):
            LoggingConfig(save_intermediate=True)

    def test_save_intermediate_with_output_dir_ok(self):
        c = LoggingConfig(save_intermediate=True, output_dir="/tmp/results")
        assert c.save_intermediate is True
        assert c.output_dir == "/tmp/results"


class TestBackendConfig:
    def test_defaults(self):
        c = BackendConfig()
        assert c.type == "numpy"
        assert c.device == "cpu"
        assert c.precision == "float64"

    @pytest.mark.parametrize("backend", ["numpy", "jax", "pytorch"])
    def test_backends_accepted(self, backend):
        assert BackendConfig(type=backend).type == backend

    def test_numpy_gpu_rejected(self):
        with pytest.raises(ValidationError, match="NumPy backend does not support GPU"):
            BackendConfig(type="numpy", device="gpu")

    def test_jax_gpu_accepted(self):
        c = BackendConfig(type="jax", device="gpu")
        assert c.device == "gpu"

    def test_invalid_precision_rejected(self):
        with pytest.raises(ValidationError):
            BackendConfig(precision="float16")


class TestPicardConfigCanonical:
    """Canonical-path coverage only. Alias coverage in test_picard_relaxation_alias.py."""

    def test_defaults(self):
        c = PicardConfig()
        assert c.max_iterations == 100
        assert c.tolerance == 1e-6
        assert c.relaxation == 0.5
        assert c.relaxation_M is None
        assert c.adaptive_relaxation is False
        assert c.anderson_memory == 0
        assert c.verbose is True

    def test_relaxation_range(self):
        PicardConfig(relaxation=0.01)  # lower bound ok
        PicardConfig(relaxation=1.0)  # upper bound ok
        with pytest.raises(ValidationError):
            PicardConfig(relaxation=0.0)
        with pytest.raises(ValidationError):
            PicardConfig(relaxation=1.5)

    def test_anderson_memory_cannot_exceed_max_iterations(self):
        with pytest.raises(ValidationError, match="anderson_memory cannot exceed"):
            PicardConfig(max_iterations=10, anderson_memory=20)

    def test_anderson_memory_negative_rejected(self):
        with pytest.raises(ValidationError):
            PicardConfig(anderson_memory=-1)

    def test_schedule_enum(self):
        for s in ("constant", "harmonic", "sqrt", "exponential"):
            assert PicardConfig(relaxation_schedule=s).relaxation_schedule == s

    def test_invalid_schedule_rejected(self):
        with pytest.raises(ValidationError):
            PicardConfig(relaxation_schedule="arithmetic")


class TestMFGSolverConfig:
    def test_default_construction_populates_all_subconfigs(self):
        c = MFGSolverConfig()
        # All sub-configs should be non-None after default_factory runs
        assert c.hjb is not None
        assert c.fp is not None
        assert c.picard is not None
        assert c.backend is not None
        assert c.logging is not None

    def test_solverconfig_is_alias_for_mfgsolverconfig(self):
        assert SolverConfig is MFGSolverConfig

    def test_nested_override(self):
        c = MFGSolverConfig(picard=PicardConfig(max_iterations=50))
        assert c.picard.max_iterations == 50
        # Other sub-configs keep defaults
        assert c.logging.level == "INFO"

    def test_model_dump_roundtrip(self):
        original = MFGSolverConfig(
            picard=PicardConfig(relaxation=0.3, max_iterations=200),
            backend=BackendConfig(type="jax", device="gpu"),
        )
        dumped = original.model_dump()
        rebuilt = MFGSolverConfig(**dumped)
        assert rebuilt.picard.relaxation == 0.3
        assert rebuilt.picard.max_iterations == 200
        assert rebuilt.backend.type == "jax"
        assert rebuilt.backend.device == "gpu"

    def test_model_dump_yaml_excludes_none(self):
        c = MFGSolverConfig()
        dumped = c.model_dump_yaml()
        # None-valued fields should not appear in YAML output
        assert "relaxation_M" not in dumped.get("picard", {}) or dumped["picard"].get("relaxation_M") is not None
