"""Equivalence tests for FixedPointIterator damping_* -> relaxation_* rename (v0.19.2).

Per CLAUDE.md deprecation policy: old API must produce identical behavior to
new API. The legacy `damping_*` ctor kwargs are accepted via
`@deprecated_parameter` decorators that emit `DeprecationWarning` and then
redirect internally. Backward-compat attribute access (`iter.damping_factor`)
is provided via silent `@property` aliases (no warning on read; loop-friendly).

Removal of legacy kwargs and properties scheduled for v0.25.0.
"""

from __future__ import annotations

import warnings

import pytest

from mfgarchon.alg.numerical.coupling import FixedPointIterator
from mfgarchon.alg.numerical.fp_solvers import FPFDMSolver
from mfgarchon.alg.numerical.hjb_solvers import HJBFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem


def _make_test_problem():
    """Minimal MFG problem for iterator construction tests."""
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )
    components = MFGComponents(
        hamiltonian=H,
        u_terminal=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )
    return MFGProblem(Nx=11, xmin=0.0, xmax=1.0, T=0.2, Nt=5, sigma=0.3, components=components)


@pytest.fixture
def solvers():
    """Build a (problem, hjb, fp) triple for iterator construction."""
    problem = _make_test_problem()
    return problem, HJBFDMSolver(problem), FPFDMSolver(problem)


class TestLegacyKwargsEquivalent:
    """Prove every legacy `damping_*` ctor kwarg yields the same instance state as canonical."""

    @pytest.mark.parametrize(
        ("legacy_name", "canonical_name", "value"),
        [
            ("damping_factor", "relaxation", 0.7),
            ("damping_factor_M", "relaxation_M", 0.3),
            ("adaptive_damping", "adaptive_relaxation", True),
            ("adaptive_damping_decay", "adaptive_relaxation_decay", 0.8),
            ("adaptive_damping_min", "adaptive_relaxation_min", 0.01),
            ("damping_schedule", "relaxation_schedule", "harmonic"),
            ("damping_schedule_M", "relaxation_schedule_M", "sqrt"),
        ],
    )
    def test_legacy_kwarg_assigns_same_canonical_attribute(self, solvers, legacy_name, canonical_name, value):
        problem, hjb, fp = solvers
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy_iter = FixedPointIterator(problem, hjb, fp, **{legacy_name: value})
        canonical_iter = FixedPointIterator(problem, hjb, fp, **{canonical_name: value})
        assert getattr(legacy_iter, canonical_name) == getattr(canonical_iter, canonical_name)

    def test_all_seven_legacy_kwargs_together(self, solvers):
        problem, hjb, fp = solvers
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy_iter = FixedPointIterator(
                problem,
                hjb,
                fp,
                damping_factor=0.6,
                damping_factor_M=0.2,
                adaptive_damping=True,
                adaptive_damping_decay=0.8,
                adaptive_damping_min=0.01,
                damping_schedule="exponential",
                damping_schedule_M="harmonic",
            )
        canonical_iter = FixedPointIterator(
            problem,
            hjb,
            fp,
            relaxation=0.6,
            relaxation_M=0.2,
            adaptive_relaxation=True,
            adaptive_relaxation_decay=0.8,
            adaptive_relaxation_min=0.01,
            relaxation_schedule="exponential",
            relaxation_schedule_M="harmonic",
        )
        for attr in [
            "relaxation",
            "relaxation_M",
            "adaptive_relaxation",
            "adaptive_relaxation_decay",
            "adaptive_relaxation_min",
            "relaxation_schedule",
            "relaxation_schedule_M",
        ]:
            assert getattr(legacy_iter, attr) == getattr(canonical_iter, attr)


class TestDeprecationWarnings:
    """Ctor must emit DeprecationWarning once per legacy kwarg passed."""

    @pytest.mark.parametrize(
        ("legacy_name", "value"),
        [
            ("damping_factor", 0.7),
            ("damping_factor_M", 0.3),
            ("adaptive_damping", True),
            ("damping_schedule", "harmonic"),
        ],
    )
    def test_each_legacy_kwarg_warns(self, solvers, legacy_name, value):
        problem, hjb, fp = solvers
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            FixedPointIterator(problem, hjb, fp, **{legacy_name: value})
            dep_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning) and legacy_name in str(x.message)
            ]
        assert len(dep_warnings) >= 1, f"Expected DeprecationWarning for '{legacy_name}'"

    def test_canonical_kwargs_emit_no_warning(self, solvers):
        problem, hjb, fp = solvers
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            FixedPointIterator(
                problem,
                hjb,
                fp,
                relaxation=0.7,
                relaxation_M=0.3,
                adaptive_relaxation=True,
                relaxation_schedule="harmonic",
            )
            dep_warnings = [
                x
                for x in w
                if issubclass(x.category, DeprecationWarning)
                and ("damping" in str(x.message) or "relaxation" in str(x.message))
            ]
        assert len(dep_warnings) == 0


class TestBackwardCompatAttributes:
    """Legacy `iter.damping_*` attribute access must still work (silent alias via @property)."""

    def test_damping_factor_property_reads_relaxation(self, solvers):
        problem, hjb, fp = solvers
        iter_ = FixedPointIterator(problem, hjb, fp, relaxation=0.8)
        assert iter_.damping_factor == 0.8
        assert iter_.damping_factor == iter_.relaxation

    def test_all_legacy_properties_return_canonical_values(self, solvers):
        problem, hjb, fp = solvers
        iter_ = FixedPointIterator(
            problem,
            hjb,
            fp,
            relaxation=0.6,
            relaxation_M=0.2,
            adaptive_relaxation=True,
            adaptive_relaxation_decay=0.7,
            adaptive_relaxation_min=0.05,
            relaxation_schedule="harmonic",
            relaxation_schedule_M="sqrt",
        )
        assert iter_.damping_factor == 0.6
        assert iter_.damping_factor_M == 0.2
        assert iter_.adaptive_damping is True
        assert iter_.adaptive_damping_decay == 0.7
        assert iter_.adaptive_damping_min == 0.05
        assert iter_.damping_schedule == "harmonic"
        assert iter_.damping_schedule_M == "sqrt"

    def test_property_read_emits_no_warning(self, solvers):
        """Property reads are silent — no DeprecationWarning flooding inside hot loops."""
        problem, hjb, fp = solvers
        iter_ = FixedPointIterator(problem, hjb, fp, relaxation=0.5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = iter_.damping_factor
            _ = iter_.damping_factor_M
            _ = iter_.adaptive_damping
            _ = iter_.damping_schedule
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) == 0
