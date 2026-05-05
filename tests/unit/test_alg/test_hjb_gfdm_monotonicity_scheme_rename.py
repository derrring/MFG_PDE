"""Equivalence tests for v0.18.0 monotonicity_scheme rename.

mfgarchon CLAUDE.md deprecation policy requires:
  - Immediate redirection: old API calls new API internally
  - Equivalence test: old API == new API give identical behavior

This module verifies that for each of the four legacy
`qp_optimization_level` values, the corresponding new
(`monotonicity_scheme`, `monotonicity_application`) tuple produces
identical solver state (same scheme/application/method-name and
same Laplacian/gradient stencil weights).

Issue #XXXX. Removal blocker: equivalence_test → check_internal_deprecation.py.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mfgarchon.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfgarchon.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfgarchon.core.mfg_components import MFGComponents
from mfgarchon.core.mfg_problem import MFGProblem
from mfgarchon.geometry import TensorProductGrid
from mfgarchon.geometry.boundary import no_flux_bc


def _problem_2d_quasi_uniform():
    """2D problem with N=11x11 quasi-uniform interior + boundary indices.

    Layout: 121 collocation points on [0,1]^2, with 4 corners + edges as boundary.
    Returns (problem, points, boundary_indices).
    """
    bc = no_flux_bc(dimension=2)
    domain = TensorProductGrid(
        bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11], boundary_conditions=bc
    )
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )
    components = MFGComponents(
        m_initial=lambda x: 1.0, u_terminal=lambda x: 0.0, hamiltonian=H,
    )
    problem = MFGProblem(geometry=domain, T=1.0, Nt=10, sigma=0.5, components=components)
    pts = problem.geometry.get_spatial_grid()
    if pts.ndim == 1:
        pts = np.atleast_2d(pts).T
    bdry = []
    n = pts.shape[0]
    for i, p in enumerate(pts):
        if min(p[0], 1.0 - p[0], p[1], 1.0 - p[1]) < 1e-9:
            bdry.append(i)
    return problem, pts, np.array(bdry)


# ---------------------------------------------------------------------------
# Mapping table (legacy → new) per docstring of HJBGFDMSolver
# ---------------------------------------------------------------------------
LEGACY_TO_NEW = {
    "none":       ("none", None),         # application=None → "ignored"
    "auto":       ("qp_m_matrix", "adaptive"),
    "always":     ("qp_m_matrix", "always"),
    "precompute": ("qp_m_matrix", "precompute"),
}


@pytest.fixture(scope="module")
def setup():
    return _problem_2d_quasi_uniform()


# ---------------------------------------------------------------------------
# Equivalence tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("legacy_value", list(LEGACY_TO_NEW.keys()))
def test_scheme_application_match(setup, legacy_value):
    """For each legacy value, verify new (scheme, application) produces identical
    self.monotonicity_scheme, self.monotonicity_application, and self.qp_optimization_level."""
    problem, pts, bdry = setup
    new_scheme, new_app = LEGACY_TO_NEW[legacy_value]

    # New API
    s_new = HJBGFDMSolver(
        problem, collocation_points=pts, boundary_indices=bdry, delta=0.3,
        monotonicity_scheme=new_scheme, monotonicity_application=new_app,
    )

    # Legacy API
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        s_old = HJBGFDMSolver(
            problem, collocation_points=pts, boundary_indices=bdry, delta=0.3,
            qp_optimization_level=legacy_value,
        )

    assert s_new.monotonicity_scheme == s_old.monotonicity_scheme, \
        f"scheme mismatch for legacy={legacy_value}: new={s_new.monotonicity_scheme}, old={s_old.monotonicity_scheme}"
    assert s_new.monotonicity_application == s_old.monotonicity_application, \
        f"application mismatch for legacy={legacy_value}: new={s_new.monotonicity_application}, old={s_old.monotonicity_application}"
    assert s_new.qp_optimization_level == s_old.qp_optimization_level, \
        f"legacy alias mismatch for legacy={legacy_value}"
    assert s_new.hjb_method_name == s_old.hjb_method_name, \
        f"method_name mismatch for legacy={legacy_value}"


@pytest.mark.parametrize("legacy_value", list(LEGACY_TO_NEW.keys()))
def test_stencil_weights_identical(setup, legacy_value):
    """Beyond config equivalence, verify that the actual Laplacian and gradient
    stencil weights are bit-identical between old and new API."""
    problem, pts, bdry = setup
    new_scheme, new_app = LEGACY_TO_NEW[legacy_value]

    s_new = HJBGFDMSolver(
        problem, collocation_points=pts, boundary_indices=bdry, delta=0.3,
        monotonicity_scheme=new_scheme, monotonicity_application=new_app,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        s_old = HJBGFDMSolver(
            problem, collocation_points=pts, boundary_indices=bdry, delta=0.3,
            qp_optimization_level=legacy_value,
        )

    op_new = s_new._gfdm_operator
    op_old = s_old._gfdm_operator
    n_pts = pts.shape[0]
    for i in range(n_pts):
        w_new = op_new.get_derivative_weights(i)
        w_old = op_old.get_derivative_weights(i)
        if w_new is None and w_old is None:
            continue
        assert (w_new is None) == (w_old is None), f"mismatch at i={i}: weights None status differs"
        np.testing.assert_array_equal(
            w_new["neighbor_indices"], w_old["neighbor_indices"],
            err_msg=f"neighbor_indices mismatch at i={i} for legacy={legacy_value}",
        )
        np.testing.assert_allclose(
            w_new["lap_weights"], w_old["lap_weights"], rtol=1e-12,
            err_msg=f"lap_weights mismatch at i={i} for legacy={legacy_value}",
        )
        np.testing.assert_allclose(
            w_new["grad_weights"], w_old["grad_weights"], rtol=1e-12,
            err_msg=f"grad_weights mismatch at i={i} for legacy={legacy_value}",
        )


def test_mutual_exclusion(setup):
    """Passing both new and legacy params must raise ValueError."""
    problem, pts, bdry = setup
    with pytest.raises(ValueError, match="Specify at most one"):
        HJBGFDMSolver(
            problem, collocation_points=pts, boundary_indices=bdry, delta=0.3,
            monotonicity_scheme="qp_m_matrix", qp_optimization_level="auto",
        )


def test_invalid_scheme_value(setup):
    """Passing a legacy bundle value to monotonicity_scheme= raises ValueError."""
    problem, pts, bdry = setup
    with pytest.raises(ValueError, match="monotonicity_scheme must be one of"):
        HJBGFDMSolver(
            problem, collocation_points=pts, boundary_indices=bdry, delta=0.3,
            monotonicity_scheme="auto",   # Wrong axis — "auto" is application, not scheme
        )


def test_invalid_application_value(setup):
    """Passing an unrecognized application value raises ValueError."""
    problem, pts, bdry = setup
    with pytest.raises(ValueError, match="monotonicity_application must be one of"):
        HJBGFDMSolver(
            problem, collocation_points=pts, boundary_indices=bdry, delta=0.3,
            monotonicity_scheme="qp_m_matrix", monotonicity_application="bogus",
        )


def test_default_application_per_scheme(setup):
    """When monotonicity_application=None, scheme-recommended default is used."""
    problem, pts, bdry = setup
    # qp_m_matrix → adaptive
    s = HJBGFDMSolver(problem, collocation_points=pts, boundary_indices=bdry, delta=0.3,
                     monotonicity_scheme="qp_m_matrix")
    assert s.monotonicity_application == "adaptive"
    # joint_socp → precompute (will warn since not yet implemented)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = HJBGFDMSolver(problem, collocation_points=pts, boundary_indices=bdry, delta=0.3,
                         monotonicity_scheme="joint_socp")
    assert s.monotonicity_application == "precompute"


def test_legacy_emits_deprecation_warning(setup):
    """Using qp_optimization_level= must emit a DeprecationWarning naming the
    replacement parameter."""
    problem, pts, bdry = setup
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        HJBGFDMSolver(
            problem, collocation_points=pts, boundary_indices=bdry, delta=0.3,
            qp_optimization_level="auto",
        )
        dep = [w for w in caught
               if issubclass(w.category, DeprecationWarning)
               and "qp_optimization_level" in str(w.message)
               and "monotonicity_scheme" in str(w.message)]
        assert len(dep) >= 1, \
            f"Expected DeprecationWarning naming qp_optimization_level + monotonicity_scheme; got: {[str(w.message) for w in caught]}"


def test_joint_socp_placeholder_warns(setup):
    """joint_socp scheme is reserved (Phase 1B) — currently behaves as 'none'
    with a warning."""
    problem, pts, bdry = setup
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s = HJBGFDMSolver(problem, collocation_points=pts, boundary_indices=bdry, delta=0.3,
                         monotonicity_scheme="joint_socp")
        joint_warns = [w for w in caught if "joint_socp" in str(w.message)]
        assert len(joint_warns) >= 1, "Expected warning about joint_socp not yet implemented"
    # Behaves as none for legacy field
    assert s.qp_optimization_level == "none"
