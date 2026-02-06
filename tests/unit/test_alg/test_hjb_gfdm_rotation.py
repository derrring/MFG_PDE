"""
Unit tests for GFDM Local Coordinate Rotation (LCR) correctness.

Tests the rotation matrix construction in HJBGFDMSolver to ensure
proper transformation of coordinate frames at boundary points.

Issue #531: GFDM boundary stencil degeneracy fix verification.
"""

import pytest

import numpy as np

from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver
from mfg_pde.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfg_pde.core.mfg_components import MFGComponents
from mfg_pde.geometry import TensorProductGrid, neumann_bc


def _default_hamiltonian():
    """Default Hamiltonian for testing (Issue #670: explicit specification required)."""
    return SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: m,
        coupling_dm=lambda m: 1.0,
    )


def _default_components_2d():
    """Default MFGComponents for 2D testing (Issue #670: explicit specification required)."""

    def m_initial_2d(x):
        x_arr = np.asarray(x)
        return np.exp(-10 * np.sum((x_arr - 0.5) ** 2))

    return MFGComponents(
        m_initial=m_initial_2d,
        u_terminal=lambda x: 0.0,
        hamiltonian=_default_hamiltonian(),
    )


class TestRotationMatrixCorrectness:
    """Test suite for _build_rotation_matrix() correctness."""

    @pytest.fixture
    def solver(self):
        """Create a minimal GFDM solver for testing rotation matrices."""
        # Create simple 2D geometry
        geometry = TensorProductGrid(
            bounds=[(0.0, 10.0), (0.0, 10.0)],
            Nx=[11, 11],
            boundary_conditions=neumann_bc(dimension=2),
        )

        # Create problem with geometry
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=10, components=_default_components_2d())

        # Get collocation points from geometry
        collocation_points = geometry.get_spatial_grid()

        # Create solver with LCR enabled
        solver = HJBGFDMSolver(
            problem,
            collocation_points=collocation_points,
            delta=1.5,
            use_local_coordinate_rotation=True,
        )

        return solver

    def test_rotation_maps_ex_to_normal_2d(self, solver):
        """
        Test that rotation matrix R satisfies: R @ e_x = n (normal).

        This is the fundamental property: the first column of R should be
        the normal vector itself.
        """
        # Test normals for different boundary orientations
        test_cases = [
            (np.array([-1.0, 0.0]), "Left wall"),
            (np.array([1.0, 0.0]), "Right wall"),
            (np.array([0.0, -1.0]), "Bottom wall"),
            (np.array([0.0, 1.0]), "Top wall"),
            (np.array([-1.0, -1.0]) / np.sqrt(2), "Bottom-left corner"),
            (np.array([1.0, -1.0]) / np.sqrt(2), "Bottom-right corner"),
            (np.array([-1.0, 1.0]) / np.sqrt(2), "Top-left corner"),
            (np.array([1.0, 1.0]) / np.sqrt(2), "Top-right corner"),
        ]

        e_x = np.array([1.0, 0.0])

        for normal, description in test_cases:
            R = solver._boundary_handler.build_rotation_matrix(normal)
            result = R @ e_x

            assert np.allclose(result, normal, atol=1e-10), (
                f"{description}: R @ e_x = {result}, expected {normal}. Matrix R = \n{R}"
            )

    def test_rotation_preserves_norm_2d(self, solver):
        """Test that rotation matrix is orthogonal (preserves vector norms)."""
        test_normals = [
            np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([1.0, 1.0]) / np.sqrt(2),
        ]

        test_vector = np.array([3.0, 4.0])  # norm = 5.0
        expected_norm = np.linalg.norm(test_vector)

        for normal in test_normals:
            R = solver._boundary_handler.build_rotation_matrix(normal)
            rotated = R @ test_vector
            actual_norm = np.linalg.norm(rotated)

            assert np.allclose(actual_norm, expected_norm, atol=1e-10), (
                f"Rotation with normal {normal} changed norm from {expected_norm} to {actual_norm}"
            )

    def test_rotation_is_orthogonal_2d(self, solver):
        """Test that R^T @ R = I (rotation matrix is orthogonal)."""
        test_normals = [
            np.array([-1.0, 0.0]),
            np.array([0.0, -1.0]),
            np.array([1.0, 1.0]) / np.sqrt(2),
        ]

        for normal in test_normals:
            R = solver._boundary_handler.build_rotation_matrix(normal)
            product = R.T @ R

            assert np.allclose(product, np.eye(2), atol=1e-10), f"R^T @ R != I for normal {normal}. Got:\n{product}"

    def test_rotation_determinant_is_one_2d(self, solver):
        """Test that det(R) = 1 (proper rotation, not reflection)."""
        test_normals = [
            np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([1.0, -1.0]) / np.sqrt(2),
        ]

        for normal in test_normals:
            R = solver._boundary_handler.build_rotation_matrix(normal)
            det = np.linalg.det(R)

            assert np.allclose(det, 1.0, atol=1e-10), f"det(R) = {det}, expected 1.0 for normal {normal}"

    def test_horizontal_boundaries_regression(self, solver):
        """
        Regression test for Issue #531 rotation bug.

        The original bug inverted the y-component at horizontal boundaries:
        - Bottom wall (0, -1) mapped to (0, +1)
        - Top wall (0, +1) mapped to (0, -1)

        This test ensures the fix is correct.
        """
        e_x = np.array([1.0, 0.0])

        # Bottom wall: normal = (0, -1)
        R_bottom = solver._boundary_handler.build_rotation_matrix(np.array([0.0, -1.0]))
        result_bottom = R_bottom @ e_x
        assert np.allclose(result_bottom, [0.0, -1.0], atol=1e-10), (
            f"Bottom wall regression: R @ e_x = {result_bottom}, expected [0, -1]"
        )

        # Top wall: normal = (0, +1)
        R_top = solver._boundary_handler.build_rotation_matrix(np.array([0.0, 1.0]))
        result_top = R_top @ e_x
        assert np.allclose(result_top, [0.0, 1.0], atol=1e-10), (
            f"Top wall regression: R @ e_x = {result_top}, expected [0, +1]"
        )


class TestDerivativeRotationBackTransform:
    """Test suite for _rotate_derivatives_back() correctness."""

    @pytest.fixture
    def solver(self):
        """Create a minimal GFDM solver for testing."""
        geometry = TensorProductGrid(
            bounds=[(0.0, 10.0), (0.0, 10.0)],
            Nx=[11, 11],
            boundary_conditions=neumann_bc(dimension=2),
        )
        problem = MFGProblem(geometry=geometry, T=1.0, Nt=10, components=_default_components_2d())
        collocation_points = geometry.get_spatial_grid()

        solver = HJBGFDMSolver(
            problem,
            collocation_points=collocation_points,
            delta=1.5,
            use_local_coordinate_rotation=True,
        )
        return solver

    def test_gradient_rotation_identity(self, solver):
        """Test that rotating and rotating back recovers original gradient."""
        # Original gradient in physical frame
        grad_orig = np.array([2.0, -3.0])

        # Rotation matrix (arbitrary normal)
        normal = np.array([1.0, 1.0]) / np.sqrt(2)
        R = solver._boundary_handler.build_rotation_matrix(normal)

        # Rotate gradient to normal-aligned frame: grad' = R @ grad
        grad_rotated = R @ grad_orig

        # Create derivatives dict in rotated frame
        derivs_rotated = {
            (1, 0): grad_rotated[0],  # ∂u/∂x' (normal direction)
            (0, 1): grad_rotated[1],  # ∂u/∂y' (tangential direction)
        }

        # Rotate back
        derivs_back = solver._boundary_handler.rotate_derivatives_back(derivs_rotated, R)

        # Check recovery
        grad_back = np.array([derivs_back[(1, 0)], derivs_back[(0, 1)]])

        assert np.allclose(grad_back, grad_orig, atol=1e-10), (
            f"Gradient rotation roundtrip failed: {grad_back} != {grad_orig}"
        )

    def test_hessian_rotation_preserves_laplacian(self, solver):
        """Test that Laplacian (trace of Hessian) is rotation-invariant."""
        # Create symmetric Hessian in original frame
        H_orig = np.array([[2.0, 0.5], [0.5, 3.0]])  # Symmetric

        # Laplacian = trace = 2.0 + 3.0 = 5.0
        lap_orig = np.trace(H_orig)

        # Rotation matrix
        normal = np.array([0.0, 1.0])
        R = solver._boundary_handler.build_rotation_matrix(normal)

        # Rotate Hessian: H' = R @ H @ R^T
        H_rotated = R @ H_orig @ R.T

        # Create derivatives dict in rotated frame
        derivs_rotated = {
            (1, 0): 0.0,  # ∂u/∂x' (not used)
            (0, 1): 0.0,  # ∂u/∂y' (not used)
            (2, 0): H_rotated[0, 0],  # ∂²u/∂x'²
            (1, 1): H_rotated[0, 1],  # ∂²u/∂x'∂y'
            (0, 2): H_rotated[1, 1],  # ∂²u/∂y'²
        }

        # Rotate back
        derivs_back = solver._boundary_handler.rotate_derivatives_back(derivs_rotated, R)

        # Compute Laplacian from rotated-back derivatives
        lap_back = derivs_back[(2, 0)] + derivs_back[(0, 2)]

        assert np.allclose(lap_back, lap_orig, atol=1e-10), f"Laplacian not preserved: {lap_back} != {lap_orig}"


def test_smoke_rotation_matrix():
    """Quick smoke test that can run without pytest."""
    print("Testing rotation matrix correctness...")

    # Create simple geometry and problem
    geometry = TensorProductGrid(
        bounds=[(0.0, 10.0), (0.0, 10.0)],
        Nx=[11, 11],
        boundary_conditions=neumann_bc(dimension=2),
    )
    problem = MFGProblem(geometry=geometry, T=1.0, Nt=10, components=_default_components_2d())
    collocation_points = geometry.get_spatial_grid()

    solver = HJBGFDMSolver(
        problem,
        collocation_points=collocation_points,
        delta=1.5,
        use_local_coordinate_rotation=True,
    )

    # Test key orientations
    test_normals = [
        (np.array([-1.0, 0.0]), "Left"),
        (np.array([1.0, 0.0]), "Right"),
        (np.array([0.0, -1.0]), "Bottom"),
        (np.array([0.0, 1.0]), "Top"),
        (np.array([-1.0, -1.0]) / np.sqrt(2), "Corner"),
    ]

    e_x = np.array([1.0, 0.0])
    all_passed = True

    for normal, label in test_normals:
        R = solver._boundary_handler.build_rotation_matrix(normal)
        result = R @ e_x

        if np.allclose(result, normal, atol=1e-10):
            print(f"✓ {label:12s}: R @ e_x = {result} (correct)")
        else:
            print(f"✗ {label:12s}: R @ e_x = {result}, expected {normal} (FAILED)")
            all_passed = False

    if all_passed:
        print("\n✓ All rotation matrix tests passed!")
    else:
        print("\n✗ Some tests failed!")
        return False

    return True


if __name__ == "__main__":
    # Run smoke test
    success = test_smoke_rotation_matrix()
    exit(0 if success else 1)
