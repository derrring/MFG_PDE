"""
Standalone verification of rotation matrix fix for Issue #531.

Tests that _build_rotation_matrix() correctly implements:
    R @ e_x = n  (maps x-axis to normal direction)

The bug was in the 2D case where horizontal boundaries had inverted y-components.
"""

import numpy as np


def build_rotation_matrix(normal: np.ndarray) -> np.ndarray:
    """
    Build rotation matrix that aligns first axis with the given normal vector.

    This is the FIXED implementation from hjb_gfdm.py.
    """
    dim = len(normal)

    if dim == 1:
        return np.array([[np.sign(normal[0]) if abs(normal[0]) > 1e-10 else 1.0]])

    elif dim == 2:
        # 2D: Rotation matrix that maps e_x to normal
        # R = [n_x, -n_y]
        #     [n_y,  n_x]
        # Verification: R @ e_x = R @ [1,0]^T = [n_x, n_y]^T = n
        n_x, n_y = normal
        return np.array([[n_x, -n_y], [n_y, n_x]])  # FIXED VERSION

    elif dim == 3:
        # 3D: Use Rodrigues' rotation formula
        e_x = np.array([1.0, 0.0, 0.0])

        dot = np.dot(e_x, normal)
        if abs(dot - 1.0) < 1e-10:
            return np.eye(3)
        if abs(dot + 1.0) < 1e-10:
            return np.diag([1.0, -1.0, -1.0])

        v = np.cross(e_x, normal)
        s = np.linalg.norm(v)
        c = dot

        v_skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + v_skew + v_skew @ v_skew * (1 - c) / (s * s)
        return R

    else:
        # Higher dimensions: Householder reflection
        e_1 = np.zeros(dim)
        e_1[0] = 1.0

        if np.allclose(normal, e_1):
            return np.eye(dim)
        if np.allclose(normal, -e_1):
            R = np.eye(dim)
            R[0, 0] = -1.0
            return R

        v = e_1 - normal
        v = v / np.linalg.norm(v)
        return np.eye(dim) - 2.0 * np.outer(v, v)


def main():
    """Run verification tests."""
    print("=" * 70)
    print("ROTATION MATRIX FIX VERIFICATION (Issue #531)")
    print("=" * 70)
    print()
    print("Testing: R @ e_x = n (rotation maps x-axis to boundary normal)")
    print()

    # Test cases: (normal, description, was_buggy)
    test_cases = [
        (np.array([-1.0, 0.0]), "Left wall (x=0)", False),
        (np.array([1.0, 0.0]), "Right wall (x=L)", False),
        (np.array([0.0, -1.0]), "Bottom wall (y=0)", True),  # BUG WAS HERE
        (np.array([0.0, 1.0]), "Top wall (y=L)", True),  # BUG WAS HERE
        (np.array([-1.0, -1.0]) / np.sqrt(2), "Bottom-left corner", True),
        (np.array([1.0, -1.0]) / np.sqrt(2), "Bottom-right corner", True),
        (np.array([-1.0, 1.0]) / np.sqrt(2), "Top-left corner", True),
        (np.array([1.0, 1.0]) / np.sqrt(2), "Top-right corner", True),
    ]

    e_x = np.array([1.0, 0.0])
    all_passed = True

    print(f"{'Boundary':25s} {'R @ e_x':20s} {'Expected':20s} {'Status':10s}")
    print("-" * 70)

    for normal, label, was_buggy in test_cases:
        R = build_rotation_matrix(normal)
        result = R @ e_x
        error = np.linalg.norm(result - normal)

        passed = error < 1e-10
        status_icon = "✓ PASS" if passed else "✗ FAIL"
        bug_marker = " (was buggy)" if was_buggy else ""

        result_str = f"[{result[0]:6.3f}, {result[1]:6.3f}]"
        expected_str = f"[{normal[0]:6.3f}, {normal[1]:6.3f}]"

        print(f"{label:25s} {result_str:20s} {expected_str:20s} {status_icon:10s}{bug_marker}")

        if not passed:
            all_passed = False
            print(f"  ERROR: ||R@e_x - n|| = {error:.2e}")
            print(f"  Matrix R = {R}")

    print()
    print("=" * 70)

    if all_passed:
        print("✓ ALL TESTS PASSED")
        print()
        print("The rotation matrix bug has been FIXED.")
        print("Horizontal boundaries (y=0, y=L) now correctly rotate to their normals.")
        print("This should eliminate the inverted gradient problem in GFDM-LCR solver.")
    else:
        print("✗ TESTS FAILED")
        print()
        print("The rotation matrix still has errors!")
        print("Check the implementation in hjb_gfdm.py:_build_rotation_matrix()")

    print("=" * 70)

    # Additional orthogonality checks
    print()
    print("Additional Validation: Orthogonality Properties")
    print("-" * 70)

    ortho_passed = True
    for normal, label, _ in test_cases:
        R = build_rotation_matrix(normal)

        # Check R^T @ R = I
        product = R.T @ R
        identity_error = np.linalg.norm(product - np.eye(2))

        # Check det(R) = 1
        det = np.linalg.det(R)
        det_error = abs(det - 1.0)

        ortho_ok = identity_error < 1e-10 and det_error < 1e-10
        status = "✓" if ortho_ok else "✗ FAIL"

        print(f"{status} {label:25s}: ||R^T@R - I|| = {identity_error:.2e}, |det(R)-1| = {det_error:.2e}")

        if not ortho_ok:
            ortho_passed = False

    print("-" * 70)
    if ortho_passed:
        print("✓ All rotation matrices are orthogonal (R^T @ R = I, det(R) = 1)")
    else:
        print("✗ Some matrices failed orthogonality checks")

    print("=" * 70)

    return all_passed and ortho_passed


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
