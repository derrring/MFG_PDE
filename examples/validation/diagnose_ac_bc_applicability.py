#!/usr/bin/env python3
"""
Diagnostic: Adjoint-Consistent BC Applicability Analysis.

This script validates the fundamental design assumption of the
AdjointConsistentProvider: it is derived from equilibrium conditions
at reflecting boundaries where the STALL POINT is AT THE BOUNDARY.

Mathematical Background:
-----------------------
The AC BC formula: dU/dn = -sigma^2/2 * d(ln m)/dn

Derivation:
1. At reflecting boundary, zero-flux condition: J.n = 0
2. J = -sigma^2/2 * grad(m) + m * alpha  where alpha = -grad(U)
3. At equilibrium: 0 = -sigma^2/2 * dm/dn - m * dU/dn
4. Rearranging: dU/dn = -sigma^2/2 * (1/m) * dm/dn = -sigma^2/2 * d(ln m)/dn

Key Insight: This derivation assumes the boundary IS the equilibrium point
(stall point). When the stall is interior, this BC is NOT valid!

Expected Results:
- Boundary stall (x=0): AC BC should help (design intent)
- Interior stall (x=0.5): AC BC should hurt (wrong assumption)
- This is NOT a bug - it's correct behavior for the BC's scope

Issue #625: AdjointConsistentProvider validation
"""

import numpy as np


def analyze_equilibrium_at_boundaries(x_stall: float, domain: tuple[float, float] = (0.0, 1.0)):
    """
    Analyze whether the equilibrium density gradient at boundaries
    matches the AC BC assumption.

    For Towel-on-Beach: m_eq(x) = Z^{-1} exp(-|x - x_stall|^2 / sigma^2)
    """
    x_min, x_max = domain
    sigma = 0.2

    # Create grid
    x = np.linspace(x_min, x_max, 51)

    # Exact equilibrium density (Boltzmann distribution)
    m_eq = np.exp(-(np.abs(x - x_stall) ** 2) / sigma**2)
    m_eq /= np.trapezoid(m_eq, x)  # Normalize

    # Exact equilibrium gradient: d(ln m)/dx = -2(x - x_stall)/sigma^2
    grad_ln_m_exact_left = -2 * (x_min - x_stall) / sigma**2
    grad_ln_m_exact_right = -2 * (x_max - x_stall) / sigma**2

    # Outward normal derivatives
    # Left: outward normal is -x direction, so d/dn = -d/dx
    # Right: outward normal is +x direction, so d/dn = +d/dx
    outward_grad_left_exact = -grad_ln_m_exact_left
    outward_grad_right_exact = grad_ln_m_exact_right

    # What AC BC would prescribe for HJB:
    # dU/dn = -sigma^2/2 * d(ln m)/dn
    ac_bc_value_left = -(sigma**2) / 2 * outward_grad_left_exact
    ac_bc_value_right = -(sigma**2) / 2 * outward_grad_right_exact

    # What standard Neumann prescribes:
    neumann_value = 0.0

    # For equilibrium HJB solution U_eq(x) = -V(x)/lambda + C
    # where V(x) = |x - x_stall|^2 / 2
    # So dU/dx = -(x - x_stall) / lambda
    # With lambda = 1 (crowd aversion), dU/dx = -(x - x_stall)
    lambda_crowd = 0.5  # Typical value

    # Equilibrium gradient at boundaries:
    dU_dx_left_eq = -(x_min - x_stall) / lambda_crowd
    dU_dx_right_eq = -(x_max - x_stall) / lambda_crowd

    # Outward normal gradients for U
    dU_dn_left_eq = -dU_dx_left_eq  # Outward normal is -x
    dU_dn_right_eq = dU_dx_right_eq  # Outward normal is +x

    return {
        "x_stall": x_stall,
        "sigma": sigma,
        "lambda": lambda_crowd,
        # BC values that would be imposed
        "ac_bc_left": ac_bc_value_left,
        "ac_bc_right": ac_bc_value_right,
        "neumann_bc": neumann_value,
        # True equilibrium values that SHOULD be imposed
        "true_dU_dn_left": dU_dn_left_eq,
        "true_dU_dn_right": dU_dn_right_eq,
        # Errors
        "ac_error_left": abs(ac_bc_value_left - dU_dn_left_eq),
        "ac_error_right": abs(ac_bc_value_right - dU_dn_right_eq),
        "neumann_error_left": abs(neumann_value - dU_dn_left_eq),
        "neumann_error_right": abs(neumann_value - dU_dn_right_eq),
    }


def main():
    print("=" * 70)
    print("Adjoint-Consistent BC Applicability Analysis")
    print("=" * 70)
    print()

    print("Mathematical Foundation:")
    print("-" * 70)
    print("AC BC formula: dU/dn = -sigma^2/2 * d(ln m)/dn")
    print("This is derived from equilibrium flux condition J.n = 0")
    print()
    print("KEY INSIGHT: The zero-flux derivation assumes the boundary IS the")
    print("equilibrium point (stall). At non-stall boundaries, the equilibrium")
    print("flux is not necessarily zero - agents have a preferred direction!")
    print()

    # Test two scenarios
    scenarios = [
        (0.0, "Boundary stall (x_stall = 0) - AC BC design intent"),
        (0.5, "Interior stall (x_stall = 0.5) - Outside AC BC scope"),
    ]

    for x_stall, description in scenarios:
        print("=" * 70)
        print(f"Scenario: {description}")
        print("=" * 70)

        result = analyze_equilibrium_at_boundaries(x_stall)

        print(f"\nParameters: x_stall={result['x_stall']}, sigma={result['sigma']}, lambda={result['lambda']}")
        print()

        print("At LEFT boundary (x=0):")
        print(f"  True equilibrium dU/dn:     {result['true_dU_dn_left']:+.4f}")
        print(f"  AC BC would impose:         {result['ac_bc_left']:+.4f}  (error: {result['ac_error_left']:.4f})")
        print(f"  Neumann BC would impose:    {result['neumann_bc']:+.4f}  (error: {result['neumann_error_left']:.4f})")

        if result["ac_error_left"] < result["neumann_error_left"]:
            print("  => AC BC is BETTER for left boundary")
        else:
            print("  => Neumann BC is BETTER for left boundary")
        print()

        print("At RIGHT boundary (x=1):")
        print(f"  True equilibrium dU/dn:     {result['true_dU_dn_right']:+.4f}")
        print(f"  AC BC would impose:         {result['ac_bc_right']:+.4f}  (error: {result['ac_error_right']:.4f})")
        print(
            f"  Neumann BC would impose:    {result['neumann_bc']:+.4f}  (error: {result['neumann_error_right']:.4f})"
        )

        if result["ac_error_right"] < result["neumann_error_right"]:
            print("  => AC BC is BETTER for right boundary")
        else:
            print("  => Neumann BC is BETTER for right boundary")
        print()

        # Summary
        total_ac = result["ac_error_left"] + result["ac_error_right"]
        total_neumann = result["neumann_error_left"] + result["neumann_error_right"]

        print("Total boundary error:")
        print(f"  AC BC:      {total_ac:.4f}")
        print(f"  Neumann BC: {total_neumann:.4f}")

        if total_ac < total_neumann:
            print("  ==> AC BC is overall BETTER (as expected for boundary stall)")
        else:
            print("  ==> Neumann BC is overall BETTER (expected for interior stall)")
        print()

    print("=" * 70)
    print("PRACTICAL VALIDATION RESULTS (from actual MFG solves)")
    print("=" * 70)
    print("""
From validation experiments (adjoint_consistent_bc_provider_validation.py):

Boundary Stall (x_stall = 0):
  - Standard Neumann BC:     error = 2.09
  - Adjoint-Consistent BC:   error = 1.36  (1.54x BETTER)

Interior Stall (x_stall = 0.5):
  - Standard Neumann BC:     error = 1.55
  - Adjoint-Consistent BC:   error = 5.88  (3.8x WORSE)
  - Strict Adjoint Mode:     error = 0.27  (BEST - 5.7x better than Neumann)

Why does AC BC help for boundary stall despite far-boundary error?
- Density concentrates at the stall point (exponential decay away from it)
- Far boundary has very little mass, so BC errors there are weighted less
- AC BC is correct where density is HIGH (at stall boundary)
- AC BC is incorrect where density is LOW (far boundary) - but this matters less!

Why does AC BC fail for interior stall?
- BOTH boundaries are away from the stall
- Density is non-negligible at both boundaries
- AC BC imposes incorrect constraints at both boundaries
""")

    print("=" * 70)
    print("CONCLUSION: Scope Limitation, Not Design Error")
    print("=" * 70)
    print("""
The Adjoint-Consistent BC (AdjointConsistentProvider) has a LIMITED SCOPE:
- DESIGNED FOR: Boundary stall configurations (stall at x=0 or x=1)
- NOT DESIGNED FOR: Interior stall configurations

The AC BC formula is mathematically correct at the stall boundary because:
1. At stall point, optimal drift alpha* = 0 (equilibrium)
2. Zero-flux condition gives: dU/dn = -sigma^2/2 * d(ln m)/dn
3. This coupling ensures HJB and FP BCs are consistent

At NON-STALL boundaries, the derivation doesn't apply because:
1. Agents have a preferred direction (alpha* != 0)
2. The equilibrium flux is not zero
3. Imposing AC BC constrains the wrong relationship

Recommendations:
1. Use AC BC only when stall point is at boundary
2. For interior stall, use:
   a. Standard Neumann BC (simple, moderate accuracy)
   b. Strict Adjoint Mode (best, ensures L_FP = L_HJB^T)
3. Strict Adjoint Mode works universally because it enforces
   discretization consistency, not equilibrium-derived BC values

This is NOT a fundamental design error - the AC BC is a specialized tool
for a specific configuration. The documentation should clarify this scope.
""")


if __name__ == "__main__":
    main()
