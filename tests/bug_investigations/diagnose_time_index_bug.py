"""
Diagnostic: Check if time-index mismatch in M-coupling causes oscillations.

Bug hypothesis: In base_hjb.py:763, when solving HJB at time t_n, the code uses:
    M_n_plus_1_prev_picard = M_density_from_prev_picard[n_idx_hjb + 1, :]

This is M(t_{n+1}), not M(t_n)!

The HJB equation should use M(t_n) because:
    ∂u/∂t + H(x, ∇u(t,x), m(t,x)) = 0
The Hamiltonian at time t depends on m at the SAME time.

Test: Print out M values at different times to see if they differ significantly.
"""

from benchmarks.validation.test_2d_crowd_motion import CrowdMotion2D
from mfg_pde.factory import create_basic_solver

print("=" * 70)
print("  Diagnostic: Time-Index Mismatch in M-Coupling")
print("=" * 70)
print()
print("Testing on simple 8×8 grid with 10 timesteps")
print()

problem = CrowdMotion2D(
    grid_resolution=8,
    time_horizon=0.4,
    num_timesteps=10,
)

# Run ONE Picard iteration to get M evolution
solver = create_basic_solver(
    problem,
    damping=0.6,
    max_iterations=1,  # Just one iteration
    tolerance=1e-4,
)

result = solver.solve()

print("M values at different times (center point):")
print()
center_x, center_y = 4, 4  # Center of 8×8 grid

Nt = result.M.shape[0]
print(f"Time evolution of M[t, {center_x}, {center_y}]:")
print()

for t_idx in range(Nt):
    m_val = result.M[t_idx, center_x, center_y]
    t = problem.T * t_idx / (Nt - 1)
    print(f"  t_idx={t_idx:2d} (t={t:.3f}): M = {m_val:.6f}")

# Check differences
print()
print("Differences M(t_{n+1}) - M(t_n):")
print()

for t_idx in range(Nt - 1):
    m_n = result.M[t_idx, center_x, center_y]
    m_n_plus_1 = result.M[t_idx + 1, center_x, center_y]
    diff = abs(m_n_plus_1 - m_n)
    rel_diff = diff / (abs(m_n) + 1e-12) * 100
    print(f"  t_idx={t_idx:2d}: |M[{t_idx + 1}] - M[{t_idx}]| = {diff:.6f} ({rel_diff:.1f}%)")

print()
print("Analysis:")
print()

max_rel_diff = 0
for t_idx in range(Nt - 1):
    m_n = result.M[t_idx, center_x, center_y]
    m_n_plus_1 = result.M[t_idx + 1, center_x, center_y]
    diff = abs(m_n_plus_1 - m_n)
    rel_diff = diff / (abs(m_n) + 1e-12) * 100
    max_rel_diff = max(max_rel_diff, rel_diff)

if max_rel_diff > 10:
    print(f"✗ SIGNIFICANT DIFFERENCES: M changes by up to {max_rel_diff:.1f}% between timesteps")
    print("  Using M(t_{n+1}) instead of M(t_n) introduces significant error")
    print("  This could cause Picard oscillations!")
else:
    print(f"✓ SMALL DIFFERENCES: M changes by at most {max_rel_diff:.1f}% between timesteps")
    print("  Time-index mismatch is unlikely to be the root cause")

print()
print("=" * 70)
