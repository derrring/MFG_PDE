"""
Analyze FP system matrix to check mass conservation property

For a conservative FDM scheme, row sums should equal 1/dt
(only the time derivative contributes to the diagonal)
"""

import numpy as np
import scipy.sparse as sparse

from mfg_pde.alg.numerical.fp_solvers import fp_fdm
from mfg_pde.utils.aux_func import npart, ppart

print("=" * 70)
print("  Bug #8: FP Matrix Analysis for Mass Conservation")
print("=" * 70)
print()


# Create 1D problem for analysis (simpler to understand)
class Simple1DProblem:
    def __init__(self):
        self.Nx = 7  # 8 grid points including boundaries
        self.Dx = 0.1
        self.Dt = 0.01
        self.sigma = 0.1
        self.coupling_coefficient = 1.0
        self.Nt = 1


# Create FP solver
problem = Simple1DProblem()
fp_solver = fp_fdm.FPFDMSolver(problem)

# Build matrix for a single timestep
# We need to manually construct the matrix to inspect it
Nx = problem.Nx
Dx = problem.Dx
Dt = problem.Dt
sigma = problem.sigma
coupling_coefficient = problem.coupling_coefficient

# Create dummy U field (constant velocity for simplicity)
u_field = np.linspace(0, 1, Nx)  # Linear potential -> constant velocity

# Build matrix (copying logic from fp_fdm.py)
row_indices = []
col_indices = []
data_values = []

print(f"Building FP matrix for {Nx} interior points")
print(f"Parameters: Dx={Dx}, Dt={Dt}, sigma={sigma}, coupling_coefficient={coupling_coefficient}")
print()

# Use no-flux boundary conditions with ghost cell method (Bug #8 fix)
for i in range(Nx):
    if i == 0:
        # Left boundary with ghost cell reflection
        # Ghost: m[-1] = m[1], u[-1] = u[1] (symmetric)
        # Stencil becomes: [m[1], m[0], m[1]] so m[1] coefficient is sum of lower+upper
        val_A_ii = 1.0 / Dt + sigma**2 / Dx**2

        # Velocity gradients using ghost cell
        u_grad_forward = u_field[1] - u_field[0]
        u_grad_backward = -u_grad_forward  # Symmetric reflection

        val_A_ii += float(coupling_coefficient * (npart(u_grad_forward) + ppart(u_grad_backward)) / Dx**2)

        row_indices.append(i)
        col_indices.append(i)
        data_values.append(val_A_ii)

        # Coupling to m[1]: combines both ghost (i-1→1) and real neighbor (i+1=1)
        val_A_i_im1 = -(sigma**2) / (2 * Dx**2)
        val_A_i_im1 += float(-coupling_coefficient * npart(u_grad_backward) / Dx**2)

        val_A_i_ip1 = -(sigma**2) / (2 * Dx**2)
        val_A_i_ip1 += float(-coupling_coefficient * ppart(u_grad_forward) / Dx**2)

        # Add combined coefficient to m[1]
        row_indices.append(i)
        col_indices.append(1)
        data_values.append(val_A_i_im1 + val_A_i_ip1)

    elif i == Nx - 1:
        # Right boundary with ghost cell reflection
        # Ghost: m[N] = m[N-2], u[N] = u[N-2] (symmetric)
        # Stencil becomes: [m[N-2], m[N-1], m[N-2]] so m[N-2] coefficient is sum of lower+upper
        val_A_ii = 1.0 / Dt + sigma**2 / Dx**2

        # Velocity gradients using ghost cell
        u_grad_backward = u_field[Nx - 1] - u_field[Nx - 2]
        u_grad_forward = -u_grad_backward  # Symmetric reflection

        val_A_ii += float(coupling_coefficient * (npart(u_grad_forward) + ppart(u_grad_backward)) / Dx**2)

        row_indices.append(i)
        col_indices.append(i)
        data_values.append(val_A_ii)

        # Coupling to m[N-2]: combines both real neighbor (i-1=N-2) and ghost (i+1→N-2)
        val_A_i_im1 = -(sigma**2) / (2 * Dx**2)
        val_A_i_im1 += float(-coupling_coefficient * npart(u_grad_backward) / Dx**2)

        val_A_i_ip1 = -(sigma**2) / (2 * Dx**2)
        val_A_i_ip1 += float(-coupling_coefficient * ppart(u_grad_forward) / Dx**2)

        # Add combined coefficient to m[N-2]
        row_indices.append(i)
        col_indices.append(Nx - 2)
        data_values.append(val_A_i_im1 + val_A_i_ip1)

    else:
        # Interior points
        val_A_ii = 1.0 / Dt + sigma**2 / Dx**2
        val_A_ii += float(coupling_coefficient * (npart(u_field[i + 1] - u_field[i]) + ppart(u_field[i] - u_field[i - 1])) / Dx**2)

        row_indices.append(i)
        col_indices.append(i)
        data_values.append(val_A_ii)

        val_A_i_im1 = -(sigma**2) / (2 * Dx**2)
        val_A_i_im1 += float(-coupling_coefficient * npart(u_field[i] - u_field[i - 1]) / Dx**2)
        row_indices.append(i)
        col_indices.append(i - 1)
        data_values.append(val_A_i_im1)

        val_A_i_ip1 = -(sigma**2) / (2 * Dx**2)
        val_A_i_ip1 += float(-coupling_coefficient * ppart(u_field[i + 1] - u_field[i]) / Dx**2)
        row_indices.append(i)
        col_indices.append(i + 1)
        data_values.append(val_A_i_ip1)

A_matrix = sparse.coo_matrix((data_values, (row_indices, col_indices)), shape=(Nx, Nx)).tocsr()

# Analyze row sums
print("Matrix Analysis:")
print("-" * 60)

row_sums = A_matrix.sum(axis=1).A1  # Convert to 1D array

expected_row_sum = 1.0 / Dt

print(f"Expected row sum (1/Dt): {expected_row_sum:.6f}")
print()
print("Row  |  Sum  |  Diff from 1/Dt  |  Type")
print("-" * 60)

for i in range(Nx):
    diff = row_sums[i] - expected_row_sum
    row_type = "Boundary (left)" if i == 0 else ("Boundary (right)" if i == Nx - 1 else "Interior")
    print(f"{i:3d}  |  {row_sums[i]:8.4f}  |  {diff:+10.6f}  |  {row_type}")

print("-" * 60)
print()

# Summary statistics
print("Summary:")
print(f"  Min row sum: {np.min(row_sums):.6f}")
print(f"  Max row sum: {np.max(row_sums):.6f}")
print(f"  Mean row sum: {np.mean(row_sums):.6f}")
print(f"  Expected: {expected_row_sum:.6f}")
print()

# Check mass conservation
max_deviation = np.max(np.abs(row_sums - expected_row_sum))
print(f"  Max deviation from 1/Dt: {max_deviation:.6e}")

if max_deviation < 1e-10:
    print("  ✓ Matrix CONSERVES mass (row sums = 1/Dt)")
else:
    print("  ✗ Matrix DOES NOT conserve mass!")
    print(f"    Row sums deviate by up to {max_deviation:.6e}")
    print(f"    Relative error: {max_deviation / expected_row_sum * 100:.2f}%")

print()

# Estimate mass loss rate from matrix error
# If row sums are wrong, mass changes as: m_new = A^{-1} (m_old / Dt)
# Expected: m_new = m_old (for conservative scheme)
if max_deviation > 1e-10:
    # Solve system for uniform initial density
    m_init = np.ones(Nx)
    b_rhs = m_init / Dt

    m_next = sparse.linalg.spsolve(A_matrix, b_rhs)

    total_mass_init = np.sum(m_init) * Dx
    total_mass_next = np.sum(m_next) * Dx

    mass_change_percent = (total_mass_next - total_mass_init) / total_mass_init * 100

    print("Mass Loss Test (uniform density → 1 timestep):")
    print(f"  Initial mass: {total_mass_init:.6f}")
    print(f"  Final mass: {total_mass_next:.6f}")
    print(f"  Change: {mass_change_percent:+.2f}%")
    print()

    if abs(mass_change_percent - 12.6) < 5:  # Within 5% of observed
        print("  ⚠ This matches the observed 12.6% per-timestep loss!")
        print("  → Confirms matrix discretization is the root cause")

print()
print("=" * 70)
