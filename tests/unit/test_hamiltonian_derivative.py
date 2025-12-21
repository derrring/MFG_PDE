"""
Quick validation of Hamiltonian derivative sign fix.

Default Hamiltonian: H = 0.5*c*|p|² - V(x) - m²
Correct derivative: dH/dm = -2m
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from mfg_pde import MFGProblem
from mfg_pde.core.derivatives import DerivativeTensors
from mfg_pde.geometry import TensorProductGrid

# Create simple problem
geometry = TensorProductGrid(dimension=1, bounds=[(0, 1)], Nx=[10])
problem = MFGProblem(geometry=geometry, T=1.0, Nt=10, diffusion=0.1)

# Test dH_dm at various m values
test_values = [0.0, 0.5, 1.0, 2.0, -1.0]

# Create dummy derivatives (not needed for dH/dm but required by API)
dummy_derivs = DerivativeTensors.from_arrays(grad=np.array([0.0]))

print("Hamiltonian Derivative Validation")
print("=" * 60)
print("Default H = 0.5*c*|p|² - V(x) - m²")
print("Expected dH/dm = -2m")
print("=" * 60)

all_correct = True
for m in test_values:
    dH_dm = problem.dH_dm(x_idx=0, m_at_x=m, derivs=dummy_derivs)
    expected = -2.0 * m
    diff = abs(dH_dm - expected)

    status = "✓" if diff < 1e-10 else "✗"
    if diff >= 1e-10:
        all_correct = False

    print(f"{status} m = {m:6.2f}  →  dH/dm = {dH_dm:8.4f}  (expected: {expected:8.4f}, error: {diff:.2e})")

print("=" * 60)
if all_correct:
    print("✓ All tests PASSED - dH/dm sign is CORRECT")
    sys.exit(0)
else:
    print("✗ Tests FAILED - dH/dm has incorrect sign or value")
    sys.exit(1)
