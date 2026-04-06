"""Diagnose corner interference in BC enforcement.

When applying BCs sequentially, later BCs overwrite corner values set by earlier BCs.

Example for 4-boundary case (5x5 grid):
1. x_min: field[0, :] = 1.0  → corners [0,0]=1.0, [0,4]=1.0
2. x_max: field[-1, :] = 2.0 → corners [-1,0]=2.0, [-1,4]=2.0
3. y_min: field[:, 0] = 3.0  → corners [0,0]=3.0, [-1,0]=3.0  (OVERWRITES x values!)
4. y_max: field[:, -1] = 4.0 → corners [0,4]=4.0, [-1,4]=4.0  (OVERWRITES x values!)

Result: Corners have y-boundary values, not x-boundary values.
"""

import numpy as np

# Simulate 5x5 field
field = np.zeros((5, 5))

print("Sequential BC application (mimics current implementation):")
print("=" * 60)

# Apply x_min
field[0, :] = 1.0
print("\nAfter x_min (value=1.0):")
print(f"Corner [0,0] = {field[0, 0]:.1f}, Corner [0,4] = {field[0, 4]:.1f}")

# Apply x_max
field[-1, :] = 2.0
print("\nAfter x_max (value=2.0):")
print(f"Corner [-1,0] = {field[-1, 0]:.1f}, Corner [-1,4] = {field[-1, 4]:.1f}")

# Apply y_min (OVERWRITES corners!)
field[:, 0] = 3.0
print("\nAfter y_min (value=3.0) ← OVERWRITES corners!")
print(f"Corner [0,0] = {field[0, 0]:.1f} (was 1.0, now 3.0)")
print(f"Corner [-1,0] = {field[-1, 0]:.1f} (was 2.0, now 3.0)")

# Apply y_max (OVERWRITES corners!)
field[:, -1] = 4.0
print("\nAfter y_max (value=4.0) ← OVERWRITES corners!")
print(f"Corner [0,4] = {field[0, 4]:.1f} (was 1.0, now 4.0)")
print(f"Corner [-1,4] = {field[-1, 4]:.1f} (was 2.0, now 4.0)")

print("\n" + "=" * 60)
print("Final field:")
print(field)

print("\n" + "=" * 60)
print("Boundary means (what test measures):")
left_mean = field[0, :].mean()
right_mean = field[-1, :].mean()
bottom_mean = field[:, 0].mean()
top_mean = field[:, -1].mean()

print(f"Left (x=0):   {left_mean:.6f} (expected 1.0, got {left_mean:.2f} due to corners)")
print(f"Right (x=1):  {right_mean:.6f} (expected 2.0, got {right_mean:.2f} due to corners)")
print(f"Bottom (y=0): {bottom_mean:.6f} (expected 3.0) ✓")
print(f"Top (y=1):    {top_mean:.6f} (expected 4.0) ✓")

print("\n" + "=" * 60)
print("Analysis:")
print(f"Left mean = (1.0*3 + 3.0 + 4.0) / 5 = {(1.0 * 3 + 3.0 + 4.0) / 5:.2f}")
print(f"Right mean = (2.0*3 + 3.0 + 4.0) / 5 = {(2.0 * 3 + 3.0 + 4.0) / 5:.2f}")
print("\nCorner values pollute x-boundary means because corners have y-values!")
print("\nThis is Issue #521: Corner handling with conflicting BC values.")
