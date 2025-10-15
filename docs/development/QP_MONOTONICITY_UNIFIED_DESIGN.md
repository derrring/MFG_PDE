# Unified QP Monotonicity Check Design

**Created**: 2025-10-15
**Purpose**: Single function design for monotonicity violation detection

---

## Unified Design: Single Function with Adaptive Flag

Instead of three separate methods, use **one method** with adaptive behavior controlled by flags:

```python
def _check_monotonicity_violation(
    self,
    D_coeffs: np.ndarray,
    point_idx: int,
    use_adaptive: bool = None
) -> bool:
    """
    Unified monotonicity violation check.

    Args:
        D_coeffs: Taylor derivative coefficients from unconstrained solve
        point_idx: Collocation point index
        use_adaptive: Override adaptive mode. If None, use qp_optimization_level

    Returns:
        True if QP constraints are needed

    Behavior by optimization level:
        - "none": Never called (QP disabled)
        - "basic": use_adaptive=False → strict criteria, no adaptation
        - "smart"/"tuned": use_adaptive=True → adaptive threshold to hit qp_usage_target
    """
```

---

## Implementation Strategy

### Step 1: Always compute violation criteria

Regardless of mode, compute all three criteria:

```python
# Find multi-index locations
laplacian_idx = None
gradient_idx = None
for k, beta in enumerate(self.multi_indices):
    if sum(beta) == 2:
        laplacian_idx = k
    elif sum(beta) == 1 and gradient_idx is None:
        gradient_idx = k

if laplacian_idx is None:
    return False  # Can't check without Laplacian

D_laplacian = D_coeffs[laplacian_idx]
tolerance = 1e-12

# Criterion 1: Laplacian negativity
violation_1 = (D_laplacian >= -tolerance)

# Criterion 2: Gradient boundedness
violation_2 = False
if gradient_idx is not None:
    D_gradient = D_coeffs[gradient_idx]
    sigma = getattr(self.problem, 'sigma', 1.0)
    scale_factor = 10.0 * max(sigma**2, 0.1)
    laplacian_mag = abs(D_laplacian) + 1e-10
    gradient_mag = abs(D_gradient)
    violation_2 = (gradient_mag > scale_factor * laplacian_mag)

# Criterion 3: Higher-order control
higher_order_norm = sum(
    abs(D_coeffs[k]) for k in range(len(D_coeffs))
    if sum(self.multi_indices[k]) >= 3
)
laplacian_mag = abs(D_laplacian) + 1e-10
violation_3 = (higher_order_norm > laplacian_mag)

# Basic violation check (any criterion violated)
has_violation = (violation_1 or violation_2 or violation_3)
```

### Step 2: Decide based on mode

```python
# Determine adaptive mode
if use_adaptive is None:
    use_adaptive = (self.qp_optimization_level in ["smart", "tuned"])

if not use_adaptive:
    # BASIC MODE: Strict enforcement
    return has_violation
else:
    # ADAPTIVE MODE: Threshold-based decision
    severity = self._compute_violation_severity(
        D_coeffs, laplacian_idx, gradient_idx
    )

    # Initialize adaptive state on first call
    if not hasattr(self, '_adaptive_qp_state'):
        self._init_adaptive_qp_state()

    # Adaptive threshold decision
    needs_qp = (severity > self._adaptive_qp_state['threshold'])

    # Update statistics and adapt threshold
    self._update_adaptive_threshold(needs_qp, severity)

    return needs_qp
```

---

## Complete Implementation

```python
def _init_enhanced_qp_features(self) -> None:
    """
    Initialize enhanced QP features for smart/tuned optimization levels.

    Sets up adaptive threshold state and tracking statistics.
    """
    self._adaptive_qp_state = {
        'threshold': 0.0,  # Start permissive (catch all violations)
        'qp_count': 0,
        'total_count': 0,
        'severity_history': [],
    }
    self._current_point_idx = 0
    self._current_time_ratio = 0.0
    self._current_newton_iter = 0


def _check_monotonicity_violation(
    self,
    D_coeffs: np.ndarray,
    point_idx: int,
    use_adaptive: bool = None
) -> bool:
    """
    Check if unconstrained solution violates monotonicity.

    Unified method supporting both basic (strict) and adaptive (threshold) modes.

    Args:
        D_coeffs: Taylor derivative coefficients from unconstrained solve
        point_idx: Collocation point index
        use_adaptive: Override adaptive mode. If None, use qp_optimization_level
            - False: BASIC mode - strict enforcement of all criteria
            - True: ADAPTIVE mode - threshold-based with target QP usage

    Returns:
        True if QP constraints are needed

    Mathematical Criteria (see docs/development/QP_MONOTONICITY_CRITERIA.md):
        1. Laplacian negativity: D₂ < 0
        2. Gradient boundedness: |D₁| ≤ C·σ²·|D₂|
        3. Higher-order control: Σ|Dₖ| < |D₂| for |k| ≥ 3

    Modes:
        - BASIC: Return True if ANY criterion violated
        - ADAPTIVE: Return True if violation_severity > adaptive_threshold
    """
    # Find multi-index locations
    laplacian_idx = None
    gradient_idx = None

    for k, beta in enumerate(self.multi_indices):
        if sum(beta) == 2 and all(b <= 2 for b in beta):
            if laplacian_idx is None:
                laplacian_idx = k
        elif sum(beta) == 1:
            if gradient_idx is None:
                gradient_idx = k

    if laplacian_idx is None:
        return False  # Cannot check monotonicity without Laplacian term

    # Extract coefficients
    D_laplacian = D_coeffs[laplacian_idx]
    tolerance = 1e-12
    laplacian_mag = abs(D_laplacian) + 1e-10

    # ===================================================================
    # Criterion 1: Laplacian Negativity (Diffusion Dominance)
    # ===================================================================
    # For proper elliptic structure: D₂ < 0
    violation_1 = (D_laplacian >= -tolerance)

    # ===================================================================
    # Criterion 2: Gradient Boundedness (Prevent Advection Dominance)
    # ===================================================================
    # Advection shouldn't overwhelm diffusion: |D₁| ≤ C·σ²·|D₂|
    violation_2 = False
    if gradient_idx is not None:
        D_gradient = D_coeffs[gradient_idx]
        sigma = getattr(self.problem, 'sigma', 1.0)
        scale_factor = 10.0 * max(sigma**2, 0.1)
        gradient_mag = abs(D_gradient)
        violation_2 = (gradient_mag > scale_factor * laplacian_mag)

    # ===================================================================
    # Criterion 3: Higher-Order Control (Truncation Error)
    # ===================================================================
    # Third+ derivatives should be small: Σ|Dₖ| < |D₂| for order ≥ 3
    higher_order_norm = sum(
        abs(D_coeffs[k]) for k in range(len(D_coeffs))
        if sum(self.multi_indices[k]) >= 3
    )
    violation_3 = (higher_order_norm > laplacian_mag)

    # Basic violation check (any criterion violated)
    has_violation = (violation_1 or violation_2 or violation_3)

    # ===================================================================
    # Mode Selection: Basic vs Adaptive
    # ===================================================================
    if use_adaptive is None:
        use_adaptive = (self.qp_optimization_level in ["smart", "tuned"])

    if not use_adaptive:
        # BASIC MODE: Strict enforcement of all criteria
        return has_violation

    # ===================================================================
    # ADAPTIVE MODE: Threshold-Based Decision
    # ===================================================================
    # Compute quantitative severity (0 = no violation, >0 = violation)
    severity = 0.0

    # Severity 1: How positive is Laplacian (should be negative)
    if violation_1:
        severity = max(severity, D_laplacian + tolerance)

    # Severity 2: Excess gradient relative to diffusion
    if violation_2:
        D_gradient = D_coeffs[gradient_idx]
        sigma = getattr(self.problem, 'sigma', 1.0)
        scale_factor = 10.0 * max(sigma**2, 0.1)
        gradient_mag = abs(D_gradient)
        excess_gradient = gradient_mag / laplacian_mag - scale_factor
        severity = max(severity, excess_gradient)

    # Severity 3: Excess higher-order terms
    if violation_3:
        excess_higher_order = higher_order_norm / laplacian_mag - 1.0
        severity = max(severity, excess_higher_order)

    # Initialize adaptive state on first call
    if not hasattr(self, '_adaptive_qp_state'):
        self._init_enhanced_qp_features()

    state = self._adaptive_qp_state
    threshold = state['threshold']

    # Decision: use QP if severity exceeds adaptive threshold
    needs_qp = (severity > threshold)

    # Update statistics
    state['total_count'] += 1
    state['severity_history'].append(severity)
    if needs_qp:
        state['qp_count'] += 1

    # Adapt threshold every N evaluations to approach qp_usage_target
    adaptation_interval = 100
    if state['total_count'] % adaptation_interval == 0:
        actual_usage = state['qp_count'] / state['total_count']
        target_usage = self.qp_usage_target

        # Proportional control to approach target
        if actual_usage > target_usage * 1.2:
            # Too much QP → increase threshold (be more permissive)
            state['threshold'] = max(threshold * 1.1, 1e-10)
        elif actual_usage < target_usage * 0.8:
            # Too little QP → decrease threshold (be more strict)
            state['threshold'] = threshold * 0.9

    return needs_qp
```

---

## Advantages of Unified Design

1. **Single source of truth**: Violation criteria defined once
2. **No code duplication**: Basic and adaptive share same criteria logic
3. **Clear mode switching**: `use_adaptive` flag makes behavior explicit
4. **Easier testing**: Test one function with different flags
5. **Maintainable**: Changes to criteria apply to both modes automatically

---

## Usage Examples

```python
# Basic mode (strict enforcement)
solver = HJBGFDMSolver(
    problem,
    qp_optimization_level="basic",
    use_monotone_constraints=True
)
# Internally calls: _check_monotonicity_violation(D, idx, use_adaptive=False)

# Smart mode (adaptive threshold)
solver = HJBGFDMSolver(
    problem,
    qp_optimization_level="smart",
    qp_usage_target=0.1,  # Target 10% QP usage
    use_monotone_constraints=True
)
# Internally calls: _check_monotonicity_violation(D, idx, use_adaptive=True)

# Tuned mode (same as smart, just different name)
solver = HJBGFDMSolver(
    problem,
    qp_optimization_level="tuned",
    qp_usage_target=0.05,  # Target 5% QP usage
    use_monotone_constraints=True
)
```

---

## Implementation Checklist

- [ ] Add `_init_enhanced_qp_features()` method
- [ ] Add unified `_check_monotonicity_violation()` method
- [ ] Update line 461-464 to call unified method
- [ ] Remove references to `_enhanced_check_monotonicity_violation()` (never implement)
- [ ] Add tests for both basic and adaptive modes
- [ ] Document in GFDM_MONOTONICITY_ANALYSIS.md

---

**Last Updated**: 2025-10-15
**Status**: Design approved, ready for implementation
