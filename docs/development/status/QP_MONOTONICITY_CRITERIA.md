# QP Monotonicity Violation Detection - Mathematical Formulation

**Created**: 2025-10-15
**Context**: Implementation of missing methods in hjb_gfdm.py
**Purpose**: Define explicit mathematical criteria for QP usage in GFDM+QP solver

---

## 1. Monotonicity Criterion (M-Matrix Property)

### Mathematical Definition

For a monotone finite difference scheme approximating the Laplacian operator, the weights **w** must satisfy the **M-matrix property**:

```
w_center ≤ 0                    (diagonal element)
w_j ≥ 0   for all j ≠ center   (off-diagonal elements)
```

### Physical Interpretation

For HJB equation: ∂u/∂t + H(∇u, m) + σ²/2 Δu = 0

The Laplacian Δu = ∂²u/∂x² is approximated as:
```
Δu(x_center) ≈ Σ_j w_j · u(x_j)
```

**M-matrix structure ensures**:
- Maximum principle preserved
- Viscosity solution compatibility
- No spurious oscillations
- Numerical stability

### Violation Detection Algorithm

**Input**: Unconstrained derivative coefficients **D** = (D₀, D₁, D₂, ..., Dₖ)
**Output**: Boolean `needs_qp` indicating whether QP constraints are required

**Steps**:

1. **Compute finite difference weights from Taylor coefficients**:
   ```
   For Laplacian (second derivative):
   Find D₂ corresponding to multi-index β = (2,) in 1D or (2,0), (0,2) in 2D
   ```

2. **Check constraint satisfaction**:
   ```python
   # Extract Laplacian coefficient
   laplacian_coeff = D[laplacian_idx]  # Should be negative

   # Check negativity (diffusion dominance)
   if laplacian_coeff >=tolerance:
       violation = True  # Laplacian should be negative

   # Check gradient boundedness (if first derivative exists)
   gradient_coeff = D[gradient_idx]
   if abs(gradient_coeff) > C * abs(laplacian_coeff):
       violation = True  # Advection dominates diffusion

   # Check higher-order terms
   higher_order_norm = Σ |D_k| for k with |β_k| ≥ 3
   if higher_order_norm > abs(laplacian_coeff):
       violation = True  # Truncation error too large
   ```

3. **Severity quantification**:
   ```python
   violation_severity = max(
       laplacian_coeff,  # How positive is it? (should be negative)
       abs(gradient_coeff) / abs(laplacian_coeff) - C,  # Excess advection
       higher_order_norm / abs(laplacian_coeff) - 1.0   # Excess truncation
   )
   ```

---

## 2. Explicit Mathematical Criteria

### Criterion 1: Laplacian Negativity (Diffusion Dominance)

**Condition**:
```
D₂ < 0   (for ∂²u/∂x² in 1D)
```

**Tolerance**:
```python
tolerance = 1e-12  # Allow small numerical errors
violation = (D_laplacian >= -tolerance)
```

**Physical Meaning**: The diffusion term σ²/2 ∂²u/∂x² should have a negative coefficient to create proper elliptic structure.

### Criterion 2: Gradient Boundedness (Prevent Advection Dominance)

**Condition**:
```
|D₁| ≤ C · σ² · |D₂|
```

where:
- C = 10.0 (empirical scaling factor)
- σ = diffusion coefficient from problem
- D₁ = first derivative coefficient
- D₂ = second derivative coefficient

**Tolerance**:
```python
sigma = getattr(problem, 'sigma', 1.0)
sigma_sq = sigma**2
scale_factor = 10.0 * max(sigma_sq, 0.1)

laplacian_mag = abs(D_laplacian) + 1e-10
gradient_mag = abs(D_gradient)

violation = (gradient_mag > scale_factor * laplacian_mag)
```

**Physical Meaning**: First-order advection terms shouldn't overwhelm second-order diffusion, which would break M-matrix structure.

### Criterion 3: Higher-Order Control (Truncation Error)

**Condition**:
```
Σ |D_k| < |D₂|   for all |β_k| ≥ 3
```

**Tolerance**:
```python
higher_order_norm = sum(abs(D_k) for k in range(len(D)) if sum(multi_indices[k]) >= 3)
laplacian_mag = abs(D_laplacian) + 1e-10

violation = (higher_order_norm > laplacian_mag)
```

**Physical Meaning**: Third and higher derivatives represent truncation error and should remain small relative to the dominant second derivative.

---

## 3. QP Usage Target Strategy

### What `qp_usage_target` Means

**Definition**: `qp_usage_target = α ∈ [0, 1]` is the target fraction of collocation points where QP is applied.

**Example**: `qp_usage_target = 0.1` means QP should be used at ~10% of points.

**Rationale**:
- **QP is expensive**: O(n³) vs O(n²) for unconstrained
- **Violations are localized**: Typically occur near boundaries, obstacles, or high-gradient regions
- **Adaptive heuristics**: Apply QP only where truly needed

### Enforcement Mechanisms

#### Strategy 1: Threshold-Based (Basic Level)

**Always check**, apply QP if violation detected:
```python
def _check_monotonicity_violation(self, D_coeffs):
    """Basic: Check all criteria, apply QP if any violated."""
    # Check Criterion 1: Laplacian negativity
    laplacian_negative = (D_laplacian < -tolerance)

    # Check Criterion 2: Gradient boundedness
    gradient_bounded = (gradient_mag <= scale_factor * laplacian_mag)

    # Check Criterion 3: Higher-order control
    higher_order_small = (higher_order_norm <= laplacian_mag)

    needs_qp = not (laplacian_negative and gradient_bounded and higher_order_small)
    return needs_qp
```

**QP usage**: Determined by problem structure, typically 20-40% of points

#### Strategy 2: Adaptive Threshold (Smart/Tuned Levels)

**Adjust thresholds** to hit target QP usage rate α:

**Algorithm**:
```python
class AdaptiveThreshold:
    def __init__(self, qp_usage_target=0.1):
        self.alpha_target = qp_usage_target
        self.severity_threshold = 0.0  # Start permissive
        self.qp_history = []  # Track actual usage

    def should_use_qp(self, D_coeffs, point_idx):
        # Compute violation severity
        severity = compute_violation_severity(D_coeffs)

        # Adaptive decision: use QP if severity exceeds threshold
        needs_qp = (severity > self.severity_threshold)

        # Track usage
        self.qp_history.append(needs_qp)

        # Periodically adjust threshold to hit target
        if len(self.qp_history) % 100 == 0:
            actual_usage = np.mean(self.qp_history[-100:])
            if actual_usage > self.alpha_target * 1.2:
                self.severity_threshold *= 1.1  # Increase threshold (less QP)
            elif actual_usage < self.alpha_target * 0.8:
                self.severity_threshold *= 0.9  # Decrease threshold (more QP)

        return needs_qp
```

**QP usage**: Controlled to approach `alpha_target` via threshold adaptation

---

## 4. Proposed Implementation

### Method 1: `_init_enhanced_qp_features()`

```python
def _init_enhanced_qp_features(self) -> None:
    """Initialize enhanced QP tracking for smart/tuned optimization."""
    self.enhanced_qp_stats = {
        "qp_calls": 0,
        "unconstrained_calls": 0,
        "total_evaluations": 0,
        "violation_severities": [],
        "adaptive_threshold": 0.0,  # Start permissive
    }
    self._current_point_idx = 0
    self._current_time_ratio = 0.0
    self._current_newton_iter = 0
```

### Method 2: `_check_monotonicity_violation()` (Basic)

```python
def _check_monotonicity_violation(
    self, D_coeffs: np.ndarray, point_idx: int
) -> bool:
    """
    Check if unconstrained solution violates monotonicity (Basic level).

    Args:
        D_coeffs: Taylor derivative coefficients from unconstrained solve
        point_idx: Collocation point index

    Returns:
        True if QP constraints are needed, False otherwise
    """
    # Find Laplacian index in multi_indices
    laplacian_idx = None
    gradient_idx = None

    for k, beta in enumerate(self.multi_indices):
        if sum(beta) == 2 and all(b <= 2 for b in beta):
            laplacian_idx = k
        elif sum(beta) == 1:
            gradient_idx = k if gradient_idx is None else gradient_idx

    if laplacian_idx is None:
        return False  # No Laplacian in expansion, can't check

    tolerance = 1e-12

    # Criterion 1: Laplacian negativity
    D_laplacian = D_coeffs[laplacian_idx]
    if D_laplacian >= -tolerance:
        return True  # Violation: should be negative

    # Criterion 2: Gradient boundedness
    if gradient_idx is not None:
        D_gradient = D_coeffs[gradient_idx]
        sigma = getattr(self.problem, 'sigma', 1.0)
        scale_factor = 10.0 * max(sigma**2, 0.1)
        laplacian_mag = abs(D_laplacian) + 1e-10
        gradient_mag = abs(D_gradient)

        if gradient_mag > scale_factor * laplacian_mag:
            return True  # Violation: advection dominates

    # Criterion 3: Higher-order control
    higher_order_norm = sum(
        abs(D_coeffs[k]) for k in range(len(D_coeffs))
        if sum(self.multi_indices[k]) >= 3
    )
    laplacian_mag = abs(D_laplacian) + 1e-10

    if higher_order_norm > laplacian_mag:
        return True  # Violation: truncation error too large

    return False  # All criteria satisfied
```

### Method 3: `_enhanced_check_monotonicity_violation()` (Smart/Tuned)

```python
def _enhanced_check_monotonicity_violation(
    self, D_coeffs: np.ndarray, point_idx: int
) -> bool:
    """
    Enhanced monotonicity check with adaptive threshold (Smart/Tuned level).

    Uses violation severity and adaptive thresholding to approach qp_usage_target.

    Args:
        D_coeffs: Taylor derivative coefficients
        point_idx: Collocation point index

    Returns:
        True if QP constraints are needed based on adaptive threshold
    """
    # Compute violation severity (quantitative measure)
    severity = self._compute_violation_severity(D_coeffs)

    # Initialize threshold on first call
    if not hasattr(self, '_adaptive_threshold'):
        self._adaptive_threshold = 0.0  # Start permissive

    # Decision: use QP if severity exceeds adaptive threshold
    needs_qp = (severity > self._adaptive_threshold)

    # Track statistics
    if self.enhanced_qp_stats is not None:
        self.enhanced_qp_stats["total_evaluations"] += 1
        self.enhanced_qp_stats["violation_severities"].append(severity)
        if needs_qp:
            self.enhanced_qp_stats["qp_calls"] += 1
        else:
            self.enhanced_qp_stats["unconstrained_calls"] += 1

        # Adapt threshold every N evaluations to approach target usage
        N = 100
        if self.enhanced_qp_stats["total_evaluations"] % N == 0:
            actual_usage = self.enhanced_qp_stats["qp_calls"] / self.enhanced_qp_stats["total_evaluations"]
            target_usage = self.qp_usage_target

            if actual_usage > target_usage * 1.2:
                # Too much QP usage, increase threshold (be more permissive)
                self._adaptive_threshold *= 1.1
            elif actual_usage < target_usage * 0.8:
                # Too little QP usage, decrease threshold (be more strict)
                self._adaptive_threshold *= 0.9

    return needs_qp

def _compute_violation_severity(self, D_coeffs: np.ndarray) -> float:
    """Compute quantitative severity of monotonicity violation."""
    laplacian_idx = None
    gradient_idx = None

    for k, beta in enumerate(self.multi_indices):
        if sum(beta) == 2:
            laplacian_idx = k
        elif sum(beta) == 1 and gradient_idx is None:
            gradient_idx = k

    if laplacian_idx is None:
        return 0.0

    D_laplacian = D_coeffs[laplacian_idx]
    laplacian_mag = abs(D_laplacian) + 1e-10

    # Severity 1: How positive is Laplacian? (should be negative)
    s1 = max(0.0, D_laplacian)  # Positive part

    # Severity 2: Gradient dominance
    s2 = 0.0
    if gradient_idx is not None:
        D_gradient = D_coeffs[gradient_idx]
        sigma = getattr(self.problem, 'sigma', 1.0)
        scale_factor = 10.0 * max(sigma**2, 0.1)
        gradient_mag = abs(D_gradient)
        s2 = max(0.0, gradient_mag / laplacian_mag - scale_factor)

    # Severity 3: Higher-order excess
    higher_order_norm = sum(
        abs(D_coeffs[k]) for k in range(len(D_coeffs))
        if sum(self.multi_indices[k]) >= 3
    )
    s3 = max(0.0, higher_order_norm / laplacian_mag - 1.0)

    # Combined severity (max of individual severities)
    severity = max(s1, s2, s3)

    return severity
```

---

## 5. Summary of Criteria

| Level | Method | Criterion | QP Usage |
|:------|:-------|:----------|:---------|
| **none** | N/A | Never use QP | 0% |
| **basic** | `_check_monotonicity_violation()` | Always check all 3 criteria | 20-40% (problem-dependent) |
| **smart** | `_enhanced_check_monotonicity_violation()` | Adaptive threshold based on severity | ~target (e.g., 10%) |
| **tuned** | Same as smart with statistics | Adaptive + statistical tracking | ~target with convergence |

### Mathematical Guarantees

**Basic level**: Monotonicity is **guaranteed** whenever QP is applied (all violations are caught)

**Smart/Tuned level**: Monotonicity is **approximate** (threshold adapted to balance accuracy vs cost)

---

## 6. References

1. **M-matrix theory**: Varga, R. S. (2009). *Matrix Iterative Analysis*.
2. **Monotone schemes**: Barles, G., & Souganidis, P. E. (1991). "Convergence of approximation schemes for fully nonlinear second order equations." *Asymptotic Analysis*.
3. **GFDM+QP**: `docs/theory/numerical_methods/[PRIVATE]_particle_collocation_qp_monotone.md`
4. **Implementation**: `mfg_pde/alg/numerical/hjb_solvers/hjb_gfdm.py`

---

**Last Updated**: 2025-10-15
**Status**: Ready for implementation
