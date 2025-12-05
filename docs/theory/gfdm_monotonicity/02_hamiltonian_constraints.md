# GFDM Hamiltonian Gradient Constraints - Mathematical Theory

**Created**: 2025-11-13
**Updated**: 2025-12-05
**Purpose**: Mathematical foundation for direct Hamiltonian gradient constraints in GFDM HJB solver
**Status**: ✅ IMPLEMENTED (Issue #271, PR #376)
**Related**: `mfg_pde/alg/numerical/hjb_solvers/hjb_gfdm.py:1608-1729`

---

## 1. Overview

This document provides the mathematical foundation for implementing **direct Hamiltonian gradient constraints** in the GFDM HJB solver as an alternative to the current indirect approach.

### Current vs Proposed Approach

| Approach | Method | Guarantees | Complexity |
|:---------|:-------|:-----------|:-----------|
| **Indirect (Current)** | Constraints on Taylor coefficients D | Approximate monotonicity | Simpler |
| **Direct (Proposed)** | Constraints on Hamiltonian gradients ∂H_h/∂u_j | Strict monotonicity | More rigorous |

**Key Insight**: Direct constraints enforce the monotone scheme property at the **Hamiltonian level**, which is the natural setting for HJB equations.

---

## 2. Mathematical Foundation

### 2.1 HJB Equation in MFG Context

The HJB equation for mean field games:

$$
-\frac{\partial u}{\partial t} = H(\nabla u, m(t,x), x) + \frac{\sigma^2}{2} \Delta u
$$

where the **Hamiltonian** for congestion games is:

$$
H(\nabla u, m, x) = \frac{1}{2}|\nabla u|^2 + \gamma m(t,x)|\nabla u|^2 + V(x)
$$

**Components**:
- $\frac{1}{2}|\nabla u|^2$: Kinetic energy (control cost)
- $\gamma m|\nabla u|^2$: Congestion cost (density-dependent)
- $V(x)$: Potential field
- $\frac{\sigma^2}{2}\Delta u$: Diffusion term

### 2.2 Monotone Scheme Requirement

A **monotone finite difference scheme** satisfies:

$$
F(x, u(x), u(y_1), ..., u(y_N)) \text{ is non-decreasing in each } u(y_i)
$$

where $F$ is the numerical scheme operator.

**Equivalent Condition (for differentiable F)**:

$$
\frac{\partial F}{\partial u_j} \geq 0 \quad \text{for all neighbors } j \neq j_0
$$

**Physical Interpretation**: Increasing the value function at a neighboring point should not decrease the update at the center point.

### 2.3 Hamiltonian Gradient Constraint

For the numerical Hamiltonian $H_h$ constructed from finite difference approximations:

$$
H_h(u_{j_0}, \{u_j\}_{j \in \mathcal{N}}) = \frac{1}{2}|\nabla_h u|^2 + \gamma m|\nabla_h u|^2 + V(x_{j_0})
$$

**Monotonicity requires**:

$$
\frac{\partial H_h}{\partial u_j} \geq 0 \quad \text{for all } j \neq j_0
$$

---

## 3. Derivation of Direct Constraints

### 3.1 Finite Difference Gradient

The GFDM approximates the gradient using collocation weights:

$$
\nabla_h u(x_{j_0}) = \sum_{j=1}^{N} \mathbf{c}_{j_0,j} u_j
$$

where $\mathbf{c}_{j_0,j} \in \mathbb{R}^d$ are **vectorial collocation weights** for the gradient operator.

**Key Property**: These weights are derived from Taylor coefficient matrix $D$ via:

$$
\mathbf{c}_{j_0} = \beta\text{-th row of } (A^T W A)^{-1} A^T W
$$

where $\beta = (1, 0, ..., 0)$ for $\partial/\partial x_1$, etc.

### 3.2 Hamiltonian Gradient Computation

Substituting the finite difference gradient into the Hamiltonian:

$$
H_h = \frac{1 + 2\gamma m}{2} \left|\sum_{j} \mathbf{c}_{j_0,j} u_j\right|^2 + V(x_{j_0})
$$

Taking the derivative with respect to $u_k$:

$$
\frac{\partial H_h}{\partial u_k} = (1 + 2\gamma m) \left(\sum_{l} \mathbf{c}_{j_0,l} u_l\right) \cdot \mathbf{c}_{j_0,k}
$$

**Geometric Interpretation**: The Hamiltonian gradient is the **inner product** of:
- The finite difference gradient $\nabla_h u$
- The collocation weight vector $\mathbf{c}_{j_0,k}$

### 3.3 Direct Monotonicity Constraint

**For monotonicity, require**:

$$
(1 + 2\gamma m) \left(\sum_{l} \mathbf{c}_{j_0,l} u_l\right) \cdot \mathbf{c}_{j_0,k} \geq 0 \quad \text{for all } k \neq j_0
$$

**Since $(1 + 2\gamma m) > 0$ (assuming $\gamma < 1/(2m)$), this simplifies to**:

$$
\left(\sum_{l} \mathbf{c}_{j_0,l} u_l\right) \cdot \mathbf{c}_{j_0,k} \geq 0 \quad \text{for all } k \neq j_0
$$

**This is a LINEAR constraint on the collocation weights** $\mathbf{c}_{j_0,k}$.

---

## 4. Comparison with Indirect Approach

### 4.1 Current Indirect Constraints (Implemented)

The current implementation (lines 1392-1460 in `hjb_gfdm.py`) enforces:

**Constraint 1: Diffusion Dominance**
$$
D_{\text{laplacian}} < 0
$$

**Constraint 2: Gradient Boundedness**
$$
|D_{\text{gradient}}| \leq C \sigma^2 |D_{\text{laplacian}}|
$$

**Constraint 3: Truncation Error Control**
$$
\sum_{|\beta| \geq 3} |D_{\beta}| < |D_{\text{laplacian}}|
$$

**Nature**: These are constraints on the **Taylor derivative coefficients** $D$, which indirectly promote monotonicity.

### 4.2 Direct Hamiltonian Constraints (Proposed)

**Constraint**:
$$
\left(\sum_{l} \mathbf{c}_{j_0,l} u_l\right) \cdot \mathbf{c}_{j_0,k} \geq 0 \quad \text{for all neighbors } k
$$

**Nature**: This is a constraint on the **collocation weights** (equivalently, the finite difference stencil), which **directly enforces** the monotone scheme property at the Hamiltonian level.

### 4.3 Trade-offs

| Aspect | Indirect (Current) | Direct (Proposed) |
|:-------|:------------------|:-----------------|
| **Theoretical Guarantee** | Approximate | Strict monotonicity |
| **Implementation Complexity** | Simpler (3 scalar constraints) | More complex (N constraints, geometry-dependent) |
| **Computational Cost** | Lower (fewer QP constraints) | Higher (more constraints) |
| **Problem Dependence** | Heuristic thresholds ($C$, etc.) | Problem-specific ($\mathbf{c}$ depends on stencil) |
| **Dimensionality** | Scales to arbitrary D | Scales to arbitrary D |
| **Convergence** | Empirically robust | Theoretically rigorous |

---

## 5. Implementation Design

### 5.1 Computing Collocation Weights

The collocation weights $\mathbf{c}_{j_0,k}$ are **already available** during GFDM construction:

**Step 1**: Build polynomial basis matrix $A$ at stencil points
$$
A_{ij} = \Phi_j(x_i - x_{j_0})
$$

**Step 2**: Solve weighted least-squares for gradient operator
$$
\mathbf{c}_{j_0} = (A^T W A)^{-1} A^T W \quad \text{(gradient row)}
$$

**Storage**: Each collocation point needs $N \times d$ weights (N neighbors, d dimensions).

### 5.2 Constraint Construction

For each collocation point $j_0$:

**Input**:
- Current iterate $u^{(n)}$
- Collocation weights $\{\mathbf{c}_{j_0,k}\}_{k=1}^{N}$
- Density value $m(x_{j_0})$

**Compute**:
1. Finite difference gradient:
$$
\nabla_h u = \sum_{k=1}^{N} \mathbf{c}_{j_0,k} u_k
$$

2. For each neighbor $k \neq j_0$, add constraint:
$$
(\nabla_h u) \cdot \mathbf{c}_{j_0,k} \geq 0
$$

**QP Form**: These are **linear inequality constraints** compatible with standard QP solvers.

### 5.3 Pseudocode

```python
def _construct_hamiltonian_gradient_constraints(
    self,
    point_idx: int,
    u_current: np.ndarray,
    m_current: np.ndarray
) -> list[tuple[np.ndarray, float]]:
    """
    Construct direct Hamiltonian gradient constraints.

    Returns:
        List of (G, h) pairs where constraint is G @ D >= h
    """
    constraints = []

    # Get collocation weights for gradient
    c_weights = self._get_collocation_weights_gradient(point_idx)
    # c_weights[k] is a d-dimensional vector

    # Compute current finite difference gradient
    grad_h_u = np.sum([c_weights[k] * u_current[k]
                       for k in range(self.stencil_size)], axis=0)

    # For each neighbor (exclude center)
    for k in range(self.stencil_size):
        if k == point_idx:
            continue

        # Constraint: (grad_h u) · c_k >= 0
        # In QP form: c_k^T @ grad_h >= 0
        # Since grad_h depends on D through weights...
        # (This needs careful derivation of D dependence)

        G_k = # ... map c_k to Taylor coefficient space
        h_k = 0.0

        constraints.append((G_k, h_k))

    return constraints
```

### 5.4 Integration Points

**Where to add** (in `hjb_gfdm.py`):

1. **Line 1370**: Replace TODO with call to `_construct_hamiltonian_gradient_constraints()`

2. **New method** (~50-100 lines):
   - Compute collocation weights if not cached
   - Build constraints for each neighbor
   - Return list of (G, h) pairs

3. **QP solver call** (line 500-530):
   - Merge Hamiltonian constraints with existing constraints
   - Pass to `cvxpy.Problem`

---

## 6. Expected Benefits

### 6.1 Theoretical

✅ **Strict Monotonicity**: Direct enforcement of $\partial H_h / \partial u_j \geq 0$

✅ **Viscosity Consistency**: Aligns with monotone scheme theory for viscosity solutions

✅ **Convergence Guarantee**: Monotone schemes converge to viscosity solution under standard regularity

### 6.2 Practical

✅ **Robustness**: Better convergence for highly nonlinear problems (large $\gamma$, sharp obstacles)

✅ **Stability**: Reduced oscillations near shocks and discontinuities

⚠️ **Cost**: More QP constraints (~10-20 per point vs 3 currently)

⚠️ **Complexity**: Requires storing/computing collocation weights explicitly

---

## 7. Implementation Roadmap

### Phase 1: Prototype ✅ COMPLETED
- [x] Extract collocation weight computation from GFDM
- [x] Implement `_build_hamiltonian_gradient_constraints()` (lines 1608-1729)
- [x] Unit test weight computation against known stencils

### Phase 2: Constraint Construction ✅ COMPLETED
- [x] Implement `_build_hamiltonian_gradient_constraints()` method
- [x] Map collocation weights to Taylor coefficient space via direction vectors
- [x] Test constraint generation on 1D/2D problems (5 unit tests)

### Phase 3: Integration ✅ COMPLETED
- [x] Integrate with existing QP solver pipeline (lines 874-896)
- [x] Add toggle: `qp_constraint_mode="indirect"|"hamiltonian"`
- [ ] Benchmark against indirect approach (future work)

### Phase 4: Validation (Future Work)
- [ ] Convergence tests on standard MFG problems
- [ ] Compare monotonicity guarantees (direct vs indirect)
- [ ] Performance profiling (QP solve times)

**Implementation completed**: 2025-12-05 (PR #376)

---

## 8. Open Questions

1. **Constraint Relaxation**: Should we allow slack variables for infeasible cases?
   - Current indirect approach uses adaptive thresholds
   - Direct approach is strict: may fail if geometry is poor

2. **Collocation Weight Caching**: Store for all points or compute on-demand?
   - Storage: $O(N_{\text{total}} \times N_{\text{stencil}} \times d)$
   - Recomputation cost: $O(N_{\text{stencil}}^3)$ per point

3. **Hybrid Strategy**: Combine direct + indirect?
   - Use direct where geometry permits
   - Fall back to indirect for poorly-conditioned stencils

4. **Higher Dimensions**: Does constraint count become prohibitive for d > 3?
   - Stencil size grows as $O(d^p)$ for polynomial degree $p$
   - May need adaptive constraint selection

---

## 9. References

### Theoretical Background

1. **Barles & Souganidis (1991)**: "Convergence of approximation schemes for fully nonlinear second order equations"
   - Monotone scheme convergence theory

2. **Oberman (2006)**: "Convergent difference schemes for degenerate elliptic and parabolic equations"
   - Wide stencil monotone schemes

3. **Froese & Oberman (2011)**: "Convergent filtered schemes for the Monge-Ampère partial differential equation"
   - Monotonicity in high dimensions

### GFDM Background

4. **Benito et al. (2001)**: "Influence of several factors in the generalized finite difference method"
   - Collocation weight computation

5. **Liszka & Orkisz (1980)**: "The finite difference method at arbitrary irregular grids"
   - Least-squares derivative approximation

### MFG Background

6. **Achdou & Capuzzo-Dolcetta (2010)**: "Mean field games: numerical methods"
   - Monotone schemes for MFG systems

---

## 10. Summary

### Current State
The GFDM HJB solver uses **indirect constraints** on Taylor coefficients that approximately enforce monotonicity. This approach is simpler and empirically robust.

### Proposed Enhancement
Implement **direct Hamiltonian gradient constraints** that strictly enforce $\partial H_h / \partial u_j \geq 0$. This provides:
- Stronger theoretical guarantees
- Better alignment with monotone scheme theory
- Potential robustness improvements for challenging problems

### Trade-off
Direct constraints are more rigorous but also more complex (more QP constraints, collocation weight computation/storage).

### Recommendation
Implement as **optional feature** with toggle `use_hamiltonian_constraints=True/False`:
- Allows A/B comparison on benchmark problems
- Provides flexibility for users (simple vs rigorous)
- Enables future hybrid strategies

---

**Last Updated**: 2025-12-05
**Status**: ✅ IMPLEMENTED (Issue #271, PR #376)
**Code Reference**: `mfg_pde/alg/numerical/hjb_solvers/hjb_gfdm.py:1608-1729`
