# Adjoint Operators and Discrete Duality in Mean Field Games

**Author**: MFG_PDE Development Team
**Date**: 2026-01-16
**Status**: Active Theory Reference
**Related**: Issue #580 (Adjoint-aware solver pairing), Issue #578 (FP SL adjoint solver)

---

## Table of Contents

1. [Mathematical Foundation](#1-mathematical-foundation)
2. [Continuous Duality in MFG](#2-continuous-duality-in-mfg)
3. [Discrete Duality: The Core Challenge](#3-discrete-duality-the-core-challenge)
4. [Classification of Numerical Schemes](#4-classification-of-numerical-schemes)
5. [Why GFDM is Not Discretely Self-Adjoint](#5-why-gfdm-is-not-discretely-self-adjoint)
6. [Practical Implications for Solver Design](#6-practical-implications-for-solver-design)
7. [References](#7-references)

---

## 1. Mathematical Foundation

### 1.1 Continuous Adjoint Operators

**Definition**: Let $L: H_1 \to H_2$ be a linear operator between Hilbert spaces. The **adjoint** $L^*: H_2 \to H_1$ satisfies:

$$\langle Lu, v \rangle_{H_2} = \langle u, L^*v \rangle_{H_1} \quad \forall u \in H_1, v \in H_2$$

**Self-adjoint operator**: $L = L^*$ (symmetric with respect to the inner product).

**Examples**:
- The Laplacian $\Delta$ is self-adjoint on $L^2(\Omega)$ with appropriate boundary conditions
- The gradient operator $\nabla: H^1 \to (L^2)^d$ has adjoint $-\text{div}: (L^2)^d \to L^2$

### 1.2 Why Adjoints Matter in PDEs

For the PDE pair:
- **Primal**: $\partial_t u + H(\nabla u) = 0$ (backward HJB)
- **Dual**: $\partial_t m - \text{div}(m \alpha) = \sigma^2 \Delta m$ (forward FP)

The **gradient** and **divergence** operators are continuous adjoints:

$$\int_\Omega \nabla u \cdot v \, dx = -\int_\Omega u \, \text{div}(v) \, dx + \text{boundary terms}$$

This relationship ensures:
- Energy conservation: $\frac{d}{dt}\int_\Omega u \cdot m \, dx = 0$ (for appropriate BC)
- Nash equilibrium: Population distribution $m$ is consistent with value function $u$
- Duality in optimization: Primal-dual formulations

---

## 2. Continuous Duality in MFG

### 2.1 The HJB-FP System

Mean Field Games are characterized by the coupled system:

**Hamilton-Jacobi-Bellman (HJB)** - backward in time:
$$-\partial_t u + H(x, \nabla u) - \frac{\sigma^2}{2}\Delta u = F(x, m)$$

**Fokker-Planck (FP)** - forward in time:
$$\partial_t m + \text{div}(m \cdot D_p H(x, \nabla u)) - \frac{\sigma^2}{2}\Delta m = 0$$

where $D_p H$ is the Hamiltonian derivative with respect to momentum $p = \nabla u$.

### 2.2 Continuous Duality Property

**Key observation**: The transport term in FP is the **adjoint** of the Hamiltonian gradient in HJB:

$$\text{div}(m \cdot D_p H) = (\nabla \cdot D_p H)^* (m)$$

This continuous duality guarantees:
1. **Mass conservation**: $\int_\Omega m(t,x) \, dx = 1$ for all $t$
2. **Energy balance**: $\frac{d}{dt}\int_\Omega u m \, dx = $ boundary flux
3. **Weak formulation consistency**: Test functions integrate correctly

---

## 3. Discrete Duality: The Core Challenge

### 3.1 What is Discrete Duality?

When we discretize the HJB and FP equations, we replace continuous operators with matrices. **Discrete duality** means:

$$\mathbf{L}_{\text{FP}} = \mathbf{L}_{\text{HJB}}^T$$

where $\mathbf{A}^T$ is the matrix transpose.

### 3.2 Why Discrete Duality Matters

**With discrete duality** (exact):
- Mass conservation: $\mathbf{1}^T \mathbf{L}_{\text{FP}} \mathbf{m} = 0$ by construction
- Discrete energy: $\mathbf{u}^T \mathbf{m}$ evolves correctly
- Numerical stability: Consistent with continuous theory
- Nash equilibrium: No artificial drift

**Without discrete duality** (approximate):
- Mass may drift over time → requires renormalization
- Energy balance violated → numerical artifacts
- Convergence issues in MFG fixed-point iteration
- Equilibrium solutions may be inconsistent

### 3.3 The $h \to 0$ Limit

All consistent discretizations converge to the correct continuous operators as grid spacing $h \to 0$. However:

**Discrete duality**: Guarantees hold at **every** $h > 0$
**Continuous duality**: Guarantees hold **only** in limit $h \to 0$

For practical computation with finite $h$, discrete duality is highly desirable.

---

## 4. Classification of Numerical Schemes

### 4.1 Schemes with Discrete Duality (True Adjoints)

#### 4.1.1 Finite Differences on Structured Grids

**Upwind FDM**:
- HJB: Forward/backward differences for $\nabla u$ depending on Hamiltonian sign
- FP: Upwind divergence using same directional bias

**Matrix structure**:
```
L_HJB = tridiag([-1/h, 1/h, 0])      (gradient-like)
L_FP  = tridiag([0, -1/h, 1/h])      (divergence-like)
L_FP  = L_HJB^T                      ✅ Exact transpose
```

**Why it works**: On uniform Cartesian grids, the discrete gradient and divergence stencils are naturally transposes.

**Code reference**: `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py`, `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py`

#### 4.1.2 Semi-Lagrangian with Interpolation-Splatting Duality

**Semi-Lagrangian (SL)**:
- HJB: Backward characteristic + **interpolation** (gather operation)
- FP: Forward characteristic + **splatting** (scatter operation)

**Adjoint relationship**:
$$\text{Interpolate}^T = \text{Splat}$$

This is the discrete analogue of $\nabla^T = -\text{div}$.

**Why it works**:
- Interpolation: $u(x^*) = \sum_i w_i u_i$ where $w_i$ are basis function weights
- Splatting: $m_i^{new} = \sum_j w_i(x_j) m_j$ using **same** weights $w_i$
- Matrix transpose relationship is exact by construction

**Code reference**: `mfg_pde/alg/numerical/hjb_solvers/hjb_semi_lagrangian.py`, `mfg_pde/alg/numerical/fp_solvers/fp_semi_lagrangian_adjoint.py`

**Important**: The backward SL solver `fp_semi_lagrangian.py` using interpolation is **NOT** the adjoint - only forward splatting (`fp_semi_lagrangian_adjoint.py`) provides true duality.

#### 4.1.3 Finite Volume Methods (FVM)

**Numerical flux approach**:
- HJB: Discrete Hamiltonian using numerical flux $\mathcal{H}_{\text{num}}$
- FP: Conservative flux $\mathcal{F} = \partial_{p} \mathcal{H}_{\text{num}} \cdot m$

**Adjoint consistency**: If numerical flux is consistent and properly linearized, the discrete operators are transposes.

**Examples**: Lax-Friedrichs, Godunov, Engquist-Osher schemes.

**Status in MFG_PDE**: Not yet implemented (future extension).

### 4.2 Schemes with Continuous Duality Only (Approximate)

#### 4.2.1 Generalized Finite Difference Method (GFDM)

**Current usage in MFG_PDE**:
- HJB: `HJBGFDMSolver` using weighted least-squares on collocation points
- FP: `FPGFDMSolver` using same framework

**Critical limitation**: $\mathbf{L}_{\text{FP}} \neq \mathbf{L}_{\text{HJB}}^T$

**Why not discrete adjoints?** See Section 5 below.

**Practical approach**: Accept continuous duality, manually enforce mass conservation via renormalization:

```python
M[t+1] = M[t] + dt * L_FP @ M[t]
M[t+1] = M[t+1] / np.sum(M[t+1])  # Renormalize
```

**Code reference**: `mfg_pde/alg/numerical/hjb_solvers/hjb_gfdm.py`, `mfg_pde/alg/numerical/fp_solvers/fp_gfdm.py`

---

## 5. Why GFDM is Not Discretely Self-Adjoint

This is a fundamental property of meshless methods that is often misunderstood.

### 5.1 The Misconception

**Common belief**: "GFDM uses symmetric weighted least-squares, so the operators should be self-adjoint."

**Why it's wrong**: The symmetry of the weighting function $W(\|x-y\|)$ does **NOT** imply symmetry of the discrete operator matrix.

### 5.2 Three Sources of Asymmetry

#### 5.2.1 Asymmetric Neighborhoods

In scattered point clouds:
- Point $x_j$ may be in the $\delta$-neighborhood of $x_i$: $\|x_j - x_i\| < \delta$
- But $x_i$ may NOT be in the neighborhood of $x_j$ if $\delta$ varies locally

**Result**: The support of basis functions is not reciprocal.

#### 5.2.2 Asymmetric Weights

Even when both points are in each other's neighborhoods, the weighted least-squares weights $w_{ij}$ and $w_{ji}$ differ:

$$w_{ij} = f(\text{local geometry around } x_i)$$
$$w_{ji} = f(\text{local geometry around } x_j)$$

**Example**: If $x_i$ has many nearby neighbors (dense region) but $x_j$ is isolated, the weights will be very different.

#### 5.2.3 Non-Symmetric Stencil Matrix

The GFDM stencil for $\partial_x u$ at point $x_i$ is computed via:

$$\mathbf{A}_i^T \mathbf{W}_i \mathbf{A}_i \mathbf{c}_i = \mathbf{A}_i^T \mathbf{W}_i \mathbf{b}$$

where:
- $\mathbf{A}_i$ = Taylor expansion matrix (geometry around $x_i$)
- $\mathbf{W}_i$ = diagonal weight matrix (local to $x_i$)
- $\mathbf{c}_i$ = stencil coefficients

**Key observation**: $\mathbf{A}_i$ and $\mathbf{W}_i$ depend on the **local** configuration at $x_i$. When assembled globally:

$$(\mathbf{L}_{\text{GFDM}})_{ij} \neq (\mathbf{L}_{\text{GFDM}})_{ji}$$

### 5.3 Adjoint GFDM: Two Approaches

#### Approach A: Discrete Adjoint (Direct Transpose)

**Method**: After assembling HJB matrix $\mathbf{L}_{\text{HJB}}$, use $\mathbf{L}_{\text{FP}} = \mathbf{L}_{\text{HJB}}^T$ directly.

**Problem**: $\mathbf{L}_{\text{HJB}}^T$ is typically **inconsistent** - it does NOT converge to the correct FP operator as $h \to 0$.

**Consequence**: Numerical artifacts (artificial aggregation, texturing) even in smooth regions.

#### Approach B: Primal-Primal (Current MFG_PDE)

**Method**: Independently discretize both HJB and FP using GFDM.

**Property**: Both converge to correct continuous operators, but $\mathbf{L}_{\text{FP}} \neq \mathbf{L}_{\text{HJB}}^T$.

**Tradeoff**: Loss of discrete mass conservation, but both operators are consistent.

**Engineering solution**: Manual renormalization after each FP time step.

#### Approach C: Weak Form GFDM (Theoretical Ideal)

**Method**: Use Petrov-Galerkin weak formulation instead of collocation.

**Property**: Integration by parts in the weak form guarantees discrete duality.

**Problem**: Extremely difficult to implement, requires advanced quadrature on scattered points.

**Status**: Research-grade, not practical for production code.

### 5.4 GFDM in Context

**GFDM is not unique** - all strong-form meshless methods (SPH, Moving Least Squares, etc.) suffer from the same asymmetry issue.

**The fundamental tradeoff**:
- Meshless methods: Geometric flexibility (irregular domains, adaptive refinement)
- Structured methods: Discrete duality guarantees

For MFG problems on regular domains, **structured methods with discrete duality are preferred** when possible.

---

## 6. Practical Implications for Solver Design

### 6.1 Solver Pairing Requirements

To ensure discrete duality, HJB and FP solvers must be **mathematically paired**:

| HJB Solver | Compatible FP Solver | Duality Type |
|------------|---------------------|--------------|
| `HJBFDMSolver` (upwind) | `FPFDMSolver` (upwind) | Discrete ✅ |
| `HJBSemiLagrangianSolver` (interp) | `FPSLAdjointSolver` (splat) | Discrete ✅ |
| `HJBGFDMSolver` | `FPGFDMSolver` | Continuous ⚠️ |

**Anti-pattern** (breaks duality):
```python
hjb = HJBSemiLagrangianSolver(problem)  # Uses interpolation
fp = FPFDMSolver(problem)                # Uses finite differences
# ❌ WRONG: No duality relationship
```

**Correct pattern** (Issue #580):
```python
from mfg_pde.factory import create_paired_solvers, NumericalScheme

hjb, fp = create_paired_solvers(NumericalScheme.SL_LINEAR, problem)
# ✅ Guaranteed discrete duality
```

### 6.2 Testing for Discrete Duality

**Test 1: Mass conservation (zero drift)**
```python
# Pure diffusion, no drift
U = np.zeros((Nt, Nx))
M = fp_solver.solve_fp_system(m_initial, drift_field=U)

masses = np.sum(M, axis=1) * dx
assert np.allclose(masses, 1.0, atol=1e-10)  # Should be exact
```

**Test 2: Energy balance**
```python
# Check discrete energy evolution
energy = np.sum(U * M, axis=1) * dx
energy_drift = np.abs(energy[-1] - energy[0])
assert energy_drift < tol  # Should be small
```

**Test 3: Matrix transpose check** (for structured grids)
```python
# Assemble discrete operators as matrices
L_hjb = hjb_solver.assemble_matrix()
L_fp = fp_solver.assemble_matrix()

assert np.allclose(L_fp, L_hjb.T)  # Should be exact transpose
```

### 6.3 When to Use Which Scheme?

**Structured domains (rectangles, simple geometries)**:
- First choice: `SL_LINEAR` or `SL_CUBIC` (high accuracy, discrete duality)
- Fallback: `FDM_UPWIND` (robust, simple, exact duality)

**Irregular/complex domains**:
- Consider: `GFDM` (geometric flexibility)
- Cost: Manual mass renormalization required
- Alternative: Use structured method with domain embedding

**High-dimensional problems** ($d \geq 3$):
- SL methods scale better than FDM
- GFDM becomes less accurate (curse of dimensionality)

### 6.4 Code References

**Discrete duality implementations**:
- FDM: `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py:250-280`
- SL adjoint: `mfg_pde/alg/numerical/fp_solvers/fp_semi_lagrangian_adjoint.py:150-200`

**Continuous duality (with renormalization)**:
- GFDM FP: `mfg_pde/alg/numerical/fp_solvers/fp_gfdm.py:180-220`

**Factory pattern** (in development):
- Solver pairing: Issue #580
- Draft: `mfg_pde/factory/solver_factory.py` (not yet committed)

---

## 7. References

### 7.1 Mathematical Theory

1. **Benamou, J.-D., & Brenier, Y.** (2000). "A computational fluid mechanics solution to the Monge-Kantorovich mass transfer problem." *Numerische Mathematik*, 84(3), 375-393.
   - Introduces the continuous duality perspective for optimal transport

2. **Carlini, E., Ferretti, R., & Russo, G.** (2005). "A weighted essentially non-oscillatory, large time-step scheme for Hamilton-Jacobi equations." *SIAM Journal on Scientific Computing*, 27(3), 1071-1091.
   - Semi-Lagrangian schemes for HJB equations

3. **Hynd, R., & Kim, H. K.** (2018). "Finite difference approximations for first-order PDEs and the G-equation." *Communications in Partial Differential Equations*, 43(10), 1471-1502.
   - Discrete duality in first-order schemes

### 7.2 Meshless Methods and Adjoint Consistency

4. **Liszka, T., & Orkisz, J.** (1980). "The finite difference method at arbitrary irregular grids and its application in applied mechanics." *Computers & Structures*, 11(1-2), 83-95.
   - Original GFDM formulation

5. **Benito, J. J., Ureña, F., & Gavete, L.** (2001). "Influence of several factors in the generalized finite difference method." *Applied Mathematical Modelling*, 25(12), 1039-1053.
   - Analysis of GFDM accuracy and consistency

6. **Trask, N., Perego, M., & Bochev, P.** (2017). "A high-order staggered meshless method for elliptic problems." *SIAM Journal on Scientific Computing*, 39(2), A479-A502.
   - Weak form approaches for discrete conservation in meshless methods

### 7.3 MFG-Specific Literature

7. **Achdou, Y., & Capuzzo-Dolcetta, I.** (2010). "Mean field games: numerical methods." *SIAM Journal on Numerical Analysis*, 48(3), 1136-1162.
   - Foundational paper on MFG numerics, establishes duality requirements

8. **Carlini, E., & Silva, F. J.** (2014). "A fully discrete Semi-Lagrangian scheme for a first order mean field game problem." *SIAM Journal on Numerical Analysis*, 52(1), 45-67.
   - Semi-Lagrangian methods for MFG

9. **Ruthotto, L., Osher, S., Li, W., Nurbekyan, L., & Fung, S. W.** (2020). "A machine learning framework for solving high-dimensional mean field game and mean field control problems." *Proceedings of the National Academy of Sciences*, 117(17), 9183-9193.
   - Modern ML approaches still require understanding of discrete duality

---

## Appendix: Quick Reference Table

| Concept | Continuous | Discrete (with duality) | Discrete (without duality) |
|---------|-----------|------------------------|---------------------------|
| **Operator form** | $\nabla$, $\text{div}$ | $\mathbf{D}$, $\mathbf{D}^T$ | $\mathbf{L}_1$, $\mathbf{L}_2$ |
| **Relationship** | $\langle \nabla u, v \rangle = -\langle u, \text{div}(v) \rangle$ | $\mathbf{L}_{\text{FP}} = \mathbf{L}_{\text{HJB}}^T$ | $\mathbf{L}_{\text{FP}} \approx \mathbf{L}_{\text{HJB}}^T$ as $h \to 0$ |
| **Mass conservation** | $\int_\Omega m = 1$ for all $t$ | $\mathbf{1}^T \mathbf{L} \mathbf{m} = 0$ exactly | Must renormalize |
| **Convergence** | N/A | $O(h^p)$ with duality | $O(h^p)$ but energy drift |
| **Examples** | PDEs on $\Omega$ | FDM, SL | GFDM, SPH |

---

**End of Document**

For questions or corrections, please open an issue on the MFG_PDE GitHub repository.
