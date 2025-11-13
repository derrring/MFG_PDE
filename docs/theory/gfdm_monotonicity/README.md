# GFDM Monotonicity Theory

**Purpose**: Mathematical theory and implementation design for monotone Generalized Finite Difference Method (GFDM) schemes in Mean Field Games.

**Level**: Graduate/Research

---

## Overview

This subfolder contains comprehensive mathematical foundations for ensuring **monotonicity** in GFDM-based HJB solvers. Monotone schemes are essential for:

1. **Convergence to viscosity solutions** (Barles-Souganidis theory)
2. **Stability** (maximum principle preservation)
3. **Physical correctness** (no spurious oscillations)

### Key Challenge

High-order GFDM naturally produces **non-monotone stencils** (negative weights) due to polynomial reproduction requirements. We address this through:

- **Constrained optimization** (quadratic programming)
- **Direct vs indirect constraint approaches**
- **Adaptive strategies** balancing accuracy and monotonicity

---

## Document Structure

### 01. Mathematical Foundation

**File**: `01_mathematical_foundation.md`

**Content**:
- HJB equation in MFG (continuous formulation)
- Viscosity solution theory
- GFDM weight computation via weighted least-squares
- Taylor expansion and polynomial reproduction
- M-matrix property and monotone scheme theory
- Barles-Souganidis convergence theorem
- Challenges: High-order stencils, nonlinear Hamiltonians, irregular geometries

**Prerequisites**: PDEs, numerical analysis, optimization

**Key Results**:
- Monotone schemes converge to viscosity solutions
- GFDM with $p \geq 3$ violates M-matrix property
- Constrained GFDM restores monotonicity via QP

### 02. Hamiltonian Constraints (Direct Approach)

**File**: `02_hamiltonian_constraints.md`

**Content**:
- Direct enforcement of $\partial H_h / \partial u_j \geq 0$
- Collocation weight computation for gradient operators
- Comparison: Indirect (Taylor coefficients) vs Direct (Hamiltonian gradients)
- Implementation design with pseudocode
- Trade-offs: Theoretical rigor vs computational cost
- 4-phase implementation roadmap

**Prerequisites**: `01_mathematical_foundation.md`

**Key Results**:
- Direct constraints provide strict monotonicity guarantees
- Linear inequalities: $(\nabla_h u) \cdot \mathbf{c}_{j,k} \geq 0$
- Higher cost: $N$ constraints per point vs 3 (indirect)

**Status**: Theoretical design (Issue #271)

### 03. Implementation Reference (Development)

**File**: `../../development/GFDM_QP_MONOTONICITY_IMPLEMENTATION.md`

**Content**:
- Practical implementation in `hjb_gfdm.py`
- Indirect constraint implementation (IMPLEMENTED)
- QP optimization levels: `none`, `basic`, `smart`, `tuned`
- Adaptive threshold algorithm
- Usage examples and benchmarks

**Status**: Production code (lines 799-925 in `hjb_gfdm.py`)

---

## Conceptual Hierarchy

```
┌─────────────────────────────────────────────┐
│  Viscosity Solution Theory                  │
│  (Crandall-Lions, uniqueness)               │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│  Monotone Scheme Convergence                │
│  (Barles-Souganidis theorem)                │
└─────────────┬───────────────────────────────┘
              │
       ┌──────┴──────┐
       │             │
       ▼             ▼
┌──────────┐   ┌──────────────┐
│ M-Matrix │   │  Hamiltonian │
│ Property │   │  Monotonicity│
└─────┬────┘   └──────┬───────┘
      │               │
      ▼               ▼
┌──────────┐   ┌──────────────┐
│ Indirect │   │    Direct    │
│  (impl.) │   │  (proposed)  │
└──────────┘   └──────────────┘
```

---

## Mathematical Notation

### Continuous Variables

| Symbol | Description | Domain |
|:-------|:------------|:-------|
| $u(t,x)$ | Value function | $(0,T) \times \Omega$ |
| $m(t,x)$ | Density | $(0,T) \times \Omega$ |
| $H(p,m,x)$ | Hamiltonian | $\mathbb{R}^d \times \mathbb{R}_+ \times \Omega$ |
| $\Omega$ | Spatial domain | $\mathbb{R}^d$ (open, bounded) |
| $T$ | Time horizon | $\mathbb{R}_+$ |
| $\sigma$ | Diffusion coefficient | $\mathbb{R}_+$ |
| $\gamma$ | Congestion intensity | $\mathbb{R}_+$ |

### Discrete Variables

| Symbol | Description | Domain |
|:-------|:------------|:-------|
| $\mathbf{u}$ | Discrete value function | $\mathbb{R}^N$ |
| $\{x_j\}_{j=1}^N$ | Collocation points | $\Omega$ |
| $\mathcal{S}_j$ | Stencil at $x_j$ | $\{1, \ldots, N\}$ |
| $w_{j,k}$ | Laplacian weights (scalar) | $\mathbb{R}$ |
| $\mathbf{c}_{j,k}$ | Gradient weights (vector) | $\mathbb{R}^d$ |
| $\mathbf{D}_j$ | Taylor coefficients | $\mathbb{R}^M$ |

### Multi-Index Notation

| Symbol | Description |
|:-------|:------------|
| $\beta = (\beta_1, \ldots, \beta_d)$ | Multi-index |
| $\|\beta\| = \sum_{i=1}^d \beta_i$ | Total degree |
| $D^\beta u = \frac{\partial^{\|\beta\|} u}{\partial x_1^{\beta_1} \cdots \partial x_d^{\beta_d}}$ | Partial derivative |
| $\Phi_\beta(x) = x_1^{\beta_1} \cdots x_d^{\beta_d}$ | Monomial basis |

---

## Key Theorems

### Theorem 1: Barles-Souganidis Convergence

**Statement**: Let $F_h$ be a numerical scheme satisfying:
1. **Monotonicity**: $\partial F_h / \partial u_k \geq 0$ for all $k \neq j$
2. **Consistency**: $F_h(\mathbf{1}\phi) \to F(\phi)$ as $h \to 0$
3. **Stability**: $\|\mathbf{u}^n\|_\infty \leq C$ uniformly

Then $u_h \to u$ (viscosity solution) as $h \to 0$.

**Reference**: Barles & Souganidis (1991), *Asymptotic Analysis*, 4(3):271-283.

### Theorem 2: M-Matrix Monotonicity

**Statement**: A linear scheme $F_j(\mathbf{u}) = \sum_k w_{j,k} u_k$ is monotone if and only if:

$$
w_{j,j} \leq 0, \quad w_{j,k} \geq 0 \text{ for } k \neq j
$$

**Reference**: Varga (2009), *Matrix Iterative Analysis*, Springer.

### Theorem 3: Hamiltonian Monotonicity (Proposed)

**Statement**: For Hamiltonian $H_h = \frac{1+2\gamma m}{2}|\nabla_h u|^2 + V$, monotonicity requires:

$$
\left(\sum_\ell \mathbf{c}_{j,\ell} u_\ell\right) \cdot \mathbf{c}_{j,k} \geq 0, \quad \text{for all } k \neq j
$$

**Status**: Theoretical design (see `02_hamiltonian_constraints.md`)

---

## Implementation Roadmap

### Phase 1: Indirect Constraints (✅ COMPLETED)

**Status**: Production code in `hjb_gfdm.py:799-925`

**Constraints**:
1. Laplacian negativity: $D_{\beta_\Delta} < 0$
2. Gradient boundedness: $|D_{\beta_i}| \leq C\sigma^2 |D_{\beta_\Delta}|$
3. Higher-order control: $\sum_{|\beta| \geq 3} |D_\beta| < |D_{\beta_\Delta}|$

**Features**:
- QP optimization levels: `none`, `basic`, `smart`, `tuned`
- Adaptive threshold for cost control
- Target usage rate: $\alpha \in [0,1]$

### Phase 2: Direct Constraints (PROPOSED)

**Status**: Theoretical design (Issue #271)

**Requirements**:
1. Collocation weight computation/caching
2. Hamiltonian gradient constraint construction
3. Integration with existing QP pipeline
4. A/B testing framework

**Estimated Effort**: 6-10 days (see `02_hamiltonian_constraints.md`)

---

## Usage for Researchers

### Reading Order

1. **First-time readers**: Start with `01_mathematical_foundation.md`
   - Graduate-level PDE and numerical analysis background assumed
   - Self-contained mathematical derivations

2. **Implementation focus**: Jump to `../../development/GFDM_QP_MONOTONICITY_IMPLEMENTATION.md`
   - Practical examples and code references
   - Usage patterns and benchmarks

3. **Advanced theory**: Read `02_hamiltonian_constraints.md`
   - Cutting-edge approach (not yet implemented)
   - Open research questions

### Prerequisites

**Mathematics**:
- Partial differential equations (HJB, viscosity solutions)
- Numerical analysis (finite differences, convergence theory)
- Optimization (convex QP, constraint construction)

**Code**:
- `mfg_pde/alg/numerical/hjb_solvers/hjb_gfdm.py`: Main implementation
- `mfg_pde/geometry/tensor_product_grid.py`: Grid structure

---

## Open Research Questions

1. **Optimal constraint selection**: Which subset of constraints suffices for monotonicity while minimizing QP cost?

2. **Adaptive strategies**: Can we design heuristics that relax constraints where the solution is smooth?

3. **Higher dimensions**: Do Hamiltonian constraints scale gracefully to $d > 3$?

4. **Hybrid approaches**: Combine indirect (cheap) and direct (rigorous) constraints?

5. **Convergence rates**: What is the optimal trade-off between monotonicity enforcement and accuracy order?

---

## References

### Foundational Papers

1. **Barles & Souganidis (1991)**: "Convergence of approximation schemes for fully nonlinear second order equations"
   - Monotone scheme convergence theory

2. **Crandall & Lions (1983)**: "Viscosity solutions of Hamilton-Jacobi equations"
   - Viscosity solution uniqueness

3. **Oberman (2006)**: "Convergent difference schemes for degenerate elliptic and parabolic equations"
   - Wide stencil monotone schemes

### GFDM Background

4. **Benito et al. (2001)**: "Influence of several factors in the generalized finite difference method"
   - Weighted least-squares formulation

5. **Liszka & Orkisz (1980)**: "The finite difference method at arbitrary irregular grids"
   - Polynomial reproduction theory

### MFG Applications

6. **Achdou & Capuzzo-Dolcetta (2010)**: "Mean field games: numerical methods"
   - Monotone schemes for MFG systems

7. **Carlini & Silva (2014)**: "A fully discrete semi-Lagrangian scheme for a first order mean field game problem"
   - Semi-Lagrangian monotone methods

---

## Maintenance

**Last Updated**: 2025-11-13

**Maintainers**: MFG_PDE Development Team

**Status**:
- ✅ Mathematical foundation documented
- ✅ Indirect constraints implemented (production)
- 🔬 Direct constraints designed (theory)
- 📋 Open research questions identified

---

**For implementation details, see**: `docs/development/GFDM_QP_MONOTONICITY_IMPLEMENTATION.md`

**For code reference, see**: `mfg_pde/alg/numerical/hjb_solvers/hjb_gfdm.py:799-925`
