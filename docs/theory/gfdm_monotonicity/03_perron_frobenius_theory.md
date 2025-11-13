# Perron-Frobenius Theory for Monotone Schemes

**Created**: 2025-11-13
**Purpose**: Spectral theory foundation for M-matrix monotonicity and iterative convergence
**Level**: Graduate/Research
**Prerequisites**: Linear algebra, spectral theory, operator theory

---

## 1. Introduction

Perron-Frobenius theory provides the **spectral foundation** for understanding why M-matrix schemes converge and preserve monotonicity. This document establishes the connection between:

1. **Matrix structure** (M-matrices, Z-matrices, non-negative matrices)
2. **Spectral properties** (dominant eigenvalues, spectral radius)
3. **Monotonicity** (maximum principles, comparison theorems)
4. **Iterative convergence** (fixed-point iterations, iterative solvers)

### Relevance to GFDM

In GFDM-based HJB solvers:
- **Laplacian discretization** → M-matrix structure
- **Spectral radius** < 1 → Iterative convergence
- **Dominant eigenvector** → Maximum principle
- **Non-negative matrices** → Monotone evolution

---

## 2. Matrix Classifications

### 2.1 Non-Negative Matrices

**Definition**: $A = (a_{ij}) \in \mathbb{R}^{n \times n}$ is **non-negative** if:

$$
a_{ij} \geq 0 \quad \text{for all } i, j
$$

**Notation**: $A \geq 0$

**Properties**:
- Preserves cone: $Ax \geq 0$ whenever $x \geq 0$
- Represents monotone operators
- Relevant for Fokker-Planck equations

### 2.2 Z-Matrices

**Definition**: $A$ is a **Z-matrix** if:

$$
a_{ij} \leq 0 \quad \text{for all } i \neq j
$$

**Canonical form**:

$$
A = \alpha I - B, \quad B \geq 0, \quad \alpha \in \mathbb{R}
$$

**Physical interpretation**:
- Diagonal: Self-interaction (accumulation)
- Off-diagonal: Neighbor interaction (diffusion to neighbors)

**Examples**:
- Laplacian discretization
- Diffusion operators
- HJB viscosity terms

### 2.3 M-Matrices

**Definition**: $A$ is an **M-matrix** if:

1. $A$ is a Z-matrix ($a_{ij} \leq 0$ for $i \neq j$)
2. $A$ is **invertible** with $A^{-1} \geq 0$

**Equivalent conditions** (any of these suffices):

1. $A = \alpha I - B$ with $B \geq 0$ and $\rho(B) < \alpha$ (spectral radius condition)
2. All principal minors of $A$ are positive
3. $A$ has **positive diagonal dominance**: $a_{ii} > \sum_{j \neq i} |a_{ij}|$ for all $i$
4. There exists $x > 0$ with $Ax > 0$ (strong maximum principle)

**Properties**:
- **Monotone**: $Ax \geq 0 \implies x \geq 0$
- **Maximum principle**: Solution of $Au = f$ satisfies max/min bounds
- **Convergence**: Iterative methods converge

**Relevance to GFDM**: The M-matrix structure ensures monotone finite difference schemes converge to viscosity solutions.

---

## 3. Perron-Frobenius Theorem (Classical)

### 3.1 Statement for Non-Negative Matrices

**Theorem** (Perron-Frobenius, 1907-1912): Let $A \geq 0$ be a **non-negative irreducible matrix**. Then:

1. **Dominant eigenvalue** $\lambda_1 = \rho(A)$ is **positive** and **simple** (algebraic multiplicity 1)

2. **Dominant eigenvector** $v_1 > 0$ is **strictly positive**

3. **Spectral separation**: For all other eigenvalues $\lambda_k$:
   $$
   |\lambda_k| < \lambda_1 = \rho(A)
   $$

4. **Uniqueness**: $\lambda_1$ is the **only** eigenvalue with a non-negative eigenvector

### 3.2 Definitions

**Irreducible matrix**: $A$ is irreducible if its directed graph is **strongly connected** (can reach any node from any node).

**Spectral radius**:

$$
\rho(A) = \max_k |\lambda_k(A)|
$$

**Reducible decomposition**: If reducible, $A$ can be permuted to block upper-triangular form:

$$
PAP^T = \begin{pmatrix}
A_{11} & A_{12} \\
0 & A_{22}
\end{pmatrix}
$$

### 3.3 Geometric Interpretation

**Dominant eigenvector** $v_1 > 0$ defines the **long-term behavior**:

$$
A^k x \approx \lambda_1^k (v_1^T x) v_1 \quad \text{as } k \to \infty
$$

for any $x$ with $v_1^T x > 0$.

**Physical meaning**:
- $v_1$: Equilibrium distribution
- $\lambda_1$: Growth/decay rate
- $\lambda_1 < 1$: Asymptotic decay to zero
- $\lambda_1 = 1$: Steady state
- $\lambda_1 > 1$: Exponential growth

---

## 4. M-Matrix Spectral Properties

### 4.1 Spectral Characterization

**Theorem** (M-matrix spectral bounds): Let $A = \alpha I - B$ with $B \geq 0$ be an M-matrix. Then:

1. **All eigenvalues** have **positive real part**: $\text{Re}(\lambda_k) > 0$ for all $k$

2. **Dominant eigenvalue** is **real and minimum**:
   $$
   \lambda_{\min}(A) = \alpha - \rho(B) > 0
   $$

3. **Condition number** bounded by:
   $$
   \kappa(A) = \frac{\lambda_{\max}(A)}{\lambda_{\min}(A)}
   $$

### 4.2 Inverse Properties

**Theorem** (M-matrix inverse): If $A$ is an M-matrix, then:

1. $A^{-1} \geq 0$ (non-negative inverse)

2. **Maximum principle**: For $Au = f$ with $f \geq 0$, then $u = A^{-1}f \geq 0$

3. **Comparison theorem**: If $f_1 \geq f_2$, then $u_1 = A^{-1}f_1 \geq u_2 = A^{-1}f_2$

**Proof sketch**: Write $A = \alpha I - B$ with $\rho(B) < \alpha$. Then:

$$
A^{-1} = \frac{1}{\alpha}(I - \tfrac{1}{\alpha}B)^{-1} = \frac{1}{\alpha}\sum_{k=0}^\infty \left(\frac{B}{\alpha}\right)^k \geq 0
$$

since $B \geq 0$ and series converges for $\rho(B) < \alpha$.

---

## 5. Iterative Methods and Convergence

### 5.1 Fixed-Point Iteration

Consider the linear system $Au = f$ with $A$ an M-matrix. Write:

$$
A = D - L - U
$$

where:
- $D = \text{diag}(a_{11}, \ldots, a_{nn})$ (diagonal)
- $-L$: Strict lower triangular
- $-U$: Strict upper triangular

**Jacobi iteration**:

$$
u^{(k+1)} = D^{-1}(L + U)u^{(k)} + D^{-1}f = T_J u^{(k)} + c_J
$$

where $T_J = D^{-1}(L + U)$ is the **Jacobi iteration matrix**.

**Gauss-Seidel iteration**:

$$
u^{(k+1)} = (D - L)^{-1}U u^{(k)} + (D - L)^{-1}f = T_{GS} u^{(k)} + c_{GS}
$$

where $T_{GS} = (D - L)^{-1}U$.

### 5.2 Convergence Theorem

**Theorem** (M-matrix iterative convergence): Let $A$ be an M-matrix. Then:

1. **Jacobi converges** if and only if $\rho(T_J) < 1$

2. **Gauss-Seidel converges** if and only if $\rho(T_{GS}) < 1$

3. If $A$ has **diagonal dominance** ($a_{ii} > \sum_{j \neq i} |a_{ij}|$), then:
   $$
   \rho(T_J) < 1 \quad \text{and} \quad \rho(T_{GS}) < \rho(T_J)
   $$

**Convergence rate**:

$$
\|u^{(k)} - u^*\| \leq \rho(T)^k \|u^{(0)} - u^*\|
$$

where $T \in \{T_J, T_{GS}\}$ and $u^* = A^{-1}f$.

### 5.3 Optimal Relaxation

**Successive Over-Relaxation (SOR)**:

$$
u^{(k+1)} = (D - \omega L)^{-1}[(1-\omega)D + \omega U]u^{(k)} + \omega(D - \omega L)^{-1}f
$$

where $\omega \in (0, 2)$ is the **relaxation parameter**.

**Optimal $\omega$** minimizes $\rho(T_{SOR}(\omega))$.

---

## 6. Application to GFDM Monotonicity

### 6.1 Laplacian M-Matrix Structure

For the **1D Laplacian** on uniform grid with $\Delta x = h$:

$$
\Delta_h u_j = \frac{u_{j-1} - 2u_j + u_{j+1}}{h^2}
$$

**Discrete Laplacian matrix**:

$$
L_h = \frac{1}{h^2}\begin{pmatrix}
-2 & 1 & & \\
1 & -2 & 1 & \\
& \ddots & \ddots & \ddots \\
& & 1 & -2
\end{pmatrix}
$$

**M-matrix form**: $A = -L_h$ (note sign flip for HJB)

$$
A = \frac{1}{h^2}\begin{pmatrix}
2 & -1 & & \\
-1 & 2 & -1 & \\
& \ddots & \ddots & \ddots \\
& & -1 & 2
\end{pmatrix}
$$

**Properties**:
- $a_{ii} = 2/h^2 > 0$ (positive diagonal)
- $a_{ij} = -1/h^2 < 0$ for $|i - j| = 1$ (negative off-diagonal)
- Row sum: $a_{ii} + a_{i,i-1} + a_{i,i+1} = 0$ (consistency)
- Irreducible (tridiagonal with non-zero sub/super diagonals)

**Eigenvalues** (known analytically):

$$
\lambda_k = \frac{2}{h^2}\left(1 - \cos\frac{k\pi}{n+1}\right), \quad k = 1, \ldots, n
$$

**Spectral bounds**:

$$
\lambda_{\min} = \frac{2}{h^2}\left(1 - \cos\frac{\pi}{n+1}\right) \approx \frac{\pi^2}{(n+1)^2 h^2} = \frac{\pi^2}{L^2}
$$

$$
\lambda_{\max} = \frac{2}{h^2}\left(1 - \cos\frac{n\pi}{n+1}\right) \approx \frac{4}{h^2}
$$

**Condition number**:

$$
\kappa(A) = \frac{\lambda_{\max}}{\lambda_{\min}} \approx \frac{4L^2}{\pi^2 h^2} = O(h^{-2})
$$

This **quadratic growth** in condition number is typical for second-order elliptic operators.

### 6.2 GFDM Weights and M-Matrix

For **general GFDM** with stencil $\mathcal{S}_j = \{j_1, \ldots, j_N\}$ and weights $\{w_{j,k}\}$:

$$
\Delta_h u_j = \sum_{k \in \mathcal{S}_j} w_{j,k} u_k
$$

**M-matrix requirements**:

1. **Diagonal negative**: $w_{j,j} < 0$

2. **Off-diagonal non-negative**: $w_{j,k} \geq 0$ for $k \neq j$

3. **Row sum zero** (consistency): $\sum_{k \in \mathcal{S}_j} w_{j,k} = 0$

**Why GFDM violates M-matrix**: High-order polynomial reproduction ($p \geq 3$) introduces **negative weights** at far neighbors to cancel higher-order truncation errors.

**Constrained GFDM solution**: Enforce $w_{j,k} \geq 0$ via quadratic programming (see `01_mathematical_foundation.md`, Section 7).

### 6.3 Iterative HJB Solvers

For **time-implicit HJB discretization**:

$$
\frac{u^{n+1} - u^n}{\Delta t} = F(u^{n+1}, m^n)
$$

**Linearized iteration** (Newton or fixed-point):

$$
(I - \Delta t J_F)u^{n+1,(k+1)} = u^n + \Delta t F(u^{n+1,(k)}, m^n) - \Delta t J_F u^{n+1,(k)}
$$

where $J_F$ is the Jacobian of $F$.

**Convergence condition** (Perron-Frobenius):

$$
\rho(I - \Delta t J_F) < 1 \iff \text{Re}(\lambda_{\min}(J_F)) > 0
$$

**M-matrix structure** ensures $\lambda_{\min}(J_F) > 0$, guaranteeing iterative convergence.

---

## 7. Maximum Principles

### 7.1 Discrete Maximum Principle

**Theorem** (Discrete maximum principle): Let $A$ be an M-matrix and $Au = f$. Then:

1. **Non-negative principle**: If $f \geq 0$, then $u = A^{-1}f \geq 0$

2. **Maximum principle**:
   $$
   \max_i u_i \leq \max\left(\max_i \frac{f_i}{a_{ii}}, 0\right)
   $$

3. **Minimum principle**:
   $$
   \min_i u_i \geq \min\left(\min_i \frac{f_i}{a_{ii}}, 0\right)
   $$

**Proof**: Uses $A^{-1} \geq 0$ and row sum properties.

### 7.2 Comparison Theorem

**Theorem** (Comparison theorem): Let $A$ be an M-matrix. If $u, v$ satisfy:

$$
Au \geq Av
$$

then $u \geq v$ (componentwise).

**Proof**: Write $w = u - v$. Then $Aw \geq 0$, so $w = A^{-1}(Aw) \geq 0$ by non-negative inverse.

**Application to HJB**: If $u_1, u_2$ are supersolutions ($Au_i \geq f$), then $\max(u_1, u_2)$ is also a supersolution. This justifies **value function comparison** in optimal control.

---

## 8. Stability Analysis

### 8.1 Von Neumann Stability

For **linear time-dependent PDE** $\partial_t u = L u$ discretized as:

$$
\frac{u^{n+1} - u^n}{\Delta t} = L_h u^{n+1}
$$

**Amplification factor**:

$$
u^{n+1} = (I - \Delta t L_h)^{-1} u^n = G u^n
$$

where $G = (I - \Delta t L_h)^{-1}$ is the **amplification matrix**.

**Stability condition** (von Neumann):

$$
\rho(G) \leq 1
$$

**For M-matrix $A = -L_h$**:

$$
G = (I + \Delta t A)^{-1}
$$

**Eigenvalue analysis**:

$$
\mu_k(G) = \frac{1}{1 + \Delta t \lambda_k(A)}
$$

Since $\lambda_k(A) > 0$ for all $k$ (M-matrix), we have $0 < \mu_k < 1$, ensuring **unconditional stability**.

### 8.2 CFL Condition

For **explicit scheme** $u^{n+1} = (I + \Delta t L_h)u^n$:

**Stability requires**:

$$
\rho(I + \Delta t L_h) < 1 \iff -1 < -\Delta t \lambda_{\max}(A) < 1
$$

$$
\Delta t < \frac{2}{\lambda_{\max}(A)} \approx \frac{h^2}{2\sigma^2}
$$

This is the **CFL condition** for explicit diffusion.

**M-matrix implicit schemes** avoid this restriction, allowing $\Delta t$ independent of $h$.

---

## 9. Connection to Monotone Schemes

### 9.1 Monotone Operator Definition

**Definition**: An operator $F : \mathbb{R}^n \to \mathbb{R}^n$ is **monotone** if:

$$
u \geq v \implies F(u) \geq F(v) \quad \text{(componentwise)}
$$

### 9.2 M-Matrix as Monotone Operator

**Theorem**: If $A$ is an M-matrix, then $F(u) = Au$ is a **monotone operator**.

**Proof**: Let $u \geq v$. Then $w = u - v \geq 0$. We have:

$$
F(u) - F(v) = A(u - v) = Aw \geq 0
$$

since $A$ has $A^{-1} \geq 0$, which implies $Aw \geq 0$ whenever $w \geq 0$ (M-matrix property).

### 9.3 Barles-Souganidis Convergence

**Theorem** (Barles-Souganidis, 1991): A **monotone, consistent, stable** numerical scheme converges to the unique viscosity solution.

**Requirements**:
1. **Monotonicity**: $F_h$ non-decreasing in all neighbor values
2. **Consistency**: $F_h \to F$ as $h \to 0$ on smooth functions
3. **Stability**: Uniform bounds on discrete solutions

**M-matrix structure ensures monotonicity** → Convergence to viscosity solution.

---

## 10. Summary and Cross-References

### Key Results

1. **Perron-Frobenius theorem**: Non-negative irreducible matrices have positive dominant eigenvalue with positive eigenvector

2. **M-matrices**: Z-matrices with non-negative inverse, ensuring maximum principles

3. **Spectral properties**: M-matrices have positive real eigenvalues, guaranteeing iterative convergence

4. **Monotone operators**: M-matrix structure implies monotonicity, leading to viscosity solution convergence

### Cross-References

**Within GFDM monotonicity series**:
- See `01_mathematical_foundation.md`: M-matrix definition and monotone schemes
- See `02_hamiltonian_constraints.md`: Application to HJB equations
- See `README.md`: Overview and theorem statements

**Related theory**:
- `docs/theory/viscosity_solutions/`: Viscosity solution theory (if exists)
- `docs/theory/numerical_methods/`: General finite difference methods (if exists)

### Connections to Implementation

**Code references**:
- `mfg_pde/alg/numerical/hjb_solvers/hjb_gfdm.py:799-925`: M-matrix constraint enforcement
- `mfg_pde/alg/numerical/hjb_solvers/hjb_gfdm.py:1399-1460`: Monotonicity checks

---

## References

### Perron-Frobenius Theory

1. **Perron, O. (1907)**: "Zur Theorie der Matrices", *Mathematische Annalen*, 64:248-263.
   - Original Perron theorem for positive matrices

2. **Frobenius, G. (1912)**: "Über Matrizen aus nicht negativen Elementen", *Sitzungsberichte der Königlich Preussischen Akademie der Wissenschaften*, 456-477.
   - Extension to non-negative irreducible matrices

3. **Berman & Plemmons (1994)**: *Nonnegative Matrices in the Mathematical Sciences*, SIAM.
   - Comprehensive reference on non-negative matrices and M-matrices

### M-Matrix Theory

4. **Varga, R. S. (2009)**: *Matrix Iterative Analysis*, 2nd ed., Springer.
   - Chapter 3: M-matrices and convergence theory

5. **Horn & Johnson (2013)**: *Matrix Analysis*, 2nd ed., Cambridge University Press.
   - Chapter 8: Perron-Frobenius theory

### Applications to PDEs

6. **Samarskii, A. A. (2001)**: *The Theory of Difference Schemes*, CRC Press.
   - Chapter 4: Maximum principles and M-matrices

7. **Morton & Mayers (2005)**: *Numerical Solution of Partial Differential Equations*, Cambridge University Press.
   - Chapter 2: Stability and convergence analysis

---

**Last Updated**: 2025-11-13
**Status**: Graduate-level spectral theory foundation for GFDM monotonicity
**Next**: See application in `01_mathematical_foundation.md` Section 5
