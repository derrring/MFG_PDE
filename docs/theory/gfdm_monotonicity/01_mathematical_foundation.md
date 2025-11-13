# GFDM Monotonicity: Mathematical Foundation

**Created**: 2025-11-13
**Purpose**: Rigorous mathematical theory for monotone GFDM schemes
**Level**: Graduate/Research

---

## 1. Introduction

This document establishes the mathematical foundation for **monotone Generalized Finite Difference Method (GFDM)** schemes applied to Hamilton-Jacobi-Bellman (HJB) equations in Mean Field Games (MFG).

### Notational Conventions

- **Spatial domain**: $\Omega \subset \mathbb{R}^d$ (open, bounded, $d \geq 1$)
- **Time horizon**: $[0,T]$ with $T > 0$
- **Value function**: $u : [0,T] \times \Omega \to \mathbb{R}$
- **Density**: $m : [0,T] \times \Omega \to \mathbb{R}_+$
- **Grid/Collocation points**: $\{x_j\}_{j=1}^{N} \subset \Omega$
- **Discrete value function**: $\mathbf{u} = (u_1, \ldots, u_N)^T \in \mathbb{R}^N$

---

## 2. HJB Equation in MFG

### 2.1 Continuous Formulation

The value function $u(t,x)$ solves the HJB equation:

$$
\begin{cases}
-\dfrac{\partial u}{\partial t} = H(\nabla u, m, x) + \dfrac{\sigma^2}{2} \Delta u & \text{in } (0,T) \times \Omega \\[0.5em]
u(T, x) = g(x, m(T,x)) & \text{in } \Omega \\[0.5em]
+ \text{ boundary conditions} & \text{on } \partial\Omega
\end{cases}
$$

**Hamiltonian** (congestion game):

$$
H(p, m, x) = \frac{1}{2}|p|^2 + \gamma m(x) |p|^2 + V(x)
$$

where:
- $p = \nabla u \in \mathbb{R}^d$: Momentum/control gradient
- $|p|^2 = \sum_{i=1}^d p_i^2$: Euclidean norm squared
- $\gamma > 0$: Congestion intensity parameter
- $V : \Omega \to \mathbb{R}$: External potential field
- $\sigma > 0$: Diffusion coefficient

**Assumptions**:
1. $V \in C^2(\Omega)$ with $\|V\|_{C^2} < \infty$
2. $\gamma < 1/(2\|m\|_\infty)$ ensures $(1 + 2\gamma m) > 0$
3. $\sigma > 0$ ensures strong ellipticity

### 2.2 Weak (Viscosity) Solution

**Definition** (Viscosity solution): A function $u \in C([0,T] \times \overline{\Omega})$ is a viscosity solution if for all $\phi \in C^2((0,T) \times \Omega)$:

1. **Subsolution**: At local maxima of $u - \phi$,
$$
-\frac{\partial \phi}{\partial t} - H(\nabla \phi, m, x) - \frac{\sigma^2}{2}\Delta\phi \leq 0
$$

2. **Supersolution**: At local minima of $u - \phi$,
$$
-\frac{\partial \phi}{\partial t} - H(\nabla \phi, m, x) - \frac{\sigma^2}{2}\Delta\phi \geq 0
$$

**Uniqueness**: Under the above assumptions, the viscosity solution is unique (Crandall-Lions theory).

---

## 3. Finite Difference Framework

### 3.1 Grid Function Space

**Discrete solution space**:

$$
V_h = \{\mathbf{v} = (v_1, \ldots, v_N)^T : v_j \approx u(x_j)\}
$$

**Semi-discrete HJB equation** (method of lines):

$$
-\frac{du_j}{dt} = F_j(\mathbf{u}, \mathbf{m}), \quad j = 1, \ldots, N
$$

where $F_j : \mathbb{R}^N \times \mathbb{R}^N \to \mathbb{R}$ is the **numerical scheme operator**.

### 3.2 Gradient Approximation (GFDM)

At collocation point $x_j$, approximate the gradient using neighboring values:

$$
\nabla u(x_j) \approx \nabla_h u_j = \sum_{k=1}^{N_j} \mathbf{c}_{j,k} u_k
$$

where:
- $N_j$: Number of neighbors in stencil $\mathcal{S}_j = \{k : x_k \in B(x_j, r_j)\}$
- $\mathbf{c}_{j,k} \in \mathbb{R}^d$: **Vectorial collocation weights** for gradient

**Matrix notation**:

$$
\nabla_h u_j = \mathbf{C}_j \mathbf{u}_{\mathcal{S}_j}
$$

where $\mathbf{C}_j \in \mathbb{R}^{d \times N_j}$ is the gradient weight matrix.

### 3.3 Laplacian Approximation (GFDM)

Similarly, for the diffusion term:

$$
\Delta u(x_j) \approx \Delta_h u_j = \sum_{k=1}^{N_j} w_{j,k} u_k
$$

where $w_{j,k} \in \mathbb{R}$ are **scalar collocation weights** for Laplacian.

**Stencil notation**:

$$
\Delta_h u_j = \sum_{k \in \mathcal{S}_j} w_{j,k} u_k = w_{j,j} u_j + \sum_{k \in \mathcal{S}_j \setminus \{j\}} w_{j,k} u_k
$$

---

## 4. GFDM Weight Computation via Taylor Expansion

### 4.1 Polynomial Reproduction

**Goal**: Choose weights $\{w_{j,k}\}$ such that $\Delta_h p_\beta = \Delta p_\beta$ for all polynomials $p_\beta$ up to degree $p$.

**Multi-index notation**: $\beta = (\beta_1, \ldots, \beta_d) \in \mathbb{N}_0^d$ with $|\beta| = \beta_1 + \cdots + \beta_d$.

**Monomial basis**:

$$
\Phi_\beta(x) = x_1^{\beta_1} \cdots x_d^{\beta_d}, \quad |\beta| \leq p
$$

**Number of basis functions**: $M = \binom{d+p}{p}$

### 4.2 Taylor Expansion at Stencil Points

For $u \in C^{p+1}$, Taylor expansion around $x_j$:

$$
u(x_k) = \sum_{|\beta| \leq p} \frac{D^\beta u(x_j)}{\beta!} (x_k - x_j)^\beta + O(|x_k - x_j|^{p+1})
$$

where:
- $D^\beta u = \frac{\partial^{|\beta|} u}{\partial x_1^{\beta_1} \cdots \partial x_d^{\beta_d}}$
- $(x_k - x_j)^\beta = (x_{k,1} - x_{j,1})^{\beta_1} \cdots (x_{k,d} - x_{j,d})^{\beta_d}$

**Matrix form**: Let $\mathbf{D}_j \in \mathbb{R}^M$ be the vector of Taylor coefficients:

$$
\mathbf{D}_j = \left(\frac{D^\beta u(x_j)}{\beta!}\right)_{|\beta| \leq p}
$$

Then:

$$
u(x_k) \approx \sum_{\beta} D_\beta(x_j) \Phi_\beta(x_k - x_j) = \mathbf{A}_{k,:} \mathbf{D}_j
$$

where $\mathbf{A} \in \mathbb{R}^{N_j \times M}$ with $A_{k,\beta} = \Phi_\beta(x_k - x_j)$.

### 4.3 Weighted Least-Squares

**Objective**: Find $\mathbf{D}_j$ minimizing weighted residual:

$$
\min_{\mathbf{D}} \sum_{k \in \mathcal{S}_j} \omega_k \left|u_k - \sum_\beta D_\beta \Phi_\beta(x_k - x_j)\right|^2
$$

where $\omega_k > 0$ are spatial weights (e.g., $\omega_k = |x_k - x_j|^{-1}$).

**Normal equations**:

$$
(\mathbf{A}^T \mathbf{W} \mathbf{A}) \mathbf{D}_j = \mathbf{A}^T \mathbf{W} \mathbf{u}_{\mathcal{S}_j}
$$

where $\mathbf{W} = \text{diag}(\omega_1, \ldots, \omega_{N_j})$.

**Solution**:

$$
\mathbf{D}_j = (\mathbf{A}^T \mathbf{W} \mathbf{A})^{-1} \mathbf{A}^T \mathbf{W} \mathbf{u}_{\mathcal{S}_j} = \mathbf{B} \mathbf{u}_{\mathcal{S}_j}
$$

where $\mathbf{B} = (\mathbf{A}^T \mathbf{W} \mathbf{A})^{-1} \mathbf{A}^T \mathbf{W} \in \mathbb{R}^{M \times N_j}$.

### 4.4 Extracting Derivative Weights

**Laplacian index**: $\beta_{\Delta} = (2, 0, \ldots, 0) + (0, 2, 0, \ldots, 0) + \cdots$ (sum over coordinate Laplacians)

**Weights**: The $\beta_{\Delta}$-th row of $\mathbf{B}$ gives:

$$
w_{j,k} = B_{\beta_{\Delta}, k}, \quad k = 1, \ldots, N_j
$$

**Gradient index**: $\beta_i = (0, \ldots, 1, \ldots, 0)$ with 1 in position $i$

**Vector weights**:

$$
\mathbf{c}_{j,k} = (B_{\beta_1, k}, \ldots, B_{\beta_d, k})^T \in \mathbb{R}^d
$$

---

## 5. Monotone Scheme Theory

### 5.1 Definition of Monotone Scheme

**Definition** (Barles-Souganidis, 1991): The numerical scheme $F_j(\mathbf{u})$ is **monotone** if:

$$
u_k \geq v_k \text{ for all } k \neq j \implies F_j(\mathbf{u}) \geq F_j(\mathbf{v})
$$

**Equivalent condition** (for differentiable $F_j$):

$$
\frac{\partial F_j}{\partial u_k} \geq 0, \quad \text{for all } k \neq j
$$

### 5.2 M-Matrix Property

For **linear schemes** (e.g., diffusion only):

$$
F_j(\mathbf{u}) = \sum_{k \in \mathcal{S}_j} w_{j,k} u_k
$$

**M-matrix structure** ensures monotonicity:

$$
\begin{aligned}
w_{j,j} &\leq 0 \quad &&\text{(diagonal negative)} \\
w_{j,k} &\geq 0 \quad &&\text{(off-diagonal non-negative), } k \neq j
\end{aligned}
$$

**Row sum condition** (consistency):

$$
\sum_{k \in \mathcal{S}_j} w_{j,k} = 0
$$

### 5.3 Convergence Theorem

**Theorem** (Barles-Souganidis): Let $F_j$ be a monotone, consistent, and stable scheme. Then the discrete solution $u_h$ converges to the unique viscosity solution $u$ as $h \to 0$.

**Requirements**:
1. **Monotonicity**: $\partial F_j / \partial u_k \geq 0$ for $k \neq j$
2. **Consistency**: $F_j(\mathbf{1} \phi) \to -\partial_t \phi - H(\nabla \phi) - \frac{\sigma^2}{2}\Delta \phi$ as $h \to 0$
3. **Stability**: $\|\mathbf{u}^n\|_\infty \leq C$ uniformly in $n$ and $h$

---

## 6. Challenges for GFDM Monotonicity

### 6.1 High-Order Stencils Break M-Matrix

For polynomial degree $p \geq 3$, the least-squares solution $\mathbf{D}_j$ often produces weights that violate:

$$
w_{j,k} \geq 0 \quad \text{for } k \neq j
$$

**Reason**: Higher-order Taylor terms introduce **negative weights** for oscillation cancellation.

**Example** (1D, 5-point stencil, quartic polynomial):

$$
\Delta_h u \approx -\frac{1}{12h^2}(u_{j-2} - 16u_{j-1} + 30u_j - 16u_{j+1} + u_{j+2})
$$

has negative coefficient at $k = j \pm 2$.

### 6.2 Nonlinear Hamiltonian Complicates Monotonicity

For $H = \frac{1}{2}|\nabla u|^2$, the scheme becomes:

$$
F_j(\mathbf{u}) = \frac{1}{2}\left|\sum_{k} \mathbf{c}_{j,k} u_k\right|^2 + \frac{\sigma^2}{2}\sum_k w_{j,k} u_k
$$

**Derivative**:

$$
\frac{\partial F_j}{\partial u_\ell} = \left(\sum_k \mathbf{c}_{j,k} u_k\right) \cdot \mathbf{c}_{j,\ell} + \frac{\sigma^2}{2} w_{j,\ell}
$$

**Challenge**: Even if $w_{j,\ell} \geq 0$, the gradient term can be negative depending on $\mathbf{u}$ and geometry of $\mathbf{c}_{j,\ell}$.

### 6.3 Irregular Geometries and Boundaries

Near boundaries or obstacles, stencils become:
- Asymmetric
- Poorly conditioned
- Have larger truncation errors

This exacerbates monotonicity violations.

---

## 7. Constrained GFDM: QP Formulation

### 7.1 Monotonicity as Constraints

Instead of solving unconstrained least-squares, solve:

$$
\begin{aligned}
\min_{\mathbf{D}} \quad & \|\mathbf{A}\mathbf{D} - \mathbf{u}_{\mathcal{S}_j}\|_{\mathbf{W}}^2 \\
\text{s.t.} \quad & \mathbf{G} \mathbf{D} \leq \mathbf{h}
\end{aligned}
$$

where $\mathbf{G} \mathbf{D} \leq \mathbf{h}$ encodes monotonicity requirements.

**Quadratic Program (QP)**: This is a convex QP with:
- $M$ variables (Taylor coefficients)
- $O(N_j)$ inequality constraints

### 7.2 Constraint Construction (Indirect Approach)

Based on $\mathbf{D} = (D_0, D_{\beta_1}, \ldots, D_{\beta_d}, D_{\beta_\Delta}, \ldots)$:

**Constraint 1: Laplacian negativity**

$$
D_{\beta_\Delta} \leq -\epsilon, \quad \epsilon > 0 \text{ (tolerance)}
$$

ensures diffusion dominates.

**Constraint 2: Gradient boundedness**

$$
|D_{\beta_i}| \leq C \sigma^2 |D_{\beta_\Delta}|, \quad i = 1, \ldots, d
$$

prevents advection from overwhelming diffusion.

**Constraint 3: Higher-order control**

$$
\sum_{|\beta| \geq 3} |D_\beta| \leq |D_{\beta_\Delta}|
$$

limits truncation error.

**Matrix form**: Each constraint becomes a row in $\mathbf{G}$:

$$
\mathbf{G} = \begin{pmatrix}
-\mathbf{e}_{\beta_\Delta}^T \\
\pm \mathbf{e}_{\beta_i}^T - C\sigma^2 |\mathbf{e}_{\beta_\Delta}|^T \\
\vdots
\end{pmatrix}, \quad
\mathbf{h} = \begin{pmatrix}
-\epsilon \\
0 \\
\vdots
\end{pmatrix}
$$

---

## 8. Mathematical Optimality

### 8.1 Bias-Variance Trade-off

**Unconstrained solution**: Minimizes variance (interpolation error) but may have large bias (non-monotone).

**Constrained solution**: Accepts increased variance to reduce bias (enforce monotonicity).

**Optimal trade-off** depends on:
- Problem regularity ($u \in C^k$)
- Stencil geometry
- Desired accuracy order

### 8.2 Convergence Rate

**Theorem** (Formal): If the constrained GFDM weights satisfy:
1. Consistency: $\Delta_h u_j - \Delta u(x_j) = O(h^q)$ for some $q \geq 1$
2. M-matrix property: $w_{j,k} \geq 0$ for $k \neq j$
3. Stability: $\|\mathbf{w}_j\|_1 = O(1)$

Then the solution $u_h$ converges to the viscosity solution with rate $O(h^q)$.

**Practical observation**: Constrained GFDM typically achieves $q = 2$ (second-order) even with higher polynomial degree $p \geq 3$.

---

## 9. Summary

### Key Results

1. **HJB equation** in MFG has a unique viscosity solution under standard assumptions
2. **GFDM** approximates derivatives via weighted least-squares on polynomial basis
3. **Monotone schemes** converge to viscosity solutions (Barles-Souganidis)
4. **High-order GFDM** naturally violates monotonicity (negative weights)
5. **Constrained GFDM** restores monotonicity via QP with indirect constraints

### Open Questions

1. **Direct Hamiltonian constraints**: Can we enforce $\partial H_h / \partial u_k \geq 0$ directly?
2. **Optimal constraint selection**: Which subset of constraints suffices for monotonicity?
3. **Adaptive strategies**: When can we relax constraints without losing convergence?

---

## References

### Monotone Schemes
1. **Barles & Souganidis (1991)**: "Convergence of approximation schemes for fully nonlinear second order equations", *Asymptotic Analysis*, 4(3):271-283.

2. **Oberman (2006)**: "Convergent difference schemes for degenerate elliptic and parabolic equations: Hamilton-Jacobi equations and free boundary problems", *SIAM J. Numer. Anal.*, 44(2):879-895.

### GFDM Theory
3. **Benito et al. (2001)**: "Influence of several factors in the generalized finite difference method", *Applied Mathematical Modelling*, 25(12):1039-1053.

4. **Liszka & Orkisz (1980)**: "The finite difference method at arbitrary irregular grids and its application in applied mechanics", *Computers & Structures*, 11(1-2):83-95.

### Viscosity Solutions
5. **Crandall & Lions (1983)**: "Viscosity solutions of Hamilton-Jacobi equations", *Trans. Amer. Math. Soc.*, 277(1):1-42.

6. **Fleming & Soner (2006)**: *Controlled Markov Processes and Viscosity Solutions*, 2nd ed., Springer.

---

**Last Updated**: 2025-11-13
**Next**: See `02_hamiltonian_constraints.md` for direct constraint theory
