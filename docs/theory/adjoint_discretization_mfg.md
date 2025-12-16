# Adjoint Discretization for Mean Field Games

## The Structure-Preserving Principle

In numerical methods for Mean Field Games (MFG), the discretization of the Fokker-Planck (FP) equation is **not an independent choice**. The Achdou-Capuzzo-Dolcetta framework establishes that the FP operator must be the **discrete adjoint** of the linearized Hamilton-Jacobi-Bellman (HJB) operator:

$$
A_{\text{FP}} = (A_{\text{HJB}})^\top
$$

This document provides the mathematical foundations for this principle.

---

## 1. Continuous Duality

### 1.1 The MFG System

Consider a stationary MFG system on a bounded domain $\Omega \subset \mathbb{R}^d$:

$$
\begin{cases}
-\nu \Delta u + H(x, \nabla u) = F[m](x) & \text{(HJB)} \\
-\nu \Delta m - \nabla \cdot (m \, \nabla_p H(x, \nabla u)) = 0 & \text{(FP)} \\
\int_\Omega m \, dx = 1, \quad m \geq 0 & \text{(Probability constraint)}
\end{cases}
$$

where:
- $u(x)$: value function
- $m(x)$: population density
- $H(x, p)$: Hamiltonian (convex in $p$)
- $F[m]$: coupling term (e.g., $F[m] = f(x, m)$)
- $\nu > 0$: viscosity/diffusion coefficient

### 1.2 Optimal Control Interpretation

The optimal drift (feedback control) is:

$$
\alpha^*(x) = -\nabla_p H(x, \nabla u(x))
$$

For the quadratic Hamiltonian $H(x, p) = \frac{1}{2}|p|^2$, this gives $\alpha^* = -\nabla u$.

### 1.3 The Formal $L^2$-Adjoint Structure

Define the **linearized HJB operator** at a solution $u$:

$$
\mathcal{L}_u w := -\nu \Delta w + \nabla_p H(x, \nabla u) \cdot \nabla w
$$

The formal $L^2$-adjoint is:

$$
\mathcal{L}_u^* \phi := -\nu \Delta \phi - \nabla \cdot (\phi \, \nabla_p H(x, \nabla u))
$$

**Key observation:** The FP equation is exactly $\mathcal{L}_u^* m = 0$.

This is not a coincidence. It reflects the **optimality conditions** of the underlying control problem:
- HJB: first-order condition for the value function
- FP: adjoint equation for the density (Lagrange multiplier for the probability constraint)

### 1.4 Variational Formulation

For potential MFG (where the coupling derives from a potential), the system is the gradient of a functional:

$$
\mathcal{J}(u, m) = \int_\Omega \left[ \nu \nabla u \cdot \nabla m + m H(x, \nabla u) - m F[m] \right] dx
$$

The Euler-Lagrange equations are:
- $\frac{\delta \mathcal{J}}{\delta m} = 0 \Rightarrow$ HJB
- $\frac{\delta \mathcal{J}}{\delta u} = 0 \Rightarrow$ FP

The **Hessian** of $\mathcal{J}$ has the block structure:

$$
D^2 \mathcal{J} = \begin{pmatrix} D^2_{uu} \mathcal{J} & D^2_{um} \mathcal{J} \\ D^2_{mu} \mathcal{J} & D^2_{mm} \mathcal{J} \end{pmatrix}
$$

The off-diagonal blocks satisfy $(D^2_{um} \mathcal{J})^* = D^2_{mu} \mathcal{J}$, which is the continuous manifestation of the adjoint structure.

---

## 2. Discrete HJB: Monotone Schemes

### 2.1 Grid Setup

Consider a uniform grid in 1D: $x_i = ih$ for $i = 0, 1, \ldots, N$, with grid spacing $h$.

Define finite difference operators:
- **Forward difference:** $D^+ u_i = \frac{u_{i+1} - u_i}{h}$
- **Backward difference:** $D^- u_i = \frac{u_i - u_{i-1}}{h}$
- **Central second difference:** $D^2 u_i = \frac{u_{i+1} - 2u_i + u_{i-1}}{h^2}$

### 2.2 Monotone Numerical Hamiltonians

The discrete HJB equation requires a **monotone numerical Hamiltonian** $\mathcal{H}(x, p^-, p^+)$ satisfying:

1. **Consistency:** $\mathcal{H}(x, p, p) = H(x, p)$
2. **Monotonicity:**
   - $\frac{\partial \mathcal{H}}{\partial p^-} \leq 0$ (non-increasing in backward argument)
   - $\frac{\partial \mathcal{H}}{\partial p^+} \geq 0$ (non-decreasing in forward argument)

#### Example: Godunov Flux

For the quadratic Hamiltonian $H(p) = \frac{1}{2}|p|^2$:

$$
\mathcal{H}^{\text{God}}(p^-, p^+) = \frac{1}{2} \max(p^-, 0)^2 + \frac{1}{2} \min(p^+, 0)^2
$$

This selects the "upwind" direction based on the sign of the gradient.

#### Example: Engquist-Osher Flux

$$
\mathcal{H}^{\text{EO}}(p^-, p^+) = \int_0^{p^-} \max(H'(s), 0) \, ds + \int_0^{p^+} \min(H'(s), 0) \, ds + H(0)
$$

For $H(p) = \frac{1}{2}p^2$:

$$
\mathcal{H}^{\text{EO}}(p^-, p^+) = \frac{1}{2}(p^-)_+^2 + \frac{1}{2}(p^+)_-^2
$$

where $(x)_+ = \max(x, 0)$ and $(x)_- = \min(x, 0)$.

### 2.3 The Discrete HJB System

The fully discrete HJB equation at interior node $i$:

$$
-\nu D^2 u_i + \mathcal{H}(x_i, D^- u_i, D^+ u_i) = f_i
$$

Expanding:

$$
-\nu \frac{u_{i+1} - 2u_i + u_{i-1}}{h^2} + \mathcal{H}\left(x_i, \frac{u_i - u_{i-1}}{h}, \frac{u_{i+1} - u_i}{h}\right) = f_i
$$

This defines a nonlinear system $\mathbf{G}(\mathbf{u}) = \mathbf{f}$ where $\mathbf{u} = (u_1, \ldots, u_{N-1})^\top$.

### 2.4 Linearization: The Jacobian Matrix

To solve via Newton's method, we compute the Jacobian $A_{\text{HJB}} = \frac{\partial \mathbf{G}}{\partial \mathbf{u}}$.

At node $i$, the equation $G_i(\mathbf{u})$ depends on $u_{i-1}, u_i, u_{i+1}$:

$$
\frac{\partial G_i}{\partial u_{i-1}} = -\frac{\nu}{h^2} + \frac{1}{h} \frac{\partial \mathcal{H}}{\partial p^-}
$$

$$
\frac{\partial G_i}{\partial u_i} = \frac{2\nu}{h^2} - \frac{1}{h} \frac{\partial \mathcal{H}}{\partial p^-} + \frac{1}{h} \frac{\partial \mathcal{H}}{\partial p^+}
$$

$$
\frac{\partial G_i}{\partial u_{i+1}} = -\frac{\nu}{h^2} - \frac{1}{h} \frac{\partial \mathcal{H}}{\partial p^+}
$$

where the partial derivatives are evaluated at $(x_i, D^- u_i, D^+ u_i)$.

#### Matrix Structure

Define:
- $a_i^- := -\frac{\partial \mathcal{H}}{\partial p^-}(x_i, D^- u_i, D^+ u_i) \geq 0$ (by monotonicity)
- $a_i^+ := \frac{\partial \mathcal{H}}{\partial p^+}(x_i, D^- u_i, D^+ u_i) \geq 0$ (by monotonicity)

The Jacobian has the tridiagonal structure:

$$
(A_{\text{HJB}})_{i,i-1} = -\frac{\nu}{h^2} - \frac{a_i^-}{h}
$$

$$
(A_{\text{HJB}})_{i,i} = \frac{2\nu}{h^2} + \frac{a_i^- + a_i^+}{h}
$$

$$
(A_{\text{HJB}})_{i,i+1} = -\frac{\nu}{h^2} - \frac{a_i^+}{h}
$$

**Key property:** $A_{\text{HJB}}$ is an **M-matrix**:
- Diagonal entries are positive
- Off-diagonal entries are non-positive
- Row sums are non-negative (strictly positive with suitable boundary conditions)

This guarantees:
- Existence and uniqueness of solutions
- Discrete maximum principle
- Monotone convergence of iterative methods

---

## 3. Discrete FP: The Adjoint Operator

### 3.1 The Transpose Construction

We define the discrete FP operator as:

$$
A_{\text{FP}} := (A_{\text{HJB}})^\top
$$

Explicitly:

$$
(A_{\text{FP}})_{i,i-1} = (A_{\text{HJB}})_{i-1,i} = -\frac{\nu}{h^2} - \frac{a_{i-1}^+}{h}
$$

$$
(A_{\text{FP}})_{i,i} = (A_{\text{HJB}})_{i,i} = \frac{2\nu}{h^2} + \frac{a_i^- + a_i^+}{h}
$$

$$
(A_{\text{FP}})_{i,i+1} = (A_{\text{HJB}})_{i+1,i} = -\frac{\nu}{h^2} - \frac{a_{i+1}^-}{h}
$$

### 3.2 Interpretation as a Conservative Scheme

The discrete FP equation $A_{\text{FP}} \mathbf{m} = \mathbf{0}$ can be written as:

$$
-\nu D^2 m_i - \frac{1}{h}\left( F_{i+1/2} - F_{i-1/2} \right) = 0
$$

where the numerical fluxes are:

$$
F_{i+1/2} = a_i^+ m_i - a_{i+1}^- m_{i+1}
$$

$$
F_{i-1/2} = a_{i-1}^+ m_{i-1} - a_i^- m_i
$$

**Interpretation:**
- $a_i^+ m_i$: mass flux from cell $i$ to cell $i+1$ (rightward)
- $a_i^- m_i$: mass flux from cell $i$ to cell $i-1$ (leftward)

This is precisely an **upwind flux scheme** where:
- If $a_i^+ > 0$ (drift to the right), mass flows from $i$ to $i+1$
- If $a_i^- > 0$ (drift to the left), mass flows from $i$ to $i-1$

### 3.3 The "Hidden" Flux-Vector Splitting

The adjoint construction automatically implements **flux-vector splitting**:

$$
\nabla \cdot (m \alpha^*) \approx \nabla \cdot (m \alpha^+) + \nabla \cdot (m \alpha^-)
$$

where $\alpha^+ \geq 0$ and $\alpha^- \leq 0$ are the positive and negative parts of the drift.

For the positive part: use backward difference (upwind from left)
For the negative part: use forward difference (upwind from right)

The HJB monotonicity conditions ensure this splitting is consistent with the optimal control.

---

## 4. Structural Properties

### 4.1 Mass Conservation

**Theorem:** The discrete FP operator conserves total mass.

**Proof:** Let $\mathbf{1} = (1, 1, \ldots, 1)^\top$. We show that $\mathbf{1}^\top A_{\text{FP}} = \mathbf{0}^\top$.

Since $A_{\text{FP}} = A_{\text{HJB}}^\top$, we have $\mathbf{1}^\top A_{\text{FP}} = (A_{\text{HJB}} \mathbf{1})^\top$.

For each row $i$ of $A_{\text{HJB}}$:

$$
\sum_j (A_{\text{HJB}})_{i,j} = \left(-\frac{\nu}{h^2} - \frac{a_i^-}{h}\right) + \left(\frac{2\nu}{h^2} + \frac{a_i^- + a_i^+}{h}\right) + \left(-\frac{\nu}{h^2} - \frac{a_i^+}{h}\right) = 0
$$

Therefore $A_{\text{HJB}} \mathbf{1} = \mathbf{0}$, which implies $\mathbf{1}^\top A_{\text{FP}} = \mathbf{0}^\top$.

**Consequence:** For any $\mathbf{m}$, we have $\mathbf{1}^\top (A_{\text{FP}} \mathbf{m}) = 0$, so the total mass $\sum_i m_i$ is unchanged by the FP dynamics. $\square$

### 4.2 Positivity Preservation

**Theorem:** If $A_{\text{HJB}}$ is an M-matrix, then $A_{\text{FP}}$ preserves non-negativity.

**Proof:** $A_{\text{FP}} = A_{\text{HJB}}^\top$ is also an M-matrix (the M-matrix property is preserved under transposition).

For implicit time-stepping of the FP equation:

$$
\frac{\mathbf{m}^{n+1} - \mathbf{m}^n}{\Delta t} + A_{\text{FP}} \mathbf{m}^{n+1} = 0
$$

Rearranging: $\left( I + \Delta t \, A_{\text{FP}} \right) \mathbf{m}^{n+1} = \mathbf{m}^n$

The matrix $B = I + \Delta t \, A_{\text{FP}}$ has:
- Positive diagonal: $B_{ii} = 1 + \Delta t (A_{\text{FP}})_{ii} > 0$
- Non-positive off-diagonal: $B_{ij} = \Delta t (A_{\text{FP}})_{ij} \leq 0$ for $i \neq j$
- Positive row sums: $\sum_j B_{ij} = 1 + \Delta t \sum_j (A_{\text{FP}})_{ij} = 1$

Therefore $B^{-1} \geq 0$ (entry-wise), and $\mathbf{m}^{n+1} = B^{-1} \mathbf{m}^n \geq 0$ whenever $\mathbf{m}^n \geq 0$. $\square$

### 4.3 Discrete Duality Pairing

**Definition:** The discrete $L^2$ inner product is $\langle \mathbf{u}, \mathbf{m} \rangle_h = h \sum_i u_i m_i$.

**Theorem:** The adjoint relationship holds:

$$
\langle A_{\text{HJB}} \mathbf{w}, \mathbf{m} \rangle_h = \langle \mathbf{w}, A_{\text{FP}} \mathbf{m} \rangle_h
$$

**Proof:** Direct computation using $A_{\text{FP}} = A_{\text{HJB}}^\top$:

$$
\langle A_{\text{HJB}} \mathbf{w}, \mathbf{m} \rangle_h = h \, \mathbf{m}^\top A_{\text{HJB}} \mathbf{w} = h \, (A_{\text{HJB}}^\top \mathbf{m})^\top \mathbf{w} = h \, (A_{\text{FP}} \mathbf{m})^\top \mathbf{w} = \langle \mathbf{w}, A_{\text{FP}} \mathbf{m} \rangle_h \quad \square
$$

---

## 5. Newton's Method for the Coupled System

### 5.1 The Full MFG System

The discrete MFG system is:

$$
\begin{cases}
\mathbf{G}(\mathbf{u}, \mathbf{m}) := -\nu D^2 \mathbf{u} + \mathcal{H}(\mathbf{u}) - \mathbf{F}(\mathbf{m}) = \mathbf{0} \\
\mathbf{K}(\mathbf{u}, \mathbf{m}) := A_{\text{FP}}(\mathbf{u}) \, \mathbf{m} = \mathbf{0} \\
\sum_i m_i = 1
\end{cases}
$$

### 5.2 The Jacobian of the Coupled System

For Newton iteration, we need the Jacobian of $(\mathbf{G}, \mathbf{K})$ with respect to $(\mathbf{u}, \mathbf{m})$:

$$
J = \begin{pmatrix}
\frac{\partial \mathbf{G}}{\partial \mathbf{u}} & \frac{\partial \mathbf{G}}{\partial \mathbf{m}} \\[6pt]
\frac{\partial \mathbf{K}}{\partial \mathbf{u}} & \frac{\partial \mathbf{K}}{\partial \mathbf{m}}
\end{pmatrix}
= \begin{pmatrix}
A_{\text{HJB}} & -\frac{\partial \mathbf{F}}{\partial \mathbf{m}} \\[6pt]
\frac{\partial A_{\text{FP}}}{\partial \mathbf{u}} \mathbf{m} & A_{\text{FP}}
\end{pmatrix}
$$

### 5.3 Symmetry for Potential Games

For potential MFG where $\mathbf{F}(\mathbf{m}) = \nabla_m \mathcal{F}(\mathbf{m})$ for some functional $\mathcal{F}$:

$$
\frac{\partial \mathbf{F}}{\partial \mathbf{m}} = D^2_m \mathcal{F}
$$

is symmetric. Combined with $A_{\text{FP}} = A_{\text{HJB}}^\top$, this gives the Jacobian a **saddle-point structure**:

$$
J \approx \begin{pmatrix}
A & -B^\top \\
B & A^\top
\end{pmatrix}
$$

where the off-diagonal blocks are related by transposition (up to lower-order terms from the $\mathbf{m}$-dependence of $A_{\text{FP}}$).

This structure is essential for:
- Proving non-singularity of $J$
- Designing efficient preconditioners
- Ensuring quadratic convergence of Newton's method

---

## 6. Extension to Multiple Dimensions

### 6.1 Tensor Product Grids

On a $d$-dimensional grid, the HJB operator becomes:

$$
(A_{\text{HJB}})_{\mathbf{i}, \mathbf{j}} = \begin{cases}
\sum_{k=1}^d \left( \frac{2\nu}{h_k^2} + \frac{a_{\mathbf{i}}^{k,-} + a_{\mathbf{i}}^{k,+}}{h_k} \right) & \text{if } \mathbf{j} = \mathbf{i} \\[6pt]
-\frac{\nu}{h_k^2} - \frac{a_{\mathbf{i}}^{k,-}}{h_k} & \text{if } \mathbf{j} = \mathbf{i} - \mathbf{e}_k \\[6pt]
-\frac{\nu}{h_k^2} - \frac{a_{\mathbf{i}}^{k,+}}{h_k} & \text{if } \mathbf{j} = \mathbf{i} + \mathbf{e}_k \\[6pt]
0 & \text{otherwise}
\end{cases}
$$

where $\mathbf{e}_k$ is the $k$-th unit vector and:

$$
a_{\mathbf{i}}^{k,\pm} = \pm \frac{\partial \mathcal{H}}{\partial p_k^\pm}(\mathbf{x}_{\mathbf{i}}, D^- \mathbf{u}_{\mathbf{i}}, D^+ \mathbf{u}_{\mathbf{i}})
$$

### 6.2 Dimensional Splitting for Hamiltonians

For separable Hamiltonians $H(\mathbf{x}, \mathbf{p}) = \sum_{k=1}^d H_k(x_k, p_k)$:

$$
\mathcal{H}(\mathbf{x}, \mathbf{p}^-, \mathbf{p}^+) = \sum_{k=1}^d \mathcal{H}_k(x_k, p_k^-, p_k^+)
$$

The adjoint construction applies dimension-by-dimension.

---

## 7. Time-Dependent Problems

### 7.1 Parabolic MFG System

$$
\begin{cases}
-\partial_t u - \nu \Delta u + H(x, \nabla u) = F[m] & \text{(HJB, backward)} \\
\partial_t m - \nu \Delta m - \nabla \cdot (m \nabla_p H) = 0 & \text{(FP, forward)}
\end{cases}
$$

with terminal condition $u(T, x) = G[m(T, \cdot)](x)$ and initial condition $m(0, x) = m_0(x)$.

### 7.2 Implicit-Explicit Time Stepping

**HJB (backward in time):**
$$
\frac{u^n - u^{n+1}}{\Delta t} - \nu D^2 u^n + \mathcal{H}(u^n) = F[m^n]
$$

**FP (forward in time):**
$$
\frac{m^{n+1} - m^n}{\Delta t} + A_{\text{FP}}(u^n) m^{n+1} = 0
$$

The adjoint relationship $A_{\text{FP}} = A_{\text{HJB}}^\top$ must hold **at each time step** with the same linearization point.

---

## 8. Implications for Implementation

### 8.1 The Standard Recipe

1. **Build HJB operator** with monotone upwind discretization
2. **Linearize** to get $A_{\text{HJB}}$ (the Jacobian)
3. **Transpose** to get $A_{\text{FP}} = A_{\text{HJB}}^\top$
4. **Solve FP** using $A_{\text{FP}}$

### 8.2 Available FDM Schemes in `mfg_pde`

Both HJB and FP solvers support selectable advection schemes:

**HJB Solver** (`HJBFDMSolver`):
- `advection_scheme="gradient_upwind"` (default) - Godunov upwind, monotone
- `advection_scheme="gradient_centered"` - Central differences, second-order

**FP Solver** (`FPFDMSolver`):
- `advection_scheme="gradient_upwind"` (default) - For standalone/non-MFG use
- `advection_scheme="gradient_centered"` - For standalone/non-MFG use
- `advection_scheme="divergence_upwind"` - For MFG coupling with HJB gradient_upwind
- `advection_scheme="divergence_centered"` - For MFG coupling with HJB gradient_centered

Independent scheme selection is valid for:

- **Standalone equations** (HJB-only or FP-only, not coupled)
- **Testing and debugging** individual components
- **Non-MFG applications** (pure advection-diffusion)
- **Research comparisons** between discretization strategies

### 8.3 When Adjoint Coupling Is Required

For **production MFG solvers** with coupled HJB-FP systems:

- Use adjoint construction to preserve variational structure
- Ensures Newton convergence for the coupled system
- Maintains mass conservation and positivity
- Required for theoretical convergence guarantees

### 8.4 Compatible HJB-FP Scheme Pairings (Without Automatic Transpose)

When users choose independent FP schemes (not using automatic transpose), they must respect the **duality of the operators**. The correct FP scheme is the **"Divergence" counterpart** of the "Gradient" scheme chosen for HJB.

#### The Pairing Rule: Integration by Parts

The pairing is dictated by **integration by parts**, which is the continuous equivalent of the matrix transpose:

$$
\int_{\Omega} \underbrace{(b \cdot \nabla u)}_{\text{Gradient Form}} \, m \, dx = - \int_{\Omega} u \, \underbrace{\nabla \cdot (b m)}_{\text{Divergence Form}} \, dx
$$

(with appropriate boundary conditions).

#### Why HJB Uses Gradient Form

**HJB is a gradient equation.** The advection term comes from the Hamiltonian $H(x, \nabla u)$. The operator acts on the **gradient** of the solution:

$$
-\nu \Delta u + b \cdot \nabla u = f
$$

Therefore, HJB requires a **Gradient** scheme (`gradient_centered` or `gradient_upwind`).

**Important:** Divergence schemes are **invalid for HJB** because HJB is not a conservation law.

#### Why FP Uses Divergence Form

**FP is a divergence equation.** The FP equation describes the conservation of probability mass. The operator acts on the **divergence** of the flux:

$$
\partial_t m - \nu \Delta m - \nabla \cdot (b m) = 0
$$

Therefore, FP requires a **Divergence** scheme (`divergence_centered` or `divergence_upwind`).

#### The Error of Using Gradient Schemes for FP

> **Formulation Error:** Using `gradient_upwind` for FP discretizes $b \cdot \nabla m$ instead of $\nabla \cdot (bm)$.

This treats the density $m$ like a passive scalar field rather than a conserved quantity, **violating mass conservation**:

$$
\frac{d}{dt} \int_\Omega m \, dx \neq 0 \quad \text{(mass not conserved)}
$$

#### The Correct Pairing Matrix

| HJB Scheme (Primal) | Correct FP Scheme (Dual) | Why? |
|:--------------------|:-------------------------|:-----|
| `gradient_upwind` | `divergence_upwind` | Matches characteristic flow. Preserves stability and positivity. |
| `gradient_centered` | `divergence_centered` | Preserves symmetric structure (though often unstable for convection). |
| `divergence_*` | *(Invalid for HJB)* | HJB is not a conservation law. |

#### Matching the Stencil (Upwind vs. Centered)

**Upwind ↔ Upwind:**
- In HJB, "upwind" means looking in the direction the agent wants to move to update the value function $u$
- In FP, "upwind" means moving the mass $m$ in that exact same direction
- If HJB uses a **backward** difference (looking left because drift is positive), the discrete transpose becomes a **forward** flux difference for the density
- Standard `divergence_upwind` implementations handle this flux splitting automatically

**Centered ↔ Centered:**
- If HJB uses central differences (only stable if viscosity is high relative to drift), the algebraic transpose is exactly a central difference divergence operator

#### Recommended Default Pairing

For most MFG problems:

```python
# Recommended: gradient_upwind (HJB) + divergence_upwind (FP)
hjb_solver = HJBFDMSolver(problem, scheme="gradient_upwind")
fp_solver = FPFDMSolver(problem, advection_scheme="divergence_upwind")
```

This pairing provides:
- First-order accuracy in advection
- Unconditional positivity for the density
- Mass conservation via flux telescoping
- Robustness for problems with sharp gradients or shocks

For smooth problems where higher accuracy is desired:

```python
# Alternative: gradient_centered (HJB) + divergence_centered (FP)
hjb_solver = HJBFDMSolver(problem, scheme="gradient_centered")
fp_solver = FPFDMSolver(problem, advection_scheme="divergence_centered")
```

This pairing provides:
- Second-order accuracy in advection
- May produce oscillations near discontinuities
- Suitable when solutions are known to be smooth

#### Common Errors to Avoid

| Configuration | Problem |
|:--------------|:--------|
| `gradient_upwind` (HJB) + `gradient_upwind` (FP) | **Formulation error.** FP should use divergence form. Breaks mass conservation. |
| `gradient_upwind` (HJB) + `divergence_centered` (FP) | Breaks duality. May cause convergence issues in MFG iteration. |
| `divergence_*` for HJB | **Invalid.** HJB is not a conservation law. |

### 8.5 Current Implementation Status

The current `mfg_pde` implementation uses **independent FP schemes**, which:
- Is simpler to implement and understand
- Works well for weakly coupled problems
- May require smaller time steps for strongly coupled problems
- Does not guarantee the discrete variational structure

**Future enhancement:** Implement adjoint-based FP operator derived from HJB linearization.

---

## 9. References

1. Achdou, Y., & Capuzzo-Dolcetta, I. (2010). *Mean field games: numerical methods*. SIAM Journal on Numerical Analysis, 48(3), 1136-1162.

2. Achdou, Y., & Laurière, M. (2020). *Mean field games and applications: numerical aspects*. In Mean Field Games, Springer. [arXiv:2003.04444](https://arxiv.org/abs/2003.04444)

3. Achdou, Y., & Porretta, A. (2018). *Mean field games with congestion*. Annales de l'IHP Analyse non linéaire, 35(2), 443-480.

4. Carlini, E., & Silva, F. J. (2014). *A fully discrete semi-Lagrangian scheme for a first order mean field game problem*. SIAM Journal on Numerical Analysis, 52(1), 45-67.

5. Benamou, J. D., & Carlier, G. (2015). *Augmented Lagrangian methods for transport optimization, mean field games and degenerate elliptic equations*. Journal of Optimization Theory and Applications, 167(1), 1-26.

---

**Document Info**
- **Last Updated:** 2025-12
- **Related Files:** `fp_fdm.py`, `hjb_fdm.py`, `advection_discretization_methods.md`
- **Author:** MFG_PDE Development Team
