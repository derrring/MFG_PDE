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

### 8.2 The Scheme Selection Is Mathematically Determined

The choice of FDM scheme for HJB and FP is **not a matter of preference**. It is determined by the mathematical structure of the equations.

#### Step 1: HJB Is Non-Divergence Form → Gradient Schemes Only

The HJB equation describes the **value** of a state. Value is transported along optimal paths; it is **not a conserved quantity**:

$$
\partial_t u + \mathbf{b} \cdot \nabla u = 0
$$

**Mathematical Form:** The advection term is $\mathbf{b} \cdot \nabla u$ (gradient of $u$).

**Numerical Consequence:** You **must** use a **Gradient Scheme** (finite difference on $\nabla u$).

> **Divergence schemes are structurally impossible for HJB.** There is no flux to diverge because HJB is not a conservation law. You cannot rewrite $\mathbf{b} \cdot \nabla u$ as $\nabla \cdot (\cdots)$.

#### Step 2: FP Is the Adjoint of HJB → Divergence Schemes Required

The FP equation is the **adjoint** (dual) of the HJB equation. The adjoint operation (integration by parts) flips the differential operator:

$$
\langle \nabla u, \mathbf{v} \rangle_{L^2} = - \langle u, \nabla \cdot \mathbf{v} \rangle_{L^2}
$$

**The adjoint of a Gradient ($\nabla$) is a negative Divergence ($-\nabla \cdot$).**

Because the HJB operator was a gradient operator, the FP operator **must** be a divergence operator:

$$
\partial_t m + \nabla \cdot (m \mathbf{b}) = 0
$$

**Numerical Consequence:** You **must** use a **Divergence Scheme** (conservative finite difference).

> **Using gradient schemes for FP in MFG is a type error** - trying to solve a conservation law using a non-conservative operator. This violates mass conservation.

#### Step 3: The Required Pairings

You do not actually "choose" two separate schemes. You make **one** choice based on the physics (upwind vs. centered), and the rest follows from the mathematics:

| Decision | HJB (Non-Divergence) | FP (Divergence) | Properties |
|:---------|:---------------------|:----------------|:-----------|
| **Stability first** | `gradient_upwind` | `divergence_upwind` | Monotone, positivity-preserving, mass-conserving |
| **Accuracy first** | `gradient_centered` | `divergence_centered` | Second-order, may oscillate, mass-conserving |

These are **mathematically necessary** pairings, not "compatible options".

### 8.3 Available Schemes in `mfg_pde`

**HJB Solver** (`HJBFDMSolver`):
- `advection_scheme="gradient_upwind"` (default) - Godunov upwind, monotone
- `advection_scheme="gradient_centered"` - Central differences, second-order

**FP Solver** (`FPFDMSolver`):
- `advection_scheme="divergence_upwind"` - **Required for MFG** with HJB `gradient_upwind`
- `advection_scheme="divergence_centered"` - **Required for MFG** with HJB `gradient_centered`
- `advection_scheme="gradient_upwind"` - For standalone/non-MFG use only
- `advection_scheme="gradient_centered"` - For standalone/non-MFG use only

> **Warning:** The gradient schemes for FP exist only for standalone advection-diffusion problems (not coupled MFG). Using them in MFG coupling is a formulation error.

### 8.4 The Duality in Detail

#### Integration by Parts: The Bridge Between Forms

The pairing is dictated by **integration by parts**, which is the continuous equivalent of the matrix transpose:

$$
\int_{\Omega} \underbrace{(\mathbf{b} \cdot \nabla u)}_{\text{Gradient Form}} \, m \, dx = - \int_{\Omega} u \, \underbrace{\nabla \cdot (\mathbf{b} m)}_{\text{Divergence Form}} \, dx
$$

(with appropriate boundary conditions).

At the discrete level, this becomes:
- **HJB Jacobian:** $A_{\text{HJB}}$ (gradient stencil)
- **FP Operator:** $A_{\text{FP}} = A_{\text{HJB}}^\top$ (divergence stencil via transpose)

#### Why Gradient Schemes Violate Mass Conservation for FP

Using `gradient_upwind` for FP discretizes $\mathbf{b} \cdot \nabla m$ instead of $\nabla \cdot (\mathbf{b} m)$:

$$
\mathbf{b} \cdot \nabla m \neq \nabla \cdot (\mathbf{b} m)
$$

The difference is $m \nabla \cdot \mathbf{b}$ (product rule). This treats density $m$ as a passive scalar rather than a conserved quantity:

$$
\frac{d}{dt} \int_\Omega m \, dx \neq 0 \quad \text{(mass not conserved)}
$$

#### Matching Upwind vs. Centered

**Upwind ↔ Upwind:**
- In HJB, "upwind" means looking in the direction the agent wants to move to update $u$
- In FP, "upwind" means moving mass $m$ in that same direction
- If HJB uses a **backward** difference, the transpose becomes a **forward** flux difference
- Divergence upwind implementations handle this automatically

**Centered ↔ Centered:**
- Central differences for HJB transpose to central differences for FP
- Only stable when viscosity dominates advection

### 8.5 Usage Examples

**Recommended for most MFG problems:**

```python
# Stability-first: upwind pairing
hjb_solver = HJBFDMSolver(problem, advection_scheme="gradient_upwind")
fp_solver = FPFDMSolver(problem, advection_scheme="divergence_upwind")
```

**For smooth problems where accuracy matters:**

```python
# Accuracy-first: centered pairing
hjb_solver = HJBFDMSolver(problem, advection_scheme="gradient_centered")
fp_solver = FPFDMSolver(problem, advection_scheme="divergence_centered")
```

### 8.6 Consequences of Incorrect Scheme Pairing

What happens if the FDM scheme pairing violates the mathematical structure? This section analyzes the failure modes.

#### Case 1: Gradient Scheme for FP in MFG (Type Error)

**Configuration:** `gradient_upwind` (HJB) + `gradient_upwind` (FP)

**Mathematical Error:**

You discretize the wrong operator:

$$
\text{Correct FP:} \quad \partial_t m + \nabla \cdot (\mathbf{b} m) = 0
$$

$$
\text{What you solve:} \quad \partial_t m + \mathbf{b} \cdot \nabla m = 0
$$

These differ by the product rule:

$$
\nabla \cdot (\mathbf{b} m) = \mathbf{b} \cdot \nabla m + m (\nabla \cdot \mathbf{b})
$$

**Consequences:**

1. **Mass conservation violated:**
   $$
   \frac{d}{dt} \int_\Omega m \, dx = -\int_\Omega m (\nabla \cdot \mathbf{b}) \, dx \neq 0
   $$
   The total mass drifts over time instead of remaining constant.

2. **Adjoint relationship broken:** $A_{\text{FP}} \neq A_{\text{HJB}}^\top$

3. **Positivity may fail:** Density can become negative in regions where drift is non-solenoidal.

4. **Wrong steady state:** The fixed-point iteration may converge, but to a solution of the wrong PDE.

**Observable Symptoms:**
- Total mass $\sum_i m_i$ drifts from 1.0 over Picard iterations
- Negative density values appear
- Solution doesn't match analytical benchmarks
- **Insidious:** Code runs without errors, producing plausible-looking but incorrect results

#### Case 2: Mismatched Stencils (Upwind + Centered)

**Configuration:** `gradient_upwind` (HJB) + `divergence_centered` (FP)

**Mathematical Error:**

Both operators discretize the correct PDE form, but the discrete adjoint relationship is broken:

$$
A_{\text{FP}}^{\text{centered}} \neq (A_{\text{HJB}}^{\text{upwind}})^\top
$$

The upwind HJB Jacobian has the structure:

$$
(A_{\text{HJB}})_{i,i-1} = -\frac{\nu}{h^2} - \frac{a_i^-}{h}, \quad
(A_{\text{HJB}})_{i,i+1} = -\frac{\nu}{h^2} - \frac{a_i^+}{h}
$$

Its transpose would give an upwind divergence operator. But centered divergence has:

$$
(A_{\text{FP}}^{\text{centered}})_{i,i-1} = -\frac{\nu}{h^2} - \frac{a_{i-1}}{2h}, \quad
(A_{\text{FP}}^{\text{centered}})_{i,i+1} = -\frac{\nu}{h^2} + \frac{a_{i+1}}{2h}
$$

**Consequences:**

1. **Coupled Jacobian loses saddle-point structure:**
   $$
   J = \begin{pmatrix} A_{\text{HJB}} & -B \\ C & A_{\text{FP}} \end{pmatrix}
   $$
   where $A_{\text{FP}} \neq A_{\text{HJB}}^\top$. The off-diagonal symmetry required for efficient solution is lost.

2. **Newton convergence degraded:** Quadratic convergence may reduce to linear or fail entirely.

3. **Fixed-point iteration unstable:** The spectral radius of the iteration operator increases.

**Observable Symptoms:**
- Picard iteration requires many more iterations (10× or more)
- Residual oscillates instead of decreasing monotonically
- Newton method fails to converge or requires very small damping
- Solution quality degrades as coupling strength increases

#### Case 3: Divergence Scheme for HJB (Structural Impossibility)

**Configuration:** Attempting `divergence_upwind` for HJB

**Mathematical Error:**

You're solving a fundamentally different equation:

$$
\text{Correct HJB:} \quad \partial_t u + \mathbf{b} \cdot \nabla u = 0 \quad \text{(advection of value)}
$$

$$
\text{What you solve:} \quad \partial_t u + \nabla \cdot (\mathbf{b} u) = 0 \quad \text{(conservation of "value flux")}
$$

But value is **not** a conserved quantity. The value function $u$ represents the cost-to-go; it doesn't "flow" through the domain like mass does.

**Consequences:**

1. **Wrong PDE entirely:** The computed $u$ satisfies a conservation law that has no physical meaning for optimal control.

2. **Optimal control is wrong:**
   $$
   \alpha^* = -\nabla_p H(x, \nabla u)
   $$
   Since $u$ is wrong, the gradient $\nabla u$ is wrong, and the control $\alpha^*$ sends agents in incorrect directions.

3. **Boundary/terminal conditions violated:** The conservation form changes how boundary data propagates.

**Observable Symptoms:**
- Value function has incorrect shape and magnitude
- Terminal condition $u(T, x) = G(x)$ is not satisfied accurately
- Optimal trajectories don't reach intended targets
- **This error is usually caught:** most implementations don't offer divergence schemes for HJB

#### Summary: Failure Mode Classification

| Mismatch Type | Adjoint $A_{\text{FP}} = A_{\text{HJB}}^\top$? | Mass Conserved? | Convergence | Solution |
|:--------------|:----------------------------------------------|:----------------|:------------|:---------|
| `gradient` + `gradient` | **Broken** | **No** | May converge | **Wrong PDE** |
| `upwind` + `centered` | **Broken** | Yes | Slow/unstable | Degraded |
| `centered` + `upwind` | **Broken** | Yes | Slow/unstable | Degraded |
| `divergence` for HJB | N/A | N/A | N/A | **Wrong PDE** |
| ✓ `gradient_upwind` + `divergence_upwind` | **Preserved** | **Yes** | Fast | Correct |
| ✓ `gradient_centered` + `divergence_centered` | **Preserved** | **Yes** | Fast | Correct |

#### The Insidious Nature of Case 1

Case 1 (`gradient` + `gradient`) is the most dangerous because:

1. **No runtime errors:** Code executes successfully
2. **Plausible output:** Solutions look reasonable at first glance
3. **Iterations converge:** Fixed-point may reach a steady state
4. **But the answer is wrong:** You've solved a different PDE

The only way to detect this error is:
- Check mass conservation: $|\sum_i m_i^{(k)} - 1| > \epsilon$ growing over iterations
- Compare against analytical solutions (when available)
- Verify against known benchmarks

### 8.7 Current Implementation Status

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
