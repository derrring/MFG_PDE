# [WIP] Periodic Boundary Conditions (PBC) Implementation in MFG_PDE

**Status**: [WIP] - Under Development and Review
**Version**: 0.4 (Added compatibility requirements)
**Date**: 2026-02-04
**Last Updated**: 2026-02-04 (Section 4.1 added based on exp06b validation)

## 1. Introduction

Periodic Boundary Conditions (PBCs) play a crucial role in many physics and engineering problems, especially when simulating infinite periodic systems or eliminating artificial boundary effects. In the field of Mean Field Games (MFG), PBCs are also widely used to model the behavior of particles, crowds, or agents in macroscopically large or infinite spaces.

This report aims to detail the implementation strategy and architectural considerations for PBCs within the `MFG_PDE` library, with the goal of building a solution that is both general-purpose and high-performance. We will analyze the topological nature of PBCs, explore the two core implementation approaches—real space and Fourier space—and propose a "three-tier" implementation architecture. We will also highlight potential "performance killers" and "logical pitfalls" in numerical implementation.

## 2. The Topological Nature and Numerical Differences of PBCs

### 2.1 Conceptual Model

The core of PBCs lies in their topological nature: they do not impose local constraints on the domain's edge but rather "stitch" opposite boundaries together to form a closed manifold (e.g., a 1D line segment becomes a circle; a 2D rectangle becomes a torus). This topological identification leads to fundamental differences in their numerical implementation compared to traditional boundary conditions (e.g., Dirichlet, Neumann, Robin).

*   **Traditional BCs**: Operate on a **bounded domain** by imposing **local constraints** at the geometric boundaries.
*   **PBCs**: Operate on a **topologically closed domain** by **identifying** opposite boundaries, thus changing the global connectivity of the domain.

### 2.2 Discretization Scheme and Matrix Structure

This topological property is directly reflected in the algebraic structure of the discretized differential operators. Consider the 1D Laplacian operator $\Delta u = \frac{\partial^2 u}{\partial x^2}$ with a second-order central difference scheme on a uniform grid $x_i = i \cdot \Delta x$:

*   **Interior Points ($0 < i < N-1$):**
    $$ (\Delta u)_i \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{(\Delta x)^2} $$
    This corresponds to the `[..., 1, -2, 1, ...]` structure in row `i` of the matrix `A`.

*   **Dirichlet Boundary ($u_0=u_N=0$):**
    At $i=1$, since $u_0=0$, the formula becomes $\frac{u_2 - 2u_1}{(\Delta x)^2}$. The resulting matrix `A` has a typical **Tridiagonal Band Matrix** structure.

*   **Periodic Boundary (PBC):**
    At the boundary points $i=0$ and $i=N-1$, neighbor indices "wrap around" due to topological identification:
    *   The left neighbor of $i=0$ is $i=N-1$.
    *   The right neighbor of $i=N-1$ is $i=0$.
    
    The discretization formulas become:
    $$ (\Delta u)_0 \approx \frac{u_1 - 2u_0 + u_{N-1}}{(\Delta x)^2} $$
    $$ (\Delta u)_{N-1} \approx \frac{u_0 - 2u_{N-1} + u_{N-2}}{(\Delta x)^2} $$

    This introduces non-zero elements in the top-right and bottom-left corners of the matrix `A`, forming a **Circulant Matrix**:
    $$ A \propto \begin{pmatrix}
    -2 & 1 & 0 & \cdots & 1 \\
    1 & -2 & 1 & \cdots & 0 \\
    0 & 1 & -2 & \cdots & 0 \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    1 & 0 & \cdots & 1 & -2
    \end{pmatrix} $$
    This fundamental difference in algebraic structure is the primary reason why PBCs must be treated distinctly from other BCs in implementation.

## 3. PBC Implementation Architecture in MFG_PDE: The "Three-Tier" Model

To balance generality, robustness, and performance, `MFG_PDE` will adopt a tiered architecture for handling PBCs.

### 3.1. Foundational Tier: Real Space Topology

This is the core and foundation for handling PBCs in `MFG_PDE`, providing the broadest generality.

#### **Implementation Strategy: Ghost Cells (Halo Regions)**

Ghost cells are key to enabling high-performance parallel computing by transforming a topological problem into a data synchronization problem.

**Engineering Practice: The Ghost Cell Workflow**
```
// Pseudocode: Applying ghost cells in a computation step

// 1. Define Data Structures
// The extended_domain includes the real domain plus ghost cells on each side.
// e.g., for a 1D domain of size N and G ghost cells on each side:
// total_size = N + 2*G
real_field[N];
extended_field[N + 2*G];

// 2. Halo Exchange / Ghost Cell Synchronization
// This must be performed before any operation that requires neighbor interaction, like computing derivatives.
function synchronize_ghost_cells(extended_field):
    // Assuming G=1 for simplicity
    // Populate ghost cells from the boundaries of the real domain
    extended_field[G-1] = extended_field[N+G-1]; // left ghost cell = real right boundary
    extended_field[N+G] = extended_field[G];     // right ghost cell = real left boundary
    
    // In parallel computing (MPI), this step involves MPI_Sendrecv operations
    // to exchange boundary data with neighbor processes to fill the ghost cells.

// 3. Computation (e.g., calculating a derivative)
function compute_derivative(extended_field):
    // The loop only iterates over the real domain [G, N+G-1].
    // However, the stencil can safely read values from the ghost cells.
    for i from G to N+G-1:
        derivative[i-G] = (extended_field[i+1] - extended_field[i-1]) / (2*dx);
    return derivative;

// 4. Main Loop
for t in timesteps:
    synchronize_ghost_cells(current_field);
    new_field = compute_step(current_field);
    current_field = new_field;
```

#### **Manifold-Aware Metric Tensor**

To support non-orthogonal coordinate systems (e.g., oblique periodicity), the `Grid` object must carry metric information. The Laplacian operator generalizes to the Laplace-Beltrami operator:
$$ \Delta_g f = \frac{1}{\sqrt{\det(g)}} \partial_i \left( \sqrt{\det(g)} g^{ij} \partial_j f \right) $$
where $g^{ij}$ is the inverse of the metric tensor $g$. The operator discretization module must use this formula instead of simple Cartesian differences to achieve true geometric generality.

### 3.2. High-Performance Specialization Tier: Fourier Space / FFT

This tier provides extreme performance for specific, compatible problems, acting as a specialized engine.

#### **Implementation Strategy: Pseudo-spectral Method**

The pseudo-spectral method is key to handling non-linear terms. For an HJB equation like $\partial_t u = -H(\nabla u) + \nu \Delta u$:

**Algorithm Flow: Solving HJB with the Pseudo-spectral Method**
1.  **Initialize**: Given the values of $u(t)$ in real space.
2.  **FFT**: Compute the Fourier representation $\hat{u}(k) = \mathcal{F}(u(x))$.
3.  **Compute Linear Term**: The diffusion term is a simple multiplication in Fourier space:
    $$ \mathcal{F}(\nu \Delta u) = -\nu |k|^2 \hat{u}(k) $$
4.  **Compute Non-linear Term ($H(\nabla u)$)**:
    a.  **Calculate Gradient**: Compute the Fourier representation of the gradient:
        $$ \widehat{\nabla u}(k) = i k \hat{u}(k) $$
    b.  **Inverse Transform (IFFT)**: Transform $\widehat{\nabla u}(k)$ back to real space to get $\nabla u(x) = \mathcal{F}^{-1}(i k \hat{u}(k))$.
    c.  **Real Space Computation**: Compute the non-linear term $H(\nabla u(x))$ point-wise in real space.
    d.  **Forward Transform (FFT)**: Transform the result $H(\nabla u(x))$ back to Fourier space to get $\widehat{H}(k) = \mathcal{F}(H(\nabla u(x)))$.
5.  **Time Stepping**: Update $\hat{u}$ in Fourier space:
    $$ \hat{u}(t+\Delta t) = \text{TimeStepper}(\hat{u}(t), \Delta t, -\widehat{H}(k) - \nu |k|^2 \hat{u}(k)) $$
    (e.g., using simple Euler stepping or a more stable Runge-Kutta method).
6.  **Inverse Transform**: If needed, transform $\hat{u}(t+\Delta t)$ back to real space to get $u(t+\Delta t)$.

#### **Engineering Practice: Mandatory Anti-Aliasing**

**The 3/2 Rule**:
1.  **Padding**: Assume the original grid has size `N`, with `N` Fourier modes. Embed the Fourier coefficient array $\hat{u}$ into a larger array of size `N' = 3/2 * N`. The high-frequency portion (from `N/2` to `N'/2`）is padded with zeros.
2.  **Transform**: Perform an IFFT on this zero-padded array of size `N'` to get a real-space field of size `N'`.
3.  **Compute**: Calculate the non-linear product in this larger real-space field.
4.  **Transform**: Perform an FFT of size `N'` on the result.
5.  **Truncate**: From the `N'` Fourier coefficients, retain only the original `N` low-frequency modes, discarding the high-frequency part. The discarded modes are where aliasing errors would have contaminated the signal.

### 3.3. User Interface / Factory Tier

This tier provides a clean and convenient interface for users to select and configure how PBCs are handled.

*   **Principle**: Adhere to the API design principles in `CLAUDE.md`, providing a clear, unified entry point that allows for explicit or automatic selection of the best solution strategy.
*   **Implementation**: The `problem.solve()` method will support a `method` parameter:
    *   `method="auto"`: The default. A factory function will intelligently inspect the `MFGProblem` properties (grid type, boundary conditions, PDE linearity) and dispatch to the most appropriate solver (`SpectralSolver` if compatible, or the general real-space solver otherwise).
    *   `method="fdm"` (or other specific methods): Forces the use of the general-purpose solver based on real-space topology (FDM, FEM, etc.).
    *   `method="spectral"`: Forces the use of the Fourier-based `SpectralSolver`. It should raise a clear error if the problem is not compatible.

## 4. Key Numerical Considerations and Pitfalls

### 4.1 Physical Model Compatibility with Periodic Topology (CRITICAL)

**Issue Discovered**: 2026-02-04 (exp06b_periodic EOC validation)

When using periodic boundary conditions, the physical model (potential, cost functions) must be **topologically compatible** with the periodic domain. A common mistake is using a quadratic potential on a periodic domain.

#### The Problem: Quadratic Potential on Periodic Domain

Consider the standard crowd evacuation potential:
$$V_{quad}(x) = C_1 \left[(x - L/2)^2 + (y - L/2)^2\right]$$

**Gradient discontinuity at boundary**:
- At $x = 0$: $\partial V/\partial x = -C_1 L$
- At $x = L$ (same point topologically): $\partial V/\partial x = +C_1 L$

This creates a **gradient jump of $2C_1 L$** at the periodic boundary, causing:
1. GFDM stencils to produce extreme gradient values near boundaries
2. Numerical instability in HJB time-stepping
3. Large U_diff values even with gradient clamping

**Validation evidence** (exp06b_periodic):
- GFDM gradient of linear function $U = x + y$: max error **7.77** (should be 0)
- GFDM gradient of periodic function $\cos(2\pi x/L)$: max error **0.14** (converges correctly)

#### The Solution: Periodic-Compatible Potential

For truly periodic MFG problems, use a potential with **continuous derivatives at boundaries**:

$$V_{periodic}(x) = \frac{C_1}{2}\left[1 + \cos\left(\frac{2\pi x}{L}\right)\right] + \frac{C_1}{2}\left[1 + \cos\left(\frac{2\pi y}{L}\right)\right]$$

**Properties**:
- $V_{min} = 0$ at center $(L/2, L/2)$
- $V_{max} = 2C_1$ at corners $(0,0), (L,0), (0,L), (L,L)$
- Gradient $\partial V/\partial x = -\frac{C_1 \pi}{L} \sin(2\pi x/L)$ is **continuous everywhere**
- At boundaries: $\partial V/\partial x|_{x=0} = \partial V/\partial x|_{x=L} = 0$

**Equilibrium density** (Boltzmann-Gibbs):
$$m^*(x) = \frac{1}{Z} \exp\left(-\frac{V_{periodic}(x)}{T_{eff}}\right)$$
where $Z = \iint \exp(-V/T_{eff}) \, dx\,dy$ and $T_{eff} = C_2 + \sigma^2/2$.

#### Decision Guide

| Physical Setup | Recommended BC | Potential Form |
|----------------|----------------|----------------|
| Bounded domain with walls | No-flux (Neumann) | Quadratic $V \propto r^2$ |
| Infinite periodic lattice | Periodic | Cosine $V \propto \cos(kx)$ |
| Torus topology | Periodic | Cosine $V \propto \cos(kx)$ |

**Rule of thumb**: If using periodic BC, all physical quantities (potential, running cost, terminal cost) must be periodic functions with continuous derivatives.

### 4.2 The Zero-Frequency Mode and Solution Uniqueness

**Mathematical Origin**: For pure Neumann or periodic boundaries, the null space of the Laplacian operator is non-trivial (it contains the constant functions). This means its discrete matrix `A` is singular (has a zero eigenvalue), and the linear system `Au=f` may not have a unique solution.

**Engineering Countermeasures**:
1.  **Constrained Solver (Lagrange Multiplier)**:
    Replace the original problem `Au=f` with a larger, non-singular saddle-point problem. For instance, to enforce the constraint $\sum u_i = 0$, we solve:
    $$ \begin{pmatrix}
    A & \mathbf{1} \\
    \mathbf{1}^T & 0
    \end{pmatrix}
    \begin{pmatrix}
    u \\
    \lambda
    \end{pmatrix}
    = \begin{pmatrix}
    f \\
    0
    \end{pmatrix} $$
    where $\mathbf{1}$ is a vector of all ones.

2.  **Projection Method**:
    First, ensure the solvability condition for `f` is met (i.e., `f` is orthogonal to the null space, $\sum f_i = 0$). If not, project `f` onto the range of `A`: $f' = f - \text{mean}(f)$。Then, solve `Au=f'` using an iterative solver that can handle semi-definite systems (like Conjugate Gradients). Finally, ensure the solution is unique by projecting it onto the space orthogonal to the null space: $u' = u - \text{mean}(u)$.

3.  **Regularization**:
    Replace the matrix `A` with a perturbed version `A' = A + \epsilon I`, where `\epsilon` is a small positive number. This makes the matrix non-singular but introduces an error of `O(\epsilon)`. This is not preferred for production but can be useful for rapid prototyping.

4.  **For `SpectralSolver`**:
    In Fourier space, the zero-frequency mode corresponds to the $k=0$ coefficient, $\hat{u}(0)$. When solving the equivalent of `Au=f`, which is $\hat{A}(k)\hat{u}(k)=\hat{f}(k)$, the case $k=0$ must be handled separately since $\hat{A}(0)=0$. If $\hat{f}(0) \neq 0$, there is no solution. If $\hat{f}(0) = 0$, then $\hat{u}(0)$ can be any value. It is conventional to set $\hat{u}(0)=0$, which is equivalent to enforcing a zero mean on the solution in real space.

... (Remaining sections on `Evolution Path` and `References` would be translated similarly) ...

## 5. Core Evolution Path for PBC in MFG_PDE

This architectural report establishes the **"Topology First, Performance Specialized"** roadmap. The evolution of PBC support in `MFG_PDE` will follow these steps:

1.  **Phase 1: Foundational Bedrock (Real Space Topology)**
    *   Refine the `Grid` object's topology management to **fully support Ghost-Cell-based periodic boundaries**, ensuring interfaces like `get_neighbors(index)` can handle the wrap-around logic seamlessly.
    *   Incorporate the **manifold-aware metric tensor ($g^{ij}$)** into the `Grid` or its coordinate system to support more general periodic geometries (like oblique periodicity).
    *   Ensure the FDM-based real-space solver can robustly handle various PBC test cases.
    *   Implement a **proper handling mechanism for the zero-frequency mode** (e.g., mean value constraint) in the real-space solver's matrix assembly and solving stages.

2.  **Phase 2: High-Performance Specialization (Fourier Space)**
    *   Implement a `SpectralSolver` based on the **pseudo-spectral method** for handling linear periodic PDEs.
    *   Mandatorily integrate **anti-aliasing techniques** (like the 3/2 rule) and **explicit handling of the zero-frequency mode** into the `SpectralSolver`.

3.  **Phase 3: Intelligent Integration and User Experience (Factory API)**
    *   Implement the **`method="auto"` logic** in the factory layer (`problem.solve()`) to intelligently route to the most appropriate solver based on problem characteristics.
    *   Provide clear documentation and error messages to guide users in selecting and debugging PBC-related issues.

By following this architecture and its implementation details, `MFG_PDE` will offer powerful, flexible, and high-performance handling of periodic boundary conditions to meet the demands of Mean Field Game research and applications.

---

### References

*   [1] Boyd, J. P. (2001). Chebyshev and Fourier Spectral Methods. Courier Corporation.
*   [2] Trefethen, L. N. (2000). Spectral Methods in MATLAB. SIAM.
*   [3] Canuto, C., et al. (2006). Spectral Methods: Fundamentals in Single Domains. Springer Science & Business Media.
