# Semi-Lagrangian Methods for HJB Equations in Mean Field Games

**Date**: 2025-11-12
**Status**: Technical Reference
**Related Issues**: #298
**Implementation**: `mfg_pde/alg/numerical/hjb_solvers/hjb_semi_lagrangian.py`

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Framework](#mathematical-framework)
3. [Explicit Semi-Lagrangian Method](#explicit-semi-lagrangian-method)
4. [Implicit Semi-Lagrangian Method](#implicit-semi-lagrangian-method)
5. [Semi-Implicit Variants](#semi-implicit-variants)
6. [Stability Analysis](#stability-analysis)
7. [Implementation Considerations](#implementation-considerations)
8. [References](#references)

## Introduction

Semi-Lagrangian (SL) methods solve advection-dominated PDEs by following characteristics backward in time. For Mean Field Games (MFG), the Hamilton-Jacobi-Bellman (HJB) equation has the form:

$$
-\frac{\partial u}{\partial t} + H(x, \nabla u, m) - \frac{\sigma^2}{2} \Delta u = 0, \quad (t,x) \in [0,T] \times \Omega
$$

with terminal condition $u(T, x) = g(x)$ and appropriate boundary conditions.

The key advantage of SL methods is **relaxed CFL conditions** compared to explicit finite difference schemes, enabling larger time steps.

## Mathematical Framework

### 1. Hamiltonian and Optimal Control

For standard MFG with quadratic control cost:

$$
H(x, p, m) = \frac{|p|^2}{2} + f(x, m)
$$

where:
- $p = \nabla u$ is the costate variable
- $f(x, m)$ is the running cost (congestion, interaction)

The optimal control satisfies:

$$
\alpha^*(x, t) = \arg\max_{\alpha} \left[ -p \cdot \alpha - \frac{|\alpha|^2}{2} \right] = -p = -\nabla u
$$

### 2. Method of Characteristics

The HJB equation can be written along characteristics $X(s; t, x)$ satisfying:

$$
\frac{dX}{ds} = \frac{\partial H}{\partial p}(X, \nabla u(s, X), m(s, X)) = p = \nabla u
$$

with $X(t; t, x) = x$.

Along characteristics, $u$ satisfies:

$$
\frac{du}{ds}(s, X(s)) = -H(X, \nabla u, m) + \frac{\sigma^2}{2} \Delta u
$$

### 3. Backward Time Discretization

Discretize time: $t_n = n \Delta t$, $n = 0, \ldots, N_t$ with $t_{N_t} = T$.

For each grid point $x_i$ at time $t_{n+1}$, we trace the characteristic **backward** to time $t_n$:

$$
X_i^n = x_i - \nabla u^{n+1}(x_i) \cdot \Delta t + O(\Delta t^2)
$$

Then update:

$$
u^n(x_i) = u^{n+1}(X_i^n) - \Delta t \left[ H(x_i, \nabla u^{n+1}, m^{n+1}) - \frac{\sigma^2}{2} \Delta u^{n+1} \right] + O(\Delta t^2)
$$

## Explicit Semi-Lagrangian Method

### Algorithm

**Given**: $u^{n+1}(x)$, $m^{n+1}(x)$ at time $t_{n+1}$

**Goal**: Compute $u^n(x_i)$ for all grid points $x_i$

**Steps**:

1. **Compute gradient** from known solution:
   $$
   p_i^{n+1} = \nabla u^{n+1}(x_i)
   $$

2. **Trace characteristic** backward (explicit):
   $$
   X_i^n = x_i - p_i^{n+1} \cdot \Delta t
   $$

3. **Interpolate** at departure point:
   $$
   \tilde{u}_i^n = \mathcal{I}[u^{n+1}](X_i^n)
   $$
   where $\mathcal{I}$ is an interpolation operator (linear, cubic, etc.)

4. **Apply diffusion** (operator splitting):
   $$
   u_i^n = \tilde{u}_i^n - \Delta t \left[ H(x_i, p_i^{n+1}, m_i^{n+1}) - \frac{\sigma^2}{2} \Delta u^{n+1}(x_i) \right]
   $$

### Gradient Computation

For 1D problems:
$$
\nabla u^{n+1}(x_i) = \begin{cases}
\frac{u_{i+1}^{n+1} - u_i^{n+1}}{\Delta x} & i = 0 \text{ (forward)} \\
\frac{u_{i+1}^{n+1} - u_{i-1}^{n+1}}{2\Delta x} & 0 < i < N_x \text{ (central)} \\
\frac{u_i^{n+1} - u_{i-1}^{n+1}}{\Delta x} & i = N_x \text{ (backward)}
\end{cases}
$$

For nD problems:
$$
\nabla u^{n+1} = \left( \frac{\partial u}{\partial x_1}, \ldots, \frac{\partial u}{\partial x_d} \right)
$$

using `np.gradient(u, dx, axis=k, edge_order=2)` for each dimension $k$.

### CFL Condition

The explicit method requires:

$$
\text{CFL} = \frac{\max_i |\nabla u^{n+1}(x_i)| \cdot \Delta t}{\Delta x} < C_{\text{cfl}}
$$

where $C_{\text{cfl}} \approx 1$ for stability. If violated, characteristics jump multiple grid cells, causing interpolation errors and potential instability.

### Stability Enhancement: Gradient Clipping

To prevent CFL violation, optionally clip gradients:

$$
\tilde{p}_i = \begin{cases}
p_i & |p_i| \leq p_{\max} \\
p_{\max} \frac{p_i}{|p_i|} & |p_i| > p_{\max}
\end{cases}
$$

This introduces artificial control bounds but ensures numerical stability.

**Trade-off**:
- **Pros**: Unconditionally prevents CFL violation
- **Cons**: Not mathematically rigorous for unbounded control problems

### Pseudocode

```python
def explicit_semi_lagrangian_step(u_next, m_next, dt, dx):
    """
    Explicit semi-Lagrangian for one time step.

    Args:
        u_next: Solution at t^{n+1}, shape (Nx,)
        m_next: Density at t^{n+1}, shape (Nx,)
        dt: Time step
        dx: Spatial step

    Returns:
        u_current: Solution at t^n, shape (Nx,)
    """
    # 1. Compute gradient
    grad_u = np.gradient(u_next, dx, edge_order=2)

    # 2. Optional: Check CFL and clip
    cfl = np.max(np.abs(grad_u)) * dt / dx
    if cfl > 1.0:
        logger.warning(f"CFL = {cfl:.2f} > 1.0")
    if max_gradient is not None:
        grad_u = np.clip(grad_u, -max_gradient, max_gradient)

    # 3. Trace characteristics and interpolate
    u_star = np.zeros_like(u_next)
    for i in range(len(x_grid)):
        x_departure = x_grid[i] - grad_u[i] * dt
        u_star[i] = interpolate(u_next, x_departure)

        # 4. Apply Hamiltonian and diffusion
        H_value = 0.5 * grad_u[i]**2 + f(x_grid[i], m_next[i])
        diffusion = laplacian(u_next, i, dx)
        u_star[i] -= dt * (H_value - 0.5 * sigma**2 * diffusion)

    return u_star
```

## Implicit Semi-Lagrangian Method

### Formulation

The implicit method solves for $u^n$ such that characteristics are computed using the **unknown** gradient $\nabla u^n$:

$$
X_i^n = x_i - \nabla u^n(x_i) \cdot \Delta t
$$

This leads to a **nonlinear system**:

$$
F_i(u^n) = u_i^n - \mathcal{I}[u^{n+1}](x_i - \nabla u^n(x_i) \cdot \Delta t) + \Delta t H(\ldots) = 0
$$

for $i = 1, \ldots, N_x$.

### Fixed-Point Iteration

Solve by iterating:

$$
u^{n,k+1}_i = \mathcal{I}[u^{n+1}](x_i - \nabla u^{n,k}_i \cdot \Delta t) - \Delta t H(\ldots)
$$

until $\|u^{n,k+1} - u^{n,k}\| < \epsilon$.

**Convergence criterion**:
$$
\|u^{n,k+1} - u^{n,k}\|_\infty < \epsilon_{\text{tol}}
$$

or maximum iterations reached.

### Algorithm

**Given**: $u^{n+1}(x)$, $m^{n+1}(x)$

**Initialize**: $u^{n,0} = u^{n+1}$ (or better initial guess)

**Iterate** for $k = 0, 1, 2, \ldots$ until convergence:

1. **Compute gradient** from current estimate:
   $$
   p_i^{n,k} = \nabla u^{n,k}(x_i)
   $$

2. **Trace characteristic** using current gradient:
   $$
   X_i^{n,k} = x_i - p_i^{n,k} \cdot \Delta t
   $$

3. **Update solution**:
   $$
   u_i^{n,k+1} = \mathcal{I}[u^{n+1}](X_i^{n,k}) - \Delta t H(x_i, p_i^{n,k}, m_i^{n+1})
   $$

4. **Check convergence**:
   $$
   \text{if } \|u^{n,k+1} - u^{n,k}\|_\infty < \epsilon_{\text{tol}}, \text{ stop}
   $$

**Output**: $u^n = u^{n,k+1}$

### Pseudocode

```python
def implicit_semi_lagrangian_step(u_next, m_next, dt, dx, max_iter=10, tol=1e-6):
    """
    Implicit semi-Lagrangian with fixed-point iteration.

    Args:
        u_next: Solution at t^{n+1}
        m_next: Density at t^{n+1}
        dt: Time step
        dx: Spatial step
        max_iter: Maximum fixed-point iterations
        tol: Convergence tolerance

    Returns:
        u_current: Solution at t^n
    """
    # Initial guess
    u_current = u_next.copy()

    for iteration in range(max_iter):
        u_old = u_current.copy()

        # 1. Compute gradient from current estimate
        grad_u = np.gradient(u_current, dx, edge_order=2)

        # 2. Trace characteristics
        u_new = np.zeros_like(u_next)
        for i in range(len(x_grid)):
            x_departure = x_grid[i] - grad_u[i] * dt

            # 3. Interpolate and update
            u_new[i] = interpolate(u_next, x_departure)
            H_value = 0.5 * grad_u[i]**2 + f(x_grid[i], m_next[i])
            u_new[i] -= dt * H_value

        u_current = u_new

        # 4. Check convergence
        residual = np.linalg.norm(u_current - u_old, np.inf)
        if residual < tol:
            logger.info(f"Converged in {iteration+1} iterations")
            break
    else:
        logger.warning(f"Did not converge in {max_iter} iterations")

    return u_current
```

### Newton's Method

For faster convergence, solve $F(u^n) = 0$ using Newton's method:

$$
u^{n,k+1} = u^{n,k} - J^{-1}(u^{n,k}) F(u^{n,k})
$$

where the Jacobian is:

$$
J_{ij} = \frac{\partial F_i}{\partial u_j^n} = \delta_{ij} + \Delta t \frac{\partial}{\partial u_j^n} \left[ \mathcal{I}[u^{n+1}](X_i^n) \right]
$$

Computing $J$ requires derivatives of the interpolation operator, which is complex.

**Approximation**: Use finite differences for Jacobian-vector products in Krylov methods (GMRES).

## Semi-Implicit Variants

### 1. Extrapolated Gradient Method

Use gradient extrapolation to avoid fixed-point iteration:

$$
\nabla u^{n}_{\text{pred}} = 2 \nabla u^{n+1} - \nabla u^{n+2}
$$

Then solve explicitly:

$$
X_i^n = x_i - \nabla u^{n}_{\text{pred}}(x_i) \cdot \Delta t
$$

$$
u_i^n = \mathcal{I}[u^{n+1}](X_i^n) - \Delta t H(x_i, \nabla u^{n}_{\text{pred}}, m_i^{n+1})
$$

**Properties**:
- **Cost**: Single evaluation per time step (like explicit)
- **Stability**: Better than explicit, worse than implicit
- **Accuracy**: Second-order in time if extrapolation is accurate

### 2. Predictor-Corrector Scheme

**Predictor** (explicit):
$$
u^{n,*} = \text{Explicit-SL}(u^{n+1}, m^{n+1})
$$

**Corrector** (implicit with predictor):
$$
\nabla u^{n}_{\text{avg}} = \frac{1}{2}(\nabla u^{n,*} + \nabla u^{n+1})
$$

$$
u^n = \text{Explicit-SL}(u^{n+1}, m^{n+1}, \text{grad} = \nabla u^{n}_{\text{avg}})
$$

**Properties**: Improved stability with moderate cost increase.

## Stability Analysis

### Linear Stability Analysis

Consider the linear advection equation:

$$
\frac{\partial u}{\partial t} + a \frac{\partial u}{\partial x} = 0
$$

with constant advection speed $a = |\nabla u|$.

#### Explicit SL

The amplification factor for Fourier mode $\exp(i k x)$ is:

$$
G = \exp(-i k a \Delta t)
$$

which is **bounded** ($|G| = 1$) for all $k, \Delta t$, suggesting **unconditional stability**.

However, for **nonlinear** equations with $a = a(u)$, the CFL condition reemerges due to:
1. Interpolation errors when $a \Delta t / \Delta x \gg 1$
2. Gradient estimation errors accumulating

#### Implicit SL

The implicit scheme solves:

$$
u_i^n = u^{n+1}(x_i - a(u_i^n) \Delta t)
$$

This is **unconditionally stable** even for nonlinear $a(u)$ because the characteristic speed is evaluated implicitly.

### Numerical Stability in Practice

| Method | CFL Constraint | Stability | Iterations per Step |
|:-------|:---------------|:----------|:--------------------|
| Explicit SL | $\frac{\|\nabla u\| \Delta t}{\Delta x} < 1$ | Conditional | 1 |
| Explicit SL + Clipping | None (artificial) | Unconditional* | 1 |
| Semi-Implicit | $\frac{\|\nabla u\| \Delta t}{\Delta x} < C$ ($C > 1$) | Relaxed | 1 |
| Implicit SL (Fixed-Point) | None | Unconditional | 3-10 |
| Implicit SL (Newton) | None | Unconditional | 2-5 |

*Stability from clipping, not mathematical analysis.

## Implementation Considerations

### 1. Gradient Computation

**1D Case**:
```python
grad_u = np.gradient(u, dx, edge_order=2)
```

**nD Case**:
```python
grad_components = []
for axis in range(ndim):
    grad_axis = np.gradient(u, spacing[axis], axis=axis, edge_order=2)
    grad_components.append(grad_axis)
grad_u = np.stack(grad_components, axis=0)  # Shape: (d, Nx1, ..., Nxd)
```

**Why `edge_order=2`?**
Ensures second-order accuracy at boundaries using one-sided finite differences.

### 2. Interpolation Methods

**Linear Interpolation**:
- Fast, unconditionally stable
- First-order accurate: $O(\Delta x)$ error

**Cubic Interpolation**:
- Smoother solutions
- Third-order accurate: $O(\Delta x^3)$ error
- May introduce spurious oscillations (Gibbs phenomenon)

**Radial Basis Functions (RBF)**:
- Handles scattered/irregular grids
- Higher cost, better for complex geometries

### 3. Diffusion Treatment

**Explicit Diffusion**:
$$
\Delta u^{n+1}(x_i) \approx \frac{u_{i+1}^{n+1} - 2u_i^{n+1} + u_{i-1}^{n+1}}{\Delta x^2}
$$

Requires CFL: $\frac{\sigma^2 \Delta t}{\Delta x^2} < \frac{1}{2}$ (1D).

**Crank-Nicolson (Recommended)**:
$$
\frac{u^n - u^*}{\Delta t} = \frac{\sigma^2}{2} \left( \Delta u^{n+1} + \Delta u^n \right)
$$

Unconditionally stable, second-order accurate. Requires solving tridiagonal system.

### 4. Boundary Conditions

**Dirichlet**: $u(t, x_{\text{boundary}}) = g(t)$

- Set boundary values directly
- Interpolation clamps to boundary

**Neumann**: $\frac{\partial u}{\partial n}(t, x_{\text{boundary}}) = h(t)$

- Modify gradient computation at boundaries
- Extrapolate characteristics if they exit domain

**Periodic**:
- Wrap-around indexing for interpolation
- Gradient uses periodic differences

### 5. Computational Cost

**Explicit SL**:
- **Per time step**: $O(N_x \cdot d)$ gradient evaluations + $O(N_x)$ interpolations
- **Total**: $O(N_t \cdot N_x \cdot d)$

**Implicit SL (Fixed-Point)**:
- **Per time step**: $k_{\text{iter}} \times O(N_x \cdot d)$ where $k_{\text{iter}} \approx 5$
- **Total**: $O(k_{\text{iter}} \cdot N_t \cdot N_x \cdot d)$

**Trade-off**: Implicit allows larger $\Delta t$, so:
$$
\text{Cost}_{\text{implicit}} \sim \frac{k_{\text{iter}} \cdot N_t^{\text{explicit}}}{\alpha}
$$
where $\alpha = \Delta t_{\text{implicit}} / \Delta t_{\text{explicit}} > 1$.

If $\alpha > k_{\text{iter}}$, implicit is **faster overall**.

## Comparison Summary

| Feature | Explicit SL | Explicit + Clipping | Semi-Implicit | Implicit SL |
|:--------|:------------|:-------------------|:--------------|:------------|
| **CFL Constraint** | Yes | No (artificial) | Relaxed | No |
| **Iterations/Step** | 1 | 1 | 1 | 3-10 |
| **Time Step Size** | Small | Medium | Medium | Large |
| **Accuracy** | $O(\Delta t, \Delta x^p)$ | $O(\Delta t, \Delta x^p)$ | $O(\Delta t^2, \Delta x^p)$ | $O(\Delta t^2, \Delta x^p)$ |
| **Implementation** | Simple | Simple | Medium | Complex |
| **Recommended For** | Small problems | General use | Stiff problems | Very stiff problems |

where $p$ is the interpolation order (1 for linear, 3 for cubic).

## Recommendations

### For MFG_PDE Library

**Current Implementation** (Explicit + Clipping):
- **Use Case**: General MFG problems with moderate gradients
- **Default**: `max_gradient = 10.0`, `check_cfl = True`
- **When to Adjust**:
  - Increase `max_gradient` if physical problem has bounded controls
  - Set `max_gradient = None` and reduce `dt` for unbounded problems

**Future Enhancements**:

1. **Semi-Implicit** (Priority: High)
   - Cost similar to explicit
   - Significant stability improvement
   - Good intermediate solution

2. **Adaptive Time-Stepping**
   ```python
   if cfl > cfl_target:
       dt_new = dt * cfl_target / cfl
       logger.info(f"Reducing dt: {dt} -> {dt_new}")
   ```

3. **Implicit SL** (Priority: Medium)
   - For problems where explicit fails even with clipping
   - Implement fixed-point iteration first
   - Add option: `time_discretization="implicit"`

### Usage Guidelines

**Choose Explicit SL when**:
- Solution is smooth (low $|\nabla u|$)
- Fine spatial grid available
- Computational budget allows small $\Delta t$

**Choose Implicit SL when**:
- Gradients are large or unbounded
- Coarse grid required (computational constraints)
- Long-time simulations ($T \gg 1$)

**Choose Semi-Implicit when**:
- Moderate gradients
- Balance between cost and stability needed

## References

1. **Falcone, M., & Ferretti, R. (2013)**. *Semi-Lagrangian Approximation Schemes for Linear and Hamilton-Jacobi Equations*. SIAM.

2. **Carlini, E., Ferretti, R., & Russo, G. (2005)**. A weighted essentially non-oscillatory, large time-step scheme for Hamilton-Jacobi equations. *SIAM Journal on Scientific Computing*, 27(3), 1071-1091.

3. **Achdou, Y., & Capuzzo-Dolcetta, I. (2010)**. Mean field games: Numerical methods. *SIAM Journal on Numerical Analysis*, 48(3), 1136-1162.

4. **Benamou, J. D., Carlier, G., & Santambrogio, F. (2017)**. Variational mean field games. In *Active Particles, Volume 1* (pp. 141-171). Birkhäuser.

5. **Carlini, E., & Silva, F. J. (2014)**. A semi-Lagrangian scheme for a degenerate second order mean field game system. *Discrete & Continuous Dynamical Systems*, 35(9), 4269-4292.

---

**Document History**:
- 2025-11-12: Initial version (Issue #298 fix)
- Implementation: `HJBSemiLagrangianSolver._compute_gradient()` (line 213)
- Related PR: #300
