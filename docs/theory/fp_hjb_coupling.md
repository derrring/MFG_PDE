# Mean Field Game: Forward-Backward FP-HJB Coupling

**Date**: 2025-10-22
**Purpose**: Explain the correct strategy for solving coupled FP-HJB systems with initial mass and terminal cost
**Audience**: Researchers implementing MFG solvers

---

## 1. Problem Formulation

### 1.1 The Mean Field Game System

A standard Mean Field Game (MFG) consists of two coupled PDEs:

**Hamilton-Jacobi-Bellman (HJB) equation** (backward in time):
```
-∂u/∂t + H(x, ∇u, m) = 0    in [0,T] × Ω
u(T, x) = g(x)               terminal condition
```

**Fokker-Planck (FP) equation** (forward in time):
```
∂m/∂t - σ²Δm + div(m ∇_p H(x, ∇u, m)) = 0    in [0,T] × Ω
m(0, x) = m₀(x)                               initial condition
```

where:
- `u(t,x)`: Value function (cost-to-go from state x at time t)
- `m(t,x)`: Density of agents (probability distribution)
- `g(x)`: Terminal cost (penalizes final position)
- `m₀(x)`: Initial distribution
- `H(x,p,m)`: Hamiltonian (typically H = ½ν|p|² + κF(m))
- `σ`: Diffusion coefficient
- `ν`: Control cost
- `κ`: Congestion cost

### 1.2 Coupling Mechanism

The two equations are **coupled**:
1. HJB depends on m through congestion term F(m)
2. FP depends on u through drift velocity: `v = -∇_p H(·, ∇u, m) = -∇u/ν` (for quadratic Hamiltonian)

### 1.3 Boundary Conditions

Typical boundary conditions:
- **Periodic**: u(t, x+L) = u(t, x), same for m
- **Neumann (no-flux)**: ∂u/∂n = 0, ∂m/∂n = 0
- **Obstacle domains**: Homogeneous Neumann on obstacle boundaries

---

## 2. Time Direction and Causality

### 2.1 Why HJB is Backward

The HJB equation is **backward in time** because:
- `u(t,x)` represents **cost-to-go**: expected cost from state x at time t **until terminal time T**
- Terminal condition `u(T,x) = g(x)` specifies cost at final time
- Dynamic programming principle: solve from T → 0

**Physical interpretation**:
- At T: agent knows terminal cost g(x)
- At t < T: agent plans optimal path from x to minimize total cost until T
- Earlier times depend on later times

### 2.2 Why FP is Forward

The FP equation is **forward in time** because:
- `m(t,x)` represents **probability density**: where agents are at time t
- Initial condition `m(0,x) = m₀(x)` specifies distribution at start
- Conservation of probability: evolve from 0 → T

**Physical interpretation**:
- At t=0: agents start at m₀(x)
- At t > 0: agents move according to optimal policy (derived from u)
- Later times depend on earlier times

### 2.3 Terminal Cost Influences FP Indirectly

**Key insight**: Terminal cost g(x) affects FP **indirectly** through HJB solution!

1. **HJB with terminal condition**: Solve `-∂u/∂t + H(∇u, m) = 0` with `u(T,x) = g(x)`
   - Low g(x) near goal → low u(t,x) near goal at all times
   - Gradient ∇u points **away from low-cost regions** (uphill direction)

2. **Optimal velocity**: `v = -∇u/ν`
   - Points **toward low-cost regions** (downhill direction)
   - Agents naturally attracted to regions where g is LOW

3. **FP evolves with drift**: `∂m/∂t + div(m v) = σ²Δm`
   - Density flows along velocity field v
   - Accumulates near goal where g is LOW

**Example**:
```
Goal at x*: g(x*) = 0
Elsewhere:  g(x) = |x - x*|²

→ u(T, x) small near x*
→ u(t, x) small near x* for all t (by backward HJB)
→ ∇u points away from x*
→ v = -∇u points toward x*
→ m(t, x) flows toward x* (by forward FP)
```

---

## 3. Iterative Coupling Strategy (Fixed-Point Iteration)

Since HJB and FP are coupled, we use **Picard iteration** (fixed-point iteration):

### 3.1 Algorithm Structure

```
Given: m₀(x), g(x), H, σ, T
Initialize: m^(0)(t,x) = m₀(x) for all t (e.g., extend uniformly)

for k = 0, 1, 2, ... until convergence:

    # Step 1: Solve HJB BACKWARD with current m^(k)
    u^(k+1) = solve_HJB_backward(m^(k), g)

    # Step 2: Solve FP FORWARD with current u^(k+1)
    m^(k+1) = solve_FP_forward(u^(k+1), m₀)

    # Step 3: Check convergence
    if ||m^(k+1) - m^(k)|| < tol:
        break
```

### 3.2 Key Principles

1. **HJB always backward**: Terminal condition u(T,x) = g(x) is FIXED
2. **FP always forward**: Initial condition m(0,x) = m₀(x) is FIXED
3. **Terminal cost g enters through HJB**: Never directly in FP
4. **Initial mass m₀ enters through FP**: Never in HJB (except through coupling)

### 3.3 Damping for Stability

To improve convergence, use **damping**:
```
m^(k+1) := (1-α) m^(k) + α m^(k+1)_raw
```
where α ∈ (0,1] is damping parameter (typical: α = 0.5).

Only damp density m, NOT value function u (HJB solution is more stable).

---

## 4. Finite Difference Implementation

### 4.1 Grid Setup

**Spatial discretization**:
```
Domain: Ω = [xmin, xmax]^d
Grid: x_i, i = 1,...,Nx
Spacing: Δx = (xmax - xmin) / (Nx - 1)
```

**Temporal discretization**:
```
Time: t_n = n·Δt, n = 0,...,Nt
Time step: Δt = T / Nt
```

**Arrays**:
- `U[n, i]`: u(t_n, x_i) stored for n = 0,...,Nt (full time history)
- `M[n, i]`: m(t_n, x_i) stored for n = 0,...,Nt (full time history)

### 4.2 HJB Solver (Backward)

**Discretization** of `-∂u/∂t + H(x, ∇u, m) = 0`:

Using **backward Euler** in time (implicit, stable):
```
-(U[n,i] - U[n+1,i])/Δt + H(x_i, ∇U[n,i], M[n,i]) = 0

Rearranging:
U[n,i] = U[n+1,i] - Δt · H(x_i, ∇U[n,i], M[n,i])
```

**Algorithm** (backward sweep from T to 0):
```python
# Terminal condition
U[Nt, :] = g(x)  # Apply terminal cost

# Backward time loop
for n in range(Nt-1, -1, -1):  # n = Nt-1, Nt-2, ..., 0
    for i in range(Nx):
        # Compute gradient
        grad_U = central_difference(U[n+1, :], i, Δx)

        # Hamiltonian
        H_val = 0.5/ν * |grad_U|² + κ * F(M[n, i])

        # Backward Euler update
        U[n, i] = U[n+1, i] - Δt * H_val
```

**Key point**: Start from `U[Nt, :]` (known terminal condition) and work backward to `U[0, :]`.

### 4.3 FP Solver (Forward)

**Discretization** of `∂m/∂t - σ²Δm + div(m ∇_p H) = 0`:

For quadratic Hamiltonian `H = ½ν|p|² + κF(m)`:
- `∇_p H = p/ν = ∇u/ν`
- Drift velocity: `v = -∇_p H = -∇u/ν`

**FP becomes**:
```
∂m/∂t - σ²Δm - div(m ∇u/ν) = 0
∂m/∂t = σ²Δm + div(m ∇u/ν)
```

Using **forward Euler** in time:
```
(M[n+1,i] - M[n,i])/Δt = σ² Δm[n,i] + div(m[n] ∇u[n]/ν)[i]

Rearranging:
M[n+1, i] = M[n, i] + Δt · (σ² Δm[n,i] + div(m[n] ∇u[n]/ν)[i])
```

**Algorithm** (forward sweep from 0 to T):
```python
# Initial condition
M[0, :] = m₀(x)  # Apply initial density

# Forward time loop
for n in range(Nt):  # n = 0, 1, 2, ..., Nt-1
    for i in range(Nx):
        # Compute gradient of u
        grad_U = central_difference(U[n, :], i, Δx)

        # Drift velocity
        v[i] = -grad_U / ν

        # Advection term: div(m·v)
        flux = M[n, :] * v
        div_flux = divergence(flux, Δx)

        # Diffusion term: σ² Δm
        laplacian_M = laplacian(M[n, :], i, Δx)
        diffusion = σ² * laplacian_M

        # Forward Euler update
        M[n+1, i] = M[n, i] + Δt * (-div_flux[i] + diffusion)

    # Enforce positivity
    M[n+1, :] = max(M[n+1, :], 0)

    # Normalize mass (if closed domain)
    M[n+1, :] = M[n+1, :] / (sum(M[n+1, :]) * Δx)
```

**Key point**: Start from `M[0, :]` (known initial condition) and work forward to `M[Nt, :]`.

### 4.4 Spatial Operators (Central Differences)

**Gradient** (central difference):
```python
def gradient(u, i, Δx):
    return (u[i+1] - u[i-1]) / (2*Δx)
```

**Laplacian** (central difference):
```python
def laplacian(u, i, Δx):
    return (u[i+1] - 2*u[i] + u[i-1]) / Δx²
```

**Divergence** (central difference):
```python
def divergence(flux, Δx):
    div = np.zeros_like(flux)
    for i in range(1, len(flux)-1):
        div[i] = (flux[i+1] - flux[i-1]) / (2*Δx)
    return div
```

---

## 5. Complete Picard Iteration (FDM)

### 5.1 Full Algorithm

```python
def solve_MFG_picard_FDM(m₀, g, ν, κ, σ, T, Nx, Nt, max_iter=50, tol=1e-6, α=0.5):
    """
    Solve MFG system using Picard iteration with finite differences.

    Args:
        m₀: Initial density function
        g: Terminal cost function
        ν: Control cost
        κ: Congestion cost
        σ: Diffusion coefficient
        T: Time horizon
        Nx: Spatial grid points
        Nt: Temporal grid points
        max_iter: Maximum Picard iterations
        tol: Convergence tolerance
        α: Damping parameter

    Returns:
        U: Value function (Nt+1, Nx)
        M: Density (Nt+1, Nx)
    """

    # Grid setup
    x = np.linspace(xmin, xmax, Nx)
    Δx = (xmax - xmin) / (Nx - 1)
    Δt = T / Nt

    # Initialize
    M = np.zeros((Nt+1, Nx))
    M[0, :] = m₀(x)  # Initial density FIXED
    for n in range(Nt+1):
        M[n, :] = M[0, :]  # Extend uniformly for first iteration

    # Picard iteration
    for k in range(max_iter):
        M_old = M.copy()

        # ===== STEP 1: Solve HJB BACKWARD =====
        U = np.zeros((Nt+1, Nx))
        U[Nt, :] = g(x)  # Terminal condition FIXED

        for n in range(Nt-1, -1, -1):  # Backward: Nt-1 → 0
            for i in range(1, Nx-1):
                # Gradient
                grad_U = (U[n+1, i+1] - U[n+1, i-1]) / (2*Δx)

                # Hamiltonian
                H_val = 0.5/ν * grad_U**2 + κ * congestion(M_old[n, i])

                # Backward Euler
                U[n, i] = U[n+1, i] - Δt * H_val

            # Boundary conditions
            apply_BC(U[n, :])

        # ===== STEP 2: Solve FP FORWARD =====
        M_new = np.zeros((Nt+1, Nx))
        M_new[0, :] = m₀(x)  # Initial condition FIXED

        for n in range(Nt):  # Forward: 0 → Nt-1
            for i in range(1, Nx-1):
                # Drift velocity
                grad_U = (U[n, i+1] - U[n, i-1]) / (2*Δx)
                v = -grad_U / ν

                # Advection: div(m·v)
                flux_left = M_new[n, i-1] * v[i-1]
                flux_right = M_new[n, i+1] * v[i+1]
                div_flux = (flux_right - flux_left) / (2*Δx)

                # Diffusion: σ² Δm
                laplacian_M = (M_new[n, i+1] - 2*M_new[n, i] + M_new[n, i-1]) / Δx**2
                diffusion = σ**2 * laplacian_M

                # Forward Euler
                M_new[n+1, i] = M_new[n, i] + Δt * (-div_flux + diffusion)

            # Boundary conditions
            apply_BC(M_new[n+1, :])

            # Positivity
            M_new[n+1, :] = np.maximum(M_new[n+1, :], 0)

            # Mass conservation (normalize)
            total_mass = np.sum(M_new[n+1, :]) * Δx
            if total_mass > 1e-12:
                M_new[n+1, :] /= total_mass

        # ===== STEP 3: Damping =====
        M = (1 - α) * M_old + α * M_new

        # ===== STEP 4: Convergence check =====
        error = np.linalg.norm(M - M_old) / np.linalg.norm(M_old)
        print(f"Iteration {k}: error = {error:.6e}")

        if error < tol:
            print(f"Converged in {k+1} iterations")
            break

    return U, M
```

### 5.2 Key Implementation Points

1. **Terminal condition in HJB**:
   ```python
   U[Nt, :] = g(x)  # ALWAYS apply before backward sweep
   ```

2. **Initial condition in FP**:
   ```python
   M[0, :] = m₀(x)  # ALWAYS apply before forward sweep
   ```

3. **Backward HJB time loop**:
   ```python
   for n in range(Nt-1, -1, -1):  # n = Nt-1, Nt-2, ..., 0
       U[n, :] = ...  # Compute from U[n+1, :]
   ```

4. **Forward FP time loop**:
   ```python
   for n in range(Nt):  # n = 0, 1, 2, ..., Nt-1
       M[n+1, :] = ...  # Compute from M[n, :]
   ```

5. **Damping only M**:
   ```python
   M = (1 - α) * M_old + α * M_new  # Damp density
   U = ...  # DON'T damp value function
   ```

---

## 6. Common Mistakes and Debugging

### 6.1 Sign Errors

**Common mistake**: Wrong sign in drift velocity
```python
# ❌ WRONG
v = ∇u / ν  # This would push AWAY from goal!

# ✅ CORRECT
v = -∇u / ν  # Pushes TOWARD low-cost regions
```

**How to check**:
- If g(x*) = 0 at goal x*, then u(t, x*) should be small
- Gradient ∇u should point AWAY from x* (uphill)
- Velocity v = -∇u should point TOWARD x* (downhill)

### 6.2 Time Direction Errors

**Common mistake**: Solving HJB forward or FP backward
```python
# ❌ WRONG: HJB forward
for n in range(Nt):
    U[n+1, :] = ...  # NO! Terminal condition at Nt, not initial!

# ✅ CORRECT: HJB backward
U[Nt, :] = g(x)  # Start here
for n in range(Nt-1, -1, -1):
    U[n, :] = ...  # Work backward
```

### 6.3 Terminal Cost Not Applied

**Common mistake**: Forgetting to apply g(x)
```python
# ❌ WRONG
U[Nt, :] = 0  # Missing terminal cost!

# ✅ CORRECT
U[Nt, :] = g(x)  # Apply terminal cost
```

### 6.4 Validation Tests

**Test 1: Terminal cost influence**
```python
# Setup: Goal at x* = 0.5, g(x*) = 0, g elsewhere high
# Expected: m(T, x) should concentrate near x*

M_final = M[Nt, :]
mass_near_goal = sum(M_final[abs(x - 0.5) < 0.1]) * Δx
assert mass_near_goal > 0.5, "At least 50% mass should be near goal"
```

**Test 2: Mass conservation**
```python
# Check mass is conserved over time (closed domain)
for n in range(Nt+1):
    total_mass = sum(M[n, :]) * Δx
    assert abs(total_mass - 1.0) < 1e-6, f"Mass not conserved at t={n*Δt}"
```

**Test 3: Value function decreases to terminal cost**
```python
# At goal x*, u(t, x*) should decrease from u(0, x*) to g(x*) = 0
i_goal = argmin(abs(x - x_goal))
assert U[Nt, i_goal] == g(x_goal), "Terminal condition not satisfied"
assert U[0, i_goal] >= U[Nt, i_goal], "Value should decrease to terminal cost"
```

---

## 7. Summary Checklist

When implementing FP-HJB coupling:

- [ ] **HJB solver**:
  - [ ] Solve BACKWARD in time (T → 0)
  - [ ] Apply terminal condition u(T,x) = g(x) BEFORE loop
  - [ ] Use M^(k) for congestion term

- [ ] **FP solver**:
  - [ ] Solve FORWARD in time (0 → T)
  - [ ] Apply initial condition m(0,x) = m₀(x) BEFORE loop
  - [ ] Use U^(k+1) for drift velocity v = -∇u/ν
  - [ ] Sign: v = **-∇u/ν** (negative gradient!)

- [ ] **Coupling**:
  - [ ] Terminal cost g enters ONLY through HJB terminal condition
  - [ ] Initial mass m₀ enters ONLY through FP initial condition
  - [ ] Damp M between iterations, NOT U

- [ ] **Validation**:
  - [ ] Density flows toward regions where g is LOW
  - [ ] Mass conserved (closed domains)
  - [ ] Value function satisfies u(T,x) = g(x)

---

## 8. References

1. **Lasry & Lions (2007)**: "Mean field games", Japanese Journal of Mathematics
2. **Achdou & Capuzzo-Dolcetta (2010)**: "Mean field games: numerical methods", SIAM J. Numer. Anal.
3. **Cardaliaguet & Lehalle (2018)**: "Mean field game of controls and an application to trade crowding", Mathematics and Financial Economics

---

**Document Status**: Reference implementation guide for MFG solvers
**Next**: Apply this to debug particle-collocation FP solver
