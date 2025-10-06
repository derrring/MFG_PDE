# Evacuation Mean Field Games: Mathematical Formulation

**Author**: MFG_PDE Development Team
**Date**: October 2025
**Status**: Theoretical Foundation
**Related**: `anisotropic_mfg_mathematical_formulation.md`, `examples/advanced/anisotropic_crowd_dynamics_2d/`

---

## Overview

This document provides the mathematical formulation for **evacuation games** using Mean Field Game theory, addressing the fundamental challenge: in real evacuations, **mass decreases** as agents exit, violating the standard MFG assumption of mass conservation.

## The Mass Conservation Paradox

### Standard MFG Framework

Classical MFG systems assume **mass conservation**:

$$\frac{d}{dt} \int_\Omega m(t,x) \, dx = 0$$

This is suitable for:
- Financial markets (traders redistribute capital)
- Traffic flow on loops (vehicles circulate)
- Resource competition (fixed population)

### Evacuation Reality

In evacuation scenarios:

$$\frac{d}{dt} \int_\Omega m(t,x) \, dx < 0$$

Agents **leave the system** through exits, requiring modified MFG formulations.

---

## Three Approaches to Evacuation Games

### Approach 1: Absorbing Boundary Conditions

**Idea**: Add killing term to Fokker-Planck equation to remove mass near exits.

#### Mathematical Formulation

**Modified MFG System**:

```
HJB:  -∂u/∂t + H(x, ∇u, m) = f(x)              x ∈ Ω, t ∈ [0,T]
FP:   ∂m/∂t + ∇·(m ∇_p H) - σΔm = -k(x)m       x ∈ Ω, t ∈ [0,T]
```

**Exit Rate Function** $k(x) \geq 0$:
```
k(x) = κ · exp(-α · d(x, Γ_exit)²)
```

where:
- $\Gamma_{exit}$ = exit doors/boundaries
- $d(x, \Gamma_{exit})$ = distance to nearest exit
- $\kappa$ = absorption strength (exit capacity)
- $\alpha$ = spatial sharpness parameter

**Mass Balance**:
```
M(t) = ∫_Ω m(t,x) dx
dM/dt = -∫_Ω k(x)m(t,x) dx ≤ 0
```

**Evacuated Count**:
```
E(t) = ∫_0^t ∫_Ω k(x)m(s,x) dx ds
```

**Conservation Check**:
```
M(t) + E(t) = M(0)  (should hold numerically)
```

#### Implementation

```python
def exit_rate(x, y, door_locations, kappa=5.0, alpha=50.0):
    """
    Exit rate function for absorption.

    Args:
        x, y: Spatial coordinates
        door_locations: List of (x_door, y_door) tuples
        kappa: Absorption strength
        alpha: Spatial concentration

    Returns:
        k: Exit rate field
    """
    k = np.zeros_like(x)
    for x_door, y_door in door_locations:
        dist_sq = (x - x_door)**2 + (y - y_door)**2
        k += kappa * np.exp(-alpha * dist_sq)
    return k

# Modified FP equation
m_new = m - dt * div_flux + dt * diffusion - dt * k * m
```

#### Numerical Challenges

**Stiffness**: The term $-k(x)m$ creates stiff ODE when $k$ is large.

**Stability Condition** (explicit Euler):
```
dt · max(k) < 1
```

**Solutions**:
1. **Implicit-Explicit (IMEX)**: Treat absorption implicitly
   ```
   m^(n+1) = m^n - dt·div(flux) + dt·σΔm - dt·k·m^(n+1)
   (1 + dt·k)m^(n+1) = RHS
   m^(n+1) = RHS / (1 + dt·k)
   ```

2. **Semi-Implicit**:
   ```
   m^(n+1) = (m^n + dt·transport + dt·diffusion) / (1 + dt·k)
   ```

3. **Operator Splitting**: Solve transport-diffusion and absorption separately

---

### Approach 2: Target Domain with Terminal Payoff

**Idea**: Keep mass-conserving FP, but give large reward for reaching exit regions.

#### Mathematical Formulation

**Standard MFG System**:
```
HJB:  -∂u/∂t + H(x, ∇u, m) = f(x)
FP:   ∂m/∂t + ∇·(m ∇_p H) - σΔm = 0
```

**Modified Terminal Condition**:
```
u(T, x) = g(x)  where  g(x) = {
    -C_exit  if x ∈ Ω_exit  (large negative = high reward)
    0        otherwise
}
```

Typical values: $C_{exit} \in [100, 1000]$

**Exit Region** $\Omega_{exit}$:
```
Ω_exit = {x ∈ Ω : d(x, Γ_exit) < r_exit}
```

where $r_{exit}$ is the exit capture radius (e.g., door width).

#### Evacuation Metrics

**Success Metric** (concentration at exits):
```
Success = ∫_{Ω_exit} m(T, x) dx / ∫_Ω m(0, x) dx
```

**Evacuation Time Distribution**:
```
τ(x₀) = argmin_t {x(t) ∈ Ω_exit | x(0) = x₀}
```

where $x(t)$ follows optimal trajectory.

#### Implementation

```python
def terminal_cost_evacuation(x, y, door_locations, C_exit=500.0, r_exit=0.1):
    """
    Terminal cost with large reward near exits.

    Args:
        x, y: Spatial coordinates
        door_locations: Exit positions
        C_exit: Reward magnitude (negative cost)
        r_exit: Exit capture radius

    Returns:
        g: Terminal cost (negative near exits)
    """
    g = np.zeros_like(x)
    for x_door, y_door in door_locations:
        dist = np.sqrt((x - x_door)**2 + (y - y_door)**2)
        # Large negative cost (reward) near exits
        g = np.where(dist < r_exit, -C_exit, g)

    # Smooth transition (optional)
    # g = -C_exit * np.exp(-((dist - r_exit)/0.05)**2)

    return g
```

#### Advantages

✅ **Numerically stable**: Uses standard mass-conserving solvers
✅ **No stiffness issues**: No absorption term
✅ **Well-tested**: Leverages existing MFG infrastructure
✅ **Clear game interpretation**: Agents maximize reward

#### Disadvantages

❌ **Mass artificially conserved**: People don't actually "leave"
❌ **Indirect metric**: Must measure concentration, not evacuation count
❌ **Boundary effects**: Need careful tuning of $C_{exit}$ and $r_{exit}$

---

### Approach 3: Free Boundary Problem

**Idea**: Model evacuation as moving boundary where $m = 0$ at doors (Dirichlet BC).

#### Mathematical Formulation

**Domain Evolution**:
```
Ω(t) = {x ∈ Ω : m(t,x) > 0}  (occupied region)
∂Ω(t) = evacuation front
```

**MFG System with Free Boundary**:
```
HJB:  -∂u/∂t + H(x, ∇u, m) = f(x)        x ∈ Ω(t), t ∈ [0,T]
FP:   ∂m/∂t + ∇·(m ∇_p H) - σΔm = 0      x ∈ Ω(t), t ∈ [0,T]

Boundary Conditions on ∂Ω(t) ∩ Γ_exit:
  m = 0                (absorbing)
  u = g(x)             (terminal value)
```

**Front Propagation**:

Stefan-like condition:
```
v_n = -σ (∇m · n) / m  on ∂Ω(t)
```

where $v_n$ is normal velocity of front.

#### Level Set Formulation

Represent front implicitly:
```
φ(t,x) = 0  defines ∂Ω(t)
φ > 0  interior (occupied)
φ < 0  exterior (evacuated)
```

**Evolution**:
```
∂φ/∂t + V|∇φ| = 0
```

where $V$ is front speed derived from FP equation.

#### Numerical Methods

1. **Level Set Method** (Osher-Sethian)
2. **Phase Field** (Allen-Cahn smoothing)
3. **Variational Inequality** (obstacle problem formulation)

---

## Comparison Table

| Approach | Mass Behavior | Numerical Difficulty | Realism | Production-Ready |
|:---------|:--------------|:---------------------|:--------|:-----------------|
| **Absorbing Boundaries** | Decreases (realistic) | High (stiff) | High | ⚠️ Needs IMEX |
| **Target Payoff** | Conserved | Low | Medium | ✅ Yes |
| **Free Boundary** | Decreases | Very High | High | ❌ Research |

**Recommendation for Production**: **Target Payoff (#2)**
- Most stable
- Proven solvers
- Good enough approximation for practical evacuation design

---

## Application: Two-Door Room Evacuation

### Problem Setup

**Geometry**:
- Room: $\Omega = [0,1] \times [0,1]$
- Doors: $(0.3, 1.0)$ and $(0.7, 1.0)$ on top wall
- Initial crowd: Gaussian centered at $(0.5, 0.4)$

### Approach 2 Implementation (Target Payoff)

**Terminal Cost** (dual-well structure):
```python
def terminal_cost(x, y):
    """Distance to nearest door."""
    dist_left = (x - 0.3)**2 + (y - 1.0)**2
    dist_right = (x - 0.7)**2 + (y - 1.0)**2
    return np.minimum(dist_left, dist_right)
```

**Hamiltonian** (with congestion):
```python
def hamiltonian(p_x, p_y, m, gamma=0.1):
    """
    H = 0.5|p|² + γm|p|²

    Congestion term γm|p|² slows movement in crowds.
    """
    kinetic = 0.5 * (p_x**2 + p_y**2)
    congestion = gamma * m * (p_x**2 + p_y**2)
    return kinetic + congestion
```

**Running Cost**:
```python
f(x) = 1  (minimize time)
```

### Evacuation Metrics

**Success Rate**:
```python
def compute_success_rate(m_final, x, y, door_locations, r_exit=0.15):
    """
    Fraction of crowd that reached exit regions.
    """
    exit_mask = np.zeros_like(m_final, dtype=bool)
    for x_door, y_door in door_locations:
        dist = np.sqrt((x - x_door)**2 + (y - y_door)**2)
        exit_mask |= (dist < r_exit)

    mass_at_exits = np.sum(m_final[exit_mask]) * dx * dy
    total_mass = np.sum(m_final) * dx * dy

    return mass_at_exits / total_mass
```

**Evacuation Time** (average):
```python
def compute_avg_evacuation_time(u_initial, x, y, m_initial):
    """
    Average value function at initial positions.

    u(0,x) ≈ expected time to reach exit from x.
    """
    avg_time = np.sum(u_initial * m_initial) / np.sum(m_initial)
    return avg_time
```

---

## Anisotropic Extensions

Combine with anisotropic diffusion for realistic crowd behavior.

### Anisotropic Hamiltonian

```
H = (1/2) p^T A(x) p + γm · p^T A(x) p

A(x) = [1    ρ(x)]
       [ρ(x)   1 ]
```

where $\rho(x) \in (-1, 1)$ encodes directional preferences.

**Physical Interpretation**:
- $\rho > 0$: Diagonal movement preferred (corridors, pathways)
- $\rho < 0$: Axis-aligned movement preferred (rooms, open spaces)
- $\rho = 0$: Isotropic (standard MFG)

**Enhancement Near Exits**:
```python
def anisotropy_evacuation(x, y, door_locations):
    """
    Anisotropy enhanced near exits to model channeling.
    """
    # Base checkerboard pattern
    rho_base = 0.3 * np.sin(2*np.pi*x) * np.cos(2*np.pi*y)

    # Enhancement near doors (channeling effect)
    for x_door, y_door in door_locations:
        dist = np.sqrt((x - x_door)**2 + (y - y_door)**2)
        enhancement = 0.5 * np.exp(-5 * dist)
        rho_base += enhancement

    return np.clip(rho_base, -0.9, 0.9)
```

**See**: `anisotropic_mfg_mathematical_formulation.md` for full details.

---

## References

### Evacuation MFG Theory

1. **Achdou, Y., & Capuzzo-Dolcetta, I.** (2010). "Mean field games: numerical methods." *SIAM Journal on Numerical Analysis*, 48(3), 1136-1162.
   - Discusses boundary conditions and exit strategies

2. **Lachapelle, A., & Wolfram, M. T.** (2011). "On a mean field game approach modeling congestion and aversion in pedestrian crowds." *Transportation Research Part B*, 45(10), 1572-1589.
   - Evacuation dynamics with congestion

3. **Dogbé, C.** (2010). "Modeling crowd dynamics by the mean-field limit approach." *Mathematical and Computer Modelling*, 52(9-10), 1506-1520.
   - Absorbing boundaries for pedestrian flow

4. **Maury, B., Roudneff-Chupin, A., & Santambrogio, F.** (2010). "A macroscopic crowd motion model of gradient flow type." *Mathematical Models and Methods in Applied Sciences*, 20(10), 1787-1821.
   - Exit capacity constraints

### Anisotropic Crowd Dynamics

5. **Appert-Rolland, C., et al.** (2020). "Microscopic and macroscopic dynamics of a pedestrian cross-flow: Part I, experimental analysis." *Physica A*, 549, 124295.
   - Empirical anisotropy in pedestrian flow

6. **Hoogendoorn, S. P., & Bovy, P. H. L.** (2004). "Pedestrian route-choice and activity scheduling theory and models." *Transportation Research Part B*, 38(2), 169-190.
   - Directional preferences in evacuation

### Numerical Methods

7. **Carlini, E., & Silva, F. J.** (2013). "A semi-Lagrangian scheme for a degenerate second order mean field game system." *Discrete and Continuous Dynamical Systems*, 35(9), 4269-4292.
   - Stable schemes for MFG with boundaries

8. **Benamou, J. D., & Carlier, G.** (2015). "Augmented Lagrangian methods for transport optimization, mean field games and degenerate elliptic equations." *Journal of Optimization Theory and Applications*, 167(1), 1-26.
   - Handling absorbing boundaries

---

## Implementation Notes

### Code References

**Examples**:
- `examples/advanced/anisotropic_crowd_dynamics_2d/room_evacuation_two_doors.py` - Visualization setup
- `examples/advanced/anisotropic_crowd_dynamics_2d/run_two_door_evacuation.py` - Full simulation (Approach #2)
- `examples/advanced/anisotropic_crowd_dynamics_2d/evacuation_with_absorption.py` - Approach #1 demo (numerical issues)
- `examples/advanced/anisotropic_crowd_dynamics_2d/production_solver_demo.py` - Mass conservation validation

**Theory**:
- `docs/theory/anisotropic_mfg_mathematical_formulation.md` - Anisotropic framework

### Production Recommendation

For **practical evacuation simulations**, use:

1. **Target Payoff Approach** (#2) with large terminal reward at exits
2. **Anisotropic Hamiltonian** to model realistic crowd channeling
3. **Congestion term** $\gamma m |p|^2$ to slow movement in crowds
4. **Success metric**: Concentration at exit regions

**Avoid** explicit absorption for production use due to stiffness issues unless using implicit methods.

---

**Last Updated**: October 2025
**Version**: 1.0
**Contributors**: MFG_PDE Development Team
