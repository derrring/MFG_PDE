# Complete BC Workflow: From Specification to Application

**Date**: 2025-12-17
**Status**: Target Architecture (after Issues #493-496)

---

## 1. BC Classification in MFG Systems

MFG systems have **two orthogonal BC dimensions**:

```
                        SPATIAL DOMAIN
                    ┌─────────────────────┐
                    │                     │
         t=T  ──────┤  u_fin (HJB)        │  ← Terminal condition
                    │                     │
    TIME            │    Ω × [0,T]        │     SPATIAL BC
    DOMAIN          │                     │     (walls, exits)
                    │                     │
         t=0  ──────┤  m_init (FP)        │  ← Initial condition
                    │                     │
                    └─────────────────────┘
                         ∂Ω (boundary)
```

### 1.1 Temporal BC (Initial/Terminal Conditions)

| Equation | Condition | Mathematical Form | Physical Meaning |
|----------|-----------|-------------------|------------------|
| **FP** | Initial | m(0, x) = m₀(x) | Initial population distribution |
| **HJB** | Terminal | u(T, x) = u_T(x) | Terminal cost/reward |

### 1.2 Spatial BC (Boundary Conditions)

| Type | Mathematical Form | Physical Meaning |
|------|-------------------|------------------|
| **Dirichlet** | u(t, x) = g(t) on ∂Ω | Fixed value at boundary |
| **Neumann** | ∂u/∂n = g(t) on ∂Ω | Fixed flux at boundary |
| **No-flux** | ∂m/∂n = 0 on ∂Ω | No mass leaves domain |
| **Periodic** | u(t, x_min) = u(t, x_max) | Wrap-around domain |
| **Robin** | αu + β∂u/∂n = g | Mixed condition |
| **Absorbing** | m = 0 on ∂Ω_exit | Agents leave at exits |

---

## 2. Two Paradigms: Grid vs Particle

### 2.1 Grid-Based Methods (FDM, FEM, GFDM)

BC applied to **discrete field arrays**:

```
┌─────────────────────────────────────────────────────────────┐
│                    GRID-BASED BC APPLICATION                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Field u[i,j] on grid:                                    │
│                                                             │
│   BC Zone    Interior    BC Zone                           │
│   ◄─────►   ◄────────►   ◄─────►                           │
│                                                             │
│   [g][g][ ][ ][ ][ ][ ][ ][ ][g][g]   ← Ghost points       │
│      ↑                         ↑                            │
│   Neumann:                  Dirichlet:                      │
│   u_g = u_1 - 2Δx·flux     u_g = 2·g - u_N                 │
│                                                             │
│   Matrix modification (implicit):                          │
│   ┌                    ┐   ┌   ┐   ┌   ┐                   │
│   │ 1  0  0  ...  0  0 │   │u_0│   │ g │  ← Dirichlet row  │
│   │ a  b  c  ...  0  0 │   │u_1│   │f_1│                   │
│   │ 0  a  b  ...  0  0 │ × │ . │ = │ . │                   │
│   │ .  .  .  ...  .  . │   │ . │   │ . │                   │
│   │ 0  0  0  ...  b  c │   │u_N│   │f_N│                   │
│   │ 0  0  0  ...  0  1 │   │u_N│   │ g │  ← Dirichlet row  │
│   └                    ┘   └   ┘   └   ┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Particle Methods (Monte Carlo, SDE)

BC applied to **particle trajectories**:

```
┌─────────────────────────────────────────────────────────────┐
│                   PARTICLE-BASED BC APPLICATION             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Particle X_t follows SDE:                                │
│   dX_t = b(X_t, m_t) dt + σ dW_t                           │
│                                                             │
│   When X_t hits boundary ∂Ω:                               │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              Domain Ω                                │  │
│   │                                                      │  │
│   │    X_t ──────•                                      │  │
│   │              │ hits boundary                        │  │
│   │              ▼                                      │  │
│   ├──────────────●──────────────────────────────────────┤  │
│   │           ∂Ω (boundary)                             │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   BC Type        Action on Particle                        │
│   ─────────────────────────────────────────────────────    │
│   Reflecting     X_t ← reflect(X_t, normal)               │
│   (No-flux)      Particle bounces back                    │
│                                                             │
│   Absorbing      Remove particle, record exit time        │
│   (Dirichlet)    Particle absorbed at boundary            │
│                                                             │
│   Periodic       X_t ← wrap(X_t, domain)                  │
│                  Particle re-enters from opposite side    │
│                                                             │
│   Partial        With prob p: absorb, else: reflect       │
│   absorption     Models partially permeable walls         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Complete Workflow (Target Architecture)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE BC WORKFLOW                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║                     USER SPECIFICATION                                 ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                                                                             │
│  # 1. Spatial BC specification                                             │
│  spatial_bc = mixed_bc([                                                   │
│      BCSegment("walls", BCType.NO_FLUX, boundary=["left", "right"]),      │
│      BCSegment("exit", BCType.DIRICHLET, value=0.0, boundary="top"),      │
│      BCSegment("inlet", BCType.NEUMANN, value=1.0, boundary="bottom"),    │
│  ])                                                                        │
│                                                                             │
│  # 2. Geometry with spatial BC (SSOT)                                      │
│  geometry = TensorProductGrid(                                             │
│      dimension=2,                                                          │
│      bounds=[(0, 10), (0, 5)],                                            │
│      Nx=[100, 50],                                                         │
│      boundary_conditions=spatial_bc,  # ← Spatial BC stored here          │
│  )                                                                         │
│                                                                             │
│  # 3. Temporal conditions (initial/terminal)                               │
│  m_init = gaussian_distribution(geometry)   # FP initial: m(0,x)          │
│  u_fin = terminal_cost(geometry)            # HJB terminal: u(T,x)        │
│                                                                             │
│  # 4. Problem definition                                                   │
│  problem = MFGProblem(                                                     │
│      geometry=geometry,      # Contains spatial BC                         │
│      m_init=m_init,          # Temporal BC for FP                         │
│      u_fin=u_fin,            # Temporal BC for HJB                        │
│      T=1.0, Nt=100,                                                        │
│      sigma=0.1,                                                            │
│  )                                                                         │
│                                                                             │
│                                    │                                        │
│                                    ▼                                        │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║                     PROBLEM INTERFACE                                  ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                                                                             │
│  problem.get_boundary_conditions()  → BoundaryConditions (spatial)         │
│      └── Delegates to: geometry.get_boundary_conditions()                  │
│                                                                             │
│  problem.get_m_init()  → np.ndarray (temporal BC for FP)                  │
│  problem.get_u_fin()   → np.ndarray (temporal BC for HJB)                 │
│                                                                             │
│                                    │                                        │
│                    ┌───────────────┼───────────────┐                       │
│                    ▼               ▼               ▼                       │
│  ╔═════════════════════╗ ╔═════════════════════╗ ╔═════════════════════╗  │
│  ║    HJB SOLVER       ║ ║    FP SOLVER        ║ ║  COUPLING SOLVER    ║  │
│  ║    (Backward)       ║ ║    (Forward)        ║ ║  (Orchestrator)     ║  │
│  ╚═════════════════════╝ ╚═════════════════════╝ ╚═════════════════════╝  │
│                                                                             │
│                                    │                                        │
│          ┌─────────────────────────┼─────────────────────────┐             │
│          ▼                         ▼                         ▼             │
│  ┌───────────────┐        ┌───────────────┐        ┌───────────────┐      │
│  │  GRID-BASED   │        │  GRID-BASED   │        │   PARTICLE    │      │
│  │  HJB (FDM)    │        │  FP (FDM)     │        │   FP (SDE)    │      │
│  └───────────────┘        └───────────────┘        └───────────────┘      │
│          │                         │                         │             │
│          ▼                         ▼                         ▼             │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║                     BC APPLICATION                                     ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ GRID-BASED APPLICATION                                              │   │
│  │                                                                      │   │
│  │ # Get BC spec from geometry (SSOT)                                  │   │
│  │ bc = problem.get_boundary_conditions()                              │   │
│  │                                                                      │   │
│  │ # Get applicator for this method                                    │   │
│  │ applicator = geometry.get_boundary_applicator(method="fdm")         │   │
│  │                                                                      │   │
│  │ # HJB: Ghost values BEFORE Hamiltonian (Issue #494)                 │   │
│  │ u_padded = applicator.get_ghost_layer(u, bc, t)                     │   │
│  │ H = hamiltonian(gradient(u_padded))  # Uses ghost-aware stencil    │   │
│  │                                                                      │   │
│  │ # FP: Apply BC after advection-diffusion step                       │   │
│  │ m_new = explicit_step(m, u, dt)                                     │   │
│  │ m_new = applicator.apply(m_new, bc, t)  # Enforce BC                │   │
│  │                                                                      │   │
│  │ # Or implicit: modify matrix                                        │   │
│  │ A, b = applicator.apply_to_system(A, b, bc, t)                      │   │
│  │ m_new = solve(A, b)                                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PARTICLE-BASED APPLICATION                                          │   │
│  │                                                                      │   │
│  │ # Get BC spec from geometry (SSOT)                                  │   │
│  │ bc = problem.get_boundary_conditions()                              │   │
│  │                                                                      │   │
│  │ # Get particle BC handler                                           │   │
│  │ handler = ParticleBCHandler(bc, geometry.get_bounds())              │   │
│  │                                                                      │   │
│  │ # Evolve particles with SDE                                         │   │
│  │ for t in timesteps:                                                 │   │
│  │     X_new = X + drift(X, m) * dt + sigma * sqrt(dt) * randn()      │   │
│  │                                                                      │   │
│  │     # Check boundary crossing                                       │   │
│  │     for each particle i:                                            │   │
│  │         if handler.crosses_boundary(X[i], X_new[i]):               │   │
│  │             X_new[i] = handler.apply_bc(X[i], X_new[i], bc_type)   │   │
│  │             # Reflecting: bounce back                               │   │
│  │             # Absorbing: mark for removal                           │   │
│  │             # Periodic: wrap to opposite side                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Method-Specific BC Application

### 4.1 FDM (Finite Difference Method)

```python
class FDMBCApplicator:
    """BC applicator for structured FDM grids."""

    def get_ghost_layer(self, u: NDArray, bc: BoundaryConditions, t: float) -> NDArray:
        """
        Pad field with ghost values for stencil computation.
        Called BEFORE differentiation (critical for HJB upwind).
        """
        padded = np.pad(u, 1, mode='constant')

        for seg in bc.segments:
            if seg.bc_type == BCType.DIRICHLET:
                # Ghost: u_g = 2*g - u_interior (linear extrapolation)
                g = seg.value(t) if callable(seg.value) else seg.value
                self._set_ghost_dirichlet(padded, g, seg.boundary)

            elif seg.bc_type == BCType.NEUMANN:
                # Ghost: u_g = u_interior - 2*dx*flux (2nd order)
                flux = seg.value(t) if callable(seg.value) else seg.value
                self._set_ghost_neumann(padded, flux, seg.boundary)

            elif seg.bc_type == BCType.NO_FLUX:
                # Ghost: u_g = u_interior (zero gradient)
                self._set_ghost_neumann(padded, 0.0, seg.boundary)

            elif seg.bc_type == BCType.PERIODIC:
                # Ghost: wrap from opposite side
                self._set_ghost_periodic(padded, seg.boundary)

        return padded

    def apply(self, u: NDArray, bc: BoundaryConditions, t: float) -> NDArray:
        """
        Apply BC directly to field (post-step enforcement).
        Used for FP explicit schemes.
        """
        for seg in bc.segments:
            if seg.bc_type == BCType.DIRICHLET:
                g = seg.value(t) if callable(seg.value) else seg.value
                self._set_boundary_value(u, g, seg.boundary)
        return u
```

### 4.2 Particle Method

```python
class ParticleBCHandler:
    """BC handler for particle/Monte Carlo methods."""

    def __init__(self, bc: BoundaryConditions, bounds: tuple):
        self.bc = bc
        self.bounds = bounds  # ((xmin, xmax), (ymin, ymax), ...)

    def apply_bc(self, X_old: NDArray, X_new: NDArray) -> tuple[NDArray, NDArray]:
        """
        Apply BC to particle positions.

        Returns:
            X_new: Updated positions
            mask: Boolean mask of active (non-absorbed) particles
        """
        mask = np.ones(len(X_new), dtype=bool)

        for dim in range(X_new.shape[1]):
            x_min, x_max = self.bounds[dim]
            seg = self._get_segment_for_dim(dim)

            # Particles below min boundary
            below = X_new[:, dim] < x_min
            # Particles above max boundary
            above = X_new[:, dim] > x_max

            if seg.bc_type == BCType.REFLECTING:
                # Reflect: x_new = x_min + (x_min - x_new)
                X_new[below, dim] = 2 * x_min - X_new[below, dim]
                X_new[above, dim] = 2 * x_max - X_new[above, dim]

            elif seg.bc_type == BCType.ABSORBING:
                # Mark for removal
                mask[below | above] = False

            elif seg.bc_type == BCType.PERIODIC:
                # Wrap around
                L = x_max - x_min
                X_new[:, dim] = x_min + (X_new[:, dim] - x_min) % L

        return X_new, mask
```

---

## 5. Temporal BC Application

Temporal BCs are simpler - they set initial/terminal values:

```python
class MFGSolver:
    """Coupling solver orchestrating HJB and FP."""

    def solve(self):
        # Get temporal BCs
        M_initial = self.problem.get_m_init()    # m(0, x) = m₀(x)
        U_terminal = self.problem.get_u_fin()    # u(T, x) = u_T(x)

        # Initialize
        M = np.zeros((Nt+1, *spatial_shape))
        U = np.zeros((Nt+1, *spatial_shape))

        # Set temporal BC
        M[0, :] = M_initial       # FP starts at t=0
        U[-1, :] = U_terminal     # HJB starts at t=T

        # Fixed-point iteration
        for iteration in range(max_iter):
            # HJB: backward from T to 0
            U = self.hjb_solver.solve_backward(U_terminal, M)

            # FP: forward from 0 to T
            M = self.fp_solver.solve_forward(M_initial, U)

            if converged(U, M):
                break

        return U, M
```

---

## 6. Summary: BC Sources and Consumers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      BC SOURCES AND CONSUMERS                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SOURCE                    BC TYPE              CONSUMER                │
│  ──────                    ───────              ────────                │
│                                                                         │
│  Geometry                                                               │
│  └── boundary_conditions ─► Spatial BC ──────► HJB Solver              │
│      (SSOT)                 (walls, exits)     FP Solver               │
│                                                 Particle Handler        │
│                                                                         │
│  MFGProblem                                                            │
│  ├── m_init ─────────────► FP Initial ───────► FP Solver (t=0)        │
│  │                          m(0,x) = m₀(x)                             │
│  │                                                                      │
│  └── u_fin ──────────────► HJB Terminal ─────► HJB Solver (t=T)       │
│                             u(T,x) = u_T(x)                            │
│                                                                         │
│  MFGComponents (Legacy)                                                │
│  └── boundary_conditions ─► Spatial BC ──────► (fallback if no geom)  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Differences: Grid vs Particle BC

| Aspect | Grid-Based (FDM/FEM) | Particle-Based (SDE) |
|--------|---------------------|---------------------|
| **When applied** | Every timestep, to arrays | When particle hits boundary |
| **How applied** | Modify values/matrix | Modify trajectory |
| **Dirichlet** | Set u[boundary] = g | Absorb particle |
| **Neumann** | Ghost point extrapolation | N/A (use reflecting) |
| **No-flux** | Zero gradient ghost | Reflect particle |
| **Periodic** | Copy from opposite side | Wrap position |
| **Complexity** | O(boundary_points) | O(particles × checks) |
| **Vectorization** | Fully vectorized | Per-particle or batched |

---

## 8. Open Issues

| Issue | Description | Status |
|-------|-------------|--------|
| #493 | Geometry owns spatial BC (SSOT) | ✅ Implemented |
| #494 | HJB ghost values for upwind | 🔴 HIGH |
| #495 | Optional BC dimension | 🟡 MEDIUM |
| #496 | Time parameter in BC applicators | 🟡 MEDIUM |
| #497 | Particle SDF-based complex geometry | 🟡 MEDIUM |

---

## 9. Audit Status

**Final Audit Verdict: APPROVED as Master Blueprint**

### Verified Correct

| Component | Verification |
|-----------|--------------|
| Dirichlet ghost: `u_g = 2g - u_in` | ✅ 2nd-order for cell-centered grids |
| Neumann ghost: `u_g = u_in - 2Δx·flux` | ✅ Central difference correct |
| Particle reflection: `X = 2·x_min - X` | ✅ Standard Euler-Maruyama |
| HJB ghost-before-Hamiltonian | ✅ Optimal for upwind schemes |

### Known Gaps (Tracked)

| Gap | Issue | Risk |
|-----|-------|------|
| Particle handler ignores SDF regions | #497 | Medium |
| Corner cases in particle reflection | #497 | Low (OK for rectangular) |
| Time `t` passed to BC applicators | #496 | Medium |
