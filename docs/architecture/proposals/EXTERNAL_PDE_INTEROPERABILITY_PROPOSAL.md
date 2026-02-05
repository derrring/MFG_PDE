# External PDE Framework Interoperability & Problem Extension

**Status**: Future Proposal (Implementation deferred to post-v1.0; analysis is current)
**Author**: External Expert Review + Internal Analysis
**Date**: 2026-02-05
**Related Issue**: TBD
**Prerequisite**: v1.0.0 release with stable core API

---

## 1. Executive Summary

This document analyzes MFG_PDE's structural relationship to general PDE frameworks and proposes a phased interoperability strategy. Implementation is deferred, but the analysis and design thinking are recorded now while the strategic context is fresh.

**Core observation**: Mean Field Games are inherently PDE-constrained optimal control problems. The HJB equation *is* the optimality condition for controlling the Fokker-Planck dynamics. As MFG_PDE matures toward richer problem classes (state constraints, multi-physics coupling, PDE control), interaction with general PDE solvers becomes structurally necessary, not merely convenient.

**Core thesis**: MFG_PDE should remain the specialized orchestrator for MFG-specific logic (backward-forward coupling, density-value iteration, control extraction), while providing clean delegation interfaces for the generic PDE capabilities that external frameworks do better (complex geometry, advanced discretization, parallel linear algebra).

**Current State**: Self-contained with NumPy/SciPy. No external PDE framework integration. Problem definition limited to standard separable HJB-FP coupling.

**Proposed State** (phased):
- **Phase A** (pre-v1.0): No implementation. Record design thinking (this document).
- **Phase B** (v1.0+): Data exchange formats + one concrete solver backend (PETSc).
- **Phase C** (v1.x+): Problem composition extensions driven by research demand.

---

## 2. Why Interoperability Is Structurally Necessary

### 2.1 MFG as PDE-Constrained Optimal Control

The standard MFG system is not merely "two coupled PDEs." It is the first-order optimality system (KKT conditions) for a PDE-constrained optimization problem:

$$\min_{\alpha(\cdot)} \int_0^T \int_\Omega \left[ L(x, \alpha, m) \, m(t,x) + F(m(t,x)) \right] dx \, dt + \int_\Omega G(m(T,x)) \, dx$$

subject to the **Fokker-Planck constraint** (state equation):

$$\partial_t m - \frac{\sigma^2}{2} \Delta m + \nabla \cdot (m \, \alpha) = 0, \quad m(0) = m_0$$

The HJB equation emerges as the **adjoint equation** (the PDE analog of the Lagrange multiplier):

$$-\partial_t u - \frac{\sigma^2}{2} \Delta u + H(\nabla u, m) = 0, \quad u(T) = G'(m(T))$$

And the optimal control is recovered from the value function: $\alpha^* = -D_p H(\nabla u, m)$.

This structure has three consequences for interoperability:

1. **The FP equation is the "physics"** --- any enrichment of the underlying dynamics (convection, reaction, nonlinear diffusion, obstacles) directly changes the PDE constraint, requiring PDE solver capabilities beyond what MFG_PDE currently provides.

2. **The HJB equation adapts automatically** --- when the FP constraint changes, the adjoint (HJB) equation changes correspondingly. The Hamiltonian encodes this relationship. MFG_PDE's Hamiltonian class hierarchy is well-positioned to express these variations.

3. **New problem classes = new PDE constraints** --- MFG with congestion, MFG with obstacles, MFG on networks with continuum limits, multi-population MFG with inter-species dynamics --- all introduce additional PDE components that may benefit from external solver capabilities.

### 2.2 Concrete Problem Classes Requiring Extended PDE Capabilities

| Problem Class | Additional PDE Component | Why External Solvers Help |
|:--------------|:------------------------|:--------------------------|
| **Crowd motion with obstacles** | Eikonal equation $\|\nabla d\| = 1$ for distance field | FMM/FSM algorithms; complex geometry |
| **MFG with congestion** | Nonlinear diffusion $\nabla \cdot (m^\alpha \nabla u)$ | Requires careful nonlinear solver; FEM natural |
| **Fluid-coupled MFG** | Navier-Stokes for background flow | Multi-physics; delegation essential |
| **MFG with state constraints** | Obstacle problem $\min(u, \psi) = 0$ (variational inequality) | Specialized VI solvers |
| **MFG on manifolds** | Surface PDE discretization | FEM on triangulated surfaces |
| **Second-order MFG (common noise)** | Stochastic PDE in $(t, x, \omega)$ | High-dimensional; Monte Carlo + PDE |
| **MFG planning problem** | Both $m(0)$ and $m(T)$ prescribed; two-point BVP | Augmented Lagrangian methods |
| **MFG with major player** | Coupled ODE-PDE system | External ODE integrators |

These are not speculative. Published MFG literature covers all of these, and any serious MFG infrastructure will eventually need to handle a subset.

### 2.3 What MFG_PDE Owns vs. What It Delegates

The key architectural question is: **where does MFG_PDE's responsibility end and the external framework's begin?**

```
MFG_PDE's Domain (KEEP)                External Framework's Domain (DELEGATE)
┌─────────────────────────┐            ┌─────────────────────────────────┐
│ Problem definition:     │            │ Spatial discretization:         │
│   Hamiltonian H(x,m,p,t)│            │   FEM assembly on complex mesh  │
│   Cost functional J[m,α]│            │   Unstructured grid management  │
│   Boundary semantics    │            │   AMR (adaptive mesh refinement)│
│                         │            │                                 │
│ Coupling logic:         │            │ Linear algebra:                 │
│   Picard / Newton outer │            │   Parallel sparse solvers       │
│   Backward-forward time │            │   Preconditioners (AMG, ILU)    │
│   Convergence monitoring│            │   GPU-accelerated mat-vec       │
│   Mass conservation     │            │                                 │
│                         │            │ Auxiliary PDE solvers:           │
│ Control extraction:     │            │   Eikonal (FMM/FSM)             │
│   α* = -D_pH(∇u, m)    │            │   Navier-Stokes                 │
│   Policy evaluation     │            │   Variational inequalities      │
│                         │            │                                 │
│ MFG-specific validation:│            │ I/O and visualization:          │
│   ∫m dx = 1             │            │   ParaView / VTK export         │
│   NaN/divergence detect │            │   HDF5 time series              │
│   Adjoint consistency   │            │   Mesh format conversion        │
└─────────────────────────┘            └─────────────────────────────────┘
```

**Principle**: MFG_PDE is the **control-theoretic orchestrator**. External frameworks are **computational backends**.

---

## 3. General PDE Framework Landscape

### 3.1 Major Frameworks

| Framework | Language | Primary Method | Key Strength | MFG Relevance |
|-----------|----------|---------------|--------------|---------------|
| **FEniCS/FEniCSx** | Python/C++ | FEM (variational) | Near-math DSL (UFL), auto code generation | Complex geometry, nonlinear PDE |
| **deal.II** | C++ | FEM | AMR, parallel scalability | Large-scale 2D/3D problems |
| **PETSc** | C/Python | Linear algebra toolkit | Scalable parallel solvers | Direct backend for $Ax=b$ |
| **Trilinos** | C++ | Linear algebra toolkit | Modular solver packages | Alternative to PETSc |
| **OpenFOAM** | C++ | FVM | CFD | Fluid-coupled MFG |
| **SciML (Julia)** | Julia | Multi-paradigm | ML + PDE, multiple dispatch | Autodiff, neural operators |
| **Firedrake** | Python/C | FEM (variational) | Composable solvers, PETSc backend | Closest philosophy to MFG_PDE |
| **scikit-fem** | Python | FEM | Lightweight, Pythonic | Quick FEM prototyping |
| **DUNE** | C++ | Multi-method | Flexible grid interface | Advanced discretization |

### 3.2 Why MFG_PDE Is Not (and Should Not Become) a General PDE Framework

MFG_PDE exploits the specific **structure of the HJB-FP coupling**:

1. **Backward-forward time structure**: HJB solves backward, FP solves forward. General frameworks don't optimize for this.
2. **Picard iteration semantics**: The outer loop alternates between two PDEs with coupling through density $m$ and value function $u$. This is domain-specific.
3. **Control-theoretic quantities**: $D_pH$ (optimal control), $D_mH$ (density coupling) are first-class objects in MFG_PDE; in a general framework they would be user-level derived quantities.
4. **Validation semantics**: Mass conservation $\int m \, dx = 1$, coupled convergence criteria, adjoint-consistent boundary conditions --- all MFG-specific invariants.

General frameworks provide the building blocks (mesh, assembly, linear solve) but not the orchestration logic.

### 3.3 Capability Comparison

| Criterion | MFG_PDE (Specialized) | General PDE Framework |
|:----------|:----------------------|:----------------------|
| **HJB-FP coupling** | Native, optimized | Manual, no special support |
| **Complex geometry** | Limited (structured grids, GFDM) | Full (unstructured, AMR) |
| **Parallel scalability** | Single-node (NumPy/SciPy) | Distributed (MPI, GPU) |
| **Discretization variety** | FDM, GFDM, neural | FEM, FVM, DG, spectral |
| **Problem setup cost** | Low (MFGComponents API) | High (manual weak forms) |
| **Control extraction** | Native ($\alpha^* = -D_pH$) | Manual post-processing |
| **Convergence monitoring** | MFG-aware (both $u$ and $m$) | Generic residual norms |

---

## 4. Interoperability Architecture

### 4.1 Design Principle

MFG_PDE delegates to external frameworks at well-defined boundaries, without coupling its internal representations to any specific external API. The adapter pattern keeps MFG_PDE's core independent.

```
MFG_PDE Core          Adapter Layer           External Framework
+--------------+     +---------------+     +------------------+
| MFGProblem   |---->| GeometryExport|---->| FEniCS Mesh      |
| HJB/FP solve |     | FieldExport   |     | PETSc Mat/Vec    |
| Picard loop  |<----| ResultImport  |<----| Solution vectors  |
+--------------+     +---------------+     +------------------+
```

### 4.2 Interoperability Tiers

#### Tier 1: Data Exchange (Low cost, high value)

Standard I/O formats for offline interoperability. No runtime coupling.

| Format | Use Case | Effort | Priority |
|--------|----------|--------|----------|
| **HDF5** | Solution fields $u(t,x)$, $m(t,x)$, convergence history | ~100 LOC | High |
| **XDMF** | Time-dependent fields for ParaView | ~150 LOC | High |
| **VTK/VTU** | Snapshot visualization, post-processing | ~100 LOC (via meshio) | Medium |
| **meshio** | Mesh import/export (GMSH, Exodus, etc.) | ~200 LOC adapter | Medium |

This enables:
- Visualization with ParaView / VisIt (publication-quality figures)
- Comparison with FEniCS solutions (offline validation)
- Initialization from external computations (warm-starting)
- Data archival and reproducibility (HDF5 is self-describing)

**Conceptual interface**:

```python
class FieldExporter(Protocol):
    """Export MFG solution fields to standard formats."""

    def export_hdf5(
        self,
        path: Path,
        u: np.ndarray,           # (Nt, *spatial_shape)
        m: np.ndarray,           # (Nt, *spatial_shape)
        geometry: GeometryProtocol,
        time_grid: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    def export_xdmf(
        self,
        path: Path,
        u: np.ndarray,
        m: np.ndarray,
        geometry: GeometryProtocol,
        time_grid: np.ndarray,
    ) -> None: ...


class GeometryAdapter(Protocol):
    """Convert between MFG_PDE geometry and external mesh formats."""

    def to_meshio(self, geometry: GeometryProtocol) -> meshio.Mesh: ...
    def from_meshio(self, mesh: meshio.Mesh) -> GeometryProtocol: ...
```

#### Tier 2: Solver Backend Delegation (Medium cost, targeted value)

Replace internal linear algebra with external solvers at runtime. MFG_PDE's discretization and iteration logic remain unchanged; only the inner $Ax = b$ solve is delegated.

**Current internal state** --- MFG_PDE already has partial infrastructure for this tier:

- **`SparseSolver`** (`mfg_pde/utils/sparse_operations.py:466-620`): Unified interface supporting `direct`/`cg`/`gmres`/`bicgstab` methods with ILU/Jacobi preconditioning and CuPy GPU backend. However, only 1-2 files use it; the remaining ~15 `spsolve` call sites across 9 files (`hjb_gfdm.py`, `base_hjb.py`, `fp_fdm.py`, `fp_fdm_time_stepping.py`, `fp_network.py`, `hjb_network.py`, `nonlinear_solvers.py`, `sparse_operations.py`, `optimization.py`) bypass it with direct `scipy.sparse.linalg.spsolve` imports.
- **`BaseBackend`** (`mfg_pde/backends/`): Array operation abstraction (NumPy, JAX, PyTorch, Numba) with a `solve(A, b)` method, but for **dense** systems only. Sparse linear algebra is not covered by the backend hierarchy.
- **`SPEC_LINEAR_SOLVER.md`** (`docs/architecture/spacetime-bc/`): Draft spec for trait-based auto-selection of linear solvers based on matrix properties (SPD → CG, nonsymmetric → GMRES, etc.). Covers similar scope to Tier 2 but is internal-only (no external framework delegation).

**Tier 2 builds on this foundation**: The `LinearSolverBackend` protocol below would (a) replace the scattered `spsolve` calls with a single delegation point, (b) subsume `SparseSolver` as the default `ScipySolver` implementation, and (c) extend the `SPEC_LINEAR_SOLVER.md` auto-selection logic as a strategy layer on top of `LinearSolverBackend`. The existing `BaseBackend` for dense operations remains orthogonal.

**Primary candidate**: PETSc via `petsc4py`.

```python
class LinearSolverBackend(Protocol):
    """Pluggable linear solver backend.

    MFG_PDE assembles the system; the backend solves it.
    This is the narrowest possible delegation boundary.
    """

    def solve(
        self,
        A: scipy.sparse.spmatrix,
        b: np.ndarray,
    ) -> np.ndarray:
        """Solve Ax = b. Returns x."""
        ...

    @property
    def name(self) -> str: ...

    @property
    def supports_parallel(self) -> bool: ...


class ScipySolver:
    """Current default: scipy.sparse.linalg.spsolve."""
    name = "scipy"
    supports_parallel = False

    def solve(self, A, b):
        return scipy.sparse.linalg.spsolve(A, b)


class PETScSolver:
    """Delegates to PETSc KSP with configurable preconditioner.

    Converts scipy sparse → PETSc Mat, solves, converts back.
    """
    name = "petsc"
    supports_parallel = True

    def __init__(self, ksp_type: str = "gmres", pc_type: str = "ilu"):
        self.ksp_type = ksp_type
        self.pc_type = pc_type

    def solve(self, A, b):
        # Convert scipy CSR → PETSc AIJ matrix
        petsc_A = PETSc.Mat().createAIJ(
            size=A.shape, csr=(A.indptr, A.indices, A.data)
        )
        petsc_b = PETSc.Vec().createWithArray(b)
        petsc_x = petsc_b.duplicate()

        ksp = PETSc.KSP().create()
        ksp.setOperators(petsc_A)
        ksp.setType(self.ksp_type)
        ksp.getPC().setType(self.pc_type)
        ksp.solve(petsc_b, petsc_x)

        return petsc_x.getArray().copy()
```

**Why PETSc first**:
- Operates at linear algebra level --- minimal semantic gap
- MFG_PDE keeps full control of FDM/GFDM discretization and Picard iteration
- `petsc4py` is mature and well-maintained
- Immediate benefit: parallel solvers, AMG preconditioners for large 2D/3D problems
- No changes to MFG_PDE's problem definition or solver structure

**Integration point in MFG_PDE**: The `LinearSolverBackend` would be injected into HJB/FP solvers where they currently call `scipy.sparse.linalg.spsolve`. This is a single point of delegation per solver.

#### Tier 3: Discretization Delegation (High cost, specialized value)

Full coupling where an external framework handles spatial discretization while MFG_PDE orchestrates the MFG-specific backward-forward iteration.

**Scenario**: FEniCS assembles FEM stiffness matrices for the HJB equation on a complex 2D domain with holes. MFG_PDE's Picard iterator calls FEniCS at each iteration to solve the spatial problem, then feeds the result into the FP solver.

**Key design decision**: The `SpatialSolverBackend` should NOT have MFG-specific methods like `solve_hjb_step()` or `solve_fp_step()`. That would leak MFG semantics into what should be a generic spatial solver. Instead, MFG_PDE's Picard iterator translates MFG semantics into generic PDE coefficients, and the backend sees only a general evolution PDE.

**Why not `ParabolicCoefficients`?** The name "parabolic" assumes second-order MFG ($\sigma > 0$). But first-order MFG ($\sigma = 0$) produces a Hamilton-Jacobi equation (hyperbolic HJB) and a continuity equation (hyperbolic FP). Nonlinear diffusion, obstacle problems, and conservation laws further blur PDE type boundaries. The structure we actually need is the **convection-diffusion-reaction (CDR) form** --- a type-agnostic representation of a linearized evolution PDE at each Picard step:

$$\partial_t u = \nabla \cdot (D \nabla u) - v \cdot \nabla u + R$$

where $D$ can be zero (hyperbolic), positive scalar (parabolic), or a tensor (anisotropic). This CDR form covers all MFG variants within a single interface:

```python
@dataclass
class PDECoefficients:
    """Convection-diffusion-reaction (CDR) coefficients for a generic evolution PDE.

    Represents: ∂u/∂t = ∇·(D ∇u) - v·∇u + R

    This is PDE-type-agnostic:
    - D > 0, v = 0: parabolic (heat equation)
    - D = 0, v ≠ 0: hyperbolic (advection / Hamilton-Jacobi)
    - D > 0, v ≠ 0: convection-diffusion (second-order MFG)
    - D = 0, v = 0, R ≠ 0: reaction equation
    - Steady state (no ∂t): elliptic (Poisson, Laplace)

    MFG_PDE maps both HJB and FP to this form before delegation.
    The backend never sees MFG-specific types.
    """
    diffusion: np.ndarray | float       # D(x): diffusion coefficient or tensor (0 for 1st-order MFG)
    advection: np.ndarray | None        # v(x): advection velocity field
    reaction: np.ndarray | None         # R(x): reaction/source term
    boundary_conditions: BoundaryConditions


class SpatialSolverBackend(Protocol):
    """Delegate spatial PDE solve to an external framework.

    The backend is PDE-generic: it sees evolution equations in CDR
    coefficient form, NOT MFG-specific quantities. MFG_PDE's Picard
    iterator handles the translation.
    """

    def solve_step(
        self,
        coefficients: PDECoefficients,
        u_prev: np.ndarray,            # Solution at previous time step
        dt: float,
    ) -> np.ndarray:
        """Solve one time step of the evolution PDE defined by coefficients."""
        ...

    def interpolate_to_mfg(self, field: Any) -> np.ndarray:
        """Convert external framework's field representation to numpy array."""
        ...

    def interpolate_from_mfg(self, array: np.ndarray) -> Any:
        """Convert numpy array to external framework's field representation."""
        ...
```

**How MFG_PDE maps to CDR coefficients** --- the Picard iterator translates MFG semantics at each iteration:

```
Second-order MFG (σ > 0):

    HJB step (backward):
        coefficients = PDECoefficients(
            diffusion = σ²/2,
            advection = D_pH(∇u, m),       # optimal drift from Hamiltonian
            reaction  = -coupling(m),        # density coupling term
            bc        = hjb_bc,
        )
        u^{k+1}_{n-1} = backend.solve_step(coefficients, u^{k+1}_n, dt)

    FP step (forward):
        coefficients = PDECoefficients(
            diffusion = σ²/2,
            advection = D_pH(∇u^{k+1}, m),  # drift from updated value function
            reaction  = 0,                   # standard FP has no reaction
            bc        = fp_bc,
        )
        m^{k+1}_{n+1} = backend.solve_step(coefficients, m^{k+1}_n, dt)

First-order MFG (σ = 0):

    HJB step (backward, hyperbolic):
        coefficients = PDECoefficients(
            diffusion = 0,                   # no viscosity
            advection = D_pH(∇u, m),
            reaction  = -coupling(m),
            bc        = hjb_bc,
        )
        # Backend must use upwind/Godunov scheme, not central differences

    FP step (forward, continuity equation):
        coefficients = PDECoefficients(
            diffusion = 0,
            advection = D_pH(∇u^{k+1}, m),
            reaction  = 0,
            bc        = fp_bc,
        )
        # Backend must use conservative discretization
```

This separation means: (a) the backend adapter is reusable for any evolution PDE, not just MFG; (b) new MFG problem classes only require new coefficient mappings in the iterator, not new backend methods; (c) the backend never imports MFG_PDE types; (d) first-order and second-order MFG use the same interface, differing only in coefficient values.

**Challenges** (honest assessment):

| Challenge | Description | Severity |
|:----------|:-----------|:---------|
| DOF ordering | FEniCS numbers unknowns differently than MFG_PDE's structured grids | High |
| Quadrature mismatch | Different numerical integration rules give slightly different matrices | Medium |
| Time discretization mismatch | MFG_PDE defaults to backward Euler; external frameworks may use Crank-Nicolson or $\theta$-schemes. Different time integrators produce different spatial operators at each step. If the HJB backend uses a different temporal scheme than the FP backend, the Picard iteration may not converge to the correct coupled solution. **Mitigation**: the `PDECoefficients` + `dt` interface above intentionally delegates only the spatial discretization per time step, keeping time stepping under MFG_PDE's control. | High |
| Scheme selection for hyperbolic case | When $D = 0$ (first-order MFG), the backend must use upwind/Godunov schemes, not central differences. The `PDECoefficients` interface must communicate whether the PDE is advection-dominated so the backend selects an appropriate discretization. | Medium |
| Memory ownership | Python ↔ C++ boundary requires careful reference management | Medium |
| Error debugging | Failures span two frameworks with different error models | High |
| Conceptual gap | FEniCS: variational forms. MFG_PDE: finite differences. Different abstractions. | High |
| Performance | Format conversion overhead may negate solver speedup for small problems | Medium |

**Estimated effort**: Multi-person-month per adapter. Justified only when MFG research requires complex 2D/3D geometry that structured grids cannot handle.

#### Tier 4: Auxiliary PDE Solver Plugins (Medium cost, high research value)

MFG_PDE calls external solvers for **auxiliary equations** that arise in extended MFG formulations, while the core HJB-FP coupling remains internal.

```python
class AuxiliarySolverPlugin(Protocol):
    """Plugin for auxiliary PDE solves within the MFG iteration.

    Examples:
    - Eikonal solver for distance-to-obstacle field
    - Poisson solver for electrostatic potential
    - Stokes solver for background fluid velocity
    """

    def solve(
        self,
        geometry: GeometryProtocol,
        source: np.ndarray | None,
        boundary_conditions: BoundaryConditions | None,
        **params: Any,
    ) -> np.ndarray:
        """Solve the auxiliary equation, return field on MFG_PDE's grid."""
        ...

    @property
    def equation_type(self) -> str:
        """E.g., 'eikonal', 'poisson', 'stokes'."""
        ...
```

**Why this tier is valuable**: Many MFG extensions need a "pre-computed field" (distance to obstacle, background velocity, external potential) that is computed once or infrequently, then fed into the Hamiltonian. This is a **loose coupling** --- the auxiliary solve is essentially a preprocessing step. The semantic gap is small because MFG_PDE only needs the result as a numpy array on its grid.

**Examples**:
- `EikonalPlugin(scikit_fmm)`: Compute signed distance field for obstacle avoidance
- `PoissonPlugin(fenics)`: Solve $-\Delta \phi = \rho$ for external potential
- `StokesPlugin(fenics)`: Compute background flow for crowd-in-fluid MFG

---

## 5. Problem Composition & Extension (PCE)

### 5.1 The Mathematical Landscape of MFG Extensions

The standard MFG system $\{$HJB, FP$\}$ is the simplest case. Published research introduces numerous extensions, each adding PDE components:

#### 5.1.1 Extended FP Dynamics

The Fokker-Planck equation gains additional terms:

$$\partial_t m - \frac{\sigma^2}{2} \Delta m + \nabla \cdot (m \, \alpha^*) = \underbrace{R(m)}_{\text{reaction}} + \underbrace{\nabla \cdot (m \, v_{\text{ext}})}_{\text{external transport}} + \underbrace{S(x,t)}_{\text{source/sink}}$$

Each new term requires:
- Discretization (may need different stencils than diffusion)
- Stability analysis (reaction terms can be stiff)
- Modified adjoint (the HJB equation changes correspondingly)

#### 5.1.2 Constrained MFG

State constraints $m(t,x) \in K$ or control constraints $\alpha \in U$ transform the optimality system into a **variational inequality**:

$$\min(-\partial_t u - \frac{\sigma^2}{2}\Delta u + H(\nabla u, m), \, u - \psi) = 0$$

where $\psi$ is the obstacle. This requires:
- Obstacle problem solvers (projected iteration, penalty methods)
- Complementarity condition checking
- Modified convergence criteria

#### 5.1.3 MFG with Major Player

A "major player" (government, platform) controls a parameter $\theta(t)$ affecting all agents:

$$\min_\theta \int_0^T C(\theta, m) \, dt \quad \text{s.t.} \quad \text{MFG system with } \theta$$

This is a **bilevel optimization**: the outer problem optimizes $\theta$, the inner problem is the full MFG system. Requires:
- Sensitivity analysis of MFG w.r.t. $\theta$
- Gradient computation (adjoint of the adjoint)
- Outer optimization loop (possibly calling external optimizer)

#### 5.1.4 Multi-Physics MFG

Agents move in a physical environment governed by its own PDE:

- **Crowd + fluid**: Agents create body forces on fluid; fluid velocity affects agent drift
- **Crowd + heat**: Agent density generates heat; temperature affects agent preferences
- **Traffic + emissions**: Vehicle density produces pollution; pollution field affects route choice

Each coupling adds a PDE that must be solved at each Picard iteration.

### 5.2 Extension Architecture: How MFG_PDE Should Grow

Rather than building a generic extension framework now, identify the **minimal structural changes** that would make future extensions clean.

These hooks operate at **different architectural layers** — this is intentional, not accidental:

```
Layer             Hook                    Plugs Into
─────────────────────────────────────────────────────────────────────
Problem def.      HamiltonianBase         MFGComponents (existing)
                  StateConstraint         MFGProblem.constraints (new)
PDE assembly      FPSourceTerm            FP solver RHS (new)
Preprocessing     AuxiliaryField          Hamiltonian params / BCs (new)
Iteration logic   AuxiliarySolverPlugin   Picard iterator (new, Tier 4)
```

Each hook is independent — implementing one doesn't require the others:

#### 5.2.1 Hamiltonian Extensibility (Already Good)

MFG_PDE's `HamiltonianBase` class hierarchy already supports custom Hamiltonians:

```python
class CongestionHamiltonian(HamiltonianBase):
    """H(x, m, p, t) = |p|^2 / (2 * g(m)) where g(m) models congestion."""
    def __call__(self, x, m, p, t=0.0):
        g_m = self.congestion_function(m)
        return 0.5 * np.dot(p, p) / g_m
```

This is the right abstraction --- the Hamiltonian encodes how density affects optimality.

#### 5.2.2 Source Term Hook (Not Yet Needed, Design Sketch)

A future `SourceTerm` could be injected into the FP solver:

```python
class FPSourceTerm(Protocol):
    """Additional source/sink term for the Fokker-Planck equation.

    The FP equation becomes:
        ∂m/∂t - σ²/2 Δm + ∇·(m α*) = source(x, t, m)
    """
    def evaluate(
        self, x: np.ndarray, t: float, m: np.ndarray
    ) -> np.ndarray:
        """Return source term value at given points."""
        ...

# Usage in FP solver (conceptual):
# rhs = diffusion_term + advection_term + source.evaluate(x, t, m)
```

#### 5.2.3 Auxiliary Field Hook (Not Yet Needed, Design Sketch)

Pre-computed fields that feed into the Hamiltonian or BCs:

```python
class AuxiliaryField:
    """A field computed by an auxiliary equation and injected into MFG.

    Example: distance-to-obstacle field d(x) that enters the Hamiltonian
    as a penalty: H(x,m,p,t) = H_0(p,m) + λ/d(x)
    """
    name: str
    values: np.ndarray          # On MFG_PDE's spatial grid
    solver: AuxiliarySolverPlugin | None  # Optional: recompute during iteration

    def needs_update(self, iteration: int) -> bool:
        """Whether this field should be recomputed at this iteration."""
        ...
```

#### 5.2.4 Constraint Hook (Not Yet Needed, Design Sketch)

```python
class StateConstraint(Protocol):
    """Constraint on the MFG state during iteration.

    Example: density must remain below a maximum:
        m(t,x) ≤ m_max  (room capacity)
    """
    def project(self, m: np.ndarray) -> np.ndarray:
        """Project density onto feasible set."""
        ...

    def is_feasible(self, m: np.ndarray) -> bool:
        """Check if current density satisfies constraint."""
        ...

    def penalty(self, m: np.ndarray) -> float:
        """Penalty for constraint violation (for penalty methods)."""
        ...
```

### 5.3 When to Implement Each Extension

| Extension | Trigger | Prerequisite |
|:----------|:--------|:-------------|
| Source terms in FP | Research problem with birth/death dynamics | Stable FP solver API |
| State constraints | Research problem with capacity limits | Variational inequality solver |
| Auxiliary PDE fields | Research problem with obstacles or external physics | Tier 4 plugin system |
| Multi-physics coupling | Research problem coupling MFG to fluid/heat | Tier 3 interop + auxiliary solver |
| Major player control | Research problem with bilevel optimization | Sensitivity/adjoint framework |

**Guiding principle**: Each extension is added **only when**:
1. A concrete MFG problem variant requires it
2. A working prototype exists in mfg-research
3. The extension has been validated on at least one research experiment
4. The API surface is understood through practical use

---

## 6. Phased Roadmap

### Phase A: Core Stabilization + Design Documentation (Current -- v1.0)

**Implementation**: None. **Documentation**: This document.

Focus on:
- Complete validation initiative (#685 series)
- Stabilize `MFGProblem.solve()` pipeline
- Ensure convergence monitoring is reliable
- Document the existing API comprehensively

**Design work** (no code):
- Record interface sketches in this document (done)
- Identify which internal APIs would need to change for each tier
- Survey which external frameworks have stable Python APIs

**Rationale**: Building interop adapters on an unstable API creates maintenance burden. But recording design thinking is free and prevents losing context.

### Phase B: Data Exchange + Linear Algebra Backend (v1.0 -- v1.x)

**Deliverables**:
1. `FieldExporter` with HDF5/XDMF support for $u(t,x)$ and $m(t,x)$
2. `GeometryAdapter` with `meshio` backend (enables GMSH mesh import)
3. `LinearSolverBackend` trait with `ScipySolver` (default) and `PETScSolver` (optional)
4. Benchmark: PETSc vs SciPy on 2D MFG with $N > 10^4$ grid points

**Validation criterion**: A 2D MFG problem solved with PETSc backend produces identical results (within solver tolerance) to SciPy backend, with measurable speedup for large problems.

**Estimated effort**: 2-3 weeks for Tier 1 + Tier 2. This assumes `SparseSolver` is used as the starting point for `ScipySolver`, and `meshio` does the heavy lifting for geometry I/O.

**Testing strategy for optional dependencies**:

External frameworks (PETSc, FEniCS, meshio) are optional dependencies. The adapter test suite must handle their absence gracefully:

```python
# Pattern: pytest.importorskip for optional backend tests
petsc4py = pytest.importorskip("petsc4py", reason="PETSc not installed")

def test_petsc_solver_matches_scipy():
    """PETSc backend produces identical results to SciPy (within tolerance)."""
    A, b = assemble_test_system()
    scipy_x = ScipySolver().solve(A, b)
    petsc_x = PETScSolver().solve(A, b)
    np.testing.assert_allclose(petsc_x, scipy_x, rtol=1e-10)
```

- **CI**: Run adapter tests only when the external package is available. Use `@pytest.mark.optional` or conditional collection.
- **Mock tests**: For each `LinearSolverBackend` / `SpatialSolverBackend`, write a `MockBackend` that verifies the protocol contract without requiring the external package.
- **Equivalence tests**: Every adapter must demonstrate that it produces identical results (within solver tolerance) to the SciPy baseline on a standard 1D and 2D MFG problem.
- **Performance regression**: Track solve time for the PETSc adapter to ensure format conversion overhead doesn't negate the speedup.

### Phase C: Auxiliary Solvers + Problem Extensions (v1.x -- v2.0, demand-driven)

**Trigger**: A specific MFG research problem in mfg-research requires capabilities beyond the standard HJB-FP system.

**Expected first candidates** (based on research literature):
1. Eikonal solver plugin for obstacle-avoidance MFG (Tier 4)
2. Source term hook for population dynamics MFG
3. FEniCS spatial backend for complex-geometry MFG (Tier 3)

**Process**:
1. Implement the extension as a prototype in mfg-research
2. Validate on research experiments with quantified metrics
3. Extract the minimal necessary abstraction
4. Add to MFG_PDE with tests and documentation
5. Update this document with lessons learned

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|:-----|:-----------|:-------|:-----------|
| Premature abstraction | High (if implemented before v1.0) | Wasted engineering, wrong interfaces | Strict phasing; design only in Phase A |
| Interface sketches don't match reality | Medium | Redesign needed in Phase B | Treat sketches as starting points, not specifications |
| Adapter maintenance burden | Medium | Each adapter is a bug surface | Start with one (PETSc), validate pattern before expanding |
| Framework API changes | Medium | External frameworks evolve | Pin versions, test adapters in CI |
| Over-engineering PCE specs | High | Trait explosion, factory proliferation | Only add with working implementations |
| Research demand for extensions is delayed | Low | Document sits unused | No cost; design thinking preserved |

---

## 8. References

### External Frameworks
- FEniCS Project: https://fenicsproject.org/
- PETSc: https://petsc.org/
- deal.II: https://www.dealii.org/
- SciML (Julia): https://sciml.ai/
- Firedrake: https://www.firedrakeproject.org/
- scikit-fem: https://github.com/kinnala/scikit-fem
- meshio: https://github.com/nschloe/meshio
- scikit-fmm: https://github.com/scikit-fmm/scikit-fmm

### MFG Literature (Problem Classes)
- Lasry & Lions (2007): Original MFG formulation
- Achdou & Capuzzo-Dolcetta (2010): FDM for MFG
- Cardaliaguet et al. (2015): MFG planning problem
- Achdou et al. (2020): MFG with congestion
- Carmona & Delarue (2018): Probabilistic MFG (common noise)
- Huang, Caines & Malhamé (2006): Large population stochastic control

### Internal Documentation
- `docs/architecture/spacetime-bc/` --- Active architecture project
- `docs/architecture/spacetime-bc/SPEC_LINEAR_SOLVER.md` --- Linear solver trait design
- `docs/architecture/spacetime-bc/SPEC_OPERATOR_SYSTEM.md` --- PDE operator system
- `docs/architecture/proposals/MFG_COUPLING_SOLVERS_ARCHITECTURE.md` --- Coupling solver family
- `docs/user/guides/backend_usage.md` --- Current backend system (NumPy/PyTorch/JAX)
- `mfg_pde/core/hamiltonian.py` --- HamiltonianBase class hierarchy

---

**Decision**: Implementation deferred to post-v1.0. Design thinking and interface sketches recorded in this document for future reference. Revisit Phase B scoping after v1.0.0 release.
