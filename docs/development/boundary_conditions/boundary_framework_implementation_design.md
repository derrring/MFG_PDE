# Boundary Framework Implementation Design

**Status**: Active Development Plan
**Created**: 2025-01-03
**GitHub Issues**: [#535](https://github.com/zvezda/MFG_PDE/issues/535) (Framework Enhancement), [#536](https://github.com/zvezda/MFG_PDE/issues/536) (Particle Absorbing BC)
**Theory Reference**: [boundary_framework_mathematical_foundation.md](../theory/boundary_framework_mathematical_foundation.md)

---

## Overview

This document outlines the implementation design for enhancing the MFG_PDE boundary condition framework. It complements the mathematical theory document with concrete architecture, API design, and implementation phases.

**Design Philosophy**: Specification-Application-Validation separation with solver-agnostic BC definition and solver-specific interpretation.

---

## Architecture Overview

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER SPECIFICATION                                 │
│                                                                             │
│   BCSegment(name="exit_A", bc_type=DIRICHLET, value=0, region={...})       │
│   BCSegment(name="wall", bc_type=REFLECTING, boundary="all")                │
│                              │                                              │
└──────────────────────────────┼──────────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BoundaryConditions                                    │
│                                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                    │
│   │  segments[] │───▶│ get_bc_at   │───▶│ matches_    │                    │
│   │             │    │ _point()    │    │ point()     │                    │
│   └─────────────┘    └─────────────┘    └─────────────┘                    │
│                              │                                              │
└──────────────────────────────┼──────────────────────────────────────────────┘
                               ▼
        ┌──────────────────────┴──────────────────────┐
        │                                             │
        ▼                                             ▼
┌───────────────────┐                     ┌───────────────────┐
│   FIELD SOLVERS   │                     │  PARTICLE SOLVER  │
│                   │                     │                   │
│  HJB-FDM/GFDM     │                     │  FPParticleSolver │
│  FP-FDM/GFDM      │                     │                   │
│                   │                     │                   │
│  DIRICHLET:       │                     │  DIRICHLET:       │
│    u = g          │                     │    ABSORB         │
│                   │                     │                   │
│  NEUMANN:         │                     │  REFLECTING:      │
│    du/dn = g      │                     │    BOUNCE         │
│                   │                     │                   │
│  PERIODIC:        │                     │  PERIODIC:        │
│    wrap grid      │                     │    wrap coords    │
└───────────────────┘                     └───────────────────┘
```

### Phase Dependencies

```
                    ┌─────────────────────────────┐
                    │  Phase 1: Particle Absorb   │
                    │  (Issue #536)               │
                    │                             │
                    │  - Segment-aware BC query   │
                    │  - Absorbing at DIRICHLET   │
                    │  - Exit flux tracking       │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │  Phase 2: L-S Validation    │
                    │  (Issue #535)               │
                    │                             │
                    │  - Well-posedness check     │
                    │  - Stability warnings       │
                    │  - HJB-FP compatibility     │
                    └─────────────┬───────────────┘
                                  │
          ┌───────────────────────┴───────────────────────┐
          │                                               │
          ▼                                               ▼
┌─────────────────────────┐                 ┌─────────────────────────┐
│  Phase 3: BC Matrices   │                 │  Phase 4: Neural BC     │
│  (Issue #535)           │                 │  (Issue #535)           │
│                         │                 │                         │
│  - Sparse BC encoding   │                 │  - Symbolic loss terms  │
│  - Elimination/Lagrange │                 │  - PINN/DGM interface   │
│  - GFDM integration     │                 │  - Auto-diff compatible │
└─────────────────────────┘                 └─────────────────────────┘
```

### Particle Solver BC Flow (Phase 1)

```
                         ┌─────────────────┐
                         │  Particle Step  │
                         │  x += drift*dt  │
                         │  x += noise     │
                         └────────┬────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Is particle outside    │
                    │  domain bounds?         │
                    └─────────────┬───────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
              ┌─────────┐                 ┌─────────┐
              │   NO    │                 │   YES   │
              └────┬────┘                 └────┬────┘
                   │                           │
                   ▼                           ▼
              ┌─────────┐         ┌────────────────────────┐
              │  KEEP   │         │  bc = get_bc_at_point  │
              │         │         │  (particle_position)   │
              └─────────┘         └───────────┬────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
                    ▼                         ▼                         ▼
           ┌────────────────┐       ┌────────────────┐       ┌────────────────┐
           │ bc == DIRICHLET│       │ bc == REFLECT  │       │ bc == PERIODIC │
           └───────┬────────┘       └───────┬────────┘       └───────┬────────┘
                   │                        │                        │
                   ▼                        ▼                        ▼
           ┌────────────────┐       ┌────────────────┐       ┌────────────────┐
           │    ABSORB      │       │    BOUNCE      │       │     WRAP       │
           │                │       │                │       │                │
           │ absorbed[i]=T  │       │ v = -v         │       │ x = x mod L    │
           │ record exit    │       │ x = reflect(x) │       │                │
           └────────────────┘       └────────────────┘       └────────────────┘
```

### L-S Validation Flow (Phase 2)

```
┌──────────────────────────────────────────────────────────────────┐
│                      Problem Setup                                │
│                                                                  │
│  HJB: -u_t - (sigma^2/2) Delta u + H(x, Du) = F[m]              │
│  FP:   m_t - (sigma^2/2) Delta m - div(m * D_p H) = 0           │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                    LSValidator.validate()                         │
│                                                                  │
│  For each BCSegment:                                             │
│    1. Identify PDE type (elliptic/parabolic/hyperbolic)         │
│    2. Check diffusion coefficient                                │
│    3. Analyze drift direction at boundary                        │
│    4. Apply L-S criterion                                        │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   L-S PASSED    │    │   L-S WARNING   │    │   L-S FAILED    │
│                 │    │                 │    │                 │
│  sigma > 0      │    │  sigma ~ 0      │    │  Outflow +      │
│  Dirichlet OK   │    │  Advection dom  │    │  Dirichlet      │
│  Neumann OK     │    │  Check drift    │    │  = overdetermined│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
    [Proceed]            [Warn + Proceed]         [Error/Suggest]
```

### Boundary Matrix Assembly (Phase 3)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Original System                                  │
│                                                                         │
│                    A * u = b                                            │
│                                                                         │
│   ┌                           ┐   ┌     ┐     ┌     ┐                  │
│   │  a_11  a_12  ...  a_1n   │   │ u_1 │     │ b_1 │                  │
│   │  a_21  a_22  ...  a_2n   │   │ u_2 │     │ b_2 │                  │
│   │   :     :    ...   :     │ * │  :  │  =  │  :  │                  │
│   │  a_n1  a_n2  ...  a_nn   │   │ u_n │     │ b_n │                  │
│   └                           ┘   └     ┘     └     ┘                  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    BC Matrix Builder                                     │
│                                                                         │
│   For boundary node i with Dirichlet BC u_i = g:                        │
│                                                                         │
│   Row i:  [0, 0, ..., 1, ..., 0]    (1 at position i)                  │
│   RHS i:  g                                                             │
│                                                                         │
│   For boundary node i with Neumann BC du/dn = h:                        │
│                                                                         │
│   Row i:  [..., -1/dx, 1/dx, ...]   (FD stencil for normal derivative) │
│   RHS i:  h                                                             │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Modified System                                  │
│                                                                         │
│   Method: Elimination                                                   │
│   ┌                           ┐   ┌     ┐     ┌     ┐                  │
│   │  1     0    ...   0      │   │ u_1 │     │ g_1 │  <- BC row       │
│   │  0    a_22  ...  a_2n    │   │ u_2 │     │ b'_2│                  │
│   │  :     :    ...   :      │ * │  :  │  =  │  :  │                  │
│   │  0    a_n2  ...  a_nn    │   │ u_n │     │ b'_n│                  │
│   └                           ┘   └     ┘     └     ┘                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Current Architecture

### Existing Components

```
mfg_pde/geometry/boundary/
├── types.py          # BCType enum, BCSegment dataclass
├── conditions.py     # BoundaryConditions class, factory functions
├── masks.py          # boundary_segment_mask(), region utilities
├── ghost.py          # Ghost cell utilities for FDM
├── dispatch.py       # apply_bc() dispatcher (not fully wired)
├── applicator_fdm.py       # FDMApplicator (ghost cells)
├── applicator_fem.py       # FEMApplicator (matrix assembly)
├── applicator_meshfree.py  # MeshfreeApplicator, ParticleReflector
└── applicator_graph.py     # GraphApplicator (network domains)
```

### Integration Gaps

| Component | Status | Gap |
|:----------|:-------|:----|
| BCSegment | Complete | Well-designed for mixed BC |
| BoundaryConditions | Complete | `get_bc_at_point()` exists |
| FDM Solvers | Integrated | Use ghost buffers |
| GFDM Solvers | Partial | Custom implementation, not using applicators |
| Particle Solver | **Gap** | Only reads uniform topology, ignores segments |
| L-S Validation | Missing | No algebraic verification |
| Neural Interface | Missing | No symbolic BC for neural solvers |

---

## Implementation Phases

### Phase 1: Particle Absorbing BC (Issue #536)

**Goal**: Enable segment-aware absorbing boundaries for `FPParticleSolver`.

**Architecture Change**:
```
FPParticleSolver
├── _apply_boundaries()         # Current: uniform periodic/reflecting
│   └── calls _periodic_bc() or _reflecting_bc()
│
└── _apply_boundaries()         # Enhanced: segment-aware
    └── for each boundary point:
        ├── bc = self.bc.get_bc_at_point(point)
        ├── if bc.bc_type == DIRICHLET → mark absorbed
        ├── if bc.bc_type == REFLECTING → elastic reflection
        └── if bc.bc_type == PERIODIC → wrap coordinates
```

**Key Insight**: `DIRICHLET` for particles = absorbing (solver interprets semantics).

**Implementation Steps**:

1. **Add BC query to particle step** (`fp_particle.py`):
   ```python
   def _apply_segment_aware_boundaries(self, particles, domain_bounds):
       """Apply boundary conditions based on BCSegment matching."""
       absorbed_mask = np.zeros(len(particles), dtype=bool)

       for i, p in enumerate(particles):
           if self._is_at_boundary(p, domain_bounds):
               bc = self.bc.get_bc_at_point(p, domain_bounds)
               if bc.bc_type == BCType.DIRICHLET:
                   absorbed_mask[i] = True  # Mark for removal
               elif bc.bc_type == BCType.REFLECTING:
                   particles[i] = self._reflect(p, domain_bounds)
               elif bc.bc_type == BCType.PERIODIC:
                   particles[i] = self._wrap(p, domain_bounds)

       return particles[~absorbed_mask], absorbed_mask
   ```

2. **Track absorbed particles for mass flux computation**:
   ```python
   self.exit_flux_history = []  # Particles absorbed per timestep
   self.exit_locations = []      # Where they exited (for multi-exit analysis)
   ```

3. **Backward compatibility**: Default behavior unchanged when BC is uniform.

**Testing**:
- Unit test: single exit, verify particles absorbed
- Integration test: multi-exit, verify correct exit attribution
- Regression: periodic BC behavior unchanged

---

### Phase 2: L-S Stability Validation (Issue #535)

**Goal**: Compile-time or setup-time verification of BC well-posedness.

**Architecture**:
```
mfg_pde/geometry/boundary/validation/
├── lopatinski_shapiro.py   # L-S condition checker
├── compatibility.py        # HJB-FP BC compatibility
└── diagnostics.py          # Warnings, error messages
```

**Key Classes**:

```python
@dataclass
class LSValidationResult:
    """Result of Lopatinski-Shapiro condition check."""
    is_satisfied: bool
    boundary_region: str
    pde_type: str  # "elliptic", "parabolic", "hyperbolic"
    details: str
    recommendations: list[str]

class LSValidator:
    """Validates boundary conditions against L-S stability criterion."""

    def validate(
        self,
        bc: BoundaryConditions,
        pde_type: str,
        diffusion: float = 1.0,
        drift: np.ndarray | None = None,
    ) -> list[LSValidationResult]:
        """
        Check L-S condition for each boundary segment.

        Args:
            bc: Boundary conditions to validate
            pde_type: "hjb_parabolic", "fp_parabolic", "elliptic"
            diffusion: Diffusion coefficient (sigma^2/2)
            drift: Optional drift field for advection-diffusion

        Returns:
            List of validation results, one per segment
        """
        ...
```

**L-S Logic by PDE Type**:

| PDE Type | BC Type | L-S Status | Condition |
|:---------|:--------|:-----------|:----------|
| Elliptic | Dirichlet | OK | Always |
| Elliptic | Neumann | OK | Always |
| Elliptic | Tangential | FAIL | Always |
| Parabolic (diffusion) | Dirichlet | OK | sigma > 0 |
| Parabolic (diffusion) | Neumann | OK | sigma > 0 |
| Parabolic (advection) | Dirichlet (outflow) | WARN | May be overdetermined |
| Hyperbolic | Dirichlet (inflow) | OK | characteristic entering |
| Hyperbolic | Dirichlet (outflow) | FAIL | characteristic leaving |

**Usage**:
```python
validator = LSValidator()
results = validator.validate(bc, pde_type="fp_parabolic", diffusion=0.1)
for r in results:
    if not r.is_satisfied:
        warnings.warn(f"L-S failure at {r.boundary_region}: {r.details}")
```

---

### Phase 3: Boundary Matrix Assembly (Issue #535)

**Goal**: Unified BC encoding in matrix form for implicit solvers.

**Interface**:
```python
class BoundaryMatrixBuilder:
    """Builds sparse matrices encoding boundary conditions."""

    def build_bc_matrices(
        self,
        bc: BoundaryConditions,
        mesh: StructuredGrid | UnstructuredMesh,
        pde_type: str,
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        """
        Build constraint matrix B and RHS g such that Bu = g.

        Returns:
            B: Sparse constraint matrix (n_constraints x n_dofs)
            g: RHS vector (n_constraints,)
        """
        ...

    def apply_to_system(
        self,
        A: sparse.csr_matrix,
        b: np.ndarray,
        method: str = "elimination",
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        """
        Apply BC to linear system Ax = b.

        Methods:
            - "elimination": Zero rows/cols, set diagonal to 1
            - "lagrange": Add constraint rows
            - "penalty": Large diagonal penalty
        """
        ...
```

**Matrix Assembly for BCTypes**:

| BC Type | Matrix Row | RHS |
|:--------|:-----------|:----|
| Dirichlet | `[0, ..., 1, ..., 0]` (identity at DOF) | `g` |
| Neumann | `[..., -1/h, 1/h, ...]` (FD normal derivative) | `g` |
| Robin | `[..., alpha + beta/h, -beta/h, ...]` | `g` |
| Periodic | `[1, 0, ..., 0, -1]` (wrap) | `0` |

---

### Phase 4: Neural BC Interface (Issue #535)

**Goal**: Symbolic BC specification for PINN/DGM solvers.

**Architecture**:
```python
class NeuralBCInterface:
    """Symbolic boundary conditions for neural network solvers."""

    def get_loss_terms(
        self,
        bc: BoundaryConditions,
        boundary_points: np.ndarray,
        model: Callable,
    ) -> dict[str, Callable]:
        """
        Generate loss functions for each BC type.

        Returns:
            Dict mapping BC names to loss functions.
        """
        loss_terms = {}

        for segment in bc.segments:
            if segment.bc_type == BCType.DIRICHLET:
                loss_terms[segment.name] = lambda x: (model(x) - segment.value)**2
            elif segment.bc_type == BCType.NEUMANN:
                loss_terms[segment.name] = lambda x: (
                    grad(model)(x) @ segment.normal - segment.value
                )**2

        return loss_terms
```

---

## Applicator Pattern

### Design Decision: Pattern A (Solver Owns Applicator)

Solvers internally select the appropriate applicator for their discretization method.
Users specify BC semantically; implementation details are hidden.

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Code                                │
│                                                                 │
│   bc = mixed_bc(segments=[                                      │
│       BCSegment("exit", BCType.DIRICHLET, value=0, region=...) │
│       BCSegment("wall", BCType.REFLECTING, boundary="all")     │
│   ])                                                            │
│   solver = FPParticleSolver(problem, boundary_conditions=bc)    │
│   result = solver.solve()   # Applicator handles BC internally  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Applicator Hierarchy

```
                    ┌─────────────────────┐
                    │  BaseApplicator     │
                    │  (Protocol/ABC)     │
                    │                     │
                    │  + apply(data, bc)  │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  FDMApplicator  │   │ MeshfreeAppl.   │   │ ParticleAppl.   │
│                 │   │                 │   │                 │
│  Ghost cells    │   │  Weight matrix  │   │  Absorb/Reflect │
│  Padding modes  │   │  Normal-based   │   │  Wrap coords    │
└─────────────────┘   └─────────────────┘   └─────────────────┘
         │                     │                     │
         ▼                     ▼                     ▼
    HJB-FDM              HJB-GFDM             FPParticleSolver
    FP-FDM               FP-GFDM
    WENO
```

### Solver-Applicator Binding

Each solver internally instantiates the correct applicator:

```python
class FPParticleSolver:
    """Lagrangian FP solver with segment-aware BC."""

    def __init__(
        self,
        problem: MFGProblem,
        boundary_conditions: BoundaryConditions | None = None,
        **kwargs,
    ):
        self.bc = boundary_conditions or self._default_bc()
        self._applicator = ParticleApplicator()  # Internal, auto-selected

    def _apply_boundaries(self, particles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Delegate BC enforcement to applicator."""
        return self._applicator.apply(
            particles=particles,
            bc=self.bc,
            domain_bounds=self.domain_bounds,
        )


class HJBFDMSolver:
    """FDM-based HJB solver."""

    def __init__(self, problem, boundary_conditions=None, **kwargs):
        self.bc = boundary_conditions or self._default_bc()
        self._applicator = FDMApplicator()  # Internal, auto-selected

    def _apply_ghost_cells(self, u: np.ndarray) -> np.ndarray:
        """Delegate ghost cell computation to applicator."""
        return self._applicator.apply(u, self.bc, self.domain_bounds)
```

### BCType Interpretation Matrix

```
                          APPLICATOR TYPE
                    ┌────────────────────────────────────────────────────┐
                    │   FDMApplicator  │  MeshfreeAppl. │ ParticleAppl. │
┌───────────────────┼──────────────────┼────────────────┼───────────────┤
│  DIRICHLET        │  u_ghost = g     │  Modify row    │  ABSORB       │
│  (u = g)          │                  │  u_i = g       │  Remove       │
├───────────────────┼──────────────────┼────────────────┼───────────────┤
│  NEUMANN          │  u_ghost =       │  Normal deriv  │  (N/A)        │
│  (∂u/∂n = g)      │  u_int + g*dx    │  constraint    │               │
├───────────────────┼──────────────────┼────────────────┼───────────────┤
│  NO_FLUX          │  u_ghost =       │  Zero normal   │  REFLECT      │
│  (J·n = 0)        │  u_interior      │  derivative    │  Bounce       │
├───────────────────┼──────────────────┼────────────────┼───────────────┤
│  REFLECTING       │  (= NO_FLUX)     │  (= NO_FLUX)   │  REFLECT      │
│                   │                  │                │  Bounce       │
├───────────────────┼──────────────────┼────────────────┼───────────────┤
│  PERIODIC         │  u_ghost =       │  DOF coupling  │  WRAP         │
│                   │  u_opposite      │                │  x mod L      │
├───────────────────┼──────────────────┼────────────────┼───────────────┤
│  ROBIN            │  Linear combo    │  Mixed BC      │  Partial      │
│  (αu + β∂u/∂n = g)│  ghost cell      │  constraint    │  absorption   │
└───────────────────┴──────────────────┴────────────────┴───────────────┘
```

---

## API Design

### Solver Interface Contract

All solvers accept `BoundaryConditions` and internally bind an applicator:

```python
class BaseSolver:
    def __init__(
        self,
        problem: MFGProblem,
        boundary_conditions: BoundaryConditions | None = None,
        **kwargs,
    ):
        # BC resolution order:
        # 1. Explicit parameter
        # 2. problem.geometry.boundary_conditions
        # 3. Solver-specific default
        self.bc = boundary_conditions or \
                  getattr(problem.geometry, 'boundary_conditions', None) or \
                  self._default_bc()

        # Applicator: solver knows its own discretization type
        self._applicator = self._create_applicator()

    def _create_applicator(self) -> BaseApplicator:
        """Override in subclass to return appropriate applicator."""
        raise NotImplementedError
```

### User-Facing API (Unchanged)

```python
# User specifies BC semantically - no knowledge of applicators needed
bc = mixed_bc(
    dimension=2,
    segments=[
        BCSegment("exit_A", BCType.DIRICHLET, value=0, boundary="right", region={"y": (4, 6)}),
        BCSegment("exit_B", BCType.DIRICHLET, value=0, boundary="left", region={"y": (4, 6)}),
        BCSegment("walls", BCType.REFLECTING, boundary="all", priority=-1),
    ]
)

solver = FPParticleSolver(problem, boundary_conditions=bc)
result = solver.solve()  # Applicator handles absorb/reflect internally
```

---

## Testing Strategy

### Unit Tests

```
tests/unit/geometry/boundary/
├── test_lopatinski_shapiro.py     # L-S validation
├── test_boundary_matrix.py        # Matrix assembly
├── test_particle_absorbing.py     # Particle BC
└── test_neural_interface.py       # Neural BC
```

### Integration Tests

```
tests/integration/
├── test_mixed_bc_crowd_motion.py  # Multi-exit scenario
├── test_ls_warnings.py            # Stability diagnostics
└── test_bc_cross_solver.py        # Same BC, different solvers
```

### Validation Experiments

```
examples/advanced/boundary_conditions/
├── multi_exit_comparison.py       # Compare particle vs field absorbing
├── ls_stability_demo.py           # Show L-S failure cases
└── mass_conservation_analysis.py  # Track mass under different BCs
```

---

## Dependencies

### New Files to Create

| File | Phase | Purpose |
|:-----|:------|:--------|
| `validation/lopatinski_shapiro.py` | 2 | L-S checker |
| `validation/compatibility.py` | 2 | HJB-FP consistency |
| `matrix/builder.py` | 3 | Sparse BC matrices |
| `neural/interface.py` | 4 | Symbolic BC for PINN |

### Files to Modify

| File | Phase | Changes |
|:-----|:------|:--------|
| `fp_solvers/fp_particle.py` | 1 | Add segment-aware BC |
| `conditions.py` | 2 | Add validation hooks |
| `hjb_gfdm.py` | 3 | Use matrix builder |
| `fp_gfdm.py` | 3 | Use matrix builder |

---

## Timeline Estimate

| Phase | Scope | Effort |
|:------|:------|:-------|
| Phase 1 | Particle absorbing BC | 1-2 days |
| Phase 2 | L-S validation | 2-3 days |
| Phase 3 | Boundary matrices | 3-5 days |
| Phase 4 | Neural interface | 2-3 days |

**Total**: 8-13 days of focused development.

---

## References

- **Theory**: [boundary_framework_mathematical_foundation.md](../theory/boundary_framework_mathematical_foundation.md)
- **Capability Matrix**: [BC_CAPABILITY_MATRIX.md](./BC_CAPABILITY_MATRIX.md)
- **Practical Reference**: [BOUNDARY_CONDITIONS_REFERENCE.md](../theory/BOUNDARY_CONDITIONS_REFERENCE.md)
- **Issue #535**: BC Framework Enhancement (L-S, matrices, neural)
- **Issue #536**: Particle Absorbing BC

---

*This document provides the implementation roadmap. For mathematical foundations, see the theory document.*
