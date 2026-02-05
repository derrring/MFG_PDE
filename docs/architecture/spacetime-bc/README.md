# Space-Time Solvers & Boundary Conditions Architecture

**Status**: DRAFT DISCUSSION
**Created**: 2026-02-05
**Specs**: MFG-SPEC-ST-0.8, MFG-SPEC-BC-0.2, MFG-SPEC-ADD-01

---

## 1. Vision

This project explores a paradigm shift for MFG_PDE's PDE substrate:

- **The Cylinder Manifold**: Treat time $t$ as a dimension parallel to space $x$.
  The computational domain becomes $\mathcal{Q} = [0,T] \times \Omega$.
- **Boundary Unification**: IC, TC, and spatial BCs are boundary data on $\partial\mathcal{Q}$.
- **Compositional BCs**: Boundary conditions defined by 4 orthogonal axes
  (Region x MathType x ValueSource x Enforcement).
- **Solver Agnosticism**: Sequential marching and global space-time solve
  under a unified `TrajectorySolver` protocol.

## 2. Scope Guard

> **MFG_PDE is production-ready infrastructure for Mean Field Games.**
> It is NOT a general-purpose PDE framework.

These specs contain ideas that serve MFG directly and ideas that belong
to a general PDE meta-compiler (FEniCSx/Firedrake scope). This project
explicitly separates the two.

| Category | Examples | Decision |
|:---------|:---------|:---------|
| **Build for MFG** | SpacetimeBoundaryData, 3-axis BC formalization, applicator consolidation | ✅ |
| **Build post-v1.0** | TrajectorySolver protocol, StepOperator extraction | ⏳ |
| **Out of scope** | GlobalSpacetimeSolver, SBP-SAT, Parareal, ALE, TPMS periodicity | ❌ |

See `MIGRATION_PLAN.md` for full phasing.

## 3. Documents

| Document | Purpose | Read Time |
|:---------|:--------|:----------|
| `CURRENT_STATE_ANALYSIS.md` | Gap analysis: current code vs proposed architecture | 15 min |
| `SPEC_SPACETIME_SOLVERS.md` | MFG-SPEC-ST-0.8: Space-time cylinder, TrajectorySolver, StoragePolicy | 20 min |
| `SPEC_COMPOSITIONAL_BC.md` | MFG-SPEC-BC-0.2: 4-axis BC framework + GKS/Lopatinskii-Shapiro | 20 min |
| `SPEC_PERIODIC_IMPLICIT.md` | MFG-SPEC-ADD-01: Periodic BCs on implicit geometries | 10 min |
| `MIGRATION_PLAN.md` | Phased approach: what to build, when, and why | 10 min |

## 4. Related Issues

### Directly Covered

| Issue | Title | Relevance |
|:------|:------|:----------|
| #712 | Consolidate ImplicitApplicator/MeshfreeApplicator | Subsumed by enforcement axis formalization |
| #637 | ImplicitApplicator for dual geometry BC handling | Covered by Interaction Protocol B/C |
| #658 | Operator Library Cleanup | Operator trait system (extension spec) |
| #607 | FEM face marker abstraction | Region axis of BC spec |
| #517 | BC Semantic Dispatch Factory | Compositional BC framework |
| #476 | Coupling Module Architecture Review | TrajectorySolver protocol |
| #729 | Clean up spacetime/BC documentation | This project consolidates that content |

### Context Issues

| Issue | Title | Relevance |
|:------|:------|:----------|
| #679 | Validate IC/BC compatibility with geometry | Corner consistency protocol |
| #682 | Geometry-agnostic IC/BC validation | SpacetimeBoundaryData validation |
| #707 | True Adjoint Mode | Scheme pairing relates to operator traits |
| #634 | HJB solver parameter explosion | StepOperator extraction motivation |

## 5. Key Architectural Decisions

### Decision 1: Enforcement stays solver-side

The spec proposes `BCSpec.enforcement = Strong|Weak|GhostFluid`. We reject this
at the problem-specification level. Enforcement is a **solver concern**, not a
**problem concern**. A user declares "Dirichlet u=0 on the wall"; the solver
decides how to enforce it (ghost cells for FDM, penalty for FEM).

This matches the current architecture where BCSegment specifies *what* and
applicator classes implement *how*.

### Decision 2: GKS/SBP-SAT — partially relevant, formal infrastructure deferred

- **Second-order MFG** ($\sigma > 0$): Parabolic. Standard energy estimates suffice.
- **First-order MFG** ($\sigma = 0$): HJB is Hamilton-Jacobi (hyperbolic), FP is a
  conservation law. Characteristic-aware BC treatment IS relevant.

The codebase already handles this via Godunov upwinding, WENO schemes, and
viscosity-solution-based solvers. Formal SBP-SAT infrastructure (provably
energy-stable operators) is deferred until formal stability proofs are needed.

### Decision 3: StoragePolicy deferred

Current problems fit in memory. Adding Streaming/Checkpointed/OutCore
infrastructure is premature optimization before v1.0.

---

**Last Updated**: 2026-02-05
