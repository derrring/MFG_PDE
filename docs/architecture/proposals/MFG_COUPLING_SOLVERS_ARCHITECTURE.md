# MFG Coupling Solvers Architecture

**Status**: Proposal
**Author**: Claude Code
**Date**: 2025-12-17
**Related Issue**: TBD

---

## 1. Executive Summary

This document proposes an architecture for extending the MFG coupling solver infrastructure beyond the current damped fixed-point iterator. The goal is to provide a family of iteration methods (Newton, Quasi-Newton, Krylov, Block methods) that users can select based on problem characteristics.

**Current State**: Single solver (`FixedPointIterator`) with optional Anderson acceleration.

**Proposed State**: Unified solver family with consistent API, leveraging existing `nonlinear_solvers.py` utilities.

---

## 2. Mathematical Background

### 2.1 The MFG Coupling Problem

The Mean Field Game system couples two PDEs:

**HJB (backward)**:
$$-\partial_t u - \frac{\sigma^2}{2}\Delta u + H(\nabla u, m) = 0, \quad u(T,x) = g(x)$$

**FP (forward)**:
$$\partial_t m - \frac{\sigma^2}{2}\Delta m - \nabla \cdot (m \cdot D_p H(\nabla u, m)) = 0, \quad m(0,x) = m_0(x)$$

After discretization, we obtain a nonlinear system:
$$\mathbf{F}(\mathbf{U}, \mathbf{M}) = \begin{pmatrix} \mathbf{F}_{HJB}(\mathbf{U}, \mathbf{M}) \\ \mathbf{F}_{FP}(\mathbf{M}, \mathbf{U}) \end{pmatrix} = \mathbf{0}$$

where $\mathbf{U} \in \mathbb{R}^{N_t \times N_x}$ and $\mathbf{M} \in \mathbb{R}^{N_t \times N_x}$.

### 2.2 Iteration Method Classification

| Category | Methods | Convergence | Key Property |
|----------|---------|-------------|--------------|
| **Stationary** | Picard, Gauss-Seidel, Jacobi | Linear O(ρ^k) | Simple, robust |
| **Accelerated Stationary** | Anderson, Aitken | Superlinear | Low overhead |
| **Newton Family** | Newton, Quasi-Newton | Quadratic O(ε^{2^k}) | Fast near solution |
| **Krylov** | GMRES, BiCGSTAB | Problem-dependent | Matrix-free possible |

### 2.3 Method Details

#### 2.3.1 Fixed-Point (Picard) Iteration [CURRENT]

```
Given: (U^k, M^k)
1. Solve HJB backward: U^{k+1} = HJB_solve(M^k)
2. Solve FP forward:   M^{k+1} = FP_solve(U^{k+1})
3. Damping: (U^{k+1}, M^{k+1}) = ω·(U^{k+1}, M^{k+1}) + (1-ω)·(U^k, M^k)
```

- **Convergence**: Linear, rate depends on coupling strength
- **Cost per iteration**: O(N_t × N_x) per PDE solve
- **Pros**: Simple, robust, preserves structure (positivity of M)
- **Cons**: Slow for strongly coupled problems

#### 2.3.2 Newton Method

```
Given: (U^k, M^k), solve:
J(U^k, M^k) · δ = -F(U^k, M^k)
(U^{k+1}, M^{k+1}) = (U^k, M^k) + α·δ
```

where Jacobian:
$$J = \begin{pmatrix} \partial_U F_{HJB} & \partial_M F_{HJB} \\ \partial_U F_{FP} & \partial_M F_{FP} \end{pmatrix}$$

- **Convergence**: Quadratic near solution
- **Cost per iteration**: O(N²) for Jacobian + O(N³) for linear solve (or O(N²) with iterative)
- **Pros**: Fast convergence, well-understood theory
- **Cons**: Requires Jacobian, expensive per iteration, needs good initial guess

#### 2.3.3 Quasi-Newton (BFGS/L-BFGS)

Approximates Jacobian from gradient history:
$$B_{k+1} = B_k + \frac{y_k y_k^T}{y_k^T s_k} - \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k}$$

- **Convergence**: Superlinear
- **Cost per iteration**: O(N) for L-BFGS (limited memory)
- **Pros**: No explicit Jacobian needed
- **Cons**: May not capture MFG structure well

#### 2.3.4 Block Gauss-Seidel

```
Given: (U^k, M^k)
1. Solve: ∂_U F_HJB(U, M^k) · δU = -F_HJB(U^k, M^k)  → U^{k+1}
2. Solve: ∂_M F_FP(M, U^{k+1}) · δM = -F_FP(M^k, U^{k+1})  → M^{k+1}
```

- **Convergence**: Linear (typically faster than Jacobi)
- **Pros**: Natural for MFG (respects HJB→FP causality)
- **Cons**: Still linear convergence

#### 2.3.5 Krylov Methods (GMRES, BiCGSTAB)

Solve linearized system iteratively without forming full Jacobian:
- Only need matrix-vector products: $J \cdot v$
- Can use finite differences: $J \cdot v \approx (F(x + εv) - F(x))/ε$

- **Convergence**: Depends on conditioning
- **Pros**: Matrix-free, good for large systems
- **Cons**: Requires good preconditioner

---

## 3. Current Infrastructure Analysis

### 3.1 Existing Components

```
mfg_pde/
├── utils/numerical/
│   ├── nonlinear_solvers.py      # FixedPointSolver, NewtonSolver, PolicyIterationSolver
│   ├── anderson_acceleration.py   # AndersonAccelerator
│   └── convergence.py             # Convergence metrics
│
├── alg/numerical/coupling/
│   ├── base_mfg.py               # BaseMFGSolver (abstract)
│   ├── fixed_point_iterator.py   # FixedPointIterator (main solver)
│   ├── fixed_point_utils.py      # Helper functions
│   └── ...
│
└── utils/solver_result.py        # SolverResult dataclass
```

### 3.2 What We Have

| Component | Status | Notes |
|-----------|--------|-------|
| `FixedPointSolver` | Ready | Generic utility |
| `NewtonSolver` | Ready | With autodiff, sparse, line search |
| `AndersonAccelerator` | Ready | Type-I and Type-II |
| `BaseMFGSolver` | Ready | Abstract base class |
| `FixedPointIterator` | Ready | Main coupling solver |
| `SolverResult` | Ready | Standardized output |

### 3.3 What We Need

| Component | Priority | Complexity |
|-----------|----------|------------|
| MFG residual computation | High | Medium |
| MFG Jacobian (autodiff) | High | High |
| Newton MFG solver | High | Medium |
| Preconditioner interface | Medium | Medium |
| Block Gauss-Seidel | Medium | Low |
| Krylov wrapper | Low | Medium |
| Quasi-Newton | Low | Medium |

---

## 4. Proposed Architecture

### 4.1 Directory Structure

```
mfg_pde/alg/numerical/coupling/
├── __init__.py
├── base_mfg.py                     # BaseMFGSolver [EXISTS]
│
├── # --- Stationary Methods ---
├── fixed_point_iterator.py         # Picard + Anderson [EXISTS]
├── block_iterators.py              # Gauss-Seidel, Jacobi [NEW]
│
├── # --- Newton Family ---
├── newton_mfg_solver.py            # Full Newton [NEW]
├── quasi_newton_mfg_solver.py      # L-BFGS variant [NEW, LOW PRIORITY]
│
├── # --- Krylov Methods ---
├── krylov_mfg_solver.py            # GMRES/BiCGSTAB wrapper [NEW, LOW PRIORITY]
│
├── # --- Utilities ---
├── fixed_point_utils.py            # [EXISTS]
├── mfg_residual.py                 # Residual computation [NEW]
├── mfg_jacobian.py                 # Jacobian computation [NEW]
├── preconditioners.py              # Preconditioner interface [NEW]
│
└── # --- Specialized ---
    ├── hybrid_fp_particle_hjb_fdm.py  # [EXISTS]
    └── network_mfg_solver.py          # [EXISTS]
```

### 4.2 Class Hierarchy

```
BaseMFGSolver (abstract)
├── FixedPointIterator          # Picard iteration [EXISTS]
├── BlockIteratorMFG            # Gauss-Seidel, Jacobi [NEW]
├── NewtonMFGSolver             # Full Newton [NEW]
├── QuasiNewtonMFGSolver        # L-BFGS [NEW]
├── KrylovMFGSolver             # GMRES wrapper [NEW]
└── HybridFPParticleHJBFDM      # Specialized [EXISTS]
```

### 4.3 Core Interfaces

#### 4.3.1 MFG Residual Interface

```python
# mfg_pde/alg/numerical/coupling/mfg_residual.py

class MFGResidual:
    """Compute MFG system residual F(U, M) = 0."""

    def __init__(
        self,
        problem: MFGProblem,
        hjb_solver: BaseHJBSolver,
        fp_solver: BaseFPSolver,
    ):
        self.problem = problem
        self.hjb_solver = hjb_solver
        self.fp_solver = fp_solver

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Compute residual from packed state vector."""
        U, M = self.unpack_state(state)
        R_hjb = self.compute_hjb_residual(U, M)
        R_fp = self.compute_fp_residual(M, U)
        return self.pack_state(R_hjb, R_fp)

    def compute_hjb_residual(self, U: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Compute HJB PDE residual."""
        # -∂_t U - (σ²/2)ΔU + H(∇U, M) - interior equations
        ...

    def compute_fp_residual(self, M: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Compute FP PDE residual."""
        # ∂_t M - (σ²/2)ΔM - ∇·(M·α(U)) - interior equations
        ...

    def pack_state(self, U: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Pack (U, M) into single state vector."""
        return np.concatenate([U.ravel(), M.ravel()])

    def unpack_state(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Unpack state vector into (U, M)."""
        n = state.size // 2
        U = state[:n].reshape(self.shape)
        M = state[n:].reshape(self.shape)
        return U, M
```

#### 4.3.2 Newton MFG Solver Interface

```python
# mfg_pde/alg/numerical/coupling/newton_mfg_solver.py

class NewtonMFGSolver(BaseMFGSolver):
    """
    Newton method for coupled MFG system.

    Solves F(U, M) = 0 using Newton iteration with automatic
    Jacobian computation (autodiff or finite differences).

    Features:
        - Leverages existing NewtonSolver utility
        - Supports JAX autodiff for efficient Jacobian
        - Optional line search for robustness
        - Warm start from FixedPointIterator solution

    Recommended usage:
        1. Run FixedPointIterator for initial iterations
        2. Switch to Newton for fast final convergence
    """

    def __init__(
        self,
        problem: MFGProblem,
        hjb_solver: BaseHJBSolver,
        fp_solver: BaseFPSolver,
        jacobian_method: Literal["autodiff", "finite_diff"] = "autodiff",
        line_search: bool = True,
        use_sparse: bool = True,
    ):
        super().__init__(problem)
        self.hjb_solver = hjb_solver
        self.fp_solver = fp_solver
        self.jacobian_method = jacobian_method
        self.line_search = line_search
        self.use_sparse = use_sparse

        # Build residual functor
        self.residual = MFGResidual(problem, hjb_solver, fp_solver)

        # Configure Newton solver from utilities
        from mfg_pde.utils.numerical.nonlinear_solvers import NewtonSolver
        self.newton = NewtonSolver(
            sparse=use_sparse,
            line_search=line_search,
            use_jax_autodiff=(jacobian_method == "autodiff"),
        )

    def solve(
        self,
        max_iterations: int = 30,
        tolerance: float = 1e-8,
        initial_guess: tuple[np.ndarray, np.ndarray] | None = None,
        **kwargs,
    ) -> SolverResult:
        """Solve MFG system using Newton method."""

        # Get initial guess
        if initial_guess is not None:
            U0, M0 = initial_guess
        elif self.has_warm_start_data:
            U0, M0 = self.get_warm_start_data()
        else:
            # Cold start: use boundary conditions
            U0, M0 = self._initialize_cold_start()

        # Pack into state vector
        state0 = self.residual.pack_state(U0, M0)

        # Configure Newton solver
        self.newton.max_iterations = max_iterations
        self.newton.tolerance = tolerance

        # Solve
        state_sol, info = self.newton.solve(self.residual, state0)

        # Unpack solution
        U, M = self.residual.unpack_state(state_sol)

        return SolverResult(
            U=U,
            M=M,
            iterations=info.iterations,
            error_history_U=np.array(info.residual_history),
            error_history_M=np.array(info.residual_history),
            solver_name="NewtonMFGSolver",
            converged=info.converged,
            metadata={"jacobian_evals": info.extra.get("jacobian_evals", 0)},
        )
```

#### 4.3.3 Solver Selection Factory

```python
# mfg_pde/alg/numerical/coupling/__init__.py

def create_mfg_coupling_solver(
    problem: MFGProblem,
    hjb_solver: BaseHJBSolver,
    fp_solver: BaseFPSolver,
    method: Literal["picard", "anderson", "gauss_seidel", "newton"] = "picard",
    **kwargs,
) -> BaseMFGSolver:
    """
    Factory for MFG coupling solvers.

    Args:
        problem: MFG problem definition
        hjb_solver: HJB solver instance
        fp_solver: FP solver instance
        method: Iteration method
            - "picard": Damped fixed-point (default, robust)
            - "anderson": Picard + Anderson acceleration (faster)
            - "gauss_seidel": Block Gauss-Seidel
            - "newton": Full Newton (fast convergence, needs good initial)
        **kwargs: Method-specific options

    Returns:
        Configured MFG coupling solver

    Example:
        >>> solver = create_mfg_coupling_solver(
        ...     problem, hjb_solver, fp_solver,
        ...     method="anderson",
        ...     damping_factor=0.7,
        ...     anderson_depth=5,
        ... )
        >>> result = solver.solve(max_iterations=100)
    """
    if method == "picard":
        return FixedPointIterator(
            problem, hjb_solver, fp_solver,
            use_anderson=False,
            **kwargs,
        )
    elif method == "anderson":
        return FixedPointIterator(
            problem, hjb_solver, fp_solver,
            use_anderson=True,
            anderson_depth=kwargs.pop("anderson_depth", 5),
            **kwargs,
        )
    elif method == "gauss_seidel":
        return BlockIteratorMFG(
            problem, hjb_solver, fp_solver,
            block_method="gauss_seidel",
            **kwargs,
        )
    elif method == "newton":
        return NewtonMFGSolver(
            problem, hjb_solver, fp_solver,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
```

---

## 5. Implementation Roadmap

### Phase 1: Foundation (Priority: High)

| Task | Effort | Dependencies |
|------|--------|--------------|
| Create `mfg_residual.py` | 2-3 days | HJB/FP solvers with residual methods |
| Add residual methods to HJB/FP solvers | 2-3 days | None |
| Create `newton_mfg_solver.py` | 2-3 days | `mfg_residual.py` |
| Tests for Newton solver | 1-2 days | Newton solver |

### Phase 2: Block Methods (Priority: Medium)

| Task | Effort | Dependencies |
|------|--------|--------------|
| Create `block_iterators.py` | 1-2 days | Phase 1 |
| Implement Gauss-Seidel | 1 day | Block iterator base |
| Implement Jacobi | 1 day | Block iterator base |

### Phase 3: Advanced (Priority: Low)

| Task | Effort | Dependencies |
|------|--------|--------------|
| Preconditioner interface | 2-3 days | Phase 1 |
| Krylov wrapper | 2-3 days | Preconditioner |
| Quasi-Newton (L-BFGS) | 2-3 days | Phase 1 |

---

## 6. Method Selection Guide

### 6.1 Decision Tree

```
Start
  │
  ├─ Problem size < 1000 unknowns?
  │    ├─ Yes → Newton (fast convergence)
  │    └─ No  → Continue
  │
  ├─ Strongly coupled (γ large)?
  │    ├─ Yes → Anderson-accelerated Picard
  │    └─ No  → Picard with damping ω ≈ 0.7
  │
  ├─ Need high accuracy (tol < 1e-10)?
  │    ├─ Yes → Picard → Newton (hybrid)
  │    └─ No  → Picard sufficient
  │
  └─ JAX available?
       ├─ Yes → Newton with autodiff Jacobian
       └─ No  → Quasi-Newton or Picard
```

### 6.2 Performance Expectations

| Method | Iterations to 1e-6 | Time/iter | Total time |
|--------|-------------------|-----------|------------|
| Picard (ω=0.5) | 50-200 | Fast | Moderate |
| Anderson-Picard | 20-50 | Fast | Fast |
| Gauss-Seidel | 30-100 | Fast | Moderate |
| Newton | 5-15 | Slow | Fast (small N) |

---

## 7. Testing Strategy

### 7.1 Unit Tests

- Residual computation correctness
- Jacobian accuracy (compare autodiff vs finite diff)
- Convergence on simple problems (LQ-MFG)

### 7.2 Integration Tests

- Compare Newton vs Picard solutions
- Verify same solution for all methods
- Mass conservation

### 7.3 Benchmark Problems

1. **Linear-Quadratic MFG**: Analytical solution available
2. **Crowd Motion 1D**: Standard benchmark
3. **Crowd Motion 2D**: Current experiment

---

## 8. Open Questions

1. **Jacobian sparsity**: MFG Jacobian is structured but not sparse in the classical sense. How to exploit?

2. **Preconditioners**: What preconditioner works well for MFG systems?
   - Block diagonal (separate HJB/FP)?
   - Incomplete LU?
   - Multigrid?

3. **Hybrid strategies**: When to switch from Picard to Newton?
   - After fixed number of iterations?
   - When residual drops below threshold?
   - Adaptive based on convergence rate?

4. **Memory for large problems**: Newton requires O(N²) storage for Jacobian. L-BFGS only O(N) but may not capture MFG structure.

---

## 9. References

1. Achdou, Y., & Capuzzo-Dolcetta, I. (2010). Mean field games: Numerical methods.
2. Carlini, E., & Silva, F. J. (2014). A fully discrete semi-Lagrangian scheme for MFG.
3. Benamou, J. D., & Carlier, G. (2015). Augmented Lagrangian methods for MFG.
4. Nocedal, J., & Wright, S. J. (2006). Numerical Optimization.
5. Walker, H. F., & Ni, P. (2011). Anderson acceleration for fixed-point iterations.

---

## Appendix A: Jacobian Structure

The MFG Jacobian has block structure:

```
J = [ ∂F_HJB/∂U   ∂F_HJB/∂M ]
    [ ∂F_FP/∂U    ∂F_FP/∂M  ]
```

where:
- `∂F_HJB/∂U`: Spatial operator (diffusion + Hamiltonian gradient)
- `∂F_HJB/∂M`: Coupling term from H(∇u, m)
- `∂F_FP/∂U`: Drift dependence on value function
- `∂F_FP/∂M`: Spatial operator (diffusion + drift)

Each block is sparse banded for FDM discretization.
