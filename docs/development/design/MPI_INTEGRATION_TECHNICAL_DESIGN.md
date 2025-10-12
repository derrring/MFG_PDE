# MPI Integration Technical Design

**Document Version**: 1.0
**Created**: October 7, 2025
**Status**: 🔵 PLANNING
**Phase**: Phase 3 Foundation Work

## 🎯 Objective

Design and implement distributed-memory parallel computing capabilities for MFG_PDE using MPI (Message Passing Interface), enabling:
- **3D problems** at production scale (10⁷ grid points)
- **2D large-scale problems** with 10× performance improvement (10⁶ points)
- **Linear scaling** up to 1000+ CPU cores

---

## 📐 Current Architecture Analysis

### Solver Architecture (Three Layers)

```
┌─────────────────────────────────────────────────────────┐
│         MFG Solver (Orchestration Layer)                │
│  mfg_pde/alg/numerical/mfg_solvers/                     │
│                                                          │
│  ┌─────────────────────────────────────────────┐       │
│  │ FixedPointIterator                          │       │
│  │                                              │       │
│  │  for iter in range(max_iterations):         │       │
│  │      U_new = hjb_solver.solve(M_old, ...)   │  ◄────┼─ Backward (HJB)
│  │      M_new = fp_solver.solve(U_new, ...)    │  ◄────┼─ Forward (FP)
│  │      check_convergence(U_new, M_new)        │  ◄────┼─ Global reduction
│  │                                              │       │
│  └─────────────────────────────────────────────┘       │
└───────────────────────┬─────────────┬───────────────────┘
                        │             │
        ┌───────────────┘             └───────────────┐
        │                                             │
┌───────▼────────────────────┐          ┌─────────────▼──────────────┐
│   HJB Solver (Backward)    │          │   FP Solver (Forward)       │
│  hjb_solvers/hjb_fdm.py    │          │  fp_solvers/fp_fdm.py       │
│                            │          │                             │
│  - Grid-based stencils     │          │  - Grid-based stencils      │
│  - Newton iteration        │          │  - Implicit time stepping   │
│  - Sparse matrix solve     │          │  - Sparse matrix solve      │
│  - Needs: M(t,x), U(T,x)   │          │  - Needs: U(t,x), M(0,x)    │
└────────────────────────────┘          └─────────────────────────────┘
```

### Key Operations Requiring MPI

**1. Stencil Operations** (Local with ghost cells)
```python
# From fp_fdm.py lines 80-100
for i in range(Nx):
    im1 = (i - 1 + Nx) % Nx  # Previous neighbor
    ip1 = (i + 1) % Nx        # Next neighbor
    # Compute: diffusion + advection terms
```

**2. Global Reductions** (All-reduce operations)
```python
# From fixed_point_iterator.py lines 331-341
l2distu_abs = np.linalg.norm(U - U_old) * norm_factor  # Global norm
l2distm_abs = np.linalg.norm(M - M_old) * norm_factor  # Global norm
```

**3. Boundary Conditions** (Ghost cell exchange)
```python
# From fp_fdm.py lines 38-50
if boundary_conditions.type == "dirichlet":
    m[0, 0] = left_value   # Left boundary
    m[0, -1] = right_value # Right boundary
elif boundary_conditions.type == "no_flux":
    # Ghost cell enforcement
```

---

## 🏗️ MPI Integration Design

### Design Principle: **Minimal Invasiveness**

**Strategy**: Implement MPI parallelization with **minimal changes** to existing solver code:

1. **Domain Decomposition Layer**: New abstraction managing spatial domain partitioning
2. **Communication Layer**: Handle ghost cell exchanges transparently
3. **Solver Wrappers**: MPI-aware wrappers around existing FDM solvers
4. **Factory Integration**: Automatic MPI solver creation when available

**Non-goal**: Refactoring existing solvers extensively (preserve backward compatibility)

### Architecture: Layered Approach

```
┌──────────────────────────────────────────────────────────┐
│  User Code (No Changes Required)                         │
│                                                           │
│  from mfg_pde import ExampleMFGProblem                   │
│  from mfg_pde.factory import create_solver               │
│                                                           │
│  problem = ExampleMFGProblem(...)                        │
│  solver = create_solver(problem, parallel="mpi")  ◄──────┼─ New parameter
│  result = solver.solve()                                 │
└──────────────────────────────────────────────────────────┘
                         │
                         │ (Factory detects MPI, creates MPISolver)
                         ▼
┌──────────────────────────────────────────────────────────┐
│  MPI Layer (New - mfg_pde/parallel/)                     │
│                                                           │
│  ┌────────────────────────────────────────────┐         │
│  │  MPIFixedPointIterator                     │         │
│  │  - Wraps FixedPointIterator                │         │
│  │  - Manages MPI communicator                │         │
│  │  - Coordinates global operations           │         │
│  │                                             │         │
│  │  for iter in range(max_iterations):        │         │
│  │      U_local = hjb_mpi.solve(M_local)      │  ◄──────┼─ Local + ghost
│  │      M_local = fp_mpi.solve(U_local)       │  ◄──────┼─ Local + ghost
│  │      error = self.global_norm(U, M)        │  ◄──────┼─ MPI_Allreduce
│  │                                             │         │
│  └────────────────────────────────────────────┘         │
│                       │           │                      │
│          ┌────────────┘           └───────────┐         │
│          │                                    │         │
│  ┌───────▼───────────────┐   ┌───────────────▼───────┐ │
│  │  MPIHJBSolver         │   │  MPIFPSolver          │ │
│  │  - Ghost exchange     │   │  - Ghost exchange     │ │
│  │  - Local solve        │   │  - Local solve        │ │
│  │  - Boundary sync      │   │  - Boundary sync      │ │
│  └───────────────────────┘   └───────────────────────┘ │
│                       │           │                      │
│          ┌────────────┘           └───────────┐         │
│          ▼                                    ▼         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  DomainDecomposition                             │  │
│  │  - 1D/2D/3D partitioning strategies              │  │
│  │  - Ghost cell management                         │  │
│  │  - Neighbor communication topology               │  │
│  └──────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
                         │
                         │ (Delegates to existing solvers)
                         ▼
┌──────────────────────────────────────────────────────────┐
│  Existing Solvers (Unchanged or Minimal Changes)         │
│                                                           │
│  hjb_solvers/hjb_fdm.py   (works on local domain)       │
│  fp_solvers/fp_fdm.py     (works on local domain)       │
│  fixed_point_iterator.py  (wrapped by MPI version)      │
└──────────────────────────────────────────────────────────┘
```

---

## 🔧 Implementation Components

### Component 1: Domain Decomposition (`mfg_pde/parallel/domain_decomposition.py`)

**Purpose**: Partition spatial domain across MPI ranks

**Key Classes**:

```python
class DomainDecomposition:
    """
    Manages spatial domain partitioning for MPI parallelization.

    Supports:
    - 1D decomposition: Partition along x-axis
    - 2D decomposition: 2D processor grid
    - 3D decomposition: 3D processor grid
    """

    def __init__(self, problem: MFGProblem, comm: MPI.Comm):
        self.problem = problem
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        # Determine optimal decomposition
        self.proc_grid = self._compute_processor_grid()

        # Compute local domain indices
        self.local_indices = self._compute_local_indices()

        # Identify neighbors for ghost cell exchange
        self.neighbors = self._identify_neighbors()

    def _compute_processor_grid(self) -> tuple[int, ...]:
        """
        Compute optimal processor grid dimensions.

        1D: (size,)
        2D: Factor size into (px, py) minimizing communication
        3D: Factor size into (px, py, pz) minimizing surface area
        """

    def _compute_local_indices(self) -> dict:
        """
        Compute local domain indices for this rank.

        Returns:
            {
                'x_start': int,     # Starting x index (global)
                'x_end': int,       # Ending x index (global)
                'nx_local': int,    # Local grid points (excluding ghosts)
                'nx_with_ghosts': int,  # Including ghost cells
            }
        """

    def _identify_neighbors(self) -> dict:
        """
        Identify neighbor ranks for ghost exchange.

        Returns:
            {
                'left': rank or None,
                'right': rank or None,
                # For 2D/3D:
                'top': rank or None,
                'bottom': rank or None,
                # For 3D:
                'front': rank or None,
                'back': rank or None,
            }
        """

    def global_to_local(self, global_index: int) -> int:
        """Convert global index to local index (with ghost offset)."""

    def local_to_global(self, local_index: int) -> int:
        """Convert local index (with ghost) to global index."""
```

**Decomposition Strategies**:

**1D Decomposition** (simplest, for 1D/2D problems):
```
Global domain: [0, Nx]
Ranks: 4

Rank 0: [0, Nx/4] + ghost_right
Rank 1: ghost_left + [Nx/4, Nx/2] + ghost_right
Rank 2: ghost_left + [Nx/2, 3Nx/4] + ghost_right
Rank 3: ghost_left + [3Nx/4, Nx]
```

**2D Decomposition** (for 2D problems):
```
Global domain: [0, Nx] × [0, Ny]
Ranks: 4 (2×2 grid)

┌──────────┬──────────┐
│  Rank 0  │  Rank 1  │  Each with ghost cells
├──────────┼──────────┤  on shared boundaries
│  Rank 2  │  Rank 3  │
└──────────┴──────────┘
```

### Component 2: Ghost Cell Communication (`mfg_pde/parallel/ghost_exchange.py`)

**Purpose**: Handle neighbor-to-neighbor data exchange

**Key Functions**:

```python
def exchange_ghost_cells_1d(
    local_data: np.ndarray,
    decomp: DomainDecomposition,
    num_ghosts: int = 1,
) -> np.ndarray:
    """
    Exchange ghost cells with neighbors in 1D decomposition.

    Args:
        local_data: Local array with ghost cells [ghost_left | interior | ghost_right]
        decomp: Domain decomposition info
        num_ghosts: Number of ghost cell layers (default 1 for FDM)

    Returns:
        local_data with updated ghost cells

    Implementation:
        1. Pack data to send (boundary cells → ghost cells of neighbors)
        2. Use MPI_Sendrecv for simultaneous send/receive
        3. Unpack received data into ghost cells
    """
    comm = decomp.comm

    # Extract boundary data to send
    if decomp.neighbors['left'] is not None:
        send_left = local_data[num_ghosts : 2*num_ghosts]  # Send to left neighbor

    if decomp.neighbors['right'] is not None:
        send_right = local_data[-2*num_ghosts : -num_ghosts]  # Send to right neighbor

    # Simultaneous exchange with left neighbor
    if decomp.neighbors['left'] is not None:
        recv_left = comm.sendrecv(
            send_left, dest=decomp.neighbors['left'],
            source=decomp.neighbors['left']
        )
        local_data[:num_ghosts] = recv_left

    # Simultaneous exchange with right neighbor
    if decomp.neighbors['right'] is not None:
        recv_right = comm.sendrecv(
            send_right, dest=decomp.neighbors['right'],
            source=decomp.neighbors['right']
        )
        local_data[-num_ghosts:] = recv_right

    return local_data


def exchange_ghost_cells_2d(
    local_data: np.ndarray,
    decomp: DomainDecomposition,
    num_ghosts: int = 1,
) -> np.ndarray:
    """
    Exchange ghost cells in 2D decomposition (4 or 8 neighbors).

    Similar to 1D but handles:
    - Horizontal exchanges (left/right)
    - Vertical exchanges (top/bottom)
    - Corner exchanges (diagonal neighbors for 2nd-order stencils)
    """
```

### Component 3: Global Operations (`mfg_pde/parallel/global_ops.py`)

**Purpose**: Implement collective operations (reductions, norms)

```python
def global_l2_norm(local_array: np.ndarray, dx: float, comm: MPI.Comm) -> float:
    """
    Compute global L2 norm across all ranks.

    L2 norm = sqrt(sum_i |u_i|^2 * dx)

    Implementation:
        1. Compute local contribution: local_sum = sum(local_array**2) * dx
        2. MPI_Allreduce with SUM operation: global_sum = sum(local_sum_i)
        3. Return sqrt(global_sum)
    """
    local_sum = np.sum(local_array**2) * dx
    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    return np.sqrt(global_sum)


def global_convergence_check(
    U_new: np.ndarray, U_old: np.ndarray,
    M_new: np.ndarray, M_old: np.ndarray,
    dx: float, dt: float,
    decomp: DomainDecomposition,
) -> dict:
    """
    Compute global convergence metrics.

    Returns:
        {
            'l2distu_abs': float,
            'l2distm_abs': float,
            'l2distu_rel': float,
            'l2distm_rel': float,
        }
    """
    norm_factor = np.sqrt(dx * dt)

    # Global L2 differences
    l2distu_abs = global_l2_norm(U_new - U_old, dx, decomp.comm) * norm_factor
    l2distm_abs = global_l2_norm(M_new - M_old, dx, decomp.comm) * norm_factor

    # Global L2 norms for relative error
    norm_U = global_l2_norm(U_new, dx, decomp.comm) * norm_factor
    norm_M = global_l2_norm(M_new, dx, decomp.comm) * norm_factor

    l2distu_rel = l2distu_abs / norm_U if norm_U > 1e-12 else l2distu_abs
    l2distm_rel = l2distm_abs / norm_M if norm_M > 1e-12 else l2distm_abs

    return {
        'l2distu_abs': l2distu_abs,
        'l2distm_abs': l2distm_abs,
        'l2distu_rel': l2distu_rel,
        'l2distm_rel': l2distm_rel,
    }
```

### Component 4: MPI Solver Wrappers

#### `MPIFixedPointIterator` (`mfg_pde/parallel/mpi_fixed_point_iterator.py`)

**Purpose**: MPI-parallel version of FixedPointIterator

```python
class MPIFixedPointIterator(BaseMFGSolver):
    """
    MPI-parallel Fixed Point Iterator for MFG systems.

    Wraps existing FixedPointIterator with MPI parallelization:
    - Domain decomposition across ranks
    - Ghost cell communication
    - Global convergence checking
    """

    def __init__(
        self,
        problem: MFGProblem,
        hjb_solver: BaseHJBSolver,
        fp_solver: BaseFPSolver,
        comm: MPI.Comm | None = None,
        decomposition: DomainDecomposition | None = None,
        **kwargs
    ):
        super().__init__(problem)

        # MPI setup
        if comm is None:
            comm = MPI.COMM_WORLD
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        # Domain decomposition
        if decomposition is None:
            self.decomp = DomainDecomposition(problem, comm)
        else:
            self.decomp = decomposition

        # Wrap solvers with MPI versions
        self.hjb_solver = MPIHJBSolver(hjb_solver, self.decomp)
        self.fp_solver = MPIFPSolver(fp_solver, self.decomp)

        # Use original solver parameters
        self.thetaUM = kwargs.get('thetaUM', 0.5)

    def solve(self, max_iterations: int, tolerance: float = 1e-5, **kwargs):
        """
        Solve MFG system with MPI parallelization.

        Key differences from serial version:
        1. Initialize local arrays (with ghost cells)
        2. Each rank solves its local subdomain
        3. Ghost cell exchanges after each HJB/FP solve
        4. Global convergence check via MPI_Allreduce
        """
        # Initialize local arrays
        U_local, M_local = self._initialize_local_arrays()

        for iiter in range(max_iterations):
            U_old_local = U_local.copy()
            M_old_local = M_local.copy()

            # 1. Solve HJB backward (local subdomain)
            U_new_local = self.hjb_solver.solve_hjb_system(M_old_local, ...)

            # Exchange ghost cells for U
            U_new_local = exchange_ghost_cells_1d(U_new_local, self.decomp)

            # 2. Solve FP forward (local subdomain)
            M_new_local = self.fp_solver.solve_fp_system(U_new_local, ...)

            # Exchange ghost cells for M
            M_new_local = exchange_ghost_cells_1d(M_new_local, self.decomp)

            # 3. Apply damping
            U_local = self.thetaUM * U_new_local + (1 - self.thetaUM) * U_old_local
            M_local = self.thetaUM * M_new_local + (1 - self.thetaUM) * M_old_local

            # 4. Global convergence check
            conv_metrics = global_convergence_check(
                U_local, U_old_local, M_local, M_old_local,
                self.problem.Dx, self.problem.Dt, self.decomp
            )

            # Root rank prints convergence
            if self.rank == 0:
                print(f"Iter {iiter+1}: U_err={conv_metrics['l2distu_rel']:.2e}, "
                      f"M_err={conv_metrics['l2distm_rel']:.2e}")

            # Check convergence (all ranks have same metrics via allreduce)
            if conv_metrics['l2distu_rel'] < tolerance and conv_metrics['l2distm_rel'] < tolerance:
                break

        # Gather full solution on root (optional - for visualization)
        U_global, M_global = self._gather_solution(U_local, M_local)

        return U_global, M_global, iiter+1, conv_metrics
```

#### `MPIHJBSolver` and `MPIFPSolver`

**Purpose**: Wrap existing HJB/FP solvers to work with local domains

```python
class MPIHJBSolver:
    """Wrapper for HJB solver to work with MPI decomposition."""

    def __init__(self, base_solver: BaseHJBSolver, decomp: DomainDecomposition):
        self.base_solver = base_solver
        self.decomp = decomp

        # Create local problem with adjusted dimensions
        self.local_problem = self._create_local_problem()

    def _create_local_problem(self) -> MFGProblem:
        """
        Create local problem with adjusted spatial dimensions.

        Key adjustments:
        - Nx_local = decomp.nx_local (interior points only)
        - xmin_local, xmax_local from decomp
        - Same Nt, T, sigma, coefCT as global problem
        """

    def solve_hjb_system(self, M_local, U_final, U_prev):
        """
        Solve HJB on local subdomain.

        Steps:
        1. Extract interior data (exclude ghost cells)
        2. Call base_solver.solve_hjb_system() with local data
        3. Return result with ghost cells (to be filled by exchange)
        """
        # Extract interior
        M_interior = M_local[:, self.decomp.ghost_width:-self.decomp.ghost_width]

        # Solve on interior using existing solver
        U_interior = self.base_solver.solve_hjb_system(M_interior, U_final, U_prev)

        # Pad with ghost cells (will be filled by exchange)
        U_local_with_ghosts = self._add_ghost_cells(U_interior)

        return U_local_with_ghosts
```

---

## 📋 Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Goal**: Basic MPI infrastructure and 1D decomposition

**Tasks**:
1. ✅ Create `mfg_pde/parallel/` package structure
2. ✅ Implement `DomainDecomposition` for 1D
3. ✅ Implement `exchange_ghost_cells_1d()`
4. ✅ Implement `global_l2_norm()` and convergence checking
5. ✅ Add mpi4py to optional dependencies

**Deliverable**: Core MPI utilities tested with simple 1D arrays

### Phase 2: MPI Solver Wrappers (Week 2-3)

**Goal**: Working MPI fixed-point iterator for 1D problems

**Tasks**:
1. ✅ Implement `MPIFixedPointIterator`
2. ✅ Implement `MPIHJBSolver` wrapper
3. ✅ Implement `MPIFPSolver` wrapper
4. ✅ Add factory support: `create_solver(problem, parallel="mpi")`
5. ✅ Create 1D test problem

**Deliverable**: 1D MFG problem solvable with MPI (2-4 ranks)

### Phase 3: Validation & Optimization (Week 3-4)

**Goal**: Verify correctness and measure performance

**Tasks**:
1. ✅ Unit tests: ghost exchange, global reductions
2. ✅ Integration test: Compare MPI vs serial results
3. ✅ Weak scaling study: 1D problem, 1→16 ranks
4. ✅ Strong scaling study: Fixed problem size, 1→16 ranks
5. ✅ Identify performance bottlenecks

**Deliverable**: Validated 1D MPI solver with scaling benchmarks

### Phase 4: 2D/3D Extension (Week 5-6)

**Goal**: Extend to 2D and 3D problems

**Tasks**:
1. ✅ Implement 2D domain decomposition
2. ✅ Implement `exchange_ghost_cells_2d()`
3. ✅ Test 2D MFG problem (32×32 → 128×128)
4. ✅ Implement 3D decomposition (optional)
5. ✅ Benchmark 2D/3D scaling

**Deliverable**: 2D/3D MPI solver with scaling validation

---

## 🧪 Testing Strategy

### Unit Tests

**Test File**: `tests/parallel/test_domain_decomposition.py`
```python
def test_1d_decomposition_4_ranks():
    """Test 1D decomposition correctness."""
    # Mock MPI comm with 4 ranks
    # Verify:
    # - Each rank has correct local indices
    # - No gaps in global coverage
    # - Neighbors correctly identified

def test_ghost_exchange_1d():
    """Test ghost cell exchange correctness."""
    # Setup: Each rank with unique data
    # Exchange ghosts
    # Verify: Ghost cells match neighbor's boundary
```

**Test File**: `tests/parallel/test_global_ops.py`
```python
def test_global_l2_norm():
    """Test global L2 norm computation."""
    # Each rank with known data
    # Compute global norm
    # Verify against serial computation

def test_convergence_check():
    """Test global convergence metrics."""
    # Mock U/M arrays
    # Compute convergence
    # Verify all ranks get same result
```

### Integration Tests

**Test File**: `tests/integration/test_mpi_solver.py`
```python
def test_mpi_vs_serial_1d():
    """Compare MPI solver to serial solver."""
    # Problem: 1D LQ-MFG, Nx=64, Nt=40
    # Solve with serial solver
    # Solve with MPI solver (4 ranks)
    # Assert solutions match within tolerance

@pytest.mark.parametrize("num_ranks", [2, 4, 8])
def test_mpi_scaling(num_ranks):
    """Test MPI solver with different rank counts."""
    # Problem: 1D, Nx=128
    # Solve with specified num_ranks
    # Verify convergence
    # Measure solve time
```

### Performance Benchmarks

**Benchmark File**: `benchmarks/mpi_evaluation/mpi_scaling_benchmark.py`
```python
def benchmark_weak_scaling():
    """
    Weak scaling: Problem size grows with ranks.

    Ideal: Constant solve time
    Nx=64 (1 rank) → Nx=512 (8 ranks) → Nx=1024 (16 ranks)
    """

def benchmark_strong_scaling():
    """
    Strong scaling: Fixed problem size.

    Ideal: Solve time ~ 1/num_ranks
    Nx=512, ranks: 1, 2, 4, 8, 16
    """
```

---

## 📊 Expected Performance

### Theoretical Scaling

**Communication Cost Analysis**:
- **Ghost exchange**: O(boundary_size) = O(N^(d-1)) for d-dimensional problem
- **Global reduction**: O(log P) for P ranks
- **Computation**: O(N^d / P) per rank

**Communication-to-Computation Ratio**:
```
C/C_ratio = O(N^(d-1)) / O(N^d / P) = O(P / N)
```

**Implication**: For good scaling, need N >> P (large problems benefit most)

### Target Performance (Phase 3 Goals)

| Problem Size | Serial Time | MPI Target (16 ranks) | Speedup |
|:-------------|:------------|:----------------------|:--------|
| **1D (10³)** | 0.5s | 0.3s | 1.7× |
| **2D (10⁶)** | 300s | 30s | 10× |
| **3D (10⁷)** | Not feasible | <5 min | ∞ → finite |

### Scaling Efficiency

**Target Parallel Efficiency**:
- **E(16) > 0.8**: 16 ranks achieve >80% of ideal speedup
- **E(64) > 0.6**: 64 ranks achieve >60% of ideal speedup
- **E(256) > 0.4**: 256 ranks achieve >40% of ideal speedup

**Where**: E(P) = T₁ / (P · Tₚ), T₁ = serial time, Tₚ = parallel time with P ranks

---

## 🔗 Dependencies

### Required Packages

**mpi4py** (Python MPI bindings):
```bash
pip install mpi4py
```

**System MPI Implementation** (one of):
- **OpenMPI**: `brew install open-mpi` (macOS) or `apt install openmpi-bin` (Linux)
- **MPICH**: Alternative MPI implementation
- **Intel MPI**: For Intel systems

### Optional but Recommended

**h5py** (parallel HDF5 for checkpoint I/O):
```bash
CC=mpicc HDF5_MPI="ON" pip install --no-binary=h5py h5py
```

---

## 📚 References

### MPI Resources
- **MPI Standard 4.0**: https://www.mpi-forum.org/docs/
- **mpi4py Documentation**: https://mpi4py.readthedocs.io/
- **Parallel Programming Patterns**: Gropp et al., "Using MPI" (3rd edition)

### Domain Decomposition Literature
- **PDE Solvers**: Smith et al., "Domain Decomposition Methods for Partial Differential Equations"
- **Load Balancing**: Catalyurek & Aykanat, "Hypergraph-partitioning-based decomposition"

### Related MFG Implementations
- None known with MPI support - MFG_PDE will be **first production MPI implementation**

---

## ✅ Success Criteria

### Correctness
- ✅ MPI solver produces same results as serial solver (tolerance: 1e-6)
- ✅ Mass conservation maintained across domain boundaries
- ✅ Convergence rates match serial implementation

### Performance
- ✅ 1D problems: Linear scaling up to 16 ranks (E > 0.8)
- ✅ 2D problems: 10× speedup on 16-64 ranks for 10⁶ grid points
- ✅ 3D problems: Feasibility demonstrated for 10⁷ grid points

### Usability
- ✅ Factory integration: `create_solver(problem, parallel="mpi")` works
- ✅ No code changes required for existing problems
- ✅ Automatic MPI detection and fallback to serial

---

**Status**: 🔵 **DESIGN COMPLETE** - Ready for Phase 1 implementation (Foundation work)

**Next Step**: Begin Phase 1 implementation (Domain decomposition and ghost exchange)

**Last Updated**: October 7, 2025
**Review Date**: November 2025 (after Phase 1 complete)
