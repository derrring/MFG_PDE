# PyTorch Compatibility Strategy

**Status**: 📋 Planning
**Priority**: High (RL/DL infrastructure dependency)
**Target**: Phase 3.x (RL/Neural paradigms)

## Strategic Principle

**Core Insight**: PyTorch compatibility is **critical for RL/DL paradigms**, but **optional for classical numerical methods**.

### Priorities by Paradigm

| Paradigm | PyTorch Priority | Rationale |
|:---------|:-----------------|:----------|
| **Neural** (PINN, DGM, FNO) | 🔴 **CRITICAL** | Neural networks ARE PyTorch |
| **RL** (Actor-Critic, DDPG, Nash-Q) | 🔴 **CRITICAL** | RL libraries built on PyTorch |
| **Optimization** (JKO, Wasserstein) | 🟡 **OPTIONAL** | May benefit from autodiff |
| **Numerical** (FDM, Particle, WENO) | 🟢 **LOW** | NumPy/JAX sufficient |

## Implementation Strategy

### Tier 1: MUST be PyTorch-Compatible (RL/DL Core)

**Components that MUST work with PyTorch tensors:**

1. **Neural Paradigm Solvers** (Already PyTorch-native):
   - `mfg_pde/alg/neural/pinn/` - Physics-Informed Neural Networks
   - `mfg_pde/alg/neural/dgm/` - Deep Galerkin Method
   - `mfg_pde/alg/neural/fno/` - Fourier Neural Operator (future)
   - ✅ **Status**: Already PyTorch-native

2. **RL Paradigm Infrastructure** (Critical for RL):
   - `mfg_pde/alg/rl/` - Reinforcement Learning solvers
   - Actor-Critic, DDPG, Nash Q-Learning implementations
   - Gymnasium environment integration
   - ✅ **Status**: Already PyTorch-native (Stable-Baselines3 dependency)

3. **Hybrid Interfaces** (RL ↔ Numerical):
   - Environment wrappers that call numerical solvers
   - Reward computation from PDE solutions
   - **Required**: Backend-agnostic array handling
   - ⚠️ **Action Needed**: Ensure NumPy ↔ PyTorch conversions at boundaries

### Tier 2: Backend-Agnostic (Numerical Methods)

**Components that should support multiple backends via abstraction:**

1. **Classical Numerical Solvers** (Keep backend-agnostic):
   - `mfg_pde/alg/numerical/hjb_solvers/` - HJB solvers (FDM, WENO, Semi-Lagrangian)
   - `mfg_pde/alg/numerical/fp_solvers/` - FP solvers (Particle, FDM)
   - `mfg_pde/alg/numerical/mfg_solvers/` - Coupled MFG solvers

   **Strategy**: Use backend abstraction layer
   ```python
   # Solver uses backend API, not direct NumPy/PyTorch
   def solve(self):
       u = self.backend.zeros(shape)  # Works with any backend
       grad = self.backend.gradient(u)  # Backend handles implementation
       return self.backend.to_numpy(u)  # Convert at boundary
   ```

2. **Backend Abstraction Layer** (Already exists):
   - `mfg_pde/backends/numpy_backend.py` ✅
   - `mfg_pde/backends/torch_backend.py` ✅
   - `mfg_pde/backends/jax_backend.py` ✅
   - **Principle**: Solvers call backend methods, backends handle tensor types

### Tier 3: NumPy-Only (Legacy/Special Cases)

**Components that can remain NumPy-only:**

1. **Utilities and Helpers**:
   - Plotting/visualization functions
   - Logging and diagnostics
   - Configuration management
   - ✅ **Status**: Can use NumPy exclusively (convert at boundaries)

2. **Legacy Examples** (if any):
   - Old demonstration scripts
   - Archive materials
   - ✅ **Status**: Convert to NumPy at output if needed

## Boundary Conversion Pattern

**Key Principle**: Convert at boundaries, not internally.

### Pattern 1: RL Environment ← Numerical Solver

```python
class MFGEnvironment(gym.Env):
    """RL environment that uses numerical solver internally."""

    def __init__(self, problem):
        self.solver = create_standard_solver(problem)
        # Solver uses NumPy/JAX backend internally (faster)

    def step(self, action):
        # Convert PyTorch action to NumPy
        action_np = action.cpu().numpy() if torch.is_tensor(action) else action

        # Solve with numerical backend (fast, NumPy)
        result = self.solver.compute_reward(action_np)

        # Convert back to PyTorch for RL algorithm
        reward = torch.from_numpy(result).float()

        return observation, reward, done, info
```

### Pattern 2: Neural Solver → Uses PyTorch Throughout

```python
class PINNSolver:
    """Neural solver - PyTorch native, no conversion needed."""

    def __init__(self, problem):
        self.network = torch.nn.Sequential(...)  # Pure PyTorch
        self.optimizer = torch.optim.Adam(...)

    def solve(self):
        # Everything is PyTorch tensors
        loss = self.compute_loss(x_torch, t_torch)
        loss.backward()
        self.optimizer.step()
        return solution_torch  # Return PyTorch, convert later if needed
```

### Pattern 3: Hybrid Neural-Numerical

```python
class HybridSolver:
    """Uses neural network + numerical validation."""

    def solve(self):
        # Neural forward pass (PyTorch)
        u_neural = self.neural_network(x_torch)

        # Convert to NumPy for numerical solver
        u_init = u_neural.detach().cpu().numpy()

        # Numerical refinement (NumPy backend)
        self.numerical_solver.set_initial_guess(u_init)
        u_refined = self.numerical_solver.solve()

        # Convert back to PyTorch if needed for gradient
        u_final = torch.from_numpy(u_refined)
        return u_final
```

## Action Plan for MPS Support

### ✅ Already Compatible (No Work Needed)

1. **Neural paradigm**: Pure PyTorch, works with MPS
2. **RL paradigm**: Pure PyTorch, works with MPS
3. **Backend abstraction**: Exists and works

### ⚠️ Needs Attention (Boundary Conversions)

**Issue**: Numerical solvers create NumPy arrays, passed to PyTorch backend causes type error.

**Solution**: Fix boundary conversions in solver factories and iterators.

#### Location 1: `FixedPointIterator` (MFG Solver Core)

**File**: `mfg_pde/alg/numerical/mfg_solvers/fixed_point_iterator.py`

**Current Problem**:
```python
def solve(self):
    U = np.zeros(...)  # NumPy array
    M = np.zeros(...)  # NumPy array

    # If backend is PyTorch, these assignments fail
    result = self.backend.some_operation(U)  # Type error!
```

**Fix**:
```python
def solve(self):
    # Use backend-native array creation
    U = self.backend.zeros(shape, dtype=self.backend.float64)
    M = self.backend.zeros(shape, dtype=self.backend.float64)

    # Now everything is consistent
    result = self.backend.some_operation(U)  # Works!
```

#### Location 2: Solver Initialization

**Files**: `hjb_fdm.py`, `fp_particle.py`, etc.

**Fix Pattern**:
```python
class HJBFDMSolver:
    def __init__(self, problem, backend=None):
        self.backend = backend or create_backend("numpy")  # Default NumPy

    def solve(self):
        # Use backend methods, not np.* directly
        u = self.backend.zeros(self.shape)  # Not np.zeros!
        grad = self.backend.gradient(u)
        return u  # Backend-native type
```

### 🎯 Minimal Fix Strategy

**Goal**: Enable PyTorch MPS for RL/Neural paradigms without touching all numerical code.

**Approach**: Fix only the **interface points** where RL/Neural meets Numerical.

**Files to Modify** (Estimated 5-10 files):

1. **`FixedPointIterator`** - Main MFG coupling solver
2. **Solver factories** - `create_standard_solver()`, etc.
3. **RL environment wrappers** - Where RL calls numerical solvers
4. **Backend interface** - Add explicit conversion helpers

**Estimated Effort**: 3-5 days (not weeks!)

## Verification Strategy

### Test Suite for PyTorch Compatibility

```python
@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_solver_backend_compatibility(backend):
    """Test solver works with different backends."""
    problem = create_lq_problem()

    # Create solver with specified backend
    solver = create_standard_solver(problem, backend=backend)

    # Should work regardless of backend
    result = solver.solve()

    # Verify result type matches backend
    if backend == "torch":
        assert torch.is_tensor(result.U)
    else:
        assert isinstance(result.U, np.ndarray)
```

### MPS-Specific Test

```python
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_acceleration():
    """Test PyTorch MPS acceleration works end-to-end."""
    problem = create_lq_problem()

    # Create solver with MPS backend
    backend = create_backend("torch", device="mps")
    solver = create_standard_solver(problem, backend=backend)

    # Should complete without type errors
    result = solver.solve()

    # Verify computation happened on MPS
    assert result.U.device.type == "mps"
```

## Decision Matrix

### When to Use Each Backend

| Use Case | Recommended Backend | Rationale |
|:---------|:-------------------|:----------|
| **Pure RL training** | PyTorch (MPS/CUDA) | Leverage GPU, integrate with RL libs |
| **Pure Neural (PINN/DGM)** | PyTorch (MPS/CUDA) | Neural networks ARE PyTorch |
| **Numerical benchmarking** | NumPy or JAX | Mature, well-tested, stable |
| **Hybrid RL-Numerical** | PyTorch + convert | Use PyTorch for RL, NumPy for speed |
| **Production inference** | JAX (XLA compile) | Fastest for deployment |
| **Prototyping** | NumPy | Simplest, most familiar |

## Summary: What Actually Needs Changing?

### ❌ Do NOT Change (Keep as-is):
- Numerical solver algorithms (HJB, FP logic)
- Individual scheme implementations (upwind, WENO, etc.)
- Utility functions (plotting, logging)
- Examples (unless demonstrating backend features)

### ✅ DO Change (Boundary interfaces only):
1. **Array initialization**: Use `backend.zeros()` not `np.zeros()`
2. **Solver factories**: Pass backend through to solvers
3. **FixedPointIterator**: Use backend-native arrays
4. **RL-Numerical interfaces**: Add explicit conversions
5. **Backend tests**: Verify all backends work

### 🎯 Practical Timeline

**Week 1-2**: Fix FixedPointIterator and solver initialization
**Week 3**: Add RL environment boundary conversions
**Week 4**: Test suite for multi-backend compatibility
**Week 5**: MPS validation and performance benchmarks

**Total**: ~1 month for robust PyTorch/MPS support across RL/Neural paradigms

## References

1. **Backend Abstraction**: `mfg_pde/backends/`
2. **FixedPointIterator**: `mfg_pde/alg/numerical/mfg_solvers/fixed_point_iterator.py`
3. **RL Infrastructure**: `mfg_pde/alg/rl/`
4. **Neural Solvers**: `mfg_pde/alg/neural/`
5. **PyTorch MPS Docs**: https://pytorch.org/docs/stable/notes/mps.html

---

**Last Updated**: October 8, 2025
**Next Action**: Create issue for "PyTorch MPS boundary conversion fixes"
**Owner**: @derrring
