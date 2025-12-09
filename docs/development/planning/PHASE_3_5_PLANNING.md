# Phase 3.5: Continuous Environments Library - Implementation Plan

**Date**: October 3, 2025
**Status**: Planning & Design
**Branch**: `feature/continuous-environments-library`
**Issue**: [#64](https://github.com/derrring/MFG_PDE/issues/64)
**Estimated Effort**: 2-3 weeks

---

## 🎯 Objectives

Create a comprehensive benchmark suite of continuous control environments for demonstrating and comparing DDPG, TD3, and SAC algorithms on realistic Mean Field Game problems.

### Success Criteria

- ✅ 5 diverse environments implemented with consistent API (LQ-MFG + 4 application domains)
- ✅ Base environment class with standardized interface
- ✅ Working demonstration for each environment
- ✅ Comprehensive test coverage (60+ tests)
- ✅ Benchmark comparing all three algorithms (DDPG, TD3, SAC)
- ✅ Theory documentation with mathematical formulations
- ✅ User guide with environment selection guidelines

---

## 📋 Implementation Phases

### Phase 1: Foundation (Days 1-2)

#### 1.1 Base Environment Class
**File**: `mfg_pde/alg/reinforcement/environments/continuous_mfg_env_base.py`

**Features**:
- Gymnasium-compatible API
- Population distribution tracking (histogram-based)
- Mean field coupling computation
- Standardized observation/action spaces
- Episode termination logic

**Interface**:
```python
class ContinuousMFGEnvBase(gym.Env):
    """Base class for continuous action MFG environments."""

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        population_bins: int = 100,
        dt: float = 0.01,
        max_steps: int = 200
    ):
        pass

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Reset environment to initial state."""

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one timestep."""

    def get_population_state(self) -> np.ndarray:
        """Return population density histogram."""

    def compute_mean_field_coupling(
        self,
        state: np.ndarray,
        population: np.ndarray
    ) -> float:
        """Compute MF interaction term for reward."""
```

**Testing**:
- API compliance (Gymnasium interface)
- Population tracking accuracy
- Seed reproducibility
- Edge cases (empty/full populations)

**Estimated**: 4-6 hours

---

### Phase 2: Core Environments (Days 3-12)

#### 2.1 LQ-MFG Environment (Linear-Quadratic) ⭐ NEW
**File**: `mfg_pde/alg/reinforcement/environments/lq_mfg_env.py`

**Purpose**: Simple validation environment to test base class and algorithms

**State Space** (dim=2):
- Position: `x` ∈ [-10, 10]
- Velocity: `v` ∈ [-5, 5]

**Action Space** (dim=1):
- Control: `u` ∈ [-2, 2]

**Dynamics**:
```python
x_{t+1} = x_t + v_t * dt
v_{t+1} = v_t + u_t * dt + σ * noise
```

**Reward**:
```python
r = -x² - 0.1*u² - 0.5 * mean_field_term
```

**Mean Field Coupling**:
```python
mean_field_term = ∫(x - y)² * m(y) dy  # Quadratic repulsion
```

**Why Start Here**:
- Simplest possible MFG environment
- Validates base class API works correctly
- Fast training for algorithm sanity checks
- Known analytical solution for validation
- Template for more complex environments

**Estimated**: 4-6 hours

---

#### 2.2 Crowd Navigation Environment
**File**: `mfg_pde/alg/reinforcement/environments/crowd_navigation_env.py`

**State Space** (dim=6):
- Position: `(x, y)` ∈ [0, L]²
- Velocity: `(vₓ, vᵧ)`
- Goal: `(x_goal, y_goal)`

**Action Space** (dim=2):
- Acceleration: `(aₓ, aᵧ)` ∈ [-a_max, a_max]²

**Dynamics**:
```python
x_{t+1} = x_t + v_t * dt
v_{t+1} = v_t + a_t * dt + σ * noise
```

**Reward**:
```python
r = -c_dist * ||x - x_goal||²
    - c_velocity * ||v||²
    - c_crowd * mean_field_term
    - c_collision * I(collision)
```

**Mean Field Coupling**:
- Repulsive potential from nearby agents
- Based on kernel: `K(x,y) = exp(-||x-y||²/2σ²)`

**Estimated**: 8-10 hours

---

#### 2.2 Price Formation Environment
**File**: `mfg_pde/alg/reinforcement/environments/price_formation_env.py`

**State Space** (dim=4):
- Inventory: `q` ∈ [-Q_max, Q_max]
- Mid-price: `p`
- Price velocity: `dp/dt`
- Market depth: `m(p, q)` (aggregated)

**Action Space** (dim=2):
- Bid spread: `δ_bid` ∈ [0, δ_max]
- Ask spread: `δ_ask` ∈ [0, δ_max]

**Dynamics**:
```python
p_{t+1} = p_t + σ * dW + λ * order_imbalance
q_{t+1} = q_t + orders_filled(δ_bid, δ_ask)
```

**Reward**:
```python
r = pnl - γ * q² - κ * inventory_risk - spread_cost
```

**Mean Field Coupling**:
- Price impact from aggregate order flow
- Liquidity depletion at popular price levels

**Estimated**: 10-12 hours

---

#### 2.3 Resource Allocation Environment
**File**: `mfg_pde/alg/reinforcement/environments/resource_allocation_env.py`

**State Space** (dim=2n):
- Current allocation: `w = (w₁, ..., wₙ)` with `∑wᵢ = 1`
- Asset values: `v = (v₁, ..., vₙ)`

**Action Space** (dim=n):
- Allocation changes: `Δw` ∈ [-Δ_max, Δ_max]ⁿ with `∑Δwᵢ = 0`

**Dynamics**:
```python
v_{t+1} = v_t * (1 + μ + σ * dW)
w_{t+1} = project_simplex(w_t + Δw_t)
```

**Reward**:
```python
r = w^T * returns
    - λ * w^T * Σ * w
    - congestion_cost(m)
```

**Mean Field Coupling**:
- Congestion in popular assets (reduced returns)
- Market impact proportional to population concentration

**Estimated**: 8-10 hours

---

#### 2.4 Traffic Flow Environment
**File**: `mfg_pde/alg/reinforcement/environments/traffic_flow_env.py`

**State Space** (dim=5):
- Position on network: `(x, y)`
- Velocity: `v` ∈ [0, v_max]
- Destination: `(x_dest, y_dest)`

**Action Space** (dim=2):
- Acceleration: `a` ∈ [-a_max, a_max]
- Lane position: `lane` ∈ [0, num_lanes-1] (continuous)

**Dynamics**:
```python
v_{t+1} = clip(v_t + a_t * dt, 0, v_max * (1 - density_factor))
x_{t+1} = x_t + v_t * dt * direction(lane)
```

**Reward**:
```python
r = -travel_time
    - fuel_cost(a)
    - accident_risk(v, density)
```

**Mean Field Coupling**:
- Speed reduction based on local traffic density
- Congestion on popular routes

**Estimated**: 8-10 hours

---

### Phase 3: Demonstrations (Days 11-13)

Create working demonstrations for each environment:

**Files**: `examples/advanced/{env_name}_demo.py`

**Each Demo Should**:
1. Train all three algorithms (DDPG, TD3, SAC)
2. Use consistent hyperparameters across algorithms
3. Track training progress (rewards, losses)
4. Visualize learned policies
5. Show population evolution over episodes
6. Compare final performance with statistics
7. Generate publication-quality plots

**Template Structure**:
```python
# 1. Environment setup
env = CrowdNavigationEnv(num_agents=100, ...)

# 2. Algorithm initialization
ddpg = MeanFieldDDPG(env, ...)
td3 = MeanFieldTD3(env, ...)
sac = MeanFieldSAC(env, ...)

# 3. Training
for algo_name, algo in [("DDPG", ddpg), ("TD3", td3), ("SAC", sac)]:
    stats = algo.train(num_episodes=1000)
    # Track and visualize

# 4. Comparison
# Statistical analysis, plots, Nash equilibrium analysis
```

**Estimated**: 12-16 hours (4 environments × 3-4 hours each)

---

### Phase 4: Testing (Days 14-16)

#### 4.1 Unit Tests
**Files**: `tests/unit/test_{env_name}_env.py`

**Test Coverage Per Environment** (10+ tests each):
- Environment initialization
- Reset functionality with/without seed
- Step function returns (state, reward, done, truncated, info)
- Action bounds enforcement
- Population state computation
- Mean field coupling calculation
- Episode termination conditions
- Edge cases (boundary states, max steps)

**Base Class Tests**:
- Gymnasium API compliance
- Abstract method enforcement
- Common functionality

**Estimated**: 12-16 hours

---

#### 4.2 Integration Tests
**File**: `tests/integration/test_continuous_environments_training.py`

**Tests**:
- Train each algorithm on each environment (12 combinations)
- Verify convergence (reward improvement)
- Check population state consistency
- Validate Nash equilibrium properties
- Memory usage within bounds
- Reproducibility with seeds

**Estimated**: 6-8 hours

---

### Phase 5: Benchmarking (Days 17-19)

#### 5.1 Comprehensive Benchmark
**File**: `benchmarks/continuous_environments_benchmark.py`

**Metrics**:
- Final performance: mean ± std over 10 seeds
- Sample efficiency: reward vs timesteps
- Robustness: variance across seeds
- Computational cost: wall-clock time, memory
- Convergence speed: episodes to 90% of final performance

**Comparison Dimensions**:
- Algorithms: DDPG vs TD3 vs SAC
- Environments: 4 environments
- Seeds: 10 independent runs each

**Output**:
- Performance tables (LaTeX-ready)
- Learning curves (all algorithms per environment)
- Statistical significance tests
- Computational cost analysis

**Estimated**: 10-14 hours

---

### Phase 6: Documentation (Days 20-21)

#### 6.1 User Guide
**File**: `docs/user/continuous_environments_guide.md`

**Content**:
- Overview of each environment
- When to use which environment
- Hyperparameter recommendations
- How to create custom environments
- Troubleshooting common issues

**Estimated**: 4-6 hours

---

#### 6.2 Theory Documentation
**File**: `docs/theory/reinforcement_learning/continuous_environment_formulations.md`

**Content Per Environment**:
- Mathematical formulation
- State/action space specifications
- Dynamics and transition kernel
- Reward structure derivation
- Mean field coupling formulation
- Nash equilibrium characterization
- References to literature

**Estimated**: 6-8 hours

---

## 📊 Implementation Timeline

| Days | Phase | Deliverables | Hours |
|------|-------|--------------|-------|
| 1-2 | Foundation | Base class, tests | 6-8 |
| 3-4 | LQ-MFG (Validation) | Env + tests | 4-6 |
| 5-7 | Crowd Navigation | Env + tests | 10-12 |
| 8-10 | Price Formation | Env + tests | 12-14 |
| 11-13 | Resource Allocation | Env + tests | 10-12 |
| 14-15 | Traffic Flow | Env + tests | 10-12 |
| 16-19 | Demonstrations | 5 demos | 15-20 |
| 20-23 | Testing | Unit + integration | 20-26 |
| 24-27 | Benchmarking | Comprehensive | 12-16 |
| 28-31 | Documentation | User + theory | 12-16 |

**Total Estimated Hours**: 111-142 hours (~3-4 weeks of full-time work)

---

## 🎯 Code Quality Standards

### Type Hints
- All functions with complete type annotations
- Use `numpy.typing.NDArray` for arrays
- Gymnasium types for env interfaces

### Documentation
- Comprehensive docstrings with mathematical notation
- LaTeX expressions for equations
- Examples in docstrings

### Testing
- Minimum 90% code coverage
- Both unit and integration tests
- Edge case coverage
- Performance regression tests

### Pre-commit Hooks
- All ruff checks passing
- Formatting consistent
- No trailing whitespace
- Yaml validity

---

## 🔗 Dependencies

### Required
- Phase 3.3 (Continuous Actions - Mean Field DDPG/TD3/SAC) ✅
- PyTorch
- Gymnasium
- NumPy, SciPy
- Matplotlib, Plotly

### Optional
- JAX (for accelerated dynamics)
- tqdm (progress bars)
- pandas (benchmark analysis)

---

## 📈 Success Metrics

### Completeness
- ✅ All 5 environments implemented (LQ-MFG + 4 application domains)
- ✅ All environments pass Gymnasium checks
- ✅ 60+ tests with 100% pass rate
- ✅ Working demos for each environment
- ✅ Benchmark results documented

### Quality
- ✅ 90%+ test coverage
- ✅ Type checking passes
- ✅ Documentation complete
- ✅ Pre-commit hooks pass

### Performance
- ✅ Training converges on all environments
- ✅ Algorithms achieve reasonable performance
- ✅ Benchmark completes in reasonable time (optimized with parallel execution)

---

## 🚀 Next Actions

1. **Start**: Implement base environment class
2. **Priority Order**:
   - Base class (foundation for all)
   - LQ-MFG (simple validation environment)
   - Crowd Navigation (2D navigation)
   - Traffic Flow (builds on crowd)
   - Resource Allocation (constrained optimization)
   - Price Formation (most complex)

3. **Parallel Work Opportunities**:
   - Environments can be developed independently after base class
   - Tests can be written alongside implementation
   - Documentation can start early with mathematical formulations

4. **Implementation Notes**:
   - LQ-MFG serves as "hello world" to validate base class API
   - Consider adding rendering support for visualizations
   - Benchmark optimization via parallel execution and reduced seeds if needed

---

**Document Version**: 1.1
**Created**: October 3, 2025
**Updated**: October 3, 2025 (Added LQ-MFG validation environment, adjusted timeline to 3-4 weeks)
**Status**: Planning Complete - Ready to Start Implementation
