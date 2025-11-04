# Detailed Test Exclusion List: File-by-File Breakdown

Generated: 2025-11-04  
Purpose: Specific implementation guide for marking tests for CI reduction

---

## TIER 1: PyTorch-Dependent Tests (120 tests) - `@pytest.mark.optional_torch`

### Unit Tests (15 files)

1. **tests/unit/test_mean_field_sac.py**
   - Lines: ~400
   - Tests: ~15 (parametrized with configurations)
   - Marker: Apply to entire file or class
   ```python
   @pytest.mark.optional_torch
   class TestMeanFieldSAC:
   ```

2. **tests/unit/test_mean_field_ddpg.py**
   - Lines: ~350
   - Tests: ~12
   - Marker: Apply to module-level decorator or class

3. **tests/unit/test_mean_field_td3.py**
   - Lines: ~350
   - Tests: ~12
   - Marker: Module-level

4. **tests/unit/test_multi_population_config.py**
   - Lines: ~400
   - Tests: ~10 (torch-specific tests)
   - Note: Contains both torch and non-torch tests
   - Action: Apply `@pytest.mark.optional_torch` to individual tests using TORCH_AVAILABLE checks

5. **tests/unit/test_multi_population_sac.py**
   - Lines: ~400
   - Tests: ~12
   - Marker: Module-level or class

6. **tests/unit/test_multi_population_ddpg.py**
   - Lines: ~350
   - Tests: ~10
   - Marker: Module-level

7. **tests/unit/test_multi_population_td3.py**
   - Lines: ~400
   - Tests: ~10
   - Marker: Module-level

8. **tests/unit/test_multi_population_algorithms_basic.py**
   - Lines: ~300
   - Tests: ~8
   - Marker: Module-level

9. **tests/unit/test_multi_population_environment.py**
   - Lines: ~300
   - Tests: ~8
   - Marker: Module-level

10. **tests/unit/test_multi_population_networks.py**
    - Lines: ~300
    - Tests: ~6
    - Marker: Module-level

11. **tests/unit/test_density_estimation.py**
    - Lines: ~250
    - Tests: ~8 (4 torch-specific)
    - Note: Mixed torch/scipy tests
    - Action: Apply to individual tests with torch imports

12. **tests/unit/test_backends/test_backend_factory.py**
    - Lines: ~200
    - Tests: ~5 (torch backend tests)
    - Note: Contains numpy backend tests (keep)
    - Action: Apply selectively to torch-related tests

13. **tests/unit/test_alg/test_neural/test_core/test_training.py**
    - Lines: ~350
    - Tests: ~10
    - Marker: Module-level (entire neural training suite)

14. **tests/unit/test_alg/test_numerical/test_fp_solvers/test_base_fp.py**
    - Lines: ~200
    - Tests: ~6 (torch solver tests)
    - Action: Apply selectively

15. **tests/unit/test_config/test_structured_schemas.py**
    - Lines: ~200
    - Tests: ~5 (torch config tests)
    - Action: Apply selectively

### Integration Tests (4 files)

16. **tests/integration/test_particle_gpu_pipeline.py**
    - Lines: ~150
    - Tests: ~4
    - Marker: Module-level
    - Note: Also tagged `@pytest.mark.gpu`

17. **tests/integration/test_torch_kde.py**
    - Lines: ~200
    - Tests: ~8
    - Marker: Module-level
    - Note: Also optional feature

18. **tests/integration/test_mps_scaling.py**
    - Lines: ~250
    - Tests: ~6
    - Marker: Module-level + `@pytest.mark.benchmark`

19. **tests/integration/test_cross_backend_consistency.py**
    - Lines: ~300
    - Tests: ~4 (torch-specific tests)
    - Note: Contains numpy and scipy tests (keep)
    - Action: Apply selectively to torch consistency tests

### Root-Level Tests (2 files)

20. **tests/test_dgm_foundation.py**
    - Lines: ~400
    - Tests: ~8
    - Marker: Module-level
    - Note: Deep Galerkin Method uses torch

21. **tests/test_neural_operators.py**
    - Lines: ~500
    - Tests: ~15
    - Marker: Module-level
    - Note: Neural operators implemented in torch

---

## TIER 2: Benchmark Tests (48 tests) - `@pytest.mark.benchmark`

### Unit Tests (8 subdirectory files)

1. **tests/unit/test_benchmarks/** (entire directory)
   - test_performance_tracker.py (~15 tests)
   - test_solver_performance.py (~15 tests)
   - test_backend_performance.py (~12 tests)
   - test_memory_management.py (~6 tests, partial)
   - Marker: Module-level for all files

2. **tests/unit/geometry/test_geometry_benchmarks.py**
   - Lines: ~350
   - Tests: ~25 (parametrized scaling tests)
   - Marker: Apply to class or individual parametrized tests
   ```python
   @pytest.mark.parametrize("dimension", [2, 5, 10])
   @pytest.mark.benchmark
   def test_hyperrectangle_sampling(dimension):
       ...
   ```

### Integration Tests (1 file)

3. **tests/integration/test_mps_scaling.py**
    - Lines: ~250
    - Tests: ~6
    - Marker: Module-level + `@pytest.mark.optional_torch`
    - Note: Dual-marked for torch and benchmark

---

## TIER 3: Experimental/Research Tests (55 tests) - `@pytest.mark.experimental`

### Unit Tests (5 files)

1. **tests/unit/test_cellular_automata.py**
   - Lines: ~350
   - Tests: ~12
   - Marker: Class-level `@pytest.mark.experimental`
   - Status: Research implementation, API unstable

2. **tests/unit/test_mcmc.py**
   - Lines: ~500
   - Tests: ~20
   - Marker: Module-level
   - Status: MCMC sampling (research feature)

3. **tests/unit/test_monte_carlo.py**
   - Lines: ~450
   - Tests: ~15
   - Marker: Module-level
   - Status: Monte Carlo integration (experimental)

4. **tests/unit/test_functional_calculus.py**
   - Lines: ~300
   - Tests: ~8
   - Marker: Module-level
   - Status: Advanced math operations

### Root-Level Tests (1 file)

5. **tests/test_anderson_multidim.py**
   - Lines: ~150
   - Tests: ~8
   - Marker: Module-level
   - Status: Anderson acceleration (advanced)

### Note on Experimental
- Keep these tests running locally
- Exclude from standard CI to reduce noise
- Run separately for research validation

---

## TIER 4: Environment-Specific Tests (60 tests) - `@pytest.mark.environment`

### Unit Tests (10 files - Problem Environments)

1. **tests/unit/test_lq_mfg_env.py** (~10 tests)
   - Marker: Module-level

2. **tests/unit/test_crowd_navigation_env.py** (~12 tests)
   - Marker: Module-level

3. **tests/unit/test_traffic_flow_env.py** (~10 tests)
   - Marker: Module-level

4. **tests/unit/test_price_formation_env.py** (~10 tests)
   - Marker: Module-level

5. **tests/unit/test_continuous_mfg_env_base.py** (~14 tests)
   - Marker: Module-level

6. **tests/unit/test_resource_allocation_env.py** (~8 tests)
   - Marker: Module-level

7. **tests/unit/test_mfg_maze_env.py** (~12 tests)
   - Marker: Module-level

### Unit Tests (5 files - Maze/Geometry Generation)

8. **tests/unit/test_maze_generator.py** (~8 tests)
   - Marker: Module-level

9. **tests/unit/test_maze_config.py** (~8 tests)
   - Marker: Module-level

10. **tests/unit/test_maze_postprocessing.py** (~15 tests)
    - Marker: Module-level

11. **tests/unit/test_hybrid_maze.py** (~14 tests)
    - Marker: Module-level

12. **tests/unit/test_voronoi_maze.py** (~12 tests)
    - Marker: Module-level

---

## OPTIONAL TIER 5: Interactive Visualization Tests (35 tests) - `@pytest.mark.visualization`

### Unit Tests (2 files)

1. **tests/unit/test_visualization/test_mathematical_plots.py**
   - Lines: ~250
   - Tests: ~20
   - Marker: Module-level
   - Note: Uses plotly (optional)

2. **tests/unit/test_visualization/test_mfg_analytics.py**
   - Lines: ~200
   - Tests: ~15
   - Marker: Module-level
   - Note: Interactive visualization

---

## IMPLEMENTATION GUIDE BY FILE

### Format for Adding Markers

**Module-level marker** (most common):
```python
# At top of file after imports
import pytest

pytestmark = pytest.mark.optional_torch

# Rest of file unchanged
```

**Class-level marker**:
```python
@pytest.mark.optional_torch
class TestClassName:
    def test_something(self):
        ...
```

**Individual test marker**:
```python
@pytest.mark.optional_torch
def test_specific_torch_feature():
    ...

def test_regular_test():
    # Runs in all CI configurations
    ...
```

---

## VALIDATION CHECKLIST

After adding all markers, verify:

```bash
# Should show 120 torch tests
pytest --collect-only -m optional_torch -q | tail -1

# Should show 48 benchmark tests
pytest --collect-only -m benchmark -q | tail -1

# Should show 55 experimental tests
pytest --collect-only -m experimental -q | tail -1

# Should show 60 environment tests
pytest --collect-only -m environment -q | tail -1

# Core suite (should be ~3,217)
pytest --collect-only -m "not slow and not optional_torch and not benchmark and not experimental" -q | tail -1
```

---

## PHASED IMPLEMENTATION SCHEDULE

### Phase 1: Critical Path (2 hours)
- [ ] Update pytest.ini with all new markers
- [ ] Mark all PyTorch files (15 files)
- [ ] Mark all benchmark files (8 files)
- [ ] Validate: `pytest -m "not optional_torch" --co | tail -1`

### Phase 2: Extended Coverage (1 hour)
- [ ] Mark experimental tests (5 files)
- [ ] Mark environment tests (10 files)
- [ ] Validate: `pytest -m "not optional_torch and not experimental and not benchmark" --co | tail -1`

### Phase 3: Polish (1 hour)
- [ ] Mark visualization tests (2 files)
- [ ] Mark GPU/gymnasium tests where needed
- [ ] Update CI configuration

### Phase 4: Testing (1-2 hours)
- [ ] Run full suite locally: verify pass
- [ ] Run aggressive suite: verify 3,217 tests
- [ ] Check no markers applied to core tests
- [ ] Verify conftest.py still works

---

## QUICK REFERENCE: Files to Modify

### Core Files (1 file)
- `pytest.ini` - Add marker definitions

### Test Files (38 files)

**PyTorch** (21 files):
- unit/test_mean_field_*.py (3)
- unit/test_multi_population_*.py (6)
- unit/test_*_sac.py, *_ddpg.py, *_td3.py (3)
- unit/test_density_estimation.py (1)
- unit/test_backends/ (1)
- integration/ (4)
- root level (2)

**Benchmark** (8 files):
- unit/test_benchmarks/ (3)
- unit/geometry/test_geometry_benchmarks.py (1)
- integration/ (1)

**Experimental** (5 files):
- unit/test_cellular_automata.py
- unit/test_mcmc.py
- unit/test_monte_carlo.py
- unit/test_functional_calculus.py
- root/test_anderson_multidim.py

**Environment** (10 files):
- unit/test_*_env.py (7)
- unit/test_maze_*.py (5)

**Visualization** (2 files):
- unit/test_visualization/ (2)

---

## Expected Results After Implementation

```
Before markers:
  Total: 3,500 tests
  CI with -m "not slow": 3,443 tests
  Execution time: ~120-150 seconds

After aggressive markers (Option A):
  Total: 3,500 tests
  CI with markers: 3,217 tests (92%)
  Execution time: ~70-90 seconds (40% reduction)
  Missing: PyTorch (120), Benchmark (48), Experimental (55)

After balanced markers (Option B):
  Total: 3,500 tests
  CI with markers: 3,272 tests (94%)
  Execution time: ~100-115 seconds (25% reduction)
  Missing: PyTorch (120), Benchmark (48)
```
