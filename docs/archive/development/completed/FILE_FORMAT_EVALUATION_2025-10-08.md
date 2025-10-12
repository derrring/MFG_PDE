# File Format Analysis for MFG_PDE: Parquet vs HDF5 vs Zarr

**Date**: 2025-10-08
**Purpose**: Determine optimal file format(s) for MFG solver data persistence

---

## Executive Summary

**Recommendation**: Use **multiple formats** for different use cases:
1. **HDF5** - Primary format for solver state/results (already partially integrated)
2. **Parquet** - Secondary for tabular analysis data (already integrated via Polars)
3. **Zarr** - Optional for large-scale array data requiring cloud storage

**Rationale**: MFG_PDE has diverse data needs that no single format optimally serves.

---

## Current Integration Status

### ✅ Parquet (via Polars)
**Files**: `mfg_pde/utils/data/polars_integration.py`
- Integrated for data analysis and parameter sweeps
- Used for tabular data (convergence logs, parameter studies)
- Polars provides: `read_parquet()`, `write_parquet()`

### ✅ HDF5 (Partial)
**Files**: `mfg_pde/solvers/fixed_point.py:338`
- Basic export capability exists
- Used for solution snapshots
- h5py integration (optional dependency)

### ❌ Zarr (Not integrated)
- No current usage
- Would require new integration

---

## Data Types in MFG_PDE

### 1. **Solver Solutions** (Primary use case)
**Data**: `U(t,x)`, `M(t,x)` arrays
- **Shape**: (Nt+1, Nx+1) typically 50×50 to 500×500
- **Type**: Float64 arrays (2D spatio-temporal)
- **Access Pattern**: Read entire solution for visualization/analysis
- **Size**: 10KB - 10MB per solution

### 2. **Convergence Metadata** (Analysis use case)
**Data**: Iteration logs, residuals, timing
- **Shape**: Tables with columns (iteration, residual, time, etc.)
- **Type**: Mixed (int, float, timestamps)
- **Access Pattern**: Query/filter for analysis
- **Size**: Small (1KB - 1MB)

### 3. **Parameter Sweep Results** (Batch analysis)
**Data**: Many solutions with parameter combinations
- **Shape**: Collections of (params, solution, metrics)
- **Type**: Mixed structured data
- **Access Pattern**: Query by parameter ranges
- **Size**: 100MB - 10GB

### 4. **Checkpoint State** (Resume capability)
**Data**: Complete solver state for resuming
- **Shape**: Multiple arrays + metadata
- **Type**: NumPy arrays + dictionaries
- **Access Pattern**: Full state save/restore
- **Size**: Similar to solution

---

## Format Comparison Matrix

| Feature | HDF5 | Parquet | Zarr |
|---------|------|---------|------|
| **NumPy Integration** | ✅ Excellent | ❌ Poor | ✅ Excellent |
| **Multidimensional Arrays** | ✅ Native | ❌ Limited | ✅ Native |
| **Metadata Support** | ✅ Rich | ⚠️ Limited | ✅ Good |
| **Compression** | ✅ Good | ✅ Excellent | ✅ Good |
| **Query Performance** | ⚠️ Moderate | ✅ Excellent | ⚠️ Moderate |
| **Cloud Storage** | ❌ Poor | ✅ Good | ✅ Excellent |
| **Maturity** | ✅ Very Mature | ✅ Mature | ⚠️ Newer |
| **Scientific Community** | ✅ Standard | ⚠️ Emerging | ⚠️ Emerging |
| **File Size** | ⚠️ Larger | ✅ Smaller | ⚠️ Moderate |
| **Read Speed** | ✅ Fast | ✅ Fast | ✅ Fast |
| **Write Speed** | ✅ Fast | ⚠️ Moderate | ✅ Fast |
| **Partial I/O** | ✅ Yes | ❌ No | ✅ Yes |
| **Python Support** | ✅ h5py | ✅ polars/pyarrow | ✅ zarr |

---

## Detailed Format Analysis

### HDF5 (Hierarchical Data Format 5)

**Best For**: Solver solutions and checkpoints

**Pros**:
- ✅ **Perfect for NumPy arrays** - direct mapping to/from ndarray
- ✅ **Hierarchical structure** - can store U, M, metadata in one file
- ✅ **Scientific standard** - widely used in computational science
- ✅ **Partial I/O** - read subsets without loading full array
- ✅ **Rich metadata** - attach arbitrary attributes
- ✅ **Mature ecosystem** - h5py, PyTables, HDF View

**Cons**:
- ❌ **Cloud storage** - poor performance on S3/GCS (file locking)
- ❌ **Concurrent writes** - difficult (needs special setup)
- ❌ **Query capability** - not designed for SQL-like queries
- ⚠️ **File size** - can be larger than compressed alternatives

**Use Cases in MFG_PDE**:
```python
# Save solver results
with h5py.File('solution.h5', 'w') as f:
    f.create_dataset('U', data=U, compression='gzip')
    f.create_dataset('M', data=M, compression='gzip')
    f.attrs['converged'] = True
    f.attrs['iterations'] = 47
    f.attrs['timestamp'] = time.time()

# Load for analysis
with h5py.File('solution.h5', 'r') as f:
    U = f['U'][:]
    M = f['M'][:]
    metadata = dict(f.attrs)
```

**Verdict**: ✅ **Ideal for solver solutions**

---

### Parquet (Columnar Storage)

**Best For**: Convergence logs, parameter sweep analysis

**Pros**:
- ✅ **Excellent compression** - columnar format highly efficient
- ✅ **Query performance** - optimized for analytics (Polars/DuckDB)
- ✅ **Cloud storage** - works well on S3/GCS
- ✅ **Schema evolution** - can add columns without rewriting
- ✅ **Industry standard** - used in big data (Spark, Pandas, Polars)
- ✅ **File size** - typically smallest format

**Cons**:
- ❌ **Array storage** - not designed for multidimensional arrays
- ❌ **NumPy integration** - requires conversion via Arrow
- ⚠️ **Metadata** - limited compared to HDF5
- ⚠️ **Write overhead** - columnar encoding has cost

**Use Cases in MFG_PDE**:
```python
# Save convergence history
import polars as pl

convergence_data = pl.DataFrame({
    'iteration': range(iterations),
    'residual_u': residuals_u,
    'residual_m': residuals_m,
    'time_elapsed': times,
    'solver': ['FixedPoint'] * iterations
})
convergence_data.write_parquet('convergence.parquet')

# Query analysis
df = pl.read_parquet('parameter_sweep/*.parquet')
results = df.filter(pl.col('sigma') > 0.1).group_by('method').agg([
    pl.col('solve_time').mean(),
    pl.col('iterations').median()
])
```

**Verdict**: ✅ **Ideal for tabular analysis data**

---

### Zarr (Chunked Array Storage)

**Best For**: Large-scale distributed computations (future use)

**Pros**:
- ✅ **Cloud-native** - designed for S3/GCS from ground up
- ✅ **Parallel I/O** - concurrent reads/writes
- ✅ **Chunking** - flexible chunk sizes for different access patterns
- ✅ **NumPy-like API** - feels like working with arrays
- ✅ **Compression** - per-chunk compression (flexible)
- ✅ **Growing ecosystem** - Xarray, Dask integration

**Cons**:
- ⚠️ **Newer** - less mature than HDF5 (but stable)
- ⚠️ **Fragmented storage** - many files for one array
- ❌ **Local filesystem** - more overhead than HDF5
- ⚠️ **Learning curve** - chunk size tuning important

**Use Cases in MFG_PDE**:
```python
import zarr

# Save large parameter sweep (distributed)
store = zarr.DirectoryStore('sweep_results.zarr')
root = zarr.group(store=store)
root.create_dataset('solutions',
                    shape=(n_params, Nt, Nx),
                    chunks=(1, Nt, Nx),  # One param at a time
                    compression='blosc')

# Parallel writes from different processes
root['solutions'][param_idx, :, :] = solution
```

**Verdict**: ⚠️ **Overkill for current needs, good for future scale**

---

## Recommendations by Use Case

### 1. **Primary Solver Output** → HDF5
```python
# mfg_pde/utils/io/hdf5_export.py
def save_solution(U, M, metadata, filename):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('U', data=U, compression='gzip', compression_opts=4)
        f.create_dataset('M', data=M, compression='gzip', compression_opts=4)
        f.create_dataset('grid/x', data=x_grid)
        f.create_dataset('grid/t', data=t_grid)

        # Metadata as attributes
        for key, value in metadata.items():
            f.attrs[key] = value
```

**Why**: Direct NumPy integration, hierarchical structure, scientific standard

---

### 2. **Convergence Logs & Analytics** → Parquet
```python
# Already integrated via polars_integration.py
def save_convergence_log(convergence_history, filename):
    df = pl.DataFrame(convergence_history)
    df.write_parquet(filename, compression='snappy')
```

**Why**: Excellent compression, fast queries, already integrated

---

### 3. **Parameter Sweep Results** → Hybrid (HDF5 + Parquet)
```python
# Solutions: HDF5 (one file per parameter combo)
save_solution(U, M, metadata, f'results/param_{idx}.h5')

# Summary table: Parquet (for fast querying)
summary = pl.DataFrame({
    'param_id': param_ids,
    'sigma': sigmas,
    'converged': convergence_flags,
    'solve_time': times,
    'final_residual': residuals,
    'filepath': filepaths
})
summary.write_parquet('results/summary.parquet')

# Query: Fast lookup, then load specific solutions
matches = pl.read_parquet('results/summary.parquet').filter(
    (pl.col('sigma') > 0.1) & pl.col('converged')
)
for filepath in matches['filepath']:
    load_solution(filepath)  # Load HDF5 on demand
```

**Why**: Best of both worlds - fast queries + efficient array storage

---

### 4. **Checkpointing** → HDF5
```python
def save_checkpoint(solver, filename):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('U', data=solver.U)
        f.create_dataset('M', data=solver.M)
        f.create_dataset('residuals', data=solver.residuals)
        f.attrs['iteration'] = solver.iteration
        f.attrs['timestamp'] = time.time()
```

**Why**: Full state serialization, easy resume

---

### 5. **Cloud Workflows** (Future) → Zarr
**Only if needed**:
- Large-scale distributed parameter sweeps on cloud
- Collaborative research with shared cloud storage
- Very large 3D problems (high-dimensional MFGs)

---

## Implementation Roadmap

### Phase 1: Enhance HDF5 Support (Priority: High)
**Effort**: 2-3 days

1. Create `mfg_pde/utils/io/hdf5_utils.py`:
   ```python
   def save_solution(U, M, metadata, filename)
   def load_solution(filename)
   def save_checkpoint(solver, filename)
   def load_checkpoint(filename)
   ```

2. Add HDF5 export to SolverResult:
   ```python
   result = solver.solve()
   result.save('solution.h5')  # HDF5 format
   ```

3. Update documentation with HDF5 examples

**Deliverable**: Complete HDF5 integration for all solvers

---

### Phase 2: Standardize Parquet Usage (Priority: Medium)
**Effort**: 1-2 days

1. Document Parquet usage patterns
2. Add examples for parameter sweeps
3. Create utility functions in `polars_integration.py`:
   ```python
   def save_parameter_sweep_summary(results, filename)
   def load_and_query_sweeps(pattern, filters)
   ```

**Deliverable**: Clear patterns for using Parquet

---

### Phase 3: Zarr (Optional, Future)
**Effort**: 1 week
**Trigger**: User request for cloud workflows OR 3D problems become common

1. Add `mfg_pde/utils/io/zarr_utils.py`
2. Cloud storage examples
3. Dask integration for parallel processing

**Deliverable**: Zarr support for advanced use cases

---

## Decision Matrix for Users

| **Your Need** | **Recommended Format** |
|---------------|----------------------|
| Save solver solution (U, M) | HDF5 |
| Log convergence history | Parquet |
| Parameter sweep (100+ runs) | HDF5 (solutions) + Parquet (summary) |
| Checkpoint for resume | HDF5 |
| Share results with collaborators | HDF5 (small), Parquet (summaries) |
| Cloud-based workflows | Zarr (if large-scale) |
| Quick prototyping | NumPy `.npz` (simplest) |

---

## Comparison to NumPy `.npz`

**Current simple option**:
```python
np.savez_compressed('solution.npz', U=U, M=M, **metadata)
```

**Pros**: Simple, no dependencies, works
**Cons**: No query capability, limited metadata, poor for large files

**When to use**: Quick scripts, temporary storage, small problems

---

## Final Recommendation

### For MFG_PDE Package:

**Implement**:
1. ✅ **HDF5 as primary** - solver solutions, checkpoints
2. ✅ **Keep Parquet** - analysis data (already integrated)
3. ❌ **Skip Zarr for now** - not needed until proven necessary

**Dependency Strategy**:
```python
# Required
- numpy

# Optional (feature flags)
- h5py        # For HDF5 export
- polars      # For Parquet analysis (already integrated)
- zarr        # Future, if needed
```

**User Guidance**:
- Default: HDF5 for solutions (add to `solver.solve()`)
- Advanced: Parquet for parameter sweep analysis
- Future: Zarr for cloud workflows

---

## Code Changes Needed

### 1. Add HDF5 utilities (new file)
`mfg_pde/utils/io/hdf5_utils.py` (~200 lines)

### 2. Integrate into SolverResult
```python
class SolverResult:
    def save(self, filename, format='hdf5'):
        if format == 'hdf5':
            save_solution_hdf5(self.u, self.m, self.metadata, filename)
        elif format == 'npz':
            np.savez_compressed(filename, u=self.u, m=self.m, **self.metadata)
```

### 3. Update examples
- Add HDF5 save/load examples
- Show Parquet for parameter sweeps

---

## Conclusion

**Answer**: Use **HDF5 as primary + Parquet for analytics**

**Why**:
- HDF5 perfect for NumPy array storage (solver solutions)
- Parquet excellent for tabular analysis (already integrated)
- Zarr overkill for current research workflows

**Next Steps**:
1. Implement comprehensive HDF5 utilities
2. Document Parquet patterns for parameter sweeps
3. Defer Zarr until demonstrated need

**Total Implementation**: 3-4 days for HDF5 enhancement + documentation
