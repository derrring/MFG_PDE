# Grid-Specific vs Universal Properties

**Created**: 2025-11-10
**Context**: User correctly questioned whether `grid_shape` and `grid_spacing` should be universal MFGProblem properties

---

## The Question

Should `grid_shape` and `grid_spacing` be properties of `MFGProblem`, or should they remain geometry-specific?

**User's insight**: "grid_shape and grid_spacing properties seems for grid uniquely, not for all xSpace"

**Answer**: User is correct! They should remain **geometry-specific**.

---

## Universal vs Grid-Specific Concepts

### ✅ Universal Concepts (All Geometries)

| Property | Meaning | Grid | Mesh | Network |
|:---------|:--------|:-----|:-----|:--------|
| `xSpace` / `get_spatial_grid()` | Spatial point locations | Grid points | Mesh vertices | Node positions |
| `num_spatial_points` | Total number of points | Nx×Ny×Nz | N_vertices | N_nodes |
| `dimension` | Spatial dimension | 1/2/3/nD | 2/3 | Varies |
| `get_bounds()` | Bounding box | [(xmin,xmax), ...] | Mesh bounds | Graph bounds |

### ❌ Grid-Only Concepts (CartesianGrid Only)

| Property | Meaning | Grid | Mesh | Network |
|:---------|:--------|:-----|:-----|:--------|
| `get_grid_shape()` | Shape tuple (Nx, Ny, Nz) | ✅ (50, 50) | ❌ N/A | ❌ N/A |
| `get_grid_spacing()` | Uniform spacing [dx, dy, dz] | ✅ [0.02, 0.02] | ❌ Varies | ❌ N/A |

**Why meshes don't have these**:
- **No uniform shape**: Mesh has N_vertices, but no (Nx × Ny) structure
- **No uniform spacing**: Triangle sizes vary throughout mesh

---

## Current Architecture (Correct!)

### Geometry Hierarchy

```python
class Geometry(ABC):
    """Base class for ALL geometry types."""

    @abstractmethod
    def get_spatial_grid(self) -> NDArray:
        """Universal - all geometries have spatial points."""
        ...

    def get_grid_spacing(self) -> list[float] | None:
        """Returns None for non-grid geometries (meshes, networks)."""
        return None

    def get_grid_shape(self) -> tuple[int, ...] | None:
        """Returns None for non-grid geometries (meshes, networks)."""
        return None


class CartesianGrid(Geometry):
    """Abstract base for structured grids only."""

    @abstractmethod
    def get_grid_spacing(self) -> list[float]:  # NOT Optional!
        """Guaranteed to return [dx, dy, dz]."""
        ...

    @abstractmethod
    def get_grid_shape(self) -> tuple[int, ...]:  # NOT Optional!
        """Guaranteed to return (Nx, Ny, Nz)."""
        ...


class UnstructuredMesh(Geometry):
    """Unstructured mesh - NO grid_shape or grid_spacing!"""

    def get_spatial_grid(self) -> NDArray:
        """Returns mesh vertices (scattered points)."""
        return self.vertices  # (N_vertices, dimension)

    # Inherits get_grid_spacing() → None
    # Inherits get_grid_shape() → None
```

**Type-safe usage**:
```python
# Solver checks geometry type before accessing grid properties
if isinstance(geometry, CartesianGrid):
    dx = geometry.get_grid_spacing()  # Guaranteed non-None
    shape = geometry.get_grid_shape()  # Guaranteed non-None
else:
    # Use mesh-specific properties instead
    num_verts = geometry.num_spatial_points
```

---

## Should MFGProblem Expose These?

### ❌ Bad Idea: Add as General Properties

```python
class MFGProblem:
    @property
    def grid_shape(self):
        """❌ BAD: Not all problems have grids!"""
        if self.geometry is not None:
            shape = self.geometry.get_grid_shape()
            if shape is not None:
                return shape
        # What to return here?
        # - Error? Breaks non-grid problems
        # - None? Caller has to check anyway
        # - Compute from legacy Nx? Fragile
        raise AttributeError("Problem does not have grid structure")
```

**Problems**:
1. **Not all MFGProblems have grids** (meshes, networks exist!)
2. **Error handling complexity**: Must check geometry type
3. **Violates type safety**: Property exists but may fail at runtime
4. **Misleading API**: Suggests all problems have grids

### ✅ Good Design: Access via Geometry

```python
# Clear, type-safe access
problem = MFGProblem(geometry=grid, T=1.0, Nt=50)

# Check geometry type first
if isinstance(problem.geometry, CartesianGrid):
    # Type system guarantees these exist
    shape = problem.geometry.get_grid_shape()    # (50, 50)
    spacing = problem.geometry.get_grid_spacing()  # [0.02, 0.02]

    # Use grid-specific solver
    solver = HJBFDMSolver(problem)
else:
    # Use mesh/network-specific properties and solvers
    num_verts = problem.geometry.num_spatial_points
    solver = FEMSolver(problem)
```

**Benefits**:
1. **Type-safe**: Properties only exist where they make sense
2. **Explicit checking**: Caller knows when grids vs meshes
3. **Clean separation**: Geometry handles geometry-specific concerns
4. **No ambiguity**: Mesh users never see misleading grid properties

---

## What About Legacy 1D Mode?

**Current state**:
```python
# Legacy 1D mode (no geometry object)
problem = MFGProblem(Nx=100, xmin=0.0, xmax=1.0, T=1.0, Nt=50)

# How to get grid properties?
problem.Nx  # ✅ Exists (scalar)
problem.Dx  # ✅ Exists (scalar)
problem.xSpace  # ✅ Exists (array)
```

**Problem**: Legacy mode has `Nx`, `Dx` as scalars (not arrays!)

**Solution**: This is Issue #243 (array notation migration) - separate concern!

---

## Revised Protocol Compliance Plan

### What Needs Fixing: Capitalization Only

**GridProblem Protocol** (for grid-based problems):
```python
@runtime_checkable
class GridProblem(Protocol):
    """Protocol for GRID-BASED problems only."""

    # ⚠️ Fix capitalization
    dt: float       # Currently Dt (should be lowercase)
    dx: float       # Currently Dx (should be lowercase - 1D only!)

    # ✅ Already correct
    xSpace: NDArray
    tSpace: NDArray
    Nx: int         # For 1D mode only
```

**Recommendation**: Just fix capitalization (`Dt` → `dt`, `Dx` → `dx`)

### What Should NOT Be Added to MFGProblem

- ❌ `grid_shape` property (geometry-specific, not universal)
- ❌ `grid_spacing` property (geometry-specific, not universal)

**Access these via geometry instead**:
```python
if isinstance(problem.geometry, CartesianGrid):
    shape = problem.geometry.get_grid_shape()
    spacing = problem.geometry.get_grid_spacing()
```

---

## Summary

**User is correct!**

1. **`xSpace`** - Universal (all geometries)
2. **`grid_shape` / `grid_spacing`** - Grid-only (CartesianGrid subclasses)
3. **Don't add to MFGProblem** - Keep geometry-specific properties on geometry objects

**What we should actually do**:
- ✅ Fix capitalization: `Dt` → `dt`, `Dx` → `dx`
- ❌ Don't add `grid_shape` or `grid_spacing` to MFGProblem
- ✅ Document how to access grid properties via geometry

**Architecture is already correct** - no need for new properties!
