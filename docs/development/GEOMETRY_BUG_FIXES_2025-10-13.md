# Geometry Bug Fixes - October 13, 2025

## Overview

Fixed all three major geometry bugs documented in Phase 2.3 testing, improving test coverage from 313 passing (15 skipped) to **323 passing (5 skipped)**.

**Branch**: `fix/geometry-bugs`
**Date**: 2025-10-13
**Status**: ✅ All fixes complete and tested

---

## Bug Fixes Summary

### 1. ✅ Domain2D Holes Feature (commit 8accb0b)

**Problem**: Gmsh kernel incompatibility
**Location**: `mfg_pde/geometry/domain_2d.py:279` in `_add_holes_gmsh()`
**Root Cause**: Mixing incompatible Gmsh kernels (built-in `geo` and OpenCASCADE `occ`)

**Solution**:
- Complete rewrite of `_add_holes_gmsh()` method (lines 221-308)
- Recreate main domain entirely in OCC kernel using `occ.addRectangle()` or `occ.addDisk()`
- Create hole entities in OCC kernel
- Use proper boolean operations: `occ.cut(main_entities, hole_entities)`
- Handle physical group conflicts with `contextlib.suppress(Exception)`

**Tests Fixed**: 3 hole tests now passing without warnings
- `test_rectangle_with_circular_hole`
- `test_rectangle_with_rectangular_hole`
- `test_domain_with_multiple_holes`

**Technical Details**:
```python
# Before: Mixed kernels (geo + occ) - INCOMPATIBLE
gmsh.model.geo.addRectangle(...)  # geo kernel
gmsh.model.occ.addDisk(...)       # occ kernel - ERROR!

# After: Pure OCC kernel
gmsh.model.occ.addRectangle(...)  # occ kernel
gmsh.model.occ.addDisk(...)       # occ kernel
result, _result_map = gmsh.model.occ.cut(main_entities, hole_entities)
```

---

### 2. ✅ Domain3D Mesh Generation (commits 8accb0b, 710de9f)

**Problem**: Boundary surface numbering and element mapping errors
**Location**: Multiple issues in `mfg_pde/geometry/domain_3d.py`

#### Issue 2a: Surface Numbering (line 139)

**Root Cause**: Enumerate-based surface assignment with arbitrary ordering from Gmsh

**Solution**: Geometric detection using center-of-mass coordinates (lines 138-184)
```python
# Compute tolerance based on domain size
domain_size = max(self.xmax - self.xmin, self.ymax - self.ymin, self.zmax - self.zmin)
tol = domain_size * 1e-6  # Relative tolerance

for surf_tag in surface_tags:
    mass = gmsh.model.occ.getCenterOfMass(2, surf_tag)
    x, y, z = mass[0], mass[1], mass[2]

    # Identify surface by which coordinate is at boundary
    if abs(x - self.xmin) < tol:
        surface_map[1] = surf_tag  # xmin (left)
    elif abs(x - self.xmax) < tol:
        surface_map[2] = surf_tag  # xmax (right)
    # ... etc for all 6 faces
```

#### Issue 2b: Element Mapping (lines 425, 447)

**Root Cause**: Confusing mesh element tags with geometry entity tags in `getPhysicalGroupsForEntity()`

**Solution**: Create element-to-physical-group mappings (lines 416-466)
```python
# Surface element mapping
surface_element_map = {}
for dim, phys_tag in physical_surfaces:
    entities = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag)
    for entity in entities:
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim, entity)
        for elem_type, tags in zip(elem_types, elem_tags, strict=False):
            if elem_type == 2:  # Triangle
                for tag in tags:
                    surface_element_map[int(tag)] = phys_tag

# Volume element mapping (similar pattern for tetrahedra)
volume_element_map = {}
# ... (analogous logic for 3D elements)
```

#### Issue 2c: Flaky test_mesh_size_effect

**Root Cause**: Insufficient difference in mesh sizes (0.5 vs 0.2) led to identical element counts

**Solution**: Use more distinct mesh sizes and realistic threshold
```python
# Before: Too similar, unrealistic expectation
domain_coarse = Domain3D(bounds=(...), mesh_size=0.5)
domain_fine = Domain3D(bounds=(...), mesh_size=0.2)
assert len(mesh_fine.elements) > len(mesh_coarse.elements)  # Could fail!

# After: More distinct, realistic threshold
domain_coarse = Domain3D(bounds=(...), mesh_size=0.8)
domain_fine = Domain3D(bounds=(...), mesh_size=0.15)
assert len(mesh_fine.elements) > len(mesh_coarse.elements) * 1.2  # 20% margin
```

**Tests Fixed**: 5 Domain3D mesh generation tests
- `test_mesh_data_structure`
- `test_mesh_size_effect`
- `test_mesh_quality_metrics`
- `test_element_volumes`
- `test_boundary_faces_extraction`

**Remaining Skipped**: 2 tests still skipped (unrelated to these fixes)

---

### 3. ✅ Mesh Export KeyError (commit 25ee338)

**Problem**: meshio KeyError when exporting meshes
**Location**: `mfg_pde/geometry/base_geometry.py:137` in `to_meshio()` method
**Error**: `KeyError: 0` when calling `meshio.Mesh(...)`

**Root Cause**: Incorrect `cell_data` format for meshio API

**Solution**: Corrected cell_data format and removed boundary_tags (lines 120-146)
```python
# Before: INCORRECT format - causes KeyError
cell_data = {
    self.element_type: {"region": self.element_tags}  # ❌ WRONG!
}
point_data = {"boundary": self.boundary_tags}  # ❌ Wrong length!

# After: CORRECT format
cell_data = {
    "region": [self.element_tags]  # ✅ List of arrays, one per cell block
}
# boundary_tags removed - they don't map to meshio's data model
```

**Why boundary_tags removed**:
- `boundary_tags` represent boundary elements (edges in 2D, faces in 3D)
- meshio's `point_data` requires arrays with length = number of vertices
- meshio's `cell_data` is for bulk elements, not boundary elements
- No appropriate place in meshio's data model for boundary element tags

**Tests Fixed**: 2 mesh export tests
- `test_export_mesh_vtk`
- `test_export_mesh_requires_generation`

---

## Test Results

### Before Fixes
```
313 passed, 15 skipped
```

### After Fixes
```
323 passed, 5 skipped, 6 warnings
```

**Improvement**: +10 tests passing, -10 tests skipped

### Warnings (Expected)
All warnings are expected and related to features under development:
- Boundary condition validation warnings (incompatibility checks working as designed)
- Named region support warnings (feature not yet fully implemented)

---

## Technical Lessons Learned

### 1. Gmsh Kernel Compatibility
**Lesson**: Gmsh's built-in `geo` kernel and OpenCASCADE `occ` kernel are **incompatible**. Never mix them in the same geometry construction.

**Best Practice**: Choose one kernel at the start and stick with it:
- `geo`: Simpler, faster, but limited features
- `occ`: More powerful (boolean operations, STEP/IGES import), required for complex geometry

### 2. Gmsh API Distinctions
**Lesson**: Gmsh distinguishes between:
- **Geometry entity tags**: Model entities (surfaces, volumes) created during geometry construction
- **Mesh element tags**: Mesh elements (triangles, tetrahedra) created during meshing

**Best Practice**: Never use mesh element tags with geometry API functions like `getPhysicalGroupsForEntity()`. Create explicit mappings when needed.

### 3. Meshio Data Model
**Lesson**: meshio has strict data structure requirements:
- `cells`: List of (type, connectivity) tuples
- `cell_data`: Dict of {data_name: [array_per_block]} format
- `point_data`: Dict of {data_name: array} where len(array) == num_vertices

**Best Practice**: Only export data that fits meshio's data model. Boundary element tags don't fit, so store them separately in custom metadata.

### 4. Test Robustness
**Lesson**: Tests depending on numerical values (like mesh size effects) can be flaky if thresholds are too tight.

**Best Practice**: Use:
- Distinct parameter values (0.8 vs 0.15, not 0.5 vs 0.2)
- Realistic thresholds with margin (1.2x instead of 1.0x)
- Consider stochastic nature of mesh generation

---

## Files Modified

### Source Code
1. **`mfg_pde/geometry/domain_2d.py`** (lines 221-308)
   - Complete rewrite of `_add_holes_gmsh()` method
   - Pure OCC kernel implementation

2. **`mfg_pde/geometry/domain_3d.py`** (multiple locations)
   - Lines 138-184: Geometric surface detection
   - Lines 416-443: Surface element mapping
   - Lines 445-466: Volume element mapping

3. **`mfg_pde/geometry/base_geometry.py`** (lines 120-146)
   - Fixed `to_meshio()` cell_data format
   - Removed boundary_tags from export
   - Added documentation

### Tests
4. **`tests/unit/test_geometry/test_domain_2d.py`**
   - Lines 394, 413, 433: Removed skip markers from hole tests
   - Lines 503, 514: Removed skip markers from mesh export tests

5. **`tests/unit/test_geometry/test_domain_3d.py`**
   - Lines 313-321: Updated mesh size test parameters
   - Multiple lines: Removed skip markers from mesh generation tests

---

## Commits

1. **`8accb0b`**: "fix: Domain2D holes and improved Domain3D surface detection"
   - Fixed Domain2D holes OCC kernel issue
   - Improved Domain3D geometric surface detection
   - Removed skip markers from 8 tests

2. **`710de9f`**: "fix: Domain3D volume element mapping and improve mesh size test"
   - Fixed volume element mapping bug
   - Improved test_mesh_size_effect robustness
   - All Domain3D mesh generation tests passing

3. **`25ee338`**: "fix: MeshData.to_meshio() KeyError - correct cell_data format"
   - Fixed meshio cell_data format
   - Removed boundary_tags from export
   - Mesh export tests now passing

---

## Next Steps

### Immediate
- [x] All Phase 2.3 documented bugs fixed
- [ ] Merge `fix/geometry-bugs` to `main`
- [ ] Update Phase 2.3 documentation with completion status

### Future Improvements
1. **Boundary element export**: Design proper format for exporting boundary element tags
2. **Domain3D sphere mesh**: Investigate remaining 2 skipped tests for sphere meshes
3. **Named regions**: Complete implementation of named region support in boundary conditions
4. **Performance**: Profile mesh generation for large domains

---

## References

- **Phase 2.3 Summary**: `docs/development/PHASE2-3_COMPLETE_SUMMARY.md`
- **Gmsh Documentation**: https://gmsh.info/doc/texinfo/gmsh.html
- **meshio Documentation**: https://github.com/nschloe/meshio

---

**Status**: ✅ **ALL BUGS FIXED** - Ready for merge to main
