# AMR Documentation Archive

**Status**: ARCHIVED (December 2025)

**Reason**: AMR implementation was removed from MFG_PDE. The custom AMR code (~3100 lines) was replaced with a minimal API stub that will wrap external libraries when AMR is needed.

## Recommended External Libraries

For adaptive mesh refinement, use established external libraries:

- **pyAMReX**: Block-structured AMR, GPU support (https://github.com/AMReX-Codes/pyamrex)
- **Clawpack/AMRClaw**: Hyperbolic PDEs, Berger-Oliger-Colella AMR
- **pyAMG**: Mesh adaptation for complex 2D/3D geometries (Inria)
- **p4est**: Scalable octree AMR

## Current API

The AMR module now provides only:
- `AdaptiveGeometry` protocol (defines interface)
- `is_adaptive()` helper function
- `create_amr_grid()` stub (raises `AMRNotImplementedError`)

## Archived Files

- `amr_for_mfg.md` - Theory of AMR for MFG problems
- `amr_mesh_types_analysis.md` - Analysis of mesh element types
- `amr_quick_reference.md` - User guide for AMR usage
- `tutorial_amr.md` - AMR tutorial
- `v0.10.1_amr_geometry_protocol.md` - Historical protocol design

These documents describe the removed implementation. They are preserved for historical reference but are no longer applicable to the current codebase.
