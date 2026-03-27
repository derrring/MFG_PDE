"""
Finite Element Method (FEM) solvers for MFG systems.

Uses scikit-fem (skfem) as the assembly backend for stiffness, mass, and
advection matrices on unstructured meshes. The coupling layer, boundary
condition handling, and MFG-specific logic remain in-house.

Requires: pip install scikit-fem

Issue #773: scikit-fem integration
"""
