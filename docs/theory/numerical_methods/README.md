# Numerical Methods for Mean Field Games

Theoretical foundations and mathematical formulations of numerical methods used in MFG_PDE.

## 📚 Contents

- **[adaptive_mesh_refinement.md](adaptive_mesh_refinement.md)** - AMR system design and theoretical framework
- **[semi_lagrangian_methods.md](semi_lagrangian_methods.md)** - Semi-Lagrangian solver implementation and theory
- **[lagrangian_formulation.md](lagrangian_formulation.md)** - Lagrangian formulation of MFG systems

## 🎯 Overview

This directory contains mathematical foundations and theoretical analyses for numerical methods implemented in MFG_PDE, including:

1. **Adaptive Methods**: Mesh refinement for spatially varying solutions
2. **Lagrangian Approaches**: Particle-based and trajectory-following methods
3. **Advanced Discretizations**: High-order and specialized numerical schemes

## 📖 Document Summaries

### adaptive_mesh_refinement.md
Covers:
- AMR theoretical framework for MFG
- Error estimation and refinement criteria
- Mesh hierarchy and grid management
- Performance considerations

### semi_lagrangian_methods.md
Includes:
- Semi-Lagrangian discretization theory
- Characteristic tracing for HJB equations
- Interpolation schemes
- Implementation strategies

### lagrangian_formulation.md
Describes:
- Lagrangian formulation of MFG
- Particle methods
- Trajectory-based approaches
- Connection to Eulerian formulation

## 🔗 Related Documentation

- **Implementation**: See [/docs/development/](../../development/) for implementation details
- **User Guide**: See [/docs/user/](../../user/) for practical usage
- **Mathematical Background**: [/docs/theory/mathematical_background.md](../mathematical_background.md)

---

**Target Audience**: Numerical analysts, algorithm developers, computational scientists
**Prerequisites**: Numerical PDEs, finite difference methods, MFG theory
