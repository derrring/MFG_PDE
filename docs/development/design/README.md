# System Design Documents

Architectural design documents for major features and systems in MFG_PDE.

## 🏗️ Contents

### **Core Systems**
- [**GEOMETRY_SYSTEM_DESIGN.md**](GEOMETRY_SYSTEM_DESIGN.md) - Geometry and domain architecture
- [**ADAPTIVE_MESH_REFINEMENT_DESIGN.md**](ADAPTIVE_MESH_REFINEMENT_DESIGN.md) - AMR system design
- [**API_REDESIGN_PLAN.md**](API_REDESIGN_PLAN.md) - API architecture and evolution
- [**BENCHMARK_DESIGN.md**](BENCHMARK_DESIGN.md) - Benchmarking infrastructure design

### **Advanced Features**
- [**HYBRID_MAZE_GENERATION_DESIGN.md**](HYBRID_MAZE_GENERATION_DESIGN.md) - Hybrid maze algorithm design
- [**LAGRANGIAN_MFG_SYSTEM.md**](LAGRANGIAN_MFG_SYSTEM.md) - Lagrangian formulation design
- [**SEMI_LAGRANGIAN_IMPLEMENTATION.md**](SEMI_LAGRANGIAN_IMPLEMENTATION.md) - Semi-Lagrangian solver design

### **Performance & Architecture**
- [**amr_performance_architecture.md**](amr_performance_architecture.md) - AMR performance considerations
- [**geometry_amr_consistency_analysis.md**](geometry_amr_consistency_analysis.md) - Geometry-AMR integration analysis
- [**continuous_action_architecture_sketch.py**](continuous_action_architecture_sketch.py) - Continuous action RL architecture (code sketch)

## 🎯 Quick Reference

**Looking for:**
- **Overall architecture?** → See [ARCHITECTURAL_CHANGES.md](../ARCHITECTURAL_CHANGES.md) (kept at root)
- **Detailed architecture docs?** → See [architecture/](../architecture/) directory
- **Geometry design?** → [GEOMETRY_SYSTEM_DESIGN.md](GEOMETRY_SYSTEM_DESIGN.md)
- **AMR design?** → [ADAPTIVE_MESH_REFINEMENT_DESIGN.md](ADAPTIVE_MESH_REFINEMENT_DESIGN.md)
- **Continuous action design?** → [continuous_action_architecture_sketch.py](continuous_action_architecture_sketch.py)

## 📋 Design Philosophy

All design documents follow these principles:
1. **Mathematical Rigor**: Precise formulations with proper notation
2. **Implementation Clarity**: Clear path from theory to code
3. **Performance Awareness**: Computational complexity analysis
4. **Extensibility**: Design for future enhancements

---

**Related**: [Development Index](../README.md) | [Roadmaps](../roadmaps/) | [Architecture](../architecture/)
