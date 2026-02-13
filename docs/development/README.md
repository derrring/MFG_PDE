# MFG_PDE Development Documentation

**Last Updated**: 2026-02-13
**Status**: Production-Ready Framework v0.17.8

This directory contains comprehensive development documentation for MFG_PDE contributors and maintainers.

---

## 📁 **Directory Structure**

### **Core Subdirectories**

- **[boundary_conditions/](boundary_conditions/)** - Boundary condition architecture and design
  - BC enforcement architecture, capability matrix, solver integration
  - Corner handling, periodic BC, adjoint consistency
  - 14 documents covering the full BC subsystem

- **[guides/](guides/)** - Developer guides and how-to documents
  - Code quality, API style, consistency guidelines
  - Architecture patterns (1D vs nD solvers, grid vs universal)

- **[planning/](planning/)** - Strategic roadmaps and development plans
  - STRATEGIC_DEVELOPMENT_ROADMAP_2026, priority lists
  - Phase-specific planning documents, BC/geometry roadmaps

- **[typing/](typing/)** - Type system documentation and MyPy guides
  - Strategic typing framework
  - CI/CD troubleshooting reference

- **[maintenance/](maintenance/)** - Audit reports and maintenance procedures
  - Solver infrastructure audits, collocation point reports
  - Silent fallback audit, protocol compliance

- **[status/](status/)** - Current status reports and tracking
  - Code quality status, implementation progress

- **[paradigms/](paradigms/)** - Solver paradigm overviews (numerical, neural, RL, optimization)

### **Root-Level Documents**

Active design and analysis documents (17 files):
- Solver analysis (HJB, FP), operator architecture
- Active issue investigations (#576, #583, #598)
- Migration plans (DerivativeTensors, Semi-Lagrangian)
- Foundational policies (deprecation lifecycle, organization)

---

## 🎯 **Quick Navigation**

### **I want to...**

| Task | Go to |
|------|-------|
| **Understand code standards** | [guides/CONSISTENCY_GUIDE.md](guides/CONSISTENCY_GUIDE.md) |
| **See strategic direction** | [planning/STRATEGIC_DEVELOPMENT_ROADMAP_2026.md](planning/STRATEGIC_DEVELOPMENT_ROADMAP_2026.md) |
| **Understand BC architecture** | [boundary_conditions/BOUNDARY_HANDLING.md](boundary_conditions/BOUNDARY_HANDLING.md) |
| **Fix type errors** | [typing/CI_CD_TROUBLESHOOTING_QUICK_REFERENCE.md](typing/CI_CD_TROUBLESHOOTING_QUICK_REFERENCE.md) |
| **Check code quality status** | [status/CODE_QUALITY_STATUS.md](status/CODE_QUALITY_STATUS.md) |
| **Review audit reports** | [maintenance/](maintenance/) |

---

## 🏆 **Major Achievements (2025)**

### **✅ Strategic Typing Excellence Framework**
**366 → 0 MyPy errors (100% reduction)** - Complete typing perfection achieved:

- **[typing/CI_CD_STRATEGIC_TYPING_EXPERIENCE_GUIDE.md](typing/CI_CD_STRATEGIC_TYPING_EXPERIENCE_GUIDE.md)** - Complete experience guide
- **[typing/STRATEGIC_TYPING_PATTERNS_REFERENCE.md](typing/STRATEGIC_TYPING_PATTERNS_REFERENCE.md)** - Production-tested code patterns
- **[typing/CI_CD_TROUBLESHOOTING_QUICK_REFERENCE.md](typing/CI_CD_TROUBLESHOOTING_QUICK_REFERENCE.md)** - Emergency diagnostic procedures

### **🚀 Research-Optimized CI/CD Pipeline**
**Complete workflow modernization** balancing development productivity with quality assurance:
- Environment compatibility handling (local vs CI/CD differences)
- Strategic validation patterns (strict formatting, informational linting/typing)
- Research codebase optimization for rapid iteration

### **📁 Geometry Module Consolidation (v0.12.5)**
**Systematic organization** of geometry infrastructure:
- Reorganized into subdirectories: boundary/, grids/, meshes/, amr/
- Clear separation of concerns
- Maintained backward compatibility

---

## 🎯 **Strategic Typing Methodology**

The **strategic typing approach** represents a breakthrough in scientific computing type safety:

### **Key Principles**
- **Production-health-first**: Zero breaking changes throughout typing improvements
- **Strategic ignores**: Targeted type ignores for type system limitations vs systematic annotation
- **Environment compatibility**: Handle local vs CI/CD differences gracefully
- **80/20 efficiency**: Maximum typing benefits with minimal development overhead

### **Quantitative Results**
- **Error Reduction**: 366 → 0 (100% success rate)
- **Files Processed**: 91 source files with complete coverage
- **Strategic Ignores**: ~30 targeted, documented ignores
- **Maintenance Overhead**: Low (quarterly review sufficient)

See [typing/](typing/) directory for complete documentation.

---

## 📊 **Framework Evolution Timeline**

### **2025 Major Milestones**
- **v0.16.8** (Dec 2025) - BaseSolver convergence integration, ConvergenceConfig support
- **v0.16.0** (Dec 2025) - Convergence module consolidation, Protocol-based checkers
- **v0.15.0** (Nov 2025) - Architecture refactoring, unified config system
- **v0.12.5** (Nov 2025) - Geometry module consolidation
- **v0.12.0** (Oct 2025) - Strategic typing excellence complete

### **Historical Development (2024-2025)**
1. **v1.0** (2024-07-25) - Repository Reorganization
2. **v1.1** (2024-07-25) - Solver Renaming Convention
3. **v1.2** (2024-07-25) - Advanced Convergence Criteria
4. **v1.3** (2024-07-25) - Adaptive Convergence Decorator Pattern
5. **v1.4** (2025-07-26) - Mathematical Visualization Consistency Fixes
6. **v2.0** (2025-09-26) - Strategic Typing Excellence Framework 🏆

---

## 🚀 **Using This Documentation**

### **For Scientific Computing Projects**
The strategic typing guides provide a **complete framework** for achieving 100% type safety in complex scientific codebases while maintaining development velocity. Start with [typing/](typing/).

### **For CI/CD Optimization**
The troubleshooting guides offer **emergency procedures** and **diagnostic tools** for resolving environment compatibility issues. See [typing/CI_CD_TROUBLESHOOTING_QUICK_REFERENCE.md](typing/CI_CD_TROUBLESHOOTING_QUICK_REFERENCE.md).

### **For Development Teams**
The guides document **lessons learned** and **proven methodologies** for balancing type safety, development productivity, and research flexibility. Browse [guides/](guides/) and [governance/](governance/).

### **For Feature Development**
Follow the roadmaps in [planning/](planning/), and use guides in [guides/](guides/).

---

## 📂 **Directory Quick Reference**

| Directory | Purpose | Files |
|-----------|---------|-------|
| **[boundary_conditions/](boundary_conditions/)** | BC architecture, design, and analysis | 14 |
| **[guides/](guides/)** | Developer how-to guides and patterns | 13 |
| **[planning/](planning/)** | Strategic roadmaps and development plans | 19 |
| **[typing/](typing/)** | Type system and MyPy documentation | 8 |
| **[maintenance/](maintenance/)** | Audit reports and maintenance procedures | 9 |
| **[status/](status/)** | Current status reports and tracking | 9 |
| **[paradigms/](paradigms/)** | Solver paradigm overviews | 4 |
| **Root** | Active design docs and policies | 17 |

---

**Documentation Status**: Production-ready, reorganized 2026-02-13 (Issue #768)
**Applicability**: Scientific computing, research codebases, complex Python projects
**Archived docs**: Completed issue summaries moved to `docs/archive/development/issues/`
